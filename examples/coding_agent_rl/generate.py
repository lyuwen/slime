"""Coding-Agent RL: per-sample generate() function for slime.

    --custom-generate-function-path examples.coding_agent_rl.generate.generate

generate() is a four-stage orchestrator: swe.prepare_workspace + harness.run
-> swe.git_diff -> swe.run_evaluation -> adapter.finish_session. The (harness,
adapter) pair is chosen by the SWE_AGENT env var (claude_code | codex); see
_AGENTS below.
Sandbox-side work is split across three layers: the provider-agnostic sandbox
contract (slime.agent.sandbox), the swappable harness lifecycle
(slime.agent.harness), and the SWE task layer (examples.coding_agent_rl.swe --
dataset parsing, workspace prep, diff, eval). LLM plumbing (Anthropic / OpenAI
<-> SGLang /generate, token capture, segment split) is the matching
slime.agent.adapters adapter. swe.get_metadata documents the dataset row schema
and produces the md dict consumed below.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import secrets
import tempfile
import time
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, TemplateError

from slime.agent.adapters import AnthropicAdapter, OpenAIAdapter
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.agent.harness import ClaudeCodeHarness, CodexHarness, OpenHandsHarness
from slime.agent.sandbox import E2BSandbox, is_fresh_sandbox_retryable
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from . import swe

logger = logging.getLogger(__name__)
logging.getLogger("e2b").setLevel(logging.WARNING)

_AGENTS = {
    "claude_code": (ClaudeCodeHarness, AnthropicAdapter),
    "codex": (CodexHarness, OpenAIAdapter),
    "openhands": (OpenHandsHarness, OpenAIAdapter),
}
AGENT_NAME = os.environ.get("SWE_AGENT", "claude_code")
if AGENT_NAME not in _AGENTS:
    raise ValueError(f"SWE_AGENT={AGENT_NAME!r} not in {sorted(_AGENTS)}")
HARNESS_CLS, ADAPTER_CLS = _AGENTS[AGENT_NAME]


@dataclass(frozen=True)
class SweConfig:
    eval_protocol: str  # eval-path schema/grader (SWE_EVAL_PROTOCOL)
    train_protocol: str  # train-path schema/grader (SWE_TRAIN_PROTOCOL)
    adapter_public_host: str | None
    adapter_bind_host: str
    adapter_port: int
    fork_merge_threshold: int | None
    agent_time_budget_sec: int
    eval_timeout_sec: int
    eval_max_attempts: int  # NEW — SWE_EVAL_MAX_ATTEMPTS
    rollout_guard_sec: int
    boot_concurrency: int
    boot_retries: int
    rollout_retries: int
    retry_policy: str
    oh_fake_user: bool
    oh_max_iterations: int
    oh_tools: list[str]
    oh_extra_envs: dict[str, str]

    @classmethod
    def from_env(cls) -> SweConfig:
        agent_time_budget = int(os.environ.get("SWE_AGENT_TIME_BUDGET_SEC", "1800"))
        eval_timeout = int(os.environ.get("SWE_EVAL_TIMEOUT_SEC", "600"))
        _raw_max_attempts = int(os.environ.get("SWE_EVAL_MAX_ATTEMPTS", "2"))
        if _raw_max_attempts < 1:
            raise ValueError(f"SWE_EVAL_MAX_ATTEMPTS must be >= 1, got {_raw_max_attempts!r}")
        eval_max_attempts = _raw_max_attempts
        guard = int(os.environ.get("SWE_ROLLOUT_GUARD_SEC", "0") or 0) or (
            agent_time_budget + eval_timeout * eval_max_attempts + 180
        )
        fork = int(v) if (v := os.environ.get("SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS")) else None
        oh_tools_raw = os.environ.get("SWE_OH_TOOLS", "file_editor,terminal,task_tracker,think,finish")
        oh_tools = [t.strip() for t in oh_tools_raw.split(",") if t.strip()]
        oh_extra_envs_raw = os.environ.get("SLIME_AGENT_OH_EXTRA_ENVS", "").strip()
        oh_extra_envs = json.loads(oh_extra_envs_raw) if oh_extra_envs_raw else {}
        if not isinstance(oh_extra_envs, dict):
            raise ValueError("SLIME_AGENT_OH_EXTRA_ENVS must be a JSON object")
        rollout_retries = int(os.environ.get("SWE_ROLLOUT_RETRIES", "2"))
        if rollout_retries < 0:
            raise ValueError("SWE_ROLLOUT_RETRIES must be non-negative")
        retry_policy = os.environ.get("SWE_ROLLOUT_RETRY_POLICY", "pre-launch")
        if retry_policy not in ("pre-launch", "retry-from-scratch", "always-fail"):
            raise ValueError(
                f"SWE_ROLLOUT_RETRY_POLICY={retry_policy!r} invalid; "
                f"must be pre-launch|retry-from-scratch|always-fail"
            )
        return cls(
            eval_protocol=os.environ.get("SWE_EVAL_PROTOCOL", swe.PROTOCOL_SCALESWE),
            train_protocol=os.environ.get("SWE_TRAIN_PROTOCOL", swe.PROTOCOL_SCALESWE),
            adapter_public_host=os.environ.get("ADAPTER_PUBLIC_HOST"),
            adapter_bind_host=os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0"),
            adapter_port=int(os.environ.get("ADAPTER_PORT", "18001")),
            fork_merge_threshold=fork,
            agent_time_budget_sec=agent_time_budget,
            eval_timeout_sec=eval_timeout,
            eval_max_attempts=eval_max_attempts,
            rollout_guard_sec=guard,
            boot_concurrency=int(os.environ.get("SWE_BOOT_CONCURRENCY", "16")),
            boot_retries=int(os.environ.get("SWE_BOOT_RETRIES", "2")),
            rollout_retries=rollout_retries,
            retry_policy=retry_policy,
            oh_fake_user=os.environ.get("SWE_OH_FAKE_USER", "0") not in ("0", "", "false", "False"),
            oh_max_iterations=int(os.environ.get("SWE_OH_MAX_ITERATIONS", "100")),
            oh_tools=oh_tools,
            oh_extra_envs=oh_extra_envs,
        )


CONFIG = SweConfig.from_env()


def get_prompt(md: dict) -> str:
    """Generate the agent prompt, optionally using a Jinja2 template.

    If SWE_PROMPT_TEMPLATE_PATH is set, loads and renders that template
    with the metadata dict. Otherwise falls back to the hardcoded SWE_PROMPT.

    Args:
        md: Metadata dict from swe.get_metadata(), containing:
            - problem_statement: The task description
            - workdir: Repository path in the sandbox
            - instance_id: Unique identifier
            - image: Docker image name
            - protocol: Evaluation protocol

    Returns:
        Rendered prompt string to pass to the harness
    """
    template_path_str = os.environ.get("SWE_PROMPT_TEMPLATE_PATH", "").strip()

    if not template_path_str:
        return swe.SWE_PROMPT

    template_path = Path(template_path_str)

    try:
        # Load template
        if not template_path.exists():
            logger.warning(
                "[coding_agent_rl] Template file not found: %s; falling back to SWE_PROMPT",
                template_path,
            )
            return swe.SWE_PROMPT

        # Set up Jinja2 environment
        template_dir = template_path.parent
        template_name = template_path.name
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        template = env.get_template(template_name)

        # Build context with instance alias for legacy.j2 compatibility
        context = {
            "instance": {
                "problem_statement": md.get("problem_statement", ""),
                "repo_path": md.get("workdir", ""),
                "workdir": md.get("workdir", ""),
                "instance_id": md.get("instance_id", ""),
                "image": md.get("image", ""),
            },
            "md": md,
        }

        # Render template
        rendered = template.render(context)
        logger.info("[coding_agent_rl] Using template prompt from %s", template_path)
        return rendered

    except TemplateError as e:
        logger.error(
            "[coding_agent_rl] Template rendering failed (%s): %s: %s; falling back to SWE_PROMPT",
            template_path,
            type(e).__name__,
            str(e),
        )
        return swe.SWE_PROMPT
    except Exception as e:
        logger.error(
            "[coding_agent_rl] Unexpected error loading template (%s): %s: %s; falling back to SWE_PROMPT",
            template_path,
            type(e).__name__,
            str(e),
        )
        return swe.SWE_PROMPT


_BOOT_SEM = asyncio.Semaphore(CONFIG.boot_concurrency)


@asynccontextmanager
async def boot_agent_sandbox(image: str, instance_id: str) -> AsyncIterator[E2BSandbox]:
    """Boot a fresh E2B sandbox and install the selected harness toolchain.

    Create the sandbox from the dataset image, install Node 22 + the harness CLI
    from host tarballs, retry transient boot/install failures, and close the
    sandbox when the caller leaves the context.
    """
    sb = None
    last_err: Exception | None = None
    for attempt in range(CONFIG.boot_retries):
        cand = E2BSandbox(image)
        try:
            async with _BOOT_SEM:
                await cand.__aenter__()
                try:
                    # Temporary sanitation for recycled Kruise sandboxes.
                    # Remove transient state potentially left by a previous claimant.
                    await cand.exec(
                        "rm -f /tmp/.run.sh /tmp/.run-*.sh /etc/gitconfig.lock; chmod 1777 /tmp",
                        user="root",
                        check=True,
                        timeout=30,
                    )

                    await HARNESS_CLS().install_cli(cand)
                except BaseException:
                    await cand.__aexit__(None, None, None)
                    raise
            sb = cand
            break
        except Exception as e:
            last_err = e
            # always-fail policy fails any exception immediately (README/design):
            # do not consume the internal boot_retries budget under this policy.
            if CONFIG.retry_policy == "always-fail":
                raise
            logger.warning(
                "[coding_agent_rl] %s: provision attempt %d/%d failed: %s: %s",
                instance_id,
                attempt + 1,
                CONFIG.boot_retries,
                type(e).__name__,
                str(e)[:200],
            )
            await asyncio.sleep(1 + attempt + random.random())
    if sb is None:
        assert last_err is not None
        raise last_err
    try:
        yield sb
    finally:
        await sb.__aexit__(None, None, None)


class _AdapterService(metaclass=SingletonMeta):
    def __init__(self, args) -> None:
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)
        self.tool_parser = getattr(args, "sglang_tool_call_parser", None) or None
        self.reasoning_parser = getattr(args, "sglang_reasoning_parser", None) or None
        sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        if not CONFIG.adapter_public_host:
            raise RuntimeError(
                "ADAPTER_PUBLIC_HOST is not set. Export it to the host IP that "
                "sandboxes can reach for reverse-connection to the adapter; "
                "without it the sandbox cannot dial back and the rollout aborts."
            )
        self.adapter = ADAPTER_CLS(
            tokenizer=self.tokenizer,
            sglang_url=sglang_url,
            tool_parser=self.tool_parser,
            reasoning_parser=self.reasoning_parser,
            fork_threshold_tokens=CONFIG.fork_merge_threshold,
            chat_template_kwargs=getattr(args, "apply_chat_template_kwargs", None),
        )
        # handler_cancellation=True so a client disconnect cancels the handler
        # coroutine, arming the fire-and-forget /abort_request in the adapter.
        # Otherwise a cancelled client leaves an inflight sglang /generate that
        # races the next release_memory_occupation and trips its idle assertion.
        self.app_handle = run_app_in_thread(
            self.adapter.app,
            host=CONFIG.adapter_bind_host,
            port=CONFIG.adapter_port,
            thread_name="anthropic-adapter",
            runner_kwargs={
                "handler_cancellation": True,
                "access_log_class": FilteredAccessLogger,
            },
        )
        self.adapter_url = f"http://{CONFIG.adapter_public_host}:{self.app_handle.port}"
        logger.info(
            "[coding_agent_rl] tokenizer=%s adapter=%s max_context_len=%s tool_parser=%s reasoning_parser=%s",
            args.hf_checkpoint,
            self.adapter_url,
            self.max_context_len,
            self.tool_parser,
            self.reasoning_parser,
        )


def _is_retryable(e: Exception) -> bool:
    """Determine if an exception represents a transient failure worth retrying.

    Retryable errors include E2B RPC failures, workspace setup issues, and
    certain network/sandbox errors. Non-retryable errors include dataset
    issues, adapter problems, and application logic failures.
    """
    # Kernel classifier is the single source of truth for generic
    # SandboxException / E2B transport errors. Consult it first.
    if is_fresh_sandbox_retryable(e):
        return True
    # For a SandboxException the kernel classifier is authoritative in BOTH
    # directions: it already returns True for transient provider errors and
    # False for the permanent auth/quota/billing denylist. Do not let the
    # message heuristics below (e.g. "create") resurrect a rejected one.
    if type(e).__name__ == "SandboxException":
        return False

    msg = str(e).lower()
    exc_type = type(e).__name__.lower()

    # E2B exec failures (especially exit=255) are often transient
    if "e2b exec failed" in msg or "exit=255" in msg or "exit 255" in msg:
        return True

    # General sandbox/RPC errors
    if "runtimeerror" in exc_type and ("sandbox" in msg or "exec" in msg or "e2b" in msg):
        return True

    # Connection/network issues (but not asyncio.TimeoutError — that's the wall-clock guard)
    if any(x in msg for x in ["connection", "timeout", "network", "unavailable"]) and "asyncio" not in exc_type:
        return True

    # Sandbox boot failures
    if "boot" in msg or "create" in msg:
        return True

    return False


async def generate(args, base_sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False):
    """Per-sample agent function with wall-clock guard and setup retries."""
    state = _AdapterService(args)
    protocol = CONFIG.eval_protocol if evaluation else CONFIG.train_protocol
    md = swe.get_metadata(base_sample, protocol)
    instance_id = md["instance_id"]
    if not md["image"] or not md["workdir"]:
        return _abort_result(base_sample, "missing_image_or_workdir", instance_id)
    reason = swe.evaluability_check(md)
    if reason:
        return _abort_result(base_sample, f"unevaluatable:{reason}", instance_id)

    base_session_id = _session_id(base_sample, instance_id)
    session_id = base_session_id
    session_open = False
    t0 = time.time()
    traj_data = None

    try:
        async with asyncio.timeout(CONFIG.rollout_guard_sec):
            for attempt in range(CONFIG.rollout_retries + 1):
                session_id = f"{base_session_id}-a{attempt}"
                session_open = False
                try:
                    async with boot_agent_sandbox(md["image"], instance_id) as sb:
                        await swe.prepare_workspace(sb, md["workdir"], md)
                        state.adapter.open_session(
                            session_id,
                            sampling_defaults=sampling_params,
                            max_context_tokens=state.max_context_len,
                        )
                        session_open = True
                        oh_kwargs = (
                            {
                                "fake_user": CONFIG.oh_fake_user,
                                "max_iterations": CONFIG.oh_max_iterations,
                                "tools": CONFIG.oh_tools,
                                "extra_envs": CONFIG.oh_extra_envs,
                            }
                            if AGENT_NAME == "openhands"
                            else {}
                        )
                        agent_exit_code = await HARNESS_CLS().run(
                            sb,
                            workdir=md["workdir"],
                            session_id=session_id,
                            adapter_url=state.adapter_url,
                            time_budget_sec=CONFIG.agent_time_budget_sec,
                            prompt=get_prompt(md),
                            **oh_kwargs,
                        )
                        diff_text = await swe.git_diff(sb, md["workdir"])
                        if AGENT_NAME == "openhands":
                            traj_data = await _read_sandbox_trajectory(sb, OpenHandsHarness.trajectory_sandbox_path)
                    # Only the winning sid is used downstream (finish_session/eval).
                    base_sample.session_id = session_id
                    break
                except Exception as error:
                    if CONFIG.retry_policy == "always-fail":
                        raise
                    turns_recorded = state.adapter.manager.has_session(session_id)
                    if not _is_retryable(error) or attempt >= CONFIG.rollout_retries:
                        raise
                    if CONFIG.retry_policy == "pre-launch" and turns_recorded:
                        raise
                    # retry permitted
                    if session_open:
                        await state.adapter.drop_session(session_id, wait_timeout=30)
                        session_open = False
                    backoff = 2**attempt
                    logger.warning(
                        "[coding_agent_rl] %s: setup attempt %d/%d failed: %s, retrying in %ds...",
                        instance_id,
                        attempt + 1,
                        CONFIG.rollout_retries + 1,
                        error,
                        backoff,
                    )
                    await asyncio.sleep(backoff)

            reward, applied_cleanly = await swe.run_evaluation(
                md,
                diff_text=diff_text,
                timeout_sec=CONFIG.eval_timeout_sec,
                max_attempts=CONFIG.eval_max_attempts,
            )
            if traj_data is not None:
                _persist_trajectory(
                    _trajectory_root(),
                    base_sample,
                    messages=traj_data["messages"],
                    tools=traj_data["tools"],
                    metrics=traj_data.get("metrics"),
                    diff_text=diff_text,
                    reward=float(reward),
                    applied_cleanly=bool(applied_cleanly),
                    instance_id=instance_id,
                    session_id=session_id,
                    agent_exit_code=agent_exit_code,
                )
            if evaluation:
                logger.info(
                    "[coding_agent_rl] %s: reward=%.2f applied=%s agent_exit_code=%d elapsed=%.1fs (eval-only)",
                    instance_id,
                    float(reward),
                    bool(applied_cleanly),
                    agent_exit_code,
                    time.time() - t0,
                )
                return _eval_result(
                    base_sample,
                    reward=float(reward),
                    applied_cleanly=bool(applied_cleanly),
                    agent_exit_code=agent_exit_code,
                    instance_id=instance_id,
                )

            samples = await state.adapter.finish_session(
                session_id,
                base_sample=base_sample,
                reward=float(reward),
                extra_metadata={
                    "grading_solved": float(reward) == 1.0,
                    "instance_id": instance_id,
                },
            )
            if not samples:
                return _abort_result(base_sample, "adapter_session_empty", instance_id)

            for sample in samples:
                sample.metadata = {**(sample.metadata or {}), "agent_exit_code": agent_exit_code}
            if agent_exit_code != 0:
                reason = "time budget exceeded" if agent_exit_code < 0 else f"CLI error (exit {agent_exit_code})"
                logger.warning(
                    "[coding_agent_rl] %s: agent_exit_code=%d (%s)",
                    instance_id,
                    agent_exit_code,
                    reason,
                )
            logger.info(
                "[coding_agent_rl] %s: reward=%.2f applied=%s agent_exit_code=%d elapsed=%.1fs segments=%d",
                instance_id,
                float(reward),
                bool(applied_cleanly),
                agent_exit_code,
                time.time() - t0,
                len(samples),
            )
            return samples

    except asyncio.TimeoutError:
        _log_timeout_diagnostic(t0, instance_id)
        return _abort_result(base_sample, "wall_clock_timeout", instance_id)
    except Exception as error:
        logger.warning(
            "[coding_agent_rl] %s: rollout failed: %s\n%s",
            instance_id,
            error,
            traceback.format_exc(),
        )
        return _abort_result(base_sample, f"exception:{type(error).__name__}", instance_id)
    finally:
        if session_open:
            await state.adapter.drop_session(session_id, wait_timeout=30)  # cleanup only, idempotent
            await asyncio.sleep(10)


def _log_timeout_diagnostic(t0: float, instance_id: str) -> None:
    # Dump pending-task names when the wall-clock guard fires. Must not crash.
    try:
        elapsed = time.time() - t0
        pending = [t for t in asyncio.all_tasks() if not t.done()]
        stuck = []
        for t in pending[:5]:  # cap to avoid log spam
            coro = getattr(t, "_coro", None)
            stuck.append(getattr(coro, "__qualname__", repr(coro)))
        logger.warning(
            "[coding_agent_rl] %s: wall_clock_timeout after %.1fs "
            "(guard=%ds); %d tasks pending; sample of stuck: %s",
            instance_id,
            elapsed,
            CONFIG.rollout_guard_sec,
            len(pending),
            stuck,
        )
    except Exception:  # pragma: no cover - diag must never crash
        pass


def _session_id(sample: Sample, instance_id: str) -> str:
    if sample.session_id:
        return sample.session_id
    if sample.index is not None and sample.group_index is not None:
        return f"cagent-{instance_id}-{sample.index}-{sample.group_index}"
    return f"cagent-{instance_id}-{secrets.token_hex(8)}"


def _trajectory_root() -> str:
    return os.environ.get("SWE_TRAJECTORY_DIR", "trajectories")


def _trajectory_final_path(root: str, sample: Sample) -> str:
    group_index = "unknown" if sample.group_index is None else str(sample.group_index)
    index = "unknown" if sample.index is None else str(sample.index)
    return os.path.join(root, group_index, f"{group_index}_{index}.json")


def _warn_trajectory(message: str, *args) -> None:
    try:
        logger.warning(message, *args)
    except Exception:
        pass


async def _read_sandbox_trajectory(sb, path: str) -> dict | None:
    """Return the parsed sandbox trajectory object, or None on read/parse failure.

    Accepts only an object with ``messages`` (list) and ``tools`` (list).
    Any other shape — bare list, missing key, or non-list field — is rejected
    with a warning. All warning failures are swallowed (best-effort).
    """
    try:
        raw = await sb.read_file(path, user="agent")
    except Exception as error:
        _warn_trajectory("[coding_agent_rl] trajectory read failed for %s: %s", path, error)
        return None
    if not raw:
        _warn_trajectory("[coding_agent_rl] trajectory content empty or missing at %s", path)
        return None
    try:
        data = json.loads(raw)
    except Exception as error:
        _warn_trajectory("[coding_agent_rl] trajectory read or parse failed for %s: %s", path, error)
        return None
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("messages"), list)
        or not isinstance(data.get("tools"), list)
    ):
        _warn_trajectory(
            "[coding_agent_rl] trajectory has unexpected shape at %s: %s",
            path,
            type(data).__name__,
        )
        return None
    return data


def _persist_trajectory(
    root: str,
    sample: Sample,
    *,
    messages: list,
    tools: list,
    diff_text: str,
    reward: float,
    applied_cleanly: bool,
    instance_id: str,
    session_id: str,
    agent_exit_code: int | None,
    metrics: dict | None = None,
) -> None:
    """Best-effort atomic write of the enriched trajectory document."""
    try:
        path = _trajectory_final_path(root, sample)
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        doc = {
            "messages": messages,
            "tools": tools,
            "diff_text": diff_text,
            "reward": reward,
            "applied_cleanly": applied_cleanly,
            "instance_id": instance_id,
            "group_index": sample.group_index,
            "index": sample.index,
            "session_id": session_id,
            "agent_exit_code": agent_exit_code,
        }
        if metrics is not None:
            doc["metrics"] = metrics
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".traj_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as file:
                json.dump(doc, file)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    except Exception as error:
        _warn_trajectory("[coding_agent_rl] %s: trajectory persist skipped: %s", instance_id, error)


def _abort_result(sample: Sample, reason: str, instance_id: str) -> list[Sample]:
    """Mark ``sample`` aborted in place and return it in the list shape this
    fan-out generate function always yields."""
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = 0.0
    sample.remove_sample = True
    sample.status = Sample.Status.ABORTED
    sample.metadata = {
        **(sample.metadata or {}),
        "abort_reason": reason,
        "instance_id": instance_id,
    }
    logger.warning("[coding_agent_rl] %s aborted: %s", instance_id, reason)
    return [sample]


def _eval_result(
    sample: Sample,
    *,
    reward: float,
    applied_cleanly: bool,
    agent_exit_code: int | None,
    instance_id: str,
) -> list[Sample]:
    """Eval-path placeholder: only ``reward`` matters for ``eval/sweb``."""

    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = float(reward)
    sample.remove_sample = True
    sample.status = Sample.Status.COMPLETED
    sample.metadata = {
        **(sample.metadata or {}),
        "instance_id": instance_id,
        "grading_solved": float(reward) == 1.0,
        "applied_cleanly": applied_cleanly,
        "agent_exit_code": agent_exit_code,
    }
    return [sample]
