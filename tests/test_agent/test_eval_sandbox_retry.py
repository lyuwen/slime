"""CPU-only tests for the evaluation sandbox retry mechanism.

Covers:
  §1  is_fresh_sandbox_retryable() exception classification
  §2  run_evaluation() retry loop orchestration (swe.py)
  §3  SweConfig env-var parsing (generate.py)
  §4  Derived rollout guard formula
  §5  Log messages emitted during retry
  §6  Grader execs are non-idempotent
  §7  Real fresh-sandbox lifecycle (distinct sandboxes, cleanup, dispatch)
  §8  read_file(strict=) contract (fresh-retryable re-raise vs "" masking)
  §9  SWE-bench strict-output integration (real exec_and_wait + real grader)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.agent import sandbox as sandbox_mod  # noqa: E402


# ---------------------------------------------------------------------------
# §1  is_fresh_sandbox_retryable predicate
# ---------------------------------------------------------------------------


def _exc(name: str, msg: str = "") -> BaseException:
    """Build a fake exception whose __class__.__name__ == name."""
    cls = type(name, (Exception,), {})
    return cls(msg)


def test_fresh_retryable_transport_errors():
    """All HTTP/transport errors that _is_transient_rpc_error recognises are
    also fresh-sandbox retryable."""
    transport_names = [
        "ReadError",
        "WriteError",
        "ConnectError",
        "ProtocolError",
        "LocalProtocolError",
        "RemoteProtocolError",
        "SSLError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "TimeoutException",
    ]
    for name in transport_names:
        e = _exc(name)
        assert sandbox_mod.is_fresh_sandbox_retryable(e), f"{name} should be fresh-retryable"


def test_fresh_retryable_sandbox_stopped():
    """SandboxException with 'STOPPED state' IS fresh-sandbox retryable even
    though it is NOT same-sandbox retryable."""
    e = _exc("SandboxException", "sandbox abc123 in STOPPED state")
    assert sandbox_mod.is_fresh_sandbox_retryable(e)


def test_fresh_retryable_sandbox_not_exist():
    """SandboxException with 'does not exist' IS fresh-sandbox retryable."""
    e = _exc("SandboxException", "sandbox xyz does not exist")
    assert sandbox_mod.is_fresh_sandbox_retryable(e)


def test_not_fresh_retryable_key_error():
    """KeyError (programmer error) is never retried."""
    assert not sandbox_mod.is_fresh_sandbox_retryable(KeyError("missing_key"))


def test_not_fresh_retryable_json_decode():
    """JSONDecodeError is never retried."""
    import json

    e = json.JSONDecodeError("msg", "", 0)
    assert not sandbox_mod.is_fresh_sandbox_retryable(e)


def test_not_fresh_retryable_cancelled():
    """asyncio.CancelledError is never retried."""
    assert not sandbox_mod.is_fresh_sandbox_retryable(asyncio.CancelledError())


def test_not_fresh_retryable_timeout():
    """asyncio.TimeoutError is never retried."""
    assert not sandbox_mod.is_fresh_sandbox_retryable(asyncio.TimeoutError())


def test_not_fresh_retryable_permanent_marker_sandbox_exception():
    """A SandboxException carrying a permanent auth/quota marker ("quota") is
    NOT fresh-retryable: a fresh sandbox cannot recover a quota failure."""
    e = _exc("SandboxException", "quota exceeded")
    # "quota exceeded" matches the permanent marker "quota", so it stays False.
    assert not sandbox_mod.is_fresh_sandbox_retryable(e)


def test_fresh_retryable_generic_transient_sandbox_exception():
    """A generic SandboxException (transient gateway) IS fresh-sandbox retryable
    now that grader execs bypass same-sandbox retry."""
    e = _exc("SandboxException", "gateway timeout talking to sandbox service")
    assert sandbox_mod.is_fresh_sandbox_retryable(e)


def test_not_fresh_retryable_permanent_auth_sandbox_exception():
    """Permanent auth/quota SandboxExceptions are NOT retried."""
    for msg in ("401 Unauthorized", "quota exceeded", "invalid API key", "billing required"):
        e = _exc("SandboxException", msg)
        assert not sandbox_mod.is_fresh_sandbox_retryable(e), msg


def test_same_sandbox_stopped_not_retried():
    """Confirm the existing same-sandbox loop still does NOT retry STOPPED."""
    sb = sandbox_mod.E2BSandbox.__new__(sandbox_mod.E2BSandbox)
    e = _exc("SandboxException", "sandbox in STOPPED state")
    assert not sb._is_transient_rpc_error(e)


# ---------------------------------------------------------------------------
# §2  run_evaluation() retry orchestration
# ---------------------------------------------------------------------------

import examples.coding_agent_rl.swe as swe_mod  # noqa: E402


def _read_error() -> Exception:
    return _exc("ReadError", "connection reset")


def _make_eval_result(reward: float = 1.0) -> swe_mod.EvalResult:
    return swe_mod.EvalResult(reward, True)


def test_retry_on_read_error_succeeds_on_second_attempt():
    """First attempt raises ReadError; second attempt succeeds.
    The dispatcher must be called exactly twice and return the second result."""
    second_result = _make_eval_result(1.0)
    call_count = 0

    async def fake_once(md, diff_text, timeout_sec):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _read_error()
        return second_result

    async def run():
        with patch.object(swe_mod, "_run_evaluation_once", fake_once):
            return await swe_mod.run_evaluation(
                {"protocol": "scaleswe", "instance_id": "inst-1"},
                diff_text="diff",
                timeout_sec=10,
                max_attempts=2,
            )

    result = asyncio.run(run())
    assert result == second_result
    assert call_count == 2


def test_no_retry_on_zero_reward():
    """EvalResult(0.0, True) is a valid outcome and must NOT be retried."""
    call_count = 0

    async def fake_once(md, diff_text, timeout_sec):
        nonlocal call_count
        call_count += 1
        return swe_mod.EvalResult(0.0, True)

    async def run():
        with patch.object(swe_mod, "_run_evaluation_once", fake_once):
            return await swe_mod.run_evaluation(
                {"protocol": "scaleswe", "instance_id": "inst-2"},
                diff_text="",
                timeout_sec=10,
                max_attempts=3,
            )

    result = asyncio.run(run())
    assert result == swe_mod.EvalResult(0.0, True)
    assert call_count == 1


def test_no_retry_on_apply_failure():
    """EvalResult(0.0, False) (patch apply failure) must NOT be retried."""
    call_count = 0

    async def fake_once(md, diff_text, timeout_sec):
        nonlocal call_count
        call_count += 1
        return swe_mod.EvalResult(0.0, False)

    async def run():
        with patch.object(swe_mod, "_run_evaluation_once", fake_once):
            return await swe_mod.run_evaluation(
                {"protocol": "scaleswe", "instance_id": "inst-3"},
                diff_text="diff",
                timeout_sec=10,
                max_attempts=3,
            )

    result = asyncio.run(run())
    assert result == swe_mod.EvalResult(0.0, False)
    assert call_count == 1


def test_exhausted_attempts_reraises_last_exception():
    """When all max_attempts raise ReadError, the last exception is re-raised."""
    call_count = 0

    async def fake_once(md, diff_text, timeout_sec):
        nonlocal call_count
        call_count += 1
        raise _read_error()

    async def run():
        with patch.object(swe_mod, "_run_evaluation_once", fake_once):
            await swe_mod.run_evaluation(
                {"protocol": "scaleswe", "instance_id": "inst-4"},
                diff_text="diff",
                timeout_sec=10,
                max_attempts=3,
            )

    with pytest.raises(Exception, match="connection reset"):
        asyncio.run(run())
    assert call_count == 3


def test_non_retryable_exception_propagates_immediately():
    """A KeyError (non-infra) propagates after exactly one call."""
    call_count = 0

    async def fake_once(md, diff_text, timeout_sec):
        nonlocal call_count
        call_count += 1
        raise KeyError("bad_key")

    async def run():
        with patch.object(swe_mod, "_run_evaluation_once", fake_once):
            await swe_mod.run_evaluation(
                {"protocol": "scaleswe", "instance_id": "inst-5"},
                diff_text="diff",
                timeout_sec=10,
                max_attempts=3,
            )

    with pytest.raises(KeyError):
        asyncio.run(run())
    assert call_count == 1


def test_cancelled_error_not_swallowed():
    """asyncio.CancelledError must never be caught as a retry trigger."""
    call_count = 0

    async def fake_once(md, diff_text, timeout_sec):
        nonlocal call_count
        call_count += 1
        raise asyncio.CancelledError()

    async def run():
        with patch.object(swe_mod, "_run_evaluation_once", fake_once):
            await swe_mod.run_evaluation(
                {"protocol": "scaleswe", "instance_id": "inst-6"},
                diff_text="",
                timeout_sec=10,
                max_attempts=3,
            )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())
    assert call_count == 1


def test_max_attempts_one_single_attempt():
    """max_attempts=1 makes exactly one evaluation attempt; a retryable
    infrastructure error still propagates (no fresh-sandbox retry). This is not
    byte-identical to pre-retry behaviour — grader execs are now non-idempotent —
    but the attempt count is one."""
    call_count = 0

    async def fake_once(md, diff_text, timeout_sec):
        nonlocal call_count
        call_count += 1
        raise _read_error()

    async def run():
        with patch.object(swe_mod, "_run_evaluation_once", fake_once):
            await swe_mod.run_evaluation(
                {"protocol": "scaleswe", "instance_id": "inst-7"},
                diff_text="",
                timeout_sec=10,
                max_attempts=1,
            )

    with pytest.raises(Exception, match="connection reset"):
        asyncio.run(run())
    assert call_count == 1


def test_scaleswe_and_swebench_both_use_retry_boundary():
    """Both protocol values reach _run_evaluation_once under the retry loop."""
    results = []

    async def fake_once(md, diff_text, timeout_sec):
        results.append(md["protocol"])
        return swe_mod.EvalResult(1.0, True)

    async def run():
        with patch.object(swe_mod, "_run_evaluation_once", fake_once):
            await swe_mod.run_evaluation(
                {"protocol": "scaleswe", "instance_id": "s1"},
                diff_text="",
                timeout_sec=10,
                max_attempts=1,
            )
            await swe_mod.run_evaluation(
                {"protocol": "swebench", "instance_id": "s2"},
                diff_text="",
                timeout_sec=10,
                max_attempts=1,
            )

    asyncio.run(run())
    assert results == ["scaleswe", "swebench"]


# ---------------------------------------------------------------------------
# §3  SweConfig env-var parsing
# ---------------------------------------------------------------------------

import os  # noqa: E402
import types  # noqa: E402

# Importing generate pulls in transformers via slime.utils.processing_utils and
# uses asyncio.timeout() (3.11+). Stub/shim both so the import resolves on the
# CPU-only 3.10 CI env (mirrors tests/test_agent/test_agent_rollout_cpu.py).
if "transformers" not in sys.modules:
    _tf_stub = types.ModuleType("transformers")
    for _name in ("AutoProcessor", "AutoTokenizer", "PreTrainedTokenizerBase", "ProcessorMixin"):
        setattr(_tf_stub, _name, type(_name, (), {}))
    sys.modules["transformers"] = _tf_stub

if not hasattr(asyncio, "timeout"):
    import contextlib

    @contextlib.asynccontextmanager
    async def _timeout_shim(_delay):
        yield

    asyncio.timeout = _timeout_shim

import examples.coding_agent_rl.generate as gen_mod  # noqa: E402


def _make_config(**env_overrides):
    """Build a SweConfig.from_env() with selective env overrides."""
    base = {
        "SWE_AGENT_TIME_BUDGET_SEC": "1800",
        "SWE_EVAL_TIMEOUT_SEC": "600",
        "ADAPTER_PUBLIC_HOST": "127.0.0.1",
    }
    base.update(env_overrides)
    with patch.dict(os.environ, base, clear=False):
        # Unset guard so it is derived, unless caller sets it explicitly.
        env = {**base}
        if "SWE_ROLLOUT_GUARD_SEC" not in env_overrides:
            env.pop("SWE_ROLLOUT_GUARD_SEC", None)
        with patch.dict(os.environ, env, clear=False):
            # Remove guard if not explicitly set so derived formula applies.
            saved = os.environ.pop("SWE_ROLLOUT_GUARD_SEC", None)
            try:
                return gen_mod.SweConfig.from_env()
            finally:
                if saved is not None:
                    os.environ["SWE_ROLLOUT_GUARD_SEC"] = saved


def test_default_eval_max_attempts_is_two():
    cfg = _make_config()
    assert cfg.eval_max_attempts == 2


def test_eval_max_attempts_parsed_from_env():
    cfg = _make_config(SWE_EVAL_MAX_ATTEMPTS="4")
    assert cfg.eval_max_attempts == 4


def test_eval_max_attempts_one_is_valid():
    cfg = _make_config(SWE_EVAL_MAX_ATTEMPTS="1")
    assert cfg.eval_max_attempts == 1


def test_eval_max_attempts_zero_raises():
    """Values below 1 must fail configuration parsing early."""
    with pytest.raises((ValueError, SystemExit)):
        _make_config(SWE_EVAL_MAX_ATTEMPTS="0")


def test_eval_max_attempts_negative_raises():
    with pytest.raises((ValueError, SystemExit)):
        _make_config(SWE_EVAL_MAX_ATTEMPTS="-1")


# ---------------------------------------------------------------------------
# §4  Derived rollout guard formula
# ---------------------------------------------------------------------------


def test_derived_guard_uses_eval_max_attempts():
    """guard = agent_budget + eval_timeout * eval_max_attempts + 180."""
    cfg = _make_config(
        SWE_AGENT_TIME_BUDGET_SEC="1800",
        SWE_EVAL_TIMEOUT_SEC="600",
        SWE_EVAL_MAX_ATTEMPTS="2",
    )
    assert cfg.rollout_guard_sec == 1800 + 600 * 2 + 180  # 3180


def test_derived_guard_single_attempt():
    """With max_attempts=1, guard = agent + eval*1 + 180."""
    cfg = _make_config(
        SWE_AGENT_TIME_BUDGET_SEC="1800",
        SWE_EVAL_TIMEOUT_SEC="600",
        SWE_EVAL_MAX_ATTEMPTS="1",
    )
    assert cfg.rollout_guard_sec == 1800 + 600 * 1 + 180  # 2580


def test_explicit_guard_not_overridden():
    """An explicit SWE_ROLLOUT_GUARD_SEC overrides the derived formula."""
    with patch.dict(
        os.environ,
        {
            "SWE_AGENT_TIME_BUDGET_SEC": "1800",
            "SWE_EVAL_TIMEOUT_SEC": "600",
            "SWE_EVAL_MAX_ATTEMPTS": "5",
            "SWE_ROLLOUT_GUARD_SEC": "9999",
            "ADAPTER_PUBLIC_HOST": "127.0.0.1",
        },
        clear=False,
    ):
        cfg = gen_mod.SweConfig.from_env()
    assert cfg.rollout_guard_sec == 9999


# ---------------------------------------------------------------------------
# §5  Logging during retry
# ---------------------------------------------------------------------------


def test_retry_warning_contains_instance_and_attempt(caplog):
    """The retry warning must include instance_id, attempt number, and exception type."""
    import logging

    call_count = 0

    async def fake_once(md, diff_text, timeout_sec):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _exc("ReadError", "connection reset")
        return swe_mod.EvalResult(1.0, True)

    async def run():
        with patch.object(swe_mod, "_run_evaluation_once", fake_once):
            with caplog.at_level(logging.WARNING, logger="examples.coding_agent_rl.swe"):
                return await swe_mod.run_evaluation(
                    {"protocol": "scaleswe", "instance_id": "astropy_pr44"},
                    diff_text="diff",
                    timeout_sec=10,
                    max_attempts=2,
                )

    asyncio.run(run())
    warning_texts = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("astropy_pr44" in t for t in warning_texts), "instance_id missing from warning"
    assert any("scaleswe" in t for t in warning_texts), "protocol missing from warning"
    assert any("1/2" in t for t in warning_texts), "attempt count missing from warning"
    assert any("ReadError" in t for t in warning_texts), "exception type missing from warning"
    assert any("fresh evaluator sandbox" in t for t in warning_texts), "fresh sandbox note missing"


def test_success_after_retry_emits_info_log(caplog):
    """After recovering via retry, an info log must record total attempts and reward."""
    import logging

    call_count = 0

    async def fake_once(md, diff_text, timeout_sec):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _exc("ReadError", "blip")
        return swe_mod.EvalResult(1.0, True)

    async def run():
        with patch.object(swe_mod, "_run_evaluation_once", fake_once):
            with caplog.at_level(logging.INFO, logger="examples.coding_agent_rl.swe"):
                return await swe_mod.run_evaluation(
                    {"protocol": "scaleswe", "instance_id": "inst-log"},
                    diff_text="",
                    timeout_sec=10,
                    max_attempts=2,
                )

    asyncio.run(run())
    info_texts = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert any("inst-log" in t for t in info_texts), "instance_id missing from success info log"
    assert any("scaleswe" in t for t in info_texts), "protocol missing from success info log"
    assert any("2" in t for t in info_texts), "attempt count missing from success info log"


# ---------------------------------------------------------------------------
# §6  grader execs are non-idempotent (spec: no same-sandbox grader re-run)
# ---------------------------------------------------------------------------


class _RecordingSandbox:
    """Minimal Sandbox stand-in that records each exec's idempotent flag."""

    sandbox_id = "rec-1"

    def __init__(self):
        self.execs = []  # list of (cmd, idempotent)

    async def exec(self, cmd, *, user="root", env=None, timeout=120, check=False, idempotent=True):
        self.execs.append((cmd, idempotent))
        return (0, "", "")

    async def write_file(self, path, content, *, user="root"):
        return None

    async def read_file(self, path, *, user="root", strict=False):
        return ""


def test_eval_cmd_grader_exec_is_non_idempotent():
    """The eval_cmd grading command must be marked idempotent=False so the
    same-sandbox RPC retry loop never re-runs the grader after a stream break."""
    ev = _RecordingSandbox()

    async def run():
        return await swe_mod._run_eval_cmd(ev, "/workspace/repo", "pytest -x", timeout=30)

    asyncio.run(run())
    grading_execs = [(cmd, idem) for cmd, idem in ev.execs if "pytest -x" in cmd]
    assert grading_execs, "grading command was never exec'd"
    assert all(idem is False for _cmd, idem in grading_execs), "grader exec must be idempotent=False"


def test_f2p_grader_exec_is_non_idempotent():
    """The f2p_script pytest run must be idempotent=False."""
    ev = _RecordingSandbox()

    async def run():
        return await swe_mod._run_f2p_script(ev, "/workspace/repo", "import sys, pytest; sys.exit(0)", timeout=30)

    asyncio.run(run())
    grading_execs = [(cmd, idem) for cmd, idem in ev.execs if "python" in cmd and "__cagent_f2p__" in cmd]
    assert grading_execs, "f2p grading command was never exec'd"
    assert all(idem is False for _cmd, idem in grading_execs), "f2p grader exec must be idempotent=False"


# ---------------------------------------------------------------------------
# §7  real fresh-sandbox lifecycle (drives run_evaluation -> _run_evaluation_once
#      -> _grade_scaleswe; DOES NOT patch _run_evaluation_once)
#
# Every earlier retry test stubs _run_evaluation_once, so nothing exercised the
# real grader path that constructs and tears down E2BSandbox instances. These
# tests drive the real grader against a lifecycle-recording fake sandbox and
# prove the spec's core guarantee: each retry boots a DISTINCT fresh evaluator
# sandbox and BOTH are cleaned up (async-with __aexit__), with real protocol
# dispatch in _run_evaluation_once for both scaleswe and swebench.
# ---------------------------------------------------------------------------
class _LifecycleSandbox:
    """E2BSandbox stand-in that records construction + cleanup in a shared
    registry so the test can assert distinct instances and async-with teardown.

    The grading path exercised is the scaleswe ``eval_cmd`` one: empty
    ``diff_text`` short-circuits ``_apply_diff`` to True (no patch tooling), so
    the only execs are ``ensure_agent_user`` (id-agent probe) and the grading
    command itself. ``ensure_agent_user`` only calls ``sb.exec``, which this
    fake satisfies by returning ``(0, "", "")``.
    """

    def __init__(self, image, registry, *, fail_first_grading):
        self.image = image
        self.index = len(registry)  # 0 = attempt 1, 1 = attempt 2, ...
        self.sandbox_id = f"sb-{self.index}"
        self.entered = False
        self.exited = False
        self._fail_first_grading = fail_first_grading
        self.exec_cmds = []
        registry.append(self)

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *exc):
        self.exited = True
        return False

    async def exec(self, cmd, *, user="root", env=None, timeout=120, check=False, idempotent=True):
        self.exec_cmds.append(cmd)
        # The FIRST sandbox fails the grading command to force a fresh-sandbox
        # retry; the scanner's ensure_agent_user probe never matches "pytest -x".
        if "pytest -x" in cmd and self.index == 0 and self._fail_first_grading:
            raise _exc("ReadError", "stream broke mid-grade")
        return (0, "", "")

    async def write_file(self, path, content, *, user="root"):
        return None

    async def read_file(self, path, *, user="root", strict=False):
        return ""


def _scaleswe_md() -> dict:
    """scaleswe metadata routing _grade_scaleswe down its eval_cmd path."""
    return {
        "protocol": "scaleswe",
        "instance_id": "lc-1",
        "image": "img",
        "workdir": "/w",
        "grading": {"eval_cmd": "pytest -x"},
    }


def _patch_lifecycle_sandbox(monkeypatch, fail_first_grading):
    registry = []
    monkeypatch.setattr(
        swe_mod,
        "E2BSandbox",
        lambda image, **kw: _LifecycleSandbox(image, registry, fail_first_grading=fail_first_grading),
    )
    return registry


def test_retry_creates_two_distinct_sandboxes_and_cleans_up_both(monkeypatch):
    """Attempt 1 grades in a fresh sandbox whose grading exec fails retryably;
    attempt 2 gets a DISTINCT second sandbox. Both must be cleaned up."""
    registry = _patch_lifecycle_sandbox(monkeypatch, fail_first_grading=True)

    async def run():
        return await swe_mod.run_evaluation(_scaleswe_md(), diff_text="", timeout_sec=10, max_attempts=2)

    result = asyncio.run(run())
    assert result.reward == 1.0  # second attempt's eval_cmd exit 0 -> solved
    assert len(registry) == 2  # two attempts -> two sandboxes
    assert registry[0] is not registry[1]  # distinct instance objects
    assert registry[0].sandbox_id != registry[1].sandbox_id
    assert registry[0].entered and registry[1].entered  # both actually used
    assert registry[0].exited and registry[1].exited  # both __aexit__-cleaned-up


def test_single_attempt_creates_one_sandbox(monkeypatch):
    """A clean first grade needs one sandbox and no retry."""
    registry = _patch_lifecycle_sandbox(monkeypatch, fail_first_grading=False)

    async def run():
        return await swe_mod.run_evaluation(_scaleswe_md(), diff_text="", timeout_sec=10, max_attempts=2)

    result = asyncio.run(run())
    assert result.reward == 1.0
    assert len(registry) == 1  # success on attempt 1 -> no retry
    assert registry[0].entered
    assert registry[0].exited


def test_run_evaluation_once_dispatches_both_protocols(monkeypatch):
    """The real _run_evaluation_once dispatch (only the leaf graders stubbed)
    routes scaleswe -> _grade_scaleswe and swebench -> _grade_swebench. Stubbing
    the graders (not the dispatcher) is the exact seam the review flagged."""
    seen = {}

    async def fake_scaleswe(md, diff_text, timeout_sec):
        seen["scaleswe"] = seen.get("scaleswe", 0) + 1
        return swe_mod.EvalResult(1.0, True)

    async def fake_swebench(md, diff_text, timeout_sec):
        seen["swebench"] = seen.get("swebench", 0) + 1
        return swe_mod.EvalResult(1.0, True)

    monkeypatch.setattr(swe_mod, "_grade_scaleswe", fake_scaleswe)
    monkeypatch.setattr(swe_mod, "_grade_swebench", fake_swebench)

    async def run():
        await swe_mod.run_evaluation(
            {"protocol": "scaleswe", "instance_id": "s1"}, diff_text="", timeout_sec=10, max_attempts=1
        )
        await swe_mod.run_evaluation(
            {"protocol": "swebench", "instance_id": "s2"}, diff_text="", timeout_sec=10, max_attempts=1
        )

    asyncio.run(run())
    assert seen == {"scaleswe": 1, "swebench": 1}


# ---------------------------------------------------------------------------
# §8  read_file(strict=) contract
#
# The evaluator grader reads its result file with strict=True so a transient
# infra read failure re-raises (reaching run_evaluation's fresh-sandbox retry
# boundary) instead of being masked as "" -> {"tests": []} -> a false reward=0.
# The rollout/harness read paths keep strict=False (always-returns-a-string).
# These tests isolate read_file by stubbing _rpc_retry with the raising factory.
# ---------------------------------------------------------------------------


def _bare_sandbox() -> sandbox_mod.E2BSandbox:
    """An E2BSandbox instance without running __init__ (mirrors line ~122)."""
    return sandbox_mod.E2BSandbox.__new__(sandbox_mod.E2BSandbox)


def _read_with_rpc_raising(exc: BaseException, *, strict: bool) -> str:
    sb = _bare_sandbox()

    async def boom(_op, _factory):
        raise exc

    async def run():
        with patch.object(sb, "_rpc_retry", boom):
            return await sb.read_file("/tmp/result.json", user="agent", strict=strict)

    return asyncio.run(run())


def test_read_file_strict_reraises_fresh_retryable():
    """strict=True re-raises a fresh-retryable infra error (ReadError) so the
    retry boundary can recreate the sandbox."""
    with pytest.raises(Exception, match="connection reset"):
        _read_with_rpc_raising(_exc("ReadError", "connection reset"), strict=True)


def test_read_file_strict_reraises_generic_sandbox_exception():
    """A generic (transient gateway) SandboxException is fresh-retryable and so
    also re-raises under strict=True."""
    with pytest.raises(Exception, match="gateway timeout"):
        _read_with_rpc_raising(_exc("SandboxException", "gateway timeout"), strict=True)


def test_read_file_strict_swallows_permanent_sandbox_exception():
    """A permanent-marker SandboxException is NOT fresh-retryable; even under
    strict=True it degrades to "" rather than raising (a fresh sandbox cannot
    recover it, so retrying is pointless)."""
    assert _read_with_rpc_raising(_exc("SandboxException", "quota exceeded"), strict=True) == ""


def test_read_file_strict_swallows_missing_file_sandbox_exception():
    """'does not exist' means the file is genuinely absent on THIS sandbox --
    but is_fresh_sandbox_retryable treats a stopped/missing *sandbox* as
    retryable, so a fresh sandbox is booted. This documents that a missing file
    surfaced as a stopped/missing-sandbox SandboxException re-raises; a plain
    absent file (no infra exception -> empty read) still yields ""."""
    with pytest.raises(Exception, match="does not exist"):
        _read_with_rpc_raising(_exc("SandboxException", "sandbox does not exist"), strict=True)


def test_read_file_non_strict_never_raises():
    """Default strict=False preserves the always-returns-a-string contract for
    the rollout/harness read paths -- even a fresh-retryable infra error yields
    "" so those callers are unaffected by the evaluator opt-in."""
    assert _read_with_rpc_raising(_exc("ReadError", "connection reset"), strict=False) == ""
    assert _read_with_rpc_raising(_exc("SandboxException", "gateway timeout"), strict=False) == ""


def test_read_file_non_retryable_swallowed_under_strict():
    """A non-infra error (KeyError) is not fresh-retryable; strict=True still
    degrades it to "" (only fresh-retryable errors re-raise)."""
    assert _read_with_rpc_raising(KeyError("boom"), strict=True) == ""


# ---------------------------------------------------------------------------
# §9  SWE-bench strict-output integration (real exec_and_wait + real grader)
#
# The prior test gap the review flagged: every swebench retry test either
# stubbed _run_evaluation_once (skipping the real grader) or asserted the
# strict flag only on read_file in isolation. Nothing proved that
# _grade_swebench passes strict_output=True INTO the real exec_and_wait, that
# exec_and_wait forwards it as read_file(strict=True), and that a fresh-
# retryable OUTPUT read failure there propagates out of _grade_swebench ->
# _run_evaluation_once -> run_evaluation to boot a second fresh sandbox instead
# of returning a false reward=0 from an empty log.
#
# These tests drive the REAL call chain:
#   run_evaluation -> _run_evaluation_once -> _grade_swebench
#     -> real sandbox.exec_and_wait(..., strict_output=True)
#       -> fake sb.read_file(strict=True)
# exec_and_wait is NOT mocked; only the swebench-package leaves
# (_build_test_spec / _eval_report_from_log / _apply_model_patch) and the
# E2BSandbox class are stubbed so the test runs on the CPU env without swebench.
# ---------------------------------------------------------------------------

# Recover the marker path from exec_and_wait's setsid launcher and match its
# ``test -f X && cat X`` poll -- mirrors tests/test_agent/_fakes.py helpers so
# the real exec_and_wait handshake resolves against this fake.
import re  # noqa: E402

_POLL_RE = re.compile(r"test -f (\S+) && cat ")


def _done_path_from_launch(cmd: str) -> str | None:
    m = re.search(r"setsid bash (\S+)\.sh\b", cmd)
    return f"{m.group(1)}.done" if m else None


class _SwebenchLifecycleSandbox:
    """E2BSandbox stand-in that drives the REAL exec_and_wait detached-launch /
    poll-marker handshake and records the ``strict`` flag every ``read_file``
    receives, plus construction + cleanup, in a shared registry.

    The first sandbox (index 0) raises a fresh-retryable ``ReadError`` from the
    eval-output ``read_file`` *only when* ``strict=True`` -- so the test both
    proves strict propagation and forces exactly one fresh-sandbox retry. The
    second sandbox returns the captured eval log so grading proceeds.
    """

    def __init__(self, image, registry, *, out_payload, fail_first_strict_read):
        self.image = image
        self.index = len(registry)
        self.sandbox_id = f"sweb-sb-{self.index}"
        self.entered = False
        self.exited = False
        self._out_payload = out_payload
        self._fail_first_strict_read = fail_first_strict_read
        self.files: dict[str, str] = {}
        self.exec_cmds: list[str] = []
        self.read_strict_flags: list[tuple[str, bool]] = []  # (path, strict)
        registry.append(self)

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *exc):
        self.exited = True
        return False

    async def exec(self, cmd, *, user="root", env=None, timeout=120, check=False, idempotent=True):
        self.exec_cmds.append(cmd)
        # Detached launch: exec_and_wait fires the command fully detached, then
        # polls a done-marker. Drop the marker so the very next poll returns 0.
        if "setsid bash" in cmd:
            done = _done_path_from_launch(cmd)
            if done:
                self.files[done] = "0\n"
            return (0, "", "")
        # Marker poll: ``test -f {done} && cat {done}``.
        m = _POLL_RE.search(cmd)
        if m:
            path = m.group(1)
            if path in self.files:
                return (0, self.files[path], "")
            return (1, "", "")
        return (0, "", "")

    async def write_file(self, path, content, *, user="root"):
        self.files[path] = content

    async def read_file(self, path, *, user="root", strict=False):
        # Record what strict value exec_and_wait forwarded for the output read.
        self.read_strict_flags.append((path, strict))
        if self.index == 0 and self._fail_first_strict_read and strict:
            # Fresh-retryable infra read failure on the eval output read: with
            # strict=True the real E2BSandbox.read_file would re-raise this, so
            # the fake models that by raising directly. is_fresh_sandbox_retryable
            # classifies ReadError as retryable -> run_evaluation boots a fresh sb.
            raise _exc("ReadError", "output read stream broke")
        return self.files.get(path, self._out_payload)


def _swebench_md() -> dict:
    """swebench metadata routing run_evaluation -> _grade_swebench."""
    return {
        "protocol": "swebench",
        "instance_id": "sweb-inst-1",
        "image": "img",
        "workdir": "/testbed",
        "grading": {"sweb_instance": {"instance_id": "sweb-inst-1"}},
    }


def _patch_swebench_grader(monkeypatch, registry, *, resolved, out_payload, fail_first_strict_read):
    """Stub only the swebench-package leaves + E2BSandbox so the REAL
    _grade_swebench sandbox/exec_and_wait path runs on the CPU env."""
    monkeypatch.setattr(swe_mod, "_SWEBENCH_IMPORT_ERROR", None, raising=False)

    class _FakeTS:
        eval_script = "#!/bin/bash\necho grading\n"

    monkeypatch.setattr(swe_mod, "_build_test_spec", lambda inst: _FakeTS())

    async def _apply_ok(ev, workdir):
        return True

    monkeypatch.setattr(swe_mod, "_apply_model_patch", _apply_ok)

    def _report(ts, instance_id, diff_text, log):
        # Prove the grader received the second sandbox's real output log.
        assert log == out_payload, f"grader saw wrong log: {log!r}"
        return {instance_id: {"resolved": resolved, "patch_successfully_applied": True}}

    monkeypatch.setattr(swe_mod, "_eval_report_from_log", _report)

    monkeypatch.setattr(
        swe_mod,
        "E2BSandbox",
        lambda image, **kw: _SwebenchLifecycleSandbox(
            image, registry, out_payload=out_payload, fail_first_strict_read=fail_first_strict_read
        ),
    )


def test_swebench_strict_output_failure_triggers_fresh_sandbox_retry(monkeypatch):
    """REAL chain: run_evaluation -> _run_evaluation_once -> _grade_swebench ->
    real exec_and_wait(strict_output=True) -> sb.read_file(strict=True).

    The first evaluator sandbox's strict output read raises a fresh-retryable
    ReadError; that must propagate out of _grade_swebench and cause
    run_evaluation(max_attempts=2) to boot a DISTINCT second sandbox that reads
    the real eval log and grades resolved -> reward=1.0. Both sandboxes are
    cleaned up, and strict_output=True is proven to have reached read_file."""
    registry = []
    _patch_swebench_grader(
        monkeypatch,
        registry,
        resolved=True,
        out_payload="grading-log-output",
        fail_first_strict_read=True,
    )

    async def run():
        return await swe_mod.run_evaluation(_swebench_md(), diff_text="the-model-diff", timeout_sec=10, max_attempts=2)

    result = asyncio.run(run())

    # Retry recovered a real grade rather than a false empty-log reward=0.
    assert result.reward == 1.0
    # Two attempts -> two DISTINCT sandboxes, both entered and cleaned up.
    assert len(registry) == 2
    assert registry[0] is not registry[1]
    assert registry[0].sandbox_id != registry[1].sandbox_id
    assert registry[0].entered and registry[0].exited
    assert registry[1].entered and registry[1].exited
    # strict_output=True was forwarded as read_file(strict=True) on BOTH the
    # failing first read and the successful second read -- the whole point.
    first_strict = [s for _p, s in registry[0].read_strict_flags]
    second_strict = [s for _p, s in registry[1].read_strict_flags]
    assert first_strict and all(first_strict), "first sandbox never got strict=True output read"
    assert second_strict and all(second_strict), "second sandbox never got strict=True output read"


def test_swebench_strict_output_single_sandbox_when_read_succeeds(monkeypatch):
    """Control: when the strict output read does not fail, exactly ONE sandbox
    is created (no retry) and it still receives read_file(strict=True) -- so the
    retry in the sibling test is caused by the read failure, not by always
    double-booting."""
    registry = []
    _patch_swebench_grader(
        monkeypatch,
        registry,
        resolved=True,
        out_payload="grading-log-output",
        fail_first_strict_read=False,
    )

    async def run():
        return await swe_mod.run_evaluation(_swebench_md(), diff_text="the-model-diff", timeout_sec=10, max_attempts=2)

    result = asyncio.run(run())
    assert result.reward == 1.0
    assert len(registry) == 1  # clean first grade -> no retry
    assert registry[0].entered and registry[0].exited
    strict_flags = [s for _p, s in registry[0].read_strict_flags]
    assert strict_flags and all(strict_flags), "output read must be strict=True even on the happy path"
