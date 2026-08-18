"""CPU-only tests for the evaluation sandbox retry mechanism.

Covers:
  §1  is_fresh_sandbox_retryable() exception classification
  §2  run_evaluation() retry loop orchestration (swe.py)
  §3  SweConfig env-var parsing (generate.py)
  §4  Derived rollout guard formula
  §5  Log messages emitted during retry
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


def test_not_fresh_retryable_generic_sandbox_exception():
    """A SandboxException without stopped/missing text is NOT fresh-retryable
    (unknown error — preserve existing behaviour)."""
    e = _exc("SandboxException", "quota exceeded")
    # quota exceeded is NOT a fresh-retryable infra failure
    # (sandbox still exists, retrying won't help)
    assert not sandbox_mod.is_fresh_sandbox_retryable(e)


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


def test_max_attempts_one_preserves_single_attempt_behavior():
    """max_attempts=1 (old default) means one call, exception propagates."""
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

    async def read_file(self, path, *, user="root"):
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
