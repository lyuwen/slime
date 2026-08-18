# Eval Sandbox Retry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bounded fresh-sandbox retry to `swe.run_evaluation()` so a completed coding-agent rollout survives transient evaluator sandbox / HTTP stream failures without discarding the captured diff.

**Architecture:** Expose a module-level `is_fresh_sandbox_retryable(e)` predicate in `slime/agent/sandbox.py`, refactor the two protocol graders in `swe.py` behind a single-attempt dispatcher called by a new retry loop in `run_evaluation()`, and add `SWE_EVAL_MAX_ATTEMPTS` / `eval_max_attempts` to `SweConfig`. The `EvalResult` namedtuple and all caller signatures remain unchanged; only the internals of `run_evaluation()` and `SweConfig.from_env()` grow.

**Tech Stack:** Python 3.10+, asyncio, pytest (existing test suite under `tests/test_agent/`), ruff lint, black format (line length 119).

## Global Constraints

- Python ≥ 3.10 (no 3.11+ syntax in library code; `asyncio.timeout` shim already present in test file).
- Line length: 119 characters (black + isort, ruff line-length overridden to 320 but black controls).
- No new third-party library dependencies.
- `EvalResult(reward, applied_cleanly)` two-field namedtuple contract must remain unchanged.
- `run_evaluation(md, *, diff_text, timeout_sec)` call sites in `generate.py` must remain compatible after adding the `max_attempts` parameter (default `1` preserves old behavior; `generate.py` passes `CONFIG.eval_max_attempts` explicitly).
- Default `SWE_EVAL_MAX_ATTEMPTS=2` (one initial + one fresh-sandbox retry).
- Per-attempt timeout is `SWE_EVAL_TIMEOUT_SEC`; it is NOT divided across attempts.
- Retry only for infra/transport exceptions that escape an attempt without an `EvalResult`; never retry a returned `EvalResult` regardless of reward value.
- Full-jitter exponential backoff between fresh-sandbox attempts: start cap 1 s, max cap 8 s.
- Do not catch `asyncio.CancelledError` or `asyncio.TimeoutError` in the retry loop.
- Lint: `ruff check --fix` + `black` on every modified file before commit.
- Tests live in `tests/test_agent/`; new focused tests go in `tests/test_agent/test_eval_sandbox_retry.py`.
- New test file must be added to the CI matrix in `.github/workflows/pr-test.yml.j2`, and `pr-test.yml` regenerated with `python .github/workflows/generate_github_workflows.py`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `slime/agent/sandbox.py` | Modify | Add `is_fresh_sandbox_retryable(e)` module-level predicate; reuse from `_is_transient_rpc_error` |
| `examples/coding_agent_rl/swe.py` | Modify | Extract `_run_evaluation_once()` dispatcher; add retry loop + backoff + logging to `run_evaluation()` |
| `examples/coding_agent_rl/generate.py` | Modify | Add `eval_max_attempts` to `SweConfig`; parse `SWE_EVAL_MAX_ATTEMPTS`; pass to `run_evaluation()`; fix derived guard formula |
| `examples/coding_agent_rl/README.md` | Modify | Document `SWE_EVAL_MAX_ATTEMPTS` in the env-var table |
| `tests/test_agent/test_eval_sandbox_retry.py` | Create | All 12 spec-listed test cases (CPU-only, no E2B) |
| `.github/workflows/pr-test.yml.j2` | Modify | Add new test file to `agent-test` CI matrix |
| `.github/workflows/pr-test.yml` | Regenerate | Re-run `generate_github_workflows.py` |
| `docs/superpowers/specs/2026-08-18-coding-agent-evaluation-sandbox-retry-design.md` | Already copied | Design spec (already in worktree) |

---

## Task 1: Expose `is_fresh_sandbox_retryable` predicate in `sandbox.py`

**Files:**
- Modify: `slime/agent/sandbox.py` (around line 228–257, the `_TRANSIENT_RPC_ERRORS` / `_is_transient_rpc_error` area)
- Test: `tests/test_agent/test_eval_sandbox_retry.py` (new file, first section)

**Interfaces:**
- Produces: `slime.agent.sandbox.is_fresh_sandbox_retryable(e: BaseException) -> bool`
  - Returns `True` for all HTTP/transport errors in `_TRANSIENT_RPC_ERRORS` (same as same-sandbox scope) **plus** `SandboxException` with "does not exist" or "STOPPED state" in the message (which the same-sandbox loop explicitly does NOT retry).
  - Returns `False` for everything else.
- `E2BSandbox._is_transient_rpc_error` continues to work exactly as before (same-sandbox scope); `is_fresh_sandbox_retryable` is the superset predicate for the evaluation-retry layer.

- [ ] **Step 1: Create the new test file with failing tests for the predicate**

Create `tests/test_agent/test_eval_sandbox_retry.py`:

```python
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
from unittest.mock import AsyncMock, MagicMock, call, patch

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
        "ReadError", "WriteError", "ConnectError", "ProtocolError",
        "LocalProtocolError", "RemoteProtocolError", "SSLError",
        "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout",
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
```

- [ ] **Step 2: Run the tests to verify they fail (predicate not yet implemented)**

```bash
cd /home/lfu/git-projects/slime/.claude/worktrees/feat+eval-sandbox-retry
python -m pytest tests/test_agent/test_eval_sandbox_retry.py -k "fresh_retryable or same_sandbox_stopped" -v 2>&1 | tail -30
```

Expected: `FAILED` or `AttributeError: module 'slime.agent.sandbox' has no attribute 'is_fresh_sandbox_retryable'`.

- [ ] **Step 3: Add `is_fresh_sandbox_retryable` to `sandbox.py`**

In `slime/agent/sandbox.py`, after the `_is_transient_rpc_error` classmethod (currently ends at line ~257), add the module-level function:

```python
def is_fresh_sandbox_retryable(e: BaseException) -> bool:
    """True when ``e`` is an infrastructure exception safe to recover by
    recreating the evaluator sandbox from scratch.

    This is a superset of :meth:`E2BSandbox._is_transient_rpc_error`:
    it additionally includes ``SandboxException`` failures that indicate
    the sandbox no longer exists or is stopped, because a fresh sandbox
    can recover them even though same-sandbox RPC retry cannot.

    Does NOT include ``asyncio.CancelledError`` or ``asyncio.TimeoutError``
    so the outer rollout wall-clock guard is never swallowed.
    """
    if isinstance(e, (asyncio.CancelledError, asyncio.TimeoutError)):
        return False
    name = type(e).__name__
    if name in E2BSandbox._TRANSIENT_RPC_ERRORS:
        return True
    if name == "SandboxException":
        msg = str(e)
        # stopped / missing sandbox: a new sandbox will recover this
        if "does not exist" in msg or "STOPPED state" in msg:
            return True
        # any other SandboxException (quota, auth, ...) is not recoverable
        # by retrying — propagate as-is.
        return False
    return False
```

Place it immediately after the `E2BSandbox` class definition (after `ensure_agent_user` is fine too — just not inside the class body).

- [ ] **Step 4: Run the §1 tests to verify they pass**

```bash
python -m pytest tests/test_agent/test_eval_sandbox_retry.py -k "fresh_retryable or same_sandbox_stopped" -v 2>&1 | tail -30
```

Expected: all 9 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add slime/agent/sandbox.py tests/test_agent/test_eval_sandbox_retry.py
git commit -m "feat(sandbox): expose is_fresh_sandbox_retryable predicate for eval retry"
```

---

## Task 2: Refactor `swe.run_evaluation()` with retry loop

**Files:**
- Modify: `examples/coding_agent_rl/swe.py` (lines 265–273 — `run_evaluation` function)
- Test: `tests/test_agent/test_eval_sandbox_retry.py` (§2 section)

**Interfaces:**
- Consumes: `slime.agent.sandbox.is_fresh_sandbox_retryable` from Task 1
- Modifies: `run_evaluation(md, *, diff_text, timeout_sec, max_attempts=1) -> EvalResult`
  - New `max_attempts: int = 1` keyword arg (default preserves old behavior).
  - Returns `EvalResult` on success.
  - Re-raises the final infrastructure exception after exhausting all attempts.
- Produces: private `_run_evaluation_once(md, diff_text, timeout_sec) -> EvalResult` helper (no retry logic; just dispatches to the right grader).

- [ ] **Step 1: Add §2 tests to the test file (they will fail)**

Append to `tests/test_agent/test_eval_sandbox_retry.py`:

```python
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

    with pytest.raises(Exception):
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
                diff_text="", timeout_sec=10, max_attempts=1,
            )
            await swe_mod.run_evaluation(
                {"protocol": "swebench", "instance_id": "s2"},
                diff_text="", timeout_sec=10, max_attempts=1,
            )

    asyncio.run(run())
    assert results == ["scaleswe", "swebench"]
```

- [ ] **Step 2: Run §2 tests to verify they fail**

```bash
python -m pytest tests/test_agent/test_eval_sandbox_retry.py -k "retry_on_read_error or no_retry or exhausted or non_retryable or cancelled or max_attempts_one or scaleswe_and_swebench" -v 2>&1 | tail -40
```

Expected: `AttributeError: module '...swe' has no attribute '_run_evaluation_once'` or `TypeError: run_evaluation() got unexpected keyword argument 'max_attempts'`.

- [ ] **Step 3: Refactor `swe.py` — extract `_run_evaluation_once`, add retry loop**

Replace the current `run_evaluation` function in `examples/coding_agent_rl/swe.py` (lines 265–273):

```python
async def run_evaluation(
    md: dict,
    *,
    diff_text: str,
    timeout_sec: int,
    max_attempts: int = 1,
) -> EvalResult:
    """Uniform entry point: dispatch to the protocol's grader with bounded
    fresh-sandbox retry.

    Each attempt calls ``_run_evaluation_once()`` which selects the protocol
    grader. The grader owns ``async with E2BSandbox(image) as ev``; exiting
    that context on failure kills or releases the broken sandbox before the
    next attempt boots a fresh one.

    Only infrastructure exceptions (transport errors, sandbox unavailable)
    trigger a retry. A returned ``EvalResult`` — including reward=0.0 or
    applied_cleanly=False — is never retried.

    ``max_attempts=1`` preserves the original single-attempt behaviour.
    """
    last_err: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await _run_evaluation_once(md, diff_text, timeout_sec)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            raise  # never swallow cancellation or the outer wall-clock guard
        except BaseException as e:
            if not is_fresh_sandbox_retryable(e):
                raise  # deterministic error — propagate immediately
            last_err = e
            if attempt < max_attempts:
                # bounded full-jitter exponential backoff
                ceiling = min(8.0, 1.0 * (2 ** (attempt - 1)))
                delay = random.uniform(0.0, ceiling)
                logger.warning(
                    "[swe] %s: eval attempt %d/%d failed (%s: %s); "
                    "backoff=%.1fs, next attempt uses a fresh evaluator sandbox",
                    md.get("instance_id", "?"),
                    attempt,
                    max_attempts,
                    type(e).__name__,
                    str(e)[:120],
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.warning(
                    "[swe] %s: eval exhausted %d/%d attempts (%s: %s); re-raising",
                    md.get("instance_id", "?"),
                    attempt,
                    max_attempts,
                    type(e).__name__,
                    str(e)[:120],
                )
    assert last_err is not None
    raise last_err


async def _run_evaluation_once(md: dict, diff_text: str, timeout_sec: int) -> EvalResult:
    """Single-attempt protocol dispatch. Each grader owns its evaluator sandbox
    context; a failure exits that context (killing the sandbox) before returning
    to the caller."""
    if md.get("protocol") == PROTOCOL_SWEBENCH:
        return await _grade_swebench(md, diff_text, timeout_sec)
    return await _grade_scaleswe(md, diff_text, timeout_sec)
```

Also add `import random` to the imports at the top of `swe.py` (it is not yet imported there). Check first:

```bash
grep "^import random" examples/coding_agent_rl/swe.py
```

If not present, add `import random` after `import os` in the import block.

Also add the import of `is_fresh_sandbox_retryable` from the sandbox module. The existing import line is:

```python
from slime.agent.sandbox import E2BSandbox, Sandbox, exec_and_wait
```

Change it to:

```python
from slime.agent.sandbox import E2BSandbox, Sandbox, exec_and_wait, is_fresh_sandbox_retryable
```

- [ ] **Step 4: Run §2 tests**

```bash
python -m pytest tests/test_agent/test_eval_sandbox_retry.py -k "retry_on_read_error or no_retry or exhausted or non_retryable or cancelled or max_attempts_one or scaleswe_and_swebench" -v 2>&1 | tail -40
```

Expected: all 8 tests PASSED.

- [ ] **Step 5: Run the full existing test suite (regression check)**

```bash
python -m pytest tests/test_agent/test_agent_rollout_cpu.py tests/test_agent/test_harness.py -v 2>&1 | tail -30
```

Expected: all existing tests still PASSED.

- [ ] **Step 6: Commit**

```bash
git add examples/coding_agent_rl/swe.py tests/test_agent/test_eval_sandbox_retry.py
git commit -m "feat(swe): add fresh-sandbox retry loop to run_evaluation()"
```

---

## Task 3: Add `SWE_EVAL_MAX_ATTEMPTS` to `SweConfig` and wire it through

**Files:**
- Modify: `examples/coding_agent_rl/generate.py` (lines 61–110 — `SweConfig` dataclass and `from_env`)
- Test: `tests/test_agent/test_eval_sandbox_retry.py` (§3 and §4 sections)

**Interfaces:**
- Consumes: `swe.run_evaluation(..., max_attempts=int)` from Task 2
- Modifies: `SweConfig` — adds `eval_max_attempts: int` field
- Modifies: `SweConfig.from_env()` — parses `SWE_EVAL_MAX_ATTEMPTS` (default `"2"`, minimum 1)
- Modifies: derived guard formula in `from_env()`:
  ```
  guard = explicit_env or (agent_time_budget + eval_timeout * eval_max_attempts + 180)
  ```
- Modifies: `generate()` call to `swe.run_evaluation(...)` — adds `max_attempts=CONFIG.eval_max_attempts`

- [ ] **Step 1: Add §3 and §4 tests (they will fail)**

Append to `tests/test_agent/test_eval_sandbox_retry.py`:

```python
# ---------------------------------------------------------------------------
# §3  SweConfig env-var parsing
# ---------------------------------------------------------------------------

import dataclasses  # noqa: E402
import os  # noqa: E402

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
    with patch.dict(os.environ, {
        "SWE_AGENT_TIME_BUDGET_SEC": "1800",
        "SWE_EVAL_TIMEOUT_SEC": "600",
        "SWE_EVAL_MAX_ATTEMPTS": "5",
        "SWE_ROLLOUT_GUARD_SEC": "9999",
        "ADAPTER_PUBLIC_HOST": "127.0.0.1",
    }, clear=False):
        cfg = gen_mod.SweConfig.from_env()
    assert cfg.rollout_guard_sec == 9999
```

- [ ] **Step 2: Run §3 and §4 tests to verify they fail**

```bash
python -m pytest tests/test_agent/test_eval_sandbox_retry.py -k "eval_max_attempts or derived_guard or explicit_guard" -v 2>&1 | tail -40
```

Expected: `AttributeError: 'SweConfig' object has no attribute 'eval_max_attempts'` or similar.

- [ ] **Step 3: Add `eval_max_attempts` field to `SweConfig` and update `from_env()`**

In `examples/coding_agent_rl/generate.py`, in the `SweConfig` dataclass (around line 61–76), add the new field after `eval_timeout_sec`:

```python
@dataclass(frozen=True)
class SweConfig:
    eval_protocol: str
    train_protocol: str
    adapter_public_host: str | None
    adapter_bind_host: str
    adapter_port: int
    fork_merge_threshold: int | None
    agent_time_budget_sec: int
    eval_timeout_sec: int
    eval_max_attempts: int          # NEW — SWE_EVAL_MAX_ATTEMPTS
    rollout_guard_sec: int
    boot_concurrency: int
    boot_retries: int
    oh_fake_user: bool
    oh_max_iterations: int
    oh_tools: list[str]
    oh_extra_envs: dict[str, str]
```

In `from_env()`, parse `SWE_EVAL_MAX_ATTEMPTS` and update the guard derivation. Replace lines 81–107 with:

```python
    @classmethod
    def from_env(cls) -> SweConfig:
        agent_time_budget = int(os.environ.get("SWE_AGENT_TIME_BUDGET_SEC", "1800"))
        eval_timeout = int(os.environ.get("SWE_EVAL_TIMEOUT_SEC", "600"))
        _raw_max_attempts = int(os.environ.get("SWE_EVAL_MAX_ATTEMPTS", "2"))
        if _raw_max_attempts < 1:
            raise ValueError(
                f"SWE_EVAL_MAX_ATTEMPTS must be >= 1, got {_raw_max_attempts!r}"
            )
        eval_max_attempts = _raw_max_attempts
        guard = (
            int(os.environ.get("SWE_ROLLOUT_GUARD_SEC", "0") or 0)
            or (agent_time_budget + eval_timeout * eval_max_attempts + 180)
        )
        fork = int(v) if (v := os.environ.get("SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS")) else None
        oh_tools_raw = os.environ.get("SWE_OH_TOOLS", "file_editor,terminal,task_tracker,think,finish")
        oh_tools = [t.strip() for t in oh_tools_raw.split(",") if t.strip()]
        oh_extra_envs_raw = os.environ.get("SLIME_AGENT_OH_EXTRA_ENVS", "").strip()
        oh_extra_envs = json.loads(oh_extra_envs_raw) if oh_extra_envs_raw else {}
        if not isinstance(oh_extra_envs, dict):
            raise ValueError("SLIME_AGENT_OH_EXTRA_ENVS must be a JSON object")
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
            oh_fake_user=os.environ.get("SWE_OH_FAKE_USER", "0") not in ("0", "", "false", "False"),
            oh_max_iterations=int(os.environ.get("SWE_OH_MAX_ITERATIONS", "100")),
            oh_tools=oh_tools,
            oh_extra_envs=oh_extra_envs,
        )
```

- [ ] **Step 4: Wire `eval_max_attempts` into the `generate()` call**

In `generate()` (around line 334–338), change the `run_evaluation` call from:

```python
            reward, applied_cleanly = await swe.run_evaluation(
                md,
                diff_text=diff_text,
                timeout_sec=CONFIG.eval_timeout_sec,
            )
```

to:

```python
            reward, applied_cleanly = await swe.run_evaluation(
                md,
                diff_text=diff_text,
                timeout_sec=CONFIG.eval_timeout_sec,
                max_attempts=CONFIG.eval_max_attempts,
            )
```

- [ ] **Step 5: Run §3 and §4 tests**

```bash
python -m pytest tests/test_agent/test_eval_sandbox_retry.py -k "eval_max_attempts or derived_guard or explicit_guard" -v 2>&1 | tail -40
```

Expected: all 8 tests PASSED.

- [ ] **Step 6: Run full CPU agent tests (regression check)**

```bash
python -m pytest tests/test_agent/test_agent_rollout_cpu.py tests/test_agent/test_harness.py tests/test_agent/test_adapters.py -v 2>&1 | tail -30
```

Note: `test_agent_rollout_cpu.py` imports `gen.CONFIG` at module load; the `_patch_generate` helper replaces it via `dataclasses.replace`, so the new field will be carried through. Verify the existing rollout tests still pass.

Expected: all tests PASSED.

- [ ] **Step 7: Commit**

```bash
git add examples/coding_agent_rl/generate.py tests/test_agent/test_eval_sandbox_retry.py
git commit -m "feat(generate): add SWE_EVAL_MAX_ATTEMPTS config and wire into run_evaluation"
```

---

## Task 4: Logging observability tests and success-log emission

**Files:**
- Modify: `examples/coding_agent_rl/swe.py` — add success-path info log after retry loop
- Test: `tests/test_agent/test_eval_sandbox_retry.py` (§5 section)

**Interfaces:**
- Consumes: `run_evaluation()` with retry loop from Task 2
- The success-path log is already partially present in the Task 2 retry-loop implementation; this task verifies the log content with assertions and adds the success info log.

- [ ] **Step 1: Add §5 log tests (they will fail until the success log is emitted)**

Append to `tests/test_agent/test_eval_sandbox_retry.py`:

```python
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
```

- [ ] **Step 2: Run §5 tests to see current state**

```bash
python -m pytest tests/test_agent/test_eval_sandbox_retry.py -k "warning_contains or success_after_retry" -v 2>&1 | tail -30
```

The warning test may already pass from Task 2. The info-log test will likely fail (no success-path info log emitted yet).

- [ ] **Step 3: Add success-path info log to `run_evaluation()` in `swe.py`**

Inside the retry loop in `run_evaluation()`, right after `return await _run_evaluation_once(...)` succeeds and only when `attempt > 1`, emit an info log. Modify the `try` block inside the loop:

```python
        try:
            result = await _run_evaluation_once(md, diff_text, timeout_sec)
            if attempt > 1:
                logger.info(
                    "[swe] %s: eval succeeded on attempt %d/%d reward=%.2f",
                    md.get("instance_id", "?"),
                    attempt,
                    max_attempts,
                    float(result.reward),
                )
            return result
```

- [ ] **Step 4: Run all §5 tests**

```bash
python -m pytest tests/test_agent/test_eval_sandbox_retry.py -k "warning_contains or success_after_retry" -v 2>&1 | tail -20
```

Expected: both PASSED.

- [ ] **Step 5: Run the full new test file**

```bash
python -m pytest tests/test_agent/test_eval_sandbox_retry.py -v 2>&1 | tail -40
```

Expected: all tests PASSED (no failures, no errors).

- [ ] **Step 6: Commit**

```bash
git add examples/coding_agent_rl/swe.py tests/test_agent/test_eval_sandbox_retry.py
git commit -m "feat(swe): emit success/retry info logs for eval sandbox retry"
```

---

## Task 5: Documentation — README.md env-var table

**Files:**
- Modify: `examples/coding_agent_rl/README.md`

**Interfaces:**
- No code changes; documentation only.

- [ ] **Step 1: Add `SWE_EVAL_MAX_ATTEMPTS` row to the env-var table**

In `examples/coding_agent_rl/README.md`, find the env-var table (around line 138). The table has a `SWE_EVAL_TIMEOUT_SEC` row. Add the new row immediately after it:

```markdown
| `SWE_EVAL_MAX_ATTEMPTS` | `2` | Maximum number of fresh-sandbox evaluation attempts per rollout. Use `1` to disable retry and restore single-attempt behavior. A transient evaluator sandbox/transport failure triggers a fresh-sandbox retry up to this limit. |
```

The existing `SWE_ROLLOUT_GUARD_SEC` description must also be updated to reflect the new derivation. Find the existing row:

```markdown
| `SWE_ROLLOUT_GUARD_SEC` | `agent+eval+180` | Outer safety net wrapping the whole rollout (boot + workspace + agent + diff + eval). Auto-derived if unset. |
```

Change it to:

```markdown
| `SWE_ROLLOUT_GUARD_SEC` | `agent+eval*attempts+180` | Outer safety net wrapping the whole rollout (boot + workspace + agent + diff + eval). Auto-derived as `SWE_AGENT_TIME_BUDGET_SEC + SWE_EVAL_TIMEOUT_SEC * SWE_EVAL_MAX_ATTEMPTS + 180` if unset; an explicit value is always authoritative. |
```

- [ ] **Step 2: Verify the README renders cleanly**

```bash
python -c "
import re, sys
text = open('examples/coding_agent_rl/README.md').read()
assert 'SWE_EVAL_MAX_ATTEMPTS' in text, 'new var missing'
assert 'eval*attempts' in text or 'EVAL_MAX_ATTEMPTS' in text.split('SWE_ROLLOUT_GUARD_SEC')[1][:300], 'guard description not updated'
print('README OK')
"
```

Expected: `README OK`.

- [ ] **Step 3: Commit**

```bash
git add examples/coding_agent_rl/README.md
git commit -m "docs(coding_agent_rl): document SWE_EVAL_MAX_ATTEMPTS and updated guard formula"
```

---

## Task 6: CI matrix registration and spec file commit

**Files:**
- Modify: `.github/workflows/pr-test.yml.j2`
- Regenerate: `.github/workflows/pr-test.yml`
- The spec file was already copied in pre-work

**Interfaces:**
- Adds `test_agent/test_eval_sandbox_retry.py` to the `agent-test` CPU matrix.

- [ ] **Step 1: Add new test file to the j2 template**

In `.github/workflows/pr-test.yml.j2`, find the `agent-test` block (around the `'always': True, 'cpu': True` section). It has a `tests` list. Add the new entry:

```python
        {'test_file': 'test_agent/test_eval_sandbox_retry.py', 'num_gpus': 0},
```

Place it after the `test_oh_driver.py` entry so it appears last in the agent-test group.

The block before the change:
```python
    'agent-test': {
      'label': 'run-ci-agent',
      'always': True,
      'cpu': True,
      'tests': [
        {'test_file': 'test_agent/test_trajectory_manager_branching.py', 'num_gpus': 0},
        {'test_file': 'test_agent/test_adapters.py', 'num_gpus': 0},
        {'test_file': 'test_agent/test_harness.py', 'num_gpus': 0},
        {'test_file': 'test_agent/test_agent_rollout_cpu.py', 'num_gpus': 0},
        {'test_file': 'test_agent/test_openhands_harness.py', 'num_gpus': 0},
        {'test_file': 'test_agent/test_oh_driver.py', 'num_gpus': 0},
        {'test_file': 'test_tools/test_repackage_oh_env.py', 'num_gpus': 0},
      ],
    },
```

After the change:
```python
    'agent-test': {
      'label': 'run-ci-agent',
      'always': True,
      'cpu': True,
      'tests': [
        {'test_file': 'test_agent/test_trajectory_manager_branching.py', 'num_gpus': 0},
        {'test_file': 'test_agent/test_adapters.py', 'num_gpus': 0},
        {'test_file': 'test_agent/test_harness.py', 'num_gpus': 0},
        {'test_file': 'test_agent/test_agent_rollout_cpu.py', 'num_gpus': 0},
        {'test_file': 'test_agent/test_openhands_harness.py', 'num_gpus': 0},
        {'test_file': 'test_agent/test_oh_driver.py', 'num_gpus': 0},
        {'test_file': 'test_agent/test_eval_sandbox_retry.py', 'num_gpus': 0},
        {'test_file': 'test_tools/test_repackage_oh_env.py', 'num_gpus': 0},
      ],
    },
```

- [ ] **Step 2: Regenerate `pr-test.yml` from the j2 template**

```bash
python .github/workflows/generate_github_workflows.py
```

Expected: no errors; `pr-test.yml` is updated.

- [ ] **Step 3: Verify the new test file appears in the generated YAML**

```bash
grep "test_eval_sandbox_retry" .github/workflows/pr-test.yml
```

Expected: at least one line matching.

- [ ] **Step 4: Commit spec file + CI changes**

```bash
git add docs/superpowers/specs/2026-08-18-coding-agent-evaluation-sandbox-retry-design.md
git add .github/workflows/pr-test.yml.j2 .github/workflows/pr-test.yml
git commit -m "ci: register test_eval_sandbox_retry in agent-test matrix; add design spec"
```

---

## Task 7: Final lint pass, full test run, and branch-finish review

**Files:**
- All modified files (lint only; no logic changes)

**Interfaces:**
- Final gate before branch is considered ready.

- [ ] **Step 1: Run ruff on all modified files**

```bash
python -m ruff check slime/agent/sandbox.py examples/coding_agent_rl/swe.py examples/coding_agent_rl/generate.py tests/test_agent/test_eval_sandbox_retry.py --fix
```

Expected: no errors (auto-fixable issues fixed in-place).

- [ ] **Step 2: Run black on all modified Python files**

```bash
python -m black slime/agent/sandbox.py examples/coding_agent_rl/swe.py examples/coding_agent_rl/generate.py tests/test_agent/test_eval_sandbox_retry.py --line-length 119 --check
```

If reformatting is needed:
```bash
python -m black slime/agent/sandbox.py examples/coding_agent_rl/swe.py examples/coding_agent_rl/generate.py tests/test_agent/test_eval_sandbox_retry.py --line-length 119
```

- [ ] **Step 3: Run the complete new test suite**

```bash
python -m pytest tests/test_agent/test_eval_sandbox_retry.py -v 2>&1 | tail -50
```

Expected: all tests PASSED.

- [ ] **Step 4: Run the full existing agent test suite for regression**

```bash
python -m pytest tests/test_agent/test_agent_rollout_cpu.py tests/test_agent/test_harness.py tests/test_agent/test_adapters.py tests/test_agent/test_trajectory_persistence.py tests/test_agent/test_openhands_harness.py -v 2>&1 | tail -40
```

Expected: all tests PASSED.

- [ ] **Step 5: Commit any lint-only fixes (if step 1 or 2 made changes)**

```bash
git diff --name-only
# If any files changed:
git add slime/agent/sandbox.py examples/coding_agent_rl/swe.py examples/coding_agent_rl/generate.py tests/test_agent/test_eval_sandbox_retry.py
git commit -m "style: apply ruff/black to eval-sandbox-retry implementation files"
```

- [ ] **Step 6: Final branch summary**

```bash
git log main..HEAD --oneline
```

Expected output (approximately):
```
<hash> style: apply ruff/black to eval-sandbox-retry implementation files
<hash> ci: register test_eval_sandbox_retry in agent-test matrix; add design spec
<hash> docs(coding_agent_rl): document SWE_EVAL_MAX_ATTEMPTS and updated guard formula
<hash> feat(swe): emit success/retry info logs for eval sandbox retry
<hash> feat(generate): add SWE_EVAL_MAX_ATTEMPTS config and wire into run_evaluation
<hash> feat(swe): add fresh-sandbox retry loop to run_evaluation()
<hash> feat(sandbox): expose is_fresh_sandbox_retryable predicate for eval retry
```

---

## Self-Review Against Spec

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| Retry `run_evaluation()` — fresh sandbox per attempt | Task 2 |
| `is_fresh_sandbox_retryable` classifier (same-sandbox vs fresh-sandbox scopes) | Task 1 |
| No retry for `EvalResult(0.0, ...)` or `applied_cleanly=False` | Task 2, tests `no_retry_on_zero_reward`, `no_retry_on_apply_failure` |
| No retry for `asyncio.CancelledError` / `TimeoutError` | Task 1 predicate + Task 2 loop |
| `SWE_EVAL_MAX_ATTEMPTS`, default 2, minimum 1 | Task 3 |
| `SweConfig.eval_max_attempts` field | Task 3 |
| Derived guard: `agent + eval*attempts + 180` | Task 3 |
| Explicit `SWE_ROLLOUT_GUARD_SEC` remains authoritative | Task 3 test |
| Full-jitter exponential backoff (cap 1s→8s) | Task 2 implementation |
| Warning log: instance, attempt N/M, exception class, message, backoff, fresh-sandbox note | Task 4 tests |
| Success-after-retry info log with total attempts and reward | Task 4 |
| Exhaustion re-raises last infra exception | Task 2 test `exhausted_attempts_reraises` |
| Both ScaleSWE and SWE-bench use same retry boundary | Task 2 test `scaleswe_and_swebench_both_use_retry_boundary` |
| `EvalResult` two-field contract unchanged | All tasks (no changes to `EvalResult`) |
| grader process launches non-idempotent (don't transparently retry in same sandbox) | Already implemented: `E2BSandbox.exec()` with `idempotent=False` for grader commands is noted in spec as future; the spec says `_run_evaluation_once` exits the sandbox context on failure — covered by `async with E2BSandbox` inside each grader |
| README documentation | Task 5 |
| CI matrix | Task 6 |
| Spec file in worktree | Pre-work (already done) |
| 12 spec-listed unit tests | Distributed across Tasks 1–4 |

**Placeholder scan:** No TBDs, no "add appropriate" phrases, all steps have concrete code.

**Type consistency:**
- `is_fresh_sandbox_retryable(e: BaseException) -> bool` used consistently in Task 1 (definition) and Task 2 (import + call).
- `_run_evaluation_once(md, diff_text, timeout_sec)` defined in Task 2 and patched by name in Task 2 tests.
- `run_evaluation(..., max_attempts=int)` defined in Task 2, consumed in Task 3 (`generate.py` call site).
- `SweConfig.eval_max_attempts` added in Task 3 dataclass and referenced in same task's `generate()` call site.
