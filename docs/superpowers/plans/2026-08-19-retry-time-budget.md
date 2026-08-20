# Retry Time-Budget Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent the coding-agent rollout retry loop from starting a fresh attempt when too little of the agent time budget remains, so late failures fail fast instead of restarting into a doomed run.

**Architecture:** Add a `retry_min_budget_sec` field to `SweConfig` (env `SWE_AGENT_RETRY_MIN_BUDGET_SEC`, default `0.5 * agent_time_budget_sec`). In the existing retry `except` block, after the retryable/attempt-count check, re-raise the original error when `agent_time_budget_sec - (time.time() - t0) < retry_min_budget_sec`. The gate applies to all retry policies.

**Tech Stack:** Python, pytest, `unittest.mock`, frozen `@dataclass`.

## Global Constraints

- Line length 119 (black, isort black profile); `ruff` E/F/B/UP; E402/E501 ignored.
- The gate re-raises the original error — no new exception type, no new ABORTED reason.
- `retry_min_budget_sec` is measured against `agent_time_budget_sec`, not `rollout_guard_sec`.
- Default floor: `0.5 * agent_time_budget_sec`; explicit env override wins; value must be `>= 0`.
- Tests follow the repo convention: `pytest.main([__file__])` under `if __name__ == "__main__"`.

---

### Task 1: Add `retry_min_budget_sec` config field

**Files:**
- Modify: `examples/coding_agent_rl/generate.py` (`SweConfig` dataclass ~line 61-80; `from_env` ~line 82-128)
- Test: `examples/coding_agent_rl/test_retry.py`

**Interfaces:**
- Produces: `SweConfig.retry_min_budget_sec: float`, parsed from `SWE_AGENT_RETRY_MIN_BUDGET_SEC`, defaulting to `0.5 * agent_time_budget_sec`.

- [ ] **Step 1: Write the failing tests**

Add to `examples/coding_agent_rl/test_retry.py`:

```python
def test_retry_min_budget_defaults_to_half_agent_budget(monkeypatch):
    monkeypatch.setenv("SWE_AGENT_TIME_BUDGET_SEC", "1800")
    monkeypatch.delenv("SWE_AGENT_RETRY_MIN_BUDGET_SEC", raising=False)
    cfg = SweConfig.from_env()
    assert cfg.retry_min_budget_sec == 900.0


def test_retry_min_budget_explicit_override(monkeypatch):
    monkeypatch.setenv("SWE_AGENT_TIME_BUDGET_SEC", "1800")
    monkeypatch.setenv("SWE_AGENT_RETRY_MIN_BUDGET_SEC", "300")
    cfg = SweConfig.from_env()
    assert cfg.retry_min_budget_sec == 300.0


def test_retry_min_budget_zero_allowed(monkeypatch):
    monkeypatch.setenv("SWE_AGENT_RETRY_MIN_BUDGET_SEC", "0")
    cfg = SweConfig.from_env()
    assert cfg.retry_min_budget_sec == 0.0


def test_retry_min_budget_negative_rejected(monkeypatch):
    monkeypatch.setenv("SWE_AGENT_RETRY_MIN_BUDGET_SEC", "-1")
    with pytest.raises(ValueError):
        SweConfig.from_env()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest examples/coding_agent_rl/test_retry.py -k retry_min_budget -v`
Expected: FAIL — `SweConfig` has no field `retry_min_budget_sec` / unexpected keyword argument.

- [ ] **Step 3: Add the field and parsing**

In the `SweConfig` dataclass, add the field after `retry_policy: str`:

```python
    retry_policy: str
    retry_min_budget_sec: float
```

In `from_env`, after the `retry_policy` validation block (line ~108) and before `return cls(`:

```python
        _min_budget_env = os.environ.get("SWE_AGENT_RETRY_MIN_BUDGET_SEC")
        retry_min_budget_sec = float(_min_budget_env) if _min_budget_env else 0.5 * agent_time_budget
        if retry_min_budget_sec < 0:
            raise ValueError("SWE_AGENT_RETRY_MIN_BUDGET_SEC must be >= 0")
```

In the `return cls(` block, add after `retry_policy=retry_policy,`:

```python
            retry_min_budget_sec=retry_min_budget_sec,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest examples/coding_agent_rl/test_retry.py -k retry_min_budget -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add examples/coding_agent_rl/generate.py examples/coding_agent_rl/test_retry.py
git commit -m "feat(coding_agent_rl): add SWE_AGENT_RETRY_MIN_BUDGET_SEC config"
```

---

### Task 2: Gate retries on remaining agent budget

**Files:**
- Modify: `examples/coding_agent_rl/generate.py` (retry `except` block, lines 409-430)
- Test: `examples/coding_agent_rl/test_retry.py`

**Interfaces:**
- Consumes: `CONFIG.retry_min_budget_sec`, `CONFIG.agent_time_budget_sec`, `t0` (set at `generate.py:367`), `_is_retryable`, `CONFIG.rollout_retries`.

- [ ] **Step 1: Write the failing test**

This test drives the retry loop with a retryable error and patches `time.time` so elapsed exceeds the budget-minus-floor, expecting the original error to propagate (no retry, single boot attempt). Model it on the existing `always-fail` test at the tail of `test_retry.py` — reuse its fixtures (`_make_sample_mock`, `fake_e2b_cls`, `fast_sleep`). Add:

```python
@pytest.mark.asyncio
async def test_budget_gate_blocks_retry_when_budget_low(monkeypatch):
    from examples.coding_agent_rl import generate as gen_mod

    # Fresh sandbox that always raises a retryable error on enter.
    fake_sb = AsyncMock()
    fake_sb.__aenter__ = AsyncMock(side_effect=RuntimeError("e2b connection lost"))
    fake_sb.__aexit__ = AsyncMock(return_value=False)
    fake_e2b_cls = MagicMock(return_value=fake_sb)

    state = _make_state_mock()  # same helper the always-fail test uses
    mock_harness_cls = MagicMock(return_value=AsyncMock())
    mock_swe = MagicMock()
    mock_swe.get_metadata.return_value = _make_sample_mock().metadata["swe_metadata"]
    mock_swe.evaluability_check.return_value = None
    mock_swe.prepare_workspace = AsyncMock()

    # Clock: first call = t0, second call (in gate) = t0 + 1000s elapsed.
    # agent_time_budget_sec=1800, floor=900 → remaining=800 < 900 → no retry.
    times = iter([1000.0, 2000.0, 2000.0, 2000.0])
    monkeypatch.setattr(gen_mod.time, "time", lambda: next(times, 2000.0))

    with (
        patch("examples.coding_agent_rl.generate._AdapterService", return_value=state),
        patch("examples.coding_agent_rl.generate.E2BSandbox", fake_e2b_cls),
        patch("examples.coding_agent_rl.generate.HARNESS_CLS", mock_harness_cls),
        patch("examples.coding_agent_rl.generate.swe", mock_swe),
        patch("examples.coding_agent_rl.generate._session_id", return_value="sess-123"),
        patch("examples.coding_agent_rl.generate.get_prompt", return_value="prompt"),
        patch("examples.coding_agent_rl.generate._abort_result", side_effect=lambda s, r, i: [r]),
        patch.object(
            gen_mod, "CONFIG",
            replace(gen_mod.CONFIG, retry_policy="retry-from-scratch",
                    rollout_retries=2, agent_time_budget_sec=1800,
                    retry_min_budget_sec=900),
        ),
        patch("examples.coding_agent_rl.generate.asyncio.sleep", side_effect=fast_sleep),
    ):
        result = await generate(MagicMock(), _make_sample_mock(), {})

    # Budget gate fired: provisioning attempted exactly once, no retry.
    assert fake_e2b_cls.call_count == 1
```

Note: if `_make_state_mock` does not already exist, extract it from the existing `always-fail` test's inline `state` setup into a module-level helper as part of this step, and have both tests use it.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest examples/coding_agent_rl/test_retry.py -k budget_gate -v`
Expected: FAIL — `fake_e2b_cls.call_count == 3` (loop retries because the gate does not exist yet).

- [ ] **Step 3: Add the budget gate**

In `generate.py`, in the `except Exception as error:` block, insert the gate immediately after the `if not _is_retryable(error) or attempt >= CONFIG.rollout_retries: raise` line (currently line 413-414), before the `pre-launch` gate:

```python
                    if not _is_retryable(error) or attempt >= CONFIG.rollout_retries:
                        raise
                    remaining = CONFIG.agent_time_budget_sec - (time.time() - t0)
                    if remaining < CONFIG.retry_min_budget_sec:
                        logger.warning(
                            "[coding_agent_rl] %s: %.0fs agent budget remaining < "
                            "retry_min_budget %.0fs; not retrying",
                            instance_id,
                            remaining,
                            CONFIG.retry_min_budget_sec,
                        )
                        raise
                    if CONFIG.retry_policy == "pre-launch" and turns_recorded:
                        raise
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest examples/coding_agent_rl/test_retry.py -k budget_gate -v`
Expected: PASS

- [ ] **Step 5: Add the complementary "gate passes" test**

Add a test where remaining budget is ample (elapsed small), confirming the loop still retries the full `rollout_retries + 1` times:

```python
@pytest.mark.asyncio
async def test_budget_gate_allows_retry_when_budget_ample(monkeypatch):
    from examples.coding_agent_rl import generate as gen_mod

    fake_sb = AsyncMock()
    fake_sb.__aenter__ = AsyncMock(side_effect=RuntimeError("e2b connection lost"))
    fake_sb.__aexit__ = AsyncMock(return_value=False)
    fake_e2b_cls = MagicMock(return_value=fake_sb)

    state = _make_state_mock()
    mock_harness_cls = MagicMock(return_value=AsyncMock())
    mock_swe = MagicMock()
    mock_swe.get_metadata.return_value = _make_sample_mock().metadata["swe_metadata"]
    mock_swe.evaluability_check.return_value = None
    mock_swe.prepare_workspace = AsyncMock()

    # Clock always near t0 → remaining ~= full budget, well above floor.
    monkeypatch.setattr(gen_mod.time, "time", lambda: 1000.0)

    with (
        patch("examples.coding_agent_rl.generate._AdapterService", return_value=state),
        patch("examples.coding_agent_rl.generate.E2BSandbox", fake_e2b_cls),
        patch("examples.coding_agent_rl.generate.HARNESS_CLS", mock_harness_cls),
        patch("examples.coding_agent_rl.generate.swe", mock_swe),
        patch("examples.coding_agent_rl.generate._session_id", return_value="sess-123"),
        patch("examples.coding_agent_rl.generate.get_prompt", return_value="prompt"),
        patch("examples.coding_agent_rl.generate._abort_result", side_effect=lambda s, r, i: [r]),
        patch.object(
            gen_mod, "CONFIG",
            replace(gen_mod.CONFIG, retry_policy="retry-from-scratch",
                    rollout_retries=2, agent_time_budget_sec=1800,
                    retry_min_budget_sec=900),
        ),
        patch("examples.coding_agent_rl.generate.asyncio.sleep", side_effect=fast_sleep),
    ):
        result = await generate(MagicMock(), _make_sample_mock(), {})

    # All attempts consumed: initial + 2 retries = 3 provisioning attempts.
    assert fake_e2b_cls.call_count == 3
```

- [ ] **Step 6: Run the full retry suite**

Run: `pytest examples/coding_agent_rl/test_retry.py -v`
Expected: PASS (all existing tests plus the new budget tests)

- [ ] **Step 7: Commit**

```bash
git add examples/coding_agent_rl/generate.py examples/coding_agent_rl/test_retry.py
git commit -m "feat(coding_agent_rl): gate rollout retries on remaining agent budget"
```

---

### Task 3: Document the env var

**Files:**
- Modify: `examples/coding_agent_rl/README.md` (retry policy section documenting `SWE_ROLLOUT_RETRY_POLICY` / `SWE_ROLLOUT_RETRIES`)

**Interfaces:** none (docs only).

- [ ] **Step 1: Locate the retry env-var docs**

Run: `grep -rn "SWE_ROLLOUT_RETRY_POLICY" examples/coding_agent_rl/README.md`
Expected: the retry-policy documentation block added in commit `485f743`.

- [ ] **Step 2: Add the new env var next to the existing retry docs**

Add an entry describing:

```
- `SWE_AGENT_RETRY_MIN_BUDGET_SEC` (default: half of `SWE_AGENT_TIME_BUDGET_SEC`):
  minimum agent time budget that must remain before a retry is started. If a
  failure happens late enough that less than this many seconds of the agent
  budget remain, the rollout fails instead of retrying into a run that cannot
  finish. Set to `0` to disable the check. Applies to all retry policies.
```

Match the surrounding list's exact formatting.

- [ ] **Step 3: Commit**

```bash
git add examples/coding_agent_rl/README.md
git commit -m "docs(coding_agent_rl): document SWE_AGENT_RETRY_MIN_BUDGET_SEC"
```

---

### Task 4: Lint

**Files:** all modified.

- [ ] **Step 1: Run pre-commit on changed files**

Run: `pre-commit run --files examples/coding_agent_rl/generate.py examples/coding_agent_rl/test_retry.py examples/coding_agent_rl/README.md`
Expected: black / isort / ruff / autoflake all pass (or auto-fix; re-stage and re-run if they modify files).

- [ ] **Step 2: Commit any lint fixes**

```bash
git add -A
git commit -m "style(coding_agent_rl): lint retry budget changes"
```
