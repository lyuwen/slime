# Time-budget gate for rollout retries

**Date:** 2026-08-19
**Component:** `examples/coding_agent_rl/generate.py`
**Status:** design approved, pending spec review

## Problem

The per-rollout retry loop in `generate()` wraps all attempts in a single
`asyncio.timeout(rollout_guard_sec)` (`generate.py:371`). The retry design
(`2026-08-18-rollout-retry-run-failures-design.md:244`) assumed retries fire
only for fast *pre-launch* failures, so their cost against the guard is
negligible.

That assumption breaks for late failures. Under `retry-from-scratch`, a
retryable error partway into a run (for example, a sandbox dying ~15 minutes
into a 30-minute agent budget) discards the partial trajectory and restarts the
whole agent run — but the shared guard is not reset, so the fresh attempt has
little time left. It races the guard, likely aborts anyway, and in the meantime
makes the rollout manager wait on a straggler that was doomed from the restart.
This risk grows as retries are treated more uniformly ("flat") across attempts.

## Goal

Do not start a retry unless enough of the agent time budget remains for a fresh
attempt to plausibly make progress. Otherwise, fail exactly as retry-exhaustion
does today (re-raise the original error). The gate applies to all retry
policies; in practice it only bites the late-failure case, since pre-launch
failures happen early when plenty of budget remains.

## Configuration

New field on `SweConfig` (`generate.py:61-128`), parsed in `from_env`:

- **Env var:** `SWE_AGENT_RETRY_MIN_BUDGET_SEC`
- **Field:** `retry_min_budget_sec: float`
- **Default:** `0.5 * agent_time_budget_sec` (derived after `agent_time_budget`
  is read; ~900s with the 1800s default).
- **Override:** if the env var is set, its value is used verbatim.
- **Validation:** must be `>= 0` (raise `ValueError` otherwise), matching the
  existing "non-negative budget" convention. `0` disables the time gate.

Parsing mirrors the existing derived-default pattern (`generate.py:90-92`):

```python
_min_budget_env = os.environ.get("SWE_AGENT_RETRY_MIN_BUDGET_SEC")
retry_min_budget_sec = (
    float(_min_budget_env) if _min_budget_env else 0.5 * agent_time_budget
)
if retry_min_budget_sec < 0:
    raise ValueError("SWE_AGENT_RETRY_MIN_BUDGET_SEC must be >= 0")
```

## Control flow

One added condition in the existing `except` block (`generate.py:409-430`),
after the retryable/attempt-count check so non-retryable or exhausted errors
still raise for their original reason:

```python
if not _is_retryable(error) or attempt >= CONFIG.rollout_retries:
    raise
remaining = CONFIG.agent_time_budget_sec - (time.time() - t0)
if remaining < CONFIG.retry_min_budget_sec:
    logger.warning(
        "%s: %.0fs of agent budget remaining < retry_min_budget %.0fs; not retrying",
        base_session_id, remaining, CONFIG.retry_min_budget_sec,
    )
    raise
if CONFIG.retry_policy == "pre-launch" and turns_recorded:
    raise
# existing drop-session / backoff / sleep / loop
```

- `t0 = time.time()` already exists (`generate.py:367`).
- `remaining` is measured against `agent_time_budget_sec`, not the guard, so the
  gate is self-contained on one budget: "retry only if less than
  `retry_min_budget_sec` of the agent budget has been consumed." A failure
  15 min into a 30-min budget leaves `remaining = 900s`, exactly at the default
  floor; past that point, no retry.
- **Behavior when the gate fires:** re-raise the original `error`. It lands in
  the same outer handling as retry-exhaustion (`generate.py:502-516`). No new
  exception type, no ABORTED-reason change — remaining behavior is unchanged.
- **Placement:** before the `pre-launch and turns_recorded` gate so the warning
  log distinguishes budget-gated stops from other raises. Both paths re-raise,
  so ordering does not change outcomes.

## Testing

Tests in `examples/coding_agent_rl/test_retry.py` (existing home for
`_is_retryable` and policy tests), following the `pytest.main([__file__])`
convention.

- **Config default:** `SWE_AGENT_TIME_BUDGET_SEC` set, min-budget unset →
  `retry_min_budget_sec == 0.5 * agent_time_budget_sec`.
- **Config override:** explicit env var wins over the derived default.
- **Config validation:** negative raises `ValueError`; `0` is accepted.
- **Gate fires:** retryable error, attempts remaining, but
  `remaining < retry_min_budget_sec` → original error propagates, no retry.
  Controlled with a small/patched clock or a tiny `agent_time_budget_sec` plus a
  nonzero floor so the gate trips on the first retry.
- **Gate passes:** same conditions with ample remaining budget → retry proceeds.
- **Interaction:** non-retryable error or exhausted attempts still raises for
  its original reason regardless of budget.

## Out of scope

- Per-attempt deadline resets (would change guard semantics).
- Changing `rollout_guard_sec` derivation or eval budgeting.
- Any change to the `pre-launch` / `always-fail` policy definitions.
