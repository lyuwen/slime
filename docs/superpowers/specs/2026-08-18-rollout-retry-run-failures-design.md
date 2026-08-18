# Rollout Retry for Pre-Launch Failures

**Date:** 2026-08-18  
**Status:** Approved  
**Scope:** `slime/agent/sandbox.py` (kernel), `examples/coding_agent_rl/generate.py` (example), `examples/coding_agent_rl/README.md` (documentation)

## Problem

Under high-concurrency sandbox launches, the coding-agent rollout occasionally fails with:

```
e2b.exceptions.SandboxException: 500: error creating file: open /tmp/.run-52f4c6ffae4a.sh: permission denied
```

This error occurs in `HARNESS_CLS().run()` when writing the launcher script as `user="agent"`. The failure happens **after** `open_session` but **before** any agent turn is recorded, so no trajectory data exists yet — but today's retry logic refuses to retry because `session_open=True` acts as a blunt "already started generating" signal. The sample aborts, wasting the entire rollout attempt.

### Root Cause

The permission-denied error is a **readiness race under load**, not a bad image. Most launches succeed; failures cluster when many sandboxes boot concurrently. The hypothesis: under high-concurrency launches, `/tmp` (or the `agent` user, or the filesystem endpoint) isn't fully ready in the first moment after `create` returns. The existing `chmod 1777 /tmp` sanitation runs as **root** and succeeds, but the subsequent launcher write as **agent** hits `EACCES`. The race is exacerbated by concurrency structure: `_BOOT_SEM` (default 16) only gates `create + install_cli` inside `boot_agent_sandbox`; the sandbox is **yielded and the semaphore released** before `run()` issues the first `/tmp` write. So launcher writes storm the gateway with unbounded concurrency precisely when sandboxes are still settling.

The error is already classified transient by `_is_transient_rpc_error` (any `SandboxException` not containing `"does not exist"` or `"STOPPED state"`), so `write_file` → `_rpc_retry` already retries it up to 6× with backoff. But same-sandbox retry can't outrun a readiness window that sometimes exceeds the retry budget under load — and the only action that clears it is a **fresh sandbox** (which gets a fresh readiness window).

## Design

Two components: **move launcher to user home** (primary fix, eliminates the `/tmp` race), and **widen rollout retry** (general pre-launch backstop for any setup failure).

---

## Component 1: Move Launcher to User Home (Primary Fix)

**File:** `slime/agent/sandbox.py`  
**Function:** `exec_and_wait`

The launcher and **all internally managed files** (`done_file`, `lock_dir`, default `out_file`) currently live in `/tmp/`. Move them to the **user's home directory** (`~/tmp/`), which is guaranteed writable by that user once `ensure_agent_user` has run:

```python
home = "/root" if user == "root" else f"/home/{user}"
run_dir = f"{home}/tmp"
await sb.exec(f"mkdir -p {run_dir}", user=user, check=True, timeout=10)

out_file = out_file or f"{run_dir}/.{slug}.out"  # honor caller-provided path
done_file = f"{run_dir}/.{slug}.done"
launcher = f"{run_dir}/.{slug}.sh"
lock_dir = f"{run_dir}/.{slug}.spawned"
```

Moving only the launcher would be insufficient: if `/tmp` is temporarily unwritable by `agent`, `mkdir {lock_dir}` inside the launch command (`mkdir ... || exit 0`) would also fail silently, no process would start, and polling would wait the full time budget before returning -1 — no exception, so the retry never fires. Moving all four together eliminates `/tmp` dependence entirely for internally managed paths.

### Why This Fixes the Race

- The failing write is `user="agent"` in `run_agent` (the harness run path), called **after** `ensure_agent_user` has created and chowned `/home/agent`. The agent's home is stable and writable the moment it's created — no readiness race.
- Root launchers (npm install, swebench eval) move to `/root/tmp`, which root can always write.
- `/tmp` was an arbitrary choice; user home is strictly safer and sidesteps the concurrency race entirely.
- Moving all four internally managed paths together (`launcher`, `done_file`, `lock_dir`, default `out_file`) eliminates `/tmp` dependence. Caller-provided `out_file` paths are honored unchanged.

### Call-Site Invariants

`exec_and_wait` is used in three contexts:

1. **`run_agent`** (`user="agent"`) — called after `ensure_agent_user`. Agent home exists. ✅
2. **`install_npm_cli`** (`user="root"`) — called during CLI install, before `ensure_agent_user`. Uses `/root/tmp`. ✅
3. **`swe.py` eval** (`user="root"`) — evaluator sandbox, agent user may not exist. Uses `/root/tmp`. ✅

The home-derivation logic (`/root` for root, `/home/{user}` otherwise) handles all three correctly.

---

## Component 2: Widen Rollout Retry (General Pre-Launch Backstop)

**File:** `examples/coding_agent_rl/generate.py`  
**Function:** `generate`

Even with the launcher move, other pre-launch failures can occur (install flakes, workspace prep, transient RPC errors). Widen the rollout retry to cover `run()`, gated on whether any trajectory turns were actually recorded.

### Current Behavior

The retry loop wraps `boot_agent_sandbox` and workspace prep, but excludes `run()` because `session_open` is set `True` immediately after `open_session` (line 357), and the `except` does `if session_open: raise` (line 382). That guard was a proxy for "did we start generating," but it's too blunt — it blocks retry even when zero turns were recorded.

### New Behavior

Replace the `session_open` guard with a **precise signal**: did the trajectory manager actually record a turn?

```python
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
            agent_exit_code = await HARNESS_CLS().run(...)
            diff_text = await swe.git_diff(sb, md["workdir"])
            ...
        base_sample.session_id = session_id  # winning sid on success
        break
    except Exception as error:
        if CONFIG.retry_policy == "always-fail":
            raise
        turns_recorded = state.adapter.manager.has_session(session_id)
        retryable = _is_retryable(error) or is_fresh_sandbox_retryable(error)
        if not retryable or attempt >= CONFIG.rollout_retries:
            raise
        if CONFIG.retry_policy == "pre-launch" and turns_recorded:
            raise
        # retry permitted
        if session_open:
            await state.adapter.drop_session(session_id, wait_timeout=30)
            session_open = False
        backoff = 2 ** attempt
        logger.warning(
            "[coding_agent_rl] %s: setup attempt %d/%d failed: %s, retrying in %ds...",
            instance_id,
            attempt + 1,
            CONFIG.rollout_retries + 1,
            error,
            backoff,
        )
        await asyncio.sleep(backoff)
```

### Gate Logic

The retry gate is controlled by `SWE_ROLLOUT_RETRY_POLICY` (env var, default `"pre-launch"`):

**`"pre-launch"` (default, recommended)** — retry requires **both** conditions:

1. **No turns recorded** — `state.adapter.manager.has_session(session_id)` returns `True` iff at least one turn with real prompt messages was recorded. The tree is only created inside `record_turn` at `self._trees.setdefault(sid, MessageNode())` (`trajectory.py:300`), and `record_turn` early-returns on empty prompt before creating the tree (`trajectory.py:292-294`).

2. **Retryable error** — Use the existing example-level `_is_retryable(error)` from `generate.py`, extended with `is_fresh_sandbox_retryable(error)` from the kernel. The example classifier handles `RuntimeError("e2b exec failed...")` from `check=True` execs (boot, install, workspace prep) and network/connection errors. The kernel classifier adds generic `SandboxException` (including those containing `"does not exist"` or `"STOPPED state"`, which a fresh sandbox can recover), E2B transport errors, and other sandbox setup failures. The combined check avoids duplicating logic while preserving existing boot/workspace retry coverage.

Combined: **retry if no turns recorded AND the error is retryable (example OR kernel classifier); hard-fail otherwise (turns exist, or deterministic error).**

**`"retry-from-scratch"` (experimental)** — retry on any retryable error regardless of turns. Drops the partial sid (discarding any already-generated turns and their token cost), opens a fresh sid, and retries from workspace prep. Use when mid-run sandbox instability is more costly than wasted generation. Risks masking real issues by re-rolling.

**`"always-fail"` (debug)** — never retry; hard-fail on any exception. Use to debug the retry logic itself or force immediate failure for CI.

Parse at `SweConfig.from_env`:

```python
retry_policy = os.environ.get("SWE_ROLLOUT_RETRY_POLICY", "pre-launch")
if retry_policy not in ("pre-launch", "retry-from-scratch", "always-fail"):
    raise ValueError(f"SWE_ROLLOUT_RETRY_POLICY={retry_policy!r} invalid; must be pre-launch|retry-from-scratch|always-fail")
```

Gate implementation:

```python
if retry_policy == "always-fail":
    raise
turns_recorded = state.adapter.manager.has_session(session_id)
retryable = is_fresh_sandbox_retryable(error)
if not retryable or attempt >= CONFIG.rollout_retries:
    raise
if retry_policy == "pre-launch" and turns_recorded:
    raise
# retry permitted: drop session if open, backoff, continue
```

This makes the policy runtime-switchable without code changes, defaults to the safe "hard-fail after turns" behavior, and leaves the door open for experimenting with mid-run retry under controlled conditions.

### Per-Attempt Session ID

Each attempt gets a fresh sid: `session_id = f"{base_session_id}-a{attempt}"`. This sidesteps a fundamental adapter limitation: `drop_session` adds the sid to `self.closed` and **never clears it** (`common.py:231`). Reusing a sid after drop would get every subsequent turn refused with `503 session closed` (`common.py:332`). Fresh sids avoid this entirely — each attempt opens a brand-new session, and the failed attempt's `drop_session` just clears its (empty) store entry.

On success, `base_sample.session_id = winning_sid`. Only the winning sid reaches `finish_session` → `get_trajectory` → training. A dropped failed sid has no tree (no turns recorded), so it can never produce a `Sample`.

### Sandbox Lifecycle

The sandbox is already correctly torn down. The exception propagates out of `async with boot_agent_sandbox(...)` → `__aexit__` → `kill()` **before** reaching the `except` block. Each retry boots a fresh sandbox via `boot_agent_sandbox`, so the mechanism is: failed sandbox killed → backoff → fresh sandbox → fresh sid → retry.

### Timing Note

`has_session` becomes `True` **after the first turn fully completes** — after `call_sglang_generate`, after `_respond` flushes the reply, and only then does `record_turn` create the tree (`common.py:394`). There's a narrow window where generation happened but the turn failed before `record_turn` (e.g. client disconnect during flush → 499 at `common.py:384`), where `has_session` is still `False` and we'd retry despite token spend. This window can't be hit by the target failure class (pre-launch errors happen before any request reaches the adapter), so it's not worth guarding. Worth documenting as a known edge case.

---

## Testing

Extend `examples/coding_agent_rl/test_retry.py` with unit tests using a fake adapter/manager (no real sandbox):

1. **Pre-launch retry success** — policy=`"pre-launch"`, `run()` raises a retryable error on attempt 0 (before any turn), `has_session=False` → retry → succeeds on attempt 1. Only the winning sid's samples returned.
2. **Hard-fail after turns (pre-launch policy)** — policy=`"pre-launch"`, `run()` records a turn, then raises. `has_session=True` → immediate hard-fail, no retry.
3. **Hard-fail on non-retryable** — `run()` raises a non-retryable error (e.g. `ValueError`) before any turn. `has_session=False` but `is_fresh_sandbox_retryable=False` → immediate hard-fail, no retry.
4. **Exhaust retries** — all attempts fail pre-launch (retryable) → abort after `rollout_retries+1`, each failed sid dropped (store clean).
5. **Fresh sid per attempt** — verify sids are `{base}-a0`, `{base}-a1`, ... and `base_sample.session_id` ends as the winning sid.
6. **Retry-from-scratch policy** — policy=`"retry-from-scratch"`, `run()` records a turn, then raises retryable error → drops partial sid, retries with fresh sid. Verify the partial tree is discarded.
7. **Always-fail policy** — policy=`"always-fail"`, any exception → immediate hard-fail, no retry, even if retryable and no turns.
8. **Invalid policy** — `SWE_ROLLOUT_RETRY_POLICY="invalid"` → `ValueError` at `SweConfig.from_env`.

For the launcher-path change (Component 1), add explicit assertions to `tests/test_agent/test_harness.py`:

9. **Agent launcher path** — verify the launcher write goes to `/home/agent/tmp/.run-{uuid}.sh` when `user="agent"`, and that `done_file`, `lock_dir`, and default `out_file` also live in `/home/agent/tmp/`.
10. **Root launcher path** — verify the launcher write goes to `/root/tmp/.{tag}-{uuid}.sh` when `user="root"`, and that all sibling paths also live in `/root/tmp/`.

The existing tests check that *some* `run-*.sh` was written but don't assert location or user. These new assertions verify the home-directory path change is correct.

---

## Scope Notes and Limitations

### In Scope

- **Launcher path change** (`slime/agent/sandbox.py`) — kernel change, but minimal (home derivation + `mkdir` for all four artifact paths), well-justified, fixes a real concurrency bug. Aligns with CONTRIBUTING "bug fixes and optimizations."
- **Rollout retry widening** (`examples/coding_agent_rl/generate.py`) — example-only, no kernel surface. Extends existing `_is_retryable` with kernel `is_fresh_sandbox_retryable` to preserve boot/workspace retry coverage.
- **Configuration documentation** (`examples/coding_agent_rl/README.md`) — add `SWE_ROLLOUT_RETRY_POLICY` to the configuration table (line ~152), document the three policy values and their interaction with `SWE_ROLLOUT_RETRIES`.

### Known Limitations

1. **`self.closed` growth** — fresh sids mean `closed` grows by up to `rollout_retries+1` per retried sample instead of 1. This is a pre-existing unbounded-growth pattern in the kernel (never cleared); multiplying by a small constant isn't worth a kernel API change here. Noted for awareness.

2. **Backoff counts against wall-clock guard** — the retry backoffs (1s + 2s + 4s for default 3 retries) count against `rollout_guard_sec`. Since pre-launch failures are fast (no long-running agent work), the backoff overhead is negligible against the typical ~1800s+ guard.

3. **Token-spend edge case** — if a turn generates tokens but fails before `record_turn` (e.g. client disconnect during response flush), `has_session` is still `False` and we'd retry despite token spend. This can't happen for the target failure class (pre-launch, no adapter request), so it's not guarded. Documenting for completeness.

### Out of Scope

- **Kernel `reset_session` API** — a single-sid design would need this to clear `store + closed + _sid_turn_count + manager tree` atomically. Not pursued to avoid kernel surface expansion; fresh-sid sidesteps it entirely.
- **Boot readiness probe** — earlier considered as a targeted fix (probe `/tmp` agent-writability before yielding the sandbox). Made unnecessary by moving the launcher to user home; no longer needed.

---

## Why This Design

**Minimal kernel surface.** The launcher-path change is a one-line path substitution + one `mkdir`. The rollout retry is example-only.

**Policy-agnostic mechanism.** Fresh sid per attempt means the mechanism supports all three partial-failure policies (hard-fail, retry-from-scratch, ship-partial) — just flip the gate. The current `has_session` guard implements hard-fail-after-turns, but the plumbing doesn't bake it in.

**Fixes the diagnosed race without heuristics.** Moving to user home eliminates the `/tmp` readiness race entirely, no retry-budget tuning or probing needed. The rollout retry becomes a general backstop for *any* pre-launch failure, not just this one.

**Verifiable.** The launcher move is testable under synthetic concurrency; the rollout retry is unit-testable with a fake manager.

---

## Summary

- **Primary fix:** Move all internally managed `exec_and_wait` artifacts (`launcher`, `done_file`, `lock_dir`, default `out_file`) from `/tmp` to user home (`~/tmp` for non-root, `/root/tmp` for root). Sidesteps the `/tmp` readiness race under high-concurrency launches. Caller-provided `out_file` paths honored unchanged.
- **Backstop:** Widen rollout retry to cover `run()`, gated on policy (`SWE_ROLLOUT_RETRY_POLICY`, default `"pre-launch"`) + error classification (existing `_is_retryable` OR kernel `is_fresh_sandbox_retryable`). Fresh sid per attempt avoids `closed`-poisoning; only the winning sid reaches training. Default policy hard-fails after turns; `"retry-from-scratch"` enables mid-run retry (experimental); `"always-fail"` disables retry (debug).
- **Testing:** Unit tests for retry logic (fake adapter) covering all three policies + non-retryable hard-fail + invalid policy; explicit path assertions for agent/root launcher locations in harness tests.
- **Documentation:** Add `SWE_ROLLOUT_RETRY_POLICY` to `examples/coding_agent_rl/README.md` configuration table with policy descriptions.
- **Scope:** One minimal kernel change (`sandbox.py`), one example change (`generate.py`) + config parsing, one README update. No new abstractions, no upstream-sensitive refactors.
