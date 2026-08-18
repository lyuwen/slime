# Rollout Retry for Pre-Launch Failures

**Date:** 2026-08-18  
**Status:** Approved  
**Scope:** `slime/agent/sandbox.py` (kernel), `examples/coding_agent_rl/generate.py` (example)

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

The launcher currently writes to `/tmp/.{tag}-{uuid}.sh`. Move it to the **user's home directory** (`~/tmp/.{tag}-{uuid}.sh`), which is guaranteed writable by that user once `ensure_agent_user` has run:

```python
home = "/root" if user == "root" else f"/home/{user}"
launcher_dir = f"{home}/tmp"
await sb.exec(f"mkdir -p {launcher_dir}", user=user, check=True, timeout=10)
launcher = f"{launcher_dir}/.{slug}.sh"
```

### Why This Fixes the Race

- The failing write is `user="agent"` in `run_agent` (the harness run path), called **after** `ensure_agent_user` has created and chowned `/home/agent`. The agent's home is stable and writable the moment it's created — no readiness race.
- Root launchers (npm install, swebench eval) move to `/root/tmp`, which root can always write.
- `/tmp` was an arbitrary choice; user home is strictly safer and sidesteps the concurrency race entirely.

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
        turns_recorded = state.adapter.manager.has_session(session_id)
        if turns_recorded or attempt >= CONFIG.rollout_retries:
            raise
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

`state.adapter.manager.has_session(session_id)` returns `True` iff at least one turn with real prompt messages was recorded. The tree is only created inside `record_turn` at `self._trees.setdefault(sid, MessageNode())` (`trajectory.py:300`), and `record_turn` early-returns on empty prompt before creating the tree (`trajectory.py:292-294`). So `has_session` exactly encodes the policy: **retry if no turns recorded (pre-launch failure); hard-fail once turns exist (partial trajectory).**

This implements the "hard-fail after turns" policy from the clarifying questions. The mechanism supports retry-from-scratch if the policy changes later — just remove the `turns_recorded` check from the gate — but the current guard enforces hard-fail once turns exist.

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

1. **Pre-launch retry success** — `run()` raises on attempt 0 (before any turn), `has_session=False` → retry → succeeds on attempt 1. Only the winning sid's samples returned.
2. **Hard-fail after turns** — `run()` records a turn, then raises. `has_session=True` → immediate hard-fail, no retry.
3. **Exhaust retries** — all attempts fail pre-launch → abort after `rollout_retries+1`, each failed sid dropped (store clean).
4. **Fresh sid per attempt** — verify sids are `{base}-a0`, `{base}-a1`, ... and `base_sample.session_id` ends as the winning sid.

No new tests for the launcher-path change (Component 1) — it's a one-line path substitution with no new logic. Existing agent tests (`test_agent/`) exercise `exec_and_wait` and will catch breakage.

---

## Scope Notes and Limitations

### In Scope

- **Launcher path change** (`slime/agent/sandbox.py`) — kernel change, but minimal (one path derivation + one `mkdir`), well-justified, fixes a real concurrency bug. Aligns with CONTRIBUTING "bug fixes and optimizations."
- **Rollout retry widening** (`examples/coding_agent_rl/generate.py`) — example-only, no kernel surface.

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

- **Primary fix:** Move launcher from `/tmp` to user home (`~/tmp` for non-root, `/root/tmp` for root) in `exec_and_wait`. Sidesteps the `/tmp` readiness race under high-concurrency launches.
- **Backstop:** Widen rollout retry to cover `run()`, gated on `has_session` (turns recorded). Fresh sid per attempt avoids `closed`-poisoning; only the winning sid reaches training.
- **Testing:** Unit tests for retry logic (fake adapter); existing agent tests cover launcher path.
- **Scope:** One minimal kernel change (`sandbox.py`), one example change (`generate.py`). No new abstractions, no upstream-sensitive refactors.
