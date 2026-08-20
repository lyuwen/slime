# Coding-Agent RL Evaluation Sandbox Retry Design

Date: 2026-08-18  
Status: Proposed  
Area: `examples/coding_agent_rl`, `slime.agent.sandbox`

## Goal

Recover a completed coding-agent rollout when grading fails because the evaluator sandbox or its HTTP stream becomes unavailable. Retry grading in a fresh sandbox with the same image and model-produced diff, without rerunning the agent or treating a known test failure as infrastructure failure.

## Problem Statement

The current per-sample flow in `examples/coding_agent_rl/generate.py` is:

```text
agent sandbox
  -> prepare workspace
  -> run agent and adapter interaction
  -> capture git diff
  -> destroy agent sandbox
  -> swe.run_evaluation(...)
       -> create a clean evaluator sandbox
       -> apply the captured diff
       -> run the dataset grader
  -> adapter.finish_session(...)
```

An error such as `httpcore.ReadError` while `E2BSandbox.exec()` is consuming the grader response means no completed process result was received. The model trajectory and diff already exist, but `generate()` currently catches the exception as a generic rollout failure and returns an aborted sample. The evaluator sandbox is normally destroyed by context-manager cleanup, and its incomplete stdout/stderr stream may no longer be recoverable.

`E2BSandbox` already retries idempotent RPCs against the same sandbox. That mechanism cannot reliably recover a grading command whose response stream broke after process creation: rerunning the command in the same sandbox risks duplicate or contaminated execution, while reconnecting cannot prove the original exit status. Recovery therefore belongs one level higher, where the complete evaluation can be replayed in a new clean sandbox.

## Observed Incident

The motivating failure occurred on 2026-08-17 at 16:02:23 for ScaleSWE instance `astropy_sphinx-automodapi_pr44`. The RolloutManager had finished the agent phase and entered F2P grading. The representative traceback was:

```text
(RolloutManager pid=131334) [2026-08-17 16:02:23] generate.py:406 -
[coding_agent_rl] astropy_sphinx-automodapi_pr44: rollout failed:
Traceback (most recent call last):
  File "examples/coding_agent_rl/generate.py", line 334, in generate
    reward, applied_cleanly = await swe.run_evaluation(...)
  File "examples/coding_agent_rl/swe.py", line 273, in run_evaluation
    return await _grade_scaleswe(md, diff_text, timeout_sec)
  File "examples/coding_agent_rl/swe.py", line 312, in _grade_scaleswe
    r = await _run_f2p_script(ev, workdir, f2p_script, timeout_sec)
  File "examples/coding_agent_rl/swe.py", line 377, in _run_f2p_script
    ec, _, _ = await ev.exec(
        f"cd {workdir} && python {_F2P}",
        user="agent",
        check=False,
        timeout=timeout,
    )
  File "slime/agent/sandbox.py", line 360, in exec
    res = await self._rpc_retry(...)
  File "slime/agent/sandbox.py", line 293, in _rpc_retry
    raise last_err
  File "e2b/sandbox_async/commands/command.py", line 235, in run
    return proc if background else await proc.wait()
  File "e2b/sandbox_async/commands/command_handle.py", line 148, in _handle_events
    async for stdout, stderr, pty in self._iterate_events():
  File "e2b_connect/client.py", line 394, in acall_server_stream
    async for chunk in http_resp.aiter_stream():
  File "httpcore/_async/http11.py", line 203, in _receive_response_body
    event = await self._receive_event(timeout=timeout)
  File "httpcore/_backends/anyio.py", line 32, in read
    with map_exceptions(exc_map):
  File "httpcore/_exceptions.py", line 14, in map_exceptions
    raise to_exc(exc) from exc
httpcore.ReadError

[coding_agent_rl] astropy_sphinx-automodapi_pr44 aborted: exception:ReadError
```

The diagnostic distinction is that `ev.exec()` raised while consuming the streamed command response; it did not return an exit code. Therefore this trace does not establish that pytest failed or that the patch earned reward zero. It establishes only that the evaluator result was not observed.

By this point the user prompt, model/tool interaction, final agent output, and Git diff had already been produced. The recoverable unit is therefore `(metadata, image, diff_text, grader configuration)`, which can be replayed in a new evaluator sandbox. The broken command stream and any stdout/stderr not delivered before the connection loss are not recoverable from the deleted pod.

Likely infrastructure causes include evaluator pod deletion or restart, container or `envd` failure, gateway reset, and an intermittent network disconnect. The retry design intentionally does not need to distinguish among these causes: they all leave the grading result unknown and are safe to handle by recreating the evaluator.

## Requirements

- Preserve the agent trajectory, session, and captured `diff_text`; retry evaluation only.
- Use a newly created evaluator sandbox for every evaluation attempt.
- Retry only when the evaluation result is unknown because of sandbox or transport infrastructure.
- Do not retry a valid reward of `0.0`, test exit code, patch-apply failure, malformed dataset, parser result, or other deterministic evaluation outcome.
- Apply the policy uniformly to ScaleSWE and SWE-bench grading.
- Bound retries and retain the existing outer rollout wall-clock guard.
- Log enough attempt context to diagnose recovered and exhausted failures after the evaluator pod is gone.
- Preserve `EvalResult` and reward semantics for callers.

## Non-goals

- Rerunning the coding agent or replaying adapter/model calls.
- Resuming an interrupted grader process inside the failed sandbox.
- Recovering stdout/stderr that the provider never delivered before the stream broke.
- Retrying flaky tests that completed with a valid failing result.
- Changing dataset protocols, reward calculation, patch application, or sandbox providers.
- Persisting failed-evaluation trajectories in a new artifact format; this can be designed separately if operators need postmortem artifacts for exhausted retries.

## Approaches Considered

### 1. Retry `ev.exec()` against the same sandbox

This is the smallest code change, but it is unsafe for grading commands. A transport failure can occur after the remote process started, so the client cannot know whether a second `exec()` duplicates work or observes a modified workspace. It also cannot recover when the evaluator pod has already disappeared.

### 2. Retry `swe.run_evaluation()` from `generate.py`

This correctly recreates the sandbox and leaves the agent untouched. However, it makes `generate.py` own details of sandbox exception classification, and other direct callers such as `sandbox_smoke.py` do not automatically share the same evaluation policy.

### 3. Retry inside the public `swe.run_evaluation()` boundary — recommended

Keep each protocol grader as a single fresh-sandbox attempt and make `run_evaluation()` the bounded orchestration layer around protocol dispatch. This is the narrowest shared boundary that owns evaluation lifecycle, applies to both protocols, and guarantees a new sandbox after a failed attempt. The caller supplies the configured attempt budget; callers that omit it retain one-attempt behavior.

## Proposed Design

### Evaluation attempt boundary

Split the current dispatch body into a private single-attempt function and retain `run_evaluation()` as the public entry point:

```python
async def run_evaluation(
    md: dict,
    *,
    diff_text: str,
    timeout_sec: int,
    max_attempts: int = 1,
) -> EvalResult:
    ...
```

For each attempt, `run_evaluation()` calls the single-attempt dispatcher. The dispatcher selects `_grade_scaleswe()` or `_grade_swebench()`, and each grader continues to own `async with E2BSandbox(image) as ev`. If an attempt raises, exiting that context kills or releases the failed sandbox before the loop proceeds. The next attempt calls the grader again and therefore boots a clean sandbox from the original image, reapplies the unchanged `diff_text`, and reruns the same grading protocol.

No mutable evaluator state crosses attempts. The model-facing agent sandbox is already gone before evaluation begins, and `generate()` does not reopen its adapter session or call the harness again.

### Failure classification

Expose one sandbox-layer predicate in `slime/agent/sandbox.py` so the existing RPC retry loop and the evaluation retry policy do not maintain divergent exception-name lists. The predicate distinguishes two recovery scopes:

- **Same-sandbox retry:** transient connection/protocol failures that are safe for an idempotent RPC.
- **Fresh-sandbox retry:** the same transport failures plus errors indicating that the sandbox no longer exists or entered `STOPPED` state.

The fresh-sandbox evaluation policy treats the following as retryable when they escape an attempt without an `EvalResult`:

- HTTP/client transport errors already recognized by `E2BSandbox`, including `ReadError`, `WriteError`, `ConnectError`, protocol errors, SSL errors, and network read/connect/write/pool timeouts.
- Provider `SandboxException` failures that indicate an unavailable, missing, or stopped sandbox, because a new sandbox can recover them even though reconnecting to the old sandbox cannot.
- Provider-side transient sandbox gateway errors already classified as retryable by the sandbox layer.

The policy explicitly does not catch `asyncio.CancelledError` or the outer `asyncio.TimeoutError`. It also does not retry arbitrary Python exceptions such as `KeyError`, `JSONDecodeError`, assertion failures, SWE-bench report parsing failures, or programmer errors. Those retain their existing behavior.

A command that returns an exit code has a known outcome and never enters exception classification. Thus exit code `1`, a failed F2P test, `EvalResult(0.0, ...)`, and `applied_cleanly=False` all return immediately without retry.

Evaluation commands should be marked non-idempotent at the low-level `E2BSandbox.exec()` boundary where applicable. This prevents the existing same-sandbox RPC retry loop from starting the grader twice after an ambiguous stream break; the exception instead reaches `run_evaluation()`, which can safely retry the whole attempt in a clean sandbox. Setup operations that are demonstrably idempotent may retain same-sandbox RPC retry behavior.

### Configuration and time budgets

Add `SWE_EVAL_MAX_ATTEMPTS`, parsed into `SweConfig.eval_max_attempts`, with a default of `2` and a minimum of `1`. “Attempts” is used instead of “retries” to avoid off-by-one ambiguity: the default permits one initial evaluation plus one fresh-sandbox retry.

`generate.py` passes `CONFIG.eval_max_attempts` to `swe.run_evaluation()`. `SWE_EVAL_TIMEOUT_SEC` remains the timeout for each grading attempt; it is not divided across attempts.

When `SWE_ROLLOUT_GUARD_SEC` is not explicitly set, its derived value becomes:

```text
agent_time_budget
+ eval_timeout * eval_max_attempts
+ 180 seconds of boot/cleanup/backoff allowance
```

An explicitly configured outer guard remains authoritative. If it expires during an attempt or before another retry, the existing `wall_clock_timeout` path aborts the sample and no further attempt starts.

Use bounded full-jitter exponential backoff between fresh-sandbox attempts, starting with a maximum delay of one second and capped at eight seconds. No additional tuning variable is introduced initially; the retry count and outer guard are the operational controls.

### Observability

Every retry warning records:

- `instance_id` and protocol;
- failed attempt number and maximum attempts;
- exception class and a bounded exception message;
- selected backoff delay;
- that the next attempt will use a fresh evaluator sandbox.

When a later attempt succeeds, emit an info log with the total number of attempts and final reward. When all attempts are exhausted, emit one warning with the same identifiers and re-raise the last infrastructure exception so `generate()` preserves its existing abort behavior.

Do not log the full model diff, dataset test script, or unbounded traceback on every intermediate failure. `generate()` already logs the terminal traceback if retry exhaustion propagates.

Attempt counts remain in structured logs for the first version. This keeps the two-field `EvalResult` contract and returned sample metadata unchanged.

### Data flow

```text
agent rollout completes once
  -> capture diff_text once
  -> evaluation attempt 1
       -> fresh sandbox A
       -> known EvalResult --------------------------> return result
       -> non-retryable exception -------------------> propagate
       -> retryable infrastructure exception
            -> sandbox A cleanup
            -> bounded jitter
            -> evaluation attempt 2
                 -> fresh sandbox B
                 -> known EvalResult ----------------> return result
                 -> exhausted infrastructure error --> propagate
  -> finish adapter session once on successful grading
```

## Code Boundaries

- `slime/agent/sandbox.py`
  - Expose the shared sandbox error classifier with same-sandbox and fresh-sandbox scopes.
  - Reuse it from `_rpc_retry()`.
  - Ensure grader process launches are not transparently retried in the same sandbox after an ambiguous response failure.
- `examples/coding_agent_rl/swe.py`
  - Make protocol dispatch a single-attempt helper.
  - Add the bounded fresh-sandbox retry loop to `run_evaluation()`.
  - Keep the existing two-field `EvalResult` contract unchanged; log attempt counts inside the retry loop.
- `examples/coding_agent_rl/generate.py`
  - Parse and validate `SWE_EVAL_MAX_ATTEMPTS`.
  - Pass the attempt budget to `run_evaluation()`.
  - Account for all attempts in the derived rollout guard.
- `examples/coding_agent_rl/README.md` and active launchers
  - Document the new environment variable and default. Launchers need an explicit export only where they already enumerate runtime environment forwarding.
- Focused CPU tests near the existing sandbox and coding-agent tests
  - Cover classification, retry orchestration, configuration, and timeout derivation without a real E2B cluster.

The implementation should avoid a general-purpose retry framework or new base class. One classifier plus one evaluation loop is sufficient.

## Testing

Focused unit tests will verify:

1. A first-attempt `ReadError` followed by success calls the single-attempt dispatcher twice and returns the second `EvalResult`.
2. A missing/stopped-sandbox exception is non-retryable for the same-sandbox RPC scope but retryable for the fresh-sandbox evaluation scope.
3. Each retry invokes a new grader context; a fake sandbox factory records two distinct sandbox instances and confirms both are cleaned up.
4. Exhausting `max_attempts` re-raises the final infrastructure exception after exactly that many calls.
5. A non-retryable exception is raised after one call even when more attempts are configured.
6. `EvalResult(0.0, True)` and `EvalResult(0.0, False)` return after one call and are never retried.
7. ScaleSWE and SWE-bench dispatch both use the same retry boundary.
8. Cancellation and the outer wall-clock timeout are never swallowed or converted into retry attempts.
9. `SWE_EVAL_MAX_ATTEMPTS=1` preserves current single-attempt behavior; invalid values below one fail configuration parsing early.
10. The automatically derived rollout guard multiplies the per-attempt evaluation timeout by `eval_max_attempts`, while an explicit `SWE_ROLLOUT_GUARD_SEC` remains unchanged.
11. Logs contain instance, protocol, attempt count, exception type, and recovered/exhausted status without embedding the diff or full test script.
12. Existing coding-agent sandbox, ScaleSWE, SWE-bench, trajectory-persistence, and adapter tests remain passing.

Verification will run the focused CPU tests and the repository's formatting/lint checks for modified files. A live E2B smoke test should additionally inject or reproduce a response-stream disconnect during grading and confirm that the second attempt uses a different sandbox ID and returns the expected reward. Full GPU training is not required to validate the retry mechanism.

## Rollout and Operational Guidance

Ship with `SWE_EVAL_MAX_ATTEMPTS=2`. Operators can set it to `1` for immediate rollback to current behavior. During initial runs, monitor:

- fraction of evaluations requiring more than one attempt;
- success rate after retry;
- retryable errors by exception type;
- added evaluator sandbox count and evaluation latency;
- exhausted retry count and outer-guard timeouts;
- reward consistency for instances later rerun without infrastructure faults.

A high retry rate should trigger investigation of the sandbox gateway rather than increasing the attempt count indefinitely. The retry mechanism protects completed rollouts from occasional infrastructure loss; it is not a substitute for sandbox-cluster health.

## Acceptance Criteria

- A transient evaluation `ReadError` no longer discards the completed agent rollout when a fresh evaluator attempt succeeds.
- Every retry starts from the original image and reapplies the original diff in a distinct sandbox.
- Known grading failures are never retried.
- Retry exhaustion remains visible as the existing aborted rollout, with attempt-specific diagnostics.
- Default execution is bounded to two evaluation attempts and remains covered by the outer rollout guard.
- Reward and patch-application semantics are unchanged.
