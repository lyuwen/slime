# Tool-call validation & regeneration during agentic rollout

**Date:** 2026-08-20
**Status:** Design approved, pending spec review

## Problem

During agentic rollout, a model sometimes emits an assistant message whose tool
call has structurally invalid *parameters* (missing required fields, wrong
types, mutually exclusive flags, or arguments that aren't even valid JSON). The
surface text looks fine, so it passes generation, but the tool then fails when
the agent tries to execute it.

We want to detect an invalid tool call at generation time and **regenerate the
assistant message** — re-running the request that produced it — before the turn
is committed to the trajectory or flushed to the agent.

In a standalone harness this is easy: the harness owns the LLM call and can just
retry. Here, generation happens inside a slime **adapter** that sits between the
agent (running in a sandbox) and SGLang, and the resulting turn is recorded by a
**trajectory manager**. So detection and retry must happen on the adapter side,
and we must ensure a rejected assistant message leaves **no side effects**.

## Key architectural facts (from exploration)

These determine why the design is safe and where it plugs in:

- **All tool-call parsing and assistant-message generation happen in the
  adapter**, specifically `BaseAdapter._run_turn`
  (`slime/agent/adapters/common.py:323`). The harness only points an agent CLI/
  SDK at the adapter URL; it never parses tool calls.
- **`record_turn` is the only place trajectory state mutates**
  (`slime/agent/trajectory.py:302`) and it is called **last** in `_run_turn`
  (`common.py:397`), *after* generation (`common.py:355`), parsing
  (`common.py:358`), reply building (`common.py:364`), and the HTTP flush
  (`common.py:374`).
- Therefore a rejected candidate that we discard **before** `record_turn`
  produces **zero side effects**: no trajectory node, no token log, no loss
  mask, no turn-count increment. There is no "bad assistant message to drop" —
  we simply loop and regenerate before ever recording.
- **The sandbox has no side effects during generation.** The sandbox is only
  touched when the agent process executes a tool, which happens *after* the HTTP
  response is delivered. Regenerating is safe: no filesystem writes, no command
  execution.
- **`call_sglang_generate` is module-level** (`common.py:462`) specifically so
  tests can monkeypatch it, and each call mints a fresh request id
  (`common.py:491`). A retry just calls it again with a new `rid`.
- **The adapter has no `is_eval` signal today.** The `Session` dataclass
  (`common.py:33`) carries only `sampling_defaults` and `max_context_tokens`.
  The `evaluation` bool exists in the rollout layer
  (`examples/coding_agent_rl/generate.py:359`) but is not forwarded into the
  adapter.
- **The OpenAI wire message is truncated to the first tool call** (`[:1]` at
  `slime/agent/adapters/openai.py:271`). Validating all calls must read the
  full, pre-truncation tool-call list, not `reply.wire`.

## Reference validator

The concrete validator this feature is built around is
`find_invalid_tool_call` in
`benchmarks-main/vendor/software-agent-sdk/openhands-sdk/openhands/sdk/llm/utils/tool_call_validation.py`.

Its contract:

```python
def find_invalid_tool_call(response_dict: dict) -> tuple[str, str] | None
```

- Input: an OpenAI `ModelResponse.model_dump()` dict of shape
  `{"choices": [{"message": {"tool_calls": [{"function": {"name", "arguments"}}]}}]}`,
  with `arguments` as a JSON **string**.
- Returns `None` if every tool call is valid, or `(tool_name, raw_arguments)`
  for the **first** invalid call.
- It is purely static (no tool execution, no filesystem), validates **all** tool
  calls across choices, and applies a generic JSON-parse check to every call
  plus tool-specific schema checks for known tools.

The slime hook contract matches this exactly so this function drops in unchanged.

## Requirements (decisions)

1. **Detection is a custom pluggable validator** — the user owns the logic. Not
   built-in JSON-schema validation.
2. **Wired via a new slime arg** `--tool-call-validator-path`, loaded with
   `slime.utils.misc.load_function`, consistent with other `--custom-*-path`
   hooks. Feature is inert when the arg is unset.
3. **Validator input = the OpenAI wire response dict** (Approach A). The
   reference function drops in unchanged.
4. **Validator return contract matches `find_invalid_tool_call`**: `None` =
   valid, `(name, reason)` = invalid. (Note: opposite polarity from a bare
   bool — `None`/falsy means valid.)
5. **Validate all tool calls** in a turn, from the full pre-truncation list.
6. **Scope: OpenAIAdapter only** (what OpenHands / `coding_agent_rl` uses).
   `AnthropicAdapter` is unchanged.
7. **Bounded retries**, configurable via `--tool-call-max-retries` (default 3).
8. **On exhaustion: accept the last candidate and flag it** in sample metadata
   (following the `ill_formed` precedent), so downstream can filter it. Do not
   abort the session.
9. **Same sampling params on retry** (fresh request id each attempt).
10. **Train-only.** Eval rollouts pass raw model output through unchanged.

## Design

### New arguments (`slime/utils/arguments.py`)

- `--tool-call-validator-path` (str, default `None`) — importable path to the
  validator function. When `None`, the feature is inert.
- `--tool-call-max-retries` (int, default `3`) — maximum regenerations per turn
  (so up to `max_retries + 1` generate calls).

### Hook contract

```python
validator(response_dict: dict) -> tuple[str, str] | None
# None            -> all tool calls valid
# (name, raw_args) -> first invalid tool call (name + raw arguments), triggers retry
```

`response_dict` is OpenAI shape with `arguments` as a JSON string:
`{"choices": [{"message": {"tool_calls": [{"function": {"name", "arguments"}}]}}]}`.

The validator is loaded once (at adapter construction, from the resolved arg)
and stored on the adapter.

### Retry loop in `_run_turn`

Introduce an overridable, side-effect-free hook on `BaseAdapter`:

```python
def _validate_reply(self, parsed, session) -> tuple[str, str] | None:
    return None  # base: no-op
```

`OpenAIAdapter._validate_reply` returns `None` when any of these hold — the
feature is disabled (no validator configured) or `session.is_eval` is `True` —
otherwise it builds the full-tool-call `response_dict` (from the complete
`wire_tool_calls` list, **before** the `[:1]` truncation) and calls the
configured validator.

`_run_turn` wraps the existing generate→parse→build block
(`common.py:355–365`) in a bounded loop:

```python
verdict = None
for attempt in range(self.tool_call_max_retries + 1):
    turn   = await call_sglang_generate(prompt_ids, s, body, adapter=self, session_id=sid)
    raw_output = tok.decode(turn.output_ids, skip_special_tokens=False) if turn.output_ids else ""
    parsed = parse_model_output(raw_output, tools_schema=tools_schema,
                                tool_parser_name=self.tool_parser,
                                reasoning_parser_name=self.reasoning_parser)
    reply  = self._build_reply(parsed, turn.finish_reason, translated, tools_schema)
    verdict = self._validate_reply(parsed, s)
    if verdict is None:
        break
    if attempt == self.tool_call_max_retries:
        # exhausted: accept last candidate, flag it, log for debugging
        self.logger.warning("[%s] sid=%s invalid tool call after %d retries: %s",
                             self.log_prefix, sid, self.tool_call_max_retries, verdict)
        break
    # else: loop and regenerate (fresh rid inside call_sglang_generate)

turn = dataclasses.replace(turn, ill_formed=parsed.ill_formed,
                           invalid_tool_call=verdict is not None)
```

Everything downstream — `_respond` (`common.py:374`) and `record_turn`
(`common.py:397`) — runs **once**, on the accepted candidate. Rejected
candidates only ever exist in local variables.

To build the validator's `response_dict` from the full tool-call list,
`_build_reply_parts` (`openai.py:211`) is refactored so the pre-truncation
`wire_tool_calls` list is reachable by `_validate_reply` (e.g. return it or
expose a helper that constructs the wire tool calls). The externally observable
wire/manager messages are unchanged — the `[:1]` truncation on what is *sent* is
preserved.

### Train-only gating

Plumb an `is_eval` flag to the session:

- Add `is_eval: bool = False` to the `Session` dataclass (`common.py:33`).
- Add an `is_eval` parameter to `open_session(...)` (`common.py:215`) and store
  it on the created `Session`.
- Pass it at the one call site,
  `examples/coding_agent_rl/generate.py:385`, which already has `evaluation` in
  scope: `open_session(..., is_eval=evaluation)`.

`OpenAIAdapter._validate_reply` returns `None` when `session.is_eval` is `True`,
so eval rollouts never retry and measure raw model output.

### Exhaustion metadata flag

Following the `ill_formed` precedent:

- Add `invalid_tool_call: bool = False` to the `TurnRecord` dataclass
  (`trajectory.py:28`).
- Set it via `dataclasses.replace` when retries are exhausted with a still-
  invalid verdict (see loop above).
- Surface it in sample metadata alongside `ill_formed` (`trajectory.py:510`) so
  downstream filters can drop or inspect rejection-failed turns.

## Error handling

- **Client disconnect** during `_respond` is unchanged (`common.py:377`): the
  turn is not recorded. The retry loop finishes before `_respond`, so a
  disconnect never interacts with retries.
- **Context overflow** early-exit in `call_sglang_generate` (`common.py:477`)
  returns an empty `TurnRecord` with `finish_reason="length"` and no tool calls;
  the validator sees no tool calls and returns `None`, so it breaks immediately
  (no wasted retries).
- **Validator raises** — treat a raising validator as a configuration error:
  let it propagate (fail fast) rather than silently disabling validation.
- **Turn cap** (`_check_turn_cap`, `common.py:290`) still bounds runaway
  sessions independently of retries.

## Limitations (documented, not mitigated)

- **Same-params retries** may reproduce the identical invalid call at low
  temperature; retries help mainly when sampling temperature is nonzero.
- **Rejection-sampling bias:** regenerating until a valid tool call is produced
  skews the recorded turn distribution away from the raw policy. For on-policy
  RL this is a known bias; it is acknowledged and documented here, not mitigated
  (no importance weighting or loss masking of retried turns).

## Testing

Adapter tests already stub `call_sglang_generate`, so the loop is driven
deterministically. Unit tests (CPU, `tests/`, `pytest.main([__file__])`
pattern):

1. **Retry then succeed** — invalid on attempt 1, valid on attempt 2. Assert 2
   generate calls; `record_turn` called once with the valid candidate; no
   `invalid_tool_call` flag.
2. **Exhaustion** — always invalid. Assert `max_retries + 1` generate calls;
   `record_turn` called once; `TurnRecord.invalid_tool_call is True`; warning
   logged.
3. **Valid first try** — validator returns `None`. Assert exactly 1 generate
   call.
4. **Eval bypass** — `open_session(is_eval=True)` + always-invalid validator.
   Assert 1 generate call, no retries, no flag.
5. **Disabled** — no `--tool-call-validator-path`. Assert `_validate_reply`
   returns `None`; 1 generate call; feature inert.
6. **Multi-call validation** — a turn with two tool calls where the *second* is
   invalid. Assert it's detected (guards the `[:1]` truncation trap).
7. **No side effects on rejected candidate** — assert trajectory state /
   turn-count is untouched until acceptance (bumps exactly once).
8. **Contract test** — feed the real `find_invalid_tool_call` a known-bad and
   known-good `response_dict` through the hook to confirm polarity (`None` =
   valid).

CI: register the new test in the CPU matrix in
`.github/workflows/pr-test.yml.j2` (regenerate via `generate_github_workflows.py`;
never edit the `.yml` directly), per the `add-tests-and-ci` skill.

## Files touched

- `slime/utils/arguments.py` — two new args.
- `slime/agent/adapters/common.py` — `Session.is_eval`, `open_session(is_eval=)`,
  `_validate_reply` base hook, retry loop in `_run_turn`, load validator at
  construction.
- `slime/agent/adapters/openai.py` — `_validate_reply` override; refactor
  `_build_reply_parts` to expose the full tool-call list.
- `slime/agent/trajectory.py` — `TurnRecord.invalid_tool_call`; surface in
  sample metadata.
- `examples/coding_agent_rl/generate.py` — pass `is_eval=evaluation` to
  `open_session`.
- `tests/` — new test module.
- `.github/workflows/pr-test.yml.j2` (+ regenerated `.yml`) — register the test.

## Out of scope

- AnthropicAdapter support (would need Anthropic wire-dict synthesis).
- Built-in JSON-schema validation (validator is fully custom/pluggable).
- Sampling perturbation on retry.
- Any mitigation of the rejection-sampling bias.
