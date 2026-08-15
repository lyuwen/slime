# OpenHands Trajectory Persistence Design

## Goal

Persist one inspectable OpenHands agent trajectory per coding-agent rollout without affecting rollout execution or grading. The saved JSON must include the reconstructed LLM message/tool-call history, the produced Git diff, and the SWE grading outcome.

## Scope

This change applies only to the OpenHands harness used by `examples/coding_agent_rl/generate.py`. It does not change the generic custom-generate function interface, rollout indexing, Claude Code or Codex harnesses, training samples, or reward semantics.

## Output Location and Identity

The host output root is configured with `SWE_TRAJECTORY_DIR`, defaulting to `trajectories`.

The existing `Sample` fields provide the available identity. The output path is:

```text
${SWE_TRAJECTORY_DIR}/{group_index}/{group_index}_{index}.json
```

`group_index` is used for the requested directory level because the actual slime `rollout_id` is not currently passed to custom generate functions. `index` is the existing globally unique sample index; it serves as the sample ID in the filename. The implementation will not modify the generic rollout pipeline solely to expose `rollout_id`.

## Sandbox Trajectory Extraction

`oh_driver.py` owns the live local OpenHands `Conversation`, so it performs event conversion immediately after the normal or fake-user conversation loop returns.

It will:

1. Filter `conv.state.events` to `LLMConvertibleEvent` instances.
2. Convert them with `LLMConvertibleEvent.events_to_messages()` so parallel actions from one LLM response are reconstructed as one assistant message.
3. Enable reasoning serialization on copied messages and call `to_chat_dict()`.
4. Extract tool definitions by scanning events for the first `SystemPromptEvent`; convert each `ToolDefinition` via `.to_openai_tool()`. If no `ToolDefinition` is found in any `SystemPromptEvent`, fall back to the initial `tools` list passed to `Agent`. The event list is preferred because the SDK injects built-in/default tools into `SystemPromptEvent` beyond what the caller supplies.
5. Serialize an intermediate JSON object `{"messages": [...], "tools": [...]}` — not a bare list.

The driver runs as the sandbox `agent` user. The intermediate file is therefore `/home/agent/oh_trajectory.json`, a location owned and writable by that user. The OpenHands harness supplies this path in `oh_config.json` as `trajectory_path`; the driver uses the configured path rather than embedding a second independent path convention.

The driver writes the intermediate file atomically by writing a sibling temporary file and replacing the destination. The parent directory already exists and is owned by `agent`; the driver does not create privileged directories or change ownership.

## Host Transfer and Finalization

The sandbox context in `generate.py` remains alive immediately after `OpenHandsHarness.run()` returns. This is the transfer boundary.

For the OpenHands path, `generate.py` will:

1. Run the harness.
2. Read `/home/agent/oh_trajectory.json` through the existing `Sandbox.read_file()` API while the sandbox is alive.
3. Capture `diff_text` through `swe.git_diff()` before leaving the sandbox context.
4. Run `swe.run_evaluation()` on the host after sandbox teardown.
5. Enrich the parsed intermediate trace with all final metadata.
6. Atomically write the final JSON to the configured host path.

No reverse transfer to the sandbox is needed because `reward` and `applied_cleanly` only become available after the agent sandbox has been released.

## Final JSON Schema

Each final trace is a JSON object with these fields:

```json
{
  "messages": [],
  "tools": [],
  "diff_text": "",
  "reward": 0.0,
  "applied_cleanly": false,
  "instance_id": "...",
  "group_index": 0,
  "index": 0,
  "session_id": "...",
  "agent_exit_code": 0
}
```

Field semantics:

- `messages`: OpenHands events reconstructed into OpenAI-compatible chat message dictionaries, including tool calls, tool results, reasoning content when available, the initial user prompt, and fake-user nudges.
- `tools`: OpenAI-format tool definitions extracted from the `SystemPromptEvent` event log (preferred, includes SDK-injected built-in tools), falling back to `ToolDefinition` instances in the initial `Agent(tools=)` list.
- `diff_text`: the Git diff captured from the task workspace after the agent exits.
- `reward`: the numeric reward returned by the configured SWE evaluator.
- `applied_cleanly`: whether the evaluator applied the patch successfully.
- `instance_id`: the SWE instance identifier.
- `group_index`: the existing dataset sample-group counter and directory key.
- `index`: the existing globally unique sample counter and filename ID.
- `session_id`: the adapter/OpenHands session identifier.
- `agent_exit_code`: the harness process outcome.

`group_index` and `index` may be `null` only for directly constructed samples outside the normal data-source path. In that exceptional case, the writer uses the literal `unknown` for the missing path component while retaining JSON `null` values in the document.

## Error Handling

Trajectory persistence is best-effort and must not alter rollout behavior.

- If sandbox conversion or writing fails, `oh_driver.py` logs the error to stderr but preserves its conversation exit behavior.
- If the sandbox trace is missing, unreadable, or malformed, `generate.py` logs a warning and skips the final trace write.
- If local directory creation or final JSON writing fails, `generate.py` logs a warning and continues returning the original rollout result.
- Trace failures never change reward, `applied_cleanly`, sample status, `remove_sample`, adapter cleanup, or the returned samples.
- Final host writes use a temporary sibling plus `os.replace()` so readers never observe a partially written JSON document.
- The temporary filename includes enough per-sample identity to avoid collisions among concurrent rollout tasks targeting the same directory.

## Code Boundaries

- `examples/coding_agent_rl/oh_driver.py`: convert local OpenHands events and atomically save the intermediate sandbox JSON.
- `slime/agent/harness/openhands.py`: define/pass the writable sandbox trajectory path through `oh_config.json`.
- `examples/coding_agent_rl/generate.py`: read the sandbox trace before teardown, enrich it after diff/evaluation, resolve the host path, and atomically persist the final JSON.
- Focused tests near the existing OpenHands driver/harness coding-agent tests: validate conversion, configuration, enrichment, path selection, and best-effort failures.

No new abstraction or base class is introduced; the implementation follows the existing OpenHands-specific harness boundary.

## Testing

Focused tests will verify:

1. Convertible OpenHands events produce the expected `messages` JSON, including batched tool calls and reasoning-content serialization.
2. `tools_to_trajectory()` extracts `ToolDefinition` entries from `SystemPromptEvent` and skips non-`ToolDefinition` entries; falls back to the initial tools list when no `SystemPromptEvent` ToolDefinitions are found.
3. `write_trajectory()` writes an object `{"messages": [...], "tools": [...]}`, not a bare list.
4. `OpenHandsHarness.write_config()` supplies `/home/agent/oh_trajectory.json` as `trajectory_path`.
5. The driver writes only beneath `/home/agent/` under its normal `agent` execution context.
6. A successful sandbox read plus grading creates `${SWE_TRAJECTORY_DIR}/{group_index}/{group_index}_{index}.json` with the complete schema including `tools`.
7. `_read_sandbox_trajectory()` accepts only a dict with `messages` and `tools` lists; rejects bare lists, missing keys, and non-list field values.
8. Missing and malformed sandbox traces only emit warnings and do not alter rollout results.
9. Local directory or atomic-write failures only emit warnings and do not alter rollout results.
10. Existing OpenHands driver and harness tests remain passing.

Verification will run the focused unit tests and formatting/lint checks covering modified files. Full GPU training and SWE evaluation are outside the local unit-test scope.
