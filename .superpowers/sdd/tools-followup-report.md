# Tools-list persistence follow-up report

## Summary

Implemented tools-list persistence for the OpenHands trajectory pipeline.
The intermediate sandbox JSON format changed from a bare list to an object
`{"messages": [...], "tools": [...]}`. The final host JSON now includes `tools`.

## Commands and results

### Red phase — driver tests

```
python -m pytest tests/test_agent/test_oh_driver.py -q
# 3 failed, 5 passed — expected: AttributeError for missing tools_to_trajectory,
# TypeError for missing tools_to_trajectory attr in monkeypatch
```

### Red phase — persistence tests

```
python -m pytest tests/test_agent/test_trajectory_persistence.py -q
# 6 failed, 2 passed — expected:
#   TypeError: _persist_trajectory() got unexpected keyword argument 'tools'
#   AssertionError: object-shaped read accepted bare list (old reader)
#   AssertionError: bare list not rejected (warning_raises variant)
```

### Implementation

Changed files:
- `examples/coding_agent_rl/oh_driver.py`
  - Added `tools_to_trajectory(events, initial_tools) -> list[dict]`
  - Updated `write_trajectory(path, events, initial_tools=None)` to write
    `{"messages": ..., "tools": ...}` (object, not bare list)
  - Updated `main()` to call `write_trajectory(..., agent.tools)`
- `examples/coding_agent_rl/generate.py`
  - `_read_sandbox_trajectory` now returns `dict | None`, validates object
    shape with `messages` and `tools` lists; rejects bare lists, missing keys,
    non-list fields; warning message uses "unexpected shape"
  - `_persist_trajectory` gains `tools: list` kwarg, writes it to final JSON
  - Call site in `generate()` renamed `traj_messages` → `traj_data`, passes
    `traj_data["messages"]` and `traj_data["tools"]` to `_persist_trajectory`

### Green phase

```
python -m pytest tests/test_agent/test_oh_driver.py tests/test_agent/test_trajectory_persistence.py -q
# 16 passed
```

### Full test_agent suite

```
python -m pytest tests/test_agent/ -q
# 90 passed, 1 skipped (pre-existing skip in test_adapters.py)
```

### Lint

```
black --check --line-length 119 <4 files>   # 4 files unchanged
isort --check-only --profile black <4 files> # exit 0
ruff check <4 files>                         # All checks passed
```

### Docs updated

- `docs/superpowers/specs/2026-08-15-openhands-trajectory-persistence-design.md`
  - Step 4 in extraction sequence now documents `tools_to_trajectory` logic
    (SystemPromptEvent preference, ToolDefinition filter, fallback)
  - Step 5 explicitly states intermediate format is object not bare list
  - Final JSON schema gains `tools` field with description
  - Testing section updated to 10 points covering new tool extraction tests
- `docs/superpowers/plans/2026-08-15-openhands-trajectory-persistence.md`
  - Goal line includes `tools`
  - Architecture sentence updated to mention object format
  - Task 1 interface adds `tools_to_trajectory` and updates `write_trajectory`
    signature; `main()` call updated
  - Task 3 interface updates `_read_sandbox_trajectory` return type and shape
    contract; `_persist_trajectory` signature gains `tools`

## Concerns

None blocking. One note: the `_read_sandbox_trajectory` warning message was
changed from "unexpected top-level type" to "unexpected shape" — the two
existing best-effort tests that check for that string were updated to match.
The test that was previously passing a dict `{"messages": []}` to test the
old type-check now correctly fails with the new shape check (missing `tools`
key), and the new test explicitly exercises that plus bare-list rejection and
non-list-field rejection.
