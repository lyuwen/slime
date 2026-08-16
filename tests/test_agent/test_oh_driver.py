"""Unit tests for the in-sandbox OpenHands driver's host-testable logic.

Only build_tools() is exercised here: it is pure name->(module, axis) routing and
must run on the host where the OpenHands SDK is NOT installed, so the driver
imports the SDK lazily and build_tools takes an injectable tool factory +
registrar. The full LLM/Agent/Conversation loop runs only inside the sandbox and
is covered by the GPU e2e follow-up, not this CPU test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.coding_agent_rl import oh_driver  # noqa: E402


def _fakes():
    registered = []
    made = []

    def register(name):
        registered.append(name)

    def make_tool(name):
        made.append(name)
        return f"Tool({name})"

    return registered, made, register, make_tool


def test_build_tools_default_splits_axes():
    registered, made, register, make_tool = _fakes()
    tools, include_default = oh_driver.build_tools(
        ["file_editor", "terminal", "task_tracker", "think", "finish"],
        register_module=register,
        make_tool=make_tool,
    )
    assert tools == ["Tool(file_editor)", "Tool(terminal)", "Tool(task_tracker)"]
    assert include_default == ["ThinkTool", "FinishTool"]
    # each tools=-axis name registered its module first
    assert registered == ["file_editor", "terminal", "task_tracker"]


def test_build_tools_legacy_names():
    registered, made, register, make_tool = _fakes()
    tools, include_default = oh_driver.build_tools(
        ["str_replace_editor", "execute_bash", "task_tracker", "finish"],
        register_module=register,
        make_tool=make_tool,
    )
    assert tools == ["Tool(str_replace_editor)", "Tool(execute_bash)", "Tool(task_tracker)"]
    assert include_default == ["FinishTool"]
    # legacy names route through the legacy preset registrar
    assert registered == ["str_replace_editor", "execute_bash", "task_tracker"]


def test_build_tools_unknown_name_raises():
    _, _, register, make_tool = _fakes()
    try:
        oh_driver.build_tools(["not_a_tool"], register_module=register, make_tool=make_tool)
    except ValueError as e:
        assert "not_a_tool" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown tool name")


class _FakeMsg:
    def __init__(self, payload):
        self._payload = payload

    def model_copy(self, update=None):
        assert update == {"send_reasoning_content": True}
        return self

    def to_chat_dict(self):
        return self._payload


def test_events_to_trajectory_converts_and_flags_reasoning(monkeypatch):
    import types

    convertible = [object(), object()]
    converted = []

    def events_to_messages(events):
        converted.append(events)
        return [_FakeMsg({"role": "assistant", "content": "hi"})]

    fake_base = types.SimpleNamespace(
        LLMConvertibleEvent=types.SimpleNamespace(
            events_to_messages=events_to_messages,
        )
    )
    # only the two convertible objects are instances; a plain int is filtered out
    monkeypatch.setitem(sys.modules, "openhands.sdk.event.base", fake_base)
    monkeypatch.setattr(oh_driver, "_isinstance_convertible", lambda e: e in convertible, raising=False)
    out = oh_driver.events_to_trajectory(convertible + [123])
    assert converted == [convertible]
    assert out == [{"role": "assistant", "content": "hi"}]


def test_tools_to_trajectory_prefers_system_prompt_event_tools(monkeypatch):
    import types

    class ToolDefinition:
        def __init__(self, name):
            self.name = name

        def to_openai_tool(self):
            return {"type": "function", "function": {"name": self.name}}

    class SystemPromptEvent:
        def __init__(self, tools):
            self.tools = tools

    monkeypatch.setitem(sys.modules, "openhands.sdk.event", types.SimpleNamespace(SystemPromptEvent=SystemPromptEvent))
    monkeypatch.setitem(sys.modules, "openhands.sdk.tool", types.SimpleNamespace(ToolDefinition=ToolDefinition))
    event_tools = [ToolDefinition("builtin"), object()]
    initial_tools = [ToolDefinition("initial")]
    assert oh_driver.tools_to_trajectory([SystemPromptEvent(event_tools)], initial_tools) == [
        {"type": "function", "function": {"name": "builtin"}}
    ]


def test_tools_to_trajectory_falls_back_to_initial_tool_definitions(monkeypatch):
    import types

    class ToolDefinition:
        def __init__(self, name):
            self.name = name

        def to_openai_tool(self):
            return {"type": "function", "function": {"name": self.name}}

    monkeypatch.setitem(sys.modules, "openhands.sdk.event", types.SimpleNamespace(SystemPromptEvent=type("S", (), {})))
    monkeypatch.setitem(sys.modules, "openhands.sdk.tool", types.SimpleNamespace(ToolDefinition=ToolDefinition))
    assert oh_driver.tools_to_trajectory([], [ToolDefinition("initial"), object()]) == [
        {"type": "function", "function": {"name": "initial"}}
    ]


def test_write_trajectory_atomic_and_best_effort(tmp_path, monkeypatch):
    monkeypatch.setattr(oh_driver, "events_to_trajectory", lambda evs: [{"role": "user", "content": "x"}])
    monkeypatch.setattr(oh_driver, "tools_to_trajectory", lambda evs, tools: [{"type": "function"}])
    dest = tmp_path / "oh_trajectory.json"
    oh_driver.write_trajectory(str(dest), [object()], [object()])
    assert json.loads(dest.read_text()) == {
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "function"}],
    }
    assert list(tmp_path.glob(".oh_traj_*.tmp")) == []

    # os.replace failure is swallowed, leaves the prior file intact, and cleans the temp sibling
    monkeypatch.setattr(oh_driver.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("replace failed")))
    oh_driver.write_trajectory(str(dest), [object()], [object()])
    assert json.loads(dest.read_text()) == {
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "function"}],
    }
    assert list(tmp_path.glob(".oh_traj_*.tmp")) == []
    monkeypatch.undo()

    # best-effort: a conversion failure must not raise
    monkeypatch.setattr(oh_driver, "events_to_trajectory", lambda evs: (_ for _ in ()).throw(RuntimeError("boom")))
    oh_driver.write_trajectory(str(dest), [object()], [object()])  # no exception


def test_write_trajectory_is_best_effort_when_diagnostic_raises(tmp_path, monkeypatch):
    # a conversion failure AND a failing stderr diagnostic must not raise
    monkeypatch.setattr(oh_driver, "events_to_trajectory", lambda evs: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(
        oh_driver, "print", lambda *a, **k: (_ for _ in ()).throw(OSError("stderr failed")), raising=False
    )
    oh_driver.write_trajectory(str(tmp_path / "oh_trajectory.json"), [object()])  # no exception


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
