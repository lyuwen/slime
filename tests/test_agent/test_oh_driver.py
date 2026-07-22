"""Unit tests for the in-sandbox OpenHands driver's host-testable logic.

Only build_tools() is exercised here: it is pure name->(module, axis) routing and
must run on the host where the OpenHands SDK is NOT installed, so the driver
imports the SDK lazily and build_tools takes an injectable tool factory +
registrar. The full LLM/Agent/Conversation loop runs only inside the sandbox and
is covered by the GPU e2e follow-up, not this CPU test.
"""

from __future__ import annotations

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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
