"""In-sandbox OpenHands driver.

Runs under /opt/oh-env/bin/python inside the sandbox; imports the baked
OpenHands SDK. Reads oh_config.json (adapter URL, session id, tool allowlist,
fake-user flag, max iterations) and the task prompt, builds an
LLM/Agent/Conversation(LocalWorkspace) loop, and runs it to completion. Every
LLM turn dials back to slime's OpenAIAdapter over the OpenAI-compatible
base_url, so token capture is handled host-side.

build_tools() is intentionally SDK-import-free at module top: the tool name ->
(registering module, agent axis) routing is unit-tested on the host where the
SDK is absent. All openhands imports happen lazily inside functions.
"""

from __future__ import annotations

import json
import sys

# Tool name -> the import path of the module whose import calls register_tool for
# it. Default preset: file_editor/terminal/task_tracker. Legacy preset also
# registers str_replace_editor/execute_bash (only after that module is imported).
_TOOLS_AXIS_MODULES = {
    "file_editor": "openhands.tools.file_editor",
    "terminal": "openhands.tools.terminal",
    "task_tracker": "openhands.tools.task_tracker",
    "str_replace_editor": "openhands.tools.preset.legacy",
    "execute_bash": "openhands.tools.preset.legacy",
}

# Builtins live on Agent(include_default_tools=[...]) by class name, NOT in tools=.
_BUILTIN_AXIS = {
    "think": "ThinkTool",
    "finish": "FinishTool",
}


def _default_register_module(name: str) -> None:
    import importlib

    importlib.import_module(_TOOLS_AXIS_MODULES[name])


def _default_make_tool(name: str):
    from openhands.sdk.tool import Tool

    return Tool(name=name)


def build_tools(names, *, register_module=None, make_tool=None):
    """Map a flat allowlist onto OpenHands' two tool axes.

    Returns (tools, include_default): ``tools`` are Tool(name=...) objects for the
    Agent(tools=) axis (each name's registering module imported first);
    ``include_default`` is the ThinkTool/FinishTool subset for
    Agent(include_default_tools=). Unknown names raise ValueError (fail fast).

    register_module / make_tool are injectable for host-side unit tests; the
    defaults import the real SDK (only available inside the sandbox).
    """
    register_module = register_module or _default_register_module
    make_tool = make_tool or _default_make_tool

    tools = []
    include_default: list[str] = []
    for name in names:
        if name in _BUILTIN_AXIS:
            include_default.append(_BUILTIN_AXIS[name])
        elif name in _TOOLS_AXIS_MODULES:
            register_module(name)
            tools.append(make_tool(name))
        else:
            raise ValueError(
                f"unknown OpenHands tool name {name!r}; "
                f"known: {sorted(set(_TOOLS_AXIS_MODULES) | set(_BUILTIN_AXIS))}"
            )
    return tools, include_default


def _run_with_fake_user(conv, max_nudges: int = 100) -> None:
    """Minimal fake-user nudge loop (vendored, ~self-contained).

    OpenHands stops the run when the agent sends a plain message instead of using
    a tool. In eval/RL we want it to keep going until it calls the finish tool.
    After each run() returns without finishing, send a short nudge and run again,
    bounded by max_nudges.
    """
    from openhands.sdk.conversation.state import ConversationExecutionStatus

    nudge = (
        "Please continue working on the task with whatever approach you think is "
        "suitable. When you are done, call the finish tool."
    )
    for _ in range(max_nudges):
        conv.run()
        if conv.state.execution_status == ConversationExecutionStatus.FINISHED:
            return
        conv.send_message(nudge)


def main(config_path: str) -> int:
    from openhands.sdk import LLM, Agent, Conversation
    from openhands.sdk.workspace import LocalWorkspace

    with open(config_path) as f:
        cfg = json.load(f)
    with open("/home/agent/oh_prompt.txt") as f:
        prompt = f.read()

    tools, include_default = build_tools(cfg["tools"])
    llm = LLM(
        model="openai/" + cfg.get("model_label", "slime-actor"),
        base_url=cfg["adapter_url"] + "/v1",
        api_key=cfg["session_id"],
    )
    agent = Agent(
        llm=llm,
        tools=tools,
        include_default_tools=include_default,
        system_prompt_kwargs={"cli_mode": True},
    )
    conv = Conversation(
        agent=agent,
        workspace=LocalWorkspace(working_dir=cfg["workdir"]),
        max_iteration_per_run=int(cfg["max_iterations"]),
    )
    conv.send_message(prompt)
    if cfg.get("fake_user"):
        _run_with_fake_user(conv)
    else:
        conv.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
