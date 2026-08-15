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
import os
import sys
import tempfile

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


def _last_agent_outcome(conv) -> str:
    """Classify why conv.run() returned, from the tail of the event log.

    Status alone is ambiguous: the SDK reports FINISHED both when the agent
    calls the finish tool AND when it stops by sending a plain message. Only the
    latter is worth a fake-user nudge, so we walk back to the last agent turn:

    - "finished": last agent action was the finish tool -> genuinely done.
    - "plain_response": agent sent prose instead of a tool call -> nudge-eligible.
    - "other": last agent turn was some other tool call, or there is no agent
      turn to nudge -> treat as terminal.
    """
    from openhands.sdk.event import ActionEvent, MessageEvent
    from openhands.sdk.tool.builtins.finish import FinishAction

    for event in reversed(list(conv.state.events)):
        if isinstance(event, ActionEvent):
            return "finished" if isinstance(getattr(event, "action", None), FinishAction) else "other"
        if isinstance(event, MessageEvent):
            source = getattr(event, "source", None)
            if getattr(source, "value", source) == "agent":
                return "plain_response"
    return "other"


def _run_with_fake_user(conv, max_nudges: int = 100) -> None:
    """Fake-user nudge loop.

    OpenHands stops the run when the agent sends a plain message instead of using
    a tool. In eval/RL we want it to keep going until it calls the finish tool,
    so on that (and only that) case we send a short nudge and run again, bounded
    by max_nudges. Every other terminal state -- finish tool, ERROR, STUCK,
    max-iterations -- is left alone; nudging a dead conversation would just burn
    turns and hammer the adapter.
    """
    from openhands.sdk.conversation.state import ConversationExecutionStatus

    nudge = (
        "Please continue working on the task with whatever approach you think is "
        "suitable. When you are done, call the finish tool."
    )
    for _ in range(max_nudges):
        conv.run()
        # ERROR / STUCK / max-iterations report a non-FINISHED status: terminal.
        if conv.state.execution_status != ConversationExecutionStatus.FINISHED:
            return
        # FINISHED is ambiguous; only a plain agent message is nudge-eligible.
        if _last_agent_outcome(conv) != "plain_response":
            return
        conv.send_message(nudge)


def _isinstance_convertible(event) -> bool:
    from openhands.sdk.event.base import LLMConvertibleEvent

    return isinstance(event, LLMConvertibleEvent)


def events_to_trajectory(events) -> list:
    """Reconstruct the LLM message/tool-call trajectory from OpenHands events.

    Pure transform: keep only LLMConvertibleEvent records, fold them into chat
    messages via the SDK, and emit chat dicts with reasoning content retained.
    """
    from openhands.sdk.event.base import LLMConvertibleEvent

    convertible = [e for e in events if _isinstance_convertible(e)]
    messages = LLMConvertibleEvent.events_to_messages(convertible)
    return [m.model_copy(update={"send_reasoning_content": True}).to_chat_dict() for m in messages]


def write_trajectory(path: str, events) -> None:
    """Best-effort atomic dump of the converted trajectory to ``path``.

    Runs inside the sandbox as user ``agent``; any failure is swallowed so a
    trace problem never aborts an otherwise-complete run.
    """
    try:
        payload = events_to_trajectory(events)
        directory = os.path.dirname(path) or "."
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".oh_traj_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    except Exception as e:  # best-effort: never abort the run over a trace failure
        print(f"[oh_driver] trajectory persistence skipped: {type(e).__name__}: {e}", file=sys.stderr)


def main(config_path: str) -> int:
    from openhands.sdk import LLM, Agent, Conversation
    from openhands.sdk.workspace import LocalWorkspace

    with open(config_path) as f:
        cfg = json.load(f)

    # prompt_path defaults to the in-sandbox convention; override via config for
    # local (non-sandbox) runs.
    prompt_path = cfg.get("prompt_path", "/home/agent/oh_prompt.txt")
    with open(prompt_path) as f:
        prompt = f.read()

    tools, include_default = build_tools(cfg["tools"])

    # Two LLM construction paths:
    #  "llm" key present  → rich config dict, already resolved by the caller
    #                        (see OmegaConf.to_container + LLM.model_validate_json
    #                        in benchmarks/scaleswe/run_infer.py for the pattern).
    #  fallback           → slime-adapter triple (adapter_url / session_id /
    #                        model_label), the original in-sandbox path.
    if "llm" in cfg:
        llm = LLM.model_validate_json(json.dumps(cfg["llm"]))
    else:
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
    trajectory_path = cfg.get("trajectory_path")
    if trajectory_path:
        write_trajectory(trajectory_path, conv.state.events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
