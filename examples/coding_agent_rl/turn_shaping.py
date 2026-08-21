"""Per-turn tool-call reward shaping for the coding-agent RL rollout.

Two halves, both living here so slime core never imports the annotator:

* Rollout side: ``make_turn_scorer`` returns a callback that scores one
  generated assistant turn (a ``slime.agent.trajectory.MessageNode``) by the
  number of tool-call errors the external ``toolcall_annotation`` package
  detects. ``TrajectoryManager`` applies ``-beta`` and the per-trajectory
  budget cap and writes the dense vector to ``Sample.metadata``.
* Train side: ``compute_advantage`` (wired via
  ``--custom-advantage-function-path``) adds the per-token shaping vector onto
  GRPO returns.

The annotator is imported lazily so runs with the feature off (beta == 0, the
default) never require it installed.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

_TOOL_CHECKS = None  # lazily populated dispatch table


def _load_checks():
    """Import the external annotator's pure check functions, once.

    Raises a clear error if the package is missing while the feature is enabled.
    """
    global _TOOL_CHECKS
    if _TOOL_CHECKS is not None:
        return _TOOL_CHECKS
    try:
        from toolcall_annotation.annotators import toolcall_correctness_impl as impl
    except ImportError as e:  # pragma: no cover - environment-specific
        raise ImportError(
            "toolcall-annotation is required for tool-call reward shaping "
            "(SWE_TOOLCALL_SHAPING_BETA != 0). Install it on every Ray worker: "
            "pip install toolcall-annotation"
        ) from e
    _TOOL_CHECKS = impl
    return _TOOL_CHECKS


def _as_wire_tool_call(tc: dict) -> dict:
    """Adapt slime's canonical tool-call (arguments as dict) to the annotator's
    wire shape (function.arguments as a JSON string)."""
    fn = tc.get("function", {}) or {}
    args = fn.get("arguments", {})
    if not isinstance(args, str):
        args = json.dumps(args)
    return {"id": tc.get("id"), "function": {"name": fn.get("name"), "arguments": args}}


def count_turn_toolcall_errors(assistant_message: dict, tool_response: dict | None) -> int:
    """Total detected tool-call errors across one assistant turn's tool calls.

    Dispatches each tool call to the matching annotator ``check_*`` and sums the
    number of error types returned. Unknown tools and turns without tool calls
    contribute 0.
    """
    if not assistant_message:
        return 0
    tool_calls = assistant_message.get("tool_calls") or []
    if not tool_calls:
        return 0

    impl = _load_checks()
    total = 0
    for tc in tool_calls:
        wire = _as_wire_tool_call(tc)
        name = wire["function"]["name"]
        # JSON validity applies to every tool call.
        _, args_valid = impl.parse_arguments(wire)
        if not args_valid:
            total += 1
            continue
        if name == "task_tracker":
            total += len(impl.check_task_tracker(wire, tool_response))
        elif name == "finish":
            total += len(impl.check_finish(wire, tool_response))
        elif name == "think":
            total += len(impl.check_think(wire, tool_response))
        elif name == "str_replace_editor":
            total += len(impl.check_str_replace_editor(wire, tool_response))
        elif name == "execute_bash":
            res = impl.check_execute_bash(wire, tool_response, None)
            total += len(res.get("errors", []))
        # unknown tools: no check, 0
    return total


def make_turn_scorer():
    """Return a callback scoring one generated assistant turn node.

    The callback takes a ``MessageNode`` and returns its errored-tool-call count.
    It locates the turn's tool response as a ``tool`` child node when present
    (the follow-up turn mounts it below the assistant); otherwise scores against
    ``None`` (the annotator tolerates a missing response).
    """

    def score(node) -> int:
        assistant_message = node.message or {}
        tool_response = None
        for child in getattr(node, "children", []) or []:
            if child.role == "tool" and child.message is not None:
                tool_response = child.message
                break
        return count_turn_toolcall_errors(assistant_message, tool_response)

    return score


def compute_advantage(args, rollout_data) -> None:
    """Custom advantage: GRPO returns plus the per-token tool-call shaping vector.

    Wired via ``--custom-advantage-function-path``. Called by
    ``compute_advantages_and_returns`` AFTER KL is computed; must set
    ``rollout_data['advantages']`` and ``rollout_data['returns']``.
    """
    import torch

    from slime.utils.ppo_utils import get_grpo_returns

    kl = rollout_data["kl"]
    rewards = torch.tensor(rollout_data["rewards"], dtype=torch.float32, device=kl[0].device)
    returns = get_grpo_returns(rewards, kl)

    shaping = rollout_data.get("toolcall_turn_shaping")
    if shaping is not None:
        for i in range(len(returns)):
            s = shaping[i]
            if not isinstance(s, torch.Tensor):
                s = torch.tensor(s, dtype=torch.float32, device=returns[i].device)
            returns[i] = returns[i] + s.to(device=returns[i].device, dtype=returns[i].dtype)

    rollout_data["returns"] = returns
    rollout_data["advantages"] = [r for r in returns]
