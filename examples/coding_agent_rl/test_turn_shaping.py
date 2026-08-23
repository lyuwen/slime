"""Turn scorer + custom advantage for coding-agent tool-call shaping."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _asst(tool_name, arguments, tc_id="call_0"):
    return {
        "role": "assistant",
        "tool_calls": [{"id": tc_id, "type": "function", "function": {"name": tool_name, "arguments": arguments}}],
    }


def test_count_errors_clean_finish():
    """A well-formed finish call with a message → 0 errors."""
    from examples.coding_agent_rl.turn_shaping import count_turn_toolcall_errors

    msg = _asst("finish", {"message": "done"})
    assert count_turn_toolcall_errors(msg, None) == 0


def test_count_errors_malformed_finish():
    """finish with no message → 1 error (missing_message)."""
    from examples.coding_agent_rl.turn_shaping import count_turn_toolcall_errors

    msg = _asst("finish", {})
    assert count_turn_toolcall_errors(msg, None) >= 1


def test_count_errors_no_tool_call():
    """Assistant turn with no tool call → 0."""
    from examples.coding_agent_rl.turn_shaping import count_turn_toolcall_errors

    assert count_turn_toolcall_errors({"role": "assistant", "content": "hi"}, None) == 0


def test_count_errors_multi_call_per_id_matching():
    """P0.3: each tool call is scored against ITS OWN response, matched by
    tool_call_id — not the first tool child for all calls.

    Two calls in one turn: "a" is a clean str_replace_editor edit, "b" is a
    malformed str_replace_editor whose own response reports old_str_not_found.
    Correct per-id matching attributes exactly 1 error (to "b"). The OLD
    first-child logic passed call "a"'s clean response to every check, so "b"
    would score 0 and the total would be 0 — this test fails against it.
    """
    from examples.coding_agent_rl.turn_shaping import count_turn_toolcall_errors

    msg = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "a",
                "type": "function",
                "function": {"name": "str_replace_editor", "arguments": {"command": "str_replace", "path": "/x"}},
            },
            {
                "id": "b",
                "type": "function",
                "function": {"name": "str_replace_editor", "arguments": {"command": "str_replace", "path": "/y"}},
            },
        ],
    }
    responses_by_id = {
        "a": {"role": "tool", "tool_call_id": "a", "content": "The file /x has been edited successfully."},
        "b": {
            "role": "tool",
            "tool_call_id": "b",
            "content": "No replacement was performed, old_str `foo` did not appear verbatim in /y",
        },
    }
    assert count_turn_toolcall_errors(msg, responses_by_id) == 1


def test_count_errors_failing_command_not_a_protocol_error():
    """P0.4: an execute_bash with a valid invocation whose response is a runtime
    failure (contains "Error:") scores 0 — ``other_error`` is observation-derived,
    not a protocol violation. A real protocol violation (reset=True + is_input=True)
    still counts.
    """
    from examples.coding_agent_rl.turn_shaping import count_turn_toolcall_errors

    failing = _asst("execute_bash", {"command": "pytest"}, tc_id="c0")
    fail_resp = {"c0": {"role": "tool", "tool_call_id": "c0", "content": "Error: 1 test failed (exit code 1)"}}
    assert count_turn_toolcall_errors(failing, fail_resp) == 0

    violation = _asst("execute_bash", {"command": "x", "reset": True, "is_input": True}, tc_id="c1")
    assert count_turn_toolcall_errors(violation, None) == 1


def test_compute_advantage_adds_shaping():
    """compute_advantage adds the per-token shaping onto GRPO returns."""
    import torch

    from examples.coding_agent_rl.turn_shaping import compute_advantage

    class _Args:
        advantage_estimator = "grpo"
        kl_coef = 0.0
        use_rollout_logprobs = False
        custom_advantage_function_path = "x"

    kl = [torch.zeros(3), torch.zeros(2)]
    rollout_data = {
        "rewards": [1.0, -1.0],
        "kl": kl,
        "log_probs": [torch.zeros(3), torch.zeros(2)],
        "rollout_log_probs": [torch.zeros(3), torch.zeros(2)],
        "toolcall_turn_shaping": [torch.tensor([0.0, -0.5, -0.5]), torch.tensor([0.0, 0.0])],
        "values": None,
        "response_lengths": [3, 2],
        "loss_masks": [torch.ones(3), torch.ones(2)],
        "total_lengths": [5, 4],
    }
    compute_advantage(_Args(), rollout_data)
    adv = rollout_data["advantages"]
    # sample 0: reward 1.0 broadcast + shaping [0,-0.5,-0.5]
    assert torch.allclose(adv[0], torch.tensor([1.0, 0.5, 0.5]))
    # sample 1: reward -1.0 broadcast + shaping [0,0]
    assert torch.allclose(adv[1], torch.tensor([-1.0, -1.0]))


def test_compute_advantage_without_shaping_key():
    """Absent shaping key → plain GRPO returns."""
    import torch

    from examples.coding_agent_rl.turn_shaping import compute_advantage

    class _Args:
        advantage_estimator = "grpo"
        kl_coef = 0.0
        use_rollout_logprobs = False
        custom_advantage_function_path = "x"

    rollout_data = {
        "rewards": [1.0],
        "kl": [torch.zeros(3)],
        "log_probs": [torch.zeros(3)],
        "rollout_log_probs": [torch.zeros(3)],
        "values": None,
        "response_lengths": [3],
        "loss_masks": [torch.ones(3)],
        "total_lengths": [5],
    }
    compute_advantage(_Args(), rollout_data)
    assert torch.allclose(rollout_data["advantages"][0], torch.tensor([1.0, 1.0, 1.0]))


def test_shaping_config_rejects_invalid_values(monkeypatch):
    """resolve_shaping_config must reject non-finite / negative beta and budget:
    a negative budget flips the penalty sign, beta=inf yields NaN shaping, and a
    negative beta silently enables shaping with a flipped sign (the gate is
    beta != 0.0). All must raise ValueError.
    """
    from examples.coding_agent_rl.generate import resolve_shaping_config

    for bad_beta in ("-1", "inf", "nan"):
        monkeypatch.setenv("SWE_TOOLCALL_SHAPING_BETA", bad_beta)
        monkeypatch.setenv("SWE_TOOLCALL_SHAPING_BUDGET", "1.0")
        with pytest.raises(ValueError):
            resolve_shaping_config()

    # invalid budget with a valid (enabling) beta
    monkeypatch.setenv("SWE_TOOLCALL_SHAPING_BETA", "0.3")
    monkeypatch.setenv("SWE_TOOLCALL_SHAPING_BUDGET", "-1")
    with pytest.raises(ValueError):
        resolve_shaping_config()


def test_shaping_config_from_env(monkeypatch):
    """CONFIG reads beta/budget from SWE_ env vars; scorer built only when beta!=0."""

    # beta == 0 → no scorer
    monkeypatch.setenv("SWE_TOOLCALL_SHAPING_BETA", "0.0")
    from examples.coding_agent_rl.generate import resolve_shaping_config

    beta, budget, scorer = resolve_shaping_config()
    assert beta == 0.0
    assert scorer is None

    # beta != 0 → scorer built
    monkeypatch.setenv("SWE_TOOLCALL_SHAPING_BETA", "0.3")
    monkeypatch.setenv("SWE_TOOLCALL_SHAPING_BUDGET", "2.0")
    beta, budget, scorer = resolve_shaping_config()
    assert beta == 0.3
    assert budget == 2.0
    assert callable(scorer)


if __name__ == "__main__":
    pytest.main([__file__])
