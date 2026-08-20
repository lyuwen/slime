"""Turn scorer + custom advantage for coding-agent tool-call shaping."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _asst(tool_name, arguments):
    return {
        "role": "assistant",
        "tool_calls": [{"type": "function", "function": {"name": tool_name, "arguments": arguments}}],
    }


def test_count_errors_clean_finish():
    """A well-formed finish call with a message → 0 errors."""
    from examples.coding_agent_rl.turn_shaping import count_turn_toolcall_errors

    msg = _asst("finish", {"message": "done"})
    assert count_turn_toolcall_errors(msg, tool_response=None) == 0


def test_count_errors_malformed_finish():
    """finish with no message → 1 error (missing_message)."""
    from examples.coding_agent_rl.turn_shaping import count_turn_toolcall_errors

    msg = _asst("finish", {})
    assert count_turn_toolcall_errors(msg, tool_response=None) >= 1


def test_count_errors_no_tool_call():
    """Assistant turn with no tool call → 0."""
    from examples.coding_agent_rl.turn_shaping import count_turn_toolcall_errors

    assert count_turn_toolcall_errors({"role": "assistant", "content": "hi"}, None) == 0


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
