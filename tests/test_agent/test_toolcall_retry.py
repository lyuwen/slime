"""Unit tests for tool-call validation & regeneration in the OpenAI adapter."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_agent._fakes import FakeTokenizer  # noqa: E402

from slime.agent.adapters import common, openai  # noqa: E402


def test_open_session_defaults_is_eval_false_and_accepts_flag():
    adapter = openai.OpenAIAdapter(tokenizer=FakeTokenizer(), sglang_url="http://x")
    adapter.open_session("s-train")
    adapter.open_session("s-eval", is_eval=True)
    assert adapter.store["s-train"].is_eval is False
    assert adapter.store["s-eval"].is_eval is True


from slime.agent.trajectory import TurnRecord  # noqa: E402


def test_turn_record_has_invalid_tool_call_default_false():
    tr = TurnRecord(prompt_ids=[1], output_ids=[2], finish_reason="stop")
    assert tr.invalid_tool_call is False
    tr2 = TurnRecord(prompt_ids=[1], output_ids=[2], finish_reason="stop", invalid_tool_call=True)
    assert tr2.invalid_tool_call is True


from slime.agent.parsing import ParsedModelOutput  # noqa: E402


def test_wire_tool_calls_returns_all_calls_as_json_string_args():
    parsed = ParsedModelOutput(
        reasoning="",
        text="",
        tool_uses=[
            {"name": "a", "input": {"x": 1}},
            {"name": "b", "input": {"y": 2}},
        ],
        ill_formed=False,
    )
    calls = openai._wire_tool_calls(parsed)
    assert [c["function"]["name"] for c in calls] == ["a", "b"]
    # arguments must be a JSON string (OpenAI wire shape), not a dict
    assert calls[0]["function"]["arguments"] == '{"x": 1}'
    assert calls[1]["function"]["arguments"] == '{"y": 2}'


if __name__ == "__main__":
    pytest.main([__file__])
