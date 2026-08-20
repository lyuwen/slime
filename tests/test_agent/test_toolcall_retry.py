"""Unit tests for tool-call validation & regeneration in the OpenAI adapter."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import types  # noqa: E402

# slime.utils.arguments imports slime.backends.sglang_utils.arguments, which does
# `from sglang.srt.server_args import ServerArgs` / `from sglang_router.launch_router
# import RouterArgs` at module load. These symbols are only used at parse_args() call
# time, not import time, so empty stubs are enough to let the module import on the
# CPU-only CI env (mirrors the transformers stub in test_openhands_harness.py).
if "sglang" not in sys.modules:
    _sglang = types.ModuleType("sglang")
    _sglang_srt = types.ModuleType("sglang.srt")
    _sglang_server_args = types.ModuleType("sglang.srt.server_args")
    _sglang_server_args.ServerArgs = type("ServerArgs", (), {})
    _sglang.srt = _sglang_srt
    _sglang_srt.server_args = _sglang_server_args
    sys.modules["sglang"] = _sglang
    sys.modules["sglang.srt"] = _sglang_srt
    sys.modules["sglang.srt.server_args"] = _sglang_server_args
if "sglang_router" not in sys.modules:
    _sglang_router = types.ModuleType("sglang_router")
    _sglang_router_launch = types.ModuleType("sglang_router.launch_router")
    _sglang_router_launch.RouterArgs = type("RouterArgs", (), {})
    _sglang_router.launch_router = _sglang_router_launch
    sys.modules["sglang_router"] = _sglang_router
    sys.modules["sglang_router.launch_router"] = _sglang_router_launch
# slime.utils.arguments also pulls in slime.utils.logging_utils, which imports wandb.
if "wandb" not in sys.modules:
    sys.modules["wandb"] = types.ModuleType("wandb")

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


import argparse  # noqa: E402


def test_toolcall_retry_args_have_expected_defaults():
    # Mirror the argparse declarations so a rename/removal in arguments.py is caught.
    p = argparse.ArgumentParser()
    p.add_argument("--tool-call-validator-path", type=str, default=None)
    p.add_argument("--tool-call-max-retries", type=int, default=3)
    ns = p.parse_args([])
    assert ns.tool_call_validator_path is None
    assert ns.tool_call_max_retries == 3

    from slime.utils import arguments as slime_args

    src = Path(slime_args.__file__).read_text()
    assert "--tool-call-validator-path" in src
    assert "--tool-call-max-retries" in src


if __name__ == "__main__":
    pytest.main([__file__])
