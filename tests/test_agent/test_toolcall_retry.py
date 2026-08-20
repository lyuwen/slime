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


import dataclasses  # noqa: E402

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from slime.agent.adapters.common import TurnRecord as _TR  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

# A tool schema the adapter will advertise; parsing needs a known tool name.
_TOOLS = [
    {"type": "function", "function": {"name": "good_tool", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "bad_tool", "parameters": {"type": "object", "properties": {}}}},
]


def _reject_bad_tool(response_dict):
    """Validator: invalid iff any tool call is named 'bad_tool'."""
    for choice in response_dict.get("choices") or []:
        for call in (choice.get("message") or {}).get("tool_calls") or []:
            fn = call.get("function") or {}
            if fn.get("name") == "bad_tool":
                return (fn["name"], fn.get("arguments", ""))
    return None


def _scripted_generate(tokenizer, texts):
    """Drop-in for common.call_sglang_generate yielding one scripted text per call.

    Records how many times it was invoked on the returned closure's `.calls`.
    """
    queue = list(texts)
    state = {"calls": 0}

    async def _fake(prompt_ids, session, body, *, adapter, session_id=None):
        state["calls"] += 1
        assert queue, "unexpected generate call (script exhausted)"
        text = queue.pop(0)
        output_ids = tokenizer.encode(text)
        return _TR(
            prompt_ids=list(prompt_ids),
            output_ids=output_ids,
            finish_reason="stop",
            output_log_probs=[0.0] * len(output_ids),
        )

    _fake.state = state
    return _fake


def _xml_call(name):
    # parse_xml_tool_uses fallback shape; no parameters needed for these tools.
    return f"<tool_call><function={name}></function></tool_call>"


async def _run_one_turn(adapter, sid):
    client = TestClient(TestServer(adapter.app))
    await client.start_server()
    try:
        resp = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {sid}"},
            json={"model": "m", "max_tokens": 8, "tools": _TOOLS,
                  "messages": [{"role": "user", "content": "go"}]},
        )
        await resp.json()
    finally:
        await client.close()
    return await adapter.finish_session(sid, base_sample=Sample(index=0, prompt=""), reward=1.0)


def _make_adapter(tok, validator=None, max_retries=3):
    return openai.OpenAIAdapter(
        tokenizer=tok,
        sglang_url="http://unused",
        tool_call_validator=validator,
        tool_call_max_retries=max_retries,
    )


def test_retry_then_succeed_records_valid_candidate(monkeypatch):
    async def run_case():
        tok = FakeTokenizer()
        gen = _scripted_generate(tok, [_xml_call("bad_tool"), _xml_call("good_tool")])
        monkeypatch.setattr(common, "call_sglang_generate", gen)
        adapter = _make_adapter(tok, validator=_reject_bad_tool)
        adapter.open_session("s1")
        samples = await _run_one_turn(adapter, "s1")
        assert gen.state["calls"] == 2  # one retry
        assert samples and all(s.metadata.get("invalid_tool_call") is False for s in samples)

    asyncio.run(run_case())


def test_exhaustion_accepts_last_and_flags_metadata(monkeypatch):
    async def run_case():
        tok = FakeTokenizer()
        gen = _scripted_generate(tok, [_xml_call("bad_tool")] * 4)  # 1 + 3 retries
        monkeypatch.setattr(common, "call_sglang_generate", gen)
        adapter = _make_adapter(tok, validator=_reject_bad_tool, max_retries=3)
        adapter.open_session("s2")
        samples = await _run_one_turn(adapter, "s2")
        assert gen.state["calls"] == 4
        assert samples and any(s.metadata.get("invalid_tool_call") is True for s in samples)

    asyncio.run(run_case())


def test_valid_first_try_no_retry(monkeypatch):
    async def run_case():
        tok = FakeTokenizer()
        gen = _scripted_generate(tok, [_xml_call("good_tool")])
        monkeypatch.setattr(common, "call_sglang_generate", gen)
        adapter = _make_adapter(tok, validator=_reject_bad_tool)
        adapter.open_session("s3")
        await _run_one_turn(adapter, "s3")
        assert gen.state["calls"] == 1

    asyncio.run(run_case())


def test_eval_session_bypasses_retry(monkeypatch):
    async def run_case():
        tok = FakeTokenizer()
        gen = _scripted_generate(tok, [_xml_call("bad_tool")])
        monkeypatch.setattr(common, "call_sglang_generate", gen)
        adapter = _make_adapter(tok, validator=_reject_bad_tool)
        adapter.open_session("s4", is_eval=True)
        samples = await _run_one_turn(adapter, "s4")
        assert gen.state["calls"] == 1  # no retries in eval
        # accepted as-is; flag not set because validation was skipped
        assert samples and all(s.metadata.get("invalid_tool_call") is False for s in samples)

    asyncio.run(run_case())


def test_disabled_when_no_validator(monkeypatch):
    async def run_case():
        tok = FakeTokenizer()
        gen = _scripted_generate(tok, [_xml_call("bad_tool")])
        monkeypatch.setattr(common, "call_sglang_generate", gen)
        adapter = _make_adapter(tok, validator=None)
        adapter.open_session("s5")
        await _run_one_turn(adapter, "s5")
        assert gen.state["calls"] == 1  # feature inert

    asyncio.run(run_case())


def test_generate_example_wires_validator_and_eval_flag():
    gen_src = (REPO_ROOT / "examples" / "coding_agent_rl" / "generate.py").read_text()
    # validator loaded from the arg and passed to the adapter
    assert "tool_call_validator_path" in gen_src
    assert "tool_call_validator=" in gen_src
    assert "tool_call_max_retries=" in gen_src
    # eval flag threaded into the session
    assert "is_eval=evaluation" in gen_src


def _find_invalid_tool_call_like_reference(response_dict):
    """Mirror of openhands.sdk ... find_invalid_tool_call polarity:
    None if all tool calls have JSON-object arguments, else (name, raw_args)."""
    import json as _json

    for choice in response_dict.get("choices") or []:
        for call in (choice.get("message") or {}).get("tool_calls") or []:
            fn = call.get("function") or {}
            name, args = fn.get("name"), fn.get("arguments")
            try:
                parsed = _json.loads(args) if isinstance(args, str) else args
            except (ValueError, TypeError):
                return (name, args if isinstance(args, str) else repr(args))
            if not isinstance(parsed, dict):
                return (name, args if isinstance(args, str) else repr(args))
    return None


def test_validate_reply_matches_reference_polarity():
    tok = FakeTokenizer()
    adapter = _make_adapter(tok, validator=_find_invalid_tool_call_like_reference)
    session = common.Session(is_eval=False)

    good = ParsedModelOutput(reasoning="", text="", tool_uses=[{"name": "good_tool", "input": {"a": 1}}], ill_formed=False)
    assert adapter._validate_reply(good, session) is None

    # A tool_use whose input isn't a dict becomes {"_raw_arguments": "..."} in the
    # wire call, which IS a JSON object -> reference treats it as valid.
    # To exercise the invalid branch, feed a raw non-JSON arguments string directly.
    bad_dict = {"choices": [{"message": {"tool_calls": [
        {"function": {"name": "good_tool", "arguments": "{not json"}}
    ]}}]}
    verdict = _find_invalid_tool_call_like_reference(bad_dict)
    assert verdict is not None and verdict[0] == "good_tool"


if __name__ == "__main__":
    pytest.main([__file__])
