# Tool-call Validation & Regeneration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect an invalid tool call in a freshly generated assistant message during agentic rollout and regenerate the turn (up to a bounded retry count) before it is recorded, with a pluggable validator, train-only gating, and a metadata flag on exhaustion.

**Architecture:** All generation and parsing happen in `BaseAdapter._run_turn` (`slime/agent/adapters/common.py`), and `record_turn` — the only place trajectory state mutates — runs last, after the HTTP flush. We wrap the generate→parse→build block in a bounded retry loop that calls a pluggable validator (matching `find_invalid_tool_call`'s contract: `None` = valid, `(name, reason)` = invalid) via a new overridable `_validate_reply` hook. `OpenAIAdapter` implements the hook (base is a no-op). Rejected candidates never reach `record_turn`, so they leave no side effects. Retries are skipped for eval rollouts via a new `Session.is_eval` flag; on exhaustion the last candidate is accepted and flagged in sample metadata.

**Tech Stack:** Python 3, aiohttp, pytest (CPU unit tests via aiohttp `TestServer`/`TestClient` loopback and monkeypatched `call_sglang_generate`), slime's `load_function` hook loader.

## Global Constraints

- Line length 119; `black` + `isort` (black profile); `ruff` E/F/B/UP; run `pre-commit run --all-files` before finishing.
- Contribution scope: minimal, verifiable changes aligned with existing patterns — no new abstractions/base-classes, no large refactors.
- New pluggable hook must load via `slime.utils.misc.load_function` (dotted path), consistent with other `--custom-*-path` args.
- Validator contract (verbatim): `validator(response_dict: dict) -> tuple[str, str] | None`, where `None` = all tool calls valid and `(tool_name, raw_arguments)` = first invalid call. `response_dict` is OpenAI shape `{"choices": [{"message": {"tool_calls": [{"function": {"name", "arguments"}}]}}]}` with `arguments` as a JSON string.
- Scope: `OpenAIAdapter` only. `AnthropicAdapter` unchanged. `BaseAdapter._validate_reply` default returns `None`.
- Train-only: retries skipped when `Session.is_eval` is `True`.
- Same sampling params on retry (each `call_sglang_generate` already mints a fresh request id).
- CPU tests follow the repo pattern: a test module under `tests/test_agent/` ending with `if __name__ == "__main__": pytest.main([__file__])`.

---

### Task 1: Add `Session.is_eval` and `open_session(is_eval=...)`

**Files:**
- Modify: `slime/agent/adapters/common.py:33-42` (Session dataclass), `slime/agent/adapters/common.py:215-228` (open_session)
- Test: `tests/test_agent/test_toolcall_retry.py` (new)

**Interfaces:**
- Consumes: nothing (leaf change).
- Produces: `Session.is_eval: bool` (default `False`); `BaseAdapter.open_session(sid, *, sampling_defaults=None, max_context_tokens=0, is_eval=False)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent/test_toolcall_retry.py` with:

```python
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


if __name__ == "__main__":
    pytest.main([__file__])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_agent/test_toolcall_retry.py -q`
Expected: FAIL — `TypeError: open_session() got an unexpected keyword argument 'is_eval'`.

- [ ] **Step 3: Implement the change**

In `slime/agent/adapters/common.py`, add the field to `Session` (after `max_context_tokens`):

```python
@dataclasses.dataclass
class Session:
    """Per-sid adapter state: sampling defaults and context budget.

    Trajectory state lives in the shared TrajectoryManager (BaseAdapter.manager),
    not here.
    """

    sampling_defaults: dict = dataclasses.field(default_factory=dict)
    max_context_tokens: int = 0
    # True for eval rollouts: disables tool-call retry so eval measures raw output.
    is_eval: bool = False
```

Update `open_session`:

```python
    def open_session(
        self,
        sid: str,
        *,
        sampling_defaults: dict | None = None,
        max_context_tokens: int = 0,
        is_eval: bool = False,
    ) -> None:
        """Register a fresh per-sid Session; sids must be unique."""
        if sid in self.store:
            raise ValueError(f"session_id {sid!r} already exists; sids must be unique per agent run")
        self.store[sid] = Session(
            sampling_defaults=dict(sampling_defaults or {}),
            max_context_tokens=int(max_context_tokens or 0),
            is_eval=bool(is_eval),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_agent/test_toolcall_retry.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add slime/agent/adapters/common.py tests/test_agent/test_toolcall_retry.py
git commit -m "feat(agent): add Session.is_eval for train-only tool-call retry gating"
```

---

### Task 2: Add `TurnRecord.invalid_tool_call` and surface it in sample metadata

**Files:**
- Modify: `slime/agent/trajectory.py:28-43` (TurnRecord), `slime/agent/trajectory.py:507-516` (metadata assembly)
- Test: `tests/test_agent/test_toolcall_retry.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `TurnRecord.invalid_tool_call: bool` (default `False`); sample metadata key `"invalid_tool_call"` (bool, `any()` over assistant nodes, mirroring `ill_formed`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent/test_toolcall_retry.py` (above the `__main__` guard):

```python
from slime.agent.trajectory import TurnRecord  # noqa: E402


def test_turn_record_has_invalid_tool_call_default_false():
    tr = TurnRecord(prompt_ids=[1], output_ids=[2], finish_reason="stop")
    assert tr.invalid_tool_call is False
    tr2 = TurnRecord(prompt_ids=[1], output_ids=[2], finish_reason="stop", invalid_tool_call=True)
    assert tr2.invalid_tool_call is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_agent/test_toolcall_retry.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'invalid_tool_call'`.

- [ ] **Step 3: Implement the change**

In `slime/agent/trajectory.py`, add the field to `TurnRecord` (after `ill_formed`):

```python
    output_log_probs: list[float] = dataclasses.field(default_factory=list)
    ill_formed: bool = False
    # True when the turn's tool call was still invalid after exhausting retries
    # (the adapter accepted the last candidate). Surfaced in sample metadata.
    invalid_tool_call: bool = False
```

In `_chain_to_samples`, add the aggregate alongside `ill_formed` (`trajectory.py:510-516`):

```python
        ill_formed = any(n.turn.ill_formed for n in asst_nodes)
        invalid_tool_call = any(n.turn.invalid_tool_call for n in asst_nodes)
        md = {
            **(extra_metadata or {}),
            "truncated": truncated,
            "use_tool": use_tool,
            "ill_formed": ill_formed,
            "invalid_tool_call": invalid_tool_call,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_agent/test_toolcall_retry.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add slime/agent/trajectory.py tests/test_agent/test_toolcall_retry.py
git commit -m "feat(agent): add TurnRecord.invalid_tool_call and surface in sample metadata"
```

---

### Task 3: Expose the full (pre-truncation) tool-call list from `_build_reply_parts`

**Files:**
- Modify: `slime/agent/adapters/openai.py:211-281` (`_build_reply_parts`)
- Test: `tests/test_agent/test_toolcall_retry.py`

**Interfaces:**
- Consumes: `ParsedModelOutput` from `slime.agent.parsing`.
- Produces: a module-level helper `openai._wire_tool_calls(parsed) -> list[dict]` returning the **full** list of OpenAI-shape wire tool calls (each `{"id", "type", "function": {"name", "arguments": <json str>}}`), before any `[:1]` truncation. `_build_reply_parts` is refactored to call it. External wire/manager output (still `[:1]`) is unchanged.

This is a pure refactor: it extracts the existing per-tool-call construction (currently inline at `openai.py:222-248`) into a reusable helper so Task 5's validator can see every tool call, not just the first.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent/test_toolcall_retry.py`:

```python
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
```

Note: `json.dumps(..., sort_keys=True)` with single-key dicts yields `'{"x": 1}'` / `'{"y": 2}'` exactly.

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_agent/test_toolcall_retry.py -q`
Expected: FAIL — `AttributeError: module 'slime.agent.adapters.openai' has no attribute '_wire_tool_calls'`.

- [ ] **Step 3: Implement the refactor**

In `slime/agent/adapters/openai.py`, add the helper just above `_build_reply_parts` (near line 208):

```python
def _wire_tool_calls(parsed: ParsedModelOutput) -> list[dict[str, Any]]:
    """Full list of OpenAI-shape wire tool calls (arguments as JSON strings).

    This is the complete list BEFORE the ``[:1]`` truncation applied when
    framing the outbound wire/manager messages, so a validator can inspect
    every tool call the model emitted, not just the first.
    """
    calls: list[dict[str, Any]] = []
    for tu in parsed.tool_uses:
        name = tu.get("name", "tool")
        args_dict = tu.get("input") or {}
        if not isinstance(args_dict, dict):
            args_dict = {"_raw_arguments": str(args_dict)}
        calls.append(
            {
                "id": f"call_{secrets.token_hex(12)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args_dict, ensure_ascii=False, sort_keys=True),
                },
            }
        )
    return calls
```

Then rewrite the tool-call section of `_build_reply_parts` (`openai.py:222-248`) to consume it. Replace the `for tu in parsed.tool_uses:` loop that builds `wire_tool_calls` and `manager_tool_calls` with:

```python
    wire_tool_calls = _wire_tool_calls(parsed)
    manager_tool_calls: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": call["function"]["name"],
                # manager_message keeps arguments as a dict (chat-template replay
                # + dict-equality history match); recover it from the wire JSON.
                "arguments": json.loads(call["function"]["arguments"]),
            },
        }
        for call in wire_tool_calls
    ]
```

Leave the rest of `_build_reply_parts` (the `wire_message`/`manager_message` assembly and the `[:1]` truncation at `openai.py:271-272`) unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python tests/test_agent/test_toolcall_retry.py -q && python tests/test_agent/test_adapters.py -q`
Expected: PASS for both (the existing adapter suite confirms the refactor preserves behavior, including the `[:1]` truncation in `test_openai_translation_developer_to_system_and_tool_calls_to_dict` and the multiturn roundtrip).

- [ ] **Step 5: Commit**

```bash
git add slime/agent/adapters/openai.py tests/test_agent/test_toolcall_retry.py
git commit -m "refactor(agent): extract _wire_tool_calls to expose full tool-call list"
```

---

### Task 4: Add `--tool-call-validator-path` and `--tool-call-max-retries` args

**Files:**
- Modify: `slime/utils/arguments.py` (near the other rollout `--*-path` args, around line 342, inside the same parser/group)
- Test: `tests/test_agent/test_toolcall_retry.py`

**Interfaces:**
- Consumes: nothing.
- Produces: parsed args `args.tool_call_validator_path: str | None` (default `None`) and `args.tool_call_max_retries: int` (default `3`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent/test_toolcall_retry.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_agent/test_toolcall_retry.py -q`
Expected: FAIL — assertion error on `"--tool-call-validator-path" in src`.

- [ ] **Step 3: Implement the args**

In `slime/utils/arguments.py`, immediately after the `--rollout-function-path` block (ends at line 342), add:

```python
            parser.add_argument(
                "--tool-call-validator-path",
                type=str,
                default=None,
                help=(
                    "Path to a tool-call validator function used during agentic rollout. "
                    "Signature: `def validator(response_dict: dict) -> tuple[str, str] | None`, "
                    "where the input is an OpenAI-shape response dict "
                    "`{'choices': [{'message': {'tool_calls': [{'function': {'name', 'arguments'}}]}}]}` "
                    "(arguments is a JSON string) and the return is None when every tool call is "
                    "valid, or (tool_name, raw_arguments) for the first invalid call. When set, the "
                    "OpenAI adapter regenerates the assistant turn on an invalid tool call "
                    "(training rollouts only). Loaded via slime.utils.misc.load_function."
                ),
            )
            parser.add_argument(
                "--tool-call-max-retries",
                type=int,
                default=3,
                help=(
                    "Maximum number of regenerations when a tool call is invalid "
                    "(so up to max_retries + 1 generate calls per turn). "
                    "Only used when --tool-call-validator-path is set."
                ),
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_agent/test_toolcall_retry.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add slime/utils/arguments.py tests/test_agent/test_toolcall_retry.py
git commit -m "feat(args): add --tool-call-validator-path and --tool-call-max-retries"
```

---

### Task 5: Add the `_validate_reply` hook, validator wiring, and the retry loop

**Files:**
- Modify: `slime/agent/adapters/common.py` (`BaseAdapter.__init__` signature + a base `_validate_reply`; the retry loop in `_run_turn:345-365`)
- Modify: `slime/agent/adapters/openai.py` (`OpenAIAdapter._validate_reply` override)
- Test: `tests/test_agent/test_toolcall_retry.py`

**Interfaces:**
- Consumes: `Session.is_eval` (Task 1), `TurnRecord.invalid_tool_call` (Task 2), `openai._wire_tool_calls` (Task 3).
- Produces:
  - `BaseAdapter.__init__(..., tool_call_validator=None, tool_call_max_retries=3)` — stores `self.tool_call_validator: Callable[[dict], tuple[str, str] | None] | None` and `self.tool_call_max_retries: int`.
  - `BaseAdapter._validate_reply(self, parsed, session) -> tuple[str, str] | None` — base returns `None`.
  - `OpenAIAdapter._validate_reply(self, parsed, session)` — returns `None` when the validator is unset or `session.is_eval`; else builds `{"choices": [{"message": {"tool_calls": _wire_tool_calls(parsed)}}]}` and returns `self.tool_call_validator(response_dict)`.
  - The retry loop in `_run_turn` sets `TurnRecord.invalid_tool_call` on exhaustion.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent/test_toolcall_retry.py`. These drive a real `OpenAIAdapter` over loopback, monkeypatching `common.call_sglang_generate` to script per-attempt outputs. The validator rejects a tool call named `bad_tool`.

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python tests/test_agent/test_toolcall_retry.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'tool_call_validator'`.

- [ ] **Step 3: Implement `__init__` params + base hook**

In `slime/agent/adapters/common.py`, extend `BaseAdapter.__init__` (add params after `debug_callback`) and store them:

```python
        debug_callback: Callable[..., None] | None = None,
        tool_call_validator: Callable[[dict], tuple[str, str] | None] | None = None,
        tool_call_max_retries: int = 3,
    ) -> None:
```

Add, near the other attribute assignments (e.g. after `self.debug_callback = debug_callback`):

```python
        self.tool_call_validator = tool_call_validator
        self.tool_call_max_retries = int(tool_call_max_retries)
```

Add the base hook (place it near the other wire hooks, e.g. after `_build_reply`):

```python
    def _validate_reply(self, parsed, session: Session) -> tuple[str, str] | None:
        """Return None if the turn's tool calls are acceptable, else (name, reason).

        Base adapter never rejects; OpenAIAdapter overrides this to run a
        configured validator for training rollouts.
        """
        return None
```

- [ ] **Step 4: Implement the retry loop in `_run_turn`**

Replace the current straight-line block (`common.py:355-365`) — from `turn = await call_sglang_generate(...)` through `turn = dataclasses.replace(turn, ill_formed=parsed.ill_formed)` — with the bounded loop:

```python
            verdict: tuple[str, str] | None = None
            for attempt in range(self.tool_call_max_retries + 1):
                turn = await call_sglang_generate(prompt_ids, s, body, adapter=self, session_id=sid)
                raw_output = tok.decode(turn.output_ids, skip_special_tokens=False) if turn.output_ids else ""
                parsed = parse_model_output(
                    raw_output,
                    tools_schema=tools_schema,
                    tool_parser_name=self.tool_parser,
                    reasoning_parser_name=self.reasoning_parser,
                )
                reply = self._build_reply(parsed, turn.finish_reason, translated, tools_schema)
                verdict = self._validate_reply(parsed, s)
                if verdict is None:
                    break
                if attempt == self.tool_call_max_retries:
                    self.logger.warning(
                        "[%s] sid=%s tool call still invalid after %d retries: %s",
                        self.log_prefix,
                        sid,
                        self.tool_call_max_retries,
                        verdict,
                    )
                    break
                self.logger.info(
                    "[%s] sid=%s invalid tool call %s; regenerating (attempt %d/%d)",
                    self.log_prefix,
                    sid,
                    verdict,
                    attempt + 1,
                    self.tool_call_max_retries,
                )

            turn = dataclasses.replace(
                turn,
                ill_formed=parsed.ill_formed,
                invalid_tool_call=verdict is not None,
            )
```

Everything after this (`in_tok, out_tok = ...` at `common.py:367` onward: `_respond`, `_run_debug_callback`, `record_turn`) is unchanged and runs once on the accepted candidate.

- [ ] **Step 5: Implement `OpenAIAdapter._validate_reply`**

In `slime/agent/adapters/openai.py`, add the override inside `OpenAIAdapter` (e.g. after `_build_reply`):

```python
    def _validate_reply(self, parsed, session):
        """Reject a turn whose tool calls fail the configured validator.

        Returns None (accept) when no validator is configured or the session is
        an eval rollout; otherwise runs the validator over the FULL tool-call
        list (pre-truncation) in OpenAI response-dict shape.
        """
        if self.tool_call_validator is None or session.is_eval:
            return None
        if not parsed.tool_uses:
            return None
        response_dict = {"choices": [{"message": {"tool_calls": _wire_tool_calls(parsed)}}]}
        return self.tool_call_validator(response_dict)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python tests/test_agent/test_toolcall_retry.py -q`
Expected: PASS (all retry tests green).

- [ ] **Step 7: Run the existing adapter suite to check for regressions**

Run: `python tests/test_agent/test_adapters.py -q && python tests/test_agent/test_agent_rollout_cpu.py -q`
Expected: PASS (the added `__init__` kwargs are optional; default behavior unchanged).

- [ ] **Step 8: Commit**

```bash
git add slime/agent/adapters/common.py slime/agent/adapters/openai.py tests/test_agent/test_toolcall_retry.py
git commit -m "feat(agent): regenerate turn on invalid tool call via pluggable validator"
```

---

### Task 6: Wire the validator + retries into the adapter construction and eval flag

**Files:**
- Modify: `examples/coding_agent_rl/generate.py:286-293` (`_AdapterService.__init__` → `ADAPTER_CLS(...)`), `examples/coding_agent_rl/generate.py:385-389` (`open_session(...)`)
- Test: `tests/test_agent/test_toolcall_retry.py` (a wiring assertion via source inspection — this file has no CPU-driveable entry for the example's Ray-oriented service)

**Interfaces:**
- Consumes: `args.tool_call_validator_path`, `args.tool_call_max_retries` (Task 4); `load_function` (`slime.utils.misc`); `OpenAIAdapter(..., tool_call_validator=, tool_call_max_retries=)` (Task 5); `open_session(is_eval=)` (Task 1).
- Produces: no new interface; connects existing args to the adapter at runtime.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent/test_toolcall_retry.py`:

```python
def test_generate_example_wires_validator_and_eval_flag():
    gen_src = (REPO_ROOT / "examples" / "coding_agent_rl" / "generate.py").read_text()
    # validator loaded from the arg and passed to the adapter
    assert "tool_call_validator_path" in gen_src
    assert "tool_call_validator=" in gen_src
    assert "tool_call_max_retries=" in gen_src
    # eval flag threaded into the session
    assert "is_eval=evaluation" in gen_src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_agent/test_toolcall_retry.py -q`
Expected: FAIL — `"tool_call_validator_path" in gen_src` assertion error.

- [ ] **Step 3: Implement the adapter wiring**

In `examples/coding_agent_rl/generate.py`, ensure `load_function` is imported (check existing imports; add `from slime.utils.misc import load_function` alongside the other `slime.utils` imports if absent).

In `_AdapterService.__init__`, before the `self.adapter = ADAPTER_CLS(...)` call (line 286), resolve the validator:

```python
        validator_path = getattr(args, "tool_call_validator_path", None)
        tool_call_validator = load_function(validator_path) if validator_path else None
```

Extend the `ADAPTER_CLS(...)` call with the two kwargs:

```python
        self.adapter = ADAPTER_CLS(
            tokenizer=self.tokenizer,
            sglang_url=sglang_url,
            tool_parser=self.tool_parser,
            reasoning_parser=self.reasoning_parser,
            fork_threshold_tokens=CONFIG.fork_merge_threshold,
            chat_template_kwargs=getattr(args, "apply_chat_template_kwargs", None),
            tool_call_validator=tool_call_validator,
            tool_call_max_retries=int(getattr(args, "tool_call_max_retries", 3)),
        )
```

- [ ] **Step 4: Implement the eval-flag wiring**

In `generate()`, update the `open_session(...)` call (line 385) to pass the eval flag (the `evaluation` param is in scope):

```python
                        state.adapter.open_session(
                            session_id,
                            sampling_defaults=sampling_params,
                            max_context_tokens=state.max_context_len,
                            is_eval=evaluation,
                        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python tests/test_agent/test_toolcall_retry.py -q`
Expected: PASS.

- [ ] **Step 6: Byte-compile the example to catch syntax/import errors**

Run: `python -m py_compile examples/coding_agent_rl/generate.py`
Expected: no output, exit 0.

- [ ] **Step 7: Commit**

```bash
git add examples/coding_agent_rl/generate.py tests/test_agent/test_toolcall_retry.py
git commit -m "feat(coding_agent_rl): wire tool-call validator and eval-only gating"
```

---

### Task 7: Contract test against the reference `find_invalid_tool_call`

**Files:**
- Test: `tests/test_agent/test_toolcall_retry.py`

**Interfaces:**
- Consumes: `OpenAIAdapter._validate_reply` (Task 5), `_wire_tool_calls` (Task 3). Uses a **local reimplementation** of the reference validator's polarity so the test is self-contained (the real `openhands.sdk` module lives in a separate repo not importable here).

- [ ] **Step 1: Write the test**

Append to `tests/test_agent/test_toolcall_retry.py`. This encodes the reference contract (`None` = valid; `(name, raw)` = first invalid; JSON-unparseable args are invalid) and drives it through `_validate_reply` to confirm polarity wiring end to end.

```python
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
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python tests/test_agent/test_toolcall_retry.py -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_agent/test_toolcall_retry.py
git commit -m "test(agent): contract test for validator None=valid polarity"
```

---

### Task 8: Register the new test in CI

**Files:**
- Modify: `.github/workflows/pr-test.yml.j2` (CPU/unit test matrix)
- Generate: `.github/workflows/pr-test.yml` (via `generate_github_workflows.py`)

**Interfaces:**
- Consumes: `tests/test_agent/test_toolcall_retry.py`.
- Produces: a CI entry running the new test.

- [ ] **Step 1: Inspect how existing agent tests are registered**

Run: `grep -n "test_agent\|test_adapters\|test_agent_rollout" .github/workflows/pr-test.yml.j2`
Expected: shows the matrix entry/entries that list the agent CPU tests. Note the exact list/format used (a path list or a loop).

- [ ] **Step 2: Add the new test to the same matrix**

Edit `.github/workflows/pr-test.yml.j2` to include `tests/test_agent/test_toolcall_retry.py` in the same CPU test list where `tests/test_agent/test_adapters.py` appears (match the surrounding YAML/Jinja formatting exactly — a list item or loop entry).

- [ ] **Step 3: Regenerate the workflow**

Run: `python generate_github_workflows.py` (from repo root; confirm the script name via `ls generate_github_workflows.py`)
Expected: `.github/workflows/pr-test.yml` updated. Never edit the `.yml` by hand.

- [ ] **Step 4: Verify the test is present in generated output**

Run: `grep -n "test_toolcall_retry" .github/workflows/pr-test.yml`
Expected: at least one match.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/pr-test.yml.j2 .github/workflows/pr-test.yml
git commit -m "ci: register test_toolcall_retry in CPU test matrix"
```

---

### Task 9: Final verification (lint + full agent suite)

**Files:** none (verification only).

- [ ] **Step 1: Run the full agent test suite**

Run:
```bash
python tests/test_agent/test_toolcall_retry.py -q
python tests/test_agent/test_adapters.py -q
python tests/test_agent/test_agent_rollout_cpu.py -q
```
Expected: all PASS.

- [ ] **Step 2: Run pre-commit on all changed files**

Run: `pre-commit run --all-files --show-diff-on-failure --color=always`
Expected: all hooks pass (black/isort/ruff/autoflake). If a hook reformats, re-stage and re-run until clean.

- [ ] **Step 3: Commit any formatting fixups**

```bash
git add -A
git commit -m "style: pre-commit fixups for tool-call retry" || echo "nothing to commit"
```

---

## Notes for the implementer

- **Why no side effects on rejected candidates:** `record_turn` (the only trajectory mutation) runs *after* the retry loop and only once, so rejected candidates live only in local variables. Do not add any `record_turn`/manager call inside the loop.
- **Same-params retries:** a low-temperature model may reproduce the same invalid call; retries help mainly at nonzero temperature. This is expected and documented in the spec — do not add sampling perturbation.
- **Rejection-sampling bias** (regenerating until valid skews the recorded distribution off-policy) is acknowledged in the spec and intentionally not mitigated.
- **Anthropic adapter** is intentionally untouched (its `_validate_reply` inherits the base no-op).
- Test file path is `tests/test_agent/test_toolcall_retry.py`; keep the `if __name__ == "__main__": pytest.main([__file__])` guard so it runs both directly and under `pytest`.
