# OpenHands Trajectory Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist one enriched OpenHands agent trajectory JSON per coding-agent rollout (messages + tools + diff_text + SWE reward/applied_cleanly) without affecting rollout execution or grading.

**Architecture:** `oh_driver.py` (runs as sandbox user `agent`) converts the live `Conversation` events to chat messages + tool definitions and writes them atomically as `{"messages": [...], "tools": [...]}` to a config-supplied sandbox path under `/home/agent/`. `OpenHandsHarness` supplies that path via `oh_config.json`. `generate.py` reads the sandbox JSON object while the sandbox is alive (right after `git_diff`), enriches it with grading results after evaluation, and atomically writes the final JSON under `SWE_TRAJECTORY_DIR`. All persistence is best-effort.

**Tech Stack:** Python 3, OpenHands SDK (`openhands.sdk.event.base.LLMConvertibleEvent`), pytest, existing slime agent test fakes.

## Global Constraints

- Line length 119; formatting via `black` + `isort` (black profile); lint `ruff` E/F/B/UP. Run `pre-commit run --all-files` before final commit.
- Trace persistence MUST be best-effort: never change reward, `applied_cleanly`, sample status, `remove_sample`, adapter cleanup, or returned samples.
- Sandbox driver writes only beneath `/home/agent/` (user `agent`, no root, no privileged mkdir/chown).
- Host output root: env `SWE_TRAJECTORY_DIR`, default `trajectories`. Final path: `${SWE_TRAJECTORY_DIR}/{group_index}/{group_index}_{index}.json`; missing id → literal `unknown` in the path only, JSON keeps `null`.
- All file writes are atomic (temp sibling + `os.replace`).
- Change only the OpenHands path; do not modify the generic custom-generate interface, rollout indexing, or Claude/Codex harnesses.
- CPU unit tests end with `raise SystemExit(pytest.main([__file__, "-v"]))` and add `REPO_ROOT` to `sys.path` (match existing `tests/test_agent/` files).

---
### Task 1: Driver-side trajectory conversion + atomic write

**Files:**
- Modify: `examples/coding_agent_rl/oh_driver.py`
- Test: `tests/test_agent/test_oh_driver.py`

**Interfaces:**
- Consumes: `conv.state.events` (list of SDK events); `agent.tools` (initial Tool list); `cfg` dict from `oh_config.json`.
- Produces:
  - `oh_driver.events_to_trajectory(events) -> list[dict]` — pure; filters to `LLMConvertibleEvent`, calls `LLMConvertibleEvent.events_to_messages`, returns `to_chat_dict()` per message with `send_reasoning_content=True`.
  - `oh_driver.tools_to_trajectory(events, initial_tools) -> list[dict]` — pure; scans events for `SystemPromptEvent`, converts `ToolDefinition` entries via `.to_openai_tool()`; falls back to filtering `initial_tools` for `ToolDefinition` instances. Prefers event list because the SDK injects built-in tools there beyond what the caller passes.
  - `oh_driver.write_trajectory(path, events, initial_tools=None) -> None` — best-effort atomic JSON write of `{"messages": events_to_trajectory(events), "tools": tools_to_trajectory(events, initial_tools)}`; swallows all exceptions. Intermediate format is an object, not a bare list.
  - `main()` calls `write_trajectory(trajectory_path, conv.state.events, agent.tools)` after the run/fake-user loop when `cfg.get("trajectory_path")` is set.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent/test_oh_driver.py`:

```python
import json


class _FakeMsg:
    def __init__(self, payload):
        self._payload = payload

    def model_copy(self, update=None):
        assert update == {"send_reasoning_content": True}
        return self

    def to_chat_dict(self):
        return self._payload


def test_events_to_trajectory_converts_and_flags_reasoning(monkeypatch):
    import types

    convertible = [object(), object()]
    fake_base = types.SimpleNamespace(
        LLMConvertibleEvent=types.SimpleNamespace(
            events_to_messages=lambda evs: [_FakeMsg({"role": "assistant", "content": "hi"})],
        )
    )
    # only the two convertible objects are instances; a plain int is filtered out
    fake_base.LLMConvertibleEvent.__class__ = type(fake_base.LLMConvertibleEvent)
    monkeypatch.setitem(sys.modules, "openhands.sdk.event.base", fake_base)
    monkeypatch.setattr(
        oh_driver, "_isinstance_convertible", lambda e: e in convertible, raising=False
    )
    out = oh_driver.events_to_trajectory(convertible + [123])
    assert out == [{"role": "assistant", "content": "hi"}]


def test_write_trajectory_atomic_and_best_effort(tmp_path, monkeypatch):
    monkeypatch.setattr(
        oh_driver, "events_to_trajectory", lambda evs: [{"role": "user", "content": "x"}]
    )
    dest = tmp_path / "oh_trajectory.json"
    oh_driver.write_trajectory(str(dest), [object()])
    assert json.loads(dest.read_text()) == [{"role": "user", "content": "x"}]
    # best-effort: a conversion failure must not raise
    monkeypatch.setattr(oh_driver, "events_to_trajectory", lambda evs: (_ for _ in ()).throw(RuntimeError("boom")))
    oh_driver.write_trajectory(str(dest), [object()])  # no exception
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_agent/test_oh_driver.py`
Expected: FAIL — `AttributeError: module 'oh_driver' has no attribute 'events_to_trajectory'`.

- [ ] **Step 3: Write minimal implementation**

In `examples/coding_agent_rl/oh_driver.py`, add imports at top (`import os`, `import tempfile` alongside existing `import json`, `import sys`), then add:

```python
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
```

Then in `main()`, after the `if cfg.get("fake_user"): ... else: conv.run()` block and before `return 0`:

```python
    trajectory_path = cfg.get("trajectory_path")
    if trajectory_path:
        write_trajectory(trajectory_path, conv.state.events)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_agent/test_oh_driver.py`
Expected: PASS (all tests including the two new ones).

- [ ] **Step 5: Commit**

```bash
git add examples/coding_agent_rl/oh_driver.py tests/test_agent/test_oh_driver.py
git commit -m "feat(oh): convert and atomically persist trajectory in sandbox driver"
```

---
### Task 2: Harness advertises the sandbox trajectory path

**Files:**
- Modify: `slime/agent/harness/openhands.py`
- Test: `tests/test_agent/test_openhands_harness.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `OpenHandsHarness.trajectory_sandbox_path = "/home/agent/oh_trajectory.json"` (class attr).
  - `oh_config.json` gains key `"trajectory_path"` set to `self.trajectory_sandbox_path`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent/test_openhands_harness.py`:

```python
def test_write_config_advertises_trajectory_path():
    async def run_case():
        sb = FakeSandbox()
        with patch.object(OpenHandsHarness, "driver_host_path", Path(__file__)):
            await OpenHandsHarness().write_config(
                sb,
                _ctx(),
                prompt="p",
                fake_user=False,
                max_iterations=1,
                tools=[],
                extra_envs={},
            )
        cfg = json.loads(sb.files["/home/agent/oh_config.json"])
        assert cfg["trajectory_path"] == "/home/agent/oh_trajectory.json"

    asyncio.run(run_case())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_agent/test_openhands_harness.py`
Expected: FAIL — `KeyError: 'trajectory_path'`.

- [ ] **Step 3: Write minimal implementation**

In `slime/agent/harness/openhands.py`, add the class attribute beside the other sandbox paths:

```python
    prompt_sandbox_path = "/home/agent/oh_prompt.txt"
    trajectory_sandbox_path = "/home/agent/oh_trajectory.json"
```

And in `write_config`, add the key to the `config` dict:

```python
        config = {
            "adapter_url": ctx.adapter_url,
            "session_id": ctx.session_id,
            "model_label": ctx.model_label,
            "workdir": ctx.workdir,
            "fake_user": bool(fake_user),
            "max_iterations": int(max_iterations),
            "tools": list(tools),
            "trajectory_path": self.trajectory_sandbox_path,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_agent/test_openhands_harness.py`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add slime/agent/harness/openhands.py tests/test_agent/test_openhands_harness.py
git commit -m "feat(oh): advertise sandbox trajectory path via oh_config"
```

---

### Task 3: Host-side read, enrich, atomic write

**Files:**
- Modify: `examples/coding_agent_rl/generate.py`
- Test: `tests/test_agent/test_trajectory_persistence.py` (create)

**Interfaces:**
- Consumes: `OpenHandsHarness.trajectory_sandbox_path` (Task 2); `sb.read_file(path, user="agent")` (returns `""` on any failure); `base_sample.group_index`, `base_sample.index`, `base_sample.session_id`.
- Produces (all module-level in `generate.py`):
  - `_trajectory_root() -> str` — returns `SWE_TRAJECTORY_DIR`, defaulting to `"trajectories"` (persistence is always on for the OpenHands path).
  - `_trajectory_final_path(root, base_sample) -> str` — builds `{root}/{grp}/{grp}_{idx}.json`, using `"unknown"` for a `None` id in the path.
  - `_read_sandbox_trajectory(sb, path) -> dict | None` — best-effort: parse JSON from `sb.read_file`; accept only an object with `messages` (list) and `tools` (list); return `None` on empty/malformed/wrong-shape (bare list, missing key, non-list field).
  - `_persist_trajectory(root, base_sample, *, messages, tools, diff_text, reward, applied_cleanly, instance_id, session_id, agent_exit_code) -> None` — best-effort atomic write of the enriched document including `tools`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent/test_trajectory_persistence.py`:

```python
"""Host-side OpenHands trajectory persistence: path building, best-effort
read of the sandbox JSON, and atomic enriched write. All logic is CPU-only and
independent of a real sandbox or the OpenHands SDK."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os

os.environ.setdefault("ADAPTER_PUBLIC_HOST", "127.0.0.1")

from examples.coding_agent_rl import generate as G  # noqa: E402
from slime.utils.types import Sample  # noqa: E402


def _sample(group_index=2, index=7, session_id="sess-9"):
    s = Sample(prompt="p")
    s.group_index = group_index
    s.index = index
    s.session_id = session_id
    return s


def test_final_path_uses_group_and_index():
    p = G._trajectory_final_path("/root", _sample(group_index=2, index=7))
    assert p == "/root/2/2_7.json"


def test_final_path_falls_back_to_unknown():
    p = G._trajectory_final_path("/root", _sample(group_index=None, index=None))
    assert p == "/root/unknown/unknown_unknown.json"


def test_persist_writes_enriched_document(tmp_path):
    G._persist_trajectory(
        str(tmp_path),
        _sample(group_index=1, index=3, session_id="sid"),
        messages=[{"role": "user", "content": "hi"}],
        diff_text="diff --git a b",
        reward=1.0,
        applied_cleanly=True,
        instance_id="inst-1",
        session_id="sid",
        agent_exit_code=0,
    )
    doc = json.loads((tmp_path / "1" / "1_3.json").read_text())
    assert doc["messages"] == [{"role": "user", "content": "hi"}]
    assert doc["diff_text"] == "diff --git a b"
    assert doc["reward"] == 1.0
    assert doc["applied_cleanly"] is True
    assert doc["instance_id"] == "inst-1"
    assert doc["group_index"] == 1 and doc["index"] == 3
    assert doc["session_id"] == "sid" and doc["agent_exit_code"] == 0


def test_persist_is_best_effort_on_bad_root(monkeypatch):
    # os.replace into a path whose parent can't be made must not raise
    monkeypatch.setattr(G.os, "makedirs", lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    G._persist_trajectory(
        "/root",
        _sample(),
        messages=[],
        diff_text="",
        reward=0.0,
        applied_cleanly=False,
        instance_id="i",
        session_id="s",
        agent_exit_code=1,
    )  # no exception


class _FakeSB:
    def __init__(self, payload):
        self._payload = payload

    async def read_file(self, path, *, user="root"):
        return self._payload


def test_read_sandbox_trajectory_parses_and_tolerates_garbage():
    import asyncio

    good = asyncio.run(G._read_sandbox_trajectory(_FakeSB('[{"role":"user"}]'), "/p"))
    assert good == [{"role": "user"}]
    assert asyncio.run(G._read_sandbox_trajectory(_FakeSB(""), "/p")) is None
    assert asyncio.run(G._read_sandbox_trajectory(_FakeSB("not json"), "/p")) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_agent/test_trajectory_persistence.py`
Expected: FAIL — `AttributeError: module ... has no attribute '_trajectory_final_path'`.

- [ ] **Step 3: Write minimal implementation**

In `examples/coding_agent_rl/generate.py`, add helpers near the other module-level `_`-helpers (after `_session_id`):

```python
def _trajectory_root() -> str:
    return os.environ.get("SWE_TRAJECTORY_DIR", "trajectories")


def _trajectory_final_path(root: str, sample: Sample) -> str:
    grp = "unknown" if sample.group_index is None else str(sample.group_index)
    idx = "unknown" if sample.index is None else str(sample.index)
    return os.path.join(root, grp, f"{grp}_{idx}.json")


async def _read_sandbox_trajectory(sb, path: str):
    """Best-effort: return the parsed sandbox trajectory list, or None.

    sb.read_file already returns "" on any read failure; malformed or empty
    content collapses to None so the caller simply skips persistence."""
    try:
        raw = await sb.read_file(path, user="agent")
        if not raw:
            return None
        data = json.loads(raw)
        return data if isinstance(data, list) else None
    except Exception:
        return None


def _persist_trajectory(
    root: str,
    sample: Sample,
    *,
    messages: list,
    diff_text: str,
    reward: float,
    applied_cleanly: bool,
    instance_id: str,
    session_id: str,
    agent_exit_code: int | None,
) -> None:
    """Best-effort atomic write of the enriched trajectory document."""
    try:
        path = _trajectory_final_path(root, sample)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        doc = {
            "messages": messages,
            "diff_text": diff_text,
            "reward": reward,
            "applied_cleanly": applied_cleanly,
            "instance_id": instance_id,
            "group_index": sample.group_index,
            "index": sample.index,
            "session_id": session_id,
            "agent_exit_code": agent_exit_code,
        }
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".traj_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(doc, f)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    except Exception as e:  # best-effort: never disturb the rollout
        logger.warning("[coding_agent_rl] %s: trajectory persist skipped: %s", instance_id, e)
```

Add `import tempfile` to the imports block (alongside `import time`).

Then wire capture into `generate()`. Only for the OpenHands agent, read the sandbox trajectory while the sandbox is alive — immediately after `diff_text = await swe.git_diff(...)` (still inside the `async with boot_agent_sandbox(...)` block):

```python
                diff_text = await swe.git_diff(sb, md["workdir"])
                traj_messages = None
                if AGENT_NAME == "openhands":
                    traj_messages = await _read_sandbox_trajectory(
                        sb, OpenHandsHarness.trajectory_sandbox_path
                    )
```

After `reward, applied_cleanly = await swe.run_evaluation(...)` returns (train and eval paths both reach it), persist when we captured messages. Place it right after the `run_evaluation` call, before the `if evaluation:` branch:

```python
            if traj_messages is not None:
                _persist_trajectory(
                    _trajectory_root(),
                    base_sample,
                    messages=traj_messages,
                    diff_text=diff_text,
                    reward=float(reward),
                    applied_cleanly=bool(applied_cleanly),
                    instance_id=instance_id,
                    session_id=session_id,
                    agent_exit_code=agent_exit_code,
                )
```

Note: `traj_messages` must be initialized to `None` before the `async with boot_agent_sandbox` block so the name exists on the timeout/exception paths — add `traj_messages = None` just after `t0 = time.time()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_agent/test_trajectory_persistence.py`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add examples/coding_agent_rl/generate.py tests/test_agent/test_trajectory_persistence.py
git commit -m "feat(oh): read, enrich, and atomically persist trajectory host-side"
```

---

### Task 4: Full suite + lint verification

**Files:** none (verification only).

- [ ] **Step 1: Run the agent test suite**

Run: `python -m pytest tests/test_agent/ -v`
Expected: PASS (all tests, including the three modified/created files).

- [ ] **Step 2: Lint/format**

Run: `pre-commit run --all-files --show-diff-on-failure --color=always`
Expected: PASS (or auto-fixes applied; re-stage and re-run until clean).

- [ ] **Step 3: Commit any lint fixes**

```bash
git add -A
git commit -m "chore(oh): lint/format trajectory persistence"
```


