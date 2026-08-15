"""Host-side OpenHands trajectory persistence: path building, best-effort
read of the sandbox JSON, and atomic enriched write. All logic is CPU-only and
independent of a real sandbox or the OpenHands SDK."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("ADAPTER_PUBLIC_HOST", "127.0.0.1")

from examples.coding_agent_rl import generate as G  # noqa: E402
from slime.utils.types import Sample  # noqa: E402


def _sample(group_index=2, index=7, session_id="sess-9"):
    sample = Sample(prompt="p")
    sample.group_index = group_index
    sample.index = index
    sample.session_id = session_id
    return sample


def test_final_path_uses_group_and_index():
    path = G._trajectory_final_path("/root", _sample(group_index=2, index=7))
    assert path == "/root/2/2_7.json"


def test_final_path_falls_back_to_unknown():
    path = G._trajectory_final_path("/root", _sample(group_index=None, index=None))
    assert path == "/root/unknown/unknown_unknown.json"


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
    assert list((tmp_path / "1").glob(".traj_*.tmp")) == []


def test_persist_replace_failure_is_best_effort_and_cleans_temp(tmp_path, monkeypatch):
    directory = tmp_path / "2"
    directory.mkdir()
    final = directory / "2_7.json"
    final.write_text("existing")
    monkeypatch.setattr(G.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("replace failed")))

    G._persist_trajectory(
        str(tmp_path),
        _sample(),
        messages=[],
        diff_text="",
        reward=0.0,
        applied_cleanly=False,
        instance_id="i",
        session_id="s",
        agent_exit_code=1,
    )

    assert final.read_text() == "existing"
    assert list(directory.glob(".traj_*.tmp")) == []


def test_persist_is_best_effort_on_bad_root(monkeypatch):
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
    )


class _FakeSB:
    def __init__(self, payload):
        self._payload = payload

    async def read_file(self, path, *, user="root"):
        return self._payload


def test_read_sandbox_trajectory_parses_and_warns_on_garbage(caplog):
    import asyncio

    good = asyncio.run(G._read_sandbox_trajectory(_FakeSB('[{"role":"user"}]'), "/p"))
    assert good == [{"role": "user"}]
    assert asyncio.run(G._read_sandbox_trajectory(_FakeSB(""), "/p")) is None
    assert asyncio.run(G._read_sandbox_trajectory(_FakeSB("not json"), "/p")) is None
    assert asyncio.run(G._read_sandbox_trajectory(_FakeSB('{"messages": []}'), "/p")) is None
    messages = [record.getMessage() for record in caplog.records]
    assert any("empty or missing" in message for message in messages)
    assert any("read or parse failed" in message for message in messages)
    assert any("unexpected top-level type" in message for message in messages)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
