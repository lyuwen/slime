"""Unit tests for the OpenHands SDK harness.

Mirrors tests/test_agent/test_harness.py: a FakeSandbox records every exec /
write_file so we assert on the issued commands (tar unpack, import check, config
drop, launch command + env) without a real sandbox, tarball, or Python 3.12.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_agent._fakes import FakeSandbox  # noqa: E402

from slime.agent.harness import HarnessContext  # noqa: E402
from slime.agent.harness import OpenHandsHarness  # noqa: E402
from slime.agent.harness import common as hc  # noqa: E402

_REAL_SLEEP = asyncio.sleep


async def _fast_sleep(_secs):
    await _REAL_SLEEP(0)


def _ctx(workdir="/workspace/repo", sid="sess-1", url="http://host:18001") -> HarnessContext:
    return HarnessContext(workdir=workdir, session_id=sid, adapter_url=url)


def test_install_cli_verifies_prefix_present():
    async def run_case():
        # Env is delivered as an image volume mounted at /opt/oh-env, so
        # install_cli only probes that the baked interpreter is present.
        # FakeSandbox.exec defaults to exit 0, so the probe passes.
        sb = FakeSandbox()
        await OpenHandsHarness().install_cli(sb)
        probe = next(c for c, _ in sb.exec_log if "test -x" in c)
        assert "/opt/oh-env/bin/python" in probe

    asyncio.run(run_case())


def test_install_cli_raises_when_prefix_missing():
    async def run_case():
        # A non-zero probe means the oh-env layer was not mounted: fail fast.
        sb = FakeSandbox(responses=[("test -x", (1, "", ""))])
        with pytest.raises(RuntimeError, match="oh-env image volume"):
            await OpenHandsHarness().install_cli(sb)

    asyncio.run(run_case())


def test_write_config_drops_driver_config_and_prompt():
    async def run_case():
        sb = FakeSandbox()
        with patch.object(OpenHandsHarness, "driver_host_path", Path(__file__)):  # any real file to stream in
            await OpenHandsHarness().write_config(
                sb,
                _ctx(sid="sess-oh", url="http://host:18001"),
                prompt="fix the bug",
                fake_user=False,
                max_iterations=42,
                tools=["file_editor", "terminal", "think", "finish"],
                extra_envs={},
            )
        # driver + prompt written
        assert "/home/agent/oh_driver.py" in sb.files
        assert sb.files["/home/agent/oh_prompt.txt"] == "fix the bug"
        # config carries the structured knobs the driver parses
        cfg = json.loads(sb.files["/home/agent/oh_config.json"])
        assert cfg["adapter_url"] == "http://host:18001"
        assert cfg["session_id"] == "sess-oh"
        assert cfg["workdir"] == "/workspace/repo"
        assert cfg["fake_user"] is False
        assert cfg["max_iterations"] == 42
        assert cfg["tools"] == ["file_editor", "terminal", "think", "finish"]

    asyncio.run(run_case())


def test_launch_command_and_extra_env_merge():
    async def run_case():
        captured = {}

        async def agent(env):
            captured["env"] = env
            return 0

        sb = FakeSandbox(on_launch=agent)
        with patch.object(OpenHandsHarness, "driver_host_path", Path(__file__)):
            with patch.object(hc.asyncio, "sleep", new=_fast_sleep):
                rc = await OpenHandsHarness().launch_and_wait(
                    sb,
                    _ctx(sid="sess-oh", url="http://host:18001"),
                    prompt="do it",
                    time_budget_sec=30,
                    fake_user=False,
                    max_iterations=10,
                    tools=["terminal", "finish"],
                    extra_envs={"HTTPS_PROXY": "http://p:8080", "HOME": "/override"},
                )
        assert rc == 0
        # the driver is launched with the baked interpreter against the config file
        body = next(v for k, v in sb.files.items() if k.endswith("run.sh"))
        assert "/opt/oh-env/bin/python /home/agent/oh_driver.py /home/agent/oh_config.json" in body
        # extra_envs reach the agent process, merged LAST (override wins)
        env = captured["env"]
        assert env["HTTPS_PROXY"] == "http://p:8080"
        assert env["HOME"] == "/override"

    asyncio.run(run_case())


def test_run_wires_steps_in_order():
    async def run_case():
        async def agent(_env):
            return 0

        sb = FakeSandbox(on_launch=agent)
        with patch.object(OpenHandsHarness, "driver_host_path", Path(__file__)):
            with patch.object(hc.asyncio, "sleep", new=_fast_sleep):
                rc = await OpenHandsHarness().run(
                    sb,
                    workdir="/workspace/repo",
                    session_id="sess-run",
                    adapter_url="http://host:18001",
                    time_budget_sec=30,
                    prompt="go",
                    tools=["terminal", "finish"],
                )
        assert rc == 0
        joined = " ".join(c for c, _ in sb.exec_log)
        order = [k for k in ("useradd", "oh_config.json", "setsid") if k in joined]
        assert order == ["useradd", "oh_config.json", "setsid"]

    asyncio.run(run_case())


def test_write_config_rejects_non_dict_extra_envs():
    async def run_case():
        sb = FakeSandbox()
        with patch.object(OpenHandsHarness, "driver_host_path", Path(__file__)):
            try:
                await OpenHandsHarness().write_config(
                    sb,
                    _ctx(),
                    prompt="x",
                    fake_user=False,
                    max_iterations=1,
                    tools=[],
                    extra_envs="NOT_A_DICT",
                )
            except TypeError as e:
                assert "extra_envs must be a dict" in str(e)
            else:
                raise AssertionError("expected TypeError for non-dict extra_envs")

    asyncio.run(run_case())


def test_sweconfig_parses_openhands_knobs(monkeypatch):
    monkeypatch.setenv("SWE_OH_FAKE_USER", "1")
    monkeypatch.setenv("SWE_OH_MAX_ITERATIONS", "55")
    monkeypatch.setenv("SWE_OH_TOOLS", "terminal, finish")
    monkeypatch.setenv("SLIME_AGENT_OH_EXTRA_ENVS", '{"HTTPS_PROXY": "http://p:8080"}')
    from examples.coding_agent_rl.generate import SweConfig

    cfg = SweConfig.from_env()
    assert cfg.oh_fake_user is True
    assert cfg.oh_max_iterations == 55
    assert cfg.oh_tools == ["terminal", "finish"]
    assert cfg.oh_extra_envs == {"HTTPS_PROXY": "http://p:8080"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
