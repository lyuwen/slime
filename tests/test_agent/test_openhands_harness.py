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

from slime.agent.harness import OpenHandsHarness  # noqa: E402
from slime.agent.harness import HarnessContext  # noqa: E402
from slime.agent.harness import common as hc  # noqa: E402

_REAL_SLEEP = asyncio.sleep


async def _fast_sleep(_secs):
    await _REAL_SLEEP(0)


def _ctx(workdir="/workspace/repo", sid="sess-1", url="http://host:18001") -> HarnessContext:
    return HarnessContext(workdir=workdir, session_id=sid, adapter_url=url)


def test_install_cli_untars_env_and_verifies_import():
    async def run_case():
        # install_cli uses exec_and_wait (detached setsid + done-marker poll), so
        # the fake must drive the launch handshake: on_launch returns exit 0 so
        # the marker is written and the poll succeeds on the next tick.
        async def installer(_env):
            return 0

        sb = FakeSandbox(on_launch=installer)
        with patch.dict("os.environ", {"SLIME_AGENT_OH_ENV_TARBALL": "/host/oh-env.tar"}):
            with patch.object(hc.asyncio, "sleep", new=_fast_sleep):
                await OpenHandsHarness().install_cli(sb)
        # the env tarball is streamed in, then untarred + import-checked in the
        # detached launcher body (exec_and_wait writes it to /tmp/.oh-install.sh)
        assert "/tmp/oh-env.tar" in sb.files
        body = " ".join(str(v) for v in sb.files.values())
        assert "tar xf /tmp/oh-env.tar -C /" in body
        # import self-check uses the baked interpreter
        assert "/opt/oh-env/bin/python" in body
        assert "import openhands.sdk" in body and "import openhands.tools" in body

    asyncio.run(run_case())


def test_write_config_drops_driver_config_and_prompt():
    async def run_case():
        sb = FakeSandbox()
        with patch.object(
            OpenHandsHarness, "driver_host_path", Path(__file__)  # any real file to stream in
        ):
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
                    sb, _ctx(), prompt="x", fake_user=False,
                    max_iterations=1, tools=[], extra_envs="NOT_A_DICT",
                )
            except TypeError as e:
                assert "extra_envs must be a dict" in str(e)
            else:
                raise AssertionError("expected TypeError for non-dict extra_envs")

    asyncio.run(run_case())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
