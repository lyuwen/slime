"""OpenHands software-agent-sdk harness.

Unlike the CLI harnesses (claude_code / codex), OpenHands is a Python agent loop.
This harness expects a prebuilt, self-contained Python-3.12 venv (with the
OpenHands SDK editable-installed) already present in the sandbox at the fixed
prefix ``/opt/oh-env``, drops a small in-sandbox driver (``oh_driver.py``) plus
its JSON config and the task prompt, then runs the driver detached via the
shared ``run_agent`` transport. The driver's litellm traffic dials back to
slime's OpenAIAdapter, so token capture is identical to the codex path.

Env delivery is an image volume: the oh-env layer (built by
``tools/oh-env-image.Dockerfile``) is mounted at ``/opt/oh-env`` when the
sandbox image is provisioned, so boot needs no tarball upload, tar, node, npm,
pip, or network egress -- the harness only verifies the prefix is present. The
mount MUST land at ``/opt/oh-env`` (editable-install paths depend on the fixed
prefix).
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from slime.agent import sandbox as _sandbox
from slime.agent.sandbox import Sandbox

from .common import BaseHarness, HarnessContext, run_agent

_ENV_PREFIX = "/opt/oh-env"
_PY = f"{_ENV_PREFIX}/bin/python"


class OpenHandsHarness(BaseHarness):
    name = "openhands"

    # host paths + knobs, all under the agent-layer SLIME_AGENT_* prefix
    extra_envs_env = "SLIME_AGENT_OH_EXTRA_ENVS"

    env_prefix = _ENV_PREFIX
    driver_host_path = Path(__file__).resolve().parents[3] / "examples/coding_agent_rl/oh_driver.py"
    driver_sandbox_path = "/home/agent/oh_driver.py"
    config_sandbox_path = "/home/agent/oh_config.json"
    prompt_sandbox_path = "/home/agent/oh_prompt.txt"

    async def install_cli(self, sb: Sandbox) -> None:
        """Verify the oh-env image volume is mounted at the fixed prefix.

        The env is delivered as an image volume (see the module docstring), not
        unpacked at boot, so there is nothing to install -- we only fail fast if
        ``/opt/oh-env`` is missing, which means the sandbox image was provisioned
        without the oh-env layer mounted."""
        exit_code, out, err = await sb.exec(
            f"test -x {_PY}",
            user="root",
            timeout=15,
            check=False,
        )
        if exit_code != 0:
            raise RuntimeError(
                f"{_PY} not found in the sandbox. The oh-env image volume "
                f"(tools/oh-env-image.Dockerfile) must be mounted at {_ENV_PREFIX}; "
                f"check the sandbox template/image provisioning."
            )

    async def write_config(
        self,
        sb: Sandbox,
        ctx: HarnessContext,
        *,
        prompt: str,
        fake_user: bool,
        max_iterations: int,
        tools: list[str],
        extra_envs: dict[str, str],
    ) -> None:
        """Drop the driver, its JSON config, and the prompt into the sandbox.

        The adapter URL / sid / tool list are STRUCTURED knobs the driver parses
        from oh_config.json. Raw agent env vars (extra_envs) are NOT put here --
        they go into the launch env in launch_and_wait, so the agent's shell sees
        them. ``extra_envs`` is accepted here only to fail fast on a bad type.
        """
        if not isinstance(extra_envs, dict):
            raise TypeError(f"extra_envs must be a dict, got {type(extra_envs).__name__}")
        await sb.write_file(self.driver_sandbox_path, self.driver_host_path, user="agent")
        await sb.write_file(self.prompt_sandbox_path, prompt, user="agent")
        config = {
            "adapter_url": ctx.adapter_url,
            "session_id": ctx.session_id,
            "model_label": ctx.model_label,
            "workdir": ctx.workdir,
            "fake_user": bool(fake_user),
            "max_iterations": int(max_iterations),
            "tools": list(tools),
        }
        await sb.write_file(self.config_sandbox_path, json.dumps(config), user="agent")
        await sb.exec(
            f"chown agent:agent {self.driver_sandbox_path} " f"{self.config_sandbox_path} {self.prompt_sandbox_path}",
            user="root",
            check=True,
            timeout=30,
        )

    async def launch_and_wait(
        self,
        sb: Sandbox,
        ctx: HarnessContext,
        prompt: str,
        time_budget_sec: int,
        *,
        fake_user: bool,
        max_iterations: int,
        tools: list[str],
        extra_envs: dict[str, str],
    ) -> int:
        await self.write_config(
            sb,
            ctx,
            prompt=prompt,
            fake_user=fake_user,
            max_iterations=max_iterations,
            tools=tools,
            extra_envs=extra_envs,
        )
        start_cmd = f"{_PY} {shlex.quote(self.driver_sandbox_path)} {shlex.quote(self.config_sandbox_path)}"
        # extra_envs merged LAST so a launcher-supplied value overrides the default.
        env = {"HOME": "/home/agent", **extra_envs}
        return await run_agent(sb, workdir=ctx.workdir, start_cmd=start_cmd, env=env, time_budget_sec=time_budget_sec)

    async def run(
        self,
        sb: Sandbox,
        *,
        workdir: str,
        session_id: str,
        adapter_url: str,
        time_budget_sec: int,
        prompt: str,
        fake_user: bool = False,
        max_iterations: int = 100,
        tools: list[str] | None = None,
        extra_envs: dict[str, str] | None = None,
    ) -> int:
        """OpenHands variant of BaseHarness.run: same step order (ensure user ->
        config -> launch), but threads the OpenHands-specific kwargs through."""
        await self.install_cli(sb)
        await _sandbox.ensure_agent_user(sb, workdir)
        ctx = HarnessContext(workdir=workdir, session_id=session_id, adapter_url=adapter_url)
        return await self.launch_and_wait(
            sb,
            ctx,
            prompt,
            time_budget_sec,
            fake_user=fake_user,
            max_iterations=max_iterations,
            tools=list(tools or []),
            extra_envs=dict(extra_envs or {}),
        )
