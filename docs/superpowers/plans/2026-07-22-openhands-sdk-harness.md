# OpenHands SDK Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the OpenHands software-agent-sdk as a new coding-agent harness in slime's `examples/coding_agent_rl`, running end-to-end scaleswe RL against the existing `E2BSandbox`.

**Architecture:** A new `OpenHandsHarness` (subclass of `BaseHarness`) unpacks a prebuilt, self-contained Python-3.12 venv (with the OpenHands SDK editable-installed) into the sandbox, drops a small in-sandbox `oh_driver.py`, and launches it detached via the existing `run_agent` transport. The driver builds an OpenHands `LLM`/`Agent`/`Conversation(LocalWorkspace)` loop whose litellm traffic dials back to slime's existing `OpenAIAdapter` (OpenAI `/v1/chat/completions`, session keyed on `Authorization: Bearer <sid>`) — so token capture, trajectory handling, and eval reuse are unchanged. The tool allowlist and arbitrary agent env vars are configurable from the RL launch end.

**Tech Stack:** Python 3.12 (python-build-standalone), OpenHands software-agent-sdk (dev/vendored), litellm, aiohttp, slime agent-rollout stack, pytest (CPU unit tests).

## Global Constraints

- Reuse the existing `OpenAIAdapter` — **no changes** to `slime/agent/adapters/`, `slime/agent/trajectory.py`, `slime/agent/parsing.py`, `slime/agent/sandbox.py`, or any Megatron/Ray training code.
- The runtime-service (docker/k8s) sandbox backend is a **separate follow-up spec** — this plan boots via the existing `E2BSandbox` only. Do **not** add a backend selector or `make_sandbox`.
- Env-var prefix convention: agent-library knobs use `SLIME_AGENT_*`; this example's task knobs use `SWE_OH_*` (task layer) / `SWE_*`. Keep every new var on the prefix that matches the layer that reads it.
- Fixed in-sandbox env prefix: `/opt/oh-env` (the tarball always unpacks here; editable install paths depend on it).
- Fake-user nudges default **off**.
- Tool allowlist default: `file_editor,terminal,task_tracker,think,finish`.
- CPU unit tests are plain scripts ending in `raise SystemExit(pytest.main([__file__, "-v"]))`; reuse `tests/test_agent/_fakes.py::FakeSandbox`.
- Line length 119; `black` + `isort` (black profile) + `ruff` (E/F/B/UP). E402 is ignored (the `sys.path` + late-import test pattern is expected).
- Spec: `docs/superpowers/specs/2026-07-21-scaleswe-openhands-harness-design.md`.

---

## File Structure

**New:**
- `slime/agent/harness/openhands.py` — `OpenHandsHarness(BaseHarness)`: tar-unpack install, config/driver drop, detached launch, env assembly.
- `examples/coding_agent_rl/oh_driver.py` — in-sandbox loop; `build_tools()` + `main()`. Runs under `/opt/oh-env/bin/python`, imports the baked SDK. Must stay import-light at module top so its pure-logic helpers (`build_tools`) are unit-testable on the host without the SDK installed.
- `tools/repackage_oh_env.py` — host-side dev tool: relink fresh SDK source into an env tarball without a venv rebuild.
- `examples/coding_agent_rl/run_qwen36_35b_a3b_scaleswe_openhands_8nodes.sh` — launcher.
- `tests/test_agent/test_openhands_harness.py` — harness + env-propagation unit tests.
- `tests/test_agent/test_oh_driver.py` — `build_tools` unit tests.
- `tests/test_tools/test_repackage_oh_env.py` — repackage tool unit test.

**Modified:**
- `slime/agent/harness/__init__.py` — export `OpenHandsHarness`.
- `examples/coding_agent_rl/generate.py` — add `_AGENTS["openhands"]`; add `SWE_OH_*` fields to `SweConfig`; pass them into the harness.
- `examples/coding_agent_rl/README.md` — OpenHands harness, env-tarball build, tool allowlist, env forwarding.

---

## Task 1: `OpenHandsHarness` — install + config drop

**Files:**
- Create: `slime/agent/harness/openhands.py`
- Modify: `slime/agent/harness/__init__.py`
- Test: `tests/test_agent/test_openhands_harness.py`

**Interfaces:**
- Consumes: `slime.agent.harness.common.BaseHarness`, `HarnessContext` (fields `workdir`, `session_id`, `adapter_url`, `model_label="slime-actor"`); `slime.agent.sandbox.Sandbox`; `slime.agent.sandbox.exec_and_wait`.
- Produces:
  - `OpenHandsHarness()` with class attrs `name="openhands"`, `env_tarball_env="SLIME_AGENT_OH_ENV_TARBALL"`, `driver_host_path` (host path to `oh_driver.py`), `env_prefix="/opt/oh-env"`, `driver_sandbox_path="/home/agent/oh_driver.py"`, `config_sandbox_path="/home/agent/oh_config.json"`, `prompt_sandbox_path="/home/agent/oh_prompt.txt"`.
  - `async install_cli(sb)` — untar `SLIME_AGENT_OH_ENV_TARBALL` to `/`, verify import.
  - `async write_config(sb, ctx, *, prompt, fake_user, max_iterations, tools, extra_envs)` — writes driver, `oh_config.json`, `oh_prompt.txt`. (NB: extends the base signature; see Task 3 for how `run` passes these.)

- [ ] **Step 1: Write the failing test for `install_cli`**

Create `tests/test_agent/test_openhands_harness.py`:

```python
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
        joined = " ".join(cmd for cmd, _ in sb.exec_log)
        # the env tarball is streamed in, then untarred + import-checked in the
        # detached launcher body (exec_and_wait writes it to /tmp/.oh-install.sh)
        assert "/tmp/oh-env.tar" in sb.files
        body = " ".join(str(v) for v in sb.files.values())
        assert "tar xf /tmp/oh-env.tar -C /" in body
        # import self-check uses the baked interpreter
        assert "/opt/oh-env/bin/python" in body
        assert "import openhands.sdk" in body and "import openhands.tools" in body

    asyncio.run(run_case())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_agent/test_openhands_harness.py`
Expected: FAIL — `ImportError: cannot import name 'OpenHandsHarness'`.

- [ ] **Step 3: Write `openhands.py` install_cli + skeleton**

Create `slime/agent/harness/openhands.py`:

```python
"""OpenHands software-agent-sdk harness.

Unlike the CLI harnesses (claude_code / codex), OpenHands is a Python agent loop.
This harness unpacks a prebuilt, self-contained Python-3.12 venv (with the
OpenHands SDK editable-installed) into the sandbox at the fixed prefix
``/opt/oh-env``, drops a small in-sandbox driver (``oh_driver.py``) plus its
JSON config and the task prompt, then runs the driver detached via the shared
``run_agent`` transport. The driver's litellm traffic dials back to slime's
OpenAIAdapter, so token capture is identical to the codex path.

Env delivery is a single prebuilt tarball (SLIME_AGENT_OH_ENV_TARBALL); boot is
pure ``tar x`` -- no node, npm, pip, or network egress. The tarball MUST unpack
to ``/opt/oh-env`` (editable-install paths depend on the fixed prefix).
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

from slime.agent.sandbox import Sandbox, exec_and_wait

from .common import BaseHarness, HarnessContext, run_agent

_ENV_PREFIX = "/opt/oh-env"
_PY = f"{_ENV_PREFIX}/bin/python"


class OpenHandsHarness(BaseHarness):
    name = "openhands"

    # host paths + knobs, all under the agent-layer SLIME_AGENT_* prefix
    env_tarball_env = "SLIME_AGENT_OH_ENV_TARBALL"
    extra_envs_env = "SLIME_AGENT_OH_EXTRA_ENVS"

    env_prefix = _ENV_PREFIX
    driver_host_path = Path(__file__).resolve().parents[3] / "examples/coding_agent_rl/oh_driver.py"
    driver_sandbox_path = "/home/agent/oh_driver.py"
    config_sandbox_path = "/home/agent/oh_config.json"
    prompt_sandbox_path = "/home/agent/oh_prompt.txt"

    async def install_cli(self, sb: Sandbox) -> None:
        """Untar the prebuilt env to the fixed prefix and verify the SDK imports."""
        tarball = Path(os.environ[self.env_tarball_env])
        await sb.write_file("/tmp/oh-env.tar", tarball)
        exit_code, log = await exec_and_wait(
            sb,
            cmd=(
                f"tar xf /tmp/oh-env.tar -C / && "
                f"{_PY} -c 'import openhands.sdk, openhands.tools'"
            ),
            user="root",
            time_budget_sec=300,
            tag="oh-install",
        )
        if exit_code != 0:
            raise RuntimeError(f"OpenHands env install failed (exit={exit_code}):\n{log[-1000:]}")
```

- [ ] **Step 4: Export from `__init__.py`**

Modify `slime/agent/harness/__init__.py`:

```python
"""Swappable coding-agent harnesses (Claude Code, Codex, OpenHands, ...)."""

from __future__ import annotations

from .claude_code import ClaudeCodeHarness
from .codex import CodexHarness
from .common import BaseHarness, HarnessContext
from .openhands import OpenHandsHarness

__all__ = [
    "BaseHarness",
    "HarnessContext",
    "ClaudeCodeHarness",
    "CodexHarness",
    "OpenHandsHarness",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python tests/test_agent/test_openhands_harness.py`
Expected: PASS (`test_install_cli_untars_env_and_verifies_import`).

- [ ] **Step 6: Write the failing test for `write_config`**

Append to `tests/test_agent/test_openhands_harness.py` (before the `__main__` guard):

```python
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
```

- [ ] **Step 7: Run test to verify it fails**

Run: `python tests/test_agent/test_openhands_harness.py -k write_config`
Expected: FAIL — `AttributeError: 'OpenHandsHarness' object has no attribute 'write_config'`.

- [ ] **Step 8: Implement `write_config`**

Add to `OpenHandsHarness` in `slime/agent/harness/openhands.py`:

```python
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
            f"chown agent:agent {self.driver_sandbox_path} "
            f"{self.config_sandbox_path} {self.prompt_sandbox_path}",
            user="root",
            check=True,
            timeout=30,
        )
```

- [ ] **Step 9: Run test to verify it passes**

Run: `python tests/test_agent/test_openhands_harness.py`
Expected: PASS (both tests).

- [ ] **Step 10: Commit**

```bash
git add slime/agent/harness/openhands.py slime/agent/harness/__init__.py tests/test_agent/test_openhands_harness.py
git commit -m "feat(agent): OpenHandsHarness install_cli + write_config"
```

---

## Task 2: `OpenHandsHarness` — launch + env assembly + `run`

**Files:**
- Modify: `slime/agent/harness/openhands.py`
- Test: `tests/test_agent/test_openhands_harness.py`

**Interfaces:**
- Consumes: `run_agent(sb, *, workdir, start_cmd, env, time_budget_sec) -> int` (from `common`); `BaseHarness.run` skeleton (calls `ensure_agent_user` → `write_config` → `launch_and_wait`).
- Produces:
  - `async launch_and_wait(sb, ctx, prompt, time_budget_sec, *, fake_user, max_iterations, tools, extra_envs) -> int` — builds `start_cmd = "/opt/oh-env/bin/python /home/agent/oh_driver.py /home/agent/oh_config.json"`; env dict is `{"HOME": "/home/agent", **extra_envs}` (extra_envs merged last so it overrides defaults); calls `write_config` then `run_agent`.
  - Overridden `async run(sb, *, workdir, session_id, adapter_url, time_budget_sec, prompt, fake_user, max_iterations, tools, extra_envs) -> int` — `ensure_agent_user` → `launch_and_wait` (which drops config). The base `run` can't be reused verbatim because OpenHands needs the extra kwargs; this override keeps the same step order.

- [ ] **Step 1: Write the failing test for launch command + env merge**

Append to `tests/test_agent/test_openhands_harness.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_agent/test_openhands_harness.py -k launch_command`
Expected: FAIL — `AttributeError: ... 'launch_and_wait'`.

- [ ] **Step 3: Implement `launch_and_wait` and `run`**

Add to `OpenHandsHarness` in `slime/agent/harness/openhands.py` (and add the import of `ensure_agent_user`):

At the top imports, change:

```python
from slime.agent.sandbox import Sandbox, exec_and_wait
```

to:

```python
from slime.agent import sandbox as _sandbox
from slime.agent.sandbox import Sandbox, exec_and_wait
```

Then add the methods:

```python
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
        return await run_agent(
            sb, workdir=ctx.workdir, start_cmd=start_cmd, env=env, time_budget_sec=time_budget_sec
        )

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_agent/test_openhands_harness.py -k launch_command`
Expected: PASS.

- [ ] **Step 5: Write the failing test for `run` step order + malformed extra_envs guard**

Append to `tests/test_agent/test_openhands_harness.py`:

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python tests/test_agent/test_openhands_harness.py`
Expected: PASS (all 5 tests). `oh_config.json` chown is issued before `setsid`, so the order assertion holds.

- [ ] **Step 7: Commit**

```bash
git add slime/agent/harness/openhands.py tests/test_agent/test_openhands_harness.py
git commit -m "feat(agent): OpenHandsHarness launch_and_wait + run with env forwarding"
```

---

## Task 3: `oh_driver.py` — `build_tools` allowlist mapping

**Files:**
- Create: `examples/coding_agent_rl/oh_driver.py`
- Test: `tests/test_agent/test_oh_driver.py`

**Interfaces:**
- Produces: `build_tools(names: list[str]) -> tuple[list, list[str]]` returning `(tools, include_default)` where `tools` is a list of `openhands.sdk.tool.Tool(name=...)` for the `tools=`-axis names and `include_default` is the subset of `["ThinkTool", "FinishTool"]` for builtins. Imports each `tools=`-axis name's registering module before constructing `Tool`. Raises `ValueError` on an unknown name.
- Consumes (at runtime only, inside the sandbox): `openhands.sdk`, `openhands.tools.*`. The module MUST import these lazily inside functions so `build_tools`'s name→module/axis routing is unit-testable on the host where the SDK is absent — the test monkeypatches the importer.

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent/test_oh_driver.py`:

```python
"""Unit tests for the in-sandbox OpenHands driver's host-testable logic.

Only build_tools() is exercised here: it is pure name->(module, axis) routing and
must run on the host where the OpenHands SDK is NOT installed, so the driver
imports the SDK lazily and build_tools takes an injectable tool factory +
registrar. The full LLM/Agent/Conversation loop runs only inside the sandbox and
is covered by the GPU e2e follow-up, not this CPU test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.coding_agent_rl import oh_driver  # noqa: E402


def _fakes():
    registered = []
    made = []

    def register(name):
        registered.append(name)

    def make_tool(name):
        made.append(name)
        return f"Tool({name})"

    return registered, made, register, make_tool


def test_build_tools_default_splits_axes():
    registered, made, register, make_tool = _fakes()
    tools, include_default = oh_driver.build_tools(
        ["file_editor", "terminal", "task_tracker", "think", "finish"],
        register_module=register,
        make_tool=make_tool,
    )
    assert tools == ["Tool(file_editor)", "Tool(terminal)", "Tool(task_tracker)"]
    assert include_default == ["ThinkTool", "FinishTool"]
    # each tools=-axis name registered its module first
    assert registered == ["file_editor", "terminal", "task_tracker"]


def test_build_tools_legacy_names():
    registered, made, register, make_tool = _fakes()
    tools, include_default = oh_driver.build_tools(
        ["str_replace_editor", "execute_bash", "task_tracker", "finish"],
        register_module=register,
        make_tool=make_tool,
    )
    assert tools == ["Tool(str_replace_editor)", "Tool(execute_bash)", "Tool(task_tracker)"]
    assert include_default == ["FinishTool"]
    # legacy names route through the legacy preset registrar
    assert registered == ["str_replace_editor", "execute_bash", "task_tracker"]


def test_build_tools_unknown_name_raises():
    _, _, register, make_tool = _fakes()
    try:
        oh_driver.build_tools(["not_a_tool"], register_module=register, make_tool=make_tool)
    except ValueError as e:
        assert "not_a_tool" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown tool name")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_agent/test_oh_driver.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'examples.coding_agent_rl.oh_driver'`.

- [ ] **Step 3: Write `oh_driver.py` with `build_tools`**

Create `examples/coding_agent_rl/oh_driver.py`:

```python
"""In-sandbox OpenHands driver.

Runs under /opt/oh-env/bin/python inside the sandbox; imports the baked
OpenHands SDK. Reads oh_config.json (adapter URL, session id, tool allowlist,
fake-user flag, max iterations) and the task prompt, builds an
LLM/Agent/Conversation(LocalWorkspace) loop, and runs it to completion. Every
LLM turn dials back to slime's OpenAIAdapter over the OpenAI-compatible
base_url, so token capture is handled host-side.

build_tools() is intentionally SDK-import-free at module top: the tool name ->
(registering module, agent axis) routing is unit-tested on the host where the
SDK is absent. All openhands imports happen lazily inside functions.
"""

from __future__ import annotations

import json
import sys

# Tool name -> the import path of the module whose import calls register_tool for
# it. Default preset: file_editor/terminal/task_tracker. Legacy preset also
# registers str_replace_editor/execute_bash (only after that module is imported).
_TOOLS_AXIS_MODULES = {
    "file_editor": "openhands.tools.file_editor",
    "terminal": "openhands.tools.terminal",
    "task_tracker": "openhands.tools.task_tracker",
    "str_replace_editor": "openhands.tools.preset.legacy",
    "execute_bash": "openhands.tools.preset.legacy",
}

# Builtins live on Agent(include_default_tools=[...]) by class name, NOT in tools=.
_BUILTIN_AXIS = {
    "think": "ThinkTool",
    "finish": "FinishTool",
}


def _default_register_module(name: str) -> None:
    import importlib

    importlib.import_module(_TOOLS_AXIS_MODULES[name])


def _default_make_tool(name: str):
    from openhands.sdk.tool import Tool

    return Tool(name=name)


def build_tools(names, *, register_module=None, make_tool=None):
    """Map a flat allowlist onto OpenHands' two tool axes.

    Returns (tools, include_default): ``tools`` are Tool(name=...) objects for the
    Agent(tools=) axis (each name's registering module imported first);
    ``include_default`` is the ThinkTool/FinishTool subset for
    Agent(include_default_tools=). Unknown names raise ValueError (fail fast).

    register_module / make_tool are injectable for host-side unit tests; the
    defaults import the real SDK (only available inside the sandbox).
    """
    register_module = register_module or _default_register_module
    make_tool = make_tool or _default_make_tool

    tools = []
    include_default: list[str] = []
    for name in names:
        if name in _BUILTIN_AXIS:
            include_default.append(_BUILTIN_AXIS[name])
        elif name in _TOOLS_AXIS_MODULES:
            register_module(name)
            tools.append(make_tool(name))
        else:
            raise ValueError(
                f"unknown OpenHands tool name {name!r}; "
                f"known: {sorted(set(_TOOLS_AXIS_MODULES) | set(_BUILTIN_AXIS))}"
            )
    return tools, include_default
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_agent/test_oh_driver.py`
Expected: PASS (all 3 tests).

- [ ] **Step 5: Commit**

```bash
git add examples/coding_agent_rl/oh_driver.py tests/test_agent/test_oh_driver.py
git commit -m "feat(coding_agent_rl): oh_driver build_tools allowlist mapping"
```

---

## Task 4: `oh_driver.py` — the agent loop `main()`

**Files:**
- Modify: `examples/coding_agent_rl/oh_driver.py`

**Interfaces:**
- Consumes: `build_tools` (Task 3); at runtime `openhands.sdk.{LLM, Agent, Conversation}`, `openhands.sdk.workspace.LocalWorkspace`.
- Produces: `main(config_path: str) -> int` — reads config + prompt, builds the loop, runs it (fake-user off → `conv.run()`; on → vendored nudge loop), returns process exit code. `if __name__ == "__main__": raise SystemExit(main(sys.argv[1]))`.

This task has no CPU unit test — the loop only runs inside the sandbox with the SDK present (covered by the GPU e2e follow-up). It is a separate task because it is an independently reviewable deliverable (the loop wiring) and the module must remain importable for Task 3's test after this is added.

- [ ] **Step 1: Add `main()` and the fake-user helper to `oh_driver.py`**

Append to `examples/coding_agent_rl/oh_driver.py`:

```python
def _run_with_fake_user(conv, max_nudges: int = 100) -> None:
    """Minimal fake-user nudge loop (vendored, ~self-contained).

    OpenHands stops the run when the agent sends a plain message instead of using
    a tool. In eval/RL we want it to keep going until it calls the finish tool.
    After each run() returns without finishing, send a short nudge and run again,
    bounded by max_nudges.
    """
    from openhands.sdk.conversation.state import ConversationExecutionStatus

    nudge = (
        "Please continue working on the task with whatever approach you think is "
        "suitable. When you are done, call the finish tool."
    )
    for _ in range(max_nudges):
        conv.run()
        if conv.state.execution_status == ConversationExecutionStatus.FINISHED:
            return
        conv.send_message(nudge)


def main(config_path: str) -> int:
    from openhands.sdk import LLM, Agent, Conversation
    from openhands.sdk.workspace import LocalWorkspace

    with open(config_path) as f:
        cfg = json.load(f)
    with open("/home/agent/oh_prompt.txt") as f:
        prompt = f.read()

    tools, include_default = build_tools(cfg["tools"])
    llm = LLM(
        model="openai/" + cfg.get("model_label", "slime-actor"),
        base_url=cfg["adapter_url"] + "/v1",
        api_key=cfg["session_id"],
    )
    agent = Agent(
        llm=llm,
        tools=tools,
        include_default_tools=include_default,
        system_prompt_kwargs={"cli_mode": True},
    )
    conv = Conversation(
        agent=agent,
        workspace=LocalWorkspace(working_dir=cfg["workdir"]),
        max_iteration_per_run=int(cfg["max_iterations"]),
    )
    conv.send_message(prompt)
    if cfg.get("fake_user"):
        _run_with_fake_user(conv)
    else:
        conv.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
```

- [ ] **Step 2: Verify the module still imports (Task 3 test still green)**

Run: `python tests/test_agent/test_oh_driver.py`
Expected: PASS (module-top import of `oh_driver` still succeeds — all SDK imports are inside functions).

- [ ] **Step 3: Verify the driver parses as valid Python under 3.12 syntax**

Run: `python -c "import ast; ast.parse(open('examples/coding_agent_rl/oh_driver.py').read())"`
Expected: no output, exit 0.

- [ ] **Step 4: Commit**

```bash
git add examples/coding_agent_rl/oh_driver.py
git commit -m "feat(coding_agent_rl): oh_driver LLM/Agent/Conversation loop"
```

---

## Task 5: Wire `openhands` into `generate.py`

**Files:**
- Modify: `examples/coding_agent_rl/generate.py`
- Test: `tests/test_agent/test_openhands_harness.py`

**Interfaces:**
- Consumes: `OpenHandsHarness` (Task 1-2), `OpenAIAdapter`, `SweConfig.from_env`.
- Produces: `_AGENTS["openhands"] = (OpenHandsHarness, OpenAIAdapter)`; `SweConfig` gains `oh_fake_user: bool`, `oh_max_iterations: int`, `oh_tools: list[str]`, `oh_extra_envs: dict[str, str]`; the `HARNESS_CLS().run(...)` call in `generate()` passes these when the harness is OpenHands.

The harness `run` signature differs between CLI harnesses (no OH kwargs) and OpenHands. Keep `generate()` generic by passing the OH kwargs only for OpenHands via a small dict that is empty for the others.

- [ ] **Step 1: Write the failing test for SweConfig OH fields**

Append to `tests/test_agent/test_openhands_harness.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_agent/test_openhands_harness.py -k sweconfig`
Expected: FAIL — `AttributeError: ... 'oh_fake_user'` (or import error if fields absent).

- [ ] **Step 3: Add OH fields to `SweConfig` and the `_AGENTS` row**

Modify `examples/coding_agent_rl/generate.py`:

Change the import (line ~34):

```python
from slime.agent.harness import ClaudeCodeHarness, CodexHarness, OpenHandsHarness
```

Change `_AGENTS` (line ~45):

```python
_AGENTS = {
    "claude_code": (ClaudeCodeHarness, AnthropicAdapter),
    "codex": (CodexHarness, OpenAIAdapter),
    "openhands": (OpenHandsHarness, OpenAIAdapter),
}
```

Add fields to the `SweConfig` dataclass (after `boot_retries: int`):

```python
    oh_fake_user: bool
    oh_max_iterations: int
    oh_tools: list[str]
    oh_extra_envs: dict[str, str]
```

In `SweConfig.from_env`, before the `return cls(`, add:

```python
        oh_tools_raw = os.environ.get("SWE_OH_TOOLS", "file_editor,terminal,task_tracker,think,finish")
        oh_tools = [t.strip() for t in oh_tools_raw.split(",") if t.strip()]
        oh_extra_envs_raw = os.environ.get("SLIME_AGENT_OH_EXTRA_ENVS", "").strip()
        oh_extra_envs = json.loads(oh_extra_envs_raw) if oh_extra_envs_raw else {}
        if not isinstance(oh_extra_envs, dict):
            raise ValueError("SLIME_AGENT_OH_EXTRA_ENVS must be a JSON object")
```

Add to the `return cls(` kwargs:

```python
            oh_fake_user=os.environ.get("SWE_OH_FAKE_USER", "0") not in ("0", "", "false", "False"),
            oh_max_iterations=int(os.environ.get("SWE_OH_MAX_ITERATIONS", "100")),
            oh_tools=oh_tools,
            oh_extra_envs=oh_extra_envs,
```

Add the `json` import at the top of `generate.py` if not present (it imports `os`, `time`, etc. — add `import json` in the stdlib group).

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_agent/test_openhands_harness.py -k sweconfig`
Expected: PASS.

- [ ] **Step 5: Pass OH kwargs into the harness `run` call**

In `generate()` in `examples/coding_agent_rl/generate.py`, find the `await HARNESS_CLS().run(` call and add the OpenHands-only kwargs. Replace:

```python
                agent_exit_code = await HARNESS_CLS().run(
                    sb,
                    workdir=md["workdir"],
                    session_id=session_id,
                    adapter_url=state.adapter_url,
                    time_budget_sec=CONFIG.agent_time_budget_sec,
                    prompt=swe.SWE_PROMPT,
                )
```

with:

```python
                oh_kwargs = (
                    {
                        "fake_user": CONFIG.oh_fake_user,
                        "max_iterations": CONFIG.oh_max_iterations,
                        "tools": CONFIG.oh_tools,
                        "extra_envs": CONFIG.oh_extra_envs,
                    }
                    if AGENT_NAME == "openhands"
                    else {}
                )
                agent_exit_code = await HARNESS_CLS().run(
                    sb,
                    workdir=md["workdir"],
                    session_id=session_id,
                    adapter_url=state.adapter_url,
                    time_budget_sec=CONFIG.agent_time_budget_sec,
                    prompt=swe.SWE_PROMPT,
                    **oh_kwargs,
                )
```

- [ ] **Step 6: Verify generate.py imports cleanly**

Run: `python -c "import ast; ast.parse(open('examples/coding_agent_rl/generate.py').read())"`
Expected: exit 0.

Run: `python tests/test_agent/test_openhands_harness.py`
Expected: PASS (all tests, incl. sweconfig).

- [ ] **Step 7: Commit**

```bash
git add examples/coding_agent_rl/generate.py tests/test_agent/test_openhands_harness.py
git commit -m "feat(coding_agent_rl): wire openhands harness into generate + SweConfig knobs"
```

---

## Task 6: `repackage_oh_env.py` — host-side dev relink tool

**Files:**
- Create: `tools/repackage_oh_env.py`
- Test: `tests/test_tools/test_repackage_oh_env.py`

**Interfaces:**
- Produces: `relink(env_tar: Path, sdk_src: Path, out_tar: Path, *, prefix_src="opt/oh-env/src/software-agent-sdk") -> None` — unpack `env_tar` to a temp dir, rsync/replace `sdk_src` over `<tmp>/<prefix_src>`, re-tar to `out_tar`. Plus a `main()` argparse wrapper. Pure filesystem/tar ops, host-testable with tiny fixtures.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tools/test_repackage_oh_env.py`:

```python
"""Unit test for tools/repackage_oh_env.py relink logic (tiny tar fixtures)."""

from __future__ import annotations

import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import repackage_oh_env  # noqa: E402


def _make_env_tar(path: Path, sdk_marker: str) -> None:
    root = path.parent / "envroot"
    src = root / "opt/oh-env/src/software-agent-sdk/openhands-sdk"
    src.mkdir(parents=True)
    (src / "marker.txt").write_text(sdk_marker)
    (root / "opt/oh-env/bin").mkdir(parents=True)
    (root / "opt/oh-env/bin/python").write_text("#!interp\n")
    with tarfile.open(path, "w") as tf:
        tf.add(root / "opt", arcname="opt")


def test_relink_replaces_sdk_source(tmp_path):
    env_tar = tmp_path / "oh-env.tar"
    _make_env_tar(env_tar, sdk_marker="OLD")

    new_src = tmp_path / "software-agent-sdk"
    (new_src / "openhands-sdk").mkdir(parents=True)
    (new_src / "openhands-sdk" / "marker.txt").write_text("NEW")

    out_tar = tmp_path / "oh-env.relinked.tar"
    repackage_oh_env.relink(env_tar, new_src, out_tar)

    extract = tmp_path / "check"
    with tarfile.open(out_tar) as tf:
        tf.extractall(extract)
    marker = extract / "opt/oh-env/src/software-agent-sdk/openhands-sdk/marker.txt"
    assert marker.read_text() == "NEW"
    # the interpreter (untouched non-src content) survives the repackage
    assert (extract / "opt/oh-env/bin/python").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_tools/test_repackage_oh_env.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.repackage_oh_env'` (create `tests/test_tools/__init__.py` if pytest can't import the package; an empty file is fine).

- [ ] **Step 3: Implement `repackage_oh_env.py`**

Create `tools/repackage_oh_env.py`:

```python
"""Relink fresh OpenHands SDK source into a prebuilt env tarball.

The env tarball is built once with the SDK editable-installed from an in-prefix
source path (/opt/oh-env/src/software-agent-sdk). Because an editable install
tracks the PATH, not the content, swapping the source needs no reinstall: unpack
the env, replace the source dir, re-tar. Use this after editing SDK code to ship
a new tarball without rebuilding the venv or re-resolving deps.

Rebuild the full env (not this tool) only when third-party DEPENDENCIES change.
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path

_DEFAULT_PREFIX_SRC = "opt/oh-env/src/software-agent-sdk"


def relink(env_tar: Path, sdk_src: Path, out_tar: Path, *, prefix_src: str = _DEFAULT_PREFIX_SRC) -> None:
    """Unpack env_tar, replace <prefix_src> with sdk_src, re-tar to out_tar."""
    env_tar = Path(env_tar)
    sdk_src = Path(sdk_src)
    out_tar = Path(out_tar)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        with tarfile.open(env_tar) as tf:
            tf.extractall(tmp)
        dst = tmp / prefix_src
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(sdk_src, dst)
        # re-tar the top-level 'opt' tree so it unpacks to the fixed prefix again
        top = prefix_src.split("/")[0]
        with tarfile.open(out_tar, "w") as tf:
            tf.add(tmp / top, arcname=top)


def main() -> None:
    p = argparse.ArgumentParser(description="Relink SDK source into an OpenHands env tarball.")
    p.add_argument("--env-tarball", required=True, type=Path)
    p.add_argument("--sdk-src", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--prefix-src", default=_DEFAULT_PREFIX_SRC)
    args = p.parse_args()
    relink(args.env_tarball, args.sdk_src, args.out, prefix_src=args.prefix_src)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_tools/test_repackage_oh_env.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/repackage_oh_env.py tests/test_tools/
git commit -m "feat(tools): repackage_oh_env relink tool for dev SDK swaps"
```

---

## Task 7: Launcher script

**Files:**
- Create: `examples/coding_agent_rl/run_qwen36_35b_a3b_scaleswe_openhands_8nodes.sh`

**Interfaces:**
- Consumes: everything above via env. No test (shell launcher); validated by `bash -n` syntax check and a grep of required knobs.

- [ ] **Step 1: Create the launcher by cloning the existing one with the OpenHands deltas**

Create `examples/coding_agent_rl/run_qwen36_35b_a3b_scaleswe_openhands_8nodes.sh` by copying `run_qwen36_35b_a3b_swe_8nodes.sh` and applying these edits (keep all Megatron/GRPO/SGLang/PERF/OPT args identical):

In the `ROLLOUT_ARGS`, change the custom-generate path comment/value to keep `examples.coding_agent_rl.generate.generate` (unchanged — same entry point).

Replace the SWE/claude-code knob block (the section starting `export SWE_AGENT=...` through the `SLIME_AGENT_CC_EXTRA_ARGS` / `AGENTS_JSON` / `SETTINGS_JSON` lines) with:

```bash
# ============ SWE / OpenHands rollout knobs ============

export SWE_AGENT="${SWE_AGENT:-openhands}"
export SWE_TRAIN_PROTOCOL="${SWE_TRAIN_PROTOCOL:-scaleswe}"
export E2B_API_KEY="${E2B_API_KEY:-e2b_0000000000000000000000000000000000000000}"
export SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY="${SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY:-image}"

# Prebuilt, self-contained Python-3.12 venv (OpenHands SDK editable-installed),
# unpacks to the fixed prefix /opt/oh-env. Build once on the host; relink dev SDK
# source with tools/repackage_oh_env.py. See README.
export SLIME_AGENT_OH_ENV_TARBALL="${SLIME_AGENT_OH_ENV_TARBALL:-/path/to/oh-env.tar}"

# OpenHands agent knobs.
export SWE_OH_FAKE_USER="${SWE_OH_FAKE_USER:-0}"
export SWE_OH_MAX_ITERATIONS="${SWE_OH_MAX_ITERATIONS:-100}"
export SWE_OH_TOOLS="${SWE_OH_TOOLS:-file_editor,terminal,task_tracker,think,finish}"
# Arbitrary extra env vars forwarded verbatim into the OH agent's shell (JSON obj).
# export SLIME_AGENT_OH_EXTRA_ENVS='{"HTTPS_PROXY":"http://proxy:8080"}'

# ADAPTER_PUBLIC_HOST must be routable from inside the sandbox (not 127.0.0.1).
export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}}"
export ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
export ADAPTER_PORT="${ADAPTER_PORT:-18001}"

export SWE_AGENT_TIME_BUDGET_SEC="${SWE_AGENT_TIME_BUDGET_SEC:-1800}"
export SWE_EVAL_TIMEOUT_SEC="${SWE_EVAL_TIMEOUT_SEC:-600}"
export SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-16}"
```

- [ ] **Step 2: Replace the `RUNTIME_ENV_JSON` builder with a prefix pass-through**

In the new launcher, replace the `RUNTIME_ENV_JSON=$(python3 - <<PY ... PY)` block with one that both keeps the explicit cluster keys AND forwards every `SLIME_AGENT_*` / `SWE_*` var (§5.1a). Use this body:

```bash
RUNTIME_ENV_JSON=$(python3 - <<PY
import json, os
# Explicit cluster/network keys that don't fit a forwarding prefix.
keys = (
    "no_proxy", "NO_PROXY",
    "E2B_API_KEY", "ADAPTER_PUBLIC_HOST",
    "ADAPTER_BIND_HOST", "ADAPTER_PORT",
)
env = {k: os.environ[k] for k in keys if k in os.environ}
# Prefix pass-through: forward every agent-library / SWE knob automatically so
# new vars need no per-var edit (see spec §5.1a). SLIME_AGENT_OH_EXTRA_ENVS rides
# this rule too.
for k, v in os.environ.items():
    if k.startswith("SLIME_AGENT_") or k.startswith("SWE_"):
        env[k] = v
env["MASTER_ADDR"] = os.environ["MASTER_ADDR"]
env["MASTER_PORT"] = os.environ.get("MASTER_PORT", "")
env["GLOO_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["TP_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["NCCL_SOCKET_IFNAME"] = os.environ["NCCL_SOCKET_IFNAME"]
env["PYTHONPATH"] = f"/root/Megatron-LM/:{os.environ['SLIME_DIR']}"
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
env["NCCL_NVLS_ENABLE"] = "0"
print(json.dumps({"env_vars": env}))
PY
)
```

Also remove the `SLIME_AGENT_NODE_TARBALL` / `SLIME_AGENT_CC_TARBALL` export lines (OpenHands env is self-contained — no node/npm).

- [ ] **Step 3: Syntax-check the launcher**

Run: `bash -n examples/coding_agent_rl/run_qwen36_35b_a3b_scaleswe_openhands_8nodes.sh`
Expected: exit 0, no output.

- [ ] **Step 4: Grep required knobs are present**

Run:
```bash
grep -c "SWE_AGENT:-openhands\|SLIME_AGENT_OH_ENV_TARBALL\|SWE_OH_TOOLS\|startswith(\"SLIME_AGENT_\")" examples/coding_agent_rl/run_qwen36_35b_a3b_scaleswe_openhands_8nodes.sh
```
Expected: `4`.

- [ ] **Step 5: Commit**

```bash
git add examples/coding_agent_rl/run_qwen36_35b_a3b_scaleswe_openhands_8nodes.sh
git commit -m "feat(coding_agent_rl): openhands 8-node launcher with env prefix pass-through"
```

---

## Task 8: README documentation

**Files:**
- Modify: `examples/coding_agent_rl/README.md`

- [ ] **Step 1: Add an OpenHands section to the README**

Add a new section after the existing harness discussion documenting:
- `SWE_AGENT=openhands` selects `(OpenHandsHarness, OpenAIAdapter)`.
- How to build `SLIME_AGENT_OH_ENV_TARBALL`: a python-build-standalone 3.12 venv at `/opt/oh-env`, with the 4 OpenHands packages editable-installed from `/opt/oh-env/src/software-agent-sdk/`, all deps installed, then `tar` the `/opt/oh-env` prefix. Include the exact `tar xf ... -C /` unpack contract (fixed prefix).
- `tools/repackage_oh_env.py --env-tarball ... --sdk-src thirdparty/benchmarks-main/vendor/software-agent-sdk --out ...` for dev SDK swaps.
- The tool allowlist: `SWE_OH_TOOLS` default and the legacy variant string.
- Env forwarding: the two mechanisms (`SLIME_AGENT_*`/`SWE_OH_*` prefix pass-through, and `SLIME_AGENT_OH_EXTRA_ENVS` JSON) with the five-hop path summary.
- Note that eval is unchanged (reuses `swe.py` scaleswe grader) and the runtime-service backend is a follow-up.

Use this block:

```markdown
## OpenHands SDK Harness (`SWE_AGENT=openhands`)

Selects `(OpenHandsHarness, OpenAIAdapter)`. Unlike claude-code/codex (self-contained
CLIs), OpenHands is a Python agent loop that runs *inside* the sandbox with a
`LocalWorkspace`; its litellm traffic dials back to the same `OpenAIAdapter` used
by codex, so token capture and trajectory handling are unchanged.

### The environment tarball (`SLIME_AGENT_OH_ENV_TARBALL`)

OpenHands needs Python 3.12 and a dep tree too heavy to `pip install` per boot.
Build a self-contained env **once** on the host and ship it as a tarball that
unpacks with a single `tar x` (no node/npm/pip/egress at boot):

1. Materialize a python-build-standalone CPython 3.12 at the fixed prefix `/opt/oh-env`.
2. Place the 4 OpenHands packages under `/opt/oh-env/src/software-agent-sdk/`.
3. `/opt/oh-env/bin/pip install -e` those packages (editable — records the path),
   then install all deps.
4. `tar cf oh-env.tar -C / opt/oh-env` (must unpack back to `/opt/oh-env`).

Boot-time `install_cli` runs `tar xf /tmp/oh-env.tar -C /` and verifies
`import openhands.sdk, openhands.tools`.

Swap fresh SDK source without rebuilding the venv:

    python tools/repackage_oh_env.py \
      --env-tarball oh-env.tar \
      --sdk-src thirdparty/benchmarks-main/vendor/software-agent-sdk \
      --out oh-env.relinked.tar

### Tool allowlist (`SWE_OH_TOOLS`)

Comma-separated; default `file_editor,terminal,task_tracker,think,finish`. Legacy
tool set: `SWE_OH_TOOLS=str_replace_editor,execute_bash,task_tracker,think,finish`.
`think`/`finish` are builtins (routed to `Agent(include_default_tools=...)`); the
rest are `Agent(tools=...)` entries whose registering module is imported first.

### Forwarding env vars into the agent

Two mechanisms carry launch-side vars all the way into the OH agent's shell:
1. **Prefix pass-through** — any `SLIME_AGENT_*` / `SWE_OH_*` / `SWE_*` var exported
   in the launcher is auto-forwarded through `RUNTIME_ENV_JSON` to the RolloutManager
   process (no per-var edit).
2. **`SLIME_AGENT_OH_EXTRA_ENVS`** — a JSON object of arbitrary `{"NAME":"value"}`
   pairs merged (last) into the driver's process env, so proxies/tokens/`PIP_*`/etc.
   reach the agent's Terminal-tool subprocesses.

Path: launcher → `RUNTIME_ENV_JSON` → RolloutManager `os.environ` → harness
`sb.exec(env=...)` → detached driver process → agent tool subprocesses.

Eval/reward is unchanged (reuses `swe.py`'s scaleswe grader). The docker/k8s
runtime-service sandbox backend is a separate follow-up; this path runs on `E2BSandbox`.
```

- [ ] **Step 2: Commit**

```bash
git add examples/coding_agent_rl/README.md
git commit -m "docs(coding_agent_rl): document OpenHands harness, env tarball, tool allowlist, env forwarding"
```

---

## Task 9: Full suite + lint gate

**Files:** none (verification only)

- [ ] **Step 1: Run all new/affected unit tests**

Run:
```bash
python tests/test_agent/test_openhands_harness.py
python tests/test_agent/test_oh_driver.py
python tests/test_tools/test_repackage_oh_env.py
python tests/test_agent/test_harness.py
```
Expected: all PASS (the existing `test_harness.py` must remain green — the `__init__.py` export change and the new harness must not break it).

- [ ] **Step 2: Run the broader agent test module via pytest**

Run: `pytest tests/test_agent/ -q`
Expected: PASS (no regressions in adapters / trajectory / rollout tests).

- [ ] **Step 3: Lint / format**

Run: `pre-commit run --files slime/agent/harness/openhands.py slime/agent/harness/__init__.py examples/coding_agent_rl/oh_driver.py examples/coding_agent_rl/generate.py tools/repackage_oh_env.py tests/test_agent/test_openhands_harness.py tests/test_agent/test_oh_driver.py tests/test_tools/test_repackage_oh_env.py`
Expected: all hooks pass (black/isort/ruff/autoflake). Fix any reported diffs and re-run.

- [ ] **Step 4: Final commit if lint made changes**

```bash
git add -A
git commit -m "chore: lint/format openhands harness additions" || echo "nothing to commit"
```

---

## Self-Review notes

- **Spec coverage:** §4 env tarball → Task 1 + Task 6 + README; §5 harness → Tasks 1-2; §5.1 env propagation → Task 2 (extra_envs merge) + Task 5 (SweConfig parse) + Task 7 (prefix pass-through) + Task 9; §6 driver → Tasks 3-4; §6.1 tool allowlist → Task 3; §7 eval → unchanged (no task needed, asserted in README); §8 run script → Task 7; §9 testing → Tasks 1-6 + Task 9; §10 file list → all tasks. All covered.
- **Deferred (spec §11), intentionally no task:** runtime-service backend, GPU e2e smoke test.
- **Type consistency:** `build_tools(names, *, register_module, make_tool) -> (tools, include_default)` consistent across Task 3 (def + test) and Task 4 (call with defaults). `OpenHandsHarness.run(..., fake_user, max_iterations, tools, extra_envs)` consistent across Task 2 (def), Task 5 (call). `relink(env_tar, sdk_src, out_tar, *, prefix_src)` consistent across Task 6 def + test.
