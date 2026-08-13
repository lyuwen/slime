#!/usr/bin/env python3
"""MVP: run oh_driver.py against a live LLM, either locally or in a sandbox.

Local mode (default)
--------------------
Runs the driver in-process using the local openhands install (e.g. the
swebench conda env's Python). No sandbox is created. Useful for fast
iteration on the driver logic.

    python test-oh-driver-mvp.py \
        --llm-config llm-config-example.json \
        --prompt "Create a file hello.txt with 'hi'. Then call the finish tool."

Sandbox mode (--sandbox)
------------------------
Boots an E2B sandbox whose image already has /opt/oh-env pre-mounted (built
from the oh-env scratch image). Sends oh_driver.py to the sandbox and runs it
under /opt/oh-env/bin/python. No tarball upload needed.

    python test-oh-driver-mvp.py \
        --sandbox \
        --image  <sandbox-image> \
        --llm-config llm-config-example.json \
        --prompt "Create a file hello.txt with 'hi'. Then call the finish tool."

The driver config is constructed here and written to the sandbox as
/home/agent/oh_config.json (same paths the harness uses).

LLM config
----------
--llm-config must point at an OmegaConf JSON/YAML file whose env-var
interpolations (${oc.env:VAR}) have been satisfied by real env vars.
This is the same file format and resolution path used by
benchmarks/scaleswe/run_infer.py (LLM.model_validate_json).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# examples/coding_agent_rl/ is two levels below the repo root
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
_DRIVER_PATH = _THIS_DIR / "oh_driver.py"

# Fixed paths used by the harness inside the sandbox.
_SANDBOX_DRIVER = "/home/agent/oh_driver.py"
_SANDBOX_CONFIG = "/home/agent/oh_config.json"
_SANDBOX_PROMPT = "/home/agent/oh_prompt.txt"
_ENV_PREFIX = "/opt/oh-env"
_PY = f"{_ENV_PREFIX}/bin/python"

_DEFAULT_TOOLS = ["file_editor", "terminal", "task_tracker", "think", "finish"]


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def resolve_llm_config(path: str) -> dict:
    """OmegaConf-resolve an llm-config JSON/YAML, matching run_infer.py's pattern.

    Concretely:  json.dumps(OmegaConf.to_container(OmegaConf.load(path), resolve=True))
                 then LLM.model_validate_json(...) on the caller side.
    """
    try:
        from omegaconf import OmegaConf
    except ImportError:
        sys.exit(
            "omegaconf is required to resolve ${oc.env:...} interpolations in the LLM config.\n"
            "Install it with:  pip install omegaconf"
        )
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def build_driver_config(
    llm_dict: dict,
    workdir: str,
    prompt_path: str | None,
    tools: list[str],
    fake_user: bool,
    max_iterations: int,
) -> dict:
    """Build the JSON dict that oh_driver.py reads as its config."""
    cfg: dict = {
        "llm": llm_dict,
        "workdir": workdir,
        "tools": tools,
        "fake_user": fake_user,
        "max_iterations": max_iterations,
    }
    if prompt_path is not None:
        cfg["prompt_path"] = prompt_path
    return cfg


# ---------------------------------------------------------------------------
# Local mode
# ---------------------------------------------------------------------------
def run_local(args: argparse.Namespace) -> int:
    """Run the driver in a subprocess under the specified Python interpreter."""
    log("=== LOCAL MODE ===")

    python = args.python or sys.executable
    log(f"Python:      {python}")
    log(f"Driver:      {_DRIVER_PATH}")
    log(f"LLM config:  {args.llm_config}")

    llm_dict = resolve_llm_config(args.llm_config)
    log(f"LLM model:   {llm_dict.get('model', '<not set>')}")
    log(f"LLM base_url:{llm_dict.get('base_url', '<not set>')}")

    with tempfile.TemporaryDirectory(prefix="oh_driver_mvp_") as tmp:
        workdir = os.path.join(tmp, "work")
        os.makedirs(workdir)

        prompt_path = os.path.join(tmp, "prompt.txt")
        Path(prompt_path).write_text(args.prompt)

        config_path = os.path.join(tmp, "config.json")
        cfg = build_driver_config(
            llm_dict=llm_dict,
            workdir=workdir,
            prompt_path=prompt_path,
            tools=args.tools,
            fake_user=args.fake_user,
            max_iterations=args.max_iterations,
        )
        Path(config_path).write_text(json.dumps(cfg, indent=2))

        log(f"Workdir:     {workdir}")
        log(f"Prompt:      {args.prompt!r}")
        log(f"Tools:       {args.tools}")
        log(f"Max iters:   {args.max_iterations}")
        log("")

        t0 = time.monotonic()
        try:
            result = subprocess.run(
                [python, str(_DRIVER_PATH), config_path],
                timeout=args.time_budget,
            )
            ec = result.returncode
        except subprocess.TimeoutExpired:
            log(f"TIMEOUT after {args.time_budget}s")
            ec = 124

        elapsed = time.monotonic() - t0
        log("")
        log(f"Driver exit code: {ec}  elapsed: {elapsed:.1f}s")

        # Show what the agent produced.
        log("=== workdir contents ===")
        for root, dirs, files in os.walk(workdir):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fname in files:
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, workdir)
                size = os.path.getsize(fpath)
                log(f"  {rel}  ({size} bytes)")

    return ec


# ---------------------------------------------------------------------------
# Sandbox mode
# ---------------------------------------------------------------------------
async def run_sandbox(args: argparse.Namespace) -> int:
    """Boot a sandbox, verify /opt/oh-env is pre-mounted, send the driver,
    write config + prompt, run under /opt/oh-env/bin/python."""
    sys.path.insert(0, str(_REPO_ROOT))
    from slime.agent.sandbox import E2BSandbox, ensure_agent_user, exec_and_wait

    log("=== SANDBOX MODE ===")
    log(f"Image:       {args.image}")
    log(f"Size:        {args.size}")
    log(f"LLM config:  {args.llm_config}")

    llm_dict = resolve_llm_config(args.llm_config)
    log(f"LLM model:   {llm_dict.get('model', '<not set>')}")
    log(f"LLM base_url:{llm_dict.get('base_url', '<not set>')}")
    log(f"Prompt:      {args.prompt!r}")
    log(f"Tools:       {args.tools}")
    log(f"Max iters:   {args.max_iterations}")
    log("")

    workdir = args.workdir
    sb = E2BSandbox(args.image, timeout=args.sandbox_timeout, size=args.size)

    overall_t0 = time.monotonic()
    entered = False

    try:
        # -- create sandbox --------------------------------------------------
        t0 = time.monotonic()
        await sb.__aenter__()
        entered = True
        log(f"Sandbox created: {sb.sandbox_id}  ({time.monotonic()-t0:.1f}s)")

        # -- verify /opt/oh-env is pre-mounted --------------------------------
        ec, out, err = await sb.exec(
            f"test -x {_PY} && echo OK || echo MISSING",
            user="root",
            timeout=15,
            check=False,
        )
        status = (out + err).strip()
        if "MISSING" in status or ec != 0:
            raise RuntimeError(
                f"{_PY} not found in the sandbox image. "
                "Make sure the image was built with the oh-env layer mounted at /opt/oh-env."
            )
        log(f"/opt/oh-env/bin/python: present (no tarball needed)")

        # -- ensure agent user + workdir -------------------------------------
        await ensure_agent_user(sb, workdir)
        log(f"Agent user ready, workdir: {workdir}")

        # -- write driver, config, prompt ------------------------------------
        cfg = build_driver_config(
            llm_dict=llm_dict,
            workdir=workdir,
            prompt_path=None,  # use the default /home/agent/oh_prompt.txt
            tools=args.tools,
            fake_user=args.fake_user,
            max_iterations=args.max_iterations,
        )

        await sb.write_file(_SANDBOX_DRIVER, _DRIVER_PATH, user="agent")
        await sb.write_file(_SANDBOX_CONFIG, json.dumps(cfg), user="agent")
        await sb.write_file(_SANDBOX_PROMPT, args.prompt, user="agent")
        log("Driver, config, and prompt written to sandbox")

        # -- run driver ------------------------------------------------------
        start_cmd = f"{_PY} {_SANDBOX_DRIVER} {_SANDBOX_CONFIG}"
        env = {"HOME": "/home/agent"}
        meta_dir = f"{workdir}/.harness"
        await sb.exec(
            f"mkdir -p {meta_dir} && chown agent:agent {meta_dir}",
            user="root",
            check=True,
            timeout=30,
        )
        log(f"Running driver under {_PY} …")
        t0 = time.monotonic()
        ec, output = await exec_and_wait(
            sb,
            cmd=start_cmd,
            user="agent",
            env=env,
            workdir=workdir,
            out_file=f"{meta_dir}/driver.out",
            time_budget_sec=args.time_budget,
            tag="oh-driver",
            want_output=True,
        )
        elapsed = time.monotonic() - t0

        log(f"Driver exit code: {ec}  elapsed: {elapsed:.1f}s")
        if output:
            log("=== driver output ===")
            print(output)

        # -- show what the agent produced ------------------------------------
        _, ls_out, _ = await sb.exec(
            f"find {workdir} -not -path '*/.harness*' -not -name '.*' -type f",
            user="agent",
            timeout=15,
            check=False,
        )
        if ls_out.strip():
            log("=== workdir files ===")
            for line in ls_out.strip().splitlines():
                print(f"  {line}")

        log("")
        log(f"Total elapsed: {time.monotonic()-overall_t0:.1f}s")
        log("SUCCESS" if ec == 0 else f"FAILED (exit={ec})")
        return ec

    except Exception as e:
        log(f"FAILED: {type(e).__name__}: {e}")
        raise

    finally:
        if entered and not args.keep:
            try:
                await sb.__aexit__(None, None, None)
                log(f"Sandbox {sb.sandbox_id} killed")
            except Exception as e:
                log(f"Sandbox cleanup failed: {type(e).__name__}: {e}")
        elif entered:
            log(f"Keeping sandbox: {sb.sandbox_id}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MVP: run oh_driver.py with a live LLM locally or in a sandbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--llm-config",
        required=True,
        metavar="PATH",
        help="OmegaConf JSON/YAML with ${oc.env:VAR} resolved from env (same format as llm-config-example.json)",
    )
    p.add_argument(
        "--prompt",
        default="Create a file named hello.txt containing the text 'hello from openhands'. "
        "Then call the finish tool.",
        help="Task prompt sent to the agent",
    )
    p.add_argument(
        "--tools",
        nargs="+",
        default=_DEFAULT_TOOLS,
        metavar="TOOL",
        help=f"Tool allowlist (default: {_DEFAULT_TOOLS})",
    )
    p.add_argument("--fake-user", action="store_true", help="Enable fake-user nudge loop")
    p.add_argument("--max-iterations", type=int, default=20, metavar="N")
    p.add_argument(
        "--time-budget",
        type=int,
        default=300,
        metavar="SEC",
        help="Wall-clock budget for the driver run (seconds, default 300)",
    )

    # local-mode options
    local = p.add_argument_group("local mode (default)")
    local.add_argument(
        "--python",
        metavar="PATH",
        help="Python interpreter to use (must have openhands installed). " "Defaults to the current interpreter.",
    )

    # sandbox-mode options
    sb = p.add_argument_group("sandbox mode (--sandbox)")
    sb.add_argument("--sandbox", action="store_true", help="Run in an E2B sandbox instead of locally")
    sb.add_argument("--image", metavar="IMAGE", help="Sandbox container image (required with --sandbox)")
    sb.add_argument(
        "--workdir",
        default="/home/agent/work",
        metavar="PATH",
        help="Working directory inside the sandbox (default: /home/agent/work)",
    )
    sb.add_argument("--size", default=os.environ.get("SLIME_AGENT_E2B_SANDBOX_SIZE", "md"))
    sb.add_argument("--sandbox-timeout", type=int, default=3600, metavar="SEC")
    sb.add_argument("--keep", action="store_true", help="Keep the sandbox alive after the run")

    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.sandbox:
        if not args.image:
            sys.exit("--image is required in sandbox mode")
        if not os.environ.get("E2B_API_KEY"):
            sys.exit("E2B_API_KEY must be set for sandbox mode")
        return asyncio.run(run_sandbox(args))
    else:
        return run_local(args)


if __name__ == "__main__":
    raise SystemExit(main())
