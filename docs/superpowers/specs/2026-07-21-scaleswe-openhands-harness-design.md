# Design: scaleswe coding-agent RL with the OpenHands SDK harness

**Date:** 2026-07-21
**Status:** Approved design, ready for implementation plan
**Scope:** Add OpenHands software-agent-sdk as a new coding-agent harness in
slime's existing agent-RL example, running end-to-end scaleswe RL against the
**existing `E2BSandbox`**. The runtime-service (docker/k8s) sandbox backend is a
**separate follow-up spec** and is out of scope here.

---

## 1. Background & goal

slime already ships an end-to-end SWE coding-agent RL example
(`examples/coding_agent_rl/`). Its design:

- A self-contained coding-agent CLI (claude-code or codex) runs **inside** a
  fresh E2B sandbox per sample and "dials back" to a slime HTTP adapter
  (`AnthropicAdapter` / `OpenAIAdapter`). The adapter renders each turn with the
  served model's chat template, calls SGLang `/generate` with
  `return_logprob=True`, and captures the exact sampled token ids into a
  `TrajectoryManager`, which linearizes into training `Sample`s
  (string-in / token-out contract).
- The model produces a `git diff`; it is graded in a **second, clean** sandbox
  by `swe.py` (no test-cheating), yielding `reward ∈ {0.0, 1.0}`.
- `generate.py` is a per-sample `generate()` wired via
  `--custom-generate-function-path`, orchestrating boot → workspace prep →
  harness → diff → eval → `adapter.finish_session`.

**Goal:** run the same loop with the **OpenHands software-agent-sdk** as the
harness instead of claude-code/codex, for the scaleswe task, and train with GRPO
end-to-end.

### Why OpenHands is architecturally different (and why it still fits)

claude-code and codex are self-contained CLIs. OpenHands is a **Python agent
loop**: it reaches its LLM over an OpenAI-compatible `base_url` via litellm, and
executes tools in a *workspace* (`LocalWorkspace` = local process/filesystem;
`Docker`/`Flex`/`Remote` = an `agent-server` container driven over HTTP/WebSocket).

The key enabling fact: **OpenHands' litellm traffic is plain
`/v1/chat/completions`, which slime's `OpenAIAdapter` already serves and already
keys on `Authorization: Bearer <sid>`** — the exact seam Codex uses. So by
running the OpenHands loop *inside* the sandbox with a `LocalWorkspace`, and
pointing its `LLM.base_url` at the slime adapter, token capture / logprobs /
trajectory handling work with **zero changes** to the adapter or trajectory
layers.

### Orchestration model (context that drove backend decisions)

slime has **no distributed dispatch of rollout jobs**. `RolloutManager` is a
single `@ray.remote` actor (`num_cpus=1, num_gpus=0`) on the head node; its
`generate_rollout` runs an asyncio event loop that fans out all per-sample
`generate()` coroutines concurrently (Semaphore-gated). Each `generate()` boots
its sandbox via a **remote provisioning call** (E2B today). The GPU-bearing,
distributed Ray actors are the SGLang engines, reached over HTTP. This is why
the eventual docker/k8s backend should be a *remote runtime service*, not
local-docker-per-node — but that is the follow-up spec.

---

## 2. Decisions (locked)

| Decision | Choice |
| --- | --- |
| Deliverable | Full end-to-end RL training (runnable `run-*.sh`). |
| Where the OpenHands loop runs | **Inside the sandbox**, `LocalWorkspace`, tools hit the sandbox FS directly; LLM dials back to slime's `OpenAIAdapter`. |
| SDK version | **Dev version** (the vendored `thirdparty/benchmarks-main/vendor/software-agent-sdk`), not the official release. Must remain swappable during active SDK development. |
| Env delivery | Prebuilt, self-contained **python-build-standalone 3.12 venv at a fixed prefix** (`/opt/oh-env`), with the 4 OpenHands packages **editable-installed** from an in-prefix source path. Baked into one tarball; boot is pure `tar x`. A host-side repackage tool relinks fresh SDK source without a full env rebuild. |
| Fake-user nudges | Configurable via `SWE_OH_FAKE_USER`, **default off** (`conv.run()` only). |
| Eval / reward | **Reuse slime's existing `swe.py` scaleswe grader** unchanged. Do NOT use the benchmark's `ScaleSWEJudge`/AweAgent. |
| Sandbox backend | **Existing `E2BSandbox`** for this spec. Runtime-service (docker/k8s) backend = separate follow-up spec. |

---

## 3. Architecture

```
train.py  →  RolloutManager (single Ray actor, head node)
   └─ generate_rollout: asyncio fan-out of generate_and_rm_group  (Semaphore-gated)
        └─ examples/coding_agent_rl/generate.py :: generate()      [≈unchanged]
             ├─ boot_agent_sandbox → E2BSandbox(image_url)          [unchanged]
             ├─ swe.prepare_workspace (pre_commands + PROBLEM_STATEMENT.md)  [unchanged]
             ├─ OpenHandsHarness.run(sb, workdir, sid, adapter_url, prompt)   ← NEW harness
             │     · install_cli:   tar x  SLIME_AGENT_OH_ENV_TARBALL → /opt/oh-env ; verify import
             │     · write_config:  drop oh_driver.py + oh_config.json + prompt file into sandbox
             │     · launch_and_wait: /opt/oh-env/bin/python oh_driver.py  (detached via run_agent/exec_and_wait)
             │          in-sandbox loop:
             │              LLM(model="openai/slime-actor", base_url=adapter_url+"/v1", api_key=sid)
             │              → Agent(get_default_tools(enable_browser=False), cli_mode=True)
             │              → Conversation(agent, LocalWorkspace(workdir))
             │              → send_message(prompt) → conv.run()  [+ optional fake-user nudges]
             │          every LLM turn dials back:
             │              litellm → OpenAIAdapter /v1/chat/completions
             │              → SGLang /generate return_logprob → TrajectoryManager   [unchanged]
             ├─ swe.git_diff                                        [unchanged]
             ├─ swe.run_evaluation (fresh E2B sandbox, scaleswe grader)   [unchanged]
             └─ adapter.finish_session → list[Sample]               [unchanged]
```

`_AGENTS` in `generate.py` gains one row:
`"openhands": (OpenHandsHarness, OpenAIAdapter)`. Selected by `SWE_AGENT=openhands`.

---

## 4. Component: prebuilt OpenHands environment tarball

### 4.1 Build (host-side, once)

1. Materialize a **python-build-standalone** CPython 3.12 at fixed prefix
   `/opt/oh-env` (self-contained: bundles its own libpython, independent of the
   base image's Python/glibc — important because scaleswe base images span many
   distros).
2. Place the 4 OpenHands packages as source at a fixed in-prefix path, e.g.
   `/opt/oh-env/src/software-agent-sdk/{openhands-sdk,openhands-tools,openhands-workspace,openhands-agent-server}`.
   (`openhands-agent-server` is not strictly needed for `LocalWorkspace` but is
   cheap to include and keeps parity with the vendored workspace.)
3. `/opt/oh-env/bin/pip install -e` the packages **from that in-prefix path**,
   then install all third-party deps (litellm, pydantic, httpx, jinja2, …).
   Editable install records the **path** (`/opt/oh-env/src/...`), valid the
   instant the tarball is unpacked to the fixed prefix.
4. `tar` the whole `/opt/oh-env` prefix → `SLIME_AGENT_OH_ENV_TARBALL`.

**Rationale:** an editable install tracks path, not content — so swapping the
source dir needs no reinstall. Single artifact, boot is pure `tar x`, no pip / no
`PYTHONPATH` juggling, dev source still swappable.

### 4.2 Repackage tool: `tools/repackage_oh_env.py`

Host-side dev tool to relink fresh SDK source into an existing env tarball
**without** rebuilding the venv or re-resolving deps:

```
python tools/repackage_oh_env.py \
  --env-tarball  oh-env.tar \
  --sdk-src      thirdparty/benchmarks-main/vendor/software-agent-sdk \
  --out          oh-env.relinked.tar
```

Behavior: unpack env to a temp dir → rsync `--delete` `sdk-src` over
`/opt/oh-env/src/software-agent-sdk/` → re-tar. Rebuild the *full* env only when
third-party **deps** change; day-to-day SDK code edits only re-pack the source.

### 4.3 Boot-time install (`OpenHandsHarness.install_cli`)

- `tar x SLIME_AGENT_OH_ENV_TARBALL -C /` (fixed prefix `/opt/oh-env`).
- Verify: `/opt/oh-env/bin/python -c "import openhands.sdk, openhands.tools"`.
- No node, no npm, no pip, no network egress at boot.

---

## 5. Component: `OpenHandsHarness` (`slime/agent/harness/openhands.py`)

Subclasses `BaseHarness` (same contract as claude_code/codex; reuses `run_agent`
/ `exec_and_wait` for the detached-spawn + done-marker transport).

Class attributes (agent-layer `SLIME_AGENT_*` prefix):
- `name = "openhands"`
- `env_tarball_env = "SLIME_AGENT_OH_ENV_TARBALL"`
- `extra_envs_env = "SLIME_AGENT_OH_EXTRA_ENVS"` (JSON object merged into the
  driver's env; escape hatch)

Methods:
- **`install_cli(sb)`** — §4.3. (Does NOT call `install_node22`/`install_npm_cli`;
  those remain for the CLI harnesses.)
- **`write_config(sb, ctx)`** — write under user `agent`:
  - `/home/agent/oh_driver.py` — the loop (see §6).
  - `/home/agent/oh_config.json` — `{adapter_url, session_id, model_label,
    workdir, fake_user, max_iterations, tools}`.
  - `/home/agent/oh_prompt.txt` — the task prompt (passed via file to avoid
    shell-quoting a long problem statement).
- **`launch_and_wait(sb, ctx, prompt, time_budget_sec)`** —
  `start_cmd = "/opt/oh-env/bin/python /home/agent/oh_driver.py /home/agent/oh_config.json"`;
  env carries only what the driver needs (the adapter URL/sid are in the config,
  not env). Merge `SLIME_AGENT_OH_EXTRA_ENVS` last. Delegate to `run_agent(...)`.

**Config knobs read from env (in `generate.py`'s `SweConfig` or harness):**
- `SWE_OH_FAKE_USER` (default `0`)
- `SWE_OH_MAX_ITERATIONS` (default matches the benchmark's `--max-iterations`,
  e.g. 100)
- `SWE_OH_TOOLS` (default `file_editor,terminal,task_tracker,think,finish`) —
  externally-specified tool allowlist, see §6.1.

---

## 6. Component: in-sandbox driver (`examples/coding_agent_rl/oh_driver.py`)

Runs inside the sandbox with `/opt/oh-env/bin/python`; imports the baked SDK.
Written at `write_config` time (so it's editable without repackaging the env).

```python
cfg = json.load(open(sys.argv[1]))
prompt = open("/home/agent/oh_prompt.txt").read()

from openhands.sdk import LLM, Agent, Conversation
from openhands.sdk.workspace import LocalWorkspace
from openhands.sdk.tool import Tool

tools, include_default = build_tools(cfg["tools"])   # see §6.1

llm = LLM(
    model="openai/slime-actor",          # openai/* → litellm OpenAI-compatible path
    base_url=cfg["adapter_url"] + "/v1",  # slime OpenAIAdapter
    api_key=cfg["session_id"],            # → Authorization: Bearer → adapter session key
)
agent = Agent(llm=llm, tools=tools,
              include_default_tools=include_default,   # controls Think/Finish
              system_prompt_kwargs={"cli_mode": True})
conv = Conversation(agent=agent, workspace=LocalWorkspace(cfg["workdir"]),
                    max_iteration_per_run=cfg["max_iterations"])
conv.send_message(prompt)

if cfg["fake_user"]:
    run_conversation_with_fake_user_response(conv, ...)  # see open item
else:
    conv.run()
```

**Open item (flag in plan):** `run_conversation_with_fake_user_response` lives in
`benchmarks/utils`, not the SDK. Default is fake-user **off**, so the primary
path is `conv.run()`. For the nudge path, **vendor the ~40-line helper into
`oh_driver.py`** (preferred — keeps the driver self-contained) rather than depend
on benchmark utils being on the baked env.

### 6.1 Externally-specified tool allowlist

The set of tools is **configurable from outside** (env → `cfg["tools"]`), not
hardcoded. Two facts from the vendored SDK drive the design:

- **Tool name is `_camel_to_snake(ClassName).removesuffix("_tool")`**, and a tool
  is only usable once the module that calls `register_tool(...)` for it has been
  imported. The default preset registers `file_editor` / `terminal` /
  `task_tracker` (importing `openhands.tools.{file_editor,terminal,task_tracker}`);
  the **legacy** preset (`openhands.tools.preset.legacy`) additionally registers
  `str_replace_editor` (`StrReplaceEditorTool`) and `execute_bash`
  (`ExecuteBashTool`). Legacy names do not exist until that module is imported.
- **Think/Finish are builtins on a different axis.** They are NOT passed in the
  `tools=` list; they are controlled by `Agent(include_default_tools=[...])` using
  **class names** `"ThinkTool"` / `"FinishTool"` (SDK default = both; `[]`
  disables all).

So `build_tools(names)` maps a flat requested list onto the two axes:

| Requested name (config) | Axis | How the driver enables it |
| --- | --- | --- |
| `file_editor` | `tools=` | import `openhands.tools.file_editor`; `Tool(name="file_editor")` |
| `str_replace_editor` | `tools=` | import `openhands.tools.preset.legacy`; `Tool(name="str_replace_editor")` |
| `terminal` | `tools=` | import `openhands.tools.terminal`; `Tool(name="terminal")` |
| `execute_bash` | `tools=` | import `openhands.tools.preset.legacy`; `Tool(name="execute_bash")` |
| `task_tracker` | `tools=` | import `openhands.tools.task_tracker`; `Tool(name="task_tracker")` |
| `think` / `ThinkTool` | `include_default_tools=` | add `"ThinkTool"` |
| `finish` / `FinishTool` | `include_default_tools=` | add `"FinishTool"` |

`build_tools` accepts a canonical set of names, imports the registering module
for each `tools=`-axis entry, routes Think/Finish to `include_default_tools`, and
raises on an unknown name (fail fast rather than silently drop). Browser tools
are intentionally not exposed (no outbound internet in the sandbox).

**Config knob:** `SWE_OH_TOOLS` — comma-separated allowlist, forwarded into
`cfg["tools"]`. **Default:**
`file_editor,terminal,task_tracker,think,finish` (the default-preset trio plus
both builtins — matches the benchmark's non-legacy default behavior). A legacy
run is expressed by setting
`SWE_OH_TOOLS=str_replace_editor,execute_bash,task_tracker,think,finish` (no
separate `SWE_OH_LEGACY_TOOLS` flag needed — the tool list *is* the switch).

---

## 7. Eval / reward

**Reuse `examples/coding_agent_rl/swe.py` unchanged.** It already implements
`PROTOCOL_SCALESWE` for the exact scaleswe row schema (`image`/`image_url` +
`pre_commands` + one of `swepro`/`eval_cmd`/`f2p_script`, graded "exit 0 ==
solved" in a fresh sandbox). Harness-agnostic by construction; the harness swap
requires no eval change. `reward ∈ {0.0, 1.0}` feeds GRPO as today.

Rejected: the benchmark's `ScaleSWEJudge`/`ExecutionBasedJudge` (delegates to
`awe_agent` + a host-side `DockerRuntime`) — duplicates grading slime owns, adds
a heavy thirdparty dep and a second container path, no signal gain. Could be
added later as an opt-in `SWE_EVAL_PROTOCOL` value if judge-parity eval is ever
wanted; not in this spec.

Dataset stays standard slime JSONL
(`--input-key prompt --label-key label --metadata-key metadata`); scaleswe rows
already carry the needed fields.

---

## 8. Run script + env knobs

New launcher `examples/coding_agent_rl/run_qwen36_35b_a3b_scaleswe_openhands_8nodes.sh`,
cloned from the existing 8-node script with these deltas:

- `SWE_AGENT=openhands`.
- **Add:** `SLIME_AGENT_OH_ENV_TARBALL` (prebuilt editable env), `SWE_OH_FAKE_USER=0`,
  optional `SWE_OH_MAX_ITERATIONS`, `SWE_OH_TOOLS`, optional `SLIME_AGENT_OH_EXTRA_ENVS`.
- **Remove:** claude-code knobs (`SLIME_AGENT_CC_TARBALL`, `SLIME_AGENT_CC_EXTRA_ARGS`,
  `AGENTS_JSON`, `SETTINGS_JSON`) and the node tarball (env is self-contained).
- Keep model-appropriate SGLang parsers
  (`--sglang-tool-call-parser qwen3_coder --sglang-reasoning-parser qwen3`); the
  `OpenAIAdapter` reuses them to parse OpenHands' tool calls.
- Update `RUNTIME_ENV_JSON` key list to propagate the new vars to Ray workers
  (add `SLIME_AGENT_OH_ENV_TARBALL`, `SWE_OH_FAKE_USER`, `SWE_OH_MAX_ITERATIONS`,
  `SWE_OH_TOOLS`, `SLIME_AGENT_OH_EXTRA_ENVS`; drop the CC keys).

Everything else (Megatron/GRPO/SGLang args, adapter host wiring,
`--rollout-max-context-len/response-len`, fan-out semantics) carries over.

---

## 9. Testing

CPU/unit tests as plain scripts calling `pytest.main([__file__])` (per
`add-tests-and-ci`):

- **`tests/test_agent/test_openhands_harness.py`** — fake in-memory `Sandbox`;
  assert `install_cli` untars to `/opt/oh-env` + runs the import check;
  `write_config` drops `oh_driver.py` / `oh_config.json` / `oh_prompt.txt` with
  correct `adapter_url` / `session_id` / `workdir`; `launch_and_wait` builds the
  right `start_cmd` and routes through `run_agent`.
- **`oh_driver.py` construction unit** — mock SDK classes; assert `LLM`/`Agent`/
  `Conversation` are built with correct `base_url` / `api_key` / `model`;
  cover fake-user on/off branches. Include a **`build_tools` unit** (§6.1):
  default list → `file_editor,terminal,task_tracker` on the `tools=` axis +
  `ThinkTool,FinishTool` on `include_default_tools`; legacy list → `str_replace_editor,execute_bash`;
  unknown name → raises; each `tools=` entry imports its registering module.
- **`tools/repackage_oh_env.py` test** — relink SDK source into an unpacked env
  fixture dir and re-tar (tmp fixtures, no real large artifact).
- **Reuse unchanged** `tests/test_agent/test_trajectory_manager_branching.py` —
  proves the token/trajectory contract the `OpenAIAdapter` already satisfies.
- **GPU e2e smoke test** — out of scope (needs real images + GPUs); note as
  follow-up, gated by CI label per existing convention.

---

## 10. File-by-file change list

**New:**
- `slime/agent/harness/openhands.py` — `OpenHandsHarness(BaseHarness)`.
- `examples/coding_agent_rl/oh_driver.py` — in-sandbox loop.
- `tools/repackage_oh_env.py` — host-side dev repackager.
- `examples/coding_agent_rl/run_qwen36_35b_a3b_scaleswe_openhands_8nodes.sh` — launcher.
- `tests/test_agent/test_openhands_harness.py` (+ driver / repackage unit tests).
- Docs: build-the-env-tarball section (how to produce `SLIME_AGENT_OH_ENV_TARBALL`
  with the editable install; how to use `repackage_oh_env.py`).

**Modified:**
- `slime/agent/harness/__init__.py` — export `OpenHandsHarness`.
- `examples/coding_agent_rl/generate.py` — add `_AGENTS` row
  `"openhands": (OpenHandsHarness, OpenAIAdapter)`; add `SWE_OH_*` to `SweConfig`
  if the harness reads them via config rather than env directly.
- `examples/coding_agent_rl/README.md` — OpenHands harness + env-tarball docs.

**Explicitly unchanged:** `OpenAIAdapter`, `TrajectoryManager`, `swe.py`,
`parsing.py`, `slime/agent/sandbox.py`, all Megatron/Ray training code.

---

## 11. Out of scope (follow-up specs)

- **Runtime-service sandbox backend** (docker/k8s): `RuntimeServiceSandbox`
  against the `Sandbox` protocol + `make_sandbox()` selector +
  `SLIME_AGENT_SANDBOX_BACKEND` knob. Separate spec; transport TBD from the
  service interface description.
- GPU end-to-end smoke test + CI matrix registration.
- Legacy-tools / judge-parity ablation knobs.

---

## 12. Risks & open items

- **Python 3.12 requirement** — handled by shipping a self-contained
  python-build-standalone interpreter; does not rely on the base image's Python.
- **Env tarball size / boot cost** — full env is large-ish (interpreter + litellm
  dep tree). Boot is a single `tar x` (fast, no network). Mitigation if needed:
  cache the unpacked prefix across samples on a warm sandbox image, or bake the
  env into the base images later. Not required for v1.
- **`cli_mode` / system prompt parity** — the benchmark uses
  `system_prompt_kwargs={"cli_mode": True}`; the driver mirrors this. Prompt
  template (the benchmark's `default.j2`) is NOT reused — slime hands the task
  instruction via `swe.SWE_PROMPT` (already overridable through the existing
  `SWE_CC_PROMPT` env knob). Confirm this simple prompt is acceptable vs.
  adopting the benchmark's richer 8-phase prompt; if the latter is wanted, add a
  dedicated `SWE_OH_PROMPT` knob rather than overloading `SWE_CC_PROMPT`.
- **Fake-user helper vendoring** — vendor into `oh_driver.py` to keep it
  self-contained.
- **litellm provider string** — `model="openai/slime-actor"` selects litellm's
  OpenAI-compatible path; verify litellm forwards `api_key` as
  `Authorization: Bearer` to `base_url` (the sid seam). Validate early in impl.
