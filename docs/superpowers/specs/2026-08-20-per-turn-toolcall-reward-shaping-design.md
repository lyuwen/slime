# Per-turn tool-call reward shaping for agentic RL

**Date:** 2026-08-20
**Status:** Design approved, pending spec review

## Problem

slime assigns a single scalar reward per trajectory. For multi-turn agentic
rollouts (`slime/agent/trajectory.py`), that outcome reward is broadcast
identically onto every turn and every response token
(`get_trajectory` → `s.reward = reward`, then GRPO
`get_grpo_returns` spreads the scalar across the whole sequence). There is no
per-turn signal: a trajectory that solved the task via clean tool use and one
that solved it while emitting many malformed tool calls receive identical
per-token advantages.

We want to shape the advantage per turn using a deterministic signal about
tool-call correctness, so the model is nudged away from malformed/incorrect
tool invocations without decoupling from whether the task was actually solved.

The signal source is the existing deterministic annotator in
`thirdparty/toolcall-annotation/` — specifically the pure detection functions in
`toolcall_annotation/annotators/toolcall_correctness_impl.py`
(`check_task_tracker`, `check_finish`, `check_think`, `check_str_replace_editor`,
`check_execute_bash`, `parse_arguments`), which take one tool call plus its tool
response and return a list of detected error types.

## Goals / non-goals

**Goals**
- Compute per-turn tool-call error counts online, in-process, during the
  coding-agent RL rollout.
- Convert them to an additive per-token shaping term on top of the existing
  GRPO outcome advantage.
- Keep total shaping per trajectory comparable to the outcome-reward scale
  regardless of trajectory length.
- Ship OFF by default; existing runs unaffected.
- Keep the annotator dependency out of slime core.

**Non-goals**
- Replacing the outcome reward with per-turn reward.
- Dense discounted (REINFORCE++-style) credit assignment over turns.
- Any change to the annotator's class/registry/CLI/report surface.
- New base classes or abstractions in the training kernel (per slime
  `CONTRIBUTING.md`).

## Key decisions (from brainstorming)

| Decision | Choice |
|---|---|
| When to score | Online, in-process during rollout |
| Execution model | Batched at `finish_session` / `get_trajectory` (off the per-turn hot path) |
| Combination with outcome reward | **Additive** shaping on top of GRPO outcome advantage |
| Per-turn value | `-beta * (errored tool-call count in that turn)` — self-bounded by number of tool calls in the turn (0 / 1 / N) |
| Length bounding | **Per-trajectory budget cap**: if the summed `|shaping|` over a trajectory exceeds `budget`, scale every turn's penalty down proportionally so the total equals `budget` (only shrinks, never grows) |
| Storage on Sample | Dense per-token vector `Sample.metadata["toolcall_turn_shaping"]`, length == `response_length`, aligned to `loss_mask` |
| Scorer injection | Constructor-injected callback on `TrajectoryManager` (core never imports the annotator); default `None` = today's behavior |
| Config | `SWE_TOOLCALL_SHAPING_BETA` (default `0.0`, off), `SWE_TOOLCALL_SHAPING_BUDGET` (default `1.0`), auto-forwarded to Ray workers by the `SWE_`-prefix loop in the run script |

## Data flow (the seam)

The one place where the message tree (with `tool_calls` + tool responses), the
per-turn token spans, and the not-yet-consumed builder state all coexist is
`TrajectoryManager.get_trajectory` (invoked from
`BaseAdapter.finish_session`). That is where scoring happens.

```
rollout (examples/coding_agent_rl/generate.py)
  └─ adapter.finish_session(reward=outcome)
       └─ manager.get_trajectory(...)                       [scoring here]
            • for each generated assistant turn node: gather its tool-response child
            • toolcall_correctness_impl → errored-call count for that turn
            • raw_turn_penalty = -beta * error_count, over that turn's token span
            • budget cap: sum |penalty| across ALL samples of the session;
              if > budget, one uniform scale factor applied to every turn
            • write dense vector → Sample.metadata["toolcall_turn_shaping"]
              (len == response_length, sliced [start:] exactly like loss_mask)
  └─ rollout.py _convert_samples_to_train_data
       • promote metadata vector → train_data["toolcall_turn_shaping"] (per-token key)
  └─ rollout.py _split_train_data_by_dp
       • add key to the per-token DP-split allowlist (line ~857) so it is
         partitioned per sample
  └─ actor.py _get_rollout_data
       • CP-slice it exactly like rollout_log_probs (the line ~273 loop)
  └─ loss.py compute_advantages_and_returns  (custom_advantage_function_path)
       • advantages = GRPO returns + rollout_data["toolcall_turn_shaping"]
```

**Five mandatory touch-points.** Per-token data that is not in the hardcoded
DP-split allowlist silently never reaches the loss function, so all five are
required: (1) trajectory manager scoring, (2) convert-to-train-data promotion,
(3) DP-split allowlist, (4) CP-slice, (5) custom advantage function.

## Per-turn → per-token mapping

The annotator works in message space; training works in token space. The
mapping must be exact or the shaping vector will not align with `loss_mask`.

- `_SampleBuilder` already sets `last_response_start_idx` immediately before
  appending each turn's `output_ids` (`trajectory.py`). That gives the token
  span `[start, start + len(output_ids))` for the turn's generated response —
  precisely the tokens to shape.
- Extend `_SampleBuilder.append_turn` to record, per turn, a
  `(response_start, response_len, turn_index, trained)` entry.
- `to_sample` emits a dense `turn_shaping` vector of length `len(loss_mask)`,
  then slices `[start:]` exactly as it does for `loss_mask` / `logprobs`, so the
  vector is automatically aligned to the response region and undergoes the same
  `max_sample_tokens` truncation.
- Untrained / re-emitted turns (`loss_mask == 0`, shared-prefix leaves) get 0 —
  they carry no gradient.

**Scoring a turn.** For each generated assistant turn node in the chain, gather
its `message` (carries `tool_calls`) and its tool-response child node(s), call
the impl's per-tool `check_*` (dispatched by tool name) via a new pure helper
`count_turn_toolcall_errors(assistant_message, tool_responses_by_id) -> int`,
and fill that turn's token span with `-beta * error_count`.

**Budget cap ordering.** The cap is per-trajectory (whole agent episode), but a
trajectory can linearize into multiple `Sample`s (forks). So:
1. Build raw per-turn penalties for every sample of the session.
2. Sum `|total|` across all those samples.
3. If `|total| > budget`, derive one `scale = budget / |total|` and apply it
   uniformly to every turn of every sample.
4. Write the (possibly scaled) vectors, then broadcast `reward` as today.

**Edge cases handled explicitly**
- REALIGN drift: a turn's span is overwritten (`_align_to_prompt`); re-score
  against the realigned span so the vector stays aligned.
- Turn with no tool call: error count 0 → no penalty.
- `beta == 0.0`: scorer returns all-zeros → whole feature is a no-op.

## Module boundaries (where code lives)

The annotator is a standalone package in `thirdparty/` (not a slime dependency;
nothing in `slime/` or `examples/` imports it today). Treat it as a pure library
and keep the coupling inside the example that opts in.

**1. Third-party package** — one additive pure helper, no other changes:
```python
# toolcall_annotation/annotators/toolcall_correctness_impl.py  (append)
def count_turn_toolcall_errors(assistant_message, tool_responses_by_id) -> int:
    """Total detected tool-call errors across one assistant turn's tool calls."""
```
Plus a unit test. Existing class/registry/CLI/report untouched.

**2. slime core** — minimal mechanical plumbing, no new abstractions:
- `slime/agent/trajectory.py`
  - `_SampleBuilder`: record per-turn `(response_start, response_len, turn_index,
    trained)`; `to_sample` emits the aligned dense vector.
  - `TrajectoryManager.__init__(..., turn_scorer: Callable | None = None)`;
    `get_trajectory` calls `turn_scorer` when set to compute per-turn error
    counts and applies `beta` + budget cap. Default `None` = current behavior
    byte-for-byte. The manager stays tokenizer-free and annotator-free — it only
    invokes the injected callback.
- `slime/agent/adapters/common.py`
  - `BaseAdapter.__init__(..., turn_scorer=None)` passthrough into the
    `TrajectoryManager` it constructs. No signature change to `finish_session`
    or `get_trajectory`.
- `slime/ray/rollout.py`
  - `_convert_samples_to_train_data`: if any sample carries
    `metadata["toolcall_turn_shaping"]`, add `train_data["toolcall_turn_shaping"]`
    (per-sample list of per-token vectors).
  - `_split_train_data_by_dp`: add `"toolcall_turn_shaping"` to the per-token
    allowlist so it is partitioned per sample.
- `slime/backends/megatron_utils/actor.py`
  - `_get_rollout_data`: CP-slice `toolcall_turn_shaping` in the existing
    `rollout_log_probs` / `teacher_log_probs` loop (same
    `slice_log_prob_with_cp` + device/dtype handling).

**3. Example wiring** (`examples/coding_agent_rl/`) — where the two worlds meet:
- New module `turn_shaping.py`:
  - `make_turn_scorer()` → a callback that imports `toolcall_annotation` and
    returns the raw per-turn **errored-call count** (an int per turn),
    annotator-specific and knob-free. `beta` and `budget` are NOT applied here:
    the scorer only knows how to count errors, keeping the annotator coupling
    purely about detection.
  - The manager owns the reward math. `TrajectoryManager` receives `beta` and
    `budget` (alongside the scorer) and applies `-beta * count` per turn plus the
    per-trajectory budget cap. Rationale: the budget cap must sum across all
    samples of a session, which only the manager sees at linearization time, so
    both scalar knobs live with it rather than in the scorer.
  - `compute_advantage(args, rollout_data)` → GRPO returns
    (`get_grpo_returns` on `rollout_data["rewards"]` + `kl`) plus
    `rollout_data["toolcall_turn_shaping"]` added per token; sets
    `rollout_data["advantages"]` and `rollout_data["returns"]`. Falls back to
    plain GRPO when the key is absent or all-zero.
- `generate.py`: read `SWE_TOOLCALL_SHAPING_BETA` / `SWE_TOOLCALL_SHAPING_BUDGET`
  from env into `CONFIG`, build the scorer once, and pass it (plus budget) when
  constructing the adapter.

**Invariant:** slime core never imports `toolcall_annotation`. Core takes a
generic per-turn scorer callback; the example supplies the annotator-backed one.

## Configuration

Consumed rollout-side; the train-side advantage function needs no knobs (it
reads an already-finalized vector).

- `SWE_TOOLCALL_SHAPING_BETA` — float, default `0.0`. `0.0` disables the feature
  (all-zero shaping vector, no-op).
- `SWE_TOOLCALL_SHAPING_BUDGET` — float, default `1.0`. Max total `|shaping|`
  per trajectory; chosen to be comparable to the (group-normed) outcome-reward
  scale.

Both are auto-forwarded to Ray workers by the existing `SWE_`-prefix loop in
`examples/coding_agent_rl/run_021_32b_a4b_scaleswe_openhands_8nodes.sh`
(`for k, v in os.environ.items(): if k.startswith("SLIME_") or
k.startswith("SWE_")`), so no per-var edit to the runtime-env block is needed.

Advantage wiring uses the standard slime hook in the run script:
```
--custom-advantage-function-path examples.coding_agent_rl.turn_shaping.compute_advantage
```
Harmless when `beta=0` (falls back to plain GRPO).

## Testing

**Third-party package** (`thirdparty/toolcall-annotation/tests/`)
- `count_turn_toolcall_errors`: clean turn → 0; one malformed-args call → 1;
  multi-call turn with 2 bad → 2; turn with no tool call → 0.

**slime core** (`tests/`, CPU, `pytest.main([__file__])` pattern)
- Inject a fake `turn_scorer` into `TrajectoryManager`; feed a 3-turn session;
  assert `Sample.metadata["toolcall_turn_shaping"]`:
  - length == `response_length`;
  - nonzero entries fall only within errored turns' spans AND inside
    `loss_mask == 1` regions;
  - all-zeros when `turn_scorer is None` (default off);
  - a synthetic 100-turn all-errored session sums to exactly `-budget`
    (budget cap).
- REALIGN-drift case: vector stays aligned after a span overwrite.

**Example** (`examples/coding_agent_rl/`, CPU)
- `make_turn_scorer` / budget math: proportional scaling correctness.
- `compute_advantage`: GRPO returns + shaping add; falls back to plain GRPO when
  the key is absent or all-zero.

**Alignment invariant** (guards the five-touch-point plumbing)
- Construct a small `rollout_data`; assert the shaping vector survives DP-split
  + CP-slice with the SAME per-sample shape as `loss_masks` after
  `_get_rollout_data`.

## Risks

- **Plumbing drift:** any per-token field not in the DP-split allowlist is
  silently dropped. Mitigated by the alignment-invariant test.
- **Scale interaction with GRPO group-norm:** shaping is added AFTER outcome
  advantage; `budget` must be tuned relative to the group-normed reward scale.
  Default `beta=0` keeps it inert until deliberately enabled.
- **CP correctness:** shaping must be CP-sliced identically to `rollout_log_probs`;
  covered by the invariant test. The target run uses `CP_SIZE=8`, so this path
  is exercised.
```
