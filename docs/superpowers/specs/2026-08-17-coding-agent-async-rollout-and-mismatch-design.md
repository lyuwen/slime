# Coding-Agent RL: Async Rollout + Train/Infer Mismatch Measurement

Date: 2026-08-17
Status: Approved (design)
Area: `examples/coding_agent_rl`

## Goal

Move the SWE coding-agent RL recipe from synchronous colocated rollout to
**fully-async, non-colocated** rollout, and add **train/inference mismatch
measurement** (metrics only, no loss correction) so we can decide later whether
to enable importance-sampling correction.

Two orthogonal axes are in play, kept separate on purpose:

1. **Sync vs. async training loop** — chosen by driver: `train.py` (sync) vs
   `train_async.py` (pipelined async), plus the fully-async rollout worker via
   `--rollout-function-path`.
2. **Colocated vs. non-colocated placement** — chosen by `--colocate`.
   `train_async.py` hard-requires non-colocated (`train_async.py:11`).

## Non-goals (this pass)

- Enabling TIS/MIS loss correction (`--use-tis`, `--use-rollout-logprobs`).
- Modifying framework code in `slime/` — the async worker and mismatch helper
  already exist and are reused as-is.
- Evaluation during the async run — the fully-async entrypoint rejects eval
  (`slime/rollout/fully_async_rollout.py:254-255`).

## Baseline

`examples/coding_agent_rl/run_021_32b_a4b_scaleswe_openhands_8nodes.sh`:
64 GPUs (8×8), `--colocate`, `--rollout-num-gpus 64` (shared), launched via
`train.py`, custom generate `examples.coding_agent_rl.generate.generate`.
This synchronous launcher stays **unchanged** for rollback and comparison.

## Approach

Add a dedicated sibling launcher:
`examples/coding_agent_rl/run_021_32b_a4b_scaleswe_openhands_8nodes_fully_async.sh`.

### GPU layout (96 total, non-colocated)

Non-colocated splits the pool into disjoint sets
(`slime/ray/placement_group.py:117`: total = actor + rollout, rollout offset =
actor GPUs).

| Role    | GPUs | Parallelism                          |
|---------|------|--------------------------------------|
| Actor   | 64   | TP=2, CP=8, EP=8, ETP=1 (unchanged)  |
| Rollout | 32   | TP=2 → 16 engines                    |

Defaults, all env-overridable:
- `ACTOR_NUM_NODES=8`, `ACTOR_NUM_GPUS_PER_NODE=8` → 64 actor GPUs.
- `ROLLOUT_NUM_GPUS=32`, `ROLLOUT_TP_SIZE=2` → 16 rollout engines.
- `SGLANG_SERVER_CONCURRENCY=4` → worker in-flight pool =
  `sglang_server_concurrency * num_engines` = 64 concurrent trajectories
  (`slime/rollout/fully_async_rollout.py:59`).

### Launcher changes vs. the sync baseline

1. Driver: `train.py` → `train_async.py`.
2. Remove `--colocate` from `MISC_ARGS`. This also drops the forced
   `offload_train`/`offload_rollout` (`slime/utils/arguments.py:1884-1904`).
3. `--rollout-num-gpus 64` → `--rollout-num-gpus ${ROLLOUT_NUM_GPUS}` (32),
   distinct from the actor set.
4. Add to `ROLLOUT_ARGS`:
   `--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async`.
5. Keep `--custom-generate-function-path examples.coding_agent_rl.generate.generate`
   (works unchanged under fully-async; both paths call `generate_and_rm_group`).
6. Keep `--rollout-global-dataset` (default on; asserted at
   `fully_async_rollout.py:195`).
7. No eval args.
8. Startup validation in the script: assert
   `ROLLOUT_NUM_GPUS % ROLLOUT_TP_SIZE == 0` and that
   `actor GPUs + ROLLOUT_NUM_GPUS` equals the expected cluster allocation.

### Mismatch measurement (metrics only)

Reuse `examples/train_infer_mismatch_helper`. Add to the launcher:

```
--get-mismatch-metrics
--custom-config-path examples/train_infer_mismatch_helper/mis.yaml
--custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
```

- Omit `--use-tis`: the helper computes metrics and returns unchanged weights
  (`mis.py:207-209`), so the training objective is untouched.
- Omit `--use-rollout-logprobs`: retain Megatron log-prob recomputation so the
  measured discrepancy is real.
- Validation requires `custom_tis_function_path` when `get_mismatch_metrics` is
  set (`slime/utils/arguments.py:1801-1811`) — satisfied.
- CP=8 is active, so the CP-aware wrapper `compute_mis_weights_with_cp` is
  required (it all-gathers/​re-slices across CP ranks).

Metrics land in TensorBoard + `run.log`: `train_rollout_logprob_abs_diff`,
`mismatch`-family KL/K3-KL, perplexity diff, and the `mis_`-prefixed
distribution/veto fractions.

## Validation plan

- `bash -n` syntax check on the new launcher.
- Run the mismatch-helper unit tests and coding-agent unit tests that don't need
  GPUs (`examples/coding_agent_rl/test_*.py`, helper tests if present).
- Argument-level check: confirm the launcher selects `train_async.py`, omits
  `--colocate`, defaults to 64/32 GPUs, sets the fully-async function path, sets
  the three metrics-only mismatch flags, and passes no eval flag. Prefer a
  lightweight repo-convention test if one fits (see `add-tests-and-ci`).
- No GPU allocation required for the above.

## Follow-up decision gate

After a measurement run, compare against the sync baseline: mismatch metric
distributions, catastrophic token/sequence fractions, reward, policy-loss
stability, effective sample utilization, and whether the 32 rollout GPUs stay
saturated. Then choose:
- small/stable mismatch → try built-in TIS (`--use-tis` with `--tis-clip*`);
- material or sequence-level-outlier mismatch → helper's YAML-driven MIS/TIS in
  a separate opt-in launcher.

## Key references

- `train_async.py:11,32-70` — async loop, colocate assertion, weight-update gate.
- `slime/rollout/fully_async_rollout.py:59,195,251-256` — worker concurrency,
  global-dataset assertion, entrypoint, eval rejection.
- `slime/ray/placement_group.py:100-137` — colocated vs non-colocated split.
- `slime/utils/arguments.py:1801-1811,1884-1904` — mismatch validation, colocate
  normalization.
- `examples/train_infer_mismatch_helper/{README.md,mis.py,mis.yaml,run-qwen3-4b-mis.sh}`.
- `slime/backends/megatron_utils/loss.py:987-1015` — TIS/mismatch dispatch.
