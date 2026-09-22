# Per-turn Tool-call Reward Shaping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional additive per-turn tool-call-correctness reward shaping to the coding-agent RL rollout, scored online in-process from the external `toolcall_annotation` package.

**Architecture:** The `TrajectoryManager` accepts an injected per-turn scorer callback plus `beta`/`budget` scalars; at linearization it computes `-beta * error_count` per turn, applies a per-trajectory budget cap, and writes a dense per-token vector to `Sample.metadata["toolcall_turn_shaping"]` aligned to `loss_mask`. That vector is promoted to a first-class per-token training field, partitioned across DP ranks, CP-sliced identically to `rollout_log_probs`, and consumed by a custom advantage function that adds it onto GRPO returns. slime core never imports the annotator; the coding_agent_rl example supplies the annotator-backed scorer and the advantage function.

**Tech Stack:** Python, PyTorch, Megatron (training), Ray (orchestration), pytest (plain-script CPU tests via `pytest.main([__file__])`). External dependency: `toolcall-annotation` (top-level module `toolcall_annotation`), pip-installed, used unmodified.

## Global Constraints

- Feature ships OFF by default: `SWE_TOOLCALL_SHAPING_BETA` default `0.0` → all-zero shaping vector, no-op. Existing runs must be byte-for-byte unaffected when disabled.
- slime core (`slime/`) MUST NOT import `toolcall_annotation`. The annotator coupling lives only in `examples/coding_agent_rl/`.
- The annotator is used UNMODIFIED (external pip package). No edits to `toolcall_annotation`.
- `TrajectoryManager(turn_scorer=None, ...)` default preserves current behavior exactly; `get_trajectory` / `finish_session` signatures MUST NOT change.
- Config env vars: `SWE_TOOLCALL_SHAPING_BETA` (float, default `0.0`), `SWE_TOOLCALL_SHAPING_BUDGET` (float, default `1.0`). Auto-forwarded to Ray workers by the existing `SWE_`-prefix loop in the run script (no runtime-env edit needed).
- The metadata key and train_data key name is exactly `"toolcall_turn_shaping"` everywhere.
- Per-token shaping vector length MUST equal `Sample.response_length` and align to `Sample.loss_mask`.
- Lint before every commit: `pre-commit run --all-files` (black/isort/ruff, line length 119).
- CPU tests are plain scripts ending in `if __name__ == "__main__": pytest.main([__file__])`; run either directly or via pytest.

---

### Task 1: Per-turn scoring in the trajectory manager

**Files:**
- Modify: `slime/agent/trajectory.py` (`_SampleBuilder`, `TrajectoryManager.__init__`, `_chain_to_samples`, `get_trajectory`; add helper `_apply_turn_shaping`)
- Test: `tests/test_agent/test_turn_shaping.py` (create)

**Interfaces:**
- Consumes: existing `TurnRecord`, `Sample`, `_SampleBuilder`, `DriftKind`.
- Produces:
  - `TrajectoryManager.__init__(self, *, fork_threshold_tokens=None, turn_scorer=None, shaping_beta=0.0, shaping_budget=1.0)` where `turn_scorer: Callable[[MessageNode], int] | None` returns the errored-tool-call count for one generated assistant turn node.
  - After `get_trajectory`, every returned `Sample` carries `metadata["toolcall_turn_shaping"]: list[float]` of length `response_length` when `turn_scorer` is set and `shaping_beta != 0`; otherwise the key is absent.
  - `_SampleBuilder` exposes `self.turn_spans: list[tuple[int, int, MessageNode, bool]]` = `(response_start, response_len, source_node, trained)` per appended turn, in token order.

**Design notes for the implementer:**
- `_SampleBuilder.append_turn` already sets `self.last_response_start_idx = len(self.tokens)` right before appending `turn.output_ids`. Record the span there. The builder currently has no reference to the originating `MessageNode`; thread it through. `_split_chain_into_builders` calls `append_turn(asst_node.turn, ...)` — change to also pass `asst_node`.
- `to_sample` strips the leading first-turn prompt via `start = self.leading_prompt_len`. The shaping vector must be built over the full `self.tokens` length, then sliced `[start:]` exactly like `loss_mask`, so it aligns to the response region and undergoes the same `max_sample_tokens` truncation.
- Budget cap operates across ALL samples of the session: compute raw penalties for every sample first, sum `abs(total)`, derive one `scale = budget / total_abs` if `total_abs > budget` else `1.0`, apply uniformly, then write vectors.
- Untrained turns (`trained=False`, shared-prefix re-emission) contribute 0 penalty and their span stays 0.

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent/test_turn_shaping.py`:

```python
"""Per-turn tool-call reward shaping in TrajectoryManager.

Drives TrajectoryManager with an injected fake turn_scorer (no annotator
dependency) and asserts the dense per-token shaping vector written to
Sample.metadata["toolcall_turn_shaping"] is correctly valued, aligned to
loss_mask, and bounded by the per-trajectory budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from slime.agent.adapters.common import TurnRecord  # noqa: E402
from slime.agent.trajectory import TrajectoryManager  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

# Token bands: system=1000, user=2000, assistant=9000, tool=3000
SYS = [1000, 1001, 1099]
USR = [2000, 2001, 2099]


def _asst_msg(label):
    return {"role": "assistant", "content": label}


def _user_msg(ids):
    return {"role": "user", "content": str(ids)}


def _turn(prompt_ids, response_ids):
    return TurnRecord(
        prompt_ids=list(prompt_ids),
        output_ids=list(response_ids),
        finish_reason="stop",
        output_log_probs=[0.0] * len(response_ids),
    )


def _two_turn_session(mgr, sid, *, r1, r2):
    """Two clean, prefix-extending turns. Returns (prompt2_len,)."""
    p1 = SYS + USR
    resp1 = [9001, 9002, 9003]  # 3 response tokens
    mgr.record_turn(
        sid, turn=_turn(p1, resp1),
        prompt_messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        response_message=_asst_msg(r1),
    )
    p2 = p1 + resp1 + [2100, 2101]  # prior + tool/user follow-up
    resp2 = [9004, 9005]  # 2 response tokens
    mgr.record_turn(
        sid, turn=_turn(p2, resp2),
        prompt_messages=[
            {"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            _asst_msg(r1), {"role": "user", "content": "f"},
        ],
        response_message=_asst_msg(r2),
    )


def test_shaping_absent_when_scorer_none():
    """Default (no scorer) leaves metadata free of the shaping key."""
    mgr = TrajectoryManager()
    _two_turn_session(mgr, "sid", r1="a1", r2="a2")
    samples = mgr.get_trajectory("sid", base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 1
    assert "toolcall_turn_shaping" not in (samples[0].metadata or {})


def test_shaping_penalizes_errored_turn_only():
    """Scorer flags turn 2 (1 error); its response tokens get -beta, turn 1 stays 0."""
    # scorer: 0 errors for first generated turn, 1 error for the second
    seen = []

    def scorer(node):
        seen.append(node)
        return 0 if len(seen) == 1 else 1

    mgr = TrajectoryManager(turn_scorer=scorer, shaping_beta=0.5, shaping_budget=100.0)
    _two_turn_session(mgr, "sid", r1="a1", r2="a2")
    samples = mgr.get_trajectory("sid", base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 1
    s = samples[0]
    vec = s.metadata["toolcall_turn_shaping"]
    assert len(vec) == s.response_length
    # response region = resp1(3) + prompt2 tail + resp2(2). Only resp2 tokens are -0.5.
    assert vec[:3] == [0.0, 0.0, 0.0]          # turn 1 response, clean
    assert vec[-2:] == [-0.5, -0.5]            # turn 2 response, 1 error * -0.5
    # non-response prompt-tail tokens between the two responses are 0
    assert set(vec[3:-2]) <= {0.0}


def test_budget_cap_scales_total():
    """Total |shaping| is capped at budget; proportions preserved."""
    def scorer(node):
        return 1  # every turn errs once

    mgr = TrajectoryManager(turn_scorer=scorer, shaping_beta=1.0, shaping_budget=1.0)
    _two_turn_session(mgr, "sid", r1="a1", r2="a2")
    samples = mgr.get_trajectory("sid", base_sample=Sample(index=0, prompt=""), reward=1.0)
    vec = samples[0].metadata["toolcall_turn_shaping"]
    total = sum(vec)
    # raw total = -(3 + 2) = -5 over 5 response tokens; capped to -1.0
    assert abs(total + 1.0) < 1e-6
    # all nonzero entries equal (uniform -beta before scaling), scaled uniformly
    nonzero = [v for v in vec if v != 0.0]
    assert len(nonzero) == 5
    assert all(abs(v - nonzero[0]) < 1e-9 for v in nonzero)


if __name__ == "__main__":
    pytest.main([__file__])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_agent/test_turn_shaping.py`
Expected: FAIL — `TrajectoryManager.__init__` got an unexpected keyword argument `turn_scorer` (and the metadata key is never written).

- [ ] **Step 3: Implement in `slime/agent/trajectory.py`**

3a. `_SampleBuilder.__init__` — add span tracking. After the existing `self.cached_tokens = 0` line:

```python
        # Per-turn token spans for reward shaping: (response_start, response_len,
        # source_node, trained), in token order. Filled by append_turn.
        self.turn_spans: list[tuple[int, int, "MessageNode", bool]] = []
```

3b. `_SampleBuilder.append_turn` — accept the source node and record the span. Change the signature and the response-append block:

```python
    def append_turn(
        self, turn: TurnRecord, kind: DriftKind, *, source_node: "MessageNode", trained: bool = True
    ) -> None:
```

Then, right after `self.last_response_start_idx = len(self.tokens)` and before `self._append_tokens(turn.output_ids, ...)`, record the span:

```python
        self.last_response_start_idx = len(self.tokens)
        self.turn_spans.append((self.last_response_start_idx, len(turn.output_ids), source_node, trained))
        self._append_tokens(
            turn.output_ids, loss_mask=int(trained), logprobs=turn.output_log_probs if trained else None
        )
```

3c. `_split_chain_into_builders` — pass `source_node`. Update both `append_turn` calls:

```python
            if not builders or (kind := builders[-1].classify_token_drift(asst_node.turn)) is DriftKind.FORK:
                builders.append(_SampleBuilder(self._fork_threshold))
                builders[-1].append_turn(asst_node.turn, DriftKind.CLEAN, source_node=asst_node, trained=trained)
            else:
                builders[-1].append_turn(asst_node.turn, kind, source_node=asst_node, trained=trained)
```

3d. `TrajectoryManager.__init__` — accept scorer + scalars:

```python
    def __init__(
        self,
        *,
        fork_threshold_tokens: int | None = None,
        turn_scorer=None,
        shaping_beta: float = 0.0,
        shaping_budget: float = 1.0,
    ) -> None:
        self._fork_threshold: int = 1024 if fork_threshold_tokens is None else fork_threshold_tokens
        self._trees: dict[str, MessageNode] = {}
        self._turn_count: dict[str, int] = {}
        self._turn_scorer = turn_scorer
        self._shaping_beta = float(shaping_beta)
        self._shaping_budget = float(shaping_budget)
```

3e. `_chain_to_samples` — return the builders alongside samples so `get_trajectory` can read `turn_spans`. Change it to attach the builder to each sample via a parallel list. Simplest: have `_chain_to_samples` also stash `(sample, builder)` pairs. Modify the loop end:

```python
        samples: list[Sample] = []
        pairs: list[tuple[Sample, "_SampleBuilder"]] = []
        for builder in self._split_chain_into_builders(chain):
            if not builder.has_trained_response():
                continue
            sample = builder.to_sample(base_sample, md, max_sample_tokens)
            sample.prefix_cache_info.cached_tokens = builder.cached_tokens
            sample.prefix_cache_info.total_prompt_tokens = builder.prompt_tokens
            sample.prefix_cache_info.completion_tokens = builder.completion_tokens
            samples.append(sample)
            pairs.append((sample, builder))
        return samples, pairs
```

Update its only caller in `get_trajectory` (see 3f) to unpack the pair list.

3f. `get_trajectory` — collect pairs, apply shaping before the reward broadcast. Replace the body's sample-collection loop and add the shaping call:

```python
        samples: list[Sample] = []
        pairs: list[tuple[Sample, _SampleBuilder]] = []
        for routing_leaf in root.leaves():
            if routing_leaf.is_root:
                continue
            chain = routing_leaf.path_from_root()
            chain_samples, chain_pairs = self._chain_to_samples(
                chain, base_sample=base_sample, extra_metadata=extra_metadata, max_sample_tokens=max_sample_tokens
            )
            samples.extend(chain_samples)
            pairs.extend(chain_pairs)

        self._apply_turn_shaping(pairs)

        for s in samples:
            s.reward = reward

        self._trees.pop(sid, None)
        self._turn_count.pop(sid, None)
        return samples
```

3g. Add the helper method to `TrajectoryManager` (after `get_trajectory`):

```python
    def _apply_turn_shaping(self, pairs: list[tuple[Sample, "_SampleBuilder"]]) -> None:
        """Write a dense per-token shaping vector to each sample's metadata.

        For every trained turn span, fill -beta * error_count over that span,
        then cap the summed |shaping| across ALL samples of the session to
        shaping_budget (proportional scale-down; never scales up). No-op when no
        scorer is configured or beta == 0.
        """
        if self._turn_scorer is None or self._shaping_beta == 0.0:
            return

        # Raw penalties per sample, in full-token space (pre first-turn strip).
        raw_full: list[list[float]] = []
        for _sample, builder in pairs:
            vec = [0.0] * len(builder.tokens)
            for start, length, node, trained in builder.turn_spans:
                if not trained or length == 0:
                    continue
                errors = int(self._turn_scorer(node))
                if errors:
                    penalty = -self._shaping_beta * errors
                    for i in range(start, start + length):
                        vec[i] = penalty
            raw_full.append(vec)

        total_abs = sum(abs(v) for vec in raw_full for v in vec)
        scale = (self._shaping_budget / total_abs) if total_abs > self._shaping_budget else 1.0

        for (sample, builder), vec in zip(pairs, raw_full, strict=True):
            start = builder.leading_prompt_len
            sliced = [v * scale for v in vec[start : start + sample.response_length]]
            # Guard: to_sample may have truncated; pad/trim to response_length.
            if len(sliced) < sample.response_length:
                sliced = sliced + [0.0] * (sample.response_length - len(sliced))
            sample.metadata = {**(sample.metadata or {}), "toolcall_turn_shaping": sliced}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_agent/test_turn_shaping.py`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the existing trajectory tests to confirm no regression**

Run: `python tests/test_agent/test_trajectory_manager_branching.py`
Expected: PASS (default `turn_scorer=None` path unchanged).

- [ ] **Step 6: Lint and commit**

```bash
pre-commit run --files slime/agent/trajectory.py tests/test_agent/test_turn_shaping.py
git add slime/agent/trajectory.py tests/test_agent/test_turn_shaping.py
git commit -m "feat(agent): per-turn tool-call reward shaping in TrajectoryManager"
```

---

### Task 2: Thread the scorer through the adapter

**Files:**
- Modify: `slime/agent/adapters/common.py` (`BaseAdapter.__init__`)
- Test: `tests/test_agent/test_turn_shaping.py` (add one case)

**Interfaces:**
- Consumes: `TrajectoryManager(turn_scorer=..., shaping_beta=..., shaping_budget=...)` from Task 1.
- Produces: `BaseAdapter.__init__(..., turn_scorer=None, shaping_beta=0.0, shaping_budget=1.0)` forwarding into the manager it constructs. Subclasses `AnthropicAdapter` / `OpenAIAdapter` inherit unchanged (they call `super().__init__(**kwargs)`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent/test_turn_shaping.py` (above the `__main__` block):

```python
def test_adapter_forwards_scorer_to_manager():
    """BaseAdapter passes turn_scorer + scalars into its TrajectoryManager."""
    from slime.agent.adapters.common import BaseAdapter

    def scorer(node):
        return 0

    class _Tok:
        def apply_chat_template(self, *a, **k):
            return {"input_ids": [1]}

        def decode(self, *a, **k):
            return ""

    ad = BaseAdapter(
        tokenizer=_Tok(),
        sglang_url="http://x",
        turn_scorer=scorer,
        shaping_beta=0.25,
        shaping_budget=2.0,
    )
    assert ad.manager._turn_scorer is scorer
    assert ad.manager._shaping_beta == 0.25
    assert ad.manager._shaping_budget == 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_agent/test_turn_shaping.py`
Expected: FAIL — `BaseAdapter.__init__` got an unexpected keyword argument `turn_scorer`.

- [ ] **Step 3: Implement in `slime/agent/adapters/common.py`**

Add the three params to `BaseAdapter.__init__` (after `debug_callback`):

```python
        debug_callback: Callable[..., None] | None = None,
        turn_scorer: Callable[..., int] | None = None,
        shaping_beta: float = 0.0,
        shaping_budget: float = 1.0,
    ) -> None:
```

And extend the manager-construction block (currently builds `mgr_kwargs` then `TrajectoryManager(**mgr_kwargs)`):

```python
        mgr_kwargs: dict[str, Any] = {}
        if fork_threshold_tokens is not None:
            mgr_kwargs["fork_threshold_tokens"] = fork_threshold_tokens
        mgr_kwargs["turn_scorer"] = turn_scorer
        mgr_kwargs["shaping_beta"] = shaping_beta
        mgr_kwargs["shaping_budget"] = shaping_budget
        self.manager = TrajectoryManager(**mgr_kwargs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_agent/test_turn_shaping.py`
Expected: PASS (4 tests).

- [ ] **Step 5: Run adapter tests for regression**

Run: `python tests/test_agent/test_adapters.py`
Expected: PASS.

- [ ] **Step 6: Lint and commit**

```bash
pre-commit run --files slime/agent/adapters/common.py tests/test_agent/test_turn_shaping.py
git add slime/agent/adapters/common.py tests/test_agent/test_turn_shaping.py
git commit -m "feat(agent): forward turn_scorer + shaping scalars through BaseAdapter"
```

---

### Task 3: Promote shaping vector into train_data and DP-split allowlist

**Files:**
- Modify: `slime/ray/rollout.py` (`_convert_samples_to_train_data`, `_split_train_data_by_dp`)
- Test: `tests/test_toolcall_shaping_pipeline.py` (create)

**Interfaces:**
- Consumes: `Sample.metadata["toolcall_turn_shaping"]: list[float]` (length == `response_length`) from Task 1.
- Produces: `train_data["toolcall_turn_shaping"]: list[list[float]]` (one per sample) when any sample carries the key; and the key is included in the per-token DP-split allowlist so `rollout_data["toolcall_turn_shaping"]` is partitioned per sample on each DP rank.

- [ ] **Step 1: Write the failing test**

Create `tests/test_toolcall_shaping_pipeline.py`:

```python
"""train_data plumbing for the per-turn tool-call shaping vector.

Unit-tests _convert_samples_to_train_data's promotion of
Sample.metadata["toolcall_turn_shaping"] into a first-class per-token
train_data key, and its presence in the DP-split allowlist.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from slime.utils.types import Sample  # noqa: E402


def _sample(idx, resp_len, shaping=None):
    md = {}
    if shaping is not None:
        md["toolcall_turn_shaping"] = shaping
    return Sample(
        index=idx,
        rollout_id=idx,
        prompt="",
        tokens=[0] * (2 + resp_len),
        response_length=resp_len,
        loss_mask=[1] * resp_len,
        rollout_log_probs=[0.0] * resp_len,
        reward=1.0,
        status=Sample.Status.COMPLETED,
        metadata=md,
    )


def test_convert_promotes_shaping_key():
    """When samples carry the metadata key, it becomes a per-sample train_data list."""
    samples = [_sample(0, 3, [0.0, -0.5, -0.5]), _sample(1, 2, [0.0, 0.0])]
    td = _convert(samples)
    assert td["toolcall_turn_shaping"] == [[0.0, -0.5, -0.5], [0.0, 0.0]]


def test_convert_omits_key_when_absent():
    """No sample carries the key → train_data has no shaping key (feature off)."""
    samples = [_sample(0, 3), _sample(1, 2)]
    td = _convert(samples)
    assert "toolcall_turn_shaping" not in td


if __name__ == "__main__":
    pytest.main([__file__])
```

Note to implementer: `_convert_samples_to_train_data` is a method on `RolloutManager` with `self`/`args` dependencies. Rather than construct a full manager, define the `_convert(samples)` helper (below) that builds a minimal stub manager and args namespace, and place it ABOVE the test functions in the file so they can call it. Concretely:

```python
def _convert(samples):
    """Invoke the real _convert_samples_to_train_data with a minimal stub."""
    import types

    from slime.ray.rollout import RolloutManager

    mgr = object.__new__(RolloutManager)  # bypass __init__
    mgr.custom_convert_samples_to_train_data_func = None
    mgr.custom_reward_post_process_func = None
    mgr.args = types.SimpleNamespace(
        reward_key=None,
        advantage_estimator="grpo",
        rewards_normalization=False,
        grpo_std_normalization=False,
        n_samples_per_prompt=1,
        rollout_batch_size=len(samples),
        rollout_top_p=1.0,
        use_rollout_routing_replay=False,
    )
    return RolloutManager._convert_samples_to_train_data(mgr, samples)
```

If `_convert_samples_to_train_data` reads args attributes not listed above, add them to the `SimpleNamespace` with values that take the default GRPO path.

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_toolcall_shaping_pipeline.py`
Expected: FAIL — `KeyError: 'toolcall_turn_shaping'` (key never added to train_data).

- [ ] **Step 3: Implement in `slime/ray/rollout.py`**

3a. In `_convert_samples_to_train_data`, after the `round_number` block (around line 786), add:

```python
        # Per-token tool-call reward shaping vector (present only when the
        # coding-agent turn scorer ran; feature off → key absent).
        if any(sample.metadata and "toolcall_turn_shaping" in sample.metadata for sample in samples):
            train_data["toolcall_turn_shaping"] = [
                (sample.metadata or {}).get("toolcall_turn_shaping", [0.0] * sample.response_length)
                for sample in samples
            ]
```

3b. In `_split_train_data_by_dp`, add `"toolcall_turn_shaping"` to the per-token allowlist list (the `for key in [...]` that includes `"rollout_log_probs"`), e.g. right after `"rollout_log_probs",`:

```python
                "rollout_log_probs",
                "toolcall_turn_shaping",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_toolcall_shaping_pipeline.py`
Expected: PASS (2 tests).

- [ ] **Step 5: Lint and commit**

```bash
pre-commit run --files slime/ray/rollout.py tests/test_toolcall_shaping_pipeline.py
git add slime/ray/rollout.py tests/test_toolcall_shaping_pipeline.py
git commit -m "feat(rollout): promote toolcall_turn_shaping into train_data + DP split"
```

---

### Task 4: CP-slice the shaping vector on the train side

**Files:**
- Modify: `slime/backends/megatron_utils/actor.py` (`_get_rollout_data`)
- Test: `tests/test_toolcall_shaping_pipeline.py` (add one case)

**Interfaces:**
- Consumes: `rollout_data["toolcall_turn_shaping"]: list[list[float]]` (per-sample, per-token) partitioned by Task 3.
- Produces: after `_get_rollout_data`, `rollout_data["toolcall_turn_shaping"]` is a list of `torch.Tensor` on the current CUDA device, CP-sliced identically to `rollout_log_probs` (same per-sample shape as the CP-sliced `loss_masks`).

**Design note:** the existing loop at `actor.py:273` iterates `["rollout_log_probs", "teacher_log_probs"]`, applies `slice_log_prob_with_cp(log_prob, total_length, response_length)`, and moves to device as float32. Add `"toolcall_turn_shaping"` to that list — it has identical per-token, response-length semantics. `slice_log_prob_with_cp` asserts `len(log_prob) == response_length`, which holds by Task 1's invariant.

- [ ] **Step 1: Write the failing test (CPU, cp_size==1 path)**

Add to `tests/test_toolcall_shaping_pipeline.py`:

```python
def test_cp_slice_included_for_shaping_key():
    """The _get_rollout_data field loop must list toolcall_turn_shaping so it is
    CP-sliced and tensorized alongside rollout_log_probs.

    We assert on source to avoid standing up Megatron/CUDA: the key must appear
    in the same slice loop as rollout_log_probs.
    """
    import inspect

    from slime.backends.megatron_utils import actor

    src = inspect.getsource(actor.MegatronTrainRayActor._get_rollout_data)
    # both keys handled by the same CP-slice loop
    assert "toolcall_turn_shaping" in src
    assert "rollout_log_probs" in src
    idx_loop = src.index("rollout_log_probs")
    idx_shape = src.index("toolcall_turn_shaping")
    # shaping key appears within ~200 chars of the rollout_log_probs loop header
    assert abs(idx_shape - idx_loop) < 400, "shaping key not in the same slice loop"


if __name__ == "__main__":
    pytest.main([__file__])
```

(Keep only one `__main__` block at the end of the file.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_toolcall_shaping_pipeline.py`
Expected: FAIL — `toolcall_turn_shaping` not found in `_get_rollout_data` source.

- [ ] **Step 3: Implement in `slime/backends/megatron_utils/actor.py`**

Change the field list in `_get_rollout_data` from:

```python
        for key in ["rollout_log_probs", "teacher_log_probs"]:
```

to:

```python
        for key in ["rollout_log_probs", "teacher_log_probs", "toolcall_turn_shaping"]:
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_toolcall_shaping_pipeline.py`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint and commit**

```bash
pre-commit run --files slime/backends/megatron_utils/actor.py tests/test_toolcall_shaping_pipeline.py
git add slime/backends/megatron_utils/actor.py tests/test_toolcall_shaping_pipeline.py
git commit -m "feat(actor): CP-slice toolcall_turn_shaping like rollout_log_probs"
```

---

### Task 5: Annotator glue + turn scorer in the example

**Files:**
- Create: `examples/coding_agent_rl/turn_shaping.py`
- Test: `examples/coding_agent_rl/test_turn_shaping.py` (create)

**Interfaces:**
- Consumes: external `toolcall_annotation.annotators.toolcall_correctness_impl` (`parse_arguments`, `find_tool_response`, `check_task_tracker`, `check_finish`, `check_think`, `check_str_replace_editor`, `check_execute_bash`, `find_previous_bash_response`) — imported lazily. And `MessageNode` from `slime.agent.trajectory` (the scorer receives a node).
- Produces:
  - `count_turn_toolcall_errors(assistant_message: dict, tool_response: dict | None) -> int`
  - `make_turn_scorer() -> Callable[[MessageNode], int]`
  - `compute_advantage(args, rollout_data) -> None` (custom advantage hook; sets `rollout_data["advantages"]` and `["returns"]`).

**Design notes:**
- A `MessageNode` (from `slime/agent/trajectory.py`) has `.message` (the assistant dict with `tool_calls`) and `.children` (the tool-response nodes are among the assistant node's descendants — but in the routing tree the tool response is the assistant node's *parent-side* sibling? No: the tool response follows the assistant in the NEXT prompt). For scoring, the tool response for turn T appears as a routing-only `tool` node mounted below the assistant node in a later turn. The scorer must locate it. Simplest robust approach: walk `node.children` for a `tool` child; if absent (no follow-up turn recorded yet, e.g. last turn), treat as no response (`None`). The annotator's `check_*` already tolerates `tool_response=None`.
- The assistant `message` stores tool calls in slime's canonical shape `{"type": "function", "function": {"name", "arguments"}}` where `arguments` is a **dict**, not a JSON string (see `common.tool_call_dict`). The annotator's `parse_arguments` expects `function.arguments` to be a JSON **string**. Bridge this: if `arguments` is a dict, wrap it so the checks see a JSON string. Provide a small adapter inside `count_turn_toolcall_errors`.

- [ ] **Step 1: Write the failing test**

Create `examples/coding_agent_rl/test_turn_shaping.py`:

```python
"""Turn scorer + custom advantage for coding-agent tool-call shaping."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _asst(tool_name, arguments):
    return {
        "role": "assistant",
        "tool_calls": [{"type": "function", "function": {"name": tool_name, "arguments": arguments}}],
    }


def test_count_errors_clean_finish():
    """A well-formed finish call with a message → 0 errors."""
    from examples.coding_agent_rl.turn_shaping import count_turn_toolcall_errors

    msg = _asst("finish", {"message": "done"})
    assert count_turn_toolcall_errors(msg, tool_response=None) == 0


def test_count_errors_malformed_finish():
    """finish with no message → 1 error (missing_message)."""
    from examples.coding_agent_rl.turn_shaping import count_turn_toolcall_errors

    msg = _asst("finish", {})
    assert count_turn_toolcall_errors(msg, tool_response=None) >= 1


def test_count_errors_no_tool_call():
    """Assistant turn with no tool call → 0."""
    from examples.coding_agent_rl.turn_shaping import count_turn_toolcall_errors

    assert count_turn_toolcall_errors({"role": "assistant", "content": "hi"}, None) == 0


def test_compute_advantage_adds_shaping():
    """compute_advantage adds the per-token shaping onto GRPO returns."""
    import torch

    from examples.coding_agent_rl.turn_shaping import compute_advantage

    class _Args:
        advantage_estimator = "grpo"
        kl_coef = 0.0
        use_rollout_logprobs = False
        custom_advantage_function_path = "x"

    kl = [torch.zeros(3), torch.zeros(2)]
    rollout_data = {
        "rewards": [1.0, -1.0],
        "kl": kl,
        "log_probs": [torch.zeros(3), torch.zeros(2)],
        "rollout_log_probs": [torch.zeros(3), torch.zeros(2)],
        "toolcall_turn_shaping": [torch.tensor([0.0, -0.5, -0.5]), torch.tensor([0.0, 0.0])],
        "values": None,
        "response_lengths": [3, 2],
        "loss_masks": [torch.ones(3), torch.ones(2)],
        "total_lengths": [5, 4],
    }
    compute_advantage(_Args(), rollout_data)
    adv = rollout_data["advantages"]
    # sample 0: reward 1.0 broadcast + shaping [0,-0.5,-0.5]
    assert torch.allclose(adv[0], torch.tensor([1.0, 0.5, 0.5]))
    # sample 1: reward -1.0 broadcast + shaping [0,0]
    assert torch.allclose(adv[1], torch.tensor([-1.0, -1.0]))


def test_compute_advantage_without_shaping_key():
    """Absent shaping key → plain GRPO returns."""
    import torch

    from examples.coding_agent_rl.turn_shaping import compute_advantage

    class _Args:
        advantage_estimator = "grpo"
        kl_coef = 0.0
        use_rollout_logprobs = False
        custom_advantage_function_path = "x"

    rollout_data = {
        "rewards": [1.0],
        "kl": [torch.zeros(3)],
        "log_probs": [torch.zeros(3)],
        "rollout_log_probs": [torch.zeros(3)],
        "values": None,
        "response_lengths": [3],
        "loss_masks": [torch.ones(3)],
        "total_lengths": [5],
    }
    compute_advantage(_Args(), rollout_data)
    assert torch.allclose(rollout_data["advantages"][0], torch.tensor([1.0, 1.0, 1.0]))


if __name__ == "__main__":
    pytest.main([__file__])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python examples/coding_agent_rl/test_turn_shaping.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'examples.coding_agent_rl.turn_shaping'`.

- [ ] **Step 3: Implement `examples/coding_agent_rl/turn_shaping.py`**

```python
"""Per-turn tool-call reward shaping for the coding-agent RL rollout.

Two halves, both living here so slime core never imports the annotator:

* Rollout side: ``make_turn_scorer`` returns a callback that scores one
  generated assistant turn (a ``slime.agent.trajectory.MessageNode``) by the
  number of tool-call errors the external ``toolcall_annotation`` package
  detects. ``TrajectoryManager`` applies ``-beta`` and the per-trajectory
  budget cap and writes the dense vector to ``Sample.metadata``.
* Train side: ``compute_advantage`` (wired via
  ``--custom-advantage-function-path``) adds the per-token shaping vector onto
  GRPO returns.

The annotator is imported lazily so runs with the feature off (beta == 0, the
default) never require it installed.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

_TOOL_CHECKS = None  # lazily populated dispatch table


def _load_checks():
    """Import the external annotator's pure check functions, once.

    Raises a clear error if the package is missing while the feature is enabled.
    """
    global _TOOL_CHECKS
    if _TOOL_CHECKS is not None:
        return _TOOL_CHECKS
    try:
        from toolcall_annotation.annotators import toolcall_correctness_impl as impl
    except ImportError as e:  # pragma: no cover - environment-specific
        raise ImportError(
            "toolcall-annotation is required for tool-call reward shaping "
            "(SWE_TOOLCALL_SHAPING_BETA != 0). Install it on every Ray worker: "
            "pip install toolcall-annotation"
        ) from e
    _TOOL_CHECKS = impl
    return _TOOL_CHECKS


def _as_wire_tool_call(tc: dict) -> dict:
    """Adapt slime's canonical tool-call (arguments as dict) to the annotator's
    wire shape (function.arguments as a JSON string)."""
    fn = tc.get("function", {}) or {}
    args = fn.get("arguments", {})
    if not isinstance(args, str):
        args = json.dumps(args)
    return {"id": tc.get("id"), "function": {"name": fn.get("name"), "arguments": args}}


def count_turn_toolcall_errors(assistant_message: dict, tool_response: dict | None) -> int:
    """Total detected tool-call errors across one assistant turn's tool calls.

    Dispatches each tool call to the matching annotator ``check_*`` and sums the
    number of error types returned. Unknown tools and turns without tool calls
    contribute 0.
    """
    if not assistant_message:
        return 0
    tool_calls = assistant_message.get("tool_calls") or []
    if not tool_calls:
        return 0

    impl = _load_checks()
    total = 0
    for tc in tool_calls:
        wire = _as_wire_tool_call(tc)
        name = wire["function"]["name"]
        # JSON validity applies to every tool call.
        _, args_valid = impl.parse_arguments(wire)
        if not args_valid:
            total += 1
            continue
        if name == "task_tracker":
            total += len(impl.check_task_tracker(wire, tool_response))
        elif name == "finish":
            total += len(impl.check_finish(wire, tool_response))
        elif name == "think":
            total += len(impl.check_think(wire, tool_response))
        elif name == "str_replace_editor":
            total += len(impl.check_str_replace_editor(wire, tool_response))
        elif name == "execute_bash":
            res = impl.check_execute_bash(wire, tool_response, None)
            total += len(res.get("errors", []))
        # unknown tools: no check, 0
    return total


def make_turn_scorer():
    """Return a callback scoring one generated assistant turn node.

    The callback takes a ``MessageNode`` and returns its errored-tool-call count.
    It locates the turn's tool response as a ``tool`` child node when present
    (the follow-up turn mounts it below the assistant); otherwise scores against
    ``None`` (the annotator tolerates a missing response).
    """

    def score(node) -> int:
        assistant_message = node.message or {}
        tool_response = None
        for child in getattr(node, "children", []) or []:
            if child.role == "tool" and child.message is not None:
                tool_response = child.message
                break
        return count_turn_toolcall_errors(assistant_message, tool_response)

    return score


def compute_advantage(args, rollout_data) -> None:
    """Custom advantage: GRPO returns plus the per-token tool-call shaping vector.

    Wired via ``--custom-advantage-function-path``. Called by
    ``compute_advantages_and_returns`` AFTER KL is computed; must set
    ``rollout_data['advantages']`` and ``rollout_data['returns']``.
    """
    import torch

    from slime.utils.ppo_utils import get_grpo_returns

    kl = rollout_data["kl"]
    rewards = torch.tensor(rollout_data["rewards"], dtype=torch.float32, device=kl[0].device)
    returns = get_grpo_returns(rewards, kl)

    shaping = rollout_data.get("toolcall_turn_shaping")
    if shaping is not None:
        for i in range(len(returns)):
            s = shaping[i]
            if not isinstance(s, torch.Tensor):
                s = torch.tensor(s, dtype=torch.float32, device=returns[i].device)
            returns[i] = returns[i] + s.to(device=returns[i].device, dtype=returns[i].dtype)

    rollout_data["returns"] = returns
    rollout_data["advantages"] = [r for r in returns]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python examples/coding_agent_rl/test_turn_shaping.py`
Expected: PASS (5 tests). Requires `toolcall_annotation` importable (it is, per environment check).

- [ ] **Step 5: Lint and commit**

```bash
pre-commit run --files examples/coding_agent_rl/turn_shaping.py examples/coding_agent_rl/test_turn_shaping.py
git add examples/coding_agent_rl/turn_shaping.py examples/coding_agent_rl/test_turn_shaping.py
git commit -m "feat(coding_agent_rl): turn scorer + custom advantage for tool-call shaping"
```

---

### Task 6: Wire config + scorer into generate.py and the run script

**Files:**
- Modify: `examples/coding_agent_rl/generate.py` (read env into `CONFIG`, build scorer, pass to adapter constructor)
- Modify: `examples/coding_agent_rl/run_021_32b_a4b_scaleswe_openhands_8nodes.sh` (export env vars + add `--custom-advantage-function-path`)
- Test: `examples/coding_agent_rl/test_turn_shaping.py` (add a config-parsing case)

**Interfaces:**
- Consumes: `make_turn_scorer` from Task 5; `BaseAdapter(..., turn_scorer=, shaping_beta=, shaping_budget=)` from Task 2.
- Produces: adapter instances constructed with the scorer when `SWE_TOOLCALL_SHAPING_BETA != 0`.

**Design note:** Find where `generate.py` constructs `ADAPTER_CLS(...)` and where `CONFIG` is built (the `@dataclass` config with `agent_time_budget_sec` etc., populated around line 86 from env). Add two fields and read them the same way. Build the scorer once (module level or in CONFIG) and pass it into the adapter constructor call. When beta == 0, pass `turn_scorer=None` so the annotator is never imported.

- [ ] **Step 1: Write the failing test**

Add to `examples/coding_agent_rl/test_turn_shaping.py`:

```python
def test_shaping_config_from_env(monkeypatch):
    """CONFIG reads beta/budget from SWE_ env vars; scorer built only when beta!=0."""
    from examples.coding_agent_rl.turn_shaping import make_turn_scorer

    # beta == 0 → no scorer
    monkeypatch.setenv("SWE_TOOLCALL_SHAPING_BETA", "0.0")
    from examples.coding_agent_rl.generate import resolve_shaping_config

    beta, budget, scorer = resolve_shaping_config()
    assert beta == 0.0
    assert scorer is None

    # beta != 0 → scorer built
    monkeypatch.setenv("SWE_TOOLCALL_SHAPING_BETA", "0.3")
    monkeypatch.setenv("SWE_TOOLCALL_SHAPING_BUDGET", "2.0")
    beta, budget, scorer = resolve_shaping_config()
    assert beta == 0.3
    assert budget == 2.0
    assert callable(scorer)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python examples/coding_agent_rl/test_turn_shaping.py`
Expected: FAIL — `ImportError: cannot import name 'resolve_shaping_config'`.

- [ ] **Step 3: Implement in `examples/coding_agent_rl/generate.py`**

3a. Add the import near the other local imports (`from . import swe`):

```python
from .turn_shaping import make_turn_scorer
```

3b. Add a module-level helper (near the CONFIG construction):

```python
def resolve_shaping_config():
    """Read tool-call shaping knobs from env.

    Returns (beta, budget, scorer). scorer is None when disabled (beta == 0),
    so the annotator is never imported for runs that don't use shaping.
    """
    beta = float(os.environ.get("SWE_TOOLCALL_SHAPING_BETA", "0.0"))
    budget = float(os.environ.get("SWE_TOOLCALL_SHAPING_BUDGET", "1.0"))
    scorer = make_turn_scorer() if beta != 0.0 else None
    return beta, budget, scorer
```

3c. At the adapter construction site (where `ADAPTER_CLS(...)` / the adapter is built), resolve config and pass it. Locate the adapter instantiation and add:

```python
        _beta, _budget, _scorer = resolve_shaping_config()
        adapter = ADAPTER_CLS(
            # ... existing kwargs (tokenizer=..., sglang_url=..., etc.) ...
            turn_scorer=_scorer,
            shaping_beta=_beta,
            shaping_budget=_budget,
        )
```

(Preserve all existing kwargs; only add the three. If the adapter is built via a helper, thread the three params through that helper.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python examples/coding_agent_rl/test_turn_shaping.py`
Expected: PASS (6 tests).

- [ ] **Step 5: Update the run script**

In `examples/coding_agent_rl/run_021_32b_a4b_scaleswe_openhands_8nodes.sh`, add to the SWE knobs block (after `SWE_BOOT_CONCURRENCY`, around line 211):

```bash
# ============ tool-call reward shaping (off by default) ============
# beta=0.0 disables the feature entirely (no annotator import). When enabled,
# toolcall-annotation must be installed on every Ray worker.
export SWE_TOOLCALL_SHAPING_BETA="${SWE_TOOLCALL_SHAPING_BETA:-0.0}"
export SWE_TOOLCALL_SHAPING_BUDGET="${SWE_TOOLCALL_SHAPING_BUDGET:-1.0}"
```

And add to the `ALGO_ARGS` array (so the train side uses the custom advantage; harmless when beta=0):

```bash
   --custom-advantage-function-path examples.coding_agent_rl.turn_shaping.compute_advantage
```

(The `SWE_`-prefix loop at lines ~242-244 auto-forwards both new env vars to workers; no runtime-env edit needed.)

- [ ] **Step 6: Lint and commit**

```bash
pre-commit run --files examples/coding_agent_rl/generate.py examples/coding_agent_rl/test_turn_shaping.py
git add examples/coding_agent_rl/generate.py examples/coding_agent_rl/test_turn_shaping.py examples/coding_agent_rl/run_021_32b_a4b_scaleswe_openhands_8nodes.sh
git commit -m "feat(coding_agent_rl): wire tool-call shaping config + advantage hook"
```

---

### Task 7: End-to-end alignment invariant test

**Files:**
- Test: `tests/test_toolcall_shaping_pipeline.py` (add the invariant case)

**Interfaces:**
- Consumes: everything from Tasks 1-4.

**Purpose:** guard the five-touch-point plumbing — a shaping vector produced by the manager must survive convert → DP-split with the same per-sample shape as `loss_masks`. (CP-slice is covered by Task 4's source assertion since CUDA/Megatron isn't available in CPU CI.)

- [ ] **Step 1: Write the test**

Add to `tests/test_toolcall_shaping_pipeline.py`:

```python
def test_shaping_survives_convert_with_loss_mask_shape():
    """Per-sample shaping vectors match loss_mask lengths through convert."""
    samples = [
        _sample(0, 3, [0.0, -0.5, -0.5]),
        _sample(1, 2, [0.0, -0.1]),
    ]
    td = _convert(samples)
    assert "toolcall_turn_shaping" in td
    for shp, lm in zip(td["toolcall_turn_shaping"], td["loss_masks"], strict=True):
        assert len(shp) == len(lm)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python tests/test_toolcall_shaping_pipeline.py`
Expected: PASS (4 tests) — this asserts the Task 3 promotion keeps shapes aligned.

- [ ] **Step 3: Full regression sweep**

Run:
```bash
python tests/test_agent/test_turn_shaping.py
python tests/test_agent/test_trajectory_manager_branching.py
python tests/test_agent/test_adapters.py
python tests/test_toolcall_shaping_pipeline.py
python examples/coding_agent_rl/test_turn_shaping.py
```
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
pre-commit run --files tests/test_toolcall_shaping_pipeline.py
git add tests/test_toolcall_shaping_pipeline.py
git commit -m "test: alignment invariant for toolcall_turn_shaping pipeline"
```

---

## Notes for the implementer

- **Registering the CPU tests in CI** is a separate follow-up governed by the `add-tests-and-ci` skill (edit `.github/workflows/pr-test.yml.j2` and regenerate — never the `.yml`). Not in scope for these tasks unless the reviewer asks; the tests run standalone via `python <file>` regardless.
- If `_convert_samples_to_train_data` reads `self.args` attributes beyond those stubbed in Task 3, extend the `SimpleNamespace` — do not change production code to accommodate the test.
- The `MessageNode` tool-response location (Task 5) is best-effort: on the final turn there is no recorded tool response yet, and the scorer correctly scores against `None`. This matches the annotator's tolerance for missing responses and does not break alignment (the assistant tokens are still shaped by the tool-call's own validity checks).
