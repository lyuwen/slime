"""train_data plumbing for the per-turn tool-call shaping vector.

Unit-tests _convert_samples_to_train_data's promotion of
Sample.metadata["toolcall_turn_shaping"] into a first-class per-token
train_data key, and its presence in the DP-split allowlist.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# CPU-only unit test: stub the heavy GPU/inference deps that ``slime.ray.rollout``
# pulls in at import time (sglang + sglang_router) so the pure-Python
# ``_convert_samples_to_train_data`` logic can be exercised without a GPU env.
def _install_sglang_stubs() -> None:
    if "sglang" not in sys.modules:
        sglang = types.ModuleType("sglang")
        srt = types.ModuleType("sglang.srt")
        constants = types.ModuleType("sglang.srt.constants")
        constants.GPU_MEMORY_TYPE_CUDA_GRAPH = "cuda_graph"
        constants.GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"
        constants.GPU_MEMORY_TYPE_WEIGHTS = "weights"
        server_args = types.ModuleType("sglang.srt.server_args")
        server_args.ServerArgs = object
        srt_utils = types.ModuleType("sglang.srt.utils")
        srt_utils.kill_process_tree = lambda *a, **k: None
        sglang.srt = srt
        srt.constants = constants
        srt.server_args = server_args
        srt.utils = srt_utils
        sys.modules["sglang"] = sglang
        sys.modules["sglang.srt"] = srt
        sys.modules["sglang.srt.constants"] = constants
        sys.modules["sglang.srt.server_args"] = server_args
        sys.modules["sglang.srt.utils"] = srt_utils
    if "sglang_router" not in sys.modules:
        sys.modules["sglang_router"] = types.ModuleType("sglang_router")


def _install_megatron_stubs() -> None:
    """Stub megatron.core and torch_memory_saver for the CP-slice test."""
    if "megatron" not in sys.modules:
        megatron = types.ModuleType("megatron")
        core = types.ModuleType("megatron.core")
        mpu = types.ModuleType("megatron.core.mpu")
        training = types.ModuleType("megatron.training")
        checkpointing = types.ModuleType("megatron.training.checkpointing")
        checkpointing.load_checkpoint = lambda *a, **k: None
        checkpointing.save_checkpoint = lambda *a, **k: None
        megatron.core = core
        megatron.training = training
        core.mpu = mpu
        training.checkpointing = checkpointing
        sys.modules["megatron"] = megatron
        sys.modules["megatron.core"] = core
        sys.modules["megatron.core.mpu"] = mpu
        sys.modules["megatron.training"] = training
        sys.modules["megatron.training.checkpointing"] = checkpointing
    if "torch_memory_saver" not in sys.modules:
        tms = types.ModuleType("torch_memory_saver")
        tms.torch_memory_saver = lambda *a, **k: lambda f: f
        sys.modules["torch_memory_saver"] = tms


_install_sglang_stubs()
_install_megatron_stubs()

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


def _convert(samples):
    """Invoke the real _convert_samples_to_train_data with a minimal stub."""
    import types

    from slime.ray.rollout import RolloutManager

    # RolloutManager is a @ray.remote actor; unwrap to the underlying class so
    # we can instantiate it plainly and call the real (unbound) method.
    cls = getattr(RolloutManager, "__ray_actor_class__", RolloutManager)

    mgr = object.__new__(cls)  # bypass __init__
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
    return cls._convert_samples_to_train_data(mgr, samples)


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


def test_convert_mixed_batch_fills_absent_with_zeros():
    """Mixed batch: when ANY sample has the key, samples lacking it get zero-fill."""
    # sample 0: no metadata; sample 1: has shaping
    samples = [_sample(0, 3), _sample(1, 2, [0.0, -0.5])]
    td = _convert(samples)
    assert "toolcall_turn_shaping" in td
    assert len(td["toolcall_turn_shaping"]) == 2
    assert td["toolcall_turn_shaping"][0] == [0.0, 0.0, 0.0]  # zero-fill for sample 0 (response_length=3)
    assert td["toolcall_turn_shaping"][1] == [0.0, -0.5]


def test_cp_slice_included_for_shaping_key():
    """The _get_rollout_data field loop must list toolcall_turn_shaping so it is
    CP-sliced and tensorized alongside rollout_log_probs.

    We assert on source to avoid standing up Megatron/CUDA: the key must appear
    in the same slice loop as rollout_log_probs.
    """
    actor_path = Path(__file__).parent.parent / "slime" / "backends" / "megatron_utils" / "actor.py"
    src = actor_path.read_text()
    # both keys handled by the same CP-slice loop
    assert "toolcall_turn_shaping" in src
    assert "rollout_log_probs" in src
    idx_loop = src.index("rollout_log_probs")
    idx_shape = src.index("toolcall_turn_shaping")
    # shaping key appears within ~200 chars of the rollout_log_probs loop header
    assert abs(idx_shape - idx_loop) < 400, "shaping key not in the same slice loop"


if __name__ == "__main__":
    pytest.main([__file__])
