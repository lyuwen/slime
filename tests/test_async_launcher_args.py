"""Argument-level checks for the fully-async coding-agent launcher.

No GPUs or live cluster required. The test verifies that the launcher:
  - selects train_async.py (not train.py)
  - omits --colocate
  - defaults to 64 actor + 32 rollout GPUs
  - wires the fully-async rollout function path
  - enables the three metrics-only mismatch flags
  - does not set any eval flag
"""
import pathlib
import pytest

LAUNCHER = (
    pathlib.Path(__file__).parent.parent
    / "examples"
    / "coding_agent_rl"
    / "run_021_32b_a4b_scaleswe_openhands_8nodes_fully_async.sh"
)


@pytest.fixture(scope="module")
def launcher_text():
    return LAUNCHER.read_text()


def test_launcher_exists():
    assert LAUNCHER.exists(), f"launcher not found: {LAUNCHER}"


def test_uses_train_async(launcher_text):
    assert "train_async.py" in launcher_text, "should launch train_async.py"
    # train.py (without _async) must not appear as the driver argument
    # (it still appears as a substring of train_async.py, so check the full token)
    lines_with_trainpy = [
        l for l in launcher_text.splitlines()
        if "train.py" in l and "train_async.py" not in l and not l.strip().startswith("#")
    ]
    assert not lines_with_trainpy, (
        f"found non-async train.py reference in active lines: {lines_with_trainpy}"
    )


def test_no_colocate(launcher_text):
    active_lines = [
        l for l in launcher_text.splitlines()
        if "--colocate" in l and not l.strip().startswith("#")
    ]
    assert not active_lines, f"--colocate must not appear in active lines: {active_lines}"


def test_default_actor_gpus(launcher_text):
    assert 'ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-' in launcher_text
    assert 'ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"' in launcher_text
    # Default node count should be 8 (8*8 = 64 actor GPUs)
    assert 'ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-${MLP_WORKER_NUM:-8}}"' in launcher_text


def test_default_rollout_gpus(launcher_text):
    assert 'ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-32}"' in launcher_text
    assert '--rollout-num-gpus "${ROLLOUT_NUM_GPUS}"' in launcher_text


def test_fully_async_rollout_function(launcher_text):
    assert (
        "slime.rollout.fully_async_rollout.generate_rollout_fully_async" in launcher_text
    ), "fully-async rollout function path must be present"
    assert "--rollout-function-path" in launcher_text


def test_mismatch_metrics_enabled(launcher_text):
    assert "--get-mismatch-metrics" in launcher_text
    assert "examples/train_infer_mismatch_helper/mis.yaml" in launcher_text
    assert (
        "examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp"
        in launcher_text
    )


def test_no_use_tis(launcher_text):
    active_lines = [
        l for l in launcher_text.splitlines()
        if "--use-tis" in l and not l.strip().startswith("#")
    ]
    assert not active_lines, "--use-tis must not appear (metrics-only pass)"


def test_no_eval_args(launcher_text):
    for flag in ("--eval-interval", "--eval-prompt-data", "--eval-config"):
        active_lines = [
            l for l in launcher_text.splitlines()
            if flag in l and not l.strip().startswith("#")
        ]
        assert not active_lines, f"{flag} must not appear (fully-async rejects eval)"


def test_startup_validation_present(launcher_text):
    # The script should guard against a mismatched ROLLOUT_TP_SIZE
    assert "ROLLOUT_NUM_GPUS % ROLLOUT_TP_SIZE" in launcher_text


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
