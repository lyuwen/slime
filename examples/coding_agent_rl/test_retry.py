"""Unit tests for rollout retry mechanism in generate.py.

Tests _is_retryable() directly (synchronous, no sandbox needed) and the
retry loop in generate() using mocks that simulate transient E2B failures.
Run with: pytest examples/coding_agent_rl/test_retry.py
"""

import asyncio
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.coding_agent_rl.generate import CONFIG, SweConfig, _is_retryable  # noqa: E402


# ---------------------------------------------------------------------------
# _is_retryable — pure function, no async needed
# ---------------------------------------------------------------------------


def test_retryable_e2b_exec_exit_255():
    assert _is_retryable(RuntimeError("e2b exec failed (exit=255): useradd failed"))


def test_retryable_e2b_exec_exit_255_alt_format():
    assert _is_retryable(RuntimeError("e2b exec failed (exit 255)"))


def test_retryable_sandbox_runtimeerror():
    assert _is_retryable(RuntimeError("sandbox boot failed: container not ready"))


def test_retryable_e2b_runtimeerror():
    assert _is_retryable(RuntimeError("e2b connection lost"))


def test_retryable_connection_error():
    assert _is_retryable(RuntimeError("connection refused"))


def test_retryable_network_unavailable():
    assert _is_retryable(RuntimeError("network unavailable"))


def test_not_retryable_value_error():
    assert not _is_retryable(ValueError("missing required field in metadata"))


def test_not_retryable_key_error():
    assert not _is_retryable(KeyError("instance_id"))


def test_not_retryable_asyncio_timeout():
    # asyncio.TimeoutError must never be retried — it's the wall-clock guard
    assert not _is_retryable(asyncio.TimeoutError())


def test_not_retryable_generic_runtime_error():
    # RuntimeError without sandbox/e2b/exec context is not retryable
    assert not _is_retryable(RuntimeError("division by zero"))


class SandboxException(Exception):
    """Stand-in matching the kernel classifier's name-based check
    (is_fresh_sandbox_retryable matches on type(e).__name__)."""


def test_is_retryable_kernel_sandbox_exception():
    # Generic SandboxException is classified as fresh-retryable by the kernel.
    assert _is_retryable(SandboxException("500: error creating file: ... permission denied"))


def test_is_retryable_example_runtime_error():
    # Example-specific check: RuntimeError from a check=True exec.
    assert _is_retryable(RuntimeError("e2b exec failed exit=255"))


def test_is_retryable_non_retryable():
    assert not _is_retryable(ValueError("invalid config"))


def test_retry_config_rejects_negative_value():
    with patch.dict("os.environ", {"SWE_ROLLOUT_RETRIES": "-1"}):
        with pytest.raises(ValueError, match="must be non-negative"):
            SweConfig.from_env()


def test_retry_config_rejects_non_integer_value():
    with patch.dict("os.environ", {"SWE_ROLLOUT_RETRIES": "invalid"}):
        with pytest.raises(ValueError, match="invalid literal"):
            SweConfig.from_env()


# ---------------------------------------------------------------------------
# generate() retry loop — integration tests with mocked E2B
# ---------------------------------------------------------------------------


def _make_state_mock():
    """Minimal _AdapterService mock."""
    state = MagicMock()
    state.adapter.open_session = MagicMock()
    state.adapter.drop_session = AsyncMock()
    state.adapter.finish_session = AsyncMock(return_value=[MagicMock()])
    state.adapter_url = "http://test:18001"
    state.max_context_len = 131072
    state.tool_parser = "glm47"
    state.reasoning_parser = "deepseek-r1"
    return state


def _make_sample_mock():
    sample = MagicMock()
    sample.metadata = {
        "instance_id": "test_repo_pr1",
        "swe_metadata": {
            "instance_id": "test_repo_pr1",
            "image": "test-image:latest",
            "workdir": "/workspace/test",
        },
    }
    sample.session_id = None
    return sample


@pytest.fixture
def base_patches():
    """Patches shared by all generate() integration tests."""
    with (
        patch("examples.coding_agent_rl.generate._AdapterService") as mock_svc_cls,
        patch("examples.coding_agent_rl.generate.boot_agent_sandbox") as mock_boot,
        patch("examples.coding_agent_rl.generate.swe") as mock_swe,
        patch("examples.coding_agent_rl.generate.HARNESS_CLS") as mock_harness_cls,
        patch("examples.coding_agent_rl.generate._session_id", return_value="sess-123"),
        patch("examples.coding_agent_rl.generate.get_prompt", return_value="prompt"),
        patch("examples.coding_agent_rl.generate._abort_result", side_effect=lambda s, r, i: [r]),
        patch("examples.coding_agent_rl.generate.AGENT_NAME", "openhands"),
        patch(
            "examples.coding_agent_rl.generate.CONFIG",
            replace(CONFIG, rollout_retries=2),
        ),
        patch("examples.coding_agent_rl.generate._read_sandbox_trajectory", new_callable=AsyncMock, return_value=None),
    ):
        state = _make_state_mock()
        mock_svc_cls.return_value = state

        mock_sb = AsyncMock()
        mock_boot.return_value.__aenter__ = AsyncMock(return_value=mock_sb)
        mock_boot.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_swe.get_metadata.return_value = _make_sample_mock().metadata["swe_metadata"]
        mock_swe.evaluability_check.return_value = None
        mock_swe.git_diff = AsyncMock(return_value="diff --git a/f.py b/f.py")
        mock_swe.run_evaluation = AsyncMock(return_value=(1.0, True))

        mock_harness_inst = AsyncMock()
        mock_harness_inst.run = AsyncMock(return_value=0)
        mock_harness_cls.return_value = mock_harness_inst

        yield {
            "state": state,
            "mock_swe": mock_swe,
            "mock_harness_inst": mock_harness_inst,
        }


@pytest.mark.anyio
async def test_success_on_first_attempt(base_patches):
    """No failures — returns samples, drop_session called once."""
    from examples.coding_agent_rl.generate import generate

    mock_swe = base_patches["mock_swe"]
    mock_swe.prepare_workspace = AsyncMock()

    await generate(MagicMock(), _make_sample_mock(), {})

    assert mock_swe.prepare_workspace.call_count == 1
    base_patches["state"].adapter.drop_session.assert_awaited_once()


@pytest.mark.anyio
async def test_retry_succeeds_on_second_attempt(base_patches):
    """First attempt raises a retryable error; second succeeds."""
    from examples.coding_agent_rl.generate import generate

    mock_swe = base_patches["mock_swe"]
    mock_swe.prepare_workspace = AsyncMock(side_effect=[RuntimeError("e2b exec failed (exit=255): useradd"), None])

    sleep_calls = []

    async def fast_sleep(n):
        sleep_calls.append(n)

    with patch("examples.coding_agent_rl.generate.asyncio.sleep", side_effect=fast_sleep):
        result = await generate(MagicMock(), _make_sample_mock(), {})

    assert mock_swe.prepare_workspace.call_count == 2
    # Backoff sleep after first failure (2^0 = 1s), plus the cleanup sleep in finally (10s)
    assert 1 in sleep_calls
    assert sleep_calls[0] == 1
    # Result is samples, not an abort string
    assert result is not None
    assert not isinstance(result, list) or result[0] != "exception:RuntimeError"


@pytest.mark.anyio
async def test_retry_exhausted_returns_abort(base_patches):
    """All attempts fail with retryable error — abort after max_retries+1 tries."""
    from examples.coding_agent_rl.generate import generate

    mock_swe = base_patches["mock_swe"]
    mock_swe.prepare_workspace = AsyncMock(side_effect=RuntimeError("e2b exec failed (exit=255): persistent"))

    sleep_calls = []

    async def fast_sleep(n):
        sleep_calls.append(n)

    with patch("examples.coding_agent_rl.generate.asyncio.sleep", side_effect=fast_sleep):
        result = await generate(MagicMock(), _make_sample_mock(), {})

    assert mock_swe.prepare_workspace.call_count == 3  # 1 original + 2 retries
    # Backoff sleeps: 2^0=1s and 2^1=2s; finally block adds a 10s cleanup sleep
    assert sleep_calls[:2] == [1, 2]
    assert result == ["exception:RuntimeError"]


@pytest.mark.anyio
async def test_non_retryable_aborts_immediately(base_patches):
    """Non-retryable error — no sleep, no retry, abort on first attempt."""
    from examples.coding_agent_rl.generate import generate

    mock_swe = base_patches["mock_swe"]
    mock_swe.prepare_workspace = AsyncMock(side_effect=ValueError("bad metadata"))

    sleep_calls = []

    async def fast_sleep(n):
        sleep_calls.append(n)

    with patch("examples.coding_agent_rl.generate.asyncio.sleep", side_effect=fast_sleep):
        await generate(MagicMock(), _make_sample_mock(), {})

    assert mock_swe.prepare_workspace.call_count == 1
    # The only sleep is the 10s cleanup in the finally block — no backoff sleeps
    assert 1 not in sleep_calls
    assert 2 not in sleep_calls


@pytest.mark.anyio
async def test_cleanup_called_on_success(base_patches):
    """drop_session is always called, even on success."""
    from examples.coding_agent_rl.generate import generate

    base_patches["mock_swe"].prepare_workspace = AsyncMock()

    await generate(MagicMock(), _make_sample_mock(), {})

    base_patches["state"].adapter.drop_session.assert_awaited_once()


@pytest.mark.anyio
async def test_no_cleanup_when_setup_never_opens_session(base_patches):
    """Setup exhaustion does not drop a session that was never opened."""
    from examples.coding_agent_rl.generate import generate

    base_patches["mock_swe"].prepare_workspace = AsyncMock(side_effect=RuntimeError("e2b exec failed (exit=255)"))

    async def fast_sleep(n):
        pass

    with patch("examples.coding_agent_rl.generate.asyncio.sleep", side_effect=fast_sleep):
        await generate(MagicMock(), _make_sample_mock(), {})

    base_patches["state"].adapter.drop_session.assert_not_awaited()


@pytest.mark.anyio
async def test_failure_after_session_open_is_not_retried(base_patches):
    """Once the harness starts recording turns, failures abort without retry."""
    from examples.coding_agent_rl.generate import generate

    mock_swe = base_patches["mock_swe"]
    mock_swe.prepare_workspace = AsyncMock()
    base_patches["mock_harness_inst"].run = AsyncMock(
        side_effect=RuntimeError("e2b exec failed (exit=255) after agent start")
    )

    async def fast_sleep(_):
        pass

    with patch("examples.coding_agent_rl.generate.asyncio.sleep", side_effect=fast_sleep):
        result = await generate(MagicMock(), _make_sample_mock(), {})

    assert mock_swe.prepare_workspace.call_count == 1
    assert base_patches["mock_harness_inst"].run.await_count == 1
    assert result == ["exception:RuntimeError"]
    base_patches["state"].adapter.drop_session.assert_awaited_once()
