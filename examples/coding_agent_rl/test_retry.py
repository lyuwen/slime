"""Unit tests for rollout retry mechanism in generate.py.

Tests _is_retryable() directly (synchronous, no sandbox needed) and the
retry loop in generate() using mocks that simulate transient E2B failures.
Run with: pytest examples/coding_agent_rl/test_retry.py
"""

import asyncio
import sys
from dataclasses import dataclass, replace
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


def test_permanent_sandbox_exception_not_resurrected_by_create_heuristic():
    # FIX A: the kernel classifier rejects a permanent auth SandboxException, and
    # the word "create" in the message must NOT flip it back to retryable via the
    # fallthrough heuristic. is_fresh_sandbox_retryable is the single source of
    # truth for SandboxException.
    assert not _is_retryable(SandboxException("unauthorized to create sandbox"))


def test_transient_sandbox_exception_still_retryable():
    # A transient SandboxException (missing/stopped sandbox) stays retryable.
    assert _is_retryable(SandboxException("sandbox does not exist"))


def test_transient_boot_sandbox_exception_retryable_via_classifier_only():
    # Boot/create failures CAN surface as SandboxException (AsyncSandbox.create
    # raises it). A generic transient provider/gateway boot failure must stay
    # retryable — and the message here deliberately contains NO fallthrough
    # heuristic keyword (no "boot"/"create"/"connection"/"e2b"/"sandbox"...), so
    # the only path that can classify it retryable is the kernel classifier at
    # the top of _is_retryable. If that call were removed, this fails.
    assert _is_retryable(SandboxException("502 bad gateway from provider"))


def test_retry_config_rejects_negative_value():
    with patch.dict("os.environ", {"SWE_ROLLOUT_RETRIES": "-1"}):
        with pytest.raises(ValueError, match="must be non-negative"):
            SweConfig.from_env()


def test_retry_config_rejects_non_integer_value():
    with patch.dict("os.environ", {"SWE_ROLLOUT_RETRIES": "invalid"}):
        with pytest.raises(ValueError, match="invalid literal"):
            SweConfig.from_env()


def test_retry_policy_default():
    import os

    os.environ.pop("SWE_ROLLOUT_RETRY_POLICY", None)
    try:
        assert SweConfig.from_env().retry_policy == "pre-launch"
    finally:
        os.environ.pop("SWE_ROLLOUT_RETRY_POLICY", None)


def test_retry_policy_custom():
    import os

    os.environ["SWE_ROLLOUT_RETRY_POLICY"] = "retry-from-scratch"
    try:
        assert SweConfig.from_env().retry_policy == "retry-from-scratch"
    finally:
        os.environ.pop("SWE_ROLLOUT_RETRY_POLICY", None)


def test_retry_policy_invalid():
    import os

    os.environ["SWE_ROLLOUT_RETRY_POLICY"] = "invalid-policy"
    try:
        with pytest.raises(ValueError, match="SWE_ROLLOUT_RETRY_POLICY.*invalid"):
            SweConfig.from_env()
    finally:
        os.environ.pop("SWE_ROLLOUT_RETRY_POLICY", None)


# ---------------------------------------------------------------------------
# generate() retry loop — integration tests with mocked E2B
# ---------------------------------------------------------------------------


def _make_state_mock():
    """Minimal _AdapterService mock."""
    state = MagicMock()
    state.adapter.open_session = MagicMock()
    state.adapter.drop_session = AsyncMock()
    state.adapter.finish_session = AsyncMock(return_value=[MagicMock()])
    # No turn recorded by default; retry gate treats failures before a turn as retryable.
    state.adapter.manager.has_session = MagicMock(return_value=False)
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
async def test_generate_sets_winning_sid_with_attempt_suffix(base_patches):
    """Real generate(): first attempt fails retryably after open, second succeeds.

    Asserts the REAL retry loop stamps base_sample.session_id with the winning
    attempt's ``-a{N}`` suffix (not the inline FakeAdapter copy). ``_session_id``
    is patched to "sess-123" in base_patches, so the winning sid after the
    second attempt is "sess-123-a1". The first attempt fails inside the harness
    run (after open_session) with a retryable error and no turns recorded
    (has_session=False), so the per-attempt sessions "-a0" (opened then dropped)
    and "-a1" (winning) are both observable.
    """
    from examples.coding_agent_rl.generate import generate

    mock_swe = base_patches["mock_swe"]
    mock_swe.prepare_workspace = AsyncMock()
    base_patches["mock_harness_inst"].run = AsyncMock(
        side_effect=[RuntimeError("e2b exec failed (exit=255): mid-launch"), 0]
    )

    async def fast_sleep(_):
        pass

    sample = _make_sample_mock()
    with patch("examples.coding_agent_rl.generate.asyncio.sleep", side_effect=fast_sleep):
        await generate(MagicMock(), sample, {})

    # Second attempt won → winning sid carries the -a1 suffix.
    assert sample.session_id == "sess-123-a1"
    # Per-attempt sids advance a0 → a1; the failed a0 session is dropped.
    open_sids = [c.args[0] for c in base_patches["state"].adapter.open_session.call_args_list]
    assert open_sids == ["sess-123-a0", "sess-123-a1"]
    drop_sids = [c.args[0] for c in base_patches["state"].adapter.drop_session.await_args_list]
    assert "sess-123-a0" in drop_sids


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
    # A turn was recorded before the run() failure → pre-launch policy hard-fails.
    base_patches["state"].adapter.manager.has_session = MagicMock(return_value=True)

    async def fast_sleep(_):
        pass

    with patch("examples.coding_agent_rl.generate.asyncio.sleep", side_effect=fast_sleep):
        result = await generate(MagicMock(), _make_sample_mock(), {})

    assert mock_swe.prepare_workspace.call_count == 1
    assert base_patches["mock_harness_inst"].run.await_count == 1
    assert result == ["exception:RuntimeError"]
    base_patches["state"].adapter.drop_session.assert_awaited_once()


# ---------------------------------------------------------------------------
# Gate-logic tests — per-attempt session id + has_session/policy gating
#
# These exercise the EXACT gate ordering used inside generate()'s retry loop
# with a self-contained FakeAdapter, rather than driving the real generate().
# The local SandboxException stand-in (defined above) is classified retryable
# by _is_retryable via the kernel classifier, matching e2b's real one by name.
# ---------------------------------------------------------------------------


@dataclass
class FakeAdapter:
    """Minimal fake adapter mirroring the pieces the retry gate touches:
    a manager with has_session()/record_turn() and async drop_session()."""

    class FakeManager:
        def __init__(self):
            self._sessions = set()

        def has_session(self, sid: str) -> bool:
            return sid in self._sessions

        def record_turn(self, sid: str):
            self._sessions.add(sid)

    def __init__(self):
        self.manager = self.FakeManager()
        self._open_sessions = set()

    def open_session(self, sid: str, **kwargs):
        self._open_sessions.add(sid)

    async def drop_session(self, sid: str, **kwargs):
        self._open_sessions.discard(sid)


@pytest.mark.anyio
async def test_pre_launch_retry_success_kernel_error():
    """Pre-launch policy: kernel-classified error with no turns → retry → success."""
    adapter = FakeAdapter()
    attempt_count = 0

    async def fake_run():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            raise SandboxException("500: error creating file: permission denied")
        return 0  # success on second attempt

    config = replace(CONFIG, retry_policy="pre-launch", rollout_retries=3)

    # Simulate the generate() retry loop's gate ordering.
    base_sid = "test-session"
    session_id = None
    for attempt in range(config.rollout_retries + 1):
        session_id = f"{base_sid}-a{attempt}"
        session_open = False

        try:
            adapter.open_session(session_id)
            session_open = True
            exit_code = await fake_run()
            assert exit_code == 0
            break
        except Exception as error:
            if config.retry_policy == "always-fail":
                raise
            turns_recorded = adapter.manager.has_session(session_id)
            if not _is_retryable(error) or attempt >= config.rollout_retries:
                raise
            if config.retry_policy == "pre-launch" and turns_recorded:
                raise
            if session_open:
                await adapter.drop_session(session_id)
                session_open = False
            await asyncio.sleep(0.01)  # minimal sleep for test

    assert attempt_count == 2, "Should retry once and succeed on attempt 2"
    assert session_id == f"{base_sid}-a1", "Winning sid should be attempt 1"


@pytest.mark.anyio
async def test_hard_fail_after_turns():
    """Pre-launch policy: error after turns recorded → immediate hard-fail, no retry."""
    adapter = FakeAdapter()
    attempt_count = 0

    async def fake_run_with_turn():
        nonlocal attempt_count
        attempt_count += 1
        adapter.manager.record_turn("test-session-a0")  # a turn lands, then we die
        raise SandboxException("sandbox died mid-run")

    config = replace(CONFIG, retry_policy="pre-launch", rollout_retries=3)

    base_sid = "test-session"
    raised = False

    for attempt in range(config.rollout_retries + 1):
        session_id = f"{base_sid}-a{attempt}"

        try:
            adapter.open_session(session_id)
            await fake_run_with_turn()
        except Exception as error:
            if config.retry_policy == "always-fail":
                raised = True
                break
            turns_recorded = adapter.manager.has_session(session_id)
            if not _is_retryable(error) or attempt >= config.rollout_retries:
                raised = True
                break
            if config.retry_policy == "pre-launch" and turns_recorded:
                raised = True
                break
            pytest.fail("Should have hard-failed after turns")

    assert raised, "Should have raised without retry"
    assert attempt_count == 1, "Should only run once (no retry after turns)"


# ---------------------------------------------------------------------------
# Remaining policy coverage — driven through the REAL generate() retry loop
# via base_patches, so each row asserts against the shipped gate, not a copy.
#
# Failures are injected inside HARNESS_CLS().run (post-open_session) so a
# per-attempt session is actually opened and can be observed being dropped.
# has_session (turns_recorded) and CONFIG.retry_policy are varied per test.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_pre_launch_retry_success_example_classifier(base_patches):
    """Pre-launch (default): example-classified RuntimeError inside run(), no
    turns recorded → real generate() retries and the second attempt wins.

    Distinct from test_generate_sets_winning_sid_with_attempt_suffix only in
    intent: this asserts the *example* classifier branch of _is_retryable
    ("e2b exec failed") drives a real retry, complementing the kernel-classifier
    FakeAdapter unit test above.
    """
    from examples.coding_agent_rl.generate import generate

    base_patches["mock_swe"].prepare_workspace = AsyncMock()
    base_patches["mock_harness_inst"].run = AsyncMock(
        side_effect=[RuntimeError("e2b exec failed (exit=255): mid-launch"), 0]
    )

    async def fast_sleep(_):
        pass

    sample = _make_sample_mock()
    with patch("examples.coding_agent_rl.generate.asyncio.sleep", side_effect=fast_sleep):
        result = await generate(MagicMock(), sample, {})

    assert base_patches["mock_harness_inst"].run.await_count == 2
    assert sample.session_id == "sess-123-a1"
    assert result == base_patches["state"].adapter.finish_session.return_value


@pytest.mark.anyio
async def test_non_retryable_after_open_aborts_without_retry(base_patches):
    """A non-retryable error raised inside run() (after open_session) aborts on
    the first attempt: run() is invoked once and the opened sid is dropped."""
    from examples.coding_agent_rl.generate import generate

    base_patches["mock_swe"].prepare_workspace = AsyncMock()
    base_patches["mock_harness_inst"].run = AsyncMock(side_effect=ValueError("not a sandbox problem"))

    async def fast_sleep(_):
        pass

    with patch("examples.coding_agent_rl.generate.asyncio.sleep", side_effect=fast_sleep):
        result = await generate(MagicMock(), _make_sample_mock(), {})

    assert base_patches["mock_harness_inst"].run.await_count == 1
    assert result == ["exception:ValueError"]
    base_patches["state"].adapter.drop_session.assert_awaited_once()


@pytest.mark.anyio
async def test_exhaust_retries_drops_every_attempt_sid(base_patches):
    """Every attempt fails retryably inside run() → abort after rollout_retries+1
    attempts, with each per-attempt session opened and then dropped.

    base_patches pins rollout_retries=2, so attempts a0/a1/a2 all run; each opens
    a session and each is dropped (the last via the finally-block cleanup).
    """
    from examples.coding_agent_rl.generate import generate

    base_patches["mock_swe"].prepare_workspace = AsyncMock()
    base_patches["mock_harness_inst"].run = AsyncMock(
        side_effect=RuntimeError("e2b exec failed (exit=255): persistent")
    )

    async def fast_sleep(_):
        pass

    with patch("examples.coding_agent_rl.generate.asyncio.sleep", side_effect=fast_sleep):
        result = await generate(MagicMock(), _make_sample_mock(), {})

    assert base_patches["mock_harness_inst"].run.await_count == 3  # a0, a1, a2
    assert result == ["exception:RuntimeError"]

    open_sids = [c.args[0] for c in base_patches["state"].adapter.open_session.call_args_list]
    assert open_sids == ["sess-123-a0", "sess-123-a1", "sess-123-a2"]
    drop_sids = [c.args[0] for c in base_patches["state"].adapter.drop_session.await_args_list]
    # Every opened session is dropped (retry drops a0/a1, finally drops the last a2).
    assert set(open_sids) <= set(drop_sids)


@pytest.mark.anyio
async def test_retry_from_scratch_retries_after_turns(base_patches):
    """retry-from-scratch: a retryable error after a turn was recorded still
    retries (unlike pre-launch), dropping the partial session, and a fresh sid wins.
    """
    from examples.coding_agent_rl import generate as gen_mod
    from examples.coding_agent_rl.generate import generate

    base_patches["mock_swe"].prepare_workspace = AsyncMock()
    base_patches["mock_harness_inst"].run = AsyncMock(
        side_effect=[RuntimeError("e2b exec failed (exit=255): died mid-turn"), 0]
    )
    # A turn WAS recorded on the failing attempt; pre-launch would hard-fail here,
    # retry-from-scratch must retry anyway.
    base_patches["state"].adapter.manager.has_session = MagicMock(return_value=True)

    async def fast_sleep(_):
        pass

    sample = _make_sample_mock()
    with (
        patch.object(gen_mod, "CONFIG", replace(gen_mod.CONFIG, retry_policy="retry-from-scratch")),
        patch("examples.coding_agent_rl.generate.asyncio.sleep", side_effect=fast_sleep),
    ):
        result = await generate(MagicMock(), sample, {})

    assert base_patches["mock_harness_inst"].run.await_count == 2
    assert sample.session_id == "sess-123-a1"
    assert result == base_patches["state"].adapter.finish_session.return_value
    # The partial (turns-recorded) first session was dropped before retrying.
    drop_sids = [c.args[0] for c in base_patches["state"].adapter.drop_session.await_args_list]
    assert "sess-123-a0" in drop_sids


@pytest.mark.anyio
async def test_always_fail_policy_aborts_on_first_retryable_error(base_patches):
    """always-fail: even a retryable error inside run() aborts immediately with
    no retry (run() invoked exactly once)."""
    from examples.coding_agent_rl import generate as gen_mod
    from examples.coding_agent_rl.generate import generate

    base_patches["mock_swe"].prepare_workspace = AsyncMock()
    base_patches["mock_harness_inst"].run = AsyncMock(
        side_effect=RuntimeError("e2b exec failed (exit=255): retryable but policy=always-fail")
    )

    async def fast_sleep(_):
        pass

    with (
        patch.object(gen_mod, "CONFIG", replace(gen_mod.CONFIG, retry_policy="always-fail")),
        patch("examples.coding_agent_rl.generate.asyncio.sleep", side_effect=fast_sleep),
    ):
        result = await generate(MagicMock(), _make_sample_mock(), {})

    assert base_patches["mock_harness_inst"].run.await_count == 1
    assert result == ["exception:RuntimeError"]


@pytest.mark.anyio
async def test_always_fail_policy_aborts_provisioning_without_boot_retries():
    """FIX B: under always-fail, a retryable *provisioning* error inside
    boot_agent_sandbox must fail immediately — boot_agent_sandbox's own internal
    retry loop (CONFIG.boot_retries) must NOT retry it. The real boot loop runs
    (boot_agent_sandbox is NOT patched); only E2BSandbox construction/entry is
    patched to raise. Asserts the sandbox is created exactly ONCE despite
    boot_retries defaulting to 2.
    """
    from examples.coding_agent_rl import generate as gen_mod
    from examples.coding_agent_rl.generate import generate

    async def fast_sleep(_):
        pass

    state = _make_state_mock()

    fake_sb = MagicMock()
    fake_sb.__aenter__ = AsyncMock(side_effect=SandboxException("does not exist: provisioning gateway"))
    fake_sb.__aexit__ = AsyncMock(return_value=None)

    fake_e2b_cls = MagicMock(return_value=fake_sb)

    mock_harness_inst = AsyncMock()
    mock_harness_inst.install_cli = AsyncMock()
    mock_harness_cls = MagicMock(return_value=mock_harness_inst)

    mock_swe = MagicMock()
    mock_swe.get_metadata.return_value = _make_sample_mock().metadata["swe_metadata"]
    mock_swe.evaluability_check.return_value = None
    mock_swe.prepare_workspace = AsyncMock()

    with (
        patch("examples.coding_agent_rl.generate._AdapterService", return_value=state),
        patch("examples.coding_agent_rl.generate.E2BSandbox", fake_e2b_cls),
        patch("examples.coding_agent_rl.generate.HARNESS_CLS", mock_harness_cls),
        patch("examples.coding_agent_rl.generate.swe", mock_swe),
        patch("examples.coding_agent_rl.generate._session_id", return_value="sess-123"),
        patch("examples.coding_agent_rl.generate.get_prompt", return_value="prompt"),
        patch("examples.coding_agent_rl.generate._abort_result", side_effect=lambda s, r, i: [r]),
        patch.object(
            gen_mod,
            "CONFIG",
            replace(gen_mod.CONFIG, retry_policy="always-fail", boot_retries=2, rollout_retries=2),
        ),
        patch("examples.coding_agent_rl.generate.asyncio.sleep", side_effect=fast_sleep),
    ):
        result = await generate(MagicMock(), _make_sample_mock(), {})

    # Provisioning attempted exactly once: boot_retries did NOT loop under always-fail.
    assert fake_e2b_cls.call_count == 1
    assert fake_sb.__aenter__.await_count == 1
    assert result == ["exception:SandboxException"]
