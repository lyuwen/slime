"""CPU-only tests for the evaluation sandbox retry mechanism.

Covers:
  §1  is_fresh_sandbox_retryable() exception classification
  §2  run_evaluation() retry loop orchestration (swe.py)
  §3  SweConfig env-var parsing (generate.py)
  §4  Derived rollout guard formula
  §5  Log messages emitted during retry
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.agent import sandbox as sandbox_mod  # noqa: E402


# ---------------------------------------------------------------------------
# §1  is_fresh_sandbox_retryable predicate
# ---------------------------------------------------------------------------


def _exc(name: str, msg: str = "") -> BaseException:
    """Build a fake exception whose __class__.__name__ == name."""
    cls = type(name, (Exception,), {})
    return cls(msg)


def test_fresh_retryable_transport_errors():
    """All HTTP/transport errors that _is_transient_rpc_error recognises are
    also fresh-sandbox retryable."""
    transport_names = [
        "ReadError",
        "WriteError",
        "ConnectError",
        "ProtocolError",
        "LocalProtocolError",
        "RemoteProtocolError",
        "SSLError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "TimeoutException",
    ]
    for name in transport_names:
        e = _exc(name)
        assert sandbox_mod.is_fresh_sandbox_retryable(e), f"{name} should be fresh-retryable"


def test_fresh_retryable_sandbox_stopped():
    """SandboxException with 'STOPPED state' IS fresh-sandbox retryable even
    though it is NOT same-sandbox retryable."""
    e = _exc("SandboxException", "sandbox abc123 in STOPPED state")
    assert sandbox_mod.is_fresh_sandbox_retryable(e)


def test_fresh_retryable_sandbox_not_exist():
    """SandboxException with 'does not exist' IS fresh-sandbox retryable."""
    e = _exc("SandboxException", "sandbox xyz does not exist")
    assert sandbox_mod.is_fresh_sandbox_retryable(e)


def test_not_fresh_retryable_key_error():
    """KeyError (programmer error) is never retried."""
    assert not sandbox_mod.is_fresh_sandbox_retryable(KeyError("missing_key"))


def test_not_fresh_retryable_json_decode():
    """JSONDecodeError is never retried."""
    import json

    e = json.JSONDecodeError("msg", "", 0)
    assert not sandbox_mod.is_fresh_sandbox_retryable(e)


def test_not_fresh_retryable_cancelled():
    """asyncio.CancelledError is never retried."""
    assert not sandbox_mod.is_fresh_sandbox_retryable(asyncio.CancelledError())


def test_not_fresh_retryable_timeout():
    """asyncio.TimeoutError is never retried."""
    assert not sandbox_mod.is_fresh_sandbox_retryable(asyncio.TimeoutError())


def test_not_fresh_retryable_generic_sandbox_exception():
    """A SandboxException without stopped/missing text is NOT fresh-retryable
    (unknown error — preserve existing behaviour)."""
    e = _exc("SandboxException", "quota exceeded")
    # quota exceeded is NOT a fresh-retryable infra failure
    # (sandbox still exists, retrying won't help)
    assert not sandbox_mod.is_fresh_sandbox_retryable(e)


def test_same_sandbox_stopped_not_retried():
    """Confirm the existing same-sandbox loop still does NOT retry STOPPED."""
    sb = sandbox_mod.E2BSandbox.__new__(sandbox_mod.E2BSandbox)
    e = _exc("SandboxException", "sandbox in STOPPED state")
    assert not sb._is_transient_rpc_error(e)
