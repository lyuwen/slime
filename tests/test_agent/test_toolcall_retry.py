"""Unit tests for tool-call validation & regeneration in the OpenAI adapter."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_agent._fakes import FakeTokenizer  # noqa: E402

from slime.agent.adapters import common, openai  # noqa: E402


def test_open_session_defaults_is_eval_false_and_accepts_flag():
    adapter = openai.OpenAIAdapter(tokenizer=FakeTokenizer(), sglang_url="http://x")
    adapter.open_session("s-train")
    adapter.open_session("s-eval", is_eval=True)
    assert adapter.store["s-train"].is_eval is False
    assert adapter.store["s-eval"].is_eval is True


if __name__ == "__main__":
    pytest.main([__file__])
