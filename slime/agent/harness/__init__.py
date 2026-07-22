"""Swappable coding-agent harnesses (Claude Code, Codex, OpenHands, ...)."""

from __future__ import annotations

from .claude_code import ClaudeCodeHarness
from .codex import CodexHarness
from .common import BaseHarness, HarnessContext
from .openhands import OpenHandsHarness

__all__ = [
    "BaseHarness",
    "HarnessContext",
    "ClaudeCodeHarness",
    "CodexHarness",
    "OpenHandsHarness",
]
