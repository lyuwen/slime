import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.coding_agent_rl.generate import get_prompt  # noqa: E402


def test_get_prompt_fallback_when_no_template_path():
    """When SWE_PROMPT_TEMPLATE_PATH is not set, should return swe.SWE_PROMPT"""
    # Ensure env var is not set
    os.environ.pop("SWE_PROMPT_TEMPLATE_PATH", None)

    md = {
        "problem_statement": "Fix the bug",
        "workdir": "/workspace/repo",
        "instance_id": "test-123",
        "image": "test:latest",
        "protocol": "scaleswe",
    }

    result = get_prompt(md)

    # Should contain the default prompt text
    assert "Read PROBLEM_STATEMENT.md" in result or "resolve the issue" in result.lower()


def test_get_prompt_with_template():
    """When template path is set, should render template with md context"""
    # Create a temporary template file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".j2", delete=False) as f:
        f.write("Task: {{ instance.problem_statement }}\nWorkdir: {{ instance.repo_path }}")
        template_path = f.name

    try:
        os.environ["SWE_PROMPT_TEMPLATE_PATH"] = template_path

        md = {
            "problem_statement": "Fix the authentication bug",
            "workdir": "/workspace/myrepo",
            "instance_id": "auth-bug-001",
            "image": "myimage:v1",
            "protocol": "scaleswe",
        }

        result = get_prompt(md)

        assert "Task: Fix the authentication bug" in result
        assert "Workdir: /workspace/myrepo" in result
    finally:
        os.environ.pop("SWE_PROMPT_TEMPLATE_PATH", None)
        Path(template_path).unlink(missing_ok=True)


def test_get_prompt_template_not_found():
    """When template file doesn't exist, should fall back to swe.SWE_PROMPT"""
    os.environ["SWE_PROMPT_TEMPLATE_PATH"] = "/nonexistent/template.j2"

    md = {
        "problem_statement": "Fix the bug",
        "workdir": "/workspace/repo",
        "instance_id": "test-456",
        "image": "test:latest",
        "protocol": "scaleswe",
    }

    result = get_prompt(md)

    # Should fall back to default prompt
    assert "Read PROBLEM_STATEMENT.md" in result or "resolve the issue" in result.lower()
    os.environ.pop("SWE_PROMPT_TEMPLATE_PATH", None)


def test_get_prompt_template_render_error():
    """When template has syntax error, should fall back to swe.SWE_PROMPT"""
    # Create a template with invalid Jinja2 syntax
    with tempfile.NamedTemporaryFile(mode="w", suffix=".j2", delete=False) as f:
        f.write("{{ instance.problem_statement")  # Missing closing }}
        template_path = f.name

    try:
        os.environ["SWE_PROMPT_TEMPLATE_PATH"] = template_path

        md = {
            "problem_statement": "Fix the bug",
            "workdir": "/workspace/repo",
            "instance_id": "test-789",
            "image": "test:latest",
            "protocol": "scaleswe",
        }

        result = get_prompt(md)

        # Should fall back to default prompt
        assert "Read PROBLEM_STATEMENT.md" in result or "resolve the issue" in result.lower()
    finally:
        os.environ.pop("SWE_PROMPT_TEMPLATE_PATH", None)
        Path(template_path).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
