# Template Prompt Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Jinja2 template rendering support to `examples/coding_agent_rl/generate.py` so prompts can be customized per-sample using metadata fields

**Architecture:** Add a `get_prompt(md: dict) -> str` helper function that checks for `SWE_PROMPT_TEMPLATE_PATH` environment variable, loads and renders Jinja2 templates with instance metadata, or falls back to the existing `swe.SWE_PROMPT` string

**Tech Stack:** Python, Jinja2, existing slime infrastructure

## Global Constraints

- Python 3.8+ compatibility (existing project requirement)
- Follow existing code style: black + isort (line length 119)
- Use existing logger patterns (`logger.info`, `logger.warning`, `logger.error`)
- No changes to argument parsing, configuration, or harness interfaces
- Maintain backward compatibility with `swe.SWE_PROMPT`

---

### Task 1: Add Jinja2 imports and get_prompt function

**Files:**
- Modify: `examples/coding_agent_rl/generate.py:18-41` (imports section)
- Modify: `examples/coding_agent_rl/generate.py:207-248` (generate function)

**Interfaces:**
- Consumes: `md` dict from `swe.get_metadata()` with keys: `problem_statement`, `workdir`, `instance_id`, `image`, `protocol`
- Produces: `get_prompt(md: dict) -> str` function that returns the rendered prompt string

- [ ] **Step 1: Write failing test for get_prompt function**

Create `examples/coding_agent_rl/test_generate_prompt.py`:

```python
import os
import tempfile
from pathlib import Path

import pytest

from examples.coding_agent_rl.generate import get_prompt


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
    with tempfile.NamedTemporaryFile(mode='w', suffix='.j2', delete=False) as f:
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
    with tempfile.NamedTemporaryFile(mode='w', suffix='.j2', delete=False) as f:
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lfu/git-projects/slime/.claude/worktrees/template-prompt-support && python examples/coding_agent_rl/test_generate_prompt.py`

Expected: Import error or NameError for `get_prompt`

- [ ] **Step 3: Add Jinja2 imports to generate.py**

In `examples/coding_agent_rl/generate.py`, add after line 23 (after `import os`):

```python
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, TemplateError
```

- [ ] **Step 4: Implement get_prompt function**

In `examples/coding_agent_rl/generate.py`, add after the `CONFIG` initialization (after line 106) and before `_BOOT_SEM`:

```python
def get_prompt(md: dict) -> str:
    """Generate the agent prompt, optionally using a Jinja2 template.
    
    If SWE_PROMPT_TEMPLATE_PATH is set, loads and renders that template
    with the metadata dict. Otherwise falls back to the hardcoded SWE_PROMPT.
    
    Args:
        md: Metadata dict from swe.get_metadata(), containing:
            - problem_statement: The task description
            - workdir: Repository path in the sandbox
            - instance_id: Unique identifier
            - image: Docker image name
            - protocol: Evaluation protocol
            
    Returns:
        Rendered prompt string to pass to the harness
    """
    template_path_str = os.environ.get("SWE_PROMPT_TEMPLATE_PATH", "").strip()
    
    if not template_path_str:
        return swe.SWE_PROMPT
    
    template_path = Path(template_path_str)
    
    try:
        # Load template
        if not template_path.exists():
            logger.warning(
                "[coding_agent_rl] Template file not found: %s; falling back to SWE_PROMPT",
                template_path,
            )
            return swe.SWE_PROMPT
        
        # Set up Jinja2 environment
        template_dir = template_path.parent
        template_name = template_path.name
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        template = env.get_template(template_name)
        
        # Build context with instance alias for legacy.j2 compatibility
        context = {
            "instance": {
                "problem_statement": md.get("problem_statement", ""),
                "repo_path": md.get("workdir", ""),
                "workdir": md.get("workdir", ""),
                "instance_id": md.get("instance_id", ""),
                "image": md.get("image", ""),
            },
            "md": md,
        }
        
        # Render template
        rendered = template.render(context)
        logger.info("[coding_agent_rl] Using template prompt from %s", template_path)
        return rendered
        
    except TemplateError as e:
        logger.error(
            "[coding_agent_rl] Template rendering failed (%s): %s: %s; falling back to SWE_PROMPT",
            template_path,
            type(e).__name__,
            str(e),
        )
        return swe.SWE_PROMPT
    except Exception as e:
        logger.error(
            "[coding_agent_rl] Unexpected error loading template (%s): %s: %s; falling back to SWE_PROMPT",
            template_path,
            type(e).__name__,
            str(e),
        )
        return swe.SWE_PROMPT
```

- [ ] **Step 5: Update generate() to use get_prompt**

In `examples/coding_agent_rl/generate.py`, replace line 246 from:
```python
                    prompt=swe.SWE_PROMPT,
```

To:
```python
                    prompt=get_prompt(md),
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /home/lfu/git-projects/slime/.claude/worktrees/template-prompt-support && python examples/coding_agent_rl/test_generate_prompt.py`

Expected: All 4 tests pass

- [ ] **Step 7: Run linter to verify code style**

Run: `cd /home/lfu/git-projects/slime/.claude/worktrees/template-prompt-support && pre-commit run --files examples/coding_agent_rl/generate.py examples/coding_agent_rl/test_generate_prompt.py`

Expected: No errors (or auto-fixed formatting)

- [ ] **Step 8: Commit**

```bash
git add examples/coding_agent_rl/generate.py examples/coding_agent_rl/test_generate_prompt.py
git commit -m "feat(coding_agent_rl): add Jinja2 template support for prompts

Add get_prompt() function that checks SWE_PROMPT_TEMPLATE_PATH env var
and renders Jinja2 templates with instance metadata, falling back to
swe.SWE_PROMPT when template is not configured or fails to render.

Template context includes instance.problem_statement, instance.repo_path
(aliased from workdir), instance.workdir, instance.instance_id, and the
full md dict for custom templates.

Co-authored-by: Claude <noreply@anthropic.com>"
```

---

### Task 2: Integration test with legacy.j2 template

**Files:**
- Test: `examples/coding_agent_rl/test_generate_prompt.py` (add integration test)

**Interfaces:**
- Consumes: `get_prompt()` from Task 1, `thirdparty/legacy.j2` template file
- Produces: Integration test confirming legacy.j2 works with generate.py

- [ ] **Step 1: Write integration test**

Add to `examples/coding_agent_rl/test_generate_prompt.py`:

```python
def test_get_prompt_with_legacy_template():
    """Integration test: legacy.j2 should render with slime metadata"""
    # Point to the actual legacy.j2 template
    legacy_template = Path(__file__).parent.parent.parent / "thirdparty" / "legacy.j2"
    
    if not legacy_template.exists():
        pytest.skip(f"legacy.j2 not found at {legacy_template}")
    
    os.environ["SWE_PROMPT_TEMPLATE_PATH"] = str(legacy_template)
    
    try:
        md = {
            "problem_statement": "Migrate testing from nose to pytest",
            "workdir": "/workspace/beets",
            "instance_id": "beetbox_beets_pr3661",
            "image": "registry.example.com/scaleswe:beetbox_beets_pr3661",
            "protocol": "scaleswe",
        }
        
        result = get_prompt(md)
        
        # Check that template rendered with expected content
        assert "Migrate testing from nose to pytest" in result
        assert "/workspace/beets" in result or "uploaded" in result.lower()
        # legacy.j2 has phases like "Phase 1. READING"
        assert "Phase" in result or "phase" in result
    finally:
        os.environ.pop("SWE_PROMPT_TEMPLATE_PATH", None)
```

- [ ] **Step 2: Run integration test**

Run: `cd /home/lfu/git-projects/slime/.claude/worktrees/template-prompt-support && python examples/coding_agent_rl/test_generate_prompt.py`

Expected: All 5 tests pass (4 from Task 1 + 1 new integration test)

- [ ] **Step 3: Commit**

```bash
git add examples/coding_agent_rl/test_generate_prompt.py
git commit -m "test(coding_agent_rl): add integration test for legacy.j2 template

Verify that the actual thirdparty/legacy.j2 template renders correctly
with slime metadata structure.

Co-authored-by: Claude <noreply@anthropic.com>"
```

---

### Task 3: Documentation and final verification

**Files:**
- Create: `examples/coding_agent_rl/README-template-prompts.md`

**Interfaces:**
- Consumes: Completed implementation from Tasks 1-2
- Produces: User-facing documentation

- [ ] **Step 1: Create documentation file**

Create `examples/coding_agent_rl/README-template-prompts.md`:

```markdown
# Template Prompt Support

## Overview

The coding agent RL rollout supports Jinja2 template rendering for prompts via the `SWE_PROMPT_TEMPLATE_PATH` environment variable. This allows you to customize the agent instruction per-sample using metadata fields.

## Usage

### Basic Usage

Set the environment variable to point to your template:

```bash
export SWE_PROMPT_TEMPLATE_PATH=thirdparty/legacy.j2
```

Run your rollout as usual. The template will be loaded and rendered with instance metadata.

### Template Context

Templates receive the following context:

```python
{
    "instance": {
        "problem_statement": str,  # Full task description
        "repo_path": str,          # Alias for workdir (legacy.j2 compat)
        "workdir": str,            # Repository path in sandbox
        "instance_id": str,        # Unique instance identifier
        "image": str,              # Docker image name
    },
    "md": dict,  # Full metadata dict for advanced templates
}
```

### Example Template

```jinja2
I've uploaded a repository at {{ instance.repo_path }}.

<issue_description>
{{ instance.problem_statement }}
</issue_description>

Please resolve the issue described above.
```

### Fallback Behavior

If `SWE_PROMPT_TEMPLATE_PATH` is:
- Not set: uses `swe.SWE_PROMPT` (current default behavior)
- Set but file not found: logs warning, falls back to `swe.SWE_PROMPT`
- Set but template has errors: logs error, falls back to `swe.SWE_PROMPT`

## Compatibility

### With legacy.j2

The `thirdparty/legacy.j2` template references `instance.repo_path` and `instance.base_commit`. The context maps:
- `instance.repo_path` → `md["workdir"]`
- `instance.base_commit` → empty string (not in slime metadata)

### With SWE_PROMPT

The existing `SWE_PROMPT` environment variable (or hardcoded default) continues to work when `SWE_PROMPT_TEMPLATE_PATH` is not set.

## Testing

Run the test suite:

```bash
python examples/coding_agent_rl/test_generate_prompt.py
```

Tests cover:
- Fallback to `SWE_PROMPT` when template path not set
- Template rendering with instance context
- Error handling for missing template files
- Error handling for template syntax errors
- Integration with `legacy.j2` template
```

- [ ] **Step 2: Run full test suite**

Run: `cd /home/lfu/git-projects/slime/.claude/worktrees/template-prompt-support && python examples/coding_agent_rl/test_generate_prompt.py`

Expected: All 5 tests pass

- [ ] **Step 3: Verify no regressions in generate.py**

Run a syntax check:
```bash
cd /home/lfu/git-projects/slime/.claude/worktrees/template-prompt-support
python -m py_compile examples/coding_agent_rl/generate.py
```

Expected: No errors

- [ ] **Step 4: Run pre-commit on all changed files**

Run: `cd /home/lfu/git-projects/slime/.claude/worktrees/template-prompt-support && pre-commit run --files examples/coding_agent_rl/generate.py examples/coding_agent_rl/test_generate_prompt.py examples/coding_agent_rl/README-template-prompts.md`

Expected: No errors

- [ ] **Step 5: Commit documentation**

```bash
git add examples/coding_agent_rl/README-template-prompts.md
git commit -m "docs(coding_agent_rl): add template prompt usage guide

Document SWE_PROMPT_TEMPLATE_PATH environment variable, template context
structure, fallback behavior, and compatibility notes.

Co-authored-by: Claude <noreply@anthropic.com>"
```

- [ ] **Step 6: Final verification - review all commits**

Run: `git log --oneline -5`

Expected: See 4 commits (design spec + 3 implementation commits)

---

## Self-Review Checklist

**Spec coverage:**
- ✓ Template loading via `SWE_PROMPT_TEMPLATE_PATH` (Task 1, Step 4)
- ✓ Jinja2 rendering with metadata context (Task 1, Step 4)
- ✓ Instance field mapping (repo_path alias) (Task 1, Step 4)
- ✓ Fallback to `swe.SWE_PROMPT` (Task 1, Step 4)
- ✓ Error handling for missing files (Task 1, Step 4 + test in Step 1)
- ✓ Error handling for template errors (Task 1, Step 4 + test in Step 1)
- ✓ Integration with `harness.run()` (Task 1, Step 5)
- ✓ Testing strategy (Task 1 + Task 2)
- ✓ Documentation (Task 3)

**Placeholder scan:**
- ✓ No TBD, TODO, or "implement later"
- ✓ All code blocks complete and executable
- ✓ All test assertions specific
- ✓ All file paths exact

**Type consistency:**
- ✓ `get_prompt(md: dict) -> str` signature used consistently
- ✓ `md` dict structure documented and used consistently
- ✓ Return type `str` matches `harness.run(prompt=...)` signature

**Execution readiness:**
- ✓ Each step is independently executable
- ✓ Expected outputs documented for verification
- ✓ Commit messages follow conventional format
- ✓ Pre-commit checks integrated into workflow
