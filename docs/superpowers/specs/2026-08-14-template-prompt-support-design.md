# Template Prompt Support for Coding Agent RL

**Date:** 2026-08-14  
**Component:** `examples/coding_agent_rl/generate.py`  
**Goal:** Add Jinja2 template rendering support for agent prompts, mirroring the benchmark harness capability

## Overview

Currently, `examples/coding_agent_rl/generate.py` uses a hardcoded string prompt (`swe.SWE_PROMPT`) to instruct the agent. This change adds optional Jinja2 template rendering so prompts can be customized per-sample using metadata fields like `problem_statement` and `workdir`.

The implementation follows the same pattern as the benchmark harness (`thirdparty/benchmarks-main/benchmarks/scaleswe/run_infer.py`), allowing users to specify a template file path via environment variable while maintaining backward compatibility with the current simple string approach.

## Current vs New Flow

### Current Prompt Flow
```
CONFIG.from_env() 
  → swe.SWE_PROMPT (env or hardcoded string)
  → harness.run(prompt=swe.SWE_PROMPT)
```

### New Prompt Flow
```
CONFIG.from_env()
  → check SWE_PROMPT_TEMPLATE_PATH
  ↓
get_prompt(md)
  ├─ if template_path: 
  │    load template → render with context → return rendered string
  └─ else: 
       return swe.SWE_PROMPT
  ↓
harness.run(prompt=rendered_prompt)
```

## Implementation Design

### New Function: `get_prompt(md: dict) -> str`

Add a helper function to `generate.py` that handles both template rendering and fallback:

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
```

### Template Context Structure

The function builds a context dict that makes existing templates like `thirdparty/legacy.j2` work with minimal changes:

```python
context = {
    "instance": {
        "problem_statement": md.get("problem_statement", ""),
        "repo_path": md.get("workdir", ""),           # alias for legacy.j2
        "workdir": md.get("workdir", ""),             # also keep original
        "instance_id": md.get("instance_id", ""),
        "image": md.get("image", ""),
    },
    "md": md,  # full metadata dict for custom templates
}
```

**Field Mapping:**
- `instance.repo_path` → `md["workdir"]` (alias for compatibility with `legacy.j2`)
- `instance.problem_statement` → `md["problem_statement"]`
- `instance.workdir` → `md["workdir"]` (keep original name too)
- `instance.instance_id` → `md["instance_id"]`
- `instance.image` → `md["image"]`
- `md` → full metadata dict for advanced templates

**Dropped Fields:**
- `base_commit` is not included (no direct mapping from slime metadata structure)

### Integration Point

The function is called in `generate()` after metadata extraction and before harness invocation:

```python
# In generate() function, after this line:
md = swe.get_metadata(base_sample, protocol)
# ... existing checks ...

# NEW: Get prompt with optional template rendering
prompt = get_prompt(md)

agent_exit_code = await HARNESS_CLS().run(
    sb,
    workdir=md["workdir"],
    session_id=session_id,
    adapter_url=state.adapter_url,
    time_budget_sec=CONFIG.agent_time_budget_sec,
    prompt=prompt,  # changed from swe.SWE_PROMPT
    **oh_kwargs,
)
```

## Error Handling

The implementation handles errors gracefully with fallback to maintain rollout stability:

**File Not Found:**
- Log warning: `logger.warning(f"[coding_agent_rl] Template file not found: {template_path}; falling back to SWE_PROMPT")`
- Fall back to `swe.SWE_PROMPT`
- Continue rollout (non-fatal)

**Template Rendering Error:**
- Catch Jinja2 exceptions (undefined variables, syntax errors)
- Log error: `logger.error(f"[coding_agent_rl] Template rendering failed ({template_path}): {type(e).__name__}: {e}; falling back to SWE_PROMPT")`
- Fall back to `swe.SWE_PROMPT`
- Continue rollout (non-fatal)

**Missing Context Data:**
- Jinja2 returns empty string for undefined variables by default
- Common fields (`problem_statement`, `workdir`) are guaranteed by `swe.get_metadata()`
- Templates can safely access any metadata field

**Success Logging:**
- Log on first use: `logger.info(f"[coding_agent_rl] Using template prompt from {template_path}")`

## Configuration

**Environment Variable:** `SWE_PROMPT_TEMPLATE_PATH`

- If set: path to a Jinja2 template file (absolute or relative to working directory)
- If not set or empty: use `swe.SWE_PROMPT` (current behavior)

**Example:**
```bash
export SWE_PROMPT_TEMPLATE_PATH=thirdparty/legacy.j2
```

## Backward Compatibility

- `swe.SWE_PROMPT` continues to work exactly as before when `SWE_PROMPT_TEMPLATE_PATH` is not set
- `swe.prepare_workspace()` still writes `PROBLEM_STATEMENT.md` to the sandbox (harmless redundancy)
- No changes to existing command-line arguments or configuration
- Template errors fall back to current behavior rather than failing the rollout

## Example Usage

Using `thirdparty/legacy.j2`:

```bash
export SWE_PROMPT_TEMPLATE_PATH=thirdparty/legacy.j2
python -m examples.coding_agent_rl.generate ...
```

The template receives:
- `{{ instance.problem_statement }}` → full task description from metadata
- `{{ instance.repo_path }}` → `/workspace/beets` (from `md["workdir"]`)
- `{{ instance.base_commit }}` → empty string (not in metadata)

## Files Changed

- `examples/coding_agent_rl/generate.py`:
  - Add `get_prompt(md: dict) -> str` function
  - Import `jinja2.Environment` and `jinja2.FileSystemLoader` at module level
  - Change `harness.run(prompt=swe.SWE_PROMPT, ...)` to `harness.run(prompt=get_prompt(md), ...)`

## Non-Changes

- `swe.py` remains unchanged (it's the task layer, not the prompt layer)
- `prepare_workspace()` still writes `PROBLEM_STATEMENT.md` (harmless, maintains compatibility)
- No changes to argument parsing or configuration loading
- No changes to harness implementations

## Testing Strategy

1. **Template rendering:** Test `get_prompt()` with a sample `md` dict and verify context structure
2. **Fallback behavior:** Verify that missing template path or render errors fall back to `swe.SWE_PROMPT`
3. **Integration:** Run one SWE sample with a template and verify the rendered prompt reaches the harness
4. **Backward compat:** Run one sample without `SWE_PROMPT_TEMPLATE_PATH` and verify current behavior unchanged
