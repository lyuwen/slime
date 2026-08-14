# Template Prompt Support

This document describes how to use Jinja2 templates to customize the agent prompt in the coding_agent_rl example.

## Overview

The `generate.py` module now supports rendering agent prompts from Jinja2 templates via the `get_prompt()` function. This allows you to customize the prompt structure without modifying the hardcoded `SWE_PROMPT`.

## Usage

Set the `SWE_PROMPT_TEMPLATE_PATH` environment variable to the path of your Jinja2 template file:

```bash
export SWE_PROMPT_TEMPLATE_PATH=/path/to/your/template.j2
```

If this variable is not set or the template file is not found, the system falls back to the default `swe.SWE_PROMPT`.

## Template Context

Your template receives two context variables:

### `instance` (dict)
Contains metadata about the coding task:
- `problem_statement`: The task description
- `repo_path`: Repository path in the sandbox (alias for `workdir`)
- `workdir`: Repository path in the sandbox
- `instance_id`: Unique identifier for this instance
- `image`: Docker image name

### `md` (dict)
The raw metadata dictionary from `swe.get_metadata()`, containing all the same fields as `instance` plus any additional metadata.

## Example Template

```jinja2
Task: {{ instance.problem_statement }}

Working Directory: {{ instance.workdir }}
Instance ID: {{ instance.instance_id }}
Image: {{ instance.image }}

Please resolve the issue described above.
```

## Error Handling

The template rendering system is designed to be resilient:

- **Template file not found**: Falls back to `swe.SWE_PROMPT` and logs a warning
- **Template syntax error**: Falls back to `swe.SWE_PROMPT` and logs an error with details
- **Rendering error**: Falls back to `swe.SWE_PROMPT` and logs an error with details

All errors are logged but do not interrupt the rollout process.

## Testing

The template support is tested in `test_generate_prompt.py` with the following test cases:

1. `test_get_prompt_fallback_when_no_template_path`: Verifies fallback when env var is not set
2. `test_get_prompt_with_template`: Tests successful template rendering with metadata
3. `test_get_prompt_template_not_found`: Tests fallback when template file doesn't exist
4. `test_get_prompt_template_render_error`: Tests fallback when template has syntax errors
5. `test_get_prompt_with_legacy_template`: Integration test with `thirdparty/legacy.j2`

Run the tests:

```bash
python examples/coding_agent_rl/test_generate_prompt.py
```

## Implementation Details

The `get_prompt()` function:
- Checks for `SWE_PROMPT_TEMPLATE_PATH` environment variable
- Loads the template using Jinja2's `FileSystemLoader`
- Builds context with both `instance` (structured) and `md` (raw) dicts
- Renders the template and returns the result
- Falls back to `swe.SWE_PROMPT` on any error

This approach maintains backward compatibility while enabling flexible prompt customization for different evaluation protocols or experimental setups.
