"""Unit tests for per-sample protocol detection (``swe.detect_protocol`` and the
``"auto"`` dispatch in ``swe.get_metadata``).

These let a single dataset interleave scaleswe and swebench rows: with
``SWE_TRAIN_PROTOCOL=auto`` / ``SWE_EVAL_PROTOCOL=auto`` each row is routed to
its matching metadata parser and grader from its own shape.
"""

import json
import sys
from pathlib import Path

import pytest

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.coding_agent_rl import swe  # noqa: E402
from slime.utils.types import Sample  # noqa: E402


def _sample(metadata: dict) -> Sample:
    s = Sample.__new__(Sample)
    s.metadata = metadata
    s.prompt = ""
    s.label = None
    return s


def test_detect_scaleswe_from_eval_cmd():
    assert swe.detect_protocol(_sample({"eval_cmd": "pytest -q"})) == swe.PROTOCOL_SCALESWE


def test_detect_scaleswe_from_swepro():
    assert swe.detect_protocol(_sample({"swepro": {"before_repo_set_cmd": "x"}})) == swe.PROTOCOL_SCALESWE


def test_detect_scaleswe_from_f2p_script():
    md = {"remote_env_info": {"f2p_script": "import sys; sys.exit(0)"}}
    assert swe.detect_protocol(_sample(md)) == swe.PROTOCOL_SCALESWE


def test_detect_swebench_from_test_patch():
    md = {"remote_env_info": {"test_patch": "diff --git a b"}}
    assert swe.detect_protocol(_sample(md)) == swe.PROTOCOL_SWEBENCH


def test_scaleswe_wins_over_test_patch():
    """A row carrying both a scaleswe grader field and a swebench test_patch is
    scaleswe: the scaleswe grader fields are the stronger, more specific signal."""
    md = {"eval_cmd": "pytest", "remote_env_info": {"test_patch": "diff"}}
    assert swe.detect_protocol(_sample(md)) == swe.PROTOCOL_SCALESWE


def test_empty_row_defaults_to_scaleswe():
    """No signal → scaleswe, so the row is rejected by scaleswe's evaluability
    check rather than mis-routed into the swebench grader."""
    assert swe.detect_protocol(_sample({})) == swe.PROTOCOL_SCALESWE
    assert swe.detect_protocol(_sample({"remote_env_info": {"test_patch": "   "}})) == swe.PROTOCOL_SCALESWE


def test_auto_dispatches_to_swebench_metadata():
    md = swe.get_metadata(
        _sample(
            {
                "remote_env_info": {
                    "test_patch": "diff",
                    "repo": "org/repo",
                    "base_commit": "abc123",
                    "instance_id": "org__repo-1",
                }
            }
        ),
        protocol=swe.PROTOCOL_AUTO,
    )
    assert md["protocol"] == swe.PROTOCOL_SWEBENCH
    assert md["grading"]["sweb_instance"]["repo"] == "org/repo"


def test_auto_dispatches_to_scaleswe_metadata():
    md = swe.get_metadata(
        _sample({"eval_cmd": "pytest", "image": "img:1", "workdir": "/w"}), protocol=swe.PROTOCOL_AUTO
    )
    assert md["protocol"] == swe.PROTOCOL_SCALESWE
    assert md["grading"]["eval_cmd"] == "pytest"


def test_explicit_protocol_still_honored():
    """A fixed protocol overrides shape: forcing swebench on a scaleswe-looking
    row routes to the swebench parser (back-compat with non-auto runs)."""
    md = swe.get_metadata(_sample({"eval_cmd": "pytest"}), protocol=swe.PROTOCOL_SWEBENCH)
    assert md["protocol"] == swe.PROTOCOL_SWEBENCH


# ---------------------------------------------------------------------------
# convert_scaleswe_data: merged output + slime-path verify_load
# ---------------------------------------------------------------------------
from examples.coding_agent_rl import convert_scaleswe_data as conv  # noqa: E402


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_merged_interleaves_both_grader_types(tmp_path):
    """--merged writes one file mixing f2p_script (scaleswe/remote_env_info) and
    f2p_patch (scaleswe/eval_cmd) rows; a graderless row is skipped."""
    src = _write(
        tmp_path / "raw.jsonl",
        [
            {"instance_id": "s1", "image_url": "i:1", "workdir": "/w", "problem_statement": "a", "f2p_script": "x"},
            {
                "instance_id": "p1",
                "image_url": "i:2",
                "workdir": "/w",
                "problem_statement": "b",
                "f2p_patch": "diff",
                "FAIL_TO_PASS": '["t1"]',
                "PASS_TO_PASS": '["t2"]',
            },
            {"instance_id": "skip1", "image_url": "i:3", "workdir": "/w", "problem_statement": "c"},
        ],
    )
    out = str(tmp_path / "out.jsonl")
    n_script, n_patch, n_skip = conv.convert_merged(src, out)
    assert (n_script, n_patch, n_skip) == (1, 1, 1)

    rows = [json.loads(line) for line in Path(out).read_text().splitlines()]
    assert len(rows) == 2
    by_id = {r["label"]: r["metadata"] for r in rows}
    assert by_id["s1"]["remote_env_info"]["f2p_script"] == "x"
    assert "eval_cmd" in by_id["p1"]


def test_verify_load_routes_mixed_scaleswe_and_swebench(tmp_path):
    """A file mixing an admissible scaleswe row and an admissible swebench row
    loads row-by-row and each routes to its own protocol under auto -- exactly
    what slime does at train time. (swebench evaluability is excused when the
    swebench package is absent on the prep host.)"""
    path = Path(
        _write(
            tmp_path / "mixed.jsonl",
            [
                {
                    "prompt": [{"role": "user", "content": "a"}],
                    "label": "s1",
                    "metadata": {"image": "i:1", "workdir": "/w", "eval_cmd": "pytest"},
                },
                {
                    "prompt": [{"role": "user", "content": "b"}],
                    "label": "sb1",
                    "metadata": {
                        "instance_id": "sb1",
                        "remote_env_info": {
                            "image": "i:2",
                            "workdir": "/testbed",
                            "repo": "o/r",
                            "base_commit": "c",
                            "test_patch": "diff",
                        },
                    },
                },
            ],
        )
    )
    assert conv.verify_load(str(path)) == (2, 0)


def test_verify_load_flags_missing_image_workdir(tmp_path):
    """A scaleswe row with a grader but no image/workdir would abort at rollout
    time (generate.py's missing_image_or_workdir gate); verify_load counts it bad
    rather than silently passing it (regression guard for a tautological check)."""
    path = Path(
        _write(
            tmp_path / "thin.jsonl",
            [{"prompt": [{"role": "user", "content": "a"}], "label": "s1", "metadata": {"eval_cmd": "pytest"}}],
        )
    )
    assert conv.verify_load(str(path)) == (0, 1)


def test_verify_load_skips_malformed_json_like_slime(tmp_path):
    """slime's read_file skips a malformed line and continues; verify_load mirrors
    that -- the bad line is counted, not fatal, and a valid line still passes."""
    path = tmp_path / "broken.jsonl"
    path.write_text(
        "{not json}\n"
        + json.dumps(
            {
                "prompt": [{"role": "user", "content": "a"}],
                "label": "s1",
                "metadata": {"image": "i", "workdir": "/w", "eval_cmd": "pytest"},
            }
        )
        + "\n"
    )
    assert conv.verify_load(str(path)) == (1, 1)


if __name__ == "__main__":
    pytest.main([__file__])
