#!/usr/bin/env python3
"""Convert a raw scaleswe dump into slime coding-agent training JSONL.

Input: the raw scaleswe rows (one JSON object per line) with flat fields:
``instance_id, image_url, workdir, problem_statement, pre_commands`` plus one of
two grader payloads:

  * ``f2p_script`` -- a self-contained pytest file ending in
    ``sys.exit(pytest.main([...]))``. Graded directly by swe.py's
    ``_run_f2p_script`` (exit 0 == solved).
  * ``f2p_patch`` + ``FAIL_TO_PASS`` / ``PASS_TO_PASS`` -- a test-adding diff and
    the node-id lists. swe.py has no native path for this shape, so we synthesize
    a ``metadata.eval_cmd`` that mirrors AweAgent's scaleswe judge
    (``BeyondSWEEvaluator._eval_beyondswe``): restore test files -> apply
    ``f2p_patch`` -> run merged F2P+P2P via pytest (exit 0 == solved).

Output: standard slime rows (see examples/coding_agent_rl/README.md "Dataset
Format"). By default they are split into two files by grader type:

  * ``<out>.f2pscript.jsonl`` -- rows carrying a native f2p_script.
  * ``<out>.f2ppatch.jsonl``  -- rows graded by the synthesized eval_cmd.

Both are the ``scaleswe`` protocol; the split is purely by which grader a row
uses. With ``--merged`` the two grader types are interleaved into a single
``<out>.jsonl`` instead -- the natural input for a mixed-protocol run
(``SWE_TRAIN_PROTOCOL=auto``), which detects each row's grader per-sample and no
longer needs the files pre-split. ``prompt`` is emitted as a chat-message list
(``[{"role":"user",...}]``) because slime's data loader requires list-form
prompts whenever the checkpoint ships an AutoProcessor (a bare string trips an
assert in slime/utils/data.py).

``--verify-load`` reloads each written file the way slime actually loads it --
per-line ``json.loads`` with each row's ``metadata`` taken as an independent dict
(``slime/utils/data.py::read_file``) -- and confirms every row routes to a
concrete protocol under ``auto`` (see ``swe.detect_protocol``) carrying the
fields ``generate.py`` reads. Note this is *not* ``datasets.load_dataset``: slime
never unifies one Arrow schema across rows, so a mixed scaleswe/swebench file
that ``load_dataset`` would reject on a nested type clash still trains fine here.
scaleswe and swebench keep their grader payloads under disjoint ``metadata`` /
``metadata.remote_env_info`` keys, so per-sample routing stays unambiguous.

Wire the outputs up with:
    --input-key prompt --label-key label --metadata-key metadata

Usage:
    python -m examples.coding_agent_rl.convert_scaleswe_data \
        scale-swe-021-rl-sample913-ecs.jsonl \
        --out-prefix scale-swe-021-rl
"""

from __future__ import annotations

import argparse
import base64
import json
import sys


def _base_metadata(row: dict, problem: str, instance_id: str) -> dict:
    return {
        "instance_id": instance_id,
        "image": row.get("image_url"),
        "workdir": row.get("workdir"),
        "problem_statement": problem,
        "pre_commands": row.get("pre_commands"),
    }


def _synth_eval_cmd(row: dict) -> str:
    """Build an eval_cmd mirroring AweAgent BeyondSWEEvaluator._eval_beyondswe.

    Steps (as a single shell string run by swe.py's ``_run_eval_cmd`` under
    ``cd {workdir} && ...`` as the agent user):

      1. Restore test files to HEAD (undo any agent test tampering). Non-gating.
      2. Apply ``f2p_patch`` (the diff introducing the failing tests). Gating.
         Carried base64 to sidestep all shell quoting of the diff body.
      3. Run merged FAIL_TO_PASS + PASS_TO_PASS via pytest. pytest's exit code
         carries pass/fail: exit 0 == every selected test passed == solved.
    """
    f2p = json.loads(row["FAIL_TO_PASS"])
    p2p = json.loads(row.get("PASS_TO_PASS") or "[]")
    ids = list(dict.fromkeys([*f2p, *p2p]))  # F2P first, dedup, preserve order
    quoted = " ".join("'" + i.replace("'", "'\\''") + "'" for i in ids)
    b64 = base64.b64encode(row["f2p_patch"].encode()).decode()
    apply_ladder = (
        "( git apply --verbose /tmp/__f2p_patch__.diff "
        "|| git apply --verbose --reject /tmp/__f2p_patch__.diff "
        "|| patch --batch --fuzz=5 -p1 -i /tmp/__f2p_patch__.diff )"
    )
    return (
        "git checkout HEAD -- tests/ test/ Test/ Tests/ 2>/dev/null || true; "
        "git checkout HEAD -- "
        "$(git ls-files '**/test_*.py' '**/*_test.py' '**/conftest.py' 2>/dev/null) "
        "2>/dev/null || true; "
        f"printf %s '{b64}' | base64 -d > /tmp/__f2p_patch__.diff && "
        f"{apply_ladder} && "
        f"python -m pytest -vv -o addopts= --rootdir=. -p no:cacheprovider {quoted}"
    )


def _build_row(row: dict) -> tuple[dict | None, str]:
    """Turn one raw scaleswe row into a slime row, tagged by grader.

    Returns ``(slime_row, kind)`` where ``kind`` is ``"script"`` or ``"patch"``;
    a row with no usable grader yields ``(None, "skip")``.
    """
    instance_id = row.get("instance_id") or "unknown"
    problem = row.get("problem_statement") or ""
    # prompt as chat-message list: slime's loader requires list-form when
    # a processor is present (AutoProcessor loads non-None for VL-capable
    # checkpoints); a bare string trips an assert in slime/utils/data.py.
    prompt = [{"role": "user", "content": problem}]
    md = _base_metadata(row, problem, instance_id)

    if row.get("f2p_script"):
        md["remote_env_info"] = {"f2p_script": row["f2p_script"]}
        kind = "script"
    elif row.get("f2p_patch") and row.get("FAIL_TO_PASS"):
        md["eval_cmd"] = _synth_eval_cmd(row)
        kind = "patch"
    else:
        return None, "skip"

    return {"prompt": prompt, "label": instance_id, "metadata": md}, kind


def convert(src: str, out_script: str, out_patch: str) -> tuple[int, int, int]:
    n_script = n_patch = n_skip = 0
    with open(src) as fin, open(out_script, "w") as fscript, open(out_patch, "w") as fpatch:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            slime_row, kind = _build_row(row)
            if slime_row is None:
                print(
                    f"[skip] {row.get('instance_id') or 'unknown'}: " "no f2p_script and no f2p_patch+FAIL_TO_PASS",
                    file=sys.stderr,
                )
                n_skip += 1
                continue
            out = fscript if kind == "script" else fpatch
            out.write(json.dumps(slime_row, ensure_ascii=False) + "\n")
            n_script += kind == "script"
            n_patch += kind == "patch"
    return n_script, n_patch, n_skip


def convert_merged(src: str, out_merged: str) -> tuple[int, int, int]:
    """Like :func:`convert` but interleaves both grader types into one file."""
    n_script = n_patch = n_skip = 0
    with open(src) as fin, open(out_merged, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            slime_row, kind = _build_row(row)
            if slime_row is None:
                print(
                    f"[skip] {row.get('instance_id') or 'unknown'}: " "no f2p_script and no f2p_patch+FAIL_TO_PASS",
                    file=sys.stderr,
                )
                n_skip += 1
                continue
            fout.write(json.dumps(slime_row, ensure_ascii=False) + "\n")
            n_script += kind == "script"
            n_patch += kind == "patch"
    return n_script, n_patch, n_skip


def verify_load(path: str) -> int:
    """Reload ``path`` the way slime does and prove every row is trainable;
    return the row count. Raises on the first row that fails to route.

    slime reads JSONL per line (``slime/utils/data.py::read_file`` -> per-line
    ``json.loads``) and hands each row's ``metadata`` to ``Sample`` untouched --
    there is no cross-row schema unification, so heterogeneous metadata across
    scaleswe and swebench rows is fine (unlike ``datasets.load_dataset``, which
    unifies one Arrow schema and would reject a nested type clash slime never
    sees). The meaningful check is therefore per-row: does each row route to a
    concrete protocol under ``auto`` and carry the fields ``generate.py`` reads?
    """
    from slime.utils.types import Sample

    from . import swe

    n = 0
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)  # mirrors read_file's jsonl reader
            sample = Sample.__new__(Sample)
            sample.metadata = data.get("metadata") or {}
            sample.prompt = data.get("prompt")
            sample.label = data.get("label")
            md = swe.get_metadata(sample, swe.PROTOCOL_AUTO)
            if md["protocol"] not in (swe.PROTOCOL_SCALESWE, swe.PROTOCOL_SWEBENCH):
                raise ValueError(f"{path}:{line_num}: row did not route to a concrete protocol: {md['protocol']!r}")
            if not md.get("instance_id") or "grading" not in md:
                raise ValueError(
                    f"{path}:{line_num}: routed row missing instance_id/grading: {md.get('instance_id')!r}"
                )
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", help="raw scaleswe JSONL dump")
    p.add_argument(
        "--out-prefix",
        default=None,
        help="output prefix; writes <prefix>.f2pscript.jsonl and <prefix>.f2ppatch.jsonl "
        "(or <prefix>.jsonl with --merged) (default: derived from src filename)",
    )
    p.add_argument(
        "--merged",
        action="store_true",
        help="interleave both grader types into a single <prefix>.jsonl instead of "
        "splitting by grader (the natural input for SWE_TRAIN_PROTOCOL=auto)",
    )
    p.add_argument(
        "--verify-load",
        action="store_true",
        help="reload each written file the way slime does (per-line json, per-row "
        "metadata) and confirm every row routes to a concrete protocol under `auto`",
    )
    args = p.parse_args(argv)

    prefix = args.out_prefix or args.src.rsplit(".jsonl", 1)[0]

    if args.merged:
        out_merged = f"{prefix}.jsonl"
        n_script, n_patch, n_skip = convert_merged(args.src, out_merged)
        outputs = [out_merged]
        print(f"merged         : {n_script + n_patch:5d} rows -> {out_merged}")
        print(f"  f2p_script   : {n_script:5d} rows")
        print(f"  f2p_patch    : {n_patch:5d} rows")
    else:
        out_script = f"{prefix}.f2pscript.jsonl"
        out_patch = f"{prefix}.f2ppatch.jsonl"
        n_script, n_patch, n_skip = convert(args.src, out_script, out_patch)
        outputs = [out_script, out_patch]
        print(f"f2p_script set : {n_script:5d} rows -> {out_script}")
        print(f"f2p_patch  set : {n_patch:5d} rows -> {out_patch}")

    if n_skip:
        print(f"skipped        : {n_skip:5d} rows (no usable grader)")
    print(f"total written  : {n_script + n_patch:5d}")

    if args.verify_load:
        for path in outputs:
            n = verify_load(path)
            print(f"verify-load    : {n:5d} rows load cleanly <- {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
