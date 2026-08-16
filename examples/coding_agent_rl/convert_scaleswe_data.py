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
per-line ``json.loads`` that *skips* a malformed line and continues, with each
row's ``metadata`` taken as an independent dict (``slime/utils/data.py::
read_file``) -- and reports any row that would abort at rollout time, mirroring
``generate.py``'s gate: missing ``image``/``workdir`` or a failing
``evaluability_check`` (``swebench_import_failed`` is excused as an environment,
not data, condition). It exits non-zero if any row is inadmissible. Note this is
*not* ``datasets.load_dataset``: slime never unifies one Arrow schema across rows,
so a mixed scaleswe/swebench file that ``load_dataset`` would reject on a nested
type clash still trains fine here. scaleswe and swebench keep their grader
payloads under disjoint ``metadata`` / ``metadata.remote_env_info`` keys, so
per-sample routing stays unambiguous.

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


def _convert_rows(src: str, sink) -> tuple[int, int, int]:
    """Shared conversion loop: parse each raw row, build the slime row, skip
    graderless rows, count by kind. ``sink(kind, slime_row)`` writes the row to
    wherever the caller wants (split files or one merged file)."""
    n_script = n_patch = n_skip = 0
    with open(src) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            slime_row, kind = _build_row(row)
            if slime_row is None:
                print(
                    f"[skip] {row.get('instance_id') or 'unknown'}: no f2p_script and no f2p_patch+FAIL_TO_PASS",
                    file=sys.stderr,
                )
                n_skip += 1
                continue
            sink(kind, slime_row)
            n_script += kind == "script"
            n_patch += kind == "patch"
    return n_script, n_patch, n_skip


def convert(src: str, out_script: str, out_patch: str) -> tuple[int, int, int]:
    with open(out_script, "w") as fscript, open(out_patch, "w") as fpatch:

        def sink(kind: str, slime_row: dict) -> None:
            out = fscript if kind == "script" else fpatch
            out.write(json.dumps(slime_row, ensure_ascii=False) + "\n")

        return _convert_rows(src, sink)


def convert_merged(src: str, out_merged: str) -> tuple[int, int, int]:
    """Like :func:`convert` but interleaves both grader types into one file."""
    with open(out_merged, "w") as fout:

        def sink(_kind: str, slime_row: dict) -> None:
            fout.write(json.dumps(slime_row, ensure_ascii=False) + "\n")

        return _convert_rows(src, sink)


def verify_load(path: str) -> tuple[int, int]:
    """Reload ``path`` the way slime does and confirm each row is admissible;
    return ``(n_ok, n_bad)``. Does not raise for data defects -- it reports them.

    slime reads JSONL per line (``slime/utils/data.py::read_file`` -> per-line
    ``json.loads``), *skips* a malformed line and continues, and hands each row's
    ``metadata`` to ``Sample`` untouched -- there is no cross-row schema
    unification, so heterogeneous scaleswe/swebench metadata is fine (unlike
    ``datasets.load_dataset``, which unifies one Arrow schema and would reject a
    nested type clash slime never sees). We mirror that: malformed lines are
    counted as bad and skipped, not fatal.

    "Admissible" mirrors ``generate.py``'s per-sample gate: the row must resolve
    a concrete protocol under ``auto``, carry a usable ``image``/``workdir``, and
    pass ``swe.evaluability_check`` -- i.e. exactly the rows that would NOT abort
    at rollout time. The one exception is ``swebench_import_failed``, an
    environment condition (no ``swebench`` installed on the prep host), not a data
    defect, so it does not fail verification.
    """
    from slime.utils.types import Sample

    from . import swe

    n_ok = n_bad = 0
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)  # mirrors read_file's jsonl reader
            except json.JSONDecodeError as e:
                print(f"[bad] {path}:{line_num}: malformed JSON: {e}", file=sys.stderr)
                n_bad += 1
                continue
            reason = _admission_reason(swe, Sample, data)
            if reason is None:
                n_ok += 1
            else:
                label = data.get("label") or (data.get("metadata") or {}).get("instance_id") or "unknown"
                print(f"[bad] {path}:{line_num}: {label}: {reason}", file=sys.stderr)
                n_bad += 1
    return n_ok, n_bad


def _admission_reason(swe, sample_cls, data: dict) -> str | None:
    """Return why ``data`` would abort at rollout time, or ``None`` if admissible.

    Mirrors the gate in ``generate.py`` (missing image/workdir, then
    ``evaluability_check``) so verify-load flags exactly the rows a run would drop
    -- with ``swebench_import_failed`` excused as an environment, not data, issue.
    """
    sample = sample_cls.__new__(sample_cls)
    sample.metadata = data.get("metadata") or {}
    sample.prompt = data.get("prompt")
    sample.label = data.get("label")
    md = swe.get_metadata(sample, swe.PROTOCOL_AUTO)
    if not md.get("image") or not md.get("workdir"):
        return "missing_image_or_workdir"
    reason = swe.evaluability_check(md)
    if reason and not reason.startswith("swebench_import_failed"):
        return f"unevaluatable:{reason}"
    return None


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
        help="reload each written file the way slime does (per-line json, skipping "
        "malformed lines) and report rows that would abort at rollout time "
        "(missing image/workdir or failing evaluability_check); exit 1 if any",
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
        total_bad = 0
        for path in outputs:
            n_ok, n_bad = verify_load(path)
            total_bad += n_bad
            status = f"{n_ok:5d} admissible" + (f", {n_bad} bad" if n_bad else "")
            print(f"verify-load    : {status} <- {path}")
        if total_bad:
            print(f"verify-load    : FAILED, {total_bad} inadmissible row(s)", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
