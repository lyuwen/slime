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
Format"), split into two files by grader type:

  * ``<out>.f2pscript.jsonl`` -- rows carrying a native f2p_script.
  * ``<out>.f2ppatch.jsonl``  -- rows graded by the synthesized eval_cmd.

Both are the ``scaleswe`` protocol; the split is purely by which grader a row
uses. ``prompt`` is emitted as a chat-message list (``[{"role":"user",...}]``)
because slime's data loader requires list-form prompts whenever the checkpoint
ships an AutoProcessor (a bare string trips an assert in slime/utils/data.py).

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


def convert(src: str, out_script: str, out_patch: str) -> tuple[int, int, int]:
    n_script = n_patch = n_skip = 0
    with open(src) as fin, open(out_script, "w") as fscript, open(out_patch, "w") as fpatch:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            instance_id = row.get("instance_id") or "unknown"
            problem = row.get("problem_statement") or ""
            # prompt as chat-message list: slime's loader requires list-form when
            # a processor is present (AutoProcessor loads non-None for VL-capable
            # checkpoints); a bare string trips an assert in slime/utils/data.py.
            prompt = [{"role": "user", "content": problem}]
            md = _base_metadata(row, problem, instance_id)

            if row.get("f2p_script"):
                md["remote_env_info"] = {"f2p_script": row["f2p_script"]}
                out, counter = fscript, "script"
            elif row.get("f2p_patch") and row.get("FAIL_TO_PASS"):
                md["eval_cmd"] = _synth_eval_cmd(row)
                out, counter = fpatch, "patch"
            else:
                # No usable grader -- would always reward 0; drop it.
                print(f"[skip] {instance_id}: no f2p_script and no f2p_patch+FAIL_TO_PASS", file=sys.stderr)
                n_skip += 1
                continue

            out.write(json.dumps({"prompt": prompt, "label": instance_id, "metadata": md}, ensure_ascii=False) + "\n")
            n_script += counter == "script"
            n_patch += counter == "patch"
    return n_script, n_patch, n_skip


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", help="raw scaleswe JSONL dump")
    p.add_argument(
        "--out-prefix",
        default=None,
        help="output prefix; writes <prefix>.f2pscript.jsonl and <prefix>.f2ppatch.jsonl "
        "(default: derived from src filename)",
    )
    args = p.parse_args(argv)

    prefix = args.out_prefix or args.src.rsplit(".jsonl", 1)[0]
    out_script = f"{prefix}.f2pscript.jsonl"
    out_patch = f"{prefix}.f2ppatch.jsonl"

    n_script, n_patch, n_skip = convert(args.src, out_script, out_patch)
    print(f"f2p_script set : {n_script:5d} rows -> {out_script}")
    print(f"f2p_patch  set : {n_patch:5d} rows -> {out_patch}")
    if n_skip:
        print(f"skipped        : {n_skip:5d} rows (no usable grader)")
    print(f"total written  : {n_script + n_patch:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
