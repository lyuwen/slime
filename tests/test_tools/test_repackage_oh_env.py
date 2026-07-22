"""Unit test for tools/repackage_oh_env.py relink logic (tiny tar fixtures)."""

from __future__ import annotations

import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import repackage_oh_env  # noqa: E402


def _make_env_tar(path: Path, sdk_marker: str) -> None:
    root = path.parent / "envroot"
    src = root / "opt/oh-env/src/software-agent-sdk/openhands-sdk"
    src.mkdir(parents=True)
    (src / "marker.txt").write_text(sdk_marker)
    (root / "opt/oh-env/bin").mkdir(parents=True)
    (root / "opt/oh-env/bin/python").write_text("#!interp\n")
    with tarfile.open(path, "w") as tf:
        tf.add(root / "opt", arcname="opt")


def test_relink_replaces_sdk_source(tmp_path):
    env_tar = tmp_path / "oh-env.tar"
    _make_env_tar(env_tar, sdk_marker="OLD")

    new_src = tmp_path / "software-agent-sdk"
    (new_src / "openhands-sdk").mkdir(parents=True)
    (new_src / "openhands-sdk" / "marker.txt").write_text("NEW")

    out_tar = tmp_path / "oh-env.relinked.tar"
    repackage_oh_env.relink(env_tar, new_src, out_tar)

    extract = tmp_path / "check"
    with tarfile.open(out_tar) as tf:
        tf.extractall(extract)
    marker = extract / "opt/oh-env/src/software-agent-sdk/openhands-sdk/marker.txt"
    assert marker.read_text() == "NEW"
    # the interpreter (untouched non-src content) survives the repackage
    assert (extract / "opt/oh-env/bin/python").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
