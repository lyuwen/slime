"""Relink fresh OpenHands SDK source into a prebuilt env tarball.

The env tarball is built once with the SDK editable-installed from an in-prefix
source path (/opt/oh-env/src/software-agent-sdk). Because an editable install
tracks the PATH, not the content, swapping the source needs no reinstall: unpack
the env, replace the source dir, re-tar. Use this after editing SDK code to ship
a new tarball without rebuilding the venv or re-resolving deps.

Rebuild the full env (not this tool) only when third-party DEPENDENCIES change.
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path

_DEFAULT_PREFIX_SRC = "opt/oh-env/src/software-agent-sdk"


def relink(env_tar: Path, sdk_src: Path, out_tar: Path, *, prefix_src: str = _DEFAULT_PREFIX_SRC) -> None:
    """Unpack env_tar, replace <prefix_src> with sdk_src, re-tar to out_tar."""
    env_tar = Path(env_tar)
    sdk_src = Path(sdk_src)
    out_tar = Path(out_tar)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        with tarfile.open(env_tar) as tf:
            tf.extractall(tmp, filter="data")
        dst = tmp / prefix_src
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(sdk_src, dst)
        # re-tar the top-level 'opt' tree so it unpacks to the fixed prefix again
        top = prefix_src.split("/")[0]
        with tarfile.open(out_tar, "w") as tf:
            tf.add(tmp / top, arcname=top)


def main() -> None:
    p = argparse.ArgumentParser(description="Relink SDK source into an OpenHands env tarball.")
    p.add_argument("--env-tarball", required=True, type=Path)
    p.add_argument("--sdk-src", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--prefix-src", default=_DEFAULT_PREFIX_SRC)
    args = p.parse_args()
    relink(args.env_tarball, args.sdk_src, args.out, prefix_src=args.prefix_src)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
