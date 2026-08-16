#!/usr/bin/env bash
# Build the prebuilt OpenHands env tarball (SLIME_AGENT_OH_ENV_TARBALL).
#
# Produces a single self-contained tarball that unpacks with a plain `tar x`
# (no python/npm/pip/network egress at sandbox boot). It bundles a
# python-build-standalone CPython 3.12 at the fixed prefix /opt/oh-env with the
# 4 OpenHands SDK packages editable-installed from an in-prefix source path.
#
# The editable install records the PATH (/opt/oh-env/src/software-agent-sdk),
# not the content, so the tarball is valid the instant it is unpacked back to
# /opt/oh-env. To swap fresh SDK source WITHOUT a full rebuild, use
# tools/repackage_oh_env.py instead; rebuild with this script only when the
# third-party DEPENDENCIES change.
#
# Usage:
#   tools/build_oh_env.sh \
#     --sdk-src thirdparty/benchmarks-main/vendor/software-agent-sdk \
#     --out     oh-env.tar
#
# Optional:
#   --python-url URL   python-build-standalone CPython 3.12 tarball to fetch.
#                      Defaults to a pinned 3.12 x86_64-linux-gnu build.
#   --prefix PATH      Fixed install prefix (default /opt/oh-env). Must match
#                      the harness (slime/agent/harness/openhands.py) and the
#                      unpack target -- do not change without updating both.
#
# Requires root (writes to /opt/oh-env) OR a writable --prefix, plus curl/tar.
set -euo pipefail

PREFIX="/opt/oh-env"
SDK_SRC=""
OUT=""
# python-build-standalone: self-contained CPython 3.12, bundles its own
# libpython so it is independent of the base image's Python/glibc (scaleswe base
# images span many distros).
PYTHON_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20250808/cpython-3.12.11+20250808-x86_64-unknown-linux-gnu-install_only.tar.gz"

# The 4 workspace packages, in dependency order (sdk first).
PACKAGES=(openhands-sdk openhands-tools openhands-workspace openhands-agent-server)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sdk-src)    SDK_SRC="$2"; shift 2 ;;
    --out)        OUT="$2"; shift 2 ;;
    --python-url) PYTHON_URL="$2"; shift 2 ;;
    --prefix)     PREFIX="$2"; shift 2 ;;
    -h|--help)    sed -n '2,33p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$SDK_SRC" ]] || { echo "error: --sdk-src is required" >&2; exit 2; }
[[ -n "$OUT" ]]     || { echo "error: --out is required" >&2; exit 2; }
[[ -d "$SDK_SRC" ]] || { echo "error: --sdk-src not a directory: $SDK_SRC" >&2; exit 2; }

SDK_SRC="$(cd "$SDK_SRC" && pwd)"
# Resolve OUT to an absolute path before we start creating dirs.
OUT_DIR="$(cd "$(dirname "$OUT")" && pwd)"
OUT="$OUT_DIR/$(basename "$OUT")"

PY="$PREFIX/bin/python"
# In-prefix source path the editable install records. Mirrors the default in
# tools/repackage_oh_env.py (opt/oh-env/src/software-agent-sdk).
PREFIX_SRC="$PREFIX/src/software-agent-sdk"

echo ">> building OpenHands env at prefix $PREFIX"

# A stale prefix would poison the tarball (leftover packages / old source).
if [[ -e "$PREFIX" ]]; then
  echo "error: $PREFIX already exists; remove it before rebuilding" >&2
  exit 1
fi

# 1. Materialize python-build-standalone CPython 3.12 at the fixed prefix.
echo ">> [1/4] fetching python-build-standalone: $PYTHON_URL"
mkdir -p "$PREFIX"
TMP_PY_TAR="$(mktemp)"
trap 'rm -f "$TMP_PY_TAR"' EXIT
curl -fSL "$PYTHON_URL" -o "$TMP_PY_TAR"
# The archive contains a top-level `python/` dir; strip it onto the prefix.
tar xzf "$TMP_PY_TAR" -C "$PREFIX" --strip-components=1
[[ -x "$PY" ]] || { echo "error: $PY missing after extract" >&2; exit 1; }
"$PY" --version

# 2. Place the 4 OpenHands packages as source at the fixed in-prefix path.
echo ">> [2/4] staging SDK source into $PREFIX_SRC"
mkdir -p "$PREFIX_SRC"
for pkg in "${PACKAGES[@]}"; do
  [[ -d "$SDK_SRC/$pkg" ]] || { echo "error: package missing in sdk-src: $pkg" >&2; exit 1; }
  cp -a "$SDK_SRC/$pkg" "$PREFIX_SRC/$pkg"
done
# Root workspace files can matter for consistent metadata; copy if present.
for f in pyproject.toml uv.lock README.md LICENSE; do
  [[ -e "$SDK_SRC/$f" ]] && cp -a "$SDK_SRC/$f" "$PREFIX_SRC/$f" || true
done

# 3. Editable-install the packages FROM the in-prefix path, then their deps.
#    Editable records the path (/opt/oh-env/src/...), valid once unpacked to the
#    fixed prefix. Installing from within PREFIX_SRC keeps the recorded path
#    inside the prefix so it survives the tar round-trip.
echo ">> [3/4] pip install -e the 4 packages + third-party deps"
"$PY" -m pip install --upgrade pip
for pkg in "${PACKAGES[@]}"; do
  echo ">>   pip install -e $PREFIX_SRC/$pkg"
  "$PY" -m pip install -e "$PREFIX_SRC/$pkg"
done

# Verify the SDK imports out of the freshly built prefix before we tar it. This
# is the same check the harness runs at boot (install_cli).
"$PY" -c 'import openhands.sdk; import openhands.tools; print("import OK")'

# 4. Tar the whole prefix so it unpacks back to the fixed prefix.
#    -C / with the prefix relative path => `tar x -C /` restores $PREFIX exactly.
echo ">> [4/4] writing tarball: $OUT"
PREFIX_REL="${PREFIX#/}"   # opt/oh-env
tar cf "$OUT" -C / "$PREFIX_REL"

echo ">> done: $OUT"
echo ">> set  export SLIME_AGENT_OH_ENV_TARBALL=$OUT"
