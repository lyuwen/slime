# OpenHands Environment Image Build Methods

Three Dockerfiles produce the same output contract: a `scratch` image whose rootfs contains the OpenHands SDK environment (python-build-standalone CPython 3.12 + the 4 editable-installed OpenHands SDK packages + third-party deps), meant to be mounted as an image volume at `/opt/oh-env` in the agent sandbox container.

## 1. Tarball-based build (`tools/oh-env-image.Dockerfile`)

**Use when:** You already have a prebuilt tarball from `tools/build_oh_env.sh`.

**What it does:** Extracts the tarball, strips the `opt/oh-env` prefix so the contents land at the image root, and packs into a `scratch` image.

**Build:**
```bash
# First, build the tarball on the host (requires root or writable /opt/oh-env):
tools/build_oh_env.sh \
  --sdk-src thirdparty/benchmarks-main/vendor/software-agent-sdk \
  --out oh-env.tar

# Then, build the image (run from the directory containing oh-env.tar):
DOCKER_BUILDKIT=1 docker build \
  -f tools/oh-env-image.Dockerfile \
  --build-arg OH_ENV_TARBALL=oh-env.tar \
  -t oh-env:latest .
```

**Pros:** The tarball can be reused across machines; `tools/repackage_oh_env.py` can swap SDK source without rebuilding deps.

**Cons:** Requires a separate host-side build step; needs root or a writable prefix.

---

## 2. From-scratch build (`tools/oh-env-source-image.Dockerfile`)

**Use when:** You want a single Docker command to build the full environment from source, without prebuilt artifacts.

**What it does:** Fetches python-build-standalone from astral-sh, bind-mounts the SDK source from the build context, editable-installs the 4 packages + third-party deps at the real `/opt/oh-env` prefix inside the builder, verifies the SDK imports, then copies the prefix contents to a `scratch` image root.

**Build (from repo root):**
```bash
DOCKER_BUILDKIT=1 docker build \
  -f tools/oh-env-source-image.Dockerfile \
  -t oh-env:latest .
```

**Build args (all optional, with defaults):**
- `PYTHON_URL` — python-build-standalone CPython 3.12 tarball URL (default: pinned 20250808 x86_64-linux-gnu build from astral-sh/python-build-standalone)
- `PIP_INDEX_URL` — PyPI mirror (default: empty, uses pip's default)
- `OH_SDK_SRC` — SDK source dir relative to build context (default: `thirdparty/benchmarks-main/vendor/software-agent-sdk`)
- `PREFIX` — fixed install prefix (default: `/opt/oh-env`)

**Override example (local mirrors):**
```bash
DOCKER_BUILDKIT=1 docker build \
  -f tools/oh-env-source-image.Dockerfile \
  --build-arg PYTHON_URL='http://10.15.64.50:8000/cpython-3.12.11%2B20250808-x86_64-unknown-linux-gnu-install_only.tar.gz' \
  --build-arg PIP_INDEX_URL='https://mirrors.ustc.edu.cn/pypi/simple' \
  -t oh-env:latest .
```

**Pros:** Single command; no host-side prefix pollution; reproducible from source; BuildKit caches the python fetch + dep resolution layers.

**Cons:** SDK source is baked into image layers at build time — to update SDK code you rebuild the image (cached layers make it ~fast, but not instant).

---

## 3. SDK source relink (`tools/oh-env-relink-image.Dockerfile`)

**Use when:** You have an existing oh-env image (from method 1 or 2) and want to swap in fresh SDK source WITHOUT reinstalling dependencies.

**What it does:** Takes a base oh-env image, removes the old SDK source at `/src/software-agent-sdk`, bind-mounts fresh source from the build context to copy in the 4 packages + workspace files, verifies the SDK imports with the new source, and packs into a fresh `scratch` image. Because the SDK is editable-installed, no `pip install` is needed — the recorded paths point to `/opt/oh-env/src/...`, so swapping the source is sufficient.

**Build (from repo root):**
```bash
DOCKER_BUILDKIT=1 docker build \
  -f tools/oh-env-relink-image.Dockerfile \
  --build-arg BASE_IMAGE=oh-env:latest \
  -t oh-env:dev .
```

**Build args:**
- `BASE_IMAGE` — base oh-env image to relink (required; default: `oh-env:latest`)
- `OH_SDK_SRC` — SDK source dir relative to build context (default: `thirdparty/benchmarks-main/vendor/software-agent-sdk`)

**Pros:** Fast SDK iteration (no dep reinstall); useful for testing local SDK changes.

**Cons:** Only swaps SDK source — if third-party dependencies change, rebuild with method 1 or 2.

---

## Workflow recommendations

- **Initial setup / dependency changes:** Use method 2 (`oh-env-source-image.Dockerfile`) to build the full environment from source. BuildKit caches the expensive layers (python fetch, pip install deps).
  
- **SDK iteration during development:** Use method 3 (`oh-env-relink-image.Dockerfile`) to swap SDK source into an existing base image. Much faster than rebuilding deps.

- **Tarball-based CI/production builds:** Use method 1 (`oh-env-image.Dockerfile`) when you have a prebuilt tarball artifact that can be versioned and reused across environments.

---

## Output contract (all methods)

The resulting image is a `scratch` image whose rootfs contains:
- `/bin/python` → `python3.12` (from python-build-standalone)
- `/lib/` — Python stdlib + third-party packages
- `/src/software-agent-sdk/` — the 4 editable-installed OpenHands packages
- Editable-install paths recorded as absolute `/opt/oh-env/src/...` paths

**Mount the image at `/opt/oh-env` in the sandbox container:**
```bash
# containerd / k8s image volume
docker run --rm -v oh-env-img:/opt/oh-env:ro <sandbox-image> /opt/oh-env/bin/python -c 'import openhands.sdk'

# docker volume from image
docker run --rm -v <oh-env-image>:/opt/oh-env:ro <sandbox-image> /opt/oh-env/bin/python -c 'import openhands.sdk'
```

The harness (`slime/agent/harness/openhands.py`) expects the mount at `/opt/oh-env` — the editable-install paths depend on this fixed prefix.
