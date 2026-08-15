# Relink fresh OpenHands SDK source into a prebuilt oh-env image.
#
# This is the Docker counterpart to tools/repackage_oh_env.py. Takes a base
# oh-env image (built by tools/oh-env-source-image.Dockerfile or
# tools/oh-env-image.Dockerfile), replaces the SDK source at
# /opt/oh-env/src/software-agent-sdk with fresh source from the build context,
# and produces a new image. Because the SDK packages are editable-installed, the
# paths recorded in site-packages point to /opt/oh-env/src/... — swapping the
# source WITHOUT reinstalling is sufficient; the new code is imported the moment
# the prefix is mounted.
#
# Use this after editing SDK code to ship an updated image without rebuilding the
# venv or re-resolving third-party deps (which is expensive). Rebuild the full
# oh-env image (via oh-env-source-image.Dockerfile) only when third-party
# DEPENDENCIES change.
#
# Build from the REPO ROOT so the SDK source under thirdparty/ is in context:
#   DOCKER_BUILDKIT=1 docker build \
#     -f tools/oh-env-relink-image.Dockerfile \
#     --build-arg BASE_IMAGE=oh-env:latest \
#     -t oh-env:dev .
#
# Override the SDK source via --build-arg if needed:
#   --build-arg OH_SDK_SRC=path/to/software-agent-sdk   (relative to context)

# syntax=docker/dockerfile:1.7

# Base oh-env image whose SDK source we'll replace. Must be a valid oh-env image
# (rootfs is the prefix contents: /bin/python, /lib/, /src/, ...).
ARG BASE_IMAGE=oh-env:latest

# ---- stage 0: materialize base image as a named stage --------------------------
FROM ${BASE_IMAGE} AS base

# ---- stage 1: replace the SDK source -------------------------------------------
# The base oh-env image is scratch-derived (no /bin/sh), so we can't RUN in it.
# Copy its rootfs into a debian builder, swap the SDK source there, then verify.
FROM debian:bookworm-slim AS relink
ARG OH_SDK_SRC=thirdparty/benchmarks-main/vendor/software-agent-sdk

# Copy the base oh-env rootfs (prefix contents at /) into /rootfs.
COPY --from=base / /rootfs

# Remove old SDK source and bind-mount fresh source from the build context.
# The editable install tracks /opt/oh-env/src/..., so we replace /rootfs/src/...
# (which becomes /src/... when repacked, then /opt/oh-env/src/... when mounted).
RUN --mount=type=bind,source=.,target=/ctx,ro \
    rm -rf /rootfs/src/software-agent-sdk && \
    mkdir -p /rootfs/src/software-agent-sdk && \
    for pkg in openhands-sdk openhands-tools openhands-workspace openhands-agent-server; do \
        test -d "/ctx/${OH_SDK_SRC}/${pkg}" || { echo "error: missing package: ${pkg}" >&2; exit 1; }; \
        cp -a "/ctx/${OH_SDK_SRC}/${pkg}" "/rootfs/src/software-agent-sdk/${pkg}"; \
    done && \
    for f in pyproject.toml uv.lock README.md LICENSE; do \
        if [ -e "/ctx/${OH_SDK_SRC}/${f}" ]; then \
            cp -a "/ctx/${OH_SDK_SRC}/${f}" "/rootfs/src/software-agent-sdk/${f}"; \
        fi; \
    done

# Verify the SDK imports with the fresh source (same check the harness runs).
# Symlink /opt/oh-env -> /rootfs so python (at /rootfs/bin/python) can resolve
# the editable paths (/opt/oh-env/src/...).
RUN mkdir -p /opt && \
    ln -sfn /rootfs /opt/oh-env && \
    /rootfs/bin/python -c 'import openhands.sdk; import openhands.tools; print("import OK with fresh source")'

# ---- stage 2: pack the result into a fresh scratch image --------------------
FROM scratch
COPY --from=relink /rootfs/ /
