# Build a scratch image whose entire rootfs IS the OpenHands env prefix.
#
# The oh-env tarball (built by tools/build_oh_env.sh) packs everything under the
# fixed prefix path `opt/oh-env/...`. This image strips that prefix so the
# contents land at the image root: `opt/oh-env/bin/python` -> `/bin/python`.
#
# The result is meant to be consumed as an *image volume* (containerd / k8s
# `image` volume, or `docker run -v <img>:...`) mounted at /opt/oh-env inside
# the sandbox -- so the rootfs contents reappear at the fixed prefix the harness
# (slime/agent/harness/openhands.py) expects, with no big tarball to ship or
# unpack at sandbox boot.
#
# Build (BuildKit required -- default on modern Docker):
#   DOCKER_BUILDKIT=1 docker build \
#     -f tools/oh-env-image.Dockerfile \
#     --build-arg OH_ENV_TARBALL=oh-env.tar \
#     -t oh-env:latest .
#
# OH_ENV_TARBALL is a path RELATIVE TO THE BUILD CONTEXT (the final `.` above).
# Run the build from the dir that holds the tarball, or point the context there.

# syntax=docker/dockerfile:1.7

# ---- stage 1: extract the prefix contents out of the tarball ----------------
# GNU tar (debian) for reliable --strip-components; a bind mount keeps the
# multi-hundred-MB tarball out of the image layers entirely.
FROM debian:bookworm-slim AS extract
ARG OH_ENV_TARBALL=oh-env.tar
RUN --mount=type=bind,source=.,target=/ctx,ro \
    mkdir -p /rootfs && \
    tar xf "/ctx/${OH_ENV_TARBALL}" -C /rootfs --strip-components=2 opt/oh-env && \
    test -x /rootfs/bin/python

# ---- stage 2: pack the extracted contents into a bare scratch image ---------
FROM scratch
COPY --from=extract /rootfs/ /
