# zenfleet-worker base-* tags — the shared bake-everything FOUNDATION.
#
# These are base-* TAGS on the existing ghcr.io/imazen/zenfleet-worker package
# (NOT a separate package — variants are TAGS, see docs/GHCR_PACKAGES.md), built
# from this one multi-stage Dockerfile:
#
#   :base-x86       = target `nocuda`, linux/amd64   (CPU; no CUDA)
#   :base-arm       = target `nocuda`, linux/arm64   (Hetzner CAX / Oracle ARM)
#   :base-x86-cuda  = target `cuda`,   linux/amd64   (CUDA 12.6 runtime + dlsym stub)
#
# Every leaf image (zenmetrics-sweep, zen-train, the exec variants) does
# `FROM ghcr.io/imazen/zenfleet-worker:base-<tag>` and adds ONLY its precompiled
# binary + entrypoint, so the heavy apt/CUDA layers are pulled once per node and
# the leaves are tiny. Per the bake-everything rule: NO apt/pip/cuda at boot, NO
# Rust toolchain here (binaries are built in CI and COPY'd into the leaves).
#
# Built by .github/workflows/base-image.yml on NATIVE runners (no QEMU).
# Build context = repo root (needs scripts/sweep/cuda_dlsym_stub.c for L-cuda).
#
# Layering is cheap-changes-rarely -> expensive-changes-often so the buildkit
# registry cache + ghcr layer dedup keep base pulls warm across rebuilds.

# ─────────────────────────────────────────────────────────────────────────────
# Stage `common`: deps shared by every variant + every arch.
# ─────────────────────────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS common

LABEL org.opencontainers.image.source="https://github.com/imazen/zenmetrics"
LABEL org.opencontainers.image.description="zenfleet-worker base-* tags: shared bake-everything foundation for zen fleet/train images (R2 tooling + python/pyarrow + optional CUDA)"
LABEL org.opencontainers.image.licenses="AGPL-3.0-or-later"

ENV DEBIAN_FRONTEND=noninteractive
# TARGETARCH (amd64 | arm64) is provided by buildx so the same Dockerfile yields
# a native image per arch — required for the ARM tiers.
ARG TARGETARCH

# Pinned helper versions (bump deliberately; do not float).
ARG S5CMD_VERSION=2.2.2
ARG JQ_VERSION=1.7.1

# ── stable apt deps (R2/runtime libs + python + a C toolchain for pip/cuda stub)
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg \
        libssl3 \
        python3 python3-pip \
        gcc libc6-dev \
        tar \
    && rm -rf /var/lib/apt/lists/*

# ── python data-stack baked (pyarrow for parquet IO, numpy for array math).
#    Ubuntu 24.04 python is externally-managed, hence --break-system-packages
#    (matches the proven sweep image). Heavier libs (pandas/sklearn/torch) are
#    a LEAF concern, not baked here.
RUN pip3 install --quiet --break-system-packages --no-cache-dir \
        'pyarrow>=15.0,<22.0' 'numpy>=1.26,<3.0' \
    && python3 -c "import pyarrow.parquet, numpy; print('pyarrow+numpy OK')"

# ── R2 / S3 tooling baked: aws-cli v2 + s5cmd, arch-selected. unzip is used
#    ONCE to install aws (preserving its exec bit) and then purged.
RUN apt-get update -qq && apt-get install -y --no-install-recommends unzip \
    && case "$TARGETARCH" in \
         amd64) S5_ARCH="Linux-64bit"; AWS_ARCH="x86_64";  JQ_ARCH="amd64" ;; \
         arm64) S5_ARCH="Linux-arm64"; AWS_ARCH="aarch64"; JQ_ARCH="arm64" ;; \
         *) echo "unsupported TARGETARCH=$TARGETARCH" >&2; exit 1 ;; \
       esac \
    && curl -fsSL "https://github.com/peak/s5cmd/releases/download/v${S5CMD_VERSION}/s5cmd_${S5CMD_VERSION}_${S5_ARCH}.tar.gz" \
         | tar -xz -C /usr/local/bin s5cmd \
    && chmod +x /usr/local/bin/s5cmd \
    && curl -fsSL "https://github.com/jqlang/jq/releases/download/jq-${JQ_VERSION}/jq-linux-${JQ_ARCH}" \
         -o /usr/local/bin/jq \
    && chmod +x /usr/local/bin/jq \
    && curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${AWS_ARCH}.zip" -o /tmp/aws.zip \
    && unzip -q /tmp/aws.zip -d /tmp && /tmp/aws/install && rm -rf /tmp/aws.zip /tmp/aws \
    && apt-get purge -y unzip && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* \
    && s5cmd version && jq --version && aws --version

ENV PATH=/usr/local/sbin:/usr/sbin:/sbin:/usr/local/bin:/usr/bin:/bin \
    ZENMETRICS_CACHE_DIR=/root/.cache/zenmetrics
RUN mkdir -p /root/.cache/zenmetrics

# ─────────────────────────────────────────────────────────────────────────────
# Stage `nocuda`: the :x86 (amd64) and :arm (arm64) base images. No CUDA.
# ─────────────────────────────────────────────────────────────────────────────
FROM common AS nocuda
RUN echo "zenfleet-worker base nocuda (${TARGETARCH:-?}) ready" \
    && python3 -c "import pyarrow.parquet, numpy" && s5cmd version >/dev/null && jq --version >/dev/null
# Base image: no ENTRYPOINT; leaf images set their own. bash for ad-hoc use.
CMD ["bash"]

# ─────────────────────────────────────────────────────────────────────────────
# Stage `cuda`: the :x86-cuda base image. amd64 only.
# CUDA 12.6 NVRTC + cudart + the cuda_dlsym_stub shim — identical to the proven
# sweep image (Dockerfile.sweep.v27 L4-L7). Binaries that link cubecl-cuda are
# built with CUDARC_CUDA_VERSION=12000 (covers all driver versions) in the LEAF.
# ─────────────────────────────────────────────────────────────────────────────
FROM common AS cuda
ARG TARGETARCH
ARG CUDA_PKG_VERSION=12-6
ARG CUDA_DIR_VERSION=12.6
RUN [ "$TARGETARCH" = "amd64" ] || { echo "zenfleet-worker base:cuda is amd64-only (got $TARGETARCH)" >&2; exit 1; }

RUN export PATH="/usr/local/sbin:/usr/sbin:/sbin:${PATH}" \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
         -o /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb \
    && apt-get update -qq \
    && apt-get install -y --no-install-recommends \
         cuda-nvrtc-${CUDA_PKG_VERSION} cuda-cudart-${CUDA_PKG_VERSION} cuda-cudart-dev-${CUDA_PKG_VERSION} \
    && rm -rf /var/lib/apt/lists/* \
    && echo "/usr/local/cuda-${CUDA_DIR_VERSION}/lib64" > /etc/ld.so.conf.d/cuda.conf \
    && ldconfig \
    && ln -sf /usr/local/cuda-${CUDA_DIR_VERSION} /usr/local/cuda \
    && test -f /usr/local/cuda/include/cuda_runtime.h \
    && test -f /usr/local/cuda/lib64/libcudart.so.12 \
    && ldconfig -p | grep -q libnvrtc.so.12

# cuda_dlsym_stub.so — LD_PRELOAD shim cubecl-cuda needs for CUDA 13.x drivers
# (stubs coredump-callback symbols removed from libcuda.so). Same source the
# sweep image bakes.
COPY scripts/sweep/cuda_dlsym_stub.c /tmp/cuda_dlsym_stub.c
RUN gcc -shared -fPIC -O2 -o /usr/local/lib/cuda_dlsym_stub.so /tmp/cuda_dlsym_stub.c -ldl \
    && rm /tmp/cuda_dlsym_stub.c \
    && test -s /usr/local/lib/cuda_dlsym_stub.so \
    && nm -D --defined-only /usr/local/lib/cuda_dlsym_stub.so 2>/dev/null | grep -q ' T dlsym$' \
    && echo "zenfleet-worker base cuda_dlsym_stub baked"

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    CUDA_PATH=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/sbin:/sbin:/usr/local/bin:/usr/bin:/bin \
    LD_PRELOAD=/usr/local/lib/cuda_dlsym_stub.so
RUN ldconfig -p | grep -q libnvrtc.so.12 \
    && test -f /usr/local/lib/cuda_dlsym_stub.so \
    && python3 -c "import pyarrow.parquet, numpy" \
    && echo "zenfleet-worker base cuda (amd64, CUDA ${CUDA_DIR_VERSION}) ready"
CMD ["bash"]
