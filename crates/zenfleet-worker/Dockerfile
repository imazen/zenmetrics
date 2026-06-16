# Baked, ready-to-run fleet worker image (goal H). Per the bake-everything rule: zenfleet-worker +
# zenfleet-gc + aws-cli v2 + s5cmd are all in the image, so a fleet box does ZERO boot-time installs —
# the entrypoint just claims work off the R2 queue. (The 2026-05-30 ad-hoc test hit two bugs this
# image designs out: python-unzip dropping the aws exec bit, and a container with no keep-alive.)
#
# MULTI-ARCH (amd64 + arm64): the base images are already multi-arch and the build stage compiles
# natively per arch; the only arch-specific bits are the s5cmd + aws-cli downloads, selected by the
# buildx-provided TARGETARCH below. arm64 is what lets the named Oracle ARM (free) and Hetzner cax ARM
# tiers join the fleet. The CI workflow builds each arch on its own native runner (no QEMU).
#
# Build context = repo root. Builds from the trimmed Cargo.deploy.toml (no sibling/CUDA deps).

FROM rust:1-slim-bookworm AS build
WORKDIR /app
# arrow/parquet (zstd-sys, lz4) need a C toolchain + cmake.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config && rm -rf /var/lib/apt/lists/*
COPY . .
RUN cp Cargo.deploy.toml Cargo.toml && rm -f Cargo.lock && \
    cargo build --release -p zenfleet-worker --bin zenfleet-worker --bin zenfleet-gc

FROM debian:bookworm-slim
# Bake the S3 tooling the worker shells out to. unzip is used ONCE here (preserving the aws exec bit —
# the thing python-unzip got wrong) and removed; tar is essential (do not purge it).
# TARGETARCH is set by buildx (amd64 / arm64) so the same Dockerfile produces a native image on each
# arch of the multi-arch manifest — required for the named Oracle ARM (free) + Hetzner cax ARM tiers.
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl unzip && \
    case "$TARGETARCH" in \
      amd64) S5_ARCH="Linux-64bit"; AWS_ARCH="x86_64" ;; \
      arm64) S5_ARCH="Linux-arm64"; AWS_ARCH="aarch64" ;; \
      *)     echo "unsupported TARGETARCH=$TARGETARCH" >&2; exit 1 ;; \
    esac && \
    curl -fsSL "https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_${S5_ARCH}.tar.gz" \
      | tar -xz -C /usr/local/bin s5cmd && \
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${AWS_ARCH}.zip" -o /tmp/aws.zip && \
    unzip -q /tmp/aws.zip -d /tmp && /tmp/aws/install && rm -rf /tmp/aws.zip /tmp/aws && \
    apt-get purge -y unzip && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
COPY --from=build /app/target/release/zenfleet-worker /usr/local/bin/zenfleet-worker
COPY --from=build /app/target/release/zenfleet-gc /usr/local/bin/zenfleet-gc
COPY --chmod=0755 crates/zenfleet-worker/fleet-entrypoint.sh /usr/local/bin/fleet-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/fleet-entrypoint.sh"]
