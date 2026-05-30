# Baked, ready-to-run fleet worker image (goal H). Per the bake-everything rule: zen-jobworker +
# zen-jobgc + aws-cli v2 + s5cmd are all in the image, so a fleet box does ZERO boot-time installs —
# the entrypoint just claims work off the R2 queue. (The 2026-05-30 ad-hoc test hit two bugs this
# image designs out: python-unzip dropping the aws exec bit, and a container with no keep-alive.)
#
# Build context = repo root. Builds from the trimmed Cargo.deploy.toml (no sibling/CUDA deps).

FROM rust:1-slim-bookworm AS build
WORKDIR /app
# arrow/parquet (zstd-sys, lz4) need a C toolchain + cmake.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config && rm -rf /var/lib/apt/lists/*
COPY . .
RUN cp Cargo.deploy.toml Cargo.toml && rm -f Cargo.lock && \
    cargo build --release -p zen-jobworker --bin zen-jobworker --bin zen-jobgc

FROM debian:bookworm-slim
# Bake the S3 tooling the worker shells out to. unzip is used ONCE here (preserving the aws exec bit —
# the thing python-unzip got wrong) and removed; tar is essential (do not purge it).
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl unzip && \
    curl -fsSL https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz \
      | tar -xz -C /usr/local/bin s5cmd && \
    curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/aws.zip && \
    unzip -q /tmp/aws.zip -d /tmp && /tmp/aws/install && rm -rf /tmp/aws.zip /tmp/aws && \
    apt-get purge -y unzip && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
COPY --from=build /app/target/release/zen-jobworker /usr/local/bin/zen-jobworker
COPY --from=build /app/target/release/zen-jobgc /usr/local/bin/zen-jobgc
COPY --chmod=0755 crates/zen-jobworker/fleet-entrypoint.sh /usr/local/bin/fleet-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/fleet-entrypoint.sh"]
