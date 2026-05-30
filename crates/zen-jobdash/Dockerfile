# Railway image for the zen-jobdash control plane.
# Build context is the repo root (workspace); we build only the dashboard crate.
# Runtime ships s5cmd because zen-ledger reads s3:// ledgers by shelling out to it.

FROM rust:1-slim-bookworm AS build
WORKDIR /app
# arrow/parquet (zstd-sys, lz4) need a C toolchain + cmake at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config && \
    rm -rf /var/lib/apt/lists/*
COPY . .
# Build against a trimmed workspace (only the 3 dashboard crates) — the full workspace has a member
# with a path dep on the jxl-encoder sibling repo, which isn't in the deploy context.
RUN cp Cargo.deploy.toml Cargo.toml && rm -f Cargo.lock && \
    cargo build --release -p zen-jobdash --bin zen-jobdash

FROM debian:bookworm-slim
# ca-certificates for HTTPS to R2; s5cmd is the s3:// transport zen-ledger invokes.
# tar is already present (essential) — do NOT install/purge it (that breaks apt's resolver).
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && \
    curl -fsSL https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz \
      | tar -xz -C /usr/local/bin s5cmd && \
    rm -rf /var/lib/apt/lists/*
COPY --from=build /app/target/release/zen-jobdash /usr/local/bin/zen-jobdash
ENV PORT=3000
EXPOSE 3000
CMD ["zen-jobdash"]
