# Railway image for the zenfleet-dash control plane.
# Build context is the repo root (workspace). Three stages: build the shadcn SPA (node), build the
# dashboard binary (rust), assemble a slim runtime that serves both. Runtime ships s5cmd because
# zenfleet-ledger reads s3:// ledgers by shelling out to it.

# ---- 1. shadcn SPA (Vite + React + Tailwind v4) ----
FROM node:22-slim AS web
WORKDIR /web
# Install deps first (cached unless lockfile changes), then build the static bundle.
COPY crates/zenfleet-dash/web/package.json crates/zenfleet-dash/web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY crates/zenfleet-dash/web/ ./
RUN npm run build

# ---- 2. dashboard binary ----
FROM rust:1-slim-bookworm AS build
WORKDIR /app
# arrow/parquet (zstd-sys, lz4) need a C toolchain + cmake at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config && \
    rm -rf /var/lib/apt/lists/*
COPY . .
# Build against a trimmed workspace (only the deploy crates) — the full workspace has a member with a
# path dep on the jxl-encoder sibling repo, which isn't in the deploy context.
RUN cp Cargo.deploy.toml Cargo.toml && rm -f Cargo.lock && \
    cargo build --release -p zenfleet-dash --bin zenfleet-dash

# ---- 3. runtime ----
FROM debian:bookworm-slim
# ca-certificates for HTTPS to R2 / Hetzner; s5cmd is the s3:// transport zenfleet-ledger invokes.
# tar is already present (essential) — do NOT install/purge it (that breaks apt's resolver).
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && \
    curl -fsSL https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz \
      | tar -xz -C /usr/local/bin s5cmd && \
    rm -rf /var/lib/apt/lists/*
COPY --from=build /app/target/release/zenfleet-dash /usr/local/bin/zenfleet-dash
COPY --from=web /web/dist /app/web
ENV PORT=3000 ZEN_WEB_DIR=/app/web
EXPOSE 3000
CMD ["zenfleet-dash"]
