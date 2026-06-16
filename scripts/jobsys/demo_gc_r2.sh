#!/usr/bin/env bash
# Live demo of safe garbage collection (goal G) against R2. Runs the self-asserting gc_live example:
# puts a referenced blob + two cheap-regenerable blobs (old/new) + an irreplaceable orphan, then runs
# gc_execute with a cap that fits only the newest cheap blob and verifies the safety guarantees.
#
#   - referenced blob: KEPT (reachability GC never deletes referenced)
#   - cheap LRU tail (oldest): EVICTED with a tombstone (bounded cache, lossless rebuild)
#   - newest cheap blob: KEPT (fits the cap)
#   - unreferenced irreplaceable: REFUSED — surfaced, never auto-deleted
#
# Also exposes the `zenfleet-gc` CLI for real use against a Parquet blob index + ledger (dry-run default).
# Requires: R2_* env, s5cmd, cargo.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export AWS_REGION=auto AWS_DEFAULT_REGION=auto
cargo run -q -p zenfleet-worker --example gc_live
