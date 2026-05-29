# task142 — reserve_staging submit_blocking cost measurement

**Date:** 2026-05-29
**Host:** RTX 5070 (12 GB), CUDA 13.2, AMD Ryzen 9 7950X, native (not docker)
**zenmetrics commit:** eb57605d (bench) -> next (findings)
**zenforks-cubecl-work commit:** d73c5b3a
**Bench source:** `crates/butteraugli-gpu/examples/bench_staging_block.rs`
**Command:** `cargo run --release -p butteraugli-gpu --features cuda --example bench_staging_block`
**Grid:** 4096x4096 (16 MP), N=8 back-to-back uploads, u32-packed pinned staging = 67,108,864 bytes/upload

## Context

cubecl maintainer's PR #1334 review: "submit_blocking is expensive and
shouldn't be used for data transfer, only for data fetching" + "the new
reserve staging doesn't seem to be implemented at the right level."

Our `reserve_staging` (zenforks-cubecl-runtime client.rs) wraps
`server.staging(...)` in `submit_blocking`. butteraugli-gpu uses it in
production (pipeline.rs:782/1332/1657) to pack sRGB u8x3 -> pinned u32
host memory before the device upload, inside a pipelined orchestrator.

## Results (two representative runs; medians shown)

| Measurement | run1 | run2 |
|---|---|---|
| (A) reserve_staging round-trip, idle runner (mean) | 0.009 ms | 0.006 ms |
| (E) reserve_staging round-trip, loaded runner (mean) | 0.317 ms | 0.186 ms |
| (B) pinned pack+create per upload (caller-thread, mean) | 21.0 ms | ~11 ms |
| (B) pinned 8-upload wall, post-sync | 170.7 ms | 93.4 ms |
| (C) create_from_slice (pageable->pinned, double memcpy) per upload | 101 ms | ~107 ms |
| (C) create_from_slice 8-upload wall, post-sync | 870 ms | 916 ms |
| (D) pipelined pinned upload+compute_handles | 47.2 ms/pair | 43.5 ms/pair |
| (F) async-probe (reserve+pack+upload on runner) per call | 0.008 ms | n/a |
| (F) async-probe 8-upload wall, post-sync | 920 ms | 885 ms |
| (G) pipelined async-probe upload+compute_handles | 82.9 ms/pair | 69.7 ms/pair |

(Run-to-run variance reflects thermal/turbo + first-touch page mapping on
the 64 MB pinned buffers; the *relative* ordering is stable.)

## Findings

1. **The submit_blocking round-trip is cheap, not a GPU stall.**
   `ComputeServer::staging` -> CUDA `reserve_cpu`/`reserve_pinned` is a
   pure host-side pinned-memory-pool allocation. It never touches the
   CUDA stream, never syncs the GPU. The block is a CPU-thread
   round-trip to the device-runner thread: 0.006-0.32 ms even when the
   runner is loaded with queued submit() closures. That is <=2% of the
   ~12-21 ms per-upload cost and <1% of the 43-47 ms/pair pipelined cost.

2. **The pinned win is real and large (~6.5x), and it is the whole point.**
   (B) pinned path = ~12-21 ms/upload vs (C) standard create_from_slice =
   ~100 ms/upload. The slice path does TWO host memcpys (caller &[u8] ->
   pageable Vec via to_vec(), then staging copies pageable->pinned). The
   pinned helper packs once directly into pinned. Removing reserve_staging
   would forfeit this.

3. **The maintainer's "right level" async alternative is WORSE for us.**
   (F)/(G) prototype the literal suggestion: reserve + pack + upload all
   on the runner thread via one non-blocking submit() (probe method
   create_from_slice_pinned_async_probe). The caller-thread per-call time
   drops to 0.008 ms (no block) — but end-to-end it is ~5x slower (F) and
   pipelined throughput regresses 1.6-1.8x (G: 70-83 ms/pair vs D:
   43-47 ms/pair). Reason: the heavy ~10 ms u8x3->u32 pack now runs on the
   shared runner thread and serializes against kernel-launch dispatch.
   The current design (heavy pack on the caller thread, block only for the
   cheap pinned reservation) is correct for a pipelined orchestrator that
   wants the runner thread free for compute dispatch.

## Decision

**Do NOT change the fork.** The maintainer's general rule is sound, but
for our heavy-pack image/codec pipeline the actual block cost is
negligible and the "right level" async path measurably regresses
throughput. Keep `reserve_staging` + caller-thread pack. The probe method
stays as a doc-hidden experiment in the fork (not exported / not relied on).

**Data point for the maintainer:** "create_from_slice is mostly used for
testing" does NOT hold for image/codec pipelines. Our production input is
`&[u8]` sRGB bytes from a decoder, so the slice path IS the production
upload path. The pinned variant exists precisely to make that production
path fast.
