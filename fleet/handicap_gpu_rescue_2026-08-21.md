# gpu_metric handicap derivation — the avifgen rescue clean window (2026-08-21)

Replaces the seeded 1.0/1.0 `gpu_metric` rows (r7900x reference-card seed + node-2 G-Z3
parity seed) with values measured from the only clean (post-hint) concurrent GPU-scoring
window: run `jobs/avifgen-sf-gpu-rescue-20260808` (proportional per-MP ResourceHints,
admission ≤2 GPU jobs/box — the OOM-storm window is excluded by construction because the
whole run postdates the hints).

## Method

- Source: the run's full ledger (10,298 sidecars pulled to
  `~/tmp/avifgen/rescue_ledger_final/`; R2 listing with mtimes saved as
  `~/tmp/avifgen/rescue_ledger_listing_final.txt`). Ledger `ts` is pass-quantized, so
  activity timing comes from sidecar upload mtimes (one sidecar ≈ one 300 s chunk).
- Unit = done ScoreFile-job rows per sidecar, bucketed per 5 minutes per worker.
- Two bases, both computed 2026-08-21:
  1. **When-active throughput**: done rows / active 5-min buckets.
  2. **Same-queue concurrent basis** (the registration basis, matching the `metric`
     rows' same-window convention): buckets where BOTH boxes uploaded ≥1 sidecar.

## Measured

| worker (card at the time) | active buckets | done rows | when-active rate | concurrent-window rows (9 shared buckets) |
|---|---|---|---|---|
| lianli → r7900x (RTX 2080 8GB) | 137 | 17,634 | ~1,545 jobs/hr | 1,016 |
| node-2 → i134 (RTX 3070 8GB) | 62 | 3,406 | ~659 jobs/hr | 616 |

Ratio (concurrent, same queue): **3070 = 0.61 × 2080** in this workload — host-stall
dominated (fresh-process CUDA context/JIT per job; node-2's dmon in the status doc showed
the worst stalls), so this is a BOX number, not a card number. Confound noted honestly:
the boxes' non-overlapping windows carried different job mixes (first rescue declare vs
union); only the 9 concurrent buckets are mix-matched, hence they are the basis.

Also recorded: node-2 exited the rescue early via repeated `status=137` (SIGKILL) worker
deaths on 2026-08-08; r7900x carried the drain until scoped-cred expiry 2026-08-11
09:37Z. The 2026-08-21 revival residue (262 union jobs) drained on node-2/i134 alone.

## Registration outcome

- `r7900x.gpu_metric = 0.0` — EXCLUDED: the measured card (RTX 2080) left the box
  (~2026-08-1x, to the dev node); current GTX 1060 6GB is unmeasured + not enrolled.
- `node-2.gpu_metric = 1.0` and the key-migrated `i134.gpu_metric = 1.0` — the sole
  enrolled GPU scorer becomes the column reference; the 0.61 ratio above re-anchors the
  column when a second measured card enrolls.
