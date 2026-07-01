# Score-many-from-persisted-variants optimization (2026-07-01)

Filling extra perceptual metrics onto a canonical picker dataset from PERSISTED encoded variants
(no re-encode) is the `JobKind::ScoreFile` path (`crates/zenmetrics-cli/src/jobexec.rs::run_score_file`).
This doc records the two optimizations landed here, the real-GPU A/B that validated them, and the
fleet-config recommendation.

## The problem (measured)
`run_score_file` already decodes the reference once + each variant once + scores all metrics on that
one decode. Two remaining costs made it I/O/upload-bound:

1. **Per-variant R2 fetch** — each variant was a byte-range GET from the remote `variants.tar`
   (`fetch_variant`). One R2 round-trip per variant.
2. **Per-variant reference RE-upload** — every `run_metric` call re-uploads the reference to the GPU.
   nsys (RTX 5070, variants already local): `cuMemcpyHtoDAsync` = **54%** of CUDA API time; GPU-side
   826 ms H2D vs 170 ms compute; **GPU sm util ~10%**. The GPU starves on uploads, not compute.

## Part 1 — TAR-SHARD local variant source (`$ZEN_VARIANTS_LOCAL_DIR`)
When set, `fetch_variant` reads `<dir>/<name>` off local disk instead of a byte-range GET. The worker
pulls + extracts the per-box tar ONCE at onstart, then a box that scores every variant in that tar
issues **ZERO per-variant R2 requests**. The variant index gains an optional 4th `name` column
(`sha\toffset\tsize\tname`; `build_scorefile_manifest.py` already computes the member basename);
3-column legacy indices still parse (name="") and fall back to the byte-range GET. `fetch_variant`
returns `(path, owned)` — TAR-SHARD reads are borrowed (shared extract dir, never deleted);
byte-range temps are owned. Real-box measured: tar pulled once (6-27 s), tens of thousands of
variants extracted, 0 per-variant R2.

## Part 2 — warm-reference batch scoring (`ZEN_SCOREFILE_WARMREF=1`)
All variants in a ScoreFile job share ONE reference. The orchestrator-eligible metrics
(cvvdp/dssim/iwssim/ssim2 — everything except butteraugli, which is `metric_orchestrator_eligible ==
false`) are routed through one `Orchestrator::run_all` batch. `run_all` groups tasks by `ref_hash`
and warm-holds the reference precompute device-resident (`session_pool::score` →
`score_with_warm_ref`), so the reference uploads ONCE per source instead of per variant — the fix for
the 54% H2D. butteraugli (ineligible) + zensim (needs its 372-feature sidecar) stay on the inline
path over the SAME decoded buffers (no double decode). Default OFF = byte-identical one-shot path.

## Real-GPU A/B (vast RTX 3060, 12 GB, driver 570; 8 sources × 315 variants × 6 metrics = 15,120 scorings)
| | baseline (one-shot) | warm-ref (Part 1+2) |
|---|---|---|
| wall | 529.1 s | **331.6 s** |
| GPU sm util | mean 23% / max 41% | mean 18% / **max 70%** |
| metric rows | 13,224 | 13,224 |
| errors | 0 | 0 |

- **Speedup: 1.60× wall** (measured). Correctness: max |diff| = 9.5e-7 (f32 noise), scores identical.
- The A/B harness spawns one `jobexec` PROCESS per source (8/side), so warm-ref pays
  orchestrator-init + warm-bench PER PROCESS — the 1.60× is a **FLOOR**. Production `--serve` (one
  warm process across many sources) amortizes that init → larger win. WSL2 could not measure this
  (it serializes GPU work + masks the H2D behaviour) — this is why the A/B ran on a real Linux box.

## Full-fill cost (MEASURED basis, 5.74M rows × 4 metrics, scaled 4/6 from the 6-metric A/B)
- baseline ~223 GPU-h; **warm-ref ~140 GPU-h**.
- warm-ref: **~$7-13, 4.7-7 h wall** on 20-30 cheap 3060-class boxes ($0.05-0.09/hr).

## Legacy=1 + fleet-image (the concurrency race did NOT reproduce)
See the CLAUDE.md KADIS-700k section. Modern orchestrator + `--bench-on-start no` + per-box GPU
self-test is the recommended default; `ZENMETRICS_SWEEP_LEGACY=1` is an escape hatch only. GPU fleet
images MUST bake CUDA dev headers (a runtime image lacks `cuda_runtime.h` → cubecl NVRTC JIT fails to
compile cvvdp/dssim/butteraugli).
