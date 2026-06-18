# Heterogeneous all-codec, all-metric sweep — scoping + plan (2026-06-18)

**Status: vast GPU-scoring link PROVEN on real hardware (2026-06-18). Done: (1) full
zenjpeg GPU sweep+score local (2976/2976, 0 fail, 4 metric cols 0-null, 372 feat_);
(2) even-only re-cluster fixing holdout contamination; (3) vast smoke — zenjpeg sweep +
GPU ssim2/butteraugli via CUDA on a real CUDA-13.0 box, RD-sane, clean teardown, ~$0.007.
NEXT: render the even-only clustered crop/scale set (SDR + HDR-76), then the heterogeneous
Hetzner-encode/vast-score run.** Survives compaction — canonical reference for the effort.

## v27 fleet worker image DELIVERED + validated (2026-06-18)

**`ghcr.io/imazen/zen-metrics-sweep:v27`** (+ `:v27-2026-06-18`, digest sha256:f73030d3… (latest main, 2026-06-19),
**public**) — the orchestrator-aware fleet worker, built + pushed + validated on real GPU
hardware. Satisfies the goal end-to-end:

- **Latest main of each codec** — binary built from the current sibling checkouts (zenjpeg
  +38 / zenjxl +12 / zenwebp +10 / zenpng +6 / zenavif +3 past the 6/12 pins) + the
  zenjxl-decoder At<Error> path-patch. `CUDARC_CUDA_VERSION=12000` (runs on all vast 12/13 boxes).
- **Features**: `sweep,gpu,gpu-cuda,orchestrator,orchestrator-cuda,orchestrator-cpu-all` +
  defaults (all 5 codecs + cpu-metrics). All 11 metrics baked: ssim2(-gpu), butteraugli(-gpu),
  dssim(-gpu), zensim(-gpu), iwssim(-gpu), cvvdp.
- **Highest-level API** = the orchestrator (default scheduler; OOM GPU→strip→CPU ladder + cpu-all
  reference adapters).
- **GPU-conditional** ("if a gpu is installed"): on a no-GPU box the orchestrator's in-process
  cubecl-cuda client panics (`CUDA_ERROR_NO_DEVICE`); fixed in `main.rs` — `detect_gpu().present`
  (safe nvidia-smi probe) forces the legacy CPU scheduler when no GPU, so the worker still
  encodes + runs CPU metrics. Proven locally both ways (no-GPU → CPU, no panic; GPU → orchestrator).
- **Efficient / bake-everything**: v27 layered image (`Dockerfile.sweep.v27`), no apt/pip/build at
  boot, precompiled binaries COPY'd in.
- **Validated on real hardware** (RTX 3060, driver 580.126.09 = CUDA 13.0): all 5 codecs encoded +
  scored via the orchestrator — zenjpeg 24/24, zenwebp 3/3, zenavif 8/8, zenjxl 9/9, zenpng 3/3
  (lossless ssim2 100.0), **0 encode/decode/score failures**; columns prove CPU+GPU metric cols
  (`score_ssim2_gpu`, `score_ssim2`, `score_butteraugli_max_gpu`, `score_butteraugli_pnorm3_gpu`,
  `score_zensim`). ~$0.015 total vast spend, 0 orphan boxes.
- **Two real bugs fixed (orchestrator-cpu-all path was never compiled before)**:
  `zenmetrics-orchestrator/src/cpu_adapter.rs` `_strip_height`→`strip_height` (2 fns, 20 errors);
  `zenmetrics-cli/src/main.rs` no-GPU panic. **Follow-up: commit these to master** (CI-safe; held
  only by the entangled zenjxl-decoder path-patch in `@`).
- Note: it's **amd64** (vast x86). Hetzner ARM CAX would need an arm64 build of the same.

## Vast smoke PROVEN + image-arch gotchas (2026-06-18)

- **The 12000-binary-on-CUDA-13 compat claim is proven on real hardware.** vast instance
  41487752 (GTX 1080, driver **580.126.20 = CUDA 13.0**), image `zen-metrics-sweep:v26`,
  ran `zen-metrics sweep --codec zenjpeg --knob-grid '{}' --q-grid 30,60,90 --metric
  ssim2-gpu --metric butteraugli-gpu --gpu-runtime cuda`: **3/3 cells, 0 fail**, RD-sane
  GPU scores (ssim2 q30→68.3, q60→77.1, q90→83.6; butteraugli max+pnorm3 both emitted) +
  a `score` GPU fallback. NOT the WSL2-snap-docker default-fallback the CVVDP task kept
  hitting — real CUDA scoring. Clean teardown, 0 orphan instances, ~$0.007 total.
- **Image arch gotchas (cost 2 wasted launches — record for next time):**
  - `zen-metrics-sweep:hetzner-v2` (06-12) is **linux/arm64 only** (the Hetzner CAX image);
    it fails `no matching manifest for linux/amd64` on x86 vast boxes.
  - `zen-metrics-sweep:v26` (05-30) is **linux/amd64** — the right image for x86 vast.
  - v26 is **pre-rename**: the CLI binary is `/usr/local/bin/zen-metrics` (not `zenmetrics`),
    version 0.6.0, and **predates `--plan`** (use classic `--knob-grid '{}'`). It IS built
    `CUDARC_CUDA_VERSION=12000`, so it directly validated the 12000-on-13 path.
  - **Implication:** for the real run we need an **amd64 image at current mains** (or
    `SWEEP_BIN_OVERRIDE` the `=12000` binary onto v26 amd64). The newest amd64 sweep image
    is the 05-30 v26; the 06-12 work shipped arm64-only.
  - vast `destroy` needs `-y`; `--onstart-cmd` arg cap is **16384 bytes** (inline base64
    source must be tiny — a 48×48 PNG fits).

- **Even-only re-cluster done:** `scripts/imazen26_recluster_even.py` →
  `imazen26_representatives_K500_even_2026-06-18.tsv` (11,902 even crop units → 500 reps /
  386 imgs, **0 odd/holdout reps**, all 11 crop labels present — clustered crops, not
  sources). Odd ids remain a natural holdout. The old K500 (202/414 odd) is superseded.

## CUDA targeting + corpus reality (2026-06-18, live-verified)

## Goal (user ask)

A single coordinated sweep across **all zen codecs at once**, via **zenfleet**,
**latest codec mains**, **all expert variant axes per codec**, producing the
training **parquets** for an **ssim2-approximating picker target** (NOT zensim
Profile-A — see [[picker-target-ssim2-not-profile-a]]). Refined by the user
2026-06-18:

- **Corpus:** dual **HDR + SDR** derived from **imazen-26**, including **scales
  and crops**.
- **Breadth:** use each codec's **new expert variant helpers** (modes_full);
  **identify codecs lacking them**.
- **Roll-out:** **heterogeneous fleet — Hetzner CPU for ENCODES + vast.ai GPU
  for METRICS**; **compute + store ALL metrics** per encode for a given source
  reference image; **smoke-test all metrics first**.

This is the **job-system** capability-routing model (Encode→CpuHeavy→Hetzner,
Metric→Gpu→vast.ai, content-addressed encode reused by every metric), NOT the
single-box chunk-sweep path.

## Dig findings (verified by 3 parallel agents 2026-06-18, file:line in agent logs)

### A. Corpus — exists on disk, NOTHING on R2

- Canonical PNG-normalized = **`imazen-26-png-v2`** at `/mnt/v/output/imazen-26-png-v2/`:
  **2,563 `.sdr.png` + 76 `.hdr.png`** (16-bit PQ, PNG 3.0 cICP `[1or12,16,0,1]`,
  SDR white 203 cd/m² per BT.2408). v2 is the in-code canonical (`sweep/hdr.rs:73`).
- SDR derivation already paired with HDR (same convert run). Distinction =
  filename suffix `.sdr.png`/`.hdr.png` + sweep output `hdr_mode=pq1000` column
  (absent for SDR).
- Scales rendered already: SDR `/mnt/v/output/imazen-26-features/train_renditions_2026-06-14/`
  (1,482 Mitchell+sharpen, downscale-only); HDR `/mnt/v/output/imazen-26-hdr-grid-2026-06-14/`
  (1,140 16-bit PQ, **linear-light** resize). Renderers: zenanalyze
  `examples/render_imazen26_variants.rs` (SDR) + `examples/extract_hdr_size_grid.rs` (HDR).
- Crops: window math exists (`zenanalyze/examples/extract_features_imazen26_crops.rs`,
  in-memory) but **NO tool writes crop PNGs to disk** → small renderer to build.
- **HDR encode reality:** only **zenjxl** has a true HDR path today (16-bit PQ via
  zencodec adapter); zenjpeg/zenwebp/zenpng/zenavif refuse HDR loudly. **`--hdr` is
  incompatible with `--plan`** (knob-grid only, accepts `{lossless,distance,noise,effort}`).
  → dual sweep = SDR across 5 codecs via plan + HDR zenjxl-only via knob-grid.
- The SDR-decode cICP tripwire (`crates/zenmetrics-cli/src/decode.rs:115`) refuses
  PQ/HLG PNGs so a `.hdr.png` can't silently enter an SDR sweep.
- **R2 upload required**: workers fetch sources at `s3://$ZEN_BUCKET/$ZEN_CORPUS_PREFIX/<image_path>`.

### B. Expert variant helpers — 5 wired, 1 unwired, 4 absent

| codec | sweep module | modes_full | cell-id grammar | feature_row | verdict |
|---|---|---|---|---|---|
| zenjpeg | `zenjpeg/src/encode/sweep.rs` | yes | `config_from_cell_id` | no | full, wired |
| zenavif | `zenavif/src/sweep.rs` | yes (+alpha) | `config_from_cell_id` | **yes (only one)** | full, wired |
| zenjxl | `zenjxl/src/sweep.rs` | yes | `variant_from_cell_id` | no | full, wired |
| zenwebp | `zenwebp/src/sweep.rs` | yes | `variant_from_cell_id` | no | full, wired |
| zenpng | `zenpng/src/sweep.rs` | yes (lossless) | `variant_from_cell_id` | no | full, wired |
| **zengif** | `zengif/src/sweep.rs` (`bd612d6`) | yes | `variant_from_cell_id` | no | **full but NOT wired into zenmetrics dispatch** |
| zentiff/image-tiff, heic, zenbitmaps, zenraw | — | — | — | — | **NONE** |

- zenmetrics dispatch: `crates/zenmetrics-cli/src/sweep/plan.rs` (`build_plan`) wires
  the 5; **add a `CodecKind::Zengif` arm** to include zengif.
- `feature_row()`/`feature_columns()` picker-training contract is **zenavif-only** —
  the other 4 expose plan/fingerprint/cell-id only. (Picker training reads the
  omni/ledger join, so this is not a hard blocker, but zenavif is the model.)
- No sweep-module drift since the 2026-06-12 grammar pins (grammars current).

### C. Heterogeneous fleet — routing exists, GPU executor + tier don't

- **Capability routing IMPLEMENTED + unit-tested** (`crates/zenfleet-core/src/job.rs`):
  `Encode{jpeg,png}`→CpuLight, `Encode{webp,jxl,avif}`→CpuHeavy, `Metric`/`Diffmap`→**Gpu**,
  `Feature`→CpuHeavy, `Bake`→HighRam. `worker_serves()` filters; worker CLI
  `--capability` / container `ZEN_CAPABILITY`. Proven local (synthetic):
  `scripts/jobsys/demo_capability_routing.sh`.
- **Gaps to a working Hetzner-encode + vast-GPU-all-metrics smoke:**
  1. **No GPU jobexec image.** `jobexec` metric path is CPU-only as built
     (`crates/zenmetrics-cli/src/metrics/mod.rs:528` gates GPU behind `gpu-*`,
     else "disabled — rebuild with --features gpu-*"). Need a CUDA-base worker
     image baking `zenmetrics --features sweep,gpu,gpu-cuda` (current
     `Dockerfile.executor` FROMs CPU debian-slim, no CUDA).
  2. **Launcher does no split.** `scripts/jobsys/launch_fleet.sh` sets
     `ZEN_CAPABILITY` on no tier; its vast tier is `num_gpus=0`. Need per-tier
     capability + a `num_gpus=1` vast GPU tier (pattern in
     `scripts/sweep/v15/launch_gpu.sh`).
  3. **Images private** — `zenfleet-worker-exec` ghcr package must be made public
     (RUNNING_JOBS.md §2); the new GPU image too.
  4. **Never run real on any remote box** (Hetzner or vast); GPU metric scoring
     never verified on real GPU in jobexec. (Adjacent proof: chunk-mode vast.ai
     sweep with `num_gpus=1` does run real GPU scoring, but single all-GPU box,
     not the split.)
  5. **"All metrics" = N Metric jobs/encode.** `run_metric` → `Vec<(col,f64)>`;
     butteraugli emits 2 sub-scores, others 1 scalar. zensim 300-feat vector NOT
     emitted by jobexec. Ledger stores `output_sha`→JSON blob (metric value in
     blob, not a typed column): `crates/zenfleet-ledger/src/lib.rs:91`.
  6. **Diffmaps declared, NO executor** (`jobexec.rs:234` rejects `Diffmap`).
  7. **No auto encode→metric join.** Two-phase: `declare_encodes` (input=source
     sha) → run → harvest `output_sha` from ledger → `declare` metrics
     (input=encode sha). No `declare_metrics_from_ledger` helper yet.

- Metric crates (all GPU+CPU capable via cubecl: cuda/wgpu/hip/cpu): butteraugli-gpu,
  dssim-gpu, ssim2-gpu, cvvdp-gpu, iwssim-gpu, zensim-gpu. Plus fast-ssim2 (CPU
  SSIMULACRA2), zenstats (correlation, not per-image).

## Ordered plan → smoke test

1. **[DONE 2026-06-18] GPU jobexec binary built + GPU scoring PROVEN native.**
   `cargo build --release -p zenmetrics-cli --features sweep,gpu,gpu-cuda` (56s,
   1.23 GiB peak). On the local RTX 5070 (CUDA 13.2, `LD_LIBRARY_PATH=/usr/lib/wsl/lib:
   /usr/local/cuda/lib64`), all 6 GPU metrics score: ssim2-gpu 57.10, butteraugli-gpu
   max 12.24/pnorm3 3.65, dssim-gpu 0.0040, iwssim-gpu 0.9932, zensim-gpu 43.28,
   cvvdp 9.71. **`jobexec` contract proven both kinds**: metric job (re-encode
   zenjpeg q80 → GPU ssim2 via Auto→CUDA = 83.95 → JSON row); encode job (→ 7395 B
   progressive JPEG). The "GATE unverifiable on WSL2" in the CVVDP task was the
   snap-docker NVML wall — native (no docker) clears it (memory [[cubecl-gpu-runtime]]).
   **GPU image cannot be validated locally** (snap-docker `--gpus all` fails on WSL2);
   only a real vast.ai GPU box validates the imaged path.
2. **Bake GPU worker image** (CUDA base + binary) → push → make public; flip CPU
   exec image public.
3. **Launcher heterogeneous split** — per-tier `ZEN_CAPABILITY` + `num_gpus=1` vast tier.
4. **One SDR reference image → R2** (full corpus after path proven).
5. **Declare**: small plan → encode cells → Hetzner; harvest `output_sha` → declare
   6 metrics → vast GPU. Add `declare_metrics_from_ledger` helper.
6. **Verify**: ledger shows `provider=hetzner` encodes + `provider=vast` metrics, 6
   metric blobs/encode.

Then **production**: full dual corpus + scales/crops to R2 (+ crop renderer), all-codec
modes_full, scale out, collect. Decide breadth with real cell counts + $ after smoke.

## Exercised + verified working 2026-06-18 (local, RTX 5070, current codec mains)

What is **already implemented and proven working** (ran it, not claimed):

1. **Plan machinery — all 5 codecs.** `modes_full --plan-budget 2000` (q-grid 11pt)
   cells/source: zenjpeg 1944, zenjxl 1980, zenwebp 1659, zenavif 1056, zenpng 9.
   **Unbudgeted = full combinatorial** (zenjpeg 1,036,800 strata/source) → a budget
   cap is mandatory. Self-describing identity `{cell,fp,plan}` + source_sha emits
   correctly (`config_from_cell_id`/`variant_from_cell_id` roundtrip).
2. **Chunk-mode source-grouped sweep** (`zenmetrics sweep --plan`). zenjpeg rd_core ×
   2 GPU metrics → omni TSV (`image_path/codec/q/knob_tuple_json/encoded_bytes/
   encode_ms/encoded_filename/decode_ms/score_<metric>`), 60 cells, scores populated,
   2s, **one ref decode** (source-grouped). `--feature-output` writes a **parquet**
   (60 rows × 377 cols, **372 `feat_*`**) keyed on identity.
3. **All 5 codecs encode + GPU-score** at current mains (rc=0, valid ssim2; zenpng
   lossless → 99.99).
4. **Two-pass heterogeneous split — PROVEN:**
   - Pass A (Hetzner side): `sweep --distorted-out-dir` → content-addressed dist PNGs
     `<img>_<srcHash>_<codec>_q<q>_<knobHash>.png`. Works encode-only.
     **GAP:** `--pairs-tsv` only emits rows when `--metric` is also set (empty in pure
     encode-only mode) — either a tiny fix, or Pass B reconstructs pairs from the
     deterministic filenames.
   - Pass B (vast-GPU side): `score-pairs --pairs-tsv --metric X --out-parquet` → per-
     metric parquet sidecar keyed on identity (ssim2-gpu 42 rows; butteraugli-gpu 2
     sub-scores max+pnorm3), 1s each. **"All metrics" = one `score-pairs` per metric
     → N sidecars**, joinable on `(image_path,codec,q,knob_tuple_json)`.
5. **jobexec** (job-system executor): encode job → bytes; metric job → re-encode + GPU
   score (Auto→CUDA) → JSON row. Both kinds proven.
6. **All 6 GPU metrics** score on the RTX 5070 natively.
7. **Job-system capability routing** (`demo_capability_routing.sh`): GPU worker
   (`--capability gpu`) claimed only the 6 Metric jobs; CPU worker
   (`--capability cpu_light cpu_heavy`) only the 9 Encode jobs — the **Hetzner/vast
   split primitive, proven** (synthetic `/bin/cat` executor).
8. **`zenfleet-ctl declare`**: spec (1 item × 6 metrics) → 6 Metric `DesiredJob`s with
   encode_sha inputs + cell identity; validates sha256 (rejected a malformed sha).

**Verdict:** the chunk-mode source-grouped two-pass pipeline (the recommended path) is
**fully working locally end-to-end** — sweep→encode→persist (Pass A) and
score-pairs→GPU→sidecar (Pass B). The single binary serves both tiers. Remaining work
is packaging (GPU image) + fleet wiring, not core function.

**Gaps found while exercising:** (a) `--pairs-tsv` needs a `--metric` to emit (blocks
the clean pure-encode Pass A → Pass B handoff); (b) `modes_full` needs a budget cap;
(c) `assemble` is R2-run-oriented (`--runs/--bucket/--r2-endpoint/--s5cmd`) — sidecars
are key-compatible so the typed join will work, but it wants a fleet run layout.

## Build fix applied 2026-06-18

zenjxl main (`3ad46e1`) migrated the decoder boundary to whereat `At<Error>` (#14
c560121 / #15 5d33e76); zenmetrics' `[patch]` pinned the stale pre-migration decoder
rev `0bd33d21` → E0599 `map_err_at`/`map_error`. Fixed: path-patch
`zenjxl-decoder = { path = "../zenjxl-decoder/zenjxl-decoder" }` (local HEAD `311a897`
carries the At<Error> API), matching zenjxl's own workspace. **CI/fleet follow-up:**
re-pin to a pushed decoder rev + advance CI sibling-clone pins before pushing
(memory: [[zenmetrics-ci-sibling-clone-drift]]).

## CUDA targeting + corpus reality (2026-06-18, live-verified)

**CUDA: pin `CUDARC_CUDA_VERSION=12000` — NOT a regression, it's full coverage.**
Live vast.ai pool (64 GPU offers, `cuda_max_good>=12`): **66% CUDA 13.x (42/64), 34% 12.x
(22/64)**, driver 525 (CUDA 12.0) still present. CUDA is backward-compatible (new driver
runs old toolkit), so a 12.6-toolkit image + `=12000` binary runs on **all 64**; a
13-targeted binary runs on only 42 (loses the 22 12.x boxes = the real regression).
Rebuilt with `=12000` (cudarc 0.19.7): zenjpeg GPU sweep works, **bit-identical scores to
the native-13 build** (ssim2 57.104932 = 57.104932 — the pin is ABI-only, zero numeric
change). This binary is the `SWEEP_BIN_OVERRIDE` artifact. Optional: bump image nvrtc
12-6→12-8/12-9 for native newest-GPU (Blackwell sm_120) codegen (12-6 covers them via
PTX-JIT); keep the `=12000` driver pin regardless. No new image needed (see hetzner-v2 reuse).

**Clustered imazen-26 set: it's a MANIFEST, not renditions; HDR absent.**
- `imazen26_representatives_K500_2026-06-14.tsv` (`/mnt/v/output/imazen-26-features/` +
  Tower) — sklearn **k-means** over 84 z-scored *content* features (geometry/size excluded),
  centroid-nearest, **K=500 → 500 rows / 414 distinct images**. Each row = (source PNG, crop
  window): `full` + `c{50,25}_{center,tl,tr,bl,br}` = 11 crops/image. Generator
  `zenanalyze/benchmarks/imazen26_cluster_ablation_2026-06-14.py`. Distinct from the FPS
  budget set (`train_renditions_2026-06-14`, 1482, resizes-only, no crops, already rendered).
- **Crops were NEVER rendered to disk** (only in-memory for feature extraction); the dense
  scale grid (`DENSE_SIZES 32..4096`, Lanczos3 downscale) is not yet applied to the clustered
  set. The clustered crop/scale set is a *selection spec*, not image files. Rendering it (SDR)
  = 414 imgs × ≤11 crops × ~10 scales. No R2 copy yet.
- **HDR duplicate ABSENT.** The HDR grid (`imazen-26-hdr-grid-2026-06-14`, 1140) is an
  independent all-76-sources × generic-size render (no crops, no cluster mirror). To duplicate
  the clustered set in HDR: (1) intersect K500 with the 76 native-HDR ids (small subset — HDR
  ceiling is 76 sources in 4 photo classes); (2) add crop support to
  `zenanalyze/examples/extract_hdr_size_grid.rs` (full-image only today; crop the u16 PQ buffer
  pre-resize — transfer-neutral); (3) render PQ crops+scales linear-light. Net-new work.

## Production decisions (defaults; redirect anytime)

- Metrics: all 6 GPU (ssim2/butteraugli/cvvdp/dssim/iwssim/zensim).
- Diffmaps: scalars-only for smoke; wire `Diffmap` executor for production.
- HDR: zenjxl-only this round; SDR across all 5 (+ zengif once wired).
- Corpus breadth: decide post-smoke with measured cell counts + Hetzner/vast $.
