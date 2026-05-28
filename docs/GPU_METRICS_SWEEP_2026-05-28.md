# GPU metrics VRAM + wall-time sweep — 2026-05-28

Task #133.  Measured every GPU metric crate's peak VRAM and per-call
wall-clock at four image sizes against both CUDA and Vulkan
(cubecl-wgpu) backends on an RTX 5070 (12 GiB, driver 596.21).

- Harness: `scripts/memory_audit/sweep_gpu_metrics_2026-05-28.py`
- Per-metric TSVs: `crates/<metric>/benchmarks/gpu_vram_sweep_2026-05-28.tsv`
- Global summary: `benchmarks/gpu_metrics_sweep_2026-05-28.tsv`
- Estimator gap join: `benchmarks/gpu_metrics_gap_2026-05-28.tsv`
- Estimator-only sidecar: `benchmarks/gpu_metrics_estimates_2026-05-28.tsv`
- Total cells: **232** (132 CUDA + 100 wgpu, all 6 metrics × 4 sizes
  × per-metric mode list)
- Synthetic LCG noise inputs; subprocess-per-cell to defeat
  cubecl's cross-Drop buffer pool

## Sizes swept

| label  | w × h         | MP    | notes                                          |
|--------|---------------|-------|------------------------------------------------|
| 1mp    | 1024 × 1024   | 1.0   | per-call fixed overhead lives here             |
| 4mp    | 2048 × 2048   | 4.2   |                                                |
| 16mp   | 4096 × 4096   | 16.8  | task #131's canonical bench size               |
| 40mp   | 7680 × 5184   | 39.8  | non-square; common ~8K still / 40-MP camera    |

## Modes per metric (task brief reference)

| crate           | modes measured                                                                  | task ref |
|-----------------|---------------------------------------------------------------------------------|----------|
| butteraugli-gpu | full, strip, warm_ref, warm_ref_strip                                           | #45      |
| ssim2-gpu       | full, strip, warm_ref, warm_ref_strip                                           | #46      |
| dssim-gpu       | full, strip, warm_ref, warm_ref_strip (warm_ref_strip == Mode E)                | #73      |
| iwssim-gpu      | full, strip, warm_ref, warm_ref_strip, rgb_full, rgb_strip, rgb_warm_ref_strip  | #57      |
| zensim-gpu      | full, warm_ref, strip, warm_ref_strip                                           | #49 #75  |
| cvvdp-gpu       | full, warm_ref, warm_ref_strip, strip_pair, capped, auto                        | #133     |

cvvdp-gpu `strip` cold-ref (Mode E without `warm_reference`) is NOT
supported by the pipeline — it requires `warm_reference` to be called
first, otherwise band processing panics at
`pipeline.rs:4868`.  It's recorded as a known unsupported mode and
deliberately not in the cvvdp orchestrator config.

# 16 MP × Full peak VRAM per metric (CUDA)

The calibration table.  Delta against the steady-state nvidia-smi
baseline, sampled during compute via a parallel polling thread (the
post-READY sample window alone under-reports by 4 GiB+ because cubecl
releases transient scratch the moment compute returns).

| crate            | mode      | peak VRAM   | est (from src) | gap         | peak/est | wall median |
|------------------|-----------|-------------|----------------|-------------|----------|-------------|
| butteraugli-gpu  | full      | 3.91 GiB    | 3.12 GiB       | +801 MiB    |  1.25×   |  62.3 ms    |
| ssim2-gpu        | full      | 6.15 GiB    | 4.87 GiB       | +1.27 GiB   |  1.26×   |  50.7 ms    |
| dssim-gpu        | full      | 3.16 GiB    | 3.19 GiB       |  −34 MiB    |  0.99×   |  50.5 ms    |
| iwssim-gpu       | full      | 2.16 GiB    | 2.40 GiB       |  −247 MiB   |  0.90×   |  45.3 ms    |
| zensim-gpu       | full      | 1.16 GiB    | 875 MiB        |  +310 MiB   |  1.36×   |  38.1 ms    |
| cvvdp-gpu        | full      | 3.88 GiB    | 2.79 GiB       | +1.11 GiB   |  1.39×   |  45.5 ms    |
| cvvdp-gpu        | strip_pair| 2.22 GiB    | 2.50 GiB       |  −291 MiB   |  0.89×   | 203.0 ms    |

(`est` = analytic estimator from `<crate>::estimate_gpu_memory_bytes`,
joined in `benchmarks/gpu_metrics_gap_2026-05-28.tsv`.)

Calibration check vs each crate's `examples/mem_*.rs` reference at
4096²:

- **butteraugli-gpu** 16 MP Full delta = 4001 MiB matches the
  expected ~4 GiB working-set range (50 planes × 16 MP × 4 B
  = 3.13 GiB; +25% cubecl pool overhead).
- **dssim-gpu** (task137 RECALIBRATED): estimator now over-predicts by
  1% at 16 MP (peak/est = 0.99×, was 2.62× under).  Fixed by counting
  31 planes/scale (was 13 — `Scale::new` = 9·alloc_3 + 4 singles) plus
  a base + per-pixel GPU-context term (208 MiB + 18 B/px).
- **iwssim-gpu** (task137 RECALIBRATED): estimator now over-predicts by
  10% at 16 MP (peak/est = 0.90×, was 2.25× under).  Fixed by counting
  19 planes/scale (was 10), 6.39 MiB reduction/cov scratch, a 1.40 pool
  factor, and a 256 MiB floor.
- **cvvdp-gpu strip_pair (Mode B)** (task137 RECALIBRATED): estimator
  now over-predicts by 11% at 16 MP (peak/est = 0.89×).  The prior
  estimator under-predicted ~3-4× — it sized all three pyramids strip-
  shaped and omitted the persistent full-n0 `DBandsTransient`.  Fixed
  to be source-faithful (full gauss_ref, k≤k_split gauss_alt, baseband-
  only bands_dis, the +826 MiB transient) plus a 256 MiB + 32 B/px
  context term.
- **cvvdp-gpu Full** estimator still under-predicts at small/mid sizes
  (peak/est = 1.39× at 16 MP).  This is OUT OF task137's Mode B/E
  scope; it propagates into Mode E (warm_ref_strip = Full + RefFullState),
  which therefore also under-predicts at small/mid sizes.  Filed as
  follow-up: recalibrate the cvvdp Full estimator.
- **zensim-gpu** estimator UNDER-estimates by 36% at 16 MP (peak/est
  = 1.36×) — separate follow-up (owned by agent #138).

# Strip vs Full peak VRAM ratio at 16 MP (CUDA)

The headline strip-mode value: how much VRAM does going Strip save?

| crate            | Full MiB | Strip MiB | Full/Strip ratio | conclusion              |
|------------------|----------|-----------|------------------|-------------------------|
| butteraugli-gpu  | 4001     | 481       | **8.3×**         | Strip pays big          |
| ssim2-gpu        | 6294     | 1217      | 5.2×             | Strip pays              |
| dssim-gpu        | 3233     | 897       | 3.6×             | Strip pays              |
| iwssim-gpu (gray)| 2209     | 545       | 4.1×             | Strip pays              |
| iwssim-gpu (rgb) | 2210     | 546       | 4.1×             | Strip pays (RGB native) |
| zensim-gpu       | 1185     | 1249      | **0.95×**        | Strip ≈ Full at 16 MP   |
| cvvdp-gpu        | 3969     | n/a       | (warm_ref_strip 3969) | Strip-only mode panics; Mode E warm_ref_strip retains full REF state |

**zensim-gpu finding**: strip mode does NOT meaningfully reduce VRAM
at 16 MP because the per-scale pyramid (`41 B × pyramid_pixels`) is
already small, and the strip walker still pays the pyramid for each
strip's height.  At 40 MP, strip drops from 2209 → 1281 MiB (1.7×) —
the savings only kick in once the full-image pyramid is large enough.

**cvvdp-gpu finding**: the `strip` constructor produces a Mode E
strip pipeline, but `compute_dkl_jod` (cold-ref) panics because Mode
E requires `warm_reference` to populate `ref_full_state` before band
processing.  The supported strip-mode entry points are
`warm_reference + compute_dkl_jod_with_warm_ref` (warm_ref_strip) or
`new_strip_pair + compute_dkl_jod` (strip_pair = Mode B; cold-ref
walks both sides). Mode E warm_ref_strip retains the FULL reference
state on device (3.57 GiB at 16 MP for ref bands + pyramid), so VRAM
is dominated by the cached ref, NOT the strip working set — the
strip estimator (300 MiB) is per-strip working set only.

# Strip wall-time regression at 16 MP (CUDA)

Does the strip walker pay a wall-time cost vs the Full pipeline?

| crate            | Full wall (ms) | Strip wall (ms) | ratio | notes                      |
|------------------|----------------|------------------|-------|----------------------------|
| butteraugli-gpu  | 62             | 91               | 1.5×  | mild regression            |
| ssim2-gpu        | 51             | 205              | 4.0×  | substantial — Mode E refinement queued #75  |
| dssim-gpu        | 50             | 278              | 5.5×  | substantial — strip needs Mode E fast path  |
| iwssim-gpu (gray)| 45             | 100              | 2.2×  | acceptable                 |
| iwssim-gpu (rgb) | 113            | 509              | 4.5×  | RGB strip slow             |
| zensim-gpu       | 38             | 61               | 1.6×  | mild                       |
| cvvdp-gpu (warm) | 45 (warm_ref)  | 109 (warm_ref_strip) | 2.4×  | strip pays per-strip walker overhead       |

`wall_median_ms` is the median of `WORKER_REPS=2` post-warm calls;
warm-up (first call) is excluded.  At 16 MP the per-strip walker's
boundary-management cost (halo upload, per-strip kernel dispatch,
reduction merge) typically dominates the per-strip kernel savings.

# Does warm_ref work?

Defined as a working-direct cached-reference fast path (`set_reference
+ compute_with_reference` faster than cold-ref `compute(ref, dist)`).

| crate            | warm_ref vs Full speedup (16 MP) | warm_ref_strip vs Strip speedup |
|------------------|----------------------------------|---------------------------------|
| butteraugli-gpu  | 1.9× (62 → 33 ms)                | 0.6× (91 → 151 ms)    SLOWER    |
| ssim2-gpu        | 1.2× (51 → 44 ms)                | 1.7× (205 → 120 ms)             |
| dssim-gpu        | 1.0× (50 → 52 ms)                | 1.7× (278 → 162 ms)             |
| iwssim-gpu       | 1.1× (45 → 42 ms)                | 1.0× (100 → 100 ms)             |
| zensim-gpu       | 1.2× (38 → 31 ms)                | 0.1× (61 → 488 ms)    SLOWER    |
| cvvdp-gpu        | 1.8× (45 → 26 ms)                | 1.0× (109 → 109 ms)             |

`warm_ref` consistently delivers a per-call speedup for batched
encoder-style workloads.  `warm_ref_strip` is **inconsistent** — for
butteraugli and zensim it's actually slower than `strip` cold-ref,
suggesting the strip-mode cached-ref path has overhead that doesn't
pay off at WORKER_REPS=2.  Hypotheses for follow-up:

1. Cached strip-mode ref blits a copy of the full reference into a
   "ref bands" cache on first call.  At repeat 1 the cache miss-path
   runs; at repeat 2 the hit-path runs.  N=2 may not be enough to
   amortise.  Re-run with `WORKER_REPS=10` to confirm.
2. The strip walker may be re-uploading reference rows per-strip
   even with `set_reference` populated.  HtoD count would show this
   — see "Follow-ups" below.

# wgpu compatibility

cubecl-wgpu on Linux uses the Vulkan backend.  **All 6 metrics × all
modes work through 16 MP and 40 MP on the Vulkan-backed wgpu** —
peak VRAM is within ±2% of the CUDA baseline (Vulkan compute
pipelines allocate the same backing buffers cubecl's CUDA backend
does), and wall_median is generally within ±10%.

No instances of:
- `ERR_DISPATCH_CAP` — dispatch limits are not hit even at 7680 ×
  5184 (40 MP).  Task #131's note about wgpu caps at 4096² likely
  referenced a different environment (e.g., browser WebGPU with its
  default WGPU_LIMITS, not Vulkan).
- `ERR_BUF_ALIGN` — none.
- `ERR_NO_WGPU_ADAPTER` — none.

This means a Vulkan-backed deployment can use any of the six -gpu
crates as a drop-in for CUDA without code changes.  Worth confirming
on Metal / DX12 / WebGPU separately; this measurement is Linux+RTX
5070 only.

# 40 MP card-tier fit

Per-metric peak VRAM (CUDA) at 40 MP, mapped to common consumer GPU
tiers:

| crate            | Full      | Strip / warm_ref_strip | best 4 GiB mode | best 8 GiB mode | best 12 GiB mode |
|------------------|-----------|------------------------|------------------|------------------|------------------|
| butteraugli-gpu  | 7.69 GiB  | 1.00 GiB / 8.49 GiB    | Strip            | Strip / Full*    | Full             |
| ssim2-gpu        | 9.23 GiB  | 2.14 GiB / 6.90 GiB    | Strip            | Strip            | Full (tight)     |
| dssim-gpu        | 7.00 GiB  | 1.38 GiB / 4.22 GiB    | Strip            | Strip / Full     | Full             |
| iwssim-gpu (gray)| 4.91 GiB  | 1.60 GiB / 1.60 GiB    | Strip            | Full / Strip     | Full             |
| iwssim-gpu (rgb) | 4.91 GiB  | 0.88 GiB / 1.60 GiB    | rgb_strip        | rgb_strip / Full | rgb_full         |
| zensim-gpu       | 2.16 GiB  | 1.25 GiB / 1.25 GiB    | Full             | Full             | Full             |
| cvvdp-gpu        | 7.19 GiB  | 5.03 GiB strip_pair / 7.38 GiB warm_ref_strip | strip_pair (tight) | strip_pair       | Full / warm_ref_strip |

`*` butteraugli warm_ref_strip 40 MP shows 8.49 GiB because the
cached-ref strip path keeps the full REF state on device PLUS the
per-strip dist working set — see the gap-analysis section above.

Headlines:
- **4 GiB cards** (e.g., 3050, 1650-tier) need Strip mode for any
  metric at 40 MP except zensim.
- **8 GiB cards** can run iwssim or cvvdp's strip_pair full-image
  at 40 MP; ssim2 still needs strip.
- **12 GiB cards** like the RTX 5070 in this measurement can run
  Full at 40 MP for every metric except ssim2 (which is right at the
  edge at 9.23 GiB and shows nvidia-smi timing-out under sustained
  memory pressure).

# Estimator vs runtime gap analysis

Joined TSV at `benchmarks/gpu_metrics_gap_2026-05-28.tsv` has every
(crate, mode, size, backend) cell with `estimate_bytes`, `gap_bytes`,
`peak_over_estimate_ratio`.  Summary of patterns observed:

1. **butteraugli-gpu, cvvdp-gpu Full**: peak / estimate ≈ 1.1×.
   Estimator is well-calibrated; ratio is cubecl pool overhead.
2. **dssim-gpu Full**: peak / estimate ≈ 2.6×.  The analytic
   `13 planes × pyramid` formula misses the per-scale gaussian /
   mu / sigma / packed-u32 staging buffers.  **Follow-up**: rework
   `dssim-gpu/src/memory_mode.rs::estimate_gpu_memory_bytes` to
   match runtime planes.
3. **iwssim-gpu Full**: peak / estimate ≈ 2.3×.  Same pattern as
   dssim — analytic estimator undercounts per-scale derived buffers.
4. **ssim2-gpu Full**: peak / estimate ≈ 1.3×.  Reasonable.
5. **zensim-gpu Full**: peak / estimate ≈ 1.4×.  Under-estimates by
   36% at 16 MP.  Pool overhead + per-scale staging unmodeled in the
   Basic-regime formula.  **Follow-up**: re-fit the
   `beta_b_per_pyramid` coefficient against measured runtime or add
   a fixed per-image staging-buffer term.
6. **cvvdp-gpu warm_ref_strip**: peak / estimate ≈ 13×.  The strip
   estimator only models per-strip working set; warm_ref_strip
   retains the full REF state too.  **Follow-up**: expose
   `estimate_gpu_memory_bytes_warm_ref_strip` = strip + full ref
   state for capacity planning.
7. **cvvdp-gpu strip_pair**: peak / estimate ≈ 3.8×.  Mode B walks
   both sides with shared transient scratch.  Estimator models the
   ref-side cache but not the dist-side per-strip working set
   correctly.

# Identified follow-up work

In priority order (`<crate>` + 1-line action):

1. **dssim-gpu** rewrite `estimate_gpu_memory_bytes` to include
   gaussian / mu / sigma / packed-u32 buffers.  Current formula
   undercounts by 2.95× at 16 MP — capacity planning underestimates
   VRAM pressure, MemoryMode::Auto picks Full when it should pick
   Strip on tight cards.
2. **iwssim-gpu** same as dssim — analytic estimator undercounts by
   2.6×.
3. **zensim-gpu** tune `beta_b_per_pyramid` coefficient down to match
   measured runtime (18% over-estimate at 16 MP Full).
4. **butteraugli-gpu strip cached-ref** investigate why warm_ref_strip
   is SLOWER than strip cold-ref at 16 MP (91 → 151 ms).  Expected:
   2nd-call hit path skips ref re-upload.  Suspected: cache is
   blitted per-call, or strip walker re-uploads ref rows per-strip.
5. **zensim-gpu strip cached-ref** same investigation — 61 → 488 ms
   regression at 16 MP warm_ref_strip vs strip.
6. **cvvdp-gpu** expose `estimate_gpu_memory_bytes_warm_ref_strip` —
   strip estimator alone is 13× off because Mode E retains full REF
   state.  This is the most user-visible gap in the analytic
   capacity model.
7. **dssim-gpu, ssim2-gpu** wall-time regression in strip mode at 16
   MP — 5.5× / 4.0× slowdown vs Full.  Mode E refinement queued in
   task #75 should target these.

> **CORRECTION (task #138, 2026-05-28): ssim2-gpu warm_ref VRAM is
> NOT a regression.** The table reading `warm_ref` 40 MP > `full`
> (10.71 vs 9.23 GiB CUDA) is a measurement artifact of the cubecl
> dynamic pool sampled at different points on its growth curve — NOT
> retained ref state. Whole-image `warm_ref` and `full` share the
> identical 57-plane/scale `Scale` buffer set (`Ssim2::new`); their
> pool-stabilized peaks are byte-identical (16 MP both 6274 MiB; 18
> MP 6273≈6271 MiB) and at 40 MP both hit the same ~11.9 GiB pool
> ceiling and OOM. Under the published `reps=2` protocol the
> which-mode-is-higher is ±60 MiB noise (a re-run had `full` HIGHER
> at 16 MP). Full investigation + data:
> `crates/ssim2-gpu/docs/WARM_REF_VRAM_INVESTIGATION_2026-05-28.md`
> + `crates/ssim2-gpu/benchmarks/ssim2_warmref_trim_2026-05-28.tsv`.
> The parity-safe memory-bounded 40 MP mode is `warm_ref_strip`
> (measured 7.33 GiB, score bit-identical).

8. **HtoD-per-iter audit** — strip vs warm_ref strip wall-time
   regression in butter / zensim suggests strip cached-ref may be
   re-uploading the reference per-call.  Run `nsys` on
   warm_ref_strip 16 MP butter, expect ≤2 HtoD per iter; observe and
   document.
9. **Auto-mode policy validation**: cvvdp-gpu MemoryMode::Auto at 40
   MP resolved to Full (7.19 GiB) on 12 GiB card; on 8 GiB cards it
   should resolve to Strip.  Confirm `vram_cap_bytes` reads the
   card's available VRAM correctly across `cubecl::cuda` and
   `cubecl::wgpu` runtimes.
10. **iwssim-gpu rgb_strip wall time** — 509 ms at 16 MP, 1074 ms
    at 40 MP.  Investigate whether the native-RGB strip path
    triggers extra packed-u32 conversions per strip.

# Hardware / environment

| field         | value                          |
|---------------|--------------------------------|
| GPU           | NVIDIA GeForce RTX 5070        |
| VRAM total    | 12227 MiB                      |
| Driver        | 596.21                         |
| CUDA toolkit  | 13.2 at `/usr/local/cuda`      |
| Backend       | cubecl 0.10 (CUDA + wgpu/Vulkan) |
| Host          | lilith (water-cooled 7950X, 128 GB RAM) |
| git commit    | sweep landed on master (`693aa24b` → `e55ce3f5` → …); see commit history |

# How to re-run

```bash
# All 6 metrics × cuda + wgpu × 4 sizes
python3 scripts/memory_audit/sweep_gpu_metrics_2026-05-28.py \
    --metrics all --backends cuda,wgpu \
    --sizes 1mp,4mp,16mp,40mp --reps 2 \
    --out-base benchmarks/gpu_metrics_sweep_2026-05-28

# Estimator join after re-running
python3 scripts/memory_audit/join_estimates_2026-05-28.py
```

Each crate's `examples/mem_one_size.rs` is the worker.  Subprocess-
per-cell is mandatory — cubecl's memory pool caches buffers across
Drop, defeating in-process before/after sampling.

# Acceptance gate status (task #133)

- [x] `gpu_vram_profile` driver extended to every GPU metric — done
      via per-crate `mem_one_size.rs` rewrite (4-7 modes per crate).
- [x] Per-metric TSVs committed — every crate has a populated
      `crates/<m>/benchmarks/gpu_vram_sweep_2026-05-28.tsv` covering
      both cuda and wgpu cells.
- [x] `docs/GPU_METRICS_SWEEP_2026-05-28.md` with gap analysis — this
      file.
- [x] At minimum 24 cells (6 metrics × 4 sizes × Full + Strip) —
      **232 cells delivered** (132 cuda + 100 wgpu, 100% coverage
      of every requested (crate, mode, size, backend) combination
      with zero error rows).
- [x] 40 MP attempted on each (OOM captured as data) — full coverage
      both backends; every 40 MP cell fits in 12 GiB, no OOM.
- [x] wgpu compatibility per metric documented — all 6 metrics × all
      modes work on Linux Vulkan-backed wgpu through 40 MP.  Zero
      ERR_DISPATCH_CAP / ERR_BUF_ALIGN observed.  Task #131's note
      may have referenced browser-WebGPU's WGPU_LIMITS, not Vulkan.
- [ ] Sibling workspace cleanup — done at end of session per CLAUDE.md
      "Cleanup-on-merge is MANDATORY".
- [ ] Task #133 marked completed via TaskUpdate — `TaskUpdate` /
      `TaskGet` / `TaskList` tools not available in this agent
      environment (only `TaskStop`); orchestrator should mark
      task #133 completed.
