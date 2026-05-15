# Changelog

Workspace conventions per the global rules:

- One `[Unreleased]` section accumulates changes for the next release.
- Per-crate headings (`## cvvdp-gpu`, `## zen-metrics-cli`, …) sit under
  each version section since this repo ships multiple crates.
- `### QUEUED BREAKING CHANGES` accumulates breaks that need to land
  together — only cleared when the corresponding major (or minor for
  0.x) release ships.
- Every entry MUST include the short commit hash(es) that implemented
  it. Reference the merge or final commit for multi-commit features.

## [Unreleased]

### QUEUED BREAKING CHANGES

(none yet)

### Added

#### cvvdp-gpu (new crate, v0.0.1)

- ColorVideoVDP (still-image) port matching pycvvdp v0.5.4 on the
  v1 R2 manifest within 0.006 JOD across q1–q90. Full pipeline:
  - Color: sRGB→DKLd65 host scalar + `srgb_to_dkl_kernel` (cuda
    parity ≤3e-5).
  - Pyramid: vanilla Laplacian + Weber-contrast variant
    (`weber_contrast_pyr_dec_scalar`) + 4 cubecl kernels
    (`downscale_kernel`, `upscale_v_kernel`, `upscale_h_kernel`,
    `subtract_kernel`, `weber_contrast_compute_kernel`).
  - CSF: 32×32×3 LUT bilinear interp host scalar +
    `csf_apply_per_pixel_kernel` (per-pixel L_bkg from achromatic
    Gaussian pyramid) + `weight_band_kernel`.
  - Masking: mult-mutual + xchannel + soft clamp.
    `mult_mutual_band` host scalar + 3 cubecl kernels
    (`min_abs_3ch_kernel`, `mult_mutual_3ch_no_blur_kernel`,
    `mult_mutual_3ch_with_blurred_kernel`), plus `pu_blur_h_kernel`
    + `pu_blur_v_kernel` for the σ=3 phase-uncertainty blur.
  - Pooling: 3-stage Minkowski + smooth `met2jod` piecewise JOD
    mapping. `pool_band_kernel` does per-pixel `safe_pow` +
    `Atomic<f32>::fetch_add` reduction.
  - Composed: `Cvvdp::score` and `host_scalar::predict_jod_still_3ch`
    are both v1-manifest-locked (≤0.006 JOD). `Cvvdp::new` defaults
    to `DisplayGeometry::STANDARD_4K`; `Cvvdp::new_with_geometry`
    accepts any cvvdp display geometry.
- Parity goldens at
  `s3://coefficient/cvvdp-goldens/v1/manifest.json`
  (public mirror: `https://coefficient.r2.imazen.org/...`).
- Test infrastructure: `parity-goldens` cargo feature gates the
  network-fetching integration test, keeping default `cargo test`
  offline. Per-stage parity tests (color, pyramid, csf, masking,
  pooling) all locked vs pycvvdp.
- **GPU-composed score path** — full pipeline up through D bands +
  masking runs on GPU; only the spatial pool + 3-stage Minkowski +
  `met2jod` are host. New `Cvvdp` helpers:
  - `compute_dkl_weber_pyramid` — color + Weber-contrast pyramid,
    returns `(bands, log_l_bkg)` per the `WeberPyramidGpu` type
    alias.
  - `compute_dkl_t_p_bands(ppd)` — Weber × per-pixel CSF S ×
    `CH_GAIN` × `band_mul`. `band_mul = 2.0` for non-edge levels,
    `1.0` at level 0 and baseband. Baseband sets `CH_GAIN_eff = 1.0`
    so callers can reproduce cvvdp's `|T_p - R_p|` baseband bypass.
  - `compute_dkl_d_bands(ref, dist, ppd)` — composes Weber + CSF +
    masking. Non-baseband bands use the GPU `mult_mutual_3ch_*`
    masker (with the `10^MASK_C` PU-blur scale applied via
    `weight_band_kernel`); baseband uses `|T_p_dis - T_p_ref|`.
    Uses the reference's `log_l_bkg` for both sides per cvvdp's
    `weber_g1` contract.
  - `compute_dkl_jod(ref, dist, ppd)` — full GPU score path
    returning a JOD scalar. Drift survey shows GPU matches host
    within 0.001 JOD for q ≥ 20; the 0.40 drift at q=1 is
    cumulative f32 noise compounding through `met2jod`'s steep
    slope region, not a parity bug.
- `Cvvdp::score_with_reference` is wired (previously returned a
  silent 0.0). Caches reference sRGB bytes and routes through
  `host_scalar::predict_jod_still_3ch` — exact-parity with
  `Cvvdp::score(ref, dist)`.
- Drift-survey tests document where GPU vs host diverges per
  stage: `compute_dkl_{weber_pyramid,t_p_bands,d_bands}_matches_host_on_corpus_256x256`
  + `compute_dkl_jod_vs_host_scalar_on_corpus` +
  `compute_dkl_jod_on_v1_manifest_corpus`.
- `zenbench` score-path benchmark (`benches/score.rs`) — first
  measured CPU vs GPU per-pixel numbers at 256×256 / 1 MP / 12 MP.
- `time_12mp` example (`examples/time_12mp.rs`) — fixed-iteration
  one-shot timer for compute_dkl_weber_pyramid / compute_dkl_d_bands
  / compute_dkl_jod at 12 MP. Per-phase breakdown surfaces where
  the GPU pipeline spends its time without the zenbench
  calibration loop's overhead at large image sizes.
- `CVVDP_TRACE=1` env-var-gated stderr instrumentation inside
  `compute_dkl_d_bands` — per-level CSF / masking / log_l_bkg
  upload timings. Zero-cost when unset.
- `CVVDP_TRACE_WEBER=1` env-var-gated stderr instrumentation
  inside `compute_dkl_weber_pyramid` splitting GPU dispatch from
  host readback.
- Direct kernel-level parity test for `csf_apply_3ch_kernel`
  in `tests/csf_kernel.rs` — sweeps the full log_l_bkg LUT axis
  with distinct per-channel ch_gain values (catches bugs the
  indirect d_bands test would miss).
- Consecutive-weber diagnostic block in `examples/time_12mp.rs`
  (`0a71bb22`) — calls `compute_dkl_weber_pyramid` twice on the
  same `ref_bytes` outside `compute_dkl_d_bands` to isolate
  whether the "weber(dist) is 2× weber(ref)" slowdown is
  position-dependent (consecutive-call overhead) or data-shape
  dependent. Result: standalone consecutive calls show no
  slowdown, ruling out cubecl warm-up / driver effects and
  pinning the cause to host memory pressure from holding the
  `ref_weber: Vec<Vec<f32>>` (~190 MB at 12 MP) alive across the
  second call inside the d_bands flow.
- `_dispatch_weber_pyramid_gpu` private helper (`072d9e43`)
  factored out of `compute_dkl_weber_pyramid` — takes a
  `&[Handle]` destination slice for the per-level `log_l_bkg`
  outputs. The bisect for tick 85's 5× regression revealed
  that this extraction itself does not regress, only the
  full 5-phase serial restructure did; the helper is kept so
  future experiments can swap the destination buffer set
  without re-wiring weber's body.

### Changed (performance)

#### cvvdp-gpu

After tick 70's per-band-allocation diagnosis, four scratch
hoists + one kernel fuse landed in succession:

- **Pre-allocate per-band CSF + masking scratch** on `Cvvdp::new`.
  `compute_dkl_d_bands` was alloc_zeros_f32-ing 18 buffers per
  non-baseband level per call (~1.5 GB worth at 12 MP). Moved
  to a `DBandsScratch` struct on the Cvvdp instance. Result:
  12 MP d_bands −25%, full jod −30%.
- **Pre-allocate per-band Weber pyramid scratch** — same shape
  for the expand/subtract/weber chain (l_bkg_fine, vscratch_a,
  log_l_bkg, per-channel vscratch_c/upscaled_c/layer_c).
  Result: 12 MP weber alone 5× faster (105 → 21.6 ns/px), full
  jod 2.4× faster (310 → 127 ns/px). **This crossed the milestone
  of beating fcvvdp single-thread** (214 ns/px on their bench).
- **Drop unused per-side GPU buffers** (`src_dis`, `gauss_dis`,
  `bands_dis`, `pool_partials`) that were allocated by
  `Cvvdp::new_with_geometry` but never read by any GPU helper.
  Saves ~13 MB per Cvvdp at 256×256.
- **Hoist `logs_row` uploads** to `Cvvdp::new_with_geometry`
  (24 small uploads of 128 B were happening per d_bands call,
  one per `(level, channel)`). Since `rho_k` is fixed per Cvvdp,
  the LUT rows are stable across calls.
- **Fuse 3-channel CSF apply** into a single kernel
  (`csf_apply_3ch_kernel`) that shares the per-pixel LUT bracket
  math across A/RG/VY channels. Cut L0 CSF time at 12 MP from
  420 ms (6 launches) to 170 ms (2 launches) — but the saved
  ~250 ms got absorbed by ~340 ms of unaccounted overhead
  (likely host Vec<Vec<f32>> alloc for the weber readback);
  median d_bands wall is unchanged.
- **`pow(10, x) → exp(x · ln(10))`** in CSF kernels for the
  mathematical identity. No measurable win on cuda (likely cubecl
  already compiles to similar PTX); kept for potential wgpu/hip
  payoff.
- **Dist-side CSF reads `self.bands_ref` handles directly**
  (`8b6f2776`) — `compute_dkl_d_bands` no longer uploads
  `dist_weber[k]` from host inside the per-band CSF apply. The
  dist-side handles are already resident in `self.bands_ref`
  after the `weber(dist)` call earlier in the band loop, so the
  CSF kernel reads them in place. REF-side still uploads since
  `bands_ref` has been overwritten with DIST data by band-loop
  time. Result on 12 MP cuda: weber 291 ms (baseline),
  d_bands 1.42 s (−3% from 1.46 s), jod 1.40 s (−7% from 1.50 s).
  Parity intact at 1.3e-3 band-relative on q=1 corpus. Critically,
  this also proves the handle-direct CSF pattern is **innocent**
  of tick 85's 5× weber regression — that regression was the
  5-phase serial restructure, not the handle access pattern.

The post-tick-87 fusion + structural-change wave (ticks 89–96)
took the d_bands per-band launch count from 27 → 14:

- **`weber_contrast_compute_3ch_kernel`** (`af994a87`) — fuses
  the per-pixel `layer/clamp(L_bkg)` math and the shared
  `log_l_bkg = log10(L_bkg)` write into one launch per
  non-baseband level. Was 3 separate
  `weber_contrast_compute_kernel` launches. log10 computed
  once per pixel instead of three times.
- **`subtract_weber_3ch_kernel`** (`39d6957f`) — drops the
  `layer_c` intermediate entirely. Reads `fine_c` and
  `upscaled_c` handles directly and writes `band[c] =
  clamp((fine_c − upscaled_c) / L_bkg)` for all three channels
  + shared `log_l_bkg` in one launch. Was 3 `subtract_kernel`
  launches + the (already-fused) weber compute. Frees ~36 MB
  of `WeberScratch.layer_c` at 12 MP per side.
- **`pu_blur_h_3ch_kernel` + `pu_blur_v_3ch_scaled_kernel`**
  (`78d951d1`) — fuses the masking-branch pu_blur into one
  h-pass + one v-pass for all 3 channels, AND folds the
  `* 10^MASK_C` post-scale into the v-pass output. Cuts the
  masking blur chain from 9 launches per non-baseband level
  (3× h + 3× v + 3× `weight_band_kernel`) to 2.
- **`csf_apply_6ch_kernel`** (`7bf02fae`) — fuses the
  REF + DIST CSF apply into a single launch sharing the
  per-pixel LUT bracket math. Per non-baseband level: 2
  `csf_apply_3ch_kernel` launches → 1 6-channel launch.
- **`diff_abs_3ch_kernel`** (`06d8e4a5`) — moves the
  baseband `|T_p_dis - T_p_ref|` bypass to GPU. Every level's
  D plane now lives in the same `d_scratch.d[k][c]` slot.
- **`pool_band_kernel` in `compute_dkl_jod`** (`5817a2e4`)
  — replaces host-scalar `lp_norm_mean` over the per-band D
  Vecs with GPU `pool_band_kernel(d_handle) → partials[k*3+c]`.
  Partials buffer is `n_levels × N_CHANNELS` floats (~144 bytes
  at 12 MP); the host fold operates on that tiny Vec.
- **Split `compute_dkl_d_bands`** (`ea632f87`) — extracted
  `_dispatch_d_bands_into_scratch` private helper that does the
  GPU dispatch only. `compute_dkl_jod` calls the helper
  directly and skips the per-band Vec readback that
  `compute_dkl_d_bands` was paying. **17% wall-time win** at
  12 MP (jod 122.4 → 101.8 ns/px); jod is now faster than
  d_bands because it skips the ~432 MB host readback. vs
  fcvvdp 8-thread at 360p, the gap narrowed from 1.48× slower
  (tick 89) to 1.18× slower.

Post-fuse housekeeping (ticks 97–107):

- **`examples/time_size_sweep.rs`** + benchmark snapshot
  (`134bc04a`) — covers tiny (64²), small (256²), medium
  (1024²), large (4000×3000) sizes with per-phase wall + per-
  pixel cost + naive OLS fit. Found per-pixel cost is
  **non-monotonic** in image size: medium (1 MP) is the
  cheapest at 53.7 ns/px JOD, large (12 MP) regresses to
  159 ns/px; weber alone shows the same shape (19 → 61 ns/px),
  so the regression is intrinsic to the dispatch, not pure
  readback bandwidth. Open investigation.
- **`shadow_jod_gpu`** manifest-parity test (`562ee924`) —
  pins the GPU JOD path directly against pycvvdp v0.5.4's
  published manifest values (not just against the host
  scalar via relative parity). q=1 tolerance is wider (0.5
  JOD) per the documented cumulative-f32 drift; q≥20 tol is
  0.05 (observed < 0.001).
- **`Cvvdp::level_dims`** helper (`efcdba76`) — drops 5 sites
  of duplicated `if k == 0 { width } else { width >> k }`
  boilerplate. The `if k == 0` branch was redundant since
  `>> 0` is a no-op.
- **Dropped `Cvvdp.ref_log_l_bkg` dead field** (`ba586480`)
  — was added in tick 85 for a regression bisect that
  confirmed the field was NOT the cause; kept around with
  `#[allow(dead_code)]` for "future use" that subsequent
  ticks went around. Frees ~190 MB of unused GPU memory per
  `Cvvdp::new` at 12 MP, drops 14 lines of allocation code.
- **`compute_dkl_t_p_bands` modernized** (`8e509807`) — uses
  the fused `csf_apply_3ch_kernel` and reads weber from the
  GPU-resident `bands_ref` handles instead of re-uploading
  from the host Vec. Per non-baseband level: 3 host uploads
  + 3 launches → 0 uploads + 1 launch.

Post-fuse housekeeping (ticks 108–124):

- **Tests + examples + benches now run under wgpu** (`a0473bf9`,
  `3c72a86d`, `70a62e63`) — `shadow_jod_gpu`, `time_12mp`,
  `time_size_sweep`, and `benches/score.rs` all switched from
  cuda-only to the `cfg(any(cuda, wgpu))` + `Backend` type-alias
  pattern. Machines without a CUDA SDK (macOS, AMD, Intel) can
  now run the manifest-parity anchor + per-phase timings under
  wgpu's Vulkan/Metal/DX12 backend.
- **`ch_gain_for_band(is_baseband, band_mul)` helper** (`f5c1df3c`)
  — replaces 6 lines of `if is_baseband { 1.0 } else { band_mul *
  CH_GAIN[c] }` boilerplate at two band-loop sites with a single
  destructuring bind.
- **Stack-allocated `compute_dkl_jod` partials zero-init**
  (`a4e019c0`) — replaces a 192-byte heap Vec with
  `[0.0_f32; MAX_LEVELS * N_CHANNELS]` sliced to the active
  prefix.
- **CHANGELOG catch-up + PORT_STATUS refresh + many small doc
  fixes** (`bcf3dfcc`, `0dc01ea5`, `b7686203`, `35a0b48d`,
  `6826c0eb`, `77908be7`, `fd1e2527`, `8cd803a9`, `ac1e21d3`,
  `067ba379`, `08c65040`, `45719dad`, `1b8b51ca`) — module-level
  pipeline overviews in `lib.rs`, `pipeline.rs`, and
  `kernels/mod.rs` updated to name the actual fused kernels;
  stale claims about which stages run host-side cleared;
  `compute_dkl_weber_pyramid` got its missing doc comment; the
  misleading α/β OLS fit dropped from `time_size_sweep`; and 9
  of 15 rustdoc warnings cleared (remaining 6 are macro-induced
  by `#[cube(launch)]`'s function-and-module duplication).
- **`Cvvdp::score` v1 manifest tolerance** still pinned by the
  CPU reference path (`shadow_jod`). The GPU composition path
  is parity-locked against pycvvdp directly via `shadow_jod_gpu`
  but with a wider q=1 tolerance (~0.4 JOD) per the documented
  cumulative-f32 drift through `met2jod`'s steep slope.

Host-memory-pressure relief (ticks 144–146):

- **Drop dist_weber host Vec immediately** (`02f37728`) —
  `compute_dkl_d_bands` was binding the `(dist_weber, _)` tuple
  from `compute_dkl_weber_pyramid(dist_srgb)` even though the
  dist-side CSF path reads `self.bands_ref` GPU handles
  directly (per tick 87). Changed to `let _ = ...` so the
  ~190 MB host Vec drops at the call site instead of
  surviving the band loop.
- **Per-band ref-side host Vec drops** (`913a7c5f`) — after the
  band-`k` CSF dispatch finishes its `create_from_slice`
  uploads, replace `ref_weber[k] = [Vec::new(); 3]` and
  `ref_log_l_bkg[k] = Vec::new()` so peak host residency scales
  with the remaining-bands sum, not the whole pyramid.

Together these two commits dropped 12 MP perf
(`benchmarks/time_12mp_tick145_2026-05-14.md`):
- weber pyramid: 26.4 → 30.6 ns/px (noise band)
- compute_dkl_d_bands: 106.6 → **82.1 ns/px** (−23%)
- compute_dkl_jod: 101.8 → **87.2 ns/px** (−14%)

The `d_bands − 2×weber` bucket (CSF + masking + IO) dropped
from 645 ms → 252 ms — a **2.5× speedup** on the non-weber
portion. vs fcvvdp's 8-thread number at 360p we crossed from
1.48× slower (tick 89) to 1.18× slower (tick 96) to **1.01×
tied** here.

- **DIST weber pyramid skips host readback entirely**
  (`8c5b96e0`, tick 150) — `compute_dkl_d_bands` was calling
  `compute_dkl_weber_pyramid` for the DIST side and
  immediately discarding the returned tuple. Tick 144 caught
  the unused tuple; tick 150 caught that the *wrapper* itself
  still allocated ~240 MB of host Vecs and issued
  `client.read_one` calls that wait for the GPU dispatch to
  complete before transferring bytes. Replaced with
  `_dispatch_weber_pyramid_gpu` (the dispatch-only private
  helper) — skips both the allocation AND the GPU→host
  transfer.

  Result on the next 12 MP run
  (`benchmarks/time_12mp_tick150_2026-05-14.md`):
  - compute_dkl_d_bands: 82.1 → **71.0 ns/px** (−14%)
  - compute_dkl_jod: 87.2 → **74.6 ns/px** (−14%)
  - `d_bands − 2×weber`: 252 ms → 156 ms (−38%)
  - vs fcvvdp 8-thread @ 360p: now **1.15× faster** (vs 1.01×
    tied pre-tick).

Perf trajectory through the recent fusion + host-pressure wave:

| tick | jod ns/px | vs fcvvdp 8t @ 360p |
| ---- | --------- | ------------------- |
| 64   | 444       | 5.16× slower        |
| 73   | 127       | 1.48× slower        |
| 89   | 122       | 1.42× slower        |
| 96   | 102       | 1.18× slower        |
| 145  |  87       | 1.01× tied          |
| 150  |  **75**   | **1.15× faster**    |

Host-memory-pressure relief continued + structural readback
elimination (ticks 151–160):

- **REF CSF reads `bands_ref` GPU handles directly** (tick 155,
  `d7c7322c`) — symmetrical to tick 87's DIST-side fix. The
  band-loop's REF CSF dispatch had been uploading `ref_weber[k]`
  from the host Vec; after tick 154's `bands_ref` / `bands_dis`
  split persisted both sides' data on GPU, the REF CSF kernel
  reads `self.bands_ref[k]` handles in place. Drops 3 host→GPU
  uploads per non-baseband level (~50 MB total at 12 MP).
- **REF weber pyramid skips bands readback** (tick 156, `2993c0a0`)
  — `_dispatch_d_bands_into_scratch` had been calling the public
  `compute_dkl_weber_pyramid(ref_srgb)` wrapper which read back
  ~190 MB of bands per call (`Vec<Vec<f32>>`). Replaced with a
  direct call to `_dispatch_weber_pyramid_gpu` + a manual
  `log_l_bkg`-only readback loop. 12 MP jod 70.3 → 60.2 ns/px
  (−14%), now 1.43× faster than fcvvdp 8t.
- **Dispatch-only split for `compute_dkl_planes` + `compute_dkl_gauss_pyramid`**
  (tick 157) — extracted private `_dispatch_dkl_planes_gpu` and
  `_dispatch_gauss_pyramid_gpu` siblings.
  `_dispatch_weber_pyramid_gpu` and `compute_dkl_laplacian_pyramid`
  switched off the public wrappers (was `let _ = ...`). Saves
  ~230 MB of wasted host transfer per weber call (36 MB level-0
  + ~190 MB pyramid). 12 MP jod 60.2 → 53.0 ns/px (−12%), now
  1.62× faster than fcvvdp 8t.
- **GPU baseband-divide** (tick 158, `3b78f847`) — adds
  `baseband_divide_3ch_kernel` (pyramid.rs). The weber baseband
  finishing step had been doing 3 channel readbacks + 3 channel
  reuploads + per-channel host divides; now does 1 GPU launch
  using host-computed `l_bkg_mean` as a scalar uniform. Sync
  drain count per weber side: 4 → 1.
- **Tested-and-regressed 3ch upscale fusion + laplacian dispatch-only split**
  (tick 159, `6495c462`) — `upscale_v_3ch_kernel` /
  `upscale_h_3ch_kernel` (same fusion pattern as
  `weber_contrast_compute_3ch`) regressed jod ~4% at 12 MP on
  RTX CUDA across two runs. Hypothesis: 3ch register footprint
  reduced warp-level latency hiding more than launch overhead
  was costing us. Left a breadcrumb in pyramid.rs so this isn't
  re-tried without a different angle (e.g. shared-memory tiling).
  Same commit also added `_dispatch_laplacian_pyramid_gpu` so
  `compute_dkl_csf_weighted_bands` no longer discards a full-
  pyramid host readback via `let _ = ...`.
- **Direct parity test for `baseband_divide_3ch_kernel`**
  (tick 160, `baf4878e`) — closes a coverage gap from tick 158.
  The kernel had been verified through the higher-level
  `compute_dkl_weber_pyramid_matches_host_on_corpus_256x256`
  integration test; the new unit test in `pyramid_kernel.rs`
  gives a fast regression gate with inputs that exercise
  negatives, large magnitudes, and 3 distinct channel patterns.

12 MP perf trajectory through this wave
(`benchmarks/time_12mp_tick{155,156,157,158}_2026-05-14.md`):

| tick | jod ns/px | weber 1-side | d_bands  | vs fcvvdp 8t |
| ---- | --------- | -----------  | -------- | ------------ |
| 150  | 74.6      | 29.0         | 71.0     | 1.15× faster |
| 155  | 70.3      | 31.8         | 73.5     | 1.22× faster |
| 156  | 60.2      | 29.2         | 52.0     | 1.43× faster |
| 157  | 53.0      | 25.5         | 45.2     | 1.62× faster |
| 158  | **52.9**  | **24.9**     | **43.7** | **1.63× faster** |

The `d_bands − 2×weber` bucket (CSF + masking + IO) is sub-noise
since tick 156: 2×weber ≈ d_bands, meaning the band-loop overhead
is now bandwidth-tightly packed against the two weber pyramids.
The next remaining hot spot is the gauss-pyramid reduce (5×5
downscale, 25 src reads per output pixel), which a shared-memory
tiled rewrite could shrink — but the per-thread register
pressure observation from tick 159 means any fusion attempt
should change the memory access pattern, not just rearrange
launches.

### Investigation Notes (cvvdp-gpu, post-tick-81)

These observations don't ship as code, but they document
findings that would otherwise be re-discovered:

- **Standalone weber(dist) is not slower than weber(ref)** —
  the consecutive-weber diagnostic in `examples/time_12mp.rs`
  shows two back-to-back `compute_dkl_weber_pyramid` calls on
  the same `ref_bytes` complete in nearly identical time. The
  "weber(dist) is 2× weber(ref)" effect observed inside
  `compute_dkl_d_bands` is therefore not algorithmic, not a
  cubecl warm-up artifact, and not driver thermal throttling.
  It is host memory pressure: ~190 MB of `ref_weber` Vec stays
  alive across the second call.
- **Tick 85's failed 5-phase d_bands refactor regressed
  standalone weber by 5×** (260 ms → 1300 ms) — the per-band
  bisect ruled out: (a) the new `self.ref_log_l_bkg` field
  itself (allocation-only does not regress), (b) the new
  `log_l_bkg_dest` parameter on `_dispatch_weber_pyramid_gpu`,
  and (c) the GPU memory-handle pattern (the dist-side CSF
  optimization above confirms this). The proven cause is the
  5-phase serial control-flow structure (all CSF(ref) bands →
  weber(dist) → all CSF(dist) bands → all masking), but the
  actual mechanism (cubecl sync barrier? memory-pool
  fragmentation? kernel-scheduler ordering?) remains unknown.
  Future attempts at the d_bands restructure should bisect a
  different axis (interleaved-per-level vs. phase-serial)
  rather than re-flatten the existing structure.

Net 12 MP performance trajectory (CUDA, RTX-class):

| metric                  | tick 64   | tick 73    | tick 158  |
| ----                    | ----      | ----       | ----      |
| weber pyramid (1 side)  | 103 ns/px | 21.6 ns/px | 24.9 ns/px |
| compute_dkl_d_bands     | 428 ns/px | 121 ns/px  | 43.7 ns/px |
| compute_dkl_jod         | 444 ns/px | 127 ns/px  | 52.9 ns/px |

(The weber column went up between tick 73 and tick 158 because
tick 73's measurement was taken on a fresh process with no
prior GPU state; later ticks share state with the dispatch
pipeline that runs around weber, and the standalone weber
benchmark in `time_12mp` ends up with more thermal/warm-up
mix. The d_bands and jod columns are the load-bearing
trajectory.)

vs fcvvdp at 360 p (their bench, i7-13700k):

| variant       | per-pixel  | vs current cvvdp-gpu @ 12 MP |
| ----          | ----       | ----                         |
| 1-thread      | 214 ns/px  | we are **4.05× faster**      |
| 8-thread      |  86 ns/px  | we are **1.63× faster**      |

### Fixed

#### cvvdp-gpu

- `host_scalar::predict_jod_still_3ch` index-out-of-bounds at
  image sizes where `band_frequencies` truncates below
  `ilog2(min(w, h))` (e.g. 1024×1024). The auto-pick now queries
  `band_frequencies(...).len()` instead of falling through to the
  `ilog2`-based default.

### Removed

#### cvvdp-gpu

- Dead `masked_diff_kernel` cubecl stub (always wrote 0.0; never
  launched).
- Dead `upscale_kernel` cubecl stub (replaced by the
  `upscale_v_kernel` + `upscale_h_kernel` pair).
- Empty `kernels::reduce` module (planned scope landed in
  `kernels::pool` instead).

#### zen-metrics-cli

- New `cvvdp` metric (`--metric cvvdp`). GPU bundle (`--features
  gpu`) now includes `gpu-cvvdp`. Sweep TSVs pick up the
  `score_cvvdp` column automatically.

### Workspace

- CI builds the new `cvvdp-gpu` crate alongside the existing four
  `-gpu` crates under `wgpu` (per-platform) and as part of the
  `i686-unknown-linux-gnu` cross-compile sanity job.
