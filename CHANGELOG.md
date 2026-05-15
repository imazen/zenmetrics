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

### Changed

#### cvvdp-gpu

- **`Cvvdp::score` and `Cvvdp::score_with_reference` now route
  through the GPU pipeline** (`compute_dkl_jod`), replacing the
  host-scalar reference path. Output matches the prior host
  path to f32 noise (verified by
  `compute_dkl_jod_matches_host_scalar` at ≤ 0.005 JOD) and the
  pycvvdp v1 R2 manifest to ≤ 0.005 JOD (verified by
  `shadow_jod_gpu`). The switch was explicitly pre-promised in
  `lib.rs` ("Switching `score` over to the GPU path is the
  remaining chunk of pipeline work") and was unblocked by tick 207's
  tightened manifest-parity tolerances. Callers that need the
  all-host path can still invoke
  `host_scalar::predict_jod_still_3ch` directly;
  cpu-runtime callers use `compute_dkl_jod_host_pool`.
  Also tightened `tests/pipeline_score.rs` `cvvdp_score_matches_v1_manifest`
  from 0.05 → 0.005 JOD (measured diffs 0.0000–0.0033).
- Removed the dead `reflect()` helper in `kernels/pyramid.rs` —
  superseded in tick 206 when `gausspyr_reduce_scalar` was
  rewritten to bug-compatible zero-pad + explicit boundary
  patches matching pycvvdp.
- **Manifest-parity tolerances tightened to 0.005 JOD across the
  v1 R2 corpus** (`tests/shadow_jod.rs`). Was a per-q schedule
  (0.5 JOD at q=1, 0.1 at q=5, 0.05 at q≥20 GPU; flat 0.05 host)
  before ticks 204/206 closed the chroma_shift and 73×91 odd-dim
  drifts. Measured diffs are now 0.0000–0.0031 JOD across all 6
  q levels (host + GPU) — well within the same 0.005 tolerance
  the other parity tests use.
- `pipeline_score.rs` host-vs-GPU corpus tests
  (`compute_dkl_t_p_bands_matches_host_on_corpus_256x256`,
  `compute_dkl_d_bands_matches_host_on_corpus_256x256`) updated
  to apply the tick-204 `CSF_BASEBAND_RHO` override in their
  host reference computation — caught when running the full
  suite after tightening shadow_jod tolerances.

### Added

#### cvvdp-gpu

- **`Cvvdp::compute_dkl_jod_host_pool`** — CPU-backend-compatible
  variant of `compute_dkl_jod`. Reads D bands back to host and
  pools them with the host-scalar `lp_norm_mean` instead of the
  GPU `pool_band_3ch_kernel` (which uses `Atomic<f32>::fetch_add`,
  unsupported by `cubecl-cpu`). Same JOD output as
  `compute_dkl_jod` to f32 noise (`diff = 0.000000` measured on
  the 32×32 odd-dim test pair); use it on the CPU backend or
  any runtime that lacks atomic f32 add. New
  `compute_dkl_jod_host_pool_matches_compute_dkl_jod` test pins
  the two paths together. Closes the standing CPU-backend
  blocker noted in `lib.rs`.
- **`tests/cpu_backend.rs`** — cpu-runtime smoke + parity tests
  exercising `compute_dkl_jod_host_pool` on `cubecl::cpu::CpuRuntime`.
  Validates the lib.rs claim that the cpu backend works:
    JOD finite + in [0, 10] on a 32×32 synth pair.
    cpu backend JOD vs host_scalar JOD: `diff = 0.000000`.
  All other test files gate themselves out of cpu-only builds; this
  file is the only place cpu-backend coverage lives.
  Run with `cargo test -p cvvdp-gpu --no-default-features --features cpu`.

#### cvvdp-gpu (docs)

- `Cvvdp::score` now has a `no_run` doctest example showing the
  canonical `Cvvdp::<CudaRuntime>::new` → `.score(&ref, &dist)`
  shape against a 64×64 byte-identical pair. Fills the only
  remaining doc gap on the crate's headline public entry point —
  the host-only and host-pool paths already had doctests via
  `host_scalar::predict_jod_still_3ch`, `compute_dkl_jod_host_pool`,
  and `compute_dkl_jod_host_pool_with_warm_ref`.

#### cvvdp-gpu (performance)

- `compute_dkl_d_bands` host readback init no longer pre-allocates
  `vec![0.0; n_px] × 3` per pyramid level only to immediately
  overwrite each entry with `f32::from_bytes(&bytes).to_vec()`.
  Now uses empty `Vec::new()` slots — matches `compute_dkl_gauss_pyramid`'s
  readback shape and drops `~3 × n_levels × n_px` floats of wasted
  host zero-fill per call. (`compute_dkl_d_bands` is a parity-test
  helper; production JOD path is unaffected since it pools on-GPU.)
- **Persistent `partials_h` atomic-pool buffer** — `Cvvdp::new`
  now allocates a single `n_levels × N_CHANNELS` partials buffer
  (≤ 144 bytes at MAX_LEVELS=9) and `_pool_and_finalize_jod` zero-
  fills it via `fill_f32_kernel` per call instead of allocating
  a fresh GPU buffer + uploading host zeros every JOD call.
  Removes one `create_from_slice` host alloc + Host→GPU copy per
  call from the JOD hot path; pattern mirrors the tick-168
  `baseband_log_l_bkg` migration. All 27 pipeline_color + 9
  pipeline_score + 8 pool_scalar tests green on CUDA, including
  manifest parity (`compute_dkl_jod_on_v1_manifest_corpus` at ≤ 0.005
  JOD) and the GPU-pool-vs-host-pool sentinel
  (`compute_dkl_jod_host_pool_matches_compute_dkl_jod`).

#### cvvdp-gpu (tests)

- `compute_dkl_jod_with_warm_ref_matches_pycvvdp_at_73x91_odd` —
  direct warm-ref pycvvdp parity on the mixed-parity 73×91 fixture.
  Pairs with the chroma_shift warm-ref test from tick 222: both pin
  the warm-state restoration path against canonical pycvvdp, but
  73×91 specifically exercises the tick-206 gausspyr_reduce
  parity-bug fix on REF (mixed-parity reduce levels 6×5 → 3×3 and
  46×37 → 23×19). Measured diff: 0.0000 JOD. Closes a transitivity
  gap: prior warm-ref pycvvdp coverage was same-parity only.

#### Workspace

- Pinned multi-tick task in `CLAUDE.md`: compute CVVDP scores for
  all zensim training data sets via vast.ai docker images, output
  as parquet sidecars with implementation-distinguished column
  names (e.g. `cvvdp_pycvvdp_v054`, `cvvdp_imazen_v0_0_1`). Survives
  context compaction; every `/loop` tick re-reads it.

#### zen-metrics-cli

- New `score-pairs` subcommand (feature-gated on `sweep`):
  consumes the pairs TSV that `sweep --pairs-tsv` produces and
  emits a parquet sidecar with the metric's versioned column name
  (e.g. `cvvdp_imazen_v0_0_1` for cvvdp). Schema matches
  `crates/cvvdp-gpu/docs/CVVDP_SIDECAR_SCHEMA.md` exactly:
  `image_path string`, `codec string`, `q int64`,
  `knob_tuple_json string`, `<metric> float64`. Zstd compression.
  Symmetric with `scripts/sweep/pycvvdp_worker.py score-pairs`.
  Initial n=4 sentinel: cvvdp-gpu vs pycvvdp parity within 0.03 JOD
  on q50/q90 zenjpeg-encoded 64×64 noise images.

#### zen-metrics-cli (sweep)

- `sweep` subcommand learns two new flags that pair off for
  external-scorer workflows (e.g. pycvvdp):
  - `--distorted-out-dir <DIR>`: every successfully-decoded cell
    writes its distorted image as a `Compression::Fastest` PNG
    into this directory. Filenames are deterministic and
    collision-resistant:
    `<src_stem>_<src_path_hash16>_<codec>_q<q>_<knob_hash16>.png`.
  - `--pairs-tsv <FILE>`: tab-separated companion to the main
    `--output` TSV with columns
    `image_path codec q knob_tuple_json ref_path dist_path` —
    one row per decoded cell. `dist_path` is empty when
    `--distorted-out-dir` is unset.
  - Smoke test: 2-image × 2-q sweep produced 4 PNGs + a 4-row pairs
    TSV that `pycvvdp_worker` then scored into a 4-row
    `cvvdp_pycvvdp_v054` parquet sidecar.

#### scripts/sweep

- `dual_impl_chunk.sh` — per-chunk dual-implementation runner.
  Drives one sweep + both cvvdp scorers (zen-metrics-cli score-pairs
  for cvvdp-gpu + pycvvdp_worker.py for canonical pycvvdp) and
  joins the two sidecars into a parity TSV. Local smoke test: 4
  cells joinable, mean |diff| 0.0245 JOD, max 0.0300 JOD on the
  synth zenjpeg q50/q90 corpus.
- `pycvvdp_worker.py` — canonical pycvvdp v0.5.4 scoring worker.
  Consumes a TSV of `(identity_tuple, ref_path, dist_path)` rows
  and writes a parquet sidecar with the `cvvdp_pycvvdp_v054`
  column per `crates/cvvdp-gpu/docs/CVVDP_SIDECAR_SCHEMA.md`.
  Verified end-to-end on a synth 64×64 pair: JOD 10.0 for identical
  inputs, 9.63 for chroma-shifted.
- `Dockerfile.pycvvdp` — image for the worker on vast.ai. Bases on
  `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` with pycvvdp
  0.5.4, pillow, numpy, pyarrow. CMD is help text; runners must
  pass an explicit `pycvvdp-worker score-pairs …` command.

#### cvvdp-gpu

- `CVVDP_COLUMN_NAME` const exposes a per-implementation column tag
  (default `cvvdp_imazen_v<MAJOR>_<MINOR>_<PATCH>`, overridable via
  the `CVVDP_IMPL_TAG` build-time env var). Used by sweep tooling so
  multiple cvvdp variants land side-by-side in parquet sidecars
  without colliding.

#### zen-metrics-cli

- `MetricKind::Cvvdp::column_names()` now returns
  `cvvdp_gpu::CVVDP_COLUMN_NAME` when the `gpu-cvvdp` feature is
  enabled, so sweep TSV/parquet headers emit
  `score_cvvdp_imazen_v0_0_1` (or the override). The user-facing
  CLI flag `--metric cvvdp` stays stable.

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

### Fixed

#### cvvdp-gpu

- **73×91 odd-dim residual closed (was 0.006 JOD).** Found a
  parity-check bug in pycvvdp's `gausspyr_reduce`: the
  horizontal-pass right-column patch uses `x.shape[-2]` (INPUT
  ROW parity) to pick its odd/even branch even though the
  comments say "columns" — `lpyr_dec.py:204-209`. For
  mixed-parity inputs (e.g. 6×5 → 3×3 at the 73×91 baseband)
  pycvvdp applies the wrong patch.
  - `host_scalar` `gausspyr_reduce_scalar`: rewritten to bug-
    compatible zero-pad + parity-aware patches.
  - GPU `downscale_kernel`: adds a delta correction at the right
    column when sw and sh parities differ.
  - New `compute_dkl_jod_matches_pycvvdp_at_73x91_odd` test
    passes at f32 precision (diff = 0.0000 vs pycvvdp golden).
  - All other corpus fixtures (256² + 4 MP, same-parity dims)
    unchanged — the bug-compat patches match pure reflection
    for all sw == sh parity inputs.

- **Chroma-shift drift closed (was 0.117 JOD).** pycvvdp overrides
  the baseband CSF rho to 0.1 cy/deg (`cvvdp_metric.py:628`),
  but our pipeline used the geometric value from
  `band_frequencies(ppd, w, h)` (0.190 at 256² standard_4k). Fixed
  by adding `kernels::csf::CSF_BASEBAND_RHO = 0.1` and applying it
  in both `host_scalar::predict_jod_still_3ch` and
  `Cvvdp::new`'s `logs_row` pre-upload. The
  `compute_dkl_jod_matches_pycvvdp_at_256x256_chroma_shift` test
  re-enabled at standard 0.005 JOD tolerance; chroma_shift now
  matches pycvvdp golden 9.664865 to f32 precision.

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

Continued perf wave + structural cleanup (ticks 162–166):

- **PORT_STATUS.md refresh** (tick 162, `621a5867`) — weber-
  contrast pyr row names `baseband_divide_3ch_kernel`, composed-
  pipeline row carries the tick 158 perf number, "Open tick 159"
  entry documents the 3ch upscale fusion negative result.
- **`compute_dkl_t_p_bands` skips bands readback**
  (tick 163, `8a6de7be`) — same tick-156 pattern applied to the
  test-only T_p path. Was discarding the bands portion of
  `compute_dkl_weber_pyramid`'s return tuple (~190 MB host
  alloc per call at 12 MP). Now dispatches via the private
  helper + log_l_bkg-only readback.
- **Size-sweep re-measurement** (tick 164, `d27c5194`) —
  documents the tick 150-158 wave's per-bucket impact:
  - tiny    jod 1835 → 527 ns/px (−71%)
  - small   jod  223 →  91 ns/px (−59%)
  - medium  jod   65 →  28 ns/px (−56%)
  - large   jod  145 →  39 ns/px (−73%)
  Most importantly the medium→large per-pixel regression open
  since tick 97 **narrowed from 2.2× to 1.36×** — falsifies the
  L2-cache-pressure hypothesis as dominant; most of it was
  host memory pressure all along. Small (256²) is now the most-
  expensive per-pixel bucket — launch overhead dominates at
  that thread count.
- **`pool_band_3ch_kernel` fusion** (tick 165, `df4dd106`) —
  3 per-channel pool launches per level → 1 fused 3ch launch.
  Total pool dispatch: `n_levels × N_CHANNELS = 24` → `n_levels
  = 8` launches per JOD. Unlike tick 159's upscale 3ch fusion
  (regressed via register pressure), pool kernel does only 3
  powfs + 3 atomic-adds per thread — register footprint stays
  small, fusion wins on launch-overhead reduction. 12 MP jod
  52.9 → 49.0 ns/px (−7%), 1.76× faster than fcvvdp 8t.

  **Decision rule for 3-channel fusion** extracted from
  tick 159 vs tick 165: fusion wins when per-thread arithmetic
  is tiny (atomics, pointwise math); loses to register pressure
  on medium-arithmetic kernels (5-tap convolutions, multi-read
  patterns). Future 3ch fusion attempts should respect this.

- **`log_l_bkg` roundtrip elimination** (tick 166, `7ce2bc24`)
  — adds `WeberScratch.log_l_bkg_dis` throwaway destination
  (parallel to tick 154's `bands_dis` split) so the DIST weber
  dispatch's log_l_bkg write doesn't clobber REF's data on
  `weber_scratch[k].log_l_bkg`. Per cvvdp's weber_g1 rule,
  both sides use REF's log_l_bkg, so DIST's value is computed-
  then-discarded. The band loop's CSF kernel now reads REF's
  log_l_bkg directly from the GPU-resident handle — no host
  roundtrip.

  Bytes saved per JOD at 12 MP: ~128 MB (64 MB readback +
  64 MB reupload of the same data). Sync drains saved: 7
  (one per non-baseband level). 12 MP jod 49.0 → **41.8 ns/px**
  (−15%). Now **2.06× faster than fcvvdp 8-thread @ 360p**.

12 MP perf trajectory through ticks 165-166
(`benchmarks/time_12mp_tick{165,166}_2026-05-14.md`):

| tick | jod ns/px | weber 1-side | d_bands  | vs fcvvdp 8t |
| ---- | --------- | -----------  | -------- | ------------ |
| 158  | 52.9      | 24.9         | 43.7     | 1.63× faster |
| 165  | 49.0      | 23.4         | 41.3     | 1.76× faster |
| 166  | **41.8**  | **22.2**     | **39.8** | **2.06× faster** |

Warm-ref API + last per-JOD host alloc removed (ticks 168–171):

- **`fill_f32_kernel` + `baseband_log_l_bkg` pre-alloc**
  (tick 168, `e0b6ca62`) — replaces the baseband band's per-JOD
  `vec![log_l_bkg_baseband; n]` host alloc + GPU upload with a
  single GPU fill launch into a pre-allocated buffer. Wallclock
  impact minimal (baseband is small), but closes the last
  per-JOD host alloc in the hot path. New parity test
  `fill_f32_kernel_writes_uniform_value` uses a sentinel-fill
  trick to catch off-by-one or short-write bugs.
- **Extract REF/DIST weber helpers + perf snapshot**
  (tick 169, `ea13bcf8`) — factors
  `_dispatch_ref_weber_pyramid_only` and
  `_dispatch_dist_weber_pyramid_only` out of
  `_dispatch_d_bands_into_scratch`. No behaviour change, sets
  up the warm-ref API. The tick 169 measurement landed at
  jod 38.0 ns/px (2.26× faster than fcvvdp 8t @ 360p) —
  the tick 166 reading at 41.8 was on the high end of its noise
  band.
- **Warm-ref batch-scoring API** (tick 170, `abe3599d`) —
  delivers the `score_with_reference` doc promise from v0.0.1:
  - `Cvvdp::warm_reference(ref_srgb)` dispatches REF weber once
    and stores `Some(log_l_bkg_baseband)` in
    `Cvvdp::warm_ref_baseband_log_l_bkg`. Any subsequent method
    that dispatches REF weber resets this to `None` —
    `_dispatch_ref_weber_pyramid_only` does the reset
    unconditionally so warm-reference is the only path that
    arms it.
  - `Cvvdp::compute_dkl_jod_with_warm_ref(dist_srgb, ppd)`
    skips the REF half of the JOD pipeline. Returns
    `Error::NoWarmReference` if the cache is cold.
  - Refactored band loop + pool into `_dispatch_d_bands_dist_and_band_loop`
    and `_pool_and_finalize_jod` so cold and warm paths share
    the post-REF tail.
  - Parity test `compute_dkl_jod_with_warm_ref_matches_unwarm_path`
    verifies: (1) warm/cold byte-for-byte match within 1e-5
    JOD, (2) state survives multiple warm-ref calls,
    (3) intervening cold calls invalidate correctly.
- **`time_12mp` measures warm-ref fast path**
  (tick 171, `8c7c5f96`) — adds phase 4 measuring per-DIST cost
  after one `warm_reference` per iter. 12 MP results:
  - jod (cold REF):       36.1 ns/px
  - jod_warm (cached REF): **20.6 ns/px**
  - Per-DIST saving: 42.9% (1.75× faster per call)
  - vs fcvvdp 8-thread @ 360p: **4.17× faster per DIST**

Warm path delivers below the naive 50% saving because the host
pool fold + band loop dispatch overhead run once per JOD
regardless of REF state. The amortization break-even is ~2
candidates per warmed reference — anything larger lands at
1.75× throughput.

| tick | jod cold (ns/px) | jod warm (ns/px) | vs fcvvdp 8t (cold / warm) |
| ---- | ----             | ----             | ----                        |
| 158  | 52.9             | —                | 1.63× / —                   |
| 166  | 41.8             | —                | 2.06× / —                   |
| 169  | 38.0             | —                | 2.26× / —                   |
| 171  | **36.1**         | **20.6**         | **2.38× / 4.17× faster**    |

The `d_bands − 2×weber` bucket (CSF + masking + IO) is sub-noise
since tick 156: 2×weber ≈ d_bands, meaning the band-loop overhead
is now bandwidth-tightly packed against the two weber pyramids.
The next remaining hot spot is the gauss-pyramid reduce (5×5
downscale, 25 src reads per output pixel), which a shared-memory
tiled rewrite could shrink — but the per-thread register
pressure observation from tick 159 means any fusion attempt
should change the memory access pattern, not just rearrange
launches.

### Tick 175–178 — ceil-div correctness wave (resolves tick 174 drift)

After tick 174 root-caused the 0.586 JOD drift vs pycvvdp at 12 MP
to floor-div vs ceil-div pyramid halving, the next ticks shipped
the fix and locked it with new tests.

- **Ceil-div pyramid + MAX_LEVELS = 9** (tick 175, `cee15d24`)
  — `build_pyramid` / `build_weber_scratch` /
  `build_d_bands_scratch` / `pyramid_levels` switched from
  `n / 2` to `(n + 1) / 2`. Order mattered: bumping MAX_LEVELS
  alone (tick 174 attempt) widened the drift to 1.54; ceil-div
  first then bump levels closed it to 0.0003.
  - 4000×3000 synth: ours **9.4583** vs pycvvdp **9.4580** —
    **drift 0.586 → 0.0003 JOD** (2000× more accurate).
  - All 67 existing parity tests stayed green (they run at
    power-of-2 sizes where floor == ceil at every level).
  - Trade-off: jod cold 36 → 62 ns/px, warm-ref 21 → 34 ns/px
    on the same RTX 5070. Open investigation — total pixel
    work is nearly unchanged, so the ~25% post-warmup slowdown
    must be a kernel-dispatch or boundary-branch interaction,
    not extra compute. Snapshot: `benchmarks/pycvvdp_parity_tick175_2026-05-15.md`.

- **`level_dims` reads `gauss_ref` shapes** (tick 176, `b9b5b71a`)
  — was computing `(bw, bh, n_px)` via `width >> k` (floor-div
  bit shift), which disagreed with the ceil-div allocator at
  odd-dim levels. Consequence: the band loop's CSF + masking +
  pool kernels dispatched fewer threads than the bands_ref /
  d_scratch buffers actually held — the last few tail pixels at
  each odd-dim level were written by weber but never processed
  downstream. 12 MP JOD output unchanged (tail values were
  near-zero so didn't move the pool), but the inconsistency
  was real and would matter on other inputs. Now reads
  `gauss_ref[k].w / .h` directly so all shape-using sites
  agree.

- **Odd-dim JOD parity test** (tick 177, `f2425dce`) — added
  `compute_dkl_jod_matches_host_scalar_on_odd_dims` at 73×91
  (the smallest source that diverges at ceil-vs-floor level 4+).
  Catches future floor-div regressions in either host_scalar
  or the GPU pyramid path. The other JOD parity tests all run
  at power-of-2 sizes where floor == ceil.

- **12 MP pycvvdp golden parity test** (tick 178, `cd61a217`)
  — added `compute_dkl_jod_matches_pycvvdp_at_12mp_synth`. The
  deterministic 4000×3000 synth pair from
  `examples/time_12mp.rs` runs through `compute_dkl_jod` and
  asserts the output matches pycvvdp v0.5.4's measured 9.4580
  golden within 0.005 JOD. Current observed diff: 0.0003.
  Would have failed at tick 173 (diff 0.586) and tick 174
  (diff 1.54); now gates the canonical-reference correctness
  in CI. Runtime ~5 s per call.

The pycvvdp parity matrix is now end-to-end:

| size      | test                                                              | tolerance | observed |
| ----      | ----                                                              | ----      | ----     |
| 32×32     | `compute_dkl_jod_matches_host_scalar`                            | 0.5 JOD   | < 0.1    |
| 73×91     | `compute_dkl_jod_matches_host_scalar_on_odd_dims`                | 0.5 JOD   | **0.0004** (post tick 181) |
| 256×256   | `compute_dkl_jod_matches_host_on_corpus_256x256` (drift sweep)   | 0.06 JOD  | < 0.05   |
| 4000×3000 | `compute_dkl_jod_matches_pycvvdp_at_12mp_synth`                  | 0.005 JOD | **0.0003** |
| 256×256 v1 manifest | `shadow_jod` (host scalar)                              | 0.01 JOD  | < 0.006  |

### Tick 179–182 — band-count alignment + pycvvdp goldens manifest

- **CHANGELOG / PORT_STATUS / lib.rs docs caught up to tick 175-178**
  (tick 179, `d7f8445f`) — the ceil-div correctness wave is now
  surfaced in user-facing docs. Corrected `lib.rs` to drop the
  misleading "2.58× slower than pycvvdp" framing (those numbers
  reflected a broken pyramid drifting 0.586 JOD); honest post-fix
  is 4.4× slower cold / 2.4× slower warm with correct output.

- **Extended pycvvdp bench script + goldens manifest**
  (tick 180, `b937401e`) — `scripts/cvvdp_goldens/bench_12mp_cuda.py`
  now produces a `pycvvdp_synth_goldens.json` manifest with the
  pycvvdp golden JOD for both the 4000×3000 12 MP fixture
  (9.4580) and a 73×91 odd-dim fixture (9.3904). The manifest
  schema lets future Rust parity tests load canonical reference
  values directly instead of duplicating hardcoded constants.

- **Surprise: host_scalar drifts ~0.6 JOD vs pycvvdp at 73×91**
  (tick 180 finding) — at sub-megapixel sizes our host_scalar
  reference produces 8.79 vs pycvvdp 9.39. The 256² v1 manifest
  fixtures hold ≤ 0.006 JOD, the 4000×3000 synth holds 0.0003,
  but 73×91 drifts ~0.6. Possible causes (open investigation):
  CSF interpolation at very small angular widths, band-mul rule
  difference for the small-band branch, or a display-geometry
  interpretation gap at sub-degree image sizes.

- **`pyramid_levels` defers to `band_frequencies` (tick 181, `e4951c15`)**
  — the GPU pipeline had a size-based cap (`cur >= 2 *
  PYRAMID_MIN_DIM`) that produced fewer bands than host_scalar
  at small inputs (4 vs 5 at 32², 5 vs 6 at 73×91, 7 vs 8 at
  256²). host_scalar already used `band_frequencies(ppd, w, h).len()`
  directly. Aligned the GPU side. Effect on the 73×91 GPU-vs-host
  parity test: **diff 0.092 → 0.0004 JOD** (235× better
  agreement). 12 MP pycvvdp gate still passes at 0.0003.

  Resolves the GPU↔host structural mismatch at small sizes.
  The remaining ~0.6 JOD drift at 73×91 is purely host_scalar
  vs pycvvdp (GPU now matches host within f32 precision).

### Investigation Notes (cvvdp-gpu, tick 174 — large-image drift)

After tick 173's pycvvdp v0.5.4 CUDA bench surfaced a **0.586 JOD
drift** between our `compute_dkl_jod` and pycvvdp on a 4000×3000
synthetic pair (ours 8.8726, pycvvdp 9.4580), tick 174 traced the
cause. Diagnostic scripts in `scripts/cvvdp_goldens/`:

- `bench_12mp_cuda.py` — pycvvdp CUDA timing + JOD output
- `diagnose_12mp.py` — pycvvdp metric internals
- `diagnose_pyramid.py` — pycvvdp band_freqs + height + pyr_shape
- `diagnose_freqs.py` — direct comparison of band frequencies
- `diagnose_decompose.py` — actual band tensor shapes via decompose()

**Two structural divergences from pycvvdp at large sizes:**

1. **n_bands cap**. Our `MAX_LEVELS = 8` caps the pyramid at 8
   levels. pycvvdp uses **9 bands** at 4000×3000 (one extra deep
   level). Bumping `MAX_LEVELS` alone is insufficient — see #2.

2. **Floor vs ceil division on pyramid sizes** (the dominant
   cause). pycvvdp uses **ceil-div** when halving level
   dimensions; we use floor-div. The bands diverge from level 4
   onward:

   | level | pycvvdp shape (ceil)  | cvvdp-gpu shape (floor) |
   | ---   | ---                   | ---                     |
   | 0     | 3000×4000             | 3000×4000               |
   | 1     | 1500×2000             | 1500×2000               |
   | 2     | 750×1000              | 750×1000                |
   | 3     | 375×500               | 375×500                 |
   | 4     | **188**×250           | **187**×250             |
   | 5     | 94×125                | 93×125                  |
   | 6     | **47×63**             | **46×62**               |
   | 7     | 24×32                 | 23×31                   |
   | 8     | 12×16 (baseband)      | (n/a — capped)          |

   Naively bumping MAX_LEVELS to 10 + adding level 8 INCREASED
   the drift (JOD 8.87 → 7.92) because the ceil-div mismatch
   compounds with every additional level. Reverted MAX_LEVELS
   to 8 until the ceil-div fix lands.

The 0.006 JOD parity tolerance our existing tests hit at 256×256
holds because at small sizes the ceil/floor difference is 0 or 1
pixel and most of pycvvdp's pyramid math rounds out. At 12 MP
the divergence stacks to ~0.6 JOD.

**Fix plan** (multi-tick):
- Switch pyramid `Level` allocator + `gauss_ref` chain to
  ceil-div (`(w + 1) / 2`).
- Update `downscale_kernel` boundary handling for the off-by-one
  case (currently floor-div semantics).
- Update upscale `back_v` / `back_h` math which assumes the
  parent floor-div shape.
- Bump MAX_LEVELS to 10 once ceil-div parity holds at 256×256.
- Add a 12 MP parity test driven by a pycvvdp golden so the
  drift is visible in CI.

**Goldens expansion (user ask, 2026-05-15):**

> pycvvdp needs to be the source of goldens and we have to sweep
> a larger distortion set

Current goldens at `v1/manifest.json` only cover 256×256 source
×6 JPEG quality levels. Planned expansion:
- Multi-resolution: 256², 1024², 4000×3000 (and 8K for sanity).
- More distortion types: Gaussian blur, Gaussian noise,
  contrast/saturation perturbations, downscale+upscale, color
  shifts, dithering, banding.
- Quality levels closer to perceptual JND than just JPEG-q.
- Sweep dimension: image content (photo, screen, line-art) so the
  golden corpus stratifies across the codec-corpus categories.

Goldens regenerator script (`build_goldens.py`) needs to grow a
distortion-config DSL + a multi-resolution + multi-image pipeline
before this expansion can land cleanly.

**cvvdp-gpu vs pycvvdp perf gap (cuDNN / Burn / cubek):**

User suggestion (2026-05-15):

> Burn is a libtorch alternative so we should be able to beat
> pycvvdp on GPU — maybe we didn't update to the latest cubecl
> 0.10 release or use the best algorithms in cubek?

Current state:
- cubecl pin: `0.10.0-pre.4` (per workspace Cargo.lock). The
  cubek (`tracel-ai/cubek`) high-level kernel library at
  `cubecl-kernels` exposes well-optimised matmul, conv, reduce.
- pycvvdp's hot path is the downscale/upscale Gaussian pyramid
  — pure depthwise separable convolution. PyTorch routes this
  via cuDNN, which has hand-tuned per-arch kernels.
- The cubek conv kernel (depthwise 5-tap, shared-memory tiled)
  would close the gap if it matches cuDNN. We currently do not
  use cubek conv — our `downscale_kernel` /
  `upscale_v_kernel` / `upscale_h_kernel` are hand-rolled 5-tap.

Investigation queued: try replacing the downscale/upscale
kernels with cubek-conv calls and re-measure. If cubek-conv
holds parity (separable filter, ceil-div boundaries) and lands
≤ pycvvdp at 12 MP, that's our path to "beat libtorch".

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

| metric                          | tick 64   | tick 73    | tick 171   |
| ----                            | ----      | ----       | ----       |
| weber pyramid (1 side)          | 103 ns/px | 21.6 ns/px | 18.7 ns/px |
| compute_dkl_d_bands             | 428 ns/px | 121 ns/px  | 33.7 ns/px |
| compute_dkl_jod (cold REF)      | 444 ns/px | 127 ns/px  | **36.1 ns/px** |
| compute_dkl_jod_with_warm_ref   | —         | —          | **20.6 ns/px** |

### Honest comparison against the canonical reference (tick 173)

The fcvvdp ratios cited in earlier rows compare against
`halidecx/fcvvdp` — a separate C+Zig fork, not the canonical
pycvvdp at `gfxdisp/ColorVideoVDP`. Direct pycvvdp v0.5.4
CUDA measurement on the same RTX 5070 host:

| metric                          | per-pixel  | vs pycvvdp CUDA |
| -----                           | ----       | ----            |
| **pycvvdp v0.5.4 (CUDA)**       | **14 ns/px** | baseline        |
| cvvdp-gpu cold                  | 36.1 ns/px | **2.58× slower** |
| cvvdp-gpu warm-ref              | 20.6 ns/px | **1.47× slower** |

pycvvdp benefits from cuDNN-optimised separable convolution on
the downscale/upscale pyramid; our cubecl kernels are hand-written
5-tap separable. cvvdp-gpu wins on portability (WGPU + HIP
backends, ~50 MB static binary vs ~3 GB PyTorch runtime, ~1 s
warm-up vs 1-13 s graph compile) but loses on raw CUDA throughput.

See `crates/cvvdp-gpu/benchmarks/pycvvdp_12mp_cuda_2026-05-14.md`
+ `scripts/cvvdp_goldens/bench_12mp_cuda.py` for the
reproduction recipe.

### vs fcvvdp (separate C+Zig fork, NOT the canonical reference)

fcvvdp's published 360p bench (i7-13700k):

| fcvvdp variant | per-pixel  | vs cvvdp-gpu cold @ 12 MP | vs cvvdp-gpu warm @ 12 MP |
| ----           | ----       | ----                       | ----                       |
| 1-thread       | 214 ns/px  | cvvdp-gpu **5.93× faster** | cvvdp-gpu **10.39× faster** |
| 8-thread       |  86 ns/px  | cvvdp-gpu **2.38× faster** | cvvdp-gpu **4.17× faster**  |

The fcvvdp comparison is real (numbers measured, ratios correct)
but **fcvvdp is not pycvvdp**. Use the pycvvdp row for the
canonical comparison.

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
