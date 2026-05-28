# Crate Graph Audit — Phase 8c.1

Audit date: 2026-05-27. Auditor: Phase 8c.1 sibling workspace at
`/home/lilith/work/zen/zenmetrics--phase8c1-audit/`.
Parent branch: `master` at `89068c37` ("bench(phase8f): parity sweep
54/54 PASS-EXACT vs zenforks-cubecl-* 0.10.1").

This document satisfies Phase A of the Phase 8c.1 brief: survey the
metric crate pairs, inventory shared types and dep directions, and rank
opportunities. Phase B (the dep-direction flip + any clear-cut
follow-on) executes against the recommendations below.

---

## A.1 — Per-crate-pair inventory

The six metric crates split into three categories by their CPU-side
provenance:

1. **In-tree CPU + in-tree GPU (one pair):** `cvvdp` (in-tree) +
   `cvvdp-gpu` (in-tree). This is the *only* pair where we own both
   sides and the dep direction is fully under our control.
2. **Upstream CPU (sibling repo) + in-tree GPU (five pairs):**
   `butteraugli` / `ssimulacra2` / `dssim-core` / `zensim` (all
   external) + the matching `-gpu` crate (in-tree).
3. **No-CPU-pair (one pair):** `iwssim-gpu` — there is no in-tree
   CPU implementation of IW-SSIM; the orchestrator's `CpuAdapter`
   surfaces `Unavailable` for that metric.

### A.1.1 — cvvdp pair (ONLY pair where we own both sides)

**CPU crate**

- Location: `crates/cvvdp/`
- Cargo: `name = "cvvdp"`, `version = "0.1.0"`, `publish = false`.
- Source layout: `color.rs`, `csf.rs`, `diffmap.rs`, `masking.rs`,
  `pipeline.rs`, `pool.rs`, `pyramid.rs`, `scratch.rs`, `simd_math.rs`,
  `simd_pyramid.rs` (10 files, all SIMD-optimized host code via
  archmage / magetypes).
- Public types: `pipeline::Cvvdp`, `Error`, `Result<T>`,
  `CVVDP_COLUMN_NAME: &str`, `N_CHANNELS: usize`, `diffmap` module.
- Public API:
  - `Cvvdp::new(width, height, params) -> Result<Self>`
  - `Cvvdp::score(&ref_srgb, &dist_srgb) -> Result<f32>` (raw JOD)
  - `Cvvdp::score_with_diffmap(...)` — diffmap variant
  - `Cvvdp::warm_reference(&ref_srgb)` + `Cvvdp::score_with_warm_ref(&dist_srgb)`
  - `Cvvdp::score_with_warm_ref_diffmap(...)`
  - `Cvvdp::score_from_linear_planes(...)` + diffmap + warm variants
  - **Returns raw `f32` (JOD)**, NOT a `Score` struct.
- Re-exports (from cvvdp-gpu — inverse-dep):
  - `pub use cvvdp_gpu::PYCVVDP_REFERENCE_VERSION;`
  - `pub use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry, DisplayModel, PerfMode};`

**GPU crate**

- Location: `crates/cvvdp-gpu/`
- Cargo: `name = "cvvdp-gpu"`, `version = "0.0.1"`, `publish = false`.
- Source layout: `heatmap.rs`, `host_scalar.rs` (pure scalar reference
  pipeline — NO GPU), `kernels/{color,csf,diffmap,masking,pool,pyramid}.rs`
  (52 `#[cube]` kernels + the constants they share with the host scalar),
  `lib.rs`, `memory_mode.rs`, `opaque.rs`, `params.rs` (display / EOTF /
  primaries / DKL matrices / parameter bundles, 1544 lines), `pipeline.rs`
  (generic `Cvvdp<R: Runtime>`), `presets.rs` (named-display registry).
- Public types: `Cvvdp<R>`, `CvvdpOpaque`, `Backend`, `Score`, `Error`,
  `Result<T>`, `CvvdpParams`, `PerfMode`, `MemoryMode`,
  `CVVDP_COLUMN_NAME`, `PYCVVDP_REFERENCE_VERSION`, `MAX_LEVELS`,
  `PYRAMID_MIN_DIM`, `N_CHANNELS`.
- Public API (opaque): `CvvdpOpaque::{new, new_with_memory_mode,
  new_with_geometry, new_with_geometry_and_memory_mode, dims,
  compute_srgb_u8, compute_pixels, compute_handles,
  compute_srgb_u8_with_diffmap, warm_reference_srgb,
  has_warm_reference, compute_with_warm_ref_srgb,
  warm_reference_from_linear_planes}`.
- Public API (typed, gated behind `cubecl-types`): `Cvvdp<R: Runtime>`
  with the same shape plus `compute_dkl_jod`, `compute_dkl_jod_host_pool`,
  `compute_dkl_jod_with_warm_ref`, etc.

**Shared types that currently live in `cvvdp-gpu` but should live in
the CPU crate** (these are the targets for the Phase B.1 flip):

| Type / constant | Current location | Canonical owner | Notes |
|---|---|---|---|
| `CvvdpParams` (struct) | `cvvdp-gpu::params` | `cvvdp` | Pure data + impls; no GPU/cubecl deps |
| `PerfMode` (enum) | `cvvdp-gpu::params` | `cvvdp` | Pure enum |
| `Eotf` (enum) | `cvvdp-gpu::params` | `cvvdp` | Pure enum + scalar `forward` helpers |
| `Primaries` (enum) | `cvvdp-gpu::params` | `cvvdp` | Pure enum + scalar helpers |
| `DisplayModel` (struct) | `cvvdp-gpu::params` | `cvvdp` | Pure data + scalar lin-to-cd helpers |
| `DisplayGeometry` (struct) | `cvvdp-gpu::params` | `cvvdp` | Pure data |
| `SRGB_LINEAR_TO_DKL`, `BT2020_LINEAR_TO_DKL`, `DISPLAY_P3_LINEAR_TO_DKL` (3×3 f32 matrices) | `cvvdp-gpu::params` | `cvvdp` | Pure constants |
| `CsfParams`, `MaskingParams`, `PoolingParams`, `JodParams` | `cvvdp-gpu::params` | `cvvdp` | Pure scaffolding structs |
| `srgb_eotf_scalar`, `pq_eotf_scalar`, `hlg_inverse_oetf_scalar`, `hlg_system_gamma` | `cvvdp-gpu::params` | `cvvdp` | Pure scalar functions |
| `PYCVVDP_REFERENCE_VERSION: &str` | `cvvdp-gpu::lib` | `cvvdp` | Pure const; both crates re-export |
| `N_CHANNELS`, `MAX_LEVELS`, `PYRAMID_MIN_DIM` | `cvvdp-gpu::lib` | `cvvdp` | Pure consts (`cvvdp` already exports its own `N_CHANNELS`) |
| `kernels::pool::{BETA_SPATIAL, BETA_BAND, BETA_CH, IMAGE_INT, JOD_A, JOD_EXP, PER_CH_W, BASEBAND_W}` | `cvvdp-gpu::kernels::pool` | `cvvdp` | Pure scalar constants; CPU already uses |
| `kernels::pool::{lp_norm_mean, lp_norm_sum, met2jod, do_pooling_and_jod_still_3ch}` | `cvvdp-gpu::kernels::pool` | `cvvdp` | Pure scalar fns; CPU already uses |
| `kernels::masking::{CH_GAIN, MASK_P, MASK_Q, MASK_C, D_MAX, XCM_3X3, PU_PADSIZE, PU_BLUR_KERNEL_1D}` | `cvvdp-gpu::kernels::masking` | `cvvdp` | Pure constants; CPU already uses |
| `kernels::masking::{safe_pow, clamp_diff_soft, phase_uncertainty_no_blur, gaussian_blur_sigma3, mult_mutual_band, reflect_idx_for_blur}` | `cvvdp-gpu::kernels::masking` | `cvvdp` | Pure scalar fns; CPU consumes from tests + a few from runtime |
| `kernels::pyramid::{KERNEL_A, GAUSS5, gausspyr_reduce_scalar, gausspyr_expand_scalar, Band, band_frequencies, laplacian_pyramid_dec_scalar, WeberPyramid, weber_contrast_pyr_dec_scalar}` | `cvvdp-gpu::kernels::pyramid` | `cvvdp` | Pure scalar; CPU consumes from runtime |
| `kernels::csf::{SENSITIVITY_CORRECTION_DB, CSF_BASEBAND_RHO, CsfChannel, N_L_BKG, N_RHO, sensitivity_scalar, sensitivity_corrected_scalar, precompute_logs_row}` | `cvvdp-gpu::kernels::csf` | `cvvdp` | Pure scalar |
| `kernels::csf::csf_lut_v0_5_4::*` (sensitivity LUT) | `cvvdp-gpu::kernels::csf::csf_lut` | `cvvdp` | Pure 32×32×3 f32 LUT |
| `kernels::color::{SRGB8_TO_LINEAR_LUT, srgb_byte_to_dkl_scalar, display_byte_to_dkl_scalar, display_linear_rgb_to_dkl_scalar, eotf_tag::*, eotf_tag_and_gamma}` | `cvvdp-gpu::kernels::color` | `cvvdp` | Pure scalar |
| `host_scalar::predict_jod_still_3ch` | `cvvdp-gpu::host_scalar` | `cvvdp` | Pure scalar reference algorithm — the canonical "no GPU" implementation, ironically living in the GPU crate today |
| `kernels::diffmap::{bilinear_sample_scalar, channel_pool_scalar}` | `cvvdp-gpu::kernels::diffmap` | `cvvdp` | Pure scalar |
| `presets::*` (`DisplayModel::by_name`, etc.) | `cvvdp-gpu::presets` | `cvvdp` | Pure data + JSON parse; no GPU |

**Items that stay in `cvvdp-gpu`** (depend on cubecl):

- Everything decorated with `#[cube]` or `#[cube(launch)]` (52 kernels).
- `pipeline::Cvvdp<R: Runtime>` generic type and its `compute_dkl_jod*`
  methods.
- `opaque::CvvdpOpaque`, `opaque::Backend`, `opaque::Score`.
- `memory_mode::*` (depends on the pipeline's runtime).
- `flatten_band_weights`, `precomputed_band_weights` (helpers that
  shape data for GPU upload — could move, but their consumers are
  GPU-side, so keeping them where they're used is fine).

**Current dep direction**

```
[INVERSE / PROBLEMATIC TODAY]
cvvdp (CPU)
  ↓ depends on
cvvdp-gpu (GPU)
```

CPU pulls cubecl + all GPU machinery transitively just to access pure
constants and scalar reference functions. This is the issue Phase 8c.1
fixes.

**After Phase B.1 (the flip):**

```
[CORRECT — what Phase B.1 ships]
cvvdp-gpu (GPU)
  ↓ depends on
cvvdp (CPU)
```

GPU crate consumes shared constants + scalar reference helpers from
`cvvdp`. CPU crate has zero awareness of GPU machinery — buildable on
embedded / no-cubecl targets.

---

### A.1.2 — butteraugli pair

- **CPU crate**: `butteraugli` (upstream sibling repo at `~/work/zen/butteraugli/`,
  workspace path-dep `butteraugli = { path = "../butteraugli", … }`).
- **GPU crate**: `crates/butteraugli-gpu/`.
- **Dep direction**: independent. `butteraugli-gpu`'s `[dependencies]`
  has NO direct dep on `butteraugli`; it's only in `[dev-dependencies]`
  for the parity tests. The CPU side is consumed by `zen-metrics-cli` /
  `zenmetrics-orchestrator` via the upstream crate.
- **Shared types**: each crate has its own `ButteraugliParams` /
  `Score`. Upstream `butteraugli` returns `f64` from
  `butteraugli::api::compute_butteraugli_2d`. The GPU `Score` carries
  `value: f64`. No type-coupling between CPU + GPU exists today.
- **Constraint**: per CLAUDE.md "NEVER touch other repos", we cannot
  move `ButteraugliParams` into the upstream crate to share. The
  GPU crate's `ButteraugliParams` is unrelated to the upstream's
  options struct.
- **Action**: NONE. The dep direction is already correct (independent).
  Type duplication is acceptable because the upstream crate is owned
  by a third party (libjxl).

### A.1.3 — ssim2 pair

- **CPU crate**: `ssimulacra2` (upstream sibling, `crates.io` published —
  see `Cargo.toml`'s `ssimulacra2 = "..."` line if present).
- **GPU crate**: `crates/ssim2-gpu/`.
- **Dep direction**: independent. `ssim2-gpu` has NO direct dep on
  `ssimulacra2`; orchestrator consumes both separately.
- **Shared types**: same situation as butteraugli — `Ssim2Params`
  lives in `ssim2-gpu`; the upstream crate has its own option struct.
- **Action**: NONE.
- **Rename opportunity**: `ssim2-gpu` could be renamed to `ssimulacra2-gpu`
  for naming alignment with the upstream crate. **Defer** — naming
  inconsistency isn't a correctness issue, and the parity sweep
  script + `MetricKind::Ssim2` + `metric_name = "ssim2"` strings would
  all need rewrites.

### A.1.4 — dssim pair

- **CPU crate**: `dssim-core` (upstream sibling, `crates.io` published).
- **GPU crate**: `crates/dssim-gpu/`.
- **Dep direction**: independent.
- **Shared types**: `DssimParams` lives in `dssim-gpu`; upstream
  `dssim-core` has its own.
- **Action**: NONE. Naming asymmetry (`dssim-core` vs `dssim-gpu`) is
  consistent with the upstream-crate naming convention; do NOT rename.

### A.1.5 — iwssim pair

- **CPU crate**: NONE (no clean reference CPU implementation exists in
  the workspace; orchestrator's `CpuAdapter` returns `Unavailable`).
- **GPU crate**: `crates/iwssim-gpu/`.
- **Dep direction**: trivially correct (no CPU crate to depend on).
- **Action**: NONE. A future tier-3 priority could be to extract a
  pure-CPU IW-SSIM reference from `iwssim-gpu/src/host_scalar.rs`
  equivalent (if one exists), but that's out of scope for 8c.1.
- **Side note**: this is the one metric where `Metric::compute_pixels`
  currently routes through `iwssim-gpu` exclusively. If a future
  `iwssim` (CPU) crate lands, its constants + scalar helpers should
  follow the same "pure stuff lives CPU-side, GPU consumes" pattern
  that Phase B.1 establishes for cvvdp.

### A.1.6 — zensim pair

- **CPU crate**: `zensim` (sibling, in-tree at `~/work/zen/zensim/`, path
  dep). This one is in-workspace but it's a separate top-level crate, not
  in `zenmetrics-api`'s tree.
- **GPU crate**: `crates/zensim-gpu/`.
- **Dep direction**: `zensim-gpu` depends on `zensim` for the
  weights-loading + score helpers. `zensim-gpu/Cargo.toml`:
  `zensim = { workspace = true }` in `[dependencies]`. **THIS IS THE
  CORRECT DIRECTION** — pattern matches what Phase B.1 will establish
  for cvvdp.
- **Shared types**: `ZensimParams` lives in `zensim-gpu` today; the
  CPU `zensim` crate has its own `Zensim` scorer. Some overlap is
  possible but `zensim` is a "sibling" by CLAUDE.md's definition
  (it's a separate published crate even though in-workspace) — moving
  types into it requires the same care as butteraugli/dssim-core.
- **Action**: NONE in 8c.1. The dep direction is already correct.
  (zensim is the *role model* for what cvvdp-gpu→cvvdp should look
  like after Phase B.1.)

---

## A.2 — zenpixels interface consistency check

All six `-gpu` crates expose:

- `pixels` feature flag (matches: `pixels = ["dep:zenpixels",
  "dep:zenpixels-convert"]` on every crate).
- `compute_pixels(r: PixelSlice<'_>, d: PixelSlice<'_>) -> Result<Score>`
  on the opaque shim.
- `compute_handles(&ref_handle, &dist_handle) -> Result<Score>` for
  pre-uploaded buffers (gated behind `cubecl-types`).

The orchestrator's `Metric::compute_pixels` (in `zenmetrics-api`)
dispatches uniformly across all 6 variants — no per-metric divergence.

`MetricContext::upload_pair` lives in `zenmetrics-api::context` and
produces a `PairHandles` with `ref_handle` / `dist_handle` / `generation`.
Every metric's `compute_handles(&PairHandles)` consumes these via
`Metric::compute_handles` in the umbrella. **One exception**: `zensim`
returns `Error::Metric { kind: "zensim", message: "compute_handles not
wired for zensim-gpu (Phase 4 deferred — see umbrella commit)" }`. This
is a documented gap, not an inconsistency.

**Result**: zenpixels interface is **consistent across all 6
opaque crates**. No gaps to fill in B.3.

The only follow-on to note: when Phase B.1 moves `CvvdpParams` into
`cvvdp`, the umbrella's `MetricParams::Cvvdp(cvvdp_gpu::CvvdpParams)`
variant must change to `MetricParams::Cvvdp(cvvdp::CvvdpParams)`.
That's a one-line edit + a `pub use` for the GPU crate to keep
backward-compat re-exports if desired.

---

## A.3 — API pattern consistency check

### Constructor

All six opaque shims agree:

```rust
pub fn new(backend: Backend, width: u32, height: u32, params: <P>) -> Result<Self>
pub fn new_with_memory_mode(backend, width, height, params, mode) -> Result<Self>
```

`cvvdp-gpu::CvvdpOpaque` adds `new_with_geometry` /
`new_with_geometry_and_memory_mode` for explicit display geometry —
not a violation of the pattern, just an extension. Other metrics
don't have a geometry concept.

### One-shot scoring

All six agree:

```rust
pub fn compute_srgb_u8(&mut self, r: &[u8], d: &[u8]) -> Result<Score>
pub fn compute_pixels(&mut self, r: PixelSlice, d: PixelSlice) -> Result<Score>
pub fn compute_handles(&mut self, ref_h, dis_h) -> Result<Score>   // cubecl-types gated
```

### Cached-reference / warm-reference

**The inconsistency lives here.** Two naming conventions split the six
metrics:

| Metric | "Set" method | "Has" method | "Score" method |
|---|---|---|---|
| butter, dssim, iwssim, ssim2 | `set_reference_srgb_u8` | `has_cached_reference` | `compute_with_cached_reference_srgb_u8` |
| cvvdp | `warm_reference_srgb` | `has_warm_reference` | `compute_with_warm_ref_srgb` |
| zensim | `set_reference_srgb_u8` | — | `compute_with_cached_reference_score_srgb_u8` |

The umbrella `zenmetrics-api::Metric` already papers over this with
`set_reference_srgb_u8` / `has_cached_reference` /
`compute_with_cached_reference_srgb_u8` dispatching to the per-crate
method (see `metric.rs:803-911`). Callers using the umbrella see one
consistent name; only direct consumers of `CvvdpOpaque` /
`ZensimOpaque` see the per-crate names.

**Recommendation**: low priority. The umbrella surface is already
consistent. Renaming `CvvdpOpaque::warm_reference_srgb` to
`set_reference_srgb_u8` (and adding deprecated aliases) would close
the gap fully but is API churn for no end-user benefit. **Defer**.

### CPU-crate signatures (cvvdp only)

`cvvdp::Cvvdp` returns `Result<f32>` (raw JOD) from `score`, NOT a
`Score` struct. This is intentional — the umbrella's `convert_score`
adapter is `Score-only`, and the CPU adapter
(`zenmetrics-orchestrator::cpu_adapter`) builds an umbrella `Score`
from the raw f32 + per-metric metadata.

**Result**: the umbrella surface is consistent; only per-crate direct
callers see the warm/cached naming split. Phase B does NOT pursue
this — see Recommendations.

---

## A.4 — Ranked opportunities

| Priority | Crate(s) | Issue | Recommended action | Risk | Phase |
|---|---|---|---|---|---|
| REQUIRED | `cvvdp` + `cvvdp-gpu` | CPU crate depends on GPU crate (inverse direction); CPU pulls cubecl transitively to access pure constants + the scalar reference algorithm | Move shared constants + `host_scalar` + the scalar fns out of `cvvdp-gpu::{params, host_scalar, kernels::*}` into `cvvdp` (new modules `cvvdp::params`, `cvvdp::host_scalar`, `cvvdp::kernels::*`); flip Cargo.toml so `cvvdp-gpu` depends on `cvvdp` | LOW — internal refactor; both crates have `publish = false`; existing CPU code already consumes these items via `pub use cvvdp_gpu::*` so re-pointing imports is mechanical; parity sweep gates the result | **B.1** |
| HIGH | `cvvdp-gpu`, `zenmetrics-api`, `zen-metrics-cli` | `pub use cvvdp_gpu::CvvdpParams` (and similar re-exports) become stale after the flip — callers should reach for `cvvdp::CvvdpParams` | After B.1, add `pub use cvvdp::{CvvdpParams, DisplayModel, ...}` from `cvvdp-gpu` so existing code (umbrella, CLI) keeps building unchanged; the umbrella's `zenmetrics_api::cvvdp` alias continues to surface the right types | LOW — `pub use` is the canonical compat shim | **B.1** (same commit set) |
| MEDIUM | `cvvdp-gpu` | `host_scalar::predict_jod_still_3ch` is the canonical *scalar* reference (no GPU dependency) living in the GPU crate — misleading and prevents users who only want the reference from avoiding the cubecl dep | Move `host_scalar` to `cvvdp::host_scalar` as part of B.1 | LOW — sibling module move; tests follow | **B.1** |
| MEDIUM | `cvvdp-gpu::presets` | Display-preset registry (JSON-loaded named displays) is pure data + JSON parse; no GPU dep | Move `presets` to `cvvdp::presets` as part of B.1 | LOW | **B.1** |
| LOW | All `-gpu` opaque crates | cached-ref naming split (`set_reference_*` vs `warm_reference_*`) | Add deprecated alias `set_reference_srgb_u8` → `warm_reference_srgb` on `CvvdpOpaque`; umbrella already normalizes | LOW but API churn | **DEFER** — umbrella already normalizes; direct consumers are few |
| LOW | `ssim2-gpu` | Naming inconsistency with upstream `ssimulacra2` crate | Rename `ssim2-gpu` → `ssimulacra2-gpu` | MEDIUM — every consumer (CLI, orchestrator, parity script, sweep workers, docs, R2 column-names) carries the `ssim2` short name; rename ripples broadly | **DEFER** — cost > benefit |
| LOW | `iwssim-gpu` | No CPU reference crate exists; orchestrator returns `Unavailable` | Extract scalar reference path from `iwssim-gpu/src/host_scalar*` (if present) into a new `iwssim` CPU crate, follow the same B.1 pattern | MEDIUM — new crate; depends on whether there's a clean scalar reference today | **DEFER** to a future phase |
| LOW | `cvvdp` (CPU) | `score` returns `Result<f32>`, not a `Score` struct (asymmetric with `-gpu` shims) | Add a `score_with_metadata` returning `Score`-like struct; keep `score` as-is for callers that just want the f32 | LOW but adds API surface | **DEFER** |
| LOW | `zenmetrics-api` | `MetricParams::Cvvdp(cvvdp_gpu::CvvdpParams)` will become `cvvdp::CvvdpParams` after B.1 | One-line edit in B.1 commit set | LOW | **B.1** (same commit set) |

---

## A.5 — Recommendations: what Phase B ships

### REQUIRED (Phase B.1 — the flip)

Move from `cvvdp-gpu` to `cvvdp`:

1. **`cvvdp-gpu::params::*`** → **`cvvdp::params`** (new module).
   Moves: `Eotf`, `Primaries`, `DisplayModel`, `DisplayGeometry`,
   `CvvdpParams`, `PerfMode`, `CsfParams`, `MaskingParams`,
   `PoolingParams`, `JodParams`, `SRGB_LINEAR_TO_DKL`,
   `BT2020_LINEAR_TO_DKL`, `DISPLAY_P3_LINEAR_TO_DKL`,
   `srgb_eotf_scalar`, `pq_eotf_scalar`, `hlg_inverse_oetf_scalar`,
   `hlg_system_gamma`.

2. **`cvvdp-gpu::kernels::pool::*` (scalar items only)** → **`cvvdp::kernels::pool`**.
   Moves: `BETA_SPATIAL`, `BETA_BAND`, `BETA_CH`, `IMAGE_INT`,
   `JOD_A`, `JOD_EXP`, `PER_CH_W`, `BASEBAND_W`, `lp_norm_mean`,
   `lp_norm_sum`, `met2jod`, `do_pooling_and_jod_still_3ch`.
   Leaves behind: `pool_band_kernel`, `pool_band_3ch_kernel`,
   `pool_band_3ch_offset_kernel`, `pool_band_3ch_lds_kernel`,
   `POOL_LDS_BLOCK_DIM` (all `#[cube]` or GPU-launch-config).

3. **`cvvdp-gpu::kernels::masking::*` (scalar items only)** → **`cvvdp::kernels::masking`**.
   Moves: `CH_GAIN`, `MASK_P`, `MASK_Q`, `MASK_C`, `D_MAX`, `XCM_3X3`,
   `PU_PADSIZE`, `PU_BLUR_KERNEL_1D`, `safe_pow`, `clamp_diff_soft`,
   `phase_uncertainty_no_blur`, `gaussian_blur_sigma3`,
   `mult_mutual_band`, `reflect_idx_for_blur` (if `pub`).
   Leaves behind: the `#[cube(launch)]` kernels (13 of them).

4. **`cvvdp-gpu::kernels::pyramid::*` (scalar items only)** → **`cvvdp::kernels::pyramid`**.
   Moves: `KERNEL_A`, `GAUSS5`, `gausspyr_reduce_scalar`,
   `gausspyr_expand_scalar`, `Band`, `band_frequencies`,
   `laplacian_pyramid_dec_scalar`, `WeberPyramid`,
   `weber_contrast_pyr_dec_scalar`.
   Leaves behind: `downscale_kernel`, `downscale_strip_kernel`,
   `DOWNSCALE_TILED_BLOCK_DIM`, `downscale_tiled_kernel`,
   `upscale_v_kernel`, `upscale_v_strip_kernel`, `upscale_h_kernel`,
   `upscale_h_strip_kernel` (`#[cube(launch)]`).

5. **`cvvdp-gpu::kernels::csf::*` (scalar items only)** → **`cvvdp::kernels::csf`**.
   Moves: `SENSITIVITY_CORRECTION_DB`, `CSF_BASEBAND_RHO`,
   `CsfChannel`, `N_L_BKG`, `N_RHO`, `sensitivity_scalar`,
   `sensitivity_corrected_scalar`, `precompute_logs_row`,
   `csf_lut_v0_5_4::*` (the 32×32×3 sensitivity LUT — pure data).
   Leaves behind: `csf_apply_per_pixel_kernel`, `csf_apply_3ch_kernel`,
   `csf_apply_6ch_kernel`, `weight_band_kernel`,
   `precomputed_band_weights`, `flatten_band_weights` (GPU-shape
   helpers).

6. **`cvvdp-gpu::kernels::color::*` (scalar items only)** → **`cvvdp::kernels::color`**.
   Moves: `SRGB8_TO_LINEAR_LUT`, `srgb_byte_to_dkl_scalar`,
   `display_byte_to_dkl_scalar`, `display_linear_rgb_to_dkl_scalar`,
   `eotf_tag::*`, `eotf_tag_and_gamma`.
   Leaves behind: `srgb_to_dkl_kernel` (`#[cube(launch)]`).

7. **`cvvdp-gpu::kernels::diffmap::{bilinear_sample_scalar, channel_pool_scalar}`** → **`cvvdp::kernels::diffmap`**.
   The four `#[cube(launch)]` kernels stay.

8. **`cvvdp-gpu::host_scalar`** → **`cvvdp::host_scalar`**.
   The whole module — it's pure scalar with no GPU dep.

9. **`cvvdp-gpu::presets`** → **`cvvdp::presets`**.
   JSON-loaded display registry; no GPU dep. Vendored JSON files
   under `cvvdp-gpu/data/` move to `cvvdp/data/` (and the
   `include_str!()` paths follow).

10. **`cvvdp-gpu::PYCVVDP_REFERENCE_VERSION`** → **`cvvdp::PYCVVDP_REFERENCE_VERSION`**.

11. **`cvvdp-gpu::{N_CHANNELS, MAX_LEVELS, PYRAMID_MIN_DIM}`** → **`cvvdp::{N_CHANNELS, MAX_LEVELS, PYRAMID_MIN_DIM}`**.
    Note `cvvdp::N_CHANNELS` already exists with the same value; the
    `MAX_LEVELS` + `PYRAMID_MIN_DIM` pin tests stay in cvvdp-gpu but
    reference the cvvdp consts.

**`cvvdp-gpu` keeps**:

- All `#[cube]` / `#[cube(launch)]` kernels (52 total).
- The runtime-generic `Cvvdp<R: Runtime>` pipeline.
- `opaque::CvvdpOpaque`, `opaque::Backend`, `opaque::Score`,
  `opaque::Error`.
- `memory_mode::*` (depends on the pipeline's runtime).
- `pub use cvvdp::{CvvdpParams, PerfMode, ...}` re-exports to keep
  existing consumer code (umbrella, CLI, conformance) building
  unchanged.

**Cargo.toml deltas**:

- `cvvdp/Cargo.toml`: remove `cvvdp-gpu = { workspace = true, … }` from
  `[dependencies]`.
- `cvvdp-gpu/Cargo.toml`: add `cvvdp = { workspace = true,
  default-features = false }` to `[dependencies]`. (The `cvvdp`
  crate has feature flags `std`, `parallel`, `pixels` —
  `cvvdp-gpu` should pull `std` at minimum; `parallel` is fine to
  opt out of for the GPU crate since GPU drives parallelism.)

**Backwards-compat re-exports** in `cvvdp-gpu/src/lib.rs`:

```rust
// Keep existing public types reachable through cvvdp-gpu so callers
// don't have to switch import paths. The canonical definitions now
// live in cvvdp; cvvdp-gpu re-exports them.
pub use cvvdp::{
    CvvdpParams, DisplayGeometry, DisplayModel, Eotf, PerfMode, Primaries,
    PYCVVDP_REFERENCE_VERSION, MAX_LEVELS, N_CHANNELS, PYRAMID_MIN_DIM,
};
pub use cvvdp::host_scalar;
pub use cvvdp::params;
pub use cvvdp::presets;
// kernels::* re-exports preserve the existing import path so the
// scalar items stay reachable at their old paths.
pub mod kernels {
    pub use cvvdp::kernels::{color, csf, diffmap, masking, pool, pyramid};
    // GPU-only kernel items (the #[cube(launch)] items) live in a
    // private inner module; this `pub use` flattens them onto
    // `cvvdp_gpu::kernels::*` as before.
    pub use crate::kernels_gpu::*;
}
```

Tests + callers continue to use `cvvdp_gpu::kernels::pool::lp_norm_mean`
etc. exactly as today. Long-term we'd rewrite call-sites to use
`cvvdp::kernels::pool::lp_norm_mean` directly, but that's not in scope
for 8c.1.

### CLI / umbrella deltas (same commit set)

- `crates/zenmetrics-api/src/metric.rs`: `MetricParams::Cvvdp(cvvdp_gpu::CvvdpParams)`
  → `MetricParams::Cvvdp(cvvdp::CvvdpParams)` (or use the
  re-export, which keeps the type identifier stable).
- `crates/zenmetrics-api/src/lib.rs`: `pub use cvvdp_gpu as cvvdp;`
  stays — the umbrella's `zenmetrics_api::cvvdp` alias maps to the
  *GPU* crate, which now re-exports everything from the CPU crate. To
  reduce confusion, we could add an additional `pub use cvvdp as cvvdp_cpu;`
  but that's optional.
- `crates/zen-metrics-cli/src/metrics/cvvdp_gpu.rs`: continues to
  `use zenmetrics_api::cvvdp;` — the indirection through the umbrella
  insulates it.

### NOT in Phase B (defer)

- B.2 highest-leverage item: skipped. The B.1 work covers the user's
  primary directive (gpu-depends-on-cpu) and is large enough that
  bundling a second refactor would muddy the parity-gate test signal.
  No other Phase A finding is clearly Low-risk + clearly bounded.
- B.3 zenpixels gaps: there are none — every `-gpu` crate has
  `compute_pixels` and umbrella dispatch is uniform.

### Risk + parity gate

Phase B.1 is a mechanical move. Every constant, scalar function,
data structure being moved is `pub` and is consumed by the CPU
crate today via `pub use cvvdp_gpu::*`. After the move, both crates
still see the same symbol path because the GPU crate re-exports
from the CPU crate.

**Parity gate**: `scripts/orchestrator_parity_sweep.py` runs 54 cells
(6 metrics × 3 sizes × 3 qs) against `target/release/zen-metrics` and
requires every cell to land within the per-metric tolerance. After
B.1: re-build `zen-metrics`, re-run the sweep, confirm 54/54
PASS-EXACT. Any divergence is the signal that the move broke
something — honest-stop.

---

## Reading guide for the next session

If you're resuming work on Phase 8c.1 after a context reset:

1. The user's directive (2026-05-27 brief in the agent prompt) is:
   "gpu versions should depend on cpu versions if there is a
   dependency relationship; zenpixels interfaces and consistent api
   patterns. review crate dep graph and see if we should shift
   responsibilities among them or rename".
2. A.1.1's "Shared types" table is the canonical move-list.
3. A.5's "REQUIRED" subsection is the literal Phase B.1 todo list.
4. A.5's "NOT in Phase B" subsection is the rationale for what's
   intentionally skipped.
5. The parity gate is `python3 scripts/orchestrator_parity_sweep.py`
   on a fresh `cargo build --release -p zen-metrics-cli --features
   sweep,png,gpu,gpu-cuda,orchestrator,orchestrator-all` (or
   whatever feature set the script needs — read its top docstring
   before running).
