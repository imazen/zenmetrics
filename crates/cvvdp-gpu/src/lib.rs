//! Multi-vendor GPU implementation of ColorVideoVDP (still-image mode),
//! built on [CubeCL](https://github.com/tracel-ai/cubecl) so the same
//! `#[cube]` kernel source dispatches across:
//!
//! - **CUDA** (NVIDIA) via the cubecl CUDA runtime
//! - **WGPU/Vulkan** (cross-platform) via the cubecl wgpu runtime
//! - **WGPU/DX12** (Windows) via the cubecl wgpu runtime
//! - **WGPU/Metal** — **NOT SUPPORTED on the production pool path**.
//!   `pool_band_3ch_lds_kernel` (the workgroup-LDS pool used by
//!   `compute_dkl_jod`) commits per-workgroup sums via
//!   `Atomic<f32>::fetch_add`, which cubecl-wgpu's Metal backend
//!   silently no-ops — every reduction returns zero and the JOD
//!   score collapses to the default (10.0 for identical, ~10.0 for
//!   different). Root cause + upstream patch:
//!   [`crates/zenmetrics-api/docs/CUBECL_METAL_ATOMIC_FIX.md`].
//!   `compute_dkl_jod_host_pool` (the host-pool fallback used for
//!   cubecl-cpu, see below) works on Metal because it reads D bands
//!   back to host before pooling. Use that path for Metal
//!   deployments until the upstream fix lands and the workspace
//!   cubecl pin bumps to a fork rev with `feat/metal-atomic-fix`.
//! - **HIP** (AMD ROCm) when the `hip` feature is enabled
//! - **CPU** via the cubecl CPU runtime — supported through
//!   [`Cvvdp::compute_dkl_jod_host_pool`] (tick 208), which reads
//!   D bands back to host and pools with the host-scalar
//!   `lp_norm_mean` instead of the GPU `pool_band_3ch_kernel`
//!   (which uses `Atomic<f32>::fetch_add`, unsupported by
//!   cubecl-cpu). For CUDA / wgpu-Vulkan / wgpu-DX12 prefer
//!   [`Cvvdp::compute_dkl_jod`] — it keeps the spatial reduction
//!   on-device and skips the per-band readback. **Metal users
//!   must use `compute_dkl_jod_host_pool` until the upstream fix
//!   lands** — see the Metal note above.
//!
//! ## Scope: still images, JOD score
//!
//! Targets bit-stable parity with the published `ColorVideoVDP` Python
//! reference (gfxdisp/ColorVideoVDP **v0.5.4**) for the **still-image**
//! code path. Video / temporal channels (sustained + transient) are
//! intentionally out of scope for v0; defer until still-mode parity is
//! locked.
//!
//! ## Algorithm shape
//!
//! Per (reference, distorted) sRGB-u8 pair:
//!
//! 1. **Display model**: sRGB byte → linear → display-emitted luminance
//!    (gamma + peak luminance + ambient).
//! 2. **Color transform**: linear RGB → DKL opponent space
//!    `(A, RG, VY)`.
//! 3. **Pyramid**: per-channel **Weber-contrast** pyramid
//!    (`contrast="weber_g1"`) — non-baseband bands are
//!    `clip(layer / max(L_bkg, 0.01), ±1000)` with `L_bkg` taken from
//!    the per-pixel expanded achromatic gauss. Baseband bypasses
//!    Weber and feeds directly into pooling. ~7 levels for a
//!    1024-wide image.
//! 4. **CSF**: per-pixel LUT lookup of castleCSF
//!    `weber_fixed_size` — bilinear interp over
//!    `(log_rho, log_L_bkg)` for the three `omega = 0` channels
//!    (achromatic A, red-green RG, violet-yellow VY), then `T_p =
//!    weber × S × CH_GAIN`.
//! 5. **Masking**: cvvdp's `mult-mutual` with cross-channel pooling
//!    (`MASK_P / MASK_Q / MASK_C / D_MAX` + the `XCM_3X3` matrix).
//!    Bands smaller than `pu_padsize = 6` skip the σ = 3 PU blur;
//!    larger bands run separable 13-tap Gaussian blur first.
//! 6. **Pooling**: 3-stage Minkowski fold per `(band, channel)` →
//!    per-channel → overall `D`.
//! 7. **JOD**: piecewise [`kernels::pool::met2jod`] — two
//!    `jod_a/b/c` regimes joined continuously at `Q = 0.1`.
//!
//! ## Status
//!
//! Still-image score matches pycvvdp v0.5.4 within 0.005 JOD across
//! q=1–90 fixtures on the v1 R2 manifest under the default
//! [`PerfMode::Strict`] (tightened from ~0.006 in tick 207 after
//! ticks 204/206 closed the chroma_shift and 73×91 odd-dim drifts).
//! Measured GPU vs pycvvdp diffs: 0.0000–0.0031. [`PerfMode::Fast`]
//! is the opt-in entry point for stage-level relaxations; today
//! it's a no-op, so Strict and Fast produce identical output (see
//! `tests/pipeline_score.rs::perf_mode_fast_matches_strict_today`).
//!
//! The full GPU composition path is wired through
//! [`Cvvdp::compute_dkl_jod`]: color, Weber pyramid, CSF, masking,
//! and spatial pool all run on GPU; only the 3-stage Minkowski fold
//! and the `met2jod` mapping happen host-side, on a ~144-byte
//! partials Vec. The parity tests
//! `compute_dkl_jod_matches_host_scalar`,
//! `compute_dkl_jod_on_v1_manifest_corpus`, and
//! `compute_dkl_jod_vs_host_scalar_on_corpus` all lock the GPU path
//! within f32-precision tolerance of the host scalar reference.
//!
//! For batch scoring (one reference vs many distorted candidates),
//! [`Cvvdp::warm_reference`] + [`Cvvdp::compute_dkl_jod_with_warm_ref`]
//! caches the REF GPU state and skips that half of the pipeline per
//! candidate — ~1.6× faster per DIST at 12 MP (~2.1 → ~1.3 ns/px;
//! see the "How we compare to the canonical reference" section
//! below for the source). Parity vs the cold path is locked at
//! ≤ 1e-5 JOD by `compute_dkl_jod_with_warm_ref_matches_unwarm_path`.
//! The warm-state invalidation contract (which helpers reset the
//! cache vs preserve it) is pinned by
//! `warm_state_invalidates_after_each_documented_dispatcher`,
//! `set_reference_does_not_invalidate_warm_state`, and
//! `gauss_chain_helpers_do_not_invalidate_warm_state`. See
//! `docs/PORT_STATUS.md`'s "Resolved ticks 236-249" entry for
//! the audit history.
//!
//! ## How we compare to the canonical reference
//!
//! On an RTX 5070 at 12 MP (4000×3000) the **canonical** pycvvdp
//! v0.5.4 CUDA path lands at ~14 ns/px steady-state (after a 1–13 s
//! PyTorch graph compile). Our cold path runs **~2.1 ns/px** and our
//! warm-ref path **~1.3 ns/px** — **6.5× / 10.7× faster than
//! pycvvdp** (measured 2026-05-25, commit `6d3444de`). Both paths
//! return JOD 9.4580, matching pycvvdp exactly.
//!
//! Earlier versions of this text reported 62 / 34 ns/px (4.4× /
//! 2.4× *slower*). Those numbers were measured at tick 175
//! (2026-05-15) immediately after a ceil-div correctness fix. The
//! intervening 10 days of optimization work — buffer recycling
//! (tick 313: -90% alloc churn), SIMD masking pow/exp (ticks
//! 316–320), pyramid dispatch flattening — closed the gap and
//! then some. See `benchmarks/pycvvdp_12mp_cuda_2026-05-14.md`
//! for the original pycvvdp measurement, and run
//! `cargo run --release --example time_12mp -p cvvdp-gpu --features
//! cuda,cubecl-types --no-default-features` to reproduce on the
//! current code.
//!
//! Where we also win: multi-vendor backends (WGPU + HIP; pycvvdp
//! is CUDA-only via PyTorch), static-binary deployment (~50 MB vs
//! ~3 GB PyTorch runtime), and ~877 ms cold start vs 1–13 s.
//!
//! The public [`Cvvdp::score`] API now routes through the full GPU
//! composition path ([`Cvvdp::compute_dkl_jod`]) — tick 213 made
//! the switch after `shadow_jod_gpu` confirmed all 6 v1 R2
//! manifest q-levels match pycvvdp at ≤ 0.005 JOD. For the
//! host-scalar reference (slower but doesn't need a working GPU
//! pool), use [`host_scalar::predict_jod_still_3ch`] directly; for
//! the cpu cubecl runtime (no atomic f32), use
//! [`Cvvdp::compute_dkl_jod_host_pool`].
//!
//! ## Debug tracing env vars
//!
//! Two opt-in environment variables emit per-phase stderr timing
//! lines. Both default off (zero cost when unset — single
//! `var_os` lookup per call) and exist for ad-hoc profiling
//! without committing instrumentation. Set either to any
//! non-empty value to enable.
//!
//! - `CVVDP_TRACE=1` — instruments the JOD dispatch path
//!   ([`Cvvdp::compute_dkl_jod`] and the host-pool variants).
//!   Emits these stderr lines per call:
//!
//!   ```text
//!   [trace] weber(ref):  …               REF weber pyramid pass
//!   [trace] weber(dist): …               DIST weber pyramid pass
//!   [trace] L{k} log_l_bkg source ({bw}×{bh}): …
//!   [trace] L{k} csf 1 fused launch:     …   per-band CSF dispatch
//!   [trace] L{k} mask:                   …   per-band masking + (band total)
//!   [trace] band loop total ({n} levels): …
//!   ```
//!
//!   Useful for narrowing down which pyramid level dominates
//!   the per-image budget on a given GPU.
//!
//! - `CVVDP_TRACE_WEBER=1` — instruments
//!   [`Cvvdp::compute_dkl_weber_pyramid`] only, splitting GPU
//!   dispatch from host readback so we can see which side
//!   dominates when the weber pyramid is the target of a
//!   microbenchmark. Emits these stderr lines per call:
//!
//!   ```text
//!   [weber-trace] GPU dispatch + baseband host (before readback): …
//!   [weber-trace] bands readback ({n} levels): …
//!   [weber-trace] log_l_bkg readback: …
//!   ```
//!
//! Both variables are read once per call via `std::env::var_os`
//! — fine for one-off diagnostics, but if you need sub-call
//! granularity, prefer an external profiler over toggling these
//! mid-run. The output goes to stderr only; release builds
//! emit nothing unless explicitly enabled.

#![allow(clippy::needless_range_loop)]
// cvvdp parameters + the per-(rho, L_bkg, channel) CSF LUT are imported
// verbatim from pycvvdp v0.5.4 source. The literals carry more digits
// than f32 can represent so the values document the source even though
// LLVM rounds at compile time.
#![allow(clippy::excessive_precision)]
// Tick 516: pin the missing_docs-clean state established at tick
// 514. Crate-level `warn` (not `deny`) so a new undocumented item
// surfaces as a warning during local dev + cargo doc, but doesn't
// hard-block. The kernel files override to `allow` via their own
// inner attribute to silence `#[cube(launch)]` macro-emitted items
// (their own non-kernel pub items remain documented).
#![warn(missing_docs)]

// `heatmap` is a diffmap-visualization feature with no in-tree consumer
// yet; kept reachable (external viz tools) but `#[doc(hidden)]`.
#[doc(hidden)]
pub mod heatmap;
// `host_scalar` (the CPU scalar reference) and `kernels` are reached
// by-path from this crate's own parity benches/examples/tests and, for
// `kernels`, cross-crate (the cvvdp CPU crate re-exports the scalar
// kernels; cvvdp-conformance asserts against them). `#[doc(hidden)]`:
// reachable for those harnesses, but not a supported per-crate API.
#[doc(hidden)]
pub mod host_scalar;
#[doc(hidden)]
pub mod kernels;
pub mod memory_mode;
pub(crate) mod opaque;
pub mod params;
// `pipeline` is reached by-path cross-crate (the cvvdp CPU crate's
// strip walker) — `#[doc(hidden)]`, like `session`.
#[doc(hidden)]
pub mod pipeline;
// `presets` re-exports the cvvdp CPU crate's display registry so
// `cvvdp_gpu::presets::*` callsites resolve; `#[doc(hidden)]` (the
// canonical owner is `cvvdp::presets`).
#[doc(hidden)]
pub mod presets;

// Stream-bound session plumbing for `zenmetrics_api::MetricSession`
// (issue #17). `#[doc(hidden)]` internal surface, gated `cubecl-types`
// (the stream binding needs the cubecl client type). Not a supported
// per-crate API — use `zenmetrics_api::MetricSession`.
#[cfg(feature = "cubecl-types")]
#[doc(hidden)]
pub mod session;

// Unified MemoryMode surface — see `memory_mode.rs`.
pub use memory_mode::{
    MemoryMode, ResolvedMode, ScoreResourceEstimate, estimate_gpu_memory_bytes_usize,
    estimate_score_resources, estimate_score_time_ms, vram_cap_bytes,
};

// CvvdpParams stays unconditionally public — it's part of the opaque
// API surface and contains no cubecl types.
pub use params::{CvvdpParams, PerfMode};

// Uniform opaque API (Phase 2). See `opaque.rs`.
pub use opaque::{Backend, CvvdpOpaque, Score};

// Typed-generic API (gated behind `cubecl-types`).
#[cfg(feature = "cubecl-types")]
pub use pipeline::{
    Cvvdp, PARALLEL_SAFETY_FACTOR, estimate_gpu_memory_bytes, estimate_gpu_memory_bytes_capped,
    estimate_gpu_memory_bytes_strip, estimate_gpu_memory_bytes_strip_pair, recommend_parallel,
};

/// Number of color channels in DKL opponent space (achromatic +
/// red-green + violet-yellow).
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::N_CHANNELS;
/// // cvvdp's still-image path is fixed at 3 DKL channels.
/// assert_eq!(N_CHANNELS, 3);
/// ```
pub const N_CHANNELS: usize = 3;

/// Maximum pyramid depth supported by the kernel allocations.
/// `pipeline::pyramid_levels` caps the per-image pyramid depth at
/// this value, so images with `min(w, h) > PYRAMID_MIN_DIM ×
/// 2^MAX_LEVELS` (≈ 1024 with the defaults) get only `MAX_LEVELS`
/// bands — coarser frequency content above the cap is folded into
/// the baseband.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::MAX_LEVELS;
/// // Pinned by tests/lib_constants.rs::max_levels_cap_at_nine.
/// // A bump requires resizing logs_row / partials / weights buffers.
/// assert_eq!(MAX_LEVELS, 9);
/// ```
pub const MAX_LEVELS: usize = 9;

/// Smallest logical width/height at which the pyramid keeps
/// building further coarse levels. Once `min(w, h) < 2 ×
/// PYRAMID_MIN_DIM`, the current level becomes the baseband.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::PYRAMID_MIN_DIM;
/// // 2 × PYRAMID_MIN_DIM = 8 is the minimum image dim accepted by
/// // Cvvdp::new and estimate_gpu_memory_bytes (smaller returns Err /
/// // None respectively). Pinned by tests/lib_constants.rs.
/// assert_eq!(PYRAMID_MIN_DIM, 4);
/// assert_eq!(PYRAMID_MIN_DIM * 2, 8);
/// ```
pub const PYRAMID_MIN_DIM: u32 = 4;

/// Stable column-name identifier for this implementation snapshot.
///
/// Used by sweep tooling (`zenmetrics-cli` and downstream
/// pipelines) to land cvvdp scores in parquet sidecars without
/// colliding with other cvvdp variants such as the canonical
/// pycvvdp reference (`cvvdp_pycvvdp_v054`) or a future alternative
/// implementation. (A Burn-based port was investigated and
/// abandoned tick 324; see `docs/BURN_PORT_PLAN.md`'s banner.)
/// See the PINNED TASK in `CLAUDE.md` at repo root.
///
/// Default form: `cvvdp_imazen_v<MAJOR>_<MINOR>_<PATCH>` derived
/// from `CARGO_PKG_VERSION` with `.` rewritten to `_`. The
/// `CVVDP_IMPL_TAG` build-time env var overrides the entire
/// string when set (e.g. CI bakes in a git short hash to
/// distinguish iterations within the same crate version).
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::CVVDP_COLUMN_NAME;
/// // Must claim the `cvvdp_imazen_*` namespace (avoids collision
/// // with `cvvdp_pycvvdp_v054` reference + `cvvdp_burn_*` reserved
/// // namespace). Pinned by tests/column_name.rs.
/// assert!(CVVDP_COLUMN_NAME.starts_with("cvvdp_"));
/// // The full string is parquet-safe: only ASCII alphanumerics
/// // and `_`. No whitespace, path separators, or shell metachars
/// // — survives every downstream tool (parquet columns, TSV
/// // headers, R2 filename derivations, Python attribute access).
/// for c in CVVDP_COLUMN_NAME.chars() {
///     assert!(c.is_ascii_alphanumeric() || c == '_');
/// }
/// ```
pub const CVVDP_COLUMN_NAME: &str = match option_env!("CVVDP_IMPL_TAG") {
    Some(t) => t,
    None => concat!(
        "cvvdp_imazen_v",
        env!("CARGO_PKG_VERSION_MAJOR"),
        "_",
        env!("CARGO_PKG_VERSION_MINOR"),
        "_",
        env!("CARGO_PKG_VERSION_PATCH"),
    ),
};

/// The pinned [`gfxdisp/ColorVideoVDP`](https://github.com/gfxdisp/ColorVideoVDP)
/// reference version this implementation tracks for parity.
///
/// Compile-time lockstep enforcement: bumping this const will FAIL
/// TO COMPILE unless these files are updated in the same commit
/// (ticks 588-595 added one pin per site to `tests/version_lockstep.rs`,
/// which runs on every `cargo check / test` — NOT only under
/// `--features parity-goldens`):
///
/// - `scripts/cvvdp_goldens/requirements.txt` (`cvvdp==X.Y.Z` pip
///   pin; matched against the bare version after stripping the
///   leading `v`)
/// - `src/kernels/csf_lut/v0_5_4.rs` (vendored sensitivity LUT;
///   matched against the auto-generated header comment)
/// - `docs/PORT_STATUS.md` ("Reference version pin" section)
/// - `README.md` (algorithm-parity claim + Status section)
/// - `Cargo.toml` (parity-goldens feature comment)
/// - `docs/CVVDP_SIDECAR_SCHEMA.md` (reserved column-name tags)
/// - `tests/parity.rs::manifest_fetches` (runtime manifest version
///   check — gated behind `--features parity-goldens`)
///
/// Plus 3 compile-time format invariants on the const itself:
/// non-empty, starts with `v`, contains `.`.
///
/// Sites NOT pinnable to this const (Rust identifiers / paths the
/// compiler doesn't expose as strings):
///
/// - `kernels::csf::csf_lut_v0_5_4` (re-export module name)
/// - `src/kernels/csf_lut/v0_5_4.rs` (filesystem path)
///
/// Sites INTENTIONALLY NOT pinned (historical / abandoned material,
/// not current-state docs):
///
/// - `docs/CHROMA_DRIFT_INVESTIGATION.md` (tick-200-era bug-hunt log)
/// - `docs/BURN_PORT_PLAN.md` (abandoned tick 324)
///
/// Separately, `tests/it/common/mod.rs` has `GOLDEN_VERSION = "v1"`,
/// which is the **R2 bucket prefix** version (a different version
/// space from this const). Goldens under `/v1/` were captured
/// against pycvvdp v0.5.4. Both bumps are needed when the goldens
/// are regenerated — see `docs/PORT_STATUS.md#reference-version-pin`
/// for the full procedure.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::PYCVVDP_REFERENCE_VERSION;
/// // v<major>.<minor>.<patch> format with leading 'v'. The three
/// // format invariants are also enforced at compile time in
/// // `tests/version_lockstep.rs` so a malformed bump fails to
/// // build.
/// assert!(PYCVVDP_REFERENCE_VERSION.starts_with('v'));
/// assert!(PYCVVDP_REFERENCE_VERSION.contains('.'));
/// assert!(!PYCVVDP_REFERENCE_VERSION.is_empty());
/// // Strip the leading 'v' to derive the bare pip-style version
/// // that `scripts/cvvdp_goldens/requirements.txt` matches.
/// assert!(PYCVVDP_REFERENCE_VERSION[1..].split('.').all(|s| s.parse::<u32>().is_ok()));
/// ```
pub const PYCVVDP_REFERENCE_VERSION: &str = "v0.5.4";

/// Failure modes for `Cvvdp::*` methods. Implements
/// `std::error::Error` so callers can use `?` against
/// `Box<dyn Error>` or `anyhow::Error` as usual.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::Error;
///
/// // `DimensionMismatch` carries both expected + actual lengths
/// // so callers can surface a precise diagnostic.
/// let e = Error::DimensionMismatch { expected: 12_288, got: 3_072 };
/// let msg = e.to_string();
/// assert!(msg.contains("expected"));
/// assert!(msg.contains("12288") || msg.contains("12_288"));
///
/// // The four zero-payload variants share a Display impl that
/// // surfaces an actionable hint pointing at the caller-fix.
/// assert!(Error::NoCachedReference.to_string().contains("set_reference"));
/// assert!(Error::NoWarmReference.to_string().contains("warm_reference"));
/// assert!(!Error::InvalidImageSize.to_string().is_empty());
///
/// // Error implements std::error::Error → bubbles through `?` against
/// // any `Box<dyn Error>` / `anyhow::Error` return type.
/// fn _bubble() -> Result<(), Box<dyn std::error::Error>> {
///     Err(Error::NoCachedReference)?;
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub enum Error {
    /// Buffer length doesn't match `width × height × 3`.
    DimensionMismatch {
        /// Required buffer length: `width × height × 3` bytes.
        expected: usize,
        /// Actual length of the buffer passed by the caller.
        got: usize,
    },
    /// `Cvvdp::score_with_reference` was called without a prior
    /// `Cvvdp::set_reference`.
    NoCachedReference,
    /// `Cvvdp::compute_dkl_jod_with_warm_ref` or
    /// `Cvvdp::compute_dkl_jod_host_pool_with_warm_ref` was called
    /// without a prior `Cvvdp::warm_reference`, **or** the warm
    /// state was invalidated by an intervening REF-dispatching
    /// method. The canonical invalidator list lives on
    /// `Cvvdp::warm_reference`'s docstring; the
    /// `warm_state_invalidates_after_each_documented_dispatcher`
    /// regression test pins each method to the contract.
    NoWarmReference,
    /// Image is too small for the configured pyramid, **or** a GPU
    /// read-back / dispatch failed. The two get the same variant
    /// because cubecl's read errors aren't easily separable yet —
    /// callers in tests / production should treat this as "GPU
    /// pipeline failed, retry or surface to user".
    InvalidImageSize,
    /// The requested [`MemoryMode`](crate::MemoryMode) variant isn't
    /// implemented in cvvdp-gpu. Kept in the error enum for API
    /// stability — the unified `MemoryMode` enum may in the future
    /// gain variants this crate doesn't support, and the umbrella
    /// `From` conversions map unsupported umbrella variants down to
    /// `Auto` so this error never fires today. Currently supported
    /// variants: `Full`, `Auto`, `Strip { h_body }` (Mode E),
    /// `StripPair { h_body }` (Mode B), `CappedPyramid { levels }`.
    /// See `docs/STRIP_PROCESSING.md` for the strip-walker design and
    /// the rolled-back capped-levels variant that task #77 removed
    /// (capped-levels Strip changed JOD; the current CappedPyramid
    /// is the explicit JOD-shifting opt-in successor).
    ModeUnsupported(&'static str),
    /// [`MemoryMode::Auto`](crate::MemoryMode) couldn't fit the image
    /// into the VRAM cap, even after attempting [`MemoryMode::Strip`]
    /// fallback.
    TooBigForFull {
        /// Estimated working-set bytes Full would allocate.
        needed: usize,
        /// Configured VRAM cap.
        cap: usize,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::DimensionMismatch { expected, got } => write!(
                f,
                "dimension mismatch: expected {expected} bytes, got {got}"
            ),
            Error::NoCachedReference => write!(f, "no cached reference; call set_reference first"),
            Error::NoWarmReference => write!(
                f,
                "no warm GPU reference; call warm_reference first (or warm state was invalidated by an intervening REF dispatch)"
            ),
            Error::InvalidImageSize => write!(
                f,
                "image too small for the configured pyramid, or GPU readback/dispatch failed (see the InvalidImageSize variant docs — cubecl's read errors aren't separable yet so both surface as this variant)"
            ),
            Error::ModeUnsupported(variant) => write!(
                f,
                "MemoryMode::{variant} is not supported in cvvdp-gpu \
                 (currently Full / Auto / Strip / StripPair / CappedPyramid \
                 are supported; see docs/STRIP_PROCESSING.md for the strip-mode lineage)"
            ),
            Error::TooBigForFull { needed, cap } => write!(
                f,
                "Auto could not fit image in {cap} byte cap even after \
                 trying Strip fallback; needs at least {needed} bytes — \
                 raise ZENMETRICS_VRAM_CAP_BYTES, use a smaller image, or \
                 construct directly via Cvvdp::new_strip_pair for one-shot \
                 scoring (which has a lower memory floor than Auto's \
                 preferred picks)"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// `Result<T, cvvdp_gpu::Error>` — the crate's standard fallible
/// return type. Every `Cvvdp::*` constructor and dispatch method
/// returns this.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::{Error, Result};
///
/// // Construct an Ok / Err via the type alias.
/// let ok: Result<f32> = Ok(7.5);
/// let err: Result<f32> = Err(Error::NoCachedReference);
/// assert_eq!(ok.ok(), Some(7.5));
/// assert!(err.is_err());
///
/// // The alias is just `std::result::Result<T, cvvdp_gpu::Error>` —
/// // composes with `?` against the same return type without
/// // any `.map_err(Into::into)`.
/// fn doubles(input: Result<f32>) -> Result<f32> {
///     Ok(input? * 2.0)
/// }
/// assert_eq!(doubles(Ok(3.0)).ok(), Some(6.0));
/// ```
pub type Result<T> = std::result::Result<T, Error>;
