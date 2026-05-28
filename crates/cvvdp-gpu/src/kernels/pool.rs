//! Pooling + JOD for still-image cvvdp.
//!
//! cvvdp v0.5.4's pipeline collapses per-pixel masked differences `D`
//! into a scalar quality-in-JOD via a 3-stage Minkowski pool plus a
//! piecewise transform:
//!
//! 1. **Spatial pooling per band per channel** (beta = 2 = RMS):
//!    `Q_per_ch[c, k] = (mean over pixels of D[c, :, :]^2)^(1/2)`.
//! 2. **Band pooling per channel** (beta_sch = 4):
//!    `Q_sc[c] = (sum_k (Q_per_ch[c, k] * per_ch_w[c] * per_sband_w[c, k])^4)^(1/4)`
//!    where `per_sband_w[c, k] = 1` for `k < n_levels - 1` and
//!    `per_sband_w[c, last] = baseband_weight[c]`.
//! 3. **Channel pooling** (beta_tch = 4):
//!    `Q_tc = (sum_c Q_sc[c]^4)^(1/4)`.
//! 4. **Image integration**: `Q = Q_tc * image_int`.
//! 5. **JOD mapping**: piecewise (smooth at Q = 0.1).
//!
//! Constants are baked from `cvvdp_parameters.json`. Still-image
//! 3-channel only — temporal channel (no_frames > 1) lives outside
//! this module.

// Tick 514: silence missing_docs warnings on items emitted by the
// #[cube(launch)] macro (see kernels/color.rs for full rationale).
#![allow(missing_docs)]

use cubecl::prelude::*;

// Phase 8c.1-C: scalar items (BETA_SPATIAL, BETA_BAND, BETA_CH,
// IMAGE_INT, JOD_A, JOD_EXP, PER_CH_W, BASEBAND_W constants plus the
// `lp_norm_mean` / `lp_norm_sum` / `met2jod` /
// `do_pooling_and_jod_still_3ch` host-scalar helpers) live in
// `cvvdp::kernels::pool` so the CPU crate owns the canonical scalar
// implementation. Re-export the surface so existing
// `cvvdp_gpu::kernels::pool::*` callsites resolve unchanged.
//
// The cube-macro `#[cube(launch)]` kernels below (pool_band_kernel,
// pool_band_3ch_kernel, pool_band_3ch_offset_kernel,
// pool_band_3ch_lds_kernel, fill_f32_kernel, copy_f32_kernel) all
// take `beta` as a runtime scalar kernel argument — they do NOT
// reference any of the moved constants by name inside the cube
// body. `POOL_LDS_BLOCK_DIM` (the LDS workgroup size used by
// pool_band_3ch_lds_kernel) stays in cvvdp-gpu since it's GPU
// launch-configuration metadata, not a scalar reference.
pub use cvvdp::kernels::pool::{
    BASEBAND_W, BETA_BAND, BETA_CH, BETA_SPATIAL, IMAGE_INT, JOD_A, JOD_EXP, PER_CH_W,
    do_pooling_and_jod_still_3ch, lp_norm_mean, lp_norm_sum, met2jod,
};

/// One thread per pixel computes cvvdp's `safe_pow(|x|, β) =
/// (|x| + 1e-5)^β - 1e-5^β` for the pixel and atomically adds it
/// into the f32 accumulator at `partials[partial_idx]`. Host folds
/// the partial to the final lp_norm via:
///
/// ```text
/// Q = safe_pow(partial / n_pixels, 1/β)
///   = ((partial / n_pixels) + 1e-5)^(1/β) - 1e-5^(1/β)
/// ```
///
/// `partial_idx` lets the caller pack multiple (band, channel)
/// partials into the same buffer. Works on cubecl backends with
/// `Atomic<f32>::fetch_add` support — CUDA, DX12, HIP (per
/// butteraugli-gpu's notes; Metal silently no-ops on the f32 add).
///
/// **Not dispatched by `Cvvdp::compute_dkl_jod`** — the production
/// path uses the 3-channel fused [`pool_band_3ch_kernel()`] (one
/// launch per band instead of three). `pool_band_kernel` is kept
/// as a test-only entry point for the scalar parity test
/// `tests/pool_scalar.rs::pool_band_kernel_matches_host_lp_norm_mean`.
#[cube(launch)]
pub fn pool_band_kernel(
    band_diff: &Array<f32>,
    partials: &mut Array<Atomic<f32>>,
    beta: f32,
    partial_idx: u32,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    let v = band_diff[idx];
    let abs_v = if v < f32::new(0.0) { -v } else { v };
    let eps = f32::new(1e-5);
    // safe_pow_lp(|v|, beta) — accumulator gets the raw safe-pow
    // contribution; the - eps^beta and 1/beta exponentiation
    // happen host-side once per (band, channel).
    let contribution = f32::powf(abs_v + eps, beta) - f32::powf(eps, beta);
    partials[partial_idx as usize].fetch_add(contribution);
}

/// 3-channel fused version of `pool_band_kernel`. Same per-pixel
/// safe_pow math, but takes 3 input arrays and 3 partial slot
/// indices, doing 3 atomic-adds per thread (each into a different
/// slot of `partials`). Eliminates 2/3 of the launch overhead for
/// the per-band pool dispatch in `compute_dkl_jod`.
///
/// Each thread reads `band_diff_{a,rg,vy}[idx]`, computes the
/// `safe_pow` contribution for each channel, and atomically adds
/// to `partials[partial_idx_{a,rg,vy}]`. The host-side fold and
/// `pool_band_finalize` semantics are unchanged.
///
/// Pool atomics into distinct slots don't contend across channels,
/// so the atomic-throughput characteristic is the same as 3 separate
/// launches — the win is purely launch-overhead reduction (which
/// matters more at small image sizes per the tick 164 size sweep).
#[cube(launch)]
pub fn pool_band_3ch_kernel(
    band_diff_a: &Array<f32>,
    band_diff_rg: &Array<f32>,
    band_diff_vy: &Array<f32>,
    partials: &mut Array<Atomic<f32>>,
    beta: f32,
    partial_idx_a: u32,
    partial_idx_rg: u32,
    partial_idx_vy: u32,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    let eps = f32::new(1e-5);
    let eps_pow_beta = f32::powf(eps, beta);

    let v_a = band_diff_a[idx];
    let abs_a = if v_a < f32::new(0.0) { -v_a } else { v_a };
    let c_a = f32::powf(abs_a + eps, beta) - eps_pow_beta;
    partials[partial_idx_a as usize].fetch_add(c_a);

    let v_rg = band_diff_rg[idx];
    let abs_rg = if v_rg < f32::new(0.0) { -v_rg } else { v_rg };
    let c_rg = f32::powf(abs_rg + eps, beta) - eps_pow_beta;
    partials[partial_idx_rg as usize].fetch_add(c_rg);

    let v_vy = band_diff_vy[idx];
    let abs_vy = if v_vy < f32::new(0.0) { -v_vy } else { v_vy };
    let c_vy = f32::powf(abs_vy + eps, beta) - eps_pow_beta;
    partials[partial_idx_vy as usize].fetch_add(c_vy);
}

/// Strip-aware sibling of [`pool_band_3ch_kernel`] (Phase 3, task #79).
///
/// Identical math but the per-thread pixel index is `start_offset +
/// ABSOLUTE_POS`, so the kernel can be dispatched on a **slab** of a
/// larger d-plane without producing a fresh sub-array. The host side
/// dispatches it per strip:
///
/// ```text
/// for strip in 0..n_strips {
///     let start = strip * strip_n;
///     let count = (band_n - start).min(strip_n);
///     pool_band_3ch_offset_kernel::launch(.., start as u32, count as u32, band_n as u32);
/// }
/// ```
///
/// The atomic-add into `partials[partial_idx_*]` is associative, so
/// the total written across strip dispatches equals the single full-
/// band dispatch result to f32 rounding. This is the load-bearing
/// invariant for the Mode E Phase 3 strip walker.
///
/// `band_total` is the underlying array's logical length; the kernel
/// silently skips threads whose computed `start + tid` would exceed
/// it (defensive against rounding in `n`).
#[cube(launch)]
pub fn pool_band_3ch_offset_kernel(
    band_diff_a: &Array<f32>,
    band_diff_rg: &Array<f32>,
    band_diff_vy: &Array<f32>,
    partials: &mut Array<Atomic<f32>>,
    beta: f32,
    partial_idx_a: u32,
    partial_idx_rg: u32,
    partial_idx_vy: u32,
    start_offset: u32,
    n: u32,
    band_total: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid >= n as usize {
        terminate!();
    }
    let idx = tid + start_offset as usize;
    if idx >= band_total as usize {
        terminate!();
    }
    let eps = f32::new(1e-5);
    let eps_pow_beta = f32::powf(eps, beta);

    let v_a = band_diff_a[idx];
    let abs_a = if v_a < f32::new(0.0) { -v_a } else { v_a };
    let c_a = f32::powf(abs_a + eps, beta) - eps_pow_beta;
    partials[partial_idx_a as usize].fetch_add(c_a);

    let v_rg = band_diff_rg[idx];
    let abs_rg = if v_rg < f32::new(0.0) { -v_rg } else { v_rg };
    let c_rg = f32::powf(abs_rg + eps, beta) - eps_pow_beta;
    partials[partial_idx_rg as usize].fetch_add(c_rg);

    let v_vy = band_diff_vy[idx];
    let abs_vy = if v_vy < f32::new(0.0) { -v_vy } else { v_vy };
    let c_vy = f32::powf(abs_vy + eps, beta) - eps_pow_beta;
    partials[partial_idx_vy as usize].fetch_add(c_vy);
}

/// Workgroup size for the LDS-reduction pool kernel.
pub const POOL_LDS_BLOCK_DIM: u32 = 256;
const POOL_LDS_BLOCK_DIM_USIZE: usize = 256;

/// LDS-reduction 3-channel pool kernel (T1.C). Same math as
/// [`pool_band_3ch_kernel`] but with workgroup-level reduction in
/// shared memory, then one atomic add per workgroup per channel
/// to commit to the global partial.
///
/// At 12 MP per band per channel that drops the atomic count from
/// 12M to 12M/256 ≈ 47K — about a 255× reduction in atomic traffic.
///
/// **Workgroup**: `POOL_LDS_BLOCK_DIM = 256` threads, 1D. Each thread
/// loads one pixel (or contributes 0 if out of bounds), accumulates
/// `safe_pow(|v|, β) - eps^β` into `groupshared[lid]` per channel,
/// runs a 256→1 pointer-jumping reduce, and thread 0 atomic-adds
/// the three workgroup sums.
///
/// **Launch**:
///
/// ```text
/// cube_dim   = CubeDim::new_1d(POOL_LDS_BLOCK_DIM)
/// cube_count = (n.div_ceil(POOL_LDS_BLOCK_DIM), 1, 1)
/// ```
///
/// Produces the same `partials[slot]` value as the per-pixel-atomic
/// kernel (to f32 rounding). The host-side fold via
/// [`pool_band_finalize`] is unchanged.
#[cube(launch)]
pub fn pool_band_3ch_lds_kernel(
    band_diff_a: &Array<f32>,
    band_diff_rg: &Array<f32>,
    band_diff_vy: &Array<f32>,
    partials: &mut Array<Atomic<f32>>,
    beta: f32,
    partial_idx_a: u32,
    partial_idx_rg: u32,
    partial_idx_vy: u32,
    n: u32,
) {
    let tx = UNIT_POS_X;
    let idx = ABSOLUTE_POS;
    let n_usize = n as usize;

    let eps = f32::new(1e-5);
    let eps_pow_beta = f32::powf(eps, beta);

    // Safe-load index: read from `idx` if in range, else from 0 (and
    // mask the contribution to 0 below). Avoids OOB Array access.
    // `idx - idx` produces a typed-zero of `ABSOLUTE_POS`' tracked
    // `usize` type, sidestepping the `0usize` literal type mismatch
    // CubeCL surfaces in mixed-arm `if` expressions.
    let in_range = idx < n_usize;
    // `idx - idx` is an intentional typed-zero (see comment above), not
    // a mistake — clippy's eq_op can't see the CubeCL type constraint.
    #[allow(clippy::eq_op)]
    let zero_idx = idx - idx;
    let safe_idx = if in_range { idx } else { zero_idx };

    let v_a = band_diff_a[safe_idx];
    let abs_a = if v_a < f32::new(0.0) { -v_a } else { v_a };
    let c_a_raw = f32::powf(abs_a + eps, beta) - eps_pow_beta;
    let c_a = if in_range { c_a_raw } else { f32::new(0.0) };

    let v_rg = band_diff_rg[safe_idx];
    let abs_rg = if v_rg < f32::new(0.0) { -v_rg } else { v_rg };
    let c_rg_raw = f32::powf(abs_rg + eps, beta) - eps_pow_beta;
    let c_rg = if in_range { c_rg_raw } else { f32::new(0.0) };

    let v_vy = band_diff_vy[safe_idx];
    let abs_vy = if v_vy < f32::new(0.0) { -v_vy } else { v_vy };
    let c_vy_raw = f32::powf(abs_vy + eps, beta) - eps_pow_beta;
    let c_vy = if in_range { c_vy_raw } else { f32::new(0.0) };

    let mut lds_a = SharedMemory::<f32>::new(POOL_LDS_BLOCK_DIM_USIZE);
    let mut lds_rg = SharedMemory::<f32>::new(POOL_LDS_BLOCK_DIM_USIZE);
    let mut lds_vy = SharedMemory::<f32>::new(POOL_LDS_BLOCK_DIM_USIZE);

    let tx_us = tx as usize;
    lds_a[tx_us] = c_a;
    lds_rg[tx_us] = c_rg;
    lds_vy[tx_us] = c_vy;
    sync_cube();

    // Pointer-jumping reduce. Active-contiguous form (`tx < stride`),
    // which keeps active threads in a contiguous range and avoids the
    // warp-divergence of `tx % (stride*2) == 0`. Eight passes for
    // POOL_LDS_BLOCK_DIM = 256: stride 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1.
    let mut stride: u32 = 128u32;
    while stride > 0u32 {
        if tx < stride {
            let other = (tx + stride) as usize;
            lds_a[tx_us] = lds_a[tx_us] + lds_a[other];
            lds_rg[tx_us] = lds_rg[tx_us] + lds_rg[other];
            lds_vy[tx_us] = lds_vy[tx_us] + lds_vy[other];
        }
        sync_cube();
        stride /= 2u32;
    }

    // Thread 0 commits this workgroup's three sums to the global
    // atomic partials.
    if tx == 0u32 {
        partials[partial_idx_a as usize].fetch_add(lds_a[0]);
        partials[partial_idx_rg as usize].fetch_add(lds_rg[0]);
        partials[partial_idx_vy as usize].fetch_add(lds_vy[0]);
    }
}

/// Write the same `value` to every slot of `dest`. Used by the
/// baseband CSF path in `_dispatch_d_bands_into_scratch` to fill
/// `baseband_log_l_bkg` from the host-computed scalar
/// `log_l_bkg_baseband` — replaces a host `vec![value; n]` alloc
/// + GPU upload with a single GPU launch and zero host bytes.
#[cube(launch)]
pub fn fill_f32_kernel(dest: &mut Array<f32>, value: f32, n: u32) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    dest[idx] = value;
}

/// Copy `n` f32 slots from `src` to `dst`. Used by the strip-mode
/// cached-ref path to populate the dedicated `RefFullState` buffers
/// from the shared `bands_ref` / `weber_scratch.log_l_bkg` scratch
/// after a `warm_reference` dispatch, and to restore them back into
/// the shared scratch before a strip-mode dist scoring call. Both
/// arrays must be at least `n` long; threads beyond `n` exit.
#[cube(launch)]
pub fn copy_f32_kernel(src: &Array<f32>, dst: &mut Array<f32>, n: u32) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    dst[idx] = src[idx];
}

/// Finish the host-side fold for the per-band atomic-pool
/// kernels ([`pool_band_kernel()`] and the fused
/// [`pool_band_3ch_kernel()`] used in production): given the
/// atomic partial sum and pixel count for one (band, channel)
/// slot, return the lp_norm_mean(β) value matching
/// `kernels::pool::lp_norm_mean`. Same algebra regardless of
/// which kernel produced the partial — both write the raw
/// `safe_pow(|x|, β)` contribution into `partials[partial_idx]`.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::pool::pool_band_finalize;
///
/// // Zero partial → zero output (apart from the eps-tail bias,
/// // which the function explicitly cancels via `- eps.powf(1/β)`).
/// assert_eq!(pool_band_finalize(0.0, 100, 2.0), 0.0);
///
/// // Negative partial clamps to zero (the kernel sometimes produces
/// // a tiny negative from f32 atomic rounding).
/// assert_eq!(pool_band_finalize(-1e-7, 100, 2.0), 0.0);
///
/// // For a uniform |x| = c contribution, partial = N * c^β,
/// // and the finalized output is ≈ c minus the constant eps-tail
/// // `eps^(1/β)` (~ 0.056 at β=4 for eps=1e-5; ~ 0.003 at β=2).
/// // Use β=2 here so the tolerance is meaningful.
/// let c = 2.0_f32;
/// let n = 100_usize;
/// let beta = 2.0_f32;
/// let partial = (n as f32) * c.powf(beta);
/// let v = pool_band_finalize(partial, n, beta);
/// assert!((v - c).abs() < 0.01, "got {v}, expected ≈ {c}");
/// ```
#[must_use]
pub fn pool_band_finalize(partial: f32, n_pixels: usize, beta: f32) -> f32 {
    let n = n_pixels as f32;
    let eps = 1e-5_f32;
    ((partial / n).max(0.0) + eps).powf(1.0 / beta) - eps.powf(1.0 / beta)
}
