//! GPU kernels for the ColorVideoVDP still-image pipeline.
//!
//! Pipeline order (per reference/distorted side):
//!
//! 1. [`color`] — sRGB packed-u8 → linear → DKL opponent planar f32
//!    via the cvvdp RGB→LMS→DKL matrix product.
//! 2. [`pyramid`] — per-channel Weber-contrast decomposition. A
//!    Gaussian pyramid is built via `downscale_kernel`, then for
//!    each non-baseband level `upscale_v_kernel` + `upscale_h_kernel`
//!    expand `gauss[k+1]` and the fused `subtract_weber_3ch_kernel`
//!    writes `band[c] = clamp((fine[c] − upscaled[c]) /
//!    max(L_bkg, 0.01), ±1000)` plus the shared `log10(L_bkg)` for
//!    all 3 channels in one launch. Baseband bypasses Weber.
//! 3. [`csf`] — per-pixel CSF apply using the
//!    `csf_lut_weber_fixed_size` LUT, with bilinear interp over
//!    `(log_rho, log_L_bkg)` for all three `omega = 0` channels
//!    (A, RG, VY). The fused `csf_apply_6ch_kernel` runs the REF
//!    and DIST sides in a single launch per non-baseband level,
//!    sharing the per-pixel LUT bracket math across all 6 outputs.
//! 4. [`masking`] — cvvdp `mult-mutual` masking with the `XCM_3X3`
//!    cross-channel matrix. Non-baseband bands run
//!    `min_abs_3ch_kernel` → `pu_blur_h_3ch_kernel` →
//!    `pu_blur_v_3ch_scaled_kernel` (the v-pass folds the
//!    `* 10^MASK_C` post-scale) → `mult_mutual_3ch_with_blurred_kernel`,
//!    or fall back to `mult_mutual_3ch_no_blur_kernel` when either
//!    band dimension is ≤ `pu_padsize = 6`. Baseband bypasses
//!    masking via `diff_abs_3ch_kernel`.
//! 5. [`pool`] — `pool_band_kernel` writes per-(level, channel)
//!    Minkowski partials into a shared GPU buffer; the host fold
//!    reduces the resulting `n_levels × 3` Vec via
//!    `pool_band_finalize` + `do_pooling_and_jod_still_3ch` +
//!    `met2jod` piecewise.
//!
//! Numerical parity target: matches pycvvdp v0.5.4 within 0.005 JOD
//! on the v1 R2 manifest across q=1–90 fixtures (measured 0.0000–
//! 0.0031 since tick 207's tolerance tightening). Per-thread
//! accumulators stay in f64 where the reference uses f64 reductions;
//! otherwise f32.

pub mod color;
pub mod csf;
pub mod masking;
pub mod pool;
pub mod pyramid;
