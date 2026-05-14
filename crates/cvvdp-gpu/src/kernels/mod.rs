//! GPU kernels for the ColorVideoVDP still-image pipeline.
//!
//! Pipeline order (per reference/distorted side):
//!
//! 1. [`color`] — sRGB packed-u8 → linear → DKL opponent planar f32
//!    via the cvvdp RGB→LMS→DKL matrix product.
//! 2. [`pyramid`] — per-channel Weber-contrast decomposition:
//!    downscale + upscale + subtract builds the Laplacian-style layer,
//!    then `weber_contrast_compute_kernel` divides by the per-pixel
//!    achromatic `L_bkg` plane (clamped to `±1000` over
//!    `max(L_bkg, 0.01)`). Baseband bypasses Weber.
//! 3. [`csf`] — per-pixel CSF apply using the
//!    `csf_lut_weber_fixed_size` LUT, with bilinear interp over
//!    `(log_rho, log_L_bkg)` for all three `omega = 0` channels
//!    (A, RG, VY).
//! 4. [`masking`] — cvvdp `mult-mutual` masking with the `XCM_3X3`
//!    cross-channel matrix; small bands skip the σ = 3 PU blur
//!    (`pu_padsize = 6` gate).
//! 5. [`pool`] — per-band Minkowski accumulation + 3-stage host fold
//!    + `met2jod` piecewise.
//!
//! Numerical parity target: matches pycvvdp v0.5.4 within ~0.006 JOD
//! on the v1 R2 manifest across q1–q90 fixtures. Per-thread
//! accumulators stay in f64 where the reference uses f64 reductions;
//! otherwise f32.

pub mod color;
pub mod csf;
pub mod masking;
pub mod pool;
pub mod pyramid;
