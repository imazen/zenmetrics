//! Contrast masking — within-channel and cross-channel.
//!
//! cvvdp's masking model (per pixel, per band):
//!
//! ```text
//! masked_diff = |R - T|^p / (k + neighborhood_excitation^q)
//! ```
//!
//! where `R` is the reference CSF-weighted coefficient, `T` is the
//! distorted one, and `neighborhood_excitation` is a local pooled
//! contrast that includes both within-channel and cross-channel terms.
//!
//! Exponents `p`, `q`, `k` come from [`crate::params::MaskingParams`]
//! and the published cvvdp JSON.
//!
//! Compiling stubs.

use cubecl::prelude::*;

/// Per-pixel masked difference: writes `out` from (`ref_band`,
/// `dist_band`, masker tensor). One thread per pixel.
#[cube(launch)]
#[allow(unused_variables)]
pub fn masked_diff_kernel(
    ref_band: &Array<f32>,
    dist_band: &Array<f32>,
    masker: &Array<f32>,
    out: &mut Array<f32>,
    p: f32,
    q: f32,
    k: f32,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    out[idx] = 0.0;
}
