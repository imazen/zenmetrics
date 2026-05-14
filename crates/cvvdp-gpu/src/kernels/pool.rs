//! Minkowski pooling per band.
//!
//! For each band of each channel, computes `sum_i diff_i^beta` over
//! all pixels and writes a single scalar partial. The host fold then:
//!
//! 1. Per-band: `D_band = (sum / n_pixels)^(1/beta_spatial)`
//! 2. Per-channel: `D_channel = (sum_band D_band^beta_band)^(1/beta_band)`
//! 3. Overall: `D = (sum_channel D_channel^beta_channel)^(1/beta_channel)`
//! 4. JOD: `JOD = jod_a - jod_b * D^jod_c`
//!
//! Compiling stub.

use cubecl::prelude::*;

/// One thread per pixel raises `band_diff[i]^beta` and atomically adds
/// into the per-band f32 accumulator at `out[band_idx]`. Host promotes
/// to f64 on read-back to preserve precision for large images.
#[cube(launch)]
#[allow(unused_variables)]
pub fn pool_band_kernel(
    band_diff: &Array<f32>,
    out: &mut Array<f32>,
    beta: f32,
    band_idx: u32,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
}
