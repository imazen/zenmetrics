//! Contrast-sensitivity weighting per pyramid band.
//!
//! Each band has an associated spatial frequency (cy/deg, derived from
//! the display pixels-per-degree and the band's level). cvvdp's
//! castleCSF gives the sensitivity for that frequency in the
//! achromatic channel; chrom-CSF variants handle RG and VY.
//!
//! The kernel multiplies each band's coefficient by its CSF weight
//! (`csf(freq, level, channel)`). Per-band weights are precomputed
//! host-side once per `(width, height, display_model)` triple and
//! uploaded as a small `Array<f32>` of length `n_levels × N_CHANNELS`.
//!
//! Compiling stub.

use cubecl::prelude::*;

/// Multiply `band` in-place by `weights[channel * n_levels + level]`.
/// `weights` is uploaded once per pipeline init.
#[cube(launch)]
#[allow(unused_variables)]
pub fn weight_band_kernel(
    band: &mut Array<f32>,
    weights: &Array<f32>,
    weight_idx: u32,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    // body lands with the actual CSF weights pinned.
}
