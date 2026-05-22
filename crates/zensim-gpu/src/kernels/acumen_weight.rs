//! Per-band per-pixel castleCSF Mode B weight application kernels.
//!
//! Used by `ZensimParams::with_acumen_arch(AcumenArch::ModeBPerBand)`.
//! The host computes 12 weight maps (3 channels × 4 pyramid scales),
//! each at that scale's `padded_w × h` resolution. Before the fused
//! features kernel runs at each scale, we element-wise multiply the
//! pyramid's src + dst per-channel arrays by the corresponding weight
//! map. The features kernel itself is unmodified — it sees
//! CSF-spatially-weighted pyramid content as if it had been built
//! from a weighted image, but with per-band fidelity that pre-image
//! multiplication can't achieve.

use cubecl::prelude::*;

/// Element-wise multiply a single channel's src array by its
/// per-pixel weight map. `n_elements = padded_w * h` of the
/// pyramid scale. One thread per pixel.
#[cube(launch_unchecked)]
pub fn apply_weight_kernel(
    src: &mut Array<f32>,
    weight: &Array<f32>,
    n_elements: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n_elements as usize {
        terminate!();
    }
    src[idx] = src[idx] * weight[idx];
}
