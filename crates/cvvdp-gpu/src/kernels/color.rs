//! sRGB packed-u8 → linear → DKL opponent planar f32.
//!
//! Two stages fused into a single kernel pass:
//!
//! 1. **Display model**: sRGB byte → linear normalized `[0, 1]` via a
//!    256-entry LUT, then scaled by display peak luminance (+ reflected
//!    ambient) to produce display-emitted luminance in cd/m². Matches
//!    cvvdp's `display_model.forward` for the standard sRGB display.
//!
//! 2. **DKL transform**: linear RGB (cd/m²) → LMS via cvvdp's RGB→LMS
//!    matrix, then LMS → DKL via the LMS→DKL matrix. Outputs three
//!    planar f32 buffers: A (achromatic ≈ L+M), RG (red-green), VY
//!    (violet-yellow). cvvdp keeps these in absolute units rather than
//!    normalizing to [0, 1] — the CSF stage handles sensitivity.
//!
//! All matrix constants land here once the cvvdp reference JSON is
//! vendored. For now this is a compiling stub.

use cubecl::prelude::*;

/// 256-entry sRGB byte → linear normalized f32 LUT (sRGB piecewise
/// EOTF). Shared with other GPU metric crates in shape; verbatim
/// numbers will be copied from `zensim_gpu::kernels::color` when the
/// kernel body lands so the byte path is bit-identical across metrics.
#[rustfmt::skip]
pub const SRGB8_TO_LINEAR_LUT: [f32; 16] = [
    0.0, 0.000303527, 0.000607054, 0.00091058103,
    0.001214108, 0.001517635, 0.0018211621, 0.002124689,
    0.002428216, 0.002731743, 0.00303527, 0.0033465356,
    0.003676507, 0.004024717, 0.004391442, 0.0047769533,
];

/// sRGB packed-u8 → DKL planar f32.
///
/// Inputs:
/// - `src`            — `[height × width × 3]` packed sRGB bytes
///                      (laid out RGBRGB...).
/// - `lut`            — uploaded `SRGB8_TO_LINEAR_LUT` (256 entries
///                      once the kernel body lands; 16 in the stub).
///
/// Outputs (one each for A, RG, VY):
/// - `out_a`, `out_rg`, `out_vy` — `[height × width]` planar f32 in
///                                 DKL opponent space, scaled by the
///                                 display model's peak luminance.
///
/// Stub body — fills outputs with zeros. Replace once the cvvdp matrix
/// constants are pinned and golden numbers from the Python reference
/// are captured.
#[cube(launch)]
#[allow(unused_variables)]
pub fn srgb_to_dkl_kernel(
    src: &Array<u32>,
    lut: &Array<f32>,
    out_a: &mut Array<f32>,
    out_rg: &mut Array<f32>,
    out_vy: &mut Array<f32>,
    width: u32,
    height: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (width * height) as usize;
    if idx >= total {
        terminate!();
    }
    out_a[idx] = 0.0;
    out_rg[idx] = 0.0;
    out_vy[idx] = 0.0;
}
