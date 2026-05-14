//! sRGB packed-u8 → linear → DKLd65 opponent planar f32.
//!
//! Three stages fused into a single kernel pass:
//!
//! 1. **sRGB EOTF**: byte → linear via a 256-entry LUT. The LUT
//!    encodes cvvdp's `srgb2lin`:
//!    `lin = if p > 0.04045 { ((p + 0.055) / 1.055)^2.4 } else { p / 12.92 }`.
//!    Same numbers as `zensim_gpu::kernels::color::SRGB8_TO_LINEARF32_LUT`.
//!
//! 2. **Display model**: linear normalized → cd/m² emitted luminance:
//!    `L = (Y_peak - Y_black) * lin + Y_black + Y_refl`. Constants
//!    come from [`crate::params::DisplayModel`].
//!
//! 3. **DKL transform**: 3×3 matmul with the combined
//!    [`crate::params::SRGB_LINEAR_TO_DKL`] matrix.
//!
//! Output: three planar f32 buffers (A, RG, VY) in absolute DKL units.
//! cvvdp keeps DKL in cd/m²-scaled units (the CSF stage handles
//! sensitivity scaling), so no post-normalization.

use cubecl::prelude::*;

/// 256-entry sRGB byte → linear-normalized f32 LUT. Matches
/// `zensim_gpu::kernels::color::SRGB8_TO_LINEARF32_LUT` byte-for-byte
/// so the upload paths can share scratch.
#[rustfmt::skip]
pub const SRGB8_TO_LINEAR_LUT: [f32; 256] = [
    0.0, 0.000303527, 0.000607054, 0.00091058103, 0.001214108, 0.001517635, 0.0018211621, 0.002124689,
    0.002428216, 0.002731743, 0.00303527, 0.0033465356, 0.003676507, 0.004024717, 0.004391442,
    0.0047769533, 0.005181517, 0.0056053917, 0.0060488326, 0.006512091, 0.00699541, 0.0074990317,
    0.008023192, 0.008568125, 0.009134057, 0.009721218, 0.010329823, 0.010960094, 0.011612245,
    0.012286487, 0.012983031, 0.013702081, 0.014443844, 0.015208514, 0.015996292, 0.016807375,
    0.017641952, 0.018500218, 0.019382361, 0.020288562, 0.02121901, 0.022173883, 0.023153365,
    0.02415763, 0.025186857, 0.026241222, 0.027320892, 0.028426038, 0.029556843, 0.03071345, 0.03189604,
    0.033104774, 0.03433981, 0.035601325, 0.036889452, 0.038204376, 0.039546248, 0.04091521, 0.042311423,
    0.043735042, 0.045186214, 0.046665095, 0.048171833, 0.049706575, 0.051269468, 0.052860655, 0.05448028,
    0.056128494, 0.057805434, 0.05951124, 0.06124607, 0.06301003, 0.06480328, 0.06662595, 0.06847818,
    0.07036011, 0.07227186, 0.07421358, 0.07618539, 0.07818743, 0.08021983, 0.082282715, 0.084376216,
    0.086500466, 0.088655606, 0.09084173, 0.09305898, 0.095307484, 0.09758736, 0.09989874, 0.10224175,
    0.10461649, 0.10702311, 0.10946172, 0.111932434, 0.11443538, 0.116970696, 0.11953845, 0.12213881,
    0.12477186, 0.12743773, 0.13013652, 0.13286836, 0.13563336, 0.13843165, 0.14126332, 0.1441285,
    0.1470273, 0.14995982, 0.15292618, 0.1559265, 0.15896086, 0.16202943, 0.16513224, 0.16826946,
    0.17144115, 0.17464745, 0.17788847, 0.1811643, 0.18447503, 0.1878208, 0.19120172, 0.19461787,
    0.19806935, 0.2015563, 0.20507877, 0.2086369, 0.21223079, 0.21586053, 0.21952623, 0.22322798,
    0.22696589, 0.23074007, 0.23455065, 0.23839766, 0.2422812, 0.2462014, 0.25015837, 0.25415218,
    0.2581829, 0.26225072, 0.26635566, 0.27049786, 0.27467737, 0.27889434, 0.2831488, 0.2874409,
    0.2917707, 0.29613832, 0.30054384, 0.30498737, 0.30946895, 0.31398875, 0.31854683, 0.3231432,
    0.3277781, 0.33245152, 0.33716363, 0.34191442, 0.3467041, 0.35153264, 0.35640016, 0.36130676,
    0.3662526, 0.3712377, 0.37626213, 0.38132602, 0.38642946, 0.39157256, 0.39675537, 0.40197802,
    0.4072406, 0.4125432, 0.41788593, 0.42326888, 0.42869216, 0.43415588, 0.43966013, 0.445205,
    0.45079055, 0.456417, 0.46208432, 0.46779266, 0.47354212, 0.47933277, 0.48516476, 0.49103815,
    0.49695304, 0.5029096, 0.50890774, 0.5149478, 0.5210297, 0.52715355, 0.53331953, 0.5395277,
    0.5457781, 0.5520708, 0.5584061, 0.564784, 0.5712046, 0.57766795, 0.58417416, 0.5907234,
    0.59731567, 0.6039511, 0.61062974, 0.61735177, 0.62411714, 0.63092613, 0.63777864, 0.64467484,
    0.6516149, 0.6585987, 0.6656265, 0.67269844, 0.6798144, 0.6869747, 0.6941793, 0.7014284,
    0.7087221, 0.71606034, 0.72344327, 0.730871, 0.7383436, 0.7458612, 0.75342387, 0.76103175,
    0.7686849, 0.77638346, 0.7841275, 0.7919172, 0.7997525, 0.8076336, 0.81556076, 0.82353383,
    0.831553, 0.8396184, 0.8477301, 0.85588825, 0.8640929, 0.8723443, 0.88064235, 0.8889873,
    0.89737915, 0.9058182, 0.9143044, 0.92283785, 0.9314188, 0.9400473, 0.9487235, 0.9574475,
    0.9662194, 0.9750394, 0.9839074, 0.9928237, 1.0,
];

/// Host-side scalar reference for the color stage. Bit-exact with
/// `srgb_to_dkl_kernel`'s per-pixel math at f32 precision. Used by
/// unit tests and by host-side debug taps.
///
/// Returns `(dkl_a, dkl_rg, dkl_vy)` for one pixel.
#[inline]
pub fn srgb_byte_to_dkl_scalar(
    r: u8,
    g: u8,
    b: u8,
    y_peak: f32,
    y_black: f32,
    y_refl: f32,
) -> (f32, f32, f32) {
    use crate::params::SRGB_LINEAR_TO_DKL as M;

    let lin_r = SRGB8_TO_LINEAR_LUT[r as usize];
    let lin_g = SRGB8_TO_LINEAR_LUT[g as usize];
    let lin_b = SRGB8_TO_LINEAR_LUT[b as usize];

    let s = y_peak - y_black;
    let bias = y_black + y_refl;
    let lr = s * lin_r + bias;
    let lg = s * lin_g + bias;
    let lb = s * lin_b + bias;

    let a = M[0][0] * lr + M[0][1] * lg + M[0][2] * lb;
    let rg = M[1][0] * lr + M[1][1] * lg + M[1][2] * lb;
    let vy = M[2][0] * lr + M[2][1] * lg + M[2][2] * lb;
    (a, rg, vy)
}

/// sRGB packed-u8 → DKL planar f32.
///
/// Inputs:
/// - `src` — `width × height × 3` packed sRGB bytes, with each byte
///   widened to a `u32` slot. The host upload helper writes RGB triples
///   in row-major order: `[r0, g0, b0, r1, g1, b1, …]`.
/// - `lut` — uploaded [`SRGB8_TO_LINEAR_LUT`] (256 entries).
///
/// Outputs:
/// - `out_a`, `out_rg`, `out_vy` — `width × height` planar f32 in
///   DKLd65 opponent space (cd/m²-scaled).
///
/// Display constants (`y_peak`, `y_black`, `y_refl`) are pushed as
/// runtime scalars so per-display retunes don't need a recompile. The
/// 3×3 RGB→DKL matrix is captured as kernel-local f32 constants so
/// LLVM folds the linear combination at codegen time.
#[cube(launch)]
pub fn srgb_to_dkl_kernel(
    src: &Array<u32>,
    lut: &Array<f32>,
    out_a: &mut Array<f32>,
    out_rg: &mut Array<f32>,
    out_vy: &mut Array<f32>,
    width: u32,
    height: u32,
    y_peak: f32,
    y_black: f32,
    y_refl: f32,
) {
    let idx = ABSOLUTE_POS;
    let total = (width * height) as usize;
    if idx >= total {
        terminate!();
    }

    let base = idx * 3;
    let r = src[base];
    let g = src[base + 1];
    let b = src[base + 2];

    let lin_r = lut[r as usize];
    let lin_g = lut[g as usize];
    let lin_b = lut[b as usize];

    let s = y_peak - y_black;
    let bias = y_black + y_refl;
    let lr = s * lin_r + bias;
    let lg = s * lin_g + bias;
    let lb = s * lin_b + bias;

    let m00 = f32::new(0.233_201_21);
    let m01 = f32::new(0.728_830_8);
    let m02 = f32::new(0.088_995_87);
    let m10 = f32::new(0.127_620_77);
    let m11 = f32::new(-0.087_068_09);
    let m12 = f32::new(-0.036_777_39);
    let m20 = f32::new(-0.214_822_5);
    let m21 = f32::new(-0.626_253_7);
    let m22 = f32::new(0.851_403_3);

    out_a[idx] = m00 * lr + m01 * lg + m02 * lb;
    out_rg[idx] = m10 * lr + m11 * lg + m12 * lb;
    out_vy[idx] = m20 * lr + m21 * lg + m22 * lb;
}
