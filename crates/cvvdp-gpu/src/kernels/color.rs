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

// Tick 514: silence missing_docs warnings on items emitted by the
// #[cube(launch)] macro. The macro generates a sibling module +
// launcher struct + associated fn for each annotated function;
// those items don't inherit the user's rustdoc comment and trigger
// 4 warnings per kernel function. Every user-written pub item in
// this file (SRGB8_TO_LINEAR_LUT const, srgb_byte_to_dkl_scalar
// fn, and srgb_to_dkl_kernel fn itself) IS documented, so this
// allow only suppresses the macro-emitted noise.
#![allow(missing_docs)]

use cubecl::prelude::*;

// Phase 8c.1-C: scalar items (the SRGB8_TO_LINEAR_LUT table, the
// `*_to_dkl_scalar` helpers, the `eotf_tag` const module, and
// `eotf_tag_and_gamma`) live in `cvvdp::kernels::color` so the CPU
// crate owns the canonical scalar implementation. The cube-macro-
// emitted `#[cube(launch)] srgb_to_dkl_kernel` below uploads the
// LUT as a runtime `Array<f32>` (not a kernel-scope constant), so it
// does not reference SRGB8_TO_LINEAR_LUT by name inside the cube
// body — no name-resolution interaction with the macro.
// External callsites (`cvvdp_gpu::kernels::color::SRGB8_TO_LINEAR_LUT`,
// `srgb_byte_to_dkl_scalar`, `eotf_tag::*`, etc.) keep working via
// these re-exports.
pub use cvvdp::kernels::color::{
    SRGB8_TO_LINEAR_LUT, display_byte_to_dkl_scalar, display_linear_rgb_to_dkl_scalar, eotf_tag,
    eotf_tag_and_gamma, srgb_byte_to_dkl_scalar,
};

/// In-kernel EOTF apply. Branches on `eotf_tag` to mirror the host
/// [`crate::params::Eotf::forward`] dispatch.
///
/// Input `v` is the byte / linear value in 0..1 normalized space (the
/// caller divides bytes by 255). Linear EOTF accepts values >1
/// (HDR linear-light cd/m² inputs); PQ accepts 0..1 PQ-encoded.
///
/// Output is the per-channel linear-cd/m² scene-light, BEFORE the
/// HLG OOTF for HLG inputs (that step depends on the RGB triple's
/// `Y_s` and is applied separately in the kernel body). For non-HLG
/// EOTFs the output is already in cd/m² and ready for the DKL matmul.
///
/// `#[cube]` doesn't support early `return`, so the dispatch uses
/// chained `if/else` expressions — semantically equivalent to a match
/// on the tag.
#[cube]
fn apply_eotf_branch(
    v: f32,
    eotf_tag: u32,
    gamma_exp: f32,
    y_peak: f32,
    y_black: f32,
    y_refl: f32,
) -> f32 {
    let bias = y_black + y_refl;
    let scale = y_peak - y_black;

    let v_clamped = if v < f32::new(0.0) {
        f32::new(0.0)
    } else if v > f32::new(1.0) {
        f32::new(1.0)
    } else {
        v
    };

    if eotf_tag == 1u32 {
        // PQ (SMPTE ST 2084). Reference: pycvvdp `pq2lin`.
        let l_max = f32::new(10000.0);
        let m1 = f32::new(0.159_301_75);
        let m2 = f32::new(78.843_75);
        let c1 = f32::new(0.835_937_5);
        let c2 = f32::new(18.851_562);
        let c3 = f32::new(18.687_5);
        // PQ accepts raw v (no 0..1 clamp; HDR PQ-encoded can exceed
        // 1 in theory but realistic inputs are clamped upstream).
        let im_t = f32::powf(v, f32::new(1.0) / m2);
        let num_raw = im_t - c1;
        let num = if num_raw < f32::new(0.0) {
            f32::new(0.0)
        } else {
            num_raw
        };
        let den = c2 - c3 * im_t;
        let lin = l_max * f32::powf(num / den, f32::new(1.0) / m1);
        let floor_val = f32::new(0.005);
        let clamped_lo = if lin < floor_val { floor_val } else { lin };
        let clamped = if clamped_lo > y_peak {
            y_peak
        } else {
            clamped_lo
        };
        clamped + bias
    } else if eotf_tag == 2u32 {
        // HLG inverse OETF. OOTF applied by caller (depends on Y_s).
        let a = f32::new(0.178_832_77);
        let b = f32::new(1.0) - f32::new(4.0) * a;
        let c = f32::new(0.5) - a * f32::ln(f32::new(4.0) * a);
        let lin = if v_clamped <= f32::new(0.5) {
            (v_clamped * v_clamped) / f32::new(3.0)
        } else {
            (f32::exp((v_clamped - c) / a) + b) / f32::new(12.0)
        };
        scale * lin + bias
    } else if eotf_tag == 3u32 {
        // Linear-light input. Clip to [max(0.005, y_black), y_peak]
        // then add y_refl (NOT bias — Linear's path doesn't re-add
        // y_black, per pycvvdp's branch).
        let floor_val = f32::new(0.005);
        let floor_eff = if y_black > floor_val {
            y_black
        } else {
            floor_val
        };
        let clamped_lo = if v < floor_eff { floor_eff } else { v };
        let clamped = if clamped_lo > y_peak {
            y_peak
        } else {
            clamped_lo
        };
        clamped + y_refl
    } else if eotf_tag == 4u32 {
        // BT.1886 — gamma 2.4 with black-level lift. L = a · (V + b)^γ.
        let gamma = f32::new(2.4);
        let inv_gamma = f32::new(1.0) / gamma;
        let y_p_g = f32::powf(y_peak, inv_gamma);
        let y_b_g = f32::powf(y_black, inv_gamma);
        let lift_a = f32::powf(y_p_g - y_b_g, gamma);
        let lift_b = y_b_g / (y_p_g - y_b_g);
        let sum = v_clamped + lift_b;
        let sum_pos = if sum < f32::new(0.0) {
            f32::new(0.0)
        } else {
            sum
        };
        let l = lift_a * f32::powf(sum_pos, gamma);
        l + y_refl
    } else if eotf_tag == 5u32 {
        // Generic power-law gamma (Adobe RGB 2.2, Apple RGB 1.8, …).
        let lin = f32::powf(v_clamped, gamma_exp);
        scale * lin + bias
    } else {
        // Default / fallback: sRGB closed-form. The caller takes the
        // LUT path when it knows the EOTF is sRGB; this branch only
        // fires if the linear-planes / non-byte entry routes a tag-0
        // value through here.
        let lin = if v_clamped > f32::new(0.040_45) {
            f32::powf(
                (v_clamped + f32::new(0.055)) / f32::new(1.055),
                f32::new(2.4),
            )
        } else {
            v_clamped / f32::new(12.92)
        };
        scale * lin + bias
    }
}

/// HLG OOTF (system gamma applied to the linear-RGB triple per
/// BT.2100). Computes Y_s from the inverse-OETF values, derives the
/// per-pixel factor `Y_s^(γ-1)`, and re-scales each channel.
///
/// `gamma` is the precomputed HLG system gamma (host-side function
/// `hlg_system_gamma(y_peak, e_ambient_lux)` — passed in as a
/// runtime scalar since it doesn't vary per pixel).
///
/// Returns the OOTF-adjusted `(lr, lg, lb)` already in display-light
/// cd/m² (the scale + bias step is folded in).
#[cube]
fn hlg_ootf(
    lr_pre: f32,
    lg_pre: f32,
    lb_pre: f32,
    y_peak: f32,
    y_black: f32,
    y_refl: f32,
    gamma: f32,
) -> (f32, f32, f32) {
    let scale = y_peak - y_black;
    let bias = y_black + y_refl;
    // Strip the bias / scale applied by apply_eotf_branch so we get
    // back to inverse_oetf(v) in 0..12.
    let inv_r = if scale > f32::new(0.0) {
        (lr_pre - bias) / scale
    } else {
        f32::new(0.0)
    };
    let inv_g = if scale > f32::new(0.0) {
        (lg_pre - bias) / scale
    } else {
        f32::new(0.0)
    };
    let inv_b = if scale > f32::new(0.0) {
        (lb_pre - bias) / scale
    } else {
        f32::new(0.0)
    };
    // BT.2100 luma coefficients (R, G, B) = (0.2627, 0.6780, 0.0593).
    let y_s = f32::new(0.262_7) * inv_r + f32::new(0.678_0) * inv_g + f32::new(0.059_3) * inv_b;
    let factor = if y_s > f32::new(0.0) {
        f32::powf(y_s, gamma - f32::new(1.0))
    } else {
        f32::new(0.0)
    };
    let lr = scale * (inv_r * factor) + bias;
    let lg = scale * (inv_g * factor) + bias;
    let lb = scale * (inv_b * factor) + bias;
    (lr, lg, lb)
}

/// 8-bit packed-RGB → DKL planar f32, with display dispatch on EOTF
/// and primaries.
///
/// Inputs:
/// - `src` — `width × height` packed sRGB bytes (R | G<<8 | B<<16).
/// - `lut` — uploaded [`SRGB8_TO_LINEAR_LUT`] (256 entries). Read only
///   on the sRGB fast path; ignored for non-sRGB EOTFs.
///
/// Outputs:
/// - `out_a`, `out_rg`, `out_vy` — `width × height` planar f32 in
///   DKLd65 opponent space (cd/m²-scaled).
///
/// Runtime dispatch:
/// - `eotf_tag` — see [`eotf_tag`] constants. `0` = sRGB takes the
///   fast LUT path; any other value runs the closed-form EOTF via
///   [`apply_eotf_branch`].
/// - `gamma_exp` — exponent for [`eotf_tag::GAMMA`]; ignored for
///   other tags.
/// - `m00..m22` — 9 runtime scalars carrying the per-primaries
///   linear-RGB→DKL matrix ([`crate::params::Primaries::linear_rgb_to_dkl`]).
///   Pushed as scalars (not constants) so a single kernel binary
///   serves every primaries set; LLVM still folds the linear combo
///   when the values are constant across the launch.
/// - `hlg_gamma` — precomputed HLG system gamma. Only consumed when
///   `eotf_tag == eotf_tag::HLG`.
///
/// The sRGB / BT.709 fast path matches the historical
/// `srgb_to_dkl_kernel` output bit-for-bit (LUT + folded matrix
/// constants come from the same vendored numbers).
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
    eotf_tag: u32,
    gamma_exp: f32,
    hlg_gamma: f32,
    m00: f32,
    m01: f32,
    m02: f32,
    m10: f32,
    m11: f32,
    m12: f32,
    m20: f32,
    m21: f32,
    m22: f32,
) {
    let idx = ABSOLUTE_POS;
    let total = (width * height) as usize;
    if idx >= total {
        terminate!();
    }

    // T4.L (2026-05-16): packed-RGBA upload. Host packs 3 sRGB bytes
    // per pixel into one u32 (R in low byte, then G, then B; alpha
    // unused). Cuts the H→D transfer 3× vs the prior u8-widened-to-u32
    // path (144 MB → 48 MB at 12 MP); the per-iter `create_from_slice`
    // alloc shrinks in proportion. 3 bit-shifts + 3 ANDs per pixel are
    // free relative to the upload time saved.
    let packed = src[idx];
    let r_byte = packed & 0xffu32;
    let g_byte = (packed >> 8u32) & 0xffu32;
    let b_byte = (packed >> 16u32) & 0xffu32;

    // Per-channel EOTF: sRGB fast path (LUT + scale/bias) on tag=0,
    // closed-form `apply_eotf_branch` on every other tag. Linear-light
    // input is 0..1 byte/255 normalised before the branch (matches the
    // host scalar's `display_byte_to_dkl_scalar` shape).
    let inv_255 = f32::new(1.0) / f32::new(255.0);
    let s = y_peak - y_black;
    let bias = y_black + y_refl;
    let lr_pre = if eotf_tag == 0u32 {
        let lin_r = lut[r_byte as usize];
        s * lin_r + bias
    } else {
        let vr = (r_byte as f32) * inv_255;
        apply_eotf_branch(vr, eotf_tag, gamma_exp, y_peak, y_black, y_refl)
    };
    let lg_pre = if eotf_tag == 0u32 {
        let lin_g = lut[g_byte as usize];
        s * lin_g + bias
    } else {
        let vg = (g_byte as f32) * inv_255;
        apply_eotf_branch(vg, eotf_tag, gamma_exp, y_peak, y_black, y_refl)
    };
    let lb_pre = if eotf_tag == 0u32 {
        let lin_b = lut[b_byte as usize];
        s * lin_b + bias
    } else {
        let vb = (b_byte as f32) * inv_255;
        apply_eotf_branch(vb, eotf_tag, gamma_exp, y_peak, y_black, y_refl)
    };

    // HLG: per-pixel OOTF using the RGB triple's Y_s. Other EOTFs
    // already produced final display-light cd/m².
    let (lr, lg, lb) = if eotf_tag == 2u32 {
        hlg_ootf(lr_pre, lg_pre, lb_pre, y_peak, y_black, y_refl, hlg_gamma)
    } else {
        (lr_pre, lg_pre, lb_pre)
    };

    out_a[idx] = m00 * lr + m01 * lg + m02 * lb;
    out_rg[idx] = m10 * lr + m11 * lg + m12 * lb;
    out_vy[idx] = m20 * lr + m21 * lg + m22 * lb;
}
