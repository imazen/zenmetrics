//! Linear RGB → positive XYB conversion (planar in/out).
//!
//! Translates `ssimulacra2-cuda-kernel/src/xyb.rs::px_linear_rgb_to_positive_xyb`,
//! which is itself the SSIMULACRA2-flavoured XYB used by the CPU
//! `ssimulacra2` crate (= `yuvxyb::linear_rgb_to_xyb` followed by
//! `make_positive_xyb`).
//!
//! Algorithm:
//! 1. Apply opsin absorbance matrix + bias.
//! 2. Clamp negatives to 0; cube-root; subtract `cbrt(K_B0)`.
//! 3. Form opponent (X = ½(rg − gr), Y = ½(rg + gr), B = b).
//! 4. Squash into ~[0, 1] range:
//!    - X' = 14·X + 0.42
//!    - Y' = Y + 0.01
//!    - B' = B − Y + 0.55
//!
//! ## CubeCL note
//!
//! `f32::cbrt` is not exposed as a runtime op in cubecl 0.10, so we use
//! `f32::powf(x, 1/3)`. After the `max(0)` clamp the input is
//! non-negative, so the power-function path yields the same value as a
//! hardware `cbrtf` would. Sub-ulp drift relative to the CUDA
//! `cbrtf` instruction is bounded by powf's ~3 ulp error and shows up as
//! < 1e-6 in parity tests — well within the 0.1 % score tolerance.

use cubecl::prelude::*;

const K_M02: f32 = 0.078;
const K_M00: f32 = 0.30;
const K_M01: f32 = 1.0 - K_M02 - K_M00;

const K_M12: f32 = 0.078;
const K_M10: f32 = 0.23;
const K_M11: f32 = 1.0 - K_M12 - K_M10;

const K_M20: f32 = 0.243_422_69;
const K_M21: f32 = 0.204_767_45;
const K_M22: f32 = 1.0 - K_M20 - K_M21;

const K_B0: f32 = 0.003_793_073_4;
/// `cbrt(K_B0)` precomputed at the same precision as the CUDA reference.
const K_B0_ROOT: f32 = 0.155_954_2;

/// Per-pixel linear RGB → positive XYB on planar buffers.
///
/// Reads `src_r/g/b[idx]`, writes `dst_x/y/b[idx]`. Buffers must all
/// have length `n_pixels`.
#[cube(launch_unchecked)]
pub fn linear_to_xyb_planar_kernel(
    src_r: &Array<f32>,
    src_g: &Array<f32>,
    src_b: &Array<f32>,
    dst_x: &mut Array<f32>,
    dst_y: &mut Array<f32>,
    dst_bb: &mut Array<f32>,
) {
    let idx = ABSOLUTE_POS;
    let n = dst_x.len();
    if idx >= n {
        terminate!();
    }
    let (x, y, b) = px_linear_rgb_to_positive_xyb(src_r[idx], src_g[idx], src_b[idx]);
    dst_x[idx] = x;
    dst_y[idx] = y;
    dst_bb[idx] = b;
}

#[cube]
fn px_linear_rgb_to_positive_xyb(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (rg, gr, b3) = opsin_absorbance(r, g, b);
    const ONE_THIRD: f32 = 1.0 / 3.0;
    let rg_c = f32::powf(f32::max(rg, 0.0), ONE_THIRD) - K_B0_ROOT;
    let gr_c = f32::powf(f32::max(gr, 0.0), ONE_THIRD) - K_B0_ROOT;
    let b_c = f32::powf(f32::max(b3, 0.0), ONE_THIRD) - K_B0_ROOT;
    let x = 0.5 * (rg_c - gr_c);
    let y = 0.5 * (rg_c + gr_c);
    let xp = 14.0 * x + 0.42;
    let yp = y + 0.01;
    let bp = b_c - y + 0.55;
    (xp, yp, bp)
}

#[cube]
fn opsin_absorbance(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let rg = K_M00 * r + K_M01 * g + K_M02 * b + K_B0;
    let gr = K_M10 * r + K_M11 * g + K_M12 * b + K_B0;
    let bb = K_M20 * r + K_M21 * g + K_M22 * b + K_B0;
    (rg, gr, bb)
}

// ───────────────────────── PU21 HDR variant ─────────────────────────
//
// PU21 (Mantiuk & Azimi, PCS 2021) `banding_glare` parameters — the
// published gfxdisp/pu21 set, byte-identical to `zensim::pu21` /
// `zenmetrics-api::hdr` / fast-ssim2's `pu_xyb` (all pinned to the same
// independent float64 goldens). Input is **absolute-luminance** linear
// RGB (cd/m²); PU21 replaces the cube-root at the perceptual-encoding
// layer. Feeding PU-encoded values as *input* to the cube-root pipeline
// instead caps HDR correlation (UPIQ SROCC 0.59–0.61 vs 0.70 for this
// integrated form) — see imazen/zenmetrics#25.
const PU_P0: f32 = 0.353_487_9;
const PU_P1: f32 = 0.373_465_86;
const PU_P2: f32 = 8.277_049e-5;
const PU_P3: f32 = 0.906_256_26;
const PU_P4: f32 = 0.091_503_03;
const PU_P5: f32 = 0.909_951_7;
const PU_P6: f32 = 596.314_8;
const PU_L_MIN: f32 = 0.005;
const PU_L_MAX: f32 = 10000.0;
/// PU21(100 cd/m²) — 100-nit reference white normalizes to ~1.0, the
/// range the cube-root XYB white point occupies.
const PU_WHITE: f32 = 256.3;
/// Opponent X amplification in PU space (cube-root path uses 14).
const PU_X_SCALE: f32 = 4.0;

/// Per-pixel absolute-luminance linear RGB (cd/m²) → positive PU-XYB.
///
/// Same buffer contract as [`linear_to_xyb_planar_kernel`]; the
/// positive offsets are folded in (no separate make-positive step).
#[cube(launch_unchecked)]
pub fn linear_nits_to_pu_xyb_planar_kernel(
    src_r: &Array<f32>,
    src_g: &Array<f32>,
    src_b: &Array<f32>,
    dst_x: &mut Array<f32>,
    dst_y: &mut Array<f32>,
    dst_bb: &mut Array<f32>,
) {
    let idx = ABSOLUTE_POS;
    let n = dst_x.len();
    if idx >= n {
        terminate!();
    }
    let (x, y, b) = px_linear_nits_to_positive_pu_xyb(src_r[idx], src_g[idx], src_b[idx]);
    dst_x[idx] = x;
    dst_y[idx] = y;
    dst_bb[idx] = b;
}

#[cube]
fn px_linear_nits_to_positive_pu_xyb(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (rg, gr, b3) = opsin_absorbance(r, g, b);
    let c0 = pu21_encode(f32::max(rg, 0.0)) / PU_WHITE;
    let c1 = pu21_encode(f32::max(gr, 0.0)) / PU_WHITE;
    let c2 = pu21_encode(f32::max(b3, 0.0)) / PU_WHITE;
    let x = 0.5 * (c0 - c1);
    let y = 0.5 * (c0 + c1);
    let xp = PU_X_SCALE * x + 0.42;
    let yp = y + 0.01;
    let bp = c2 - y + 0.55;
    (xp, yp, bp)
}

/// PU21 encode on-device: absolute luminance (cd/m², clamped to the
/// `[0.005, 10000]` operating range) → perceptually-uniform value.
/// `V = max(p7·(((p1 + p2·Y^p4)/(1 + p3·Y^p4))^p5 − p6), 0)`.
#[cube]
fn pu21_encode(y: f32) -> f32 {
    let yc = f32::clamp(y, PU_L_MIN, PU_L_MAX);
    let yp = f32::powf(yc, PU_P3);
    let inner = (PU_P0 + PU_P1 * yp) / (1.0 + PU_P2 * yp);
    f32::max(PU_P6 * (f32::powf(inner, PU_P4) - PU_P5), 0.0)
}
