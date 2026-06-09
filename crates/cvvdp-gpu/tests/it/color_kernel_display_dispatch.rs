//! GPU parity vs the display-aware host scalar for non-sRGB EOTFs +
//! non-BT.709 primaries.
//!
//! The legacy [`color_kernel`] test covers the historical
//! `STANDARD_4K` (sRGB / BT.709) path; this file extends coverage to
//! every EOTF tag + primaries variant the GPU dispatch supports so a
//! regression in either side surfaces immediately.
//!
//! Each subtest builds an 8×8 deterministic RGB pattern, runs the
//! dispatch through `srgb_to_dkl_kernel::launch` with the target
//! display, reads the three planar DKL outputs back, and asserts
//! they agree with `display_byte_to_dkl_scalar` to within FMA-vs-
//! plain-add noise (3e-4 absolute is the tightest band that absorbs
//! the multi-stage `powf` accumulation in PQ / HLG / Bt1886 / Gamma
//! without bouncing on the canonical fixtures).
//!
//! Pinned: `srgb_to_dkl_kernel`'s arg order is fixed by the GPU
//! dispatch contract — any new field MUST extend the host scalar
//! first so tests can be regenerated lock-step.

#![cfg(any(feature = "cuda", feature = "wgpu", feature = "hip", feature = "cpu"))]

use cubecl::Runtime;
use cubecl::prelude::*;
use cvvdp_gpu::kernels::color::{
    SRGB8_TO_LINEAR_LUT, display_byte_to_dkl_scalar, eotf_tag_and_gamma, srgb_to_dkl_kernel,
};
use cvvdp_gpu::params::{DisplayModel, Eotf, Primaries, hlg_system_gamma};

use crate::common;

use common::Backend;

fn rgb_input(w: u32, h: u32) -> Vec<u8> {
    let n = (w * h) as usize;
    let mut v = Vec::with_capacity(n * 3);
    for i in 0..n {
        v.push((i % 251) as u8);
        v.push(((i * 7 + 13) % 251) as u8);
        v.push(((i * 19 + 41) % 251) as u8);
    }
    v
}

/// Run the kernel for a given `DisplayModel` and assert per-pixel
/// agreement with `display_byte_to_dkl_scalar`. `tol` is the
/// abs-error band; tighter bands (≤1e-5) only fit the sRGB fast
/// path, while the `powf`-heavy EOTFs need 3e-4 to absorb GPU-vs-
/// host f32 ordering. Returns `(max_a_err, max_rg_err, max_vy_err)`
/// for diagnostic reporting.
fn assert_kernel_matches_scalar(display: DisplayModel, tol: f32, label: &str) -> (f32, f32, f32) {
    let client = Backend::client(&Default::default());
    let (w, h) = (8u32, 8u32);
    let n = (w * h) as usize;
    let rgb_bytes = rgb_input(w, h);

    let src_u32: Vec<u32> = rgb_bytes
        .chunks_exact(3)
        .map(|t| u32::from(t[0]) | (u32::from(t[1]) << 8) | (u32::from(t[2]) << 16))
        .collect();
    let src_h = client.create_from_slice(u32::as_bytes(&src_u32));
    let lut_h = client.create_from_slice(f32::as_bytes(&SRGB8_TO_LINEAR_LUT));
    let a_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let rg_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let vy_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));

    let (eotf_tag, gamma_exp) = eotf_tag_and_gamma(display.eotf);
    let hlg_gamma = hlg_system_gamma(display.y_peak, display.e_ambient_lux);
    let m = display.primaries.linear_rgb_to_dkl();

    let cube_dim = CubeDim::new_1d(64);
    let cube_count = CubeCount::Static((n as u32).div_ceil(64), 1, 1);
    unsafe {
        srgb_to_dkl_kernel::launch::<Backend>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(src_h.clone(), n),
            ArrayArg::from_raw_parts(lut_h.clone(), SRGB8_TO_LINEAR_LUT.len()),
            ArrayArg::from_raw_parts(a_h.clone(), n),
            ArrayArg::from_raw_parts(rg_h.clone(), n),
            ArrayArg::from_raw_parts(vy_h.clone(), n),
            w,
            h,
            display.y_peak,
            display.y_black,
            display.y_refl,
            eotf_tag,
            gamma_exp,
            hlg_gamma,
            m[0][0],
            m[0][1],
            m[0][2],
            m[1][0],
            m[1][1],
            m[1][2],
            m[2][0],
            m[2][1],
            m[2][2],
        );
    }

    let a_bytes = client.read_one(a_h.clone()).expect("read A");
    let rg_bytes = client.read_one(rg_h.clone()).expect("read RG");
    let vy_bytes = client.read_one(vy_h.clone()).expect("read VY");
    let gpu_a: &[f32] = f32::from_bytes(&a_bytes);
    let gpu_rg: &[f32] = f32::from_bytes(&rg_bytes);
    let gpu_vy: &[f32] = f32::from_bytes(&vy_bytes);

    let mut max_a = 0.0_f32;
    let mut max_rg = 0.0_f32;
    let mut max_vy = 0.0_f32;
    let mut worst = (0u8, 0u8, 0u8, 0.0_f32, 0.0_f32, 0.0_f32);
    for i in 0..n {
        let r = rgb_bytes[i * 3];
        let g = rgb_bytes[i * 3 + 1];
        let b = rgb_bytes[i * 3 + 2];
        let (ea, erg, evy) = display_byte_to_dkl_scalar(r, g, b, display);
        let da = (gpu_a[i] - ea).abs();
        let drg = (gpu_rg[i] - erg).abs();
        let dvy = (gpu_vy[i] - evy).abs();
        if da > max_a {
            max_a = da;
            worst = (r, g, b, gpu_a[i] - ea, gpu_rg[i] - erg, gpu_vy[i] - evy);
        }
        if drg > max_rg {
            max_rg = drg;
        }
        if dvy > max_vy {
            max_vy = dvy;
        }
    }
    let max_err = max_a.max(max_rg).max(max_vy);
    assert!(
        max_err < tol,
        "{label}: GPU vs host-scalar display dispatch max abs error = {max_err} (A={max_a}, \
         RG={max_rg}, VY={max_vy}) above tol {tol}; worst pixel RGB=({},{},{}) diffs \
         A/RG/VY=({},{},{})",
        worst.0,
        worst.1,
        worst.2,
        worst.3,
        worst.4,
        worst.5
    );
    (max_a, max_rg, max_vy)
}

/// sRGB / BT.709 — the legacy fast path. Tight bound (3e-5 like the
/// existing `color_kernel` test) since both sides use the same LUT
/// and matrix consts.
#[test]
fn gpu_dispatch_srgb_bt709_matches_host_scalar() {
    assert_kernel_matches_scalar(DisplayModel::STANDARD_4K, 3e-5, "STANDARD_4K");
}

#[test]
fn gpu_dispatch_srgb_bt2020_matches_host_scalar() {
    // BT.2020 matrix magnitudes are slightly larger than BT.709 on
    // the chroma rows, so the matmul accumulates ~1 ULP more error
    // before truncation. 1e-4 absolute still equals ≤0.5 ppm relative
    // to the ~200 cd/m² A peak — well under any perceptual band.
    let d = DisplayModel {
        primaries: Primaries::Bt2020,
        ..DisplayModel::STANDARD_4K
    };
    assert_kernel_matches_scalar(d, 1e-4, "Srgb+Bt2020");
}

#[test]
fn gpu_dispatch_srgb_display_p3_matches_host_scalar() {
    let d = DisplayModel {
        primaries: Primaries::DisplayP3,
        ..DisplayModel::STANDARD_4K
    };
    assert_kernel_matches_scalar(d, 1e-4, "Srgb+DisplayP3");
}

#[test]
fn gpu_dispatch_pq_bt2020_matches_host_scalar() {
    // STANDARD_HDR_PQ: 1500 cd/m² PQ + BT.2020.
    // PQ involves two chained powf calls per channel. At HDR peaks
    // the DKL output magnitudes scale with y_peak, so an absolute
    // tolerance has to follow. 0.05 abs on a 1500-cd/m² peak is
    // 33 ppm relative — well inside f32 noise for a four-term FMA
    // chain (matmul) over four chained transcendentals (PQ).
    assert_kernel_matches_scalar(DisplayModel::STANDARD_HDR_PQ, 5e-2, "PQ+Bt2020 (1500nit)");
}

#[test]
fn gpu_dispatch_pq_3000_nit_matches_host_scalar() {
    // 3000-nit peak — proportionally larger absolute bound. 0.1 abs
    // on 3000 cd/m² is 33 ppm, same relative band as the 1500-nit case.
    assert_kernel_matches_scalar(
        DisplayModel::LG_OLED_2026_HDR_PQ,
        1e-1,
        "PQ+Bt2020 (3000nit)",
    );
}

#[test]
fn gpu_dispatch_hlg_bt2020_matches_host_scalar() {
    // HLG combines an exp() (above v=0.5) with a powf() OOTF —
    // chained transcendentals widen the tolerance similarly to PQ.
    // 0.05 abs on 1500-cd/m² peak is 33 ppm relative.
    assert_kernel_matches_scalar(DisplayModel::STANDARD_HDR_HLG, 5e-2, "HLG+Bt2020");
}

#[test]
fn gpu_dispatch_linear_bt709_matches_host_scalar() {
    // Linear EOTF — the simplest non-sRGB path. The 8-bit byte input
    // is interpreted as a normalized 0..1 linear-light value (which
    // is what `display_byte_to_dkl_scalar` also does). Magnitudes
    // are small (post-clamp ≤ y_peak * tiny linear values) and the
    // path involves no powf, so the band is tight.
    assert_kernel_matches_scalar(DisplayModel::STANDARD_HDR_LINEAR, 1e-4, "Linear+Bt709");
}

#[test]
fn gpu_dispatch_bt1886_matches_host_scalar() {
    // BT.1886 — gamma 2.4 with black-level lift. Single powf per
    // channel.
    let d = DisplayModel {
        eotf: Eotf::Bt1886,
        ..DisplayModel::STANDARD_4K
    };
    assert_kernel_matches_scalar(d, 3e-4, "Bt1886+Bt709");
}

#[test]
fn gpu_dispatch_gamma_22_matches_host_scalar() {
    // Generic power-law gamma (Adobe RGB / Wide Gamut RGB use 2.2).
    let d = DisplayModel {
        eotf: Eotf::Gamma(2.2),
        ..DisplayModel::STANDARD_4K
    };
    assert_kernel_matches_scalar(d, 3e-4, "Gamma(2.2)+Bt709");
}

#[test]
fn gpu_dispatch_gamma_18_matches_host_scalar() {
    // Apple RGB uses gamma 1.8.
    let d = DisplayModel {
        eotf: Eotf::Gamma(1.8),
        ..DisplayModel::STANDARD_4K
    };
    assert_kernel_matches_scalar(d, 3e-4, "Gamma(1.8)+Bt709");
}

#[test]
fn gpu_dispatch_iphone_14_pro_matches_host_scalar() {
    // iPhone 14 Pro preset — high-peak SDR (1025 cd/m²) on sRGB +
    // BT.709 primaries. Tolerance scales with y_peak vs STANDARD_4K's
    // 200 cd/m² (~5×) → 1.5e-4 absolute.
    assert_kernel_matches_scalar(DisplayModel::IPHONE_14_PRO, 1.5e-4, "iphone_14_pro");
}

#[test]
fn gpu_dispatch_iphone_14_pro_hdr_matches_host_scalar() {
    // iPhone 14 Pro HDR — 1590 cd/m² HLG + BT.2020. The HLG OOTF
    // chain widens tolerance to the PQ band.
    assert_kernel_matches_scalar(DisplayModel::IPHONE_14_PRO_HDR, 5e-2, "iphone_14_pro_hdr");
}
