//! Tests: golden vectors from the reference implementation (Octave,
//! the authors' `FR_FSIMc.m`), API contract checks, and per-tier
//! parity.

use alloc::vec::Vec;

use crate::*;

/// `gen` kind 1: `mod(37x + 91y + x·y, 256)`.
fn gen1(w: usize, h: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            v.push(((37 * x + 91 * y + x * y) % 256) as f32);
        }
    }
    v
}

/// `gen` kind 2: `mod((17x XOR 29y) + 3x + 5y, 256)`.
fn gen2(w: usize, h: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            v.push(((((17 * x) ^ (29 * y)) + 3 * x + 5 * y) % 256) as f32);
        }
    }
    v
}

/// `gen` kind 3: `mod(x + 2y, 256)` — DC-dominated gradient.
fn gen3(w: usize, h: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            v.push(((x + 2 * y) % 256) as f32);
        }
    }
    v
}

/// `gen` kind 4: gen1 with the `20<=x<40, 10<=y<30` patch replaced by
/// gen2 — mixed smooth/texture content.
fn gen4(w: usize, h: usize) -> Vec<f32> {
    let mut v = gen1(w, h);
    let g2 = gen2(w, h);
    for y in 0..h {
        for x in 0..w {
            if (20..40).contains(&x) && (10..30).contains(&y) {
                v[y * w + x] = g2[y * w + x];
            }
        }
    }
    v
}

/// `gen` kind 5: constant 128.
fn gen5(w: usize, h: usize) -> Vec<f32> {
    alloc::vec![128.0f32; w * h]
}

/// Golden rows produced by running `FR_FSIMc.m` under GNU Octave on the
/// `gen` patterns above (same integer arithmetic, so inputs are
/// bit-identical). `(name, w, h, expected_fsim)`. Regenerate with
/// `validation/gen_goldens.m`. For grayscale input the reference's
/// FSIMc equals FSIM, so one column suffices.
const GOLDENS: &[(&str, usize, usize, f64)] = &[
    ("g1_vs_g2_64", 64, 64, 0.726254017324),
    ("g1_vs_g2_65x63", 65, 63, 0.728207119568),
    ("g1_vs_g4_64", 64, 64, 0.792809724558),
    ("g3_vs_g4_96x80", 96, 80, 0.208858441622),
    ("g1_identical", 64, 64, 1.0),
    ("g2_identical_40", 40, 40, 1.0),
    ("g4_identical", 64, 64, 1.0),
    ("g3_vs_g1_32", 32, 32, 0.184680011089),
    ("g2_vs_g1_37x41", 37, 41, 0.753920819360),
    ("g1_vs_g2_300x260", 300, 260, 0.722297928694),
    ("g1_vs_g4_520x400", 520, 400, 0.994864017870),
];

fn gens_for(g: &(&str, usize, usize, f64)) -> (Vec<f32>, Vec<f32>) {
    match g.0 {
        "g1_vs_g2_64" | "g1_vs_g2_65x63" => (gen1(g.1, g.2), gen2(g.1, g.2)),
        "g1_vs_g4_64" | "g1_vs_g4_520x400" => (gen1(g.1, g.2), gen4(g.1, g.2)),
        "g3_vs_g4_96x80" => (gen3(g.1, g.2), gen4(g.1, g.2)),
        "g1_identical" => (gen1(g.1, g.2), gen1(g.1, g.2)),
        "g2_identical_40" => (gen2(g.1, g.2), gen2(g.1, g.2)),
        "g4_identical" => (gen4(g.1, g.2), gen4(g.1, g.2)),
        "g3_vs_g1_32" => (gen3(g.1, g.2), gen1(g.1, g.2)),
        "g2_vs_g1_37x41" => (gen2(g.1, g.2), gen1(g.1, g.2)),
        "g1_vs_g2_300x260" => (gen1(g.1, g.2), gen2(g.1, g.2)),
        _ => unreachable!("{}", g.0),
    }
}

#[test]
fn plane_goldens() {
    let mut worst = 0.0f64;
    for g in GOLDENS {
        let (r, d) = gens_for(g);
        let s = fsim_plane_f32(&r, &d, g.1, g.2, g.1).unwrap();
        let delta = (s - g.3).abs();
        worst = worst.max(delta);
        assert!(
            delta < 5e-5,
            "{}: {s} vs golden {} (delta {delta:e})",
            g.0,
            g.3
        );
    }
    eprintln!("worst plane golden delta: {worst:e}");
}

/// Channel plane assignment matching `validation/gen_goldens.m`:
/// `R = gen(k)`, `G = gen(mod(k,5)+1)`, `B = gen(mod(k+1,5)+1)`.
fn rgb_planes(k: usize, w: usize, h: usize) -> Vec<u8> {
    let gens = [gen1, gen2, gen3, gen4, gen5];
    let planes = [
        gens[k - 1](w, h),
        gens[k % 5](w, h),
        gens[(k + 1) % 5](w, h),
    ];
    let mut v = Vec::with_capacity(3 * w * h);
    for ((r, g), b) in planes[0].iter().zip(&planes[1]).zip(&planes[2]) {
        v.extend_from_slice(&[*r as u8, *g as u8, *b as u8]);
    }
    v
}

#[test]
fn rgb_goldens() {
    let cases: &[(&str, usize, usize, usize, usize, f64, f64)] = &[
        ("rgb_g1_g2_64", 1, 2, 64, 64, 0.810535752919, 0.775605881544),
        (
            "rgb_g4_g2_65x63",
            4,
            2,
            65,
            63,
            0.803564620437,
            0.766571972317,
        ),
        ("rgb_identical_48x40", 1, 1, 48, 40, 1.0, 1.0),
    ];
    let mut worst = 0.0f64;
    for (name, kr, kd, w, h, efsim, efsimc) in cases {
        let r = rgb_planes(*kr, *w, *h);
        let d = rgb_planes(*kd, *w, *h);
        let s = fsim_rgb8(&r, &d, *w, *h, 3 * w).unwrap();
        let d1 = (s.fsim - efsim).abs();
        let d2 = (s.fsimc - efsimc).abs();
        worst = worst.max(d1.max(d2));
        assert!(
            d1 < 5e-5 && d2 < 5e-5,
            "{name}: ({},{}) vs ({efsim},{efsimc})",
            s.fsim,
            s.fsimc
        );
    }
    eprintln!("worst rgb golden delta: {worst:e}");
}

#[test]
fn luma_goldens() {
    // Octave: L = 0.299R + 0.587G + 0.114B (unrounded) -> grayscale FSIM.
    // R,G,B = gen(1),gen(2),gen(3) vs gen(2),gen(3),gen(4) — same FSIM as
    // rgb_g1_g2's fsim column (FSIM is luma-only by definition).
    let cases: &[(&str, usize, usize, usize, usize, f64)] =
        &[("luma_g1_g2_64", 1, 2, 64, 64, 0.810535752919)];
    for (name, kr, kd, w, h, expected) in cases {
        let r = rgb_planes(*kr, *w, *h);
        let d = rgb_planes(*kd, *w, *h);
        let s = fsim_luma8(&r, &d, *w, *h, 3 * w).unwrap();
        assert!((s - expected).abs() < 5e-5, "{name}: {s} vs {expected}");
    }
}

/// Constant input degenerates the PC maps — the reference returns NaN.
#[test]
fn const_pair_is_nan() {
    let z = gen5(64, 64);
    assert!(fsim_plane_f32(&z, &z, 64, 64, 64).unwrap().is_nan());
    let zeros = alloc::vec![0.0f32; 64 * 64];
    assert!(fsim_plane_f32(&zeros, &zeros, 64, 64, 64).unwrap().is_nan());
    let g = gen1(64, 64);
    assert!(fsim_plane_f32(&z, &g, 64, 64, 64).unwrap().is_nan());
}

/// Identical inputs land on 1.0 (up to f32 noise).
#[test]
fn identical_scores_one() {
    for g in GOLDENS {
        if g.3 != 1.0 {
            continue;
        }
        let (r, d) = gens_for(g);
        let s = fsim_plane_f32(&r, &d, g.1, g.2, g.1).unwrap();
        assert!((s - 1.0).abs() < 5e-5, "{}: identical scored {s}", g.0);
    }
}

#[test]
fn buffer_length_checked() {
    let v = alloc::vec![0.0f32; 32];
    assert!(matches!(
        fsim_plane_f32(&v, &v, 16, 16, 16),
        Err(Error::BufferLength { .. })
    ));
    let u = alloc::vec![0u8; 96];
    assert!(matches!(
        fsim_rgb8(&u, &u, 16, 16, 48),
        Err(Error::BufferLength { .. })
    ));
}

/// Stride padding must not perturb the score: pack the same planes into
/// a wider buffer with garbage in the gutter.
#[test]
fn stride_invariance() {
    let (w, h, stride) = (40, 32, 56);
    let r = gen1(w, h);
    let d = gen2(w, h);
    let tight = fsim_plane_f32(&r, &d, w, h, w).unwrap();

    let mut rp = alloc::vec![-999.0f32; stride * h];
    let mut dp = alloc::vec![-999.0f32; stride * h];
    for y in 0..h {
        rp[y * stride..y * stride + w].copy_from_slice(&r[y * w..y * w + w]);
        dp[y * stride..y * stride + w].copy_from_slice(&d[y * w..y * w + w]);
    }
    let padded = fsim_plane_f32(&rp, &dp, w, h, stride).unwrap();
    assert_eq!(tight, padded);
}

/// Every SIMD tier must produce the *identical* `(fsim, fsimc)` pair:
/// each output pixel is a fixed-order f32 expression and every scalar
/// reduction keeps a fixed order, so this is exact equality.
#[cfg(all(feature = "_dev", target_arch = "x86_64"))]
#[test]
fn tier_parity() {
    use archmage::SimdToken;
    let (r, d) = (gen1(48, 40), gen2(48, 40));
    let rgb_r = rgb_planes(1, 48, 40);
    let rgb_d = rgb_planes(2, 48, 40);
    let Some(token) = archmage::X64V3Token::summon() else {
        return;
    };
    let p = kernel::Planes {
        yr: &r,
        yd: &d,
        w: 48,
        h: 40,
        iq: None,
    };
    let s_scalar = crate::dev::fsim_core_tier_scalar(archmage::ScalarToken, p);
    let s_v3 = crate::dev::fsim_core_tier_v3(token, p);
    assert_eq!(s_scalar, s_v3);

    let mk = |img: &[u8]| {
        (
            crate::plane_from_rgb8(img, 48, 40, 144, crate::Y_COEF),
            crate::plane_from_rgb8(img, 48, 40, 144, crate::I_COEF),
            crate::plane_from_rgb8(img, 48, 40, 144, crate::Q_COEF),
        )
    };
    let (yr, ir, qr) = mk(&rgb_r);
    let (yd, id, qd) = mk(&rgb_d);
    let pc = kernel::Planes {
        yr: &yr,
        yd: &yd,
        w: 48,
        h: 40,
        iq: Some((&ir, &id, &qr, &qd)),
    };
    let c_scalar = crate::dev::fsim_core_tier_scalar(archmage::ScalarToken, pc);
    let c_v3 = crate::dev::fsim_core_tier_v3(token, pc);
    assert_eq!(c_scalar, c_v3);
}
