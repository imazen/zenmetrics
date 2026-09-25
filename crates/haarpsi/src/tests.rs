//! Tests: golden vectors from the reference implementation (Octave,
//! the authors' MIT-licensed `HaarPSI.m`), API contract checks, and
//! per-tier parity.

use alloc::vec::Vec;

use crate::*;

/// Golden rows produced by running the reference `HaarPSI.m` under
/// GNU Octave on the `gen` patterns below (same integer arithmetic,
/// so the inputs are bit-identical). Columns: `(name, w, h, sub,
/// nosub)`. Regenerate with `validation/gen_goldens.m`.
struct Golden {
    name: &'static str,
    width: usize,
    height: usize,
    sub: f64,
    nosub: f64,
    make: fn() -> (Vec<f32>, Vec<f32>),
}

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

const GOLDENS: &[Golden] = &[
    Golden {
        name: "g1_vs_g2_64",
        width: 64,
        height: 64,
        sub: 0.373060134426,
        nosub: 0.248258426705,
        make: || (gen1(64, 64), gen2(64, 64)),
    },
    Golden {
        name: "g1_vs_g2_65x63",
        width: 65,
        height: 63,
        sub: 0.398646630698,
        nosub: 0.247237525378,
        make: || (gen1(65, 63), gen2(65, 63)),
    },
    Golden {
        name: "g1_vs_g4_64",
        width: 64,
        height: 64,
        sub: 0.872800039187,
        nosub: 0.660653398378,
        make: || (gen1(64, 64), gen4(64, 64)),
    },
    Golden {
        name: "g3_vs_g4_96x80",
        width: 96,
        height: 80,
        sub: 0.376908742121,
        nosub: 0.096892506921,
        make: || (gen3(96, 80), gen4(96, 80)),
    },
    Golden {
        name: "const_identical",
        width: 64,
        height: 64,
        sub: 1.0,
        nosub: 1.0,
        make: || (gen5(64, 64), gen5(64, 64)),
    },
    Golden {
        name: "g1_identical",
        width: 64,
        height: 64,
        sub: 1.0,
        nosub: 1.000000000005,
        make: || (gen1(64, 64), gen1(64, 64)),
    },
    Golden {
        name: "g2_identical_40",
        width: 40,
        height: 40,
        sub: 1.0,
        nosub: 1.0,
        make: || (gen2(40, 40), gen2(40, 40)),
    },
    Golden {
        name: "g4_identical",
        width: 64,
        height: 64,
        sub: 1.0,
        nosub: 1.000000000004,
        make: || (gen4(64, 64), gen4(64, 64)),
    },
    Golden {
        name: "g3_vs_g1_32",
        width: 32,
        height: 32,
        sub: 0.367527036348,
        nosub: 0.102738786276,
        make: || (gen3(32, 32), gen1(32, 32)),
    },
    Golden {
        name: "const_vs_g1_64",
        width: 64,
        height: 64,
        sub: 0.165689811126,
        nosub: 0.041156156723,
        make: || (gen5(64, 64), gen1(64, 64)),
    },
    Golden {
        name: "g2_vs_g1_37x41",
        width: 37,
        height: 41,
        sub: 0.406441215352,
        nosub: 0.264857039507,
        make: || (gen2(37, 41), gen1(37, 41)),
    },
];

#[test]
fn plane_goldens() {
    for g in GOLDENS {
        let (r, d) = (g.make)();
        let sub = haarpsi_plane_f32_opts(&r, &d, g.width, g.height, g.width, true).unwrap();
        assert!(
            (sub - g.sub).abs() < 5e-5,
            "{}: sub {sub} vs golden {}",
            g.name,
            g.sub
        );
        let nosub = haarpsi_plane_f32_opts(&r, &d, g.width, g.height, g.width, false).unwrap();
        assert!(
            (nosub - g.nosub).abs() < 5e-5,
            "{}: nosub {nosub} vs golden {}",
            g.name,
            g.nosub
        );
    }
}

/// Strided input must produce the same score as packed input.
#[test]
fn stride_invariance() {
    let (w, h, stride) = (64usize, 64usize, 80usize);
    let mut r = alloc::vec![0.0f32; stride * h];
    let mut d = alloc::vec![0.0f32; stride * h];
    let (rp, dp) = (gen1(w, h), gen2(w, h));
    for y in 0..h {
        r[y * stride..y * stride + w].copy_from_slice(&rp[y * w..(y + 1) * w]);
        d[y * stride..y * stride + w].copy_from_slice(&dp[y * w..(y + 1) * w]);
    }
    let a = haarpsi_plane_f32(&rp, &dp, w, h, w).unwrap();
    let b = haarpsi_plane_f32(&r, &d, w, h, stride).unwrap();
    assert_eq!(a, b);
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
    let cases: &[(&str, usize, usize, usize, usize, f64)] = &[
        ("rgb_g1_g2_64", 1, 2, 64, 64, 0.571452786952),
        ("rgb_identical_48x40", 1, 1, 48, 40, 1.0),
        ("rgb_g4_g2_65x63", 4, 2, 65, 63, 0.332309263003),
    ];
    for (name, kr, kd, w, h, expected) in cases {
        let r = rgb_planes(*kr, *w, *h);
        let d = rgb_planes(*kd, *w, *h);
        let s = haarpsi_rgb8(&r, &d, *w, *h, 3 * w).unwrap();
        assert!((s - expected).abs() < 5e-5, "{name}: {s} vs {expected}");
    }
}

#[test]
fn luma_goldens() {
    // Octave: L = 0.299R + 0.587G + 0.114B (unrounded) -> grayscale HaarPSI.
    let cases: &[(&str, usize, usize, usize, usize, f64)] = &[
        ("luma_g1_g2_64", 1, 2, 64, 64, 0.677912751857),
        ("luma_identical_48x40", 4, 4, 48, 40, 1.0),
    ];
    for (name, kr, kd, w, h, expected) in cases {
        let r = rgb_planes(*kr, *w, *h);
        let d = rgb_planes(*kd, *w, *h);
        let s = haarpsi_luma8(&r, &d, *w, *h, 3 * w).unwrap();
        assert!((s - expected).abs() < 5e-5, "{name}: {s} vs {expected}");
    }
}

#[test]
fn identical_scores_one() {
    for g in GOLDENS {
        if g.sub != 1.0 {
            continue;
        }
        let (r, d) = (g.make)();
        let s = haarpsi_plane_f32(&r, &d, g.width, g.height, g.width).unwrap();
        assert!(
            (s - 1.0).abs() < 1e-5,
            "{}: identical input scored {s}",
            g.name
        );
    }
}

/// An all-zero pair has no weight anywhere — the reference returns NaN.
#[test]
fn zero_pair_is_nan() {
    let z = alloc::vec![0.0f32; 64 * 64];
    let s = haarpsi_plane_f32(&z, &z, 64, 64, 64).unwrap();
    assert!(s.is_nan());
}

#[test]
fn buffer_length_checked() {
    let v = alloc::vec![0.0f32; 32];
    assert!(matches!(
        haarpsi_plane_f32(&v, &v, 16, 16, 16),
        Err(Error::BufferLength { .. })
    ));
    let u = alloc::vec![0u8; 96];
    assert!(matches!(
        haarpsi_rgb8(&u, &u, 16, 16, 48),
        Err(Error::BufferLength { .. })
    ));
}

/// Every SIMD tier must produce the *identical* f64 score: each output
/// pixel is a fixed-order f32 expression and the pooling uses a fixed
/// lane-grouping, so this is exact equality, not a tolerance.
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
    let pr = kernel::plane_from_f32(&r, 48, 40, 48, true);
    let pd = kernel::plane_from_f32(&d, 48, 40, 48, true);
    let s_scalar = crate::dev::haarpsi_core_tier_scalar(archmage::ScalarToken, &pr, &pd, None);
    let s_v3 = crate::dev::haarpsi_core_tier_v3(token, &pr, &pd, None);
    assert_eq!(s_scalar, s_v3);

    let mk = |img: &[u8]| {
        (
            kernel::plane_from_rgb8(img, 48, 40, 144, crate::Y_COEF, true),
            kernel::plane_from_rgb8(img, 48, 40, 144, crate::I_COEF, true),
            kernel::plane_from_rgb8(img, 48, 40, 144, crate::Q_COEF, true),
        )
    };
    let (yr, ir, qr) = mk(&rgb_r);
    let (yd, id, qd) = mk(&rgb_d);
    let c_scalar = crate::dev::haarpsi_core_tier_scalar(
        archmage::ScalarToken,
        &yr,
        &yd,
        Some((&ir, &id, &qr, &qd)),
    );
    let c_v3 = crate::dev::haarpsi_core_tier_v3(token, &yr, &yd, Some((&ir, &id, &qr, &qd)));
    assert_eq!(c_scalar, c_v3);
}
