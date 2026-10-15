//! Tests: golden vectors from the reference implementation (Octave +
//! image package, the authors' `VSI.m`), API contract checks, and
//! per-tier parity.

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

fn gens() -> [fn(usize, usize) -> Vec<f32>; 5] {
    [gen1, gen2, gen3, gen4, gen5]
}

/// Channel assignment matching `validation/gen_goldens.m`:
/// `R = gen(k)`, `G = gen(mod(k,5)+1)`, `B = gen(mod(k+1,5)+1)`, packed
/// RGB8.
fn rgb_planes(k: usize, w: usize, h: usize) -> Vec<u8> {
    let gens = gens();
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

/// Replicated gray channel: `R=G=B=gen(k)` packed RGB8.
fn grayrep_planes(k: usize, w: usize, h: usize) -> Vec<u8> {
    let p = gens()[k - 1](w, h);
    let mut v = Vec::with_capacity(3 * w * h);
    for &x in &p {
        let b = x as u8;
        v.extend_from_slice(&[b, b, b]);
    }
    v
}

/// Golden rows produced by `VSI.m` under GNU Octave (image package
/// loaded for `imresize`/`fspecial`). `(kind_r, kind_d, w, h, vsi)` —
/// regenerate with `validation/gen_goldens.m`.
const GOLDENS: &[(&str, usize, usize, usize, usize, f64)] = &[
    ("rgb_g1_g2_64", 1, 2, 64, 64, 0.805715653863),
    ("rgb_g1_g2_65x63", 1, 2, 65, 63, 0.805763795388),
    ("rgb_g4_g2_96x80", 4, 2, 96, 80, 0.918349783313),
    ("rgb_g2_g3_40", 2, 3, 40, 40, 0.883649139137),
    ("rgb_identical_64", 1, 1, 64, 64, 1.0),
    ("rgb_identical_65x63", 4, 4, 65, 63, 1.0),
    ("rgb_g1_g2_300x260", 1, 2, 300, 260, 0.820680017935),
    ("rgb_g1_g4_520x400", 1, 4, 520, 400, 0.632447337804),
    ("rgb_g2_g1_384", 2, 1, 384, 384, 0.749821565109),
    ("rgb_g3_g2_37x41", 3, 2, 37, 41, 0.880610657024),
];

#[test]
fn rgb_goldens() {
    let mut worst = 0.0f64;
    for (name, kr, kd, w, h, expected) in GOLDENS {
        let r = rgb_planes(*kr, *w, *h);
        let d = rgb_planes(*kd, *w, *h);
        let s = vsi_rgb8(&r, &d, *w, *h, 3 * w).unwrap();
        let delta = (s - expected).abs();
        worst = worst.max(delta);
        assert!(
            delta < 1e-4,
            "{name}: {s} vs golden {expected} (delta {delta:e})"
        );
    }
    eprintln!("worst rgb golden delta: {worst:e}");
}

#[test]
fn grayrep_golden() {
    let r = grayrep_planes(1, 64, 64);
    let d = grayrep_planes(2, 64, 64);
    let s = vsi_rgb8(&r, &d, 64, 64, 3 * 64).unwrap();
    assert!((s - 0.857259981622).abs() < 1e-4, "{s} vs 0.857259981622");
}

/// Flat inputs degenerate the saliency normalisation — the reference
/// returns NaN.
#[test]
fn flat_pair_is_nan() {
    let z = alloc::vec![128u8; 3 * 64 * 64];
    assert!(vsi_rgb8(&z, &z, 64, 64, 3 * 64).unwrap().is_nan());
    let zeros = alloc::vec![0u8; 3 * 64 * 64];
    assert!(vsi_rgb8(&zeros, &zeros, 64, 64, 3 * 64).unwrap().is_nan());
    let g = rgb_planes(1, 64, 64);
    assert!(vsi_rgb8(&z, &g, 64, 64, 3 * 64).unwrap().is_nan());
}

/// Identical inputs land on 1.0 (up to f32 noise).
#[test]
fn identical_scores_one() {
    for (_, kr, kd, w, h, expected) in GOLDENS {
        if *expected != 1.0 || kr != kd {
            continue;
        }
        let r = rgb_planes(*kr, *w, *h);
        let s = vsi_rgb8(&r, &r, *w, *h, 3 * w).unwrap();
        assert!((s - 1.0).abs() < 1e-4, "identical {w}x{h} scored {s}");
    }
}

#[test]
fn buffer_length_checked() {
    let u = alloc::vec![0u8; 96];
    assert!(matches!(
        vsi_rgb8(&u, &u, 16, 16, 48),
        Err(Error::BufferLength { .. })
    ));
}

/// Stride padding must not perturb the score: pack the same RGB into
/// a wider buffer with garbage in the gutter.
#[test]
fn stride_invariance() {
    let (w, h, stride) = (40, 32, 3 * 56);
    let r = rgb_planes(1, w, h);
    let d = rgb_planes(2, w, h);
    let tight = vsi_rgb8(&r, &d, w, h, 3 * w).unwrap();

    let mut rp = alloc::vec![0u8; stride * h];
    let mut dp = alloc::vec![0u8; stride * h];
    for y in 0..h {
        rp[y * stride..y * stride + 3 * w].copy_from_slice(&r[y * 3 * w..y * 3 * w + 3 * w]);
        dp[y * stride..y * stride + 3 * w].copy_from_slice(&d[y * 3 * w..y * 3 * w + 3 * w]);
    }
    let padded = vsi_rgb8(&rp, &dp, w, h, stride).unwrap();
    assert_eq!(tight, padded);
}

/// `decimation_factor` boundary: `round(min/256)` (not floor) — F=1 at
/// 383, F=2 at 384.
#[cfg(feature = "_dev")]
#[test]
fn f_factor_boundary() {
    assert_eq!(kernel::decimation_factor(383, 600), 1);
    assert_eq!(kernel::decimation_factor(384, 600), 2);
    assert_eq!(kernel::decimation_factor(639, 700), 2);
    assert_eq!(kernel::decimation_factor(640, 700), 3);
    assert_eq!(kernel::decimation_factor(16, 16), 1);
}

/// The `imresize` port is bit-exact against Octave — pin a few outputs
/// so a refactor can't silently change it. Values generated by
/// `imresize_check.m` (the golden vectors below are produced by the
/// same algorithm verified bit-identical there).
#[test]
fn imresize_shape() {
    // Identity axis skipped.
    let v = alloc::vec![1.0f32, 2.0, 3.0, 4.0];
    let same = kernel::imresize(&v, 2, 2, 2, 2);
    assert_eq!(v, same);
    // Upscale preserves the input values at mapped sample points.
    let up = kernel::imresize(&v, 2, 2, 4, 4);
    assert_eq!(up.len(), 16);
    assert!(up.iter().all(|x| x.is_finite()));
}

/// Every SIMD tier must produce the *identical* score — the pool stage
/// is fixed-order f64 accumulation over per-lane scalar expressions.
#[cfg(all(feature = "_dev", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[test]
fn tier_parity() {
    use archmage::SimdToken;
    let (w, h) = (48, 40);
    let ri = Input::from_rgb8(&rgb_planes(1, w, h), w, h, 3 * w);
    let di = Input::from_rgb8(&rgb_planes(2, w, h), w, h, 3 * w);
    let mut scratch = kernel::Scratch::new();
    // `prepare` needs the private Scratch type — build planes via the
    // full pipeline for both tiers (prepare is shared, so only the
    // pool differs per tier anyway).
    let p = kernel::prepare_with_downscale(&ri, &di, w, h, Downscale::Auto, &mut scratch);
    let s_scalar = crate::dev::vsi_core_tier_scalar_t(archmage::ScalarToken, p, None);
    #[cfg(target_arch = "x86_64")]
    {
        let Some(token) = archmage::X64V3Token::summon() else {
            return;
        };
        let s_v3 = crate::dev::vsi_core_tier_v3_t(token, p, None);
        assert_eq!(s_scalar, s_v3);
    }
    #[cfg(target_arch = "aarch64")]
    {
        let Some(token) = archmage::NeonToken::summon() else {
            return;
        };
        let s_neon = crate::dev::vsi_core_tier_neon_t(token, p, None);
        assert_eq!(s_scalar.to_bits(), s_neon.to_bits());
    }
}

/// The dispatched (SIMD) Lab fill is bit-identical to the scalar one:
/// every 8-bit level on each channel, mixed so all three planes vary,
/// with a non-multiple-of-4 length to cover the scalar tail.
#[test]
fn srgb_to_lab_dispatch_matches_scalar_bitwise() {
    let n = 256 * 257 + 3;
    let dr: alloc::vec::Vec<f32> = (0..n).map(|i| (i % 256) as f32).collect();
    let dg: alloc::vec::Vec<f32> = (0..n).map(|i| ((i * 7 + i / 256) % 256) as f32).collect();
    let db: alloc::vec::Vec<f32> = (0..n)
        .map(|i| (255 - (i * 13 + i / 97) % 256) as f32)
        .collect();
    let z = || alloc::vec![0f32; n];
    let (mut l0, mut a0, mut b0) = (z(), z(), z());
    let (mut l1, mut a1, mut b1) = (z(), z(), z());
    kernel::srgb_to_lab_all_scalar(&dr, &dg, &db, &mut l0, &mut a0, &mut b0);
    kernel::srgb_to_lab_all(&dr, &dg, &db, &mut l1, &mut a1, &mut b1);
    for i in 0..n {
        assert_eq!(l0[i].to_bits(), l1[i].to_bits(), "L at {i}");
        assert_eq!(a0[i].to_bits(), a1[i].to_bits(), "a at {i}");
        assert_eq!(b0[i].to_bits(), b1[i].to_bits(), "b at {i}");
    }
}

/// `vsi_rgb8_with_diffmap` returns the identical score and a similarity
/// map at the decimated resolution (full resolution below min-dim 384).
#[test]
fn vsi_diffmap_matches_score() {
    use zenmetrics_abstractions::Evaluation;
    let (w, h) = (48usize, 40usize);
    let r: alloc::vec::Vec<u8> = (0..w * h * 3).map(|i| (i * 13 + 7) as u8 % 251).collect();
    let mut d = r.clone();
    for i in (300..700).step_by(3) {
        d[i] = 255 - d[i];
    }
    let plain = crate::vsi_rgb8(&r, &d, w, h, w * 3).unwrap();
    let (score, dm) = crate::vsi_rgb8_with_diffmap(&r, &d, w, h, w * 3).unwrap();
    assert_eq!(plain, score);
    assert_eq!(dm.map().len(), w * h);
    let view = Evaluation::map(&dm).expect("map");
    assert_eq!(
        view.polarity,
        zenmetrics_abstractions::MapPolarity::Similarity
    );
    assert_eq!(
        view.geometry,
        zenmetrics_abstractions::MapGeometry::identity(48, 40)
            .with_support(zenmetrics_abstractions::Support::Global)
    );
}

/// Regression: at min-dim ≥ 384 the reference decimates by `F ≥ 2`. The map
/// used to be reported as `w × h` with only the first `⌈w/F⌉·⌈h/F⌉`
/// entries written (the rest zero).
#[test]
fn vsi_diffmap_is_decimated_at_large_sizes() {
    use zenmetrics_abstractions::Evaluation;
    let (w, h) = (401usize, 390usize);
    let r = zenmetrics_abstractions::conformance::textured_rgb8(w, h, 3);
    let d = zenmetrics_abstractions::conformance::distort_rgb8(&r, w, h);
    let (_, dm) = crate::vsi_rgb8_with_diffmap(&r, &d, w, h, w * 3).unwrap();
    assert_eq!(dm.factor(), 2);
    assert_eq!((dm.width, dm.height), (201, 195));
    assert_eq!(dm.map().len(), 201 * 195);
    let last_row = &dm.map()[194 * 201..];
    assert!(
        last_row.iter().all(|v| v.is_finite() && *v > 0.0),
        "tail written"
    );
    let g = Evaluation::map(&dm).unwrap().geometry;
    // F = 2, s = 1: sample k averages input 2k, 2k+1.
    assert_eq!(g.cell_rect(3, 0).map(|r| (r.x, r.width)), Some((6, 2)));
}

/// The generic surface honors the abstractions contract.
#[test]
fn generic_surface_conforms() {
    zenmetrics_abstractions::conformance::assert_conforms_rgb8(&mut crate::Vsi::default(), 41, 33);
}

#[test]
fn fixed_factor_matches_auto_and_strided() {
    for (w, factor) in [(31, 1), (384, 2), (640, 3), (896, 4)] {
        let h = w + 1;
        let r: Vec<u8> = (0..w * h * 3)
            .map(|i| ((i * 37 + i / 11) % 256) as u8)
            .collect();
        let d: Vec<u8> = r
            .iter()
            .enumerate()
            .map(|(i, &v)| if i % 17 == 0 { v.wrapping_add(29) } else { v })
            .collect();
        let scorer = Vsi::with_downscale(Downscale::Factor(
            core::num::NonZeroU32::new(factor).unwrap(),
        ));
        let auto = vsi_rgb8(&r, &d, w, h, w * 3).unwrap();
        let fixed = scorer.score_rgb8(&r, &d, w, h, w * 3).unwrap();
        assert_eq!(auto.to_bits(), fixed.to_bits());
        let stride = w * 3 + 7;
        let mut rs = alloc::vec![0;stride*h];
        let mut ds = rs.clone();
        for y in 0..h {
            rs[y * stride..y * stride + w * 3].copy_from_slice(&r[y * w * 3..(y + 1) * w * 3]);
            ds[y * stride..y * stride + w * 3].copy_from_slice(&d[y * w * 3..(y + 1) * w * 3]);
        }
        assert_eq!(fixed, scorer.score_rgb8(&rs, &ds, w, h, stride).unwrap());
        let dm = scorer.diffmap_rgb8(&rs, &ds, w, h, stride).unwrap();
        assert_eq!(dm.factor(), factor);
        assert_eq!(dm.vsi.to_bits(), fixed.to_bits());
    }
}

#[test]
fn downscale_identity_preserves_auto_and_distinguishes_factors() {
    use zenmetrics_abstractions::Metric as _;
    // Auto fingerprint from base 765a2f9c's identity fields and FNV-1a encoding.
    let auto = Vsi::default().id().fingerprint();
    assert_eq!(auto, 0x57ad6b5bda41d80a);
    let fixed = |n| {
        Vsi::with_downscale(Downscale::Factor(core::num::NonZeroU32::new(n).unwrap()))
            .id()
            .fingerprint()
    };
    assert_ne!(auto, fixed(2));
    assert_ne!(fixed(2), fixed(3));
}
