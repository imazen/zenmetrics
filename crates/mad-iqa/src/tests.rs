//! Tests: golden vectors from the authors' implementation (Octave +
//! `hi_index.m`/`lo_index.m` + `ical_std.m`/`ical_stat.m` shims ported
//! verbatim from the C mex — see `validation/`), API contract checks,
//! and per-tier parity.

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

/// Golden rows produced by the authors' `hi_index.m` + `lo_index.m`
/// under GNU Octave (with the `ical_std`/`ical_stat` `.m` shims —
/// verbatim ports of the release's C mex). `(name, kind_r, kind_d,
/// w, h, hi, lo, mad)` — regenerate with
/// `validation/gen_goldens.m`.
#[allow(clippy::type_complexity)]
const GOLDENS: &[(&str, usize, usize, usize, usize, f64, f64, f64)] = &[
    (
        "g1_g2_64",
        1,
        2,
        64,
        64,
        131572.283897184563,
        9.688765373323,
        198.154310728882,
    ),
    (
        "g1_g2_65x63",
        1,
        2,
        65,
        63,
        132302.255032831221,
        9.384997604771,
        193.943121996639,
    ),
    (
        "g4_g2_96x80",
        4,
        2,
        96,
        80,
        102819.537075095577,
        8.893306321210,
        184.469973147550,
    ),
    (
        "g1_g2_256",
        1,
        2,
        256,
        256,
        101143.088739909173,
        9.065290225621,
        186.691455459098,
    ),
    (
        "g2_g3_256",
        2,
        3,
        256,
        256,
        64478.986671323131,
        16.523224071463,
        269.257575656904,
    ),
    (
        "g1_g4_300x260",
        1,
        4,
        300,
        260,
        5168.494644471720,
        0.662500927369,
        27.084379048476,
    ),
    (
        "g3_g2_37x41",
        3,
        2,
        37,
        41,
        218003.466036476486,
        13.090370420302,
        249.271256336578,
    ),
    (
        "g1_g2_48x40",
        1,
        2,
        48,
        40,
        128490.494412795029,
        10.559435780623,
        209.883711362091,
    ),
    (
        "g5_g3_512",
        5,
        3,
        512,
        512,
        246576.072716682946,
        21.979369283852,
        359.837788298032,
    ),
    (
        "g1_g2_34",
        1,
        2,
        34,
        34,
        112251.749213334246,
        5.846236835626,
        139.471865095581,
    ),
    ("identical_256", 1, 1, 256, 256, 0.0, 0.0, 0.0),
    ("const_identical_64", 5, 5, 64, 64, 0.0, 0.0, 0.0),
    (
        "constref_vs_g1_64",
        5,
        1,
        64,
        64,
        149462.479444795055,
        16.601442321003,
        288.506837133495,
    ),
];

#[test]
fn plane_goldens() {
    let mut worst_hi = 0.0f64;
    let mut worst_lo = 0.0f64;
    let mut worst_mad = 0.0f64;
    for (name, kr, kd, w, h, e_hi, e_lo, e_mad) in GOLDENS {
        let r = gens()[*kr - 1](*w, *h);
        let d = gens()[*kd - 1](*w, *h);
        let s = mad_plane_f32(&r, &d, *w, *h, *w).unwrap();
        // MAD outputs span 0..~1e5; compare on a relative scale.
        let (d_hi, d_lo, d_mad) = (
            (s.hi - e_hi).abs() / e_hi.max(1.0),
            (s.lo - e_lo).abs() / e_lo.max(1.0),
            (s.mad - e_mad).abs() / e_mad.max(1.0),
        );
        worst_hi = worst_hi.max(d_hi);
        worst_lo = worst_lo.max(d_lo);
        worst_mad = worst_mad.max(d_mad);
        assert!(
            d_hi < 1e-5 && d_lo < 1e-5 && d_mad < 1e-5,
            "{name}: ({},{},{}) vs ({},{},{}) — rel ({d_hi:e},{d_lo:e},{d_mad:e})",
            s.hi,
            s.lo,
            s.mad,
            e_hi,
            e_lo,
            e_mad
        );
    }
    #[cfg(feature = "std")]
    std::eprintln!("worst rel deltas: hi {worst_hi:e} lo {worst_lo:e} mad {worst_mad:e}");
}

/// MAD is a distance: identical inputs score exactly 0 on both
/// strategies and the blend.
#[test]
fn identical_is_zero() {
    let p = gen1(64, 64);
    let s = mad_plane_f32(&p, &p, 64, 64, 64).unwrap();
    assert_eq!(s.hi, 0.0);
    assert_eq!(s.lo, 0.0);
    assert_eq!(s.mad, 0.0);
}

/// The reference's `mp(17:end-17)` edge kill leaves nothing to pool
/// for `min(w,h) < 34` — `norm([])/sqrt(0)` → NaN.
#[test]
fn small_image_is_nan() {
    let r = gen1(24, 24);
    let d = gen2(24, 24);
    assert!(mad_plane_f32(&r, &d, 24, 24, 24).unwrap().mad.is_nan());
}

#[test]
fn buffer_length_checked() {
    let p = gen1(64, 64);
    let err = mad_plane_f32(&p[..100], &p, 64, 64, 64).unwrap_err();
    assert_eq!(
        err,
        Error::BufferLength {
            needed: 64 * 64,
            got: 100
        }
    );
    let rgb = alloc::vec![0u8; 3 * 64 * 64];
    assert!(mad_rgb8(&rgb[..100], &rgb, 64, 64, 3 * 64).is_err());
}

/// Rows beyond `stride` must not affect the score.
#[test]
fn stride_invariance() {
    let (w, h) = (64usize, 64usize);
    let r = gen1(w, h);
    let d = gen2(w, h);
    let s0 = mad_plane_f32(&r, &d, w, h, w).unwrap();
    let pad = 37usize;
    let mut rp = alloc::vec![0.0f32; (w + pad) * h];
    let mut dp = alloc::vec![0.0f32; (w + pad) * h];
    for y in 0..h {
        rp[y * (w + pad)..y * (w + pad) + w].copy_from_slice(&r[y * w..y * w + w]);
        dp[y * (w + pad)..y * (w + pad) + w].copy_from_slice(&d[y * w..y * w + w]);
    }
    let s1 = mad_plane_f32(&rp, &dp, w, h, w + pad).unwrap();
    assert_eq!(s0.hi, s1.hi);
    assert_eq!(s0.lo, s1.lo);
    assert_eq!(s0.mad, s1.mad);
}

/// Every available SIMD tier must agree bit-for-bit with the scalar
/// tier (per-element expressions have fixed order).
#[cfg(feature = "_dev")]
#[test]
fn tier_parity() {
    use crate::dev::*;
    use archmage::SimdToken;
    let r = gen1(96, 96);
    let d = gen2(96, 96);
    let a = Args {
        im1: &r,
        im2: &d,
        w: 96,
        h: 96,
        maps: None,
    };
    let s0 = mad_core_tier_scalar_t(archmage::ScalarToken, a);
    #[cfg(target_arch = "x86_64")]
    {
        let a = Args {
            im1: &r,
            im2: &d,
            w: 96,
            h: 96,
            maps: None,
        };
        let Some(t3) = archmage::X64V3Token::summon() else {
            return;
        };
        let s3 = mad_core_tier_v3_t(t3, a);
        assert_eq!(s0, s3, "v3 diverged");
        #[cfg(feature = "avx512")]
        if let Some(t4) = archmage::X64V4xToken::summon() {
            let s4 = mad_core_tier_v4x_t(
                t4,
                Args {
                    im1: &r,
                    im2: &d,
                    w: 96,
                    h: 96,
                    maps: None,
                },
            );
            assert_eq!(s0, s4, "v4x diverged");
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let tn = archmage::NeonToken::summon().unwrap();
        let sn = mad_core_tier_neon_t(
            tn,
            Args {
                im1: &r,
                im2: &d,
                w: 96,
                h: 96,
                maps: None,
            },
        );
        assert_eq!(s0, sn, "neon diverged");
    }
}

/// `interior_norm`'s accumulation over already-cropped interior rows:
/// eight lane partials per row, carried across rows, tail into lane 0.
fn rms_like_interior_norm(rows: &[f32], iw: usize) -> f64 {
    let mut acc = [0.0f64; 8];
    let mut count = 0usize;
    for row in rows.chunks_exact(iw) {
        let mut c = 0;
        while c + 8 <= row.len() {
            for k in 0..8 {
                let v = row[c + k] as f64;
                acc[k] += v * v;
            }
            c += 8;
        }
        while c < row.len() {
            let v = row[c] as f64;
            acc[0] += v * v;
            c += 1;
        }
        count += row.len();
    }
    libm::sqrt(acc[0] + acc[1] + acc[2] + acc[3] + acc[4] + acc[5] + acc[6] + acc[7])
        / libm::sqrt(count as f64)
}

fn rgb_from_luma_like(p: &[f32]) -> Vec<u8> {
    p.iter()
        .flat_map(|&v| [v as u8, (255.0 - v) as u8, (v * 0.5) as u8])
        .collect()
}

/// The interior maps fold back to `hi` and `lo` bit for bit: `HI` is 10×
/// their RMS, `LO` the RMS of the tile values over interior pixels.
#[test]
fn strategy_maps_fold_to_the_indices_exactly() {
    for (w, h) in [(96usize, 96usize), (77, 61), (34, 40), (130, 70)] {
        let (a, b) = (
            rgb_from_luma_like(&gen1(w, h)),
            rgb_from_luma_like(&gen4(w, h)),
        );
        let plain = mad_rgb8(&a, &b, w, h, w * 3).unwrap();
        let (score, raw) = mad_rgb8_maps(&a, &b, w, h, w * 3).unwrap();
        assert_eq!(score, plain, "{w}x{h}");
        let maps = MadMaps::new(score, raw, w, h);
        let (iw, ih) = interior(w, h).unwrap();
        assert_eq!(maps.hi.len(), iw * ih);
        let hi = rms_like_interior_norm(&maps.hi, iw) * 10.0;
        assert_eq!(hi.to_bits(), plain.hi.to_bits(), "{w}x{h} hi");
        // Expand the tile map over the interior pixels it covers.
        let (tx, ty) = (
            maps.geometry[1].cells_x as usize,
            maps.geometry[1].cells_y as usize,
        );
        let mut px = Vec::with_capacity(iw * ih);
        for y in 0..ih {
            for x in 0..iw {
                let (jt, it) = (y / 4, x / 4);
                assert!(jt < ty && it < tx, "{w}x{h}: interior pixel without a tile");
                px.push(maps.lo[jt * tx + it]);
            }
        }
        let lo = rms_like_interior_norm(&px, iw);
        assert_eq!(lo.to_bits(), plain.lo.to_bits(), "{w}x{h} lo");
    }
}

/// `hi` is the contrast mask times a 16×16 local MSE of the raw planes
/// over `{x−7 … x+8}`, so a changed patch can only light up cells whose
/// box holds part of it: this pins the map's pixel alignment (cell
/// `(i, j)` is pixel `(16+i, 16+j)`). A bright patch on a flat field clears
/// the contrast threshold; a faint change in texture may not, and then the
/// mask, and the map, are legitimately 0.
#[test]
fn hi_map_alignment_matches_the_local_mse_window() {
    let (w, h) = (96usize, 80usize);
    let base = alloc::vec![64.0f32; w * h];
    let size = 3;
    for (px, py) in [(40usize, 37usize), (20, 20), (73, 60)] {
        let mut d = base.clone();
        for y in py..py + size {
            for x in px..px + size {
                d[y * w + x] = 250.0;
            }
        }
        let (a, b) = (rgb_from_luma_like(&base), rgb_from_luma_like(&d));
        let (s, raw) = mad_rgb8_maps(&a, &b, w, h, w * 3).unwrap();
        let maps = MadMaps::new(s, raw, w, h);
        let (iw, ih) = interior(w, h).unwrap();
        let mut lit = 0;
        for j in 0..ih {
            for i in 0..iw {
                if maps.hi[j * iw + i] != 0.0 {
                    let (x, y) = (16 + i, 16 + j);
                    assert!(
                        x + 8 >= px
                            && x <= px + size - 1 + 7
                            && y + 8 >= py
                            && y <= py + size - 1 + 7,
                        "patch ({px},{py}): cell at ({x},{y}) outside the box window"
                    );
                    lit += 1;
                }
            }
        }
        assert!(lit > 0, "patch ({px},{py}) not visible");
    }
}
