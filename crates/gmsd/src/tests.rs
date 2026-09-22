//! Unit tests: fast path vs the straight-line reference, tier parity,
//! stride invariance, identity, and API edges.

use super::*;
use alloc::vec;
use alloc::vec::Vec;

/// Deterministic gray test image, integer levels 0..=255 (the domain the
/// reference's rounded `rgb2gray` produces): smooth gradients + texture +
/// hard edges.
fn img(w: usize, h: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (s >> 24) as f32 / 255.0 * 40.0;
            let base = 100.0 + 60.0 * ((x as f32) * 0.07).sin() * ((y as f32) * 0.05).cos();
            let edge = if (x / 13 + y / 17) % 2 == 0 {
                40.0
            } else {
                0.0
            };
            out[y * w + x] = (base + edge + noise).clamp(0.0, 255.0).round();
        }
    }
    out
}

/// A distortion: blur-ish + quantisation + a few bright pixels.
fn distort(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = src.to_vec();
    for y in 0..h {
        for x in 1..w {
            let v = 0.6 * src[y * w + x] + 0.4 * src[y * w + x - 1];
            out[y * w + x] = ((v / 12.0).round() * 12.0).clamp(0.0, 255.0);
        }
    }
    for i in (0..w * h).step_by(97) {
        out[i] = 255.0;
    }
    out
}

fn two_pass_std(map: &[f32]) -> (f64, f64) {
    let n = map.len() as f64;
    let mean = map.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = map
        .iter()
        .map(|&v| (v as f64 - mean) * (v as f64 - mean))
        .sum::<f64>()
        / (n - 1.0);
    (var.sqrt(), mean)
}

const SIZES: &[(usize, usize)] = &[
    (4, 2),
    (5, 5),
    (16, 16),
    (17, 9),
    (33, 31),
    (64, 48),
    (130, 67),
    (257, 129),
    (512, 384),
];

#[test]
fn identity_is_exactly_zero() {
    for &(w, h) in SIZES {
        let a = img(w, h, 7);
        let g = GrayImage::packed(&a, w, h).unwrap();
        let s = gmsd(g, g).unwrap();
        assert_eq!(s.gmsd, 0.0, "{w}x{h}");
        assert_eq!(s.mean_gms, 1.0, "{w}x{h}");
    }
}

#[test]
fn map_matches_straight_line_reference_bitwise() {
    for &(w, h) in SIZES {
        let a = img(w, h, 1);
        let b = distort(&a, w, h);
        let (w2, h2) = map_dims(w, h);
        let mut map = vec![0.0f32; w2 * h2];
        let s = gmsd_with_map(
            GrayImage::packed(&a, w, h).unwrap(),
            GrayImage::packed(&b, w, h).unwrap(),
            &mut map,
        )
        .unwrap();
        let reference = kernel::reference_map(
            kernel::Plane {
                data: &a,
                stride: w,
            },
            kernel::Plane {
                data: &b,
                stride: w,
            },
            w,
            h,
        );
        for (i, (&x, &y)) in map.iter().zip(reference.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{w}x{h} sample {i}: {x} vs {y}");
        }
        let (std2, mean2) = two_pass_std(&reference);
        let rel = (s.gmsd - std2).abs() / std2.max(1e-300);
        assert!(
            rel < 1e-12,
            "{w}x{h}: one-pass {} vs two-pass {std2}",
            s.gmsd
        );
        assert!((s.mean_gms - mean2).abs() < 1e-14, "{w}x{h} mean");
    }
}

#[test]
fn score_without_map_equals_score_with_map() {
    let (w, h) = (300, 170);
    let a = img(w, h, 3);
    let b = distort(&a, w, h);
    let ga = GrayImage::packed(&a, w, h).unwrap();
    let gb = GrayImage::packed(&b, w, h).unwrap();
    let mut map = vec![0.0f32; (w / 2) * (h / 2)];
    assert_eq!(
        gmsd(ga, gb).unwrap(),
        gmsd_with_map(ga, gb, &mut map).unwrap()
    );
}

#[test]
fn strided_equals_packed_exactly() {
    for &(w, h) in SIZES {
        let a = img(w, h, 11);
        let b = distort(&a, w, h);
        let stride = w + 7;
        let pad = |p: &[f32]| {
            let mut o = vec![-1234.5f32; stride * h];
            for y in 0..h {
                o[y * stride..y * stride + w].copy_from_slice(&p[y * w..(y + 1) * w]);
            }
            o
        };
        let (pa, pb) = (pad(&a), pad(&b));
        let packed = gmsd(
            GrayImage::packed(&a, w, h).unwrap(),
            GrayImage::packed(&b, w, h).unwrap(),
        )
        .unwrap();
        let strided = gmsd(
            GrayImage::new(&pa, w, h, stride).unwrap(),
            GrayImage::new(&pb, w, h, stride).unwrap(),
        )
        .unwrap();
        assert_eq!(packed, strided, "{w}x{h}");
    }
}

#[test]
fn every_tier_matches_scalar_bitwise() {
    let (w, h) = (203, 141);
    let a = img(w, h, 5);
    let b = distort(&a, w, h);
    let (w2, h2) = map_dims(w, h);
    let band = kernel::Band {
        reference: kernel::Plane {
            data: &a,
            stride: w,
        },
        distorted: kernel::Plane {
            data: &b,
            stride: w,
        },
        w2,
        h2,
        y0: 0,
        y1: h2,
    };
    let mut m_s = vec![0.0f32; w2 * h2];
    let mut s_s = vec![(0.0, 0.0); h2];
    kernel::gmsd_band_scalar(archmage::ScalarToken, &band, &mut m_s, w2, &mut s_s);
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(t) = archmage::X64V3Token::summon() {
            let mut m = vec![0.0f32; w2 * h2];
            let mut s = vec![(0.0, 0.0); h2];
            kernel::gmsd_band_v3(t, &band, &mut m, w2, &mut s);
            assert!(m.iter().zip(&m_s).all(|(x, y)| x.to_bits() == y.to_bits()));
            // Row sums: same f64 lane grouping in both tiers.
            assert_eq!(s, s_s);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(t) = archmage::NeonToken::summon() {
            let mut m = vec![0.0f32; w2 * h2];
            let mut s = vec![(0.0, 0.0); h2];
            kernel::gmsd_band_neon(t, &band, &mut m, w2, &mut s);
            assert!(m.iter().zip(&m_s).all(|(x, y)| x.to_bits() == y.to_bits()));
            assert_eq!(s, s_s);
        }
    }
}

#[test]
fn banding_is_invisible() {
    // More rows than one band, so several bands contribute.
    let (w, h) = (96, 2 * BAND_ROWS * 2 + 38);
    let a = img(w, h, 9);
    let b = distort(&a, w, h);
    let (w2, h2) = map_dims(w, h);
    let mut map = vec![0.0f32; w2 * h2];
    gmsd_with_map(
        GrayImage::packed(&a, w, h).unwrap(),
        GrayImage::packed(&b, w, h).unwrap(),
        &mut map,
    )
    .unwrap();
    let reference = kernel::reference_map(
        kernel::Plane {
            data: &a,
            stride: w,
        },
        kernel::Plane {
            data: &b,
            stride: w,
        },
        w,
        h,
    );
    assert!(
        map.iter()
            .zip(&reference)
            .all(|(x, y)| x.to_bits() == y.to_bits())
    );
}

#[test]
fn worse_distortion_scores_higher() {
    let (w, h) = (128, 96);
    let a = img(w, h, 2);
    let mild: Vec<f32> = a.iter().map(|&v| (v + 2.0).min(255.0)).collect();
    let bad = distort(&a, w, h);
    fn g(x: &[f32], w: usize, h: usize) -> GrayImage<'_> {
        GrayImage::packed(x, w, h).unwrap()
    }
    let s_mild = gmsd(g(&a, w, h), g(&mild, w, h)).unwrap().gmsd;
    let s_bad = gmsd(g(&a, w, h), g(&bad, w, h)).unwrap().gmsd;
    assert!(s_bad > s_mild, "{s_bad} <= {s_mild}");
}

#[test]
fn rgb8_gray_matches_c_round() {
    let rgb = [
        0u8, 0, 0, 255, 255, 255, 10, 200, 30, 1, 1, 2, 100, 50, 25, 3, 4, 5,
    ];
    let mut out = [0.0f32; 6];
    rgb8_to_gray(&rgb, 6, 1, 18, &mut out).unwrap();
    for (i, px) in rgb.chunks(3).enumerate() {
        let v = 0.299 * px[0] as f64 + 0.587 * px[1] as f64 + 0.114 * px[2] as f64;
        assert_eq!(out[i], v.round() as f32, "pixel {i}");
    }
}

#[test]
fn errors() {
    let a = vec![0.0f32; 64];
    let g8 = GrayImage::packed(&a, 8, 8).unwrap();
    let g4 = GrayImage::packed(&a, 4, 4).unwrap();
    assert!(matches!(gmsd(g8, g4), Err(Error::DimensionMismatch { .. })));
    let tiny = GrayImage::packed(&a, 3, 3).unwrap();
    assert!(matches!(gmsd(tiny, tiny), Err(Error::TooSmall { .. })));
    assert!(matches!(
        GrayImage::new(&a, 8, 8, 7),
        Err(Error::StrideTooSmall { .. })
    ));
    assert!(matches!(
        GrayImage::new(&a, 8, 9, 8),
        Err(Error::BufferTooSmall { .. })
    ));
    let mut short = vec![0.0f32; 15];
    assert!(matches!(
        gmsd_with_map(g8, g8, &mut short),
        Err(Error::BufferTooSmall { .. })
    ));
}

#[cfg(feature = "pixels")]
#[test]
fn pixels_rgb8_equals_rgb8_entry() {
    let (w, h) = (70, 50);
    let a = img(w, h, 4);
    let b = distort(&a, w, h);
    let to_rgb = |p: &[f32]| -> Vec<u8> {
        p.iter()
            .enumerate()
            .flat_map(|(i, &v)| [v as u8, (v as u8).wrapping_add(i as u8 % 7), 255 - v as u8])
            .collect()
    };
    let (ra, rb) = (to_rgb(&a), to_rgb(&b));
    let desc = zenpixels::PixelDescriptor::RGB8_SRGB;
    let pa = zenpixels::PixelSlice::new(&ra, w as u32, h as u32, w * 3, desc).unwrap();
    let pb = zenpixels::PixelSlice::new(&rb, w as u32, h as u32, w * 3, desc).unwrap();
    assert_eq!(
        gmsd_pixels(&pa, &pb).unwrap(),
        gmsd_rgb8(&ra, &rb, w, h, w * 3).unwrap()
    );
}
