//! Tests for `mdsi`: bit-identity of the optimized path against the straight-line reference,
//! tier and thread parity, and the caller-controlled 116-pair author-score gate.

use super::*;

fn lcg_image(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut s = seed;
    (0..w * h * 3)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 24) as u8
        })
        .collect()
}

fn perturb(img: &[u8], seed: u32) -> Vec<u8> {
    let mut s = seed;
    img.iter()
        .map(|&v| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let d = ((s >> 24) as i32 % 41) - 20;
            (v as i32 + d).clamp(0, 255) as u8
        })
        .collect()
}

fn padded(img: &[u8], w: usize, h: usize, extra: usize) -> (Vec<u8>, usize) {
    let stride = w * 3 + extra;
    let mut out = vec![0xA5u8; stride * h];
    for y in 0..h {
        out[y * stride..y * stride + w * 3].copy_from_slice(&img[y * w * 3..(y + 1) * w * 3]);
    }
    (out, stride)
}

fn assert_bits(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: element {i}: {x} vs {y}");
    }
}

const SIZES: [(usize, usize); 14] = [
    (1, 1),
    (1, 9),
    (9, 1),
    (2, 2),
    (3, 5),
    (17, 13),
    (33, 31),
    (63, 47),
    (127, 129),
    (128, 200),
    (383, 385),
    (511, 384),
    (640, 641),
    (769, 513),
];

#[test]
fn factor_matches_float_round() {
    for s in 0..=70_000usize {
        let want = ((s as f64 / 256.0).round() as usize).max(1);
        assert_eq!(downsample_factor(s, usize::MAX / 4), want, "min {s}");
    }
}

#[test]
fn integer_box_matches_reference_for_every_factor() {
    for &(w, h) in &[(1usize, 1usize), (5, 3), (20, 20), (37, 29), (64, 65)] {
        let img = lcg_image(w, h, 3 + w as u32);
        let (buf, stride) = padded(&img, w, h, 7);
        for m in (1..=40).chain([255, 256, 257, 258, 300, 513]) {
            let s = source_with_factor(&buf, &buf, w, h, stride, m);
            let planes = prepare(&s);
            assert_eq!(
                (planes.width, planes.height),
                (w.div_ceil(m), h.div_ceil(m))
            );
            // Channel planes as the reference builds them.
            for c in 0..3usize {
                let (want, ow, oh) = reference::box_downsample(&buf, c, w, h, stride, m);
                // Lr = 0.2989 R + ... reproduces from the same averaged
                // channels, so compare through the luminance plane.
                if c == 0 {
                    let (g, _, _) = reference::box_downsample(&buf, 1, w, h, stride, m);
                    let (b, _, _) = reference::box_downsample(&buf, 2, w, h, stride, m);
                    for y in 0..oh {
                        for x in 0..ow {
                            let i = y * ow + x;
                            let l = 0.2989 * want[i] + 0.5870 * g[i] + 0.1140 * b[i];
                            let got = planes.data[(y + 1) * planes.row_len() + x + 1];
                            assert_eq!(got.to_bits(), l.to_bits(), "L m={m} ({x},{y})");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn fast_path_is_bit_identical_to_reference() {
    for &(w, h) in &SIZES {
        for extra in [0usize, 5] {
            let a = lcg_image(w, h, 1 + w as u32);
            let b = perturb(&a, 9 + h as u32);
            let (pa, stride) = padded(&a, w, h, extra);
            let (pb, _) = padded(&b, w, h, extra);
            let p = Params::default();
            let (want, want_map) = reference::score_and_map(&pa, &pb, w, h, stride, &p).unwrap();
            let (got, gcs, _) = score_with(&pa, &pb, w, h, stride, &p).unwrap();
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{w}x{h} +{extra}: {got} vs {want}"
            );
            assert_bits(&gcs, &want_map, "gcs map");
        }
    }
}

#[test]
fn negative_controls_track_the_reference_and_change_the_score() {
    let (w, h) = (129, 97);
    let a = lcg_image(w, h, 5);
    let b = perturb(&a, 6);
    let base = run(&a, &b, w, h, w * 3).unwrap();
    for wrong in [
        Params {
            c3: 5500.0,
            ..Params::default()
        },
        Params {
            c1: 280.0,
            ..Params::default()
        },
        Params {
            c2: 110.0,
            ..Params::default()
        },
        Params {
            alpha: 0.5,
            ..Params::default()
        },
    ] {
        let (fast, _, _) = score_with(&a, &b, w, h, w * 3, &wrong).unwrap();
        let (slow, _) = reference::score_and_map(&a, &b, w, h, w * 3, &wrong).unwrap();
        assert_eq!(fast.to_bits(), slow.to_bits());
        assert!((fast - base).abs() > 1e-4 * base, "{wrong:?}");
    }
}

#[test]
fn identity_is_exactly_zero() {
    for &(w, h) in &SIZES {
        let a = lcg_image(w, h, 42);
        assert_eq!(run(&a, &a, w, h, w * 3).unwrap(), 0.0, "{w}x{h}");
    }
}

#[test]
fn small_sizes_are_finite_and_non_negative() {
    for &(w, h) in &SIZES {
        let a = lcg_image(w, h, 7);
        let b = perturb(&a, 8);
        let s = run(&a, &b, w, h, w * 3).unwrap();
        assert!(s.is_finite() && s >= 0.0, "{w}x{h}: {s}");
    }
}

#[test]
fn strided_equals_packed() {
    let (w, h) = (97, 61);
    let a = lcg_image(w, h, 11);
    let b = perturb(&a, 12);
    let packed = run(&a, &b, w, h, w * 3).unwrap();
    let (pa, stride) = padded(&a, w, h, 13);
    let (pb, _) = padded(&b, w, h, 13);
    assert_eq!(
        run(&pa, &pb, w, h, stride).unwrap().to_bits(),
        packed.to_bits()
    );
}

#[test]
fn heavier_distortion_scores_higher() {
    let (w, h) = (256, 256);
    let a = lcg_image(w, h, 21);
    let light = perturb(&a, 22);
    let heavy: Vec<u8> = light
        .iter()
        .zip(&a)
        .map(|(&l, &o)| if (l ^ o) & 8 == 0 { l } else { 255 - l })
        .collect();
    assert!(run(&a, &light, w, h, w * 3).unwrap() < run(&a, &heavy, w, h, w * 3).unwrap());
}

#[test]
fn errors_are_reported() {
    let a = [0u8; 12];
    assert!(run(&a, &a, 0, 4, 12).is_err());
    assert!(run(&a, &a, 2, 2, 5).is_err());
    assert!(run(&a[..11], &a, 2, 2, 6).is_err());
}

#[test]
fn available_tiers_match_scalar() {
    let (w, h) = (300, 2 * BAND_ROWS * 2 + 11);
    let a = lcg_image(w, h, 31);
    let b = perturb(&a, 32);
    let s = source(&a, &b, w, h, w * 3);
    let oh = s.h.div_ceil(s.m);
    let row_len = PLANES * s.pitch;
    let p = Params::default();

    let mut base = vec![0.0f64; (oh + 2) * row_len];
    for (i, chunk) in base[row_len..(oh + 1) * row_len]
        .chunks_mut(BAND_ROWS * row_len)
        .enumerate()
    {
        prepare_band_scalar(archmage::ScalarToken, &s, i * BAND_ROWS, chunk);
    }
    let planes = Planes {
        data: base.clone(),
        width: s.ow,
        height: oh,
        pitch: s.pitch,
    };
    let (n, mut gs, mut cs) = (planes.width * planes.height, vec![], vec![]);
    gs.resize(n, 0.0);
    cs.resize(n, 0.0);
    for (i, (q, c)) in gs
        .chunks_mut(BAND_ROWS * planes.width)
        .zip(cs.chunks_mut(BAND_ROWS * planes.width))
        .enumerate()
    {
        maps_band_scalar(archmage::ScalarToken, &planes, i * BAND_ROWS, &p, q, c);
    }

    macro_rules! check {
        ($label:expr, $prep:ident, $maps:ident, $token:expr) => {{
            let mut got = vec![0.0f64; (oh + 2) * row_len];
            for (i, chunk) in got[row_len..(oh + 1) * row_len]
                .chunks_mut(BAND_ROWS * row_len)
                .enumerate()
            {
                $prep($token, &s, i * BAND_ROWS, chunk);
            }
            assert_bits(&got, &base, concat!($label, " planes"));
            let pl = Planes {
                data: got,
                width: s.ow,
                height: oh,
                pitch: s.pitch,
            };
            let (mut g2, mut c2) = (vec![0.0; n], vec![0.0; n]);
            for (i, (q, c)) in g2
                .chunks_mut(BAND_ROWS * pl.width)
                .zip(c2.chunks_mut(BAND_ROWS * pl.width))
                .enumerate()
            {
                $maps($token, &pl, i * BAND_ROWS, &p, q, c);
            }
            assert_bits(&g2, &gs, concat!($label, " gcs"));
            assert_bits(&c2, &cs, concat!($label, " cs"));
        }};
    }
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(t) = X64V3Token::summon() {
            check!("v3", prepare_band_v3, maps_band_v3, t);
        }
        #[cfg(feature = "avx512")]
        if let Some(t) = X64V4Token::summon() {
            check!("v4", prepare_band_v4, maps_band_v4, t);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(t) = NeonToken::summon() {
            check!("neon", prepare_band_neon, maps_band_neon, t);
        }
    }
    // The production dispatch equals the scalar tier as well.
    let prod = prepare(&s);
    assert_bits(&prod.data, &base, "production planes");
    let (pg, pc) = maps(&prod, &p);
    assert_bits(&pg, &gs, "production gcs");
    assert_bits(&pc, &cs, "production cs");
}

#[cfg(feature = "parallel")]
#[test]
fn thread_count_does_not_change_the_score() {
    let (w, h) = (700, 530);
    let a = lcg_image(w, h, 51);
    let b = perturb(&a, 52);
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| run(&a, &b, w, h, w * 3).unwrap());
    let many = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap()
        .install(|| run(&a, &b, w, h, w * 3).unwrap());
    assert_eq!(one.to_bits(), many.to_bits());
}

/// The 116-pair gate against the authors' reference scores.
///
/// The caller decides, via environment variables that the justfile and CI
/// pass explicitly: `GMSD_MDSI_GATE=require` runs the gate (and fails if
/// `GMSD_MDSI_TARGETS` does not name a readable target table);
/// `GMSD_MDSI_GATE=skip` skips it visibly; anything else, including unset,
/// fails. The table has columns `id width height ref_rgb dist_rgb
/// author_mdsi` (raw RGB8 files; relative paths resolve against the
/// table's directory).
#[cfg(feature = "std")]
#[test]
fn author_score_gate() {
    let mode = std::env::var("GMSD_MDSI_GATE").unwrap_or_default();
    match mode.as_str() {
        "skip" => {
            std::eprintln!("author_score_gate: SKIPPED by GMSD_MDSI_GATE=skip");
            return;
        }
        "require" => {}
        other => panic!(
            "GMSD_MDSI_GATE must be `require` or `skip` (got {other:?}); \
             the caller decides whether the 116-pair author-score gate runs"
        ),
    }
    let table = std::env::var("GMSD_MDSI_TARGETS")
        .expect("GMSD_MDSI_GATE=require needs GMSD_MDSI_TARGETS=<path to target.tsv>");
    let table = std::path::PathBuf::from(table);
    let text = std::fs::read_to_string(&table)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", table.display()));
    let base = table.parent().unwrap().to_path_buf();
    let (mut pairs, mut max_rel, mut max_abs) = (0usize, 0.0f64, 0.0f64);
    let mut wrong_failures = 0usize;
    let wrong = Params {
        c3: 5500.0,
        ..Params::default()
    };
    for line in text.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f.len(), 6, "bad row: {line}");
        let (w, h): (usize, usize) = (f[1].parse().unwrap(), f[2].parse().unwrap());
        let author: f64 = f[5].parse().unwrap();
        let r = std::fs::read(base.join(f[3])).unwrap();
        let d = std::fs::read(base.join(f[4])).unwrap();
        let ours = run(&r, &d, w, h, w * 3).unwrap();
        let abs = (ours - author).abs();
        let rel = if author == 0.0 {
            abs
        } else {
            abs / author.abs()
        };
        assert!(
            rel <= 1e-9,
            "{}: ours {ours:e} author {author:e} rel {rel:e}",
            f[0]
        );
        max_rel = max_rel.max(rel);
        max_abs = max_abs.max(abs);
        let (bad, _, _) = score_with(&r, &d, w, h, w * 3, &wrong).unwrap();
        let bad_rel = if author == 0.0 {
            (bad - author).abs()
        } else {
            (bad - author).abs() / author.abs()
        };
        wrong_failures += usize::from(bad_rel > 1e-9);
        pairs += 1;
    }
    std::eprintln!(
        "author_score_gate: {pairs} pairs, max abs {max_abs:e}, max rel {max_rel:e}, wrong constant fails {wrong_failures}/{pairs}"
    );
    assert_eq!(pairs, 116, "the gate table must hold all 116 pairs");
    assert!(
        wrong_failures >= 100,
        "negative control must fail most pairs"
    );
}
