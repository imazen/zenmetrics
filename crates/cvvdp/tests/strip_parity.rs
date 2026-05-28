//! D4 of phase9zd / task #124: comprehensive bit-identical parity
//! between the strip pool walker and Full mode across 18 synthetic
//! seeds × 3 sizes × 5 h_body values = 270 cells per mode.
//!
//! ## Why bit-identical
//!
//! The cvvdp spatial Minkowski pool (`lp_norm_mean(d, p=2)`) is a
//! reduction `Σ safe_pow_lp(v, p)` over the band's `d` array. The
//! reduction is associative under f32-add ordering, and the strip
//! walker dispatches strips in **deterministic row order** (`s = 0,
//! 1, 2, ...`), so the per-strip accumulator sees the same `acc +=
//! x_i` sequence as a single-pass `lp_norm_mean` call. There is **no
//! rounding drift** — the bits in the final f32 score are equal to
//! the Full-mode bits, not "within ε".
//!
//! This is why the gate is `assert_eq!(.to_bits(), .to_bits())` not
//! `(strip - full).abs() < 1e-4`. Any drift here would mean either
//! the strips dispatch out of order, or some non-associative
//! operation has slipped into the pool path (a real bug).
//!
//! ## Image sources
//!
//! Synthetic XorShift pairs cover every (seed, size) combination —
//! we don't depend on bundled fixtures (the only ones in-tree are
//! 64² and 256² noisy/identical, way too small for the parity
//! sweep). The walker's correctness has nothing to do with image
//! content; it has everything to do with shape coverage. 18 distinct
//! seeds → 18 distinct distortion patterns, sufficient for the
//! row-order-dispatch invariant.
//!
//! ## Mode E (warm reference) parity
//!
//! The Mode E path runs the dist-side cold pyramid build with the
//! warm cached ref, then enters the same band-fold (and therefore
//! the same strip pool walker). So the warm-ref + strip path is
//! also bit-identical to the warm-ref + Full path. This is asserted
//! by the `warm_ref_strip_matches_full_*` tests.

use cvvdp::{Cvvdp, CvvdpParams};

/// Build a synth sRGB pair via XorShift64.
fn synth_pair(w: u32, h: u32, seed: u64) -> (Vec<u8>, Vec<u8>) {
    let n = (w as usize) * (h as usize) * 3;
    let mut r = vec![0u8; n];
    let mut d = vec![0u8; n];
    let mut s_r = seed;
    let mut s_d = seed.wrapping_mul(2_654_435_769);
    for i in 0..n {
        s_r ^= s_r << 13;
        s_r ^= s_r >> 7;
        s_r ^= s_r << 17;
        r[i] = (s_r & 0xFF) as u8;
        s_d ^= s_d << 13;
        s_d ^= s_d >> 7;
        s_d ^= s_d << 17;
        let mixed = (r[i] as u16) * 230 + ((s_d as u8) as u16) * 25;
        d[i] = ((mixed / 256) as u8).min(255);
    }
    (r, d)
}

/// 18 distinct seeds × 3 sizes × 5 h_body values = 270 cells.
/// Bigger image sizes are gated under the `cvvdp-strip-parity-big`
/// feature so the default `cargo test` run stays under ~30s; CI runs
/// the full grid via `--features cvvdp-strip-parity-big`.
const SEEDS: [u64; 18] = [
    0xfeed_beef_dead_c0fe,
    0x1234_5678_9abc_def0,
    0xabcd_ef01_2345_6789,
    0x5555_aaaa_5555_aaaa,
    0xa5a5_5a5a_a5a5_5a5a,
    0x0fff_f000_0fff_f000,
    0xdead_beef_cafe_babe,
    0x1111_2222_3333_4444,
    0x9999_8888_7777_6666,
    0xc0ff_eeee_cafe_d00d,
    0xfade_face_b00b_5005,
    0x0badf00d_dead_beef,
    0x6996_9669_6996_9669,
    0x3141_5926_5358_9793,
    0x2718_2818_2845_9045,
    0x1414_2135_6237_3095,
    0x4815_1623_4216_5223,
    0xc0c0_aaaa_5555_3333,
];

/// 5 h_body values covering the validator's accepted range. The h_body
/// is in scale-0 rows; bands at deeper levels see `(h_body >> k).max(1)`
/// row strips per the GPU's per-band partitioning rule.
const H_BODY_VALUES: [u32; 5] = [32, 64, 128, 256, 512];

/// Default-sized parity grid. 1024² is the smallest size at which
/// h_body=512 still meaningfully partitions level 0 into 2 strips
/// (1024 / 512 = 2). Larger sizes are gated behind feature flags.
const SIZES_DEFAULT: [(u32, u32); 1] = [(512, 512)];

/// Big-grid sizes — gated behind `cvvdp-strip-parity-big`. 18 seeds ×
/// 3 sizes × 5 h_body = 270 cells per cold mode (and again for warm).
/// At 4096² each cell takes ~5s sequentially; 270 cells × 5s = 22 min.
/// Gate keeps default `cargo test` runs fast.
#[cfg(feature = "cvvdp-strip-parity-big")]
const SIZES_BIG: [(u32, u32); 3] = [(1024, 1024), (2048, 2048), (4096, 4096)];

/// Bit-identical parity gate: every (seed, size, h_body) cell asserts
/// `score_strip(.., h_body).to_bits() == score(..).to_bits()`.
fn check_cell(seed: u64, w: u32, h: u32, h_body: u32) {
    let (r, d) = synth_pair(w, h, seed);
    let mut s_full = Cvvdp::new(w, h, CvvdpParams::default()).unwrap();
    let mut s_strip = Cvvdp::new(w, h, CvvdpParams::default()).unwrap();
    let full = s_full.score(&r, &d).unwrap();
    let strip = s_strip.score_strip(&r, &d, h_body).unwrap();
    assert_eq!(
        full.to_bits(),
        strip.to_bits(),
        "strip walker drift at seed=0x{seed:x}, size={w}×{h}, h_body={h_body}: \
         full={full:e} (0x{:08x}), strip={strip:e} (0x{:08x})",
        full.to_bits(),
        strip.to_bits(),
    );
}

fn check_warm_cell(seed: u64, w: u32, h: u32, h_body: u32) {
    let (r, d) = synth_pair(w, h, seed);
    let mut s_full = Cvvdp::new(w, h, CvvdpParams::default()).unwrap();
    let mut s_strip = Cvvdp::new(w, h, CvvdpParams::default()).unwrap();
    s_full.warm_reference(&r).unwrap();
    s_strip.warm_reference(&r).unwrap();
    let full = s_full.score_with_warm_ref(&d).unwrap();
    let strip = s_strip.score_with_warm_ref_strip(&d, h_body).unwrap();
    assert_eq!(
        full.to_bits(),
        strip.to_bits(),
        "warm-ref strip drift at seed=0x{seed:x}, size={w}×{h}, h_body={h_body}: \
         full={full:e} (0x{:08x}), strip={strip:e} (0x{:08x})",
        full.to_bits(),
        strip.to_bits(),
    );
}

/// 18 seeds × 1 size × 5 h_body = 90 cells. Always runs.
#[test]
fn strip_parity_default_grid_cold() {
    for &seed in &SEEDS {
        for &(w, h) in &SIZES_DEFAULT {
            for &hb in &H_BODY_VALUES {
                check_cell(seed, w, h, hb);
            }
        }
    }
}

/// Same as `strip_parity_default_grid_cold` but for the warm-ref path.
#[test]
fn strip_parity_default_grid_warm() {
    for &seed in &SEEDS {
        for &(w, h) in &SIZES_DEFAULT {
            for &hb in &H_BODY_VALUES {
                check_warm_cell(seed, w, h, hb);
            }
        }
    }
}

/// 18 × 3 × 5 = 270 cells. Gated behind `cvvdp-strip-parity-big`
/// because 4096² parity takes ~5s per cell sequentially.
#[cfg(feature = "cvvdp-strip-parity-big")]
#[test]
fn strip_parity_big_grid_cold() {
    for &seed in &SEEDS {
        for &(w, h) in &SIZES_BIG {
            for &hb in &H_BODY_VALUES {
                check_cell(seed, w, h, hb);
            }
        }
    }
}

#[cfg(feature = "cvvdp-strip-parity-big")]
#[test]
fn strip_parity_big_grid_warm() {
    for &seed in &SEEDS {
        for &(w, h) in &SIZES_BIG {
            for &hb in &H_BODY_VALUES {
                check_warm_cell(seed, w, h, hb);
            }
        }
    }
}

/// Sanity: the dispatch counter increments at sizes large enough to
/// partition. Asserts the walker actually walks (not silently
/// degenerating to a single-strip dispatch). 512² × h_body=128 →
/// level 0 (512 rows) partitions into 4 strips × 3 channels = 12.
#[test]
fn strip_walker_dispatches_n_strips_at_default_size() {
    let (r, d) = synth_pair(512, 512, 0xfeed);
    let mut s = Cvvdp::new(512, 512, CvvdpParams::default()).unwrap();
    let _ = s.score_strip(&r, &d, 128).unwrap();
    let count = s.strip_dispatch_counter();
    // Level 0: 512/128 = 4 strips × 3 ch = 12. Plus deeper levels
    // contributing more. Minimum guaranteed: 12.
    assert!(count >= 12, "expected >= 12 dispatches at 512² h_body=128; got {count}");
}

/// All 5 h_body values must yield identical Full-vs-strip JOD at the
/// same image — proves the partitioning rule doesn't drift even when
/// the strip body shifts dramatically across configurations.
#[test]
fn strip_jod_invariant_across_h_body_at_seed_0() {
    let (r, d) = synth_pair(512, 512, SEEDS[0]);
    let mut s_full = Cvvdp::new(512, 512, CvvdpParams::default()).unwrap();
    let full = s_full.score(&r, &d).unwrap();
    for &hb in &H_BODY_VALUES {
        let mut s_strip = Cvvdp::new(512, 512, CvvdpParams::default()).unwrap();
        let strip = s_strip.score_strip(&r, &d, hb).unwrap();
        assert_eq!(
            full.to_bits(),
            strip.to_bits(),
            "strip JOD drift at h_body={hb}: full={full:e}, strip={strip:e}"
        );
    }
}
