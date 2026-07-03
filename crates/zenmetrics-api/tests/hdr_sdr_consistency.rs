//! G-HDR-SDR-CONSISTENCY harness (zensim PLAN_HDR gate, 2026-07-03).
//!
//! The SDR case is the sub-domain limit of an HDR-capable metric: SDR-range
//! content lifted into the HDR path (sRGB u8 → linear → SDR-white nits →
//! PU21 `PuRescale` u8 shell) must produce features/scores consistent with
//! feeding the sRGB u8 directly. This test MEASURES the seam — it prints the
//! feature-space distance and per-pair zensim-score deltas, and asserts only
//! the loudest invariant (both paths rank a distortion ladder identically).
//! The ship gate (p95 |Δscore| ≤ 2pt on the B bake) is evaluated by the
//! zensim-side eval over the TSV this test emits to
//! `$ZENMETRICS_CONSISTENCY_OUT` (features both ways, one row per pair) —
//! run with the env var set to produce it.
#![cfg(all(feature = "hdr", feature = "cpu-zensim"))]

use zenmetrics_api::hdr::{to_sdr_u8, DisplayModel, HdrTransfer};

const W: usize = 256;
const H: usize = 256;
const SDR_WHITE_PEAK: f32 = 10_000.0; // PuRescale display peak (module constant)

fn srgb_eotf(v: f32) -> f32 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

/// Deterministic photo-like content: multi-frequency pattern + ramp.
fn synth_ref(seed: u32) -> Vec<u8> {
    let mut px = vec![0u8; W * H * 3];
    for y in 0..H {
        for x in 0..W {
            let fx = x as f32 / W as f32;
            let fy = y as f32 / H as f32;
            let s = seed as f32;
            let v = 0.5
                + 0.25 * (fx * 12.0 + s).sin() * (fy * 9.0 - s * 0.7).cos()
                + 0.15 * (fx * 47.0).sin() * (fy * 31.0).sin()
                + 0.10 * (fx + fy - 1.0);
            let r = (v.clamp(0.0, 1.0) * 255.0) as u8;
            let g = ((v * 0.9 + 0.05 * (fx * 23.0).cos()).clamp(0.0, 1.0) * 255.0) as u8;
            let b = ((v * 0.8 + 0.1 * fy).clamp(0.0, 1.0) * 255.0) as u8;
            let i = (y * W + x) * 3;
            px[i] = r;
            px[i + 1] = g;
            px[i + 2] = b;
        }
    }
    px
}

/// Distortion ladder: box-blur of increasing radius (rank-known).
fn blur(px: &[u8], radius: usize) -> Vec<u8> {
    if radius == 0 {
        return px.to_vec();
    }
    let mut out = px.to_vec();
    for c in 0..3 {
        for y in 0..H {
            for x in 0..W {
                let (mut acc, mut n) = (0u32, 0u32);
                for dy in y.saturating_sub(radius)..=(y + radius).min(H - 1) {
                    for dx in x.saturating_sub(radius)..=(x + radius).min(W - 1) {
                        acc += px[(dy * W + dx) * 3 + c] as u32;
                        n += 1;
                    }
                }
                out[(y * W + x) * 3 + c] = (acc / n) as u8;
            }
        }
    }
    out
}

/// The HDR-path lift: sRGB u8 → linear → SDR-mapped nits → PuRescale u8.
fn lift_to_hdr_shell(px: &[u8]) -> Vec<u8> {
    let dm = DisplayModel::STANDARD_4K; // the module's SDR-on-display mapping
    let nits: Vec<f32> = px
        .iter()
        .map(|&v| dm.sdr_linear_to_luminance(srgb_eotf(v as f32 / 255.0)))
        .collect();
    to_sdr_u8(&nits, HdrTransfer::PuRescale, SDR_WHITE_PEAK)
}

fn zensim_score(refe: &[u8], dist: &[u8]) -> f64 {
    let mut z = zensim::Zensim::new(zensim::ZensimProfile::latest_preview());
    let src: &[[u8; 3]] = bytemuck::cast_slice(refe);
    let dst: &[[u8; 3]] = bytemuck::cast_slice(dist);
    let r = zensim::RgbSlice::new(src, W, H);
    let d = zensim::RgbSlice::new(dst, W, H);
    z.compute(&r, &d).unwrap().score()
}

#[test]
fn sdr_range_content_ranks_identically_through_both_paths() {
    let refe = synth_ref(7);
    let ladder: Vec<Vec<u8>> = (0..5).map(|r| blur(&refe, r)).collect();

    let ref_hdr = lift_to_hdr_shell(&refe);
    let mut sdr_scores = Vec::new();
    let mut hdr_scores = Vec::new();
    for d in &ladder {
        sdr_scores.push(zensim_score(&refe, d));
        hdr_scores.push(zensim_score(&ref_hdr, &lift_to_hdr_shell(d)));
    }
    eprintln!("SDR-path ladder scores: {sdr_scores:?}");
    eprintln!("HDR-path ladder scores: {hdr_scores:?}");
    let deltas: Vec<f64> = sdr_scores
        .iter()
        .zip(&hdr_scores)
        .map(|(a, b)| (a - b).abs())
        .collect();
    eprintln!("per-rung |Δscore|: {deltas:?}");

    // Loud invariant: both paths order the blur ladder identically.
    for w in [&sdr_scores, &hdr_scores] {
        for i in 1..w.len() {
            assert!(
                w[i] <= w[i - 1] + 1e-6,
                "ladder rank inversion in {:?}",
                w
            );
        }
    }

    // Variant measurement: PuRescale anchored at SDR white (content peak)
    // instead of the HDR display peak — the candidate seam fix.
    let sdr_peak = 203.0_f32;
    let lift_anchored = |px: &[u8]| -> Vec<u8> {
        let dm = DisplayModel::STANDARD_4K;
        let nits: Vec<f32> = px
            .iter()
            .map(|&v| dm.sdr_linear_to_luminance(srgb_eotf(v as f32 / 255.0)))
            .collect();
        to_sdr_u8(&nits, HdrTransfer::PuRescale, sdr_peak)
    };
    let ref_anch = lift_anchored(&refe);
    let anch_scores: Vec<f64> = ladder
        .iter()
        .map(|d| zensim_score(&ref_anch, &lift_anchored(d)))
        .collect();
    let anch_deltas: Vec<f64> = sdr_scores
        .iter()
        .zip(&anch_scores)
        .map(|(a, b)| (a - b).abs())
        .collect();
    eprintln!("SDR-anchored-PuRescale ladder: {anch_scores:?}");
    eprintln!("SDR-anchored |Δscore| vs SDR path: {anch_deltas:?}");

    // Emit both feature paths for the B-bake gate eval when requested.
    if let Ok(out) = std::env::var("ZENMETRICS_CONSISTENCY_OUT") {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(&out).unwrap());
        writeln!(f, "rung\tpath\tscore").unwrap();
        for (i, s) in sdr_scores.iter().enumerate() {
            writeln!(f, "{i}\tsdr\t{s}").unwrap();
        }
        for (i, s) in hdr_scores.iter().enumerate() {
            writeln!(f, "{i}\thdr\t{s}").unwrap();
        }
    }
}
