//! Composed host-scalar pipeline integration test on the
//! zenmetrics-corpus. Each (ref, dist) pair runs through the full
//! still-image cvvdp chain (color → pyramid → CSF → masking →
//! pooling → JOD) at the standard_4k display config.
//!
//! This test does NOT yet assert tight pycvvdp parity. Observed
//! shadow on the v1 corpus (standard_4k, l_bkg=100, CH_GAIN +
//! PU blur both on):
//!
//! ```text
//!   q    pycvvdp manifest   shadow scalar
//!   1    7.65               ~6.3
//!   5    8.89               ~7.0
//!   20   9.71               ~8.5
//!   45   9.83               ~8.4
//!   70   9.89               ~8.8
//!   90   9.99               ~9.0
//! ```
//!
//! Remaining gap (~1.0 JOD at high q) attributed to:
//! - **Global L_bkg approximation**: cvvdp uses per-pixel L_bkg
//!   from the reference's achromatic Gaussian-pyramid level
//!   matching each band.
//!
//! Both produce a JOD that broadly increases with q, but the
//! shadow's absolute scale is ~2-3 JOD lower. The non-monotone
//! q20→q45 dip reflects JPEG's near-flat RD curve in that range
//! amplified by the simplifications.
//!
//! The assertions enforce:
//!
//! - JOD is finite and in `[0, 10]`.
//! - `JOD(q90) - JOD(q1) > 1` (overall trend captures the q range).
//! - At least 4 of 5 adjacent (q_lo, q_hi) pairs are monotonic.
//!
//! Tight parity is the work that closes the documented gaps.

use cvvdp_gpu::host_scalar::predict_jod_still_3ch;
use cvvdp_gpu::params::{DisplayGeometry, DisplayModel};
use image::ImageReader;
use std::path::PathBuf;

fn load_rgb_bytes(path: &PathBuf, w: u32, h: u32) -> Vec<u8> {
    let img = ImageReader::open(path)
        .unwrap_or_else(|e| panic!("open {path:?}: {e}"))
        .decode()
        .unwrap_or_else(|e| panic!("decode {path:?}: {e}"))
        .to_rgb8();
    assert_eq!(img.width(), w);
    assert_eq!(img.height(), h);
    img.into_raw()
}

#[test]
fn shadow_jod_runs_and_is_monotonic_on_corpus() {
    let corpus = zenmetrics_corpus::corpus_dir();
    let (w, h) = (256u32, 256u32);

    let display = DisplayModel::STANDARD_4K;
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let l_bkg = display.y_peak / 2.0;

    let ref_bytes = load_rgb_bytes(&zenmetrics_corpus::source_png(), w, h);

    let qs = [1u32, 5, 20, 45, 70, 90];
    let mut jods = Vec::with_capacity(qs.len());
    for q in qs {
        let dist_bytes = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(q), w, h);
        let jod = predict_jod_still_3ch(
            &ref_bytes,
            &dist_bytes,
            w as usize,
            h as usize,
            display,
            ppd,
            l_bkg,
        );
        eprintln!("q={q:>2}: shadow JOD = {jod:.4}");
        jods.push((q, jod));
        assert!(jod.is_finite(), "q={q}: JOD = {jod} (not finite)");
        assert!(
            (0.0..=10.0).contains(&jod),
            "q={q}: JOD = {jod} out of [0, 10]"
        );
    }

    // Allow at most 1 of 5 adjacent pairs to flip — JPEG's RD curve
    // has near-flat regions in the q20-q45 range that the
    // simplifications amplify.
    let mut flips = 0;
    for w in jods.windows(2) {
        let (q_lo, j_lo) = w[0];
        let (q_hi, j_hi) = w[1];
        if j_hi + 1e-3 < j_lo {
            flips += 1;
            eprintln!("flip at q{q_lo}→q{q_hi}: {j_lo:.4} → {j_hi:.4}");
        }
    }
    assert!(
        flips <= 1,
        "expected ≤1 non-monotone flip in q-sweep; got {flips}"
    );

    let &(_, jod_q1) = jods.first().unwrap();
    let &(_, jod_q90) = jods.last().unwrap();
    assert!(
        jod_q90 - jod_q1 > 1.0,
        "expected JOD(q90) - JOD(q1) > 1.0 (broad trend captured); \
         got q1={jod_q1:.3} q90={jod_q90:.3}"
    );
}
