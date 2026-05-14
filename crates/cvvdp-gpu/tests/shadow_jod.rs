//! Composed host-scalar pipeline integration test on the
//! zenmetrics-corpus. Each (ref, dist) pair runs through the full
//! still-image cvvdp chain (color → pyramid → CSF → masking →
//! pooling → JOD) at the standard_4k display config.
//!
//! After tick 25 (Weber pyramid + band_mul = 2.0 + baseband bypass)
//! the host-scalar shadow matches the v1 R2 manifest values
//! (standard_4k display) within ~0.01 JOD across all 6 q-levels:
//!
//! ```text
//!   q    pycvvdp manifest   shadow scalar   |diff|
//!   1    7.6536             7.6476          0.006
//!   5    8.8889             8.8912          0.002
//!   20   9.7076             9.7089          0.001
//!   45   9.8273             9.8296          0.002
//!   70   9.8915             9.8945          0.003
//!   90   9.9930             9.9929          0.000
//! ```
//!
//! Test assertions:
//! - JOD is finite and in `[0, 10]`.
//! - Each shadow JOD is within 0.05 of the pycvvdp manifest value
//!   (loose tol to absorb f32 accumulation across the pipeline +
//!   any future minor refactor).
//! - JOD is monotonic non-decreasing across q.

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

    let ref_bytes = load_rgb_bytes(&zenmetrics_corpus::source_png(), w, h);

    // (q, pycvvdp_manifest_jod) captured 2026-05-14 from
    // build_goldens.py over the v1 corpus at standard_4k.
    let cases: &[(u32, f32)] = &[
        (1, 7.6536),
        (5, 8.8889),
        (20, 9.7076),
        (45, 9.8273),
        (70, 9.8915),
        (90, 9.9930),
    ];
    let mut jods = Vec::with_capacity(cases.len());
    for &(q, expected) in cases {
        let dist_bytes = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(q), w, h);
        let jod = predict_jod_still_3ch(
            &ref_bytes,
            &dist_bytes,
            w as usize,
            h as usize,
            display,
            ppd,
        );
        let diff = (jod - expected).abs();
        eprintln!(
            "q={q:>2}: shadow JOD = {jod:.4} (pycvvdp {expected:.4}, |diff| {diff:.4})"
        );
        assert!(jod.is_finite(), "q={q}: JOD = {jod} (not finite)");
        assert!(
            (0.0..=10.0).contains(&jod),
            "q={q}: JOD = {jod} out of [0, 10]"
        );
        assert!(
            diff < 0.05,
            "q={q}: shadow JOD {jod:.4} diverges from pycvvdp manifest {expected:.4} by {diff:.4} > 0.05"
        );
        jods.push((q, jod));
    }

    // Monotone non-decreasing across q.
    for win in jods.windows(2) {
        let (q_lo, j_lo) = win[0];
        let (q_hi, j_hi) = win[1];
        assert!(
            j_hi + 1e-3 >= j_lo,
            "non-monotone: q={q_lo} JOD={j_lo:.4} > q={q_hi} JOD={j_hi:.4}"
        );
    }
}
