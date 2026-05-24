//! Parity vs host_scalar on real corpus images.
//!
//! Loads the 256² PNG reference + 6 JPEG variants from
//! `zenmetrics_corpus`, scores both with cvvdp-cpu and with
//! cvvdp-gpu's host_scalar reference, and locks the JOD difference
//! to ≤ 1e-4. This is the closest thing we have to a real-image
//! goldens test that doesn't require fetching the pycvvdp R2
//! manifest.

use cvvdp_cpu::{Cvvdp, CvvdpParams, DisplayGeometry};
use cvvdp_gpu::host_scalar::predict_jod_still_3ch;
use cvvdp_gpu::params::DisplayModel;

fn load_rgb8(path: &std::path::Path, w: u32, h: u32) -> Vec<u8> {
    let img = image::open(path).unwrap_or_else(|e| panic!("load {path:?}: {e}"));
    let rgb = img.to_rgb8();
    assert_eq!(rgb.width(), w);
    assert_eq!(rgb.height(), h);
    rgb.into_raw()
}

#[test]
fn matches_host_scalar_on_corpus_q1_q90() {
    // The corpus references are 256×256.
    let w: u32 = 256;
    let h: u32 = 256;
    let display = DisplayModel::STANDARD_4K;
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();

    let src = load_rgb8(&zenmetrics_corpus::source_png(), w, h);
    let qualities = [1u32, 5, 20, 45, 70, 90];

    let mut cv = Cvvdp::new(w, h, CvvdpParams::default()).unwrap();
    cv.warm_reference(&src).unwrap();

    for &q in &qualities {
        let dist = load_rgb8(&zenmetrics_corpus::jpeg_at_quality(q), w, h);
        let cpu_jod = cv.score_with_warm_ref(&dist).unwrap();
        let ref_jod = predict_jod_still_3ch(&src, &dist, w as usize, h as usize, display, ppd);
        let diff = (cpu_jod - ref_jod).abs();
        eprintln!("q={q}: cpu={cpu_jod:.6}  host_scalar={ref_jod:.6}  diff={diff:.6}");
        assert!(
            diff < 1e-4,
            "q={q}: cpu={cpu_jod}, host_scalar={ref_jod}, diff={diff}"
        );
    }
}

#[test]
fn diffmap_correlates_with_quality_on_corpus() {
    // Higher quality → smaller diffmap sum.
    let w: u32 = 256;
    let h: u32 = 256;
    let src = load_rgb8(&zenmetrics_corpus::source_png(), w, h);

    let mut cv = Cvvdp::new(w, h, CvvdpParams::default()).unwrap();
    cv.warm_reference(&src).unwrap();

    let qualities = [1u32, 20, 70, 90];
    let mut prev_sum = f64::INFINITY;
    let mut prev_jod = f32::NEG_INFINITY;
    for &q in &qualities {
        let dist = load_rgb8(&zenmetrics_corpus::jpeg_at_quality(q), w, h);
        let mut dmap = Vec::new();
        let jod = cv.score_with_warm_ref_diffmap(&dist, &mut dmap).unwrap();
        let sum: f64 = dmap.iter().map(|v| *v as f64).sum();
        eprintln!("q={q}: jod={jod:.4} diffmap_sum={sum:.2}");
        assert!(
            sum <= prev_sum + 1e-3,
            "monotone decrease in diff sum across quality: q={q} sum={sum} prev={prev_sum}"
        );
        assert!(
            jod >= prev_jod - 1e-3,
            "monotone increase in JOD across quality: q={q} jod={jod} prev={prev_jod}"
        );
        prev_sum = sum;
        prev_jod = jod;
    }
}
