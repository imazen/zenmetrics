//! End-to-end gate on **real** pixels: a real 256×256 photograph and real
//! JPEG re-compressions of it, scored through the whole HDR-VDP-2.2 pipeline.
//!
//! The unit tests inside the crate use synthetic gratings, which can flatter a
//! metric — a synthetic distortion is exactly the shape the model expects.
//! This test uses `zenmetrics-corpus`' actual JPEG ladder, so the distortion is
//! real blocking, ringing and chroma loss.
//!
//! Images are reduced to 128×128 (through this crate's own MATLAB-compatible
//! `imresize`, which is exercised by the reduction too) so the debug-profile
//! runtime stays reasonable in CI. The full 256×256 six-rung ladder lives in
//! `examples/score_corpus_ladder.rs`, and its measured output is recorded in
//! `benchmarks/hdrvdp2_corpus_ladder_2026-08-28.tsv`.
//!
//! **This is a monotonicity gate, not a validation of the numbers.** Whether
//! this port reproduces the reference implementation's `Q_MOS` is settled by
//! the UPIQ SROCC measurement in zenmetrics#50 chunk 4, which is not this.

#![forbid(unsafe_code)]

use hdrvdp::resize::imresize;
use hdrvdp::{ColorEncoding, Params, hdrvdp, pix_per_deg};

const SIDE: usize = 128;
const HDR_PEAK_NITS: f64 = 1000.0;

fn srgb_to_linear(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Decode an image to interleaved RGB in `[0,1]`, reduced to `SIDE × SIDE`.
fn load(path: &std::path::Path) -> Vec<f64> {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("decoding {}: {e}", path.display()))
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let raw = img.into_raw();
    // Reduce each channel separately, then re-interleave.
    let mut out = vec![0.0; SIDE * SIDE * 3];
    for c in 0..3 {
        let plane: Vec<f64> = (0..w * h).map(|i| raw[i * 3 + c] as f64 / 255.0).collect();
        let small = imresize(&plane, w, h, SIDE, SIDE);
        for (i, v) in small.into_iter().enumerate() {
            out[i * 3 + c] = v.clamp(0.0, 1.0);
        }
    }
    out
}

#[test]
fn quality_rises_monotonically_with_jpeg_quality_on_real_content() {
    let par = Params::new(pix_per_deg(24.0, [1920.0, 1200.0], 0.5));
    let reference = load(&zenmetrics_corpus::source_png());
    let ref_nits: Vec<f64> = reference
        .iter()
        .map(|v| srgb_to_linear(*v) * HDR_PEAK_NITS)
        .collect();

    let mut prev_sdr = f64::NEG_INFINITY;
    let mut prev_hdr = f64::NEG_INFINITY;
    let mut prev_cmax = f64::INFINITY;

    for q in [1u32, 20, 90] {
        let test = load(&zenmetrics_corpus::jpeg_at_quality(q));

        let sdr = hdrvdp(
            &test,
            &reference,
            SIDE,
            SIDE,
            ColorEncoding::SrgbDisplay,
            &par,
        )
        .unwrap_or_else(|e| panic!("q{q} sdr: {e}"));

        let test_nits: Vec<f64> = test
            .iter()
            .map(|v| srgb_to_linear(*v) * HDR_PEAK_NITS)
            .collect();
        let hdr = hdrvdp(
            &test_nits,
            &ref_nits,
            SIDE,
            SIDE,
            ColorEncoding::RgbBt709,
            &par,
        )
        .unwrap_or_else(|e| panic!("q{q} hdr: {e}"));

        // Every reported number must be finite and in range — a metric that
        // NaNs on real content is worse than one that scores badly.
        for (name, v) in [("sdr", &sdr), ("hdr", &hdr)] {
            assert!(
                v.q_mos.is_finite() && (0.0..=100.0).contains(&v.q_mos),
                "q{q} {name}: Q_MOS = {}",
                v.q_mos
            );
            assert!(v.q.is_finite() && v.q < 0.0, "q{q} {name}: Q = {}", v.q);
            assert!(
                v.p_map
                    .iter()
                    .all(|p| p.is_finite() && (0.0..=1.0).contains(p)),
                "q{q} {name}: P_map left the unit interval"
            );
            assert_eq!(v.p_map.len(), SIDE * SIDE);
            assert!(
                !v.input_looks_relative,
                "q{q} {name}: feed flagged relative"
            );
        }

        assert!(
            sdr.q_mos > prev_sdr,
            "SDR Q_MOS must rise with JPEG quality: {prev_sdr} then {} at q{q}",
            sdr.q_mos
        );
        assert!(
            hdr.q_mos > prev_hdr,
            "HDR Q_MOS must rise with JPEG quality: {prev_hdr} then {} at q{q}",
            hdr.q_mos
        );
        assert!(
            sdr.c_max < prev_cmax,
            "difference magnitude must fall with JPEG quality: {prev_cmax} then {} at q{q}",
            sdr.c_max
        );

        prev_sdr = sdr.q_mos;
        prev_hdr = hdr.q_mos;
        prev_cmax = sdr.c_max;
    }

    // The worst rung must actually be scored as bad, and the best as good —
    // otherwise "monotone" could be satisfied by a metric that barely moves.
    assert!(prev_sdr > 90.0, "q90 should score well, got {prev_sdr}");
}

#[test]
fn an_identical_real_image_pair_is_invisible() {
    let par = Params::new(pix_per_deg(24.0, [1920.0, 1200.0], 0.5));
    let reference = load(&zenmetrics_corpus::source_png());
    let r = hdrvdp(
        &reference,
        &reference,
        SIDE,
        SIDE,
        ColorEncoding::SrgbDisplay,
        &par,
    )
    .unwrap();
    assert!(
        r.p_det < 1e-9,
        "P_det = {} on identical real pixels",
        r.p_det
    );
    assert_eq!(r.visible_fraction(), 0.0);
    assert!(r.q_mos > 99.99, "Q_MOS = {}", r.q_mos);
}
