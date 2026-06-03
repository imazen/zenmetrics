//! cvvdp's photometric `DisplayModel` (HDR peak luminance) threads through
//! the umbrella via `MetricParams::Cvvdp` — display-aware scoring is the whole
//! point of cvvdp, so a different display peak must yield a different JOD.
//! Proves the HDR display is reachable through `Metric` (no need to drop to
//! the per-crate `CvvdpBatchScorer`). CUDA-gated; NO GRACEFUL SKIPS.
#![cfg(all(feature = "cuda", feature = "cvvdp"))]

use zenmetrics_api::cvvdp::params::DisplayModel;
use zenmetrics_api::{Backend, Metric, MetricKind, MetricParams};

#[test]
fn cvvdp_display_model_threads_through_umbrella() {
    let (w, h) = (256u32, 256u32);
    let n = (w * h * 3) as usize;
    let r: Vec<u8> = (0..n)
        .map(|i| ((i * 2654435761usize) >> 13) as u8)
        .collect();
    let d: Vec<u8> = r
        .iter()
        .enumerate()
        .map(|(i, b)| b.wrapping_add(((i * 40503) & 0x1f) as u8))
        .collect();

    // Default = SDR reference display (STANDARD_4K, y_peak 200).
    let sdr_jod = {
        let mut m = Metric::new(
            MetricKind::Cvvdp,
            Backend::Cuda,
            w,
            h,
            MetricParams::default_for(MetricKind::Cvvdp),
        )
        .expect("Metric::new cvvdp default");
        m.compute_srgb_u8(&r, &d).expect("compute sdr").value
    };

    // HDR display: 1000 cd/m² peak via the ergonomic constructor.
    let hdr_jod = {
        let hdr = DisplayModel {
            y_peak: 1000.0,
            ..DisplayModel::STANDARD_HDR_LINEAR
        };
        let mut m = Metric::new(
            MetricKind::Cvvdp,
            Backend::Cuda,
            w,
            h,
            MetricParams::cvvdp_with_display(hdr),
        )
        .expect("Metric::new cvvdp HDR display");
        m.compute_srgb_u8(&r, &d).expect("compute hdr").value
    };

    assert!(
        sdr_jod.is_finite() && hdr_jod.is_finite(),
        "sdr={sdr_jod} hdr={hdr_jod}"
    );
    assert!(
        (sdr_jod - hdr_jod).abs() > 1e-3,
        "a different display peak must change the JOD (display-aware scoring): \
         sdr(200nit)={sdr_jod} hdr(1000nit)={hdr_jod}"
    );
}
