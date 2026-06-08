//! The FOLD (Option A): the **base** `Metric::compute_pixels` /
//! `compute_pixels_multi` is itself descriptor-driven — an HDR slice is auto-fed
//! per `hdr::hdr_feeding` at the metric's `display_peak`, an `RGB8_SRGB` slice
//! takes the native path. No `HdrScorer` wrapper, no silent SDR collapse: the
//! umbrella entry every consumer calls does the right thing for HDR and SDR.
//! CUDA-gated; NO GRACEFUL SKIPS.
#![cfg(all(
    feature = "cuda",
    feature = "hdr",
    feature = "pixels",
    feature = "ssim2"
))]

use zenmetrics_api::hdr::{HDR_PEAK_NITS, HdrScorer};
use zenmetrics_api::{Backend, Metric, MetricKind, MetricParams};
use zenpixels::{PixelDescriptor, PixelSlice};

fn gradient_u8(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            out.push(((x.wrapping_add(seed)) & 0xff) as u8);
            out.push(((y.wrapping_add(seed * 3)) & 0xff) as u8);
            out.push(((x ^ y ^ seed) & 0xff) as u8);
        }
    }
    out
}

fn srgb_eotf(c: u8) -> f32 {
    let v = c as f32 / 255.0;
    if v <= 0.040_449_936 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_ne_bytes());
    }
    out
}

/// An interleaved-linear-f32 (`RGBF32_LINEAR`) `PixelSlice` over `b`. A `fn` (not
/// a closure) so the borrow lifetime threads through cleanly.
fn lin_slice(b: &[u8], w: u32, h: u32) -> PixelSlice<'_> {
    PixelSlice::new(
        b,
        w,
        h,
        (w * 3 * 4) as usize,
        PixelDescriptor::RGBF32_LINEAR,
    )
    .unwrap()
}

/// Base `Metric::compute_pixels` on an `RGB8_SRGB` slice is bit-identical to the
/// native `compute_srgb_u8` — SDR scores preserved through the umbrella entry,
/// without an `HdrScorer`.
#[test]
fn base_metric_compute_pixels_sdr_matches_native() {
    let (w, h) = (64_u32, 64_u32);
    let ref_u8 = gradient_u8(w, h, 0);
    let dis_u8 = gradient_u8(w, h, 9);
    let row = (w * 3) as usize;

    let mut m = Metric::new(
        MetricKind::Ssim2,
        Backend::Cuda,
        w,
        h,
        MetricParams::default_for(MetricKind::Ssim2),
    )
    .expect("ssim2");
    let native = m.compute_srgb_u8(&ref_u8, &dis_u8).expect("native").value;
    let via_pixels = m
        .compute_pixels(
            PixelSlice::new(&ref_u8, w, h, row, PixelDescriptor::RGB8_SRGB).unwrap(),
            PixelSlice::new(&dis_u8, w, h, row, PixelDescriptor::RGB8_SRGB).unwrap(),
        )
        .expect("pixels")
        .value;
    let rel = (via_pixels - native).abs() / native.abs().max(1e-6);
    assert!(
        rel < 1e-4,
        "base Metric::compute_pixels SDR ({via_pixels}) must match native compute_srgb_u8 ({native}); rel {rel}"
    );
}

/// Base `Metric` (ssim2 + an HDR `display_peak`, no display-model construction
/// needed for the SSIM-family) auto-feeds an HDR linear slice through
/// `compute_pixels_multi` — bit-identical to the validated `HdrScorer`. The
/// fold's core promise: the base entry does the HDR feeding, not a collapse.
#[test]
fn base_metric_compute_pixels_multi_hdr_matches_scorer() {
    let (w, h) = (64_u32, 64_u32);
    let ref_u8 = gradient_u8(w, h, 0);
    let dis_u8 = gradient_u8(w, h, 9);
    let ref_lin: Vec<f32> = ref_u8.iter().map(|&c| srgb_eotf(c)).collect();
    let dis_lin: Vec<f32> = dis_u8.iter().map(|&c| srgb_eotf(c)).collect();
    let ref_bytes = f32_bytes(&ref_lin);
    let dis_bytes = f32_bytes(&dis_lin);

    // Base Metric with the HDR peak set directly — no HdrScorer.
    let mut m = Metric::new(
        MetricKind::Ssim2,
        Backend::Cuda,
        w,
        h,
        MetricParams::default_for(MetricKind::Ssim2),
    )
    .expect("ssim2")
    .with_display_peak(HDR_PEAK_NITS);
    let via_metric = m
        .compute_pixels_multi(lin_slice(&ref_bytes, w, h), lin_slice(&dis_bytes, w, h))
        .expect("base metric hdr")
        .primary();

    // Validated reference: the HdrScorer (which now itself delegates here).
    let mut hs =
        HdrScorer::new(MetricKind::Ssim2, Backend::Cuda, w, h, HDR_PEAK_NITS).expect("scorer");
    let via_scorer = hs
        .compute_pixels_multi(lin_slice(&ref_bytes, w, h), lin_slice(&dis_bytes, w, h))
        .expect("scorer hdr")
        .primary();

    let rel = (via_metric - via_scorer).abs() / via_scorer.abs().max(1e-6);
    assert!(
        rel < 1e-4,
        "base Metric::compute_pixels_multi HDR ({via_metric}) must match HdrScorer ({via_scorer}); rel {rel}"
    );
}

/// The base entry honors `display_peak`: the same HDR slice scored at two
/// different peaks gives different pu-rescale mappings → different scores. Proves
/// the descriptor path actually consumes the peak (i.e. is doing HDR feeding),
/// not collapsing to a peak-independent sRGB8 path.
#[test]
fn base_metric_hdr_score_depends_on_display_peak() {
    let (w, h) = (64_u32, 64_u32);
    let ref_u8 = gradient_u8(w, h, 0);
    let dis_u8 = gradient_u8(w, h, 17);
    let ref_lin: Vec<f32> = ref_u8.iter().map(|&c| srgb_eotf(c)).collect();
    let dis_lin: Vec<f32> = dis_u8.iter().map(|&c| srgb_eotf(c)).collect();
    let ref_bytes = f32_bytes(&ref_lin);
    let dis_bytes = f32_bytes(&dis_lin);

    // Two base metrics at distinct display peaks (SSIM-family: the peak is purely
    // the pu-rescale parameter, so `with_display_peak` alone configures HDR).
    let mut lo = Metric::new(
        MetricKind::Ssim2,
        Backend::Cuda,
        w,
        h,
        MetricParams::default_for(MetricKind::Ssim2),
    )
    .expect("ssim2 lo")
    .with_display_peak(100.0);
    let mut hi = Metric::new(
        MetricKind::Ssim2,
        Backend::Cuda,
        w,
        h,
        MetricParams::default_for(MetricKind::Ssim2),
    )
    .expect("ssim2 hi")
    .with_display_peak(4000.0);

    let s_lo = lo
        .compute_pixels_multi(lin_slice(&ref_bytes, w, h), lin_slice(&dis_bytes, w, h))
        .expect("lo")
        .primary();
    let s_hi = hi
        .compute_pixels_multi(lin_slice(&ref_bytes, w, h), lin_slice(&dis_bytes, w, h))
        .expect("hi")
        .primary();
    assert!(
        (s_lo - s_hi).abs() > 1e-6,
        "HDR score must depend on display_peak (peak=100 → {s_lo}, peak=4000 → {s_hi})"
    );
}

/// Every SSIM-family GPU metric auto-feeds HDR through the **base** `Metric`
/// (pu-rescale u8 → `compute_srgb_u8_multi`), bit-identical to the validated
/// `HdrScorer`. ssim2 is covered above and zensim by the scorer tests; this
/// nails dssim + iwssim. iwssim's pyramid needs ≥176 px, so this runs at 256².
#[cfg(all(feature = "dssim", feature = "iwssim"))]
#[test]
fn base_metric_hdr_dssim_and_iwssim() {
    let (w, h) = (256_u32, 256_u32);
    let ref_u8 = gradient_u8(w, h, 1);
    let dis_u8 = gradient_u8(w, h, 23);
    let ref_lin: Vec<f32> = ref_u8.iter().map(|&c| srgb_eotf(c)).collect();
    let dis_lin: Vec<f32> = dis_u8.iter().map(|&c| srgb_eotf(c)).collect();
    let ref_bytes = f32_bytes(&ref_lin);
    let dis_bytes = f32_bytes(&dis_lin);

    for kind in [MetricKind::Dssim, MetricKind::Iwssim] {
        let mut m = Metric::new(kind, Backend::Cuda, w, h, MetricParams::default_for(kind))
            .unwrap_or_else(|e| panic!("{kind:?} new: {e:?}"))
            .with_display_peak(HDR_PEAK_NITS);
        let via_metric = m
            .compute_pixels_multi(lin_slice(&ref_bytes, w, h), lin_slice(&dis_bytes, w, h))
            .unwrap_or_else(|e| panic!("{kind:?} base hdr: {e:?}"))
            .primary();

        let mut hs = HdrScorer::new(kind, Backend::Cuda, w, h, HDR_PEAK_NITS)
            .unwrap_or_else(|e| panic!("{kind:?} scorer: {e:?}"));
        let via_scorer = hs
            .compute_pixels_multi(lin_slice(&ref_bytes, w, h), lin_slice(&dis_bytes, w, h))
            .unwrap_or_else(|e| panic!("{kind:?} scorer hdr: {e:?}"))
            .primary();

        let rel = (via_metric - via_scorer).abs() / via_scorer.abs().max(1e-6);
        assert!(
            rel < 1e-4,
            "{kind:?} base Metric HDR ({via_metric}) must match HdrScorer ({via_scorer}); rel {rel}"
        );
    }
}
