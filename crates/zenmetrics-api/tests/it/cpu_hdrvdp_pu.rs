//! CPU-hdrvdp absolute-nits HDR routing (`hdr::HdrFeeding::IntegratedPuNits`):
//! `HdrScorer` on `Backend::Cpu` routes hdrvdp through
//! `cpu_dispatch::compute_pu_nits_interleaved` →
//! `hdrvdp::score(.., ColorEncoding::RgbBt709, params)` — absolute-luminance
//! interleaved linear-RGB f32 in cd/m², no u8 round-trip, no PU21 surrogate
//! (absolute nits is the metric's *native* input; its own photoreceptor/CSF
//! stack does the adaptation). Unlike the SSIM-family rows this needs NO GPU
//! feature — `MetricParams::Hdrvdp` carries `Box<hdrvdp::Params>`
//! (the 656-byte official-params struct), so a pure-CPU build wires the
//! whole path. NO graceful skips.
#![cfg(all(feature = "hdr", feature = "cpu-hdrvdp"))]

use zenmetrics_api::hdr::{HDR_PEAK_NITS, HdrScorer};
use zenmetrics_api::hdrvdp_cpu as hdrvdp;
use zenmetrics_api::{Backend, Metric, MetricKind, MetricParams};

/// Interleaved absolute-luminance linear-RGB (cd/m²): a smooth HDR gradient
/// (50..650 cd/m²) and a uniformly 10%-darker distorted copy — the same pair
/// shape `cpu_ssim2_pu.rs` / `cpu_zensim_pu.rs` / `hdr_scorer.rs` use.
fn hdr_pair(w: u32, h: u32) -> (Vec<f32>, Vec<f32>) {
    let n = (w * h) as usize;
    let mut r = vec![0.0f32; n * 3];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let v = 50.0 + 600.0 * (x + y) as f32 / (w + h) as f32;
            let i = (y * w as usize + x) * 3;
            r[i] = v;
            r[i + 1] = v;
            r[i + 2] = v;
        }
    }
    let d: Vec<f32> = r.iter().map(|&v| v * 0.9).collect();
    (r, d)
}

fn widen(nits: &[f32]) -> Vec<f64> {
    nits.iter().map(|&v| f64::from(v)).collect()
}

/// The umbrella's CPU-hdrvdp HDR score equals calling `hdrvdp::score`
/// directly on the same nits widened to f64 at `DEFAULT_PIX_PER_DEG` —
/// proving the routing reaches the native entry, the params default fills
/// the viewing geometry, and the (dis, ref) argument order is right.
/// Bit-equal: both sides run the identical deterministic pipeline in the
/// same process.
#[test]
fn cpu_hdrvdp_pu_matches_direct_score() {
    let (w, h) = (128u32, 96u32);
    let (r, d) = hdr_pair(w, h);

    let mut s = HdrScorer::new(MetricKind::Hdrvdp, Backend::Cpu, w, h, HDR_PEAK_NITS)
        .expect("HdrScorer::new hdrvdp on Backend::Cpu");
    assert_eq!(s.kind(), MetricKind::Hdrvdp);
    let umbrella = s.compute_multi(&r, &d).expect("compute_multi");
    assert_eq!(umbrella.metric_name, "hdrvdp");

    let direct = hdrvdp::score(
        &widen(&d),
        &widen(&r),
        w as usize,
        h as usize,
        hdrvdp::ColorEncoding::RgbBt709,
        &hdrvdp::Params::new(hdrvdp::DEFAULT_PIX_PER_DEG),
    )
    .expect("direct hdrvdp::score");

    assert_eq!(
        umbrella.scores[0].value, direct,
        "umbrella {umbrella:?} vs direct {direct}"
    );
    // res.Q scale sanity: a visible 10% dimming lands well below identical.
    assert!(umbrella.scores[0].value < 100.0 && umbrella.scores[0].value > 0.0);
}

/// `Metric::new` + `MetricParams::Hdrvdp` is the custom-geometry entry: a
/// non-default `pix_per_deg` must reach `hdrvdp::score` unchanged (different
/// geometry → different score, and equal to a direct call at the same ppd).
#[test]
fn cpu_hdrvdp_custom_ppd_threads_through() {
    let (w, h) = (128u32, 96u32);
    let (r, d) = hdr_pair(w, h);
    let ppd = 60.0f64;

    let mut m = Metric::new(
        MetricKind::Hdrvdp,
        Backend::Cpu,
        w,
        h,
        MetricParams::Hdrvdp(Box::new(hdrvdp::Params::new(ppd))),
    )
    .expect("Metric::new hdrvdp on Backend::Cpu");
    let umbrella = m
        .compute_pu_nits_interleaved_multi(&r, &d)
        .expect("compute_pu_nits_interleaved_multi");

    let direct = hdrvdp::score(
        &widen(&d),
        &widen(&r),
        w as usize,
        h as usize,
        hdrvdp::ColorEncoding::RgbBt709,
        &hdrvdp::Params::new(ppd),
    )
    .expect("direct hdrvdp::score");
    let at_default = hdrvdp::score(
        &widen(&d),
        &widen(&r),
        w as usize,
        h as usize,
        hdrvdp::ColorEncoding::RgbBt709,
        &hdrvdp::Params::new(hdrvdp::DEFAULT_PIX_PER_DEG),
    )
    .expect("direct hdrvdp::score @ default");

    assert_eq!(umbrella.scores[0].value, direct);
    assert_ne!(
        umbrella.scores[0].value, at_default,
        "ppd=60 must change the score vs ppd=30 (params really thread through)"
    );
}

/// `try_default_for(Hdrvdp)` must fill `pix_per_deg` with
/// `DEFAULT_PIX_PER_DEG` (30 — the measured UPIQ protocol), NOT
/// `Params::default()`'s NaN geometry. The length-mismatch and
/// display-relative `compute` paths fail loudly rather than silently
/// misfeeding buffers.
#[test]
fn cpu_hdrvdp_default_ppd_and_input_validation() {
    let params = MetricParams::try_default_for(MetricKind::Hdrvdp)
        .expect("cpu-hdrvdp enabled => try_default_for Ok");
    let MetricParams::Hdrvdp(p) = params else {
        panic!("try_default_for(Hdrvdp) returned a different variant");
    };
    assert_eq!(p.pix_per_deg, hdrvdp::DEFAULT_PIX_PER_DEG);

    let (w, h) = (64u32, 48u32);
    let (r, d) = hdr_pair(w, h);
    let mut m = Metric::new(
        MetricKind::Hdrvdp,
        Backend::Cpu,
        w,
        h,
        MetricParams::default_for(MetricKind::Hdrvdp),
    )
    .expect("Metric::new hdrvdp on Backend::Cpu");

    // Short nits buffer → explicit length error, not a panic/score.
    let short = &r[..r.len() - 3];
    let err = m
        .compute_pu_nits_interleaved_multi(short, &d)
        .expect_err("short ref nits must error");
    assert!(err.to_string().contains("length"), "got: {err}");

    // Display-relative linear planes are equally invalid input.
    let err = m
        .compute_from_linear_planes(&r, &r, &r, &d, &d, &d)
        .expect_err("linear planes must error");
    let msg = err.to_string();
    assert!(
        msg.contains("nits") || msg.contains("absolute"),
        "got: {msg}"
    );
}

/// sRGB8 is not a valid HDR-VDP feeding — the u8 entries fail loudly and
/// point at the nits path rather than misreading code values as cd/m².
#[test]
fn cpu_hdrvdp_srgb8_is_a_loud_error() {
    let (w, h) = (128u32, 96u32);
    let buf = vec![128u8; (w * h * 3) as usize];

    let mut m = Metric::new(
        MetricKind::Hdrvdp,
        Backend::Cpu,
        w,
        h,
        MetricParams::default_for(MetricKind::Hdrvdp),
    )
    .expect("Metric::new hdrvdp on Backend::Cpu");
    let err = m
        .compute_srgb_u8(&buf, &buf)
        .expect_err("sRGB8 must not score through HDR-VDP");
    let msg = err.to_string();
    assert!(
        msg.contains("nits") || msg.contains("absolute luminance"),
        "error should point at the nits feeding, got: {msg}"
    );
}
