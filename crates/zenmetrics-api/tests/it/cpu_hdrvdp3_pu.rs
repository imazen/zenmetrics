//! CPU-hdrvdp3 absolute-nits HDR routing (`hdr::HdrFeeding::IntegratedPuNits`):
//! HDR-VDP-3 (`MetricKind::Hdrvdp3`) scores via
//! `cpu_dispatch::compute_pu_nits_interleaved` → `hdrvdp::v3` — the same
//! interleaved-linear-RGB cd/m² currency as v2. Unlike v2 there is **no
//! default viewing geometry**: `MetricParams::Default`/`try_default_for`
//! must refuse (the metric would silently produce incomparable JODs);
//! the caller must pass `MetricParams::Hdrvdp3(Box<v3::Params>)`.
//! NO graceful skips.
#![cfg(all(feature = "hdr", feature = "cpu-hdrvdp"))]

use zenmetrics_api::hdrvdp_cpu::v3;
use zenmetrics_api::{Backend, Metric, MetricKind, MetricParams};

/// Interleaved absolute-luminance linear-RGB (cd/m²): a smooth HDR
/// gradient (50..650 cd/m²) and a uniformly 10%-darker distorted copy —
/// the same pair shape `cpu_hdrvdp_pu.rs` uses.
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

fn params(ppd: f64) -> v3::Params {
    v3::Params::new(
        v3::Task::Quality,
        v3::ViewingConditions::new(ppd, v3::Surround::None, 24),
        v3::InputEncoding::RgbBt709,
        v3::Emission::default_for(v3::InputEncoding::RgbBt709),
        v3::Options::reference(v3::Task::Quality),
    )
    .expect("v3::Params::new")
}

/// `Metric::new` + `MetricParams::Hdrvdp3` is the only entry — and it
/// must equal a direct `hdrvdp::v3::score` on the same nits at the same
/// explicit viewing conditions (the params really thread through).
#[test]
fn cpu_hdrvdp3_pu_matches_direct_score() {
    let (w, h) = (128u32, 96u32);
    let (r, d) = hdr_pair(w, h);
    let ppd = 45.0f64;

    let mut m = Metric::new(
        MetricKind::Hdrvdp3,
        Backend::Cpu,
        w,
        h,
        MetricParams::Hdrvdp3(Box::new(params(ppd))),
    )
    .expect("Metric::new hdrvdp3 on Backend::Cpu");
    let umbrella = m
        .compute_pu_nits_interleaved_multi(&r, &d)
        .expect("compute_pu_nits_interleaved_multi");

    let direct = v3::hdrvdp3(&widen(&d), &widen(&r), w as usize, h as usize, &params(ppd))
        .expect("direct v3::hdrvdp3");

    assert_eq!(
        umbrella.scores[0].value, direct.q_jod,
        "umbrella {umbrella:?} vs direct {direct:?}"
    );
    // q_jod scale sanity: a visible 10% dimming lands below identical.
    assert!(umbrella.scores[0].value < 10.0 && umbrella.scores[0].value > 0.0);
}

/// An identical pair is invisible: `q_jod` must be exactly 10.
#[test]
fn cpu_hdrvdp3_identical_pair_scores_ten() {
    let (w, h) = (64u32, 48u32);
    let (r, _) = hdr_pair(w, h);
    let mut m = Metric::new(
        MetricKind::Hdrvdp3,
        Backend::Cpu,
        w,
        h,
        MetricParams::Hdrvdp3(Box::new(params(30.0))),
    )
    .expect("Metric::new hdrvdp3 on Backend::Cpu");
    let umbrella = m
        .compute_pu_nits_interleaved_multi(&r, &r)
        .expect("identical pair");
    assert_eq!(umbrella.scores[0].value, 10.0);
}

/// HDR-VDP-3 has NO safe default viewing geometry: `MetricParams::Default`
/// and `try_default_for` must refuse, and sRGB8/display-relative feedings
/// must fail loudly rather than reinterpret code values as nits.
#[test]
fn cpu_hdrvdp3_requires_explicit_params_and_nits() {
    // No default params exist for v3.
    assert!(
        crate::try_params_for(MetricKind::Hdrvdp3).is_err(),
        "try_default_for(Hdrvdp3) must fail — v3 has no default geometry"
    );

    let (w, h) = (64u32, 48u32);
    let (r, d) = hdr_pair(w, h);
    let mut m = Metric::new(
        MetricKind::Hdrvdp3,
        Backend::Cpu,
        w,
        h,
        MetricParams::Hdrvdp3(Box::new(params(30.0))),
    )
    .expect("Metric::new hdrvdp3 on Backend::Cpu");

    // sRGB8 → loud error, not a score.
    let buf = vec![128u8; (w * h * 3) as usize];
    let err = m
        .compute_srgb_u8(&buf, &buf)
        .expect_err("sRGB8 must not score through HDR-VDP-3");
    assert!(
        err.to_string().contains("nits") || err.to_string().contains("absolute"),
        "error should point at the nits feeding, got: {err}"
    );

    // Display-relative linear planes → loud error.
    let err = m
        .compute_from_linear_planes(&r, &r, &r, &d, &d, &d)
        .expect_err("linear planes must error");
    let msg = err.to_string();
    assert!(
        msg.contains("nits") || msg.contains("absolute"),
        "got: {msg}"
    );

    // Short nits buffer → explicit length error, not a panic.
    let short = &r[..r.len() - 3];
    let err = m
        .compute_pu_nits_interleaved_multi(short, &d)
        .expect_err("short ref nits must error");
    assert!(err.to_string().contains("length"), "got: {err}");
}
