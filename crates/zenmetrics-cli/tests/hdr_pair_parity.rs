//! PARITY GATE: the shared HDR pair-scoring layer (`hdr::score_hdr_pair_per_score_pairs`
//! + `hdr::score_hdr_zensim_with_features_per_score_pairs`, consumed by jobexec's
//! ScoreFile HDR arm) must produce values IDENTICAL to what `cmd_score_pairs --hdr`
//! computes for the same pair.
//!
//! Side (a) below is `cmd_score_pairs --hdr`'s per-metric feeding, hand-composed from
//! the SAME primitives its HDR blocks call, line-for-line (main.rs: the
//! `hdr_u8_pair` construction — `to_cvvdp_rgb8` for the cvvdp kinds, `to_sdr_rgb8`
//! otherwise — then `run_metric` via `score_one_pair_maybe_hdr`; zensim features via
//! `run_zensim_with_features`). Composing the primitives rather than invoking
//! `cmd_score_pairs` itself keeps this test buildable while the `sweep` feature is
//! blocked on the sibling-codec ErrorCategory reshape (score-pairs is `sweep`-gated);
//! the equivalence to the real command is by construction — the referenced blocks are
//! the only code between decode-to-nits and the primitive calls. A follow-on
//! end-to-end run against the staged fleet binary covers the GPU metrics
//! (documented in the change record).
//!
//! Equality is asserted EXACT (bit-for-bit f64): both sides run the identical scoring
//! functions on identical bytes in the same process, so any drift means the shared
//! layer's feeding diverged from score-pairs' — the exact bug class this test exists
//! to catch.
#![cfg(all(feature = "hdr", feature = "cpu-metrics"))]

use zenmetrics_cli::hdr::{
    HdrImageFeeds, HdrPairScorers, HdrTransfer, NitsImage, score_hdr_pair_per_score_pairs,
    score_hdr_zensim_with_features_per_score_pairs, to_cvvdp_rgb8, to_sdr_rgb8,
};
use zenmetrics_cli::metrics::{
    GpuRuntime, MetricKind, ZensimFeatureRegime, run_metric, run_zensim_with_features,
};

/// Deterministic synthetic HDR image: a luminance gradient spanning shadow
/// (~2 cd/m²) through SDR white into HDR highlights (~600 cd/m²), plus a
/// seeded LCG texture so the metrics see structure, not a flat ramp.
fn synth_nits(seed: u32, w: u32, h: u32) -> NitsImage {
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    let mut s = seed.wrapping_add(1);
    for y in 0..h {
        for x in 0..w {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let ramp = (x as f32) / (w as f32 - 1.0); // 0..1 across the row
            let base = 2.0 + 598.0 * ramp * ramp; // 2 .. 600 cd/m², highlight-heavy
            let tex = ((s >> 16) & 0xff) as f32 / 255.0; // 0..1 texture
            let vy = (y as f32) / (h as f32 - 1.0);
            rgb.push(base * (0.85 + 0.15 * tex));
            rgb.push(base * (0.80 + 0.20 * vy));
            rgb.push(base * (0.75 + 0.25 * (1.0 - tex)));
        }
    }
    NitsImage {
        rgb,
        width: w,
        height: h,
    }
}

/// A distorted sibling: multiplicative banding + seeded noise on top of the
/// reference — visible but not destructive, so scores land mid-range.
fn distort(reference: &NitsImage, seed: u32) -> NitsImage {
    let mut s = seed.wrapping_add(7);
    let rgb = reference
        .rgb
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let band = if (i / 96).is_multiple_of(2) {
                1.06
            } else {
                0.94
            };
            let noise = 1.0 + (((s >> 20) & 0xff) as f32 / 255.0 - 0.5) * 0.08;
            (v * band * noise).max(0.0)
        })
        .collect();
    NitsImage {
        rgb,
        width: reference.width,
        height: reference.height,
    }
}

/// 192×192: above iwssim's 176-pixel minimum, small enough for CI.
const W: u32 = 192;
const H: u32 = 192;
const TRANSFER: HdrTransfer = HdrTransfer::PuRescale;

/// Side (a): `cmd_score_pairs --hdr`'s feeding for one metric, composed from the
/// exact primitives its HDR blocks call (u8-shell metrics + the cvvdp-u8 kinds;
/// the GPU faithful paths need a GPU and are covered by the fleet-binary
/// cross-check instead).
fn score_pairs_side(metric: MetricKind, r: &NitsImage, d: &NitsImage) -> Vec<(&'static str, f64)> {
    let (ru8, du8) = if matches!(metric, MetricKind::Cvvdp) {
        (to_cvvdp_rgb8(r).0, to_cvvdp_rgb8(d).0)
    } else {
        (to_sdr_rgb8(r, TRANSFER), to_sdr_rgb8(d, TRANSFER))
    };
    run_metric(metric, &ru8, &du8, GpuRuntime::Auto)
        .unwrap_or_else(|e| panic!("score-pairs-side {metric:?}: {e}"))
}

#[test]
fn shared_hdr_layer_matches_score_pairs_feeding_per_metric() {
    let r = synth_nits(11, W, H);
    let d = distort(&r, 23);

    // Side (a): score-pairs primitive composition, per CPU metric.
    let metrics = [
        MetricKind::Cvvdp,
        MetricKind::Ssim2,
        MetricKind::Butteraugli,
        MetricKind::Iwssim,
        MetricKind::Zensim,
    ];
    let expected: Vec<Vec<(&'static str, f64)>> = metrics
        .iter()
        .map(|m| score_pairs_side(*m, &r, &d))
        .collect();

    // Side (b): the shared layer jobexec's ScoreFile HDR arm calls.
    let rf = HdrImageFeeds::new(synth_nits(11, W, H), TRANSFER);
    let df = HdrImageFeeds::new(distort(&synth_nits(11, W, H), 23), TRANSFER);
    let mut scorers = HdrPairScorers::new(GpuRuntime::Auto);
    for (m, want) in metrics.iter().zip(&expected) {
        let got = score_hdr_pair_per_score_pairs(*m, &rf, &df, &mut scorers)
            .unwrap_or_else(|e| panic!("shared-layer {m:?}: {e}"));
        assert_eq!(
            got.len(),
            want.len(),
            "{m:?}: column count diverged from score-pairs"
        );
        for ((gc, gv), (wc, wv)) in got.iter().zip(want) {
            assert_eq!(gc, wc, "{m:?}: column name diverged");
            assert!(
                gv.to_bits() == wv.to_bits(),
                "{m:?}/{gc}: shared layer {gv} != score-pairs feeding {wv} (bit-exact required)"
            );
        }
        // Scores must be real numbers, not accidental NaN-equal.
        assert!(got.iter().all(|(_, v)| v.is_finite()), "{m:?}: {got:?}");
    }
}

#[test]
fn shared_hdr_zensim_features_match_score_pairs_feature_path() {
    let r = synth_nits(31, W, H);
    let d = distort(&r, 41);

    // Side (a): score-pairs' CPU zensim feature path — PU21 u8 shell into
    // run_zensim_with_features (main.rs `want_features` branch, hdr_mode).
    let (want_score, want_feats) =
        run_zensim_with_features(&to_sdr_rgb8(&r, TRANSFER), &to_sdr_rgb8(&d, TRANSFER))
            .expect("score-pairs-side zensim features");

    // Side (b): the shared layer.
    let rf = HdrImageFeeds::new(synth_nits(31, W, H), TRANSFER);
    let df = HdrImageFeeds::new(distort(&synth_nits(31, W, H), 41), TRANSFER);
    let mut scorers = HdrPairScorers::new(GpuRuntime::Auto);
    let (got_score, got_feats) = score_hdr_zensim_with_features_per_score_pairs(
        MetricKind::Zensim,
        &rf,
        &df,
        &mut scorers,
        ZensimFeatureRegime::WithIw,
    )
    .expect("shared-layer zensim features");

    assert!(
        got_score.to_bits() == want_score.to_bits(),
        "zensim score diverged: {got_score} != {want_score}"
    );
    assert_eq!(got_feats.len(), want_feats.len(), "feature width diverged");
    for (i, (g, w)) in got_feats.iter().zip(&want_feats).enumerate() {
        assert!(
            g.to_bits() == w.to_bits(),
            "feat_{i} diverged: {g} != {w} (bit-exact required)"
        );
    }
    assert!(got_score.is_finite() && !got_feats.is_empty());
}

#[test]
fn dssim_is_refused_in_hdr_mode_by_design() {
    let rf = HdrImageFeeds::new(synth_nits(5, 64, 64), TRANSFER);
    let df = HdrImageFeeds::new(distort(&synth_nits(5, 64, 64), 9), TRANSFER);
    let mut scorers = HdrPairScorers::new(GpuRuntime::Auto);
    let err = score_hdr_pair_per_score_pairs(MetricKind::Dssim, &rf, &df, &mut scorers)
        .expect_err("dssim must be refused in HDR mode");
    assert!(
        err.to_string().contains("by design"),
        "unexpected dssim error text: {err}"
    );
}

#[test]
fn dimension_mismatch_errors_before_scoring() {
    let rf = HdrImageFeeds::new(synth_nits(5, 64, 64), TRANSFER);
    let df = HdrImageFeeds::new(synth_nits(5, 48, 64), TRANSFER);
    let mut scorers = HdrPairScorers::new(GpuRuntime::Auto);
    let err = score_hdr_pair_per_score_pairs(MetricKind::Ssim2, &rf, &df, &mut scorers)
        .expect_err("dims mismatch must error");
    assert!(err.to_string().contains("dimension mismatch"), "{err}");
}

#[test]
fn transfer_choice_reaches_the_u8_shell() {
    // PQ vs PU-rescale produce different u8 shells → different scores. Guards
    // against the transfer silently not being threaded through the feeds.
    let r = synth_nits(3, 96, 96);
    let d = distort(&r, 13);
    let mut scorers = HdrPairScorers::new(GpuRuntime::Auto);
    let pu = {
        let rf = HdrImageFeeds::new(synth_nits(3, 96, 96), HdrTransfer::PuRescale);
        let df = HdrImageFeeds::new(distort(&synth_nits(3, 96, 96), 13), HdrTransfer::PuRescale);
        score_hdr_pair_per_score_pairs(MetricKind::Ssim2, &rf, &df, &mut scorers).unwrap()[0].1
    };
    let pq = {
        let rf = HdrImageFeeds::new(r, HdrTransfer::Pq);
        let df = HdrImageFeeds::new(d, HdrTransfer::Pq);
        score_hdr_pair_per_score_pairs(MetricKind::Ssim2, &rf, &df, &mut scorers).unwrap()[0].1
    };
    assert!(
        (pu - pq).abs() > 1e-9,
        "PQ and PU-rescale shells should differ: pu={pu} pq={pq}"
    );
}

/// GPU-side parity — the faithful paths (cvvdp-gpu linear planes, butteraugli-gpu
/// umbrella HdrScorer) and the GPU u8 shells (ssim2-gpu, iwssim-gpu, zensim-gpu +
/// 372 with-iw features) against the same score-pairs primitive composition.
///
/// `#[ignore]` because it needs a CUDA/wgpu GPU — the skip decision is the
/// caller's (`cargo test -- --ignored` on a GPU box), never a silent runtime
/// bail. Run + record results whenever the executor image is rebuilt.
///
/// Note on side (a) for zensim-gpu: score-pairs uses the pub(crate)
/// `MetricCache::compute_zensim_features`, unreachable from an integration
/// test; `run_zensim_gpu_with_features` makes the IDENTICAL umbrella call
/// (`ZensimParams::default_weights().with_regime(..)` →
/// `compute_features_srgb_u8`) minus the instance cache, which affects cost,
/// not values.
#[cfg(all(
    feature = "gpu-cvvdp",
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-iwssim",
    feature = "gpu-zensim"
))]
mod gpu_parity {
    use super::*;
    use zenmetrics_cli::hdr::{score_via_hdr_scorer, to_cvvdp_linear_planes};
    use zenmetrics_cli::metrics::{cvvdp_gpu, run_zensim_gpu_with_features};

    /// |a-b| within `tol·max(1,|b|)`. GPU pooling (atomic-f32) is not
    /// bit-order-deterministic across invocations, so the faithful cvvdp path
    /// gets a tiny relative tolerance; everything else must match bit-exactly
    /// (same kernel, same bytes, same reduction order).
    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0)
    }

    #[test]
    #[ignore = "needs a CUDA/wgpu GPU — run manually: cargo test --release -- --ignored"]
    fn gpu_shared_layer_matches_score_pairs_feeding() {
        let r = synth_nits(11, W, H);
        let d = distort(&r, 23);
        let rf = HdrImageFeeds::new(synth_nits(11, W, H), TRANSFER);
        let df = HdrImageFeeds::new(distort(&synth_nits(11, W, H), 23), TRANSFER);
        let mut scorers = HdrPairScorers::new(GpuRuntime::Auto);

        // cvvdp-gpu: faithful linear planes @ DisplayTarget::hdr(1000).
        let want_cvvdp = {
            let mut s = cvvdp_gpu::CvvdpBatchScorer::new_with_target(
                GpuRuntime::Auto,
                cvvdp_gpu::DisplayTarget::hdr(zenmetrics_cli::hdr::HDR_DISPLAY_PEAK_NITS),
            )
            .expect("cvvdp gpu scorer");
            let (rr, rg, rb) = to_cvvdp_linear_planes(&r);
            let (dr, dg, db) = to_cvvdp_linear_planes(&d);
            s.score_from_linear_planes(&rr, &rg, &rb, &dr, &dg, &db, W, H)
                .expect("cvvdp faithful")
        };
        let got_cvvdp =
            score_hdr_pair_per_score_pairs(MetricKind::CvvdpGpu, &rf, &df, &mut scorers)
                .expect("shared cvvdp-gpu")[0]
                .1;
        eprintln!(
            "[gpu-parity] cvvdp-gpu: shared={got_cvvdp} score-pairs={want_cvvdp} delta={}",
            (got_cvvdp - want_cvvdp).abs()
        );
        assert!(close(got_cvvdp, want_cvvdp, 1e-6), "cvvdp-gpu diverged");

        // butteraugli-gpu: faithful umbrella HdrScorer (max + pnorm_3). Side
        // (a) uses the fleet's explicit `--gpu-runtime cuda`; side (b) keeps
        // Auto through the shared layer, which must resolve to the same
        // backend via its runtime ladder (the divergence this test caught
        // when the ladder was missing).
        let want_butter = score_via_hdr_scorer(
            MetricKind::ButteraugliGpu,
            &r,
            &d,
            TRANSFER,
            GpuRuntime::Cuda,
        )
        .expect("umbrella path exists")
        .expect("butteraugli faithful");
        let got_butter =
            score_hdr_pair_per_score_pairs(MetricKind::ButteraugliGpu, &rf, &df, &mut scorers)
                .expect("shared butteraugli-gpu");
        for ((gc, gv), (wc, wv)) in got_butter.iter().zip(&want_butter) {
            eprintln!(
                "[gpu-parity] butteraugli-gpu/{gc}: shared={gv} score-pairs={wv} delta={}",
                (gv - wv).abs()
            );
            assert_eq!(gc, wc);
            assert!(
                close(*gv, *wv, 1e-9),
                "butteraugli-gpu {gc} diverged: {gv} vs {wv}"
            );
        }

        // ssim2-gpu + iwssim-gpu: PU21 u8 shell -> run_metric, bit-exact.
        for kind in [MetricKind::Ssim2Gpu, MetricKind::IwssimGpu] {
            let want = score_pairs_side(kind, &r, &d);
            let got = score_hdr_pair_per_score_pairs(kind, &rf, &df, &mut scorers)
                .unwrap_or_else(|e| panic!("shared {kind:?}: {e}"));
            for ((gc, gv), (wc, wv)) in got.iter().zip(&want) {
                eprintln!(
                    "[gpu-parity] {kind:?}/{gc}: shared={gv} score-pairs={wv} delta={}",
                    (gv - wv).abs()
                );
                assert_eq!(gc, wc);
                assert!(
                    close(*gv, *wv, 1e-9),
                    "{kind:?} {gc} diverged: {gv} vs {wv}"
                );
            }
        }

        // zensim-gpu: u8 shell + 372 with-iw features.
        let (want_zs, want_zf) = run_zensim_gpu_with_features(
            &to_sdr_rgb8(&r, TRANSFER),
            &to_sdr_rgb8(&d, TRANSFER),
            GpuRuntime::Auto,
            ZensimFeatureRegime::WithIw,
        )
        .expect("zensim gpu features");
        let (got_zs, got_zf) = score_hdr_zensim_with_features_per_score_pairs(
            MetricKind::ZensimGpu,
            &rf,
            &df,
            &mut scorers,
            ZensimFeatureRegime::WithIw,
        )
        .expect("shared zensim-gpu features");
        eprintln!(
            "[gpu-parity] zensim-gpu: shared={got_zs} score-pairs={want_zs} delta={}",
            (got_zs - want_zs).abs()
        );
        assert!(close(got_zs, want_zs, 1e-9), "zensim-gpu score diverged");
        assert_eq!(got_zf.len(), 372, "with-iw must emit 372 features on GPU");
        assert_eq!(want_zf.len(), 372);
        let max_fd = got_zf
            .iter()
            .zip(&want_zf)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0f64, f64::max);
        eprintln!("[gpu-parity] zensim-gpu features: max abs delta = {max_fd}");
        assert!(max_fd <= 1e-9, "zensim-gpu features diverged: max {max_fd}");
    }
}
