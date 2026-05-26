//! Cached-ref parity tests (Phase 2A).
//!
//! For each metric that wires cached-ref through the umbrella in
//! Phase 2A (cvvdp, zensim, iwssim), verify that:
//!
//! 1. `set_reference_srgb_u8` + `compute_with_cached_reference_srgb_u8`
//!    produces the same `Score` as the one-shot
//!    `compute_srgb_u8(ref, dist)`.
//! 2. `compute_with_cached_reference_srgb_u8` works against multiple
//!    distortions after a single `set_reference_srgb_u8` call (the
//!    sweep workload shape).
//! 3. butter / ssim2 / dssim surface a clear "Phase 2B pending"
//!    error from `set_reference_srgb_u8` — they are not silently
//!    fall-through to one-shot.
//!
//! GPU-dependent (cuda feature). No graceful skips per CLAUDE.md.

#![cfg(feature = "cuda")]

use zenmetrics_api::{Backend, MemoryMode, Metric, MetricKind, MetricParams};

const W: u32 = 256;
const H: u32 = 256;

fn make_pair(seed_a: u64, seed_b: u64) -> (Vec<u8>, Vec<u8>) {
    let n = (W as usize) * (H as usize) * 3;
    let mut r = vec![0u8; n];
    for (i, b) in r.iter_mut().enumerate() {
        *b = (((i as u64).wrapping_mul(seed_a)) & 0xFF) as u8;
    }
    let mut d = vec![0u8; n];
    for (i, b) in d.iter_mut().enumerate() {
        *b = (((i as u64).wrapping_mul(seed_b)) & 0xFF) as u8;
    }
    (r, d)
}

/// Memory mode for the cached-ref tests. Auto is correct for every
/// metric now that butter's strip-mode `set_reference` is Mode-E-
/// supported (task #45 / issue #15) — both the strip walker and the
/// whole-image walker produce the same body-row diffmaps so a single
/// tolerance covers both code paths.
fn cached_ref_memory_mode(_kind: MetricKind) -> MemoryMode {
    MemoryMode::Auto
}

/// Compare cached-ref vs one-shot for a single `(ref, dist)` pair.
///
/// **Tolerance** (`tol`): pass `0.0` for metrics whose cached-ref vs
/// one-shot kernel paths produce bit-identical output (zensim,
/// iwssim — they share the same per-call reduction order). cvvdp /
/// butter / ssim2 / dssim use `Atomic<f32>::fetch_add` reductions
/// whose scheduling can vary between the warm-ref-reuse and
/// fresh-alloc paths; that produces small (~1e-6 to 1e-4) drift on
/// a single pair. Bit-identical isn't a structural guarantee for
/// those; tight numeric agreement is.
fn assert_cached_ref_matches_one_shot(kind: MetricKind, tol: f64) {
    let params = MetricParams::default_for(kind);
    let mode = cached_ref_memory_mode(kind);
    let (r, d) = make_pair(7919, 2147483647);

    // One-shot.
    let mut m_oneshot = Metric::new_with_memory_mode(
        kind,
        Backend::Cuda,
        W,
        H,
        params.clone(),
        mode,
    )
    .unwrap_or_else(|e| panic!("one-shot Metric::new_with_memory_mode({kind:?}) failed: {e}"));
    let s_oneshot = m_oneshot
        .compute_srgb_u8(&r, &d)
        .unwrap_or_else(|e| panic!("compute_srgb_u8({kind:?}) failed: {e}"));

    // Cached-ref.
    let mut m_cached = Metric::new_with_memory_mode(
        kind,
        Backend::Cuda,
        W,
        H,
        params,
        mode,
    )
    .unwrap_or_else(|e| panic!("cached Metric::new_with_memory_mode({kind:?}) failed: {e}"));
    m_cached
        .set_reference_srgb_u8(&r)
        .unwrap_or_else(|e| panic!("set_reference_srgb_u8({kind:?}) failed: {e}"));
    let s_cached = m_cached
        .compute_with_cached_reference_srgb_u8(&d)
        .unwrap_or_else(|e| panic!("compute_with_cached_reference_srgb_u8({kind:?}) failed: {e}"));

    assert_eq!(
        s_oneshot.metric_name, s_cached.metric_name,
        "metric_name must match"
    );
    if tol == 0.0 {
        // Bit-identical — same kernels, same inputs, same reductions.
        // Any difference is a shipping bug per CLAUDE.md "ZERO
        // TOLERANCE for image corruption, distortion, or precision
        // loss".
        assert_eq!(
            s_oneshot.value.to_bits(),
            s_cached.value.to_bits(),
            "{kind:?}: cached-ref score {} differs from one-shot {} (tol=0 required)",
            s_cached.value,
            s_oneshot.value,
        );
    } else {
        let diff = (s_oneshot.value - s_cached.value).abs();
        assert!(
            diff <= tol,
            "{kind:?}: cached-ref {} vs one-shot {} (diff {}) exceeds tolerance {tol}",
            s_cached.value,
            s_oneshot.value,
            diff,
        );
    }
}

/// Cache one ref, score N distorted candidates, compare to N
/// one-shot calls. The sweep workload shape — every cell in a
/// sweep is one new distorted candidate against the same source.
///
/// **Tolerance** (`tol`): for metrics whose pipelines preserve
/// reduction order across cached-ref vs one-shot (zensim, iwssim)
/// pass `0.0` to assert bit-identical. For metrics where the
/// cached-ref path uses different on-device reduction ordering than
/// the one-shot path (cvvdp's per-band atomic-add accumulator can
/// vary slightly when reference state persists across calls) pass
/// a small absolute tolerance.
fn assert_cached_ref_n_distortions(kind: MetricKind, n: usize, tol: f64) {
    let params = MetricParams::default_for(kind);
    let mode = cached_ref_memory_mode(kind);
    let (r, _) = make_pair(7919, 2147483647);

    // Build N distortions deterministically.
    let dists: Vec<Vec<u8>> = (0..n)
        .map(|i| {
            let (_, d) = make_pair(7919, 2147483647u64.wrapping_mul((i + 1) as u64));
            d
        })
        .collect();

    // Cached-ref pass.
    let mut m = Metric::new_with_memory_mode(
        kind,
        Backend::Cuda,
        W,
        H,
        params.clone(),
        mode,
    )
    .unwrap_or_else(|e| panic!("cached Metric::new_with_memory_mode({kind:?}) failed: {e}"));
    m.set_reference_srgb_u8(&r)
        .unwrap_or_else(|e| panic!("set_reference_srgb_u8({kind:?}) failed: {e}"));
    let cached_scores: Vec<f64> = dists
        .iter()
        .map(|d| {
            m.compute_with_cached_reference_srgb_u8(d)
                .unwrap_or_else(|e| panic!("cached compute({kind:?}) failed: {e}"))
                .value
        })
        .collect();

    // One-shot pass for parity.
    let mut m_os = Metric::new_with_memory_mode(
        kind,
        Backend::Cuda,
        W,
        H,
        params,
        mode,
    )
    .unwrap_or_else(|e| panic!("oneshot Metric::new_with_memory_mode({kind:?}) failed: {e}"));
    let oneshot_scores: Vec<f64> = dists
        .iter()
        .map(|d| {
            m_os.compute_srgb_u8(&r, d)
                .unwrap_or_else(|e| panic!("one-shot compute({kind:?}) failed: {e}"))
                .value
        })
        .collect();

    if tol == 0.0 {
        let cached_bits: Vec<u64> = cached_scores.iter().map(|v| v.to_bits()).collect();
        let oneshot_bits: Vec<u64> = oneshot_scores.iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            cached_bits, oneshot_bits,
            "{kind:?}: cached-ref scores must match one-shot bit-identically (got tol=0)"
        );
    } else {
        for (i, (c, o)) in cached_scores.iter().zip(oneshot_scores.iter()).enumerate() {
            let diff = (c - o).abs();
            assert!(
                diff <= tol,
                "{kind:?}: cached_scores[{i}] = {c} vs one-shot {o} (diff {diff}) exceeds tolerance {tol}"
            );
        }
    }
}

#[cfg(feature = "cvvdp")]
#[test]
fn cached_ref_cvvdp_matches_one_shot() {
    // cvvdp's warm-ref path keeps device buffers alive that the
    // fresh path re-allocates — touches Atomic<f32> reduction order.
    // ~1e-6 JOD drift observed; well inside pycvvdp's 5e-3 parity
    // gate.
    assert_cached_ref_matches_one_shot(MetricKind::Cvvdp, 1e-4);
}

#[cfg(feature = "cvvdp")]
#[test]
fn cached_ref_cvvdp_n_distortions() {
    // cvvdp's per-band atomic-add reduction order can vary slightly
    // when reference state persists across compute calls — the warm-
    // ref path keeps device buffers alive that the fresh path
    // reallocates. Observed drift: ~1e-6 JOD on the 256×256 noise
    // fixture (well within the 5e-3 JOD pycvvdp parity gate). Bit-
    // identical isn't a structural guarantee; tight numeric agreement
    // is.
    assert_cached_ref_n_distortions(MetricKind::Cvvdp, 3, 1e-4);
}

#[cfg(feature = "zensim")]
#[test]
fn cached_ref_zensim_matches_one_shot() {
    assert_cached_ref_matches_one_shot(MetricKind::Zensim, 0.0);
}

#[cfg(feature = "zensim")]
#[test]
fn cached_ref_zensim_n_distortions() {
    assert_cached_ref_n_distortions(MetricKind::Zensim, 3, 0.0);
}

#[cfg(feature = "iwssim")]
#[test]
fn cached_ref_iwssim_matches_one_shot() {
    assert_cached_ref_matches_one_shot(MetricKind::Iwssim, 0.0);
}

#[cfg(feature = "iwssim")]
#[test]
fn cached_ref_iwssim_n_distortions() {
    assert_cached_ref_n_distortions(MetricKind::Iwssim, 3, 0.0);
}

#[cfg(feature = "iwssim")]
#[test]
fn cached_ref_iwssim_has_cached_reference_roundtrip() {
    let params = MetricParams::default_for(MetricKind::Iwssim);
    let mut m = Metric::new(MetricKind::Iwssim, Backend::Cuda, W, H, params).unwrap();
    assert!(!m.has_cached_reference());
    let (r, _) = make_pair(7919, 2147483647);
    m.set_reference_srgb_u8(&r).unwrap();
    assert!(m.has_cached_reference());
    m.clear_reference();
    assert!(!m.has_cached_reference());
}

#[cfg(feature = "butter")]
#[test]
fn cached_ref_butter_matches_one_shot() {
    // butter uses Atomic<f32>::fetch_add for the per-octave reduction;
    // cached-ref vs one-shot may drift by ~1e-5 score units. Bit-
    // identical isn't a structural guarantee; tight numeric agreement
    // is well within butter's pycvvdp-equivalent parity gate.
    assert_cached_ref_matches_one_shot(MetricKind::Butter, 1e-4);
}

#[cfg(feature = "butter")]
#[test]
fn cached_ref_butter_n_distortions() {
    assert_cached_ref_n_distortions(MetricKind::Butter, 3, 1e-4);
}

#[cfg(feature = "ssim2")]
#[test]
fn cached_ref_ssim2_matches_one_shot() {
    // ssim2 reduction order varies (~5e-5 floor per task #52).
    assert_cached_ref_matches_one_shot(MetricKind::Ssim2, 1e-3);
}

#[cfg(feature = "ssim2")]
#[test]
fn cached_ref_ssim2_n_distortions() {
    assert_cached_ref_n_distortions(MetricKind::Ssim2, 3, 1e-3);
}

#[cfg(feature = "dssim")]
#[test]
fn cached_ref_dssim_matches_one_shot() {
    assert_cached_ref_matches_one_shot(MetricKind::Dssim, 1e-4);
}

#[cfg(feature = "dssim")]
#[test]
fn cached_ref_dssim_n_distortions() {
    assert_cached_ref_n_distortions(MetricKind::Dssim, 3, 1e-4);
}

#[cfg(feature = "butter")]
#[test]
fn cached_ref_butter_has_cached_reference_roundtrip() {
    let params = MetricParams::default_for(MetricKind::Butter);
    // butter is strip-preferred at 256x256. With Mode E (task #45)
    // the strip-mode instance accepts set_reference by allocating a
    // whole-image cache sibling — the umbrella roundtrip works
    // through both modes.
    let mut m = Metric::new(MetricKind::Butter, Backend::Cuda, W, H, params).unwrap();
    assert!(!m.has_cached_reference());
    let (r, _) = make_pair(7919, 2147483647);
    m.set_reference_srgb_u8(&r).unwrap();
    assert!(m.has_cached_reference());
    m.clear_reference();
    assert!(!m.has_cached_reference());
}

#[cfg(feature = "ssim2")]
#[test]
fn cached_ref_ssim2_has_cached_reference_roundtrip() {
    let params = MetricParams::default_for(MetricKind::Ssim2);
    let mut m = Metric::new(MetricKind::Ssim2, Backend::Cuda, W, H, params).unwrap();
    assert!(!m.has_cached_reference());
    let (r, _) = make_pair(7919, 2147483647);
    m.set_reference_srgb_u8(&r).unwrap();
    assert!(m.has_cached_reference());
    m.clear_reference();
    assert!(!m.has_cached_reference());
}

#[cfg(feature = "dssim")]
#[test]
fn cached_ref_dssim_has_cached_reference_roundtrip() {
    let params = MetricParams::default_for(MetricKind::Dssim);
    let mut m = Metric::new(MetricKind::Dssim, Backend::Cuda, W, H, params).unwrap();
    assert!(!m.has_cached_reference());
    let (r, _) = make_pair(7919, 2147483647);
    m.set_reference_srgb_u8(&r).unwrap();
    assert!(m.has_cached_reference());
    m.clear_reference();
    assert!(!m.has_cached_reference());
}
/// Mode E parity test for task #73. Forces dssim into Strip mode at
/// a size that would normally pick Full (4096×4096 fits in a 12 GB
/// fleet box's VRAM, ~3 GB measured), then verifies the cached-ref
/// strip path produces scores within 1e-4 of the Full-mode cached
/// ref path across 3 distortions sharing a single reference.
///
/// 4096×4096 is the large-enough size where mode E starts to matter
/// on small-VRAM tiers (24 MP+ images don't fit in Full at 4 GB,
/// 12 GB tiers can fit Full at 16 MP but not 96 MP). The body
/// auto-sizer picks h_body within the VRAM cap so the test runs on
/// hosts with widely different memory caps.
///
/// Tolerance follows the other dssim cached-ref tests (1e-4) — same
/// kernels, same data, but the per-strip path runs each scale's
/// reduction across many strips whereas the Full path runs it once
/// over the whole image. Atomic<f32> ordering between strips
/// introduces small (~1e-6 to 1e-5) drift.
#[cfg(feature = "dssim")]
#[test]
fn cached_ref_dssim_strip_n_distortions_24mp() {
    use zenmetrics_api::{Backend, MemoryMode, Metric, MetricKind, MetricParams};

    const WL: u32 = 4096;
    const HL: u32 = 4096;

    fn make_large_pair(seed_a: u64, seed_b: u64) -> (Vec<u8>, Vec<u8>) {
        let n = (WL as usize) * (HL as usize) * 3;
        let mut r = vec![0u8; n];
        for (i, b) in r.iter_mut().enumerate() {
            *b = (((i as u64).wrapping_mul(seed_a)) & 0xFF) as u8;
        }
        let mut d = vec![0u8; n];
        for (i, b) in d.iter_mut().enumerate() {
            *b = (((i as u64).wrapping_mul(seed_b)) & 0xFF) as u8;
        }
        (r, d)
    }

    let params = MetricParams::default_for(MetricKind::Dssim);
    let n_dists = 3usize;
    let (r, _) = make_large_pair(7919, 2147483647);
    let dists: Vec<Vec<u8>> = (0..n_dists)
        .map(|i| {
            let (_, d) = make_large_pair(7919, 2147483647u64.wrapping_mul((i + 1) as u64));
            d
        })
        .collect();

    // Cached-ref in Strip mode (mode E).
    let mut m_strip = Metric::new_with_memory_mode(
        MetricKind::Dssim,
        Backend::Cuda,
        WL,
        HL,
        params.clone(),
        MemoryMode::Strip { h_body: None },
    )
    .unwrap_or_else(|e| panic!("strip Metric::new_with_memory_mode failed: {e}"));
    m_strip
        .set_reference_srgb_u8(&r)
        .unwrap_or_else(|e| panic!("strip set_reference_srgb_u8 failed: {e}"));
    let strip_scores: Vec<f64> = dists
        .iter()
        .map(|d| {
            m_strip
                .compute_with_cached_reference_srgb_u8(d)
                .unwrap_or_else(|e| panic!("strip compute failed: {e}"))
                .value
        })
        .collect();

    // Cached-ref in Full mode (parity target).
    let mut m_full = Metric::new_with_memory_mode(
        MetricKind::Dssim,
        Backend::Cuda,
        WL,
        HL,
        params,
        MemoryMode::Full,
    )
    .unwrap_or_else(|e| panic!("full Metric::new_with_memory_mode failed: {e}"));
    m_full
        .set_reference_srgb_u8(&r)
        .unwrap_or_else(|e| panic!("full set_reference_srgb_u8 failed: {e}"));
    let full_scores: Vec<f64> = dists
        .iter()
        .map(|d| {
            m_full
                .compute_with_cached_reference_srgb_u8(d)
                .unwrap_or_else(|e| panic!("full compute failed: {e}"))
                .value
        })
        .collect();

    let tol = 1e-4_f64;
    for (i, (s, f)) in strip_scores.iter().zip(full_scores.iter()).enumerate() {
        let diff = (s - f).abs();
        assert!(
            diff <= tol,
            "dssim strip-cached[{i}] = {s} vs full-cached {f} (diff {diff}) exceeds tolerance {tol}"
        );
    }
}

/// Mode E parity test: at a size where the umbrella's Auto policy
/// resolves to Strip for butter, set_reference + N
/// compute_with_cached_reference calls must agree with N one-shot
/// compute_srgb_u8 calls under the same Atomic<f32>-tolerance band
/// the in-mode butter tests use. This exercises the row-blit strip
/// walker added in task #45 against multiple distortions.
///
/// Size: 1024×1024 — large enough that the strip walker engages
/// at body < image_h (multiple strips per image) without bloating
/// CI wall time.
///
/// Comparison: both sides use Auto-resolved Strip mode, so the
/// baseline is single-resolution strip compute (NOT multires-Full
/// which adds the half-res supersample contribution). The tight
/// numeric agreement (~1e-4) shows the cached-ref strip walker
/// produces the same body-row diffmaps as one-shot strip compute.
#[cfg(feature = "butter")]
#[test]
fn cached_ref_butter_strip_n_distortions_1mp() {
    const SW: u32 = 1024;
    const SH: u32 = 1024;
    let n = (SW as usize) * (SH as usize) * 3;
    let mut r = vec![0u8; n];
    for (i, b) in r.iter_mut().enumerate() {
        *b = (((i as u64).wrapping_mul(7919)) & 0xFF) as u8;
    }
    let dists: Vec<Vec<u8>> = (0..3)
        .map(|j| {
            let seed = 2147483647u64.wrapping_mul((j + 1) as u64);
            let mut d = vec![0u8; n];
            for (i, b) in d.iter_mut().enumerate() {
                *b = (((i as u64).wrapping_mul(seed)) & 0xFF) as u8;
            }
            d
        })
        .collect();

    let params = MetricParams::default_for(MetricKind::Butter);

    // Auto-resolved (likely Strip at 1MP — butter is strip-preferred).
    let mut m = Metric::new_with_memory_mode(
        MetricKind::Butter,
        Backend::Cuda,
        SW,
        SH,
        params.clone(),
        MemoryMode::Auto,
    )
    .expect("butter Auto Metric::new");
    m.set_reference_srgb_u8(&r)
        .expect("butter set_reference at 1MP (Mode E)");
    let cached: Vec<f64> = dists
        .iter()
        .map(|d| {
            m.compute_with_cached_reference_srgb_u8(d)
                .expect("butter cached compute")
                .value
        })
        .collect();

    // Baseline: same Auto-resolved Strip mode, one-shot compute.
    // Comparing strip-cached-ref vs strip-one-shot isolates the
    // Mode E walker's correctness from any single-res/multires
    // mode difference.
    let mut m_os = Metric::new_with_memory_mode(
        MetricKind::Butter,
        Backend::Cuda,
        SW,
        SH,
        params,
        MemoryMode::Auto,
    )
    .expect("butter Auto Metric::new (one-shot)");
    let oneshot: Vec<f64> = dists
        .iter()
        .map(|d| {
            m_os.compute_srgb_u8(&r, d)
                .expect("butter strip one-shot compute")
                .value
        })
        .collect();

    for (i, (c, o)) in cached.iter().zip(oneshot.iter()).enumerate() {
        let diff = (c - o).abs();
        // Same tolerance band as the existing butter cached-ref
        // tests at 256×256 (Atomic<f32> reduction-order drift).
        assert!(
            diff <= 1e-4,
            "cached_scores[{i}] = {c} vs strip one-shot {o} (diff {diff})"
        );
    }
}
