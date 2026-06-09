//! `Metric::compute_srgb_u8_multi` — the lossless multi-output umbrella
//! path. Asserts butter returns BOTH the max-norm and the libjxl 3-norm
//! (`pnorm_3`) the single-value path drops, and zensim returns its scalar
//! PLUS the regime-length feature vector — all without bypassing the
//! umbrella to a per-crate typed API. CUDA-gated; NO GRACEFUL SKIPS.
#![cfg(all(feature = "cuda", feature = "butter", feature = "zensim"))]

use zenmetrics_api::{Backend, MemoryMode, Metric, MetricKind, MetricParams};

fn pair(w: u32, h: u32) -> (Vec<u8>, Vec<u8>) {
    let n = (w * h * 3) as usize;
    let r: Vec<u8> = (0..n)
        .map(|i| ((i * 2654435761usize) >> 13) as u8)
        .collect();
    let d: Vec<u8> = r
        .iter()
        .enumerate()
        .map(|(i, b)| b.wrapping_add(((i * 40503) & 0x1f) as u8))
        .collect();
    (r, d)
}

#[test]
fn butter_multi_returns_max_and_pnorm3_and_matches_single() {
    let (w, h) = (256u32, 256u32);
    let (r, d) = pair(w, h);
    let mut m = Metric::new(
        MetricKind::Butter,
        Backend::Cuda,
        w,
        h,
        MetricParams::default_for(MetricKind::Butter),
    )
    .expect("Metric::new butter");

    let multi = m
        .compute_srgb_u8_multi(&r, &d)
        .expect("compute_srgb_u8_multi");
    assert_eq!(multi.metric_name, "butter");
    // Two named scalars: the max-norm (primary) + the libjxl 3-norm.
    assert_eq!(multi.scores.len(), 2, "butter must expose max + pnorm_3");
    assert_eq!(multi.scores[0].name, "max");
    assert_eq!(multi.scores[1].name, "pnorm_3");
    assert!(
        multi.features.is_empty(),
        "butter is scalar-only, no features"
    );
    let max = multi.get("max").unwrap();
    let pnorm3 = multi.get("pnorm_3").unwrap();
    assert!(max.is_finite() && pnorm3.is_finite(), "{max} {pnorm3}");
    assert!(max > 0.0 && pnorm3 > 0.0, "distorted pair: both > 0");
    assert!(pnorm3 <= max + 1e-6, "3-norm ≤ max-norm");

    // Primary == the single-value path's score (same kernel, max-norm).
    let single = m.compute_srgb_u8(&r, &d).expect("compute_srgb_u8");
    assert!(
        (multi.primary() - single.value).abs() < 1e-3,
        "multi.primary {} vs single {}",
        multi.primary(),
        single.value
    );
}

#[test]
fn butter_linear_planes_multi_returns_max_and_pnorm3() {
    // Faithful HDR path: 6 display-relative [0,1] f32 LINEAR planes (ref/dist
    // R/G/B). Use a SMOOTH diagonal gradient with a uniform 10% darkening as
    // the distortion — spatially correlated, low-frequency, image-like. (Raw
    // uniform-random noise fed *as linear light* overflows any perceptual
    // metric's HF stage; that's an input property, not a path bug.)
    let (w, h) = (256u32, 256u32);
    let n = (w * h) as usize;
    let mut rr = vec![0.0f32; n];
    for y in 0..h as usize {
        for x in 0..w as usize {
            // smooth diagonal ramp in [0.1, 0.9]
            rr[y * w as usize + x] = 0.1 + 0.8 * (x + y) as f32 / (w + h) as f32;
        }
    }
    let (rg, rb) = (rr.clone(), rr.clone());
    let darken = |p: &[f32]| -> Vec<f32> { p.iter().map(|&v| v * 0.9).collect() };
    let (dr, dg, db) = (darken(&rr), darken(&rg), darken(&rb));

    // The faithful linear-planes path needs a whole-image instance —
    // butteraugli's `Auto` is strip-preferred, and the typed linear API
    // assumes whole-image (strip is rejected with StripModeUnsupported).
    let mut m = Metric::new_with_memory_mode(
        MetricKind::Butter,
        Backend::Cuda,
        w,
        h,
        MetricParams::default_for(MetricKind::Butter),
        MemoryMode::Full,
    )
    .expect("Metric::new_with_memory_mode butter Full");

    let multi = m
        .compute_from_linear_planes_multi(&rr, &rg, &rb, &dr, &dg, &db)
        .expect("compute_from_linear_planes_multi");
    assert_eq!(multi.metric_name, "butter");
    assert_eq!(multi.scores.len(), 2);
    assert_eq!(multi.scores[0].name, "max");
    assert_eq!(multi.scores[1].name, "pnorm_3");
    let max = multi.get("max").unwrap();
    let pnorm3 = multi.get("pnorm_3").unwrap();
    assert!(max.is_finite() && pnorm3.is_finite(), "{max} {pnorm3}");
    assert!(
        max > 0.0 && pnorm3 > 0.0,
        "perturbed planes should differ: max={max} pnorm3={pnorm3}"
    );
    assert!(pnorm3 <= max + 1e-6, "3-norm ≤ max-norm");
}

#[test]
fn butter_interleaved_multi_matches_planar() {
    // The non-planar entry point just deinterleaves before the planar dispatch,
    // so on identical data it must match the planar path bit-for-bit (within
    // GPU reduction noise).
    let (w, h) = (256u32, 256u32);
    let n = (w * h) as usize;
    let mut rr = vec![0.0f32; n];
    for y in 0..h as usize {
        for x in 0..w as usize {
            rr[y * w as usize + x] = 0.1 + 0.8 * (x + y) as f32 / (w + h) as f32;
        }
    }
    let (rg, rb) = (rr.clone(), rr.clone());
    let darken = |p: &[f32]| -> Vec<f32> { p.iter().map(|&v| v * 0.9).collect() };
    let (dr, dg, db) = (darken(&rr), darken(&rg), darken(&rb));
    let interleave = |r: &[f32], g: &[f32], b: &[f32]| -> Vec<f32> {
        (0..n).flat_map(|i| [r[i], g[i], b[i]]).collect()
    };
    let ref_il = interleave(&rr, &rg, &rb);
    let dis_il = interleave(&dr, &dg, &db);

    let mut m = Metric::new_with_memory_mode(
        MetricKind::Butter,
        Backend::Cuda,
        w,
        h,
        MetricParams::default_for(MetricKind::Butter),
        MemoryMode::Full,
    )
    .expect("Metric::new_with_memory_mode butter Full");

    let planar = m
        .compute_from_linear_planes_multi(&rr, &rg, &rb, &dr, &dg, &db)
        .expect("planar");
    let inter = m
        .compute_from_linear_interleaved_multi(&ref_il, &dis_il)
        .expect("interleaved");

    assert_eq!(inter.scores.len(), 2);
    assert_eq!(inter.metric_name, "butter");
    assert!(
        (planar.primary() - inter.primary()).abs() < 1e-6,
        "max: planar {} vs interleaved {}",
        planar.primary(),
        inter.primary()
    );
    assert!(
        (planar.get("pnorm_3").unwrap() - inter.get("pnorm_3").unwrap()).abs() < 1e-6,
        "pnorm_3 mismatch"
    );

    // A non-multiple-of-3 interleaved buffer errors cleanly.
    assert!(
        m.compute_from_linear_interleaved_multi(&[0.1, 0.2], &dis_il)
            .is_err(),
        "len not divisible by 3 must error"
    );
}

#[test]
fn butter_linear_planes_rejects_strip_mode() {
    // The typed linear-planes API is whole-image only; a strip instance must
    // be rejected LOUDLY (StripModeUnsupported), not silently return garbage
    // (huge / non-finite scores — the bug this guard fixes).
    let (w, h) = (256u32, 256u32);
    let z = vec![0.5f32; (w * h) as usize];
    let mut m = Metric::new_with_memory_mode(
        MetricKind::Butter,
        Backend::Cuda,
        w,
        h,
        MetricParams::default_for(MetricKind::Butter),
        MemoryMode::Strip { h_body: None },
    )
    .expect("Metric::new_with_memory_mode butter Strip");
    let r = m.compute_from_linear_planes_multi(&z, &z, &z, &z, &z, &z);
    assert!(
        r.is_err(),
        "strip-mode butter linear path must error, not return garbage"
    );
}

#[test]
fn zensim_multi_returns_score_plus_feature_vector() {
    let (w, h) = (256u32, 256u32);
    let (r, d) = pair(w, h);
    let mut m = Metric::new(
        MetricKind::Zensim,
        Backend::Cuda,
        w,
        h,
        MetricParams::default_for(MetricKind::Zensim),
    )
    .expect("Metric::new zensim");

    let multi = m
        .compute_srgb_u8_multi(&r, &d)
        .expect("compute_srgb_u8_multi");
    assert_eq!(multi.metric_name, "zensim");
    assert_eq!(multi.scores.len(), 1);
    assert_eq!(multi.scores[0].name, "zensim");
    // zensim is a feature extractor — the regime vector comes back too.
    assert!(
        matches!(multi.features.len(), 228 | 300 | 372),
        "expected a regime-length feature vector, got {}",
        multi.features.len()
    );
    assert!(
        multi.features.iter().all(|f| f.is_finite()),
        "all features finite"
    );
    assert!(multi.primary().is_finite());

    // Primary == the single-value path's score (same extraction + scoring).
    let single = m.compute_srgb_u8(&r, &d).expect("compute_srgb_u8");
    assert!(
        (multi.primary() - single.value).abs() < 1e-2,
        "multi.primary {} vs single {}",
        multi.primary(),
        single.value
    );
}
