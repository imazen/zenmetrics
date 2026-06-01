//! `score_pair` one-shot convenience — smoke + parity vs manual
//! `Metric::new` + `compute_srgb_u8` (the convenience must not change the
//! result). CUDA-gated (+ default cvvdp); NO GRACEFUL SKIPS.
#![cfg(all(feature = "cuda", feature = "cvvdp"))]

use zenmetrics_api::{Backend, Metric, MetricKind, MetricParams, score_pair};

#[test]
fn score_pair_matches_manual_new_compute() {
    // 256×256 is comfortably above cvvdp's pyramid minimum.
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

    let one_shot = score_pair(MetricKind::Cvvdp, Backend::Cuda, w, h, &r, &d).expect("score_pair");
    assert!(
        one_shot.value.is_finite(),
        "score_pair value must be finite"
    );

    // Same kind/backend/dims/default-params + same compute → same score,
    // within cvvdp's Atomic<f32> reduction-noise band (JOD is not
    // bit-reproducible under concurrent GPU work; ~1e-6).
    let mut m = Metric::new(
        MetricKind::Cvvdp,
        Backend::Cuda,
        w,
        h,
        MetricParams::default_for(MetricKind::Cvvdp),
    )
    .expect("Metric::new");
    let manual = m.compute_srgb_u8(&r, &d).expect("compute_srgb_u8");

    assert!(
        (one_shot.value - manual.value).abs() < 1e-3,
        "score_pair ({}) must match manual new+compute ({})",
        one_shot.value,
        manual.value
    );
}
