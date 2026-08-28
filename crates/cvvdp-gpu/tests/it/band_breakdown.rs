//! zenmetrics#14 "output completeness" — `Cvvdp::score_with_band_breakdown`
//! exposes upstream pycvvdp's still-image `Q_per_ch` (per pyramid level ×
//! DKL channel pooled band scores) next to a JOD that is bit-identical to
//! `Cvvdp::score`, and the breakdown reproduces that JOD through the same
//! host finalizer the pipeline uses.

#![cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]

use cubecl::Runtime;
use cvvdp_gpu::kernels::pool::do_pooling_and_jod_still_3ch;
use cvvdp_gpu::memory_mode::STRIP_H_BODY_DEFAULT;
use cvvdp_gpu::{Cvvdp, CvvdpParams, Error, N_CHANNELS};

use crate::common;
use common::{Backend, synth_pair_with_offset_dist};

/// Atomic-reorder noise band for a repeated JOD on the same instance.
const JOD_TOL: f64 = 1e-4;

fn check_breakdown(c: &mut Cvvdp<Backend>, r: &[u8], d: &[u8]) {
    let plain = c.score(r, d).expect("score");
    let bb = c.score_with_band_breakdown(r, d).expect("breakdown");
    // Same dispatches, same pooling kernels — but the pool reduces with
    // f32 atomics, so two runs differ by reorder noise (the documented
    // ~1e-4 JOD band, `PARITY_TOL_JOD` in strip_mode_b_parity.rs), not
    // by bits.
    assert!(
        (bb.jod - plain).abs() <= JOD_TOL,
        "breakdown jod {} must match score {} within {JOD_TOL}",
        bb.jod,
        plain
    );
    assert!(
        bb.q_per_ch.len() >= 2,
        "expected ≥ 2 pyramid levels, got {}",
        bb.q_per_ch.len()
    );
    for (k, q) in bb.q_per_ch.iter().enumerate() {
        for (ch, &v) in q.iter().enumerate() {
            assert!(v.is_finite() && v >= 0.0, "level {k} channel {ch}: {v}");
        }
    }
    assert!(
        bb.q_per_ch.iter().flatten().any(|&v| v > 0.0),
        "a distorted pair must light up at least one band"
    );
    // The breakdown is exactly the finalizer's input.
    let refin = f64::from(do_pooling_and_jod_still_3ch(&bb.q_per_ch));
    assert_eq!(refin.to_bits(), bb.jod.to_bits(), "{refin} vs {}", bb.jod);
    let _ = N_CHANNELS;
}

#[test]
fn full_mode_breakdown_matches_score_and_refinalizes() {
    let (r, d) = synth_pair_with_offset_dist(256, 256);
    let client = Backend::client(&Default::default());
    let mut c = Cvvdp::<Backend>::new(client, 256, 256, CvvdpParams::PLACEHOLDER).expect("new");
    check_breakdown(&mut c, &r, &d);
    // Repeated calls on the same instance agree to atomic-reorder noise
    // and keep the same shape.
    let a = c.score_with_band_breakdown(&r, &d).expect("a");
    let b = c.score_with_band_breakdown(&r, &d).expect("b");
    assert_eq!(a.q_per_ch.len(), b.q_per_ch.len());
    assert!((a.jod - b.jod).abs() <= JOD_TOL, "{} vs {}", a.jod, b.jod);
    for (qa, qb) in a.q_per_ch.iter().zip(&b.q_per_ch) {
        for (x, y) in qa.iter().zip(qb) {
            assert!((x - y).abs() <= 1e-3 * x.abs().max(1e-3), "{x} vs {y}");
        }
    }
}

#[test]
fn strip_pair_mode_breakdown_matches_score() {
    // Single-strip Mode B (the multi-strip walker is a known Metal
    // failure — see CLAUDE.md Known Bugs); exercises the strip pool path.
    let (r, d) = synth_pair_with_offset_dist(64, 64);
    let client = Backend::client(&Default::default());
    let mut c = Cvvdp::<Backend>::new_strip_pair(
        client,
        64,
        64,
        STRIP_H_BODY_DEFAULT,
        CvvdpParams::PLACEHOLDER,
    )
    .expect("new_strip_pair");
    assert!(c.is_strip_pair_mode());
    check_breakdown(&mut c, &r, &d);
}

#[test]
fn identical_inputs_short_circuit_to_ten_with_zero_bands() {
    let (r, _) = synth_pair_with_offset_dist(64, 64);
    let client = Backend::client(&Default::default());
    let mut c = Cvvdp::<Backend>::new(client, 64, 64, CvvdpParams::PLACEHOLDER).expect("new");
    let bb = c.score_with_band_breakdown(&r, &r).expect("identity");
    assert_eq!(bb.jod, 10.0);
    assert!(!bb.q_per_ch.is_empty());
    assert!(bb.q_per_ch.iter().flatten().all(|&v| v == 0.0));
    assert_eq!(c.score(&r, &r).expect("score identity"), 10.0);
}

#[test]
fn breakdown_rejects_dimension_mismatch_before_dispatch() {
    let (r, d) = synth_pair_with_offset_dist(64, 64);
    let client = Backend::client(&Default::default());
    let mut c = Cvvdp::<Backend>::new(client, 64, 64, CvvdpParams::PLACEHOLDER).expect("new");
    let err = c
        .score_with_band_breakdown(&r[..r.len() - 3], &d)
        .expect_err("short ref must be rejected");
    assert!(matches!(err, Error::DimensionMismatch { .. }), "{err:?}");
}
