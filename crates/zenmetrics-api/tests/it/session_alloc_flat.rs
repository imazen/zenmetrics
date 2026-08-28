//! zenmetrics#47 acceptance gate 1 — "N scorings of same-shape pairs re-use
//! device buffers (allocation count flat after warmup)". The in-API
//! evidence is the session's own stream pool: `__stream_reserved_bytes`
//! (cubecl `memory_usage().bytes_reserved` for the session's stream) must
//! not grow once the first scoring has sized the working set. A per-pair
//! alloc/free pattern would show up as reserved bytes that keep climbing
//! (or churn) across the ladder; a re-used working set is flat.
//!
//! Runs on CUDA or Metal/wgpu — the pool accounting is backend-agnostic.

#![cfg(any(feature = "cuda", feature = "wgpu"))]

use zenmetrics_api::{Backend, MetricKind, MetricParams, MetricSession};

const W: u32 = 512;
const H: u32 = 512;
/// Ladder length; the first scoring sizes the pool, the rest must not.
const LADDER: usize = 6;

#[cfg(feature = "cuda")]
const BACKEND: Backend = Backend::Cuda;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
const BACKEND: Backend = Backend::Wgpu;

fn reference() -> Vec<u8> {
    let n = (W as usize) * (H as usize) * 3;
    let mut r = vec![0u8; n];
    for (i, b) in r.iter_mut().enumerate() {
        *b = ((i.wrapping_mul(2654435761usize)) >> 13) as u8;
    }
    r
}

/// A distinct distorted image per ladder rung (same shape, new bytes) so
/// the only thing that changes between scorings is the bitmap data.
fn distorted(r: &[u8], rung: usize) -> Vec<u8> {
    let amp = (rung as u8 + 1) * 3;
    r.iter()
        .enumerate()
        .map(|(i, &b)| b.wrapping_add(((i * (40503 + rung)) as u8) % amp))
        .collect()
}

#[allow(clippy::vec_init_then_push)] // per-feature `cfg` pushes — no literal fits
fn kinds() -> Vec<MetricKind> {
    let mut v = Vec::new();
    #[cfg(feature = "cvvdp")]
    v.push(MetricKind::Cvvdp);
    #[cfg(feature = "ssim2")]
    v.push(MetricKind::Ssim2);
    #[cfg(feature = "dssim")]
    v.push(MetricKind::Dssim);
    #[cfg(feature = "iwssim")]
    v.push(MetricKind::Iwssim);
    #[cfg(feature = "butter")]
    v.push(MetricKind::Butter);
    #[cfg(feature = "zensim")]
    v.push(MetricKind::Zensim);
    v
}

#[test]
fn warm_ref_ladder_keeps_the_session_pool_flat() {
    let r = reference();
    for kind in kinds() {
        let ctx = MetricSession::acquire(BACKEND).unwrap_or_else(|e| panic!("{kind:?}: {e}"));
        let stream = ctx.__stream_value();
        let mut m = ctx
            .into_metric(kind, W, H, MetricParams::default_for(kind))
            .unwrap_or_else(|e| panic!("{kind:?}: into_metric: {e}"));
        m.set_reference_srgb_u8(&r)
            .unwrap_or_else(|e| panic!("{kind:?}: set_reference: {e}"));
        let mut reserved_per_rung = Vec::with_capacity(LADDER);
        for rung in 0..LADDER {
            let d = distorted(&r, rung);
            let s = m
                .score_with_warm_ref(&d)
                .unwrap_or_else(|e| panic!("{kind:?}: rung {rung}: {e}"));
            assert!(s.value.is_finite(), "{kind:?}: rung {rung}: {}", s.value);
            let reserved = zenmetrics_api::__stream_reserved_bytes(BACKEND, stream)
                .unwrap_or_else(|| panic!("{kind:?}: stream_reserved_bytes returned None"));
            reserved_per_rung.push(reserved);
        }
        eprintln!(
            "{kind:?} {W}x{H} warm-ref ladder reserved bytes per rung: {reserved_per_rung:?}"
        );
        let after_first = reserved_per_rung[0];
        assert!(
            after_first > 0,
            "{kind:?}: the first scoring must size a working set"
        );
        for (rung, &reserved) in reserved_per_rung.iter().enumerate().skip(1) {
            assert_eq!(
                reserved, after_first,
                "{kind:?}: rung {rung} changed the session pool ({after_first} → {reserved} bytes); \
                 same-shape scorings must re-use the working set (zenmetrics#47 gate 1)"
            );
        }
        m.clear_reference();
    }
}
