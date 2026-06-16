//! Tick 486: pin the independence of the two reference-side caches
//! on `Cvvdp` (`set_reference` / `score_with_reference` cache vs
//! `warm_reference` / `compute_dkl_jod_with_warm_ref` cache) and
//! non-pollution from the one-shot scoring methods.
//!
//! Existing pins (tick 238, 249, 314, 315) prove the two caches do
//! not invalidate or clobber each other. These tests pin the duals:
//!
//! - A fresh `Cvvdp::new()` has BOTH caches empty — `score_with_reference`
//!   surfaces `NoCachedReference` and `compute_dkl_jod_with_warm_ref`
//!   surfaces `NoWarmReference`, independently.
//! - `set_reference` does NOT prime warm state.
//! - `warm_reference` does NOT prime the set_reference cache.
//! - The one-shot `score` / `compute_dkl_jod` paths do NOT pollute
//!   either cache as a side effect.
//!
//! Catches a future refactor that, say, makes `set_reference`
//! eagerly upload to the warm-state buffers (cheaper if the call is
//! followed by `compute_dkl_jod_with_warm_ref` but silently changes
//! the documented surface) — the no-cache fast paths would then
//! return a JOD instead of `NoCachedReference` / `NoWarmReference`.

#![cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]

use cubecl::Runtime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};

use crate::common;

use common::Backend;

fn make_cvvdp(w: u32, h: u32) -> Cvvdp<Backend> {
    let client = Backend::client(&Default::default());
    Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp")
}

fn synth_pair(w: u32, h: u32) -> (Vec<u8>, Vec<u8>) {
    let n = (w * h * 3) as usize;
    let r = vec![128u8; n];
    let d: Vec<u8> = r.iter().map(|b| b.saturating_add(8)).collect();
    (r, d)
}

#[test]
fn fresh_cvvdp_has_empty_caches() {
    // Both fast paths MUST surface their respective no-cache error
    // immediately after `Cvvdp::new()`, before any priming call.
    let (w, h) = (64u32, 64u32);
    let mut cvvdp = make_cvvdp(w, h);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let (_, dist_bytes) = synth_pair(w, h);

    let swr_err = cvvdp
        .score_with_reference(&dist_bytes)
        .expect_err("fresh Cvvdp must error on score_with_reference");
    match swr_err {
        cvvdp_gpu::Error::NoCachedReference => {}
        other => panic!("expected NoCachedReference on fresh Cvvdp, got {other:?}"),
    }

    let warm_err = cvvdp
        .compute_dkl_jod_with_warm_ref(&dist_bytes, ppd)
        .expect_err("fresh Cvvdp must error on compute_dkl_jod_with_warm_ref");
    match warm_err {
        cvvdp_gpu::Error::NoWarmReference => {}
        other => panic!("expected NoWarmReference on fresh Cvvdp, got {other:?}"),
    }
}

#[test]
fn set_reference_does_not_prime_warm_state() {
    // Tick 486: dual of `set_reference_does_not_invalidate_warm_state`
    // (tick 238). After ONLY `set_reference` (no `warm_reference`),
    // the warm-ref fast path MUST still fail with `NoWarmReference`.
    // The two caches are independent: set_reference only touches the
    // host-side `cached_ref` buffer, not the GPU-side warm scalar /
    // bands_ref.
    let (w, h) = (64u32, 64u32);
    let mut cvvdp = make_cvvdp(w, h);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let (ref_bytes, dist_bytes) = synth_pair(w, h);

    cvvdp.set_reference(&ref_bytes).expect("set_reference");

    let err = cvvdp
        .compute_dkl_jod_with_warm_ref(&dist_bytes, ppd)
        .expect_err("warm-ref fast path must error after set_reference only");
    match err {
        cvvdp_gpu::Error::NoWarmReference => {}
        other => {
            panic!(
                "set_reference must not prime warm state; expected NoWarmReference, got {other:?}"
            )
        }
    }

    // The set_reference cache itself is primed — score_with_reference
    // should succeed.
    let jod = cvvdp
        .score_with_reference(&dist_bytes)
        .expect("score_with_reference must succeed after set_reference");
    assert!(
        jod.is_finite(),
        "score_with_reference JOD must be finite, got {jod}"
    );
}

#[test]
fn warm_reference_does_not_prime_set_reference_cache() {
    // Tick 486: dual direction — after ONLY `warm_reference` (no
    // `set_reference`), `score_with_reference` MUST surface
    // `NoCachedReference`. The warm-ref priming touches GPU
    // bands_ref / log_l_bkg scalar but NOT the host-side
    // cached_ref Option that score_with_reference reads.
    let (w, h) = (64u32, 64u32);
    let mut cvvdp = make_cvvdp(w, h);
    let (ref_bytes, dist_bytes) = synth_pair(w, h);

    cvvdp.warm_reference(&ref_bytes).expect("warm_reference");

    let err = cvvdp
        .score_with_reference(&dist_bytes)
        .expect_err("score_with_reference must error after warm_reference only");
    match err {
        cvvdp_gpu::Error::NoCachedReference => {}
        other => panic!(
            "warm_reference must not prime set_reference cache; expected NoCachedReference, \
             got {other:?}"
        ),
    }

    // The warm cache itself is primed — warm-ref fast path should
    // succeed.
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let jod = cvvdp
        .compute_dkl_jod_with_warm_ref(&dist_bytes, ppd)
        .expect("compute_dkl_jod_with_warm_ref must succeed after warm_reference");
    assert!(jod.is_finite(), "warm-ref JOD must be finite, got {jod}");
}

#[test]
fn score_does_not_pollute_caches() {
    // Tick 486: pin that the one-shot `Cvvdp::score` does NOT prime
    // either cache as a side effect. A future "optimization" that
    // wrote the REF buffer into `cached_ref` so the next
    // `score_with_reference` call could skip the upload would change
    // the documented surface; this test catches it.
    let (w, h) = (64u32, 64u32);
    let mut cvvdp = make_cvvdp(w, h);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let (ref_bytes, dist_bytes) = synth_pair(w, h);

    let _ = cvvdp.score(&ref_bytes, &dist_bytes).expect("score");

    // Neither fast path should succeed yet.
    let swr_err = cvvdp
        .score_with_reference(&dist_bytes)
        .expect_err("score must not prime set_reference cache");
    match swr_err {
        cvvdp_gpu::Error::NoCachedReference => {}
        other => panic!("after score(): expected NoCachedReference, got {other:?}"),
    }

    let warm_err = cvvdp
        .compute_dkl_jod_with_warm_ref(&dist_bytes, ppd)
        .expect_err("score must not prime warm state");
    match warm_err {
        cvvdp_gpu::Error::NoWarmReference => {}
        other => panic!("after score(): expected NoWarmReference, got {other:?}"),
    }
}

#[test]
fn compute_dkl_jod_does_not_pollute_caches() {
    // Tick 486: same pin as `score_does_not_pollute_caches` but for
    // the f32-returning `compute_dkl_jod` one-shot. The two paths
    // share most internals but `compute_dkl_jod` is the one that
    // batch consumers (e.g. zenmetrics CLI) reach for; if a
    // refactor accidentally added cache-priming there, batch callers
    // that subsequently primed via the documented `warm_reference`
    // path could silently get stale state.
    let (w, h) = (64u32, 64u32);
    let mut cvvdp = make_cvvdp(w, h);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let (ref_bytes, dist_bytes) = synth_pair(w, h);

    let _ = cvvdp
        .compute_dkl_jod(&ref_bytes, &dist_bytes, ppd)
        .expect("compute_dkl_jod");

    let swr_err = cvvdp
        .score_with_reference(&dist_bytes)
        .expect_err("compute_dkl_jod must not prime set_reference cache");
    match swr_err {
        cvvdp_gpu::Error::NoCachedReference => {}
        other => panic!("after compute_dkl_jod(): expected NoCachedReference, got {other:?}"),
    }

    let warm_err = cvvdp
        .compute_dkl_jod_with_warm_ref(&dist_bytes, ppd)
        .expect_err("compute_dkl_jod must not prime warm state");
    match warm_err {
        cvvdp_gpu::Error::NoWarmReference => {}
        other => panic!("after compute_dkl_jod(): expected NoWarmReference, got {other:?}"),
    }
}
