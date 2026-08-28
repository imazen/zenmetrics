//! zenmetrics#30 — cooperative cancellation on the JOD dispatch:
//! `score_with_stop` / `compute_dkl_jod_with_stop` /
//! `compute_dkl_jod_with_warm_ref_with_stop` poll their `enough::Stop`
//! before each Weber stage, once per Mode B strip, and once per pyramid
//! level, returning `Error::Cancelled` (no score) when it fires, while
//! `Unstoppable` reproduces the plain entry points bit-for-bit.

#![cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]

use core::sync::atomic::{AtomicUsize, Ordering};

use almost_enough::Stopper;
use cubecl::Runtime;
use cvvdp_gpu::{Cvvdp, CvvdpParams, Error};
use enough::{Stop, StopReason};

use crate::common;
use common::{Backend, synth_pair_with_offset_dist};

/// A `Stop` that counts its polls and fires on the `cancel_at`-th one
/// (1-based; `usize::MAX` never fires). Lets a test pin *where* the
/// walker polls, not just that a pre-cancelled flag is honoured.
struct CountingStop {
    polls: AtomicUsize,
    cancel_at: usize,
}

impl CountingStop {
    fn live() -> Self {
        Self {
            polls: AtomicUsize::new(0),
            cancel_at: usize::MAX,
        }
    }
    fn firing_at(cancel_at: usize) -> Self {
        Self {
            polls: AtomicUsize::new(0),
            cancel_at,
        }
    }
    fn polls(&self) -> usize {
        self.polls.load(Ordering::Relaxed)
    }
}

impl Stop for CountingStop {
    fn check(&self) -> Result<(), StopReason> {
        let n = self.polls.fetch_add(1, Ordering::Relaxed) + 1;
        if n >= self.cancel_at {
            Err(StopReason::Cancelled)
        } else {
            Ok(())
        }
    }
}

const W: u32 = 256;
const H: u32 = 256;
// Mode B strip body: power of two, 4 strips at 256 rows → 4 polls in
// the strip-major walker (CUDA-only test, see below).
#[cfg(feature = "cuda")]
const BODY_H: u32 = 64;

fn pair() -> (Vec<u8>, Vec<u8>) {
    synth_pair_with_offset_dist(W as usize, H as usize)
}

#[test]
fn full_mode_pre_cancelled_stop_returns_cancelled() {
    let (r, d) = pair();
    let client = Backend::client(&Default::default());
    let mut c = Cvvdp::<Backend>::new(client, W, H, CvvdpParams::PLACEHOLDER).expect("new");
    let stop = Stopper::new();
    stop.cancel();
    let err = c
        .score_with_stop(&r, &d, &stop)
        .expect_err("a cancelled Stop must abort before the REF Weber stage");
    assert!(matches!(err, Error::Cancelled(_)), "got {err:?}");
    assert!(err.to_string().contains("cancelled"), "{err}");
    // The instance stays usable afterwards and the plain path agrees
    // bit-for-bit with the live-stopper path.
    let plain = c.score(&r, &d).expect("plain");
    let live = c
        .score_with_stop(&r, &d, &Stopper::new())
        .expect("live stopper");
    assert_eq!(plain.to_bits(), live.to_bits());
    let unstoppable = c
        .score_with_stop(&r, &d, &enough::Unstoppable)
        .expect("unstoppable");
    assert_eq!(plain.to_bits(), unstoppable.to_bits());
}

/// Full mode polls before the REF Weber stage, before the DIST Weber
/// stage, then once per pyramid level — so the 3rd poll is the first
/// level-loop checkpoint. Firing there proves the per-level poll exists
/// (a walker that only polled at the two stage boundaries would never
/// reach a 3rd poll and would return a score instead).
#[test]
fn full_mode_polls_per_pyramid_level() {
    let (r, d) = pair();
    let client = Backend::client(&Default::default());
    let mut c = Cvvdp::<Backend>::new(client, W, H, CvvdpParams::PLACEHOLDER).expect("new");
    let live = CountingStop::live();
    let plain = c
        .score_with_stop(&r, &d, &live)
        .expect("live counting stop");
    let n_polls = live.polls();
    assert!(
        n_polls >= 3,
        "expected ≥ 3 polls (2 stage boundaries + ≥ 1 pyramid level), got {n_polls}"
    );
    let third = CountingStop::firing_at(3);
    let err = c
        .score_with_stop(&r, &d, &third)
        .expect_err("the 3rd poll (first pyramid level) must be able to cancel");
    assert!(matches!(err, Error::Cancelled(_)), "got {err:?}");
    assert_eq!(third.polls(), 3, "the walk must stop at the firing poll");
    // Cancelling mid-walk leaves the instance reusable and the score
    // unchanged.
    let again = c.score(&r, &d).expect("reusable after mid-walk cancel");
    assert_eq!(plain.to_bits(), again.to_bits());
}

/// Mode B (StripPair) with several strips polls once per strip inside
/// the strip-major shallow walker, ahead of the per-level polls.
///
/// CUDA-only: the multi-strip Mode B walker panics inside wgpu on
/// Metal (`wgpu_core.rs` / cubecl `client.rs:105 CallError`) before any
/// of this crate's code runs — pre-existing, the same failure the
/// baseline `mode_b_walker_parity::mode_b_walker_jod_matches_full_at_*`
/// tests show on Metal (verified 2026-08-28, see CLAUDE.md Known Bugs).
/// Single-strip Mode B (64×64 / h_body 512) takes `k_split = 0` and
/// never enters the strip-major walker, so it cannot pin this poll.
#[cfg(feature = "cuda")]
#[test]
fn strip_pair_mode_polls_per_strip() {
    let (r, d) = pair();
    let client = Backend::client(&Default::default());
    let mut c = Cvvdp::<Backend>::new_strip_pair(client, W, H, BODY_H, CvvdpParams::PLACEHOLDER)
        .expect("new_strip_pair");
    assert!(c.is_strip_pair_mode());
    let full_client = Backend::client(&Default::default());
    let mut full = Cvvdp::<Backend>::new(full_client, W, H, CvvdpParams::PLACEHOLDER).expect("new");
    let full_live = CountingStop::live();
    full.score_with_stop(&r, &d, &full_live).expect("full live");
    let strip_live = CountingStop::live();
    let plain = c.score(&r, &d).expect("plain");
    let live = c
        .score_with_stop(&r, &d, &strip_live)
        .expect("strip live counting stop");
    assert_eq!(plain.to_bits(), live.to_bits());
    let n_strips = (H / BODY_H) as usize;
    assert!(
        strip_live.polls() >= full_live.polls() + n_strips - 1,
        "Mode B must add one poll per strip ({n_strips} strips) on top of the \
         stage + level polls: strip={} full={}",
        strip_live.polls(),
        full_live.polls()
    );
    let stop = Stopper::new();
    stop.cancel();
    let err = c
        .score_with_stop(&r, &d, &stop)
        .expect_err("a cancelled Stop must abort the strip-major walk");
    assert!(matches!(err, Error::Cancelled(_)), "got {err:?}");
}

#[test]
fn warm_ref_walk_honours_stop_and_keeps_the_reference() {
    let (r, d) = pair();
    let client = Backend::client(&Default::default());
    let mut c = Cvvdp::<Backend>::new(client, W, H, CvvdpParams::PLACEHOLDER).expect("new");
    c.warm_reference(&r).expect("warm_reference");
    let ppd = c.geometry_ppd_for_warm_ref();
    let plain = c
        .compute_dkl_jod_with_warm_ref(&d, ppd)
        .expect("plain warm");
    let stop = Stopper::new();
    stop.cancel();
    let err = c
        .compute_dkl_jod_with_warm_ref_with_stop(&d, ppd, &stop)
        .expect_err("a cancelled Stop must abort before the DIST Weber stage");
    assert!(matches!(err, Error::Cancelled(_)), "got {err:?}");
    // Warm state survives the cancellation and the score is unchanged.
    let again = c
        .compute_dkl_jod_with_warm_ref_with_stop(&d, ppd, &enough::Unstoppable)
        .expect("warm reference intact");
    assert_eq!(plain.to_bits(), again.to_bits());
}
