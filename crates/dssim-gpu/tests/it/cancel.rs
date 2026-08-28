//! zenmetrics#30 — cooperative cancellation on the strip walkers:
//! `compute_stripped_with_stop` / `compute_with_reference_with_stop` poll
//! their `enough::Stop` once per strip (per pass) and return
//! `Error::Cancelled` (no score) when it fires, while `Unstoppable`
//! reproduces the plain entry points bit-for-bit.

use almost_enough::Stopper;
use cubecl::Runtime;
use dssim_gpu::{Dssim, Error};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;
#[cfg(all(feature = "cpu", not(feature = "cuda"), not(feature = "wgpu")))]
type Backend = cubecl::cpu::CpuRuntime;

fn client() -> cubecl::client::ComputeClient<Backend> {
    Backend::client(&Default::default())
}

fn gradient(w: usize, h: usize, seed: u8) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            v.push(((x * 255) / w) as u8);
            v.push(((y * 255) / h) as u8 ^ seed);
            v.push((((x + y) * 255) / (w + h)) as u8);
        }
    }
    v
}

const W: u32 = 128;
const H: u32 = 128;
const BODY_H: u32 = 32; // 4 strips → 4 polls per pass

#[test]
fn pre_cancelled_stop_returns_cancelled_before_any_strip() {
    let r = gradient(W as usize, H as usize, 0);
    let d = gradient(W as usize, H as usize, 0x20);
    let mut s = Dssim::<Backend>::new_strip(client(), W, H, BODY_H).expect("new_strip");
    let stop = Stopper::new();
    stop.cancel();
    let err = s
        .compute_stripped_with_stop(&r, &d, &stop)
        .expect_err("a cancelled Stop must abort the walk");
    assert!(matches!(err, Error::Cancelled(_)), "got {err:?}");
    assert!(err.to_string().contains("cancelled"), "{err}");
    // The instance stays usable afterwards.
    let ok = s
        .compute_stripped(&r, &d)
        .expect("instance reusable after cancellation");
    assert!(ok.score.is_finite());
}

#[test]
fn unstoppable_is_bit_identical_to_the_plain_entry() {
    let r = gradient(W as usize, H as usize, 0);
    let d = gradient(W as usize, H as usize, 0x20);
    let mut s = Dssim::<Backend>::new_strip(client(), W, H, BODY_H).expect("new_strip");
    let plain = s.compute_stripped(&r, &d).expect("plain");
    let stopped = s
        .compute_stripped_with_stop(&r, &d, &enough::Unstoppable)
        .expect("unstoppable");
    assert_eq!(plain.score.to_bits(), stopped.score.to_bits());
    let live = Stopper::new();
    let stopped2 = s
        .compute_with_stop(&r, &d, &live)
        .expect("live stopper via the routing entry");
    assert_eq!(plain.score.to_bits(), stopped2.score.to_bits());
}

#[test]
fn cached_reference_walk_honours_stop_and_keeps_the_reference() {
    let r = gradient(W as usize, H as usize, 0);
    let d = gradient(W as usize, H as usize, 0x20);
    let mut s = Dssim::<Backend>::new_strip(client(), W, H, BODY_H).expect("new_strip");
    s.set_reference(&r).expect("set_reference");
    let plain = s.compute_with_reference(&d).expect("plain warm");
    let stop = Stopper::new();
    stop.cancel();
    let err = s
        .compute_with_reference_with_stop(&d, &stop)
        .expect_err("a cancelled Stop must abort the warm walk");
    assert!(matches!(err, Error::Cancelled(_)), "got {err:?}");
    // Reference survives the cancellation and the score is unchanged.
    assert!(s.has_reference());
    let again = s
        .compute_with_reference_with_stop(&d, &enough::Unstoppable)
        .expect("reference intact");
    assert_eq!(plain.score.to_bits(), again.score.to_bits());
}

#[test]
fn whole_image_mode_polls_once_before_submission() {
    let r = gradient(W as usize, H as usize, 0);
    let d = gradient(W as usize, H as usize, 0x20);
    let mut s = Dssim::<Backend>::new(client(), W, H).expect("new");
    let stop = Stopper::new();
    stop.cancel();
    let err = s
        .compute_with_stop(&r, &d, &stop)
        .expect_err("a cancelled Stop must abort before the submission");
    assert!(matches!(err, Error::Cancelled(_)), "got {err:?}");
    let plain = s.compute(&r, &d).expect("plain");
    let live = s
        .compute_with_stop(&r, &d, &Stopper::new())
        .expect("live stopper");
    assert_eq!(plain.score.to_bits(), live.score.to_bits());
}
