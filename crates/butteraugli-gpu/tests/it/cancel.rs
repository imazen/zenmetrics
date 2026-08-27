//! zenmetrics#30 — cooperative cancellation on the strip walkers: the
//! `*_with_stop` / `*_and_stop` entry points poll their `enough::Stop` once
//! per strip and return `Error::Cancelled` (no score) when it fires, while
//! `Unstoppable` reproduces the plain entry point bit-for-bit.

use almost_enough::Stopper;
use butteraugli_gpu::{Butteraugli, ButteraugliParams, Error};
use cubecl::Runtime;

#[cfg(feature = "cuda")]
type BackendT = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type BackendT = cubecl::wgpu::WgpuRuntime;
#[cfg(all(feature = "cpu", not(feature = "cuda"), not(feature = "wgpu")))]
type BackendT = cubecl::cpu::CpuRuntime;

fn client() -> cubecl::client::ComputeClient<BackendT> {
    BackendT::client(&Default::default())
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

const W: usize = 64;
const H: usize = 64;
const BODY_H: u32 = 16; // 4 strips → 4 polls

#[test]
fn pre_cancelled_stop_returns_cancelled_before_any_strip() {
    let r = gradient(W, H, 0);
    let d = gradient(W, H, 0x20);
    let mut b = Butteraugli::<BackendT>::new_strip(client(), W as u32, H as u32, BODY_H);
    let stop = Stopper::new();
    stop.cancel();
    let err = b
        .compute_strip_with_options_and_stop(&r, &d, &ButteraugliParams::default(), &stop)
        .expect_err("a cancelled Stop must abort the walk");
    assert!(matches!(err, Error::Cancelled(_)), "got {err:?}");
    assert!(err.to_string().contains("cancelled"), "{err}");
    // The instance stays usable afterwards.
    let ok = b
        .compute_strip(&r, &d)
        .expect("instance reusable after cancellation");
    assert!(ok.score.is_finite());
}

#[test]
fn unstoppable_is_bit_identical_to_the_plain_entry() {
    let r = gradient(W, H, 0);
    let d = gradient(W, H, 0x20);
    let mut b = Butteraugli::<BackendT>::new_strip(client(), W as u32, H as u32, BODY_H);
    let plain = b.compute_strip(&r, &d).expect("plain");
    let stopped = b
        .compute_strip_with_options_and_stop(
            &r,
            &d,
            &ButteraugliParams::default(),
            &enough::Unstoppable,
        )
        .expect("unstoppable");
    assert_eq!(plain.score.to_bits(), stopped.score.to_bits());
    assert_eq!(plain.pnorm_3.to_bits(), stopped.pnorm_3.to_bits());
    // A live-but-not-cancelled Stopper behaves like Unstoppable.
    let live = Stopper::new();
    let stopped2 = b
        .compute_strip_with_options_and_stop(&r, &d, &ButteraugliParams::default(), &live)
        .expect("live stopper");
    assert_eq!(plain.score.to_bits(), stopped2.score.to_bits());
}

#[test]
fn whole_image_with_reference_polls_once() {
    let r = gradient(W, H, 0);
    let d = gradient(W, H, 0x20);
    let mut b = Butteraugli::<BackendT>::new(client(), W as u32, H as u32);
    b.set_reference(&r).expect("set_reference");
    let stop = Stopper::new();
    stop.cancel();
    let err = b
        .compute_with_reference_with_stop(&d, &stop)
        .expect_err("cancelled");
    assert!(matches!(err, Error::Cancelled(_)), "got {err:?}");
    let plain = b.compute_with_reference(&d).expect("plain");
    let stopped = b
        .compute_with_reference_with_stop(&d, &enough::Unstoppable)
        .expect("unstoppable");
    assert_eq!(plain.score.to_bits(), stopped.score.to_bits());
}
