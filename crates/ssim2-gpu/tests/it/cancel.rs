//! zenmetrics#30 — cooperative cancellation on the strip walker:
//! `compute_stripped_with_mode_and_stop` polls its `enough::Stop` once per
//! strip and returns `Error::Cancelled` (no score) when it fires, while
//! `Unstoppable` reproduces the plain entry point bit-for-bit.

use almost_enough::Stopper;
use cubecl::Runtime;
use ssim2_gpu::{Error, Ssim2, Ssim2Mode};

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
const BODY_H: u32 = 32; // 4 strips → 4 polls

#[test]
fn pre_cancelled_stop_returns_cancelled_before_any_strip() {
    let r = gradient(W as usize, H as usize, 0);
    let d = gradient(W as usize, H as usize, 0x20);
    let mut s = Ssim2::<Backend>::new_strip(client(), W, H, BODY_H).expect("new_strip");
    let stop = Stopper::new();
    stop.cancel();
    let err = s
        .compute_stripped_with_mode_and_stop(Ssim2Mode::default(), &r, &d, &stop)
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
    let mut s = Ssim2::<Backend>::new_strip(client(), W, H, BODY_H).expect("new_strip");
    let plain = s.compute_stripped(&r, &d).expect("plain");
    let stopped = s
        .compute_stripped_with_mode_and_stop(Ssim2Mode::default(), &r, &d, &enough::Unstoppable)
        .expect("unstoppable");
    assert_eq!(plain.score.to_bits(), stopped.score.to_bits());
    let live = Stopper::new();
    let stopped2 = s
        .compute_stripped_with_mode_and_stop(Ssim2Mode::default(), &r, &d, &live)
        .expect("live stopper");
    assert_eq!(plain.score.to_bits(), stopped2.score.to_bits());
}
