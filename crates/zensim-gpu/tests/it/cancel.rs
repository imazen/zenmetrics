//! zenmetrics#30 — cooperative cancellation on the strip walker:
//! `compute_with_reference_vec_with_stop` polls its `enough::Stop` once per
//! strip and returns `Error::Cancelled` (no features, cached reference
//! intact) when it fires, while `Unstoppable` reproduces the plain entry
//! point bit-for-bit.

use almost_enough::Stopper;
use cubecl::Runtime;
use zensim_gpu::{Error, Zensim};

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
fn pre_cancelled_stop_returns_cancelled_and_keeps_the_reference() {
    let r = gradient(W as usize, H as usize, 0);
    let d = gradient(W as usize, H as usize, 0x20);
    let mut z = Zensim::<Backend>::new_strip(client(), W, H, BODY_H).expect("new_strip");
    z.set_reference(&r).expect("set_reference");
    let stop = Stopper::new();
    stop.cancel();
    let err = z
        .compute_with_reference_vec_with_stop(&d, &stop)
        .expect_err("a cancelled Stop must abort the walk");
    assert!(matches!(err, Error::Cancelled(_)), "got {err:?}");
    assert!(err.to_string().contains("cancelled"), "{err}");
    // Reference still cached: the plain warm call succeeds without set_reference.
    let ok = z
        .compute_with_reference_vec(&d)
        .expect("reference intact after cancellation");
    assert_eq!(ok.len(), z.regime().total_features());
}

#[test]
fn unstoppable_is_bit_identical_to_the_plain_entry() {
    let r = gradient(W as usize, H as usize, 0);
    let d = gradient(W as usize, H as usize, 0x20);
    let mut z = Zensim::<Backend>::new_strip(client(), W, H, BODY_H).expect("new_strip");
    z.set_reference(&r).expect("set_reference");
    let plain = z.compute_with_reference_vec(&d).expect("plain");
    let stopped = z
        .compute_with_reference_vec_with_stop(&d, &enough::Unstoppable)
        .expect("unstoppable");
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&plain), bits(&stopped));
    let live = Stopper::new();
    let stopped2 = z
        .compute_with_reference_vec_with_stop(&d, &live)
        .expect("live stopper");
    assert_eq!(bits(&plain), bits(&stopped2));
}

#[test]
fn whole_image_polls_once_before_submission() {
    let r = gradient(W as usize, H as usize, 0);
    let d = gradient(W as usize, H as usize, 0x20);
    let mut z = Zensim::<Backend>::new(client(), W, H).expect("new");
    z.set_reference(&r).expect("set_reference");
    let stop = Stopper::new();
    stop.cancel();
    let err = z
        .compute_with_reference_vec_with_stop(&d, &stop)
        .expect_err("cancelled");
    assert!(matches!(err, Error::Cancelled(_)), "got {err:?}");
}
