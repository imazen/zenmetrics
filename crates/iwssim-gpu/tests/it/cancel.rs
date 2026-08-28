//! zenmetrics#30 — cooperative cancellation on the strip walkers:
//! the `*_with_stop` entry points poll their `enough::Stop` once per
//! strip (per pass) and return `Error::Cancelled` (no score) when it
//! fires, while `Unstoppable` reproduces the plain entry points
//! bit-for-bit and a cancellation leaves the cached reference intact.

use almost_enough::Stopper;
use cubecl::Runtime;
use iwssim_gpu::{Error, Iwssim};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;
#[cfg(all(feature = "cpu", not(feature = "cuda"), not(feature = "wgpu")))]
type Backend = cubecl::cpu::CpuRuntime;

fn client() -> cubecl::client::ComputeClient<Backend> {
    Backend::client(&Default::default())
}

// ≥ 176 px on both axes (the 5-level IW-SSIM floor) and a strip
// allocation (`h_body + 2·halo` = 192) that clears the same floor; 4
// body strips of 128 rows → 4 polls per pass.
const W: u32 = 256;
const H: u32 = 512;
const BODY_H: u32 = 128;
const HALO: u32 = 32;

fn make_gray(seed: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity((W * H) as usize);
    for y in 0..H {
        for x in 0..W {
            let lf = ((x as f32 / W as f32) * 200.0) + ((y as f32 / H as f32) * 50.0);
            let hf = (((x * 7 + y * 13 + seed) % 17) as f32) * 1.5;
            out.push((lf + hf).clamp(0.0, 255.0));
        }
    }
    out
}

fn make_rgb(seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((W * H * 3) as usize);
    for y in 0..H {
        for x in 0..W {
            let v = ((x * 3 + y * 5 + seed) % 251) as u8;
            out.push(v);
            out.push(v.wrapping_add(17));
            out.push(v.wrapping_add(41));
        }
    }
    out
}

fn strip_instance() -> Iwssim<Backend> {
    Iwssim::<Backend>::new_strip_with_halo(client(), W, H, BODY_H, HALO).expect("new_strip")
}

#[test]
fn pre_cancelled_stop_returns_cancelled_before_any_strip() {
    let r = make_gray(0);
    let d = make_gray(9);
    let mut s = strip_instance();
    let stop = Stopper::new();
    stop.cancel();
    let err = s
        .compute_gray_stripped_with_stop(&r, &d, &stop)
        .expect_err("a cancelled Stop must abort the walk");
    assert!(matches!(err, Error::Cancelled(_)), "got {err:?}");
    assert!(err.to_string().contains("cancelled"), "{err}");
    // The instance stays usable afterwards.
    let ok = s
        .compute_gray_stripped(&r, &d)
        .expect("instance reusable after cancellation");
    assert!(ok.score.is_finite());
}

#[test]
fn unstoppable_is_bit_identical_to_the_plain_entry() {
    let r = make_gray(0);
    let d = make_gray(9);
    let mut s = strip_instance();
    let plain = s.compute_gray_stripped(&r, &d).expect("plain");
    let stopped = s
        .compute_gray_stripped_with_stop(&r, &d, &enough::Unstoppable)
        .expect("unstoppable");
    assert_eq!(plain.score.to_bits(), stopped.score.to_bits());
    let live = Stopper::new();
    let stopped2 = s
        .compute_gray_stripped_with_stop(&r, &d, &live)
        .expect("live stopper");
    assert_eq!(plain.score.to_bits(), stopped2.score.to_bits());
}

#[test]
fn cached_reference_walk_honours_stop_and_keeps_the_reference() {
    let r = make_gray(0);
    let d = make_gray(9);
    let mut s = strip_instance();
    s.set_reference_stripped(&r)
        .expect("set_reference_stripped");
    let plain = s.compute_with_reference_stripped(&d).expect("plain warm");
    let stop = Stopper::new();
    stop.cancel();
    let err = s
        .compute_with_reference_stripped_with_stop(&d, &stop)
        .expect_err("a cancelled Stop must abort the warm walk");
    assert!(matches!(err, Error::Cancelled(_)), "got {err:?}");
    assert!(s.has_cached_reference_stripped());
    let again = s
        .compute_with_reference_stripped_with_stop(&d, &enough::Unstoppable)
        .expect("reference intact");
    assert_eq!(plain.score.to_bits(), again.score.to_bits());
}

#[test]
fn rgb_native_cached_reference_walk_honours_stop() {
    let r = make_rgb(0);
    let d = make_rgb(9);
    let mut s = strip_instance();
    s.set_rgb_reference_stripped(&r)
        .expect("set_rgb_reference_stripped");
    let plain = s
        .compute_rgb_with_reference_stripped_native(&d)
        .expect("plain native warm");
    let stop = Stopper::new();
    stop.cancel();
    let err = s
        .compute_rgb_with_reference_stripped_native_with_stop(&d, &stop)
        .expect_err("a cancelled Stop must abort the native warm walk");
    assert!(matches!(err, Error::Cancelled(_)), "got {err:?}");
    let err2 = s
        .compute_rgb_with_reference_stripped_with_stop(&d, &stop)
        .expect_err("a cancelled Stop must abort the gray-converted warm walk");
    assert!(matches!(err2, Error::Cancelled(_)), "got {err2:?}");
    let err3 = s
        .compute_rgb_stripped_with_stop(&r, &d, &stop)
        .expect_err("a cancelled Stop must abort the rgb strip walk");
    assert!(matches!(err3, Error::Cancelled(_)), "got {err3:?}");
    assert!(s.has_cached_reference_stripped());
    let again = s
        .compute_rgb_with_reference_stripped_native_with_stop(&d, &enough::Unstoppable)
        .expect("reference intact");
    assert_eq!(plain.score.to_bits(), again.score.to_bits());
}
