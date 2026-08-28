//! zenmetrics#30 — cooperative cancellation through the umbrella:
//! `MetricInner::compute_srgb_u8_with_stop` /
//! `compute_with_reference_srgb_u8_with_stop` / `Metric::compute_pixels_with_stop`
//! surface a fired `enough::Stop` as the typed `Error::Cancelled` (never a
//! flattened `Error::Metric` string), leave the scorer + cached reference
//! usable, and are bit-identical to the plain entries under `Unstoppable`.

use almost_enough::Stopper;
use zenmetrics_api::{Backend, Error, Metric, MetricKind, MetricParams};

const W: u32 = 256;
const H: u32 = 256;

fn make_pair() -> (Vec<u8>, Vec<u8>) {
    let n = (W as usize) * (H as usize) * 3;
    let mut r = vec![0u8; n];
    for (i, b) in r.iter_mut().enumerate() {
        *b = (((i as u64).wrapping_mul(7919)) & 0xFF) as u8;
    }
    let mut d = vec![0u8; n];
    for (i, b) in d.iter_mut().enumerate() {
        *b = (((i as u64).wrapping_mul(2147483647)) & 0xFF) as u8;
    }
    (r, d)
}

fn cancelled() -> Stopper {
    let s = Stopper::new();
    s.cancel();
    s
}

/// Every GPU metric compiled into this build.
#[cfg(any(feature = "cuda", feature = "wgpu"))]
#[allow(clippy::vec_init_then_push)] // per-feature `cfg` pushes — no literal fits
fn gpu_kinds() -> Vec<MetricKind> {
    let mut v = Vec::new();
    #[cfg(feature = "cvvdp")]
    v.push(MetricKind::Cvvdp);
    #[cfg(feature = "butter")]
    v.push(MetricKind::Butter);
    #[cfg(feature = "ssim2")]
    v.push(MetricKind::Ssim2);
    #[cfg(feature = "dssim")]
    v.push(MetricKind::Dssim);
    #[cfg(feature = "iwssim")]
    v.push(MetricKind::Iwssim);
    #[cfg(feature = "zensim")]
    v.push(MetricKind::Zensim);
    v
}

#[cfg(feature = "cuda")]
const GPU_BACKEND: Backend = Backend::Cuda;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
const GPU_BACKEND: Backend = Backend::Wgpu;

#[cfg(any(feature = "cuda", feature = "wgpu"))]
#[test]
fn gpu_one_shot_pre_cancelled_stop_is_error_cancelled_for_every_metric() {
    let (r, d) = make_pair();
    for kind in gpu_kinds() {
        let mut m = Metric::new(kind, GPU_BACKEND, W, H, MetricParams::default_for(kind))
            .unwrap_or_else(|e| panic!("{kind:?}: new: {e}"));
        let err = m
            .compute_srgb_u8_with_stop(&r, &d, &cancelled())
            .err()
            .unwrap_or_else(|| panic!("{kind:?}: a cancelled Stop must abort"));
        assert!(matches!(err, Error::Cancelled(_)), "{kind:?}: got {err:?}");
        assert!(err.to_string().contains("cancelled"), "{kind:?}: {err}");
        // Scorer stays usable; Unstoppable + a live stopper are
        // bit-identical to the plain entry.
        let plain = m
            .compute_srgb_u8(&r, &d)
            .unwrap_or_else(|e| panic!("{kind:?}: {e}"));
        let unstoppable = m
            .compute_srgb_u8_with_stop(&r, &d, &zenmetrics_api::enough::Unstoppable)
            .unwrap_or_else(|e| panic!("{kind:?}: {e}"));
        assert_eq!(
            plain.value.to_bits(),
            unstoppable.value.to_bits(),
            "{kind:?}"
        );
        let live = m
            .compute_srgb_u8_with_stop(&r, &d, &Stopper::new())
            .unwrap_or_else(|e| panic!("{kind:?}: {e}"));
        assert_eq!(plain.value.to_bits(), live.value.to_bits(), "{kind:?}");
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
#[test]
fn gpu_cached_reference_pre_cancelled_stop_keeps_the_reference() {
    let (r, d) = make_pair();
    for kind in gpu_kinds() {
        let mut m = Metric::new(kind, GPU_BACKEND, W, H, MetricParams::default_for(kind))
            .unwrap_or_else(|e| panic!("{kind:?}: new: {e}"));
        m.set_reference_srgb_u8(&r)
            .unwrap_or_else(|e| panic!("{kind:?}: set_reference: {e}"));
        let plain = m
            .compute_with_reference_srgb_u8(&d)
            .unwrap_or_else(|e| panic!("{kind:?}: warm: {e}"));
        let err = m
            .compute_with_reference_srgb_u8_with_stop(&d, &cancelled())
            .err()
            .unwrap_or_else(|| panic!("{kind:?}: a cancelled Stop must abort the warm path"));
        assert!(matches!(err, Error::Cancelled(_)), "{kind:?}: got {err:?}");
        // The umbrella's `has_reference` is hard-wired to `false` for
        // zensim (pre-existing; `MetricInner::has_reference`), so the
        // survival proof for every kind is the warm re-score below —
        // it can only succeed if the cached reference is still there.
        if kind != MetricKind::Zensim {
            assert!(m.has_reference(), "{kind:?}: reference must survive");
        }
        let again = m
            .compute_with_reference_srgb_u8_with_stop(&d, &zenmetrics_api::enough::Unstoppable)
            .unwrap_or_else(|e| panic!("{kind:?}: {e}"));
        assert_eq!(plain.value.to_bits(), again.value.to_bits(), "{kind:?}");
    }
}

#[cfg(all(any(feature = "cuda", feature = "wgpu"), feature = "pixels"))]
#[test]
fn gpu_compute_pixels_with_stop_routes_the_sdr_path() {
    use zenpixels::{PixelDescriptor, PixelSlice};
    let (r, d) = make_pair();
    let row_bytes = (W as usize) * 3;
    // `PixelSlice` is a borrow, not `Copy` — rebuild per call.
    let rs = || PixelSlice::new(&r, W, H, row_bytes, PixelDescriptor::RGB8_SRGB).expect("ref");
    let ds = || PixelSlice::new(&d, W, H, row_bytes, PixelDescriptor::RGB8_SRGB).expect("dis");
    for kind in gpu_kinds() {
        let mut m = Metric::new(kind, GPU_BACKEND, W, H, MetricParams::default_for(kind))
            .unwrap_or_else(|e| panic!("{kind:?}: new: {e}"));
        let err = m
            .compute_pixels_with_stop(rs(), ds(), &cancelled())
            .err()
            .unwrap_or_else(|| panic!("{kind:?}: a cancelled Stop must abort"));
        assert!(matches!(err, Error::Cancelled(_)), "{kind:?}: got {err:?}");
        let plain = m
            .compute_pixels(rs(), ds())
            .unwrap_or_else(|e| panic!("{kind:?}: {e}"));
        let live = m
            .compute_pixels_with_stop(rs(), ds(), &Stopper::new())
            .unwrap_or_else(|e| panic!("{kind:?}: {e}"));
        assert_eq!(plain.value.to_bits(), live.value.to_bits(), "{kind:?}");
    }
}

/// The optimized-CPU backend has no internal checkpoint: it polls once
/// before scoring. Named with the `cpu_dispatch` substring so CI's
/// `cpu-metrics-tests` job filter (`-- backend_matrix cpu_dispatch`)
/// runs it on a GPU-less runner.
#[cfg(any(
    feature = "cpu-ssim2",
    feature = "cpu-cvvdp",
    feature = "cpu-dssim",
    feature = "cpu-butter",
    feature = "cpu-zensim",
    feature = "cpu-iwssim"
))]
#[test]
#[allow(clippy::vec_init_then_push)] // per-feature `cfg` pushes — no literal fits
fn cpu_dispatch_with_stop_polls_once_and_is_bit_identical_otherwise() {
    let (r, d) = make_pair();
    let mut kinds = Vec::new();
    #[cfg(feature = "cpu-ssim2")]
    kinds.push(MetricKind::Ssim2);
    #[cfg(feature = "cpu-dssim")]
    kinds.push(MetricKind::Dssim);
    #[cfg(feature = "cpu-butter")]
    kinds.push(MetricKind::Butter);
    #[cfg(feature = "cpu-zensim")]
    kinds.push(MetricKind::Zensim);
    #[cfg(feature = "cpu-cvvdp")]
    kinds.push(MetricKind::Cvvdp);
    #[cfg(feature = "cpu-iwssim")]
    kinds.push(MetricKind::Iwssim);
    for kind in kinds {
        let mut m = Metric::new(kind, Backend::Cpu, W, H, MetricParams::default_for(kind))
            .unwrap_or_else(|e| panic!("{kind:?}: new: {e}"));
        let err = m
            .compute_srgb_u8_with_stop(&r, &d, &cancelled())
            .err()
            .unwrap_or_else(|| panic!("{kind:?}: a cancelled Stop must abort"));
        assert!(matches!(err, Error::Cancelled(_)), "{kind:?}: got {err:?}");
        let plain = m
            .compute_srgb_u8(&r, &d)
            .unwrap_or_else(|e| panic!("{kind:?}: {e}"));
        let live = m
            .compute_srgb_u8_with_stop(&r, &d, &Stopper::new())
            .unwrap_or_else(|e| panic!("{kind:?}: {e}"));
        assert_eq!(plain.value.to_bits(), live.value.to_bits(), "{kind:?}");
        m.set_reference_srgb_u8(&r)
            .unwrap_or_else(|e| panic!("{kind:?}: set_reference: {e}"));
        let err = m
            .compute_with_reference_srgb_u8_with_stop(&d, &cancelled())
            .err()
            .unwrap_or_else(|| panic!("{kind:?}: a cancelled Stop must abort the warm path"));
        assert!(matches!(err, Error::Cancelled(_)), "{kind:?}: got {err:?}");
        assert!(m.has_reference(), "{kind:?}: reference must survive");
    }
}
