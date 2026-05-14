//! Integration test for `Cvvdp::compute_dkl_planes` — exercises the
//! upload + LUT-init + color-kernel path end-to-end through the
//! pipeline. Compares against the host scalar `srgb_byte_to_dkl_scalar`.

#![cfg(any(feature = "cuda", feature = "wgpu"))]

use cubecl::Runtime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::kernels::color::srgb_byte_to_dkl_scalar;
use cvvdp_gpu::params::{CvvdpParams, DisplayModel};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

#[test]
fn compute_dkl_planes_matches_host_scalar() {
    let client = Backend::client(&Default::default());
    let (w, h) = (16u32, 16u32);
    let n = (w * h) as usize;
    let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
        .expect("new Cvvdp");

    // Non-trivial RGB pattern.
    let mut srgb = Vec::with_capacity(n * 3);
    for i in 0..n {
        srgb.push((i % 251) as u8);
        srgb.push(((i * 7 + 13) % 251) as u8);
        srgb.push(((i * 19 + 41) % 251) as u8);
    }

    let [a, rg, vy] = cvvdp.compute_dkl_planes(&srgb).expect("compute_dkl_planes");
    assert_eq!(a.len(), n);
    assert_eq!(rg.len(), n);
    assert_eq!(vy.len(), n);

    let display = DisplayModel::STANDARD_4K;
    let mut max_err = 0.0_f32;
    for i in 0..n {
        let (ea, erg, evy) = srgb_byte_to_dkl_scalar(
            srgb[i * 3],
            srgb[i * 3 + 1],
            srgb[i * 3 + 2],
            display.y_peak,
            display.y_black,
            display.y_refl,
        );
        for d in [(a[i] - ea).abs(), (rg[i] - erg).abs(), (vy[i] - evy).abs()] {
            if d > max_err {
                max_err = d;
            }
        }
    }
    // 3e-5 absolute — same FMA-vs-non-FMA slack as the kernel-only
    // test in color_kernel.rs (DKL output magnitudes ~200, 1 ULP ≈ 1.5e-5).
    assert!(
        max_err < 3e-5,
        "compute_dkl_planes vs host scalar max-abs = {max_err}"
    );
}
