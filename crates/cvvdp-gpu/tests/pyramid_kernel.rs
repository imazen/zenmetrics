//! GPU kernel parity for `downscale_kernel` against the host scalar.
//!
//! Launches the cubecl kernel on the selected runtime (cuda by
//! default, wgpu as fallback) for the same 8×8 ramp-with-peak input
//! used by `pyramid_scalar.rs`, then checks the 4×4 output matches
//! `gausspyr_reduce_scalar` (which is itself locked against pycvvdp
//! v0.5.4 at <1e-4 max-abs).
//!
//! cubecl-cpu is intentionally NOT selected here. Same reasoning as
//! `zensim_gpu`'s parity tests: the CPU runtime in 0.10.0-pre.4
//! handles a few of cubecl's slot-array launch geometries
//! inconsistently, which is unrelated to the kernel correctness we
//! want to verify.

#![cfg(any(feature = "cuda", feature = "wgpu"))]

use cubecl::Runtime;
use cubecl::prelude::*;
use cvvdp_gpu::kernels::pyramid::{downscale_kernel, gausspyr_reduce_scalar};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

#[rustfmt::skip]
const INPUT_8X8: [f32; 64] = [
    0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,
    4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0,
    8.0,  9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
   12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
   16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
   20.0, 21.0, 22.0, 24.0, 24.0, 25.0, 26.0, 27.0,
   24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
   28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0,
];

#[test]
fn downscale_kernel_matches_host_scalar() {
    let client = Backend::client(&Default::default());

    let (sw, sh) = (8u32, 8u32);
    let (dw, dh) = (4u32, 4u32);
    let n_src = (sw * sh) as usize;
    let n_dst = (dw * dh) as usize;

    let src = client.create_from_slice(f32::as_bytes(&INPUT_8X8));
    let dst = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n_dst]));

    let cube_dim = CubeDim::new_1d(64);
    let total_threads = (n_dst as u32).div_ceil(64);
    let cube_count = CubeCount::Static(total_threads, 1, 1);

    unsafe {
        downscale_kernel::launch::<Backend>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(src.clone(), n_src),
            ArrayArg::from_raw_parts(dst.clone(), n_dst),
            sw,
            sh,
            dw,
            dh,
        );
    }

    let dst_bytes = client.read_one(dst.clone()).expect("read dst");
    let gpu_out: &[f32] = f32::from_bytes(&dst_bytes);
    assert_eq!(gpu_out.len(), n_dst);

    let mut cpu_out = Vec::new();
    let (dw_s, dh_s) =
        gausspyr_reduce_scalar(&INPUT_8X8, sw as usize, sh as usize, &mut cpu_out);
    assert_eq!((dw_s, dh_s), (dw as usize, dh as usize));

    let max_err = gpu_out
        .iter()
        .zip(&cpu_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_err < 1e-5,
        "GPU vs CPU scalar downscale max-abs error = {max_err}\n\
         gpu = {gpu_out:?}\n\
         cpu = {cpu_out:?}"
    );
}
