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
use cvvdp_gpu::kernels::pyramid::{
    downscale_kernel, gausspyr_expand_scalar, gausspyr_reduce_scalar, subtract_kernel,
    upscale_h_kernel, upscale_v_kernel, weber_contrast_compute_kernel,
};

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

#[test]
fn upscale_two_kernels_match_host_scalar() {
    // 4×4 → 8×8 expand via the cvvdp interleave-with-edge-replicate
    // scheme. The GPU path runs upscale_v_kernel (4×4 → 4×8) then
    // upscale_h_kernel (4×8 → 8×8); the CPU reference is
    // gausspyr_expand_scalar (already locked vs pycvvdp).
    let client = Backend::client(&Default::default());

    let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let (sw, sh) = (4u32, 4u32);
    let (dw, dh) = (8u32, 8u32);
    let n_src = (sw * sh) as usize;
    let n_v = (sw * dh) as usize;
    let n_dst = (dw * dh) as usize;

    let src_h = client.create_from_slice(f32::as_bytes(&src));
    let vscratch_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n_v]));
    let dst_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n_dst]));

    let cube_dim = CubeDim::new_1d(64);
    let count_v = CubeCount::Static((n_v as u32).div_ceil(64), 1, 1);
    let count_h = CubeCount::Static((n_dst as u32).div_ceil(64), 1, 1);

    unsafe {
        upscale_v_kernel::launch::<Backend>(
            &client,
            count_v,
            cube_dim,
            ArrayArg::from_raw_parts(src_h.clone(), n_src),
            ArrayArg::from_raw_parts(vscratch_h.clone(), n_v),
            sw,
            sh,
            dh,
        );
        upscale_h_kernel::launch::<Backend>(
            &client,
            count_h,
            cube_dim,
            ArrayArg::from_raw_parts(vscratch_h.clone(), n_v),
            ArrayArg::from_raw_parts(dst_h.clone(), n_dst),
            sw,
            dw,
            dh,
        );
    }

    let dst_bytes = client.read_one(dst_h.clone()).expect("read dst");
    let gpu_out: &[f32] = f32::from_bytes(&dst_bytes);

    let mut cpu_out = Vec::new();
    gausspyr_expand_scalar(
        &src,
        sw as usize,
        sh as usize,
        dw as usize,
        dh as usize,
        &mut cpu_out,
    );

    let max_err = gpu_out
        .iter()
        .zip(&cpu_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_err < 1e-4,
        "GPU vs CPU scalar 2-kernel upscale max-abs error = {max_err}\n\
         gpu = {gpu_out:?}\n\
         cpu = {cpu_out:?}"
    );
}

#[test]
fn weber_contrast_compute_kernel_matches_host_formula() {
    let client = Backend::client(&Default::default());

    // Mixed inputs: layer goes negative + positive; l_bkg spans
    // tiny-clamp to large values; one layer/lbkg ratio over 1000
    // to exercise the upper clamp.
    let layer: Vec<f32> = vec![
        0.0, 1.0, -1.0, 5.0, -5.0, 50.0, -50.0, 500.0, -500.0, 1.0e5, -1.0e5, 0.001, -0.001,
        0.0,
    ];
    let lbkg: Vec<f32> = vec![
        1.0, 1.0, 1.0, 10.0, 100.0, 0.5, 0.005, 0.5, 0.001, 50.0, 50.0, 1.0, 0.001, 0.001,
    ];
    assert_eq!(layer.len(), lbkg.len());
    let n = layer.len();

    let layer_h = client.create_from_slice(f32::as_bytes(&layer));
    let lbkg_h = client.create_from_slice(f32::as_bytes(&lbkg));
    let contrast_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let log_l_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));

    let cube_dim = CubeDim::new_1d(64);
    let cube_count = CubeCount::Static((n as u32).div_ceil(64), 1, 1);
    unsafe {
        weber_contrast_compute_kernel::launch::<Backend>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(layer_h.clone(), n),
            ArrayArg::from_raw_parts(lbkg_h.clone(), n),
            ArrayArg::from_raw_parts(contrast_h.clone(), n),
            ArrayArg::from_raw_parts(log_l_h.clone(), n),
            n as u32,
        );
    }

    let c_bytes = client.read_one(contrast_h.clone()).expect("read contrast");
    let l_bytes = client.read_one(log_l_h.clone()).expect("read log_l");
    let c_gpu: &[f32] = f32::from_bytes(&c_bytes);
    let l_gpu: &[f32] = f32::from_bytes(&l_bytes);

    for i in 0..n {
        let l = lbkg[i].max(0.01);
        let c_exp = (layer[i] / l).clamp(-1000.0, 1000.0);
        let log_l_exp = l.log10();
        assert!(
            (c_gpu[i] - c_exp).abs() < 1e-4 * c_exp.abs().max(1e-3),
            "contrast[{i}]: got {} expected {} (layer={}, lbkg={})",
            c_gpu[i],
            c_exp,
            layer[i],
            lbkg[i]
        );
        assert!(
            (l_gpu[i] - log_l_exp).abs() < 1e-5,
            "log_l_bkg[{i}]: got {} expected {} (lbkg={})",
            l_gpu[i],
            log_l_exp,
            lbkg[i]
        );
    }
}

#[test]
fn subtract_kernel_produces_band_correctly() {
    // band[i] = fine[i] - upscaled_coarse[i]. Trivial but a regression
    // gate against accidentally swapping operand order or zeroing.
    let client = Backend::client(&Default::default());

    let fine: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let coarse: Vec<f32> = (0..64).map(|i| (i * 2) as f32).collect();
    let n = fine.len();

    let fine_h = client.create_from_slice(f32::as_bytes(&fine));
    let coarse_h = client.create_from_slice(f32::as_bytes(&coarse));
    let band_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));

    let cube_dim = CubeDim::new_1d(64);
    let cube_count = CubeCount::Static((n as u32).div_ceil(64), 1, 1);

    unsafe {
        subtract_kernel::launch::<Backend>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(fine_h.clone(), n),
            ArrayArg::from_raw_parts(coarse_h.clone(), n),
            ArrayArg::from_raw_parts(band_h.clone(), n),
            n as u32,
        );
    }

    let band_bytes = client.read_one(band_h.clone()).expect("read band");
    let band: &[f32] = f32::from_bytes(&band_bytes);
    for (i, &v) in band.iter().enumerate() {
        let expected = fine[i] - coarse[i];
        assert!(
            (v - expected).abs() < 1e-7,
            "band[{i}] = {v}, expected {expected}"
        );
    }
}
