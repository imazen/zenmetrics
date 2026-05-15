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

#![cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]

use cubecl::Runtime;
use cubecl::prelude::*;
use cvvdp_gpu::kernels::pyramid::{
    downscale_kernel, gausspyr_expand_scalar, gausspyr_reduce_scalar, subtract_kernel,
    subtract_weber_3ch_kernel, upscale_h_kernel, upscale_v_kernel,
    weber_contrast_compute_3ch_kernel, weber_contrast_compute_kernel,
};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;
#[cfg(all(feature = "hip", not(feature = "cuda"), not(feature = "wgpu")))]
type Backend = cubecl::hip::HipRuntime;

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
    let (dw_s, dh_s) = gausspyr_reduce_scalar(&INPUT_8X8, sw as usize, sh as usize, &mut cpu_out);
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
        0.0, 1.0, -1.0, 5.0, -5.0, 50.0, -50.0, 500.0, -500.0, 1.0e5, -1.0e5, 0.001, -0.001, 0.0,
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
fn weber_contrast_compute_3ch_kernel_matches_per_channel_kernel() {
    // Fused 3-channel weber-contrast compute — verifies it produces
    // the same per-channel `contrast` output as three independent
    // launches of `weber_contrast_compute_kernel` AND the same
    // `log_l_bkg` field. A bug in the shared L_bkg compute would
    // show up identically across all three channels; a bug in the
    // per-channel layer math would show up only on the affected
    // channel — distinct per-channel inputs make the difference
    // detectable.
    let client = Backend::client(&Default::default());

    // Mirrors the single-channel parity test's edge-case coverage:
    // clamp triggers on both sides, tiny lbkg, large lbkg.
    let lbkg: Vec<f32> = vec![
        1.0, 1.0, 1.0, 10.0, 100.0, 0.5, 0.005, 0.5, 0.001, 50.0, 50.0, 1.0, 0.001, 0.001,
    ];
    let layer_a: Vec<f32> = vec![
        0.0, 1.0, -1.0, 5.0, -5.0, 50.0, -50.0, 500.0, -500.0, 1.0e5, -1.0e5, 0.001, -0.001, 0.0,
    ];
    // Distinct per-channel layer values so a "wrong-channel" bug
    // would mismatch.
    let layer_rg: Vec<f32> = layer_a.iter().map(|v| v * 0.5 + 0.25).collect();
    let layer_vy: Vec<f32> = layer_a.iter().map(|v| -v * 0.75 - 0.1).collect();
    let n = lbkg.len();

    let layer_a_h = client.create_from_slice(f32::as_bytes(&layer_a));
    let layer_rg_h = client.create_from_slice(f32::as_bytes(&layer_rg));
    let layer_vy_h = client.create_from_slice(f32::as_bytes(&layer_vy));
    let lbkg_h = client.create_from_slice(f32::as_bytes(&lbkg));
    let c_a_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let c_rg_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let c_vy_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let log_l_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));

    let cube_dim = CubeDim::new_1d(64);
    let cube_count = CubeCount::Static((n as u32).div_ceil(64), 1, 1);
    unsafe {
        weber_contrast_compute_3ch_kernel::launch::<Backend>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(layer_a_h.clone(), n),
            ArrayArg::from_raw_parts(layer_rg_h.clone(), n),
            ArrayArg::from_raw_parts(layer_vy_h.clone(), n),
            ArrayArg::from_raw_parts(lbkg_h.clone(), n),
            ArrayArg::from_raw_parts(c_a_h.clone(), n),
            ArrayArg::from_raw_parts(c_rg_h.clone(), n),
            ArrayArg::from_raw_parts(c_vy_h.clone(), n),
            ArrayArg::from_raw_parts(log_l_h.clone(), n),
            n as u32,
        );
    }

    let a_bytes = client.read_one(c_a_h).expect("read c_a");
    let rg_bytes = client.read_one(c_rg_h).expect("read c_rg");
    let vy_bytes = client.read_one(c_vy_h).expect("read c_vy");
    let l_bytes = client.read_one(log_l_h).expect("read log_l");
    let c_a_gpu: &[f32] = f32::from_bytes(&a_bytes);
    let c_rg_gpu: &[f32] = f32::from_bytes(&rg_bytes);
    let c_vy_gpu: &[f32] = f32::from_bytes(&vy_bytes);
    let l_gpu: &[f32] = f32::from_bytes(&l_bytes);

    let layers = [layer_a.as_slice(), layer_rg.as_slice(), layer_vy.as_slice()];
    let gpu = [c_a_gpu, c_rg_gpu, c_vy_gpu];
    for c in 0..3 {
        for i in 0..n {
            let l = lbkg[i].max(0.01);
            let exp = (layers[c][i] / l).clamp(-1000.0, 1000.0);
            assert!(
                (gpu[c][i] - exp).abs() < 1e-4 * exp.abs().max(1e-3),
                "channel {c} pixel {i}: got {} expected {} (layer={}, lbkg={})",
                gpu[c][i],
                exp,
                layers[c][i],
                lbkg[i],
            );
        }
    }
    for i in 0..n {
        let exp_log = lbkg[i].max(0.01).log10();
        assert!(
            (l_gpu[i] - exp_log).abs() < 1e-5,
            "log_l_bkg[{i}]: got {} expected {} (lbkg={})",
            l_gpu[i],
            exp_log,
            lbkg[i]
        );
    }
}

#[test]
fn subtract_weber_3ch_kernel_matches_subtract_then_weber() {
    // Fused subtract + 3-channel Weber-contrast — verifies that
    // `band[c] = clamp((fine[c] - upscaled[c]) / max(L_bkg, 0.01))`
    // matches doing subtract then weber separately, AND that the
    // shared log_l_bkg field matches log10(max(L_bkg, 0.01)).
    //
    // Distinct per-channel `fine` AND `upscaled` arrays so a
    // "wrong-channel" bug in the fused kernel is detectable.
    // Edge-case L_bkg coverage matches the single-channel weber
    // parity test.
    let client = Backend::client(&Default::default());

    let lbkg: Vec<f32> = vec![
        1.0, 1.0, 1.0, 10.0, 100.0, 0.5, 0.005, 0.5, 0.001, 50.0, 50.0, 1.0, 0.001, 0.001,
    ];
    let n = lbkg.len();

    let fine_a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.7 + 1.0).collect();
    let fine_rg: Vec<f32> = (0..n).map(|i| (i as f32) * -0.3 + 5.0).collect();
    let fine_vy: Vec<f32> = (0..n).map(|i| (i as f32) * 1.1 - 4.0).collect();

    // Cover both the upper and lower clamp on the resulting contrast
    // for at least one channel — pick upscaled_a so a few rows
    // produce |layer/L_bkg| > 1000.
    let upsc_a: Vec<f32> = vec![
        0.0, 0.0, 1001.0, 5.0, -5005.0, 50.0, -50.0, 500.0, -500.0, 1.0e5, -1.0e5, 0.001, -0.001,
        0.0,
    ];
    let upsc_rg: Vec<f32> = (0..n).map(|i| (i as f32) * 0.15 - 1.2).collect();
    let upsc_vy: Vec<f32> = (0..n).map(|i| (i as f32) * 0.42 + 0.7).collect();

    let fine_a_h = client.create_from_slice(f32::as_bytes(&fine_a));
    let fine_rg_h = client.create_from_slice(f32::as_bytes(&fine_rg));
    let fine_vy_h = client.create_from_slice(f32::as_bytes(&fine_vy));
    let upsc_a_h = client.create_from_slice(f32::as_bytes(&upsc_a));
    let upsc_rg_h = client.create_from_slice(f32::as_bytes(&upsc_rg));
    let upsc_vy_h = client.create_from_slice(f32::as_bytes(&upsc_vy));
    let lbkg_h = client.create_from_slice(f32::as_bytes(&lbkg));
    let c_a_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let c_rg_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let c_vy_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let log_l_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));

    let cube_dim = CubeDim::new_1d(64);
    let cube_count = CubeCount::Static((n as u32).div_ceil(64), 1, 1);
    unsafe {
        subtract_weber_3ch_kernel::launch::<Backend>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(fine_a_h, n),
            ArrayArg::from_raw_parts(fine_rg_h, n),
            ArrayArg::from_raw_parts(fine_vy_h, n),
            ArrayArg::from_raw_parts(upsc_a_h, n),
            ArrayArg::from_raw_parts(upsc_rg_h, n),
            ArrayArg::from_raw_parts(upsc_vy_h, n),
            ArrayArg::from_raw_parts(lbkg_h, n),
            ArrayArg::from_raw_parts(c_a_h.clone(), n),
            ArrayArg::from_raw_parts(c_rg_h.clone(), n),
            ArrayArg::from_raw_parts(c_vy_h.clone(), n),
            ArrayArg::from_raw_parts(log_l_h.clone(), n),
            n as u32,
        );
    }

    let c_a_gpu_b = client.read_one(c_a_h).expect("read c_a");
    let c_rg_gpu_b = client.read_one(c_rg_h).expect("read c_rg");
    let c_vy_gpu_b = client.read_one(c_vy_h).expect("read c_vy");
    let log_l_gpu_b = client.read_one(log_l_h).expect("read log_l");
    let c_a_gpu: &[f32] = f32::from_bytes(&c_a_gpu_b);
    let c_rg_gpu: &[f32] = f32::from_bytes(&c_rg_gpu_b);
    let c_vy_gpu: &[f32] = f32::from_bytes(&c_vy_gpu_b);
    let log_l_gpu: &[f32] = f32::from_bytes(&log_l_gpu_b);

    let fines = [fine_a.as_slice(), fine_rg.as_slice(), fine_vy.as_slice()];
    let upscs = [upsc_a.as_slice(), upsc_rg.as_slice(), upsc_vy.as_slice()];
    let gpus = [c_a_gpu, c_rg_gpu, c_vy_gpu];

    let mut saw_upper_clamp = false;
    let mut saw_lower_clamp = false;
    for c in 0..3 {
        for i in 0..n {
            let l = lbkg[i].max(0.01);
            let layer = fines[c][i] - upscs[c][i];
            let exp = (layer / l).clamp(-1000.0, 1000.0);
            if (exp - 1000.0).abs() < 1e-3 {
                saw_upper_clamp = true;
            }
            if (exp + 1000.0).abs() < 1e-3 {
                saw_lower_clamp = true;
            }
            assert!(
                (gpus[c][i] - exp).abs() < 1e-4 * exp.abs().max(1e-3),
                "channel {c} pixel {i}: got {} expected {} (fine={} upsc={} lbkg={})",
                gpus[c][i],
                exp,
                fines[c][i],
                upscs[c][i],
                lbkg[i],
            );
        }
    }
    for i in 0..n {
        let exp_log = lbkg[i].max(0.01).log10();
        assert!(
            (log_l_gpu[i] - exp_log).abs() < 1e-5,
            "log_l_bkg[{i}]: got {} expected {} (lbkg={})",
            log_l_gpu[i],
            exp_log,
            lbkg[i],
        );
    }
    assert!(saw_upper_clamp, "test inputs failed to exercise upper clamp");
    assert!(saw_lower_clamp, "test inputs failed to exercise lower clamp");
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
