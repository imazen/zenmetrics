//! GPU kernel test for `weight_band_kernel` — verifies it
//! multiplies each band pixel by the indexed weight.

#![cfg(any(feature = "cuda", feature = "wgpu"))]

use cubecl::Runtime;
use cubecl::prelude::*;
use cvvdp_gpu::kernels::csf::{
    CsfChannel, csf_apply_per_pixel_kernel, precompute_logs_row, sensitivity_corrected_scalar,
    weight_band_kernel,
};
use cvvdp_gpu::kernels::masking::CH_GAIN;

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

#[test]
fn weight_band_kernel_scales_in_place() {
    let client = Backend::client(&Default::default());
    let band_in: Vec<f32> = (0..32).map(|i| (i as f32) * 0.5 - 4.0).collect();
    let weights = vec![1.7_f32, 0.25, -2.0];
    let weight_idx = 1u32;
    let n = band_in.len();

    let band_h = client.create_from_slice(f32::as_bytes(&band_in));
    let weights_h = client.create_from_slice(f32::as_bytes(&weights));

    let cube_dim = CubeDim::new_1d(64);
    let cube_count = CubeCount::Static((n as u32).div_ceil(64), 1, 1);

    unsafe {
        weight_band_kernel::launch::<Backend>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(band_h.clone(), n),
            ArrayArg::from_raw_parts(weights_h.clone(), weights.len()),
            weight_idx,
            n as u32,
        );
    }

    let out_bytes = client.read_one(band_h.clone()).expect("read band");
    let out: &[f32] = f32::from_bytes(&out_bytes);
    let scale = weights[weight_idx as usize];
    for (i, (&got, &orig)) in out.iter().zip(&band_in).enumerate() {
        let expected = orig * scale;
        assert!(
            (got - expected).abs() < 1e-6,
            "band[{i}]: got {got}, expected {expected}"
        );
    }
}

#[test]
fn csf_apply_per_pixel_kernel_matches_host() {
    let client = Backend::client(&Default::default());

    // 64 pixels, varied log_l_bkg covering axis range.
    let n = 64usize;
    let log_l_bkg: Vec<f32> = (0..n)
        .map(|i| -2.3 + (i as f32 / (n - 1) as f32) * 6.3)
        .collect();
    let weber: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
    let rho = 4.0_f32;
    let cc = CsfChannel::Rg;
    let cc_idx = 1; // RG = channels[1]
    let ch_gain = CH_GAIN[cc_idx];

    let logs_row = precompute_logs_row(rho, cc);

    let weber_h = client.create_from_slice(f32::as_bytes(&weber));
    let log_l_h = client.create_from_slice(f32::as_bytes(&log_l_bkg));
    let logs_row_h = client.create_from_slice(f32::as_bytes(&logs_row));
    let t_p_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));

    let cube_dim = CubeDim::new_1d(64);
    let cube_count = CubeCount::Static((n as u32).div_ceil(64), 1, 1);

    unsafe {
        csf_apply_per_pixel_kernel::launch::<Backend>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(weber_h.clone(), n),
            ArrayArg::from_raw_parts(log_l_h.clone(), n),
            ArrayArg::from_raw_parts(logs_row_h.clone(), logs_row.len()),
            ArrayArg::from_raw_parts(t_p_h.clone(), n),
            ch_gain,
            n as u32,
        );
    }

    let bytes = client.read_one(t_p_h.clone()).expect("read t_p");
    let gpu: &[f32] = f32::from_bytes(&bytes);

    let mut max_rel = 0.0_f32;
    for i in 0..n {
        let s = sensitivity_corrected_scalar(rho, log_l_bkg[i], cc);
        let exp = weber[i] * s * ch_gain;
        let rel = ((gpu[i] - exp) / exp.abs().max(1e-6)).abs();
        if rel > max_rel {
            max_rel = rel;
        }
        assert!(
            rel < 1e-3,
            "pixel {i}: gpu={} exp={} rel={:.4e} log_l={:.3} weber={:.3}",
            gpu[i],
            exp,
            rel,
            log_l_bkg[i],
            weber[i]
        );
    }
    eprintln!("max rel error: {max_rel:.4e}");
}
