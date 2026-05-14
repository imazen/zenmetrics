//! GPU kernel test for `weight_band_kernel` — verifies it
//! multiplies each band pixel by the indexed weight.

#![cfg(any(feature = "cuda", feature = "wgpu"))]

use cubecl::Runtime;
use cubecl::prelude::*;
use cvvdp_gpu::kernels::csf::{
    CsfChannel, csf_apply_3ch_kernel, csf_apply_per_pixel_kernel, precompute_logs_row,
    sensitivity_corrected_scalar, weight_band_kernel,
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

#[test]
fn csf_apply_3ch_kernel_matches_host() {
    // Fused 3-channel CSF kernel — verifies it produces the same
    // per-channel output as the per-channel kernel + host scalar.
    // The fused kernel shares LUT bracket math across channels, so
    // a bug in the shared portion would show as a 3-channel mismatch.
    let client = Backend::client(&Default::default());

    let n = 128usize;
    let log_l_bkg: Vec<f32> = (0..n)
        .map(|i| -2.3 + (i as f32 / (n - 1) as f32) * 6.3)
        .collect();
    let weber_a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
    let weber_rg: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 0.13 + 0.4).cos() * 0.3)
        .collect();
    let weber_vy: Vec<f32> = (0..n)
        .map(|i| ((i as f32) - 64.0) * 0.01)
        .collect();
    let rho = 4.0_f32;
    let channels = [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy];

    // Use unusual ch_gain values to make sure each channel reads
    // its own gain (a bug would have all 3 use the same gain).
    let ch_gain_a = 1.5_f32;
    let ch_gain_rg = 0.8_f32;
    let ch_gain_vy = 2.3_f32;
    let ch_gains = [ch_gain_a, ch_gain_rg, ch_gain_vy];

    let logs_rows = [
        precompute_logs_row(rho, channels[0]),
        precompute_logs_row(rho, channels[1]),
        precompute_logs_row(rho, channels[2]),
    ];

    let weber_a_h = client.create_from_slice(f32::as_bytes(&weber_a));
    let weber_rg_h = client.create_from_slice(f32::as_bytes(&weber_rg));
    let weber_vy_h = client.create_from_slice(f32::as_bytes(&weber_vy));
    let log_l_h = client.create_from_slice(f32::as_bytes(&log_l_bkg));
    let logs_row_a_h = client.create_from_slice(f32::as_bytes(&logs_rows[0]));
    let logs_row_rg_h = client.create_from_slice(f32::as_bytes(&logs_rows[1]));
    let logs_row_vy_h = client.create_from_slice(f32::as_bytes(&logs_rows[2]));
    let t_p_a_h = client.create_from_slice(f32::as_bytes(&[0.0_f32; 128]));
    let t_p_rg_h = client.create_from_slice(f32::as_bytes(&[0.0_f32; 128]));
    let t_p_vy_h = client.create_from_slice(f32::as_bytes(&[0.0_f32; 128]));

    let cube_dim = CubeDim::new_1d(64);
    let cube_count = CubeCount::Static((n as u32).div_ceil(64), 1, 1);

    unsafe {
        csf_apply_3ch_kernel::launch::<Backend>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(weber_a_h, n),
            ArrayArg::from_raw_parts(weber_rg_h, n),
            ArrayArg::from_raw_parts(weber_vy_h, n),
            ArrayArg::from_raw_parts(log_l_h, n),
            ArrayArg::from_raw_parts(logs_row_a_h, 32),
            ArrayArg::from_raw_parts(logs_row_rg_h, 32),
            ArrayArg::from_raw_parts(logs_row_vy_h, 32),
            ArrayArg::from_raw_parts(t_p_a_h.clone(), n),
            ArrayArg::from_raw_parts(t_p_rg_h.clone(), n),
            ArrayArg::from_raw_parts(t_p_vy_h.clone(), n),
            ch_gain_a,
            ch_gain_rg,
            ch_gain_vy,
            n as u32,
        );
    }

    let a_bytes = client.read_one(t_p_a_h).expect("read t_p_a");
    let rg_bytes = client.read_one(t_p_rg_h).expect("read t_p_rg");
    let vy_bytes = client.read_one(t_p_vy_h).expect("read t_p_vy");
    let gpu_a: &[f32] = f32::from_bytes(&a_bytes);
    let gpu_rg: &[f32] = f32::from_bytes(&rg_bytes);
    let gpu_vy: &[f32] = f32::from_bytes(&vy_bytes);

    let webers = [weber_a.as_slice(), weber_rg.as_slice(), weber_vy.as_slice()];
    let gpu_planes = [gpu_a, gpu_rg, gpu_vy];

    let mut max_rel_per_channel = [0.0_f32; 3];
    for c in 0..3 {
        for i in 0..n {
            let s = sensitivity_corrected_scalar(rho, log_l_bkg[i], channels[c]);
            let exp = webers[c][i] * s * ch_gains[c];
            let rel = ((gpu_planes[c][i] - exp) / exp.abs().max(1e-6)).abs();
            if rel > max_rel_per_channel[c] {
                max_rel_per_channel[c] = rel;
            }
            assert!(
                rel < 5e-3,
                "channel {c} pixel {i}: gpu={} exp={} rel={:.4e}",
                gpu_planes[c][i],
                exp,
                rel,
            );
        }
    }
    eprintln!(
        "max rel per channel: A={:.4e}, RG={:.4e}, VY={:.4e}",
        max_rel_per_channel[0], max_rel_per_channel[1], max_rel_per_channel[2],
    );
}
