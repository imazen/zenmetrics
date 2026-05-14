//! GPU kernel parity for `mult_mutual_3ch_no_blur_kernel` against
//! the host scalar `mult_mutual_band` at small band sizes (no PU
//! Gaussian blur).

#![cfg(any(feature = "cuda", feature = "wgpu"))]

use cubecl::Runtime;
use cubecl::prelude::*;
use cvvdp_gpu::kernels::masking::{
    gaussian_blur_sigma3, mult_mutual_3ch_no_blur_kernel, mult_mutual_band, pu_blur_h_kernel,
    pu_blur_v_kernel,
};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

#[test]
fn mult_mutual_3ch_no_blur_matches_host_scalar() {
    let client = Backend::client(&Default::default());

    // 4×4 band — below PU_PADSIZE=6, so the host scalar runs the
    // no-blur path. Mix of signs + magnitudes per channel.
    let (w, h) = (4usize, 4usize);
    let n = w * h;
    let t_a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 2.0).collect();
    let t_rg: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 + 0.2).collect();
    let t_vy: Vec<f32> = (0..n).map(|i| (i as f32).sin() * 1.5).collect();
    let r_a: Vec<f32> = (0..n).map(|i| ((i + 3) as f32) * 0.3 - 2.5).collect();
    let r_rg: Vec<f32> = (0..n).map(|i| (i as f32) * 0.4 + 0.5).collect();
    let r_vy: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.7).cos() * 1.2).collect();

    let t_p = [t_a.clone(), t_rg.clone(), t_vy.clone()];
    let r_p = [r_a.clone(), r_rg.clone(), r_vy.clone()];
    let d_cpu = mult_mutual_band(&t_p, &r_p, w, h);

    let t_a_h = client.create_from_slice(f32::as_bytes(&t_a));
    let t_rg_h = client.create_from_slice(f32::as_bytes(&t_rg));
    let t_vy_h = client.create_from_slice(f32::as_bytes(&t_vy));
    let r_a_h = client.create_from_slice(f32::as_bytes(&r_a));
    let r_rg_h = client.create_from_slice(f32::as_bytes(&r_rg));
    let r_vy_h = client.create_from_slice(f32::as_bytes(&r_vy));
    let d_a_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let d_rg_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let d_vy_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));

    let cube_dim = CubeDim::new_1d(64);
    let cube_count = CubeCount::Static((n as u32).div_ceil(64), 1, 1);

    unsafe {
        mult_mutual_3ch_no_blur_kernel::launch::<Backend>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(t_a_h.clone(), n),
            ArrayArg::from_raw_parts(t_rg_h.clone(), n),
            ArrayArg::from_raw_parts(t_vy_h.clone(), n),
            ArrayArg::from_raw_parts(r_a_h.clone(), n),
            ArrayArg::from_raw_parts(r_rg_h.clone(), n),
            ArrayArg::from_raw_parts(r_vy_h.clone(), n),
            ArrayArg::from_raw_parts(d_a_h.clone(), n),
            ArrayArg::from_raw_parts(d_rg_h.clone(), n),
            ArrayArg::from_raw_parts(d_vy_h.clone(), n),
            n as u32,
        );
    }

    let d_a_bytes = client.read_one(d_a_h.clone()).expect("read D_a");
    let d_rg_bytes = client.read_one(d_rg_h.clone()).expect("read D_rg");
    let d_vy_bytes = client.read_one(d_vy_h.clone()).expect("read D_vy");
    let d_a_gpu: &[f32] = f32::from_bytes(&d_a_bytes);
    let d_rg_gpu: &[f32] = f32::from_bytes(&d_rg_bytes);
    let d_vy_gpu: &[f32] = f32::from_bytes(&d_vy_bytes);

    let mut max_rel = 0.0_f32;
    for i in 0..n {
        for (gpu, cpu, tag) in [
            (d_a_gpu[i], d_cpu[0][i], "A"),
            (d_rg_gpu[i], d_cpu[1][i], "RG"),
            (d_vy_gpu[i], d_cpu[2][i], "VY"),
        ] {
            let rel = ((gpu - cpu) / cpu.abs().max(1e-6)).abs();
            if rel > max_rel {
                max_rel = rel;
            }
            assert!(
                rel < 5e-3,
                "ch {tag} pixel {i}: gpu={gpu} cpu={cpu} rel={rel:.4e}"
            );
        }
    }
    eprintln!("max relative error: {max_rel:.4e}");
}

#[test]
fn pu_blur_two_kernels_match_host_scalar() {
    let client = Backend::client(&Default::default());

    // 16×16 band with a non-trivial pattern — large enough that
    // reflect padding spans both ends of the kernel.
    let (w, h) = (16usize, 16usize);
    let n = w * h;
    let src: Vec<f32> = (0..n)
        .map(|i| {
            let x = (i % w) as f32;
            let y = (i / w) as f32;
            (x * 0.3 + y * 0.5).sin() * 4.0 + 1.0
        })
        .collect();

    // Host reference.
    let cpu = gaussian_blur_sigma3(&src, w, h);

    // GPU: h pass then v pass via the two kernels.
    let src_h = client.create_from_slice(f32::as_bytes(&src));
    let mid_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let dst_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));

    let cube_dim = CubeDim::new_1d(64);
    let cube_count = CubeCount::Static((n as u32).div_ceil(64), 1, 1);

    unsafe {
        pu_blur_h_kernel::launch::<Backend>(
            &client,
            cube_count.clone(),
            cube_dim,
            ArrayArg::from_raw_parts(src_h.clone(), n),
            ArrayArg::from_raw_parts(mid_h.clone(), n),
            w as u32,
            h as u32,
        );
        pu_blur_v_kernel::launch::<Backend>(
            &client,
            cube_count.clone(),
            cube_dim,
            ArrayArg::from_raw_parts(mid_h.clone(), n),
            ArrayArg::from_raw_parts(dst_h.clone(), n),
            w as u32,
            h as u32,
        );
    }

    let bytes = client.read_one(dst_h.clone()).expect("read dst");
    let gpu: &[f32] = f32::from_bytes(&bytes);
    let max_err = gpu
        .iter()
        .zip(&cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_err < 1e-4,
        "PU blur GPU vs CPU max-abs = {max_err}"
    );
}
