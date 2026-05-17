//! T_y.B.1: sanity-check the FIR D=5 separable Gaussian kernel on a
//! tiny known input. Compares GPU output to a hand-rolled CPU FIR
//! with reflect padding using the same 5-tap coefficients.

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
use ssim2_gpu::kernels::blur;

const TAPS: [f32; 5] = [0.12015825, 0.23383145, 0.29202062, 0.23383145, 0.12015825];

fn cpu_h_fir(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    // Zero-padding to match libjxl IIR / our GPU kernel.
    let mut out = vec![0.0_f32; w * h];
    for y in 0..h {
        let row_base = y * w;
        for x in 0..w {
            let mut acc = 0.0_f32;
            for k in -2_i32..=2_i32 {
                let xi = x as i32 + k;
                let s = if xi >= 0 && xi < w as i32 {
                    src[row_base + xi as usize]
                } else {
                    0.0
                };
                let tap = TAPS[(k + 2) as usize];
                acc += s * tap;
            }
            out[row_base + x] = acc;
        }
    }
    out
}

fn cpu_v_fir(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0_f32;
            for k in -2_i32..=2_i32 {
                let yi = y as i32 + k;
                let s = if yi >= 0 && yi < h as i32 {
                    src[yi as usize * w + x]
                } else {
                    0.0
                };
                let tap = TAPS[(k + 2) as usize];
                acc += s * tap;
            }
            out[y * w + x] = acc;
        }
    }
    out
}

fn main() {
    let client = CudaRuntime::client(&Default::default());

    // Constant-input sanity: blur of [0.5; n] should be [0.5; n] (DC gain=1).
    {
        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize;
        let src = vec![0.5_f32; n];
        let mut cpu_iir = ssimulacra2::Blur::new(w as usize, h as usize);
        let blurred = cpu_iir.blur(&[src.clone(), src.clone(), src.clone()]);
        let mut min = blurred[0][0];
        let mut max = blurred[0][0];
        for &v in &blurred[0] {
            if v < min { min = v; }
            if v > max { max = v; }
        }
        eprintln!(
            "  cpu_iir (constant 0.5): min={:.6}, max={:.6}, expected ~0.5",
            min, max
        );

        let cpu_fir = cpu_v_fir(&cpu_h_fir(&src, w as usize, h as usize), w as usize, h as usize);
        let mut min = cpu_fir[0];
        let mut max = cpu_fir[0];
        for &v in &cpu_fir { if v < min { min = v; } if v > max { max = v; } }
        eprintln!(
            "  cpu_fir (constant 0.5): min={:.6}, max={:.6}, expected ~0.5",
            min, max
        );
    }

    let cases: &[(u32, u32)] = &[(8, 8), (16, 16), (32, 32), (100, 50), (256, 256)];
    for &(w, h) in cases {
        let n = (w * h) as usize;
        let src: Vec<f32> = (0..n)
            .map(|i| ((i * 7 + 3) & 0xff) as f32 / 255.0)
            .collect();
        let cpu_h = cpu_h_fir(&src, w as usize, h as usize);
        let cpu_hv = cpu_v_fir(&cpu_h, w as usize, h as usize);

        let src_h = client.create_from_slice(f32::as_bytes(&src));
        let mid_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
        let dst_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));

        let cubes = (n as u32).div_ceil(256).max(1);
        let cube_dim = CubeDim::new_1d(256);

        unsafe {
            blur::blur_h_fir5_kernel::launch_unchecked::<CudaRuntime>(
                &client,
                CubeCount::Static(cubes, 1, 1),
                cube_dim,
                ArrayArg::from_raw_parts(src_h.clone(), n),
                ArrayArg::from_raw_parts(mid_h.clone(), n),
                w,
                h,
            );
            blur::blur_v_fir5_kernel::launch_unchecked::<CudaRuntime>(
                &client,
                CubeCount::Static(cubes, 1, 1),
                cube_dim,
                ArrayArg::from_raw_parts(mid_h.clone(), n),
                ArrayArg::from_raw_parts(dst_h.clone(), n),
                w,
                h,
            );
        }
        let bytes = client.read_one(dst_h).expect("read");
        let gpu = f32::from_bytes(&bytes);

        let mut max_d = 0.0_f32;
        let mut max_i = 0;
        for i in 0..n {
            let d = (gpu[i] - cpu_hv[i]).abs();
            if d > max_d {
                max_d = d;
                max_i = i;
            }
        }
        let ok = max_d < 1e-5;
        eprintln!(
            "  {}×{}: max |gpu - cpu_fir| = {:.3e} at idx {} (gpu={:.6}, cpu={:.6})  {}",
            w,
            h,
            max_d,
            max_i,
            gpu[max_i],
            cpu_hv[max_i],
            if ok { "OK" } else { "FAIL" }
        );

        // Also compare against CPU IIR for shape comparison.
        let mut cpu_iir = ssimulacra2::Blur::new(w as usize, h as usize);
        let rgb_src = src.clone();
        let blurred = cpu_iir.blur(&[rgb_src.clone(), rgb_src.clone(), rgb_src.clone()]);
        let mut max_diff_vs_iir = 0.0_f32;
        for i in 0..n {
            let d = (gpu[i] - blurred[0][i]).abs();
            if d > max_diff_vs_iir {
                max_diff_vs_iir = d;
            }
        }
        eprintln!(
            "  {}×{}: max |gpu_fir - cpu_iir| = {:.3e} (expected: O(σ-truncation drift, ~1e-2))",
            w, h, max_diff_vs_iir
        );
    }
}
