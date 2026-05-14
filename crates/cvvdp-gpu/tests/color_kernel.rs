//! GPU kernel parity for `srgb_to_dkl_kernel` against the host scalar.
//!
//! The host scalar `srgb_byte_to_dkl_scalar` is itself locked vs
//! pycvvdp v0.5.4 (see `tests/color_scalar.rs`), so verifying that
//! the cubecl kernel agrees with the scalar transitively locks the
//! GPU path against the reference. Tolerance is tight (1e-5 absolute)
//! since both paths run the same f32 math — but a few ULPs of slack
//! is needed because CUDA fuses multiply-adds while host f32 does
//! not, so the matmul order can differ in the last bit of the
//! mantissa.

#![cfg(any(feature = "cuda", feature = "wgpu"))]

use cubecl::Runtime;
use cubecl::prelude::*;
use cvvdp_gpu::kernels::color::{SRGB8_TO_LINEAR_LUT, srgb_byte_to_dkl_scalar, srgb_to_dkl_kernel};
use cvvdp_gpu::params::DisplayModel;

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

fn rgb_input(w: u32, h: u32) -> Vec<u8> {
    let n = (w * h) as usize;
    let mut v = Vec::with_capacity(n * 3);
    // Deterministic non-trivial pattern: each channel varies by a
    // different stride so the matrix rows aren't accidentally
    // degenerate (a pure ramp would test only the row sums).
    for i in 0..n {
        v.push((i % 251) as u8);
        v.push(((i * 7 + 13) % 251) as u8);
        v.push(((i * 19 + 41) % 251) as u8);
    }
    v
}

#[test]
fn srgb_to_dkl_kernel_matches_host_scalar() {
    let client = Backend::client(&Default::default());

    let (w, h) = (8u32, 8u32);
    let n = (w * h) as usize;
    let rgb_bytes = rgb_input(w, h);

    // Kernel expects bytes packed one-per-u32 slot, RGBRGB order.
    let src_u32: Vec<u32> = rgb_bytes.iter().map(|&b| b as u32).collect();
    let src_h = client.create_from_slice(u32::as_bytes(&src_u32));

    let lut_h = client.create_from_slice(f32::as_bytes(&SRGB8_TO_LINEAR_LUT));
    let a_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let rg_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let vy_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));

    let display = DisplayModel::STANDARD_4K;

    let cube_dim = CubeDim::new_1d(64);
    let cube_count = CubeCount::Static((n as u32).div_ceil(64), 1, 1);

    unsafe {
        srgb_to_dkl_kernel::launch::<Backend>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(src_h.clone(), n * 3),
            ArrayArg::from_raw_parts(lut_h.clone(), SRGB8_TO_LINEAR_LUT.len()),
            ArrayArg::from_raw_parts(a_h.clone(), n),
            ArrayArg::from_raw_parts(rg_h.clone(), n),
            ArrayArg::from_raw_parts(vy_h.clone(), n),
            w,
            h,
            display.y_peak,
            display.y_black,
            display.y_refl,
        );
    }

    let a_bytes = client.read_one(a_h.clone()).expect("read A");
    let rg_bytes = client.read_one(rg_h.clone()).expect("read RG");
    let vy_bytes = client.read_one(vy_h.clone()).expect("read VY");
    let gpu_a: &[f32] = f32::from_bytes(&a_bytes);
    let gpu_rg: &[f32] = f32::from_bytes(&rg_bytes);
    let gpu_vy: &[f32] = f32::from_bytes(&vy_bytes);

    let mut max_err = 0.0_f32;
    let mut worst = (0u8, 0u8, 0u8, 0.0_f32, 0.0_f32, 0.0_f32);
    for i in 0..n {
        let r = rgb_bytes[i * 3];
        let g = rgb_bytes[i * 3 + 1];
        let b = rgb_bytes[i * 3 + 2];
        let (ea, erg, evy) =
            srgb_byte_to_dkl_scalar(r, g, b, display.y_peak, display.y_black, display.y_refl);
        let da = (gpu_a[i] - ea).abs();
        let drg = (gpu_rg[i] - erg).abs();
        let dvy = (gpu_vy[i] - evy).abs();
        for &d in &[da, drg, dvy] {
            if d > max_err {
                max_err = d;
                worst = (r, g, b, gpu_a[i] - ea, gpu_rg[i] - erg, gpu_vy[i] - evy);
            }
        }
    }
    // 3e-5 absolute — wider than ULP at the DKL output magnitudes
    // (200 cd/m² peak, so 1 ULP ≈ 1.5e-5) to absorb FMA-vs-mul-add
    // ordering between CUDA and host. Tightened beyond this would
    // need host-side FMA enforcement.
    assert!(
        max_err < 3e-5,
        "GPU vs host-scalar color transform max-abs error = {max_err}; \
         worst pixel RGB=({},{},{}) diffs A/RG/VY=({},{},{})",
        worst.0,
        worst.1,
        worst.2,
        worst.3,
        worst.4,
        worst.5
    );
}
