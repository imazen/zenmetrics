//! GPU kernel test for `weight_band_kernel` — verifies it
//! multiplies each band pixel by the indexed weight.

#![cfg(any(feature = "cuda", feature = "wgpu"))]

use cubecl::Runtime;
use cubecl::prelude::*;
use cvvdp_gpu::kernels::csf::weight_band_kernel;

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
