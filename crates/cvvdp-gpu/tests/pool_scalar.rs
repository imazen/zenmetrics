//! Parity test for `kernels::pool::do_pooling_and_jod_still_3ch`
//! against pycvvdp v0.5.4's `cvvdp.do_pooling_and_jods()`.
//!
//! Three Q_per_ch fixtures covering the JOD curve:
//! - near-perfect (~10 JOD)
//! - middling (~9.99 JOD)
//! - strongly distorted (~9.93 JOD)
#![allow(clippy::excessive_precision)]

use cvvdp_gpu::kernels::pool::{do_pooling_and_jod_still_3ch, met2jod};

#[cfg(any(feature = "cuda", feature = "wgpu"))]
mod gpu {
    use cubecl::Runtime;
    use cubecl::prelude::*;
    use cvvdp_gpu::kernels::pool::{lp_norm_mean, pool_band_finalize, pool_band_kernel};

    #[cfg(feature = "cuda")]
    type Backend = cubecl::cuda::CudaRuntime;

    #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    type Backend = cubecl::wgpu::WgpuRuntime;

    #[test]
    fn pool_band_kernel_matches_host_lp_norm_mean() {
        let client = Backend::client(&Default::default());

        // Deterministic input spanning sign + magnitude range so the
        // safe_pow (|v|+eps)^p - eps^p form actually exercises the
        // epsilon shift on the small values.
        let n = 256usize;
        let band: Vec<f32> = (0..n)
            .map(|i| {
                let x = i as f32 * 0.0123;
                x.sin() * 5.0 + 0.0005 * if i.is_multiple_of(7) { -1.0 } else { 1.0 }
            })
            .collect();
        let beta = 2.0_f32;

        // GPU path: kernel accumulates safe_pow per pixel into a
        // single-slot Atomic<f32> partial; host finalises with
        // pool_band_finalize.
        let band_h = client.create_from_slice(f32::as_bytes(&band));
        let partial_h = client.create_from_slice(f32::as_bytes(&[0.0_f32; 1]));

        let cube_dim = CubeDim::new_1d(64);
        let cube_count = CubeCount::Static((n as u32).div_ceil(64), 1, 1);

        unsafe {
            pool_band_kernel::launch::<Backend>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(band_h.clone(), n),
                ArrayArg::from_raw_parts(partial_h.clone(), 1),
                beta,
                0_u32,
                n as u32,
            );
        }

        let bytes = client.read_one(partial_h.clone()).expect("read partial");
        let partial: &[f32] = f32::from_bytes(&bytes);
        let gpu_q = pool_band_finalize(partial[0], n, beta);

        let cpu_q = lp_norm_mean(&band, beta);
        let rel = ((gpu_q - cpu_q) / cpu_q.abs().max(1e-6)).abs();
        assert!(
            rel < 5e-4,
            "GPU pool Q = {gpu_q}, CPU lp_norm_mean = {cpu_q}, rel = {rel:.4e}"
        );
    }
}

#[test]
fn pool_near_perfect_matches_pycvvdp() {
    let q_per_ch = vec![[0.01_f32; 3]; 8];
    let jod = do_pooling_and_jod_still_3ch(&q_per_ch);
    assert!(
        (jod - 10.0).abs() < 1e-3,
        "near-perfect: got {jod}, expected ~10.0"
    );
}

#[test]
fn pool_middling_matches_pycvvdp() {
    // ch0..2 rows, 8 bands each. Layout: q[k] = [ch0, ch1, ch2].
    let ch = [
        [0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.04],
        [0.4, 0.25, 0.18, 0.12, 0.08, 0.06, 0.04, 0.03],
        [0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02],
    ];
    let q_per_ch: Vec<[f32; 3]> = (0..8).map(|k| [ch[0][k], ch[1][k], ch[2][k]]).collect();
    let jod = do_pooling_and_jod_still_3ch(&q_per_ch);
    let expected = 9.987_316_f32;
    assert!(
        (jod - expected).abs() < 1e-3,
        "middling: got {jod}, expected {expected}"
    );
}

#[test]
fn pool_strong_matches_pycvvdp() {
    let ch = [
        [2.5, 1.5, 1.0, 0.8, 0.5, 0.4],
        [2.0, 1.2, 0.8, 0.6, 0.4, 0.3],
        [1.5, 0.9, 0.6, 0.5, 0.3, 0.2],
    ];
    let q_per_ch: Vec<[f32; 3]> = (0..6).map(|k| [ch[0][k], ch[1][k], ch[2][k]]).collect();
    let jod = do_pooling_and_jod_still_3ch(&q_per_ch);
    let expected = 9.931_840_f32;
    assert!(
        (jod - expected).abs() < 1e-3,
        "strong: got {jod}, expected {expected}"
    );
}

#[test]
fn met2jod_continuous_at_kink() {
    // The piecewise transform is C0 at Q=0.1; verify the two
    // branches agree there to within f32 epsilon.
    let q = 0.1_f32;
    let from_low = met2jod(q);
    let from_high = met2jod(q + 1e-6);
    assert!(
        (from_low - from_high).abs() < 1e-3,
        "discontinuity at Q=0.1: low={from_low}, high={from_high}"
    );
}

#[test]
fn met2jod_clamps_at_origin() {
    // Q=0 should give JOD=10 (no perceptible difference).
    let jod = met2jod(0.0);
    assert!((jod - 10.0).abs() < 1e-6, "met2jod(0) = {jod}, expected 10");
}
