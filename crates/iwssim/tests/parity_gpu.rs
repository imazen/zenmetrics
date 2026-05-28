//! Parity test against `iwssim-gpu` — feature-gated on
//! `gpu-parity-test` so that ordinary `cargo test -p iwssim` runs
//! don't pull cubecl + CUDA into the build graph.
//!
//! Each test scores the same synthetic fixture via both the CPU port
//! and the CUDA-backed GPU port and asserts the results agree within
//! atomic-add / SIMD-reorder tolerance.
//!
//! ## Running
//!
//! ```bash
//! # Requires a real CUDA device + libcuda on PATH.
//! cargo test -p iwssim --features gpu-parity-test --test parity_gpu -- --ignored
//! ```
//!
//! The tests are `#[ignore]`d by default. Manual runs after a port-side
//! refactor (or after a cubecl bump in `iwssim-gpu`) opt them in via
//! `--ignored`.

#![cfg(feature = "gpu-parity-test")]

use iwssim::{Iwssim, IwssimParams};

/// Tolerance for CPU↔GPU agreement. Both implementations are f32-based
/// with the same algorithmic structure; the differences come from:
///
/// - The GPU's atomic-add reductions for `C_u` accumulation are
///   nondeterministic in op ordering, costing ~1e-5 per scale.
/// - The CPU's Jacobi eigendecomposition produces eigenvalues in a
///   different sorted order than the GPU's LAPACK call. The infow sum
///   is order-invariant in exact arithmetic but loses ~1e-6 per
///   eigenvalue in f32.
///
/// 1e-3 in `[0, 1]` score space is the conservative band — production
/// scoring is far more precise than this. Tight CI gates can drop to
/// 1e-4 once the SIMD pyramid lands (currently scalar fallback dominates
/// the drift budget).
const TOL: f64 = 1e-3;

fn make_rgb_from_seed(w: u32, h: u32, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let n = (w as usize) * (h as usize) * 3;
    let mut out = vec![0u8; n];
    let xs = |s: &mut u64| -> u64 {
        let mut v = *s;
        v ^= v.wrapping_shl(13);
        v ^= v >> 7;
        v ^= v.wrapping_shl(17);
        *s = v;
        v
    };
    for i in (0..n).step_by(3) {
        let v = xs(&mut state);
        out[i] = (v & 0xFF) as u8;
        out[i + 1] = ((v >> 8) & 0xFF) as u8;
        out[i + 2] = ((v >> 16) & 0xFF) as u8;
    }
    out
}

fn run_pair(name: &str, w: u32, h: u32, seed: u64, distort_offset: i32) {
    // Reference: deterministic XorShift64 RGB.
    let ref_rgb = make_rgb_from_seed(w, h, seed);
    // Distortion: per-byte additive offset clamped to [0, 255].
    let dist_rgb: Vec<u8> = ref_rgb
        .iter()
        .map(|&v| (v as i32 + distort_offset).clamp(0, 255) as u8)
        .collect();

    // CPU score.
    let mut cpu = Iwssim::with_params(w, h, IwssimParams::default()).expect("Iwssim::new");
    let cpu_result = cpu.score(&ref_rgb, &dist_rgb).expect("cpu score");

    // GPU score via the opaque API. The cuda feature on iwssim-gpu
    // (pulled by the gpu-parity-test feature) lets us select
    // Backend::Cuda directly.
    use iwssim_gpu::{Backend, IwssimOpaque, IwssimParams as GpuParams};
    let mut gpu = IwssimOpaque::new(Backend::Cuda, w, h, GpuParams::default()).expect("gpu new");
    let gpu_result = gpu.compute_srgb_u8(&ref_rgb, &dist_rgb).expect("gpu score");

    let diff = (cpu_result.score - gpu_result.value).abs();
    eprintln!(
        "  {name:32}  cpu = {:.10}  gpu = {:.10}  diff = {:.3e}",
        cpu_result.score, gpu_result.value, diff
    );
    assert!(
        diff <= TOL,
        "{name}: CPU↔GPU drift {diff:.3e} exceeds tolerance {TOL:.3e} \
         (cpu={cpu:.10}, gpu={gpu:.10})",
        cpu = cpu_result.score,
        gpu = gpu_result.value
    );
}

#[test]
#[ignore = "requires real CUDA device — opt in with --ignored"]
fn parity_256_identical() {
    run_pair("parity_256_identical", 256, 256, 1, 0);
}

#[test]
#[ignore = "requires real CUDA device — opt in with --ignored"]
fn parity_256_offset_small() {
    run_pair("parity_256_offset_small", 256, 256, 2, 5);
}

#[test]
#[ignore = "requires real CUDA device — opt in with --ignored"]
fn parity_256_offset_large() {
    run_pair("parity_256_offset_large", 256, 256, 3, 32);
}

#[test]
#[ignore = "requires real CUDA device — opt in with --ignored"]
fn parity_320x240_identical() {
    run_pair("parity_320x240_identical", 320, 240, 10, 0);
}

#[test]
#[ignore = "requires real CUDA device — opt in with --ignored"]
fn parity_176_identical() {
    run_pair("parity_176_identical", 176, 176, 100, 0);
}
