//! Reduction determinism — task #52.
//!
//! ssim2-gpu's per-octave (Σssim, Σ|ssim − avg|) reduction historically
//! used `Atomic<f32>::fetch_add`. Atomic-add commit order across cubes
//! varies launch-to-launch on every modern GPU, so two runs of the
//! same `(reference, distorted)` pair produced final scores that
//! diverged by ~5e-5. That's fine for picker / training data but
//! breaks "million metrics" bit-reproducibility.
//!
//! Since 2026-05-26 the default reduction path is the portable
//! per-thread-partials + finalize kernel: each grid-strided thread
//! writes `(local_sum, local_p4)` to its own scratch slot, and a
//! single-cube finalize sums the 4096 partials in a fixed
//! `k = 0..n_threads` order. f32 add isn't associative but the order
//! IS — so the output is bit-identical across re-runs of the same
//! input.
//!
//! This test re-runs `compute` 10× on the same input and asserts every
//! score has identical `to_bits()`. It exists specifically to catch
//! regressions where someone re-enables a non-deterministic reduction
//! by default.
//!
//! ## Why this also restores Metal support
//!
//! cubecl-wgpu's Metal backend reports `Atomic<f32> = LoadStore|Add`
//! as supported, but the codegen silently no-ops `fetch_add` at
//! execution time. With the portable path now default, Metal users
//! get correct (and bit-identical) scores out of the box. See the
//! comment in `kernels/reduction.rs` for the codegen story.

#![cfg(any(feature = "cuda", feature = "wgpu"))]

use cubecl::Runtime;
use ssim2_gpu::Ssim2;

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

/// Deterministic synthetic pair — same scheme as `parity_lock.rs`.
fn synthetic_pair(width: usize, height: usize, mag: u8) -> (Vec<u8>, Vec<u8>) {
    let mut a = vec![0u8; width * height * 3];
    let mut b = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 220 / width.max(1)) & 0xff) as u8;
            let g = ((y * 220 / height.max(1)) & 0xff) as u8;
            let bb = (((x + y) * 200 / (width + height).max(1)) & 0xff) as u8;
            let i = (y * width + x) * 3;
            a[i] = r;
            a[i + 1] = g;
            a[i + 2] = bb;
            let bx = x / 8;
            let by = y / 8;
            let pert = if (bx ^ by) & 1 == 0 {
                mag as i32
            } else {
                -(mag as i32)
            };
            b[i] = (r as i32 + pert).clamp(0, 255) as u8;
            b[i + 1] = (g as i32 + pert).clamp(0, 255) as u8;
            b[i + 2] = (bb as i32 + pert).clamp(0, 255) as u8;
        }
    }
    (a, b)
}

/// 10 calls to `compute(ref, dist)` on the same pair must produce
/// bit-identical scores (`to_bits()` equality).
///
/// The fast-reduction `Atomic<f32>::fetch_add` path is non-deterministic
/// — this test will FAIL if someone re-adds `fast-reduction` to the
/// default features. When that's intentional, build the failing test
/// with `--features fast-reduction` and the assertion will report the
/// observed ulp drift.
#[test]
fn reduction_determinism_10x_bit_identical() {
    let (a, b) = synthetic_pair(256, 256, 4);
    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, 256, 256).expect("Ssim2::new");

    let s0 = s.compute(&a, &b).expect("compute s0").score;
    let bits0 = s0.to_bits();

    let mut all_bits = vec![bits0];
    let mut all_scores = vec![s0];
    for i in 1..10 {
        let si = s
            .compute(&a, &b)
            .unwrap_or_else(|e| panic!("compute s{i} failed: {e}"))
            .score;
        all_bits.push(si.to_bits());
        all_scores.push(si);
    }

    // Report every run before asserting so a regression dumps the
    // full distribution rather than just the first failing pair.
    eprintln!("reduction_determinism_10x_bit_identical:");
    for (i, (sc, bits)) in all_scores.iter().zip(all_bits.iter()).enumerate() {
        eprintln!("  run {i:>2}: score={sc:.15} bits=0x{bits:016x}");
    }

    let mismatched: Vec<(usize, u64)> = all_bits
        .iter()
        .enumerate()
        .filter(|(_, b)| **b != bits0)
        .map(|(i, b)| (i, *b))
        .collect();

    assert!(
        mismatched.is_empty(),
        "ssim2-gpu reduction is not bit-deterministic: run 0 bits=0x{bits0:016x} \
         (score={s0:.15}), but {} of 10 runs differ: {:?}. \
         This means a non-deterministic atomic-add path is active — \
         check ssim2-gpu's `fast-reduction` feature gate.",
        mismatched.len(),
        mismatched
    );
}

/// Second pair shape — smaller, deeper pyramid coverage. Repeats the
/// same bit-identity assertion on a fresh instance.
#[test]
fn reduction_determinism_small_image_10x_bit_identical() {
    let (a, b) = synthetic_pair(128, 96, 8);
    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, 128, 96).expect("Ssim2::new");

    let s0 = s.compute(&a, &b).expect("compute s0").score;
    let bits0 = s0.to_bits();

    let mut all_bits = vec![bits0];
    for i in 1..10 {
        let si = s
            .compute(&a, &b)
            .unwrap_or_else(|e| panic!("compute s{i} failed: {e}"))
            .score;
        all_bits.push(si.to_bits());
    }

    let mismatched: Vec<(usize, u64)> = all_bits
        .iter()
        .enumerate()
        .filter(|(_, b)| **b != bits0)
        .map(|(i, b)| (i, *b))
        .collect();

    assert!(
        mismatched.is_empty(),
        "ssim2-gpu reduction (128×96) is not bit-deterministic: \
         run 0 bits=0x{bits0:016x}, {} of 10 runs differ: {:?}",
        mismatched.len(),
        mismatched
    );
}
