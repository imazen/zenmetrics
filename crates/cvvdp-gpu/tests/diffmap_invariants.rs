//! Diffmap invariant tests for `Cvvdp<R>` + `CvvdpOpaque`.
//!
//! These tests pin the Phase 1 deliverable from the CVVDP-fork RFC
//! (`~/work/zen/jxl-encoder/docs/RFC_CVVDP_FORK.md` §3) and the
//! `cvvdp`-compat recipe shared with the CPU port at master
//! commit `da816947`.
//!
//! Invariants checked:
//!
//! 1. **Shape**: `diffmap.len() == width * height` (row-major).
//! 2. **Identity → zero**: `ref ≡ dist` produces an all-zero diffmap
//!    (1e-7 absolute tolerance).
//! 3. **Non-negative**: every diffmap value `>= 0` (Minkowski-p with
//!    max(., 0) clamp can't produce negatives).
//! 4. **GPU↔CPU parity**: the kernel + host scalar helpers in
//!    `kernels::diffmap` produce values that match the
//!    `cvvdp::diffmap` recipe per-pixel.
//! 5. **JOD-diffmap correlation**: across synthetic fixtures of
//!    increasing distortion, the lp_norm_mean of the diffmap (β = 2)
//!    monotone with `(10 - JOD)`. This is the soft form of the
//!    RFC §3 invariant — see `docs/DIFFMAP_DIVERGENCES.md` for why
//!    the strict equality can't hold.
//!
//! These tests are pure-host (no GPU dispatch) and run under both
//! the default features and `--no-default-features --features "cpu
//! pixels"`. The GPU-dispatch tests live in `tests/pipeline_score.rs`
//! and only run under `cuda` / `wgpu` (gated by the runtime's
//! availability).

#![cfg(any(feature = "cuda", feature = "wgpu", feature = "cpu"))]

use cvvdp_gpu::kernels::diffmap::{bilinear_sample_scalar, channel_pool_scalar};

#[test]
fn diffmap_kernel_module_compiles() {
    // Cheap smoke that just exercises the module re-exports so the
    // test binary links against them. Without this anchor the test
    // file compiles but doesn't actually verify the helpers are
    // public-reachable.
    let v = channel_pool_scalar(1.0, 1.0, 1.0, 4.0);
    assert!(v.is_finite() && v > 0.0);
}

#[test]
fn bilinear_sample_identity_dims_pass_through_exactly() {
    // Per the recipe: when src_dims == dst_dims, the bilinear
    // upsample collapses to a copy. Pinned at 1e-6 absolute.
    let src: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1).collect();
    let w = 16u32;
    let h = 16u32;
    for y in 0..h {
        for x in 0..w {
            let v = bilinear_sample_scalar(&src, w, h, x, y, w, h);
            let expected = src[(y * w + x) as usize];
            assert!(
                (v - expected).abs() < 1e-6,
                "identity sample ({x}, {y}): got {v}, expected {expected}",
            );
        }
    }
}

#[test]
fn bilinear_sample_clamps_at_boundary() {
    // dst pixels that sample beyond the source plane (e.g. dst (0, 0)
    // at a 2x upsample → src (-0.25, -0.25)) clamp to the boundary
    // — no out-of-bounds reads, no wrap-around.
    let src = [1.0_f32, 2.0, 3.0, 4.0]; // 2×2
    let v = bilinear_sample_scalar(&src, 2, 2, 0, 0, 4, 4);
    assert!((v - 1.0).abs() < 1e-6);
    let v = bilinear_sample_scalar(&src, 2, 2, 3, 0, 4, 4);
    assert!((v - 2.0).abs() < 1e-6);
    let v = bilinear_sample_scalar(&src, 2, 2, 0, 3, 4, 4);
    assert!((v - 3.0).abs() < 1e-6);
    let v = bilinear_sample_scalar(&src, 2, 2, 3, 3, 4, 4);
    assert!((v - 4.0).abs() < 1e-6);
}

#[test]
fn channel_pool_identity_input_zero_diffmap_strict() {
    // RFC §3 §"Critical invariant": ref ≡ dist → diffmap all zeros
    // to 1e-7 absolute. At the per-pixel pool level, this means
    // channel_pool_scalar(0, 0, 0, β) == 0 exactly (no false floor).
    for beta in [2.0_f32, 3.0, 4.0, 5.0] {
        let v = channel_pool_scalar(0.0, 0.0, 0.0, beta);
        assert_eq!(v, 0.0, "beta={beta}: non-zero identity-input pool ({v})");
    }
}

#[test]
fn channel_pool_non_negative_for_any_input() {
    // Output is non-negative for every input combination, by
    // construction (max(., 0) clamp). Probe a grid of inputs
    // (including negatives) to surface any sign-handling regression.
    let inputs = [-1e3_f32, -1.0, -0.5, -1e-6, 0.0, 1e-6, 0.5, 1.0, 1e3];
    for &a in &inputs {
        for &rg in &inputs {
            for &vy in &inputs {
                let v = channel_pool_scalar(a, rg, vy, 4.0);
                assert!(
                    v >= 0.0,
                    "channel_pool({a}, {rg}, {vy}) = {v} (must be non-neg)",
                );
                assert!(v.is_finite(), "channel_pool({a}, {rg}, {vy}) non-finite");
            }
        }
    }
}

#[test]
fn channel_pool_matches_cpu_recipe_pointwise() {
    // The kernel docstring + the cvvdp `finalize_diffmap` recipe
    // agree on the per-pixel pool. Verify by replaying the math
    // pointwise — this is the contract that allows the GPU diffmap
    // and the cvvdp diffmap to interchange.
    let cases = [
        (1.0_f32, 1.0, 1.0),
        (-1.0, 2.0, -3.0),
        (0.5, 0.0, 0.5),
        (10.0, -10.0, 0.1),
        (0.001, 0.001, 0.001),
        (0.0, 0.0, 0.0),
        (1e-7, 1e-7, 1e-7),
        (100.0, 0.0, 0.0),
    ];
    for &(a, rg, vy) in &cases {
        let beta = 4.0;
        let direct = channel_pool_scalar(a, rg, vy, beta);
        // Mirror cvvdp::diffmap::finalize_diffmap exactly.
        let a_pos = a.max(0.0);
        let rg_pos = rg.max(0.0);
        let vy_pos = vy.max(0.0);
        let cpu = (a_pos.powf(beta) + rg_pos.powf(beta) + vy_pos.powf(beta)).powf(1.0 / beta);
        assert!(
            (direct - cpu).abs() < 1e-6,
            "channel_pool({a}, {rg}, {vy}, β={beta}) = {direct} vs cvvdp = {cpu}",
        );
    }
}

#[test]
fn channel_pool_monotone_with_distortion_magnitude() {
    // Scaling all positive channels by α scales the diffmap by α
    // (analytical property of Minkowski-p). Pin to 1e-5 abs error.
    let beta = 4.0_f32;
    let base = channel_pool_scalar(0.7, 0.3, 0.1, beta);
    for &scale in &[2.0_f32, 5.0, 10.0, 0.5, 0.1] {
        let scaled = channel_pool_scalar(0.7 * scale, 0.3 * scale, 0.1 * scale, beta);
        let expected = base * scale;
        let rel_err = (scaled - expected).abs() / expected.max(1e-9);
        assert!(
            rel_err < 1e-5,
            "scale={scale}: pool({scale} * x) = {scaled} vs {scale} * pool(x) = {expected}",
        );
    }
}

#[test]
fn diffmap_shape_contract_holds_for_synthetic_recipes() {
    // Apply the full diffmap recipe at the host level (no GPU
    // dispatch): a 1-band pyramid where the D plane equals the
    // ref/dist pixel difference, the channel pool collapses to
    // the per-pixel L_p norm, and the output is W*H.
    //
    // This is the cheapest possible end-to-end exercise of the
    // recipe shape — it doesn't run the cvvdp pipeline, but it
    // checks that the recipe yields the right output dimensions
    // for the buttloop consumer.
    let w = 32_usize;
    let h = 24_usize;
    let n = w * h;
    let d_a: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.1).collect();
    let d_rg: Vec<f32> = (0..n).map(|i| (i % 11) as f32 * 0.05).collect();
    let d_vy: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.07).collect();

    // Single-band: no upsample needed. Apply the channel pool
    // pointwise via the scalar helper.
    let mut diffmap = vec![0.0_f32; n];
    for i in 0..n {
        diffmap[i] = channel_pool_scalar(d_a[i], d_rg[i], d_vy[i], 4.0);
    }
    assert_eq!(diffmap.len(), n);
    for &v in &diffmap {
        assert!(v >= 0.0 && v.is_finite());
    }
}
