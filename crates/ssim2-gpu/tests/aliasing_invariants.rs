//! Aliasing-safety invariants for the Phase 1 plane-aliasing change in
//! `pipeline.rs` (2026-05-22).
//!
//! Phase 1 replaced 30 per-`Scale` blur-intermediate buffers
//! (`{sigma11,sigma22,sigma12,mu1,mu2}_v` + `*_t` — 5 plane names × 3
//! channels × 2 orientations) with shared rolling scratch
//! `v_scratch: [Handle; 3]` + `t_scratch: [Handle; 3]`. The five blurs
//! per (scale, channel) recycle the same scratch pair sequentially.
//!
//! Safety contract verified here:
//!
//! 1. **Pair-path correctness**. `Ssim2::compute(ref, dis)` must still
//!    match the published CPU `ssimulacra2` reference at the same
//!    tolerance the pre-aliasing pipeline did (Δ < 0.1 abs OR rel <
//!    0.5%) across sizes 256², 1024², 2048², 4096².
//!
//! 2. **Cached-ref / pair agreement**. `set_reference(ref)` +
//!    `compute_with_reference(dis)` must agree with the direct
//!    `compute(ref, dis)` path within 8e-6 absolute (the documented
//!    "cached vs direct path drift" gate per the crate README). This
//!    is the strongest aliasing-safety probe because `set_reference`
//!    caches per-`Scale` reference-side products (sigma11_full,
//!    mu1_full, ref_xyb_t) — if the aliased scratch poisons any of
//!    those before they get consumed downstream, the cached path
//!    diverges from the pair path.
//!
//! 3. **Repeated-call stability**. Running `compute` (or
//!    `compute_with_reference`) multiple times against varying
//!    distorted inputs on the same `Ssim2` instance must produce
//!    scores that are stable — no drift from latent corruption in
//!    aliased scratch.
//!
//! 4. **Mode invariance**. Each `Ssim2Mode` (Full / Lossless / Fast /
//!    Faster) reads a different subset of the per-scale state. The
//!    aliasing must not change which subset is "live" at any read
//!    point. We verify pair + cached-ref agree across all four modes.
//!
//! Backend selection mirrors `parity_lock.rs` (cuda preferred, wgpu
//! fallback).

use cubecl::Runtime;
use ssim2_gpu::{Ssim2, Ssim2Mode};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
compile_error!(
    "ssim2-gpu integration tests require either the `cuda` or `wgpu` feature to select a runtime"
);

use ssimulacra2::{ColorPrimaries, Rgb, TransferCharacteristic, Xyb};

/// Build a synthetic reference + distorted pair with deterministic
/// content. Mirrors the helper in `parity_lock.rs` so the tests here
/// don't depend on a real PNG corpus and can exercise any resolution.
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

fn srgb_u8_to_xyb(bytes: &[u8], w: usize, h: usize) -> Xyb {
    let pixels: Vec<[f32; 3]> = bytes
        .chunks_exact(3)
        .map(|c| {
            [
                c[0] as f32 / 255.0,
                c[1] as f32 / 255.0,
                c[2] as f32 / 255.0,
            ]
        })
        .collect();
    Xyb::try_from(
        Rgb::new(
            pixels,
            w,
            h,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap(),
    )
    .unwrap()
}

fn assert_cpu_parity(w: u32, h: u32, mag: u8) {
    let (a, b) = synthetic_pair(w as usize, h as usize, mag);
    let cpu = ssimulacra2::compute_frame_ssimulacra2(
        srgb_u8_to_xyb(&a, w as usize, h as usize),
        srgb_u8_to_xyb(&b, w as usize, h as usize),
    )
    .expect("cpu reference");

    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, w, h).expect("Ssim2::new");
    let gpu = s.compute(&a, &b).expect("gpu compute").score;
    let d = (gpu - cpu).abs();
    let rel = if cpu.abs() > 1e-3 {
        d / cpu.abs() * 100.0
    } else {
        0.0
    };
    assert!(
        d < 0.1 || rel < 0.5,
        "pair-path parity {w}×{h}: cpu={cpu:.5}, gpu={gpu:.5}, Δ={d:.5}, rel={rel:.3}%"
    );
}

fn assert_pair_vs_cached_agreement(w: u32, h: u32, mag: u8) {
    let (a, b) = synthetic_pair(w as usize, h as usize, mag);

    let client_a = Backend::client(&Default::default());
    let mut s_direct = Ssim2::<Backend>::new(client_a, w, h).expect("direct");
    let direct = s_direct.compute(&a, &b).expect("direct compute").score;

    let client_b = Backend::client(&Default::default());
    let mut s_cached = Ssim2::<Backend>::new(client_b, w, h).expect("cached");
    s_cached.set_reference(&a).expect("set_reference");
    let cached = s_cached
        .compute_with_reference(&b)
        .expect("cached compute")
        .score;

    let d = (direct - cached).abs();
    // 8e-6 = the documented cached-vs-direct gate per the crate
    // README. Aliasing must not widen this gap.
    assert!(
        d < 8e-6,
        "pair-vs-cached drift {w}×{h}: direct={direct:.10}, cached={cached:.10}, Δ={d:.3e}"
    );
}

// ───────────────────── 1. pair-path CPU parity across sizes ─────────────────────

#[test]
fn aliasing_pair_path_256() {
    assert_cpu_parity(256, 256, 8);
}

#[test]
fn aliasing_pair_path_1024() {
    assert_cpu_parity(1024, 1024, 6);
}

#[test]
fn aliasing_pair_path_2048() {
    assert_cpu_parity(2048, 2048, 6);
}

// 4096² test: runs on both cuda AND wgpu since `cube_count_1d` got
// the 2D-when-large split (pipeline.rs::cube_count_1d). The scale-0
// kernel grid at 4096² needs 65,536 cubes, which used to exceed
// wgpu's 65535-per-dim cap (Limits::downlevel_defaults); the 2D
// split brings each dim under both wgpu's 65535 and CUDA's 2^31
// limits while keeping the kernel's `ABSOLUTE_POS` reader unchanged.
#[test]
fn aliasing_pair_path_4096() {
    // 4096² = 48 MiB raw upload + ~7.3 GB peak GPU after Phase 1
    // aliasing (was 10.4 GB pre-aliasing). RTX 5070 has 12 GB so this
    // fits with margin. On a smaller GPU this test will OOM — that's
    // a useful signal, not a defect.
    assert_cpu_parity(4096, 4096, 6);
}

// ───────────────────── 2. cached-ref vs pair-path agreement ─────────────────────

#[test]
fn aliasing_cached_vs_pair_256() {
    assert_pair_vs_cached_agreement(256, 256, 8);
}

#[test]
fn aliasing_cached_vs_pair_1024() {
    assert_pair_vs_cached_agreement(1024, 1024, 6);
}

#[test]
fn aliasing_cached_vs_pair_2048() {
    assert_pair_vs_cached_agreement(2048, 2048, 6);
}

// ───────────────────── 3. repeated-call stability ─────────────────────

#[test]
fn aliasing_repeated_calls_pair_path_stable() {
    // Run a sequence of distinct (ref, dis) pairs on the SAME `Ssim2`
    // instance. Each call writes to the aliased v_scratch / t_scratch
    // for every scale × channel. If a prior call's scratch state
    // leaks into the current call's blur read, scores will drift in
    // ways that are NOT just atomic-add noise (the cross-product
    // mul reads sigma11_in/sigma22_in/sigma12_in which are NOT
    // aliased — so a single drift would be silent until something
    // downstream consumes the wrong v_scratch state).
    //
    // We catch drift by computing the same (ref, dis) at the start
    // AND end of the sequence and asserting the two scores agree to
    // within tight atomic-add noise (~1e-4).
    let (w, h): (u32, u32) = (512, 384);
    let (a, b) = synthetic_pair(w as usize, h as usize, 6);
    let (a2, b2) = synthetic_pair(w as usize, h as usize, 12);
    let (a3, b3) = synthetic_pair(w as usize, h as usize, 4);

    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, w, h).expect("Ssim2::new");

    let first = s.compute(&a, &b).expect("first").score;
    // Interleave several other calls that all write the aliased
    // scratch with different content.
    let _ = s.compute(&a2, &b2).expect("intermediate 1").score;
    let _ = s.compute(&a3, &b3).expect("intermediate 2").score;
    let _ = s.compute(&a2, &b3).expect("intermediate 3").score;
    let _ = s.compute(&a3, &b2).expect("intermediate 4").score;
    // Identical-pair, exercises a different scratch usage pattern
    // (sigma11_in == sigma22_in == sigma12_in numerically).
    let _ = s.compute(&a, &a).expect("intermediate 5").score;
    let last = s.compute(&a, &b).expect("last").score;

    let d = (first - last).abs();
    assert!(
        d < 1e-4,
        "repeated-call drift: first={first:.10}, last={last:.10}, Δ={d:.3e}"
    );
}

#[test]
fn aliasing_repeated_calls_cached_ref_stable() {
    // Same idea, cached-reference path. set_reference once; call
    // compute_with_reference with several distinct distortions then
    // re-issue the first one and compare.
    let (w, h): (u32, u32) = (512, 384);
    let (a, b) = synthetic_pair(w as usize, h as usize, 6);
    let (_, b2) = synthetic_pair(w as usize, h as usize, 12);
    let (_, b3) = synthetic_pair(w as usize, h as usize, 4);
    let (_, b4) = synthetic_pair(w as usize, h as usize, 16);

    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, w, h).expect("Ssim2::new");
    s.set_reference(&a).expect("set_reference");

    let first = s.compute_with_reference(&b).expect("first").score;
    let _ = s.compute_with_reference(&b2).expect("intermediate 1").score;
    let _ = s.compute_with_reference(&b3).expect("intermediate 2").score;
    let _ = s.compute_with_reference(&b4).expect("intermediate 3").score;
    // Identical-pair against cached reference.
    let _ = s.compute_with_reference(&a).expect("intermediate 4").score;
    let last = s.compute_with_reference(&b).expect("last").score;

    let d = (first - last).abs();
    assert!(
        d < 1e-4,
        "cached-ref repeated-call drift: first={first:.10}, last={last:.10}, Δ={d:.3e}"
    );
}

// ───────────────────── 4. mode invariance ─────────────────────

#[test]
fn aliasing_modes_agree_pair_path() {
    // Lossless / Fast / Faster modes touch a strictly subset of the
    // scales × channels × maps the Full mode does. If aliasing
    // changes the "live at read" subset, the mode-skipping logic in
    // `pipeline.rs` (skip_error_map / skip_reduction / skip_scale)
    // would silently produce a different score from Full — because
    // the skipped read points would now be reading the wrong v/t
    // scratch state.
    //
    // We expect Lossless ≈ Full to ~1e-4 (skipmap_audit picks at
    // ~1e-6 but we relax here to the parity_lock-batch tolerance —
    // the noise floor of f32 reductions on real GPU).
    let (w, h): (u32, u32) = (512, 384);
    let (a, b) = synthetic_pair(w as usize, h as usize, 6);

    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, w, h).expect("Ssim2::new");

    let full = s
        .compute_with_mode(Ssim2Mode::Full, &a, &b)
        .expect("full")
        .score;
    let lossless = s
        .compute_with_mode(Ssim2Mode::Lossless, &a, &b)
        .expect("lossless")
        .score;
    let fast = s
        .compute_with_mode(Ssim2Mode::Fast, &a, &b)
        .expect("fast")
        .score;
    let faster = s
        .compute_with_mode(Ssim2Mode::Faster, &a, &b)
        .expect("faster")
        .score;

    // Lossless: zero-weight cells contribute exactly 0 → bit-near to
    // Full. Tolerance matches the existing skipmap_audit gate, but
    // see `tests/ssim2_skipmap_audit.rs::modes_agree_on_jpeg_corpus`
    // for the known pre-existing flake at q=5 on the JPEG corpus.
    // Synthetic content is well-behaved enough to clear 1e-4.
    let d_lossless = (lossless - full).abs();
    assert!(
        d_lossless < 1e-4,
        "Lossless ≠ Full at {w}×{h}: full={full:.10}, lossless={lossless:.10}, Δ={d_lossless:.3e}"
    );
    // Fast / Faster: bounded by threshold × Σ_values. For real data,
    // empirical rel error is much smaller than the worst case.
    let rel_f = if full.abs() > 1e-3 {
        (fast - full).abs() / full.abs()
    } else {
        0.0
    };
    let rel_x = if full.abs() > 1e-3 {
        (faster - full).abs() / full.abs()
    } else {
        0.0
    };
    assert!(
        rel_f < 5e-4,
        "Fast diverged from Full: full={full:.6}, fast={fast:.6}, rel={rel_f:.4e}"
    );
    assert!(
        rel_x < 5e-4,
        "Faster diverged from Full: full={full:.6}, faster={faster:.6}, rel={rel_x:.4e}"
    );
}

#[test]
fn aliasing_modes_agree_cached_ref_path() {
    // Same as `aliasing_modes_agree_pair_path` but via the
    // cached-reference path. This exercises the **most fragile**
    // intersection of the Phase 1 aliasing + skip-map dispatch:
    // `set_reference` pre-populates per-scale sigma11_full / mu1_full
    // / ref_xyb_t using v_scratch / t_scratch; then each subsequent
    // `compute_with_reference_with_mode` re-uses those same scratch
    // buffers for the dis-side blurs under the mode's skip mask. If
    // either Phase 1 OR the cached-ref code path lets the scratch
    // state leak across the set_reference → compute boundary, the
    // mode-comparison drift here will surface it.
    let (w, h): (u32, u32) = (512, 384);
    let (a, b) = synthetic_pair(w as usize, h as usize, 6);

    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, w, h).expect("Ssim2::new");
    s.set_reference(&a).expect("set_reference");

    let full = s
        .compute_with_reference_with_mode(Ssim2Mode::Full, &b)
        .expect("full")
        .score;
    let lossless = s
        .compute_with_reference_with_mode(Ssim2Mode::Lossless, &b)
        .expect("lossless")
        .score;
    let fast = s
        .compute_with_reference_with_mode(Ssim2Mode::Fast, &b)
        .expect("fast")
        .score;
    let faster = s
        .compute_with_reference_with_mode(Ssim2Mode::Faster, &b)
        .expect("faster")
        .score;

    let d_lossless = (lossless - full).abs();
    assert!(
        d_lossless < 1e-4,
        "cached Lossless ≠ Full: full={full:.10}, lossless={lossless:.10}, Δ={d_lossless:.3e}"
    );
    let rel_f = if full.abs() > 1e-3 {
        (fast - full).abs() / full.abs()
    } else {
        0.0
    };
    let rel_x = if full.abs() > 1e-3 {
        (faster - full).abs() / full.abs()
    } else {
        0.0
    };
    assert!(
        rel_f < 5e-4,
        "cached Fast diverged from Full: full={full:.6}, fast={fast:.6}, rel={rel_f:.4e}"
    );
    assert!(
        rel_x < 5e-4,
        "cached Faster diverged from Full: full={full:.6}, faster={faster:.6}, rel={rel_x:.4e}"
    );
}

#[test]
fn aliasing_pair_vs_cached_matches_per_mode() {
    // Cross-product of (4 modes) × (pair, cached-ref) on the same
    // (ref, dis). For each mode the two paths must agree to within
    // 8e-6 — the cached-vs-direct gate, **independently for each
    // mode**. This is a stronger test than `aliasing_cached_vs_pair`
    // at the default mode only, because per-mode skip masks change
    // which scratch state is "live" at each scale × channel.
    let (w, h): (u32, u32) = (256, 256);
    let (a, b) = synthetic_pair(w as usize, h as usize, 8);

    for mode in [
        Ssim2Mode::Full,
        Ssim2Mode::Lossless,
        Ssim2Mode::Fast,
        Ssim2Mode::Faster,
    ] {
        let client_a = Backend::client(&Default::default());
        let mut direct = Ssim2::<Backend>::new(client_a, w, h).expect("direct");
        let d = direct
            .compute_with_mode(mode, &a, &b)
            .expect("direct mode")
            .score;

        let client_b = Backend::client(&Default::default());
        let mut cached = Ssim2::<Backend>::new(client_b, w, h).expect("cached");
        cached.set_reference(&a).expect("set_reference");
        let c = cached
            .compute_with_reference_with_mode(mode, &b)
            .expect("cached mode")
            .score;

        let drift = (d - c).abs();
        assert!(
            drift < 8e-6,
            "pair-vs-cached drift for {mode:?}: pair={d:.10}, cached={c:.10}, Δ={drift:.3e}"
        );
    }
}

// ───────────────────── 5. set_reference re-arm with aliased scratch ─────────────────────

#[test]
fn aliasing_set_reference_reuse() {
    // set_reference writes per-scale sigma11_full / mu1_full / ref_xyb_t
    // using the aliased v_scratch + t_scratch. Calling set_reference
    // a second time MUST fully overwrite those _full buffers — if
    // some plane is silently left over from the prior reference (e.g.
    // because the new aliasing made the scratch dependency invisible
    // to the second pass), subsequent compute_with_reference would
    // return a score that's a blend of the two references.
    //
    // Probe: set_reference(A) → compute_with_reference(B) = scoreA.
    // Then set_reference(B) → compute_with_reference(A) = scoreB.
    // Then back to set_reference(A) → compute_with_reference(B) again.
    // The third call's score MUST match the first (atomic noise only).
    let (w, h): (u32, u32) = (512, 384);
    let (a, b) = synthetic_pair(w as usize, h as usize, 8);

    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, w, h).expect("Ssim2::new");

    s.set_reference(&a).expect("set A");
    let first = s.compute_with_reference(&b).expect("compute B").score;

    s.set_reference(&b).expect("set B");
    let _swapped = s.compute_with_reference(&a).expect("compute A").score;

    s.set_reference(&a).expect("set A again");
    let third = s.compute_with_reference(&b).expect("compute B again").score;

    let d = (first - third).abs();
    assert!(
        d < 1e-4,
        "set_reference reuse mismatch: first={first:.10}, third={third:.10}, Δ={d:.3e}"
    );
}

// ───────────────────── 6. identical-pair sanity at multiple sizes ─────────────────────

#[test]
fn aliasing_identical_pair_scores_100_across_sizes() {
    // Identical (ref, dis) at multiple sizes — score must round to
    // ~100 at each. A non-100 score at one size but not another
    // would indicate the aliased scratch is contaminating one
    // scale's reduction more than others (different scratch reuse
    // patterns activate at different pyramid depths).
    let sizes: [(u32, u32); 3] = [(256, 256), (1024, 1024), (2048, 2048)];

    for (w, h) in sizes {
        // Use the synthetic_pair `a` only (so ref == dis).
        let (a, _) = synthetic_pair(w as usize, h as usize, 0);
        let client = Backend::client(&Default::default());
        let mut s = Ssim2::<Backend>::new(client, w, h).expect("Ssim2::new");
        let r = s.compute(&a, &a).expect("identical").score;
        assert!(
            r >= 99.0 && r <= 100.05,
            "identical-pair {w}×{h}: score={r}, expected [99, 100.05]"
        );
    }
}
