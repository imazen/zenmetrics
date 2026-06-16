//! Sub-64px reflect-pad parity + size-invariance for the GPU path.
//!
//! `ZensimOpaque` reflect(mirror)-pads any image whose min dimension is
//! below the 4-scale pyramid floor (`MIN_PAD_DIM = 64`) up to that floor
//! before running the GPU pipeline — byte-for-byte the same reflect-101
//! rule the CPU `zensim::metric` funnel uses. This file pins three
//! properties of that path, exercising it DIRECTLY (the `zenmetrics`
//! CLI routes sub-64px to CPU for exact parity, which would otherwise
//! hide the native GPU sub-64 path):
//!
//! 1. Every size down to 1×1 scores without `InvalidImageSize`.
//! 2. A constant colour difference scores the same at every size — the
//!    invariant the size-invariance work exists to guarantee.
//! 3. GPU sub-64 scores match the CPU canonical path (which also
//!    reflect-pads) within the usual f32-kernel tolerance.

#![cfg(all(feature = "cubecl-types", any(feature = "cuda", feature = "wgpu")))]

use zensim::{RgbSlice, Zensim as ZensimCpu, ZensimProfile};
use zensim_gpu::{Backend, ZensimOpaque, ZensimParams};

#[cfg(feature = "cuda")]
const BACKEND_E: Backend = Backend::Cuda;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
const BACKEND_E: Backend = Backend::Wgpu;

/// Deterministic textured image; same generator as the other parity
/// tests so the fixtures are familiar.
fn make_image(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x.wrapping_add(seed)) & 0xff) as u8;
            let g = ((y.wrapping_add(seed.wrapping_mul(3))) & 0xff) as u8;
            let b = ((x ^ y ^ seed) & 0xff) as u8;
            out.extend_from_slice(&[r, g, b]);
        }
    }
    out
}

fn solid(w: u32, h: u32, c: [u8; 3]) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for _ in 0..(w * h) {
        out.extend_from_slice(&c);
    }
    out
}

fn gpu_opaque(w: u32, h: u32) -> ZensimOpaque {
    ZensimOpaque::new(
        BACKEND_E,
        w,
        h,
        ZensimParams::new().with_profile(ZensimProfile::latest_preview()),
    )
    .expect("opaque new (sub-64 reflect-pad should never error on ≥1px)")
}

fn gpu_score(w: u32, h: u32, r: &[u8], d: &[u8]) -> f64 {
    gpu_opaque(w, h)
        .compute_srgb_u8(r, d)
        .expect("gpu compute_srgb_u8")
        .value
}

fn cpu_score(w: u32, h: u32, r: &[u8], d: &[u8]) -> f64 {
    let z = ZensimCpu::new(ZensimProfile::latest_preview());
    let to_pix =
        |buf: &[u8]| -> Vec<[u8; 3]> { buf.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect() };
    let src = to_pix(r);
    let dst = to_pix(d);
    let s = RgbSlice::new(&src, w as usize, h as usize);
    let dd = RgbSlice::new(&dst, w as usize, h as usize);
    z.compute(&s, &dd).expect("cpu compute").score()
}

/// Property 1 — every size down to 1×1 produces a finite score on the
/// GPU; `dims()` reports the *logical* size, not the padded one.
#[test]
fn gpu_scores_every_size_down_to_1px() {
    for n in [1u32, 2, 3, 4, 8, 16, 32, 48, 63, 64] {
        let r = make_image(n, n, 0);
        let d = make_image(n, n, 7);
        let mut z = gpu_opaque(n, n);
        assert_eq!(z.dims(), (n, n), "dims() must report logical size at {n}px");
        let s = z.compute_srgb_u8(&r, &d).expect("gpu score").value;
        assert!(s.is_finite(), "score at {n}px must be finite, got {s}");
    }
}

/// Property 2 — a constant colour difference must NOT vary with size.
/// Reflecting a solid image yields the same solid 64×64 (or native ≥64)
/// image at every size, so the score should be effectively constant.
/// Asserts well inside the user's 2.0pt tolerance.
#[test]
fn gpu_solid_color_diff_is_size_invariant() {
    let sizes = [1u32, 2, 4, 8, 16, 32, 48, 63, 64, 96, 128];
    let scores: Vec<f64> = sizes
        .iter()
        .map(|&n| {
            let r = solid(n, n, [100, 100, 100]);
            let d = solid(n, n, [120, 120, 120]);
            gpu_score(n, n, &r, &d)
        })
        .collect();
    let lo = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "solid-diff scores by size {:?} = {:?} (spread {:.4})",
        sizes,
        scores,
        hi - lo
    );
    assert!(
        hi - lo < 0.5,
        "solid-colour difference must be size-invariant; spread {:.4} over {:?}",
        hi - lo,
        scores
    );
}

/// Property 2b — non-square sub-64 (the 1:3-style ratios the corpus
/// sweeps) also score and stay invariant for solid colour.
#[test]
fn gpu_nonsquare_sub64_solid_invariant() {
    let dims = [(8u32, 24u32), (24, 8), (16, 48), (1, 50), (50, 1), (33, 17)];
    let scores: Vec<f64> = dims
        .iter()
        .map(|&(w, h)| {
            let r = solid(w, h, [40, 90, 200]);
            let d = solid(w, h, [60, 90, 200]);
            gpu_score(w, h, &r, &d)
        })
        .collect();
    let lo = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("nonsquare solid scores {:?} = {:?}", dims, scores);
    assert!(
        scores.iter().all(|s| s.is_finite()),
        "all non-square sub-64 solid scores must be finite: {scores:?}"
    );
    assert!(
        hi - lo < 0.5,
        "non-square solid-colour difference must be size-invariant; spread {:.4}",
        hi - lo
    );
}

/// Property 3 — GPU sub-64 (reflect-padded on-device) agrees with the
/// CPU canonical path (which reflect-pads identically) within the f32
/// kernel tolerance. Both sides feed the same profile the same padded
/// dims, so only the f32-vs-f64 feature drift remains.
#[test]
fn gpu_cpu_parity_sub64_textured() {
    for n in [8u32, 16, 32, 48, 63] {
        let r = make_image(n, n, 3);
        let d = make_image(n, n, 29);
        let g = gpu_score(n, n, &r, &d);
        let c = cpu_score(n, n, &r, &d);
        eprintln!("size {n}: gpu={g:.4} cpu={c:.4} |Δ|={:.4}", (g - c).abs());
        assert!(
            (g - c).abs() < 2.0,
            "GPU/CPU sub-64 parity at {n}px: gpu={g} cpu={c} (Δ {:.4} ≥ 2.0)",
            (g - c).abs()
        );
    }
}
