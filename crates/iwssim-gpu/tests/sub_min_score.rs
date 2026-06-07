//! Sub-176px small-image handling: IW-SSIM's 5-level pyramid + 11×11
//! valid-mode SSIM needs `min(W,H) ≥ MIN_NATIVE_DIM = 176`. The opaque
//! `IwssimParams::DEFAULT` now enables the **tile** small-image strategy
//! (the empirically best of the three on the 980-pair CID22 validation,
//! `benchmarks/iwssim_smallimg/`), so the scorer returns a finite score
//! down to 1×1 instead of `InvalidImageSize`. A constant colour
//! difference scores identically for all sizes ≤ the floor (they tile to
//! the same image).

#![cfg(all(feature = "cubecl-types", any(feature = "cuda", feature = "wgpu")))]

use iwssim_gpu::{Backend, IwssimOpaque, IwssimParams};

#[cfg(feature = "cuda")]
const BACKEND_E: Backend = Backend::Cuda;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
const BACKEND_E: Backend = Backend::Wgpu;

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

fn score(w: u32, h: u32, r: &[u8], d: &[u8]) -> f64 {
    IwssimOpaque::new(BACKEND_E, w, h, IwssimParams::DEFAULT)
        .expect("opaque new (DEFAULT enables small-image tiling; must not error on ≥1px)")
        .compute_srgb_u8(r, d)
        .expect("iwssim compute")
        .value
}

#[test]
fn scores_every_size_down_to_1px() {
    for n in [1u32, 2, 4, 8, 16, 64, 128, 176, 256] {
        let r = make_image(n, n, 0);
        let d = make_image(n, n, 7);
        let mut z = IwssimOpaque::new(BACKEND_E, n, n, IwssimParams::DEFAULT).expect("new");
        assert_eq!(z.dims(), (n, n), "dims() must report logical size at {n}px");
        let s = z.compute_srgb_u8(&r, &d).expect("score").value;
        assert!(s.is_finite(), "iwssim score at {n}px must be finite, got {s}");
    }
}

/// Sizes ≤ the 176px floor all tile to the same image, so a constant-
/// colour difference scores identically across them.
#[test]
fn solid_color_diff_invariant_at_or_below_floor() {
    let sizes = [1u32, 2, 4, 8, 16, 64, 128, 176];
    let scores: Vec<f64> = sizes
        .iter()
        .map(|&n| score(n, n, &solid(n, n, [100, 100, 100]), &solid(n, n, [120, 120, 120])))
        .collect();
    let lo = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("iwssim solid-diff ≤floor {sizes:?} = {scores:?} (spread {:.6})", hi - lo);
    assert!(
        hi - lo < 1e-4,
        "iwssim solid-colour difference must be invariant for sizes ≤ 176px floor; spread {:.6}",
        hi - lo
    );
}

#[test]
fn nonsquare_sub176_scores_finite() {
    for (w, h) in [(16u32, 200u32), (200, 16), (1, 180), (180, 1), (40, 90)] {
        let s = score(w, h, &solid(w, h, [40, 90, 200]), &solid(w, h, [60, 90, 200]));
        assert!(s.is_finite(), "iwssim {w}x{h} score must be finite, got {s}");
    }
}
