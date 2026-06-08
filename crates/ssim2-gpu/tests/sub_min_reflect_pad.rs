//! Sub-8px reflect-pad: SSIMULACRA2's typed pipeline rejects `< 8×8`;
//! `Ssim2Opaque` reflect(mirror)-pads sub-8px inputs up to that floor so
//! it returns a finite score down to 1×1 instead of `InvalidImageSize`,
//! and a constant colour difference scores the same at every size.

#![cfg(all(feature = "cubecl-types", any(feature = "cuda", feature = "wgpu")))]

use ssim2_gpu::{Backend, Ssim2Opaque, Ssim2Params};

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
    Ssim2Opaque::new(BACKEND_E, w, h, Ssim2Params::DEFAULT)
        .expect("opaque new (sub-8 reflect-pad must not error on ≥1px)")
        .compute_srgb_u8(r, d)
        .expect("ssim2 compute")
        .value
}

#[test]
fn scores_every_size_down_to_1px() {
    for n in [1u32, 2, 3, 4, 7, 8, 16] {
        let r = make_image(n, n, 0);
        let d = make_image(n, n, 7);
        let mut z = Ssim2Opaque::new(BACKEND_E, n, n, Ssim2Params::DEFAULT).expect("new");
        assert_eq!(z.dims(), (n, n), "dims() must report logical size at {n}px");
        let s = z.compute_srgb_u8(&r, &d).expect("score").value;
        assert!(
            s.is_finite(),
            "ssim2 score at {n}px must be finite, got {s}"
        );
    }
}

/// The reflect-pad guarantees that every size **at or below the 8px
/// floor** maps to the same padded 8×8 image, so a constant-colour
/// difference scores identically across 1..=8 (they're the same scored
/// image). Above the floor, SSIMULACRA2 adds pyramid scales with size —
/// that inherent multi-scale size-dependence is NOT what this fix
/// addresses (the fix removes the sub-8 *error*, it doesn't flatten
/// ssim2's scale pyramid).
#[test]
fn solid_color_diff_invariant_at_or_below_floor() {
    let sizes = [1u32, 2, 3, 4, 5, 6, 7, 8];
    let scores: Vec<f64> = sizes
        .iter()
        .map(|&n| {
            score(
                n,
                n,
                &solid(n, n, [100, 100, 100]),
                &solid(n, n, [120, 120, 120]),
            )
        })
        .collect();
    let lo = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "ssim2 solid-diff ≤floor {sizes:?} = {scores:?} (spread {:.4})",
        hi - lo
    );
    assert!(
        hi - lo < 0.5,
        "ssim2 solid-colour difference must be invariant for sizes ≤ 8px floor; spread {:.4}",
        hi - lo
    );
}

#[test]
fn nonsquare_sub8_scores_finite() {
    for (w, h) in [(2u32, 24u32), (24, 2), (1, 30), (30, 1), (5, 3)] {
        let s = score(
            w,
            h,
            &solid(w, h, [40, 90, 200]),
            &solid(w, h, [60, 90, 200]),
        );
        assert!(s.is_finite(), "ssim2 {w}x{h} score must be finite, got {s}");
    }
}
