//! Sub-8px reflect-pad: cvvdp's pyramid rejects below `2 × PYRAMID_MIN_DIM
//! = 8`; `CvvdpOpaque` reflect(mirror)-pads sub-8px inputs up to that floor
//! so it returns a finite JOD down to 1×1 instead of `InvalidImageSize`,
//! and a constant colour difference scores identically for sizes ≤ floor
//! (they map to the same padded image). cvvdp is display-aware, so the
//! padded image is scored at the padded resolution's PPD — a deterministic
//! fallback for an otherwise-unscoreable input.

#![cfg(all(feature = "cubecl-types", any(feature = "cuda", feature = "wgpu")))]

use cvvdp_gpu::{Backend, CvvdpOpaque, CvvdpParams};

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
    CvvdpOpaque::new(BACKEND_E, w, h, CvvdpParams::default())
        .expect("opaque new (sub-8 reflect-pad must not error on ≥1px)")
        .compute_srgb_u8(r, d)
        .expect("cvvdp compute")
        .value
}

#[test]
fn scores_every_size_down_to_1px() {
    for n in [1u32, 2, 3, 4, 7, 8, 16] {
        let r = make_image(n, n, 0);
        let d = make_image(n, n, 7);
        let mut z = CvvdpOpaque::new(BACKEND_E, n, n, CvvdpParams::default()).expect("new");
        assert_eq!(z.dims(), (n, n), "dims() must report logical size at {n}px");
        let s = z.compute_srgb_u8(&r, &d).expect("score").value;
        assert!(s.is_finite(), "cvvdp JOD at {n}px must be finite, got {s}");
    }
}

/// Sizes ≤ the 8px floor all map to the same padded 8×8 image, so a
/// constant-colour difference scores identically across 1..=8.
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
        "cvvdp solid-diff ≤floor {sizes:?} = {scores:?} (spread {:.6})",
        hi - lo
    );
    assert!(
        hi - lo < 1e-4,
        "cvvdp solid-colour difference must be invariant for sizes ≤ 8px floor; spread {:.6}",
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
        assert!(s.is_finite(), "cvvdp {w}x{h} JOD must be finite, got {s}");
    }
}
