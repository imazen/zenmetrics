//! Strip-vs-whole-image parity for `Dssim::new_strip` /
//! `compute_stripped`.
//!
//! Goal: strip-processed score must agree with whole-image score
//! within f32 reordering noise (per-strip partial sums reorder f32
//! adds vs the whole-image atomic sum, which is well above
//! 1e-4 rel — see STRIP_PROCESSING.md).
//!
//! Backend selection mirrors `parity_lock.rs`.
//!
//! Coverage matrix:
//! * Pair path (`new` + `compute`)  vs  `new_strip` + `compute`.
//! * Cached-ref path (`set_reference` + `compute_with_reference`) vs
//!   `new_strip` + `compute`.
//! * Image sizes × `h_body` values across the matrix.
//! * Edge cases: image_h not divisible by `h_body`, single-strip
//!   (image_h ≤ h_body), full-image-as-one-strip (degenerate).
//! * Halo edge behavior: image-boundary rows must produce identical
//!   reflect/clamp output between whole-image and strip paths.

use cubecl::Runtime;
use dssim_gpu::{Dssim, Error};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
compile_error!(
    "dssim-gpu strip parity tests require either the `cuda` or `wgpu` feature to select a runtime"
);

macro_rules! make_client {
    () => {
        Backend::client(&Default::default())
    };
}

fn gradient_rgb(w: usize, h: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 255) / w.max(1)) as u8;
            let g = ((y * 255) / h.max(1)) as u8;
            let b = ((x ^ y) & 0xff) as u8;
            v.push(r);
            v.push(g);
            v.push(b);
        }
    }
    v
}

fn add_noise(data: &[u8], amount: i16) -> Vec<u8> {
    use std::num::Wrapping;
    let mut out = Vec::with_capacity(data.len());
    let mut seed = Wrapping(98765_u32);
    for &v in data {
        seed = seed * Wrapping(1103515245_u32) + Wrapping(12345_u32);
        let noise = ((seed.0 >> 16) as i16 % (amount * 2 + 1)) - amount;
        out.push((v as i16 + noise).clamp(0, 255) as u8);
    }
    out
}

/// Whole-image pair path: `Dssim::new(w, h)` + `compute(ref, dist)`.
/// Used as the primary oracle now that `compute_post_srgb` builds
/// both pyramids.
fn whole_image_pair(w: u32, h: u32, ref_rgb: &[u8], dist_rgb: &[u8]) -> f64 {
    let mut d = Dssim::<Backend>::new(make_client!(), w, h).unwrap();
    d.compute(ref_rgb, dist_rgb).unwrap().score
}

/// Whole-image cached-ref path.
fn whole_image_cached(w: u32, h: u32, ref_rgb: &[u8], dist_rgb: &[u8]) -> f64 {
    let mut d = Dssim::<Backend>::new(make_client!(), w, h).unwrap();
    d.set_reference(ref_rgb).unwrap();
    d.compute_with_reference(dist_rgb).unwrap().score
}

/// Strip path: `new_strip` + `compute` (auto-routes to compute_stripped).
fn strip_compute(w: u32, h: u32, h_body: u32, ref_rgb: &[u8], dist_rgb: &[u8]) -> f64 {
    let mut d = Dssim::<Backend>::new_strip(make_client!(), w, h, h_body).unwrap();
    d.compute(ref_rgb, dist_rgb).unwrap().score
}

/// Strip path via the explicit `compute_stripped` entry point.
fn strip_compute_explicit(w: u32, h: u32, h_body: u32, ref_rgb: &[u8], dist_rgb: &[u8]) -> f64 {
    let mut d = Dssim::<Backend>::new_strip(make_client!(), w, h, h_body).unwrap();
    d.compute_stripped(ref_rgb, dist_rgb).unwrap().score
}

const REL_TOL: f64 = 1e-4;

fn check_rel(label: &str, whole: f64, strip: f64) {
    let rel = (strip - whole).abs() / whole.max(1e-6);
    eprintln!(
        "{label}: whole={whole:.8}, strip={strip:.8}, rel={:.4}%",
        rel * 100.0
    );
    assert!(
        rel < REL_TOL,
        "{label}: strip drifted from whole by {rel:.6} rel (whole={whole}, strip={strip})"
    );
}

// ──────────────────────── Pair-path matrix ────────────────────────

/// Pair path matrix at 256×256.
#[test]
fn strip_vs_pair_256() {
    let w = 256_u32;
    let h = 256_u32;
    let r = gradient_rgb(w as usize, h as usize);
    let d = add_noise(&r, 16);
    let whole = whole_image_pair(w, h, &r, &d);

    for h_body in &[64_u32, 128] {
        let s = strip_compute(w, h, *h_body, &r, &d);
        check_rel(&format!("pair 256 h_body={h_body}"), whole, s);
    }
}

/// Pair path matrix at 512×512.
#[test]
fn strip_vs_pair_512() {
    let w = 512_u32;
    let h = 512_u32;
    let r = gradient_rgb(w as usize, h as usize);
    let d = add_noise(&r, 16);
    let whole = whole_image_pair(w, h, &r, &d);

    for h_body in &[128_u32, 256] {
        let s = strip_compute(w, h, *h_body, &r, &d);
        check_rel(&format!("pair 512 h_body={h_body}"), whole, s);
    }
}

/// Pair path matrix at 1024×1024 — the size in the verification gate.
#[test]
fn strip_vs_pair_1024() {
    let w = 1024_u32;
    let h = 1024_u32;
    let r = gradient_rgb(w as usize, h as usize);
    let d = add_noise(&r, 12);
    let whole = whole_image_pair(w, h, &r, &d);

    for h_body in &[256_u32, 512] {
        let s = strip_compute(w, h, *h_body, &r, &d);
        check_rel(&format!("pair 1024 h_body={h_body}"), whole, s);
    }
}

// ─────────────────────── Cached-ref matrix ───────────────────────

/// Cached-ref path matrix at 256×256.
#[test]
fn strip_vs_cached_256() {
    let w = 256_u32;
    let h = 256_u32;
    let r = gradient_rgb(w as usize, h as usize);
    let d = add_noise(&r, 16);
    let whole = whole_image_cached(w, h, &r, &d);

    for h_body in &[64_u32, 128] {
        let s = strip_compute(w, h, *h_body, &r, &d);
        check_rel(&format!("cached 256 h_body={h_body}"), whole, s);
    }
}

/// Cached-ref path matrix at 512×512.
#[test]
fn strip_vs_cached_512() {
    let w = 512_u32;
    let h = 512_u32;
    let r = gradient_rgb(w as usize, h as usize);
    let d = add_noise(&r, 16);
    let whole = whole_image_cached(w, h, &r, &d);

    for h_body in &[128_u32, 256] {
        let s = strip_compute(w, h, *h_body, &r, &d);
        check_rel(&format!("cached 512 h_body={h_body}"), whole, s);
    }
}

/// Cached-ref path matrix at 1024×1024.
#[test]
fn strip_vs_cached_1024() {
    let w = 1024_u32;
    let h = 1024_u32;
    let r = gradient_rgb(w as usize, h as usize);
    let d = add_noise(&r, 12);
    let whole = whole_image_cached(w, h, &r, &d);

    for h_body in &[256_u32, 512] {
        let s = strip_compute(w, h, *h_body, &r, &d);
        check_rel(&format!("cached 1024 h_body={h_body}"), whole, s);
    }
}

// ─────────────────────── Edge cases ───────────────────────

/// `image_h` not a multiple of `h_body`: last strip is partial.
/// 384 with h_body=128 → 3 full strips (no partial).
/// 400 with h_body=128 → 3 strips of 128 + 1 partial of 16
///   — but h_body=128 requires h_body divisible by 16, satisfied.
///   image_h=400 means rows 0..400 with last body 384..400 (16 rows).
#[test]
fn strip_image_h_not_divisible_by_h_body() {
    let w = 256_u32;
    let h = 400_u32;
    let r = gradient_rgb(w as usize, h as usize);
    let d = add_noise(&r, 12);
    let whole = whole_image_pair(w, h, &r, &d);

    let s = strip_compute(w, h, 128, &r, &d);
    check_rel("non-divisible h=400 h_body=128", whole, s);
}

/// `image_h <= h_body`: degenerates to a single strip whose body is
/// the full image. Strip-buffer height equals image_h after clamp.
#[test]
fn strip_single_strip_image_h_le_h_body() {
    let w = 256_u32;
    let h = 128_u32;
    let r = gradient_rgb(w as usize, h as usize);
    let d = add_noise(&r, 16);
    let whole = whole_image_pair(w, h, &r, &d);

    // h_body=256 ≥ image_h=128 → one strip of body 128.
    let s = strip_compute(w, h, 256, &r, &d);
    check_rel("single strip h=128 h_body=256", whole, s);
}

/// Full-image-as-one-strip degenerate: strip buffer covers the entire
/// image (h_body large enough that there's no second strip).
#[test]
fn strip_full_image_as_one_strip() {
    let w = 320_u32;
    let h = 256_u32;
    let r = gradient_rgb(w as usize, h as usize);
    let d = add_noise(&r, 16);
    let whole = whole_image_pair(w, h, &r, &d);

    // h_body=256 with image_h=256 → exactly one strip whose body
    // covers the whole image. Halo region clamps to image edge.
    let s = strip_compute(w, h, 256, &r, &d);
    check_rel("full-image-as-strip h=256 h_body=256", whole, s);
}

/// Halo / edge behavior: at the image's top and bottom boundary, the
/// strip path's reflect/clamp must produce the same SSIM body
/// contribution as the whole-image path. A targeted off-by-one in
/// the halo-or-data clamp would show as a measurable drift.
///
/// Constructs an image where the top + bottom 32 rows differ from
/// the middle (high-contrast band) so any edge mishandling biases
/// the score. Compares strip (which sees the boundary at every
/// strip-top and strip-bottom edge that overlaps the image edge) to
/// whole-image.
#[test]
fn strip_halo_edge_matches_whole_image() {
    let w = 256_u32;
    let h = 384_u32;
    let mut r = gradient_rgb(w as usize, h as usize);
    // Force a strong band at the top and bottom so any halo
    // mishandling biases the SSIM there.
    for y in 0..32 {
        for x in 0..(w as usize) {
            let i = (y * w as usize + x) * 3;
            r[i] = 255;
            r[i + 1] = 0;
            r[i + 2] = 255;
        }
    }
    for y in (h as usize - 32)..(h as usize) {
        for x in 0..(w as usize) {
            let i = (y * w as usize + x) * 3;
            r[i] = 0;
            r[i + 1] = 255;
            r[i + 2] = 0;
        }
    }
    let d = add_noise(&r, 20);

    let whole = whole_image_pair(w, h, &r, &d);

    // Multiple strips: each interior strip sees only halo data on
    // both sides; the first and last strips see image-edge clamp on
    // one side and data on the other.
    let s = strip_compute(w, h, 128, &r, &d);
    check_rel("halo edge h=384 h_body=128", whole, s);
}

// ─────────────────────── Cross-implementation ───────────────────────

/// `compute()` on a strip-mode instance must auto-route to
/// `compute_stripped()` (backwards compatibility for callers that
/// don't know about the strip API).
#[test]
fn compute_routes_to_compute_stripped_in_strip_mode() {
    let w = 384_u32;
    let h = 384_u32;
    let r = gradient_rgb(w as usize, h as usize);
    let d = add_noise(&r, 12);

    let auto = strip_compute(w, h, 128, &r, &d);
    let explicit = strip_compute_explicit(w, h, 128, &r, &d);
    let diff = (auto - explicit).abs();
    let rel = diff / auto.max(1e-6);
    eprintln!(
        "auto-route: auto={auto:.8}, explicit={explicit:.8}, abs_diff={diff:.2e}, rel={:.4}%",
        rel * 100.0
    );
    // Each call constructs a fresh client and re-runs the GPU
    // pipeline; the f32 reductions reorder between launches. The
    // routing parity check is "both paths land within reorder noise"
    // (matches the strip-vs-whole REL_TOL), not bit-exactness.
    assert!(
        rel < REL_TOL,
        "compute() and compute_stripped() must agree within reorder noise, rel={rel:.6}"
    );
}

/// `compute_stripped()` on a non-strip instance returns an error.
#[test]
fn compute_stripped_rejects_non_strip_instance() {
    let w = 256_u32;
    let h = 256_u32;
    let r = gradient_rgb(w as usize, h as usize);
    let mut d = Dssim::<Backend>::new(make_client!(), w, h).unwrap();
    let result = d.compute_stripped(&r, &r);
    assert!(
        result.is_err(),
        "compute_stripped on whole-image instance should error"
    );
}

// ─────────────────────── Original tests ───────────────────────

/// Cross-strip-size parity: two different strip configs should give
/// matching scores to each other, proving the drift is bounded by
/// f32 reordering noise rather than a real bug.
#[test]
fn cross_strip_size_parity() {
    let w = 768_u32;
    let h = 768_u32;
    let r = gradient_rgb(w as usize, h as usize);
    let dist = add_noise(&r, 8);

    let mut a = Dssim::<Backend>::new_strip(make_client!(), w, h, 128).unwrap();
    let mut b = Dssim::<Backend>::new_strip(make_client!(), w, h, 256).unwrap();
    let sa = a.compute(&r, &dist).unwrap().score;
    let sb = b.compute(&r, &dist).unwrap().score;
    let rel = (sa - sb).abs() / sa.max(1e-6);
    eprintln!(
        "strip h_body=128 vs 256: a={sa:.8}, b={sb:.8}, rel={:.4} %",
        rel * 100.0
    );
    // Tightened 2026-05-22 from 1e-3 to 1e-4 (measured 9e-6 on this
    // fixture; 1e-4 leaves 10× margin while still catching real
    // strip-orchestration bugs).
    assert!(
        rel < 1e-4,
        "cross-strip drift {rel:.6} rel exceeds tolerance"
    );
}

/// Identical-image strip score should be ~0.
#[test]
fn strip_identical_is_zero() {
    let w = 512_u32;
    let h = 256_u32;
    let img = gradient_rgb(w as usize, h as usize);
    let mut d = Dssim::<Backend>::new_strip(make_client!(), w, h, 128).unwrap();
    let s = d.compute(&img, &img).unwrap().score;
    eprintln!("strip identical: dssim = {s:.8}");
    // Backend-aware. CUDA is bit-exact 0.0 for identical input (measured
    // 2026-05-22), so keep the tight 1e-7 there — it catches any real
    // identical-handling regression. The wgpu/Metal backend leaves f32
    // strip-reduction residue from FMA contraction / fast-math (measured
    // 1.3e-5 on Apple Metal CI, 2026-06-01) — still negligible for DSSIM
    // and not a logic bug (same cubecl kernel source as CUDA).
    #[cfg(feature = "cuda")]
    let tol = 1e-7;
    #[cfg(not(feature = "cuda"))]
    let tol = 1e-4;
    assert!(s < tol, "expected ~0, got {s} (tol {tol:e})");
}

/// `dimensions()` returns image dims (not strip-buffer dims) for
/// strip-mode instances.
#[test]
fn strip_dimensions_reports_image_size() {
    let d = Dssim::<Backend>::new_strip(make_client!(), 1024, 600, 128).unwrap();
    assert_eq!(d.dimensions(), (1024, 600));
    assert!(d.is_strip_mode());
}

/// Whole-image construction reports `is_strip_mode() == false`.
#[test]
fn whole_image_is_not_strip_mode() {
    let d = Dssim::<Backend>::new(make_client!(), 256, 256).unwrap();
    assert!(!d.is_strip_mode());
}

/// `new_strip` rejects `h_body` that isn't a multiple of 16.
#[test]
fn strip_rejects_unaligned_body() {
    let r = Dssim::<Backend>::new_strip(make_client!(), 256, 256, 100);
    assert!(r.is_err(), "expected error for h_body=100 (not /16)");
}

#[test]
fn strip_constructor_sub_min_routes_to_padded_full() {
    // Sub-MIN_PAD_DIM strip requests route to the Full constructor,
    // which reflect-pads to the pyramid floor and scores (down to 1×1),
    // rather than rejecting. `dimensions()` reports the logical size and
    // the instance is whole-image (not strip) after the reroute.
    let mut d =
        Dssim::<Backend>::new_strip(make_client!(), 4, 4, 64).expect("4x4 strip routes to Full");
    assert_eq!(d.dimensions(), (4, 4));
    assert!(!d.is_strip_mode(), "sub-min reroute lands on the whole-image path");
    let buf = vec![0_u8; 4 * 4 * 3];
    assert!(d.compute(&buf, &buf).is_ok(), "4x4 must score");

    // 0-dim is still rejected.
    let r = Dssim::<Backend>::new_strip(make_client!(), 0, 4, 64);
    assert!(matches!(r, Err(Error::InvalidImageSize)));
}
