//! End-to-end parity-lock test against the published IW-SSIM PyTorch
//! reference (`IW_SSIM_PyTorch.py` from
//! <https://ece.uwaterloo.ca/~z70wang/research/iwssim/python_iwssim.zip>).
//!
//! Reference scores collected by patching the upstream Python file to
//! use `torch.linalg.eig` in place of the long-removed `torch.eig`,
//! then running the script's stock `images/Ref.bmp`/`images/Dist.jpg`
//! and `images/Ref2.bmp`/`images/Dist2.jpg` pairs.
//!
//! The test only runs when CUDA is available — it's a real GPU launch
//! against the canonical pair, not a unit test of one kernel. Guarded
//! by `RUN_GPU_PARITY=1` so the default `cargo test` works on machines
//! without a CUDA device.
//!
//! Required env vars when running:
//!
//! - `RUN_GPU_PARITY=1` — enables the test
//! - `IWSSIM_PARITY_REFS=<dir>` — directory containing
//!   `Ref.bmp`/`Dist.jpg`/`Ref2.bmp`/`Dist2.jpg`

#![cfg(feature = "cuda")]

use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
use image::GenericImageView;
use iwssim_gpu::Iwssim;

fn rgb2gray_bt601_round(rgba: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(rgba.len() / 3);
    for chunk in rgba.chunks_exact(3) {
        let r = chunk[0] as f64;
        let g = chunk[1] as f64;
        let b = chunk[2] as f64;
        let y = 0.2989 * r + 0.5870 * g + 0.1140 * b;
        let rounded = if y.fract().abs() == 0.5 {
            let f = y.floor();
            if (f as i64) % 2 == 0 {
                f
            } else {
                f + 1.0
            }
        } else {
            (y + 0.5).floor()
        };
        out.push(rounded as f32);
    }
    out
}

fn score_one(ref_path: &str, dis_path: &str) -> (f64, [f64; iwssim_gpu::NUM_SCALES]) {
    let r_img = image::open(ref_path).expect("open ref");
    let d_img = image::open(dis_path).expect("open dist");
    let (w, h) = r_img.dimensions();
    let r_rgb = r_img.to_rgb8().into_raw();
    let d_rgb = d_img.to_rgb8().into_raw();
    let r_gray = rgb2gray_bt601_round(&r_rgb);
    let d_gray = rgb2gray_bt601_round(&d_rgb);
    let client = CudaRuntime::client(&Default::default());
    let mut iw = Iwssim::<CudaRuntime>::new(client, w, h).expect("Iwssim::new");
    let r = iw.compute_gray(&r_gray, &d_gray).expect("compute_gray");
    (r.score, r.per_scale)
}

#[test]
fn ref_vs_dist_matches_python_reference() {
    if std::env::var("RUN_GPU_PARITY").is_err() {
        eprintln!("skipping (set RUN_GPU_PARITY=1 to enable)");
        return;
    }
    let dir =
        std::env::var("IWSSIM_PARITY_REFS").expect("set IWSSIM_PARITY_REFS=<refs-dir>");
    let r = format!("{dir}/Ref.bmp");
    let d = format!("{dir}/Dist.jpg");
    let (score, _per) = score_one(&r, &d);
    // Published Python score (PyTorch 2.12, `torch.linalg.eig` patch).
    let expected = 0.803405;
    let rel = ((score - expected) / expected).abs();
    assert!(
        rel < 5e-3,
        "score {score} differs from {expected} by rel={rel} (>0.5%)"
    );
}

#[test]
fn ref2_vs_dist2_matches_python_reference() {
    if std::env::var("RUN_GPU_PARITY").is_err() {
        return;
    }
    let dir = std::env::var("IWSSIM_PARITY_REFS").expect("set IWSSIM_PARITY_REFS");
    let r = format!("{dir}/Ref2.bmp");
    let d = format!("{dir}/Dist2.jpg");
    let (score, _per) = score_one(&r, &d);
    let expected = 0.840189;
    let rel = ((score - expected) / expected).abs();
    assert!(
        rel < 5e-3,
        "score {score} differs from {expected} by rel={rel} (>0.5%)"
    );
}

#[test]
fn ref_vs_ref_is_one() {
    if std::env::var("RUN_GPU_PARITY").is_err() {
        return;
    }
    let dir = std::env::var("IWSSIM_PARITY_REFS").expect("set IWSSIM_PARITY_REFS");
    let r = format!("{dir}/Ref.bmp");
    let (score, _per) = score_one(&r, &r);
    assert!(
        (score - 1.0).abs() < 1e-5,
        "self-identity score {score} ≠ 1.0"
    );
}

/// `compute_with_reference` should return the same score as
/// `compute_gray` on the same pair — caching the reference pyramid
/// is a perf optimization, not a numerical change.
#[test]
fn cached_reference_matches_full_compute() {
    if std::env::var("RUN_GPU_PARITY").is_err() {
        return;
    }
    let dir = std::env::var("IWSSIM_PARITY_REFS").expect("set IWSSIM_PARITY_REFS");
    let ref_path = format!("{dir}/Ref.bmp");
    let dis_path = format!("{dir}/Dist.jpg");

    let r_img = image::open(&ref_path).expect("open ref");
    let d_img = image::open(&dis_path).expect("open dist");
    let (w, h) = r_img.dimensions();
    let r_rgb = r_img.to_rgb8().into_raw();
    let d_rgb = d_img.to_rgb8().into_raw();
    let r_gray = rgb2gray_bt601_round(&r_rgb);
    let d_gray = rgb2gray_bt601_round(&d_rgb);

    let client = CudaRuntime::client(&Default::default());
    let mut iw = Iwssim::<CudaRuntime>::new(client, w, h).expect("Iwssim::new");

    let full = iw.compute_gray(&r_gray, &d_gray).expect("full").score;
    iw.set_reference(&r_gray).expect("set_reference");
    let cached = iw.compute_with_reference(&d_gray).expect("cwr").score;

    let rel = ((cached - full) / full).abs();
    assert!(
        rel < 1e-4,
        "cached score {cached} differs from full {full} by rel={rel}"
    );
}
