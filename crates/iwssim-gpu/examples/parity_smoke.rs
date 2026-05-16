//! End-to-end smoke + parity check against the published PyTorch
//! reference on the IW-SSIM author's `images/Ref.bmp` / `images/Dist.jpg`
//! pair.
//!
//! Reference scores (computed by running the patched
//! `IW_SSIM_PyTorch.py` under PyTorch 2.12 with `linalg.eig` in place
//! of the removed `torch.eig`):
//!
//! - `Ref.bmp`  vs `Dist.jpg`  → 0.803405
//! - `Ref2.bmp` vs `Dist2.jpg` → 0.840189
//! - `Ref.bmp`  vs `Ref.bmp`   → 1.0
//!
//! Usage:
//!     cargo run -p iwssim-gpu --example parity_smoke --release --no-default-features --features cuda \
//!         -- /path/to/Ref.bmp /path/to/Dist.jpg [expected_score]

use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
use image::GenericImageView;
use iwssim_gpu::Iwssim;

fn rgb2gray_bt601_round(rgba: &[u8], w: u32, h: u32) -> Vec<f32> {
    // `np.round` is banker's rounding; we replicate that on f64 here
    // so the resulting f32 buffer matches the reference exactly.
    let mut out = Vec::with_capacity((w * h) as usize);
    for chunk in rgba.chunks_exact(3) {
        let r = chunk[0] as f64;
        let g = chunk[1] as f64;
        let b = chunk[2] as f64;
        let y = 0.2989 * r + 0.5870 * g + 0.1140 * b;
        // Round-half-to-even (Python's np.round semantic).
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

fn load_rgb(path: &str) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).expect("open image");
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8().into_raw();
    (rgb, w, h)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let ref_path = args.next().expect("Usage: parity_smoke REF DIST [expected]");
    let dis_path = args.next().expect("Usage: parity_smoke REF DIST [expected]");
    let expected: Option<f64> = args.next().and_then(|s| s.parse().ok());

    let (ref_rgb, rw, rh) = load_rgb(&ref_path);
    let (dis_rgb, dw, dh) = load_rgb(&dis_path);
    assert_eq!(
        (rw, rh),
        (dw, dh),
        "ref and dist must have the same dimensions"
    );

    let ref_gray = rgb2gray_bt601_round(&ref_rgb, rw, rh);
    let dis_gray = rgb2gray_bt601_round(&dis_rgb, dw, dh);

    let client = CudaRuntime::client(&Default::default());
    let mut iw = Iwssim::<CudaRuntime>::new(client, rw, rh).expect("Iwssim::new");
    let r = iw
        .compute_gray(&ref_gray, &dis_gray)
        .expect("compute_gray");

    println!("IW-SSIM (gray): {:.6}", r.score);
    println!("per-scale:");
    for (i, v) in r.per_scale.iter().enumerate() {
        println!("  scale {}: {:.6}", i, v);
    }
    if let Some(e) = expected {
        let diff = (r.score - e).abs();
        let rel = if e != 0.0 { diff / e.abs() } else { diff };
        println!(
            "expected {:.6}, abs_diff {:.6}, rel_diff {:.6}",
            e, diff, rel
        );
        if rel > 1e-3 {
            eprintln!("PARITY FAILED: relative diff > 1e-3");
            std::process::exit(1);
        }
    }
}
