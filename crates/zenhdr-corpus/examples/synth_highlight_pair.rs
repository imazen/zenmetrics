//! Synthesize a controlled HDR pair that differs ONLY in the highlights —
//! the regime where the u8 PU-clamp goes blind but a faithful HDR metric still
//! sees the difference. Both images share an identical mid-tone background;
//! the reference has a bright highlight patch (`--ref-nits`, default 2000
//! cd/m²) and the distorted version clips that patch to `--dist-nits`
//! (default 200 cd/m²). Writes two small EXRs (absolute cd/m²).
//!
//! Usage: synth_highlight_pair <ref.exr> <dist.exr> [ref_nits] [dist_nits] [size]

use std::path::PathBuf;

use image::Rgb32FImage;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = std::env::args().skip(1);
    let ref_path =
        PathBuf::from(a.next().ok_or(
            "usage: synth_highlight_pair <ref.exr> <dist.exr> [ref_nits] [dist_nits] [size]",
        )?);
    let dist_path = PathBuf::from(a.next().ok_or("missing <dist.exr>")?);
    let ref_nits: f32 = a.next().map(|s| s.parse()).transpose()?.unwrap_or(2000.0);
    let dist_nits: f32 = a.next().map(|s| s.parse()).transpose()?.unwrap_or(200.0);
    let size: u32 = a.next().map(|s| s.parse()).transpose()?.unwrap_or(256);

    // Identical mid-tone background (~80 cd/m²); a centered quarter-size patch
    // is the only thing that differs between ref and dist.
    let bg = 80.0f32;
    let lo = size / 2 - size / 8;
    let hi = size / 2 + size / 8;
    let make = |patch: f32| -> Rgb32FImage {
        Rgb32FImage::from_fn(size, size, |x, y| {
            let v = if x >= lo && x < hi && y >= lo && y < hi {
                patch
            } else {
                bg
            };
            image::Rgb([v, v, v])
        })
    };
    make(ref_nits).save(&ref_path)?;
    make(dist_nits).save(&dist_path)?;
    eprintln!(
        "wrote {} (patch {ref_nits} cd/m²) + {} (patch {dist_nits} cd/m²), {size}x{size}",
        ref_path.display(),
        dist_path.display()
    );
    Ok(())
}
