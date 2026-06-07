#![forbid(unsafe_code)]

//! CPU butteraugli scoring via the `butteraugli` crate.
//!
//! Returns both aggregations from a single `compute()` call:
//! - **max-norm** (`ButteraugliResult::score`) — the per-block maximum,
//!   sensitive to localised distortion.
//! - **3-norm** (`ButteraugliResult::pnorm_3`) — the libjxl-style
//!   `butteraugli_main --pnorm` aggregation used by the Cloudinary CID22
//!   paper.
//!
//! Both numbers are byproducts of the same internal heatmap, so it is
//! strictly cheaper to emit both than to run butteraugli twice.

use crate::decode::Rgb8Image;

/// Compute butteraugli once and return `(max_norm, pnorm_3)`.
pub(crate) fn score_both(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    use rgb::FromSlice;
    let (w0, h0) = (reference.width as usize, reference.height as usize);
    let (dw0, dh0) = (distorted.width as usize, distorted.height as usize);
    if reference.pixels.len() != w0 * h0 * 3 || distorted.pixels.len() != dw0 * dh0 * 3 {
        return Err("butteraugli: pixel buffer is not packed w*h*3 RGB8".into());
    }

    // Reflect(mirror)-pad sub-8px inputs up to butteraugli's 8×8 minimum
    // (same reflect-101 rule as the GPU metrics) so the CPU path scores
    // down to 1×1 instead of "image too small".
    let (ref_buf, pw, ph) =
        crate::metrics::pad_rgb8_to_min(&reference.pixels, reference.width, reference.height, 8);
    let (dst_buf, dw, dh) =
        crate::metrics::pad_rgb8_to_min(&distorted.pixels, distorted.width, distorted.height, 8);
    let (w, h) = (pw as usize, ph as usize);

    let ref_pixels = ref_buf.as_rgb();
    let dst_pixels = dst_buf.as_rgb();
    if ref_pixels.len() != w * h || dst_pixels.len() != (dw as usize) * (dh as usize) {
        return Err("butteraugli: pixel count != width*height".into());
    }
    let img1 = imgref::ImgRef::new(ref_pixels, w, h);
    let img2 = imgref::ImgRef::new(dst_pixels, dw as usize, dh as usize);
    let params = butteraugli::ButteraugliParams::new();
    let result =
        butteraugli::butteraugli(img1, img2, &params).map_err(|e| format!("butteraugli: {e}"))?;
    Ok((result.score as f64, result.pnorm_3))
}
