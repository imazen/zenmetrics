#![forbid(unsafe_code)]

//! CPU SSIMULACRA2 via Imazen's local `fast-ssim2` crate (our SIMD SSIMULACRA2 implementation),
//! replacing the third-party `ssimulacra2` crate so the ssim2 score builds from local source
//! ("scores use local, crates versions banned"). `fast-ssim2`'s `imgref` feature provides the
//! `ToLinearRgb` impl for `ImgRef<[u8;3]>`, treating the 8-bit input as sRGB — same color
//! interpretation as the prior path (sRGB transfer, BT.709 primaries).

use crate::decode::Rgb8Image;

pub(crate) fn score(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<f64, Box<dyn std::error::Error>> {
    let r = to_img(reference)?;
    let d = to_img(distorted)?;
    let s = fast_ssim2::compute_ssimulacra2(r.as_ref(), d.as_ref())
        .map_err(|e| format!("fast-ssim2: {e:?}"))?;
    Ok(s)
}

/// Pack the decoder's flat RGB8 buffer into an `ImgVec<[u8; 3]>` (sRGB) that fast-ssim2 accepts.
///
/// Sub-8px images are reflect(mirror)-padded up to fast-ssim2's 8px
/// pyramid floor (same reflect-101 rule as the GPU metrics), so the CPU
/// path scores down to 1×1 instead of `InvalidImageSize`.
fn to_img(img: &Rgb8Image) -> Result<imgref::ImgVec<[u8; 3]>, Box<dyn std::error::Error>> {
    let (w, h) = (img.width as usize, img.height as usize);
    if w == 0 || h == 0 {
        return Err("ssim2: image has zero dimension".into());
    }
    if img.pixels.len() != w * h * 3 {
        return Err("ssim2: pixel buffer is not packed w*h*3 RGB8".into());
    }
    let (buf, pw, ph) = crate::metrics::pad_rgb8_to_min(&img.pixels, img.width, img.height, 8);
    let px: Vec<[u8; 3]> = buf.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    Ok(imgref::ImgVec::new(px, pw as usize, ph as usize))
}
