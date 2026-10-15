//! Image → network input conversion.

use crate::za;

/// Planar `[3, h, w]` f32 in `[-1, 1]` from strided packed sRGB8, matching the
/// upstream LPIPS loaders bit for bit: `(v as f32 / 255) * 2 - 1`, computed in
/// f32 in that order. No linearisation: LPIPS is trained on sRGB-encoded
/// values.
pub(crate) fn rgb8_to_planar_pm1(img: &za::Rgb8In<'_>) -> Vec<f32> {
    let (w, h) = (img.width, img.height);
    let plane = w * h;
    let mut out = vec![0.0f32; 3 * plane];
    let (r, rest) = out.split_at_mut(plane);
    let (g, b) = rest.split_at_mut(plane);
    for y in 0..h {
        let row = &img.pixels[y * img.stride..y * img.stride + 3 * w];
        let (ro, go, bo) = (
            &mut r[y * w..(y + 1) * w],
            &mut g[y * w..(y + 1) * w],
            &mut b[y * w..(y + 1) * w],
        );
        for (((px, ro), go), bo) in row.as_chunks::<3>().0.iter().zip(ro).zip(go).zip(bo) {
            *ro = (f32::from(px[0]) / 255.0) * 2.0 - 1.0;
            *go = (f32::from(px[1]) / 255.0) * 2.0 - 1.0;
            *bo = (f32::from(px[2]) / 255.0) * 2.0 - 1.0;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strided_matches_packed() {
        let (w, h) = (5usize, 3usize);
        let packed: Vec<u8> = (0..w * h * 3).map(|i| (i * 37 % 256) as u8).collect();
        let stride = w * 3 + 7;
        let mut strided = vec![0xAAu8; stride * h];
        for y in 0..h {
            strided[y * stride..y * stride + w * 3]
                .copy_from_slice(&packed[y * w * 3..(y + 1) * w * 3]);
        }
        let a = rgb8_to_planar_pm1(&za::Rgb8In {
            pixels: &packed,
            width: w,
            height: h,
            stride: w * 3,
        });
        let b = rgb8_to_planar_pm1(&za::Rgb8In {
            pixels: &strided,
            width: w,
            height: h,
            stride,
        });
        assert_eq!(a, b);
        assert_eq!(a[0], -1.0);
        assert_eq!(a[w * h], (37.0f32 / 255.0) * 2.0 - 1.0);
    }
}
