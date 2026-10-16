//! Image → network input conversion.

use crate::par;
use crate::za;

/// `img` → planar `[3, h, w]` with `f` applied per channel byte, rows split
/// across workers (each output element's conversion is unchanged — only the
/// schedule over rows differs).
fn planar_map<T: Copy + Default + Send>(
    img: &za::Rgb8In<'_>,
    f: impl Fn(u8) -> T + Sync,
) -> Vec<T> {
    let (w, h) = (img.width, img.height);
    let plane = w * h;
    let mut out = vec![T::default(); 3 * plane];
    if plane == 0 {
        return out;
    }
    let (r, rest) = out.split_at_mut(plane);
    let (g, b) = rest.split_at_mut(plane);
    let threads = par::threads_for(3 * plane);
    let convert = |y: usize, red: &mut [T], green: &mut [T], blue: &mut [T]| {
        let row = &img.pixels[y * img.stride..y * img.stride + 3 * w];
        for (((pixel, red), green), blue) in
            row.as_chunks::<3>().0.iter().zip(red).zip(green).zip(blue)
        {
            *red = f(pixel[0]);
            *green = f(pixel[1]);
            *blue = f(pixel[2]);
        }
    };
    if threads <= 1 {
        for y in 0..h {
            convert(
                y,
                &mut r[y * w..(y + 1) * w],
                &mut g[y * w..(y + 1) * w],
                &mut b[y * w..(y + 1) * w],
            );
        }
    } else {
        let mut rows: Vec<_> = r
            .chunks_mut(w)
            .zip(g.chunks_mut(w))
            .zip(b.chunks_mut(w))
            .enumerate()
            .collect();
        par::par_each_mut(&mut rows, threads, |row| {
            let (y, ((r, g), b)) = row;
            convert(*y, r, g, b);
        });
    }
    out
}

/// Planar `[3, h, w]` f32 in `[-1, 1]` from strided packed sRGB8, matching the
/// upstream LPIPS loaders bit for bit: `(v as f32 / 255) * 2 - 1`, computed in
/// f32 in that order. No linearisation: LPIPS is trained on sRGB-encoded
/// values.
pub(crate) fn rgb8_to_planar_pm1(img: &za::Rgb8In<'_>) -> Vec<f32> {
    planar_map(img, |v| (f32::from(v) / 255.0) * 2.0 - 1.0)
}

/// Planar `[3, h, w]` f32 in `0..=255` from strided packed sRGB8: `v as f32`,
/// exact — the way CKDN's upstream `PrefetchLoader` feeds uint8 tensors to the
/// network (its ImageNet mean/std are then applied inside the graph).
pub(crate) fn rgb8_to_planar_255(img: &za::Rgb8In<'_>) -> Vec<f32> {
    planar_map(img, f32::from)
}

/// Planar `[3, h, w]` u8 from strided packed sRGB8 — for model
/// preprocessings that resample in u8 (PIL `Image.resize` fixed-point
/// path).
pub(crate) fn rgb8_to_planar_u8(img: &za::Rgb8In<'_>) -> Vec<u8> {
    planar_map(img, std::convert::identity)
}

/// Planar `[3, h, w]` f32 in `[0, 1]` from strided packed sRGB8:
/// `v as f32 / 255`, as torchvision's `ToTensor` computes it.
pub(crate) fn rgb8_to_planar_01(img: &za::Rgb8In<'_>) -> Vec<f32> {
    planar_map(img, |v| f32::from(v) / 255.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversions_keep_identical_bits_for_one_and_many_workers() {
        let (w, h, stride) = (513, 385, 513 * 3 + 13);
        let px: Vec<u8> = (0..stride * h)
            .map(|i| ((i * 73 + i / 7 * 19) % 256) as u8)
            .collect();
        let img = za::Rgb8In {
            pixels: &px,
            width: w,
            height: h,
            stride,
        };
        for f in [rgb8_to_planar_pm1, rgb8_to_planar_255, rgb8_to_planar_01] {
            let (a, n) = par::observe_workers(Some(1), || f(&img));
            assert_eq!(n, 0);
            let (b, n) = par::observe_workers(Some(16), || f(&img));
            assert!(n > 0);
            assert_eq!(
                a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        }
        let (a, _) = par::observe_workers(Some(1), || rgb8_to_planar_u8(&img));
        let (b, _) = par::observe_workers(Some(16), || rgb8_to_planar_u8(&img));
        assert_eq!(a, b);
        for (w, h) in [(0, 5), (5, 0), (0, 0)] {
            let img = za::Rgb8In {
                pixels: &[],
                width: w,
                height: h,
                stride: 0,
            };
            assert!(rgb8_to_planar_u8(&img).is_empty());
            assert!(rgb8_to_planar_pm1(&img).is_empty());
            assert!(rgb8_to_planar_255(&img).is_empty());
            assert!(rgb8_to_planar_01(&img).is_empty());
        }
    }

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
