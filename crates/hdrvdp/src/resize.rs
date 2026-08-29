//! MATLAB-compatible `imresize`: bicubic interpolation with antialiasing.
//!
//! HDR-VDP-2's band loop resamples three things — the adapting-luminance map
//! down to each band's size, a neighbouring band's masking signal onto the
//! current band's grid, and the "pixels actually different" mask onto each
//! band. All three go through MATLAB's `imresize` with its defaults, so the
//! defaults are what this module reproduces:
//!
//! * **bicubic** kernel (`a = −0.5`), support 4;
//! * **antialiasing on** — when downsampling, the kernel is stretched by
//!   `1/scale` and rescaled by `scale`, so a ½-scale reduction averages over
//!   8 input samples rather than 4. Skipping this is the classic aliasing bug
//!   in a hand-rolled "bicubic downsample";
//! * **`reflect2` boundary** (`[1:n, n:-1:1]` — the edge sample *is*
//!   duplicated, unlike the pyramid's `reflect1`);
//! * weights renormalised to sum to 1 per output sample, which is what keeps
//!   a constant image constant.
//!
//! Written from `imresize`'s documented behaviour; no MATLAB source is
//! reproduced.

/// The bicubic kernel MATLAB's `imresize` uses (Keys, `a = −0.5`), support 4.
#[must_use]
#[inline]
pub fn cubic(x: f64) -> f64 {
    let a = x.abs();
    let a2 = a * a;
    let a3 = a2 * a;
    if a <= 1.0 {
        1.5 * a3 - 2.5 * a2 + 1.0
    } else if a <= 2.0 {
        -0.5 * a3 + 2.5 * a2 - 4.0 * a + 2.0
    } else {
        0.0
    }
}

/// Per-output-sample resampling weights and source indices for one axis.
struct Contributions {
    /// `out_length` rows of `taps` weights each.
    weights: Vec<f64>,
    /// `out_length` rows of `taps` source indices each.
    indices: Vec<usize>,
    /// Number of taps per output sample.
    taps: usize,
}

fn contributions(in_length: usize, out_length: usize) -> Contributions {
    assert!(in_length > 0 && out_length > 0);
    let scale = out_length as f64 / in_length as f64;
    // Antialiasing: stretch the kernel when shrinking.
    let (kernel_width, stretch) = if scale < 1.0 {
        (4.0 / scale, scale)
    } else {
        (4.0, 1.0)
    };
    let taps = (kernel_width.ceil() as usize) + 2;

    let mut weights = Vec::with_capacity(out_length * taps);
    let mut indices = Vec::with_capacity(out_length * taps);

    // The `reflect2` extension MATLAB clamps into: [0..n-1, n-1..0].
    let aux_len = 2 * in_length;
    let aux = |i: isize| -> usize {
        let m = i.rem_euclid(aux_len as isize) as usize;
        if m < in_length { m } else { aux_len - 1 - m }
    };

    for x in 0..out_length {
        // MATLAB's 1-based `u = x/scale + 0.5·(1 − 1/scale)`; in 0-based
        // coordinates that is the centre of output pixel `x` mapped back.
        let u = (x as f64 + 1.0) / scale + 0.5 * (1.0 - 1.0 / scale);
        let left = (u - kernel_width / 2.0).floor();

        let mut row: Vec<f64> = Vec::with_capacity(taps);
        let mut sum = 0.0;
        for t in 0..taps {
            let idx = left + t as f64; // still 1-based, like MATLAB
            let w = stretch * cubic(stretch * (u - idx));
            row.push(w);
            sum += w;
        }
        for w in &mut row {
            *w /= sum;
        }
        for (t, w) in row.into_iter().enumerate() {
            let idx1 = left as isize + t as isize; // 1-based
            indices.push(aux(idx1 - 1));
            weights.push(w);
        }
    }

    Contributions {
        weights,
        indices,
        taps,
    }
}

/// Resize a row-major `f64` image to `out_width × out_height`, matching
/// MATLAB's `imresize(im, [out_height out_width])` defaults.
///
/// Separable: rows first, then columns.
///
/// # Panics
/// If `image.len() != width · height`, or any dimension is zero.
#[must_use]
pub fn imresize(
    image: &[f64],
    width: usize,
    height: usize,
    out_width: usize,
    out_height: usize,
) -> Vec<f64> {
    assert_eq!(image.len(), width * height, "imresize: image size mismatch");
    assert!(width > 0 && height > 0 && out_width > 0 && out_height > 0);
    if (width, height) == (out_width, out_height) {
        return image.to_vec();
    }

    // Horizontal pass: width → out_width.
    let hc = contributions(width, out_width);
    let mut tmp = vec![0.0; out_width * height];
    for y in 0..height {
        let src = &image[y * width..y * width + width];
        for x in 0..out_width {
            let base = x * hc.taps;
            let mut acc = 0.0;
            for t in 0..hc.taps {
                acc += hc.weights[base + t] * src[hc.indices[base + t]];
            }
            tmp[y * out_width + x] = acc;
        }
    }

    // Vertical pass: height → out_height.
    let vc = contributions(height, out_height);
    let mut out = vec![0.0; out_width * out_height];
    for y in 0..out_height {
        let base = y * vc.taps;
        for x in 0..out_width {
            let mut acc = 0.0;
            for t in 0..vc.taps {
                acc += vc.weights[base + t] * tmp[vc.indices[base + t] * out_width + x];
            }
            out[y * out_width + x] = acc;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic_kernel_is_an_interpolant() {
        // Keys' bicubic: 1 at the origin, 0 at every other integer, and
        // identically 0 beyond ±2.
        assert!((cubic(0.0) - 1.0).abs() < 1e-15);
        for k in [1.0, -1.0, 2.0, -2.0, 3.0, -3.0] {
            assert!(cubic(k).abs() < 1e-15, "cubic({k}) = {}", cubic(k));
        }
        assert_eq!(cubic(2.5), 0.0);
        // Symmetric, and negative in the outer lobes (that is the sharpening
        // that separates bicubic from bilinear).
        assert!((cubic(0.7) - cubic(-0.7)).abs() < 1e-15);
        assert!(cubic(1.5) < 0.0);
    }

    #[test]
    fn a_constant_image_survives_any_resize() {
        // Weight renormalisation is what guarantees this; without it the
        // antialiased path loses energy near the borders.
        for (w, h, ow, oh) in [
            (7usize, 5usize, 3usize, 2usize),
            (4, 4, 16, 16),
            (16, 9, 5, 12),
            (1, 1, 8, 8),
            (33, 17, 33, 17),
        ] {
            let im = vec![4.75; w * h];
            for v in imresize(&im, w, h, ow, oh) {
                assert!((v - 4.75).abs() < 1e-12, "{w}×{h}→{ow}×{oh} gave {v}");
            }
        }
    }

    #[test]
    fn identity_size_is_a_bit_exact_passthrough() {
        let im: Vec<f64> = (0..35).map(|i| (i as f64 * 0.31).sin()).collect();
        assert_eq!(imresize(&im, 7, 5, 7, 5), im);
    }

    #[test]
    fn a_linear_ramp_stays_linear_under_upsampling() {
        // Bicubic reproduces polynomials up to degree 1 exactly (degree 3 in
        // the interior), so an upsampled ramp must stay a ramp away from the
        // borders.
        let (w, h) = (16usize, 1usize);
        let im: Vec<f64> = (0..w).map(|x| 2.0 * x as f64 + 1.0).collect();
        let out = imresize(&im, w, h, 4 * w, h);
        // Interior only: the reflect2 boundary legitimately bends the ramp.
        for (x, got) in out.iter().enumerate().take(4 * w - 12).skip(12) {
            let u = (x as f64 + 0.5) / 4.0 - 0.5;
            let want = 2.0 * u + 1.0;
            assert!((got - want).abs() < 1e-9, "at {x}: {got} vs {want}");
        }
    }

    #[test]
    fn downsampling_antialiases_instead_of_point_sampling() {
        // A Nyquist checkerboard has zero mean. Point sampling (or bicubic
        // without the kernel stretch) at ½ scale would alias it into a solid
        // field; the antialiased kernel must average it away.
        let (w, h) = (64usize, 64usize);
        let im: Vec<f64> = (0..w * h)
            .map(|i| if (i % w + i / w) % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let out = imresize(&im, w, h, w / 2, h / 2);
        let worst = out.iter().fold(0.0f64, |a, b| a.max(b.abs()));
        // The stretched bicubic does not null Nyquist completely — its own
        // leakage leaves ~0.0748 of the ±1 pattern. That is the correct
        // behaviour, not a target of zero.
        assert!(
            worst < 0.08,
            "½-scale reduction of a Nyquist checkerboard left {worst} — aliasing"
        );

        // Non-vacuous: the naive ½ reduction — point-sample every other pixel
        // — turns this checkerboard into a SOLID field of +1, the textbook
        // aliasing failure. The antialiased path is ~13× better.
        let point_sampled: Vec<f64> = (0..(h / 2))
            .flat_map(|y| (0..(w / 2)).map(move |x| (y, x)))
            .map(|(y, x)| im[(2 * y) * w + 2 * x])
            .collect();
        assert!(
            point_sampled.iter().all(|v| (*v - 1.0).abs() < 1e-12),
            "the point-sampled control should be solid +1"
        );
    }

    #[test]
    fn downsampling_preserves_the_mean_of_a_smooth_image() {
        let (w, h) = (48usize, 48usize);
        let pi = core::f64::consts::PI;
        let im: Vec<f64> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f64, (i / w) as f64);
                3.0 + (2.0 * pi * x / 24.0).sin() + 0.5 * (2.0 * pi * y / 16.0).cos()
            })
            .collect();
        let out = imresize(&im, w, h, w / 4, h / 4);
        let m_in = im.iter().sum::<f64>() / im.len() as f64;
        let m_out = out.iter().sum::<f64>() / out.len() as f64;
        assert!(
            (m_in - m_out).abs() < 0.02,
            "mean drifted {m_in} → {m_out} under ¼ reduction"
        );
    }

    #[test]
    fn resize_is_separable_and_order_independent_for_pure_scaling() {
        // Doing width then height must equal height then width (the operation
        // is a tensor product of two 1-D operators). If a future optimisation
        // fuses the passes, this catches an index slip.
        let (w, h) = (17usize, 13usize);
        let im: Vec<f64> = (0..w * h).map(|i| (i as f64 * 0.77).cos()).collect();
        let both = imresize(&im, w, h, 9, 21);
        let stage1 = imresize(&im, w, h, 9, h);
        let stage2 = imresize(&stage1, 9, h, 9, 21);
        for (a, b) in both.iter().zip(&stage2) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }
}
