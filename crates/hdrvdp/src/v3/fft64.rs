//! `f64` Fourier machinery for HDR-VDP-3.
//!
//! The HDR-VDP-3 reference (`jpeg-ai-qaf` `VDP3/`, a NumPy port of the
//! HDR-VDP-3.0.7 MATLAB code) runs the whole pipeline in `f64`, so this
//! module is the `f64` twin of [`crate::fft`]: a radix-2 Cooley–Tukey FFT
//! with a Bluestein chirp-z fallback for non-power-of-two lengths, plus
//! the padded Fourier-domain convolution (`fast_conv_fft`) that both the
//! optical MTF and the large-support Gaussian go through.
//!
//! The padding modes matter and are part of the reference's behaviour:
//! * [`Pad::Symmetric`] — `np.pad(..., 'symmetric')`: reflect the image
//!   about its edges **including** the edge sample (MATLAB `padarray`
//!   'symmetric'); used when `surround = 'none'`.
//! * [`Pad::Replicate`] — repeat the edge row/column outward; used by
//!   `hdrvdp_local_adapt`'s `fast_gauss` call.
//! * [`Pad::Constant`] — fill with a fixed value; used for the spatial
//!   probability summation and explicit surround luminances.
//!
//! Padding is always *post*-padding: the image occupies `[0..h, 0..w]` of
//! the padded lattice, matching `np.pad(X, ((0, H), (0, W)), ...)`.

use core::f64::consts::PI;

/// `f64` complex number for the transforms here.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct C64 {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl C64 {
    /// `re + i·im`.
    #[must_use]
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// `e^{iθ}`.
    #[must_use]
    #[inline]
    pub fn expi(theta: f64) -> Self {
        let (s, c) = theta.sin_cos();
        Self { re: c, im: s }
    }

    /// Complex conjugate.
    #[must_use]
    #[inline]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[must_use]
    #[inline]
    fn mul(self, o: Self) -> Self {
        Self {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
}

/// Forward DFT of `buf`, in place. Any length.
pub fn fft(buf: &mut [C64]) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    if n.is_power_of_two() {
        let stages = radix2_stages(n);
        fft_radix2(buf, &stages);
    } else {
        bluestein(buf);
    }
}

/// Inverse DFT of `buf`, in place, normalised by `1/n`.
pub fn ifft(buf: &mut [C64]) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    for v in buf.iter_mut() {
        *v = v.conj();
    }
    fft(buf);
    let s = 1.0 / n as f64;
    for v in buf.iter_mut() {
        *v = v.conj();
        v.re *= s;
        v.im *= s;
    }
}

fn radix2_stages(n: usize) -> Vec<Vec<C64>> {
    debug_assert!(n.is_power_of_two() && n >= 2);
    let mut stages = Vec::with_capacity(n.trailing_zeros() as usize);
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * PI / len as f64;
        stages.push((0..len / 2).map(|k| C64::expi(ang * k as f64)).collect());
        len <<= 1;
    }
    stages
}

fn fft_radix2(buf: &mut [C64], stages: &[Vec<C64>]) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two());
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = (i as u32).reverse_bits() >> (32 - bits);
        if j as usize > i {
            buf.swap(i, j as usize);
        }
    }
    for tw in stages {
        let half = tw.len();
        let len = half * 2;
        for start in (0..n).step_by(len) {
            for (k, &w) in tw.iter().enumerate() {
                let u = buf[start + k];
                let v = buf[start + k + half].mul(w);
                buf[start + k] = C64::new(u.re + v.re, u.im + v.im);
                buf[start + k + half] = C64::new(u.re - v.re, u.im - v.im);
            }
        }
    }
}

/// Bluestein chirp-z transform for arbitrary `n`.
fn bluestein(buf: &mut [C64]) {
    let n = buf.len();
    let m = (2 * n - 1).next_power_of_two();
    let chirp: Vec<C64> = (0..n)
        .map(|k| {
            let kk = (k as u128 * k as u128 % (2 * n as u128)) as f64;
            C64::expi(-PI * kk / n as f64)
        })
        .collect();
    let stages = radix2_stages(m);
    let mut bf = vec![C64::default(); m];
    for (k, &c) in chirp.iter().enumerate() {
        bf[k] = c.conj();
        if k > 0 {
            bf[m - k] = c.conj();
        }
    }
    fft_radix2(&mut bf, &stages);
    let mut a = vec![C64::default(); m];
    for (k, b) in buf.iter().enumerate() {
        a[k] = b.mul(chirp[k]);
    }
    fft_radix2(&mut a, &stages);
    for (x, &y) in a.iter_mut().zip(&bf) {
        *x = x.mul(y);
    }
    for v in a.iter_mut() {
        *v = v.conj();
    }
    fft_radix2(&mut a, &stages);
    let s = 1.0 / m as f64;
    for v in a.iter_mut() {
        *v = v.conj();
        v.re *= s;
        v.im *= s;
    }
    for (k, out) in buf.iter_mut().enumerate() {
        *out = a[k].mul(chirp[k]);
    }
}

/// Forward 2-D DFT of a row-major `height × width` buffer, in place.
pub fn fft2(buf: &mut [C64], width: usize, height: usize) {
    assert_eq!(buf.len(), width * height, "fft2: buffer size mismatch");
    for row in buf.chunks_exact_mut(width) {
        fft(row);
    }
    let mut col = vec![C64::default(); height];
    for x in 0..width {
        for (y, c) in col.iter_mut().enumerate() {
            *c = buf[y * width + x];
        }
        fft(&mut col);
        for (y, c) in col.iter().enumerate() {
            buf[y * width + x] = *c;
        }
    }
}

/// Inverse 2-D DFT of a row-major `height × width` buffer, in place,
/// normalised by `1/(width·height)`.
pub fn ifft2(buf: &mut [C64], width: usize, height: usize) {
    assert_eq!(buf.len(), width * height, "ifft2: buffer size mismatch");
    let mut col = vec![C64::default(); height];
    for x in 0..width {
        for (y, c) in col.iter_mut().enumerate() {
            *c = buf[y * width + x];
        }
        ifft(&mut col);
        for (y, c) in col.iter().enumerate() {
            buf[y * width + x] = *c;
        }
    }
    for row in buf.chunks_exact_mut(width) {
        ifft(row);
    }
}

/// `np.fft.fft2` of a real row-major `width × height` buffer.
#[must_use]
pub fn fft2_of(x: &[f64], width: usize, height: usize) -> Vec<C64> {
    assert_eq!(x.len(), width * height, "fft2_of: size");
    let mut buf: Vec<C64> = x.iter().map(|&v| C64::new(v, 0.0)).collect();
    fft2(&mut buf, width, height);
    buf
}

/// How the image is extended onto the padded convolution lattice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Pad {
    /// Reflect about the edges, edge sample duplicated (`np.pad`
    /// 'symmetric', MATLAB `padarray` 'symmetric'). The `surround = 'none'`
    /// behaviour.
    Symmetric,
    /// Repeat the edge row/column outward (`fast_conv_fft`'s 'replicate').
    Replicate,
    /// Fill the pad region with a constant value.
    Constant(f64),
}

/// Extend `x` (row-major `width × height`) to `pad_w × pad_h`, where the
/// image occupies the top-left corner and the rest follows `pad`.
///
/// Symmetric reflection of `padded[py]` for `py >= height` maps to image row
/// `2·height − 1 − py`; replicate clamps to `height − 1`. Both are applied
/// componentwise on each axis, which reproduces the corner block too.
#[must_use]
pub fn pad_image(
    x: &[f64],
    width: usize,
    height: usize,
    pad_w: usize,
    pad_h: usize,
    pad: Pad,
) -> Vec<f64> {
    assert_eq!(x.len(), width * height, "pad_image: size mismatch");
    assert!(pad_w >= width && pad_h >= height);
    let map = |i: usize, n: usize| -> usize {
        if i < n {
            i
        } else {
            match pad {
                Pad::Symmetric => {
                    // index n+k → n−1−k (edge duplicated).
                    (2 * n - 1).saturating_sub(i).min(n - 1)
                }
                Pad::Replicate => n - 1,
                Pad::Constant(_) => usize::MAX,
            }
        }
    };
    let mut out = vec![0.0; pad_w * pad_h];
    for py in 0..pad_h {
        let sy = map(py, height);
        for px in 0..pad_w {
            let sx = map(px, width);
            out[py * pad_w + px] = match (sy, sx) {
                (usize::MAX, _) | (_, usize::MAX) => match pad {
                    Pad::Constant(v) => v,
                    _ => unreachable!(),
                },
                _ => x[sy * width + sx],
            };
        }
    }
    out
}

/// Convolve `x` (row-major `height × width`) with the zero-phase filter
/// `filter` given as its real Fourier-domain response on a
/// `pad_h × pad_w` lattice; `x` is extended per `pad`, transformed,
/// multiplied, inverse-transformed and cropped back to `height × width`.
///
/// Equivalent to upstream `fast_conv_fft`.
#[must_use]
pub fn conv_fft_pad(
    x: &[f64],
    width: usize,
    height: usize,
    filter: &[f64],
    pad_w: usize,
    pad_h: usize,
    pad: Pad,
) -> Vec<f64> {
    assert_eq!(filter.len(), pad_w * pad_h, "conv_fft_pad: filter size");
    let padded = pad_image(x, width, height, pad_w, pad_h, pad);
    let mut buf: Vec<C64> = padded.iter().map(|&v| C64::new(v, 0.0)).collect();
    fft2(&mut buf, pad_w, pad_h);
    for (b, &f) in buf.iter_mut().zip(filter) {
        *b = C64::new(b.re * f, b.im * f);
    }
    ifft2(&mut buf, pad_w, pad_h);
    let mut out = Vec::with_capacity(width * height);
    for y in 0..height {
        for x0 in 0..width {
            out.push(buf[y * pad_w + x0].re);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_ifft_roundtrip() {
        for n in [1usize, 2, 4, 8, 16, 96, 100, 97, 180] {
            let orig: Vec<C64> = (0..n)
                .map(|k| C64::new((k as f64 * 0.37).sin(), (k as f64 * 1.1).cos()))
                .collect();
            let mut b = orig.clone();
            fft(&mut b);
            ifft(&mut b);
            for (a, g) in b.iter().zip(&orig) {
                assert!(
                    (a.re - g.re).abs() < 1e-12 && (a.im - g.im).abs() < 1e-12,
                    "n={n}: {a:?} vs {g:?}"
                );
            }
        }
    }

    #[test]
    fn conv_fft_pad_matches_direct_convolution() {
        // Zero-phase tent filter made in the Fourier domain: build the
        // spatial kernel, DFT it, then run conv_fft_pad and compare against
        // direct correlation with replicate borders on a small case.
        let (w, h) = (8usize, 6usize);
        let x: Vec<f64> = (0..w * h).map(|i| (i * 37 % 19) as f64 - 9.0).collect();
        // Spatial kernel: 5×5 separable tent, sum 1.
        let t = [1.0, 2.0, 3.0, 2.0, 1.0];
        let mut k = [0.0; 25];
        for y in 0..5 {
            for x0 in 0..5 {
                k[y * 5 + x0] = t[y] * t[x0] / 81.0;
            }
        }
        // Fourier response on the 2× padded lattice: DFT of the kernel
        // embedded at the right offset (kernel centred at origin → place taps
        // wrap-around, i.e. kernel index (ky,kx) maps to bin offset (−2,−2)).
        let (pw, ph) = (2 * w, 2 * h);
        let mut kbuf = vec![C64::default(); pw * ph];
        for ky in 0..5usize {
            for kx in 0..5usize {
                let by = (ky + ph - 2) % ph;
                let bx = (kx + pw - 2) % pw;
                kbuf[by * pw + bx] = C64::new(k[ky * 5 + kx], 0.0);
            }
        }
        fft2(&mut kbuf, pw, ph);
        let filt: Vec<f64> = kbuf.iter().map(|c| c.re).collect();
        let got = conv_fft_pad(&x, w, h, &filt, pw, ph, Pad::Replicate);
        // Direct model: the padded lattice is circular; a tap at centred
        // offset (ky-2, kx-2) reads padded[(y+ky-2) mod ph, (x+kx-2) mod pw],
        // where padded positions ≥ h/≥ w carry the replicate extension.
        let sample = |py: usize, px: usize| -> f64 {
            let sy = py.min(h - 1);
            let sx = px.min(w - 1);
            x[sy * w + sx]
        };
        for y in 0..h {
            for x0 in 0..w {
                let mut acc = 0.0;
                for ky in 0..5usize {
                    for kx in 0..5usize {
                        let py = (y + ky + ph - 2) % ph;
                        let px = (x0 + kx + pw - 2) % pw;
                        acc += k[ky * 5 + kx] * sample(py, px);
                    }
                }
                assert!(
                    (got[y * w + x0] - acc).abs() < 1e-9,
                    "conv mismatch at {y},{x0}: {} vs {acc}",
                    got[y * w + x0]
                );
            }
        }
    }
}
