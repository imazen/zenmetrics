//! `fast_gauss` — the large-support Gaussian used for local adaptation
//! (σ ≈ 0.165·ppd) and spatial probability summation (σ ≈ 0.315·ppd).
//!
//! Upstream picks between two paths on `sigma ≥ 4.3`:
//! * large σ — the Gaussian's *transfer function* on a 2× lattice, applied
//!   via `fast_conv_fft` (pad = `replicate` for local adaptation, `0` for
//!   the probability summation);
//! * small σ — a `matlab_style_gauss2D` kernel through
//!   `ndimage.convolve(mode='constant', cval=pad_value)`. Upstream passes
//!   the string `'replicate'` as `cval` in that branch, which scipy
//!   rejects — so the reference effectively requires σ ≥ 4.3 on the
//!   local-adaptation path (ppd ≳ 26). We implement the small-σ branch
//!   faithfully for the [`Pad`] modes we do support and note the upstream
//!   limitation in `DIVERGENCES.md`.
//!
//! `do_norm` matches upstream's (reversed-looking) convention: on the FFT
//! path the un-normalised transfer `K` already preserves DC, so `do_norm`
//! needs nothing; `!do_norm` divides `K` by its mean. On the direct path
//! `do_norm` normalises by `Σ h`, `!do_norm` by the centre tap.

use super::fft64::{self, Pad};

/// `matlab_style_gauss2D` — `fspecial('gaussian', k, sigma)`: an odd square
/// kernel, taps below `eps·max` zeroed, normalised to sum 1 (before the
/// `do_norm` adjustment the caller applies).
fn matlab_gauss_2d(ksize: usize, sigma: f64) -> Vec<f64> {
    debug_assert!(ksize % 2 == 1);
    let c = (ksize as f64 - 1.0) / 2.0;
    let mut h = Vec::with_capacity(ksize * ksize);
    for y in 0..ksize {
        for x in 0..ksize {
            let dx = x as f64 - c;
            let dy = y as f64 - c;
            h.push((-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp());
        }
    }
    let max = h.iter().copied().fold(0.0f64, f64::max);
    for v in h.iter_mut() {
        if *v < f64::EPSILON * max {
            *v = 0.0;
        }
    }
    let sum: f64 = h.iter().sum();
    if sum != 0.0 {
        for v in h.iter_mut() {
            *v /= sum;
        }
    }
    h
}

/// Sample `x` (w×h) with the boundary rule `pad` at `(y, x)` which may be
/// out of range. `Pad::Symmetric` mirrors with the edge sample repeated
/// (numpy `pad` 'symmetric' semantics, applied symmetrically on both sides
/// — this is the *convolution* border, unlike [`fft64::pad_image`] which
/// only extends right/down).
fn sample_pad(x: &[f64], w: usize, h: usize, y: isize, xx: isize, pad: Pad) -> f64 {
    let n = w.max(h);
    let _ = n;
    let map = |i: isize, n: usize| -> Option<usize> {
        if (0..n as isize).contains(&i) {
            return Some(i as usize);
        }
        match pad {
            Pad::Constant(_) => None,
            Pad::Replicate => Some(i.clamp(0, n as isize - 1) as usize),
            Pad::Symmetric => {
                // Reflect including the edge: −1→0, −2→1, n→n−1, n+1→n−2.
                let mut j = i;
                // Handle reflections longer than n by folding repeatedly —
                // kernels here are far smaller than the image, so one fold
                // suffices, but loop for completeness.
                while j < 0 || j >= n as isize {
                    j = if j < 0 {
                        -1 - j
                    } else {
                        2 * n as isize - 1 - j
                    };
                }
                Some(j as usize)
            }
        }
    };
    match (map(y, h), map(xx, w)) {
        (Some(sy), Some(sx)) => x[sy * w + sx],
        _ => match pad {
            Pad::Constant(v) => v,
            _ => 0.0,
        },
    }
}

/// `fast_gauss(X, sigma, do_norm, pad_value)`.
#[must_use]
pub fn fast_gauss(x: &[f64], w: usize, h: usize, sigma: f64, do_norm: bool, pad: Pad) -> Vec<f64> {
    assert_eq!(x.len(), w * h, "fast_gauss: size");
    if sigma >= 4.3 {
        // Fourier path: K = exp(−0.5·(KX²+KY²)·σ²) on the 2× lattice.
        let (pw, ph) = (2 * w, 2 * h);
        let mut k = vec![0.0; pw * ph];
        for y in 0..ph {
            let ky = ((0.5 + y as f64 / ph as f64) % 1.0 - 0.5) * core::f64::consts::TAU;
            for xx in 0..pw {
                let kx = ((0.5 + xx as f64 / pw as f64) % 1.0 - 0.5) * core::f64::consts::TAU;
                k[y * pw + xx] = (-0.5 * (kx * kx + ky * ky) * sigma * sigma).exp();
            }
        }
        if !do_norm {
            let mean = k.iter().sum::<f64>() / k.len() as f64;
            for v in k.iter_mut() {
                *v /= mean;
            }
        }
        fft64::conv_fft_pad(x, w, h, &k, pw, ph, pad)
    } else {
        // Direct path: matlab-style kernel, conv 'same'.
        let ksize = (sigma * 6.0).round() as usize;
        let ksize = ksize + 1 - ksize % 2;
        let ksize = ksize.max(1);
        let mut kern = matlab_gauss_2d(ksize, sigma);
        if !do_norm {
            let centre = kern[(ksize / 2) * ksize + ksize / 2];
            for v in kern.iter_mut() {
                *v /= centre;
            }
        }
        let c = (ksize / 2) as isize;
        let mut out = vec![0.0; w * h];
        for y in 0..h {
            for xx in 0..w {
                let mut acc = 0.0;
                for ky in 0..ksize {
                    for kx in 0..ksize {
                        acc += kern[ky * ksize + kx]
                            * sample_pad(
                                x,
                                w,
                                h,
                                y as isize + ky as isize - c,
                                xx as isize + kx as isize - c,
                                pad,
                            );
                    }
                }
                out[y * w + xx] = acc;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_gauss_preserves_constant() {
        let x = vec![3.25; 24 * 24];
        for &sigma in &[1.0, 4.3, 8.0] {
            let y = fast_gauss(&x, 24, 24, sigma, true, Pad::Replicate);
            for &v in &y {
                assert!((v - 3.25).abs() < 1e-9, "σ={sigma}: {v}");
            }
        }
    }

    #[test]
    fn fast_gauss_direct_kernel_odd() {
        // σ small enough for the direct path, impulse at centre → kernel.
        let (w, h) = (16usize, 16usize);
        let mut x = vec![0.0; w * h];
        x[8 * w + 8] = 1.0;
        let y = fast_gauss(&x, w, h, 1.0, true, Pad::Constant(0.0));
        let k = matlab_gauss_2d(7, 1.0); // ksize = round(6)+1-6%2 = 7
        // centre pixel of output should be kernel's centre tap.
        assert!((y[8 * w + 8] - k[3 * 7 + 3]).abs() < 1e-12);
    }
}
