//! Scalar f64 adaptation of the in-tree fsim/src/fft.rs radix-2/Bluestein
//! transforms. Kept private to the CW-SSIM implementation; no dependency added.
use std::f64::consts::PI;
/// Minimal complex number — enough for the transforms here, and it keeps the
/// crate dependency-free.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(super) struct Complex {
    /// Real part.
    pub(super) re: f64,
    /// Imaginary part.
    pub(super) im: f64,
}

impl Complex {
    /// `re + i·im`.
    #[must_use]
    #[inline]
    pub(super) const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Unit complex exponential.
    #[must_use]
    #[inline]
    pub(super) fn expi(theta: f64) -> Self {
        let (s, c) = theta.sin_cos();
        Self { re: c, im: s }
    }

    /// Complex conjugate.
    #[must_use]
    #[inline]
    pub(super) fn conj(self) -> Self {
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

    #[must_use]
    #[inline]
    fn add(self, o: Self) -> Self {
        Self {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }

    #[must_use]
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self {
            re: self.re - o.re,
            im: self.im - o.im,
        }
    }

    #[must_use]
    #[inline]
    fn scale(self, s: f64) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }
}

/// Per-length radix-2 or Bluestein plan.
pub(super) struct Plan {
    n: usize,
    kind: PlanKind,
}

enum PlanKind {
    /// `n <= 1`: nothing to do.
    Trivial,
    /// Power-of-two Cooley–Tukey; `stages[s]` holds the `2^s` twiddles of the
    /// stage with butterfly span `2^(s+1)`.
    Radix2 { stages: Vec<Vec<Complex>> },
    /// Bluestein chirp-z: `chirp[k] = e^{-iπk²/n}` and `bf` = the length-`m`
    /// FFT of the chirp filter, with the radix-2 plan for `m`.
    Bluestein {
        chirp: Vec<Complex>,
        bf: Vec<Complex>,
        m: usize,
        mplan: Vec<Vec<Complex>>,
    },
}

/// Per-stage twiddles for a power-of-two length.
fn radix2_stages(n: usize) -> Vec<Vec<Complex>> {
    debug_assert!(n.is_power_of_two() && n >= 2);
    let mut stages = Vec::with_capacity(n.trailing_zeros() as usize);
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * PI / len as f64;
        let half = len / 2;
        stages.push((0..half).map(|k| Complex::expi(ang * k as f64)).collect());
        len <<= 1;
    }
    stages
}

impl Plan {
    pub(super) fn new(n: usize) -> Self {
        let kind = if n <= 1 {
            PlanKind::Trivial
        } else if n.is_power_of_two() {
            PlanKind::Radix2 {
                stages: radix2_stages(n),
            }
        } else {
            let m = (2 * n - 1).next_power_of_two();
            let chirp: Vec<Complex> = (0..n)
                .map(|k| {
                    let kk = (k as u128 * k as u128 % (2 * n as u128)) as f64;
                    Complex::expi(-PI * kk / n as f64)
                })
                .collect();
            let mplan = radix2_stages(m);
            let mut bf = vec![Complex::default(); m];
            for k in 0..n {
                bf[k] = chirp[k].conj();
                if k > 0 {
                    bf[m - k] = chirp[k].conj();
                }
            }
            fft_radix2_planned(&mut bf, &mplan);
            PlanKind::Bluestein {
                chirp,
                bf,
                m,
                mplan,
            }
        };
        Self { n, kind }
    }

    fn run(&self, buf: &mut [Complex]) {
        debug_assert_eq!(buf.len(), self.n);
        match &self.kind {
            PlanKind::Trivial => {}
            PlanKind::Radix2 { stages } => fft_radix2_planned(buf, stages),
            PlanKind::Bluestein {
                chirp,
                bf,
                m,
                mplan,
            } => {
                let n = self.n;
                let mut a = vec![Complex::default(); *m];
                for k in 0..n {
                    a[k] = buf[k].mul(chirp[k]);
                }
                fft_radix2_planned(&mut a, mplan);
                for (x, y) in a.iter_mut().zip(bf) {
                    *x = x.mul(*y);
                }
                for v in a.iter_mut() {
                    *v = v.conj();
                }
                fft_radix2_planned(&mut a, mplan);
                let s = 1.0 / *m as f64;
                for v in a.iter_mut() {
                    *v = v.conj().scale(s);
                }
                for (k, out) in buf.iter_mut().enumerate() {
                    *out = a[k].mul(chirp[k]);
                }
            }
        }
    }
}
/// Iterative radix-2 with precomputed per-stage twiddles.
fn fft_radix2_planned(buf: &mut [Complex], stages: &[Vec<Complex>]) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two());

    // Bit-reversal permutation.
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = (i as u32).reverse_bits() >> (32 - bits);
        let j = j as usize;
        if j > i {
            buf.swap(i, j);
        }
    }

    for tw in stages {
        let half = tw.len();
        let len = half * 2;
        for start in (0..n).step_by(len) {
            for (k, &w) in tw.iter().enumerate() {
                let u = buf[start + k];
                let v = buf[start + k + half].mul(w);
                buf[start + k] = u.add(v);
                buf[start + k + half] = u.sub(v);
            }
        }
    }
}

/// Row/column transform, with a plan shared across all rows or columns.
pub(super) fn transform(a: &mut [Complex], w: usize, h: usize, inverse: bool) {
    if inverse {
        for x in a.iter_mut() {
            x.im = -x.im;
        }
    }
    let wp = Plan::new(w);
    let hp = Plan::new(h);
    for row in a.chunks_exact_mut(w) {
        wp.run(row);
    }
    let mut col = vec![Complex::default(); h];
    for x in 0..w {
        for y in 0..h {
            col[y] = a[y * w + x];
        }
        hp.run(&mut col);
        for y in 0..h {
            a[y * w + x] = col[y];
        }
    }
    if inverse {
        let scale = 1.0 / (w * h) as f64;
        for x in a {
            x.re *= scale;
            x.im *= -scale;
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn non_power_of_two_matches_direct_dft() {
        for n in [1, 2, 7, 16, 19] {
            let source: Vec<_> = (0..n)
                .map(|i| Complex::new((i * 7 % 13) as f64, (i * 3 % 11) as f64))
                .collect();
            let mut actual = source.clone();
            Plan::new(n).run(&mut actual);
            for (k, got) in actual.iter().enumerate() {
                let expected = source
                    .iter()
                    .enumerate()
                    .fold(Complex::default(), |s, (j, &v)| {
                        s.add(v.mul(Complex::expi(-2.0 * PI * (j * k) as f64 / n as f64)))
                    });
                assert!(
                    (got.re - expected.re).abs() < 1e-10 && (got.im - expected.im).abs() < 1e-10
                );
            }
        }
    }
    #[test]
    fn rectangular_round_trip() {
        let src: Vec<_> = (0..13 * 18)
            .map(|i| Complex::new((i % 29) as f64, 0.0))
            .collect();
        let mut a = src.clone();
        transform(&mut a, 13, 18, false);
        transform(&mut a, 13, 18, true);
        for (a, b) in a.iter().zip(src) {
            assert!((a.re - b.re).abs() < 1e-10 && a.im.abs() < 1e-10);
        }
    }
}
