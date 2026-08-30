//! Fourier-domain filtering: a self-contained complex FFT, the radial
//! cycles-per-degree grid, and zero-phase convolution with post-padding.
//!
//! ## Provenance — written independently, on purpose
//!
//! Upstream's `create_cycdeg_image.m` and `fast_conv_fft.m` are marked
//! *"experimental code for internal use. Do not redistribute."* — they carry no
//! permission grant, unlike the rest of HDR-VDP-2 (see
//! `THIRD-PARTY-NOTICES.md`). **No line of either file is reproduced here.**
//! Both do a standard thing, and this module implements those standard things
//! from their one-line descriptions:
//!
//! * a matrix holding, for every DFT bin, the radial spatial frequency in
//!   cycles per degree;
//! * convolution with a large-support kernel done in the Fourier domain, where
//!   the signal is extended to the filter's size with a constant pad value and
//!   the result cropped back.
//!
//! ### One knowable difference from upstream
//!
//! For an **odd**-length axis, upstream's frequency axis runs the positive half
//! up to the Nyquist frequency inclusive, which spaces the two halves
//! differently from the DFT's own bin frequencies. This module always uses the
//! true DFT bin frequencies (`k/N · ppd` folded to `[-ppd/2, ppd/2)`), which
//! agrees with upstream exactly for **even** lengths. Every axis the metric
//! actually transforms is padded to twice an integer size and is therefore
//! even, so the difference is unreachable in the pipeline — it is documented
//! only so nobody later "fixes" this to match an odd-size reference dump.

use core::f64::consts::PI;

/// Minimal complex number — enough for the transforms here, and it keeps the
/// crate dependency-free.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex {
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

/// Forward DFT of `buf`, in place. Any length; `O(n log n)`.
pub fn fft(buf: &mut [Complex]) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    Plan::new(n).run(buf);
}

/// A DFT plan for one length: the per-stage twiddle factors (and, for
/// non-power-of-two lengths, the Bluestein chirp and pre-transformed filter),
/// computed once and reused across every transform of that length.
///
/// Every value is produced by the *same expressions* the direct implementation
/// evaluated inline — `expi(-2π/len · k)` per stage, the `k² mod 2n` chirp —
/// so a planned transform is bit-identical to the unplanned one; the plan only
/// memoises. `fft2` reuses one plan across all rows and one across all
/// columns, and [`conv_fft_real`] across its forward and inverse passes, which
/// is where the win concentrates: the direct form recomputed `sin_cos` for
/// every butterfly of every row.
struct Plan {
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

/// Per-stage twiddles for a power-of-two length — exactly the values the
/// direct loop computed as `expi(ang · k)` per block, hoisted.
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
    fn new(n: usize) -> Self {
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

/// Iterative radix-2 with precomputed per-stage twiddles. The butterfly body,
/// its iteration order, and the twiddle values are those of the direct
/// implementation; only the `expi` calls moved out of the loop.
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

/// Inverse DFT of `buf`, in place, normalised by `1/n`.
pub fn ifft(buf: &mut [Complex]) {
    let n = buf.len();
    if n == 0 {
        return;
    }
    for v in buf.iter_mut() {
        *v = v.conj();
    }
    fft(buf);
    let s = 1.0 / n as f64;
    for v in buf.iter_mut() {
        *v = v.conj().scale(s);
    }
}

/// Forward 2D DFT of a row-major `height × width` buffer, in place.
pub fn fft2(buf: &mut [Complex], width: usize, height: usize) {
    let (wplan, hplan) = (Plan::new(width), Plan::new(height));
    fft2_planned(buf, width, height, &wplan, &hplan);
}

/// [`fft2`] with the two 1-D plans supplied, so callers doing several
/// same-size transforms (or a forward + inverse pair) build them once.
fn fft2_planned(buf: &mut [Complex], width: usize, height: usize, wplan: &Plan, hplan: &Plan) {
    debug_assert_eq!(buf.len(), width * height);
    for row in buf.chunks_exact_mut(width) {
        wplan.run(row);
    }
    fft2_cols_planned(buf, width, height, hplan);
}

/// The column half of [`fft2_planned`], split out so [`conv_fft_real`] can run
/// a smarter row pass first.
///
/// Columns go in blocks of 8: one Complex is 16 bytes, so 8 adjacent columns
/// span a whole cache line per touched row instead of using 1/8th of it. Each
/// column still gets the identical 1-D transform — columns are independent, so
/// the grouping cannot change a single bit.
fn fft2_cols_planned(buf: &mut [Complex], width: usize, height: usize, hplan: &Plan) {
    const CB: usize = 8;
    let mut cols = vec![Complex::default(); height * CB];
    let mut x = 0usize;
    while x < width {
        let nb = CB.min(width - x);
        for y in 0..height {
            let src = &buf[y * width + x..y * width + x + nb];
            for (j, &v) in src.iter().enumerate() {
                cols[j * height + y] = v;
            }
        }
        for c in cols.chunks_exact_mut(height).take(nb) {
            hplan.run(c);
        }
        for y in 0..height {
            let dst = &mut buf[y * width + x..y * width + x + nb];
            for (j, d) in dst.iter_mut().enumerate() {
                *d = cols[j * height + y];
            }
        }
        x += nb;
    }
}

/// Inverse 2D DFT (normalised by `1/(width·height)`), in place.
pub fn ifft2(buf: &mut [Complex], width: usize, height: usize) {
    let (wplan, hplan) = (Plan::new(width), Plan::new(height));
    ifft2_planned(buf, width, height, &wplan, &hplan);
}

/// [`ifft2`] with the plans supplied — same conjugate–forward–conjugate
/// construction, same normalisation.
fn ifft2_planned(buf: &mut [Complex], width: usize, height: usize, wplan: &Plan, hplan: &Plan) {
    debug_assert_eq!(buf.len(), width * height);
    for v in buf.iter_mut() {
        *v = v.conj();
    }
    fft2_planned(buf, width, height, wplan, hplan);
    let s = 1.0 / (width * height) as f64;
    for v in buf.iter_mut() {
        *v = v.conj().scale(s);
    }
}

/// Radial spatial frequency, in cycles per degree, for every bin of a
/// `height × width` DFT of an image sampled at `pix_per_deg` pixels/degree.
///
/// Row-major, in DFT bin order (bin 0 is DC, the second half holds the negative
/// frequencies). The value at bin `(y, x)` is `sqrt(fx² + fy²)` where
/// `fx = fftfreq(x, width) · pix_per_deg`.
///
/// See the module header for the one documented divergence from upstream on
/// odd-length axes (unreachable in the pipeline).
#[must_use]
pub fn cycles_per_degree_grid(width: usize, height: usize, pix_per_deg: f64) -> Vec<f64> {
    let fx = dft_bin_frequencies(width, pix_per_deg);
    let fy = dft_bin_frequencies(height, pix_per_deg);
    let mut out = Vec::with_capacity(width * height);
    for &v in &fy {
        for &u in &fx {
            out.push((u * u + v * v).sqrt());
        }
    }
    out
}

/// Signed DFT bin frequencies in cycles/degree for one axis, in bin order.
fn dft_bin_frequencies(n: usize, pix_per_deg: f64) -> Vec<f64> {
    (0..n)
        .map(|k| {
            let k = if k * 2 <= n {
                k as f64
            } else {
                k as f64 - n as f64
            };
            k / n as f64 * pix_per_deg
        })
        .collect()
}

/// Convolve `x` (a `height × width` real image) with a zero-phase filter given
/// by its real Fourier-domain response `filter` on a `pad_h × pad_w` lattice.
///
/// `x` is extended to `pad_h × pad_w` with `pad_value` (appended after the last
/// row/column, so pixel `(0, 0)` keeps its position), transformed, multiplied,
/// inverse-transformed, and cropped back to `height × width`.
///
/// Padding to twice the image size is what keeps the circular wrap-around from
/// folding real image content back onto the opposite edge — the wrapped
/// contribution comes from the constant pad region instead.
///
/// # Panics
/// If `x.len() != width·height`, `filter.len() != pad_w·pad_h`, or the padded
/// size is smaller than the image.
#[must_use]
pub fn conv_fft_real(
    x: &[f64],
    width: usize,
    height: usize,
    filter: &[f64],
    pad_w: usize,
    pad_h: usize,
    pad_value: f64,
) -> Vec<f64> {
    assert_eq!(
        x.len(),
        width * height,
        "conv_fft_real: image size mismatch"
    );
    assert_eq!(
        filter.len(),
        pad_w * pad_h,
        "conv_fft_real: filter size mismatch"
    );
    assert!(
        pad_w >= width && pad_h >= height,
        "conv_fft_real: padded size must not be smaller than the image"
    );

    let mut buf = vec![Complex::new(pad_value, 0.0); pad_w * pad_h];
    for y in 0..height {
        for xi in 0..width {
            buf[y * pad_w + xi] = Complex::new(x[y * width + xi], 0.0);
        }
    }

    // One plan pair serves the forward and the inverse transform.
    let (wplan, hplan) = (Plan::new(pad_w), Plan::new(pad_h));

    // Forward row pass, exploiting the padding structure: every row below the
    // image (`height..pad_h`) is the same all-`pad_value` row, and a DFT is a
    // deterministic function of its input — so those rows' transforms are
    // bit-identical. Transform one and copy it into the rest, then run the
    // column pass as usual.
    for row in buf[..height * pad_w].chunks_exact_mut(pad_w) {
        wplan.run(row);
    }
    if height < pad_h {
        let (first, rest) = buf[height * pad_w..].split_at_mut(pad_w);
        wplan.run(first);
        for row in rest.chunks_exact_mut(pad_w) {
            row.copy_from_slice(first);
        }
    }
    fft2_cols_planned(&mut buf, pad_w, pad_h, &hplan);

    for (b, f) in buf.iter_mut().zip(filter) {
        *b = b.scale(*f);
    }
    ifft2_planned(&mut buf, pad_w, pad_h, &wplan, &hplan);

    let mut out = Vec::with_capacity(width * height);
    for y in 0..height {
        for xi in 0..width {
            out.push(buf[y * pad_w + xi].re);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dft(x: &[Complex]) -> Vec<Complex> {
        let n = x.len();
        (0..n)
            .map(|k| {
                let mut acc = Complex::default();
                for (j, v) in x.iter().enumerate() {
                    let ang = -2.0 * PI * (k * j) as f64 / n as f64;
                    acc = acc.add(v.mul(Complex::expi(ang)));
                }
                acc
            })
            .collect()
    }

    fn seeded(n: usize) -> Vec<Complex> {
        // Deterministic pseudo-random-ish input; no dependency needed.
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        (0..n)
            .map(|_| {
                let mut next = || {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
                };
                Complex::new(next(), next())
            })
            .collect()
    }

    #[test]
    fn fft_matches_the_naive_dft_for_power_of_two() {
        for n in [1usize, 2, 4, 8, 16, 64] {
            let x = seeded(n);
            let want = naive_dft(&x);
            let mut got = x.clone();
            fft(&mut got);
            for (a, b) in got.iter().zip(&want) {
                assert!(
                    (a.re - b.re).abs() < 1e-10 && (a.im - b.im).abs() < 1e-10,
                    "n={n}: {a:?} != {b:?}"
                );
            }
        }
    }

    #[test]
    fn fft_matches_the_naive_dft_for_arbitrary_lengths() {
        // Bluestein path: primes, odd composites, and a length just over a
        // power of two.
        for n in [3usize, 5, 6, 7, 9, 11, 12, 15, 17, 33, 100] {
            let x = seeded(n);
            let want = naive_dft(&x);
            let mut got = x.clone();
            fft(&mut got);
            for (a, b) in got.iter().zip(&want) {
                assert!(
                    (a.re - b.re).abs() < 1e-9 && (a.im - b.im).abs() < 1e-9,
                    "n={n}: {a:?} != {b:?}"
                );
            }
        }
    }

    #[test]
    fn fft_ifft_round_trips() {
        for n in [1usize, 2, 7, 16, 100, 256] {
            let x = seeded(n);
            let mut y = x.clone();
            fft(&mut y);
            ifft(&mut y);
            for (a, b) in y.iter().zip(&x) {
                assert!(
                    (a.re - b.re).abs() < 1e-10 && (a.im - b.im).abs() < 1e-10,
                    "n={n}"
                );
            }
        }
    }

    #[test]
    fn fft2_ifft2_round_trips() {
        for (w, h) in [(4usize, 4usize), (8, 5), (13, 7), (32, 16)] {
            let x = seeded(w * h);
            let mut y = x.clone();
            fft2(&mut y, w, h);
            ifft2(&mut y, w, h);
            for (a, b) in y.iter().zip(&x) {
                assert!((a.re - b.re).abs() < 1e-10 && (a.im - b.im).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn cycdeg_grid_has_dc_at_the_origin_and_nyquist_at_the_fold() {
        let ppd = 30.0;
        let (w, h) = (8usize, 8usize);
        let g = cycles_per_degree_grid(w, h, ppd);
        assert_eq!(g[0], 0.0, "bin (0,0) must be DC");
        // Along the first row, |f| rises to Nyquist at k = w/2 and falls back.
        let nyq = 0.5 * ppd;
        assert!((g[w / 2] - nyq).abs() < 1e-12, "{}", g[w / 2]);
        assert!((g[1] - g[w - 1]).abs() < 1e-12, "±f must be symmetric");
        // Radial: the corner bin is sqrt(2)·Nyquist.
        let corner = g[(h / 2) * w + w / 2];
        assert!((corner - nyq * 2f64.sqrt()).abs() < 1e-12, "{corner}");
        // Non-negative and finite everywhere.
        assert!(g.iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    #[test]
    fn cycdeg_grid_scales_with_pix_per_deg() {
        let a = cycles_per_degree_grid(16, 16, 20.0);
        let b = cycles_per_degree_grid(16, 16, 40.0);
        for (x, y) in a.iter().zip(&b) {
            assert!((2.0 * x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn conv_with_an_all_pass_filter_is_the_identity() {
        let (w, h) = (5usize, 4usize);
        let x: Vec<f64> = (0..w * h).map(|i| (i as f64 * 0.37).sin()).collect();
        let filter = vec![1.0; (2 * w) * (2 * h)];
        let got = conv_fft_real(&x, w, h, &filter, 2 * w, 2 * h, 0.0);
        for (a, b) in got.iter().zip(&x) {
            assert!((a - b).abs() < 1e-10, "{a} != {b}");
        }
    }

    #[test]
    fn conv_with_a_dc_only_filter_returns_the_padded_mean() {
        // A filter that keeps only bin 0 replaces every output pixel with the
        // mean over the PADDED domain — which is how the constant pad value
        // enters the result.
        let (w, h) = (4usize, 4usize);
        let (pw, ph) = (2 * w, 2 * h);
        let x: Vec<f64> = (0..w * h).map(|i| i as f64).collect();
        let pad_value = 7.0;
        let mut filter = vec![0.0; pw * ph];
        filter[0] = 1.0;
        let got = conv_fft_real(&x, w, h, &filter, pw, ph, pad_value);
        let want =
            (x.iter().sum::<f64>() + pad_value * (pw * ph - w * h) as f64) / (pw * ph) as f64;
        for v in got {
            assert!((v - want).abs() < 1e-10, "{v} != {want}");
        }
    }

    #[test]
    fn conv_matches_direct_circular_convolution() {
        // Build a small real, symmetric spatial kernel, take its DFT as the
        // frequency response, and check the FFT path against an explicit
        // circular convolution over the padded domain.
        let (w, h) = (4usize, 3usize);
        let (pw, ph) = (2 * w, 2 * h);
        let x: Vec<f64> = (0..w * h).map(|i| (i as f64 * 1.7).cos()).collect();
        let pad_value = -0.5;

        // Spatial kernel: a 3×3 blur centred at (0,0) with wraparound.
        let mut k = vec![0.0; pw * ph];
        for dy in [ph - 1, 0, 1] {
            for dx in [pw - 1, 0, 1] {
                k[(dy % ph) * pw + (dx % pw)] += 1.0 / 9.0;
            }
        }
        let mut kf: Vec<Complex> = k.iter().map(|v| Complex::new(*v, 0.0)).collect();
        fft2(&mut kf, pw, ph);
        // The kernel is symmetric, so its response is real; drop the (tiny)
        // imaginary residue the way a zero-phase filter table would.
        let filter: Vec<f64> = kf.iter().map(|c| c.re).collect();
        assert!(kf.iter().all(|c| c.im.abs() < 1e-12));

        let got = conv_fft_real(&x, w, h, &filter, pw, ph, pad_value);

        // Direct circular convolution on the padded image.
        let mut padded = vec![pad_value; pw * ph];
        for y in 0..h {
            for xi in 0..w {
                padded[y * pw + xi] = x[y * w + xi];
            }
        }
        for y in 0..h {
            for xi in 0..w {
                let mut acc = 0.0;
                for (ky, kv) in k.chunks_exact(pw).enumerate() {
                    for (kx, kk) in kv.iter().enumerate() {
                        if *kk == 0.0 {
                            continue;
                        }
                        let sy = (y + ph - ky) % ph;
                        let sx = (xi + pw - kx) % pw;
                        acc += kk * padded[sy * pw + sx];
                    }
                }
                let g = got[y * w + xi];
                assert!((g - acc).abs() < 1e-10, "at ({y},{xi}): {g} != {acc}");
            }
        }
    }
}
