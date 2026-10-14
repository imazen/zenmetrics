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
pub(super) struct C64 {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl C64 {
    /// `re + i·im`.
    #[must_use]
    #[inline]
    pub(super) const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// `e^{iθ}`.
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

    #[cfg(test)]
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
#[cfg(test)]
pub(super) fn fft(buf: &mut [C64]) {
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
#[cfg(test)]
pub(super) fn ifft(buf: &mut [C64]) {
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

/// Radix-2 twiddle tables split into real and imaginary rows so stage
/// butterflies run on contiguous `f64` slices (the same values, and the
/// same per-element arithmetic, as the `Vec<Vec<C64>>` form).
struct Stages2 {
    /// `wre[s][k]` / `wim[s][k]` — twiddle `k` of stage `s`.
    wre: Vec<Vec<f64>>,
    /// Imaginary rows, parallel to `wre`.
    wim: Vec<Vec<f64>>,
    /// Bit-reversal permutation: batched inputs gather `src[k]` into
    /// `re8[rev[k]]` directly, so the kernel needs no swap pass.
    rev: Vec<u32>,
}

fn radix2_stages(n: usize) -> Stages2 {
    debug_assert!(n.is_power_of_two() && n >= 2);
    let mut wre = Vec::with_capacity(n.trailing_zeros() as usize);
    let mut wim = Vec::with_capacity(n.trailing_zeros() as usize);
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * PI / len as f64;
        let mut r = Vec::with_capacity(len / 2);
        let mut i = Vec::with_capacity(len / 2);
        for k in 0..len / 2 {
            let w = C64::expi(ang * k as f64);
            r.push(w.re);
            i.push(w.im);
        }
        wre.push(r);
        wim.push(i);
        len <<= 1;
    }
    let bits = n.trailing_zeros();
    let rev = (0..n as u32)
        .map(|i| i.reverse_bits() >> (32 - bits))
        .collect();
    Stages2 { wre, wim, rev }
}

/// In-place radix-2 DFT on split real/imaginary planes. The butterfly
/// arithmetic is identical to the complex-buffer version — only the
/// layout changed, so results are bit-identical.
fn fft_radix2_soa(re: &mut [f64], im: &mut [f64], stages: &Stages2) {
    let n = re.len();
    debug_assert!(n.is_power_of_two() && n == im.len());
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = (i as u32).reverse_bits() >> (32 - bits);
        if j as usize > i {
            re.swap(i, j as usize);
            im.swap(i, j as usize);
        }
    }
    for (wre, wim) in stages.wre.iter().zip(&stages.wim) {
        let half = wre.len();
        let len = half * 2;
        for start in (0..n).step_by(len) {
            let (a_re, b_re) = re[start..start + len].split_at_mut(half);
            let (a_im, b_im) = im[start..start + len].split_at_mut(half);
            for k in 0..half {
                let vr = b_re[k] * wre[k] - b_im[k] * wim[k];
                let vi = b_re[k] * wim[k] + b_im[k] * wre[k];
                b_re[k] = a_re[k] - vr;
                b_im[k] = a_im[k] - vi;
                a_re[k] += vr;
                a_im[k] += vi;
            }
        }
    }
}

/// Complex-buffer convenience wrapper: splits, transforms, merges.
#[cfg(test)]
fn fft_radix2(buf: &mut [C64], stages: &Stages2) {
    let n = buf.len();
    let mut re = Vec::with_capacity(n);
    let mut im = Vec::with_capacity(n);
    for c in buf.iter() {
        re.push(c.re);
        im.push(c.im);
    }
    fft_radix2_soa(&mut re, &mut im, stages);
    for (c, (&r, &i)) in buf.iter_mut().zip(re.iter().zip(&im)) {
        *c = C64::new(r, i);
    }
}

/// Eight length-`n` radix-2 transforms at once on `[k·8 + j]` SoA lanes —
/// the `f64` twin of `crate::fft::fft_batch8`: lane `j` is transform `j`, so
/// every butterfly (including the small stages that are strided per row) is
/// a full `f64x8` op with splat twiddles and no shuffles. The per-lane
/// arithmetic is identical to [`fft_radix2_soa`] — same expression, same
/// order — so batched results are bit-identical to the per-transform path.
///
/// Input arrives already bit-reversed: callers gather `src[k]` into
/// `re8[rev[k]·8 + j]` (see [`Stages2::rev`]), which writes the same
/// permutation the scalar swap loop would — just without a separate pass.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f64x8))]
fn fft8_inner<const RELAXED: bool>(
    token: Token,
    re8: &mut [f64],
    im8: &mut [f64],
    n: usize,
    stages: &Stages2,
) {
    debug_assert_eq!(re8.len(), n * 8);
    debug_assert_eq!(im8.len(), n * 8);
    for (wre, wim) in stages.wre.iter().zip(&stages.wim) {
        let half = wre.len();
        let len = half * 2;
        for start in (0..n).step_by(len) {
            for (k, (&wr, &wi)) in wre.iter().zip(wim.iter()).enumerate() {
                let a = (start + k) * 8;
                let b = (start + k + half) * 8;
                let wre = f64x8::splat_t(token, wr);
                let wim = f64x8::splat_t(token, wi);
                let ure = f64x8::load_t(token, (&re8[a..a + 8]).try_into().unwrap());
                let uim = f64x8::load_t(token, (&im8[a..a + 8]).try_into().unwrap());
                let vre = f64x8::load_t(token, (&re8[b..b + 8]).try_into().unwrap());
                let vim = f64x8::load_t(token, (&im8[b..b + 8]).try_into().unwrap());
                // RELAXED: fuse one product per complex half — `vim·(−wim)`
                // negates the splat scalar, never a vector (the x86 vector
                // `Neg` maps +0 to +0), and the fused term keeps one
                // rounding instead of two.
                let (tre, tim) = if RELAXED {
                    let nwim = f64x8::splat_t(token, -wi);
                    (vim.mul_add(nwim, vre * wre), vre.mul_add(wim, vim * wre))
                } else {
                    (vre * wre - vim * wim, vre * wim + vim * wre)
                };
                (ure + tre).store((&mut re8[a..a + 8]).try_into().unwrap());
                (uim + tim).store((&mut im8[a..a + 8]).try_into().unwrap());
                (ure - tre).store((&mut re8[b..b + 8]).try_into().unwrap());
                (uim - tim).store((&mut im8[b..b + 8]).try_into().unwrap());
            }
        }
    }
}

/// [`fft8_inner`] with runtime tier dispatch.
fn fft8<const RELAXED: bool>(re8: &mut [f64], im8: &mut [f64], n: usize, stages: &Stages2) {
    archmage::attuned!(
        fft8_inner::<RELAXED>(re8, im8, n, stages),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

/// Bluestein gather for a batch of eight: `sre8[rev[k]·8+j] =
/// re8[k·8+j]·chirp[k]` for `k < n`, zero for `k` in `n..m` — the SoA form
/// of `a[k] = buf[k]·chirp[k]`, written straight to bit-reversed
/// positions so the following [`fft8`] needs no swap pass.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f64x8))]
#[allow(clippy::too_many_arguments)]
fn blue8_in_inner<const RELAXED: bool>(
    token: Token,
    re8: &[f64],
    im8: &[f64],
    sre8: &mut [f64],
    sim8: &mut [f64],
    n: usize,
    m: usize,
    rev: &[u32],
    chirp_re: &[f64],
    chirp_im: &[f64],
) {
    for i in n..m {
        let d = rev[i] as usize * 8;
        f64x8::zero_t(token).store((&mut sre8[d..d + 8]).try_into().unwrap());
        f64x8::zero_t(token).store((&mut sim8[d..d + 8]).try_into().unwrap());
    }
    for k in 0..n {
        let d = rev[k] as usize * 8;
        let s = k * 8;
        let xre = f64x8::load_t(token, (&re8[s..s + 8]).try_into().unwrap());
        let xim = f64x8::load_t(token, (&im8[s..s + 8]).try_into().unwrap());
        let cr = f64x8::splat_t(token, chirp_re[k]);
        let ci = f64x8::splat_t(token, chirp_im[k]);
        let (tre, tim) = if RELAXED {
            let nci = f64x8::splat_t(token, -chirp_im[k]);
            (xim.mul_add(nci, xre * cr), xre.mul_add(ci, xim * cr))
        } else {
            (xre * cr - xim * ci, xre * ci + xim * cr)
        };
        tre.store((&mut sre8[d..d + 8]).try_into().unwrap());
        tim.store((&mut sim8[d..d + 8]).try_into().unwrap());
    }
}

/// The `x·bf` multiply and conjugate of the Bluestein inner pass — fused so
/// the imaginary part is stored negated, matching `x = x.mul(y); x = x.conj()`
/// in the scalar sequence. The negation is folded into the `bf_re` splat
/// (`xi·(−br) − xr·bi ≡ −(xr·bi + xi·br)` exactly, ±0 included) because the
/// x86 vector `Neg` lowers to `0 − x`, which maps `+0` to `+0`.
///
/// Reads `sre8`/`sim8` in natural order and writes the product to
/// `s2re8`/`s2im8` at bit-reversed positions, feeding the second
/// [`fft8`] without a swap pass.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f64x8))]
fn blue8_bf_inner<const RELAXED: bool>(
    token: Token,
    sre8: &[f64],
    sim8: &[f64],
    s2re8: &mut [f64],
    s2im8: &mut [f64],
    bf_re: &[f64],
    bf_im: &[f64],
    rev: &[u32],
) {
    for k in 0..bf_re.len() {
        let s = k * 8;
        let d = rev[k] as usize * 8;
        let xr = f64x8::load_t(token, (&sre8[s..s + 8]).try_into().unwrap());
        let xi = f64x8::load_t(token, (&sim8[s..s + 8]).try_into().unwrap());
        let br = f64x8::splat_t(token, bf_re[k]);
        let nbr = f64x8::splat_t(token, -bf_re[k]);
        let bi = f64x8::splat_t(token, bf_im[k]);
        let (tre, tim) = if RELAXED {
            let nbi = f64x8::splat_t(token, -bf_im[k]);
            (xi.mul_add(nbi, xr * br), xr.mul_add(nbi, xi * nbr))
        } else {
            (xr * br - xi * bi, xi * nbr - xr * bi)
        };
        tre.store((&mut s2re8[d..d + 8]).try_into().unwrap());
        tim.store((&mut s2im8[d..d + 8]).try_into().unwrap());
    }
}

/// Bluestein writeback: conjugate·scale fused into the chirp multiply,
/// `re8[k·8+j] = (sre·s − i·sim·s)·chirp[k]`, matching the scalar sequence
/// `v = v.conj(); v·= s; out = v·chirp[k]` exactly.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f64x8))]
fn blue8_out_inner<const RELAXED: bool>(
    token: Token,
    sre8: &[f64],
    sim8: &[f64],
    re8: &mut [f64],
    im8: &mut [f64],
    n: usize,
    chirp_re: &[f64],
    chirp_im: &[f64],
    s: f64,
) {
    let sv = f64x8::splat_t(token, s);
    // `is = −sim·s` computed as `sim·(−s)`: bitwise-identical including
    // signed zeros, and avoids the `0 − x` vector negation quirk.
    let nsv = f64x8::splat_t(token, -s);
    for k in 0..n {
        let d = k * 8;
        let rs = f64x8::load_t(token, (&sre8[d..d + 8]).try_into().unwrap()) * sv;
        let is = f64x8::load_t(token, (&sim8[d..d + 8]).try_into().unwrap()) * nsv;
        let cr = f64x8::splat_t(token, chirp_re[k]);
        let ci = f64x8::splat_t(token, chirp_im[k]);
        let (tre, tim) = if RELAXED {
            let nci = f64x8::splat_t(token, -chirp_im[k]);
            (is.mul_add(nci, rs * cr), rs.mul_add(ci, is * cr))
        } else {
            (rs * cr - is * ci, rs * ci + is * cr)
        };
        tre.store((&mut re8[d..d + 8]).try_into().unwrap());
        tim.store((&mut im8[d..d + 8]).try_into().unwrap());
    }
}

/// [`blue8_in_inner`] with runtime tier dispatch.
#[allow(clippy::too_many_arguments)]
fn blue8_in<const RELAXED: bool>(
    re8: &[f64],
    im8: &[f64],
    sre8: &mut [f64],
    sim8: &mut [f64],
    n: usize,
    m: usize,
    rev: &[u32],
    chirp_re: &[f64],
    chirp_im: &[f64],
) {
    archmage::attuned!(
        blue8_in_inner::<RELAXED>(re8, im8, sre8, sim8, n, m, rev, chirp_re, chirp_im),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

/// [`blue8_bf_inner`] with runtime tier dispatch.
fn blue8_bf<const RELAXED: bool>(
    sre8: &[f64],
    sim8: &[f64],
    s2re8: &mut [f64],
    s2im8: &mut [f64],
    bf_re: &[f64],
    bf_im: &[f64],
    rev: &[u32],
) {
    archmage::attuned!(
        blue8_bf_inner::<RELAXED>(sre8, sim8, s2re8, s2im8, bf_re, bf_im, rev),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

/// [`blue8_out_inner`] with runtime tier dispatch.
#[allow(clippy::too_many_arguments)]
fn blue8_out<const RELAXED: bool>(
    sre8: &[f64],
    sim8: &[f64],
    re8: &mut [f64],
    im8: &mut [f64],
    n: usize,
    chirp_re: &[f64],
    chirp_im: &[f64],
    s: f64,
) {
    archmage::attuned!(
        blue8_out_inner::<RELAXED>(sre8, sim8, re8, im8, n, chirp_re, chirp_im, s),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

/// Bluestein chirp-z transform for arbitrary `n`.
#[cfg(test)]
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

enum Plan {
    Trivial,
    Radix2(Stages2),
    Bluestein {
        /// `e^{−iπk²/n}` split into re/im rows.
        chirp_re: Vec<f64>,
        /// Imaginary row of the chirp.
        chirp_im: Vec<f64>,
        /// DFT of the chirp-conjugate kernel, re/im rows.
        bf_re: Vec<f64>,
        /// Imaginary row of `bf`.
        bf_im: Vec<f64>,
        stages: Stages2,
        /// Work rows (`m` each) for the scalar chirp-z sequence.
        #[cfg(test)]
        sre: Vec<f64>,
        /// Imaginary work row.
        #[cfg(test)]
        sim: Vec<f64>,
    },
}

impl Plan {
    fn new(n: usize) -> Self {
        if n <= 1 {
            return Self::Trivial;
        }
        if n.is_power_of_two() {
            return Self::Radix2(radix2_stages(n));
        }
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
        let mut bf_re = Vec::with_capacity(m);
        let mut bf_im = Vec::with_capacity(m);
        for c in &bf {
            bf_re.push(c.re);
            bf_im.push(c.im);
        }
        fft_radix2_soa(&mut bf_re, &mut bf_im, &stages);
        Self::Bluestein {
            chirp_re: chirp.iter().map(|c| c.re).collect(),
            chirp_im: chirp.iter().map(|c| c.im).collect(),
            bf_re,
            bf_im,
            stages,
            #[cfg(test)]
            sre: vec![0.0; m],
            #[cfg(test)]
            sim: vec![0.0; m],
        }
    }

    /// Forward DFT on split planes — the SoA form of `forward`, sharing
    /// the same per-element arithmetic.
    #[cfg(test)]
    fn forward_soa(&mut self, re: &mut [f64], im: &mut [f64]) {
        match self {
            Self::Trivial => {}
            Self::Radix2(stages) => fft_radix2_soa(re, im, stages),
            Self::Bluestein {
                chirp_re,
                chirp_im,
                bf_re,
                bf_im,
                stages,
                sre,
                sim,
            } => {
                sre.fill(0.0);
                sim.fill(0.0);
                for k in 0..re.len() {
                    sre[k] = re[k] * chirp_re[k] - im[k] * chirp_im[k];
                    sim[k] = re[k] * chirp_im[k] + im[k] * chirp_re[k];
                }
                fft_radix2_soa(sre, sim, stages);
                for k in 0..sre.len() {
                    let xr = sre[k] * bf_re[k] - sim[k] * bf_im[k];
                    sim[k] = sre[k] * bf_im[k] + sim[k] * bf_re[k];
                    sre[k] = xr;
                }
                for v in sim.iter_mut() {
                    *v = -*v;
                }
                fft_radix2_soa(sre, sim, stages);
                let s = 1.0 / sre.len() as f64;
                for k in 0..re.len() {
                    let vr = sre[k] * s;
                    let vi = -sim[k] * s;
                    re[k] = vr * chirp_re[k] - vi * chirp_im[k];
                    im[k] = vr * chirp_im[k] + vi * chirp_re[k];
                }
            }
        }
    }

    /// Forward DFT of `buf`, in place (complex-buffer form).
    #[cfg(test)]
    fn forward(&mut self, buf: &mut [C64]) {
        let mut re = Vec::with_capacity(buf.len());
        let mut im = Vec::with_capacity(buf.len());
        for c in buf.iter() {
            re.push(c.re);
            im.push(c.im);
        }
        self.forward_soa(&mut re, &mut im);
        for (c, (&r, &i)) in buf.iter_mut().zip(re.iter().zip(&im)) {
            *c = C64::new(r, i);
        }
    }

    #[cfg(test)]
    fn inverse_soa(&mut self, re: &mut [f64], im: &mut [f64]) {
        let n = re.len();
        if n <= 1 {
            return;
        }
        for v in im.iter_mut() {
            *v = -*v;
        }
        self.forward_soa(re, im);
        let s = 1.0 / n as f64;
        for k in 0..n {
            im[k] = -im[k] * s;
            re[k] *= s;
        }
    }

    /// Input permutation for the batched path: `Some(rev)` means the
    /// caller gathers `src[k]` into `re8[rev[k]·8 + j]` (the radix-2
    /// bit-reversal, fused into the gather); `None` means natural order
    /// (`Bluestein`'s `blue8_in` applies the internal `m`-sized reversal
    /// itself; `Trivial` is a no-op either way).
    fn gather_perm(&self) -> Option<&[u32]> {
        match self {
            Self::Trivial | Self::Bluestein { .. } => None,
            Self::Radix2(stages) => Some(&stages.rev),
        }
    }

    /// Eight forward transforms at once on `[k·8 + j]` SoA lanes —
    /// bit-identical to eight [`Self::forward_soa`] calls. The input must
    /// already be laid out per [`Self::gather_perm`]. `RELAXED` lets the
    /// kernels fuse multiply-adds; `false` keeps the bit-exact sequence.
    fn run8<const RELAXED: bool>(
        &self,
        re8: &mut [f64],
        im8: &mut [f64],
        sre8: &mut [f64],
        sim8: &mut [f64],
        s2re8: &mut [f64],
        s2im8: &mut [f64],
    ) {
        match self {
            Self::Trivial => {}
            Self::Radix2(stages) => fft8::<RELAXED>(re8, im8, re8.len() / 8, stages),
            Self::Bluestein {
                chirp_re,
                chirp_im,
                bf_re,
                bf_im,
                stages,
                ..
            } => {
                let n = re8.len() / 8;
                let m = sre8.len() / 8;
                blue8_in::<RELAXED>(re8, im8, sre8, sim8, n, m, &stages.rev, chirp_re, chirp_im);
                fft8::<RELAXED>(sre8, sim8, m, stages);
                blue8_bf::<RELAXED>(sre8, sim8, s2re8, s2im8, bf_re, bf_im, &stages.rev);
                fft8::<RELAXED>(s2re8, s2im8, m, stages);
                blue8_out::<RELAXED>(
                    s2re8,
                    s2im8,
                    re8,
                    im8,
                    n,
                    chirp_re,
                    chirp_im,
                    1.0 / m as f64,
                );
            }
        }
    }

    /// [`Self::run8`] with a runtime `FpMode` switch — `relaxed` picks the
    /// FMA-fused const instantiation, `false` the bit-exact one.
    #[allow(clippy::too_many_arguments)]
    fn run8_mode(
        &self,
        relaxed: bool,
        re8: &mut [f64],
        im8: &mut [f64],
        sre8: &mut [f64],
        sim8: &mut [f64],
        s2re8: &mut [f64],
        s2im8: &mut [f64],
    ) {
        if relaxed {
            self.run8::<true>(re8, im8, sre8, sim8, s2re8, s2im8);
        } else {
            self.run8::<false>(re8, im8, sre8, sim8, s2re8, s2im8);
        }
    }

    /// The `sre8`/`sim8` scratch length a batch call needs: `m·8` for
    /// Bluestein, `n·8` for the direct plans.
    fn work_len(&self, n: usize) -> usize {
        match self {
            Self::Trivial | Self::Radix2(_) => n,
            Self::Bluestein { bf_re, .. } => bf_re.len(),
        }
    }

    /// Inverse DFT of `buf`, in place, normalised by `1/n`.
    #[cfg(test)]
    fn inverse(&mut self, buf: &mut [C64]) {
        let mut re = Vec::with_capacity(buf.len());
        let mut im = Vec::with_capacity(buf.len());
        for c in buf.iter() {
            re.push(c.re);
            im.push(c.im);
        }
        self.inverse_soa(&mut re, &mut im);
        for (c, (&r, &i)) in buf.iter_mut().zip(re.iter().zip(&im)) {
            *c = C64::new(r, i);
        }
    }
}

struct Plan2 {
    width: usize,
    height: usize,
    rows: Plan,
    columns: Plan,
    /// Batch-of-8 transform lanes, `axis_len·8`.
    re8: Vec<f64>,
    /// Imaginary lanes.
    im8: Vec<f64>,
    /// Bluestein work lanes for the larger axis plan.
    sre8: Vec<f64>,
    /// Imaginary work lanes.
    sim8: Vec<f64>,
    /// Second Bluestein work pair — `blue8_bf` writes the second
    /// transform's input here at bit-reversed positions.
    s2re8: Vec<f64>,
    /// Imaginary work lanes.
    s2im8: Vec<f64>,
    /// Whole-plane re/im copies — the transforms all run on these.
    pre: Vec<f64>,
    /// Imaginary plane.
    pim: Vec<f64>,
}

impl Plan2 {
    fn new(width: usize, height: usize) -> Self {
        let rows = Plan::new(width);
        let columns = Plan::new(height);
        let m8 = rows.work_len(width).max(columns.work_len(height)) * 8;
        Self {
            width,
            height,
            rows,
            columns,
            re8: vec![0.0; width.max(height) * 8],
            im8: vec![0.0; width.max(height) * 8],
            sre8: vec![0.0; m8],
            sim8: vec![0.0; m8],
            s2re8: vec![0.0; m8],
            s2im8: vec![0.0; m8],
            pre: vec![0.0; width * height],
            pim: vec![0.0; width * height],
        }
    }

    /// Forward row pass: 8 rows of `buf` per block, transposed into the
    /// lane layout (bit-reversed positions for radix-2 plans),
    /// transformed, scattered into the working planes.
    fn rows_fwd(&mut self, buf: &[C64], relaxed: bool) {
        let (w, h) = (self.width, self.height);
        let perm = self.rows.gather_perm();
        for y0 in (0..h).step_by(8) {
            let rws = (h - y0).min(8);
            for j in 0..rws {
                let row = (y0 + j) * w;
                for k in 0..w {
                    let d = match perm {
                        Some(r) => r[k] as usize,
                        None => k,
                    } * 8;
                    self.re8[d + j] = buf[row + k].re;
                    self.im8[d + j] = buf[row + k].im;
                }
            }
            {
                let m = self.rows.work_len(w);
                let re8 = &mut self.re8[..w * 8];
                let im8 = &mut self.im8[..w * 8];
                let sre8 = &mut self.sre8[..m * 8];
                let sim8 = &mut self.sim8[..m * 8];
                let s2re8 = &mut self.s2re8[..m * 8];
                let s2im8 = &mut self.s2im8[..m * 8];
                self.rows
                    .run8_mode(relaxed, re8, im8, sre8, sim8, s2re8, s2im8);
            }
            for j in 0..rws {
                let row = (y0 + j) * w;
                for k in 0..w {
                    self.pre[row + k] = self.re8[k * 8 + j];
                    self.pim[row + k] = self.im8[k * 8 + j];
                }
            }
        }
    }

    /// Inverse row pass: transposed gather from the working planes with
    /// the per-axis input conjugate fused in, then the conjugate·`1/w`
    /// writeback fused into the `buf` store (`ifft` per axis is
    /// `conj∘fwd∘conj·(1/n)` — both conjugates land in the data passes).
    #[cfg(test)]
    fn rows_inv(&mut self, buf: &mut [C64], relaxed: bool) {
        let (w, h) = (self.width, self.height);
        let perm = self.rows.gather_perm();
        let s = 1.0 / w as f64;
        for y0 in (0..h).step_by(8) {
            let rws = (h - y0).min(8);
            for j in 0..rws {
                let row = (y0 + j) * w;
                for k in 0..w {
                    let d = match perm {
                        Some(r) => r[k] as usize,
                        None => k,
                    } * 8;
                    self.re8[d + j] = self.pre[row + k];
                    self.im8[d + j] = -self.pim[row + k];
                }
            }
            {
                let m = self.rows.work_len(w);
                let re8 = &mut self.re8[..w * 8];
                let im8 = &mut self.im8[..w * 8];
                let sre8 = &mut self.sre8[..m * 8];
                let sim8 = &mut self.sim8[..m * 8];
                let s2re8 = &mut self.s2re8[..m * 8];
                let s2im8 = &mut self.s2im8[..m * 8];
                self.rows
                    .run8_mode(relaxed, re8, im8, sre8, sim8, s2re8, s2im8);
            }
            for j in 0..rws {
                let row = (y0 + j) * w;
                for k in 0..w {
                    buf[row + k] = C64::new(self.re8[k * 8 + j] * s, -self.im8[k * 8 + j] * s);
                }
            }
        }
    }

    /// Forward column pass: the `[k·8 + j]` lane layout is a contiguous
    /// 8-`f64` row of the working plane; results go straight back to `buf`.
    fn columns_fwd(&mut self, buf: &mut [C64], relaxed: bool) {
        let (w, h) = (self.width, self.height);
        let perm = self.columns.gather_perm();
        for x0 in (0..w).step_by(8) {
            let cols = (w - x0).min(8);
            for y in 0..h {
                let (d, s) = (
                    match perm {
                        Some(r) => r[y] as usize,
                        None => y,
                    } * 8,
                    y * w + x0,
                );
                self.re8[d..d + cols].copy_from_slice(&self.pre[s..s + cols]);
                self.im8[d..d + cols].copy_from_slice(&self.pim[s..s + cols]);
            }
            {
                let m = self.columns.work_len(h);
                let re8 = &mut self.re8[..h * 8];
                let im8 = &mut self.im8[..h * 8];
                let sre8 = &mut self.sre8[..m * 8];
                let sim8 = &mut self.sim8[..m * 8];
                let s2re8 = &mut self.s2re8[..m * 8];
                let s2im8 = &mut self.s2im8[..m * 8];
                self.columns
                    .run8_mode(relaxed, re8, im8, sre8, sim8, s2re8, s2im8);
            }
            for y in 0..h {
                let (d, s) = (y * 8, y * w + x0);
                for j in 0..cols {
                    buf[s + j] = C64::new(self.re8[d + j], self.im8[d + j]);
                }
            }
        }
    }

    /// Inverse column pass: gather straight from `buf` with the input
    /// conjugate fused in (eight adjacent columns per block, bit-reversed
    /// positions for radix-2 plans), transform, scatter into the planes
    /// with the conjugate·`1/h` writeback fused in.
    #[cfg(test)]
    fn columns_inv(&mut self, buf: &[C64], relaxed: bool) {
        let (w, h) = (self.width, self.height);
        let perm = self.columns.gather_perm();
        let sc = 1.0 / h as f64;
        for x0 in (0..w).step_by(8) {
            let cols = (w - x0).min(8);
            for y in 0..h {
                let (d, s) = (
                    match perm {
                        Some(r) => r[y] as usize,
                        None => y,
                    } * 8,
                    y * w + x0,
                );
                for j in 0..cols {
                    self.re8[d + j] = buf[s + j].re;
                    self.im8[d + j] = -buf[s + j].im;
                }
            }
            {
                let m = self.columns.work_len(h);
                let re8 = &mut self.re8[..h * 8];
                let im8 = &mut self.im8[..h * 8];
                let sre8 = &mut self.sre8[..m * 8];
                let sim8 = &mut self.sim8[..m * 8];
                let s2re8 = &mut self.s2re8[..m * 8];
                let s2im8 = &mut self.s2im8[..m * 8];
                self.columns
                    .run8_mode(relaxed, re8, im8, sre8, sim8, s2re8, s2im8);
            }
            for y in 0..h {
                let (d, s) = (y * 8, y * w + x0);
                for j in 0..cols {
                    self.pre[s + j] = self.re8[d + j] * sc;
                    self.pim[s + j] = -self.im8[d + j] * sc;
                }
            }
        }
    }

    /// Plane variants of the four passes: identical arithmetic, but the
    /// gathers/scatters touch contiguous `f64` runs instead of `C64`
    /// pairs — the hot `conv_fft_pad` path uses these.
    fn rows_fwd_p(&mut self, re: &[f64], im: &[f64], relaxed: bool) {
        let (w, h) = (self.width, self.height);
        let perm = self.rows.gather_perm();
        for y0 in (0..h).step_by(8) {
            let rws = (h - y0).min(8);
            for j in 0..rws {
                let row = (y0 + j) * w;
                for k in 0..w {
                    let d = match perm {
                        Some(r) => r[k] as usize,
                        None => k,
                    } * 8;
                    self.re8[d + j] = re[row + k];
                    self.im8[d + j] = im[row + k];
                }
            }
            {
                let m = self.rows.work_len(w);
                let re8 = &mut self.re8[..w * 8];
                let im8 = &mut self.im8[..w * 8];
                let sre8 = &mut self.sre8[..m * 8];
                let sim8 = &mut self.sim8[..m * 8];
                let s2re8 = &mut self.s2re8[..m * 8];
                let s2im8 = &mut self.s2im8[..m * 8];
                self.rows
                    .run8_mode(relaxed, re8, im8, sre8, sim8, s2re8, s2im8);
            }
            for j in 0..rws {
                let row = (y0 + j) * w;
                for k in 0..w {
                    self.pre[row + k] = self.re8[k * 8 + j];
                    self.pim[row + k] = self.im8[k * 8 + j];
                }
            }
        }
    }

    fn rows_inv_p(&mut self, re: &mut [f64], im: &mut [f64], relaxed: bool) {
        let (w, h) = (self.width, self.height);
        let perm = self.rows.gather_perm();
        let s = 1.0 / w as f64;
        for y0 in (0..h).step_by(8) {
            let rws = (h - y0).min(8);
            for j in 0..rws {
                let row = (y0 + j) * w;
                for k in 0..w {
                    let d = match perm {
                        Some(r) => r[k] as usize,
                        None => k,
                    } * 8;
                    self.re8[d + j] = self.pre[row + k];
                    self.im8[d + j] = -self.pim[row + k];
                }
            }
            {
                let m = self.rows.work_len(w);
                let re8 = &mut self.re8[..w * 8];
                let im8 = &mut self.im8[..w * 8];
                let sre8 = &mut self.sre8[..m * 8];
                let sim8 = &mut self.sim8[..m * 8];
                let s2re8 = &mut self.s2re8[..m * 8];
                let s2im8 = &mut self.s2im8[..m * 8];
                self.rows
                    .run8_mode(relaxed, re8, im8, sre8, sim8, s2re8, s2im8);
            }
            for j in 0..rws {
                let row = (y0 + j) * w;
                for k in 0..w {
                    re[row + k] = self.re8[k * 8 + j] * s;
                    im[row + k] = -self.im8[k * 8 + j] * s;
                }
            }
        }
    }

    fn columns_fwd_p(&mut self, re: &mut [f64], im: &mut [f64], relaxed: bool) {
        let (w, h) = (self.width, self.height);
        let perm = self.columns.gather_perm();
        for x0 in (0..w).step_by(8) {
            let cols = (w - x0).min(8);
            for y in 0..h {
                let (d, s) = (
                    match perm {
                        Some(r) => r[y] as usize,
                        None => y,
                    } * 8,
                    y * w + x0,
                );
                self.re8[d..d + cols].copy_from_slice(&self.pre[s..s + cols]);
                self.im8[d..d + cols].copy_from_slice(&self.pim[s..s + cols]);
            }
            {
                let m = self.columns.work_len(h);
                let re8 = &mut self.re8[..h * 8];
                let im8 = &mut self.im8[..h * 8];
                let sre8 = &mut self.sre8[..m * 8];
                let sim8 = &mut self.sim8[..m * 8];
                let s2re8 = &mut self.s2re8[..m * 8];
                let s2im8 = &mut self.s2im8[..m * 8];
                self.columns
                    .run8_mode(relaxed, re8, im8, sre8, sim8, s2re8, s2im8);
            }
            for y in 0..h {
                let (d, s) = (y * 8, y * w + x0);
                re[s..s + cols].copy_from_slice(&self.re8[d..d + cols]);
                im[s..s + cols].copy_from_slice(&self.im8[d..d + cols]);
            }
        }
    }

    fn columns_inv_p(&mut self, re: &[f64], im: &mut [f64], relaxed: bool) {
        let (w, h) = (self.width, self.height);
        let perm = self.columns.gather_perm();
        let sc = 1.0 / h as f64;
        for x0 in (0..w).step_by(8) {
            let cols = (w - x0).min(8);
            for y in 0..h {
                let (d, s) = (
                    match perm {
                        Some(r) => r[y] as usize,
                        None => y,
                    } * 8,
                    y * w + x0,
                );
                for j in 0..cols {
                    self.re8[d + j] = re[s + j];
                    self.im8[d + j] = -im[s + j];
                }
            }
            {
                let m = self.columns.work_len(h);
                let re8 = &mut self.re8[..h * 8];
                let im8 = &mut self.im8[..h * 8];
                let sre8 = &mut self.sre8[..m * 8];
                let sim8 = &mut self.sim8[..m * 8];
                let s2re8 = &mut self.s2re8[..m * 8];
                let s2im8 = &mut self.s2im8[..m * 8];
                self.columns
                    .run8_mode(relaxed, re8, im8, sre8, sim8, s2re8, s2im8);
            }
            for y in 0..h {
                let (d, s) = (y * 8, y * w + x0);
                for j in 0..cols {
                    self.pre[s + j] = self.re8[d + j] * sc;
                    self.pim[s + j] = -self.im8[d + j] * sc;
                }
            }
        }
    }

    /// Forward transform on split real/imaginary planes — same results as
    /// [`forward`] on the equivalent `C64` buffer.
    fn forward_p(&mut self, re: &mut [f64], im: &mut [f64], relaxed: bool) {
        assert_eq!(
            (re.len(), im.len()),
            (self.width * self.height, self.width * self.height),
            "fft2: plane size mismatch"
        );
        self.rows_fwd_p(re, im, relaxed);
        self.columns_fwd_p(re, im, relaxed);
    }

    /// Inverse transform on split real/imaginary planes.
    fn inverse_p(&mut self, re: &mut [f64], im: &mut [f64], relaxed: bool) {
        assert_eq!(
            (re.len(), im.len()),
            (self.width * self.height, self.width * self.height),
            "ifft2: plane size mismatch"
        );
        self.columns_inv_p(re, im, relaxed);
        self.rows_inv_p(re, im, relaxed);
    }

    fn forward(&mut self, buf: &mut [C64], relaxed: bool) {
        assert_eq!(
            buf.len(),
            self.width * self.height,
            "fft2: buffer size mismatch"
        );
        self.rows_fwd(buf, relaxed);
        self.columns_fwd(buf, relaxed);
    }

    #[cfg(test)]
    fn inverse(&mut self, buf: &mut [C64], relaxed: bool) {
        assert_eq!(
            buf.len(),
            self.width * self.height,
            "ifft2: buffer size mismatch"
        );
        self.columns_inv(buf, relaxed);
        self.rows_inv(buf, relaxed);
    }
}

std::thread_local! {
    /// Per-thread `(width, height) → Plan2` cache. A plan carries twiddle
    /// tables plus ~5·w·h of batch/scratch lanes, so rebuilding per call is a
    /// real fixed cost in `conv_fft_pad`. Plans are a pure function of the
    /// dims — reuse is bit-identical (`reused_2d_plans_*` tests).
    static PLAN2_CACHE: core::cell::RefCell<
        std::collections::HashMap<(usize, usize), Plan2>,
    > = core::cell::RefCell::new(std::collections::HashMap::new());
}

/// Run `f` on the cached plan for `width × height`, constructing it on
/// first use. Bounded to a handful of resident shapes — the metric works a
/// fixed lattice set per call.
fn with_plan2<R>(width: usize, height: usize, f: impl FnOnce(&mut Plan2) -> R) -> R {
    PLAN2_CACHE.with(|c| {
        let mut map = c.borrow_mut();
        if map.len() >= 8 && !map.contains_key(&(width, height)) {
            map.clear();
        }
        let plan = map.entry((width, height)).or_insert_with(|| Plan2::new(width, height));
        f(plan)
    })
}

/// Forward 2-D DFT of a row-major `height × width` buffer, in place.
pub(super) fn fft2(buf: &mut [C64], width: usize, height: usize, relaxed: bool) {
    with_plan2(width, height, |p| p.forward(buf, relaxed));
}

/// Inverse 2-D DFT of a row-major `height × width` buffer, in place,
/// normalised by `1/(width·height)`.
#[cfg(test)]
fn ifft2(buf: &mut [C64], width: usize, height: usize, relaxed: bool) {
    Plan2::new(width, height).inverse(buf, relaxed);
}

/// `np.fft.fft2` of a real row-major `width × height` buffer.
#[must_use]
pub(super) fn fft2_of(x: &[f64], width: usize, height: usize, relaxed: bool) -> Vec<C64> {
    assert_eq!(x.len(), width * height, "fft2_of: size");
    let mut buf: Vec<C64> = x.iter().map(|&v| C64::new(v, 0.0)).collect();
    fft2(&mut buf, width, height, relaxed);
    buf
}

/// How the image is extended onto the padded convolution lattice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Pad {
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
pub(super) fn pad_image(
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
    // Column map hoisted once; interior columns map to themselves, so each
    // output row is `srow` followed by the reflected/clamped pad run.
    let xmap: Vec<usize> = (0..pad_w).map(|px| map(px, width)).collect();
    for py in 0..pad_h {
        let sy = map(py, height);
        let orow = &mut out[py * pad_w..py * pad_w + pad_w];
        if sy == usize::MAX {
            let Pad::Constant(v) = pad else {
                unreachable!()
            };
            orow.fill(v);
            continue;
        }
        let srow = &x[sy * width..sy * width + width];
        orow[..width].copy_from_slice(srow);
        for (o, &sx) in orow[width..].iter_mut().zip(&xmap[width..]) {
            *o = match sx {
                usize::MAX => {
                    let Pad::Constant(v) = pad else {
                        unreachable!()
                    };
                    v
                }
                sx => srow[sx],
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
pub(super) fn conv_fft_pad(
    x: &[f64],
    width: usize,
    height: usize,
    filter: &[f64],
    pad_w: usize,
    pad_h: usize,
    pad: Pad,
    relaxed: bool,
) -> Vec<f64> {
    assert_eq!(filter.len(), pad_w * pad_h, "conv_fft_pad: filter size");
    // Split real/imaginary planes end to end: the pointwise multiply is
    // contiguous and vectorizable, and no C64 (de)interleave is needed.
    let mut re = pad_image(x, width, height, pad_w, pad_h, pad);
    let mut im = vec![0.0; pad_w * pad_h];
    with_plan2(pad_w, pad_h, |plan| {
        plan.forward_p(&mut re, &mut im, relaxed);
        for ((r, v), &f) in re.iter_mut().zip(im.iter_mut()).zip(filter) {
            *r *= f;
            *v *= f;
        }
        plan.inverse_p(&mut re, &mut im, relaxed);
    });
    let mut out = Vec::with_capacity(width * height);
    for y in 0..height {
        out.extend_from_slice(&re[y * pad_w..y * pad_w + width]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn same_bits(actual: &[C64], expected: &[C64]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(a.re.to_bits(), b.re.to_bits(), "real bin {i}");
            assert_eq!(a.im.to_bits(), b.im.to_bits(), "imaginary bin {i}");
        }
    }

    #[test]
    fn reused_1d_plans_match_unplanned_bits() {
        for n in [0, 1, 2, 3, 4, 8, 17, 96, 100, 97, 180, 1021, 1024] {
            let mut plan = Plan::new(n);
            for seed in 0..4 {
                let input: Vec<C64> = (0..n)
                    .map(|k| {
                        if seed == 3 {
                            C64::default()
                        } else {
                            C64::new(
                                (k as f64 * 0.37 + seed as f64).sin(),
                                (k as f64 * 1.1 - seed as f64).cos(),
                            )
                        }
                    })
                    .collect();
                let mut expected = input.clone();
                fft(&mut expected);
                let mut got = input;
                plan.forward(&mut got);
                same_bits(&got, &expected);
                ifft(&mut expected);
                plan.inverse(&mut got);
                same_bits(&got, &expected);
            }
        }
    }

    #[test]
    fn reused_2d_plans_match_ordered_row_column_bits() {
        for (w, h) in [(1, 1), (7, 13), (64, 32), (97, 100), (128, 96)] {
            let mut plan = Plan2::new(w, h);
            for seed in 0..3 {
                let input: Vec<C64> = (0..w * h)
                    .map(|i| C64::new((i as f64 * 0.37 + seed as f64).sin(), 0.0))
                    .collect();
                let mut expected = input.clone();
                for row in expected.chunks_exact_mut(w) {
                    fft(row);
                }
                let mut col = vec![C64::default(); h];
                for x in 0..w {
                    for y in 0..h {
                        col[y] = expected[y * w + x];
                    }
                    fft(&mut col);
                    for y in 0..h {
                        expected[y * w + x] = col[y];
                    }
                }
                let mut got = input;
                plan.forward(&mut got, false);
                same_bits(&got, &expected);
                let mut wrapper_inverse = got.clone();
                ifft2(&mut wrapper_inverse, w, h, false);
                for x in 0..w {
                    for y in 0..h {
                        col[y] = expected[y * w + x];
                    }
                    ifft(&mut col);
                    for y in 0..h {
                        expected[y * w + x] = col[y];
                    }
                }
                for row in expected.chunks_exact_mut(w) {
                    ifft(row);
                }
                plan.inverse(&mut got, false);
                same_bits(&got, &expected);
                same_bits(&wrapper_inverse, &expected);
            }
        }
    }

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
        fft2(&mut kbuf, pw, ph, false);
        let filt: Vec<f64> = kbuf.iter().map(|c| c.re).collect();
        let got = conv_fft_pad(&x, w, h, &filt, pw, ph, Pad::Replicate, false);
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
