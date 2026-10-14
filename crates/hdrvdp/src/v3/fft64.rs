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
            // `pre`/`pim` rows padded to `width + 8`: column passes read
            // one 64 B lane group per row, and a pow-2 `w` makes them
            // stride-by-16 KB — every read lands in the same L1 set. The
            // +8 rotates the set index one line per row.
            re8: vec![0.0; width.max(height) * 8],
            im8: vec![0.0; width.max(height) * 8],
            sre8: vec![0.0; m8],
            sim8: vec![0.0; m8],
            s2re8: vec![0.0; m8],
            s2im8: vec![0.0; m8],
            pre: vec![0.0; height * (width + 8)],
            pim: vec![0.0; height * (width + 8)],
        }
    }

    /// Forward row pass: 8 rows of `buf` per block, transposed into the
    /// lane layout (bit-reversed positions for radix-2 plans),
    /// transformed, scattered into the working planes.
    fn rows_fwd(&mut self, buf: &[C64], relaxed: bool) {
        let (w, h) = (self.width, self.height);
        let stride = w + 8;
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
                let row = (y0 + j) * stride;
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
        let stride = w + 8;
        let perm = self.rows.gather_perm();
        let s = 1.0 / w as f64;
        for y0 in (0..h).step_by(8) {
            let rws = (h - y0).min(8);
            for j in 0..rws {
                let row = (y0 + j) * stride;
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
        let stride = w + 8;
        let perm = self.columns.gather_perm();
        for x0 in (0..w).step_by(8) {
            let cols = (w - x0).min(8);
            if cols == 8 {
                // Fixed-8 copies inline to vector moves — a runtime-length
                // `copy_from_slice` defers to `memmove` per row.
                for y in 0..h {
                    let (d, s) = (
                        match perm {
                            Some(r) => r[y] as usize,
                            None => y,
                        } * 8,
                        y * stride + x0,
                    );
                    self.re8[d..d + 8].copy_from_slice(&self.pre[s..s + 8]);
                    self.im8[d..d + 8].copy_from_slice(&self.pim[s..s + 8]);
                }
            } else {
                for y in 0..h {
                    let (d, s) = (
                        match perm {
                            Some(r) => r[y] as usize,
                            None => y,
                        } * 8,
                        y * stride + x0,
                    );
                    self.re8[d..d + cols].copy_from_slice(&self.pre[s..s + cols]);
                    self.im8[d..d + cols].copy_from_slice(&self.pim[s..s + cols]);
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
        let stride = w + 8;
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
                let (d, s) = (y * 8, y * stride + x0);
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
    /// Inverse row pass writing straight into the cropped output — `out`
    /// is `out_h` rows of `out_w` (its stride too), so the scatter is fully
    /// contiguous. The imaginary output lanes are dead in the conv path and
    /// not stored; the gather still reads all `w` columns and negates `pim`
    /// (every output needs the full conjugated input).
    /// `post` applied per output element *after* the `1/w` scale — the
    /// caller's own post-pass fused in (same expression, same bits). The
    /// store lands at `out[(row·out_w + col)·xstride + xoff]` — strided
    /// interleaves (e.g. per-channel planes) go through the scalar arm.
    fn rows_inv_out(
        &mut self,
        out: &mut [f64],
        relaxed: bool,
        out_w: usize,
        out_h: usize,
        xstride: usize,
        xoff: usize,
        clamp: Option<(f64, f64)>,
    ) {
        let (w, _h) = (self.width, self.height);
        let stride = w + 8;
        debug_assert!(out.len() >= (out_h * out_w - 1) * xstride + xoff + 1);
        let perm = self.rows.gather_perm();
        let s = 1.0 / w as f64;
        let clamp = clamp.unwrap_or((f64::NEG_INFINITY, f64::INFINITY));
        #[cfg(feature = "parallel")]
        if out_h >= 16 && w >= 64 && rayon::current_num_threads() > 1 {
            self.rows_inv_out_par(out, relaxed, out_w, out_h, xstride, xoff, clamp);
            return;
        }
        for y0 in (0..out_h).step_by(8) {
            let rws = (out_h - y0).min(8);
            if perm.is_none() && rws == 8 {
                // Tiled 8×8 transpose gather: contiguous row reads out of
                // pre/pim, contiguous 64 B lane writes (im negated).
                for k0 in (0..w).step_by(8) {
                    let kn = (k0 + 8).min(w) - k0;
                    let mut tre = [[0.0f64; 8]; 8];
                    let mut tim = [[0.0f64; 8]; 8];
                    for j in 0..8 {
                        let row = (y0 + j) * stride + k0;
                        tre[j][..kn].copy_from_slice(&self.pre[row..row + kn]);
                        tim[j][..kn].copy_from_slice(&self.pim[row..row + kn]);
                    }
                    for t in 0..kn {
                        let lane = (k0 + t) * 8;
                        for j in 0..8 {
                            self.re8[lane + j] = tre[j][t];
                            self.im8[lane + j] = -tim[j][t];
                        }
                    }
                }
            } else {
                for j in 0..rws {
                    let row = (y0 + j) * stride;
                    for k in 0..w {
                        let d = match perm {
                            Some(r) => r[k] as usize,
                            None => k,
                        } * 8;
                        self.re8[d + j] = self.pre[row + k];
                        self.im8[d + j] = -self.pim[row + k];
                    }
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
            if perm.is_none() && rws == 8 && xstride == 1 {
                for k0 in (0..out_w).step_by(8) {
                    let kn = (k0 + 8).min(out_w) - k0;
                    let mut tre = [[0.0f64; 8]; 8];
                    for t in 0..kn {
                        let lane = (k0 + t) * 8;
                        for j in 0..8 {
                            tre[j][t] = (self.re8[lane + j] * s).clamp(clamp.0, clamp.1);
                        }
                    }
                    for j in 0..8 {
                        let row = (y0 + j) * out_w + k0;
                        out[row + xoff..row + xoff + kn].copy_from_slice(&tre[j][..kn]);
                    }
                }
            } else {
                for j in 0..rws {
                    let row = (y0 + j) * out_w;
                    for k in 0..out_w {
                        out[(row + k) * xstride + xoff] =
                            (self.re8[k * 8 + j] * s).clamp(clamp.0, clamp.1);
                    }
                }
            }
        }
    }

    /// `parallel` variant of [`Self::rows_inv_out`]: `out` row-ranges are
    /// disjoint, so each chunk gathers, transforms and writes its rows
    /// straight through — `1/w` scale, clamp and interleaved store inside
    /// the task, no strip buffers or serial scatter. Identical
    /// per-element arithmetic; only loop distribution differs.
    #[cfg(feature = "parallel")]
    #[allow(clippy::too_many_arguments)]
    fn rows_inv_out_par(
        &mut self,
        out: &mut [f64],
        relaxed: bool,
        out_w: usize,
        out_h: usize,
        xstride: usize,
        xoff: usize,
        clamp: (f64, f64),
    ) {
        use rayon::prelude::*;
        let (w, _h) = (self.width, self.height);
        let stride = w + 8;
        let w8 = w * 8;
        let m8 = self.rows.work_len(w) * 8;
        let rows = &self.rows;
        let perm = rows.gather_perm();
        let pre = &self.pre;
        let pim = &self.pim;
        let nblk = out_h.div_ceil(8);
        let nchunks = nblk.min(rayon::current_num_threads());
        let chunk_len = nblk.div_ceil(nchunks);
        // Chunk `c` owns out-rows `[c·rw, (c+1)·rw)` — exactly the rows
        // its blocks `[c·chunk_len, …)` cover.
        let rw = chunk_len * 8;
        let row_bytes = out_w * xstride;
        let s = 1.0 / w as f64;
        out[..out_h * row_bytes]
            .par_chunks_mut(rw * row_bytes)
            .enumerate()
            .for_each(|(c, out_c)| {
                let b0 = c * chunk_len;
                let bn = nblk.saturating_sub(b0).min(chunk_len);
                if bn == 0 {
                    return;
                }
                let mut re8 = vec![0.0; w8];
                let mut im8 = vec![0.0; w8];
                let mut sre8 = vec![0.0; m8];
                let mut sim8 = vec![0.0; m8];
                let mut s2re8 = vec![0.0; m8];
                let mut s2im8 = vec![0.0; m8];
                for bi in b0..b0 + bn {
                    let y0 = bi * 8;
                    let rws = (out_h - y0).min(8);
                    if perm.is_none() && rws == 8 {
                        for k0 in (0..w).step_by(8) {
                            let kn = (k0 + 8).min(w) - k0;
                            let mut tre = [[0.0f64; 8]; 8];
                            let mut tim = [[0.0f64; 8]; 8];
                            for j in 0..8 {
                                let row = (y0 + j) * stride + k0;
                                tre[j][..kn].copy_from_slice(&pre[row..row + kn]);
                                tim[j][..kn].copy_from_slice(&pim[row..row + kn]);
                            }
                            for t in 0..kn {
                                let lane = (k0 + t) * 8;
                                for j in 0..8 {
                                    re8[lane + j] = tre[j][t];
                                    im8[lane + j] = -tim[j][t];
                                }
                            }
                        }
                    } else {
                        for j in 0..rws {
                            let row = (y0 + j) * stride;
                            for k in 0..w {
                                let d = match perm {
                                    Some(r) => r[k] as usize,
                                    None => k,
                                } * 8;
                                re8[d + j] = pre[row + k];
                                im8[d + j] = -pim[row + k];
                            }
                        }
                    }
                    rows.run8_mode(
                        relaxed,
                        &mut re8,
                        &mut im8,
                        &mut sre8,
                        &mut sim8,
                        &mut s2re8,
                        &mut s2im8,
                    );
                    let yl = y0 - c * rw;
                    for j in 0..rws {
                        let drow = (yl + j) * out_w;
                        for k in 0..out_w {
                            out_c[(drow + k) * xstride + xoff] =
                                (re8[k * 8 + j] * s).clamp(clamp.0, clamp.1);
                        }
                    }
                }
            });
    }

    /// `parallel` variant of [`Self::rows_fwd_p_pad`]: `pre`/`pim`
    /// row-ranges are disjoint, so each chunk gathers + transforms and
    /// writes its rows directly — no strip buffers or serial scatter.
    /// All-pad blocks past `src_h` are still skipped and replayed by the
    /// dedup tail — identical arithmetic, only loop distribution differs.
    #[cfg(feature = "parallel")]
    #[allow(clippy::too_many_arguments)]
    fn rows_fwd_p_pad_par(
        &mut self,
        x: &[f64],
        src_w: usize,
        src_h: usize,
        pad: Pad,
        relaxed: bool,
        xmap: &[usize],
        ymap: &[usize],
    ) {
        use rayon::prelude::*;
        let (w, h) = (self.width, self.height);
        let stride = w + 8;
        let w8 = w * 8;
        let m8 = self.rows.work_len(w) * 8;
        let rows = &self.rows;
        let perm = rows.gather_perm();
        let dedup_pad = !matches!(pad, Pad::Symmetric) && h > src_h;
        let nblk = h.div_ceil(8);
        let nchunks = nblk.min(rayon::current_num_threads());
        let chunk_len = nblk.div_ceil(nchunks);
        // Chunk `c` owns pre/pim rows `[c·rw, (c+1)·rw)` — exactly the
        // rows its blocks `[c·chunk_len, …)` cover.
        let rw = chunk_len * 8;
        let pre = &mut self.pre;
        let pim = &mut self.pim;
        pre.par_chunks_mut(rw * stride)
            .zip(pim.par_chunks_mut(rw * stride))
            .enumerate()
            .for_each(|(c, (pre_c, pim_c))| {
                let b0 = c * chunk_len;
                let bn = nblk.saturating_sub(b0).min(chunk_len);
                if bn == 0 {
                    return;
                }
                let mut re8 = vec![0.0; w8];
                let mut im8 = vec![0.0; w8];
                let mut sre8 = vec![0.0; m8];
                let mut sim8 = vec![0.0; m8];
                let mut s2re8 = vec![0.0; m8];
                let mut s2im8 = vec![0.0; m8];
                for bi in b0..b0 + bn {
                    let y0 = bi * 8;
                    let rws = (h - y0).min(8);
                    if dedup_pad && y0 > src_h {
                        // All-pad block — the dedup tail writes the
                        // canonical row into these rows.
                        continue;
                    }
                    im8[..w8].fill(0.0);
                    if perm.is_none() && rws == 8 {
                        let ybase: [usize; 8] = core::array::from_fn(|j| ymap[y0 + j]);
                        for k0 in (0..w).step_by(8) {
                            let kend = (k0 + 8).min(w);
                            if kend <= src_w {
                                let mut tile = [[0.0f64; 8]; 8];
                                for j in 0..8 {
                                    let sy = ybase[j];
                                    if sy == usize::MAX {
                                        let Pad::Constant(v) = pad else {
                                            unreachable!()
                                        };
                                        tile[j] = [v; 8];
                                    } else {
                                        tile[j][..kend - k0].copy_from_slice(
                                            &x[sy * src_w + k0..sy * src_w + kend],
                                        );
                                    }
                                }
                                for t in 0..kend - k0 {
                                    let lane = (k0 + t) * 8;
                                    for j in 0..8 {
                                        re8[lane + j] = tile[j][t];
                                    }
                                }
                            } else {
                                let id_end = src_w.clamp(k0, kend);
                                let mut tile = [[0.0f64; 8]; 8];
                                for j in 0..8 {
                                    let sy = ybase[j];
                                    if sy == usize::MAX {
                                        let Pad::Constant(v) = pad else {
                                            unreachable!()
                                        };
                                        tile[j] = [v; 8];
                                        continue;
                                    }
                                    let srow = &x[sy * src_w..sy * src_w + src_w];
                                    if id_end > k0 {
                                        tile[j][..id_end - k0]
                                            .copy_from_slice(&srow[k0..id_end]);
                                    }
                                    for t in id_end..kend {
                                        tile[j][t - k0] = match pad {
                                            Pad::Constant(v) => v,
                                            Pad::Replicate => srow[src_w - 1],
                                            Pad::Symmetric => srow[xmap[t]],
                                        };
                                    }
                                }
                                for t in 0..kend - k0 {
                                    let lane = (k0 + t) * 8;
                                    for j in 0..8 {
                                        re8[lane + j] = tile[j][t];
                                    }
                                }
                            }
                        }
                    } else {
                        for j in 0..rws {
                            let d_row = y0 + j;
                            let sy = ymap[d_row];
                            if sy == usize::MAX {
                                let Pad::Constant(v) = pad else {
                                    unreachable!()
                                };
                                for k in 0..w {
                                    let d = match perm {
                                        Some(r) => r[k] as usize,
                                        None => k,
                                    } * 8;
                                    re8[d + j] = v;
                                }
                                continue;
                            }
                            let srow = &x[sy * src_w..sy * src_w + src_w];
                            for k in 0..w {
                                let d = match perm {
                                    Some(r) => r[k] as usize,
                                    None => k,
                                } * 8;
                                re8[d + j] = match xmap[k] {
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
                    }
                    rows.run8_mode(
                        relaxed,
                        &mut re8,
                        &mut im8,
                        &mut sre8,
                        &mut sim8,
                        &mut s2re8,
                        &mut s2im8,
                    );
                    let yl = y0 - c * rw;
                    if perm.is_none() && rws == 8 {
                        // Transposed scatter: contiguous lane reads,
                        // contiguous row writes — same as sequential.
                        for k0 in (0..w).step_by(8) {
                            let kn = (k0 + 8).min(w) - k0;
                            let mut tre = [[0.0f64; 8]; 8];
                            let mut tim = [[0.0f64; 8]; 8];
                            for t in 0..kn {
                                let lane = (k0 + t) * 8;
                                for j in 0..8 {
                                    tre[j][t] = re8[lane + j];
                                    tim[j][t] = im8[lane + j];
                                }
                            }
                            for j in 0..8 {
                                let row = (yl + j) * stride + k0;
                                pre_c[row..row + kn].copy_from_slice(&tre[j][..kn]);
                                pim_c[row..row + kn].copy_from_slice(&tim[j][..kn]);
                            }
                        }
                    } else {
                        for j in 0..rws {
                            let row = (yl + j) * stride;
                            for k in 0..w {
                                pre_c[row + k] = re8[k * 8 + j];
                                pim_c[row + k] = im8[k * 8 + j];
                            }
                        }
                    }
                }
            });
        if dedup_pad {
            // Canonical pad row sits at `src_h`; every row of the skipped
            // all-pad blocks gets it — same copy_within as sequential.
            for y0 in (0..h).step_by(8) {
                if y0 <= src_h {
                    continue;
                }
                for row in y0..(y0 + 8).min(h) {
                    let dst = row * stride;
                    self.pre
                        .copy_within(src_h * stride..src_h * stride + w, dst);
                    self.pim
                        .copy_within(src_h * stride..src_h * stride + w, dst);
                }
            }
        }
    }

    /// Pad-index map: `i` → source coordinate, or `usize::MAX` for
    /// [`Pad::Constant`] positions. Interior maps to itself.
    fn pad_index(i: usize, n: usize, pad: Pad) -> usize {
        if i < n {
            i
        } else {
            match pad {
                Pad::Symmetric => (2 * n - 1).saturating_sub(i).min(n - 1),
                Pad::Replicate => n - 1,
                Pad::Constant(_) => usize::MAX,
            }
        }
    }

    /// Forward row transforms of a *real* image given unpadded: the gather
    /// applies the pad map straight from `x` (skipping the padded-plane
    /// round trip), the imaginary lanes are all +0, and results scatter
    /// into `pre`/`pim` for the column stage.
    fn rows_fwd_p_pad(&mut self, x: &[f64], src_w: usize, src_h: usize, pad: Pad, relaxed: bool) {
        let (w, h) = (self.width, self.height);
        let stride = w + 8;
        let perm = self.rows.gather_perm();
        let xmap: Vec<usize> = (0..w).map(|px| Self::pad_index(px, src_w, pad)).collect();
        let ymap: Vec<usize> = (0..h).map(|py| Self::pad_index(py, src_h, pad)).collect();
        // `Replicate`/`Constant` give every pad row (y ≥ src_h) the same
        // input, hence the same transform — copy the canonical row instead
        // of re-running the FFT per block. `Symmetric` rows differ.
        let dedup_pad = !matches!(pad, Pad::Symmetric) && h > src_h;
        let mut pad_row_ready = false;
        #[cfg(feature = "parallel")]
        if h >= 16 && w >= 64 && rayon::current_num_threads() > 1 {
            self.rows_fwd_p_pad_par(x, src_w, src_h, pad, relaxed, &xmap, &ymap);
            return;
        }
        for y0 in (0..h).step_by(8) {
            let rws = (h - y0).min(8);
            if dedup_pad && y0 >= src_h && pad_row_ready {
                for j in 0..rws {
                    self.pre
                        .copy_within(src_h * stride..src_h * stride + w, (y0 + j) * stride);
                    self.pim
                        .copy_within(src_h * stride..src_h * stride + w, (y0 + j) * stride);
                }
                continue;
            }
            // Imaginary lanes are all +0 for real input — one memset per
            // row block instead of an interleaved store per lane.
            self.im8[..w * 8].fill(0.0);
            if perm.is_none() && rws == 8 {
                // Tiled 8×8 transpose gather: contiguous reads out of the
                // source rows, contiguous 64 B lane writes into re8.
                let ybase: [usize; 8] = core::array::from_fn(|j| ymap[y0 + j]);
                for k0 in (0..w).step_by(8) {
                    let kend = (k0 + 8).min(w);
                    if kend <= src_w {
                        let mut tile = [[0.0f64; 8]; 8];
                        for j in 0..8 {
                            let sy = ybase[j];
                            if sy == usize::MAX {
                                let Pad::Constant(v) = pad else {
                                    unreachable!()
                                };
                                tile[j] = [v; 8];
                            } else {
                                tile[j][..kend - k0]
                                    .copy_from_slice(&x[sy * src_w + k0..sy * src_w + kend]);
                            }
                        }
                        for t in 0..kend - k0 {
                            let lane = (k0 + t) * 8;
                            for j in 0..8 {
                                self.re8[lane + j] = tile[j][t];
                            }
                        }
                    } else {
                        // Tail tile straddling/past `src_w`: contiguous
                        // copy for the identity columns, then the pad
                        // columns — constant lane-fill for `Constant`,
                        // edge broadcast for `Replicate`, mirror reads
                        // for `Symmetric`.
                        let id_end = src_w.clamp(k0, kend);
                        let mut tile = [[0.0f64; 8]; 8];
                        for j in 0..8 {
                            let sy = ybase[j];
                            if sy == usize::MAX {
                                let Pad::Constant(v) = pad else {
                                    unreachable!()
                                };
                                tile[j] = [v; 8];
                                continue;
                            }
                            let srow = &x[sy * src_w..sy * src_w + src_w];
                            if id_end > k0 {
                                tile[j][..id_end - k0].copy_from_slice(&srow[k0..id_end]);
                            }
                            for t in id_end..kend {
                                tile[j][t - k0] = match pad {
                                    Pad::Constant(v) => v,
                                    Pad::Replicate => srow[src_w - 1],
                                    Pad::Symmetric => srow[xmap[t]],
                                };
                            }
                        }
                        for t in 0..kend - k0 {
                            let lane = (k0 + t) * 8;
                            for j in 0..8 {
                                self.re8[lane + j] = tile[j][t];
                            }
                        }
                    }
                }
            } else {
                for j in 0..rws {
                    let d_row = y0 + j;
                    let sy = ymap[d_row];
                    if sy == usize::MAX {
                        let Pad::Constant(v) = pad else {
                            unreachable!()
                        };
                        for k in 0..w {
                            let d = match perm {
                                Some(r) => r[k] as usize,
                                None => k,
                            } * 8;
                            self.re8[d + j] = v;
                        }
                        continue;
                    }
                    let srow = &x[sy * src_w..sy * src_w + src_w];
                    for k in 0..w {
                        let d = match perm {
                            Some(r) => r[k] as usize,
                            None => k,
                        } * 8;
                        self.re8[d + j] = match xmap[k] {
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
            if perm.is_none() && rws == 8 {
                // Transposed scatter: contiguous 64 B lane reads, contiguous
                // row writes into pre/pim.
                for k0 in (0..w).step_by(8) {
                    let kn = (k0 + 8).min(w) - k0;
                    let mut tre = [[0.0f64; 8]; 8];
                    let mut tim = [[0.0f64; 8]; 8];
                    for t in 0..kn {
                        let lane = (k0 + t) * 8;
                        for j in 0..8 {
                            tre[j][t] = self.re8[lane + j];
                            tim[j][t] = self.im8[lane + j];
                        }
                    }
                    for j in 0..8 {
                        let row = (y0 + j) * stride + k0;
                        self.pre[row..row + kn].copy_from_slice(&tre[j][..kn]);
                        self.pim[row..row + kn].copy_from_slice(&tim[j][..kn]);
                    }
                }
            } else {
                for j in 0..rws {
                    let row = (y0 + j) * stride;
                    for k in 0..w {
                        self.pre[row + k] = self.re8[k * 8 + j];
                        self.pim[row + k] = self.im8[k * 8 + j];
                    }
                }
            }
            // The block covering `src_h` has now written the canonical pad
            // row — all later all-pad blocks can copy it.
            pad_row_ready |= dedup_pad && y0 <= src_h && src_h < y0 + rws;
        }
    }

    /// Filtered column round-trip: forward column FFT of `pre`/`pim`,
    /// multiply by the real filter, inverse (conj∘fwd∘conj·1/h) — all on
    /// the same 8-column tiles, so the transform output never leaves the
    /// lane planes. Writes only rows `< keep_rows` of `pre`/`pim` (the
    /// crop region the following row pass consumes). Every multiply and
    /// negation lands in the same order as the unfused gather/scatter
    /// sequence, so results are bit-identical.
    fn columns_filt_rt_p(&mut self, filt: &[f64], relaxed: bool, keep_rows: usize) {
        let (w, h) = (self.width, self.height);
        // `pre`/`pim` rows stride by `w + 8` (set-rotation padding); `filt`
        // is the caller's unpadded `w·h` plane — keep `y*w` for it.
        let stride = w + 8;
        let perm = self.columns.gather_perm();
        let sc = 1.0 / h as f64;
        #[cfg(feature = "parallel")]
        if w % 8 == 0 && w.div_ceil(8) >= 8 && rayon::current_num_threads() > 1 {
            self.columns_filt_rt_par(filt, relaxed, keep_rows);
            return;
        }
        for x0 in (0..w).step_by(8) {
            let cols = (w - x0).min(8);
            if cols == 8 {
                match perm {
                    None => {
                        // Contiguous dst / strided src — slice-adapted
                        // chunks keep bounds checks out of the loop.
                        for (y, (d8, d8i)) in self
                            .re8
                            .chunks_exact_mut(8)
                            .zip(self.im8.chunks_exact_mut(8))
                            .enumerate()
                            .take(h)
                        {
                            let s = y * stride + x0;
                            d8.copy_from_slice(&self.pre[s..s + 8]);
                            d8i.copy_from_slice(&self.pim[s..s + 8]);
                        }
                    }
                    Some(r) => {
                        for y in 0..h {
                            let (d, s) = (r[y] as usize * 8, y * stride + x0);
                            self.re8[d..d + 8].copy_from_slice(&self.pre[s..s + 8]);
                            self.im8[d..d + 8].copy_from_slice(&self.pim[s..s + 8]);
                        }
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
                // Filter + input-conjugate of the inverse, fused per lane:
                // `re·f`, `−(im·f)` — landed at the permuted input slots the
                // second transform reads (identity when `perm` is None).
                match perm {
                    None => {
                        for y in 0..h {
                            let f8: &[f64; 8] =
                                filt[y * w + x0..y * w + x0 + 8].try_into().unwrap();
                            let re: &mut [f64; 8] =
                                (&mut self.re8[y * 8..y * 8 + 8]).try_into().unwrap();
                            let im: &mut [f64; 8] =
                                (&mut self.im8[y * 8..y * 8 + 8]).try_into().unwrap();
                            for j in 0..8 {
                                re[j] *= f8[j];
                                im[j] = -(im[j] * f8[j]);
                            }
                        }
                    }
                    Some(r) => {
                        for y in 0..h {
                            let (d, s) = (r[y] as usize * 8, y * w + x0);
                            for j in 0..8 {
                                self.s2re8[d + j] = self.re8[y * 8 + j] * filt[s + j];
                                self.s2im8[d + j] = -(self.im8[y * 8 + j] * filt[s + j]);
                            }
                        }
                        let n = h * 8;
                        self.re8[..n].copy_from_slice(&self.s2re8[..n]);
                        self.im8[..n].copy_from_slice(&self.s2im8[..n]);
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
                for y in 0..keep_rows {
                    let (d, s) = (y * 8, y * stride + x0);
                    let re: &[f64; 8] = (&self.re8[d..d + 8]).try_into().unwrap();
                    let im: &[f64; 8] = (&self.im8[d..d + 8]).try_into().unwrap();
                    let pr: &mut [f64; 8] =
                        (&mut self.pre[s..s + 8]).try_into().unwrap();
                    let pi: &mut [f64; 8] =
                        (&mut self.pim[s..s + 8]).try_into().unwrap();
                    for j in 0..8 {
                        pr[j] = re[j] * sc;
                        pi[j] = -im[j] * sc;
                    }
                }
            } else {
                for y in 0..h {
                    let (d, s) = (
                        match perm {
                            Some(r) => r[y] as usize,
                            None => y,
                        } * 8,
                        y * stride + x0,
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
                match perm {
                    None => {
                        for y in 0..h {
                            let s = y * w + x0;
                            for j in 0..cols {
                                self.re8[y * 8 + j] *= filt[s + j];
                                self.im8[y * 8 + j] = -(self.im8[y * 8 + j] * filt[s + j]);
                            }
                        }
                    }
                    Some(r) => {
                        for y in 0..h {
                            let (d, s) = (r[y] as usize * 8, y * w + x0);
                            for j in 0..cols {
                                self.s2re8[d + j] = self.re8[y * 8 + j] * filt[s + j];
                                self.s2im8[d + j] = -(self.im8[y * 8 + j] * filt[s + j]);
                            }
                        }
                        let n = h * 8;
                        self.re8[..n].copy_from_slice(&self.s2re8[..n]);
                        self.im8[..n].copy_from_slice(&self.s2im8[..n]);
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
                for y in 0..keep_rows {
                    let (d, s) = (y * 8, y * stride + x0);
                    for j in 0..cols {
                        self.pre[s + j] = self.re8[d + j] * sc;
                        self.pim[s + j] = -self.im8[d + j] * sc;
                    }
                }
            }
        }
    }

    /// `parallel` fast path for [`Self::columns_filt_rt_p`]: column tiles
    /// are independent, so each chunk of tiles gathers, transforms,
    /// filters and inverse-transforms into a strip buffer; the
    /// `keep_rows` crop then scatters back into `pre`/`pim` by disjoint
    /// row-ranges. Identical per-element arithmetic — only loop
    /// distribution differs.
    #[cfg(feature = "parallel")]
    fn columns_filt_rt_par(&mut self, filt: &[f64], relaxed: bool, keep_rows: usize) {
        use rayon::prelude::*;
        let (w, h) = (self.width, self.height);
        let stride = w + 8;
        let h8 = h * 8;
        let sc = 1.0 / h as f64;
        let ntiles = w.div_ceil(8);
        let m8 = self.columns.work_len(h) * 8;
        let columns = &self.columns;
        let perm = columns.gather_perm();
        let pre = &self.pre;
        let pim = &self.pim;
        // One task per worker: each allocates its lane scratch once and
        // returns `(first_tile, strips)` with strips laid
        // `[tile][y][8]` for re and im.
        let nchunks = ntiles.min(rayon::current_num_threads());
        let chunk_len = ntiles.div_ceil(nchunks);
        let strips: Vec<(usize, Vec<f64>, Vec<f64>)> =
            crate::par::collect_indexed(nchunks, |c| {
                let t0 = c * chunk_len;
                let tn = ntiles.saturating_sub(t0).min(chunk_len);
                let mut re8 = vec![0.0; h8];
                let mut im8 = vec![0.0; h8];
                let mut sre8 = vec![0.0; m8];
                let mut sim8 = vec![0.0; m8];
                let mut s2re8 = vec![0.0; m8];
                let mut s2im8 = vec![0.0; m8];
                let mut sre = Vec::with_capacity(tn * keep_rows * 8);
                let mut sim = Vec::with_capacity(tn * keep_rows * 8);
                for t in t0..t0 + tn {
                    let x0 = t * 8;
                    let cols = (w - x0).min(8);
                    match perm {
                        None => {
                            for y in 0..h {
                                let s = y * stride + x0;
                                re8[y * 8..y * 8 + cols]
                                    .copy_from_slice(&pre[s..s + cols]);
                                im8[y * 8..y * 8 + cols]
                                    .copy_from_slice(&pim[s..s + cols]);
                            }
                        }
                        Some(r) => {
                            for y in 0..h {
                                let (d, s) = (r[y] as usize * 8, y * stride + x0);
                                re8[d..d + cols].copy_from_slice(&pre[s..s + cols]);
                                im8[d..d + cols].copy_from_slice(&pim[s..s + cols]);
                            }
                        }
                    }
                    columns.run8_mode(
                        relaxed,
                        &mut re8,
                        &mut im8,
                        &mut sre8,
                        &mut sim8,
                        &mut s2re8,
                        &mut s2im8,
                    );
                    // Filter + input-conjugate of the inverse; `perm`
                    // lands products at the input slots the second
                    // transform reads.
                    match perm {
                        None => {
                            for y in 0..h {
                                let s = y * w + x0;
                                for j in 0..cols {
                                    re8[y * 8 + j] *= filt[s + j];
                                    im8[y * 8 + j] = -(im8[y * 8 + j] * filt[s + j]);
                                }
                            }
                        }
                        Some(r) => {
                            for y in 0..h {
                                let (d, s) = (r[y] as usize * 8, y * w + x0);
                                for j in 0..cols {
                                    s2re8[d + j] = re8[y * 8 + j] * filt[s + j];
                                    s2im8[d + j] = -(im8[y * 8 + j] * filt[s + j]);
                                }
                            }
                            re8[..h8].copy_from_slice(&s2re8[..h8]);
                            im8[..h8].copy_from_slice(&s2im8[..h8]);
                        }
                    }
                    columns.run8_mode(
                        relaxed,
                        &mut re8,
                        &mut im8,
                        &mut sre8,
                        &mut sim8,
                        &mut s2re8,
                        &mut s2im8,
                    );
                    for y in 0..keep_rows {
                        let mut r8 = [0.0f64; 8];
                        let mut i8 = [0.0f64; 8];
                        for j in 0..cols {
                            r8[j] = re8[y * 8 + j] * sc;
                            i8[j] = -im8[y * 8 + j] * sc;
                        }
                        sre.extend_from_slice(&r8[..cols]);
                        sim.extend_from_slice(&i8[..cols]);
                    }
                }
                (t0, sre, sim)
            });
        // Scatter-back, parallel over `keep_rows` row-ranges: each task
        // copies its rows' tile-columns out of every chunk strip (strips
        // are read-only here; the row ranges are disjoint writes).
        let rb = 32usize;
        let pre_rows = &mut self.pre[..keep_rows * stride];
        let pim_rows = &mut self.pim[..keep_rows * stride];
        pre_rows
            .par_chunks_mut(stride * rb)
            .zip(pim_rows.par_chunks_mut(stride * rb))
            .enumerate()
            .for_each(|(ci, (pre_c, pim_c))| {
                let yb = ci * rb;
                let yn = pre_c.len() / stride;
                for (t0, sre, sim) in &strips {
                    let mut x0 = t0 * 8;
                    for (rstrip, istrip) in sre
                        .chunks_exact(keep_rows * 8)
                        .zip(sim.chunks_exact(keep_rows * 8))
                    {
                        for ly in 0..yn {
                            let s = ly * stride + x0;
                            let o = (yb + ly) * 8;
                            pre_c[s..s + 8].copy_from_slice(&rstrip[o..o + 8]);
                            pim_c[s..s + 8].copy_from_slice(&istrip[o..o + 8]);
                        }
                        x0 += 8;
                    }
                }
            });
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

/// Shared `(width, height) → Plan2` pool. A plan carries twiddle tables
/// plus ~5·w·h of batch/scratch lanes, so rebuilding per call is a real
/// fixed cost in `conv_fft_pad`. A per-thread cache would hoard a full
/// shape set on every rayon worker under `parallel`; the pool instead
/// retains what the workload actually needs concurrently — one plan per
/// in-flight call per shape. Plans are a pure function of the dims —
/// reuse is bit-identical (`reused_2d_plans_*` tests).
static PLAN2_POOL: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<(usize, usize), Vec<Plan2>>>,
> = std::sync::LazyLock::new(|| {
    std::sync::Mutex::new(std::collections::HashMap::new())
});

/// Run `f` on a pooled plan for `width × height`, constructing it on
/// first use.
fn with_plan2<R>(width: usize, height: usize, f: impl FnOnce(&mut Plan2) -> R) -> R {
    let mut plan = {
        let mut g = PLAN2_POOL.lock().unwrap_or_else(|e| e.into_inner());
        g.get_mut(&(width, height))
            .and_then(Vec::pop)
            .unwrap_or_else(|| Plan2::new(width, height))
    };
    let r = f(&mut plan);
    {
        let mut g = PLAN2_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let v = g.entry((width, height)).or_default();
        // Retain enough instances to cover steady-state concurrency —
        // too few and tasks thrash `Plan2::new` (~70 MB + twiddle build).
        let keep = if cfg!(feature = "parallel") { 32 } else { 4 };
        if v.len() < keep {
            v.push(plan);
        }
    }
    r
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
    let mut out = vec![0.0; width * height];
    conv_fft_pad_into(
        x,
        width,
        height,
        filter,
        pad_w,
        pad_h,
        pad,
        relaxed,
        &mut out,
        1,
        0,
        None,
    );
    out
}

/// [`conv_fft_pad`] writing into a caller slice: element `(y,x)` lands at
/// `out[(y·width + x)·xstride + xoff]` with `clamp` applied after the
/// `1/w` scale — the caller's post-pass fused in, bit-identical.
#[allow(clippy::too_many_arguments)]
pub(super) fn conv_fft_pad_into(
    x: &[f64],
    width: usize,
    height: usize,
    filter: &[f64],
    pad_w: usize,
    pad_h: usize,
    pad: Pad,
    relaxed: bool,
    out: &mut [f64],
    xstride: usize,
    xoff: usize,
    clamp: Option<(f64, f64)>,
) {
    // Fully fused: rows gather straight from the unpadded `x` (imag lanes
    // are constants), the column round-trip keeps its transform output in
    // the lane planes while folding in the real filter multiply, and the
    // inverse row pass scatters only the kept crop straight into `out`.
    with_plan2(pad_w, pad_h, |plan| {
        plan.rows_fwd_p_pad(x, width, height, pad, relaxed);
        plan.columns_filt_rt_p(filter, relaxed, height);
        plan.rows_inv_out(out, relaxed, width, height, xstride, xoff, clamp);
    });
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
