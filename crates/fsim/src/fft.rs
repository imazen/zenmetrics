//! Fourier-domain transforms: a self-contained complex FFT with 2-D
//! forward/inverse entry points and reusable per-length plans.
//!
//! ## Provenance
//!
//! Vendored verbatim (minus HDR-VDP-specific helpers) from
//! `crates/hdrvdp/src/fft.rs` — itself an independent implementation of the
//! standard DFT (the upstream `create_cycdeg_image.m` / `fast_conv_fft.m`
//! carry no permission grant; none of that code is used anywhere in this
//! file). Kept in step manually with the hdrvdp copy.
//!
//! ### Precision
//!
//! The transforms run in `f32` — plane data through the whole pipeline is
//! `f32` while reductions and one-time table/grid construction stay `f64`.
//! Twiddles are still computed in `f64` (`expi`) and cast down, so the error
//! vs an `f64` transform is plane-quantisation only.

use alloc::vec;
use alloc::vec::Vec;
use core::f64::consts::PI;

/// Minimal complex number — enough for the transforms here, and it keeps the
/// crate dependency-free.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct Complex {
    /// Real part.
    pub re: f32,
    /// Imaginary part.
    pub im: f32,
}

impl Complex {
    /// `re + i·im`.
    #[must_use]
    #[inline]
    pub(crate) const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// `e^{iθ}` — the angle is evaluated in `f64` and cast down, so stored
    /// twiddles are the correctly-rounded `f32` values.
    #[must_use]
    #[inline]
    pub(crate) fn expi(theta: f64) -> Self {
        let (s, c) = theta.sin_cos();
        Self {
            re: c as f32,
            im: s as f32,
        }
    }

    /// Complex conjugate.
    #[must_use]
    #[inline]
    pub(crate) fn conj(self) -> Self {
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
    fn scale(self, s: f32) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }
}

/// Forward DFT of `buf`, in place. Any length; `O(n log n)`.
#[allow(dead_code)] // public mirror of the vendored API; callers use the _planned forms
pub(crate) fn fft(buf: &mut [Complex]) {
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
/// `fft2` reuses one plan across all rows and one across all columns, and
/// [`conv_fft_real`] across its forward and inverse passes, which is where the
/// win concentrates: the direct form recomputed `sin_cos` for every butterfly
/// of every row.
pub(crate) struct Plan {
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
    pub(crate) fn new(n: usize) -> Self {
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
                let s = 1.0 / *m as f32;
                for v in a.iter_mut() {
                    *v = v.conj().scale(s);
                }
                for (k, out) in buf.iter_mut().enumerate() {
                    *out = a[k].mul(chirp[k]);
                }
            }
        }
    }

    /// `f32` scratch length (per side) [`run_batch8`] needs: `8·m` for
    /// Bluestein plans, `0` otherwise.
    fn batch_scratch_len(&self) -> usize {
        match &self.kind {
            PlanKind::Bluestein { m, .. } => 8 * m,
            _ => 0,
        }
    }

    /// Batched [`run`]: `re8`/`im8` hold eight independent length-`n`
    /// transforms in the SoA batch layout (element `k` of lane `j` at
    /// `[k·8 + j]`, matching [`fft_batch8`]). Every step reproduces
    /// [`run`]'s per-lane complex op sequence, so each lane's result
    /// is bit-identical. `are`/`aim` are `batch_scratch_len()` scratch.
    fn run_batch8(&self, re8: &mut [f32], im8: &mut [f32], are: &mut [f32], aim: &mut [f32]) {
        debug_assert_eq!(re8.len(), self.n * 8);
        debug_assert_eq!(im8.len(), self.n * 8);
        match &self.kind {
            PlanKind::Trivial => {}
            PlanKind::Radix2 { stages } => fft_batch8(re8, im8, self.n, stages),
            PlanKind::Bluestein {
                chirp,
                bf,
                m,
                mplan,
            } => bluestein_batch8(re8, im8, self.n, chirp, bf, *m, mplan, are, aim),
        }
    }
}

/// Batched Bluestein chirp-z, the SoA twin of the Bluestein arm in
/// [`Plan::run`]: `a = buf·chirp` → FFT(m) → `a·bf` → conj → FFT(m)
/// → conj·(1/m) → `out = a·chirp`. Each lane's op sequence is
/// identical to the scalar path — `x·y` expands to
/// `re·re − im·im`/`re·im + im·re` either way — so results are
/// bit-identical. `fft_batch8` itself re-dispatches to the same tier.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn bluestein_batch8_inner(
    token: Token,
    re8: &mut [f32],
    im8: &mut [f32],
    n: usize,
    chirp: &[Complex],
    bf: &[Complex],
    m: usize,
    mplan: &[Vec<Complex>],
    are: &mut [f32],
    aim: &mut [f32],
) {
    debug_assert_eq!(re8.len(), n * 8);
    debug_assert_eq!(im8.len(), n * 8);
    debug_assert_eq!(are.len(), m * 8);
    debug_assert_eq!(aim.len(), m * 8);
    let zero = f32x8::splat_t(token, 0.0);

    // a[k] = buf[k]·chirp[k] for k < n, zero elsewhere.
    for k in 0..n {
        let cr = f32x8::splat_t(token, chirp[k].re);
        let ci = f32x8::splat_t(token, chirp[k].im);
        let br = f32x8::load_t(token, (&re8[k * 8..k * 8 + 8]).try_into().unwrap());
        let bi = f32x8::load_t(token, (&im8[k * 8..k * 8 + 8]).try_into().unwrap());
        (br * cr - bi * ci).store((&mut are[k * 8..k * 8 + 8]).try_into().unwrap());
        (br * ci + bi * cr).store((&mut aim[k * 8..k * 8 + 8]).try_into().unwrap());
    }
    for k in n..m {
        zero.store((&mut are[k * 8..k * 8 + 8]).try_into().unwrap());
        zero.store((&mut aim[k * 8..k * 8 + 8]).try_into().unwrap());
    }

    fft_batch8(are, aim, m, mplan);

    // a[k] = a[k]·bf[k], then a[k] = conj(a[k]) — one fused sweep.
    for k in 0..m {
        let br = f32x8::splat_t(token, bf[k].re);
        let bi = f32x8::splat_t(token, bf[k].im);
        let ar = f32x8::load_t(token, (&are[k * 8..k * 8 + 8]).try_into().unwrap());
        let ai = f32x8::load_t(token, (&aim[k * 8..k * 8 + 8]).try_into().unwrap());
        (ar * br - ai * bi).store((&mut are[k * 8..k * 8 + 8]).try_into().unwrap());
        (zero - (ar * bi + ai * br)).store((&mut aim[k * 8..k * 8 + 8]).try_into().unwrap());
    }

    fft_batch8(are, aim, m, mplan);

    // out[k] = (conj(a[k])·(1/m))·chirp[k] for k < n — fused into one
    // sweep. Elements k ≥ n are never read after this point, so their
    // conj·scale is dead work the scalar path performed anyway; the
    // per-lane expression `im·(−s)` ≡ `(−im)·s` bitwise.
    let s = f32x8::splat_t(token, 1.0 / m as f32);
    let negs = f32x8::splat_t(token, -1.0 / m as f32);
    for k in 0..n {
        let cr = f32x8::splat_t(token, chirp[k].re);
        let ci = f32x8::splat_t(token, chirp[k].im);
        let ar = f32x8::load_t(token, (&are[k * 8..k * 8 + 8]).try_into().unwrap());
        let ai = f32x8::load_t(token, (&aim[k * 8..k * 8 + 8]).try_into().unwrap());
        let ore = ar * s;
        let oim = ai * negs;
        (ore * cr - oim * ci).store((&mut re8[k * 8..k * 8 + 8]).try_into().unwrap());
        (ore * ci + oim * cr).store((&mut im8[k * 8..k * 8 + 8]).try_into().unwrap());
    }
}

/// [`bluestein_batch8_inner`] with runtime tier dispatch.
#[allow(clippy::too_many_arguments)]
fn bluestein_batch8(
    re8: &mut [f32],
    im8: &mut [f32],
    n: usize,
    chirp: &[Complex],
    bf: &[Complex],
    m: usize,
    mplan: &[Vec<Complex>],
    are: &mut [f32],
    aim: &mut [f32],
) {
    archmage::attuned!(
        bluestein_batch8_inner(re8, im8, n, chirp, bf, m, mplan, are, aim),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

/// Batched radix-2 FFT: **eight** independent length-`n` transforms at once,
/// held as struct-of-arrays planes `re8`/`im8` of length `n·8` where element
/// `k` of transform `j` lives at `[k·8 + j]`.
///
/// Every butterfly is then a full `f32x8` lane operation — the batch lane is
/// the vector lane — so no shuffles are needed anywhere (an AoS layout would
/// need pair-swaps magetypes doesn't expose), and *every* stage vectorizes
/// including the small ones that would be strided in a per-row SIMD scheme.
/// `fft2` uses this for the column pass (8 columns per block, a layout the
/// old blocked transpose already produced) and for the row pass (8 rows per
/// block, transposed in and out). Bit-reversal swaps whole 8-lane rows.
///
/// Twiddles are splat per butterfly — cheap, and identical across lanes.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn fft_batch8_inner(
    token: Token,
    re8: &mut [f32],
    im8: &mut [f32],
    n: usize,
    stages: &[Vec<Complex>],
) {
    debug_assert_eq!(re8.len(), n * 8);
    debug_assert_eq!(im8.len(), n * 8);

    // Bit-reversal permutation on 8-lane blocks.
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = (i as u32).reverse_bits() >> (32 - bits);
        let j = j as usize;
        if j > i {
            let a = f32x8::load_t(token, (&re8[i * 8..i * 8 + 8]).try_into().unwrap());
            let b = f32x8::load_t(token, (&re8[j * 8..j * 8 + 8]).try_into().unwrap());
            b.store((&mut re8[i * 8..i * 8 + 8]).try_into().unwrap());
            a.store((&mut re8[j * 8..j * 8 + 8]).try_into().unwrap());
            let a = f32x8::load_t(token, (&im8[i * 8..i * 8 + 8]).try_into().unwrap());
            let b = f32x8::load_t(token, (&im8[j * 8..j * 8 + 8]).try_into().unwrap());
            b.store((&mut im8[i * 8..i * 8 + 8]).try_into().unwrap());
            a.store((&mut im8[j * 8..j * 8 + 8]).try_into().unwrap());
        }
    }

    // Stages run in fused pairs: the two butterflies of stage `s`
    // inside one stage-`s+1` span and the two `s+1` butterflies that
    // combine them are all computed in registers before storing. The
    // per-element op sequence is identical to the unfused sweep —
    // every butterfly still applies `u±v·w` in the same order — so
    // results are bit-identical; the win is halved sweep traffic.
    let mut it = stages.iter();
    loop {
        let (tw_a, tw_b) = match (it.next(), it.next()) {
            (Some(a), Some(b)) => (a, b),
            (Some(a), None) => {
                // Odd trailing stage: plain sweep.
                let half = a.len();
                let len = half * 2;
                for start in (0..n).step_by(len) {
                    for (k, &w) in a.iter().enumerate() {
                        let aa = (start + k) * 8;
                        let bb = (start + k + half) * 8;
                        let wre = f32x8::splat_t(token, w.re);
                        let wim = f32x8::splat_t(token, w.im);
                        let ure = f32x8::load_t(token, (&re8[aa..aa + 8]).try_into().unwrap());
                        let uim = f32x8::load_t(token, (&im8[aa..aa + 8]).try_into().unwrap());
                        let vre = f32x8::load_t(token, (&re8[bb..bb + 8]).try_into().unwrap());
                        let vim = f32x8::load_t(token, (&im8[bb..bb + 8]).try_into().unwrap());
                        let tre = vre * wre - vim * wim;
                        let tim = vre * wim + vim * wre;
                        (ure + tre).store((&mut re8[aa..aa + 8]).try_into().unwrap());
                        (uim + tim).store((&mut im8[aa..aa + 8]).try_into().unwrap());
                        (ure - tre).store((&mut re8[bb..bb + 8]).try_into().unwrap());
                        (uim - tim).store((&mut im8[bb..bb + 8]).try_into().unwrap());
                    }
                }
                break;
            }
            (None, _) => break,
        };
        let half1 = tw_a.len();
        let half2 = tw_b.len();
        debug_assert_eq!(half2, half1 * 2);
        let len2 = half2 * 2;
        for start in (0..n).step_by(len2) {
            for k in 0..half1 {
                // Stage-s butterflies at (start+k, +half1) and
                // (start+k+half2, +half1) share twiddle tw_a[k]; the
                // stage-s+1 butterflies then join them: upper pair uses
                // tw_b[k], lower pair tw_b[k+half1].
                let a0 = (start + k) * 8;
                let b0 = (start + k + half1) * 8;
                let a1 = (start + k + half2) * 8;
                let b1 = (start + k + half2 + half1) * 8;
                let wa = tw_a[k];
                let ware = f32x8::splat_t(token, wa.re);
                let waim = f32x8::splat_t(token, wa.im);
                let u0re = f32x8::load_t(token, (&re8[a0..a0 + 8]).try_into().unwrap());
                let u0im = f32x8::load_t(token, (&im8[a0..a0 + 8]).try_into().unwrap());
                let v0re = f32x8::load_t(token, (&re8[b0..b0 + 8]).try_into().unwrap());
                let v0im = f32x8::load_t(token, (&im8[b0..b0 + 8]).try_into().unwrap());
                let u1re = f32x8::load_t(token, (&re8[a1..a1 + 8]).try_into().unwrap());
                let u1im = f32x8::load_t(token, (&im8[a1..a1 + 8]).try_into().unwrap());
                let v1re = f32x8::load_t(token, (&re8[b1..b1 + 8]).try_into().unwrap());
                let v1im = f32x8::load_t(token, (&im8[b1..b1 + 8]).try_into().unwrap());
                // stage s, upper block: t0 = v0·wa
                let t0re = v0re * ware - v0im * waim;
                let t0im = v0re * waim + v0im * ware;
                let p0re = u0re + t0re;
                let p0im = u0im + t0im;
                let q0re = u0re - t0re;
                let q0im = u0im - t0im;
                // stage s, lower block: t1 = v1·wa
                let t1re = v1re * ware - v1im * waim;
                let t1im = v1re * waim + v1im * ware;
                let p1re = u1re + t1re;
                let p1im = u1im + t1im;
                let q1re = u1re - t1re;
                let q1im = u1im - t1im;
                // stage s+1, upper butterfly: v = p1·tw_b[k]
                let wb = tw_b[k];
                let wre = f32x8::splat_t(token, wb.re);
                let wim = f32x8::splat_t(token, wb.im);
                let tre = p1re * wre - p1im * wim;
                let tim = p1re * wim + p1im * wre;
                (p0re + tre).store((&mut re8[a0..a0 + 8]).try_into().unwrap());
                (p0im + tim).store((&mut im8[a0..a0 + 8]).try_into().unwrap());
                (p0re - tre).store((&mut re8[a1..a1 + 8]).try_into().unwrap());
                (p0im - tim).store((&mut im8[a1..a1 + 8]).try_into().unwrap());
                // stage s+1, lower butterfly: v = q1·tw_b[k+half1]
                let wb2 = tw_b[k + half1];
                let wre = f32x8::splat_t(token, wb2.re);
                let wim = f32x8::splat_t(token, wb2.im);
                let tre = q1re * wre - q1im * wim;
                let tim = q1re * wim + q1im * wre;
                (q0re + tre).store((&mut re8[b0..b0 + 8]).try_into().unwrap());
                (q0im + tim).store((&mut im8[b0..b0 + 8]).try_into().unwrap());
                (q0re - tre).store((&mut re8[b1..b1 + 8]).try_into().unwrap());
                (q0im - tim).store((&mut im8[b1..b1 + 8]).try_into().unwrap());
            }
        }
    }
}

/// [`fft_batch8_inner`] with runtime tier dispatch.
fn fft_batch8(re8: &mut [f32], im8: &mut [f32], n: usize, stages: &[Vec<Complex>]) {
    archmage::attuned!(
        fft_batch8_inner(re8, im8, n, stages),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
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

/// Inverse DFT of `buf`, in place, normalised by `1/n`.
#[allow(dead_code)] // public mirror of the vendored API; callers use the _planned forms
pub(crate) fn ifft(buf: &mut [Complex]) {
    let n = buf.len();
    if n == 0 {
        return;
    }
    for v in buf.iter_mut() {
        *v = v.conj();
    }
    fft(buf);
    let s = 1.0 / n as f32;
    for v in buf.iter_mut() {
        *v = v.conj().scale(s);
    }
}

/// Forward 2D DFT of a row-major `height × width` buffer, in place.
#[allow(dead_code)] // public mirror of the vendored API; callers use the _planned forms
pub(crate) fn fft2(buf: &mut [Complex], width: usize, height: usize) {
    let (wplan, hplan) = (Plan::new(width), Plan::new(height));
    fft2_planned(buf, width, height, &wplan, &hplan);
}

/// [`fft2`] with the two 1-D plans supplied, so callers doing several
/// same-size transforms (or a forward + inverse pair) build them once.
pub(crate) fn fft2_planned(
    buf: &mut [Complex],
    width: usize,
    height: usize,
    wplan: &Plan,
    hplan: &Plan,
) {
    debug_assert_eq!(buf.len(), width * height);
    fft2_rows_planned(buf, width, height, wplan);
    fft2_cols_planned(buf, width, height, hplan);
}

/// The row half of [`fft2_planned`]: batched eight rows per block through
/// [`fft_batch8`] when the row plan is radix-2 (each 8-row block is
/// transposed into the SoA batch layout, transformed, and written back —
/// the block fits L2, so the extra pass is in-cache). Non-power-of-two
/// plans keep the scalar per-row path (Bluestein sizes are rare and small).
fn fft2_rows_planned(buf: &mut [Complex], width: usize, height: usize, wplan: &Plan) {
    fft2_rows_conj_mul(buf, width, height, wplan, None, false);
}

/// [`fft2_rows_planned`] with an optional `conj` and a real filter multiply
/// fused into the transpose **in**: with `conj` set it loads
/// `(re·f, −im·f)` (with `f = 1` when `filter` is `None`), i.e.
/// `conj(buf·f)` — exactly what an `ifft2` whose input was just
/// pointwise-filtered needs, without a separate pass over the buffer.
fn fft2_rows_conj_mul(
    buf: &mut [Complex],
    width: usize,
    height: usize,
    wplan: &Plan,
    filter: Option<&[f32]>,
    conj: bool,
) {
    if let PlanKind::Radix2 { stages } = &wplan.kind {
        const RB: usize = 8;
        // Row blocks are disjoint slices of `buf`, so under `parallel` each
        // block's transpose+transform+writeback is an independent task.
        #[cfg(feature = "parallel")]
        if height >= 4 * RB && rayon::current_num_threads() > 1 {
            use rayon::prelude::*;
            buf.par_chunks_mut(width * RB)
                .enumerate()
                .map_init(
                    || (vec![0.0f32; width * RB], vec![0.0f32; width * RB]),
                    |(re8, im8), (ci, chunk)| {
                        let nb = chunk.len() / width;
                        for x in 0..width {
                            for j in 0..nb {
                                let i = j * width + x;
                                let v = chunk[i];
                                let f = filter.map_or(1.0, |f| f[(ci * RB + j) * width + x]);
                                let im = if conj { -v.im } else { v.im };
                                re8[x * RB + j] = v.re * f;
                                im8[x * RB + j] = im * f;
                            }
                            for j in nb..RB {
                                re8[x * RB + j] = 0.0;
                                im8[x * RB + j] = 0.0;
                            }
                        }
                        fft_batch8(re8, im8, width, stages);
                        for x in 0..width {
                            for j in 0..nb {
                                chunk[j * width + x] =
                                    Complex::new(re8[x * RB + j], im8[x * RB + j]);
                            }
                        }
                    },
                )
                .for_each(|()| ());
            return;
        }
        let mut re8 = vec![0.0f32; width * RB];
        let mut im8 = vec![0.0f32; width * RB];
        let mut y = 0usize;
        while y < height {
            let nb = RB.min(height - y);
            for x in 0..width {
                for j in 0..nb {
                    let i = (y + j) * width + x;
                    let v = buf[i];
                    let f = filter.map_or(1.0, |f| f[i]);
                    let im = if conj { -v.im } else { v.im };
                    re8[x * RB + j] = v.re * f;
                    im8[x * RB + j] = im * f;
                }
                for j in nb..RB {
                    re8[x * RB + j] = 0.0;
                    im8[x * RB + j] = 0.0;
                }
            }
            fft_batch8(&mut re8, &mut im8, width, stages);
            for x in 0..width {
                for j in 0..nb {
                    buf[(y + j) * width + x] = Complex::new(re8[x * RB + j], im8[x * RB + j]);
                }
            }
            y += nb;
        }
    } else {
        // Non-radix-2 (Bluestein) plans: batch eight rows through the
        // SoA `run_batch8` — same per-lane op sequence as the scalar
        // `Plan::run`, so results are bit-identical. `conj`/`filter`
        // fold into the transpose-in exactly like the radix-2 arm.
        const RB: usize = 8;
        #[cfg(feature = "parallel")]
        if height >= 4 * RB && rayon::current_num_threads() > 1 {
            use rayon::prelude::*;
            let scratch_len = wplan.batch_scratch_len();
            buf.par_chunks_mut(width * RB)
                .enumerate()
                .map_init(
                    || {
                        (
                            vec![0.0f32; width * RB],
                            vec![0.0f32; width * RB],
                            vec![0.0f32; scratch_len],
                            vec![0.0f32; scratch_len],
                        )
                    },
                    |(re8, im8, are, aim), (ci, chunk)| {
                        let nb = chunk.len() / width;
                        for x in 0..width {
                            for j in 0..nb {
                                let i = j * width + x;
                                let v = chunk[i];
                                let f = filter.map_or(1.0, |f| f[(ci * RB + j) * width + x]);
                                let im = if conj { -v.im } else { v.im };
                                re8[x * RB + j] = v.re * f;
                                im8[x * RB + j] = im * f;
                            }
                            for j in nb..RB {
                                re8[x * RB + j] = 0.0;
                                im8[x * RB + j] = 0.0;
                            }
                        }
                        wplan.run_batch8(re8, im8, are, aim);
                        for x in 0..width {
                            for j in 0..nb {
                                chunk[j * width + x] =
                                    Complex::new(re8[x * RB + j], im8[x * RB + j]);
                            }
                        }
                    },
                )
                .for_each(|()| ());
            return;
        }
        let scratch_len = wplan.batch_scratch_len();
        let mut re8 = vec![0.0f32; width * RB];
        let mut im8 = vec![0.0f32; width * RB];
        let mut are = vec![0.0f32; scratch_len];
        let mut aim = vec![0.0f32; scratch_len];
        let mut y = 0usize;
        while y < height {
            let nb = RB.min(height - y);
            for x in 0..width {
                for j in 0..nb {
                    let i = (y + j) * width + x;
                    let v = buf[i];
                    let f = filter.map_or(1.0, |f| f[i]);
                    let im = if conj { -v.im } else { v.im };
                    re8[x * RB + j] = v.re * f;
                    im8[x * RB + j] = im * f;
                }
                for j in nb..RB {
                    re8[x * RB + j] = 0.0;
                    im8[x * RB + j] = 0.0;
                }
            }
            wplan.run_batch8(&mut re8, &mut im8, &mut are, &mut aim);
            for x in 0..width {
                for j in 0..nb {
                    buf[(y + j) * width + x] = Complex::new(re8[x * RB + j], im8[x * RB + j]);
                }
            }
            y += nb;
        }
    }
}

/// The column half of [`fft2_planned`], split out so [`conv_fft_real`] can run
/// a smarter row pass first.
///
/// Radix-2 plans go through [`fft_batch8`] on the SoA batch layout — the
/// transpose writes column `j` into lane `j` (`re8[y·8+j]`/`im8[y·8+j]`), so
/// the whole transform is lane ops on contiguous vectors. Non-power-of-two
/// plans keep the blocked scalar path (8 adjacent columns per block = one
/// cache line per touched row, as before).
fn fft2_cols_planned(buf: &mut [Complex], width: usize, height: usize, hplan: &Plan) {
    fft2_cols_conj_scale(buf, width, height, hplan, None);
}

/// [`fft2_cols_planned`] with `conj·scale` fused into the transpose **out**:
/// with `scale` set it stores `(re·s, −im·s)`, which is `ifft2`'s trailing
/// `conj·(1/n)` — fused so it doesn't take its own pass over the buffer.
fn fft2_cols_conj_scale(
    buf: &mut [Complex],
    width: usize,
    height: usize,
    hplan: &Plan,
    scale: Option<f32>,
) {
    const CB: usize = 8;
    if let PlanKind::Radix2 { stages } = &hplan.kind {
        // Column stripes are disjoint writes but strided — not sliceable —
        // so the parallel path goes through a block-major SoA buffer in
        // bounded super-blocks (64 blocks = 512 columns ≈ 16 MB at 2048²):
        // gather (parallel over (block,row) cells), transform (parallel over
        // blocks), scatter (parallel over rows, fusing `conj·scale`).
        #[cfg(feature = "parallel")]
        if width >= 4 * CB && height >= 4 * CB && rayon::current_num_threads() > 1 {
            use rayon::prelude::*;
            const SB: usize = 64; // blocks per super-block
            let mut x0 = 0usize;
            while x0 < width {
                let nb_cols = (SB * CB).min(width - x0);
                let nblocks = nb_cols.div_ceil(CB);
                let mut re_all = vec![0.0f32; nblocks * height * CB];
                let mut im_all = vec![0.0f32; nblocks * height * CB];
                {
                    let ro: &[Complex] = buf;
                    re_all
                        .par_chunks_mut(CB)
                        .zip(im_all.par_chunks_mut(CB))
                        .enumerate()
                        .for_each(|(j, (rc, ic))| {
                            let (b, y) = (j / height, j % height);
                            for (l, (rc, ic)) in rc.iter_mut().zip(ic.iter_mut()).enumerate() {
                                let x = x0 + b * CB + l;
                                if x < width {
                                    let v = ro[y * width + x];
                                    *rc = v.re;
                                    *ic = v.im;
                                }
                            }
                        });
                }
                re_all
                    .par_chunks_mut(height * CB)
                    .zip(im_all.par_chunks_mut(height * CB))
                    .for_each(|(rc, ic)| fft_batch8(rc, ic, height, stages));
                let s = scale.unwrap_or(1.0);
                let negate = scale.is_some();
                buf.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
                    for b in 0..nblocks {
                        let base = (b * height + y) * CB;
                        for l in 0..CB {
                            let x = x0 + b * CB + l;
                            if x < width {
                                let (re, im) = (re_all[base + l], im_all[base + l]);
                                row[x] =
                                    Complex::new(re * s, if negate { -im * s } else { im * s });
                            }
                        }
                    }
                });
                x0 += nb_cols;
            }
            return;
        }
        let mut re8 = vec![0.0f32; height * CB];
        let mut im8 = vec![0.0f32; height * CB];
        let mut x = 0usize;
        while x < width {
            let nb = CB.min(width - x);
            for y in 0..height {
                for j in 0..nb {
                    let v = buf[y * width + x + j];
                    re8[y * CB + j] = v.re;
                    im8[y * CB + j] = v.im;
                }
                for j in nb..CB {
                    re8[y * CB + j] = 0.0;
                    im8[y * CB + j] = 0.0;
                }
            }
            fft_batch8(&mut re8, &mut im8, height, stages);
            match scale {
                Some(s) => {
                    for y in 0..height {
                        for j in 0..nb {
                            buf[y * width + x + j] =
                                Complex::new(re8[y * CB + j] * s, -im8[y * CB + j] * s);
                        }
                    }
                }
                None => {
                    for y in 0..height {
                        for j in 0..nb {
                            buf[y * width + x + j] = Complex::new(re8[y * CB + j], im8[y * CB + j]);
                        }
                    }
                }
            }
            x += nb;
        }
        return;
    }

    // Non-radix-2 (Bluestein) plans: batch eight columns through the
    // SoA `run_batch8` — same per-lane op sequence as the scalar
    // `Plan::run`, so results are bit-identical. `conj·scale` folds
    // into the scatter, same as the radix-2 arm.
    #[cfg(feature = "parallel")]
    if width >= 4 * CB && height >= 4 * CB && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        const SB: usize = 64; // blocks per super-block
        let scratch_len = hplan.batch_scratch_len();
        let mut x0 = 0usize;
        while x0 < width {
            let nb_cols = (SB * CB).min(width - x0);
            let nblocks = nb_cols.div_ceil(CB);
            let mut re_all = vec![0.0f32; nblocks * height * CB];
            let mut im_all = vec![0.0f32; nblocks * height * CB];
            {
                let ro: &[Complex] = buf;
                re_all
                    .par_chunks_mut(CB)
                    .zip(im_all.par_chunks_mut(CB))
                    .enumerate()
                    .for_each(|(j, (rc, ic))| {
                        let (b, y) = (j / height, j % height);
                        for (l, (rc, ic)) in rc.iter_mut().zip(ic.iter_mut()).enumerate() {
                            let x = x0 + b * CB + l;
                            if x < width {
                                let v = ro[y * width + x];
                                *rc = v.re;
                                *ic = v.im;
                            }
                        }
                    });
            }
            re_all
                .par_chunks_mut(height * CB)
                .zip(im_all.par_chunks_mut(height * CB))
                .map_init(
                    || (vec![0.0f32; scratch_len], vec![0.0f32; scratch_len]),
                    |(are, aim), (rc, ic)| hplan.run_batch8(rc, ic, are, aim),
                )
                .for_each(|()| ());
            let s = scale.unwrap_or(1.0);
            let negate = scale.is_some();
            buf.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
                for b in 0..nblocks {
                    let base = (b * height + y) * CB;
                    for l in 0..CB {
                        let x = x0 + b * CB + l;
                        if x < width {
                            let (re, im) = (re_all[base + l], im_all[base + l]);
                            row[x] = Complex::new(re * s, if negate { -im * s } else { im * s });
                        }
                    }
                }
            });
            x0 += nb_cols;
        }
        return;
    }
    let scratch_len = hplan.batch_scratch_len();
    let mut re8 = vec![0.0f32; height * CB];
    let mut im8 = vec![0.0f32; height * CB];
    let mut are = vec![0.0f32; scratch_len];
    let mut aim = vec![0.0f32; scratch_len];
    let mut x = 0usize;
    while x < width {
        let nb = CB.min(width - x);
        for y in 0..height {
            for j in 0..nb {
                let v = buf[y * width + x + j];
                re8[y * CB + j] = v.re;
                im8[y * CB + j] = v.im;
            }
            for j in nb..CB {
                re8[y * CB + j] = 0.0;
                im8[y * CB + j] = 0.0;
            }
        }
        hplan.run_batch8(&mut re8, &mut im8, &mut are, &mut aim);
        match scale {
            Some(s) => {
                for y in 0..height {
                    for j in 0..nb {
                        buf[y * width + x + j] =
                            Complex::new(re8[y * CB + j] * s, -im8[y * CB + j] * s);
                    }
                }
            }
            None => {
                for y in 0..height {
                    for j in 0..nb {
                        buf[y * width + x + j] = Complex::new(re8[y * CB + j], im8[y * CB + j]);
                    }
                }
            }
        }
        x += nb;
    }
}

/// Inverse 2D DFT (normalised by `1/(width·height)`), in place.
#[allow(dead_code)] // public mirror of the vendored API; callers use the _planned forms
pub(crate) fn ifft2(buf: &mut [Complex], width: usize, height: usize) {
    let (wplan, hplan) = (Plan::new(width), Plan::new(height));
    ifft2_planned(buf, width, height, &wplan, &hplan);
}

/// [`ifft2`] with the plans supplied — same conjugate–forward–conjugate
/// construction, same normalisation. The leading `conj` rides the row
/// transpose-in and `conj·(1/n)` rides the column transpose-out, so the
/// radix-2 path never touches the buffer outside a transform pass.
pub(crate) fn ifft2_planned(
    buf: &mut [Complex],
    width: usize,
    height: usize,
    wplan: &Plan,
    hplan: &Plan,
) {
    debug_assert_eq!(buf.len(), width * height);
    fft2_rows_conj_mul(buf, width, height, wplan, None, true);
    fft2_cols_conj_scale(
        buf,
        width,
        height,
        hplan,
        Some(1.0 / (width * height) as f32),
    );
}
#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dft(x: &[Complex]) -> Vec<Complex> {
        let n = x.len();
        (0..n)
            .map(|k| {
                let mut acc_re = 0.0f64;
                let mut acc_im = 0.0f64;
                for (j, v) in x.iter().enumerate() {
                    let ang = -2.0 * PI * (k * j) as f64 / n as f64;
                    let (s, c) = ang.sin_cos();
                    acc_re += v.re as f64 * c - v.im as f64 * s;
                    acc_im += v.re as f64 * s + v.im as f64 * c;
                }
                Complex::new(acc_re as f32, acc_im as f32)
            })
            .collect()
    }

    fn seeded(n: usize) -> Vec<Complex> {
        seeded_with(0x2545_F491_4F6C_DD1Du64, n)
    }

    fn seeded_with(seed: u64, n: usize) -> Vec<Complex> {
        // Deterministic pseudo-random-ish input; no dependency needed.
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                let mut next = || {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32
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
                    (a.re - b.re).abs() < 1e-4 && (a.im - b.im).abs() < 1e-4,
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
                    (a.re - b.re).abs() < 1e-3 && (a.im - b.im).abs() < 1e-3,
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
                    (a.re - b.re).abs() < 1e-4 && (a.im - b.im).abs() < 1e-4,
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
                assert!((a.re - b.re).abs() < 1e-4 && (a.im - b.im).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn run_batch8_matches_scalar_bitexact() {
        // The SoA batched `Plan::run` must reproduce the scalar `run`
        // bit-for-bit on every lane: identical complex op sequence per
        // element. Covers radix-2, Bluestein (prime, odd composite,
        // pow2-adjacent), and trivial lengths.
        for n in [1usize, 2, 4, 8, 16, 3, 5, 6, 7, 9, 11, 15, 17, 33, 100, 129] {
            let plan = Plan::new(n);
            let mut re8 = vec![0.0f32; n * 8];
            let mut im8 = vec![0.0f32; n * 8];
            let mut want: Vec<Complex> = Vec::with_capacity(n * 8);
            for j in 0..8 {
                let lane = seeded_with(0x09E3_79B9_7F4A_7C15_u64.wrapping_mul(j as u64 + 1), n);
                for (k, v) in lane.iter().enumerate() {
                    re8[k * 8 + j] = v.re;
                    im8[k * 8 + j] = v.im;
                }
                let mut w = lane;
                plan.run(&mut w);
                want.extend(w);
            }
            let scratch_len = plan.batch_scratch_len();
            let mut are = vec![0.0f32; scratch_len];
            let mut aim = vec![0.0f32; scratch_len];
            plan.run_batch8(&mut re8, &mut im8, &mut are, &mut aim);
            for j in 0..8 {
                for k in 0..n {
                    let got = Complex::new(re8[k * 8 + j], im8[k * 8 + j]);
                    let w = want[j * n + k];
                    assert_eq!(
                        got.re.to_bits(),
                        w.re.to_bits(),
                        "n={n} lane={j} k={k} re: {got:?} vs {w:?}"
                    );
                    assert_eq!(
                        got.im.to_bits(),
                        w.im.to_bits(),
                        "n={n} lane={j} k={k} im: {got:?} vs {w:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn fft2_planned_non_pow2_matches_naive_roundtrip() {
        // The batched Bluestein drivers (fft2/ifft2 row+col passes)
        // must still round-trip — exercises the run_batch8 path
        // end-to-end including partial 8-lane blocks.
        for (w, h) in [(7usize, 5usize), (13, 9), (33, 17)] {
            let x = seeded(w * h);
            let mut y = x.clone();
            fft2_planned(&mut y, w, h, &Plan::new(w), &Plan::new(h));
            ifft2_planned(&mut y, w, h, &Plan::new(w), &Plan::new(h));
            for (a, b) in y.iter().zip(&x) {
                assert!((a.re - b.re).abs() < 1e-4 && (a.im - b.im).abs() < 1e-4);
            }
        }
    }
}
