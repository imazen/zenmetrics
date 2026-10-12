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
pub struct Complex {
    /// Real part.
    pub re: f32,
    /// Imaginary part.
    pub im: f32,
}

impl Complex {
    /// `re + i·im`.
    #[must_use]
    #[inline]
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// `e^{iθ}` — the angle is evaluated in `f64` and cast down, so stored
    /// twiddles are the correctly-rounded `f32` values.
    #[must_use]
    #[inline]
    pub fn expi(theta: f64) -> Self {
        let (s, c) = theta.sin_cos();
        Self {
            re: c as f32,
            im: s as f32,
        }
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
    fn scale(self, s: f32) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }
}

/// Forward DFT of `buf`, in place. Any length; `O(n log n)`.
#[allow(dead_code)] // public mirror of the vendored API; callers use the _planned forms
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
    /// Power-of-two Cooley–Tukey.
    Radix2 { mplan: MPlan },
    /// Bluestein chirp-z: `chirp[k] = e^{-iπk²/n}` and `bf` = the length-`m`
    /// FFT of the chirp filter, with the mixed-radix plan for `m`.
    Bluestein {
        chirp: Vec<Complex>,
        bf: Vec<Complex>,
        m: usize,
        mplan: MPlan,
    },
}

/// A Cooley–Tukey plan over {2,3}-radix stages — the generalization of
/// the old power-of-two stage table. Stage `s` applies `radices[s]`
/// on span `Π_{t<s} radices[t]`; `tw[s]` packs its twiddles —
/// radix-2: `w[t] = W_{2s}^{t}` (len `s`); radix-3: `w[t] = W_{3s}^{t}`
/// then `w[s+t] = W_{3s}^{2t}` (len `2s`). `perm` is the input-order
/// gather — slot `p` holds `src[perm[p]]`, the mixed-radix digit
/// reversal that replaces bit-reversal.
struct MPlan {
    /// Stage radices in application order — power-of-two plans are all
    /// 2; Bluestein `m` orders the 2s first so the gather pass keeps its
    /// fused-quad shape.
    radices: Vec<u8>,
    /// Per-stage twiddles, packed as described above.
    tw: Vec<Vec<Complex>>,
    /// `slot p ← src[perm[p]]` — the digit-reversal row gather.
    perm: Vec<u32>,
    /// Transform length.
    n: usize,
}

impl MPlan {
    /// Pure radix-2 plan for power-of-two `n` — the twiddle table is
    /// [`radix2_stages`] verbatim, the permutation bit-reversal.
    fn pow2(n: usize) -> Self {
        debug_assert!(n.is_power_of_two() && n >= 2);
        let bits = n.trailing_zeros();
        let radices = vec![2u8; bits as usize];
        let tw = radix2_stages(n);
        let perm = (0..n as u32)
            .map(|i| i.reverse_bits() >> (32 - bits))
            .collect();
        Self {
            radices,
            tw,
            perm,
            n,
        }
    }

    /// Mixed-radix plan for `n = 2^a·3^b` (radix-2 stages first). The
    /// caller guarantees the factorization — `m` always comes from
    /// [`smooth23`].
    fn smooth(n: usize) -> Self {
        let (mut a, mut b, mut rem) = (0usize, 0usize, n);
        while rem % 2 == 0 {
            a += 1;
            rem /= 2;
        }
        while rem % 3 == 0 {
            b += 1;
            rem /= 3;
        }
        debug_assert_eq!(rem, 1, "MPlan::smooth needs a 2^a·3^b length");
        debug_assert!(a + b >= 1);
        let radices: Vec<u8> = [2].repeat(a).into_iter().chain([3].repeat(b)).collect();
        // Stage s: span = product of earlier radices; block = span·r.
        // r=2: tw[t] = W_{2·span}^t. r=3: tw[t] = W_{3·span}^t,
        // tw[span+t] = W_{3·span}^{2t}.
        let mut tw = Vec::with_capacity(a + b);
        let mut span = 1usize;
        for &r in &radices {
            let block = span * r as usize;
            let ang = -2.0 * PI / block as f64;
            if r == 2 {
                tw.push((0..span).map(|t| Complex::expi(ang * t as f64)).collect());
            } else {
                tw.push(
                    (0..span)
                        .map(|t| Complex::expi(ang * t as f64))
                        .chain((0..span).map(|t| Complex::expi(ang * 2.0 * t as f64)))
                        .collect(),
                );
            }
            span = block;
        }
        debug_assert_eq!(span, n);
        // perm[i] = Σ_s d_s·(n/M_s) where i = Σ d_s·M_{s−1} — the
        // digit-reversal of `i` in the stage-order positional system
        // (bit-reversal when all radices are 2).
        let mut m_acc = 1usize;
        let mut perm = vec![0u32; n];
        let mut weights = Vec::with_capacity(radices.len());
        for &r in &radices {
            weights.push(m_acc);
            m_acc *= r as usize;
        }
        for (i, p) in perm.iter_mut().enumerate() {
            let mut rev = 0usize;
            for (w, &r) in weights.iter().zip(&radices) {
                rev += (i / w % r as usize) * (n / (w * r as usize));
            }
            *p = rev as u32;
        }
        Self {
            radices,
            tw,
            perm,
            n,
        }
    }
}

/// Smallest `2^a·3^b` ≥ `need` — the Bluestein convolution length,
/// which hugs `2n−1` where `next_power_of_two` overshoots by up to 2×
/// (n=2516 → 5184 vs 8192; n=3355 → 6912).
fn smooth23(need: usize) -> usize {
    debug_assert!(need >= 2);
    let mut best = need.next_power_of_two();
    let mut p2 = 2usize;
    while p2 < best {
        let mut v = p2;
        while v < need {
            v *= 3;
        }
        best = best.min(v);
        p2 *= 2;
    }
    best
}

/// Real multiplier a row-transform gather applies on the way in — the
/// `v ↦ (re·f, ±im·f)` factor the old separate conj/filter pass wrote
/// into `buf` before transforming. The shifted-position views
/// `a[su·n+sv]` and `a[su·n+sv]·b[su·n+sv]` with `su = (u+uo) mod m`,
/// `sv = (v+vo) mod n` let the fftshift/ifftshift weight remaps ride the
/// gather directly, so no weight plane is ever materialized. `u`/`v` are
/// absolute row/column indices of the plane being transformed.
#[derive(Clone, Copy)]
pub(crate) enum RowMul<'a> {
    /// `f = 1` — no multiply.
    Id,
    /// `f = a[su·n + sv]` — one circularly shifted plane (the csf remap).
    Shift {
        /// The unshifted source plane.
        a: &'a [f32],
        /// Row shift `uo` and column shift `vo`; each wraps at most once.
        uo: usize,
        vo: usize,
        /// Plane height — the `su` modulus (`n` is ambient).
        m: usize,
    },
    /// `f = a[su·n+sv]·b[su·n+sv]` — the gabor `lg·sp` product, lazily.
    Shift2 {
        /// First shifted plane.
        a: &'a [f32],
        /// Second shifted plane.
        b: &'a [f32],
        /// Row shift.
        uo: usize,
        /// Column shift.
        vo: usize,
        /// Plane height — the `su` modulus.
        m: usize,
    },
}

impl RowMul<'_> {
    /// The multiplier at plane position `(u, v)`; `n` is the plane width.
    /// `u < m`, `v < n`, `uo < m`, `vo < n` → each shifted index wraps at
    /// most once, so one conditional subtract replaces a modulo.
    #[inline(always)]
    fn at(self, u: usize, v: usize, n: usize) -> f32 {
        match self {
            RowMul::Id => 1.0,
            RowMul::Shift { a, uo, vo, m } | RowMul::Shift2 { a, uo, vo, m, .. } => {
                let su = u + uo;
                let su = if su >= m { su - m } else { su };
                let sv = v + vo;
                let sv = if sv >= n { sv - n } else { sv };
                let av = a[su * n + sv];
                match self {
                    RowMul::Shift2 { b, .. } => av * b[su * n + sv],
                    _ => av,
                }
            }
        }
    }
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
                mplan: MPlan::pow2(n),
            }
        } else {
            // The {2,3}-smooth convolution length hugs 2n−1 — for
            // n=2516 that's 5184 (−51% stage passes vs the 8192 power
            // of two), for n=3355 it's 6912 (−29%).
            let m = smooth23(2 * n - 1);
            let chirp: Vec<Complex> = (0..n)
                .map(|k| {
                    let kk = (k as u128 * k as u128 % (2 * n as u128)) as f64;
                    Complex::expi(-PI * kk / n as f64)
                })
                .collect();
            let mplan = MPlan::smooth(m);
            let mut bf = vec![Complex::default(); m];
            for k in 0..n {
                bf[k] = chirp[k].conj();
                if k > 0 {
                    bf[m - k] = chirp[k].conj();
                }
            }
            fft_mixed_planned(&mut bf, &mplan);
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
            PlanKind::Radix2 { mplan } => fft_mixed_planned(buf, mplan),
            PlanKind::Bluestein { m, .. } => {
                let mut a = vec![Complex::default(); *m];
                self.run_bluestein(buf, &mut a, RowMul::Id, false, 0, 0);
            }
        }
    }

    /// Eight Bluestein transforms at once in the `fft_batch8` SoA layout
    /// (lane = transform). The three `m`-length radix-2 stages vectorize
    /// through [`fft_batch8`]; the chirp/`bf` multiplies and the conj·scale
    /// are per-lane scalar ops on the way in/out — bit-identical to
    /// [`Self::run_bluestein`] on each row. `bufs` are the eight `n`-length
    /// rows/columns; `re8`/`im8` are caller scratch of `m·8`.
    ///
    /// `conj`/`mul` fuse the `(re·f, −im·f)` input factor into the gather —
    /// `bufs[j]` is treated as row `r0 + j` of a `width`-strided plane for
    /// the [`RowMul`] index math (ignored for [`RowMul::Id`]).
    fn run_bluestein8(
        &self,
        bufs: &mut [&mut [Complex]],
        re8: &mut [f32],
        im8: &mut [f32],
        re8b: &mut [f32],
        im8b: &mut [f32],
        mul: RowMul,
        conj: bool,
        r0: usize,
        width: usize,
    ) {
        let PlanKind::Bluestein { chirp, m, .. } = &self.kind else {
            unreachable!("run_bluestein8 on non-Bluestein plan")
        };
        let (n, m) = (self.n, *m);
        debug_assert_eq!(bufs.len(), 8);
        {
            // The gather finishes before any scatter store, so the shared
            // reborrow of `bufs` here never aliases the mutable use below.
            let sbufs: [&[Complex]; 8] = core::array::from_fn(|i| &bufs[i][..]);
            blue8_gather_all(&sbufs, n, m, chirp, re8, im8, mul, conj, r0, width);
        }
        self.blue8_inner(re8, im8, re8b, im8b);
        let s = 1.0 / m as f32;
        for k0 in (0..n).step_by(8) {
            blue8_scatter(bufs, n, k0, chirp, s, re8, im8);
        }
    }

    /// [`Self::run_bluestein8`] reading `sbufs` and scattering to `dbufs`
    /// — the out-of-place row-pass variant, where the input plane is a
    /// shared spectrum no transform may mutate.
    #[allow(clippy::too_many_arguments)]
    fn run_bluestein8_io(
        &self,
        sbufs: &[&[Complex]],
        dbufs: &mut [&mut [Complex]],
        re8: &mut [f32],
        im8: &mut [f32],
        re8b: &mut [f32],
        im8b: &mut [f32],
        mul: RowMul,
        conj: bool,
        r0: usize,
        width: usize,
    ) {
        let PlanKind::Bluestein { chirp, m, .. } = &self.kind else {
            unreachable!("run_bluestein8_io on non-Bluestein plan")
        };
        let (n, m) = (self.n, *m);
        debug_assert_eq!(sbufs.len(), 8);
        debug_assert_eq!(dbufs.len(), 8);
        blue8_gather_all(sbufs, n, m, chirp, re8, im8, mul, conj, r0, width);
        self.blue8_inner(re8, im8, re8b, im8b);
        let s = 1.0 / m as f32;
        for k0 in (0..n).step_by(8) {
            blue8_scatter(dbufs, n, k0, chirp, s, re8, im8);
        }
    }

    /// [`Self::run_bluestein8`] for a column block read straight out of
    /// the row-major plane: `bufs[j]` would be column `x+j`, so the
    /// gather transposes 8×8 tiles from `buf` directly — the col-major
    /// scratch's input round-trip is skipped entirely. Transformed
    /// columns still scatter into `dcols` (col-major scratch, eight
    /// `height`-length columns) for the row-major write-back sweep.
    fn run_bluestein8_cols(
        &self,
        buf: &[Complex],
        width: usize,
        x: usize,
        dcols: &mut [Complex],
        re8: &mut [f32],
        im8: &mut [f32],
        re8b: &mut [f32],
        im8b: &mut [f32],
    ) {
        let PlanKind::Bluestein { chirp, m, .. } = &self.kind else {
            unreachable!("run_bluestein8_cols on non-Bluestein plan")
        };
        let (n, m) = (self.n, *m);
        debug_assert!(dcols.len() >= 8 * n);
        // Gather: a[k] = buf[k·width + x+j]·chirp[k], zeroed past n.
        // Software-pipelined: the next 8×8 tile's strided row loads
        // issue while this tile's chirp multiplies run — the 26 KB
        // strides are latency-bound, not bandwidth-bound.
        let mb = m & !7;
        let mut xtile = gather_cols_tile(buf, width, n, x, 0);
        for k0 in (0..mb).step_by(8) {
            let cur = xtile;
            if k0 + 8 < mb {
                xtile = gather_cols_tile(buf, width, n, x, k0 + 8);
            }
            blue8_gather_cols_apply(cur, n, k0, chirp, re8, im8);
        }
        for k in mb..m {
            for j in 0..8 {
                let (re, im) = if k < n {
                    let v = buf[k * width + x + j];
                    let c = chirp[k];
                    (v.re * c.re - v.im * c.im, v.re * c.im + v.im * c.re)
                } else {
                    (0.0, 0.0)
                };
                re8[k * 8 + j] = re;
                im8[k * 8 + j] = im;
            }
        }
        self.blue8_inner(re8, im8, re8b, im8b);
        let s = 1.0 / m as f32;
        let mut dbufs: Vec<&mut [Complex]> = dcols.chunks_mut(n).take(8).collect();
        for k0 in (0..n).step_by(8) {
            blue8_scatter(&mut dbufs, n, k0, chirp, s, re8, im8);
        }
    }

    /// The SoA middle of the batch Bluestein pass: `fft → ·bf → fft` on
    /// the `m·8` planes, ping-ponging the two scratch pairs.
    fn blue8_inner(&self, re8: &mut [f32], im8: &mut [f32], re8b: &mut [f32], im8b: &mut [f32]) {
        let PlanKind::Bluestein { bf, m, mplan, .. } = &self.kind else {
            unreachable!("blue8_inner on non-Bluestein plan")
        };
        let m = *m;
        debug_assert!(
            re8.len() >= m * 8 && im8.len() >= m * 8 && re8b.len() >= m * 8 && im8b.len() >= m * 8
        );
        let (re8, im8, re8b, im8b) = (
            &mut re8[..m * 8],
            &mut im8[..m * 8],
            &mut re8b[..m * 8],
            &mut im8b[..m * 8],
        );
        fft_batch8(re8, im8, re8b, im8b, mplan);
        blue8_bf(re8b, im8b, bf);
        fft_batch8(re8b, im8b, re8, im8, mplan);
    }

    /// The Bluestein pass with caller-supplied `m`-length scratch, so
    /// parallel row/column sweeps reuse one buffer per worker instead of
    /// allocating ~64 KiB per row. Results are identical to `run`.
    ///
    /// `conj`/`mul` apply the `(re·f, −im·f)` input factor on the way in
    /// — `buf` is treated as row `u` of a `width`-strided plane for the
    /// [`RowMul`] index math (ignored for [`RowMul::Id`]).
    fn run_bluestein(
        &self,
        buf: &mut [Complex],
        a: &mut [Complex],
        mul: RowMul,
        conj: bool,
        u: usize,
        width: usize,
    ) {
        debug_assert_eq!(buf.len(), self.n);
        let PlanKind::Bluestein { chirp, m, .. } = &self.kind else {
            unreachable!("run_bluestein on non-Bluestein plan")
        };
        debug_assert!(a.len() >= *m);
        let n = self.n;
        a[..*m].fill(Complex::default());
        for k in 0..n {
            let v = buf[k];
            let f = mul.at(u, k, width);
            let im = if conj { -v.im } else { v.im };
            a[k] = Complex::new(v.re * f, im * f).mul(chirp[k]);
        }
        self.bluestein_core(a);
        for (k, out) in buf.iter_mut().enumerate() {
            *out = a[k].mul(chirp[k]);
        }
    }

    /// [`Self::run_bluestein`] reading `src` and writing `dst` — the
    /// out-of-place row-pass variant.
    fn run_bluestein_io(
        &self,
        src: &[Complex],
        dst: &mut [Complex],
        a: &mut [Complex],
        mul: RowMul,
        conj: bool,
        u: usize,
        width: usize,
    ) {
        debug_assert_eq!(src.len(), self.n);
        debug_assert_eq!(dst.len(), self.n);
        let PlanKind::Bluestein { chirp, m, .. } = &self.kind else {
            unreachable!("run_bluestein_io on non-Bluestein plan")
        };
        debug_assert!(a.len() >= *m);
        let n = self.n;
        a[..*m].fill(Complex::default());
        for k in 0..n {
            let v = src[k];
            let f = mul.at(u, k, width);
            let im = if conj { -v.im } else { v.im };
            a[k] = Complex::new(v.re * f, im * f).mul(chirp[k]);
        }
        self.bluestein_core(a);
        for (k, out) in dst.iter_mut().enumerate() {
            *out = a[k].mul(chirp[k]);
        }
    }

    /// The scalar middle shared by [`Self::run_bluestein`] and
    /// [`Self::run_bluestein_io`]: `fft → ·bf → conj → fft → conj·(1/m)`
    /// on the `m`-length scratch.
    fn bluestein_core(&self, a: &mut [Complex]) {
        let PlanKind::Bluestein { bf, m, mplan, .. } = &self.kind else {
            unreachable!("bluestein_core on non-Bluestein plan")
        };
        let m = *m;
        fft_mixed_planned(&mut a[..m], mplan);
        for (x, y) in a[..m].iter_mut().zip(bf) {
            *x = x.mul(*y);
        }
        for v in a[..m].iter_mut() {
            *v = v.conj();
        }
        fft_mixed_planned(&mut a[..m], mplan);
        let s = 1.0 / m as f32;
        for v in a[..m].iter_mut() {
            *v = v.conj().scale(s);
        }
    }
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
/// block, transposed in and out).
///
/// The transform runs out-of-place: the first fused stage pair reads
/// `src[rev(p)]` — the bit-reversal as a row gather, no swap sweep at all —
/// and writes `dst` contiguously, then the remaining stages run in-place on
/// `dst`. The gather would corrupt an in-place array (later blocks still
/// need rows the earlier stores already overwrote), so `src`/`dst` must not
/// alias; `run_bluestein8` ping-pongs the pair across its two transforms.
///
/// Twiddles are splat per butterfly — cheap, and identical across lanes.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn fft_batch8_inner(
    token: Token,
    src_re: &[f32],
    src_im: &[f32],
    dst_re: &mut [f32],
    dst_im: &mut [f32],
    mplan: &MPlan,
) {
    let n = mplan.n;
    debug_assert_eq!(src_re.len(), n * 8);
    debug_assert_eq!(src_im.len(), n * 8);
    debug_assert_eq!(dst_re.len(), n * 8);
    debug_assert_eq!(dst_im.len(), n * 8);

    let perm = &mplan.perm;
    let radices = &mplan.radices;
    let stages = &mplan.tw;
    let mut s = 0usize;

    if radices.len() >= 2 && radices[0] == 2 && radices[1] == 2 {
        // The digit-reversal rides the first fused pair: logical slot `p`
        // holds `src[perm[p]]`, so loads are permuted row gathers while
        // stores stay contiguous — the src→dst split makes the gather
        // alias-safe where an in-place version would read rows it had
        // already overwritten. Stage-0/1 twiddles are exactly (1,0) and
        // {(1,0), w1[1]≈−i}, so the pass is adds, subs, and one twiddle
        // multiply per 4 elements.
        let tw1 = &stages[1];
        // Row gather for one 4-element block: `src[perm[start+q]]`.
        let load4 = |start: usize| {
            let (p0, p1, p2, p3) = (
                perm[start] as usize * 8,
                perm[start + 1] as usize * 8,
                perm[start + 2] as usize * 8,
                perm[start + 3] as usize * 8,
            );
            (
                f32x8::load_t(token, (&src_re[p0..p0 + 8]).try_into().unwrap()),
                f32x8::load_t(token, (&src_im[p0..p0 + 8]).try_into().unwrap()),
                f32x8::load_t(token, (&src_re[p1..p1 + 8]).try_into().unwrap()),
                f32x8::load_t(token, (&src_im[p1..p1 + 8]).try_into().unwrap()),
                f32x8::load_t(token, (&src_re[p2..p2 + 8]).try_into().unwrap()),
                f32x8::load_t(token, (&src_im[p2..p2 + 8]).try_into().unwrap()),
                f32x8::load_t(token, (&src_re[p3..p3 + 8]).try_into().unwrap()),
                f32x8::load_t(token, (&src_im[p3..p3 + 8]).try_into().unwrap()),
            )
        };
        // Software-pipelined: the scattered row loads for `start+4` issue a
        // full iteration before their values are needed, hiding the gather
        // latency behind this block's butterflies and stores.
        let mut x = load4(0);
        for start in (0..n).step_by(4) {
            let (x0re, x0im, x1re, x1im, x2re, x2im, x3re, x3im) = x;
            if start + 4 < n {
                x = load4(start + 4);
            }
            // Stage 0 (twiddle 1): y = x±x on both pairs.
            let (y0re, y1re) = (x0re + x1re, x0re - x1re);
            let (y0im, y1im) = (x0im + x1im, x0im - x1im);
            let (y2re, y3re) = (x2re + x3re, x2re - x3re);
            let (y2im, y3im) = (x2im + x3im, x2im - x3im);
            // Stage 1: tw1[0] is exactly (1,0) — y2 through unchanged;
            // tw1[1] is the stored ≈−i twiddle (a real cmul, kept for
            // bit-parity with the unfused path).
            let i0 = start * 8;
            let (i1, i2, i3) = (i0 + 8, i0 + 16, i0 + 24);
            (y0re + y2re).store((&mut dst_re[i0..i0 + 8]).try_into().unwrap());
            (y0im + y2im).store((&mut dst_im[i0..i0 + 8]).try_into().unwrap());
            (y0re - y2re).store((&mut dst_re[i2..i2 + 8]).try_into().unwrap());
            (y0im - y2im).store((&mut dst_im[i2..i2 + 8]).try_into().unwrap());
            let (w1re_b, w1im_b) = (
                f32x8::splat_t(token, tw1[1].re),
                f32x8::splat_t(token, tw1[1].im),
            );
            let tre = y3re * w1re_b - y3im * w1im_b;
            let tim = y3re * w1im_b + y3im * w1re_b;
            (y1re + tre).store((&mut dst_re[i1..i1 + 8]).try_into().unwrap());
            (y1im + tim).store((&mut dst_im[i1..i1 + 8]).try_into().unwrap());
            (y1re - tre).store((&mut dst_re[i3..i3 + 8]).try_into().unwrap());
            (y1im - tim).store((&mut dst_im[i3..i3 + 8]).try_into().unwrap());
        }
        s = 2;
    } else {
        // Anything but a [2,2] start (pure-3 plans, single radix-2):
        // digit-reversed gather into dst; the stage loop below applies
        // every stage's twiddles.
        for i in 0..n {
            let j = perm[i] as usize;
            let a = f32x8::load_t(token, (&src_re[j * 8..j * 8 + 8]).try_into().unwrap());
            a.store((&mut dst_re[i * 8..i * 8 + 8]).try_into().unwrap());
            let a = f32x8::load_t(token, (&src_im[j * 8..j * 8 + 8]).try_into().unwrap());
            a.store((&mut dst_im[i * 8..i * 8 + 8]).try_into().unwrap());
        }
    }

    // Consecutive radix-2 stage pairs run fused (radix-4-equivalent): the
    // two butterflies land on the same registers, so each element loads
    // and stores once per pair instead of once per stage — identical
    // arithmetic, half the passes over the planes. Radix-3 stages and
    // unpaired radix-2s take the single-stage paths.
    while s < radices.len() {
        if radices[s] == 2 {
            if s + 1 < radices.len() && radices[s + 1] == 2 {
                let (tw0, tw1) = (&stages[s], &stages[s + 1]);
                let half = tw0.len();
                let (h8, h16, h24) = (half * 8, half * 16, half * 24);
                for start in (0..n).step_by(half * 4) {
                    for k in 0..half {
                        let (w0re, w0im) = (
                            f32x8::splat_t(token, tw0[k].re),
                            f32x8::splat_t(token, tw0[k].im),
                        );
                        let (w1re_a, w1im_a) = (
                            f32x8::splat_t(token, tw1[k].re),
                            f32x8::splat_t(token, tw1[k].im),
                        );
                        let (w1re_b, w1im_b) = (
                            f32x8::splat_t(token, tw1[k + half].re),
                            f32x8::splat_t(token, tw1[k + half].im),
                        );
                        let i0 = (start + k) * 8;
                        let (i1, i2, i3) = (i0 + h8, i0 + h16, i0 + h24);
                        // Stage s: butterflies (i0,i1) and (i2,i3), same twiddle.
                        let vre = f32x8::load_t(token, (&dst_re[i1..i1 + 8]).try_into().unwrap());
                        let vim = f32x8::load_t(token, (&dst_im[i1..i1 + 8]).try_into().unwrap());
                        let tre = vre * w0re - vim * w0im;
                        let tim = vre * w0im + vim * w0re;
                        let ure = f32x8::load_t(token, (&dst_re[i0..i0 + 8]).try_into().unwrap());
                        let uim = f32x8::load_t(token, (&dst_im[i0..i0 + 8]).try_into().unwrap());
                        let (y0re, y1re) = (ure + tre, ure - tre);
                        let (y0im, y1im) = (uim + tim, uim - tim);
                        let vre = f32x8::load_t(token, (&dst_re[i3..i3 + 8]).try_into().unwrap());
                        let vim = f32x8::load_t(token, (&dst_im[i3..i3 + 8]).try_into().unwrap());
                        let tre = vre * w0re - vim * w0im;
                        let tim = vre * w0im + vim * w0re;
                        let ure = f32x8::load_t(token, (&dst_re[i2..i2 + 8]).try_into().unwrap());
                        let uim = f32x8::load_t(token, (&dst_im[i2..i2 + 8]).try_into().unwrap());
                        let (y2re, y3re) = (ure + tre, ure - tre);
                        let (y2im, y3im) = (uim + tim, uim - tim);
                        // Stage s+1: butterflies (i0,i2) with tw1[k], (i1,i3) with
                        // tw1[k+half].
                        let tre = y2re * w1re_a - y2im * w1im_a;
                        let tim = y2re * w1im_a + y2im * w1re_a;
                        (y0re + tre).store((&mut dst_re[i0..i0 + 8]).try_into().unwrap());
                        (y0im + tim).store((&mut dst_im[i0..i0 + 8]).try_into().unwrap());
                        (y0re - tre).store((&mut dst_re[i2..i2 + 8]).try_into().unwrap());
                        (y0im - tim).store((&mut dst_im[i2..i2 + 8]).try_into().unwrap());
                        let tre = y3re * w1re_b - y3im * w1im_b;
                        let tim = y3re * w1im_b + y3im * w1re_b;
                        (y1re + tre).store((&mut dst_re[i1..i1 + 8]).try_into().unwrap());
                        (y1im + tim).store((&mut dst_im[i1..i1 + 8]).try_into().unwrap());
                        (y1re - tre).store((&mut dst_re[i3..i3 + 8]).try_into().unwrap());
                        (y1im - tim).store((&mut dst_im[i3..i3 + 8]).try_into().unwrap());
                    }
                }
                s += 2;
            } else {
                let tw = &stages[s];
                let half = tw.len();
                let len = half * 2;
                for start in (0..n).step_by(len) {
                    for (k, &w) in tw.iter().enumerate() {
                        let a = (start + k) * 8;
                        let b = (start + k + half) * 8;
                        let wre = f32x8::splat_t(token, w.re);
                        let wim = f32x8::splat_t(token, w.im);
                        let ure = f32x8::load_t(token, (&dst_re[a..a + 8]).try_into().unwrap());
                        let uim = f32x8::load_t(token, (&dst_im[a..a + 8]).try_into().unwrap());
                        let vre = f32x8::load_t(token, (&dst_re[b..b + 8]).try_into().unwrap());
                        let vim = f32x8::load_t(token, (&dst_im[b..b + 8]).try_into().unwrap());
                        let tre = vre * wre - vim * wim;
                        let tim = vre * wim + vim * wre;
                        (ure + tre).store((&mut dst_re[a..a + 8]).try_into().unwrap());
                        (uim + tim).store((&mut dst_im[a..a + 8]).try_into().unwrap());
                        (ure - tre).store((&mut dst_re[b..b + 8]).try_into().unwrap());
                        (uim - tim).store((&mut dst_im[b..b + 8]).try_into().unwrap());
                    }
                }
                s += 1;
            }
        } else {
            // Radix-3 stage, span `sp`, block `3·sp`: X_q =
            // x_q·W_{3sp}^{q·t} then a 3-point DFT over q — with
            // ω₃ = −½−i√3/2 and ω₃² = conj(ω₃), the DFT-3 needs only
            // real scales of S = X₁+X₂ and D = X₁−X₂.
            let tw = &stages[s];
            let sp = tw.len() / 2;
            let (a, b) = (-0.5f32, -0.866_025_4f32);
            let (a8, b8) = (f32x8::splat_t(token, a), f32x8::splat_t(token, b));
            for start in (0..n).step_by(sp * 3) {
                for t in 0..sp {
                    let w1 = tw[t];
                    let w2 = tw[sp + t];
                    let (w1re, w1im) = (f32x8::splat_t(token, w1.re), f32x8::splat_t(token, w1.im));
                    let (w2re, w2im) = (f32x8::splat_t(token, w2.re), f32x8::splat_t(token, w2.im));
                    let i0 = (start + t) * 8;
                    let i1 = i0 + sp * 8;
                    let i2 = i0 + 2 * sp * 8;
                    let x0re = f32x8::load_t(token, (&dst_re[i0..i0 + 8]).try_into().unwrap());
                    let x0im = f32x8::load_t(token, (&dst_im[i0..i0 + 8]).try_into().unwrap());
                    let x1re = f32x8::load_t(token, (&dst_re[i1..i1 + 8]).try_into().unwrap());
                    let x1im = f32x8::load_t(token, (&dst_im[i1..i1 + 8]).try_into().unwrap());
                    let x2re = f32x8::load_t(token, (&dst_re[i2..i2 + 8]).try_into().unwrap());
                    let x2im = f32x8::load_t(token, (&dst_im[i2..i2 + 8]).try_into().unwrap());
                    let (x1re, x1im) = (x1re * w1re - x1im * w1im, x1re * w1im + x1im * w1re);
                    let (x2re, x2im) = (x2re * w2re - x2im * w2im, x2re * w2im + x2im * w2re);
                    let (sre, sim) = (x1re + x2re, x1im + x2im);
                    let (dre, dim) = (x1re - x2re, x1im - x2im);
                    // out0 = x0+S; out1 = x0 + a·S + i·b·D; out2 = x0 + a·S − i·b·D.
                    let (tre, tim) = (sre * a8, sim * a8);
                    let (ure, uim) = (dre * b8, dim * b8);
                    (x0re + sre).store((&mut dst_re[i0..i0 + 8]).try_into().unwrap());
                    (x0im + sim).store((&mut dst_im[i0..i0 + 8]).try_into().unwrap());
                    (x0re + tre - uim).store((&mut dst_re[i1..i1 + 8]).try_into().unwrap());
                    (x0im + tim + ure).store((&mut dst_im[i1..i1 + 8]).try_into().unwrap());
                    (x0re + tre + uim).store((&mut dst_re[i2..i2 + 8]).try_into().unwrap());
                    (x0im + tim - ure).store((&mut dst_im[i2..i2 + 8]).try_into().unwrap());
                }
            }
            s += 1;
        }
    }
}

/// [`fft_batch8_inner`] with runtime tier dispatch.
fn fft_batch8(
    src_re: &[f32],
    src_im: &[f32],
    dst_re: &mut [f32],
    dst_im: &mut [f32],
    mplan: &MPlan,
) {
    archmage::attuned!(
        fft_batch8_inner(src_re, src_im, dst_re, dst_im, mplan),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

/// One 8×8 k-block of the [`Plan::run_bluestein8`] gather: lane `j` reads
/// `bufs[j][k′]` (zeroed past `n`), applies the `conj·f` input factor
/// (`bufs[j]` is plane row `r0+j` for the [`RowMul`] index math), and
/// stores `·chirp[k′]` contiguously at `re8[k′·8+j]` for k′ = k0..k0+8.
/// The fixed arrays keep the 8×8 transpose register-resident so each tier
/// body lowers to unpack/permute shuffles.
#[archmage::attune(dispatch, +v4x(cfg(avx512)))]
fn blue8_gather(
    sbufs: &[&[Complex]],
    n: usize,
    k0: usize,
    chirp: &[Complex],
    re8: &mut [f32],
    im8: &mut [f32],
    mul: RowMul,
    conj: bool,
    r0: usize,
    width: usize,
) {
    let cols = n.saturating_sub(k0).min(8);
    // The per-element multiplier for this tile, match-once per spec kind.
    let mut fm = [[1.0f32; 8]; 8];
    if !matches!(mul, RowMul::Id) {
        for (j, row) in fm.iter_mut().enumerate() {
            for (w, c) in row.iter_mut().zip(0..cols) {
                *w = mul.at(r0 + j, k0 + c, width);
            }
        }
    }
    let mut br = [[0.0f32; 8]; 8];
    let mut bi = [[0.0f32; 8]; 8];
    if cols == 8 {
        // Fully fixed bounds: LLVM lowers this to ymm loads + deinterleave.
        for (j, (b, frow)) in br.iter_mut().zip(bi.iter_mut()).zip(fm.iter()).enumerate() {
            let row = sbufs[j];
            for c in 0..8 {
                let z = row[k0 + c];
                b.0[c] = z.re * frow[c];
                b.1[c] = (if conj { -z.im } else { z.im }) * frow[c];
            }
        }
    } else {
        for (j, (b, frow)) in br.iter_mut().zip(bi.iter_mut()).zip(fm.iter()).enumerate() {
            let row = sbufs[j];
            for c in 0..cols {
                let z = row[k0 + c];
                b.0[c] = z.re * frow[c];
                b.1[c] = (if conj { -z.im } else { z.im }) * frow[c];
            }
        }
    }
    for k in 0..8 {
        let d = (k0 + k) * 8;
        if k >= cols {
            // chirp is n-length: the padded lanes are zero, not c·0.
            re8[d..d + 8].fill(0.0);
            im8[d..d + 8].fill(0.0);
            continue;
        }
        let c = chirp[k0 + k];
        let (mut r, mut i) = ([0.0f32; 8], [0.0f32; 8]);
        for j in 0..8 {
            let (zr, zi) = (br[j][k], bi[j][k]);
            r[j] = zr * c.re - zi * c.im;
            i[j] = zr * c.im + zi * c.re;
        }
        re8[d..d + 8].copy_from_slice(&r);
        im8[d..d + 8].copy_from_slice(&i);
    }
}

/// The 8×8 tile read of [`Plan::run_bluestein8_cols`]'s
/// direct-from-plane gather: `(br[j][c], bi[j][c])` = the real/imag
/// parts of `buf[(k0+c)·width + x+j]` — eight contiguous 8-wide row
/// loads (fixed bounds lower to ymm loads + deinterleave), transposed
/// in-register. Split from the chirp apply so the caller can pipeline
/// the strided loads a tile ahead.
#[archmage::attune(dispatch, +v4x(cfg(avx512)))]
fn gather_cols_tile(
    buf: &[Complex],
    width: usize,
    n: usize,
    x: usize,
    k0: usize,
) -> ([[f32; 8]; 8], [[f32; 8]; 8]) {
    let rows = n.saturating_sub(k0).min(8);
    let mut br = [[0.0f32; 8]; 8];
    let mut bi = [[0.0f32; 8]; 8];
    if rows == 8 {
        for c in 0..8 {
            let src: &[Complex; 8] = buf[(k0 + c) * width + x..(k0 + c) * width + x + 8]
                .try_into()
                .unwrap();
            for j in 0..8 {
                br[j][c] = src[j].re;
                bi[j][c] = src[j].im;
            }
        }
    } else {
        for c in 0..rows {
            let src = &buf[(k0 + c) * width + x..(k0 + c) * width + x + 8];
            for j in 0..8 {
                br[j][c] = src[j].re;
                bi[j][c] = src[j].im;
            }
        }
    }
    (br, bi)
}

/// The chirp-apply half of [`Plan::run_bluestein8_cols`]'s gather:
/// `re8[k′·8+j] = tile[j][k′]·chirp[k′]` for k′ = k0..k0+8, zero-filled
/// past `n` — the same stores as the row-side [`blue8_gather`].
#[archmage::attune(dispatch, +v4x(cfg(avx512)))]
fn blue8_gather_cols_apply(
    tile: ([[f32; 8]; 8], [[f32; 8]; 8]),
    n: usize,
    k0: usize,
    chirp: &[Complex],
    re8: &mut [f32],
    im8: &mut [f32],
) {
    let (br, bi) = tile;
    let rows = n.saturating_sub(k0).min(8);
    for k in 0..8 {
        let d = (k0 + k) * 8;
        if k >= rows {
            re8[d..d + 8].fill(0.0);
            im8[d..d + 8].fill(0.0);
            continue;
        }
        let c = chirp[k0 + k];
        let (mut r, mut i) = ([0.0f32; 8], [0.0f32; 8]);
        for j in 0..8 {
            let (zr, zi) = (br[j][k], bi[j][k]);
            r[j] = zr * c.re - zi * c.im;
            i[j] = zr * c.im + zi * c.re;
        }
        re8[d..d + 8].copy_from_slice(&r);
        im8[d..d + 8].copy_from_slice(&i);
    }
}

/// The full [`Plan::run_bluestein8`]/[`Plan::run_bluestein8_io`] gather:
/// `re8[k·8+j] = (conj·f)(sbufs[j][k])·chirp[k]` for `k < n`, zero past it
/// — 8×8 [`blue8_gather`] tiles plus the scalar `m % 8` tail.
#[allow(clippy::too_many_arguments)]
fn blue8_gather_all(
    sbufs: &[&[Complex]],
    n: usize,
    m: usize,
    chirp: &[Complex],
    re8: &mut [f32],
    im8: &mut [f32],
    mul: RowMul,
    conj: bool,
    r0: usize,
    width: usize,
) {
    let mb = m & !7;
    for k0 in (0..mb).step_by(8) {
        blue8_gather(sbufs, n, k0, chirp, re8, im8, mul, conj, r0, width);
    }
    for k in mb..m {
        for (j, buf) in sbufs.iter().enumerate() {
            let (re, im) = if k < n {
                let v = buf[k];
                let c = chirp[k];
                let f = mul.at(r0 + j, k, width);
                let (zr, zi) = (v.re * f, (if conj { -v.im } else { v.im }) * f);
                (zr * c.re - zi * c.im, zr * c.im + zi * c.re)
            } else {
                (0.0, 0.0)
            };
            re8[k * 8 + j] = re;
            im8[k * 8 + j] = im;
        }
    }
}

/// The bf-multiply + conjugate pass of [`Plan::run_bluestein8`] — per k the
/// eight SoA lanes are one vector, so the whole pass is contiguous.
#[archmage::attune(dispatch, +v4x(cfg(avx512)))]
fn blue8_bf(re8: &mut [f32], im8: &mut [f32], bf: &[Complex]) {
    for (k, &b) in bf.iter().enumerate() {
        let d = k * 8;
        let mut r: [f32; 8] = re8[d..d + 8].try_into().unwrap();
        let mut i: [f32; 8] = im8[d..d + 8].try_into().unwrap();
        for j in 0..8 {
            let (zr, zi) = (r[j], i[j]);
            r[j] = zr * b.re - zi * b.im;
            i[j] = -(zr * b.im + zi * b.re);
        }
        re8[d..d + 8].copy_from_slice(&r);
        im8[d..d + 8].copy_from_slice(&i);
    }
}

/// One 8×8 k-block of the [`Plan::run_bluestein8`] scatter: the scale
/// `(re·s, −im·s)` stays fused into the chirp multiply and `bufs[j][k′]` is
/// written through a per-row contiguous copy for k′ = k0..k0+8, bounded by `n`.
#[archmage::attune(dispatch, +v4x(cfg(avx512)))]
fn blue8_scatter(
    bufs: &mut [&mut [Complex]],
    n: usize,
    k0: usize,
    chirp: &[Complex],
    s: f32,
    re8: &[f32],
    im8: &[f32],
) {
    let cols = (n - k0).min(8);
    let mut out = [[Complex::default(); 8]; 8];
    if cols == 8 {
        // Fully fixed bounds: the transpose-out lowers to unpacks and the
        // 64-byte row stores to inline ymm ops.
        for k in 0..8 {
            let c = chirp[k0 + k];
            let d = (k0 + k) * 8;
            let mut r = [0.0f32; 8];
            let mut i = [0.0f32; 8];
            r.copy_from_slice(&re8[d..d + 8]);
            i.copy_from_slice(&im8[d..d + 8]);
            for j in 0..8 {
                let (rs, ims) = (r[j] * s, -(i[j] * s));
                out[j][k] = Complex::new(rs * c.re - ims * c.im, rs * c.im + ims * c.re);
            }
        }
        for (j, o) in out.iter().enumerate() {
            bufs[j][k0..k0 + 8].copy_from_slice(o);
        }
    } else {
        for k in 0..cols {
            let c = chirp[k0 + k];
            let d = (k0 + k) * 8;
            let mut r = [0.0f32; 8];
            let mut i = [0.0f32; 8];
            r.copy_from_slice(&re8[d..d + 8]);
            i.copy_from_slice(&im8[d..d + 8]);
            for j in 0..8 {
                let (rs, ims) = (r[j] * s, -(i[j] * s));
                out[j][k] = Complex::new(rs * c.re - ims * c.im, rs * c.im + ims * c.re);
            }
        }
        for (j, o) in out.iter().enumerate() {
            bufs[j][k0..k0 + cols].copy_from_slice(&o[..cols]);
        }
    }
}

/// Gather one `nr`×`nb` tile (≤ 8×8) of the column transpose-in:
/// `ro[y0+k][x..x+nb]` rows load contiguously, the fixed 8×8 array
/// keeps the transpose register-resident (unpack/permute shuffles),
/// and each column `l` stores contiguously at `dst[l·height + y0]`.
#[archmage::attune(dispatch, +v4x(cfg(avx512)))]
fn cols_gather_tile(
    ro: &[Complex],
    width: usize,
    x: usize,
    y0: usize,
    nr: usize,
    nb: usize,
    dst: &mut [Complex],
    height: usize,
) {
    let mut t = [[Complex::default(); 8]; 8];
    if nr == 8 && nb == 8 {
        for k in 0..8 {
            let src: &[Complex; 8] = ro[(y0 + k) * width + x..(y0 + k) * width + x + 8]
                .try_into()
                .unwrap();
            for l in 0..8 {
                t[l][k] = src[l];
            }
        }
        for (l, col) in t.iter().enumerate() {
            dst[l * height + y0..l * height + y0 + 8].copy_from_slice(col);
        }
    } else {
        for k in 0..nr {
            let src = &ro[(y0 + k) * width + x..(y0 + k) * width + x + nb];
            for (l, &v) in src.iter().enumerate() {
                t[l][k] = v;
            }
        }
        for l in 0..nb {
            dst[l * height + y0..l * height + y0 + nr].copy_from_slice(&t[l][..nr]);
        }
    }
}

/// The transpose-out twin of [`cols_gather_tile`]: column `l` loads
/// contiguously from `src[l·height + y0]`, gets `conj·scale` fused in
/// (`re·s, −im·s` when `negate`), and each row `k` stores
/// contiguously at `dst[k·width + x]`.
#[archmage::attune(dispatch, +v4x(cfg(avx512)))]
#[allow(clippy::too_many_arguments)]
fn cols_scatter_tile(
    src: &[Complex],
    height: usize,
    y0: usize,
    nr: usize,
    nb: usize,
    s: f32,
    negate: bool,
    dst: &mut [Complex],
    width: usize,
    x: usize,
) {
    let mut t = [[Complex::default(); 8]; 8];
    if nr == 8 && nb == 8 {
        for (l, col) in t.iter_mut().enumerate() {
            let src_c: &[Complex; 8] = src[l * height + y0..l * height + y0 + 8]
                .try_into()
                .unwrap();
            *col = *src_c;
        }
        for k in 0..8 {
            let mut out = [Complex::default(); 8];
            for l in 0..8 {
                let v = t[l][k];
                out[l] = Complex::new(v.re * s, if negate { -v.im * s } else { v.im * s });
            }
            dst[k * width + x..k * width + x + 8].copy_from_slice(&out);
        }
    } else {
        for l in 0..nb {
            for k in 0..nr {
                t[l][k] = src[l * height + y0 + k];
            }
        }
        for k in 0..nr {
            for l in 0..nb {
                let v = t[l][k];
                dst[k * width + x + l] =
                    Complex::new(v.re * s, if negate { -v.im * s } else { v.im * s });
            }
        }
    }
}

/// [`cols_scatter_tile`] writing `|z|` instead of `z` — the final
/// conj·scale value's magnitude, identical to running the element
/// `sqrt(re²+im²)` pass over the complex result.
#[archmage::attune(dispatch, +v4x(cfg(avx512)))]
#[allow(clippy::too_many_arguments)]
fn cols_scatter_tile_mag(
    src: &[Complex],
    height: usize,
    y0: usize,
    nr: usize,
    nb: usize,
    s: f32,
    negate: bool,
    dst: &mut [f32],
    width: usize,
    x: usize,
) {
    #[inline(always)]
    fn norm(v: Complex, s: f32, negate: bool) -> f32 {
        let (re, im) = (v.re * s, if negate { -v.im * s } else { v.im * s });
        libm::sqrtf(re * re + im * im)
    }
    let mut t = [[Complex::default(); 8]; 8];
    if nr == 8 && nb == 8 {
        for (l, col) in t.iter_mut().enumerate() {
            let src_c: &[Complex; 8] = src[l * height + y0..l * height + y0 + 8]
                .try_into()
                .unwrap();
            *col = *src_c;
        }
        for k in 0..8 {
            let mut out = [0.0f32; 8];
            for l in 0..8 {
                out[l] = norm(t[l][k], s, negate);
            }
            dst[k * width + x..k * width + x + 8].copy_from_slice(&out);
        }
    } else {
        for l in 0..nb {
            for k in 0..nr {
                t[l][k] = src[l * height + y0 + k];
            }
        }
        for k in 0..nr {
            for l in 0..nb {
                dst[k * width + x + l] = norm(t[l][k], s, negate);
            }
        }
    }
}

/// Iterative mixed-radix DFT with the precomputed stage table and
/// digit-reversal permutation — identical to the old radix-2 form on
/// power-of-two plans, radix-3 stages slot in where `m` needs them.
fn fft_mixed_planned(buf: &mut [Complex], mplan: &MPlan) {
    let n = buf.len();
    debug_assert_eq!(n, mplan.n);

    // Digit-reversal permutation. Mixed-radix digit reversal is not an
    // involution, so the in-place swap trick of pure bit-reversal
    // doesn't apply — this gathers out-of-place through a scratch row.
    // The scalar path is the cold one (edge-column tails, tiny sizes,
    // `bf` setup); the batch kernel gathers straight out of `src`.
    let mut tmp = vec![Complex::default(); n];
    for (i, &p) in mplan.perm.iter().enumerate() {
        tmp[i] = buf[p as usize];
    }
    buf.copy_from_slice(&tmp);

    for (tw, &r) in mplan.tw.iter().zip(&mplan.radices) {
        if r == 2 {
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
        } else {
            // Radix-3: span `sp`, block `3·sp`; X_q = x_q·W_{3sp}^{q·t},
            // then a 3-point DFT over q. With ω₃ = −½−i√3/2 the DFT-3
            // is real scales of S = X₁+X₂ and D = X₁−X₂.
            let sp = tw.len() / 2;
            let (a, b) = (-0.5f32, -0.866_025_4f32);
            for start in (0..n).step_by(sp * 3) {
                for t in 0..sp {
                    let (i0, i1, i2) = (start + t, start + t + sp, start + t + 2 * sp);
                    let x0 = buf[i0];
                    let x1 = buf[i1].mul(tw[t]);
                    let x2 = buf[i2].mul(tw[sp + t]);
                    let s = x1.add(x2);
                    let d = x1.sub(x2);
                    buf[i0] = x0.add(s);
                    // out1 = x0 + a·S + i·b·D; out2 = x0 + a·S − i·b·D.
                    let tre = s.re * a;
                    let tim = s.im * a;
                    let ure = d.re * b;
                    let uim = d.im * b;
                    buf[i1] = Complex::new(x0.re + tre - uim, x0.im + tim + ure);
                    buf[i2] = Complex::new(x0.re + tre + uim, x0.im + tim - ure);
                }
            }
        }
    }
}

/// Inverse DFT of `buf`, in place, normalised by `1/n`.
#[allow(dead_code)] // public mirror of the vendored API; callers use the _planned forms
pub fn ifft(buf: &mut [Complex]) {
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
pub fn fft2(buf: &mut [Complex], width: usize, height: usize) {
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
    fft2_rows_conj_mul(buf, width, height, wplan, RowMul::Id, false);
}

/// [`fft2_rows_planned`] with an optional `conj` and a real filter multiply
/// fused into the transpose **in**: with `conj` set it loads
/// `(re·f, −im·f)` (with `f = 1` for [`RowMul::Id`]), i.e.
/// `conj(buf·f)` — exactly what an `ifft2` whose input was just
/// pointwise-filtered needs, without a separate pass over the buffer.
/// Bluestein plans fuse the same factor into the chirp gather.
fn fft2_rows_conj_mul(
    buf: &mut [Complex],
    width: usize,
    height: usize,
    wplan: &Plan,
    mul: RowMul,
    conj: bool,
) {
    if let PlanKind::Radix2 { mplan } = &wplan.kind {
        const RB: usize = 8;
        // Row blocks are disjoint slices of `buf`, so under `parallel` each
        // block's transpose+transform+writeback is an independent task.
        #[cfg(feature = "parallel")]
        if height >= 4 * RB {
            use rayon::prelude::*;
            buf.par_chunks_mut(width * RB)
                .enumerate()
                .map_init(
                    || {
                        (
                            vec![0.0f32; width * RB],
                            vec![0.0f32; width * RB],
                            vec![0.0f32; width * RB],
                            vec![0.0f32; width * RB],
                        )
                    },
                    |(re8, im8, re8b, im8b), (ci, chunk)| {
                        let nb = chunk.len() / width;
                        let r0 = ci * RB;
                        for x in 0..width {
                            let mut fj = [1.0f32; RB];
                            for (w, j) in fj.iter_mut().zip(0..nb) {
                                *w = mul.at(r0 + j, x, width);
                            }
                            for j in 0..nb {
                                let i = j * width + x;
                                let v = chunk[i];
                                let f = fj[j];
                                let im = if conj { -v.im } else { v.im };
                                re8[x * RB + j] = v.re * f;
                                im8[x * RB + j] = im * f;
                            }
                            for j in nb..RB {
                                re8[x * RB + j] = 0.0;
                                im8[x * RB + j] = 0.0;
                            }
                        }
                        fft_batch8(re8, im8, re8b, im8b, mplan);
                        for x in 0..width {
                            for j in 0..nb {
                                chunk[j * width + x] =
                                    Complex::new(re8b[x * RB + j], im8b[x * RB + j]);
                            }
                        }
                    },
                )
                .for_each(|()| ());
            return;
        }
        let mut re8 = vec![0.0f32; width * RB];
        let mut im8 = vec![0.0f32; width * RB];
        let mut re8b = vec![0.0f32; width * RB];
        let mut im8b = vec![0.0f32; width * RB];
        let mut y = 0usize;
        while y < height {
            let nb = RB.min(height - y);
            for x in 0..width {
                let mut fj = [1.0f32; RB];
                for (w, j) in fj.iter_mut().zip(0..nb) {
                    *w = mul.at(y + j, x, width);
                }
                for j in 0..nb {
                    let i = (y + j) * width + x;
                    let v = buf[i];
                    let f = fj[j];
                    let im = if conj { -v.im } else { v.im };
                    re8[x * RB + j] = v.re * f;
                    im8[x * RB + j] = im * f;
                }
                for j in nb..RB {
                    re8[x * RB + j] = 0.0;
                    im8[x * RB + j] = 0.0;
                }
            }
            fft_batch8(&re8, &im8, &mut re8b, &mut im8b, mplan);
            for x in 0..width {
                for j in 0..nb {
                    buf[(y + j) * width + x] = Complex::new(re8b[x * RB + j], im8b[x * RB + j]);
                }
            }
            y += nb;
        }
    } else {
        let bluestein = matches!(wplan.kind, PlanKind::Bluestein { .. });
        if !bluestein && (conj || !matches!(mul, RowMul::Id)) {
            for (i, v) in buf.iter_mut().enumerate() {
                let f = mul.at(i / width, i % width, width);
                let im = if conj { -v.im } else { v.im };
                *v = Complex::new(v.re * f, im * f);
            }
        }
        // Bluestein rows are independent transforms of disjoint rows —
        // every real-world (non-power-of-two) image lands here, so this
        // is where mad's thread scaling actually lives. Groups of eight
        // take the SoA batch path (each m-length radix-2 stage vectorizes)
        // on both the serial and the parallel path.
        let mut done = false;
        #[cfg(feature = "parallel")]
        if let PlanKind::Bluestein { m, .. } = &wplan.kind {
            use rayon::prelude::*;
            let m = *m;
            buf.par_chunks_mut(width * 8).enumerate().for_each_init(
                || {
                    (
                        vec![0.0f32; m * 8],
                        vec![0.0f32; m * 8],
                        vec![0.0f32; m * 8],
                        vec![0.0f32; m * 8],
                        vec![Complex::default(); m],
                    )
                },
                |(re8, im8, re8b, im8b, a), (ci, chunk)| {
                    let nb = chunk.len() / width;
                    let r0 = ci * 8;
                    let mut bufs: Vec<&mut [Complex]> = chunk.chunks_mut(width).collect();
                    if nb == 8 {
                        wplan.run_bluestein8(&mut bufs, re8, im8, re8b, im8b, mul, conj, r0, width);
                    } else {
                        for (j, row) in bufs.into_iter().enumerate() {
                            wplan.run_bluestein(row, a, mul, conj, r0 + j, width);
                        }
                    }
                },
            );
            done = true;
        }
        if !done {
            if let PlanKind::Bluestein { m, .. } = &wplan.kind {
                // Serial path runs the same eight-lane batches, reusing one
                // scratch set across the sweep instead of allocating per row.
                let m = *m;
                let (mut re8, mut im8, mut re8b, mut im8b, mut a) = (
                    vec![0.0f32; m * 8],
                    vec![0.0f32; m * 8],
                    vec![0.0f32; m * 8],
                    vec![0.0f32; m * 8],
                    vec![Complex::default(); m],
                );
                let mut r0 = 0usize;
                for chunk in buf.chunks_mut(width * 8) {
                    let nb = chunk.len() / width;
                    let mut bufs: Vec<&mut [Complex]> = chunk.chunks_mut(width).collect();
                    if nb == 8 {
                        wplan.run_bluestein8(
                            &mut bufs, &mut re8, &mut im8, &mut re8b, &mut im8b, mul, conj, r0,
                            width,
                        );
                    } else {
                        for (j, row) in bufs.into_iter().enumerate() {
                            wplan.run_bluestein(row, &mut a, mul, conj, r0 + j, width);
                        }
                    }
                    r0 += nb;
                }
            } else {
                for row in buf.chunks_exact_mut(width) {
                    wplan.run(row);
                }
            }
        }
    }
}

/// The out-of-place twin of [`fft2_rows_conj_mul`]: `src` rows are
/// gathered (with `conj·mul` fused) and transformed rows land in `dst` —
/// the caller's spectrum can stay shared read-only input.
#[allow(clippy::too_many_arguments)]
fn fft2_rows_conj_mul_io(
    src: &[Complex],
    dst: &mut [Complex],
    width: usize,
    height: usize,
    wplan: &Plan,
    mul: RowMul,
    conj: bool,
) {
    debug_assert_eq!(src.len(), width * height);
    debug_assert_eq!(dst.len(), width * height);
    if let PlanKind::Radix2 { mplan } = &wplan.kind {
        const RB: usize = 8;
        #[cfg(feature = "parallel")]
        if height >= 4 * RB {
            use rayon::prelude::*;
            dst.par_chunks_mut(width * RB)
                .zip(src.par_chunks(width * RB))
                .enumerate()
                .map_init(
                    || {
                        (
                            vec![0.0f32; width * RB],
                            vec![0.0f32; width * RB],
                            vec![0.0f32; width * RB],
                            vec![0.0f32; width * RB],
                        )
                    },
                    |(re8, im8, re8b, im8b), (ci, (dchunk, schunk))| {
                        let nb = dchunk.len() / width;
                        let r0 = ci * RB;
                        for x in 0..width {
                            let mut fj = [1.0f32; RB];
                            for (w, j) in fj.iter_mut().zip(0..nb) {
                                *w = mul.at(r0 + j, x, width);
                            }
                            for j in 0..nb {
                                let v = schunk[j * width + x];
                                let f = fj[j];
                                let im = if conj { -v.im } else { v.im };
                                re8[x * RB + j] = v.re * f;
                                im8[x * RB + j] = im * f;
                            }
                            for j in nb..RB {
                                re8[x * RB + j] = 0.0;
                                im8[x * RB + j] = 0.0;
                            }
                        }
                        fft_batch8(re8, im8, re8b, im8b, mplan);
                        for x in 0..width {
                            for j in 0..nb {
                                dchunk[j * width + x] =
                                    Complex::new(re8b[x * RB + j], im8b[x * RB + j]);
                            }
                        }
                    },
                )
                .for_each(|()| ());
            return;
        }
        let mut re8 = vec![0.0f32; width * RB];
        let mut im8 = vec![0.0f32; width * RB];
        let mut re8b = vec![0.0f32; width * RB];
        let mut im8b = vec![0.0f32; width * RB];
        let mut y = 0usize;
        while y < height {
            let nb = RB.min(height - y);
            for x in 0..width {
                let mut fj = [1.0f32; RB];
                for (w, j) in fj.iter_mut().zip(0..nb) {
                    *w = mul.at(y + j, x, width);
                }
                for j in 0..nb {
                    let v = src[(y + j) * width + x];
                    let f = fj[j];
                    let im = if conj { -v.im } else { v.im };
                    re8[x * RB + j] = v.re * f;
                    im8[x * RB + j] = im * f;
                }
                for j in nb..RB {
                    re8[x * RB + j] = 0.0;
                    im8[x * RB + j] = 0.0;
                }
            }
            fft_batch8(&re8, &im8, &mut re8b, &mut im8b, mplan);
            for x in 0..width {
                for j in 0..nb {
                    dst[(y + j) * width + x] = Complex::new(re8b[x * RB + j], im8b[x * RB + j]);
                }
            }
            y += nb;
        }
    } else if let PlanKind::Bluestein { m, .. } = &wplan.kind {
        let m = *m;
        // Parallel: disjoint src/dst chunk pairs are independent tasks.
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            dst.par_chunks_mut(width * 8)
                .zip(src.par_chunks(width * 8))
                .enumerate()
                .for_each_init(
                    || {
                        (
                            vec![0.0f32; m * 8],
                            vec![0.0f32; m * 8],
                            vec![0.0f32; m * 8],
                            vec![0.0f32; m * 8],
                            vec![Complex::default(); m],
                        )
                    },
                    |(re8, im8, re8b, im8b, a), (ci, (dchunk, schunk))| {
                        let nb = dchunk.len() / width;
                        let r0 = ci * 8;
                        if nb == 8 {
                            let sbufs: Vec<&[Complex]> = schunk.chunks(width).collect();
                            let mut dbufs: Vec<&mut [Complex]> = dchunk.chunks_mut(width).collect();
                            wplan.run_bluestein8_io(
                                &sbufs, &mut dbufs, re8, im8, re8b, im8b, mul, conj, r0, width,
                            );
                        } else {
                            for (j, (srow, drow)) in schunk
                                .chunks(width)
                                .zip(dchunk.chunks_mut(width))
                                .enumerate()
                            {
                                wplan.run_bluestein_io(srow, drow, a, mul, conj, r0 + j, width);
                            }
                        }
                    },
                );
        }
        #[cfg(not(feature = "parallel"))]
        {
            let (mut re8, mut im8, mut re8b, mut im8b, mut a) = (
                vec![0.0f32; m * 8],
                vec![0.0f32; m * 8],
                vec![0.0f32; m * 8],
                vec![0.0f32; m * 8],
                vec![Complex::default(); m],
            );
            let mut r0 = 0usize;
            for (dchunk, schunk) in dst.chunks_mut(width * 8).zip(src.chunks(width * 8)) {
                let nb = dchunk.len() / width;
                if nb == 8 {
                    let sbufs: Vec<&[Complex]> = schunk.chunks(width).collect();
                    let mut dbufs: Vec<&mut [Complex]> = dchunk.chunks_mut(width).collect();
                    wplan.run_bluestein8_io(
                        &sbufs, &mut dbufs, &mut re8, &mut im8, &mut re8b, &mut im8b, mul, conj,
                        r0, width,
                    );
                } else {
                    for (j, (srow, drow)) in schunk
                        .chunks(width)
                        .zip(dchunk.chunks_mut(width))
                        .enumerate()
                    {
                        wplan.run_bluestein_io(srow, drow, &mut a, mul, conj, r0 + j, width);
                    }
                }
                r0 += nb;
            }
        }
    } else {
        // Trivial length: the transform is the identity — just the
        // fused conj·f copy.
        for (i, (s, d)) in src.iter().zip(dst.iter_mut()).enumerate() {
            let f = mul.at(i / width, i % width, width);
            let im = if conj { -s.im } else { s.im };
            *d = Complex::new(s.re * f, im * f);
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
    fft2_cols_conj_scale(buf, width, height, hplan, None, None);
}

/// [`fft2_cols_planned`] with `conj·scale` fused into the transpose **out**:
/// with `scale` set it stores `(re·s, −im·s)`, which is `ifft2`'s trailing
/// `conj·(1/n)` — fused so it doesn't take its own pass over the buffer.
/// With `mag` set the scatter writes `|re·s, ±im·s|` into the `f32` plane
/// instead of the complex value into `buf` — `buf` is then pure working
/// storage and the separate magnitude pass never happens.
fn fft2_cols_conj_scale(
    buf: &mut [Complex],
    width: usize,
    height: usize,
    hplan: &Plan,
    scale: Option<f32>,
    mut mag: Option<&mut [f32]>,
) {
    const CB: usize = 8;
    if let PlanKind::Radix2 { mplan } = &hplan.kind {
        // Column stripes are disjoint writes but strided — not sliceable —
        // so the parallel path goes through a block-major SoA buffer in
        // bounded super-blocks (64 blocks = 512 columns ≈ 16 MB at 2048²):
        // gather (parallel over (block,row) cells), transform (parallel over
        // blocks), scatter (parallel over rows, fusing `conj·scale`).
        #[cfg(feature = "parallel")]
        if width >= 4 * CB && height >= 4 * CB {
            use rayon::prelude::*;
            const SB: usize = 64; // blocks per super-block
            // The super-block buffers are sized for a full sweep once and
            // reused — each iteration rewrites every cell it reads.
            let mut re_all = vec![0.0f32; SB * height * CB];
            let mut im_all = vec![0.0f32; SB * height * CB];
            let mut re_allb = vec![0.0f32; SB * height * CB];
            let mut im_allb = vec![0.0f32; SB * height * CB];
            let mut x0 = 0usize;
            while x0 < width {
                let nb_cols = (SB * CB).min(width - x0);
                let nblocks = nb_cols.div_ceil(CB);
                let re_all = &mut re_all[..nblocks * height * CB];
                let im_all = &mut im_all[..nblocks * height * CB];
                let re_allb = &mut re_allb[..nblocks * height * CB];
                let im_allb = &mut im_allb[..nblocks * height * CB];
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
                    .zip(re_allb.par_chunks_mut(height * CB))
                    .zip(im_allb.par_chunks_mut(height * CB))
                    .for_each(|(((rc, ic), rcb), icb)| fft_batch8(rc, ic, rcb, icb, mplan));
                let s = scale.unwrap_or(1.0);
                let negate = scale.is_some();
                if let Some(mag) = mag.as_deref_mut() {
                    mag.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
                        for b in 0..nblocks {
                            let base = (b * height + y) * CB;
                            for l in 0..CB {
                                let x = x0 + b * CB + l;
                                if x < width {
                                    let (re, im) = (re_allb[base + l], im_allb[base + l]);
                                    let (zr, zi) = (re * s, if negate { -im * s } else { im * s });
                                    row[x] = libm::sqrtf(zr * zr + zi * zi);
                                }
                            }
                        }
                    });
                } else {
                    buf.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
                        for b in 0..nblocks {
                            let base = (b * height + y) * CB;
                            for l in 0..CB {
                                let x = x0 + b * CB + l;
                                if x < width {
                                    let (re, im) = (re_allb[base + l], im_allb[base + l]);
                                    row[x] =
                                        Complex::new(re * s, if negate { -im * s } else { im * s });
                                }
                            }
                        }
                    });
                }
                x0 += nb_cols;
            }
            return;
        }
        let mut re8 = vec![0.0f32; height * CB];
        let mut im8 = vec![0.0f32; height * CB];
        let mut re8b = vec![0.0f32; height * CB];
        let mut im8b = vec![0.0f32; height * CB];
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
            fft_batch8(&re8, &im8, &mut re8b, &mut im8b, mplan);
            if let Some(mag) = mag.as_deref_mut() {
                let s = scale.unwrap_or(1.0);
                let negate = scale.is_some();
                for y in 0..height {
                    for j in 0..nb {
                        let (re, im) = (re8b[y * CB + j], im8b[y * CB + j]);
                        let (zr, zi) = (re * s, if negate { -im * s } else { im * s });
                        mag[y * width + x + j] = libm::sqrtf(zr * zr + zi * zi);
                    }
                }
            } else {
                match scale {
                    Some(s) => {
                        for y in 0..height {
                            for j in 0..nb {
                                buf[y * width + x + j] =
                                    Complex::new(re8b[y * CB + j] * s, -im8b[y * CB + j] * s);
                            }
                        }
                    }
                    None => {
                        for y in 0..height {
                            for j in 0..nb {
                                buf[y * width + x + j] =
                                    Complex::new(re8b[y * CB + j], im8b[y * CB + j]);
                            }
                        }
                    }
                }
            }
            x += nb;
        }
        return;
    }

    // Bluestein columns: strided writes can't be borrowed as disjoint
    // slices, so under `parallel` this mirrors the radix-2 super-block
    // structure — gather+transform per column-block into a contiguous SoA
    // buffer (parallel over blocks), then scatter back over rows (parallel,
    // fusing `conj·scale`).
    #[cfg(feature = "parallel")]
    if let PlanKind::Bluestein { m, .. } = &hplan.kind
        && width >= 4 * CB
        && height >= 4 * CB
    {
        use rayon::prelude::*;
        let m = *m;
        const SB: usize = 64;
        let s = scale.unwrap_or(1.0);
        let negate = scale.is_some();
        // Reused across super-blocks — every cell is regathered each pass.
        let mut cols_all = vec![Complex::default(); SB * height * CB];
        let mut x0 = 0usize;
        while x0 < width {
            let nb_cols = (SB * CB).min(width - x0);
            let nblocks = nb_cols.div_ceil(CB);
            let cols_all = &mut cols_all[..nblocks * height * CB];
            {
                let ro: &[Complex] = buf;
                cols_all
                    .par_chunks_mut(height * CB)
                    .enumerate()
                    .for_each_init(
                        || {
                            (
                                vec![0.0f32; m * CB],
                                vec![0.0f32; m * CB],
                                vec![0.0f32; m * CB],
                                vec![0.0f32; m * CB],
                                vec![Complex::default(); m],
                            )
                        },
                        |(re8, im8, re8b, im8b, a), (b, block)| {
                            let bx = x0 + b * CB;
                            let nb = CB.min(width - bx);
                            if nb == CB {
                                hplan.run_bluestein8_cols(
                                    ro, width, bx, block, re8, im8, re8b, im8b,
                                );
                            } else {
                                for y0 in (0..height).step_by(8) {
                                    cols_gather_tile(
                                        ro,
                                        width,
                                        bx,
                                        y0,
                                        (height - y0).min(8),
                                        nb,
                                        block,
                                        height,
                                    );
                                }
                                for c in block.chunks_mut(height).take(nb) {
                                    hplan.run_bluestein(c, a, RowMul::Id, false, 0, 0);
                                }
                            }
                        },
                    );
            }
            if let Some(mag) = mag.as_deref_mut() {
                mag.par_chunks_mut(width * 8)
                    .enumerate()
                    .for_each(|(g, rows)| {
                        let y0 = g * 8;
                        for b in 0..nblocks {
                            let bx = x0 + b * CB;
                            let nb = CB.min(width - bx);
                            cols_scatter_tile_mag(
                                &cols_all[b * CB * height..],
                                height,
                                y0,
                                rows.len() / width,
                                nb,
                                s,
                                negate,
                                rows,
                                width,
                                bx,
                            );
                        }
                    });
            } else {
                buf.par_chunks_mut(width * 8)
                    .enumerate()
                    .for_each(|(g, rows)| {
                        let y0 = g * 8;
                        for b in 0..nblocks {
                            let bx = x0 + b * CB;
                            let nb = CB.min(width - bx);
                            cols_scatter_tile(
                                &cols_all[b * CB * height..],
                                height,
                                y0,
                                rows.len() / width,
                                nb,
                                s,
                                negate,
                                rows,
                                width,
                                bx,
                            );
                        }
                    });
            }
            x0 += nb_cols;
        }
        return;
    }

    let mut cols = vec![Complex::default(); height * CB];
    // The CB-column blocks are exactly the batch-8 width, so the serial
    // Bluestein path transforms each block through `run_bluestein8` (one
    // scratch set reused across the sweep) instead of `run` per column.
    let mut blue = if let PlanKind::Bluestein { m, .. } = &hplan.kind {
        let m = *m;
        Some((
            vec![0.0f32; m * CB],
            vec![0.0f32; m * CB],
            vec![0.0f32; m * CB],
            vec![0.0f32; m * CB],
            vec![Complex::default(); m],
        ))
    } else {
        None
    };
    let mut x = 0usize;
    while x < width {
        let nb = CB.min(width - x);
        match &mut blue {
            Some((re8, im8, re8b, im8b, _)) if nb == CB => {
                // Full block: gather straight out of the plane — the
                // transformed columns land back in `cols` for the
                // scatter sweep below.
                hplan.run_bluestein8_cols(buf, width, x, &mut cols, re8, im8, re8b, im8b);
            }
            Some((_, _, _, _, a)) => {
                for y0 in (0..height).step_by(8) {
                    cols_gather_tile(
                        buf,
                        width,
                        x,
                        y0,
                        (height - y0).min(8),
                        nb,
                        &mut cols,
                        height,
                    );
                }
                for c in cols.chunks_mut(height).take(nb) {
                    hplan.run_bluestein(c, a, RowMul::Id, false, 0, 0);
                }
            }
            None => {
                for y0 in (0..height).step_by(8) {
                    cols_gather_tile(
                        buf,
                        width,
                        x,
                        y0,
                        (height - y0).min(8),
                        nb,
                        &mut cols,
                        height,
                    );
                }
                for c in cols.chunks_exact_mut(height).take(nb) {
                    hplan.run(c);
                }
            }
        }
        let s = scale.unwrap_or(1.0);
        let negate = scale.is_some();
        if let Some(mag) = mag.as_deref_mut() {
            for y0 in (0..height).step_by(8) {
                let nr = (height - y0).min(8);
                cols_scatter_tile_mag(
                    &cols,
                    height,
                    y0,
                    nr,
                    nb,
                    s,
                    negate,
                    &mut mag[y0 * width..],
                    width,
                    x,
                );
            }
        } else {
            for y0 in (0..height).step_by(8) {
                let nr = (height - y0).min(8);
                cols_scatter_tile(
                    &cols,
                    height,
                    y0,
                    nr,
                    nb,
                    s,
                    negate,
                    &mut buf[y0 * width..],
                    width,
                    x,
                );
            }
        }
        x += nb;
    }
}

/// Inverse 2D DFT (normalised by `1/(width·height)`), in place.
#[allow(dead_code)] // public mirror of the vendored API; callers use the _planned forms
pub fn ifft2(buf: &mut [Complex], width: usize, height: usize) {
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
    fft2_rows_conj_mul(buf, width, height, wplan, RowMul::Id, true);
    fft2_cols_conj_scale(
        buf,
        width,
        height,
        hplan,
        Some(1.0 / (width * height) as f32),
        None,
    );
}

/// [`ifft2_planned`] with a real (r-space) filter multiply fused into the
/// row transpose-in: element `i` is transformed as `conj(buf[i]·mul[i])`,
/// which is `conj(buf[i])·mul[i]` for real `mul` — identical arithmetic
/// to multiplying then running [`ifft2_planned`], minus the multiply's own
/// pass over the buffer. The [`RowMul`] shifted views also remove the
/// weight-plane materialization entirely.
pub(crate) fn ifft2_planned_mul(
    buf: &mut [Complex],
    width: usize,
    height: usize,
    wplan: &Plan,
    hplan: &Plan,
    mul: RowMul,
) {
    debug_assert_eq!(buf.len(), width * height);
    fft2_rows_conj_mul(buf, width, height, wplan, mul, true);
    fft2_cols_conj_scale(
        buf,
        width,
        height,
        hplan,
        Some(1.0 / (width * height) as f32),
        None,
    );
}

/// [`ifft2_planned_mul`] out-of-place and magnitude-producing: `src` stays
/// untouched shared input, `dst` is caller-owned complex scratch, and the
/// column scatter writes `|z|` into `mag` — so a filtered inverse transform
/// plus the `sqrt(re²+im²)` plane cost one copy's worth of traffic total.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ifft2_planned_mul_mag_into(
    src: &[Complex],
    dst: &mut [Complex],
    mag: &mut [f32],
    width: usize,
    height: usize,
    wplan: &Plan,
    hplan: &Plan,
    mul: RowMul,
) {
    debug_assert_eq!(src.len(), width * height);
    debug_assert_eq!(dst.len(), width * height);
    debug_assert_eq!(mag.len(), width * height);
    fft2_rows_conj_mul_io(src, dst, width, height, wplan, mul, true);
    fft2_cols_conj_scale(
        dst,
        width,
        height,
        hplan,
        Some(1.0 / (width * height) as f32),
        Some(mag),
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
        // Deterministic pseudo-random-ish input; no dependency needed.
        let mut s = 0x2545_F491_4F6C_DD1Du64;
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

    /// The scalar mixed-radix path against the naive DFT — the
    /// Bluestein `m` values hit every radix combination ([2,3], [2,2,3],
    /// pure 3, multiple 3s).
    #[test]
    fn fft_mixed_matches_the_naive_dft() {
        for n in [6usize, 12, 9, 18] {
            let mplan = MPlan::smooth(n);
            let x = seeded(n);
            let want = naive_dft(&x);
            let mut got = x.clone();
            fft_mixed_planned(&mut got, &mplan);
            for (k, (a, b)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (a.re - b.re).abs() < 1e-4 && (a.im - b.im).abs() < 1e-4,
                    "n={n} k={k}: {a:?} != {b:?} radices={:?} perm={:?}",
                    mplan.radices,
                    mplan.perm
                );
            }
        }
    }

    /// `fft_batch8` vs the scalar [`fft_mixed_planned`] — eight distinct
    /// lane inputs, so a broken row gather can't hide behind identical
    /// data (the failure mode that once masked a bad permutation). The
    /// mixed sizes (6, 12, 18, 48, 216…) exercise the radix-3 stages and
    /// the digit-reversal permutation.
    #[test]
    fn batch8_matches_scalar_per_lane() {
        for n in [4usize, 8, 16, 32, 64, 6, 12, 18, 48, 216, 6 * 8, 3 * 81] {
            let mplan = if n.is_power_of_two() {
                MPlan::pow2(n)
            } else {
                MPlan::smooth(n)
            };
            let xs: Vec<Vec<Complex>> = (0..8)
                .map(|j| {
                    seeded(n)
                        .iter()
                        .map(|v| Complex::new(v.re + j as f32, v.im - j as f32))
                        .collect()
                })
                .collect();
            let wants: Vec<Vec<Complex>> = xs
                .iter()
                .map(|x| {
                    let mut w = x.clone();
                    fft_mixed_planned(&mut w, &mplan);
                    w
                })
                .collect();
            let mut re8 = vec![0f32; n * 8];
            let mut im8 = vec![0f32; n * 8];
            let mut re8b = vec![0f32; n * 8];
            let mut im8b = vec![0f32; n * 8];
            for k in 0..n {
                for j in 0..8 {
                    re8[k * 8 + j] = xs[j][k].re;
                    im8[k * 8 + j] = xs[j][k].im;
                }
            }
            fft_batch8(&re8, &im8, &mut re8b, &mut im8b, &mplan);
            for k in 0..n {
                for j in 0..8 {
                    let d = (re8b[k * 8 + j] - wants[j][k].re).abs()
                        + (im8b[k * 8 + j] - wants[j][k].im).abs();
                    assert!(
                        d < 1e-4,
                        "n={n} k={k} j={j}: got ({},{}) want ({},{})",
                        re8b[k * 8 + j],
                        im8b[k * 8 + j],
                        wants[j][k].re,
                        wants[j][k].im
                    );
                }
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
            for (i, (a, b)) in y.iter().zip(&x).enumerate() {
                assert!(
                    (a.re - b.re).abs() < 1e-4 && (a.im - b.im).abs() < 1e-4,
                    "{w}x{h} i={i}: {a:?} vs {b:?}"
                );
            }
        }
    }
}
