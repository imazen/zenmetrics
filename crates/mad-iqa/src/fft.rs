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
        let PlanKind::Bluestein {
            chirp,
            bf,
            m,
            mplan,
        } = &self.kind
        else {
            unreachable!("run_bluestein8 on non-Bluestein plan")
        };
        let (n, m) = (self.n, *m);
        debug_assert_eq!(bufs.len(), 8);
        debug_assert!(
            re8.len() >= m * 8 && im8.len() >= m * 8 && re8b.len() >= m * 8 && im8b.len() >= m * 8
        );
        // Gather: a[k] = (conj·f)(buf[k])·chirp[k], zeroed past n — 8×8
        // k-blocks keep the transposed load in registers (scalar tail for
        // m % 8 ≠ 0).
        let mb = m & !7;
        for k0 in (0..mb).step_by(8) {
            blue8_gather(bufs, n, k0, chirp, re8, im8, mul, conj, r0, width);
        }
        for k in mb..m {
            for (j, buf) in bufs.iter().enumerate() {
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
        let (re8, im8, re8b, im8b) = (
            &mut re8[..m * 8],
            &mut im8[..m * 8],
            &mut re8b[..m * 8],
            &mut im8b[..m * 8],
        );
        fft_batch8(re8, im8, re8b, im8b, m, mplan);
        blue8_bf(re8b, im8b, bf);
        fft_batch8(re8b, im8b, re8, im8, m, mplan);
        let s = 1.0 / m as f32;
        for k0 in (0..n).step_by(8) {
            blue8_scatter(bufs, n, k0, chirp, s, re8, im8);
        }
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
        let PlanKind::Bluestein {
            chirp,
            bf,
            m,
            mplan,
        } = &self.kind
        else {
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
        fft_radix2_planned(&mut a[..*m], mplan);
        for (x, y) in a[..*m].iter_mut().zip(bf) {
            *x = x.mul(*y);
        }
        for v in a[..*m].iter_mut() {
            *v = v.conj();
        }
        fft_radix2_planned(&mut a[..*m], mplan);
        let s = 1.0 / *m as f32;
        for v in a[..*m].iter_mut() {
            *v = v.conj().scale(s);
        }
        for (k, out) in buf.iter_mut().enumerate() {
            *out = a[k].mul(chirp[k]);
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
    n: usize,
    stages: &[Vec<Complex>],
) {
    debug_assert_eq!(src_re.len(), n * 8);
    debug_assert_eq!(src_im.len(), n * 8);
    debug_assert_eq!(dst_re.len(), n * 8);
    debug_assert_eq!(dst_im.len(), n * 8);

    let bits = n.trailing_zeros();
    let mut s = 0usize;

    if stages.len() >= 2 {
        // The bit-reversal rides the first fused pair: logical slot `p`
        // holds `src[rev(p)]`, so loads are permuted row gathers while
        // stores stay contiguous — the src→dst split makes the gather
        // alias-safe where an in-place version would read rows it had
        // already overwritten. Stage-0/1 twiddles are exactly (1,0) and
        // {(1,0), w1[1]≈−i}, so the pass is adds, subs, and one twiddle
        // multiply per 4 elements.
        let tw1 = &stages[1];
        // Row gather for one 4-element block: `src[rev(start+q)]`.
        let load4 = |start: usize| {
            let rev = |q: usize| (((start + q) as u32).reverse_bits() >> (32 - bits)) as usize * 8;
            let (p0, p1, p2, p3) = (rev(0), rev(1), rev(2), rev(3));
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
        // n <= 2: bit-reversed gather into dst; the single-stage loop
        // below applies the one twiddle.
        for i in 0..n {
            let j = ((i as u32).reverse_bits() >> (32 - bits)) as usize;
            let a = f32x8::load_t(token, (&src_re[j * 8..j * 8 + 8]).try_into().unwrap());
            a.store((&mut dst_re[i * 8..i * 8 + 8]).try_into().unwrap());
            let a = f32x8::load_t(token, (&src_im[j * 8..j * 8 + 8]).try_into().unwrap());
            a.store((&mut dst_im[i * 8..i * 8 + 8]).try_into().unwrap());
        }
    }

    // Consecutive stage pairs run fused (radix-4-equivalent): the two
    // butterflies land on the same registers, so each element loads and
    // stores once per pair instead of once per stage — identical
    // arithmetic, half the passes over the planes. An odd stage count
    // leaves the final stage to the single-stage loop below.
    while s + 1 < stages.len() {
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
    }
    for tw in &stages[s..] {
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
    }
}

/// [`fft_batch8_inner`] with runtime tier dispatch.
fn fft_batch8(
    src_re: &[f32],
    src_im: &[f32],
    dst_re: &mut [f32],
    dst_im: &mut [f32],
    n: usize,
    stages: &[Vec<Complex>],
) {
    archmage::attuned!(
        fft_batch8_inner(src_re, src_im, dst_re, dst_im, n, stages),
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
    bufs: &[&mut [Complex]],
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
            let row = &*bufs[j];
            for c in 0..8 {
                let z = row[k0 + c];
                b.0[c] = z.re * frow[c];
                b.1[c] = (if conj { -z.im } else { z.im }) * frow[c];
            }
        }
    } else {
        for (j, (b, frow)) in br.iter_mut().zip(bi.iter_mut()).zip(fm.iter()).enumerate() {
            let row = &*bufs[j];
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
            for k in 0..8 {
                col[k] = src_c[k];
            }
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
    if let PlanKind::Radix2 { stages } = &wplan.kind {
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
                        fft_batch8(re8, im8, re8b, im8b, width, stages);
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
            fft_batch8(&re8, &im8, &mut re8b, &mut im8b, width, stages);
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
                    .for_each(|(((rc, ic), rcb), icb)| {
                        fft_batch8(rc, ic, rcb, icb, height, stages)
                    });
                let s = scale.unwrap_or(1.0);
                let negate = scale.is_some();
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
            fft_batch8(&re8, &im8, &mut re8b, &mut im8b, height, stages);
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
                            let mut bufs: Vec<&mut [Complex]> =
                                block.chunks_mut(height).take(nb).collect();
                            if nb == CB {
                                hplan.run_bluestein8(
                                    &mut bufs,
                                    re8,
                                    im8,
                                    re8b,
                                    im8b,
                                    RowMul::Id,
                                    false,
                                    0,
                                    0,
                                );
                            } else {
                                for c in bufs {
                                    hplan.run_bluestein(c, a, RowMul::Id, false, 0, 0);
                                }
                            }
                        },
                    );
            }
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
        match &mut blue {
            Some((re8, im8, re8b, im8b, a)) => {
                let mut bufs: Vec<&mut [Complex]> = cols.chunks_mut(height).take(nb).collect();
                if nb == CB {
                    hplan.run_bluestein8(&mut bufs, re8, im8, re8b, im8b, RowMul::Id, false, 0, 0);
                } else {
                    for c in bufs {
                        hplan.run_bluestein(c, a, RowMul::Id, false, 0, 0);
                    }
                }
            }
            None => {
                for c in cols.chunks_exact_mut(height).take(nb) {
                    hplan.run(c);
                }
            }
        }
        let s = scale.unwrap_or(1.0);
        let negate = scale.is_some();
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

    /// `fft_batch8` vs the scalar [`fft_radix2_planned`] — eight distinct
    /// lane inputs, so a broken row gather can't hide behind identical
    /// data (the failure mode that once masked a bad permutation).
    #[test]
    fn batch8_matches_scalar_per_lane() {
        for n in [4usize, 8, 16, 32, 64] {
            let stages = radix2_stages(n);
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
                    fft_radix2_planned(&mut w, &stages);
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
            fft_batch8(&re8, &im8, &mut re8b, &mut im8b, n, &stages);
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
