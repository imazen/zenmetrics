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
//!
//! ### Precision
//!
//! The transforms run in `f32` — the fleet-speed rewrite: plane data through
//! the whole pipeline is `f32` while reductions and one-time table/grid
//! construction stay `f64`. Twiddles are still computed in `f64` (`expi`)
//! and cast down, so the error vs the previous `f64` build is plane-quantisation
//! only, not a different algorithm. `cycles_per_degree_grid` stays `f64` for
//! the same reason (built once per call, scalar).

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
#[archmage::magetypes(define(f32x8), +v4, +v4x, +v3, +neon, +wasm128, +scalar)]
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
            let a = f32x8::load(token, (&re8[i * 8..i * 8 + 8]).try_into().unwrap());
            let b = f32x8::load(token, (&re8[j * 8..j * 8 + 8]).try_into().unwrap());
            b.store((&mut re8[i * 8..i * 8 + 8]).try_into().unwrap());
            a.store((&mut re8[j * 8..j * 8 + 8]).try_into().unwrap());
            let a = f32x8::load(token, (&im8[i * 8..i * 8 + 8]).try_into().unwrap());
            let b = f32x8::load(token, (&im8[j * 8..j * 8 + 8]).try_into().unwrap());
            b.store((&mut im8[i * 8..i * 8 + 8]).try_into().unwrap());
            a.store((&mut im8[j * 8..j * 8 + 8]).try_into().unwrap());
        }
    }

    for tw in stages {
        let half = tw.len();
        let len = half * 2;
        for start in (0..n).step_by(len) {
            for (k, &w) in tw.iter().enumerate() {
                let a = (start + k) * 8;
                let b = (start + k + half) * 8;
                let wre = f32x8::splat(token, w.re);
                let wim = f32x8::splat(token, w.im);
                let ure = f32x8::load(token, (&re8[a..a + 8]).try_into().unwrap());
                let uim = f32x8::load(token, (&im8[a..a + 8]).try_into().unwrap());
                let vre = f32x8::load(token, (&re8[b..b + 8]).try_into().unwrap());
                let vim = f32x8::load(token, (&im8[b..b + 8]).try_into().unwrap());
                let tre = vre * wre - vim * wim;
                let tim = vre * wim + vim * wre;
                (ure + tre).store((&mut re8[a..a + 8]).try_into().unwrap());
                (uim + tim).store((&mut im8[a..a + 8]).try_into().unwrap());
                (ure - tre).store((&mut re8[b..b + 8]).try_into().unwrap());
                (uim - tim).store((&mut im8[b..b + 8]).try_into().unwrap());
            }
        }
    }
}

/// [`fft_batch8_inner`] with runtime tier dispatch.
fn fft_batch8(re8: &mut [f32], im8: &mut [f32], n: usize, stages: &[Vec<Complex>]) {
    archmage::incant!(
        fft_batch8_inner(re8, im8, n, stages),
        [v4x, v4, v3, neon, wasm128, scalar]
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
pub fn fft2(buf: &mut [Complex], width: usize, height: usize) {
    let (wplan, hplan) = (Plan::new(width), Plan::new(height));
    fft2_planned(buf, width, height, &wplan, &hplan);
}

/// [`fft2`] with the two 1-D plans supplied, so callers doing several
/// same-size transforms (or a forward + inverse pair) build them once.
fn fft2_planned(buf: &mut [Complex], width: usize, height: usize, wplan: &Plan, hplan: &Plan) {
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
        if height >= 4 * RB {
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
        if conj || filter.is_some() {
            for (i, v) in buf.iter_mut().enumerate() {
                let f = filter.map_or(1.0, |f| f[i]);
                let im = if conj { -v.im } else { v.im };
                *v = Complex::new(v.re * f, im * f);
            }
        }
        for row in buf.chunks_exact_mut(width) {
            wplan.run(row);
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
    // The trailing `conj·scale` applies to the TRANSFORMED data — on the
    // scalar fallback it stays a separate final pass.
    if let Some(s) = scale {
        for v in buf.iter_mut() {
            *v = Complex::new(v.re, -v.im).scale(s);
        }
    }
}

/// Inverse 2D DFT (normalised by `1/(width·height)`), in place.
pub fn ifft2(buf: &mut [Complex], width: usize, height: usize) {
    let (wplan, hplan) = (Plan::new(width), Plan::new(height));
    ifft2_planned(buf, width, height, &wplan, &hplan);
}

/// [`ifft2`] with the plans supplied — same conjugate–forward–conjugate
/// construction, same normalisation. The leading `conj` rides the row
/// transpose-in and `conj·(1/n)` rides the column transpose-out, so the
/// radix-2 path never touches the buffer outside a transform pass.
fn ifft2_planned(buf: &mut [Complex], width: usize, height: usize, wplan: &Plan, hplan: &Plan) {
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

/// Radial spatial frequency, in cycles per degree, for every bin of a
/// `height × width` DFT of an image sampled at `pix_per_deg` pixels/degree.
///
/// Row-major, in DFT bin order (bin 0 is DC, the second half holds the negative
/// frequencies). The value at bin `(y, x)` is `sqrt(fx² + fy²)` where
/// `fx = fftfreq(x, width) · pix_per_deg`.
///
/// Built once per call and feeds the MTF filter (itself evaluated once), so it
/// stays `f64` — the `f32` plane conversion happens at convolution time.
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
    x: &[f32],
    width: usize,
    height: usize,
    filter: &[f32],
    pad_w: usize,
    pad_h: usize,
    pad_value: f32,
) -> Vec<f32> {
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
    // identical. Transform one and copy it into the rest, then run the column
    // pass as usual. Image rows go through the batched `fft_batch8` path.
    fft2_rows_planned(&mut buf[..height * pad_w], pad_w, height, &wplan);
    if height < pad_h {
        let (first, rest) = buf[height * pad_w..].split_at_mut(pad_w);
        wplan.run(first);
        for row in rest.chunks_exact_mut(pad_w) {
            row.copy_from_slice(first);
        }
    }
    fft2_cols_planned(&mut buf, pad_w, pad_h, &hplan);

    // The filter multiply, the leading `conj`, and the trailing
    // `conj·(1/n)` all ride the batched transposes — no separate buffer
    // passes between the forward and inverse transforms.
    fft2_rows_conj_mul(&mut buf, pad_w, pad_h, &wplan, Some(filter), true);
    fft2_cols_conj_scale(
        &mut buf,
        pad_w,
        pad_h,
        &hplan,
        Some(1.0 / (pad_w * pad_h) as f32),
    );

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
        let x: Vec<f32> = (0..w * h).map(|i| (i as f32 * 0.37).sin()).collect();
        let filter = vec![1.0f32; (2 * w) * (2 * h)];
        let got = conv_fft_real(&x, w, h, &filter, 2 * w, 2 * h, 0.0);
        for (a, b) in got.iter().zip(&x) {
            assert!((a - b).abs() < 1e-3, "{a} != {b}");
        }
    }

    #[test]
    fn conv_with_a_dc_only_filter_returns_the_padded_mean() {
        // A filter that keeps only bin 0 replaces every output pixel with the
        // mean over the PADDED domain — which is how the constant pad value
        // enters the result.
        let (w, h) = (4usize, 4usize);
        let (pw, ph) = (2 * w, 2 * h);
        let x: Vec<f32> = (0..w * h).map(|i| i as f32).collect();
        let pad_value = 7.0f32;
        let mut filter = vec![0.0f32; pw * ph];
        filter[0] = 1.0;
        let got = conv_fft_real(&x, w, h, &filter, pw, ph, pad_value);
        let mean = (x.iter().map(|v| *v as f64).sum::<f64>()
            + pad_value as f64 * (pw * ph - w * h) as f64)
            / (pw * ph) as f64;
        for (a, _) in got.iter().zip(&x) {
            assert!((*a as f64 - mean).abs() < 1e-4, "{a} != {mean}");
        }
    }
}
