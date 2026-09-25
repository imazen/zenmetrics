//! HaarPSI core: padded-plane Haar decomposition, similarity maps and
//! weighted-logistic pooling — the exact computation `HaarPSI.m`
//! performs, organized as f32 plane sweeps plus an f64 raster-order
//! pooling fold.
//!
//! # conv2 'same' semantics (verified against Octave)
//!
//! For an even `n×n` kernel, MATLAB's `'same'` anchors the *flipped*
//! kernel so that `out(i,j) = Σ_{u,v} K(u,v)·X(i+s−u, j+s−v)`
//! (0-based, `s = n/2`, `X = 0` outside). The scale-k Haar filter is
//! separable — `K(u,v) = f(u)` with `f = −2^-k` on the first `2^(k-1)`
//! rows and `+2^-k` on the rest — so each orientation reduces to a box
//! sum plus a difference of box sums:
//!
//! ```text
//! V(i,j) = 2^-k · Σ_{c=j-h+1}^{j+h} (CS(i,c) − CS(i+h,c)),
//!          CS(r,c) = Σ_{a=r-h+1}^{r} X(a,c)         (h = 2^(k-1))
//! H(i,j) = 2^-k · Σ_{a=i-h+1}^{i+h} (RS(a,j) − RS(a,j+h)),
//!          RS(r,c) = Σ_{b=c-h+1}^{c} X(r,b)
//! ```
//!
//! CS needs `h` extra computed rows below the image (partial boxes);
//! RS needs `h` extra computed cols right of it. Every other out-of-
//! range access is a genuine zero — the padded planes supply it.
//!
//! # Layout and floating-point order
//!
//! `Plane` rows live inside a `PAD`-wide zero frame (plus `HMAX`
//! writable rows below the image), so shifted loads never bounds-check
//! and never branch. Every output pixel is a fixed-order f32
//! expression independent of the lane count, and the pooling keeps
//! gmsd's fixed 8-lane f64 accumulator grouping — every tier is
//! bit-identical. Plain `*`/`+` throughout — never `mul_add`.

use alloc::vec::Vec;

/// Zero frame around every plane's computed region — `>=` the largest
/// filter half-size (`2^2 = 4`) plus vector-tail margin.
pub(crate) const PAD: usize = 8;
/// Extra writable rows below the image (CS of the coarsest scale
/// computes `h_max = 4` partial-box rows there).
pub(crate) const HMAX: usize = 4;

/// `C` — the reference's similarity constant.
const C30: f32 = 30.0;
/// `alpha` — the reference's logistic slope.
const ALPHA: f32 = 4.2;
/// `2^-k` for scales k = 1, 2, 3.
const SCALE: [f32; 3] = [0.5, 0.25, 0.125];

/// A strided f32 image plane inside a zero frame: image row `r` /
/// column `c` lives at `buf[(r + PAD) * stride + PAD + c]`; `stride =
/// w + 2·PAD`, `buf` holds `h + 2·PAD + HMAX` rows, all initialized to
/// zero. Reads of rows `[-PAD, h + PAD + HMAX)` are always in bounds;
/// rows/cols outside the computed region are zero.
#[doc(hidden)]
pub struct Plane {
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) stride: usize,
    pub(crate) buf: Vec<f32>,
}

impl Plane {
    /// Zero-filled padded plane for a `w`×`h` image.
    pub fn new(w: usize, h: usize) -> Self {
        let stride = w + 2 * PAD;
        let rows = h + 2 * PAD + HMAX;
        Self {
            w,
            h,
            stride,
            buf: alloc::vec![0.0; rows * stride],
        }
    }

    /// Padded row `r` (image coordinates; `r` may range over
    /// `[-PAD, h + PAD + HMAX)`). Includes the zero pads, so shifted
    /// loads at `±HMAX` columns stay in bounds.
    #[inline]
    pub fn prow(&self, r: isize) -> &[f32] {
        let i = (r + PAD as isize) as usize;
        &self.buf[i * self.stride..(i + 1) * self.stride]
    }

    /// Mutable padded row `r` — same domain as [`Plane::prow`].
    #[inline]
    pub fn prow_mut(&mut self, r: isize) -> &mut [f32] {
        let i = (r + PAD as isize) as usize;
        &mut self.buf[i * self.stride..(i + 1) * self.stride]
    }

    /// Inner `w`-wide view of image row `r`.
    #[inline]
    pub fn data(&self, r: usize) -> &[f32] {
        let i = (r + PAD) * self.stride + PAD;
        &self.buf[i..i + self.w]
    }

    /// Calls `f(buf_row_index, row)` for buf rows covering image rows
    /// `[r0, r1)` — sequentially, or on a rayon pool when `parallel`
    /// is enabled. Each pass writes disjoint buf rows, so results are
    /// bit-identical at every thread count.
    pub fn for_rows(&mut self, r0: isize, r1: isize, f: impl Fn(usize, &mut [f32]) + Sync) {
        let stride = self.stride;
        let lo = (r0 + PAD as isize) as usize;
        let hi = (r1 + PAD as isize) as usize;
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            self.buf
                .par_chunks_mut(stride)
                .enumerate()
                .for_each(|(ri, row)| {
                    if ri >= lo && ri < hi {
                        f(ri, row);
                    }
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for (ri, row) in self.buf.chunks_mut(stride).enumerate() {
                if ri >= lo && ri < hi {
                    f(ri, row);
                }
            }
        }
    }
}

// ------------------------------------------------------------------
// Scalar preparation (identical on every tier — no token needed).
// ------------------------------------------------------------------

/// f32 plane → padded plane, optionally 2× subsampled.
///
/// The reference's `HaarPSISubsample` is `conv2(ones(2,2)/4,'same')`
/// decimated at the odd (1-based) indices — a forward-looking 2×2 box
/// mean, zeros outside, exactly `out(y,x) = (X(2y,2x) + X(2y,2x+1) +
/// X(2y+1,2x) + X(2y+1,2x+1))/4`.
pub(crate) fn plane_from_f32(
    src: &[f32],
    w: usize,
    h: usize,
    stride: usize,
    subsample: bool,
) -> Plane {
    if !subsample {
        let mut p = Plane::new(w, h);
        for y in 0..h {
            let dst = p.prow_mut(y as isize);
            dst[PAD..PAD + w].copy_from_slice(&src[y * stride..y * stride + w]);
        }
        return p;
    }
    let (w2, h2) = (w.div_ceil(2), h.div_ceil(2));
    let mut p = Plane::new(w2, h2);
    for y in 0..h2 {
        let r0 = y * 2;
        let r1 = r0 + 1;
        let has_r1 = r1 < h;
        let dst = p.prow_mut(y as isize);
        for x in 0..w2 {
            let c0 = 2 * x;
            let right = c0 + 1 < w;
            let (a, b) = (
                src[r0 * stride + c0],
                if right {
                    src[r0 * stride + c0 + 1]
                } else {
                    0.0
                },
            );
            let (c, d) = if has_r1 {
                (
                    src[r1 * stride + c0],
                    if right {
                        src[r1 * stride + c0 + 1]
                    } else {
                        0.0
                    },
                )
            } else {
                (0.0, 0.0)
            };
            dst[PAD + x] = ((a + b) + (c + d)) * 0.25;
        }
    }
    p
}

/// sRGB8 → padded coefficient-weighted plane, optionally subsampled.
///
/// `coef` is one of the reference's YIQ rows applied to the unrounded
/// channel means: `out = (c0·mR + c1·mG) + c2·mB` computed in f64 and
/// stored f32. Subsampled output averages the four u8 channels first —
/// the same value the reference's "transform then box-mean" produces,
/// modulo f64 rounding of the mean.
pub(crate) fn plane_from_rgb8(
    rgb: &[u8],
    w: usize,
    h: usize,
    stride: usize,
    coef: [f64; 3],
    subsample: bool,
) -> Plane {
    let dot =
        |r: f64, g: f64, b: f64| -> f32 { ((coef[0] * r + coef[1] * g) + coef[2] * b) as f32 };
    if !subsample {
        let mut p = Plane::new(w, h);
        for y in 0..h {
            let row = &rgb[y * stride..y * stride + 3 * w];
            let dst = p.prow_mut(y as isize);
            for x in 0..w {
                dst[PAD + x] = dot(
                    row[3 * x] as f64,
                    row[3 * x + 1] as f64,
                    row[3 * x + 2] as f64,
                );
            }
        }
        return p;
    }
    let (w2, h2) = (w.div_ceil(2), h.div_ceil(2));
    let mut p = Plane::new(w2, h2);
    const Z: [u8; 3] = [0, 0, 0];
    for y in 0..h2 {
        let r0 = y * 2;
        let r1 = r0 + 1;
        let has_r1 = r1 < h;
        let dst = p.prow_mut(y as isize);
        for x in 0..w2 {
            let c0 = 2 * x;
            let right = c0 + 1 < w;
            let p00 = &rgb[r0 * stride + 3 * c0..r0 * stride + 3 * c0 + 3];
            let p01: &[u8] = if right {
                &rgb[r0 * stride + 3 * c0 + 3..r0 * stride + 3 * c0 + 6]
            } else {
                &Z
            };
            let (p10, p11): (&[u8], &[u8]) = if has_r1 {
                (
                    &rgb[r1 * stride + 3 * c0..r1 * stride + 3 * c0 + 3],
                    if right {
                        &rgb[r1 * stride + 3 * c0 + 3..r1 * stride + 3 * c0 + 6]
                    } else {
                        &Z
                    },
                )
            } else {
                (&Z, &Z)
            };
            let mut ch = [0.0f64; 3];
            for (i, acc) in ch.iter_mut().enumerate() {
                *acc = ((p00[i] as f64 + p01[i] as f64) + (p10[i] as f64 + p11[i] as f64)) * 0.25;
            }
            dst[PAD + x] = dot(ch[0], ch[1], ch[2]);
        }
    }
    p
}

// ------------------------------------------------------------------
// Shared leaf bodies (f32, per-output fixed order → tier-identical).
// All slices are padded `Plane` rows; `PAD` is the frame width, so a
// shifted gather at `±HMAX` stays in bounds.
// ------------------------------------------------------------------

/// `dst[x] = Σ_a rows[a][x]` for `x ∈ [0, n)` (indices in padded
/// coords starting at `PAD`). Vertical box sum: fixed tap order.
macro_rules! rowsum_body {
    ($token:ident, $F32:ident, $LANES:literal, $rows:ident, $dst:ident, $n:ident) => {{
        let n = $n;
        let mut x = 0usize;
        while x + $LANES <= n {
            let mut v = $F32::from_array($token, core::array::from_fn(|k| $rows[0][PAD + x + k]));
            for a in $rows.iter().skip(1) {
                v = v + $F32::from_array($token, core::array::from_fn(|k| a[PAD + x + k]));
            }
            v.store((&mut $dst[PAD + x..PAD + x + $LANES]).try_into().unwrap());
            x += $LANES;
        }
        while x < n {
            let mut s = $rows[0][PAD + x];
            for a in $rows.iter().skip(1) {
                s = s + a[PAD + x];
            }
            $dst[PAD + x] = s;
            x += 1;
        }
    }};
}

/// `dst[x] = Σ_{u<h} src[x − h + 1 + u]` for `x ∈ [0, n)` — a width-`h`
/// box sum *ending* at `x` (the RS pass). Shifted gathers read the
/// zero frame for out-of-range taps.
macro_rules! tailbox_body {
    ($token:ident, $F32:ident, $LANES:literal, $src:ident, $dst:ident, $n:ident, $h:ident) => {{
        let n = $n;
        let h = $h;
        let mut x = 0usize;
        while x + $LANES <= n {
            let mut v = $F32::splat($token, 0.0);
            for u in 0..h {
                v = v + $F32::from_array(
                    $token,
                    core::array::from_fn(|k| $src[PAD + x + k + u + 1 - h]),
                );
            }
            v.store((&mut $dst[PAD + x..PAD + x + $LANES]).try_into().unwrap());
            x += $LANES;
        }
        while x < n {
            let mut s = 0.0f32;
            for u in 0..h {
                s = s + $src[PAD + x + u + 1 - h];
            }
            $dst[PAD + x] = s;
            x += 1;
        }
    }};
}

/// `dst[x] = scale · Σ_{t<2h} (a[x − h + 1 + t] − b[x − h + 1 + t])`
/// for `x ∈ [0, n)` — the V-response combine (CS row `i` vs `i + h`).
macro_rules! stripdiff_body {
    ($token:ident, $F32:ident, $LANES:literal, $a:ident, $b:ident, $dst:ident, $n:ident, $h:ident, $scale:ident) => {{
        let n = $n;
        let h = $h;
        let sc = $F32::splat($token, $scale);
        let mut x = 0usize;
        while x + $LANES <= n {
            let mut v = $F32::splat($token, 0.0);
            for t in 0..2 * h {
                let i = PAD + x + t + 1 - h;
                v = v
                    + ($F32::from_array($token, core::array::from_fn(|k| $a[i + k]))
                        - $F32::from_array($token, core::array::from_fn(|k| $b[i + k])));
            }
            (sc * v).store((&mut $dst[PAD + x..PAD + x + $LANES]).try_into().unwrap());
            x += $LANES;
        }
        while x < n {
            let mut s = 0.0f32;
            for t in 0..2 * h {
                s = s + ($a[PAD + x + t + 1 - h] - $b[PAD + x + t + 1 - h]);
            }
            $dst[PAD + x] = $scale * s;
            x += 1;
        }
    }};
}

/// `dst[x] = scale · Σ_a (rows[a][x] − rows[a][x + h])` for `x ∈
/// [0, n)` — the H-response combine over `2h` RS rows.
macro_rules! rowdiff_body {
    ($token:ident, $F32:ident, $LANES:literal, $rows:ident, $dst:ident, $n:ident, $h:ident, $scale:ident) => {{
        let n = $n;
        let h = $h;
        let sc = $F32::splat($token, $scale);
        let mut x = 0usize;
        while x + $LANES <= n {
            let mut v = $F32::splat($token, 0.0);
            for a in $rows.iter() {
                v = v
                    + ($F32::from_array($token, core::array::from_fn(|k| a[PAD + x + k]))
                        - $F32::from_array($token, core::array::from_fn(|k| a[PAD + x + k + h])));
            }
            (sc * v).store((&mut $dst[PAD + x..PAD + x + $LANES]).try_into().unwrap());
            x += $LANES;
        }
        while x < n {
            let mut s = 0.0f32;
            for a in $rows.iter() {
                s = s + (a[PAD + x] - a[PAD + x + h]);
            }
            $dst[PAD + x] = $scale * s;
            x += 1;
        }
    }};
}

/// `dst[x] = |(r[x] + r[x+1]) + (s[x] + s[x+1])| · 0.25` — the I/Q
/// `abs(conv2(ones(2,2)/4,'same'))` magnitude (`r`, `s` = rows `i`,
/// `i+1`; the forward-looking 'same' anchor for even kernels).
macro_rules! box2abs_body {
    ($token:ident, $F32:ident, $LANES:literal, $r:ident, $s:ident, $dst:ident, $n:ident) => {{
        let n = $n;
        let q = $F32::splat($token, 0.25);
        let mut x = 0usize;
        while x + $LANES <= n {
            let r0 = $F32::from_array($token, core::array::from_fn(|k| $r[PAD + x + k]));
            let r1 = $F32::from_array($token, core::array::from_fn(|k| $r[PAD + x + k + 1]));
            let s0 = $F32::from_array($token, core::array::from_fn(|k| $s[PAD + x + k]));
            let s1 = $F32::from_array($token, core::array::from_fn(|k| $s[PAD + x + k + 1]));
            (q * ((r0 + r1) + (s0 + s1)))
                .abs()
                .store((&mut $dst[PAD + x..PAD + x + $LANES]).try_into().unwrap());
            x += $LANES;
        }
        while x < n {
            $dst[PAD + x] =
                (0.25 * (($r[PAD + x] + $r[PAD + x + 1]) + ($s[PAD + x] + $s[PAD + x + 1]))).abs();
            x += 1;
        }
    }};
}

/// `dst[x] = (2·|a||b| + C) / (|a|² + |b|² + C)` — the HaarPSI local
/// similarity, stored (first scale) — same op order as the reference:
/// `(2·cr·cd + C) / (cr² + cd² + C)` on magnitudes.
macro_rules! sim_body {
    ($token:ident, $F32:ident, $LANES:literal, $a:ident, $b:ident, $dst:ident, $n:ident) => {{
        let n = $n;
        let two = $F32::splat($token, 2.0);
        let cc = $F32::splat($token, C30);
        let mut x = 0usize;
        while x + $LANES <= n {
            let aa = $F32::from_array($token, core::array::from_fn(|k| $a[PAD + x + k])).abs();
            let bb = $F32::from_array($token, core::array::from_fn(|k| $b[PAD + x + k])).abs();
            let num = (two * aa) * bb + cc;
            let den = (aa * aa + bb * bb) + cc;
            (num / den).store((&mut $dst[PAD + x..PAD + x + $LANES]).try_into().unwrap());
            x += $LANES;
        }
        while x < n {
            let aa = $a[PAD + x].abs();
            let bb = $b[PAD + x].abs();
            $dst[PAD + x] = ((2.0 * aa) * bb + C30) / ((aa * aa + bb * bb) + C30);
            x += 1;
        }
    }};
}

/// `dst[x] = (dst[x] + sim(a,b)) · 0.5` — folds the second scale's
/// similarity into the map mean (reference `sum(...,3)/2`).
macro_rules! simavg_body {
    ($token:ident, $F32:ident, $LANES:literal, $a:ident, $b:ident, $dst:ident, $n:ident) => {{
        let n = $n;
        let two = $F32::splat($token, 2.0);
        let half = $F32::splat($token, 0.5);
        let cc = $F32::splat($token, C30);
        let mut x = 0usize;
        while x + $LANES <= n {
            let aa = $F32::from_array($token, core::array::from_fn(|k| $a[PAD + x + k])).abs();
            let bb = $F32::from_array($token, core::array::from_fn(|k| $b[PAD + x + k])).abs();
            let num = (two * aa) * bb + cc;
            let den = (aa * aa + bb * bb) + cc;
            let prev = $F32::from_array($token, core::array::from_fn(|k| $dst[PAD + x + k]));
            ((prev + num / den) * half)
                .store((&mut $dst[PAD + x..PAD + x + $LANES]).try_into().unwrap());
            x += $LANES;
        }
        while x < n {
            let aa = $a[PAD + x].abs();
            let bb = $b[PAD + x].abs();
            let sim = ((2.0 * aa) * bb + C30) / ((aa * aa + bb * bb) + C30);
            $dst[PAD + x] = ($dst[PAD + x] + sim) * 0.5;
            x += 1;
        }
    }};
}

/// `dst[x] = max(|a[x]|, |b[x]|)` — the orientation weight map.
macro_rules! wmax_body {
    ($token:ident, $F32:ident, $LANES:literal, $a:ident, $b:ident, $dst:ident, $n:ident) => {{
        let n = $n;
        let mut x = 0usize;
        while x + $LANES <= n {
            let aa = $F32::from_array($token, core::array::from_fn(|k| $a[PAD + x + k])).abs();
            let bb = $F32::from_array($token, core::array::from_fn(|k| $b[PAD + x + k])).abs();
            aa.max(bb)
                .store((&mut $dst[PAD + x..PAD + x + $LANES]).try_into().unwrap());
            x += $LANES;
        }
        while x < n {
            $dst[PAD + x] = $a[PAD + x].abs().max($b[PAD + x].abs());
            x += 1;
        }
    }};
}

/// `dst[x] = (a[x] + b[x]) · 0.5` — `(similarityI + similarityQ)/2`
/// and `(weights₁ + weights₂)/2`.
macro_rules! avg2_body {
    ($token:ident, $F32:ident, $LANES:literal, $a:ident, $b:ident, $dst:ident, $n:ident) => {{
        let n = $n;
        let half = $F32::splat($token, 0.5);
        let mut x = 0usize;
        while x + $LANES <= n {
            let va = $F32::from_array($token, core::array::from_fn(|k| $a[PAD + x + k]));
            let vb = $F32::from_array($token, core::array::from_fn(|k| $b[PAD + x + k]));
            ((va + vb) * half).store((&mut $dst[PAD + x..PAD + x + $LANES]).try_into().unwrap());
            x += $LANES;
        }
        while x < n {
            $dst[PAD + x] = ($a[PAD + x] + $b[PAD + x]) * 0.5;
            x += 1;
        }
    }};
}

// ------------------------------------------------------------------
// Whole-score body — instantiated once per tier.
// ------------------------------------------------------------------

/// Fixed-order logistic + weighted-sum fold of one (ls, w) map pair.
/// gmsd's 8-lane accumulator grouping: `acc[j]` sums positions
/// `≡ j (mod 8)` within each 8-block; tails append after the fold —
/// the same order on every tier. Returns `(Σ w·σ(ls), Σ w)` where
/// `σ(x) = 1/(1+exp(−αx))` in f32 (libm, deterministic).
fn pool_map(ls: &Plane, w: &Plane) -> (f64, f64) {
    let mut acc_t = [0.0f64; 8];
    let mut acc_w = [0.0f64; 8];
    let mut tail = Vec::with_capacity(ls.h * (ls.w % 8));
    let nblk = ls.w / 8;
    for r in 0..ls.h {
        let lr = ls.data(r);
        let wr = w.data(r);
        for g in 0..nblk {
            let base = 8 * g;
            for (j, (at, aw)) in acc_t.iter_mut().zip(acc_w.iter_mut()).enumerate() {
                let i = base + j;
                let l = 1.0f32 / (1.0 + libm::expf(-ALPHA * lr[i]));
                *at += (wr[i] * l) as f64;
                *aw += wr[i] as f64;
            }
        }
        for i in 8 * nblk..ls.w {
            tail.push((wr[i], lr[i]));
        }
    }
    let ft = [
        acc_t[0] + acc_t[4],
        acc_t[1] + acc_t[5],
        acc_t[2] + acc_t[6],
        acc_t[3] + acc_t[7],
    ];
    let fw = [
        acc_w[0] + acc_w[4],
        acc_w[1] + acc_w[5],
        acc_w[2] + acc_w[6],
        acc_w[3] + acc_w[7],
    ];
    let mut s_t = ft[0] + ft[1] + ft[2] + ft[3];
    let mut s_w = fw[0] + fw[1] + fw[2] + fw[3];
    for (wv, lv) in tail {
        let l = 1.0f32 / (1.0 + libm::expf(-ALPHA * lv));
        s_t += (wv * l) as f64;
        s_w += wv as f64;
    }
    (s_t, s_w)
}

/// The score pipeline on already-prepared planes: `pr`, `pd` are the
/// luma (or grayscale) planes; `iq` carries the I/Q plane pairs for
/// the color path. Runs under a tier token so every leaf body is the
/// tier's vector width — bit-identical output for any tier.
macro_rules! score_body {
    ($token:ident, $F32:ident, $LANES:literal, $pr:ident, $pd:ident, $iq:ident) => {{
        let (w, h) = ($pr.w, $pr.h);
        // Orientation similarity/weight accumulators + Haar scratch.
        let mut ls = [Plane::new(w, h), Plane::new(w, h)];
        let mut wm = [Plane::new(w, h), Plane::new(w, h)];
        let mut cs = Plane::new(w, h);
        let mut rs = Plane::new(w, h);
        let mut ca = Plane::new(w, h);
        let mut cb = Plane::new(w, h);

        // Both Haar orientations × 3 scales. `o == 0` is the reference's
        // `haarFilter` direction (vertical-difference kernel, coeffs
        // 1..3); `o == 1` is its transpose (coeffs 4..6).
        for (o, (lso, wmo)) in ls.iter_mut().zip(wm.iter_mut()).enumerate() {
            for k in 0..3usize {
                let hk = 1usize << k; // 1, 2, 4 — the filter half-size
                for (src, coeff) in [($pr, &mut ca), ($pd, &mut cb)] {
                    if o == 0 {
                        // V: CS(r,c) = Σ_{a=r-h+1}^{r} X(a,c) for r in
                        // [0, h+hk) (partial boxes extend hk rows below).
                        cs.for_rows(0, (h + hk) as isize, |ri, dst| {
                            let r = ri as isize - PAD as isize;
                            let rows: Vec<&[f32]> = (0..hk)
                                .map(|a| src.prow(r - hk as isize + 1 + a as isize))
                                .collect();
                            rowsum_body!($token, $F32, $LANES, rows, dst, w);
                        });
                        // V(i,j) = 2^-k·Σ_t (CS(i,·) − CS(i+h,·)).
                        let sc = SCALE[k];
                        coeff.for_rows(0, h as isize, |ri, dst| {
                            let i = ri as isize - PAD as isize;
                            let (a, b) = (cs.prow(i), cs.prow(i + hk as isize));
                            stripdiff_body!($token, $F32, $LANES, a, b, dst, w, hk, sc);
                        });
                    } else {
                        // H: RS(r,c) = Σ_{b=c-h+1}^{c} X(r,b) for c in
                        // [0, w+hk) (partial boxes extend hk cols right).
                        rs.for_rows(0, h as isize, |ri, dst| {
                            let r = ri as isize - PAD as isize;
                            let s = src.prow(r);
                            let n = w + hk;
                            tailbox_body!($token, $F32, $LANES, s, dst, n, hk);
                        });
                        // H(i,j) = 2^-k·Σ_a (RS(a,j) − RS(a,j+h)).
                        let sc = SCALE[k];
                        coeff.for_rows(0, h as isize, |ri, dst| {
                            let i = ri as isize - PAD as isize;
                            let rows: Vec<&[f32]> = (0..2 * hk)
                                .map(|a| rs.prow(i - hk as isize + 1 + a as isize))
                                .collect();
                            rowdiff_body!($token, $F32, $LANES, rows, dst, w, hk, sc);
                        });
                    }
                }
                // Fold the scale into the similarity/weight maps.
                if k == 2 {
                    wmo.for_rows(0, h as isize, |ri, dst| {
                        let i = ri as isize - PAD as isize;
                        let (ra, rb) = (ca.prow(i), cb.prow(i));
                        wmax_body!($token, $F32, $LANES, ra, rb, dst, w);
                    });
                } else {
                    lso.for_rows(0, h as isize, |ri, dst| {
                        let i = ri as isize - PAD as isize;
                        let (ra, rb) = (ca.prow(i), cb.prow(i));
                        if k == 0 {
                            sim_body!($token, $F32, $LANES, ra, rb, dst, w);
                        } else {
                            simavg_body!($token, $F32, $LANES, ra, rb, dst, w);
                        }
                    });
                }
            }
        }

        // Color: I/Q box magnitudes → third similarity map (cs/rs/ca are
        // free scratch once the weights exist).
        let mut ls3w3 = None;
        if let Some((ir, id, qr, qd)) = $iq {
            let mut l3 = Plane::new(w, h);
            for (src, dst_plane) in [(ir, &mut cs), (id, &mut rs)] {
                dst_plane.for_rows(0, h as isize, |ri, dst| {
                    let i = ri as isize - PAD as isize;
                    let (r, s) = (src.prow(i), src.prow(i + 1));
                    box2abs_body!($token, $F32, $LANES, r, s, dst, w);
                });
            }
            l3.for_rows(0, h as isize, |ri, dst| {
                let i = ri as isize - PAD as isize;
                let (ra, rb) = (cs.prow(i), rs.prow(i));
                sim_body!($token, $F32, $LANES, ra, rb, dst, w);
            });
            for (src, dst_plane) in [(qr, &mut cs), (qd, &mut rs)] {
                dst_plane.for_rows(0, h as isize, |ri, dst| {
                    let i = ri as isize - PAD as isize;
                    let (r, s) = (src.prow(i), src.prow(i + 1));
                    box2abs_body!($token, $F32, $LANES, r, s, dst, w);
                });
            }
            l3.for_rows(0, h as isize, |ri, dst| {
                let i = ri as isize - PAD as isize;
                let (ra, rb) = (cs.prow(i), rs.prow(i));
                simavg_body!($token, $F32, $LANES, ra, rb, dst, w);
            });
            let mut w3 = Plane::new(w, h);
            w3.for_rows(0, h as isize, |ri, dst| {
                let i = ri as isize - PAD as isize;
                let (wa, wb) = (wm[0].prow(i), wm[1].prow(i));
                avg2_body!($token, $F32, $LANES, wa, wb, dst, w);
            });
            ls3w3 = Some((l3, w3));
        }

        // Weighted-logistic pooling over all maps, fixed order.
        let (mut s_t, mut s_w) = (0.0f64, 0.0f64);
        for (l, m) in ls
            .iter()
            .zip(wm.iter())
            .chain(ls3w3.iter().map(|(l, m)| (l, m)))
        {
            let (t, wsum) = pool_map(l, m);
            s_t += t;
            s_w += wsum;
        }
        let m = s_t / s_w;
        // HaarPSILogInv(m, alpha)^2 — NaN propagates like the reference.
        let li = libm::log(m / (1.0 - m)) / ALPHA as f64;
        li * li
    }};
}

/// The `v3` tier (AVX2/FMA, 8-wide f32).
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub fn haarpsi_core_tier_v3(
    token: archmage::X64V3Token,
    pr: &Plane,
    pd: &Plane,
    iq: Option<(&Plane, &Plane, &Plane, &Plane)>,
) -> f64 {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    score_body!(token, V, 8, pr, pd, iq)
}

/// The `v4` tier (AVX-512, 16-wide f32 — crate feature `avx512`).
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
pub fn haarpsi_core_tier_v4(
    token: archmage::X64V4Token,
    pr: &Plane,
    pd: &Plane,
    iq: Option<(&Plane, &Plane, &Plane, &Plane)>,
) -> f64 {
    type V = magetypes::simd::generic::f32x16<archmage::X64V4Token>;
    score_body!(token, V, 16, pr, pd, iq)
}

/// The `neon` tier (8-wide f32).
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
pub fn haarpsi_core_tier_neon(
    token: archmage::NeonToken,
    pr: &Plane,
    pd: &Plane,
    iq: Option<(&Plane, &Plane, &Plane, &Plane)>,
) -> f64 {
    type V = magetypes::simd::generic::f32x8<archmage::NeonToken>;
    score_body!(token, V, 8, pr, pd, iq)
}

/// The `wasm128` tier (8-wide f32).
#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
pub fn haarpsi_core_tier_wasm128(
    token: archmage::Wasm128Token,
    pr: &Plane,
    pd: &Plane,
    iq: Option<(&Plane, &Plane, &Plane, &Plane)>,
) -> f64 {
    type V = magetypes::simd::generic::f32x8<archmage::Wasm128Token>;
    score_body!(token, V, 8, pr, pd, iq)
}

/// The `scalar` tier (8-wide f32, scalar-emulated lanes).
pub fn haarpsi_core_tier_scalar(
    token: archmage::ScalarToken,
    pr: &Plane,
    pd: &Plane,
    iq: Option<(&Plane, &Plane, &Plane, &Plane)>,
) -> f64 {
    type V = magetypes::simd::generic::f32x8<archmage::ScalarToken>;
    score_body!(token, V, 8, pr, pd, iq)
}

/// Runtime-dispatched score: the best tier this CPU supports.
pub(crate) fn haarpsi_core(
    pr: &Plane,
    pd: &Plane,
    iq: Option<(&Plane, &Plane, &Plane, &Plane)>,
) -> f64 {
    archmage::incant!(
        haarpsi_core_tier(pr, pd, iq),
        [v4, v3, neon, wasm128, scalar]
    )
}
