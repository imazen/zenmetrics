//! VSI kernel — see `lib.rs`'s module docs for the pipeline and
//! provenance. Scalar helpers (`sdsp`, `imresize`, `downsample_decimate`,
//! `conv2_same_3x3`) are shared by every SIMD tier; only the final
//! elementwise similarity + pooling stage is parameterized over the
//! magetypes vector width, so all tiers are bit-identical.

use alloc::vec;
use alloc::vec::Vec;

use crate::fft::{self, Complex};

/// `sdsp` works on a fixed 256×256 grid (the reference hardcodes the
/// `imresize` target — input of any size is rescaled to it).
const SDSP_N: usize = 256;

/// Reference constants (`VSI.m` top of file — "fixed").
const C_VS: f32 = 1.27;
const C_GM: f32 = 386.0;
const C_CHROM: f32 = 130.0;
const ALPHA: f32 = 0.40;
const LAMBDA: f32 = 0.020;
const SIGMA_F: f64 = 1.34;
const OMEGA0: f64 = 0.0210;
const SIGMA_D: f32 = 145.0;
const SIGMA_C: f32 = 0.001;

/// `F = max(1, round(min(w,h) / 256))` — MATLAB `round` (half away
/// from zero), single shot — the identical rule `FR_FSIMc.m` uses.
/// VSI reaches F=2 at min-dim 384.
pub(crate) fn decimation_factor(w: usize, h: usize) -> usize {
    let f = libm::round(w.min(h) as f64 / 256.0) as usize;
    f.max(1)
}

// ======================== imresize (bilinear, AA) ==========================
//
// Port of octave-image `imresize`'s `conv_interp_vec` — verified
// bit-identical to `imresize(A,[m n],'bilinear')` at 17 scale
// combinations (up, down, non-integer, both directions):
//
// * `ZI[o] = 0.5 + 0.5/scale + o/scale` (1-based source coord).
// * Triangle kernel; when `scale < 1` it is broadened:
//   `kernel(h) = scale·tri(scale·h)`, support `2/scale`.
// * Source indices are extended by whole-point *symmetric* padding
//   (`pad_indices "symmetric"`), never dropped.
// * Each output is normalised by the *full* kernel-weight sum.
// * Column pass first, then rows (Octave's axis order); an axis with
//   `scale == 1` is skipped entirely.

/// One output position's folded tap list: `(source_index, weight)` —
/// symmetric padding folded in at table-build time, weights already
/// normalised to sum 1 (Σ kernel over all shifts, padding included).
struct Axis {
    /// `(start, len)` into `taps` per output position.
    spans: Vec<(u32, u32)>,
    /// `(src_index, weight)` flat.
    taps: Vec<(u32, f32)>,
}

fn sympad(i: isize, n: usize) -> usize {
    let n = n as isize;
    let i0 = i - 1;
    let m = i0.rem_euclid(n);
    if i0.div_euclid(n) % 2 != 0 {
        (n - m) as usize
    } else {
        (m + 1) as usize
    }
}

fn tri(h: f64) -> f64 {
    let a = h.abs();
    if a <= 1.0 { 1.0 - a } else { 0.0 }
}

/// Build the tap table for one axis. Mirrors `conv_interp_vec`:
/// `shift ∈ [1−pad, pad]`, `pad = ceil(kernel_size/2) + 2`, weights
/// `kernel(shift − DZ)` with `DZ = ZI − floor(ZI)`.
fn axis_taps(in_n: usize, out_n: usize) -> Axis {
    let scale = out_n as f64 / in_n as f64;
    let shrink = scale < 1.0;
    let kernel_size = if shrink { 2.0 / scale } else { 2.0 };
    let pad = (kernel_size / 2.0).ceil() as isize + 2;
    let mut spans = Vec::with_capacity(out_n);
    let mut taps = Vec::new();
    for o in 0..out_n {
        let zi = 0.5 + 0.5 / scale + o as f64 / scale; // 1-based
        let idx = zi.floor() as isize;
        let dz = zi - idx as f64;
        let start = taps.len() as u32;
        let mut wsum = 0.0f64;
        for shift in (1 - pad)..=pad {
            let h = shift as f64 - dz;
            let w = if shrink {
                scale * tri(scale * h)
            } else {
                tri(h)
            };
            if w == 0.0 {
                continue;
            }
            wsum += w;
            // `sympad` yields the 1-based Octave padded index; store
            // 0-based for `srow`/`src` indexing.
            taps.push((sympad(idx + shift, in_n) as u32 - 1, w as f32));
        }
        if wsum == 0.0 {
            wsum = 1.0;
        }
        let len = taps.len() as u32 - start;
        // Normalise by the full kernel sum (padding included) — the
        // `sum_weights` division at the end of conv_interp_vec.
        for t in &mut taps[start as usize..] {
            t.1 = (t.1 as f64 / wsum) as f32;
        }
        spans.push((start, len));
    }
    Axis { spans, taps }
}

/// Horizontal pass: `out[i,j] = Σ_t w·src[i, idx_t]`, accumulate f64.
fn resize_h(src: &[f32], in_w: usize, in_h: usize, ax: &Axis) -> Vec<f32> {
    let out_w = ax.spans.len();
    let mut out = vec![0.0f32; in_h * out_w];
    for i in 0..in_h {
        let srow = &src[i * in_w..(i + 1) * in_w];
        let drow = &mut out[i * out_w..(i + 1) * out_w];
        for (o, &(st, len)) in ax.spans.iter().enumerate() {
            let mut acc = 0.0f64;
            for &(idx, w) in &ax.taps[st as usize..(st + len) as usize] {
                acc += w as f64 * srow[idx as usize] as f64;
            }
            drow[o] = acc as f32;
        }
    }
    out
}

/// Vertical pass over a `in_w`-strided plane.
fn resize_v(src: &[f32], w: usize, _in_h: usize, ax: &Axis) -> Vec<f32> {
    let out_h = ax.spans.len();
    let mut out = vec![0.0f32; out_h * w];
    for (o, &(st, len)) in ax.spans.iter().enumerate() {
        let orow = &mut out[o * w..(o + 1) * w];
        for (x, v) in orow.iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for &(idx, wt) in &ax.taps[st as usize..(st + len) as usize] {
                acc += wt as f64 * src[idx as usize * w + x] as f64;
            }
            *v = acc as f32;
        }
    }
    out
}

/// `imresize(A, [out_h out_w], 'bilinear')` — column pass first
/// (Octave's axis order), axis skipped when `scale == 1`.
pub(crate) fn imresize(
    src: &[f32],
    in_w: usize,
    in_h: usize,
    out_w: usize,
    out_h: usize,
) -> Vec<f32> {
    let tmp = if out_w == in_w {
        src.to_vec()
    } else {
        let ax = axis_taps(in_w, out_w);
        resize_h(src, in_w, in_h, &ax)
    };
    if out_h == in_h {
        tmp
    } else {
        let ax = axis_taps(in_h, out_h);
        resize_v(&tmp, out_w, in_h, &ax)
    }
}

// ============================ sRGB → Lab (D50) =============================
//
// `RGB2Lab` from `VSI.m`, verbatim: sRGB EOTF inverse, the reference's
// own XYZ matrix, **D50** reference white (Xr 0.9642, Yr 1.0,
// Zr 0.8251 — not the usual D65), ε = 0.008856, κ = 903.3.

fn srgb_to_lab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    fn lin(v: f64) -> f64 {
        if v <= 0.04045 {
            v / 12.92
        } else {
            libm::pow((v + 0.055) / 1.055, 2.4)
        }
    }
    let (r, g, b) = (
        lin(r as f64 / 255.0),
        lin(g as f64 / 255.0),
        lin(b as f64 / 255.0),
    );
    let x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    let y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    let z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;
    const EPS: f64 = 0.008856;
    const KAPPA: f64 = 903.3;
    fn f(t: f64) -> f64 {
        if t > EPS {
            libm::pow(t, 1.0 / 3.0)
        } else {
            (KAPPA * t + 16.0) / 116.0
        }
    }
    let (fx, fy, fz) = (f(x / 0.9642), f(y), f(z / 0.8251));
    (
        (116.0 * fy - 16.0) as f32,
        (500.0 * (fx - fy)) as f32,
        (200.0 * (fy - fz)) as f32,
    )
}

// ============================ log-Gabor grid ===============================
//
// `logGabor(256,256,ω0,σf)` from `VSI.m` (Kovesi grid): centred coords
// `([1:256]-129)/256`, mask `u1²+u2² ≤ 0.25`, `ifftshift`, radius with
// DC=1 fudge, `LG = exp(−(log(r/ω0))²/(2σf²))`, `LG(1,1) = 0`.

fn loggabor_256() -> Vec<f32> {
    let n = SDSP_N;
    let mut lg = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            // ifftshift of the centred grid: bin (i,j) holds centred
            // coordinate [(idx + n/2) % n].
            let u1 = (((j + n / 2) % n) as f64 + 1.0 - (n / 2 + 1) as f64) / n as f64;
            let u2 = (((i + n / 2) % n) as f64 + 1.0 - (n / 2 + 1) as f64) / n as f64;
            let r2 = u1 * u1 + u2 * u2;
            if r2 > 0.25 {
                continue;
            }
            let radius = libm::sqrt(r2);
            // radius==0 only at the DC bin — the reference sets
            // radius(1,1)=1 then LG(1,1)=0; the `continue` paths above
            // already write 0, and DC lands here with r=0 → log(0)
            // → −inf → exp(−inf²)=0 — same result; set it explicitly.
            if radius == 0.0 {
                lg[i * n + j] = 0.0;
                continue;
            }
            let t = libm::log(radius / OMEGA0);
            lg[i * n + j] = libm::exp(-(t * t) / (2.0 * SIGMA_F * SIGMA_F)) as f32;
        }
    }
    lg
}

// ================================ SDSP ===================================

/// `SDSP(rgb, σf, ω0, σd, σc)` — returns the w×h saliency map.
/// `rgb` is the three concatenated channel planes `[R | G | B]` of
/// the ORIGINAL (undecimated) image.
fn sdsp(rgb: &[f32], w: usize, h: usize, lg: &[f32]) -> Vec<f32> {
    let n = SDSP_N;
    let wh = w * h;
    let (r, g, b) = (&rgb[..wh], &rgb[wh..2 * wh], &rgb[2 * wh..]);
    // 1. antialiased bilinear resize of each channel to 256².
    let dr = imresize(r, w, h, n, n);
    let dg = imresize(g, w, h, n, n);
    let db = imresize(b, w, h, n, n);
    // 2. RGB → Lab (D50) on the 256² image.
    let mut lab = [
        vec![0.0f32; n * n],
        vec![0.0f32; n * n],
        vec![0.0f32; n * n],
    ];
    for i in 0..n * n {
        let (l, a, bb) = srgb_to_lab(dr[i], dg[i], db[i]);
        lab[0][i] = l;
        lab[1][i] = a;
        lab[2][i] = bb;
    }
    // 3. log-Gabor bandpass per channel: real(ifft2(fft2(ch) .* LG)).
    let plan = fft::Plan::new(n);
    let mut sf = vec![0.0f32; n * n];
    for ch in &lab {
        let mut buf: Vec<Complex> = ch.iter().map(|&v| Complex { re: v, im: 0.0 }).collect();
        fft::fft2_planned(&mut buf, n, n, &plan, &plan);
        for (z, &k) in buf.iter_mut().zip(lg.iter()) {
            z.re *= k;
            z.im *= k;
        }
        fft::ifft2_planned(&mut buf, n, n, &plan, &plan);
        for (s, z) in sf.iter_mut().zip(&buf) {
            *s += z.re * z.re;
        }
    }
    for s in &mut sf {
        *s = libm::sqrtf(*s);
    }
    // 4. centre prior: coords 1..=256, centre rows/2 = 128.
    let mut vs = vec![0.0f32; n * n];
    for i in 0..n {
        let dy = (i + 1) as f32 - (n / 2) as f32;
        for j in 0..n {
            let dx = (j + 1) as f32 - (n / 2) as f32;
            vs[i * n + j] = sf[i * n + j] * libm::expf(-(dx * dx + dy * dy) / (SIGMA_D * SIGMA_D));
        }
    }
    // 5. warm-colour prior on the resized A/B channels.
    let (mut amin, mut amax) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut bmin, mut bmax) = (f32::INFINITY, f32::NEG_INFINITY);
    for i in 0..n * n {
        amin = amin.min(lab[1][i]);
        amax = amax.max(lab[1][i]);
        bmin = bmin.min(lab[2][i]);
        bmax = bmax.max(lab[2][i]);
    }
    let (arange, brange) = (amax - amin, bmax - bmin);
    for i in 0..n * n {
        let na = (lab[1][i] - amin) / arange;
        let nb = (lab[2][i] - bmin) / brange;
        let d2 = na * na + nb * nb;
        vs[i] *= 1.0 - libm::expf(-d2 / (SIGMA_C * SIGMA_C));
    }
    // 6. resize back to the input size, then mat2gray.
    let mut out = imresize(&vs, n, n, w, h);
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in &out {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let range = hi - lo;
    for v in &mut out {
        *v = (*v - lo) / range;
    }
    out
}

// ===================== decimation + gradients ============================

/// `conv2(x, ones(F,F)/F², 'same')` then `x(1:F:end, 1:F:end)` — the
/// `same` anchor for an F-tap kernel is `full(i + ⌊F/2⌋)`, i.e. output
/// `i` averages the window `{i+⌊F/2⌋−F+1 .. i+⌊F/2⌋}` — **forward**-tilted
/// for even F (`{i, i+1}` at F=2), zero-padded. Verified against
/// Octave's `conv2`.
fn downsample_decimate(x: &[f32], w: usize, h: usize, f: usize) -> (usize, usize, Vec<f32>) {
    if f == 1 {
        return (w, h, x.to_vec());
    }
    let s = f / 2;
    let mut tmp = vec![0.0f32; w * h];
    for i in 0..h {
        let row = &x[i * w..(i + 1) * w];
        let trow = &mut tmp[i * w..(i + 1) * w];
        for (j, t) in trow.iter_mut().enumerate() {
            let lo = (j + s).saturating_sub(f - 1);
            let hi = (j + s).min(w - 1);
            let mut acc = 0.0f32;
            for v in &row[lo..=hi] {
                acc += v;
            }
            *t = acc;
        }
    }
    let w2 = w.div_ceil(f);
    let h2 = h.div_ceil(f);
    let mut out = vec![0.0f32; w2 * h2];
    let inv = 1.0 / (f * f) as f32;
    for oi in 0..h2 {
        let i = oi * f;
        let lo = (i + s).saturating_sub(f - 1);
        let hi = (i + s).min(h - 1);
        for oj in 0..w2 {
            let j = oj * f;
            let mut acc = 0.0f32;
            for r in lo..=hi {
                acc += tmp[r * w + j];
            }
            out[oi * w2 + oj] = acc * inv;
        }
    }
    (w2, h2, out)
}

/// `conv2(Y, K, 'same')` for a 3×3 kernel — flipped kernel,
/// zero-padded (same as `crates/fsim`).
fn conv2_same_3x3(y: &[f32], w: usize, h: usize, k: &[[f32; 3]; 3]) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    for i in 0..h {
        for j in 0..w {
            let mut acc = 0.0f32;
            for a in 0..3isize {
                let r = i as isize + 1 - a;
                if r < 0 || r >= h as isize {
                    continue;
                }
                for b in 0..3isize {
                    let c = j as isize + 1 - b;
                    if c < 0 || c >= w as isize {
                        continue;
                    }
                    acc += k[a as usize][b as usize] * y[r as usize * w + c as usize];
                }
            }
            out[i * w + j] = acc;
        }
    }
    out
}

/// Scharr gradient magnitude of a decimated luma plane.
fn scharr_mag(l: &[f32], w: usize, h: usize) -> Vec<f32> {
    const DX: [[f32; 3]; 3] = [
        [3.0 / 16.0, 0.0, -3.0 / 16.0],
        [10.0 / 16.0, 0.0, -10.0 / 16.0],
        [3.0 / 16.0, 0.0, -3.0 / 16.0],
    ];
    const DY: [[f32; 3]; 3] = [
        [3.0 / 16.0, 10.0 / 16.0, 3.0 / 16.0],
        [0.0, 0.0, 0.0],
        [-3.0 / 16.0, -10.0 / 16.0, -3.0 / 16.0],
    ];
    let gx = conv2_same_3x3(l, w, h, &DX);
    let gy = conv2_same_3x3(l, w, h, &DY);
    let mut out = vec![0.0f32; w * h];
    for i in 0..w * h {
        out[i] = libm::sqrtf(gx[i] * gx[i] + gy[i] * gy[i]);
    }
    out
}

/// `real(z^λ)` with the reference's complex-power semantics
/// (`(ISim·QSim)^0.02` — same helper as `crates/fsim`).
fn real_pow_lambda(z: f32) -> f32 {
    if z >= 0.0 {
        libm::powf(z, LAMBDA)
    } else {
        libm::powf(-z, LAMBDA) * libm::cosf(LAMBDA * core::f32::consts::PI)
    }
}

// ====================== shared per-tier pipeline =========================

/// The decimated planes feeding the similarity stage.
#[derive(Clone, Copy)]
pub struct Planes<'a> {
    /// Decimated `max(SM_ref, SM_dis)` inputs — `(sm_r, sm_d)`.
    pub sm: (&'a [f32], &'a [f32]),
    /// Decimated Scharr gradient magnitudes — `(g_r, g_d)`.
    pub g: (&'a [f32], &'a [f32]),
    /// Decimated `M` opponent planes — `(m_r, m_d)`.
    pub m: (&'a [f32], &'a [f32]),
    /// Decimated `N` opponent planes — `(n_r, n_d)`.
    pub n: (&'a [f32], &'a [f32]),
    /// Decimated width × height.
    pub w: usize,
    /// Decimated height.
    pub h: usize,
}

/// `rayon::join` under `parallel`, sequential otherwise — each side
/// writes its own `Vec`, so results are identical at any thread count.
fn maybe_join<A, B>(fa: impl FnOnce() -> A + Send, fb: impl FnOnce() -> B + Send) -> (A, B)
where
    A: Send,
    B: Send,
{
    #[cfg(feature = "parallel")]
    {
        rayon::join(fa, fb)
    }
    #[cfg(not(feature = "parallel"))]
    {
        (fa(), fb())
    }
}

/// Shared scalar preprocessing: SDSP both images, opponent decimation,
/// gradients — everything up to (but not including) the elementwise
/// similarity+pool stage.
pub(crate) fn prepare<'a>(
    r: &'a crate::Input,
    d: &'a crate::Input,
    w: usize,
    h: usize,
    scratch: &'a mut Scratch,
) -> Planes<'a> {
    let lg = loggabor_256();
    let (sm_r, sm_d) = maybe_join(|| sdsp(&r.rgb, w, h, &lg), || sdsp(&d.rgb, w, h, &lg));
    let f = decimation_factor(w, h);
    let ((w2, h2, l_r), (_, _, l_d)) = maybe_join(
        || downsample_decimate(&r.l, w, h, f),
        || downsample_decimate(&d.l, w, h, f),
    );
    let ((_, _, m_r), (_, _, m_d)) = maybe_join(
        || downsample_decimate(&r.m, w, h, f),
        || downsample_decimate(&d.m, w, h, f),
    );
    let ((_, _, n_r), (_, _, n_d)) = maybe_join(
        || downsample_decimate(&r.n, w, h, f),
        || downsample_decimate(&d.n, w, h, f),
    );
    let ((_, _, sm_r), (_, _, sm_d)) = maybe_join(
        || downsample_decimate(&sm_r, w, h, f),
        || downsample_decimate(&sm_d, w, h, f),
    );
    let (g_r, g_d) = maybe_join(|| scharr_mag(&l_r, w2, h2), || scharr_mag(&l_d, w2, h2));
    scratch.keep(sm_r, sm_d, g_r, g_d, m_r, m_d, n_r, n_d);
    Planes {
        sm: (&scratch.buf[0], &scratch.buf[1]),
        g: (&scratch.buf[2], &scratch.buf[3]),
        m: (&scratch.buf[4], &scratch.buf[5]),
        n: (&scratch.buf[6], &scratch.buf[7]),
        w: w2,
        h: h2,
    }
}

/// Owns the eight decimated planes so `Planes` can borrow them.
#[derive(Default)]
pub(crate) struct Scratch {
    buf: [Vec<f32>; 8],
}

impl Scratch {
    pub(crate) fn new() -> Self {
        Self::default()
    }
    #[allow(clippy::too_many_arguments)]
    fn keep(
        &mut self,
        a: Vec<f32>,
        b: Vec<f32>,
        c: Vec<f32>,
        d: Vec<f32>,
        e: Vec<f32>,
        f: Vec<f32>,
        g: Vec<f32>,
        h: Vec<f32>,
    ) {
        self.buf = [a, b, c, d, e, f, g, h];
    }
}

/// The similarity+pool stage, vectorized over the magetypes width.
/// `SimMatrixC = gradSim^α · VSSim · real((ISim·QSim)^λ) · weight`,
/// `sim = Σ SimMatrixC / Σ weight` with `weight = max(SM1, SM2)`
/// (NaN-propagating, MATLAB `max` semantics). Fixed-order f64
/// accumulation — identical on every tier.
macro_rules! pool_body {
    ($token:ident, $F32:ty, $LANES:expr, $p:ident) => {{
        let n = $p.w * $p.h;
        let (sm1, sm2) = $p.sm;
        let (g1, g2) = $p.g;
        let (m1, m2) = $p.m;
        let (n1, n2) = $p.n;
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        let mut i = 0usize;
        while i + $LANES <= n {
            let a = <$F32>::from_array($token, core::array::from_fn(|k| sm1[i + k]));
            let b = <$F32>::from_array($token, core::array::from_fn(|k| sm2[i + k]));
            let x = <$F32>::from_array($token, core::array::from_fn(|k| g1[i + k]));
            let z = <$F32>::from_array($token, core::array::from_fn(|k| g2[i + k]));
            let mi = <$F32>::from_array($token, core::array::from_fn(|k| m1[i + k]));
            let mj = <$F32>::from_array($token, core::array::from_fn(|k| m2[i + k]));
            let qi = <$F32>::from_array($token, core::array::from_fn(|k| n1[i + k]));
            let qj = <$F32>::from_array($token, core::array::from_fn(|k| n2[i + k]));
            let cvs = <$F32>::splat($token, C_VS);
            let cgm = <$F32>::splat($token, C_GM);
            let cch = <$F32>::splat($token, C_CHROM);
            let two = <$F32>::splat($token, 2.0);
            let vssim = (two * a * b + cvs) / (a * a + b * b + cvs);
            let gsim = (two * x * z + cgm) / (x * x + z * z + cgm);
            let isim = (two * mi * mj + cch) / (mi * mi + mj * mj + cch);
            let qsim = (two * qi * qj + cch) / (qi * qi + qj * qj + cch);
            // Per-lane scalar for powf/max (fixed lane order — the same
            // scalar expression as the tail loop, so tiers agree
            // bitwise).
            let va = a.to_array();
            let vb = b.to_array();
            let vg = gsim.to_array();
            let vv = vssim.to_array();
            let vi = isim.to_array();
            let vq = qsim.to_array();
            for k in 0..$LANES {
                let w = if va[k].is_nan() || vb[k].is_nan() {
                    f32::NAN
                } else {
                    va[k].max(vb[k])
                };
                let iqreal = real_pow_lambda(vi[k] * vq[k]);
                let contrib = libm::powf(vg[k], ALPHA) * vv[k] * iqreal * w;
                num += contrib as f64;
                den += w as f64;
            }
            i += $LANES;
        }
        while i < n {
            let (a, b) = (sm1[i], sm2[i]);
            let vssim = (2.0 * a * b + C_VS) / (a * a + b * b + C_VS);
            let (x, z) = (g1[i], g2[i]);
            let gsim = (2.0 * x * z + C_GM) / (x * x + z * z + C_GM);
            let (ia, ib) = (m1[i], m2[i]);
            let isim = (2.0 * ia * ib + C_CHROM) / (ia * ia + ib * ib + C_CHROM);
            let (qa, qb) = (n1[i], n2[i]);
            let qsim = (2.0 * qa * qb + C_CHROM) / (qa * qa + qb * qb + C_CHROM);
            let w = if a.is_nan() || b.is_nan() {
                f32::NAN
            } else {
                a.max(b)
            };
            let iqreal = real_pow_lambda(isim * qsim);
            let contrib = libm::powf(gsim, ALPHA) * vssim * iqreal * w;
            num += contrib as f64;
            den += w as f64;
            i += 1;
        }
        num / den
    }};
}

/// Scalar tier (the magetypes generic vec over `ScalarToken` — same
/// lane count as `v3`, so per-lane extraction stays fixed-order).
pub fn vsi_core_tier_scalar(token: archmage::ScalarToken, p: Planes<'_>) -> f64 {
    type V = magetypes::simd::generic::f32x8<archmage::ScalarToken>;
    pool_body!(token, V, 8, p)
}

/// The `v3` tier (AVX2/FMA, 8-wide f32).
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub fn vsi_core_tier_v3(token: archmage::X64V3Token, p: Planes<'_>) -> f64 {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    pool_body!(token, V, 8, p)
}

/// The `v4` tier (AVX-512, 16-wide f32 — crate feature `avx512`).
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
pub fn vsi_core_tier_v4(token: archmage::X64V4Token, p: Planes<'_>) -> f64 {
    type V = magetypes::simd::generic::f32x16<archmage::X64V4Token>;
    pool_body!(token, V, 16, p)
}

/// The `neon` tier (8-wide f32).
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
pub fn vsi_core_tier_neon(token: archmage::NeonToken, p: Planes<'_>) -> f64 {
    type V = magetypes::simd::generic::f32x8<archmage::NeonToken>;
    pool_body!(token, V, 8, p)
}

/// The `wasm128` tier (8-wide f32).
#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
pub fn vsi_core_tier_wasm(token: archmage::Wasm128Token, p: Planes<'_>) -> f64 {
    type V = magetypes::simd::generic::f32x8<archmage::Wasm128Token>;
    pool_body!(token, V, 8, p)
}

/// Runtime-dispatched score: the best tier this CPU supports.
fn vsi_core(p: Planes<'_>) -> f64 {
    archmage::incant!(vsi_core_tier(p), [v4, v3, neon, wasm128, scalar])
}

/// Full VSI pipeline on prepared inputs — preprocess scalar-shared,
/// pool dispatched to the widest available tier.
pub(crate) fn vsi_planes(r: &crate::Input, d: &crate::Input, w: usize, h: usize) -> f64 {
    let mut scratch = Scratch::new();
    let p = prepare(r, d, w, h, &mut scratch);
    vsi_core(p)
}
