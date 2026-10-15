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
    /// Weights only, same `[st, st+len)` layout as `taps` — lets the
    /// contiguous-index fast path dot two flat slices.
    wt: Vec<f32>,
    /// Per output: first source index when the span's indices are
    /// consecutive (`src[j0 + k]`), `u32::MAX` when sympad wrapped.
    j0: Vec<u32>,
    /// 8-output blocks realigned to a common source window for the
    /// small-span (upscale) case: `(j0_base, k_len, wt8_offset)`.
    blk: Vec<(u32, u32, u32)>,
    /// `blk` index per 8-aligned output block, `u32::MAX` when the
    /// block isn't usable (non-contiguous span or right-edge bounds).
    blk_idx: Vec<u32>,
    /// Transposed block weights, `[k * 8 + lane]`, zero-padded.
    wt8: Vec<f32>,
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
    let mut wt = vec![0.0f32; taps.len()];
    let mut j0 = vec![u32::MAX; spans.len()];
    for (o, &(st, len)) in spans.iter().enumerate() {
        let (st, len) = (st as usize, len as usize);
        if len == 0 {
            continue;
        }
        let base = taps[st].0;
        let mut contig = true;
        for k in 0..len {
            contig &= taps[st + k].0.wrapping_sub(base) == k as u32;
            wt[st + k] = taps[st + k].1;
        }
        if contig {
            j0[o] = base;
        }
    }
    // Transposed tables for the 8-outputs-per-vector path: worthwhile
    // when a block's union window is small (upscale) — for downscale
    // the union is ~7·stride wider than the per-o spans and loses.
    let out_n = spans.len();
    let mut blk = Vec::new();
    let mut blk_idx = vec![u32::MAX; out_n / 8];
    let mut wt8 = Vec::new();
    for b in 0..out_n / 8 {
        let o0 = b * 8;
        let j0b = (0..8).map(|l| j0[o0 + l]).min().unwrap();
        if j0b == u32::MAX {
            continue;
        }
        let hi = (0..8)
            .map(|l| {
                if j0[o0 + l] == u32::MAX {
                    u32::MAX
                } else {
                    j0[o0 + l] + spans[o0 + l].1
                }
            })
            .max()
            .unwrap();
        if hi == u32::MAX {
            continue;
        }
        let k_len = hi - j0b;
        // `srow[j0b + k .. +8]` loads need 8 lanes past the window end.
        if k_len > 24 || j0b as usize + k_len as usize + 7 > in_n {
            continue;
        }
        let off = wt8.len() as u32;
        for k in 0..k_len {
            for l in 0..8 {
                let (st, len) = spans[o0 + l];
                let d = j0b as i64 + k as i64 - j0[o0 + l] as i64;
                wt8.push(if d >= 0 && d < len as i64 {
                    wt[(st as i64 + d) as usize]
                } else {
                    0.0
                });
            }
        }
        blk_idx[b] = blk.len() as u32;
        blk.push((j0b, k_len, off));
    }
    Axis {
        spans,
        taps,
        wt,
        j0,
        blk,
        blk_idx,
        wt8,
    }
}

/// Row-map driver — `f(y, out_row)` for every row. Under `parallel`
/// rows are `par_chunks_mut`-mapped when the pool has >1 thread;
/// otherwise the serial loop runs. Writes are disjoint per row either
/// way, so results are bit-identical at any thread count.
fn for_each_row(out: &mut [f32], row_w: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        out.par_chunks_mut(row_w)
            .enumerate()
            .for_each(|(y, drow)| f(y, drow));
        return;
    }
    for (y, drow) in out.chunks_mut(row_w).enumerate() {
        f(y, drow);
    }
}

/// Horizontal pass: `out[i,j] = Σ_t w·src[i, idx_t]`, accumulate f64.
fn resize_h(src: &[f32], in_w: usize, in_h: usize, ax: &Axis) -> Vec<f32> {
    let out_w = ax.spans.len();
    let mut out = vec![0.0f32; in_h * out_w];
    for_each_row(&mut out, out_w, |i, drow| {
        let srow = &src[i * in_w..(i + 1) * in_w];
        resize_h_row(srow, drow, ax);
    });
    out
}

/// One output row of the horizontal pass. Interior spans have
/// consecutive source indices (`src[j0 + k]`), so the tap loop becomes
/// a contiguous dot product; spans whose `sympad` wrap broke
/// consecutiveness keep the scalar indexed path.
fn resize_h_row(srow: &[f32], drow: &mut [f32], ax: &Axis) {
    archmage::attuned!(
        resize_h_row_tier(srow, drow, ax),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

macro_rules! resize_h_body {
    ($token:ident, $V:ty, $N:literal, $V8:ty, $srow:ident, $drow:ident, $ax:ident) => {{
        let out_w = $ax.spans.len();
        let nblk8 = out_w / 8;
        for b in 0..nblk8 {
            let o0 = b * 8;
            let bi = $ax.blk_idx[b];
            if bi != u32::MAX {
                // Transposed block: 8 outputs share one source window;
                // zero-padded weights keep per-element k order. The
                // block table is 8 lanes wide on every tier ($V8), so
                // 16-wide tiers run 8-lane blocks here.
                let (j0b, k_len, off) = $ax.blk[bi as usize];
                let mut av = <$V8>::splat_t($token, 0.0);
                for k in 0..k_len as usize {
                    let wv = <$V8>::load_t(
                        $token,
                        $ax.wt8[off as usize + k * 8..][..8].first_chunk().unwrap(),
                    );
                    let xv = <$V8>::splat_t($token, $srow[j0b as usize + k]);
                    av = av + wv * xv;
                }
                av.store($drow[o0..][..8].first_chunk_mut().unwrap());
                continue;
            }
            for o in o0..o0 + 8 {
                per_output!($token, $V, $N, $srow, $drow, $ax, o);
            }
        }
        for o in nblk8 * 8..out_w {
            per_output!($token, $V, $N, $srow, $drow, $ax, o);
        }
    }};
}

macro_rules! per_output {
    ($token:ident, $V:ty, $N:literal, $srow:ident, $drow:ident, $ax:ident, $o:ident) => {{
        let (st, len) = $ax.spans[$o];
        let (st, len) = (st as usize, len as usize);
        let j0 = $ax.j0[$o];
        if j0 != u32::MAX && len >= $N {
            let j0 = j0 as usize;
            let wt = &$ax.wt[st..st + len];
            let sv = &$srow[j0..j0 + len];
            // Vector accumulator over tap chunks, then a lane-ordered
            // fold into the f64 accumulator. Reassociates each output's
            // sum (drift ≲1e-7, inside the crate's 1e-4 golden slack).
            let mut av = <$V>::splat_t($token, 0.0);
            let mut k = 0;
            while k + $N <= len {
                let wv = <$V>::load_t($token, wt[k..][..$N].first_chunk().unwrap());
                let xv = <$V>::load_t($token, sv[k..][..$N].first_chunk().unwrap());
                av = av + wv * xv;
                k += $N;
            }
            let mut lanes = [0.0f32; $N];
            av.store(&mut lanes);
            let mut acc = 0.0f64;
            for &p in &lanes {
                acc += p as f64;
            }
            for k in k..len {
                acc += wt[k] as f64 * sv[k] as f64;
            }
            $drow[$o] = acc as f32;
        } else {
            let mut acc = 0.0f64;
            for &(idx, w) in &$ax.taps[st..st + len] {
                acc += w as f64 * $srow[idx as usize] as f64;
            }
            $drow[$o] = acc as f32;
        }
    }};
}

/// Scalar tier — generic f32x8 polyfill.
fn resize_h_row_tier_scalar_t(
    token: archmage::ScalarToken,
    srow: &[f32],
    drow: &mut [f32],
    ax: &Axis,
) {
    type V = magetypes::simd::generic::f32x8<archmage::ScalarToken>;
    resize_h_body!(token, V, 8, V, srow, drow, ax)
}

/// The `v3` tier (AVX2, 8-wide f32).
#[cfg(target_arch = "x86_64")]
#[archmage::attune]
fn resize_h_row_tier_v3_t(token: archmage::X64V3Token, srow: &[f32], drow: &mut [f32], ax: &Axis) {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    resize_h_body!(token, V, 8, V, srow, drow, ax)
}

/// The `v4x` tier (AVX-512). Resize accumulators stay 8-wide — the
/// reassociation pattern then matches `v3` bitwise, so tier parity and
/// the flat-image NaN path behave identically on both.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::attune]
fn resize_h_row_tier_v4x_t(
    token: archmage::X64V4xToken,
    srow: &[f32],
    drow: &mut [f32],
    ax: &Axis,
) {
    type V = magetypes::simd::generic::f32x8<archmage::X64V4xToken>;
    resize_h_body!(token, V, 8, V, srow, drow, ax)
}

/// The `neon` tier (8-wide f32).
#[cfg(target_arch = "aarch64")]
#[archmage::attune]
fn resize_h_row_tier_neon_t(token: archmage::NeonToken, srow: &[f32], drow: &mut [f32], ax: &Axis) {
    type V = magetypes::simd::generic::f32x8<archmage::NeonToken>;
    resize_h_body!(token, V, 8, V, srow, drow, ax)
}

/// The `wasm128` tier (8-wide f32).
#[cfg(target_arch = "wasm32")]
#[archmage::attune(wasm128)]
fn resize_h_row_tier_wasm(
    token: archmage::Wasm128Token,
    srow: &[f32],
    drow: &mut [f32],
    ax: &Axis,
) {
    type V = magetypes::simd::generic::f32x8<archmage::Wasm128Token>;
    resize_h_body!(token, V, 8, V, srow, drow, ax)
}

#[cfg(target_arch = "wasm32")]
#[archmage::attune]
fn resize_h_row_tier_wasm128_t(
    token: archmage::Wasm128Token,
    srow: &[f32],
    drow: &mut [f32],
    ax: &Axis,
) {
    resize_h_row_tier_wasm(token, srow, drow, ax)
}

/// Vertical pass over a `in_w`-strided plane.
fn resize_v(src: &[f32], w: usize, _in_h: usize, ax: &Axis) -> Vec<f32> {
    let out_h = ax.spans.len();
    let mut out = vec![0.0f32; out_h * w];
    for_each_row(&mut out, w, |o, orow| {
        let (st, len) = ax.spans[o];
        resize_v_row(&src[..], st as usize, len as usize, &ax.taps, w, orow);
    });
    out
}

/// One output row of the vertical pass: `out[x] = Σ_k w_k·src[idx_k·W+x]`.
/// Each output's taps accumulate in k order — vector lanes carry x, so
/// per-element op order is unchanged; products/acc run in f32 (the same
/// ≲1e-7 drift class the horizontal pass documents).
fn resize_v_row(
    src: &[f32],
    st: usize,
    len: usize,
    taps: &[(u32, f32)],
    w: usize,
    orow: &mut [f32],
) {
    archmage::attuned!(
        resize_v_row_tier(src, st, len, taps, w, orow),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

macro_rules! resize_v_body {
    ($token:ident, $V:ty, $N:literal, $src:ident, $st:ident, $len:ident, $taps:ident, $w:ident, $orow:ident) => {{
        let taps = &$taps[$st..$st + $len];
        let mut x = 0;
        while x + $N <= $w {
            let mut av = <$V>::splat_t($token, 0.0);
            for &(idx, wt) in taps {
                let base = idx as usize * $w + x;
                let xv = <$V>::load_t($token, $src[base..][..$N].first_chunk().unwrap());
                av = av + <$V>::splat_t($token, wt) * xv;
            }
            av.store($orow[x..][..$N].first_chunk_mut().unwrap());
            x += $N;
        }
        for xi in x..$w {
            let mut acc = 0.0f64;
            for &(idx, wt) in taps {
                acc += wt as f64 * $src[idx as usize * $w + xi] as f64;
            }
            $orow[xi] = acc as f32;
        }
    }};
}

/// Scalar tier — generic f32x8 polyfill.
fn resize_v_row_tier_scalar_t(
    token: archmage::ScalarToken,
    src: &[f32],
    st: usize,
    len: usize,
    taps: &[(u32, f32)],
    w: usize,
    orow: &mut [f32],
) {
    type V = magetypes::simd::generic::f32x8<archmage::ScalarToken>;
    resize_v_body!(token, V, 8, src, st, len, taps, w, orow)
}

/// The `v3` tier (AVX2, 8-wide f32).
#[cfg(target_arch = "x86_64")]
#[archmage::attune]
fn resize_v_row_tier_v3_t(
    token: archmage::X64V3Token,
    src: &[f32],
    st: usize,
    len: usize,
    taps: &[(u32, f32)],
    w: usize,
    orow: &mut [f32],
) {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    resize_v_body!(token, V, 8, src, st, len, taps, w, orow)
}

/// The `v4x` tier (AVX-512) — accumulators stay 8-wide for `v3` parity.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::attune]
fn resize_v_row_tier_v4x_t(
    token: archmage::X64V4xToken,
    src: &[f32],
    st: usize,
    len: usize,
    taps: &[(u32, f32)],
    w: usize,
    orow: &mut [f32],
) {
    type V = magetypes::simd::generic::f32x8<archmage::X64V4xToken>;
    resize_v_body!(token, V, 8, src, st, len, taps, w, orow)
}

/// The `neon` tier (8-wide f32).
#[cfg(target_arch = "aarch64")]
#[archmage::attune]
fn resize_v_row_tier_neon_t(
    token: archmage::NeonToken,
    src: &[f32],
    st: usize,
    len: usize,
    taps: &[(u32, f32)],
    w: usize,
    orow: &mut [f32],
) {
    type V = magetypes::simd::generic::f32x8<archmage::NeonToken>;
    resize_v_body!(token, V, 8, src, st, len, taps, w, orow)
}

/// The `wasm128` tier (8-wide f32).
#[cfg(target_arch = "wasm32")]
#[archmage::attune(wasm128)]
fn resize_v_row_tier_wasm(
    token: archmage::Wasm128Token,
    src: &[f32],
    st: usize,
    len: usize,
    taps: &[(u32, f32)],
    w: usize,
    orow: &mut [f32],
) {
    type V = magetypes::simd::generic::f32x8<archmage::Wasm128Token>;
    resize_v_body!(token, V, 8, src, st, len, taps, w, orow)
}

#[cfg(target_arch = "wasm32")]
#[archmage::attune]
fn resize_v_row_tier_wasm128_t(
    token: archmage::Wasm128Token,
    src: &[f32],
    st: usize,
    len: usize,
    taps: &[(u32, f32)],
    w: usize,
    orow: &mut [f32],
) {
    resize_v_row_tier_wasm(token, src, st, len, taps, w, orow)
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

// ========================== RGB8 ingress planes ============================
//
// `Input::from_rgb8` from `VSI.m`: u8x3 → {L,M,N} opponent planes plus a
// planar copy of the RGB used by `sdsp`. Per-pixel arithmetic is
// `c0*r + c1*g + c2*b` (left-assoc); the vector form keeps that exact
// order per lane, so every tier is bit-identical to the scalar loop.

use crate::{L_COEF, M_COEF, N_COEF};

macro_rules! rgb8_body {
    ($token:ident, $V:ty, $N:literal, $src:ident, $l:ident, $m:ident, $n:ident, $rp:ident, $gp:ident, $bp:ident) => {{
        let w = $l.len();
        let mut x = 0;
        while x + $N <= w {
            let mut rr = [0f32; $N];
            let mut gg = [0f32; $N];
            let mut bb = [0f32; $N];
            for j in 0..$N {
                let p = (x + j) * 3;
                rr[j] = $src[p] as f32;
                gg[j] = $src[p + 1] as f32;
                bb[j] = $src[p + 2] as f32;
            }
            let rv = <$V>::load_t($token, &rr);
            let gv = <$V>::load_t($token, &gg);
            let bv = <$V>::load_t($token, &bb);
            let lv = <$V>::splat_t($token, L_COEF[0]) * rv
                + <$V>::splat_t($token, L_COEF[1]) * gv
                + <$V>::splat_t($token, L_COEF[2]) * bv;
            let mv = <$V>::splat_t($token, M_COEF[0]) * rv
                + <$V>::splat_t($token, M_COEF[1]) * gv
                + <$V>::splat_t($token, M_COEF[2]) * bv;
            let nv = <$V>::splat_t($token, N_COEF[0]) * rv
                + <$V>::splat_t($token, N_COEF[1]) * gv
                + <$V>::splat_t($token, N_COEF[2]) * bv;
            lv.store($l[x..][..$N].first_chunk_mut().unwrap());
            mv.store($m[x..][..$N].first_chunk_mut().unwrap());
            nv.store($n[x..][..$N].first_chunk_mut().unwrap());
            rv.store($rp[x..][..$N].first_chunk_mut().unwrap());
            gv.store($gp[x..][..$N].first_chunk_mut().unwrap());
            bv.store($bp[x..][..$N].first_chunk_mut().unwrap());
            x += $N;
        }
        for px in x..w {
            let (r, g, b) = (
                $src[px * 3] as f32,
                $src[px * 3 + 1] as f32,
                $src[px * 3 + 2] as f32,
            );
            $l[px] = L_COEF[0] * r + L_COEF[1] * g + L_COEF[2] * b;
            $m[px] = M_COEF[0] * r + M_COEF[1] * g + M_COEF[2] * b;
            $n[px] = N_COEF[0] * r + N_COEF[1] * g + N_COEF[2] * b;
            $rp[px] = r;
            $gp[px] = g;
            $bp[px] = b;
        }
    }};
}

/// Scalar tier — generic f32x8 polyfill, same per-lane order.
fn rgb8_row_tier_scalar_t(
    token: archmage::ScalarToken,
    src: &[u8],
    l: &mut [f32],
    m: &mut [f32],
    n: &mut [f32],
    rp: &mut [f32],
    gp: &mut [f32],
    bp: &mut [f32],
) {
    type V = magetypes::simd::generic::f32x8<archmage::ScalarToken>;
    rgb8_body!(token, V, 8, src, l, m, n, rp, gp, bp)
}

/// The `v3` tier (AVX2, 8-wide f32).
#[cfg(target_arch = "x86_64")]
#[archmage::attune]
fn rgb8_row_tier_v3_t(
    token: archmage::X64V3Token,
    src: &[u8],
    l: &mut [f32],
    m: &mut [f32],
    n: &mut [f32],
    rp: &mut [f32],
    gp: &mut [f32],
    bp: &mut [f32],
) {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    rgb8_body!(token, V, 8, src, l, m, n, rp, gp, bp)
}

/// The `v4x` tier (AVX-512, 16-wide f32 — crate feature `avx512`).
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::attune]
fn rgb8_row_tier_v4x_t(
    token: archmage::X64V4xToken,
    src: &[u8],
    l: &mut [f32],
    m: &mut [f32],
    n: &mut [f32],
    rp: &mut [f32],
    gp: &mut [f32],
    bp: &mut [f32],
) {
    type V = magetypes::simd::generic::f32x16<archmage::X64V4xToken>;
    rgb8_body!(token, V, 16, src, l, m, n, rp, gp, bp)
}

/// The `neon` tier (8-wide f32).
#[cfg(target_arch = "aarch64")]
#[archmage::attune]
fn rgb8_row_tier_neon_t(
    token: archmage::NeonToken,
    src: &[u8],
    l: &mut [f32],
    m: &mut [f32],
    n: &mut [f32],
    rp: &mut [f32],
    gp: &mut [f32],
    bp: &mut [f32],
) {
    type V = magetypes::simd::generic::f32x8<archmage::NeonToken>;
    rgb8_body!(token, V, 8, src, l, m, n, rp, gp, bp)
}

/// The `wasm128` tier (8-wide f32).
#[cfg(target_arch = "wasm32")]
#[archmage::attune(wasm128)]
fn rgb8_row_tier_wasm(
    token: archmage::Wasm128Token,
    src: &[u8],
    l: &mut [f32],
    m: &mut [f32],
    n: &mut [f32],
    rp: &mut [f32],
    gp: &mut [f32],
    bp: &mut [f32],
) {
    type V = magetypes::simd::generic::f32x8<archmage::Wasm128Token>;
    rgb8_body!(token, V, 8, src, l, m, n, rp, gp, bp)
}

#[cfg(target_arch = "wasm32")]
#[archmage::attune]
fn rgb8_row_tier_wasm128_t(
    token: archmage::Wasm128Token,
    src: &[u8],
    l: &mut [f32],
    m: &mut [f32],
    n: &mut [f32],
    rp: &mut [f32],
    gp: &mut [f32],
    bp: &mut [f32],
) {
    rgb8_row_tier_wasm(token, src, l, m, n, rp, gp, bp)
}

/// One source row → row `y` of every plane (runtime tier dispatch).
fn rgb8_row(
    src: &[u8],
    l: &mut [f32],
    m: &mut [f32],
    n: &mut [f32],
    rp: &mut [f32],
    gp: &mut [f32],
    bp: &mut [f32],
) {
    archmage::attuned!(
        rgb8_row_tier(src, l, m, n, rp, gp, bp),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

/// Build the `{L,M,N}` planes and the planar RGB copy for `sdsp`.
/// Row-parallel under `parallel`; per-element arithmetic is identical
/// to the scalar `from_rgb8` loop at any thread count.
pub(crate) fn rgb8_planes(
    data: &[u8],
    w: usize,
    h: usize,
    stride: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let wh = w * h;
    let mut l = vec![0.0f32; wh];
    let mut m = vec![0.0f32; wh];
    let mut n = vec![0.0f32; wh];
    let mut rgb = vec![0.0f32; 3 * wh];
    // Three channel views of the planar rgb — lets one row closure own
    // all six outputs without unsafe.
    let (rp, gb) = rgb.split_at_mut(wh);
    let (gp, bp) = gb.split_at_mut(wh);
    let row = |y: usize,
               lrow: &mut [f32],
               mrow: &mut [f32],
               nrow: &mut [f32],
               rrow: &mut [f32],
               grow: &mut [f32],
               brow: &mut [f32]| {
        let src = &data[y * stride..y * stride + w * 3];
        rgb8_row(src, lrow, mrow, nrow, rrow, grow, brow);
    };
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        l.par_chunks_mut(w)
            .zip(m.par_chunks_mut(w))
            .zip(n.par_chunks_mut(w))
            .zip(rp.par_chunks_mut(w))
            .zip(gp.par_chunks_mut(w))
            .zip(bp.par_chunks_mut(w))
            .enumerate()
            .for_each(|(y, (((((lr, mr), nr), rr), gr), br))| {
                row(y, lr, mr, nr, rr, gr, br);
            });
        return (l, m, n, rgb);
    }
    for (y, (((((lr, mr), nr), rr), gr), br)) in l
        .chunks_mut(w)
        .zip(m.chunks_mut(w))
        .zip(n.chunks_mut(w))
        .zip(rp.chunks_mut(w))
        .zip(gp.chunks_mut(w))
        .zip(bp.chunks_mut(w))
        .enumerate()
    {
        row(y, lr, mr, nr, rr, gr, br);
    }
    (l, m, n, rgb)
}

// ============================ sRGB → Lab (D50) =============================
//
// `RGB2Lab` from `VSI.m`, verbatim: sRGB EOTF inverse, the reference's
// own XYZ matrix, **D50** reference white (Xr 0.9642, Yr 1.0,
// Zr 0.8251 — not the usual D65), ε = 0.008856, κ = 903.3.

pub(crate) fn srgb_to_lab_all_scalar(
    dr: &[f32],
    dg: &[f32],
    db: &[f32],
    ll: &mut [f32],
    la: &mut [f32],
    lb: &mut [f32],
) {
    for i in 0..dr.len() {
        let (l, a, bb) = srgb_to_lab(dr[i], dg[i], db[i]);
        ll[i] = l;
        la[i] = a;
        lb[i] = bb;
    }
}

/// `srgb_to_lab` at `f64x4` — **bit-identical** to the scalar conversion
/// on every pixel: the two `pow` calls run through `zenmetrics-math`'s
/// lane-exact fdlibm `e_pow` port and every other op is the same `f64`
/// arithmetic in the same order, so x86 tiers agree with the scalar path
/// bitwise — x86 and aarch64 NEON alike. Branch
/// thresholds become `blend` selects — same piecewise semantics. The
/// `f(t)` pow input is blended to `1.0` on `t <= ε` lanes so edge lanes
/// never take the port's scalar fallback.
///
/// Written as a macro so the whole body — `pow64_libm_4x` included —
/// lands inside the caller's attuned target-feature context and the lane
/// ops inline (a generic fn or closure body would make each
/// `__simd_inner` op a call).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
macro_rules! srgb_to_lab_exact_body {
    ($token:ident, $F64:ident, $dr:ident, $dg:ident, $db:ident, $ll:ident, $la:ident, $lb:ident) => {{
        use magetypes::simd::generic::f64x4;
        let n = $dr.len();
        let n4 = n & !3;
        for i in (0..n4).step_by(4) {
            macro_rules! lin {
                ($p:ident) => {{
                    let v = $F64::from_array_t(
                        $token,
                        core::array::from_fn(|k| $p[i + k] as f64 / 255.0),
                    );
                    // arg in [0.0521, 1.0], result in [8e-4, 1]: `BULK`
                    // (coverage) — no edge or range lane is reachable, and
                    // arg == 1.0 computes exactly 1.0 through the poly.
                    // arg in [0.0521, 1.0], result in [8e-4, 1]: `BULK`
                    // (coverage) — no edge or range lane is reachable, and
                    // arg == 1.0 computes exactly 1.0 through the poly.
                    // arg in [0.0521, 1.0], result in [8e-4, 1]: `BULK`
                    // (coverage) — no edge or range lane is reachable, and
                    // arg == 1.0 computes exactly 1.0 through the poly.
                    let hi = zenmetrics_math::fdlibm64::pow64_libm_4x_gated::<
                        { zenmetrics_math::fdlibm::coverage::BULK },
                        _,
                    >(
                        $token,
                        (v + $F64::splat_t($token, 0.055)) / $F64::splat_t($token, 1.055),
                        $F64::splat_t($token, 2.4),
                    );
                    $F64::blend(
                        v.simd_le($F64::splat_t($token, 0.04045)),
                        v / $F64::splat_t($token, 12.92),
                        hi,
                    )
                }};
            }
            let (r, g, b) = (lin!($dr), lin!($dg), lin!($db));
            let x = r * $F64::splat_t($token, 0.4124564)
                + g * $F64::splat_t($token, 0.3575761)
                + b * $F64::splat_t($token, 0.1804375);
            let y = r * $F64::splat_t($token, 0.2126729)
                + g * $F64::splat_t($token, 0.7151522)
                + b * $F64::splat_t($token, 0.0721750);
            let z = r * $F64::splat_t($token, 0.0193339)
                + g * $F64::splat_t($token, 0.1191920)
                + b * $F64::splat_t($token, 0.9503041);
            macro_rules! cbrt_or_linear {
                ($t:expr) => {{
                    let t = $t;
                    let sel = t.simd_gt($F64::splat_t($token, 0.008856));
                    let arg = $F64::blend(sel, t, $F64::splat_t($token, 1.0));
                    let hi = zenmetrics_math::fdlibm64::pow64_libm_4x_gated::<
                        { zenmetrics_math::fdlibm::coverage::BULK },
                        _,
                    >($token, arg, $F64::splat_t($token, 1.0 / 3.0));
                    let lo = (t * $F64::splat_t($token, 903.3) + $F64::splat_t($token, 16.0))
                        / $F64::splat_t($token, 116.0);
                    $F64::blend(sel, hi, lo)
                }};
            }
            let fx = cbrt_or_linear!(x / $F64::splat_t($token, 0.9642));
            let fy = cbrt_or_linear!(y);
            let fz = cbrt_or_linear!(z / $F64::splat_t($token, 0.8251));
            let va = ($F64::splat_t($token, 116.0) * fy - $F64::splat_t($token, 16.0)).to_array();
            let vb = ($F64::splat_t($token, 500.0) * (fx - fy)).to_array();
            let vc = ($F64::splat_t($token, 200.0) * (fy - fz)).to_array();
            for k in 0..4 {
                $ll[i + k] = va[k] as f32;
                $la[i + k] = vb[k] as f32;
                $lb[i + k] = vc[k] as f32;
            }
        }
        srgb_to_lab_all_scalar(
            &$dr[n4..],
            &$dg[n4..],
            &$db[n4..],
            &mut $ll[n4..],
            &mut $la[n4..],
            &mut $lb[n4..],
        );
    }};
}

#[cfg(target_arch = "aarch64")]
#[archmage::attune]
fn srgb_to_lab_tier_neon_t(
    token: archmage::NeonToken,
    dr: &[f32],
    dg: &[f32],
    db: &[f32],
    ll: &mut [f32],
    la: &mut [f32],
    lb: &mut [f32],
) {
    srgb_to_lab_exact_body!(token, f64x4, dr, dg, db, ll, la, lb);
}

#[cfg(target_arch = "x86_64")]
#[archmage::attune]
fn srgb_to_lab_tier_v3_t(
    token: archmage::X64V3Token,
    dr: &[f32],
    dg: &[f32],
    db: &[f32],
    ll: &mut [f32],
    la: &mut [f32],
    lb: &mut [f32],
) {
    srgb_to_lab_exact_body!(token, f64x4, dr, dg, db, ll, la, lb);
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::attune]
fn srgb_to_lab_tier_v4x_t(
    token: archmage::X64V4xToken,
    dr: &[f32],
    dg: &[f32],
    db: &[f32],
    ll: &mut [f32],
    la: &mut [f32],
    lb: &mut [f32],
) {
    srgb_to_lab_exact_body!(token, f64x4, dr, dg, db, ll, la, lb);
}

/// Scalar-tier entry for `attuned!` — same signature shape as the NEON
/// twin; the actual conversion is `srgb_to_lab_all_scalar`.
#[allow(dead_code)]
fn srgb_to_lab_tier_scalar_t(
    _token: archmage::ScalarToken,
    dr: &[f32],
    dg: &[f32],
    db: &[f32],
    ll: &mut [f32],
    la: &mut [f32],
    lb: &mut [f32],
) {
    srgb_to_lab_all_scalar(dr, dg, db, ll, la, lb);
}

/// Runtime-dispatched Lab fill — `attuned!` expands to the best tier.
pub(crate) fn srgb_to_lab_all(
    dr: &[f32],
    dg: &[f32],
    db: &[f32],
    ll: &mut [f32],
    la: &mut [f32],
    lb: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    archmage::attuned!(srgb_to_lab_tier(dr, dg, db, ll, la, lb), [_neon, _scalar]);
    #[cfg(target_arch = "x86_64")]
    archmage::attuned!(
        srgb_to_lab_tier(dr, dg, db, ll, la, lb),
        [_v4x(cfg(avx512)), _v3, _scalar]
    );
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    srgb_to_lab_all_scalar(dr, dg, db, ll, la, lb);
}

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

fn loggabor_256_build() -> Vec<f32> {
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

/// The log-Gabor grid is a pure function of the crate constants — a
/// per-process constant. `std` builds compute it once and reuse it;
/// `no_std` rebuilds it per call. Either way the stored bits are the
/// same `libm` results.
#[cfg(feature = "std")]
fn loggabor_256() -> &'static [f32] {
    static LG: std::sync::LazyLock<Vec<f32>> = std::sync::LazyLock::new(loggabor_256_build);
    &LG
}
#[cfg(not(feature = "std"))]
fn loggabor_256() -> Vec<f32> {
    loggabor_256_build()
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
    let [ll, la, lb] = &mut lab;
    #[cfg(feature = "parallel")]
    let ran_par = rayon::current_num_threads() > 1 && {
        use rayon::prelude::*;
        // Chunked through `srgb_to_lab_all` so the parallel path produces
        // the same bits as the serial tier dispatch — the score must not
        // depend on the rayon pool size.
        const CHUNK: usize = 4096;
        ll.par_chunks_mut(CHUNK)
            .zip(la.par_chunks_mut(CHUNK))
            .zip(lb.par_chunks_mut(CHUNK))
            .enumerate()
            .for_each(|(ci, ((l, a), b))| {
                let off = ci * CHUNK;
                let end = off + l.len();
                srgb_to_lab_all(&dr[off..end], &dg[off..end], &db[off..end], l, a, b);
            });
        true
    };
    #[cfg(not(feature = "parallel"))]
    let ran_par = false;
    if !ran_par {
        srgb_to_lab_all(&dr, &dg, &db, ll, la, lb);
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
    // 4. centre prior: coords 1..=256, centre rows/2 = 128. The
    // Gaussian window is constant per process (see `loggabor_256`) —
    // under `std` the 65k `expf`s run once for the whole process.
    let mut vs = vec![0.0f32; n * n];
    #[cfg(feature = "std")]
    {
        static GW: std::sync::LazyLock<Vec<f32>> = std::sync::LazyLock::new(|| {
            let n = SDSP_N;
            let mut gw = vec![0.0f32; n * n];
            for i in 0..n {
                let dy = (i + 1) as f32 - (n / 2) as f32;
                for j in 0..n {
                    let dx = (j + 1) as f32 - (n / 2) as f32;
                    gw[i * n + j] = libm::expf(-(dx * dx + dy * dy) / (SIGMA_D * SIGMA_D));
                }
            }
            gw
        });
        for i in 0..n * n {
            vs[i] = sf[i] * GW[i];
        }
    }
    #[cfg(not(feature = "std"))]
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
    for (&a, &b) in lab[1].iter().zip(&lab[2]).take(n * n) {
        amin = amin.min(a);
        amax = amax.max(a);
        bmin = bmin.min(b);
        bmax = bmax.max(b);
    }
    let (arange, brange) = (amax - amin, bmax - bmin);
    let mut arg = vec![0.0f32; n * n];
    for i in 0..n * n {
        let na = (lab[1][i] - amin) / arange;
        let nb = (lab[2][i] - bmin) / brange;
        let d2 = na * na + nb * nb;
        arg[i] = -d2 / (SIGMA_C * SIGMA_C);
    }
    // expf on lanes — bit-identical to scalar `libm::expf` per element.
    let mut e = vec![0.0f32; n * n];
    zenmetrics_math::vexpf_libm_into(&arg, &mut e);
    for i in 0..n * n {
        vs[i] *= 1.0 - e[i];
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
    let w2 = w.div_ceil(f);
    let h2 = h.div_ceil(f);
    // The vertical pass only consumes columns 0, F, 2F, ... . Keep their
    // horizontal sums in a compact plane, preserving each sum's order.
    let mut tmp = vec![0.0f32; w2 * h];
    for_each_row(&mut tmp, w2, |i, trow| {
        let row = &x[i * w..(i + 1) * w];
        for (oj, t) in trow.iter_mut().enumerate() {
            let j = oj * f;
            let lo = (j + s).saturating_sub(f - 1);
            let hi = (j + s).min(w - 1);
            let mut acc = 0.0f32;
            for v in &row[lo..=hi] {
                acc += v;
            }
            *t = acc;
        }
    });
    let mut out = vec![0.0f32; w2 * h2];
    let inv = 1.0 / (f * f) as f32;
    let tmp_ref = &tmp;
    for_each_row(&mut out, w2, |oi, orow| {
        let i = oi * f;
        let lo = (i + s).saturating_sub(f - 1);
        let hi = (i + s).min(h - 1);
        for (oj, o) in orow.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for r in lo..=hi {
                acc += tmp_ref[r * w2 + oj];
            }
            *o = acc * inv;
        }
    });
    (w2, h2, out)
}

#[cfg(test)]
mod decimation_tests {
    use super::*;

    #[test]
    fn retained_columns_match_full_box_filter_bits() {
        for (w, h) in [(1, 1), (1, 17), (19, 1), (7, 11), (32, 29), (65, 63)] {
            let x: Vec<f32> = (0..w * h)
                .map(|i| ((i * 37 % 251) as f32 - 125.0) / ((i % 7 + 1) as f32))
                .collect();
            for f in 1..=12 {
                let (dw, dh, actual) = downsample_decimate(&x, w, h, f);
                assert_eq!((dw, dh), (w.div_ceil(f), h.div_ceil(f)));
                for oy in 0..dh {
                    for ox in 0..dw {
                        let mut expected = 0.0f32;
                        for dy in 0..f {
                            let y = (oy * f + f / 2) as isize - (f - 1) as isize + dy as isize;
                            if !(0..h as isize).contains(&y) {
                                continue;
                            }
                            let mut row = 0.0f32;
                            for dx in 0..f {
                                let xx = (ox * f + f / 2) as isize - (f - 1) as isize + dx as isize;
                                if (0..w as isize).contains(&xx) {
                                    row += x[y as usize * w + xx as usize];
                                }
                            }
                            expected += row;
                        }
                        expected *= 1.0 / (f * f) as f32;
                        assert_eq!(
                            actual[oy * dw + ox].to_bits(),
                            expected.to_bits(),
                            "{w}x{h}, F={f}, ({ox},{oy})"
                        );
                    }
                }
            }
        }
    }
}

/// `conv2(Y, K, 'same')` for a 3×3 kernel — flipped kernel,
/// zero-padded (same as `crates/fsim`).
fn conv2_same_3x3(y: &[f32], w: usize, h: usize, k: &[[f32; 3]; 3]) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    for_each_row(&mut out, w, |i, orow| {
        for (j, o) in orow.iter_mut().enumerate() {
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
            *o = acc;
        }
    });
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
        if rayon::current_num_threads() <= 1 {
            return (fa(), fb());
        }
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
    ($token:ident, $F32:ty, $LANES:expr, $pow:path, $p:ident, $map:ident) => {{
        let mut $map = $map;
        let n = $p.w * $p.h;
        let (sm1, sm2) = $p.sm;
        let (g1, g2) = $p.g;
        let (m1, m2) = $p.m;
        let (n1, n2) = $p.n;
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        let cos_lambda_pi = libm::cosf(LAMBDA * core::f32::consts::PI);
        let alpha_v = <$F32>::splat_t($token, ALPHA);
        let lambda_v = <$F32>::splat_t($token, LAMBDA);
        let cos_v = <$F32>::splat_t($token, cos_lambda_pi);
        let zero_v = <$F32>::zero_t($token);
        let sign_bit = <$F32>::splat_t($token, -0.0).bitcast_to_i32();
        let mut i = 0usize;
        while i + $LANES <= n {
            let a = <$F32>::from_array_t($token, core::array::from_fn(|k| sm1[i + k]));
            let b = <$F32>::from_array_t($token, core::array::from_fn(|k| sm2[i + k]));
            let x = <$F32>::from_array_t($token, core::array::from_fn(|k| g1[i + k]));
            let z = <$F32>::from_array_t($token, core::array::from_fn(|k| g2[i + k]));
            let mi = <$F32>::from_array_t($token, core::array::from_fn(|k| m1[i + k]));
            let mj = <$F32>::from_array_t($token, core::array::from_fn(|k| m2[i + k]));
            let qi = <$F32>::from_array_t($token, core::array::from_fn(|k| n1[i + k]));
            let qj = <$F32>::from_array_t($token, core::array::from_fn(|k| n2[i + k]));
            let cvs = <$F32>::splat_t($token, C_VS);
            let cgm = <$F32>::splat_t($token, C_GM);
            let cch = <$F32>::splat_t($token, C_CHROM);
            let two = <$F32>::splat_t($token, 2.0);
            let vssim = (two * a * b + cvs) / (a * a + b * b + cvs);
            let gsim = (two * x * z + cgm) / (x * x + z * z + cgm);
            let isim = (two * mi * mj + cch) / (mi * mi + mj * mj + cch);
            let qsim = (two * qi * qj + cch) / (qi * qi + qj * qj + cch);
            // Lane-exact `libm::powf` (zenmetrics-math `fdlibm`): every
            // lane is bit-identical to the scalar call, so the powers run
            // in vector. `real_pow_lambda` is transcribed per lane: for
            // `z >= 0` it is `powf(z, λ)`, otherwise (negative or NaN `z`)
            // `powf(-z, λ) * cosf(λπ)`, with `-z` a sign-bit flip like
            // Rust's `-z`.
            let gpow = $pow($token, gsim, alpha_v);
            let iq = isim * qsim;
            let nonneg = iq.simd_ge(zero_v);
            let neg_iq = <$F32>::from_i32_bitcast_t($token, iq.bitcast_to_i32() ^ sign_bit);
            let lam = $pow($token, <$F32>::blend(nonneg, iq, neg_iq), lambda_v);
            let iqreal = <$F32>::blend(nonneg, lam, lam * cos_v);
            // Per-lane scalar for max and the accumulation (fixed lane
            // order — the same scalar expression as the tail loop, so
            // tiers agree bitwise).
            let va = a.to_array();
            let vb = b.to_array();
            let vg = gpow.to_array();
            let vv = vssim.to_array();
            let vr = iqreal.to_array();
            for k in 0..$LANES {
                let w = if va[k].is_nan() || vb[k].is_nan() {
                    f32::NAN
                } else {
                    va[k].max(vb[k])
                };
                let contrib = vg[k] * vv[k] * vr[k] * w;
                num += contrib as f64;
                den += w as f64;
                // Canonical `qualitymap` = SimMatrixC (score = Σcontrib/Σw).
                if let Some(m) = $map.as_deref_mut() {
                    m[i + k] = contrib;
                }
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
            if let Some(m) = $map.as_deref_mut() {
                m[i] = contrib;
            }
            i += 1;
        }
        num / den
    }};
}

/// Scalar tier (the magetypes generic vec over `ScalarToken` — same
/// lane count as `v3`, so per-lane extraction stays fixed-order).
pub fn vsi_core_tier_scalar_t(
    token: archmage::ScalarToken,
    p: Planes<'_>,
    map: Option<&mut [f32]>,
) -> f64 {
    type V = magetypes::simd::generic::f32x8<archmage::ScalarToken>;
    pool_body!(token, V, 8, zenmetrics_math::fdlibm::powf_libm_8x, p, map)
}

/// The `v3` tier (AVX2/FMA, 8-wide f32).
#[cfg(target_arch = "x86_64")]
#[archmage::attune]
pub fn vsi_core_tier_v3_t(
    token: archmage::X64V3Token,
    p: Planes<'_>,
    map: Option<&mut [f32]>,
) -> f64 {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    pool_body!(token, V, 8, zenmetrics_math::fdlibm::powf_libm_8x, p, map)
}

/// The `v4x` tier (AVX-512, 16-wide f32 — crate feature `avx512`).
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::attune]
pub fn vsi_core_tier_v4x_t(
    token: archmage::X64V4xToken,
    p: Planes<'_>,
    map: Option<&mut [f32]>,
) -> f64 {
    type V = magetypes::simd::generic::f32x16<archmage::X64V4xToken>;
    pool_body!(token, V, 16, zenmetrics_math::fdlibm::powf_libm_16x, p, map)
}

/// The `neon` tier (8-wide f32).
#[cfg(target_arch = "aarch64")]
#[archmage::attune]
pub fn vsi_core_tier_neon_t(
    token: archmage::NeonToken,
    p: Planes<'_>,
    map: Option<&mut [f32]>,
) -> f64 {
    type V = magetypes::simd::generic::f32x8<archmage::NeonToken>;
    pool_body!(token, V, 8, zenmetrics_math::fdlibm::powf_libm_8x, p, map)
}

/// The `wasm128` tier (8-wide f32).
#[cfg(target_arch = "wasm32")]
#[archmage::attune(wasm128)]
pub fn vsi_core_tier_wasm(
    token: archmage::Wasm128Token,
    p: Planes<'_>,
    map: Option<&mut [f32]>,
) -> f64 {
    type V = magetypes::simd::generic::f32x8<archmage::Wasm128Token>;
    pool_body!(token, V, 8, zenmetrics_math::fdlibm::powf_libm_8x, p, map)
}

#[cfg(target_arch = "wasm32")]
#[archmage::attune]
fn vsi_core_tier_wasm128_t(
    token: archmage::Wasm128Token,
    p: Planes<'_>,
    map: Option<&mut [f32]>,
) -> f64 {
    vsi_core_tier_wasm(token, p, map)
}

/// Runtime-dispatched score: the best tier this CPU supports.
fn vsi_core(p: Planes<'_>, map: Option<&mut [f32]>) -> f64 {
    archmage::attuned!(
        vsi_core_tier(p, map),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

/// Full VSI pipeline on prepared inputs — preprocess scalar-shared,
/// pool dispatched to the widest available tier.
pub(crate) fn vsi_planes(
    r: &crate::Input,
    d: &crate::Input,
    w: usize,
    h: usize,
    map: Option<&mut [f32]>,
) -> f64 {
    let mut scratch = Scratch::new();
    let p = prepare(r, d, w, h, &mut scratch);
    vsi_core(p, map)
}

#[cfg(test)]
mod resize_block_tests {
    use super::*;

    /// Vectorized resize paths (contiguous dots, transposed blocks)
    /// against the scalar indexed reference — drift must stay in the
    /// f32-reassociation class (≲1e-5), never a real error.
    #[test]
    fn resize_simd_matches_scalar_within_reassoc() {
        for (in_w, out_w) in [
            (64usize, 256usize),
            (256, 64),
            (2898, 256),
            (64, 128),
            (7, 19),
            (19, 7),
        ] {
            let ax = axis_taps(in_w, out_w);
            let in_h = 3;
            let src: Vec<f32> = (0..in_w * in_h)
                .map(|i| ((i * 37 % 251) as f32 - 125.0) / 7.0)
                .collect();
            let mut want = vec![0.0f32; in_h * out_w];
            for i in 0..in_h {
                let srow = &src[i * in_w..(i + 1) * in_w];
                for (o, &(st, len)) in ax.spans.iter().enumerate() {
                    let mut acc = 0.0f64;
                    for &(idx, w) in &ax.taps[st as usize..(st + len) as usize] {
                        acc += w as f64 * srow[idx as usize] as f64;
                    }
                    want[i * out_w + o] = acc as f32;
                }
            }
            let got = resize_h(&src, in_w, in_h, &ax);
            let mut worst = 0.0f64;
            for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                let d = (g as f64 - w as f64).abs();
                assert!(d < 1e-5, "in {in_w}->out {out_w} @{i}: {d:e}");
                if d > worst {
                    worst = d;
                }
            }
            eprintln!("in {in_w} -> out {out_w}: worst {worst:e}");
        }
    }
}

#[cfg(test)]
mod pool_exact_tests {
    use super::*;

    /// The original per-element pool: scalar `libm::powf` and
    /// `real_pow_lambda` on every element (the tail-loop expression).
    fn pool_reference(p: Planes<'_>, mut map: Option<&mut [f32]>) -> f64 {
        let n = p.w * p.h;
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for i in 0..n {
            let (a, b) = (p.sm.0[i], p.sm.1[i]);
            let vssim = (2.0 * a * b + C_VS) / (a * a + b * b + C_VS);
            let (x, z) = (p.g.0[i], p.g.1[i]);
            let gsim = (2.0 * x * z + C_GM) / (x * x + z * z + C_GM);
            let (ia, ib) = (p.m.0[i], p.m.1[i]);
            let isim = (2.0 * ia * ib + C_CHROM) / (ia * ia + ib * ib + C_CHROM);
            let (qa, qb) = (p.n.0[i], p.n.1[i]);
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
            if let Some(m) = map.as_deref_mut() {
                m[i] = contrib;
            }
        }
        num / den
    }

    /// The pool as it was before the lane `powf` (a4d0b8fc): per-lane
    /// scalar `libm::powf` / `real_pow_lambda` inside the vector loop.
    macro_rules! pool_body_before {
        ($token:ident, $F32:ty, $LANES:expr, $p:ident, $map:ident) => {{
            let mut $map = $map;
            let n = $p.w * $p.h;
            let (sm1, sm2) = $p.sm;
            let (g1, g2) = $p.g;
            let (m1, m2) = $p.m;
            let (n1, n2) = $p.n;
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            let mut i = 0usize;
            while i + $LANES <= n {
                let a = <$F32>::from_array_t($token, core::array::from_fn(|k| sm1[i + k]));
                let b = <$F32>::from_array_t($token, core::array::from_fn(|k| sm2[i + k]));
                let x = <$F32>::from_array_t($token, core::array::from_fn(|k| g1[i + k]));
                let z = <$F32>::from_array_t($token, core::array::from_fn(|k| g2[i + k]));
                let mi = <$F32>::from_array_t($token, core::array::from_fn(|k| m1[i + k]));
                let mj = <$F32>::from_array_t($token, core::array::from_fn(|k| m2[i + k]));
                let qi = <$F32>::from_array_t($token, core::array::from_fn(|k| n1[i + k]));
                let qj = <$F32>::from_array_t($token, core::array::from_fn(|k| n2[i + k]));
                let cvs = <$F32>::splat_t($token, C_VS);
                let cgm = <$F32>::splat_t($token, C_GM);
                let cch = <$F32>::splat_t($token, C_CHROM);
                let two = <$F32>::splat_t($token, 2.0);
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
                    // Canonical `qualitymap` = SimMatrixC (score = Σcontrib/Σw).
                    if let Some(m) = $map.as_deref_mut() {
                        m[i + k] = contrib;
                    }
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
                if let Some(m) = $map.as_deref_mut() {
                    m[i] = contrib;
                }
                i += 1;
            }
            num / den
        }};
    }

    fn before_scalar(p: Planes<'_>, map: Option<&mut [f32]>) -> f64 {
        let token = archmage::ScalarToken;
        type V = magetypes::simd::generic::f32x8<archmage::ScalarToken>;
        pool_body_before!(token, V, 8, p, map)
    }

    #[cfg(target_arch = "x86_64")]
    #[archmage::arcane]
    fn before_v3(token: archmage::X64V3Token, p: Planes<'_>, map: Option<&mut [f32]>) -> f64 {
        type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
        pool_body_before!(token, V, 8, p, map)
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[archmage::arcane]
    fn before_v4x(token: archmage::X64V4xToken, p: Planes<'_>, map: Option<&mut [f32]>) -> f64 {
        type V = magetypes::simd::generic::f32x16<archmage::X64V4xToken>;
        pool_body_before!(token, V, 16, p, map)
    }

    #[cfg(target_arch = "aarch64")]
    #[archmage::arcane]
    fn before_neon(token: archmage::NeonToken, p: Planes<'_>, map: Option<&mut [f32]>) -> f64 {
        type V = magetypes::simd::generic::f32x8<archmage::NeonToken>;
        pool_body_before!(token, V, 8, p, map)
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 32) as u32
        }
        fn unit(&mut self) -> f32 {
            (self.next() >> 8) as f32 / (1u32 << 24) as f32
        }
    }

    /// Plane sets: metric-range values (positive saliency/gradients, signed
    /// opponent channels), identical reference/distorted planes (every
    /// similarity exactly 1, the `x = 1` powf path), opponent channels of
    /// opposite sign (`ISim·QSim < 0`, the `cos(λπ)` branch), and IEEE
    /// specials (±0, ±inf, NaN, subnormals) in every plane.
    fn plane_sets(w: usize, h: usize) -> Vec<[Vec<f32>; 8]> {
        let n = w * h;
        let mut rng = Rng(0x7651_0b1e);
        let mut sets = Vec::new();
        let mut metric = || -> [Vec<f32>; 8] {
            core::array::from_fn(|c| {
                (0..n)
                    .map(|_| match c {
                        0 | 1 => rng.unit(),
                        2 | 3 => rng.unit() * 600.0,
                        _ => (rng.unit() - 0.5) * 200.0,
                    })
                    .collect()
            })
        };
        let base = metric();
        sets.push(base.clone());
        let mut same = base.clone();
        for c in [1, 3, 5, 7] {
            same[c] = same[c - 1].clone();
        }
        sets.push(same);
        let mut flip = base.clone();
        for c in [5, 7] {
            flip[c] = flip[c - 1].iter().map(|v| -v * 1.7).collect();
        }
        sets.push(flip);
        let specials = [
            0.0f32,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            1.0,
            -1.0,
            1e30,
        ];
        let mut special = base;
        for (c, plane) in special.iter_mut().enumerate() {
            for (i, v) in plane.iter_mut().enumerate() {
                if (i * 7 + c * 3) % 5 == 0 {
                    *v = specials[(i + c) % specials.len()];
                }
            }
        }
        sets.push(special);
        sets
    }

    type Pool = dyn Fn(Planes<'_>, Option<&mut [f32]>) -> f64;

    /// `after` (the shipped tier) must equal `before` (the same tier before
    /// the lane `powf`): every qualitymap element bit for bit, NaN payloads
    /// included, and the score bit for bit unless it is NaN. Against the
    /// per-element scalar reference, every non-NaN value is bit-identical
    /// and NaN falls on the same elements (Rust leaves the NaN sign/payload
    /// of arithmetic unspecified; the vector similarity terms predate this
    /// change).
    fn check_tier(name: &str, after: &Pool, before: &Pool) {
        let same = |x: f32, y: f32| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan());
        // Widths cover whole vectors only, a tail, and a tail-only plane.
        for (w, h) in [(5usize, 1usize), (16, 4), (37, 9), (64, 33)] {
            for (s, planes) in plane_sets(w, h).iter().enumerate() {
                let p = Planes {
                    sm: (&planes[0], &planes[1]),
                    g: (&planes[2], &planes[3]),
                    m: (&planes[4], &planes[5]),
                    n: (&planes[6], &planes[7]),
                    w,
                    h,
                };
                let mut ref_map = vec![0.0f32; w * h];
                let mut old_map = vec![0.0f32; w * h];
                let mut new_map = vec![0.0f32; w * h];
                let want = pool_reference(p, Some(&mut ref_map));
                let old = before(p, Some(&mut old_map));
                let got = after(p, Some(&mut new_map));
                for i in 0..w * h {
                    assert_eq!(
                        new_map[i].to_bits(),
                        old_map[i].to_bits(),
                        "{name} {w}x{h} set {s} map[{i}]: {:#x} vs before {:#x}; planes {:?}",
                        new_map[i].to_bits(),
                        old_map[i].to_bits(),
                        planes.iter().map(|pl| pl[i]).collect::<Vec<_>>()
                    );
                    assert!(
                        same(new_map[i], ref_map[i]),
                        "{name} {w}x{h} set {s} map[{i}]: {} vs reference {}",
                        new_map[i],
                        ref_map[i]
                    );
                }
                // The score's NaN sign is not pinned even with every
                // contribution bit-identical: LLVM may commute the f64
                // `num += contrib` operands, and x86 propagates the first
                // operand's NaN. Measured on v3 for the specials set.
                assert!(
                    got.to_bits() == old.to_bits() || (got.is_nan() && old.is_nan()),
                    "{name} {w}x{h} set {s}: score {got} vs before {old}"
                );
                assert!(
                    got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                    "{name} {w}x{h} set {s}: score {got} vs reference {want}"
                );
                assert_eq!(after(p, None).to_bits(), got.to_bits());
            }
        }
    }

    /// Every tier's pool equals the per-element scalar `libm` pool bit for
    /// bit: the score and every qualitymap element.
    #[test]
    fn pool_matches_scalar_libm_reference_bitwise() {
        check_tier(
            "scalar",
            &move |p, m| vsi_core_tier_scalar_t(archmage::ScalarToken, p, m),
            &before_scalar,
        );
        #[cfg(target_arch = "x86_64")]
        {
            use archmage::SimdToken;
            if let Some(t) = archmage::X64V3Token::summon() {
                check_tier(
                    "v3",
                    &move |p, m| vsi_core_tier_v3_t(t, p, m),
                    &move |p, m| before_v3(t, p, m),
                );
            }
            #[cfg(feature = "avx512")]
            if let Some(t) = archmage::X64V4xToken::summon() {
                check_tier(
                    "v4x",
                    &move |p, m| vsi_core_tier_v4x_t(t, p, m),
                    &move |p, m| before_v4x(t, p, m),
                );
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            use archmage::SimdToken;
            if let Some(t) = archmage::NeonToken::summon() {
                check_tier(
                    "neon",
                    &move |p, m| vsi_core_tier_neon_t(t, p, m),
                    &move |p, m| before_neon(t, p, m),
                );
            }
        }
    }
}
