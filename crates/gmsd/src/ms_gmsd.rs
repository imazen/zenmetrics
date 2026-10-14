//! Paper-derived MS-GMSD/MS-GMSDc. This is not author-software parity.
//! Numerical conventions are frozen in the gmsd-chroma preregistration.
use alloc::vec::Vec;
#[cfg(target_arch = "aarch64")]
use archmage::NeonToken;
#[cfg(target_arch = "wasm32")]
use archmage::Wasm128Token;
#[cfg(target_arch = "x86_64")]
use archmage::X64V3Token;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use archmage::X64V4xToken;
use archmage::attune;

pub(crate) struct Planes {
    // Y reference/distortion, I reference/distortion, Q reference/distortion.
    data: [Vec<f64>; 6],
    pub(crate) width: usize,
    pub(crate) height: usize,
    pitch: usize,
}

impl Planes {
    fn empty(width: usize, height: usize) -> Self {
        let pitch = super::kernel::padded_pitch(width);
        Self {
            data: core::array::from_fn(|_| alloc::vec![0.0; pitch*(height+2)]),
            width,
            height,
            pitch,
        }
    }

    pub(crate) fn half(&self) -> Self {
        let mut out = Self::empty(self.width.div_ceil(2), self.height.div_ceil(2));
        let (dp, dw, sh, sw, sp) = (out.pitch, out.width, self.height, self.width, self.pitch);
        for (dest, a) in out.data.iter_mut().zip(&self.data) {
            for_each_row(&mut dest[dp..dp + out.height * dp], dp, |y, orow| {
                let y0 = 2 * y + 1;
                let y1 = (2 * y + 2).min(sh);
                let r0 = &a[y0 * sp..y0 * sp + sp];
                let r1 = &a[y1 * sp..y1 * sp + sp];
                archmage::attuned!(
                    half_row(r0, r1, orow, sw, dw),
                    [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
                );
            });
        }
        out
    }
}

/// Row-map driver — `f(y, out_row)` for every `row_w`-wide row. Under
/// `parallel` rows are `par_chunks_mut`-mapped when the pool has >1
/// thread; writes are disjoint per row, identical at any thread count.
fn for_each_row(out: &mut [f64], row_w: usize, f: impl Fn(usize, &mut [f64]) + Sync) {
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

// YIQ coefficients per channel — `data` is laid out
// `[yr, yd, cr, cd, br, bd]` (side-major pairs).
const COEF: [[f64; 3]; 3] = [
    [0.299, 0.587, 0.114],
    [0.595716, -0.274453, -0.321263],
    [0.211456, -0.522591, 0.311135],
];

/// One RGB8 source row → the Y/I/Q interior of three destination rows.
/// Each lane's arithmetic is the scalar `a*R + b*G + c*B` in f64 (u8->f64
/// conversion is exact, every output element is independent), so the
/// vector form is bit-identical to the element loop.
#[attune(_*, +v4x(cfg(avx512)), define(f64x8))]
fn fill_row_inner(
    token: Token,
    src: &[u8],
    w: usize,
    or: &mut [f64],
    oi: &mut [f64],
    oq: &mut [f64],
) {
    let cvs: [[f64x8; 3]; 3] =
        core::array::from_fn(|c| core::array::from_fn(|k| f64x8::splat_t(token, COEF[c][k])));
    let mut x = 0;
    while x + 8 <= w {
        let rv = f64x8::from_array_t(token, core::array::from_fn(|j| f64::from(src[3 * (x + j)])));
        let gv = f64x8::from_array_t(
            token,
            core::array::from_fn(|j| f64::from(src[3 * (x + j) + 1])),
        );
        let bv = f64x8::from_array_t(
            token,
            core::array::from_fn(|j| f64::from(src[3 * (x + j) + 2])),
        );
        let yw = (cvs[0][0] * rv + cvs[0][1] * gv) + cvs[0][2] * bv;
        let iw = (cvs[1][0] * rv + cvs[1][1] * gv) + cvs[1][2] * bv;
        let qw = (cvs[2][0] * rv + cvs[2][1] * gv) + cvs[2][2] * bv;
        or[1 + x..1 + x + 8].copy_from_slice(&yw.to_array());
        oi[1 + x..1 + x + 8].copy_from_slice(&iw.to_array());
        oq[1 + x..1 + x + 8].copy_from_slice(&qw.to_array());
        x += 8;
    }
    while x < w {
        let i = 3 * x;
        or[1 + x] = (COEF[0][0] * f64::from(src[i]) + COEF[0][1] * f64::from(src[i + 1]))
            + COEF[0][2] * f64::from(src[i + 2]);
        oi[1 + x] = (COEF[1][0] * f64::from(src[i]) + COEF[1][1] * f64::from(src[i + 1]))
            + COEF[1][2] * f64::from(src[i + 2]);
        oq[1 + x] = (COEF[2][0] * f64::from(src[i]) + COEF[2][1] * f64::from(src[i + 1]))
            + COEF[2][2] * f64::from(src[i + 2]);
        x += 1;
    }
}

/// `fill_row_inner` with the token threaded through `attuned!`.
macro_rules! row_entry {
    ($name:ident, $token:ident, $inner:ident, $($arg:ident : $ty:ty),*) => {
        #[archmage::attune]
        pub(crate) fn $name(token: $token, $($arg: $ty),*) {
            $inner(token, $($arg),*);
        }
    };
}
#[cfg(target_arch = "x86_64")]
row_entry!(
    fill_row_v3_t,
    X64V3Token,
    fill_row_inner_v3,
    src: &[u8],
    w: usize,
    or: &mut [f64],
    oi: &mut [f64],
    oq: &mut [f64]
);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
row_entry!(
    fill_row_v4x_t,
    X64V4xToken,
    fill_row_inner_v4x,
    src: &[u8],
    w: usize,
    or: &mut [f64],
    oi: &mut [f64],
    oq: &mut [f64]
);
#[cfg(target_arch = "aarch64")]
row_entry!(
    fill_row_neon_t,
    NeonToken,
    fill_row_inner_neon,
    src: &[u8],
    w: usize,
    or: &mut [f64],
    oi: &mut [f64],
    oq: &mut [f64]
);
#[cfg(target_arch = "wasm32")]
row_entry!(
    fill_row_wasm128_t,
    Wasm128Token,
    fill_row_inner_wasm128,
    src: &[u8],
    w: usize,
    or: &mut [f64],
    oi: &mut [f64],
    oq: &mut [f64]
);
pub(crate) fn fill_row_scalar_t(
    token: archmage::ScalarToken,
    src: &[u8],
    w: usize,
    or: &mut [f64],
    oi: &mut [f64],
    oq: &mut [f64],
) {
    fill_row_inner_scalar(token, src, w, or, oi, oq);
}

/// One output row of the 2x2 f64 decimator: reads the two source rows
/// `r0`/`r1` at stride-2 columns. The `x1` column clamp only fires on the
/// final output of an odd width, so the interior (`2x + 2 <= sw`) runs
/// eight lanes per gather group; every lane is the same independent
/// `((a + b) + c + d) * 0.25` expression as the scalar loop —
/// bit-identical.
#[attune(_*, +v4x(cfg(avx512)), define(f64x8))]
fn half_row_inner(token: Token, r0: &[f64], r1: &[f64], orow: &mut [f64], sw: usize, ow: usize) {
    let quarter = f64x8::splat_t(token, 0.25);
    let interior = sw / 2;
    let mut x = 0;
    while x + 8 <= interior {
        let p00 = f64x8::from_array_t(token, core::array::from_fn(|j| r0[2 * (x + j) + 1]));
        let p01 = f64x8::from_array_t(token, core::array::from_fn(|j| r0[2 * (x + j) + 2]));
        let p10 = f64x8::from_array_t(token, core::array::from_fn(|j| r1[2 * (x + j) + 1]));
        let p11 = f64x8::from_array_t(token, core::array::from_fn(|j| r1[2 * (x + j) + 2]));
        let v = (((p00 + p01) + p10) + p11) * quarter;
        orow[1 + x..1 + x + 8].copy_from_slice(&v.to_array());
        x += 8;
    }
    while x < ow {
        let x0 = 2 * x + 1;
        let x1 = (2 * x + 2).min(sw);
        orow[1 + x] = (((r0[x0] + r0[x1]) + r1[x0]) + r1[x1]) * 0.25;
        x += 1;
    }
}

#[cfg(target_arch = "x86_64")]
row_entry!(
    half_row_v3_t,
    X64V3Token,
    half_row_inner_v3,
    r0: &[f64],
    r1: &[f64],
    orow: &mut [f64],
    sw: usize,
    ow: usize
);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
row_entry!(
    half_row_v4x_t,
    X64V4xToken,
    half_row_inner_v4x,
    r0: &[f64],
    r1: &[f64],
    orow: &mut [f64],
    sw: usize,
    ow: usize
);
#[cfg(target_arch = "aarch64")]
row_entry!(
    half_row_neon_t,
    NeonToken,
    half_row_inner_neon,
    r0: &[f64],
    r1: &[f64],
    orow: &mut [f64],
    sw: usize,
    ow: usize
);
#[cfg(target_arch = "wasm32")]
row_entry!(
    half_row_wasm128_t,
    Wasm128Token,
    half_row_inner_wasm128,
    r0: &[f64],
    r1: &[f64],
    orow: &mut [f64],
    sw: usize,
    ow: usize
);
pub(crate) fn half_row_scalar_t(
    token: archmage::ScalarToken,
    r0: &[f64],
    r1: &[f64],
    orow: &mut [f64],
    sw: usize,
    ow: usize,
) {
    half_row_inner_scalar(token, r0, r1, orow, sw, ow);
}

/// Level 0 fused pipeline: RGB rows feed a four-row-per-channel ring
/// (~650 KB at 8 MP) that emits each full-resolution similarity-map row
/// (Y channels) and each half-scale plane row (all six channels) as soon
/// as its inputs land. The six full-size level-0 planes are never
/// materialized — about 400 MB less plane traffic per call.
///
/// Work splits into even `BAND`-row bands, each with a private ring; a
/// band refills the boundary data rows its neighbours also fill (~3%
/// duplicate work at 64 rows/band) so map and half rows land in disjoint
/// output ranges. Every value goes through the identical element
/// expression the staged version used — bit-identical at any thread count.
const STREAM_BAND: usize = 64;

#[allow(clippy::too_many_arguments)]
fn stream_band(
    r: &[u8],
    d: &[u8],
    w: usize,
    h: usize,
    stride: usize,
    m0: usize,
    m1: usize,
    pitch: usize,
    np: usize,
    now: usize,
    map_band: &mut [f64],
    mut planes: [&mut [f64]; 6],
) {
    const DEPTH: usize = 4;
    let mut ring = alloc::vec![0.0; 6 * DEPTH * pitch];
    let zero = alloc::vec![0.0f64; pitch];
    let noh = planes[0].len() / np;
    let emit_map = |ring: &[f64], m: usize, orow: &mut [f64]| {
        // Map row m reads plane rows m, m+1, m+2 — data rows m-1, m, m+1,
        // with pad rows at both ends.
        let rows: [[&[f64]; 3]; 2] = core::array::from_fn(|plane| {
            core::array::from_fn(|dy| {
                let drow = m + dy; // plane row index
                if drow == 0 || drow > h {
                    &zero[..]
                } else {
                    let base = plane * DEPTH * pitch + ((drow - 1) % DEPTH) * pitch;
                    &ring[base..base + pitch]
                }
            })
        });
        archmage::attuned!(
            similarity_row(&rows, w, 170.0, orow),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    };
    // Data rows this band must fill: [m0-1, m1] clamped to 0..h.
    let d_lo = m0.saturating_sub(1);
    let d_hi = m1.min(h - 1);
    for sy in d_lo..=d_hi {
        let slot = sy % DEPTH;
        for side in 0..2 {
            let rgb = if side == 0 { r } else { d };
            let src = &rgb[sy * stride..sy * stride + 3 * w];
            let (ch0, rest) = ring.split_at_mut(2 * DEPTH * pitch);
            let (ch1, ch2) = rest.split_at_mut(2 * DEPTH * pitch);
            let (yc, ic, qc) = if side == 0 {
                (
                    &mut ch0[slot * pitch..(slot + 1) * pitch],
                    &mut ch1[slot * pitch..(slot + 1) * pitch],
                    &mut ch2[slot * pitch..(slot + 1) * pitch],
                )
            } else {
                (
                    &mut ch0[DEPTH * pitch + slot * pitch..(DEPTH + slot + 1) * pitch],
                    &mut ch1[DEPTH * pitch + slot * pitch..(DEPTH + slot + 1) * pitch],
                    &mut ch2[DEPTH * pitch + slot * pitch..(DEPTH + slot + 1) * pitch],
                )
            };
            archmage::attuned!(
                fill_row(src, w, yc, ic, qc),
                [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
            );
        }
        if sy > d_lo {
            let m = sy - 1;
            if m >= m0 {
                emit_map(&ring, m, &mut map_band[(m - m0) * w..(m - m0 + 1) * w]);
            }
        }
        if sy % 2 == 1 && sy > m0 && sy <= m1 {
            // Half row yp reads data rows 2yp and 2yp+1 — both filled.
            let yp = (sy - 1) / 2;
            let yb = yp - (m0 / 2);
            for (ch, plane) in planes.iter_mut().enumerate() {
                let base = ch * DEPTH * pitch;
                let r0 = &ring
                    [base + ((sy - 1) % DEPTH) * pitch..base + ((sy - 1) % DEPTH) * pitch + pitch];
                let r1 = &ring[base + slot * pitch..base + slot * pitch + pitch];
                let orow = &mut plane[yb * np..(yb + 1) * np];
                archmage::attuned!(
                    half_row(r0, r1, orow, w, now),
                    [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
                );
            }
        }
    }
    // Bottom edge: map row h-1 reads the pad row h+1.
    if m1 == h {
        emit_map(&ring, h - 1, &mut map_band[(h - 1 - m0) * w..(h - m0) * w]);
        // Odd height: the last half row folds the last data row against itself.
        if h % 2 == 1 {
            let slot = (h - 1) % DEPTH;
            let yb = noh - 1;
            for (ch, plane) in planes.iter_mut().enumerate() {
                let base = ch * DEPTH * pitch;
                let r0 = &ring[base + slot * pitch..base + slot * pitch + pitch];
                let orow = &mut plane[yb * np..(yb + 1) * np];
                archmage::attuned!(
                    half_row(r0, r0, orow, w, now),
                    [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
                );
            }
        }
    }
}

fn stream_level0(
    r: &[u8],
    d: &[u8],
    w: usize,
    h: usize,
    stride: usize,
    map: &mut [f64],
    next: &mut Planes,
) {
    let pitch = super::kernel::padded_pitch(w);
    let np = next.pitch;
    let now = next.width;
    // `planes[c]` is channel c's interior rows, banded the same way as the
    // map rows: band b emits half rows [m0/2, m1/2), so each band's slice
    // starts (STREAM_BAND/2) rows into the interior.
    let band_rows = STREAM_BAND / 2;
    let ph = np * band_rows;
    let noh = next.height;
    let [d0, d1, d2, d3, d4, d5] = &mut next.data;
    let run = |b: usize,
               args: (
        &mut [f64],
        &mut [f64],
        &mut [f64],
        &mut [f64],
        &mut [f64],
        &mut [f64],
        &mut [f64],
    )| {
        let (mb, p0, p1, p2, p3, p4, p5) = args;
        let m0 = b * STREAM_BAND;
        let m1 = (m0 + STREAM_BAND).min(h);
        stream_band(
            r,
            d,
            w,
            h,
            stride,
            m0,
            m1,
            pitch,
            np,
            now,
            mb,
            [p0, p1, p2, p3, p4, p5],
        );
    };
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        map.par_chunks_mut(STREAM_BAND * w)
            .zip(d0[np..np + noh * np].par_chunks_mut(ph))
            .zip(d1[np..np + noh * np].par_chunks_mut(ph))
            .zip(d2[np..np + noh * np].par_chunks_mut(ph))
            .zip(d3[np..np + noh * np].par_chunks_mut(ph))
            .zip(d4[np..np + noh * np].par_chunks_mut(ph))
            .zip(d5[np..np + noh * np].par_chunks_mut(ph))
            .enumerate()
            .for_each(|(b, args)| {
                let ((((((mb, p0), p1), p2), p3), p4), p5) = args;
                run(b, (mb, p0, p1, p2, p3, p4, p5));
            });
        return;
    }
    for (b, args) in map
        .chunks_mut(STREAM_BAND * w)
        .zip(d0[np..np + noh * np].chunks_mut(ph))
        .zip(d1[np..np + noh * np].chunks_mut(ph))
        .zip(d2[np..np + noh * np].chunks_mut(ph))
        .zip(d3[np..np + noh * np].chunks_mut(ph))
        .zip(d4[np..np + noh * np].chunks_mut(ph))
        .zip(d5[np..np + noh * np].chunks_mut(ph))
        .enumerate()
    {
        let ((((((mb, p0), p1), p2), p3), p4), p5) = args;
        run(b, (mb, p0, p1, p2, p3, p4, p5));
    }
}

/// Staged (unfused) level-0: materialize all six planes. Kept for the
/// oracle examples that compare the staged and streamed paths, and for
/// the bitwise staged-vs-streamed test.
#[allow(dead_code)]
pub(crate) fn prepare(r: &[u8], d: &[u8], w: usize, h: usize, stride: usize) -> Planes {
    let mut p = Planes::empty(w, h);
    let pitch = p.pitch;
    let (head, rest) = p.data.split_at_mut(2);
    let (mid, tail) = rest.split_at_mut(2);
    for side in 0..2 {
        let rgb = if side == 0 { r } else { d };
        for (y, ((or, oi), oq)) in head[side][pitch..pitch + h * pitch]
            .chunks_mut(pitch)
            .zip(mid[side][pitch..pitch + h * pitch].chunks_mut(pitch))
            .zip(tail[side][pitch..pitch + h * pitch].chunks_mut(pitch))
            .enumerate()
        {
            let src = &rgb[y * stride..y * stride + 3 * w];
            archmage::attuned!(
                fill_row(src, w, or, oi, oq),
                [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
            );
        }
    }
    p
}

/// Similarity over the reference/distorted Y channel as three row triples:
/// `rows[plane][dy]` is the plane row at `y + dy` (`dy` in 0..3). Slices are
/// `pitch`-long including pad columns, so the Prewitt border reads stay the
/// same loads the `Planes` form made — bit-identical values.
#[attune(_*, +v4x(cfg(avx512)), define(f64x8))]
fn similarity_row(
    token: Token,
    rows: &[[&[f64]; 3]; 2],
    width: usize,
    stabilizer: f64,
    out: &mut [f64],
) {
    let t = f64x8::splat_t(token, 1.0 / 3.0);
    let two = f64x8::splat_t(token, 2.0);
    let alpha = f64x8::splat_t(token, 0.5);
    let c = f64x8::splat_t(token, stabilizer);
    for x in (0..width).step_by(8) {
        macro_rules! sample {
            ($plane:expr,$dy:expr,$dx:expr) => {
                f64x8::from_array_t(
                    token,
                    core::array::from_fn(|k| rows[$plane][$dy][x + k + $dx]),
                )
            };
        }
        let r = chroma_prewitt!(sample, 0, t);
        let d = chroma_prewitt!(sample, 1, t);
        let mask = (alpha * r) * d;
        let q = (((two * r) * d - mask) + c) / (((r * r + d * d) - mask) + c);
        let n = (width - x).min(8);
        out[x..x + n].copy_from_slice(&q.to_array()[..n]);
    }
}

#[cfg(target_arch = "x86_64")]
row_entry!(
    similarity_row_v3_t,
    X64V3Token,
    similarity_row_v3,
    rows: &[[&[f64]; 3]; 2],
    width: usize,
    stabilizer: f64,
    out: &mut [f64]
);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
row_entry!(
    similarity_row_v4x_t,
    X64V4xToken,
    similarity_row_v4x,
    rows: &[[&[f64]; 3]; 2],
    width: usize,
    stabilizer: f64,
    out: &mut [f64]
);
#[cfg(target_arch = "aarch64")]
row_entry!(
    similarity_row_neon_t,
    NeonToken,
    similarity_row_neon,
    rows: &[[&[f64]; 3]; 2],
    width: usize,
    stabilizer: f64,
    out: &mut [f64]
);
#[cfg(target_arch = "wasm32")]
row_entry!(
    similarity_row_wasm128_t,
    Wasm128Token,
    similarity_row_wasm128,
    rows: &[[&[f64]; 3]; 2],
    width: usize,
    stabilizer: f64,
    out: &mut [f64]
);
pub(crate) fn similarity_row_scalar_t(
    token: archmage::ScalarToken,
    rows: &[[&[f64]; 3]; 2],
    width: usize,
    stabilizer: f64,
    out: &mut [f64],
) {
    similarity_row_scalar(token, rows, width, stabilizer, out);
}

macro_rules! band {
    ($name:ident,$token:ident,$row:ident) => {
        #[archmage::attune]
        pub(crate) fn $name(token: $token, p: &Planes, y0: usize, c: f64, out: &mut [f64]) {
            for (y, row) in out.chunks_mut(p.width).enumerate() {
                let y = y0 + y;
                let rows: [[&[f64]; 3]; 2] = core::array::from_fn(|plane| {
                    core::array::from_fn(|dy| {
                        &p.data[plane][(y + dy) * p.pitch..(y + dy + 1) * p.pitch]
                    })
                });
                $row(token, &rows, p.width, c, row);
            }
        }
    };
}
#[cfg(target_arch = "x86_64")]
band!(ms_band_v3_t, X64V3Token, similarity_row_v3);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
band!(ms_band_v4x_t, X64V4xToken, similarity_row_v4x);
#[cfg(target_arch = "aarch64")]
band!(ms_band_neon_t, NeonToken, similarity_row_neon);
#[cfg(target_arch = "wasm32")]
band!(ms_band_wasm128_t, Wasm128Token, similarity_row_wasm128);
pub(crate) fn ms_band_scalar_t(
    token: archmage::ScalarToken,
    p: &Planes,
    y0: usize,
    c: f64,
    out: &mut [f64],
) {
    for (y, row) in out.chunks_mut(p.width).enumerate() {
        let y = y0 + y;
        let rows: [[&[f64]; 3]; 2] = core::array::from_fn(|plane| {
            core::array::from_fn(|dy| &p.data[plane][(y + dy) * p.pitch..(y + dy + 1) * p.pitch])
        });
        similarity_row_scalar(token, &rows, p.width, c, row);
    }
}

pub(crate) fn map(p: &Planes, c: f64) -> Vec<f64> {
    let mut out = alloc::vec![0.0; p.width*p.height];
    let run = |i: usize, out: &mut [f64]| {
        archmage::attuned!(
            ms_band(p, i * super::BAND_ROWS, c, out),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(p.width * super::BAND_ROWS)
            .enumerate()
            .for_each(|(i, q)| run(i, q));
    }
    #[cfg(not(feature = "parallel"))]
    for (i, q) in out.chunks_mut(p.width * super::BAND_ROWS).enumerate() {
        run(i, q);
    }
    out
}

/// Striped fold of `f(v)` over `q` — fixed 32-band order in every
/// build so serial and parallel results are bit-identical.
fn fold_map(q: &[f64], f: impl Fn(&f64) -> f64 + Sync) -> f64 {
    const BANDS: usize = 32;
    let bs = q.len().div_ceil(BANDS);
    let body = |r: core::ops::Range<usize>| -> f64 { q[r].iter().map(&f).sum::<f64>() };
    let mut parts = [0.0f64; BANDS];
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        parts.par_iter_mut().enumerate().for_each(|(b, p)| {
            let lo = b * bs;
            let hi = (lo + bs).min(q.len());
            *p = if lo < hi { body(lo..hi) } else { 0.0 };
        });
        return parts.iter().sum();
    }
    for (b, p) in parts.iter_mut().enumerate() {
        let lo = b * bs;
        let hi = (lo + bs).min(q.len());
        *p = if lo < hi { body(lo..hi) } else { 0.0 };
    }
    parts.iter().sum()
}

pub(crate) fn variance(q: &[f64]) -> f64 {
    let mean = fold_map(q, |&v| v) / q.len() as f64;
    fold_map(q, |&v| {
        let d = v - mean;
        d * d
    }) / q.len() as f64
}

pub(crate) fn finish(variances: [f64; 4], p: &Planes) -> (f64, f64) {
    let weights = [0.096, 0.596, 0.289, 0.019];
    let ms = super::sqrt_f64(variances.into_iter().zip(weights).map(|(v, w)| v * w).sum());
    let mut error = 0.0;
    for y in 0..p.height {
        for x in 0..p.width {
            let j = (y + 1) * p.pitch + x + 1;
            let di = p.data[2][j] - p.data[3][j];
            let dq = p.data[4][j] - p.data[5][j];
            error += di * di + dq * dq;
        }
    }
    let chroma = super::sqrt_f64(error / (p.width * p.height) as f64);
    let gamma = 2.0 / (1.0 + 0.32 * libm::exp(-15.0 * ms)) - 1.0;
    (ms, gamma * ms + (1.0 - gamma) * (0.01 * chroma))
}

pub(crate) fn run(
    r: &[u8],
    d: &[u8],
    w: usize,
    h: usize,
    stride: usize,
) -> super::Result<(f64, f64)> {
    super::check_rgb8(r, w, h, stride)?;
    super::check_rgb8(d, w, h, stride)?;
    if w == 0 || h == 0 {
        return Err(super::Error::TooSmall {
            width: w,
            height: h,
        });
    }
    // Level 0 streams RGB -> similarity map -> level-1 planes without
    // materializing the six full-size planes (~400 MB less traffic).
    let mut map0 = alloc::vec![0.0; w * h];
    let mut p = Planes::empty(w.div_ceil(2), h.div_ceil(2));
    stream_level0(r, d, w, h, stride, &mut map0, &mut p);
    let mut variances = [0.0; 4];
    variances[0] = variance(&map0);
    for (scale, v) in variances.iter_mut().enumerate().skip(1) {
        *v = variance(&map(&p, 170.0));
        if scale != 3 {
            p = p.half();
        }
    }
    Ok(finish(variances, &p))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The staged pipeline the streamed level 0 replaced: materialize all
    /// six planes, then `map`/`half` per scale. Kept in-tree via `prepare`
    /// for the oracle examples — the streamed `run` must reproduce it
    /// bit-for-bit.
    fn staged_run(r: &[u8], d: &[u8], w: usize, h: usize, stride: usize) -> (f64, f64) {
        let mut p = prepare(r, d, w, h, stride);
        let mut variances = [0.0; 4];
        for (scale, v) in variances.iter_mut().enumerate() {
            *v = variance(&map(&p, 170.0));
            if scale != 3 {
                p = p.half();
            }
        }
        finish(variances, &p)
    }

    #[test]
    fn streamed_level0_matches_staged_bitwise() {
        let mut s = 88172645463325252u64;
        for (w, h) in [
            (1, 1),
            (2, 2),
            (3, 5),
            (7, 4),
            (16, 9),
            (67, 31),
            (64, 48),
            (129, 65),
            (200, 150),
            (257, 193),
        ] {
            let mut r = alloc::vec![0u8; w * h * 3];
            let mut d = alloc::vec![0u8; w * h * 3];
            for i in 0..r.len() {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                r[i] = (s >> 32) as u8;
                d[i] = r[i].wrapping_add((s >> 11) as u8);
            }
            let (a_ms, a_c) = run(&r, &d, w, h, w * 3).unwrap();
            let (b_ms, b_c) = staged_run(&r, &d, w, h, w * 3);
            assert_eq!(
                (a_ms.to_bits(), a_c.to_bits()),
                (b_ms.to_bits(), b_c.to_bits()),
                "{w}x{h}: streamed ({a_ms:e},{a_c:e}) != staged ({b_ms:e},{b_c:e})"
            );
        }
    }

    #[test]
    fn identity_and_colour_shift() {
        for (w, h) in [(1, 1), (3, 5), (67, 31)] {
            let r = alloc::vec![128;w*h*3];
            assert_eq!(run(&r, &r, w, h, w * 3).unwrap(), (0.0, 0.0));
            let mut d = r.clone();
            for rgb in d.chunks_mut(3) {
                rgb[0] = 200;
                rgb[2] = 20;
            }
            let (_, colour) = run(&r, &d, w, h, w * 3).unwrap();
            assert!(colour > 0.0);
        }
    }
}
