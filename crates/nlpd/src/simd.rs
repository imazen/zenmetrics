//! SIMD reduction for the scale-wise mean squared distance.

#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn squared_difference_inner(token: Token, a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let (ac, at) = f32x8::partition_slice_t(token, a);
    let (bc, bt) = f32x8::partition_slice_t(token, b);
    let mut sum = f32x8::zero_t(token);
    for (aa, bb) in ac.iter().zip(bc) {
        let d = f32x8::load_t(token, aa) - f32x8::load_t(token, bb);
        sum = d.mul_add(d, sum);
    }
    let mut result = sum.to_array().into_iter().map(f64::from).sum::<f64>();
    for (&a, &b) in at.iter().zip(bt) {
        let d = a - b;
        result += f64::from(d * d);
    }
    result
}

pub(crate) fn squared_difference(a: &[f32], b: &[f32]) -> f64 {
    archmage::attuned!(
        squared_difference_inner(a, b),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn filter5_vertical_inner(
    token: Token,
    src: &[f32],
    width: usize,
    height: usize,
    dst: &mut [f32],
    _out_height: usize,
    stride: usize,
    pad: usize,
) {
    let weights = [0.05, 0.25, 0.4, 0.25, 0.05].map(|v| f32x8::splat_t(token, v));
    super::for_each_row(dst, width, |oy, dst_row| {
        let rows = core::array::from_fn::<_, 5, _>(|ky| {
            let sy = super::reflect((oy * stride + ky) as isize - pad as isize, height);
            &src[sy * width..(sy + 1) * width]
        });
        let (dst_chunks, dst_tail) = f32x8::partition_slice_mut_t(token, dst_row);
        let row_parts = rows.map(|r| f32x8::partition_slice_t(token, r));
        for (i, chunk) in dst_chunks.iter_mut().enumerate() {
            let mut sum = f32x8::zero_t(token);
            for k in 0..5 {
                sum = f32x8::load_t(token, &row_parts[k].0[i]).mul_add(weights[k], sum);
            }
            sum.store(chunk);
        }
        for (x, d) in dst_tail.iter_mut().enumerate() {
            let mut sum = 0.0_f32;
            for k in 0..5 {
                sum += row_parts[k].1[x] * [0.05, 0.25, 0.4, 0.25, 0.05][k];
            }
            *d = sum;
        }
    });
}

pub(crate) fn filter5_vertical(
    src: &[f32],
    width: usize,
    height: usize,
    dst: &mut [f32],
    out_height: usize,
    stride: usize,
    pad: usize,
) {
    // Callers pass level-0-sized scratch slabs for `dst` — iterate only
    // this level's out_height×width subregion (row chunking keys off it).
    let dst = &mut dst[..out_height * width];
    archmage::attuned!(
        filter5_vertical_inner(src, width, height, dst, out_height, stride, pad),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

/// Fused single-pass 5-tap horizontal conv: `out[i] = Σ_k src[i+k]·w_k`.
/// Bit-identical to the 5-pass accumulate form — each product is rounded
/// separately, then added in k order (no FMA). `ZINIT` replicates the
/// decimate path's `0.0`-initialised accumulator (the `0 + v0·w0` add
/// flushes a negative-zero first product to +0, matching the scalar `+=`
/// onto a cleared slot). Caller guarantees `src.len() >= out.len() + 4`.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn h5_fused_inner<const ZINIT: bool>(token: Token, src: &[f32], out: &mut [f32]) {
    debug_assert!(src.len() >= out.len() + 4);
    let w = super::LOWPASS.map(|v| f32x8::splat_t(token, v));
    let span = out.len();
    let full = span & !7;
    for i in (0..full).step_by(8) {
        let mut acc = f32x8::load_t(token, src[i..i + 8].try_into().unwrap()) * w[0];
        if ZINIT {
            acc += f32x8::zero_t(token);
        }
        for k in 1..5 {
            acc += f32x8::load_t(token, src[i + k..i + k + 8].try_into().unwrap()) * w[k];
        }
        acc.store((&mut out[i..i + 8]).try_into().unwrap());
    }
    for (i, o) in out.iter_mut().enumerate().skip(full) {
        let mut acc = src[i] * super::LOWPASS[0];
        if ZINIT {
            acc += 0.0;
        }
        for k in 1..5 {
            acc += src[i + k] * super::LOWPASS[k];
        }
        *o = acc;
    }
}

/// `h5_fused_inner` dispatch: `out[i] = Σ_k src[i+k]·LOWPASS[k]`,
/// `src.len() >= out.len() + 4`.
pub(crate) fn h5_fused<const ZINIT: bool>(src: &[f32], out: &mut [f32]) {
    archmage::attuned!(
        h5_fused_inner::<ZINIT>(src, out),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

/// Split a row into even/odd-indexed halves: `e[j] = row[2j]`,
/// `o[j] = row[2j+1]`. Written as plain scalar copies — LLVM lowers the
/// stride-2 access patterns into shuffle-based deinterleaving.
#[archmage::attune(_*_t, +v4x(cfg(avx512)))]
fn deinterleave2_inner(_token: Token, row: &[f32], e: &mut [f32], o: &mut [f32]) {
    let no = o.len();
    for j in 0..no {
        e[j] = row[2 * j];
        o[j] = row[2 * j + 1];
    }
    if e.len() > no {
        e[no] = row[2 * no];
    }
}

/// `deinterleave2_inner` dispatch.
pub(crate) fn deinterleave2(row: &[f32], e: &mut [f32], o: &mut [f32]) {
    archmage::attuned!(
        deinterleave2_inner(row, e, o),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

/// Stride-2 decimated horizontal 5-tap on a deinterleaved row, for pad
/// ∈ {1, 2}: `out[ox] = Σ_k LOWPASS[k]·row[2ox-pad+k]` for ox ∈
/// [lo, hi). With `e[j] = row[2j]` / `o[j] = row[2j+1]` each tap lands
/// on a fixed half at a fixed offset — for pad 2 that's
/// `w0·e[ox-1] + w1·o[ox-1] + w2·e[ox] + w3·o[ox] + w4·e[ox+1]` and
/// pad 1 swaps the halves (`w0·o[ox-1] + w1·e[ox] + … + w4·o[ox+1]`).
/// Same k-ordered products/adds `tap5` produces (the `+ 0.0` replicates
/// the cleared-slot `+=` exactly as `h5_fused`'s ZINIT does), but only
/// at kept positions from contiguous half-length streams. Caller
/// guarantees lo ≥ 1 and `e.len() > hi`, `o.len() > hi`.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn h5_eo_inner<const ODDPAD: bool>(
    token: Token,
    e: &[f32],
    o: &[f32],
    lo: usize,
    hi: usize,
    out: &mut [f32],
) {
    // e[ox+1] is always touched → e.len() > hi; o[ox] (pad 2) needs
    // o.len() >= hi, o[ox+1] (pad 1) needs o.len() > hi.
    debug_assert!(lo >= 1 && e.len() > hi && o.len() >= hi - ODDPAD as usize);
    let w = super::LOWPASS.map(|v| f32x8::splat_t(token, v));
    let zero = f32x8::zero_t(token);
    // Per-tap (half, signed offset from ox), parity-selected:
    //   pad 2 → row[2ox-2+k] = e[ox-1], o[ox-1], e[ox], o[ox], e[ox+1]
    //   pad 1 → row[2ox-1+k] = o[ox-1], e[ox],   o[ox], e[ox+1], o[ox+1]
    let (h0, h1, h2, h3, h4): (&[f32], &[f32], &[f32], &[f32], &[f32]) = if ODDPAD {
        (o, e, o, e, o)
    } else {
        (e, o, e, o, e)
    };
    let (s0, s1, s2, s3, s4): (usize, usize, usize, usize, usize) = if ODDPAD {
        (usize::MAX, 0, 0, 1, 1)
    } else {
        (usize::MAX, usize::MAX, 0, 0, 1)
    };
    let mut ox = lo;
    while ox + 8 <= hi {
        let lv = |h: &[f32], s: usize| {
            f32x8::load_t(
                token,
                h[ox.wrapping_add(s)..ox.wrapping_add(s) + 8]
                    .try_into()
                    .unwrap(),
            )
        };
        let mut acc = lv(h0, s0) * w[0] + zero;
        acc += lv(h1, s1) * w[1];
        acc += lv(h2, s2) * w[2];
        acc += lv(h3, s3) * w[3];
        acc += lv(h4, s4) * w[4];
        acc.store((&mut out[ox..ox + 8]).try_into().unwrap());
        ox += 8;
    }
    let sv = |h: &[f32], i: usize| h[i];
    while ox < hi {
        let mut acc = sv(h0, ox.wrapping_add(s0)) * super::LOWPASS[0] + 0.0;
        acc += sv(h1, ox.wrapping_add(s1)) * super::LOWPASS[1];
        acc += sv(h2, ox.wrapping_add(s2)) * super::LOWPASS[2];
        acc += sv(h3, ox.wrapping_add(s3)) * super::LOWPASS[3];
        acc += sv(h4, ox.wrapping_add(s4)) * super::LOWPASS[4];
        out[ox] = acc;
        ox += 1;
    }
}

/// `h5_eo_inner` dispatch. `pad` ∈ {1, 2}.
pub(crate) fn h5_eo(e: &[f32], o: &[f32], pad: usize, lo: usize, hi: usize, out: &mut [f32]) {
    if pad & 1 == 1 {
        archmage::attuned!(
            h5_eo_inner::<true>(e, o, lo, hi, out),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    } else {
        archmage::attuned!(
            h5_eo_inner::<false>(e, o, lo, hi, out),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    }
}

/// Horizontal lerp of one source row: `out[x] = row[x0]·(1−tx) +
/// row[x1]·tx`. Two lane-gathers per output; per-lane arithmetic
/// identical to the scalar formula → bit-exact.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn hlerp_row_inner(
    token: Token,
    row: &[f32],
    x0v: &[usize],
    x1v: &[usize],
    txv: &[f32],
    out: &mut [f32],
) {
    let one = f32x8::splat_t(token, 1.0);
    let span = out.len();
    let full = span & !7;
    for x in (0..full).step_by(8) {
        let g = |ix: &[usize]| f32x8::from_array_t(token, core::array::from_fn(|j| row[ix[j]]));
        let tx = f32x8::load_t(token, txv[x..x + 8].try_into().unwrap());
        (g(&x0v[x..x + 8]) * (one - tx) + g(&x1v[x..x + 8]) * tx)
            .store((&mut out[x..x + 8]).try_into().unwrap());
    }
    for x in full..span {
        let tx = txv[x];
        out[x] = row[x0v[x]] * (1.0 - tx) + row[x1v[x]] * tx;
    }
}

pub(crate) fn hlerp_row(row: &[f32], x0v: &[usize], x1v: &[usize], txv: &[f32], out: &mut [f32]) {
    archmage::attuned!(
        hlerp_row_inner(row, x0v, x1v, txv, out),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

/// `out[x] = a[x]·(1−ty) + b[x]·ty` — the vertical blend of two
/// pre-lerped rows. Same per-lane arithmetic as the scalar loop.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn blend_rows_inner(token: Token, a: &[f32], b: &[f32], ty: f32, out: &mut [f32]) {
    let tyv = f32x8::splat_t(token, ty);
    let omty = f32x8::splat_t(token, 1.0 - ty);
    let span = out.len();
    let full = span & !7;
    for x in (0..full).step_by(8) {
        let av = f32x8::load_t(token, a[x..x + 8].try_into().unwrap());
        let bv = f32x8::load_t(token, b[x..x + 8].try_into().unwrap());
        (av * omty + bv * tyv).store((&mut out[x..x + 8]).try_into().unwrap());
    }
    for x in full..span {
        out[x] = a[x] * (1.0 - ty) + b[x] * ty;
    }
}

pub(crate) fn blend_rows(a: &[f32], b: &[f32], ty: f32, out: &mut [f32]) {
    archmage::attuned!(
        blend_rows_inner(a, b, ty, out),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

/// Interleaved→planar u8 channel extract: `dst[i] = src[3i+c] / 255`.
/// i32-widened lanes keep `u8 → /255` arithmetic identical to scalar.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8, i32x8))]
fn deinterleave_u8_ch_inner(token: Token, src: &[u8], c: usize, dst: &mut [f32]) {
    let inv = f32x8::splat_t(token, 255.0);
    let n = dst.len();
    let full = n & !7;
    for i in (0..full).step_by(8) {
        let arr = core::array::from_fn(|j| src[3 * (i + j) + c] as i32);
        let v = i32x8::from_array_t(token, arr).to_f32() / inv;
        v.store((&mut dst[i..i + 8]).try_into().unwrap());
    }
    for (i, o) in dst.iter_mut().enumerate().skip(full) {
        *o = src[3 * i + c] as f32 / 255.0;
    }
}

pub(crate) fn deinterleave_u8_ch(src: &[u8], c: usize, dst: &mut [f32]) {
    archmage::attuned!(
        deinterleave_u8_ch_inner(src, c, dst),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

/// BT.709 luma ingress for the AIC-4 harness (`iqa::y709_u8_grid`):
/// `round(clamp(Y·255,0,255))/255` on `0.2126R+0.7152G+0.0722B` per pixel.
/// The weighted sum is f32-exact per lane (same product/add order as the
/// scalar form); inputs are u8 so the clamp domain is [0, ~255.02] — no
/// NaN path exists, so `max/min` is a faithful `clamp` here.
///
/// `roundf` is half-away-from-zero, but the hardware/magetypes `round` is
/// half-even — and `floor(y+0.5)` differs from `roundf` when the `+0.5`
/// add itself rounds up to k+1 (y just under k+0.5). The correction
/// `r − y > 0.5` detects exactly that case, making this bit-identical to
/// `roundf` for every y ≥ 0 below 2²².
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8, i32x8))]
fn y709_grid_inner(token: Token, src: &[u8], out: &mut [f32]) {
    let wr = f32x8::splat_t(token, 0.2126);
    let wg = f32x8::splat_t(token, 0.7152);
    let wb = f32x8::splat_t(token, 0.0722);
    let zero = f32x8::zero_t(token);
    let c255 = f32x8::splat_t(token, 255.0);
    let half = f32x8::splat_t(token, 0.5);
    let one = f32x8::splat_t(token, 1.0);
    let n = out.len();
    let full = n & !7;
    for i in (0..full).step_by(8) {
        let r = i32x8::from_array_t(token, core::array::from_fn(|j| src[3 * (i + j)] as i32));
        let g = i32x8::from_array_t(token, core::array::from_fn(|j| src[3 * (i + j) + 1] as i32));
        let b = i32x8::from_array_t(token, core::array::from_fn(|j| src[3 * (i + j) + 2] as i32));
        let y = (r.to_f32() * wr + g.to_f32() * wg + b.to_f32() * wb)
            .max(zero)
            .min(c255);
        let r0 = (y + half).floor();
        let r0 = f32x8::blend((r0 - y).simd_gt(half), r0 - one, r0);
        (r0 / c255).store((&mut out[i..i + 8]).try_into().unwrap());
    }
    for (i, o) in out.iter_mut().enumerate().skip(full) {
        let v = src[3 * i] as f32 * 0.2126
            + src[3 * i + 1] as f32 * 0.7152
            + src[3 * i + 2] as f32 * 0.0722;
        *o = v.clamp(0.0, 255.0).round() / 255.0;
    }
}

pub(crate) fn y709_grid(src: &[u8], out: &mut [f32]) {
    archmage::attuned!(
        y709_grid_inner(src, out),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

/// Interleaved→planar f32 channel extract: `dst[i] = src[3i+c]`.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn deinterleave_f32_ch_inner(token: Token, src: &[f32], c: usize, dst: &mut [f32]) {
    let n = dst.len();
    let full = n & !7;
    for i in (0..full).step_by(8) {
        let v = f32x8::from_array_t(token, core::array::from_fn(|j| src[3 * (i + j) + c]));
        v.store((&mut dst[i..i + 8]).try_into().unwrap());
    }
    for (i, o) in dst.iter_mut().enumerate().skip(full) {
        *o = src[3 * i + c];
    }
}

pub(crate) fn deinterleave_f32_ch(src: &[f32], c: usize, dst: &mut [f32]) {
    archmage::attuned!(
        deinterleave_f32_ch_inner(src, c, dst),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

// Each output band retains a 5-row ring of H-blurred `up` rows (slot
// row%5) feeding a 3-row ring of Laplacian rows (slot row%3) — the H
// conv runs inside this pass so the `horizontal` plane is never
// materialized. The vertical blur uses the same five FMAs and scalar
// tail as filter5_vertical; normalization uses ordered non-fused
// additions and an exact divide. Bit-identical to filtering into a
// scratch plane first: the ring rows are produced by the same ops.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn normalize_filtered_inner(
    token: Token,
    original: &[f32],
    low: &[f32],
    lw: usize,
    lh: usize,
    uw: usize,
    uh: usize,
    w: usize,
    h: usize,
    level: usize,
    dst: &mut [f32],
) {
    let weights = super::LOWPASS.map(|v| f32x8::splat_t(token, v));
    let [top, left, right, bottom] = super::DN[level];
    let sigma = super::SIGMAS[level];
    let tv = f32x8::splat_t(token, top);
    let lv = f32x8::splat_t(token, left);
    let rv = f32x8::splat_t(token, right);
    let bv = f32x8::splat_t(token, bottom);
    let sv = f32x8::splat_t(token, sigma);
    // The bilinear x-mapping for `(lw → uw)` — identical tables to the
    // materialized `upsample_bilinear` pass this ring chain replaces.
    let (x0v, x1v, txv) = super::upsample_xtabs(lw, uw);
    // Smooth → lap row/column maps (nearest crop `(y·uh/h, x·uw/w)`);
    // degenerate to identity on the RGB path where `uw==w`, `uh==h`.
    let sy_of = |r: usize| (r * uh / h).min(uh - 1);
    let sxv: Vec<usize> = (0..w).map(|x| (x * uw / w).min(uw - 1)).collect();
    struct Rings {
        lerp: super::LerpRing,
        up_row: Vec<f32>,
        hrows: [Vec<f32>; 5],
        hidx: [usize; 5],
        smooth: Vec<f32>,
        lap: [Vec<f32>; 3],
        lidx: [usize; 3],
    }
    super::for_each_row_init_min(
        dst,
        w,
        16,
        || Rings {
            lerp: super::LerpRing::new(uw),
            up_row: vec![0.0; uw],
            hrows: core::array::from_fn(|_| vec![0.0; uw]),
            hidx: [usize::MAX; 5],
            smooth: vec![0.0; uw],
            lap: core::array::from_fn(|_| vec![0.0; w]),
            lidx: [usize::MAX; 3],
        },
        |y, out, rings| {
            // Materialise the three lap rows this output row needs.
            for r in [
                super::reflect(y as isize - 1, h),
                y,
                super::reflect(y as isize + 1, h),
            ] {
                let slot = r % 3;
                if rings.lidx[slot] == r {
                    continue;
                }
                // Lap row r needs the smoothed row `sy(r)` — 5 H rows
                // reflect(sy-2..sy+2), each filtered straight off a
                // freshly produced upsampled row. Neither the `up` plane
                // nor the H plane nor `smooth` is ever materialized.
                let sy = sy_of(r);
                for k in 0..5usize {
                    let hr = super::reflect(sy as isize + k as isize - 2, uh);
                    let hslot = hr % 5;
                    if rings.hidx[hslot] != hr {
                        rings.lerp.produce(
                            low,
                            lw,
                            lh,
                            uh,
                            hr,
                            &x0v,
                            &x1v,
                            &txv,
                            &mut rings.up_row,
                        );
                        super::filter5_row(
                            &rings.up_row,
                            uw,
                            &mut rings.hrows[hslot],
                            2,
                            uw.saturating_sub(2).max(2),
                        );
                        rings.hidx[hslot] = hr;
                    }
                }
                let rows: [&[f32]; 5] = core::array::from_fn(|k| {
                    &rings.hrows[super::reflect(sy as isize + k as isize - 2, uh) % 5][..]
                });
                let out_lap = &mut rings.lap[slot];
                if (uw, uh) == (w, h) {
                    // Identity crop (RGB path, and even-dim IQA levels):
                    // fold the V5 directly into `original − Σ` — the
                    // smooth row is never materialized.
                    let (chunks, tail) = f32x8::partition_slice_mut_t(token, out_lap);
                    let row_parts = rows.map(|r| f32x8::partition_slice_t(token, r));
                    let original_parts =
                        f32x8::partition_slice_t(token, &original[r * w..(r + 1) * w]);
                    for (i, chunk) in chunks.iter_mut().enumerate() {
                        let mut sum = f32x8::zero_t(token);
                        for k in 0..5 {
                            let values = &row_parts[k].0[i];
                            sum = f32x8::load_t(token, values).mul_add(weights[k], sum);
                        }
                        let values = &original_parts.0[i];
                        (f32x8::load_t(token, values) - sum).store(chunk);
                    }
                    let base = w / 8 * 8;
                    for (i, o) in tail.iter_mut().enumerate() {
                        let x = base + i;
                        let mut sum = 0.0;
                        for k in 0..5 {
                            sum += rows[k][x] * super::LOWPASS[k];
                        }
                        *o = original[r * w + x] - sum;
                    }
                } else {
                    // Overshoot dims (odd-dim IQA levels): materialize the
                    // smoothed row, then apply the nearest-crop gather —
                    // `lap[r][x] = original[r][x] − smooth[sy][sx(x)]`,
                    // the scalar `x·uw/w` formula per lane.
                    {
                        let (chunks, tail) = f32x8::partition_slice_mut_t(token, &mut rings.smooth);
                        let row_parts = rows.map(|r| f32x8::partition_slice_t(token, r));
                        for (i, chunk) in chunks.iter_mut().enumerate() {
                            let mut sum = f32x8::zero_t(token);
                            for k in 0..5 {
                                let values = &row_parts[k].0[i];
                                sum = f32x8::load_t(token, values).mul_add(weights[k], sum);
                            }
                            sum.store(chunk);
                        }
                        let base = uw / 8 * 8;
                        for (i, o) in tail.iter_mut().enumerate() {
                            let x = base + i;
                            let mut sum = 0.0;
                            for k in 0..5 {
                                sum += rows[k][x] * super::LOWPASS[k];
                            }
                            *o = sum;
                        }
                    }
                    {
                        let orow = &original[r * w..(r + 1) * w];
                        let mut x = 0usize;
                        while x + 8 <= w {
                            let o = f32x8::load_t(token, orow[x..x + 8].try_into().unwrap());
                            let s = f32x8::from_array_t(
                                token,
                                core::array::from_fn(|j| rings.smooth[sxv[x + j]]),
                            );
                            (o - s).store((&mut out_lap[x..x + 8]).try_into().unwrap());
                            x += 8;
                        }
                        while x < w {
                            out_lap[x] = orow[x] - rings.smooth[sxv[x]];
                            x += 1;
                        }
                    }
                }
                rings.lidx[slot] = r;
            }
            let a = &rings.lap[super::reflect(y as isize - 1, h) % 3];
            let mid = &rings.lap[y % 3];
            let b = &rings.lap[super::reflect(y as isize + 1, h) % 3];
            let scalar = |x: usize| {
                let xm = super::reflect(x as isize - 1, w);
                let xp = super::reflect(x as isize + 1, w);
                let denom = sigma
                    + top * a[x].abs()
                    + left * mid[xm].abs()
                    + right * mid[xp].abs()
                    + bottom * b[x].abs();
                mid[x] / denom
            };
            out[0] = scalar(0);
            out[w - 1] = scalar(w - 1);
            let (chunks, tail) = f32x8::partition_slice_mut_t(token, &mut out[1..w - 1]);
            for (i, chunk) in chunks.iter_mut().enumerate() {
                let x = 1 + i * 8;
                let load =
                    |r: &[f32], x: usize| f32x8::load_t(token, r[x..x + 8].try_into().unwrap());
                let denom = sv
                    + tv * load(a, x).abs()
                    + lv * load(mid, x - 1).abs()
                    + rv * load(mid, x + 1).abs()
                    + bv * load(b, x).abs();
                (load(mid, x) / denom).store(chunk);
            }
            let base = 1 + (w - 2) / 8 * 8;
            for (i, out) in tail.iter_mut().enumerate() {
                *out = scalar(base + i);
            }
        },
    );
}

/// Fused pyramid level: `low` (lw×lh) bilinear-upsampled to `uw`×`uh`,
/// 5×5-smoothed, nearest-cropped/subtracted against `original` (w×h)
/// and divisive-normalized into `dst` — all via row rings, with no
/// materialized intermediate plane. The RGB path passes `uw==w,uh==h`
/// (identity crop); the IQA harness path passes `2·ow × 2·oh` and gets
/// the `F.interpolate`-nearest crop for free.
pub(crate) fn normalize_filtered(
    original: &[f32],
    low: &[f32],
    lw: usize,
    lh: usize,
    uw: usize,
    uh: usize,
    w: usize,
    h: usize,
    level: usize,
    dst: &mut [f32],
) {
    archmage::attuned!(
        normalize_filtered_inner(original, low, lw, lh, uw, uh, w, h, level, dst),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}
