//! The **Daala / Xiph `dump_psnrhvs`** variant of PSNR-HVS — the
//! implementation libvmaf vendored at
//! `libvmaf/src/feature/third_party/xiph/psnr_hvs.c` (tag v3.2.1) and
//! the convention behind the JPEG AIC-4 `PSNR-HVS(-Y|-Cb|-Cr)` columns.
//!
//! This is a **different algorithm family** from the Ponomarenko
//! `psnrhvs`/`psnrhvsm` port in [`crate`]: instead of an orthonormal
//! float DCT-II, it uses Daala's fixed-point bin-DCT (`od_bin_fdct8`),
//! 8×8 blocks stepped by **7** (overlapping), per-quadrant
//! variance-ratio masking with the 0.3885746225901003-scaled CSF
//! masking tables, and a different error accumulation (`(err·csf)²`
//! over the masked-down DCT coefficients, DC unmasked).
//!
//! Output convention (per plane): the masked MSE normalized by
//! `pixels·255²`, reported as `-10·log10(mse)` dB. The combined score is
//! `0.8·mse_y + 0.1·mse_cb + 0.1·mse_cr` in the *linear* domain, then
//! `-10·log10` — matching libvmaf's `psnr_hvs` feature.
//!
//! Arithmetic mirrors the C: integer butterflies/shifts for the DCT,
//! f32 for block statistics/masking/error accumulation (including the
//! f32 `ret` accumulator over all coefficients — libvmaf accumulates in
//! float, and matching that accumulation order is what keeps parity),
//! f64 only for the final `-10·log10`.
//!
//! The integer DCT processes eight independent columns in SIMD lanes,
//! with bit-shuffle transposes between its column-first passes. Runtime
//! dispatch occurs once per block row. All floating-point statistics and
//! the global coefficient fold retain their original scalar order.

/// Per-plane CSF tables (`csf_y`, `csf_cb420`, `csf_cr420`) — normalized
/// inverse-quantization matrices "at the point of transparency".
#[rustfmt::skip]
const CSF_Y: [[f32; 8]; 8] = [
    [1.6193873005, 2.2901594831, 2.08509755623, 1.48366094411, 1.00227514334, 0.678296995242, 0.466224900598, 0.3265091542],
    [2.2901594831, 1.94321815382, 2.04793073064, 1.68731108984, 1.2305666963, 0.868920337363, 0.61280991668, 0.436405793551],
    [2.08509755623, 2.04793073064, 1.34329019223, 1.09205635862, 0.875748795257, 0.670882927016, 0.501731932449, 0.372504254596],
    [1.48366094411, 1.68731108984, 1.09205635862, 0.772819797575, 0.605636379554, 0.48309405692, 0.380429446972, 0.295774038565],
    [1.00227514334, 1.2305666963, 0.875748795257, 0.605636379554, 0.448996256676, 0.352889268808, 0.283006984131, 0.226951348204],
    [0.678296995242, 0.868920337363, 0.670882927016, 0.48309405692, 0.352889268808, 0.27032073436, 0.215017739696, 0.17408067321],
    [0.466224900598, 0.61280991668, 0.501731932449, 0.380429446972, 0.283006984131, 0.215017739696, 0.168869545842, 0.136153931001],
    [0.3265091542, 0.436405793551, 0.372504254596, 0.295774038565, 0.226951348204, 0.17408067321, 0.136153931001, 0.109083846276],
];

#[rustfmt::skip]
const CSF_CB420: [[f32; 8]; 8] = [
    [1.91113096927, 2.46074210438, 1.18284184739, 1.14982565193, 1.05017074788, 0.898018824055, 0.74725392039, 0.615105596242],
    [2.46074210438, 1.58529308355, 1.21363250036, 1.38190029285, 1.33100189972, 1.17428548929, 0.996404342439, 0.830890433625],
    [1.18284184739, 1.21363250036, 0.978712413627, 1.02624506078, 1.03145147362, 0.960060382087, 0.849823426169, 0.731221236837],
    [1.14982565193, 1.38190029285, 1.02624506078, 0.861317501629, 0.801821139099, 0.751437590932, 0.685398513368, 0.608694761374],
    [1.05017074788, 1.33100189972, 1.03145147362, 0.801821139099, 0.676555426187, 0.605503172737, 0.55002013668, 0.495804539034],
    [0.898018824055, 1.17428548929, 0.960060382087, 0.751437590932, 0.605503172737, 0.514674450957, 0.454353482512, 0.407050308965],
    [0.74725392039, 0.996404342439, 0.849823426169, 0.685398513368, 0.55002013668, 0.454353482512, 0.389234902883, 0.342353999733],
    [0.615105596242, 0.830890433625, 0.731221236837, 0.608694761374, 0.495804539034, 0.407050308965, 0.342353999733, 0.295530605237],
];

#[rustfmt::skip]
const CSF_CR420: [[f32; 8]; 8] = [
    [2.03871978502, 2.62502345193, 1.26180942886, 1.11019789803, 1.01397751469, 0.867069376285, 0.721500455585, 0.593906509971],
    [2.62502345193, 1.69112867013, 1.17180569821, 1.3342742857, 1.28513006198, 1.13381474809, 0.962064122248, 0.802254508198],
    [1.26180942886, 1.17180569821, 0.944981930573, 0.990876405848, 0.995903384143, 0.926972725286, 0.820534991409, 0.706020324706],
    [1.11019789803, 1.3342742857, 0.990876405848, 0.831632933426, 0.77418706195, 0.725539939514, 0.661776842059, 0.587716619023],
    [1.01397751469, 1.28513006198, 0.995903384143, 0.77418706195, 0.653238524286, 0.584635025748, 0.531064164893, 0.478717061273],
    [0.867069376285, 1.13381474809, 0.926972725286, 0.725539939514, 0.584635025748, 0.496936637883, 0.438694579826, 0.393021669543],
    [0.721500455585, 0.962064122248, 0.820534991409, 0.661776842059, 0.531064164893, 0.438694579826, 0.375820256136, 0.330555063063],
    [0.593906509971, 0.802254508198, 0.706020324706, 0.587716619023, 0.478717061273, 0.393021669543, 0.330555063063, 0.285345396658],
];

/// `OD_UNBIASED_RSHIFT32(a, b)` — the unbiased right shift used by the
/// Daala integer transforms: `((uint32)(a) >> (32-b)) + a) >> b`, with
/// the add carried out in the u32 domain exactly as C does.
#[inline(always)]
#[cfg(test)]
fn od_rshift(a: i32, b: u32) -> i32 {
    (((a as u32) >> (32 - b)).wrapping_add(a as u32) as i32) >> b
}

/// `od_bin_fdct8` — Daala's 8-point integer forward DCT (bit-exact
/// port). `x` is read with `xstride` between taps.
#[cfg(test)]
fn od_bin_fdct8(y: &mut [i32; 8], x: &[i32], xstride: usize) {
    let mut t0 = x[0];
    let mut t4 = x[xstride];
    let mut t2 = x[2 * xstride];
    let mut t6 = x[3 * xstride];
    let mut t7 = x[4 * xstride];
    let mut t3 = x[5 * xstride];
    let mut t5 = x[6 * xstride];
    let mut t1 = x[7 * xstride];
    t1 = t0 - t1;
    let t1h = od_rshift(t1, 1);
    t0 -= t1h;
    t4 += t5;
    let t4h = od_rshift(t4, 1);
    t5 -= t4h;
    t3 = t2 - t3;
    t2 -= od_rshift(t3, 1);
    t6 += t7;
    let t6h = od_rshift(t6, 1);
    t7 = t6h - t7;
    t0 += t6h;
    t6 = t0 - t6;
    t2 = t4h - t2;
    t4 = t2 - t4;
    t0 -= (t4 * 13573 + 16384) >> 15;
    t4 += (t0 * 11585 + 8192) >> 14;
    t0 -= (t4 * 13573 + 16384) >> 15;
    t6 -= (t2 * 21895 + 16384) >> 15;
    t2 += (t6 * 15137 + 8192) >> 14;
    t6 -= (t2 * 21895 + 16384) >> 15;
    t3 += (t5 * 19195 + 16384) >> 15;
    t5 += (t3 * 11585 + 8192) >> 14;
    t3 -= (t5 * 7489 + 4096) >> 13;
    t7 = od_rshift(t5, 1) - t7;
    t5 -= t7;
    t3 = t1h - t3;
    t1 -= t3;
    t7 += (t1 * 3227 + 16384) >> 15;
    t1 -= (t7 * 6393 + 16384) >> 15;
    t7 += (t1 * 3227 + 16384) >> 15;
    t5 += (t3 * 2485 + 4096) >> 13;
    t3 -= (t5 * 18205 + 16384) >> 15;
    t5 += (t3 * 2485 + 4096) >> 13;
    y[0] = t0;
    y[1] = t1;
    y[2] = t2;
    y[3] = t3;
    y[4] = t4;
    y[5] = t5;
    y[6] = t6;
    y[7] = t7;
}

/// `od_bin_fdct8x8` — column pass then column pass over the transposed
/// intermediate, in place (C: `od_bin_fdct8(z + 8*i, x + i, xstride)`
/// then `od_bin_fdct8(y + ystride*i, z + i, 8)`). The order matters:
/// the bin-DCT's integer rounding makes row-first ≠ column-first.
#[cfg(test)]
fn od_bin_fdct8x8(y: &mut [i32; 64]) {
    let mut z = [0i32; 64];
    let mut col_in = [0i32; 8];
    for i in 0..8 {
        for j in 0..8 {
            col_in[j] = y[j * 8 + i];
        }
        let mut col_out = [0i32; 8];
        od_bin_fdct8(&mut col_out, &col_in, 1);
        z[i * 8..i * 8 + 8].copy_from_slice(&col_out);
    }
    for i in 0..8 {
        let mut col_out = [0i32; 8];
        od_bin_fdct8(&mut col_out, &z[i..], 8);
        y[i * 8..i * 8 + 8].copy_from_slice(&col_out);
    }
}

/// Eight independent integer transforms, one in each lane. The shifts
/// retain Daala's signed rounding, including the unbiased half shifts.
#[inline(always)]
fn fdct8_vectors<T: magetypes::simd::backends::I32x8Backend>(
    x: [magetypes::simd::generic::i32x8<T>; 8],
) -> [magetypes::simd::generic::i32x8<T>; 8] {
    fn half<T: magetypes::simd::backends::I32x8Backend>(
        a: magetypes::simd::generic::i32x8<T>,
    ) -> magetypes::simd::generic::i32x8<T> {
        (a.shr_logical_const::<31>() + a).shr_arithmetic_const::<1>()
    }
    let [
        mut t0,
        mut t4,
        mut t2,
        mut t6,
        mut t7,
        mut t3,
        mut t5,
        mut t1,
    ] = x;
    t1 = t0 - t1;
    let t1h = half(t1);
    t0 -= t1h;
    t4 += t5;
    let t4h = half(t4);
    t5 -= t4h;
    t3 = t2 - t3;
    t2 -= half(t3);
    t6 += t7;
    let t6h = half(t6);
    t7 = t6h - t7;
    t0 += t6h;
    t6 = t0 - t6;
    t2 = t4h - t2;
    t4 = t2 - t4;
    t0 -= (t4 * 13573 + 16384).shr_arithmetic_const::<15>();
    t4 += (t0 * 11585 + 8192).shr_arithmetic_const::<14>();
    t0 -= (t4 * 13573 + 16384).shr_arithmetic_const::<15>();
    t6 -= (t2 * 21895 + 16384).shr_arithmetic_const::<15>();
    t2 += (t6 * 15137 + 8192).shr_arithmetic_const::<14>();
    t6 -= (t2 * 21895 + 16384).shr_arithmetic_const::<15>();
    t3 += (t5 * 19195 + 16384).shr_arithmetic_const::<15>();
    t5 += (t3 * 11585 + 8192).shr_arithmetic_const::<14>();
    t3 -= (t5 * 7489 + 4096).shr_arithmetic_const::<13>();
    t7 = half(t5) - t7;
    t5 -= t7;
    t3 = t1h - t3;
    t1 -= t3;
    t7 += (t1 * 3227 + 16384).shr_arithmetic_const::<15>();
    t1 -= (t7 * 6393 + 16384).shr_arithmetic_const::<15>();
    t7 += (t1 * 3227 + 16384).shr_arithmetic_const::<15>();
    t5 += (t3 * 2485 + 4096).shr_arithmetic_const::<13>();
    t3 -= (t5 * 18205 + 16384).shr_arithmetic_const::<15>();
    t5 += (t3 * 2485 + 4096).shr_arithmetic_const::<13>();
    [t0, t1, t2, t3, t4, t5, t6, t7]
}

#[inline(always)]
fn fdct8x8_vectors<T: magetypes::simd::backends::F32x8Convert>(token: T, block: &mut [i32; 64]) {
    use magetypes::simd::generic::{f32x8, i32x8};
    let rows = block.as_chunks::<8>().0;
    let columns = fdct8_vectors(core::array::from_fn(|i| i32x8::load_t(token, &rows[i])));
    // These are bit shuffles only: the float vectors never do arithmetic.
    let mut transposed = columns.map(i32x8::bitcast_to_f32);
    f32x8::transpose_8x8(&mut transposed);
    let columns = fdct8_vectors(transposed.map(f32x8::bitcast_to_i32));
    let mut transposed = columns.map(i32x8::bitcast_to_f32);
    f32x8::transpose_8x8(&mut transposed);
    for (out, row) in block.as_chunks_mut::<8>().0.iter_mut().zip(transposed) {
        row.bitcast_to_i32().store(out);
    }
}

/// `mask[i][j] = (csf[i][j] * 0.3885746225901003)²` — the product is
/// computed in f64 and stored in f32, matching the C `float mask`.
fn mask_of(csf: &[[f32; 8]; 8]) -> [[f32; 8]; 8] {
    let mut mask = [[0.0f32; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            let v = csf[i][j] as f64 * 0.3885746225901003f64;
            mask[i][j] = (v * v) as f32;
        }
    }
    mask
}

/// `ret /= pixels; ret /= samplemax²` — both divisions in f32, as C.
fn finish_score(ret: f32, pixels: u32) -> f64 {
    let mut ret = ret / pixels as f32;
    ret /= 255.0f32 * 255.0f32;
    ret as f64
}

/// `calc_psnrhvs` — masked MSE of one u8 plane, normalized by
/// `pixels·255²` (the pre-dB `score` the C returns). `step` is the block
/// stride (libvmaf calls with 7).
fn calc_psnrhvs(
    src: &[u8],
    sstride: usize,
    dst: &[u8],
    dstride: usize,
    w: usize,
    h: usize,
    step: usize,
    csf: &[[f32; 8]; 8],
) -> f64 {
    let mask = mask_of(csf);

    // The `ret` fold is f32 in raster order — the vendored-libvmaf FFI
    // parity tests observe the exact fold, so the block raster stays
    // serial (band-reordered partials shifted the score 3.7e-4 dB).
    let mut ret = 0.0f32;
    let mut pixels = 0u32;
    let mut y = 0usize;
    while y + 7 < h {
        archmage::attuned!(
            block_row(
                src,
                sstride,
                dst,
                dstride,
                w,
                y,
                step,
                &mask,
                csf,
                &mut ret,
                &mut pixels,
            ),
            [_v3_t, _neon_t, _wasm128_t, _scalar_t]
        );
        y += step;
    }
    finish_score(ret, pixels)
}

/// One `step`-strided block row of `calc_psnrhvs` — the unchanged
/// per-block math, accumulating `(ret, pixels)` for its band.
#[allow(clippy::too_many_arguments)]
#[archmage::attune(_*_t, define(i32x8, f32x8))]
fn block_row(
    token: Token,
    src: &[u8],
    sstride: usize,
    dst: &[u8],
    dstride: usize,
    w: usize,
    y: usize,
    step: usize,
    mask: &[[f32; 8]; 8],
    csf: &[[f32; 8]; 8],
    ret: &mut f32,
    pixels: &mut u32,
) {
    // `mask`/`csf` are block-invariant: hold the rows as vectors for
    // the whole call (mask[0][0] is zeroed — the DC tap is unmasked).
    let mut mask8 = [f32x8::splat_t(token, 0.0); 8];
    let mut csf8 = [f32x8::splat_t(token, 0.0); 8];
    let mut rmask8 = [f32x8::splat_t(token, 0.0); 8];
    for i in 0..8 {
        let mut m = mask[i];
        if i == 0 {
            m[0] = 0.0;
        }
        mask8[i] = f32x8::load_t(token, &m);
        csf8[i] = f32x8::load_t(token, &csf[i]);
        // `s_mask * (1/mask)` vs `s_mask / mask` differ in the last ulp —
        // same drift class as the accepted vector reassociation (the FFI
        // gate is 2e-4 dB); it trades 8 per-block divisions for 8 hoisted
        // reciprocals per call.
        rmask8[i] = f32x8::splat_t(token, 1.0) / mask8[i];
    }
    let mut x = 0usize;
    while x + 7 < w {
        let mut dct_s = [0i32; 64];
        let mut dct_d = [0i32; 64];
        // Means are exact-integer sums (pixels ≤ 255, totals ≤ 16320 <
        // 2^24), so any accumulation order gives the same f32 bits —
        // accumulate whole row vectors and reduce lanes at the end.
        // Lane layout: rows i<4 → lanes 0-3 feed means[0], lanes 4-7
        // means[2]; rows i≥4 → lanes 0-3 means[1], lanes 4-7 means[3].
        let mut sq02 = f32x8::splat_t(token, 0.0);
        let mut sq13 = f32x8::splat_t(token, 0.0);
        let mut dq02 = f32x8::splat_t(token, 0.0);
        let mut dq13 = f32x8::splat_t(token, 0.0);
        for i in 0..8 {
            let mut s8 = [0i32; 8];
            let mut d8 = [0i32; 8];
            for j in 0..8 {
                s8[j] = src[(y + i) * sstride + (j + x)] as i32;
                d8[j] = dst[(y + i) * dstride + (j + x)] as i32;
            }
            dct_s[i * 8..][..8].copy_from_slice(&s8);
            dct_d[i * 8..][..8].copy_from_slice(&d8);
            if i < 4 {
                sq02 = sq02 + i32x8::load_t(token, &s8).to_f32();
                dq02 = dq02 + i32x8::load_t(token, &d8).to_f32();
            } else {
                sq13 = sq13 + i32x8::load_t(token, &s8).to_f32();
                dq13 = dq13 + i32x8::load_t(token, &d8).to_f32();
            }
        }
        let sq02 = sq02.to_array();
        let sq13 = sq13.to_array();
        let dq02 = dq02.to_array();
        let dq13 = dq13.to_array();
        let mut s_means = [0.0f32; 4];
        let mut d_means = [0.0f32; 4];
        s_means[0] = (sq02[0] + sq02[1]) + (sq02[2] + sq02[3]);
        s_means[1] = (sq13[0] + sq13[1]) + (sq13[2] + sq13[3]);
        s_means[2] = (sq02[4] + sq02[5]) + (sq02[6] + sq02[7]);
        s_means[3] = (sq13[4] + sq13[5]) + (sq13[6] + sq13[7]);
        d_means[0] = (dq02[0] + dq02[1]) + (dq02[2] + dq02[3]);
        d_means[1] = (dq13[0] + dq13[1]) + (dq13[2] + dq13[3]);
        d_means[2] = (dq02[4] + dq02[5]) + (dq02[6] + dq02[7]);
        d_means[3] = (dq13[4] + dq13[5]) + (dq13[6] + dq13[7]);
        let s_gmean = ((s_means[0] + s_means[1]) + (s_means[2] + s_means[3])) / 64.0;
        let d_gmean = ((d_means[0] + d_means[1]) + (d_means[2] + d_means[3])) / 64.0;
        for m in s_means.iter_mut() {
            *m /= 16.0;
        }
        for m in d_means.iter_mut() {
            *m /= 16.0;
        }
        // Variance pass: products are per-lane identical to the scalar
        // version; unlike the exact-integer means these sums reassociate,
        // shifting `s_mask` by ~ulp and flipping rare threshold boundaries
        // (observed ret drift ≤1e-7 rel — far inside the 2e-4 dB FFI gate).
        let sm_lo = f32x8::load_t(
            token,
            &[
                s_means[0], s_means[0], s_means[0], s_means[0], s_means[2], s_means[2], s_means[2],
                s_means[2],
            ],
        );
        let sm_hi = f32x8::load_t(
            token,
            &[
                s_means[1], s_means[1], s_means[1], s_means[1], s_means[3], s_means[3], s_means[3],
                s_means[3],
            ],
        );
        let dm_lo = f32x8::load_t(
            token,
            &[
                d_means[0], d_means[0], d_means[0], d_means[0], d_means[2], d_means[2], d_means[2],
                d_means[2],
            ],
        );
        let dm_hi = f32x8::load_t(
            token,
            &[
                d_means[1], d_means[1], d_means[1], d_means[1], d_means[3], d_means[3], d_means[3],
                d_means[3],
            ],
        );
        let sg8 = f32x8::splat_t(token, s_gmean);
        let dg8 = f32x8::splat_t(token, d_gmean);
        // Vector accumulators for the per-block stats. The quadrant
        // layout matches the mean vectors: lanes 0-3 / 4-7 map to
        // quadrant pairs {0,2} on rows i<4 and {1,3} on i>=4.
        let mut sgv = f32x8::splat_t(token, 0.0);
        let mut dgv = f32x8::splat_t(token, 0.0);
        let mut sv02 = f32x8::splat_t(token, 0.0);
        let mut sv13 = f32x8::splat_t(token, 0.0);
        let mut dv02 = f32x8::splat_t(token, 0.0);
        let mut dv13 = f32x8::splat_t(token, 0.0);
        for i in 0..8 {
            let xv = i32x8::load_t(token, dct_s[i * 8..][..8].first_chunk().unwrap()).to_f32();
            let ds = xv - sg8;
            sgv = sgv + ds * ds;
            let mv = if i < 4 { sm_lo } else { sm_hi };
            let vv = xv - mv;
            if i < 4 {
                sv02 = sv02 + vv * vv;
            } else {
                sv13 = sv13 + vv * vv;
            }
            let xv = i32x8::load_t(token, dct_d[i * 8..][..8].first_chunk().unwrap()).to_f32();
            let dd = xv - dg8;
            dgv = dgv + dd * dd;
            let mv = if i < 4 { dm_lo } else { dm_hi };
            let vv = xv - mv;
            if i < 4 {
                dv02 = dv02 + vv * vv;
            } else {
                dv13 = dv13 + vv * vv;
            }
        }
        let sgv = sgv.to_array();
        let dgv = dgv.to_array();
        let sv02 = sv02.to_array();
        let sv13 = sv13.to_array();
        let dv02 = dv02.to_array();
        let dv13 = dv13.to_array();
        let mut s_vars = [0.0f32; 4];
        let mut d_vars = [0.0f32; 4];
        s_vars[0] = (sv02[0] + sv02[1]) + (sv02[2] + sv02[3]);
        s_vars[1] = (sv13[0] + sv13[1]) + (sv13[2] + sv13[3]);
        s_vars[2] = (sv02[4] + sv02[5]) + (sv02[6] + sv02[7]);
        s_vars[3] = (sv13[4] + sv13[5]) + (sv13[6] + sv13[7]);
        d_vars[0] = (dv02[0] + dv02[1]) + (dv02[2] + dv02[3]);
        d_vars[1] = (dv13[0] + dv13[1]) + (dv13[2] + dv13[3]);
        d_vars[2] = (dv02[4] + dv02[5]) + (dv02[6] + dv02[7]);
        d_vars[3] = (dv13[4] + dv13[5]) + (dv13[6] + dv13[7]);
        // Pairwise-tree folds: the parens are the association order —
        // kept bit-identical to the reference reduction, not stylistic.
        #[allow(unused_parens)]
        let mut s_gvar =
            ((sgv[0] + sgv[1]) + (sgv[2] + sgv[3])) + ((sgv[4] + sgv[5]) + (sgv[6] + sgv[7]));
        let mut d_gvar =
            ((dgv[0] + dgv[1]) + (dgv[2] + dgv[3])) + ((dgv[4] + dgv[5]) + (dgv[6] + dgv[7]));
        // `*= 1/63.f*64` / `*= 1/15.f*16` — the constant folds in f32.
        s_gvar *= 1.0f32 / 63.0f32 * 64.0f32;
        d_gvar *= 1.0f32 / 63.0f32 * 64.0f32;
        for i in 0..4 {
            s_vars[i] *= 1.0f32 / 15.0f32 * 16.0f32;
            d_vars[i] *= 1.0f32 / 15.0f32 * 16.0f32;
        }
        if s_gvar > 0.0 {
            s_gvar = (s_vars[0] + s_vars[1] + s_vars[2] + s_vars[3]) / s_gvar;
        }
        if d_gvar > 0.0 {
            d_gvar = (d_vars[0] + d_vars[1] + d_vars[2] + d_vars[3]) / d_gvar;
        }
        fdct8x8_vectors(token, &mut dct_s);
        fdct8x8_vectors(token, &mut dct_d);
        let mut smv = f32x8::splat_t(token, 0.0);
        let mut dmv = f32x8::splat_t(token, 0.0);
        for i in 0..8 {
            let sq = i32x8::load_t(token, dct_s[i * 8..][..8].first_chunk().unwrap());
            smv = smv + (sq * sq).to_f32() * mask8[i];
            let sq = i32x8::load_t(token, dct_d[i * 8..][..8].first_chunk().unwrap());
            dmv = dmv + (sq * sq).to_f32() * mask8[i];
        }
        let smv = smv.to_array();
        let dmv = dmv.to_array();
        // Pairwise-tree fold: the parens are the association order — kept
        // bit-identical to the reference reduction, not stylistic.
        #[allow(unused_parens)]
        let mut s_mask =
            ((smv[0] + smv[1]) + (smv[2] + smv[3])) + ((smv[4] + smv[5]) + (smv[6] + smv[7]));
        let mut d_mask =
            ((dmv[0] + dmv[1]) + (dmv[2] + dmv[3])) + ((dmv[4] + dmv[5]) + (dmv[6] + dmv[7]));
        // C: `s_mask = sqrt(s_mask*s_gvar) / 32.f` — double sqrt of
        // the float product, double division, float store.
        s_mask = (((s_mask * s_gvar) as f64).sqrt() / 32.0) as f32;
        d_mask = (((d_mask * d_gvar) as f64).sqrt() / 32.0) as f32;
        if d_mask > s_mask {
            s_mask = d_mask;
        }
        let sm8 = f32x8::splat_t(token, s_mask);
        for i in 0..8 {
            let s8 = i32x8::load_t(token, dct_s[i * 8..][..8].first_chunk().unwrap());
            let d8 = i32x8::load_t(token, dct_d[i * 8..][..8].first_chunk().unwrap());
            let e = (s8 - d8).abs().to_f32();
            // `if err < t { 0 } else { err - t }` ≡ `max(err - t, 0)` —
            // the only NaN-producing lane is the overridden DC tap.
            let t8 = sm8 * rmask8[i];
            let e2 = (e - t8).max(f32x8::splat_t(token, 0.0));
            let ec = e2 * csf8[i];
            let mut pl = (ec * ec).to_array();
            if i == 0 {
                // The DC tap is unthresholded — replace lane 0's product
                // before the sequential fold.
                let e00 = (dct_s[0] - dct_d[0]).abs() as f32;
                pl[0] = (e00 * csf[0][0]) * (e00 * csf[0][0]);
            }
            for &v in &pl {
                *ret += v;
            }
        }
        *pixels += 64;
        x += step;
    }
}

/// Per-plane PSNR-HVS scores (dB) plus the libvmaf combined score.
#[derive(Debug, Clone, Copy)]
pub struct DaalaScore {
    /// `-10·log10` of the combined masked MSE
    /// `0.8·mse_y + 0.1·(mse_cb + mse_cr)` — the libvmaf `psnr_hvs`
    /// feature value (the AIC-4 `PSNR-HVS` column).
    pub psnr_hvs: f64,
    /// Luma plane score (`psnr_hvs_y`).
    pub psnr_hvs_y: f64,
    /// Cb plane score (`psnr_hvs_cb`).
    pub psnr_hvs_cb: f64,
    /// Cr plane score (`psnr_hvs_cr`).
    pub psnr_hvs_cr: f64,
}

fn score_db(mse: f64) -> f64 {
    // `libm::log10`, not the platform libm — crate convention for
    // bit-identical results across platforms.
    -10.0 * libm::log10(mse)
}

/// Which YUV plane a [`psnr_hvs_daala_plane_mse`] call scores — selects
/// the matching CSF table (Y / Cb420 / Cr420 are distinct matrices).
#[derive(Debug, Clone, Copy)]
pub enum DaalaPlane {
    /// Luma (`csf_y`).
    Y,
    /// Cb (`csf_cb420`).
    Cb,
    /// Cr (`csf_cr420`).
    Cr,
}

/// Daala PSNR-HVS on a single u8 plane pair (strided, `w`×`h`,
/// `step = 7`). Returns the masked MSE (pre-dB); caller applies
/// `-10·log10`.
pub fn psnr_hvs_daala_plane_mse(
    reference: &[u8],
    distorted: &[u8],
    stride: usize,
    w: usize,
    h: usize,
    plane: DaalaPlane,
) -> f64 {
    let csf = match plane {
        DaalaPlane::Y => &CSF_Y,
        DaalaPlane::Cb => &CSF_CB420,
        DaalaPlane::Cr => &CSF_CR420,
    };
    calc_psnrhvs(reference, stride, distorted, stride, w, h, 7, csf)
}

/// Full `psnr_hvs` on YUV444-style planes: `y`, `cb`, and `cr` planes
/// are ALL `w`×`h` (luma at `y_stride`, chroma at `c_stride`). This is
/// the convention the JPEG AIC-4 `PSNR-HVS` columns were computed with
/// (full-resolution chroma — verified Δ ≤ 0.03 dB vs the published
/// `PSNR-HVS-{Cb,Cr}` cells).
pub fn psnr_hvs_daala_yuv444(
    y_ref: &[u8],
    y_dis: &[u8],
    cb_ref: &[u8],
    cb_dis: &[u8],
    cr_ref: &[u8],
    cr_dis: &[u8],
    w: usize,
    h: usize,
    y_stride: usize,
    c_stride: usize,
) -> DaalaScore {
    // Each plane's `ret` fold is its own serial chain — the three
    // planes are independent, so `join2` runs them in parallel under
    // `parallel` and sequentially otherwise (bit-exact per plane either
    // way).
    let ((mse_y, mse_cb), mse_cr) = crate::kernel::join2(
        || {
            crate::kernel::join2(
                || calc_psnrhvs(y_ref, y_stride, y_dis, y_stride, w, h, 7, &CSF_Y),
                || calc_psnrhvs(cb_ref, c_stride, cb_dis, c_stride, w, h, 7, &CSF_CB420),
            )
        },
        || calc_psnrhvs(cr_ref, c_stride, cr_dis, c_stride, w, h, 7, &CSF_CR420),
    );
    DaalaScore {
        psnr_hvs: score_db(0.8 * mse_y + 0.1 * (mse_cb + mse_cr)),
        psnr_hvs_y: score_db(mse_y),
        psnr_hvs_cb: score_db(mse_cb),
        psnr_hvs_cr: score_db(mse_cr),
    }
}

/// Full `psnr_hvs` on YUV420-style planes: `y` planes are `w`×`h` at
/// `y_stride`; `cb`/`cr` planes are `(w/2)`×`(h/2)` at `c_stride`.
/// Slice lengths must cover `h·y_stride` / `ch·c_stride` elements.
pub fn psnr_hvs_daala_yuv420(
    y_ref: &[u8],
    y_dis: &[u8],
    cb_ref: &[u8],
    cb_dis: &[u8],
    cr_ref: &[u8],
    cr_dis: &[u8],
    w: usize,
    h: usize,
    y_stride: usize,
    c_stride: usize,
) -> DaalaScore {
    let cw = w / 2;
    let ch = h / 2;
    // Same independent-plane structure as the 444 path: luma and chroma
    // `ret` chains run on separate workers under `parallel`.
    let ((mse_y, mse_cb), mse_cr) = crate::kernel::join2(
        || {
            crate::kernel::join2(
                || calc_psnrhvs(y_ref, y_stride, y_dis, y_stride, w, h, 7, &CSF_Y),
                || calc_psnrhvs(cb_ref, c_stride, cb_dis, c_stride, cw, ch, 7, &CSF_CB420),
            )
        },
        || calc_psnrhvs(cr_ref, c_stride, cr_dis, c_stride, cw, ch, 7, &CSF_CR420),
    );
    DaalaScore {
        psnr_hvs: score_db(0.8 * mse_y + 0.1 * (mse_cb + mse_cr)),
        psnr_hvs_y: score_db(mse_y),
        psnr_hvs_cb: score_db(mse_cb),
        psnr_hvs_cr: score_db(mse_cr),
    }
}

#[cfg(test)]
mod transform_tests {
    use super::*;
    use archmage::SimdToken;

    fn check<T: magetypes::simd::backends::F32x8Convert>(token: T, input: [i32; 64]) {
        let mut expected = input;
        od_bin_fdct8x8(&mut expected);
        let mut actual = input;
        fdct8x8_vectors(token, &mut actual);
        assert_eq!(actual, expected, "input {input:?}");
    }

    #[test]
    fn vector_transform_matches_original_integer_coefficients() {
        let mut state = 0x9182_783bu32;
        for seed in 0..4096 {
            let input = core::array::from_fn(|i| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                match seed {
                    0 => 0,
                    1 => 255,
                    2..=65 => i32::from(i == seed - 2) * 255,
                    66 => i32::from(i % 2 == 0) * 255,
                    67 => (i as i32 * 17) % 256,
                    _ => (state >> 24) as i32,
                }
            });
            check(archmage::ScalarToken::summon().unwrap(), input);
            #[cfg(target_arch = "x86_64")]
            if let Some(token) = archmage::X64V3Token::summon() {
                check(token, input);
            }
            #[cfg(target_arch = "aarch64")]
            check(archmage::NeonToken::summon().unwrap(), input);
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            check(archmage::Wasm128Token::summon().unwrap(), input);
        }
    }
}

#[cfg(test)]
mod block_row_parity_tests {
    use super::*;
    use archmage::SimdToken;

    /// Scalar `block_row` reference — the pre-vectorization per-block
    /// math verbatim. The vectorized path must agree bitwise: every
    /// accumulator keeps the scalar i,j add order.
    #[allow(clippy::too_many_arguments)]
    fn block_row_scalar(
        src: &[u8],
        sstride: usize,
        dst: &[u8],
        dstride: usize,
        w: usize,
        y: usize,
        step: usize,
        mask: &[[f32; 8]; 8],
        csf: &[[f32; 8]; 8],
        ret: &mut f32,
        pixels: &mut u32,
    ) {
        let mut x = 0usize;
        while x + 7 < w {
            let mut dct_s = [0i32; 64];
            let mut dct_d = [0i32; 64];
            let mut s_means = [0.0f32; 4];
            let mut d_means = [0.0f32; 4];
            let mut s_gmean = 0.0f32;
            let mut d_gmean = 0.0f32;
            for i in 0..8 {
                for j in 0..8 {
                    let sub = ((i & 12) >> 2) + ((j & 12) >> 1);
                    let s = src[(y + i) * sstride + (j + x)] as i32;
                    let d = dst[(y + i) * dstride + (j + x)] as i32;
                    dct_s[i * 8 + j] = s;
                    dct_d[i * 8 + j] = d;
                    s_gmean += s as f32;
                    d_gmean += d as f32;
                    s_means[sub] += s as f32;
                    d_means[sub] += d as f32;
                }
            }
            s_gmean /= 64.0;
            d_gmean /= 64.0;
            for m in s_means.iter_mut() {
                *m /= 16.0;
            }
            for m in d_means.iter_mut() {
                *m /= 16.0;
            }
            let mut s_vars = [0.0f32; 4];
            let mut d_vars = [0.0f32; 4];
            let mut s_gvar = 0.0f32;
            let mut d_gvar = 0.0f32;
            for i in 0..64 {
                let sub = (((i / 8) & 12) >> 2) + (((i % 8) & 12) >> 1);
                let ds = dct_s[i] as f32 - s_gmean;
                let dd = dct_d[i] as f32 - d_gmean;
                s_gvar += ds * ds;
                d_gvar += dd * dd;
                let vs = dct_s[i] as f32 - s_means[sub];
                let vd = dct_d[i] as f32 - d_means[sub];
                s_vars[sub] += vs * vs;
                d_vars[sub] += vd * vd;
            }
            s_gvar *= 1.0f32 / 63.0f32 * 64.0f32;
            d_gvar *= 1.0f32 / 63.0f32 * 64.0f32;
            for i in 0..4 {
                s_vars[i] *= 1.0f32 / 15.0f32 * 16.0f32;
                d_vars[i] *= 1.0f32 / 15.0f32 * 16.0f32;
            }
            if s_gvar > 0.0 {
                s_gvar = (s_vars[0] + s_vars[1] + s_vars[2] + s_vars[3]) / s_gvar;
            }
            if d_gvar > 0.0 {
                d_gvar = (d_vars[0] + d_vars[1] + d_vars[2] + d_vars[3]) / d_gvar;
            }
            fdct8x8_vectors(archmage::ScalarToken::summon().unwrap(), &mut dct_s);
            fdct8x8_vectors(archmage::ScalarToken::summon().unwrap(), &mut dct_d);
            let mut s_mask = 0.0f32;
            let mut d_mask = 0.0f32;
            for i in 0..8 {
                for j in usize::from(i == 0)..8 {
                    s_mask += (dct_s[i * 8 + j] * dct_s[i * 8 + j]) as f32 * mask[i][j];
                    d_mask += (dct_d[i * 8 + j] * dct_d[i * 8 + j]) as f32 * mask[i][j];
                }
            }
            s_mask = (((s_mask * s_gvar) as f64).sqrt() / 32.0) as f32;
            d_mask = (((d_mask * d_gvar) as f64).sqrt() / 32.0) as f32;
            if d_mask > s_mask {
                s_mask = d_mask;
            }
            for i in 0..8 {
                for j in 0..8 {
                    let mut err = (dct_s[i * 8 + j] - dct_d[i * 8 + j]).abs() as f32;
                    if i != 0 || j != 0 {
                        let t = s_mask / mask[i][j];
                        err = if err < t { 0.0 } else { err - t };
                    }
                    *ret += (err * csf[i][j]) * (err * csf[i][j]);
                    *pixels += 1;
                }
            }
            x += step;
        }
    }

    #[test]
    fn vectorized_block_row_matches_scalar_bits() {
        let mut mask = [[0.0f32; 8]; 8];
        for i in 0..8 {
            for j in 0..8 {
                let v = CSF_Y[i][j] as f64 * 0.3885746225901003f64;
                mask[i][j] = (v * v) as f32;
            }
        }
        for (w, h) in [(64usize, 64usize), (96, 40), (300, 24)] {
            let src: Vec<u8> = (0..w * h)
                .map(|i| ((i * 37 + i / w * 91) % 251) as u8)
                .collect();
            let dst: Vec<u8> = (0..w * h)
                .map(|i| ((i * 53 + i / w * 7 + 11) % 241) as u8)
                .collect();
            let (mut got_ret, mut got_px) = (0.0f32, 0u32);
            let (mut want_ret, mut want_px) = (0.0f32, 0u32);
            let mut y = 0usize;
            let _ = h;
            while y + 7 < h {
                archmage::attuned!(
                    block_row(
                        &src,
                        w,
                        &dst,
                        w,
                        w,
                        y,
                        7,
                        &mask,
                        &CSF_Y,
                        &mut got_ret,
                        &mut got_px,
                    ),
                    [_v3_t, _neon_t, _wasm128_t, _scalar_t]
                );
                block_row_scalar(
                    &src,
                    w,
                    &dst,
                    w,
                    w,
                    y,
                    7,
                    &mask,
                    &CSF_Y,
                    &mut want_ret,
                    &mut want_px,
                );
                y += 7;
            }
            assert_eq!(got_px, want_px, "pixels {w}x{h}");
            // The per-block stats (means are exact-integer; variances and
            // mask sums are vector reassociations) can shift `s_mask` by
            // ~1 ulp, flipping an occasional threshold boundary — observed
            // ret deltas are ≤1e-7 relative, far inside the 2e-4 dB FFI
            // gate. The `ret` fold itself stays in scalar raster order.
            let delta = (got_ret - want_ret).abs() / want_ret.abs().max(1.0);
            assert!(
                delta <= 1e-6,
                "ret {w}x{h}: {got_ret} vs {want_ret} rel {delta}"
            );
        }
    }
}
