//! The HDR-VDP-3 multi-scale decomposition: an **isotropic** steerable
//! pyramid on the `sp0Filters` kernel set.
//!
//! Unlike HDR-VDP-2's `sp3Filters` pyramid (4 oriented bands per level),
//! `sp0` has a single isotropic bandpass filter per level — the pyramid is
//! effectively a Laplacian-style band stack:
//!
//! ```text
//!   hi0  = corrDn(im,  hi0filt)          — residual high-pass, full res
//!   lo0  = corrDn(im,  lo0filt)
//!   per level ℓ: band_ℓ = corrDn(lo_ℓ, bfilt);  lo_{ℓ+1} = corrDn(lo_ℓ, lofilt, ↓2)
//!   final band = lo_H                        — residual low-pass
//! ```
//!
//! so `band_count = levels + 2` and band `b` is normalised by `2^-b` on read
//! (stored raw, scaled on access — upstream's `get_band`/`set_band`
//! `band_norm`). Band frequencies are `ppd/2^(b+1)` cycles/degree.
//!
//! Boundary handling is `reflect1` (mirror without duplicating the edge
//! sample), matching `corrDn`/`upConv` in the reference's `PyrTools`.
//!
//! Ported from `VDP3/hdrvdp_spyr.py` + `VDP3/PyrTools.py` (`sp0Filters`,
//! `buildSpyr`, `reconSpyr`) of the `jpeg-ai-qaf` HDR-VDP-3 port — itself a
//! transliteration of Simoncelli's matlabPyrTools (MIT). `f64` throughout.

use crate::spyr::{max_pyr_height, reflect1};

/// [`corr_dn64`] block kernel: eight output columns at once — lane `j` is
/// output `ox0 + j`, so each lane's `acc` accumulates in the same tap order
/// as the scalar loop and the result is bit-identical. `wint` holds the
/// transposed tap windows for one `ky` (`wint[kx·8 + j]`).
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f64x8))]
#[allow(clippy::too_many_arguments)]
fn corr8_inner<const RELAXED: bool>(
    token: Token,
    image: &[f64],
    width: usize,
    xtab: &[usize],
    contig: &[bool],
    unit: bool,
    yt: &[usize],
    filter: &[f64],
    fs: usize,
    ox0: usize,
    out_row: &mut [f64],
) {
    let mut acc = f64x8::zero_t(token);
    if unit {
        // Whole block interior at step 1: lane `j`'s window is the
        // contiguous run `x0 + j + kx`, so each tap is one vector load —
        // `w[kx*8 + j] = row[x0 + kx + j]` exactly, bit-identical per lane.
        let x0 = xtab[0];
        for ky in 0..fs {
            let row = &image[yt[ky]..yt[ky] + width];
            let frow = &filter[ky * fs..ky * fs + fs];
            for kx in 0..fs {
                let t = f64x8::load_t(token, (&row[x0 + kx..x0 + kx + 8]).try_into().unwrap());
                let c = f64x8::splat_t(token, frow[kx]);
                acc = if RELAXED {
                    c.mul_add(t, acc)
                } else {
                    acc + c * t
                };
            }
        }
        acc.store((&mut out_row[ox0..ox0 + 8]).try_into().unwrap());
        return;
    }
    let mut wint = [0.0f64; 8 * 16];
    for ky in 0..fs {
        let row = &image[yt[ky]..yt[ky] + width];
        let w = &mut wint[..fs * 8];
        for (j, wt) in xtab.chunks_exact(fs).enumerate() {
            if contig[j] {
                // Interior lane: the reflected tap window is a contiguous
                // run — sequential loads instead of table lookups.
                for kx in 0..fs {
                    w[kx * 8 + j] = row[wt[0] + kx];
                }
            } else {
                for (kx, &ix) in wt.iter().enumerate() {
                    w[kx * 8 + j] = row[ix];
                }
            }
        }
        let frow = &filter[ky * fs..ky * fs + fs];
        for kx in 0..fs {
            let t = f64x8::load_t(token, (&w[kx * 8..kx * 8 + 8]).try_into().unwrap());
            let c = f64x8::splat_t(token, frow[kx]);
            acc = if RELAXED {
                c.mul_add(t, acc)
            } else {
                acc + c * t
            };
        }
    }
    acc.store((&mut out_row[ox0..ox0 + 8]).try_into().unwrap());
}

/// [`corr8_inner`] covering 32 unit-stride outputs: four independent
/// accumulator chains hide the FMA latency, each lane still summing its taps
/// in the scalar order (bit-identical to `corr8_inner` on the same block).
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f64x8))]
#[allow(clippy::too_many_arguments)]
fn corr32_inner<const RELAXED: bool>(
    token: Token,
    image: &[f64],
    width: usize,
    x0: usize,
    yt: &[usize],
    filter: &[f64],
    fs: usize,
    ox0: usize,
    out_row: &mut [f64],
) {
    let (mut a0, mut a1, mut a2, mut a3) = (
        f64x8::zero_t(token),
        f64x8::zero_t(token),
        f64x8::zero_t(token),
        f64x8::zero_t(token),
    );
    for ky in 0..fs {
        let row = &image[yt[ky]..yt[ky] + width];
        let frow = &filter[ky * fs..ky * fs + fs];
        for kx in 0..fs {
            let c = f64x8::splat_t(token, frow[kx]);
            let t0 = f64x8::load_t(token, (&row[x0 + kx..x0 + kx + 8]).try_into().unwrap());
            let t1 = f64x8::load_t(token, (&row[x0 + kx + 8..x0 + kx + 16]).try_into().unwrap());
            let t2 = f64x8::load_t(
                token,
                (&row[x0 + kx + 16..x0 + kx + 24]).try_into().unwrap(),
            );
            let t3 = f64x8::load_t(
                token,
                (&row[x0 + kx + 24..x0 + kx + 32]).try_into().unwrap(),
            );
            if RELAXED {
                a0 = c.mul_add(t0, a0);
                a1 = c.mul_add(t1, a1);
                a2 = c.mul_add(t2, a2);
                a3 = c.mul_add(t3, a3);
            } else {
                a0 = a0 + c * t0;
                a1 = a1 + c * t1;
                a2 = a2 + c * t2;
                a3 = a3 + c * t3;
            }
        }
    }
    a0.store((&mut out_row[ox0..ox0 + 8]).try_into().unwrap());
    a1.store((&mut out_row[ox0 + 8..ox0 + 16]).try_into().unwrap());
    a2.store((&mut out_row[ox0 + 16..ox0 + 24]).try_into().unwrap());
    a3.store((&mut out_row[ox0 + 24..ox0 + 32]).try_into().unwrap());
}

/// [`corr32_inner`] with runtime tier dispatch; all 32 outputs must be
/// unit-stride contiguous, i.e. `xtab` starts at the lane-0 window base.
#[allow(clippy::too_many_arguments)]
fn corr32<const RELAXED: bool>(
    image: &[f64],
    width: usize,
    x0: usize,
    yt: &[usize],
    filter: &[f64],
    fs: usize,
    ox0: usize,
    out_row: &mut [f64],
) {
    archmage::attuned!(
        corr32_inner::<RELAXED>(image, width, x0, yt, filter, fs, ox0, out_row),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

/// [`corr8_inner`] with runtime tier dispatch; `xtab` holds one `fs`-tap
/// window per lane (8 of them), `contig` flags lanes whose window is a
/// contiguous run starting at `xtab[j*fs]`.
#[allow(clippy::too_many_arguments)]
fn corr8<const RELAXED: bool>(
    image: &[f64],
    width: usize,
    xtab: &[usize],
    contig: &[bool],
    unit: bool,
    yt: &[usize],
    filter: &[f64],
    fs: usize,
    ox0: usize,
    out_row: &mut [f64],
) {
    debug_assert_eq!(xtab.len(), 8 * fs);
    debug_assert_eq!(contig.len(), 8);
    archmage::attuned!(
        corr8_inner::<RELAXED>(
            image, width, xtab, contig, unit, yt, filter, fs, ox0, out_row
        ),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

/// Pyramid low-pass pre-filter, 7×7 (upstream `sp0Filters` `lo0filt`).
#[allow(clippy::excessive_precision)]
const LO0FILT: [f64; 49] = [
    -4.514000000000000217e-04,
    -1.137099999999999938e-04,
    -3.725800000000000140e-04,
    -3.743859999999999896e-03,
    -3.725800000000000140e-04,
    -1.137099999999999938e-04,
    -4.514000000000000217e-04,
    -1.137099999999999938e-04,
    -6.119519999999999821e-03,
    -1.344159999999999973e-02,
    -7.563199999999999958e-03,
    -1.344159999999999973e-02,
    -6.119519999999999821e-03,
    -1.137099999999999938e-04,
    -3.725800000000000140e-04,
    -1.344159999999999973e-02,
    6.441487999999999381e-02,
    1.524935000000000040e-01,
    6.441487999999999381e-02,
    -1.344159999999999973e-02,
    -3.725800000000000140e-04,
    -3.743859999999999896e-03,
    -7.563199999999999958e-03,
    1.524935000000000040e-01,
    3.153017000000000181e-01,
    1.524935000000000040e-01,
    -7.563199999999999958e-03,
    -3.743859999999999896e-03,
    -3.725800000000000140e-04,
    -1.344159999999999973e-02,
    6.441487999999999381e-02,
    1.524935000000000040e-01,
    6.441487999999999381e-02,
    -1.344159999999999973e-02,
    -3.725800000000000140e-04,
    -1.137099999999999938e-04,
    -6.119519999999999821e-03,
    -1.344159999999999973e-02,
    -7.563199999999999958e-03,
    -1.344159999999999973e-02,
    -6.119519999999999821e-03,
    -1.137099999999999938e-04,
    -4.514000000000000217e-04,
    -1.137099999999999938e-04,
    -3.725800000000000140e-04,
    -3.743859999999999896e-03,
    -3.725800000000000140e-04,
    -1.137099999999999938e-04,
    -4.514000000000000217e-04,
];
/// Pyramid high-pass residual filter, 9×9 (`hi0filt`).
#[allow(clippy::excessive_precision)]
const HI0FILT: [f64; 81] = [
    5.997199999999999840e-04,
    -6.068000000000000179e-05,
    -3.324900000000000099e-04,
    -3.325599999999999737e-04,
    -2.406599999999999896e-04,
    -3.325599999999999737e-04,
    -3.324900000000000099e-04,
    -6.068000000000000179e-05,
    5.997199999999999840e-04,
    -6.068000000000000179e-05,
    1.263100000000000019e-04,
    4.927100000000000446e-04,
    1.459699999999999865e-04,
    -3.732100000000000131e-04,
    1.459699999999999865e-04,
    4.927100000000000446e-04,
    1.263100000000000019e-04,
    -6.068000000000000179e-05,
    -3.324900000000000099e-04,
    4.927100000000000446e-04,
    -1.616650000000000075e-03,
    -1.437358000000000038e-02,
    -2.420138000000000150e-02,
    -1.437358000000000038e-02,
    -1.616650000000000075e-03,
    4.927100000000000446e-04,
    -3.324900000000000099e-04,
    -3.325599999999999737e-04,
    1.459699999999999865e-04,
    -1.437358000000000038e-02,
    -6.300922999999999941e-02,
    -9.623594000000000592e-02,
    -6.300922999999999941e-02,
    -1.437358000000000038e-02,
    1.459699999999999865e-04,
    -3.325599999999999737e-04,
    -2.406599999999999896e-04,
    -3.732100000000000131e-04,
    -2.420138000000000150e-02,
    -9.623594000000000592e-02,
    8.554893000000000081e-01,
    -9.623594000000000592e-02,
    -2.420138000000000150e-02,
    -3.732100000000000131e-04,
    -2.406599999999999896e-04,
    -3.325599999999999737e-04,
    1.459699999999999865e-04,
    -1.437358000000000038e-02,
    -6.300922999999999941e-02,
    -9.623594000000000592e-02,
    -6.300922999999999941e-02,
    -1.437358000000000038e-02,
    1.459699999999999865e-04,
    -3.325599999999999737e-04,
    -3.324900000000000099e-04,
    4.927100000000000446e-04,
    -1.616650000000000075e-03,
    -1.437358000000000038e-02,
    -2.420138000000000150e-02,
    -1.437358000000000038e-02,
    -1.616650000000000075e-03,
    4.927100000000000446e-04,
    -3.324900000000000099e-04,
    -6.068000000000000179e-05,
    1.263100000000000019e-04,
    4.927100000000000446e-04,
    1.459699999999999865e-04,
    -3.732100000000000131e-04,
    1.459699999999999865e-04,
    4.927100000000000446e-04,
    1.263100000000000019e-04,
    -6.068000000000000179e-05,
    5.997199999999999840e-04,
    -6.068000000000000179e-05,
    -3.324900000000000099e-04,
    -3.325599999999999737e-04,
    -2.406599999999999896e-04,
    -3.325599999999999737e-04,
    -3.324900000000000099e-04,
    -6.068000000000000179e-05,
    5.997199999999999840e-04,
];
/// Pyramid level low-pass (decimation) filter, 13×13 (`lofilt`).
#[allow(clippy::excessive_precision)]
const LOFILT: [f64; 169] = [
    -2.257000000000000109e-04,
    -8.064399999999999563e-04,
    -5.686000000000000109e-05,
    8.741400000000000298e-04,
    -1.862800000000000122e-04,
    -1.031639999999999901e-03,
    -1.871920000000000008e-03,
    -1.031639999999999901e-03,
    -1.862800000000000122e-04,
    8.741400000000000298e-04,
    -5.686000000000000109e-05,
    -8.064399999999999563e-04,
    -2.257000000000000109e-04,
    -8.064399999999999563e-04,
    1.417620000000000078e-03,
    -1.903800000000000034e-04,
    -2.449059999999999866e-03,
    -4.596420000000000367e-03,
    -7.006740000000000017e-03,
    -6.948900000000000042e-03,
    -7.006740000000000017e-03,
    -4.596420000000000367e-03,
    -2.449059999999999866e-03,
    -1.903800000000000034e-04,
    1.417620000000000078e-03,
    -8.064399999999999563e-04,
    -5.686000000000000109e-05,
    -1.903800000000000034e-04,
    -3.059759999999999910e-03,
    -6.400999999999999572e-03,
    -6.720799999999999864e-03,
    -5.236180000000000001e-03,
    -3.781599999999999979e-03,
    -5.236180000000000001e-03,
    -6.720799999999999864e-03,
    -6.400999999999999572e-03,
    -3.059759999999999910e-03,
    -1.903800000000000034e-04,
    -5.686000000000000109e-05,
    8.741400000000000298e-04,
    -2.449059999999999866e-03,
    -6.400999999999999572e-03,
    -5.260019999999999800e-03,
    3.938620000000000315e-03,
    1.722078000000000150e-02,
    2.449600000000000041e-02,
    1.722078000000000150e-02,
    3.938620000000000315e-03,
    -5.260019999999999800e-03,
    -6.400999999999999572e-03,
    -2.449059999999999866e-03,
    8.741400000000000298e-04,
    -1.862800000000000122e-04,
    -4.596420000000000367e-03,
    -6.720799999999999864e-03,
    3.938620000000000315e-03,
    3.220743999999999690e-02,
    6.306261999999999979e-02,
    7.624673999999999341e-02,
    6.306261999999999979e-02,
    3.220743999999999690e-02,
    3.938620000000000315e-03,
    -6.720799999999999864e-03,
    -4.596420000000000367e-03,
    -1.862800000000000122e-04,
    -1.031639999999999901e-03,
    -7.006740000000000017e-03,
    -5.236180000000000001e-03,
    1.722078000000000150e-02,
    6.306261999999999979e-02,
    1.116387999999999964e-01,
    1.348998999999999893e-01,
    1.116387999999999964e-01,
    6.306261999999999979e-02,
    1.722078000000000150e-02,
    -5.236180000000000001e-03,
    -7.006740000000000017e-03,
    -1.031639999999999901e-03,
    -1.871920000000000008e-03,
    -6.948900000000000042e-03,
    -3.781599999999999979e-03,
    2.449600000000000041e-02,
    7.624673999999999341e-02,
    1.348998999999999893e-01,
    1.576508000000000076e-01,
    1.348998999999999893e-01,
    7.624673999999999341e-02,
    2.449600000000000041e-02,
    -3.781599999999999979e-03,
    -6.948900000000000042e-03,
    -1.871920000000000008e-03,
    -1.031639999999999901e-03,
    -7.006740000000000017e-03,
    -5.236180000000000001e-03,
    1.722078000000000150e-02,
    6.306261999999999979e-02,
    1.116387999999999964e-01,
    1.348998999999999893e-01,
    1.116387999999999964e-01,
    6.306261999999999979e-02,
    1.722078000000000150e-02,
    -5.236180000000000001e-03,
    -7.006740000000000017e-03,
    -1.031639999999999901e-03,
    -1.862800000000000122e-04,
    -4.596420000000000367e-03,
    -6.720799999999999864e-03,
    3.938620000000000315e-03,
    3.220743999999999690e-02,
    6.306261999999999979e-02,
    7.624673999999999341e-02,
    6.306261999999999979e-02,
    3.220743999999999690e-02,
    3.938620000000000315e-03,
    -6.720799999999999864e-03,
    -4.596420000000000367e-03,
    -1.862800000000000122e-04,
    8.741400000000000298e-04,
    -2.449059999999999866e-03,
    -6.400999999999999572e-03,
    -5.260019999999999800e-03,
    3.938620000000000315e-03,
    1.722078000000000150e-02,
    2.449600000000000041e-02,
    1.722078000000000150e-02,
    3.938620000000000315e-03,
    -5.260019999999999800e-03,
    -6.400999999999999572e-03,
    -2.449059999999999866e-03,
    8.741400000000000298e-04,
    -5.686000000000000109e-05,
    -1.903800000000000034e-04,
    -3.059759999999999910e-03,
    -6.400999999999999572e-03,
    -6.720799999999999864e-03,
    -5.236180000000000001e-03,
    -3.781599999999999979e-03,
    -5.236180000000000001e-03,
    -6.720799999999999864e-03,
    -6.400999999999999572e-03,
    -3.059759999999999910e-03,
    -1.903800000000000034e-04,
    -5.686000000000000109e-05,
    -8.064399999999999563e-04,
    1.417620000000000078e-03,
    -1.903800000000000034e-04,
    -2.449059999999999866e-03,
    -4.596420000000000367e-03,
    -7.006740000000000017e-03,
    -6.948900000000000042e-03,
    -7.006740000000000017e-03,
    -4.596420000000000367e-03,
    -2.449059999999999866e-03,
    -1.903800000000000034e-04,
    1.417620000000000078e-03,
    -8.064399999999999563e-04,
    -2.257000000000000109e-04,
    -8.064399999999999563e-04,
    -5.686000000000000109e-05,
    8.741400000000000298e-04,
    -1.862800000000000122e-04,
    -1.031639999999999901e-03,
    -1.871920000000000008e-03,
    -1.031639999999999901e-03,
    -1.862800000000000122e-04,
    8.741400000000000298e-04,
    -5.686000000000000109e-05,
    -8.064399999999999563e-04,
    -2.257000000000000109e-04,
];
/// Per-level isotropic bandpass filter, 9×9 (`bfilts`; one band per level).
#[allow(clippy::excessive_precision)]
const BFILT: [f64; 81] = [
    -9.066000000000000274e-05,
    -1.738640000000000064e-03,
    -4.942499999999999845e-03,
    -7.889389999999999598e-03,
    -1.009472999999999968e-02,
    -7.889389999999999598e-03,
    -4.942499999999999845e-03,
    -1.738640000000000064e-03,
    -9.066000000000000274e-05,
    -1.738640000000000064e-03,
    -4.625149999999999748e-03,
    -7.272540000000000081e-03,
    -7.623409999999999735e-03,
    -9.091949999999999685e-03,
    -7.623409999999999735e-03,
    -7.272540000000000081e-03,
    -4.625149999999999748e-03,
    -1.738640000000000064e-03,
    -4.942499999999999845e-03,
    -7.272540000000000081e-03,
    -2.129539999999999905e-02,
    -2.435661999999999897e-02,
    -3.487007999999999774e-02,
    -2.435661999999999897e-02,
    -2.129539999999999905e-02,
    -7.272540000000000081e-03,
    -4.942499999999999845e-03,
    -7.889389999999999598e-03,
    -7.623409999999999735e-03,
    -2.435661999999999897e-02,
    -1.730465999999999949e-02,
    -3.158604999999999746e-02,
    -1.730465999999999949e-02,
    -2.435661999999999897e-02,
    -7.623409999999999735e-03,
    -7.889389999999999598e-03,
    -1.009472999999999968e-02,
    -9.091949999999999685e-03,
    -3.487007999999999774e-02,
    -3.158604999999999746e-02,
    9.464194999999999691e-01,
    -3.158604999999999746e-02,
    -3.487007999999999774e-02,
    -9.091949999999999685e-03,
    -1.009472999999999968e-02,
    -7.889389999999999598e-03,
    -7.623409999999999735e-03,
    -2.435661999999999897e-02,
    -1.730465999999999949e-02,
    -3.158604999999999746e-02,
    -1.730465999999999949e-02,
    -2.435661999999999897e-02,
    -7.623409999999999735e-03,
    -7.889389999999999598e-03,
    -4.942499999999999845e-03,
    -7.272540000000000081e-03,
    -2.129539999999999905e-02,
    -2.435661999999999897e-02,
    -3.487007999999999774e-02,
    -2.435661999999999897e-02,
    -2.129539999999999905e-02,
    -7.272540000000000081e-03,
    -4.942499999999999845e-03,
    -1.738640000000000064e-03,
    -4.625149999999999748e-03,
    -7.272540000000000081e-03,
    -7.623409999999999735e-03,
    -9.091949999999999685e-03,
    -7.623409999999999735e-03,
    -7.272540000000000081e-03,
    -4.625149999999999748e-03,
    -1.738640000000000064e-03,
    -9.066000000000000274e-05,
    -1.738640000000000064e-03,
    -4.942499999999999845e-03,
    -7.889389999999999598e-03,
    -1.009472999999999968e-02,
    -7.889389999999999598e-03,
    -4.942499999999999845e-03,
    -1.738640000000000064e-03,
    -9.066000000000000274e-05,
];

/// `f64` `corrDn` — correlate with `reflect1` borders, keep every `step`-th
/// sample starting at 0. Filter origin is its centre `fs/2`.
#[must_use]
fn corr_dn64(
    image: &[f64],
    width: usize,
    height: usize,
    filter: &[f64],
    fs: usize,
    step: usize,
    relaxed: bool,
) -> Band3 {
    assert_eq!(image.len(), width * height, "corr_dn64: size");
    assert_eq!(filter.len(), fs * fs);
    let out_w = width.div_ceil(step);
    let out_h = height.div_ceil(step);
    let c = (fs / 2) as isize;
    // Reflected tap indices are shared by every output row/column —
    // compute them once instead of per output pixel.
    let mut xtab = vec![0usize; out_w * fs];
    // `contig[ox]` ⇔ the tap window never reflects, so lane `xtab[ox]`'s
    // window is the contiguous run `xtab[ox*fs] .. +fs`.
    let mut contig = vec![false; out_w];
    for ox in 0..out_w {
        let ix0 = ox * step;
        contig[ox] = ix0 >= c as usize && ix0 - c as usize + fs <= width;
        for kx in 0..fs {
            xtab[ox * fs + kx] = reflect1(ix0 as isize + kx as isize - c, width);
        }
    }
    let mut yoff = vec![0usize; out_h * fs];
    for oy in 0..out_h {
        let iy0 = oy * step;
        for ky in 0..fs {
            yoff[oy * fs + ky] = reflect1(iy0 as isize + ky as isize - c, height) * width;
        }
    }
    let mut out = vec![0.0; out_w * out_h];
    let full = out_w & !7;
    // Output rows are disjoint — parallel row-ranges under `parallel`.
    let rb = out_h.div_ceil(256).max(1);
    crate::par::for_each_dst_chunk(&mut out, rb * out_w, |o, out_c| {
        let oy0 = o / out_w;
        for ly in 0..out_c.len() / out_w {
        let oy = oy0 + ly;
        let yt = &yoff[oy * fs..oy * fs + fs];
        let out_row = &mut out_c[ly * out_w..ly * out_w + out_w];
        // 32-output groups chain four accumulators for FMA latency; blocks
        // covered by a `corr32` call are skipped via `skip_until`.
        let mut skip_until = 0usize;
        for ox0 in (0..full).step_by(8) {
            if ox0 < skip_until {
                continue;
            }
            // `unit` ⇔ lane `j`'s window start is `xtab[ox0*fs] + j`, i.e.
            // every lane contiguous and the block at step 1 — the kernel
            // then reads each tap as one vector load instead of gathering.
            let unit = step == 1 && contig[ox0..ox0 + 8].iter().all(|&c| c);
            if unit && ox0 + 32 <= full && contig[ox0 + 8..ox0 + 32].iter().all(|&c| c) {
                if relaxed {
                    corr32::<true>(image, width, xtab[ox0 * fs], yt, filter, fs, ox0, out_row);
                } else {
                    corr32::<false>(image, width, xtab[ox0 * fs], yt, filter, fs, ox0, out_row);
                }
                skip_until = ox0 + 32;
                continue;
            }
            if relaxed {
                corr8::<true>(
                    image,
                    width,
                    &xtab[ox0 * fs..ox0 * fs + 8 * fs],
                    &contig[ox0..ox0 + 8],
                    unit,
                    yt,
                    filter,
                    fs,
                    ox0,
                    out_row,
                );
            } else {
                corr8::<false>(
                    image,
                    width,
                    &xtab[ox0 * fs..ox0 * fs + 8 * fs],
                    &contig[ox0..ox0 + 8],
                    unit,
                    yt,
                    filter,
                    fs,
                    ox0,
                    out_row,
                );
            }
        }
        for ox in full..out_w {
            let xt = &xtab[ox * fs..ox * fs + fs];
            let mut acc = 0.0;
            for ky in 0..fs {
                let row = &image[yt[ky]..yt[ky] + width];
                for kx in 0..fs {
                    acc += filter[ky * fs + kx] * row[xt[kx]];
                }
            }
            out_row[ox] = acc;
        }
        }
    });
    Band3 {
        width: out_w,
        height: out_h,
        data: out,
    }
}

/// Interior `up_conv64` phase kernel: output phase `r` writes
/// `res_row[r + m·step]`; its aligned taps are the sub-filter
/// `filter[ky·fs + kx0 + t·step]` applied to the contiguous band window
/// `brow[m + off + t]` — one vector load per tap, same per-lane tap order as
/// the scalar loop (bit-identical). Lane `i` covers `m0 + i`.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f64x8))]
#[allow(clippy::too_many_arguments)]
fn up8_phase_inner<const RELAXED: bool>(
    token: Token,
    band: &[f64],
    bw: usize,
    rrow: &[usize],
    ron: &[bool],
    filter: &[f64],
    fs: usize,
    kx0: usize,
    ntaps: usize,
    base: usize,
    step: usize,
) -> [f64; 8] {
    let mut acc = f64x8::zero_t(token);
    for ky in 0..fs {
        if !ron[ky] {
            continue;
        }
        let brow = &band[rrow[ky]..rrow[ky] + bw];
        let frow = &filter[ky * fs..ky * fs + fs];
        for t in 0..ntaps {
            let c = f64x8::splat_t(token, frow[kx0 + t * step]);
            let w = f64x8::load_t(token, (&brow[base + t..base + t + 8]).try_into().unwrap());
            acc = if RELAXED {
                c.mul_add(w, acc)
            } else {
                acc + c * w
            };
        }
    }
    acc.to_array()
}

/// [`up8_phase_inner`] with runtime tier dispatch.
#[allow(clippy::too_many_arguments)]
fn up8_phase<const RELAXED: bool>(
    band: &[f64],
    bw: usize,
    rrow: &[usize],
    ron: &[bool],
    filter: &[f64],
    fs: usize,
    kx0: usize,
    ntaps: usize,
    base: usize,
    step: usize,
) -> [f64; 8] {
    archmage::attuned!(
        up8_phase_inner::<RELAXED>(band, bw, rrow, ron, filter, fs, kx0, ntaps, base, step),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

/// One `up_conv64` output the original full-order way — every tap in
/// sequence, zero contributions included (bit-exact signed-zero sums).
#[allow(clippy::too_many_arguments)]
fn up_pixel(
    band: &Band3,
    filter: &[f64],
    fs: usize,
    step: usize,
    out_w: usize,
    out_h: usize,
    py: usize,
    px: usize,
) -> f64 {
    let c = (fs / 2) as isize;
    let mut acc = 0.0;
    for ky in 0..fs {
        for kx in 0..fs {
            let y = reflect1(py as isize + ky as isize - c, out_h);
            let x = reflect1(px as isize + kx as isize - c, out_w);
            let v = if y.is_multiple_of(step) && x.is_multiple_of(step) {
                band.data[(y / step) * band.width + (x / step)]
            } else {
                0.0
            };
            acc += filter[ky * fs + kx] * v;
        }
    }
    acc
}

/// `f64` `upConv`: zero-upsample `band` by `step` onto the
/// `out_width × out_height` lattice and correlate with `filter` under
/// `reflect1` borders, accumulating into `res`.
fn up_conv64(
    band: &Band3,
    filter: &[f64],
    fs: usize,
    step: usize,
    out_width: usize,
    out_height: usize,
    res: &mut [f64],
    relaxed: bool,
) {
    assert_eq!(filter.len(), fs * fs);
    assert_eq!(res.len(), out_width * out_height);
    assert_eq!(
        (band.width, band.height),
        (out_width.div_ceil(step), out_height.div_ceil(step)),
        "up_conv64: band size vs out size"
    );
    let c = (fs / 2) as isize;
    let sample = |y: isize, x: isize| -> f64 {
        let y = reflect1(y, out_height);
        let x = reflect1(x, out_width);
        if !y.is_multiple_of(step) || !x.is_multiple_of(step) {
            return 0.0;
        }
        band.data[(y / step) * band.width + (x / step)]
    };
    // Only `step`-aligned taps can be nonzero; precompute the aligned
    // band row/col per (output, tap) so the inner loop walks just those.
    let mut rrow = vec![0usize; out_height * fs];
    let mut ron = vec![false; out_height * fs];
    for py in 0..out_height {
        for ky in 0..fs {
            let y = reflect1(py as isize + ky as isize - c, out_height);
            if y.is_multiple_of(step) {
                rrow[py * fs + ky] = (y / step) * band.width;
                ron[py * fs + ky] = true;
            }
        }
    }
    let mut rcol = vec![0usize; out_width * fs];
    let mut rcon = vec![false; out_width * fs];
    for px in 0..out_width {
        for kx in 0..fs {
            let x = reflect1(px as isize + kx as isize - c, out_width);
            if x.is_multiple_of(step) {
                rcol[px * fs + kx] = x / step;
                rcon[px * fs + kx] = true;
            }
        }
    }
    // Interior outputs have every tap window inside the lattice:
    // `px ∈ [c, out_w − fs + c + 1)`. There the aligned taps of phase `r`
    // form the contiguous sub-filter window `brow[m + off + t]`, handled by
    // [`up8_phase`] vector blocks; `cov_hi[r]` records how far they covered.
    let c_us = c as usize;
    let px_lo = c_us.min(out_width);
    let px_hi = out_width
        .saturating_sub(fs - 1)
        .saturating_add(c_us)
        .min(out_width);
    // Per phase `r`: `(r, kx0, ntaps, m_lo, m_vec_end, m_hi)`. Py-invariant.
    let mut phases: Vec<(usize, usize, usize, usize, usize, usize)> = Vec::with_capacity(step);
    // Interior outputs of a phase with no aligned tap still get their ±0
    // accumulation — send them through the scalar path.
    let mut uncovered: Vec<usize> = (0..px_lo).chain(px_hi..out_width).collect();
    for r in 0..step {
        let kx0 = (c - r as isize).rem_euclid(step as isize) as usize;
        let m_lo = px_lo.saturating_sub(r).div_ceil(step);
        let m_hi = px_hi.saturating_sub(r).div_ceil(step);
        if kx0 >= fs {
            for m in m_lo..m_hi {
                uncovered.push(r + m * step);
            }
            continue;
        }
        let ntaps = (fs - 1 - kx0) / step + 1;
        let m_vec = m_lo + m_hi.saturating_sub(m_lo) / 8 * 8;
        phases.push((r, kx0, ntaps, m_lo, m_vec, m_hi));
    }
    for &(r, _, _, _, vec_end, hi) in &phases {
        for m in vec_end..hi {
            uncovered.push(r + m * step);
        }
    }
    // Output rows are disjoint — parallel row-ranges under `parallel`.
    let rb = out_height.div_ceil(256).max(1);
    crate::par::for_each_dst_chunk(res, rb * out_width, |o, res_c| {
        let py0 = o / out_width;
        for ly in 0..res_c.len() / out_width {
        let py = py0 + ly;
        let rr = &rrow[py * fs..py * fs + fs];
        let rn = &ron[py * fs..py * fs + fs];
        let res_row = &mut res_c[ly * out_width..ly * out_width + out_width];
        for &(r, kx0, ntaps, m_lo, m_vec, _) in &phases {
            let off = (r as isize + kx0 as isize - c) / step as isize;
            for m0 in (m_lo..m_vec).step_by(8) {
                let base = (m0 as isize + off) as usize;
                let acc = if relaxed {
                    up8_phase::<true>(
                        &band.data, band.width, rr, rn, filter, fs, kx0, ntaps, base, step,
                    )
                } else {
                    up8_phase::<false>(
                        &band.data, band.width, rr, rn, filter, fs, kx0, ntaps, base, step,
                    )
                };
                for (j, a) in acc.iter().enumerate() {
                    // A ±0 lane can differ from the full-order sum only in
                    // the sign of zero — recompute it like the scalar path.
                    let a = if *a == 0.0 {
                        up_pixel(
                            band,
                            filter,
                            fs,
                            step,
                            out_width,
                            out_height,
                            py,
                            r + (m0 + j) * step,
                        )
                    } else {
                        *a
                    };
                    res_row[r + (m0 + j) * step] += a;
                }
            }
        }
        // `px < out_w & !7` historically ran the vector kernel: relaxed mode
        // therefore needs the FMA chain over every tap (unaligned taps add
        // f·(±0.0)), while all other outputs sum aligned taps in plain
        // mul+add order. Zero accumulators recompute full-order either way.
        let full8 = out_width & !7;
        for &px in &uncovered {
            let rc = &rcol[px * fs..px * fs + fs];
            let cn = &rcon[px * fs..px * fs + fs];
            let mut acc = 0.0;
            if relaxed && px < full8 {
                for ky in 0..fs {
                    if !rn[ky] {
                        continue;
                    }
                    let brow = &band.data[rr[ky]..rr[ky] + band.width];
                    for kx in 0..fs {
                        let v = if cn[kx] { brow[rc[kx]] } else { 0.0 };
                        acc = filter[ky * fs + kx].mul_add(v, acc);
                    }
                }
            } else {
                for ky in 0..fs {
                    if !rn[ky] {
                        continue;
                    }
                    let brow = &band.data[rr[ky]..rr[ky] + band.width];
                    for kx in 0..fs {
                        if !cn[kx] {
                            continue;
                        }
                        acc += filter[ky * fs + kx] * brow[rc[kx]];
                    }
                }
            }
            if acc == 0.0 {
                acc = 0.0;
                for ky in 0..fs {
                    for kx in 0..fs {
                        acc += filter[ky * fs + kx]
                            * sample(py as isize + ky as isize - c, px as isize + kx as isize - c);
                    }
                }
            }
            res_row[px] += acc;
        }
        }
    });
}

/// A single pyramid plane.
#[derive(Debug, Clone)]
pub(super) struct Band3 {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Row-major samples — the **stored** (un-normalised) coefficients.
    pub data: Vec<f64>,
}

/// The sp0 pyramid of one photoreceptor-response image.
///
/// `bands[0]` is the residual high-pass at full resolution, `bands[1..=H]`
/// the isotropic bandpass levels, `bands[H+1]` the low-pass baseband.
/// [`Self::band`] applies the `2^-b` normalisation upstream's `get_band`
/// uses; [`Self::set_band`] scales by `2^b` on write, matching `set_band`.
#[derive(Debug, Clone)]
pub(super) struct Spyr0 {
    bands: Vec<Band3>,
    /// `2^-(b+1) · ppd` per band — upstream `get_freqs`.
    freqs: Vec<f64>,
}

impl Spyr0 {
    /// Decompose `image` (`width × height` row-major) into the sp0 pyramid.
    ///
    /// `levels = min(ceil(log2(ppd)) − 2, maxPyrHt(img, 13))` — upstream's
    /// rule, where 13 is the `lofilt` size. Returns `None` when the image is
    /// too small for even one level (min dimension < 13).
    #[must_use]
    pub(super) fn decompose(
        image: &[f64],
        width: usize,
        height: usize,
        ppd: f64,
        relaxed: bool,
    ) -> Option<Self> {
        assert_eq!(image.len(), width * height, "spyr0: size");
        let max_ht = max_pyr_height(width, height, 13);
        let levels = (ppd.log2().ceil() as i64 - 2).min(max_ht as i64);
        if levels < 0 {
            return None;
        }
        let levels = levels as usize;

        let mut bands = Vec::with_capacity(levels + 2);
        bands.push(corr_dn64(image, width, height, &HI0FILT, 9, 1, relaxed));
        let mut lo = corr_dn64(image, width, height, &LO0FILT, 7, 1, relaxed);
        for _ in 0..levels {
            bands.push(corr_dn64(
                &lo.data, lo.width, lo.height, &BFILT, 9, 1, relaxed,
            ));
            lo = corr_dn64(&lo.data, lo.width, lo.height, &LOFILT, 13, 2, relaxed);
        }
        bands.push(lo);

        let freqs = (0..bands.len())
            .map(|b| ppd / 2f64.powi(b as i32 + 1))
            .collect();
        Some(Self { bands, freqs })
    }

    /// Number of bands (`levels + 2`).
    #[must_use]
    pub(super) fn band_count(&self) -> usize {
        self.bands.len()
    }

    /// Band frequencies in cycles/degree, one per band.
    #[must_use]
    pub(super) fn freqs(&self) -> &[f64] {
        &self.freqs
    }

    /// `(width, height)` of band `b`.
    #[must_use]
    pub(super) fn band_dims(&self, b: usize) -> (usize, usize) {
        (self.bands[b].width, self.bands[b].height)
    }

    /// Band `b`'s coefficients, normalised by `2^-b` (upstream `get_band`).
    #[must_use]
    pub(super) fn band(&self, b: usize) -> Vec<f64> {
        let norm = 2f64.powi(b as i32);
        let d = &self.bands[b].data;
        crate::par::collect_indexed(d.len(), |i| d[i] / norm)
    }

    /// Write `data` (already in the normalised domain) into band `b`,
    /// storing `data · 2^b` (upstream `set_band`).
    pub(super) fn set_band(&mut self, b: usize, data: &[f64]) {
        assert_eq!(data.len(), self.bands[b].data.len(), "set_band size");
        let norm = 2f64.powi(b as i32);
        let d = &mut self.bands[b].data;
        let cb = crate::par::chunk8(d.len());
        crate::par::for_each_dst_chunk(d, cb, |o, dc| {
            let n = dc.len();
            for (dv, &v) in dc.iter_mut().zip(&data[o..o + n]) {
                *dv = v * norm;
            }
        });
    }

    /// `reconSpyr(pyr, pind, 'sp0Filters', 'reflect1', levs, 'all')`.
    ///
    /// `levs` names pyramid *levels*: `0` is the residual high-pass,
    /// `1..=H` the isotropic bands, `H+1` the low-pass. A level contributes
    /// only when present in `levs`; deeper levels are chained through the
    /// low-pass reconstruction whenever `levs` contains a level deeper than
    /// the current one — mirroring the reference recursion exactly.
    ///
    /// The metric only ever calls this with `levs = 0..H` (everything but
    /// the last bandpass level and the low-pass) to build the masking
    /// transform.
    #[must_use]
    pub(super) fn reconstruct(&self, levs: &[usize], relaxed: bool) -> Vec<f64> {
        let (w0, h0) = self.band_dims(0);
        let n_levels = self.bands.len() - 2; // H
        let lowpass_idx = self.bands.len() - 1;
        let has = |l: usize| levs.contains(&l);

        // reconSpyrLevs: `level` is the 1-based level number, equal to the
        // band index for sp0. Level `k` descends iff some requested level
        // is deeper than `k`; at the bottom call (`level == H`) the source
        // is the low-pass row.
        fn rec(me: &Spyr0, level: usize, has: &dyn Fn(usize) -> bool, relaxed: bool) -> Band3 {
            let (bw, bh) = me.band_dims(level);
            let mut acc = vec![0.0; bw * bh];
            if (level + 1..=me.bands.len() - 1).any(has) {
                let deeper;
                let src = if level < me.bands.len() - 2 {
                    deeper = rec(me, level + 1, has, relaxed);
                    &deeper
                } else {
                    &me.bands[me.bands.len() - 1] // low-pass row
                };
                up_conv64(src, &LOFILT, 13, 2, bw, bh, &mut acc, relaxed);
            }
            if has(level) {
                up_conv64(&me.bands[level], &BFILT, 9, 1, bw, bh, &mut acc, relaxed);
            }
            Band3 {
                width: bw,
                height: bh,
                data: acc,
            }
        }

        let res1 = if n_levels == 0 {
            // `spyrHt == 0`: `res1 = pyrBand(2)` (the low-pass row) iff
            // level 1 requested, else a zero map.
            if has(1) {
                self.bands[lowpass_idx].clone()
            } else {
                Band3 {
                    width: w0,
                    height: h0,
                    data: vec![0.0; w0 * h0],
                }
            }
        } else {
            rec(self, 1, &has, relaxed)
        };

        let mut out = vec![0.0; w0 * h0];
        up_conv64(&res1, &LO0FILT, 7, 1, w0, h0, &mut out, relaxed);
        if has(0) {
            up_conv64(&self.bands[0], &HI0FILT, 9, 1, w0, h0, &mut out, relaxed);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checker(w: usize, h: usize) -> Vec<f64> {
        (0..w * h)
            .map(|i| ((i / w + i % w) % 2) as f64 * 40.0 + (i % 7) as f64 * 0.5)
            .collect()
    }

    #[test]
    fn decompose_reconstruct_roundtrip() {
        let (w, h) = (96, 96);
        let im = checker(w, h);
        let p = Spyr0::decompose(&im, w, h, 30.0, false).unwrap();
        let levs: Vec<usize> = (0..p.band_count()).collect();
        let recon = p.reconstruct(&levs, false);
        // sp0 filters are not a perfect-reconstruction set at the borders,
        // but the interior should come back within a few % of a 40-unit
        // checker amplitude.
        let mut worst = 0.0f64;
        for y in 8..h - 8 {
            for x in 8..w - 8 {
                worst = worst.max((recon[y * w + x] - im[y * w + x]).abs());
            }
        }
        assert!(worst < 2.0, "interior reconstruction error {worst}");
    }

    #[test]
    fn up_conv64_matches_up_pixel_bits() {
        // The phase-decomposed driver must reproduce the historical
        // contract bit-for-bit: vector-block outputs (px < out_w & !7) sum
        // via FMA over every tap in relaxed mode, everything else sums the
        // aligned taps in mul+add order; zero accumulators recompute
        // full-order via `up_pixel`.
        for &(fs, step) in &[(7usize, 1usize), (9, 1), (13, 2), (9, 2)] {
            let (bw, bh) = (17, 13);
            // `up_conv64` requires `bw == ow.div_ceil(step)`.
            let (ow, oh) = (bw * step - (step - 1), bh * step - (step - 1));
            let band = Band3 {
                width: bw,
                height: bh,
                data: (0..bw * bh)
                    .map(|i| ((i * 2654435761usize) % 1000) as f64 * 0.01 - 5.0)
                    .collect(),
            };
            let filter: Vec<f64> = (0..fs * fs)
                .map(|i| ((i * 40503usize) % 97) as f64 * 0.01 - 0.5)
                .collect();
            for relaxed in [false, true] {
                let mut res = (0..ow * oh)
                    .map(|i| ((i * 97usize) % 23) as f64 - 11.0)
                    .collect::<Vec<_>>();
                let c = (fs / 2) as isize;
                let full8 = ow & !7;
                let expect: Vec<f64> = res
                    .iter()
                    .enumerate()
                    .map(|(i, &r)| {
                        let (py, px) = (i / ow, i % ow);
                        let mut acc = 0.0f64;
                        if relaxed && px < full8 {
                            for ky in 0..fs {
                                let y = reflect1(py as isize + ky as isize - c, oh);
                                for kx in 0..fs {
                                    let x = reflect1(px as isize + kx as isize - c, ow);
                                    let v = if y.is_multiple_of(step) && x.is_multiple_of(step) {
                                        band.data[(y / step) * bw + (x / step)]
                                    } else {
                                        0.0
                                    };
                                    acc = filter[ky * fs + kx].mul_add(v, acc);
                                }
                            }
                        } else {
                            for ky in 0..fs {
                                let y = reflect1(py as isize + ky as isize - c, oh);
                                if !y.is_multiple_of(step) {
                                    continue;
                                }
                                for kx in 0..fs {
                                    let x = reflect1(px as isize + kx as isize - c, ow);
                                    if x.is_multiple_of(step) {
                                        acc += filter[ky * fs + kx]
                                            * band.data[(y / step) * bw + (x / step)];
                                    }
                                }
                            }
                        }
                        r + if acc == 0.0 {
                            up_pixel(&band, &filter, fs, step, ow, oh, py, px)
                        } else {
                            acc
                        }
                    })
                    .collect();
                up_conv64(&band, &filter, fs, step, ow, oh, &mut res, relaxed);
                for (i, (&a, &e)) in res.iter().zip(&expect).enumerate() {
                    if a.to_bits() != e.to_bits() {
                        eprintln!(
                            "DIFF fs={fs} step={step} relaxed={relaxed} px=({},{}) got {a:024.17e} {:016x} want {e:024.17e} {:016x}",
                            i / ow,
                            i % ow,
                            a.to_bits(),
                            e.to_bits(),
                        );
                    }
                    assert_eq!(a.to_bits(), e.to_bits(), "px {i}");
                }
            }
        }
    }

    #[test]
    fn freqs_halve_per_band() {
        let im = vec![1.0; 64 * 64];
        let p = Spyr0::decompose(&im, 64, 64, 30.0, false).unwrap();
        for (b, &f) in p.freqs().iter().enumerate() {
            assert!((f - 30.0 / 2f64.powi(b as i32 + 1)).abs() < 1e-12);
        }
    }
}
