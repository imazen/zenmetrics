use crate::{Error, ModelVariant, VmafV0Variant};
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use archmage::intrinsics::x86_64::*;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use archmage::{SimdToken, X64V3Token, arcane, rite};
#[cfg(feature = "simd")]
use archmage::{autoversion, magetypes};

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline(always)]
fn a8<T, const N: usize>(s: &[T]) -> &[T; N] {
    s.try_into().unwrap()
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline(always)]
fn a8m<T, const N: usize>(s: &mut [T]) -> &mut [T; N] {
    s.try_into().unwrap()
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline(always)]
fn v3_token() -> Option<X64V3Token> {
    #[cfg(test)]
    if FORCE_SCALAR.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    X64V3Token::summon()
}

#[cfg(all(test, feature = "simd", target_arch = "x86_64"))]
static FORCE_SCALAR: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

const ADM_BORDER_FACTOR: f64 = 0.1;
pub(crate) const ADM_MIN_DIM: usize = 33;
const DLM_WEIGHT: f64 = 0.7;
const ENHN_GAIN_LIMIT: f64 = 1.0;
const NOISE_WEIGHT: f64 = 0.02;
const MIN_VAL: f64 = 0.5;
const ONE_BY_15: i32 = 8738;
const I4_ONE_BY_15: i64 = 286331153;
const DWT2_LO: [i16; 4] = [15826, 27411, 7345, -4240];
const DWT2_HI: [i16; 4] = [-4240, -7345, 27411, -15826];
const DWT2_LO_SUM: i32 = 46342;
const DWT2_HI_SUM: i32 = 0;

const BLENDED_CSF_1080_3H: [[f32; 4]; 2] = [
    [0.01183, 0.025026, 0.04295, 0.058621],
    [0.004302, 0.011778, 0.023918, 0.035901],
];
const BLENDED_CSF_1080_5H: [[f32; 4]; 2] = [
    [0.004212, 0.014809, 0.029642, 0.047464],
    [0.000984, 0.005852, 0.0146, 0.027574],
];
const BLENDED_CSF_2160_3H: [[f32; 4]; 2] = [
    [0.00226, 0.01183, 0.025026, 0.04295],
    [0.000479, 0.004302, 0.011778, 0.023918],
];
const WATSON97_CSF_1080_3H: [[f32; 4]; 2] = [
    [
        f32::from_bits(0x3c8e63ba),
        f32::from_bits(0x3d030286),
        f32::from_bits(0x3d31a788),
        f32::from_bits(0x3d3b140b),
    ],
    [
        f32::from_bits(0x3bc106aa),
        f32::from_bits(0x3c6a46a3),
        f32::from_bits(0x3cc7dc06),
        f32::from_bits(0x3d0041c9),
    ],
];

struct AdmSettings {
    factors: &'static [[f32; 4]; 2],
    enhn_gain_limit: f64,
    noise_weight: f64,
    watson_fixed: bool,
    norm_view_dist: f64,
    ref_display_height: usize,
}

fn csf_table(variant: ModelVariant) -> &'static [[f32; 4]; 2] {
    match variant {
        ModelVariant::Phone | ModelVariant::HfrPhone => &BLENDED_CSF_1080_5H,
        ModelVariant::Consumer4k | ModelVariant::HfrConsumer4k => &BLENDED_CSF_2160_3H,
        _ => &BLENDED_CSF_1080_3H,
    }
}

fn rfactor(settings: &AdmSettings, scale: usize) -> [f32; 3] {
    let t = settings.factors;
    [t[0][scale], t[0][scale], t[1][scale]]
}

fn div_lookup() -> &'static [i32; 65537] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[i32; 65537]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [0i32; 65537];
        for i in 1..=32768i64 {
            let recip = (1i64 << 30) / i;
            t[32768 + i as usize] = recip as i32;
            t[32768 - i as usize] = -(recip as i32);
        }
        t
    })
}

fn dwt2_indices(w: usize, h: usize) -> (Vec<[i32; 4]>, Vec<[i32; 4]>) {
    let h_half = h.div_ceil(2);
    let w_half = w.div_ceil(2);
    let mut ind_y = vec![[0i32; 4]; h_half];
    let mut ind_x = vec![[0i32; 4]; w_half];
    ind_y[0] = [1, 0, 1, 2];
    ind_x[0] = [1, 0, 1, 2];
    for (i, slot) in ind_y
        .iter_mut()
        .enumerate()
        .take(h_half.saturating_sub(2))
        .skip(1)
    {
        *slot = [
            2 * i as i32 - 1,
            2 * i as i32,
            2 * i as i32 + 1,
            2 * i as i32 + 2,
        ];
    }
    for (i, slot) in ind_y.iter_mut().enumerate().skip(h_half.saturating_sub(2)) {
        let mut idx = [
            2 * i as i32 - 1,
            2 * i as i32,
            2 * i as i32 + 1,
            2 * i as i32 + 2,
        ];
        for v in idx.iter_mut() {
            if *v >= h as i32 {
                *v = 2 * h as i32 - *v - 1;
            }
        }
        *slot = idx;
    }
    for (j, slot) in ind_x
        .iter_mut()
        .enumerate()
        .take(w_half.saturating_sub(2))
        .skip(1)
    {
        *slot = [
            2 * j as i32 - 1,
            2 * j as i32,
            2 * j as i32 + 1,
            2 * j as i32 + 2,
        ];
    }
    for (j, slot) in ind_x.iter_mut().enumerate().skip(w_half.saturating_sub(2)) {
        let mut idx = [
            2 * j as i32 - 1,
            2 * j as i32,
            2 * j as i32 + 1,
            2 * j as i32 + 2,
        ];
        for v in idx.iter_mut() {
            if *v >= w as i32 {
                *v = 2 * w as i32 - *v - 1;
            }
        }
        *slot = idx;
    }
    (ind_y, ind_x)
}

#[derive(Default)]
struct BandI16 {
    h: Vec<i16>,
    v: Vec<i16>,
    d: Vec<i16>,
}

#[derive(Default)]
struct BandI32 {
    h: Vec<i32>,
    v: Vec<i32>,
    d: Vec<i32>,
}

#[cfg(any(test, not(feature = "simd")))]
fn dwt2_vertical_scalar(
    src: &[u16],
    src_stride: usize,
    w: usize,
    bit_depth: u8,
    indices: [i32; 4],
    tmplo: &mut [i16],
    tmphi: &mut [i16],
) {
    let shift_vp = bit_depth as i32;
    let add_shift_vp = 1i32 << (bit_depth - 1);
    for j in 0..w {
        let s = [
            src[indices[0] as usize * src_stride + j] as i32,
            src[indices[1] as usize * src_stride + j] as i32,
            src[indices[2] as usize * src_stride + j] as i32,
            src[indices[3] as usize * src_stride + j] as i32,
        ];
        let mut accum = 0i32;
        for k in 0..4 {
            accum += DWT2_LO[k] as i32 * s[k];
        }
        accum -= DWT2_LO_SUM * add_shift_vp;
        tmplo[j] = ((accum + add_shift_vp) >> shift_vp) as i16;
        let mut accum = 0i32;
        for k in 0..4 {
            accum += DWT2_HI[k] as i32 * s[k];
        }
        accum -= DWT2_HI_SUM * add_shift_vp;
        tmphi[j] = ((accum + add_shift_vp) >> shift_vp) as i16;
    }
}

#[cfg(feature = "simd")]
#[magetypes(define(i32x8), v3, neon, wasm128, scalar)]
fn dwt2_vertical_simd(
    token: Token,
    src: &[u16],
    src_stride: usize,
    w: usize,
    bit_depth: u8,
    indices: [i32; 4],
    tmplo: &mut [i16],
    tmphi: &mut [i16],
) {
    let shift = bit_depth as u32;
    let add_shift_vp = 1i32 << (bit_depth - 1);
    let chunks = w / 8;
    for chunk in 0..chunks {
        let col = chunk * 8;
        let s = [
            i32x8::from_array(
                token,
                core::array::from_fn(|lane| {
                    src[indices[0] as usize * src_stride + col + lane] as i32
                }),
            ),
            i32x8::from_array(
                token,
                core::array::from_fn(|lane| {
                    src[indices[1] as usize * src_stride + col + lane] as i32
                }),
            ),
            i32x8::from_array(
                token,
                core::array::from_fn(|lane| {
                    src[indices[2] as usize * src_stride + col + lane] as i32
                }),
            ),
            i32x8::from_array(
                token,
                core::array::from_fn(|lane| {
                    src[indices[3] as usize * src_stride + col + lane] as i32
                }),
            ),
        ];
        let mut lo = i32x8::zero(token);
        for k in 0..4 {
            lo += s[k] * i32x8::splat(token, DWT2_LO[k] as i32);
        }
        lo -= i32x8::splat(token, DWT2_LO_SUM * add_shift_vp);
        lo = (lo + i32x8::splat(token, add_shift_vp)).shr_arithmetic_uniform(shift);
        let mut hi = i32x8::zero(token);
        for k in 0..4 {
            hi += s[k] * i32x8::splat(token, DWT2_HI[k] as i32);
        }
        hi -= i32x8::splat(token, DWT2_HI_SUM * add_shift_vp);
        hi = (hi + i32x8::splat(token, add_shift_vp)).shr_arithmetic_uniform(shift);
        let lo_arr = lo.to_array();
        let hi_arr = hi.to_array();
        for lane in 0..8 {
            tmplo[col + lane] = lo_arr[lane] as i16;
            tmphi[col + lane] = hi_arr[lane] as i16;
        }
    }
    let shift_vp = bit_depth as i32;
    for j in chunks * 8..w {
        let s = [
            src[indices[0] as usize * src_stride + j] as i32,
            src[indices[1] as usize * src_stride + j] as i32,
            src[indices[2] as usize * src_stride + j] as i32,
            src[indices[3] as usize * src_stride + j] as i32,
        ];
        let mut accum = 0i32;
        for k in 0..4 {
            accum += DWT2_LO[k] as i32 * s[k];
        }
        accum -= DWT2_LO_SUM * add_shift_vp;
        tmplo[j] = ((accum + add_shift_vp) >> shift_vp) as i16;
        let mut accum = 0i32;
        for k in 0..4 {
            accum += DWT2_HI[k] as i32 * s[k];
        }
        accum -= DWT2_HI_SUM * add_shift_vp;
        tmphi[j] = ((accum + add_shift_vp) >> shift_vp) as i16;
    }
}

#[cfg(all(test, feature = "simd"))]
fn adm_dwt2_scalar(
    src: &[u16],
    src_stride: usize,
    dst: &mut BandI16,
    w: usize,
    _h: usize,
    dst_stride: usize,
    bit_depth: u8,
    ind_y: &[[i32; 4]],
    ind_x: &[[i32; 4]],
    a_out: &mut [i32],
) {
    adm_dwt2_core(
        src,
        src_stride,
        dst,
        w,
        dst_stride,
        bit_depth,
        ind_y,
        ind_x,
        a_out,
        dwt2_vertical_scalar,
        dwt2_horizontal_scalar,
    );
}

fn adm_dwt2(
    src: &[u16],
    src_stride: usize,
    dst: &mut BandI16,
    w: usize,
    _h: usize,
    dst_stride: usize,
    bit_depth: u8,
    ind_y: &[[i32; 4]],
    ind_x: &[[i32; 4]],
    a_out: &mut [i32],
) {
    #[cfg(feature = "simd")]
    adm_dwt2_core(
        src,
        src_stride,
        dst,
        w,
        dst_stride,
        bit_depth,
        ind_y,
        ind_x,
        a_out,
        |src, src_stride, w, bit_depth, indices, tmplo, tmphi| {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            if let Some(token) = v3_token() {
                dwt2_vertical_v3(token, src, src_stride, w, bit_depth, indices, tmplo, tmphi);
                return;
            }
            archmage::incant!(
                dwt2_vertical_simd(src, src_stride, w, bit_depth, indices, tmplo, tmphi),
                [v3, neon, wasm128, scalar]
            );
        },
        dwt2_horizontal_row,
    );
    #[cfg(not(feature = "simd"))]
    adm_dwt2_core(
        src,
        src_stride,
        dst,
        w,
        dst_stride,
        bit_depth,
        ind_y,
        ind_x,
        a_out,
        dwt2_vertical_scalar,
        dwt2_horizontal_scalar,
    );
}

fn dwt2_horizontal_at(
    tmplo: &[i16],
    tmphi: &[i16],
    ix: [i32; 4],
    i: usize,
    j: usize,
    dst_stride: usize,
    dst: &mut BandI16,
    a_row: &mut [i32],
) {
    let (j0, j1, j2, j3) = (
        ix[0] as usize,
        ix[1] as usize,
        ix[2] as usize,
        ix[3] as usize,
    );
    let s = [tmplo[j0], tmplo[j1], tmplo[j2], tmplo[j3]];
    let mut accum = 0i32;
    for k in 0..4 {
        accum += DWT2_LO[k] as i32 * s[k] as i32;
    }
    a_row[j] = ((accum + 32768) >> 16) as i16 as i32;
    let mut accum = 0i32;
    for k in 0..4 {
        accum += DWT2_HI[k] as i32 * s[k] as i32;
    }
    dst.v[i * dst_stride + j] = ((accum + 32768) >> 16) as i16;
    let s = [tmphi[j0], tmphi[j1], tmphi[j2], tmphi[j3]];
    let mut accum = 0i32;
    for k in 0..4 {
        accum += DWT2_LO[k] as i32 * s[k] as i32;
    }
    dst.h[i * dst_stride + j] = ((accum + 32768) >> 16) as i16;
    let mut accum = 0i32;
    for k in 0..4 {
        accum += DWT2_HI[k] as i32 * s[k] as i32;
    }
    dst.d[i * dst_stride + j] = ((accum + 32768) >> 16) as i16;
}

fn dwt2_horizontal_scalar(
    tmplo: &[i16],
    tmphi: &[i16],
    ind_x: &[[i32; 4]],
    i: usize,
    _w: usize,
    dst_stride: usize,
    dst: &mut BandI16,
    a_out: &mut [i32],
) {
    let a_row = &mut a_out[i * dst_stride..i * dst_stride + ind_x.len()];
    for (j, &ix) in ind_x.iter().enumerate() {
        dwt2_horizontal_at(tmplo, tmphi, ix, i, j, dst_stride, dst, a_row);
    }
}

#[cfg(feature = "simd")]
fn dwt2_horizontal_row(
    tmplo: &[i16],
    tmphi: &[i16],
    ind_x: &[[i32; 4]],
    i: usize,
    w: usize,
    dst_stride: usize,
    dst: &mut BandI16,
    a_out: &mut [i32],
) {
    let w_half = ind_x.len();
    if w_half < 11 {
        dwt2_horizontal_scalar(tmplo, tmphi, ind_x, i, w, dst_stride, dst, a_out);
        return;
    }
    let a_row = &mut a_out[i * dst_stride..i * dst_stride + w_half];
    dwt2_horizontal_at(tmplo, tmphi, ind_x[0], i, 0, dst_stride, dst, a_row);
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if let Some(token) = v3_token() {
        let mut j = 1;
        // Every output below w_half - 2 uses consecutive (non-reflected)
        // indices, so the 16-output madd kernel applies.
        while j + 16 <= w_half - 2 {
            dwt2_horizontal_16_v3(token, tmplo, tmphi, i, j, dst_stride, dst, a_row);
            j += 16;
        }
        while j < w_half {
            dwt2_horizontal_at(tmplo, tmphi, ind_x[j], i, j, dst_stride, dst, a_row);
            j += 1;
        }
        return;
    }
    let mut j = 1;
    while j + 8 <= w_half - 2 {
        let mut out = [[0i16; 8]; 4];
        archmage::incant!(
            dwt2_horizontal_simd(tmplo, tmphi, 2 * j - 1, &mut out),
            [v3, neon, wasm128, scalar]
        );
        let row = i * dst_stride + j;
        let mut a8 = [0i32; 8];
        for k in 0..8 {
            a8[k] = out[0][k] as i32;
        }
        a_row[j..j + 8].copy_from_slice(&a8);
        dst.v[row..row + 8].copy_from_slice(&out[1]);
        dst.h[row..row + 8].copy_from_slice(&out[2]);
        dst.d[row..row + 8].copy_from_slice(&out[3]);
        j += 8;
    }
    while j < w_half {
        dwt2_horizontal_at(tmplo, tmphi, ind_x[j], i, j, dst_stride, dst, a_row);
        j += 1;
    }
}

#[cfg(feature = "simd")]
#[magetypes(define(i16x16, i32x8), v3, neon, wasm128, scalar)]
fn dwt2_horizontal_simd(
    token: Token,
    tmplo: &[i16],
    tmphi: &[i16],
    base: usize,
    out: &mut [[i16; 8]; 4],
) {
    let mut acc_a_lo = i32x8::zero(token);
    let mut acc_a_hi = i32x8::zero(token);
    let mut acc_v_lo = i32x8::zero(token);
    let mut acc_v_hi = i32x8::zero(token);
    let mut acc_h_lo = i32x8::zero(token);
    let mut acc_h_hi = i32x8::zero(token);
    let mut acc_d_lo = i32x8::zero(token);
    let mut acc_d_hi = i32x8::zero(token);
    for t in 0..4 {
        let wl = i32x8::splat(token, DWT2_LO[t] as i32);
        let wh = i32x8::splat(token, DWT2_HI[t] as i32);
        let lt = i16x16::load(token, tmplo[base + t..base + t + 16].try_into().unwrap());
        let ht = i16x16::load(token, tmphi[base + t..base + t + 16].try_into().unwrap());
        let ltl = lt.widen_low();
        let lth = lt.widen_high();
        let htl = ht.widen_low();
        let hth = ht.widen_high();
        acc_a_lo += wl * ltl;
        acc_a_hi += wl * lth;
        acc_v_lo += wh * ltl;
        acc_v_hi += wh * lth;
        acc_h_lo += wl * htl;
        acc_h_hi += wl * hth;
        acc_d_lo += wh * htl;
        acc_d_hi += wh * hth;
    }
    let add = i32x8::splat(token, 32768);
    let a_lo = (acc_a_lo + add).shr_arithmetic_uniform(16).to_array();
    let a_hi = (acc_a_hi + add).shr_arithmetic_uniform(16).to_array();
    let v_lo = (acc_v_lo + add).shr_arithmetic_uniform(16).to_array();
    let v_hi = (acc_v_hi + add).shr_arithmetic_uniform(16).to_array();
    let h_lo = (acc_h_lo + add).shr_arithmetic_uniform(16).to_array();
    let h_hi = (acc_h_hi + add).shr_arithmetic_uniform(16).to_array();
    let d_lo = (acc_d_lo + add).shr_arithmetic_uniform(16).to_array();
    let d_hi = (acc_d_hi + add).shr_arithmetic_uniform(16).to_array();
    for (band, lane) in [a_lo, v_lo, h_lo, d_lo].into_iter().enumerate() {
        for k in 0..4 {
            out[band][k] = lane[2 * k] as i16;
        }
    }
    for (band, lane) in [a_hi, v_hi, h_hi, d_hi].into_iter().enumerate() {
        for k in 0..4 {
            out[band][4 + k] = lane[2 * k] as i16;
        }
    }
}

/// Direct port of `adm_dwt2_8_avx2`'s vertical pass, adapted to u16 loads
/// (libvmaf's `_16` variant keeps the vertical pass scalar; our source is u16
/// for all depths). unpack_epi16 pairs (s0,s1)/(s2,s3) so madd_epi16 computes
/// the 4-tap convolution in i32 lanes; `srl`+blend_epi16(0xAA)+packus_epi32
/// reproduces the scalar `(x + add) >> shift` then-truncate-to-i16 exactly.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane(import_intrinsics)]
#[allow(clippy::too_many_arguments)]
fn dwt2_vertical_v3(
    _token: X64V3Token,
    src: &[u16],
    src_stride: usize,
    w: usize,
    bit_depth: u8,
    indices: [i32; 4],
    tmplo: &mut [i16],
    tmphi: &mut [i16],
) {
    let shift_vp = bit_depth as i32;
    let add_shift_vp = 1i32 << (bit_depth - 1);
    let lo_sum_const = _mm256_set1_epi32(DWT2_LO_SUM * add_shift_vp);
    let hi_sum_const = _mm256_set1_epi32(DWT2_HI_SUM * add_shift_vp);
    let fl0 = _mm256_set1_epi32(((DWT2_LO[1] as i32) << 16) | (DWT2_LO[0] as u16 as i32));
    let fl1 = _mm256_set1_epi32(((DWT2_LO[3] as i32) << 16) | (DWT2_LO[2] as u16 as i32));
    let fh0 = _mm256_set1_epi32(((DWT2_HI[1] as i32) << 16) | (DWT2_HI[0] as u16 as i32));
    let fh1 = _mm256_set1_epi32(((DWT2_HI[3] as i32) << 16) | (DWT2_HI[2] as u16 as i32));
    let add_vp = _mm256_set1_epi32(add_shift_vp);
    let pad0 = _mm256_setzero_si256();
    let shift_v = _mm_cvtsi32_si128(shift_vp);
    let r = [
        indices[0] as usize * src_stride,
        indices[1] as usize * src_stride,
        indices[2] as usize * src_stride,
        indices[3] as usize * src_stride,
    ];
    let mut j = 0usize;
    while j + 16 <= w {
        let s0 = _mm256_loadu_si256(a8::<u16, 16>(&src[r[0] + j..r[0] + j + 16]));
        let s1 = _mm256_loadu_si256(a8::<u16, 16>(&src[r[1] + j..r[1] + j + 16]));
        let s2 = _mm256_loadu_si256(a8::<u16, 16>(&src[r[2] + j..r[2] + j + 16]));
        let s3 = _mm256_loadu_si256(a8::<u16, 16>(&src[r[3] + j..r[3] + j + 16]));

        let s0lo = _mm256_unpacklo_epi16(s0, s1);
        let s0hi = _mm256_unpackhi_epi16(s0, s1);
        let s1lo = _mm256_unpacklo_epi16(s2, s3);
        let s1hi = _mm256_unpackhi_epi16(s2, s3);

        let mut acc_lo =
            _mm256_add_epi32(_mm256_madd_epi16(s0lo, fl0), _mm256_madd_epi16(s1lo, fl1));
        let mut acc_hi =
            _mm256_add_epi32(_mm256_madd_epi16(s0hi, fl0), _mm256_madd_epi16(s1hi, fl1));
        acc_lo = _mm256_sub_epi32(acc_lo, lo_sum_const);
        acc_hi = _mm256_sub_epi32(acc_hi, lo_sum_const);
        acc_lo = _mm256_srl_epi32(_mm256_add_epi32(acc_lo, add_vp), shift_v);
        acc_hi = _mm256_srl_epi32(_mm256_add_epi32(acc_hi, add_vp), shift_v);
        acc_lo = _mm256_blend_epi16(acc_lo, pad0, 0xAA);
        acc_hi = _mm256_blend_epi16(acc_hi, pad0, 0xAA);
        _mm256_storeu_si256(
            a8m::<i16, 16>(&mut tmplo[j..j + 16]),
            _mm256_packus_epi32(acc_lo, acc_hi),
        );

        let mut acc_lo =
            _mm256_add_epi32(_mm256_madd_epi16(s0lo, fh0), _mm256_madd_epi16(s1lo, fh1));
        let mut acc_hi =
            _mm256_add_epi32(_mm256_madd_epi16(s0hi, fh0), _mm256_madd_epi16(s1hi, fh1));
        acc_lo = _mm256_sub_epi32(acc_lo, hi_sum_const);
        acc_hi = _mm256_sub_epi32(acc_hi, hi_sum_const);
        acc_lo = _mm256_srl_epi32(_mm256_add_epi32(acc_lo, add_vp), shift_v);
        acc_hi = _mm256_srl_epi32(_mm256_add_epi32(acc_hi, add_vp), shift_v);
        acc_lo = _mm256_blend_epi16(acc_lo, pad0, 0xAA);
        acc_hi = _mm256_blend_epi16(acc_hi, pad0, 0xAA);
        _mm256_storeu_si256(
            a8m::<i16, 16>(&mut tmphi[j..j + 16]),
            _mm256_packus_epi32(acc_lo, acc_hi),
        );
        j += 16;
    }
    while j < w {
        let s = [
            src[r[0] + j] as i32,
            src[r[1] + j] as i32,
            src[r[2] + j] as i32,
            src[r[3] + j] as i32,
        ];
        let mut accum = 0i32;
        for k in 0..4 {
            accum += DWT2_LO[k] as i32 * s[k];
        }
        accum -= DWT2_LO_SUM * add_shift_vp;
        tmplo[j] = ((accum + add_shift_vp) >> shift_vp) as i16;
        let mut accum = 0i32;
        for k in 0..4 {
            accum += DWT2_HI[k] as i32 * s[k];
        }
        accum -= DWT2_HI_SUM * add_shift_vp;
        tmphi[j] = ((accum + add_shift_vp) >> shift_vp) as i16;
        j += 1;
    }
}

/// Direct port of `adm_dwt2_16_avx2`'s horizontal pass: 16 outputs per
/// iteration, four interleaved 16-wide loads at offsets 2j-1/2j+1/2j+15/2j+17
/// feeding madd_epi16 pair-products. Callers must guarantee every output in
/// [j, j+16) uses non-reflected indices (j + 16 <= w_half - 2).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane(import_intrinsics)]
#[allow(clippy::too_many_arguments)]
fn dwt2_horizontal_16_v3(
    _token: X64V3Token,
    tmplo: &[i16],
    tmphi: &[i16],
    i: usize,
    j: usize,
    dst_stride: usize,
    dst: &mut BandI16,
    a_row: &mut [i32],
) {
    let f01_lo = _mm256_set1_epi32(((DWT2_LO[1] as i32) << 16) | (DWT2_LO[0] as u16 as i32));
    let f23_lo = _mm256_set1_epi32(((DWT2_LO[3] as i32) << 16) | (DWT2_LO[2] as u16 as i32));
    let f01_hi = _mm256_set1_epi32(((DWT2_HI[1] as i32) << 16) | (DWT2_HI[0] as u16 as i32));
    let f23_hi = _mm256_set1_epi32(((DWT2_HI[3] as i32) << 16) | (DWT2_HI[2] as u16 as i32));
    let add_hp = _mm256_set1_epi32(32768);
    let (j0, j2, j16, j18) = (2 * j - 1, 2 * j + 1, 2 * j + 15, 2 * j + 17);

    let s0 = _mm256_loadu_si256(a8::<i16, 16>(&tmplo[j0..j0 + 16]));
    let s2 = _mm256_loadu_si256(a8::<i16, 16>(&tmplo[j2..j2 + 16]));
    let s0_32 = _mm256_loadu_si256(a8::<i16, 16>(&tmplo[j16..j16 + 16]));
    let s2_32 = _mm256_loadu_si256(a8::<i16, 16>(&tmplo[j18..j18 + 16]));

    let mut acc_lo = _mm256_add_epi32(_mm256_madd_epi16(s0, f01_lo), _mm256_madd_epi16(s2, f23_lo));
    let mut acc_hi = _mm256_add_epi32(
        _mm256_madd_epi16(s0_32, f01_lo),
        _mm256_madd_epi16(s2_32, f23_lo),
    );
    acc_lo = _mm256_srai_epi32(_mm256_add_epi32(acc_lo, add_hp), 16);
    acc_hi = _mm256_srai_epi32(_mm256_add_epi32(acc_hi, add_hp), 16);
    let packed = _mm256_permute4x64_epi64(_mm256_packs_epi32(acc_lo, acc_hi), 0xD8);
    let mut a16 = [0i16; 16];
    _mm256_storeu_si256(&mut a16, packed);
    for k in 0..16 {
        a_row[j + k] = a16[k] as i32;
    }

    let mut acc_lo = _mm256_add_epi32(_mm256_madd_epi16(s0, f01_hi), _mm256_madd_epi16(s2, f23_hi));
    let mut acc_hi = _mm256_add_epi32(
        _mm256_madd_epi16(s0_32, f01_hi),
        _mm256_madd_epi16(s2_32, f23_hi),
    );
    acc_lo = _mm256_srai_epi32(_mm256_add_epi32(acc_lo, add_hp), 16);
    acc_hi = _mm256_srai_epi32(_mm256_add_epi32(acc_hi, add_hp), 16);
    let row = i * dst_stride + j;
    _mm256_storeu_si256(
        a8m::<i16, 16>(&mut dst.v[row..row + 16]),
        _mm256_permute4x64_epi64(_mm256_packs_epi32(acc_lo, acc_hi), 0xD8),
    );

    let s0 = _mm256_loadu_si256(a8::<i16, 16>(&tmphi[j0..j0 + 16]));
    let s2 = _mm256_loadu_si256(a8::<i16, 16>(&tmphi[j2..j2 + 16]));
    let s0_32 = _mm256_loadu_si256(a8::<i16, 16>(&tmphi[j16..j16 + 16]));
    let s2_32 = _mm256_loadu_si256(a8::<i16, 16>(&tmphi[j18..j18 + 16]));

    let mut acc_lo = _mm256_add_epi32(_mm256_madd_epi16(s0, f01_lo), _mm256_madd_epi16(s2, f23_lo));
    let mut acc_hi = _mm256_add_epi32(
        _mm256_madd_epi16(s0_32, f01_lo),
        _mm256_madd_epi16(s2_32, f23_lo),
    );
    acc_lo = _mm256_srai_epi32(_mm256_add_epi32(acc_lo, add_hp), 16);
    acc_hi = _mm256_srai_epi32(_mm256_add_epi32(acc_hi, add_hp), 16);
    _mm256_storeu_si256(
        a8m::<i16, 16>(&mut dst.h[row..row + 16]),
        _mm256_permute4x64_epi64(_mm256_packs_epi32(acc_lo, acc_hi), 0xD8),
    );

    let mut acc_lo = _mm256_add_epi32(_mm256_madd_epi16(s0, f01_hi), _mm256_madd_epi16(s2, f23_hi));
    let mut acc_hi = _mm256_add_epi32(
        _mm256_madd_epi16(s0_32, f01_hi),
        _mm256_madd_epi16(s2_32, f23_hi),
    );
    acc_lo = _mm256_srai_epi32(_mm256_add_epi32(acc_lo, add_hp), 16);
    acc_hi = _mm256_srai_epi32(_mm256_add_epi32(acc_hi, add_hp), 16);
    _mm256_storeu_si256(
        a8m::<i16, 16>(&mut dst.d[row..row + 16]),
        _mm256_permute4x64_epi64(_mm256_packs_epi32(acc_lo, acc_hi), 0xD8),
    );
}

fn adm_dwt2_core(
    src: &[u16],
    src_stride: usize,
    dst: &mut BandI16,
    w: usize,
    dst_stride: usize,
    bit_depth: u8,
    ind_y: &[[i32; 4]],
    ind_x: &[[i32; 4]],
    a_out: &mut [i32],
    vertical: impl Fn(&[u16], usize, usize, u8, [i32; 4], &mut [i16], &mut [i16]),
    horizontal: impl Fn(&[i16], &[i16], &[[i32; 4]], usize, usize, usize, &mut BandI16, &mut [i32]),
) {
    let mut tmplo = vec![0i16; w];
    let mut tmphi = vec![0i16; w];
    for (i, iy) in ind_y.iter().enumerate() {
        vertical(src, src_stride, w, bit_depth, *iy, &mut tmplo, &mut tmphi);
        horizontal(&tmplo, &tmphi, ind_x, i, w, dst_stride, dst, a_out);
    }
}

fn border_region(w: usize, h: usize, extra_tap: i32) -> (i32, i32, i32, i32) {
    let mut left = (w as f64 * ADM_BORDER_FACTOR - 0.5) as i32 - extra_tap;
    let mut top = (h as f64 * ADM_BORDER_FACTOR - 0.5) as i32 - extra_tap;
    let mut right = w as i32 - left + 2 * extra_tap;
    let mut bottom = h as i32 - top + 2 * extra_tap;
    if left < 0 {
        left = 0;
    }
    if right > w as i32 {
        right = w as i32;
    }
    if top < 0 {
        top = 0;
    }
    if bottom > h as i32 {
        bottom = h as i32;
    }
    (left, top, right, bottom)
}

fn border_region_inner(w: usize, h: usize) -> (i32, i32, i32, i32) {
    let left = (w as f64 * ADM_BORDER_FACTOR - 0.5) as i32;
    let top = (h as f64 * ADM_BORDER_FACTOR - 0.5) as i32;
    (left, top, w as i32 - left, h as i32 - top)
}

#[cfg_attr(feature = "simd", autoversion)]
fn adm_decouple(
    ref_b: &BandI16,
    dis_b: &BandI16,
    r: &mut BandI16,
    a: &mut BandI16,
    w: usize,
    h: usize,
    stride: usize,
    div: &[i32; 65537],
    enhn_gain_limit: f64,
) {
    let cos_1deg_sq = (std::f64::consts::PI / 180.0).cos().powi(2);
    let (left, top, right, bottom) = border_region(w, h, 1);
    for i in top..bottom {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        let j_start = if let Some(token) = v3_token() {
            adm_decouple_row_v3(
                token,
                &ref_b.h,
                &ref_b.v,
                &ref_b.d,
                &dis_b.h,
                &dis_b.v,
                &dis_b.d,
                &mut r.h,
                &mut r.v,
                &mut r.d,
                &mut a.h,
                &mut a.v,
                &mut a.d,
                i as usize,
                stride,
                left as usize,
                right as usize,
                div,
                enhn_gain_limit,
            );
            left + ((right - left) / 8) * 8
        } else {
            left
        };
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        let j_start = left;
        for j in j_start..right {
            let idx = i as usize * stride + j as usize;
            let oh = ref_b.h[idx] as i64;
            let ov = ref_b.v[idx] as i64;
            let od = ref_b.d[idx] as i64;
            let th = dis_b.h[idx] as i64;
            let tv = dis_b.v[idx] as i64;
            let td = dis_b.d[idx] as i64;
            let ot_dp = oh * th + ov * tv;
            let o_mag_sq = oh * oh + ov * ov;
            let t_mag_sq = th * th + tv * tv;
            let dp_f = ot_dp as f32 as f64 / 4096.0;
            let angle_flag = dp_f >= 0.0
                && (dp_f * dp_f
                    >= cos_1deg_sq
                        * (o_mag_sq as f32 as f64 / 4096.0)
                        * (t_mag_sq as f32 as f64 / 4096.0));

            let tmp_kh = if oh == 0 {
                32768i64
            } else {
                ((div[(oh + 32768) as usize] as i64 * th) + 16384) >> 15
            };
            let tmp_kv = if ov == 0 {
                32768i64
            } else {
                ((div[(ov + 32768) as usize] as i64 * tv) + 16384) >> 15
            };
            let tmp_kd = if od == 0 {
                32768i64
            } else {
                ((div[(od + 32768) as usize] as i64 * td) + 16384) >> 15
            };
            let kh = tmp_kh.clamp(0, 32768);
            let kv = tmp_kv.clamp(0, 32768);
            let kd = tmp_kd.clamp(0, 32768);

            let mut rst_h = (((kh * oh) + 16384) >> 15) as i16;
            let mut rst_v = (((kv * ov) + 16384) >> 15) as i16;
            let mut rst_d = (((kd * od) + 16384) >> 15) as i16;

            let rst_h_f = (kh as f32 / 32768.0) * (oh as f32 / 64.0);
            let rst_v_f = (kv as f32 / 32768.0) * (ov as f32 / 64.0);
            let rst_d_f = (kd as f32 / 32768.0) * (od as f32 / 64.0);

            if angle_flag && rst_h_f > 0.0 {
                rst_h = ((rst_h as f64 * enhn_gain_limit).min(th as f64)) as i16;
            }
            if angle_flag && rst_h_f < 0.0 {
                rst_h = ((rst_h as f64 * enhn_gain_limit).max(th as f64)) as i16;
            }
            if angle_flag && rst_v_f > 0.0 {
                rst_v = ((rst_v as f64 * enhn_gain_limit).min(tv as f64)) as i16;
            }
            if angle_flag && rst_v_f < 0.0 {
                rst_v = ((rst_v as f64 * enhn_gain_limit).max(tv as f64)) as i16;
            }
            if angle_flag && rst_d_f > 0.0 {
                rst_d = ((rst_d as f64 * enhn_gain_limit).min(td as f64)) as i16;
            }
            if angle_flag && rst_d_f < 0.0 {
                rst_d = ((rst_d as f64 * enhn_gain_limit).max(td as f64)) as i16;
            }

            r.h[idx] = rst_h;
            r.v[idx] = rst_v;
            r.d[idx] = rst_d;
            a.h[idx] = (th - rst_h as i64) as i16;
            a.v[idx] = (tv - rst_v as i64) as i16;
            a.d[idx] = (td - rst_d as i64) as i16;
        }
    }
}

/// Direct port of `shift15_64b_signExt_256`: logical >>15 plus the masked
/// high bits of the original — arithmetic >>15 on i64 lanes given products
/// stay below 2^49.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[rite]
fn shift15_64b_v3(_token: X64V3Token, a: __m256i) -> __m256i {
    _mm256_add_epi64(
        _mm256_srli_epi64(a, 15),
        _mm256_and_si256(a, _mm256_set1_epi64x(0xFFFE000000000000u64 as i64)),
    )
}

/// Safe lane-gather on an i32 table (see cambi's gather_u16_v3).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[rite]
fn gather_i32_v3(_token: X64V3Token, table: &[i32], idx: __m256i) -> __m256i {
    let mut a = [0i32; 8];
    _mm256_storeu_si256(a8m::<i32, 8>(&mut a), idx);
    _mm256_setr_epi32(
        table[a[0] as usize],
        table[a[1] as usize],
        table[a[2] as usize],
        table[a[3] as usize],
        table[a[4] as usize],
        table[a[5] as usize],
        table[a[6] as usize],
        table[a[7] as usize],
    )
}

/// Direct port of `adm_decouple_avx2`'s inner kernel for one row range.
/// Vector loop is bounded by `j + 8 <= right` (C iterates to `right_mod8`
/// which can overshoot `right` into border columns that are never read).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane(import_intrinsics)]
#[allow(clippy::too_many_arguments)]
fn adm_decouple_row_v3(
    _token: X64V3Token,
    ref_h: &[i16],
    ref_v: &[i16],
    ref_d: &[i16],
    dis_h: &[i16],
    dis_v: &[i16],
    dis_d: &[i16],
    r_h: &mut [i16],
    r_v: &mut [i16],
    r_d: &mut [i16],
    a_h: &mut [i16],
    a_v: &mut [i16],
    a_d: &mut [i16],
    row: usize,
    stride: usize,
    j_start: usize,
    j_end: usize,
    div: &[i32; 65537],
    enhn_gain_limit: f64,
) {
    let cos_1deg_sq = (std::f64::consts::PI / 180.0).cos().powi(2);
    let base = row * stride;
    let const_32768 = _mm256_set1_epi32(32768);
    let const_16384_64 = _mm256_set1_epi64x(16384);
    let const_16384_32 = _mm256_set1_epi32(16384);
    let lo16 = _mm256_set1_epi32(0xFFFF);
    let lo32_64 = _mm256_set1_epi64x(0xFFFFFFFF);
    let inv_32768 = _mm256_set1_ps(1.0 / 32768.0);
    let inv_64 = _mm256_set1_ps(1.0 / 64.0);
    let gain_d = _mm256_set1_pd(enhn_gain_limit);
    let zero = _mm256_setzero_si256();
    let zero_ps = _mm256_setzero_ps();

    let mut j = j_start;
    while j + 8 <= j_end {
        let idx = base + j;
        let oh = _mm256_cvtepi16_epi32(_mm_loadu_si128(a8::<i16, 8>(&ref_h[idx..idx + 8])));
        let ov = _mm256_cvtepi16_epi32(_mm_loadu_si128(a8::<i16, 8>(&ref_v[idx..idx + 8])));
        let od = _mm256_cvtepi16_epi32(_mm_loadu_si128(a8::<i16, 8>(&ref_d[idx..idx + 8])));
        let th = _mm256_cvtepi16_epi32(_mm_loadu_si128(a8::<i16, 8>(&dis_h[idx..idx + 8])));
        let tv = _mm256_cvtepi16_epi32(_mm_loadu_si128(a8::<i16, 8>(&dis_v[idx..idx + 8])));
        let td = _mm256_cvtepi16_epi32(_mm_loadu_si128(a8::<i16, 8>(&dis_d[idx..idx + 8])));

        let oh_ov = _mm256_or_si256(_mm256_and_si256(oh, lo16), _mm256_slli_epi32(ov, 16));
        let th_tv = _mm256_or_si256(_mm256_and_si256(th, lo16), _mm256_slli_epi32(tv, 16));

        let o_mag_sq = _mm256_madd_epi16(oh_ov, oh_ov);
        let ot_dp = _mm256_madd_epi16(oh_ov, th_tv);
        let t_mag_sq = _mm256_madd_epi16(th_tv, th_tv);

        let mut dp_arr = [0i32; 8];
        let mut oms_arr = [0i32; 8];
        let mut tms_arr = [0i32; 8];
        _mm256_storeu_si256(a8m::<i32, 8>(&mut dp_arr), ot_dp);
        _mm256_storeu_si256(a8m::<i32, 8>(&mut oms_arr), o_mag_sq);
        _mm256_storeu_si256(a8m::<i32, 8>(&mut tms_arr), t_mag_sq);
        let mut angle_flag = [0i32; 8];
        for lane in 0..8 {
            let dp_f = dp_arr[lane] as f32 as f64 / 4096.0;
            angle_flag[lane] = (dp_f >= 0.0
                && dp_f * dp_f
                    >= cos_1deg_sq
                        * (oms_arr[lane] as f32 as f64 / 4096.0)
                        * (tms_arr[lane] as f32 as f64 / 4096.0))
                as i32;
        }
        let angle_mask = _mm256_mullo_epi32(
            _mm256_setr_epi32(
                angle_flag[0],
                angle_flag[1],
                angle_flag[2],
                angle_flag[3],
                angle_flag[4],
                angle_flag[5],
                angle_flag[6],
                angle_flag[7],
            ),
            _mm256_set1_epi32(-1),
        );

        let oh_div = gather_i32_v3(_token, &div[..], _mm256_add_epi32(oh, const_32768));
        let ov_div = gather_i32_v3(_token, &div[..], _mm256_add_epi32(ov, const_32768));
        let od_div = gather_i32_v3(_token, &div[..], _mm256_add_epi32(od, const_32768));

        let mut kh_lo = _mm256_mul_epi32(oh_div, th);
        let mut kh_hi = _mm256_mul_epi32(_mm256_srli_epi64(oh_div, 32), _mm256_srli_epi64(th, 32));
        let mut kv_lo = _mm256_mul_epi32(ov_div, tv);
        let mut kv_hi = _mm256_mul_epi32(_mm256_srli_epi64(ov_div, 32), _mm256_srli_epi64(tv, 32));
        let mut kd_lo = _mm256_mul_epi32(od_div, td);
        let mut kd_hi = _mm256_mul_epi32(_mm256_srli_epi64(od_div, 32), _mm256_srli_epi64(td, 32));

        kh_lo = shift15_64b_v3(_token, _mm256_add_epi64(kh_lo, const_16384_64));
        kh_hi = shift15_64b_v3(_token, _mm256_add_epi64(kh_hi, const_16384_64));
        kv_lo = shift15_64b_v3(_token, _mm256_add_epi64(kv_lo, const_16384_64));
        kv_hi = shift15_64b_v3(_token, _mm256_add_epi64(kv_hi, const_16384_64));
        kd_lo = shift15_64b_v3(_token, _mm256_add_epi64(kd_lo, const_16384_64));
        kd_hi = shift15_64b_v3(_token, _mm256_add_epi64(kd_hi, const_16384_64));

        let mut tmp_kh = _mm256_or_si256(
            _mm256_and_si256(kh_lo, lo32_64),
            _mm256_slli_epi64(kh_hi, 32),
        );
        let mut tmp_kv = _mm256_or_si256(
            _mm256_and_si256(kv_lo, lo32_64),
            _mm256_slli_epi64(kv_hi, 32),
        );
        let mut tmp_kd = _mm256_or_si256(
            _mm256_and_si256(kd_lo, lo32_64),
            _mm256_slli_epi64(kd_hi, 32),
        );

        let eqz_oh = _mm256_cmpeq_epi32(oh, zero);
        let eqz_ov = _mm256_cmpeq_epi32(ov, zero);
        let eqz_od = _mm256_cmpeq_epi32(od, zero);
        tmp_kh = _mm256_or_si256(
            _mm256_andnot_si256(eqz_oh, tmp_kh),
            _mm256_and_si256(const_32768, eqz_oh),
        );
        tmp_kv = _mm256_or_si256(
            _mm256_andnot_si256(eqz_ov, tmp_kv),
            _mm256_and_si256(const_32768, eqz_ov),
        );
        tmp_kd = _mm256_or_si256(
            _mm256_andnot_si256(eqz_od, tmp_kd),
            _mm256_and_si256(const_32768, eqz_od),
        );

        tmp_kh = _mm256_min_epi32(_mm256_max_epi32(tmp_kh, zero), const_32768);
        tmp_kv = _mm256_min_epi32(_mm256_max_epi32(tmp_kv, zero), const_32768);
        tmp_kd = _mm256_min_epi32(_mm256_max_epi32(tmp_kd, zero), const_32768);

        let mut rst_h = _mm256_srai_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(tmp_kh, oh), const_16384_32),
            15,
        );
        let mut rst_v = _mm256_srai_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(tmp_kv, ov), const_16384_32),
            15,
        );
        let mut rst_d = _mm256_srai_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(tmp_kd, od), const_16384_32),
            15,
        );

        let rst_h_f = _mm256_mul_ps(
            _mm256_mul_ps(inv_32768, _mm256_cvtepi32_ps(tmp_kh)),
            _mm256_mul_ps(inv_64, _mm256_cvtepi32_ps(oh)),
        );
        let rst_v_f = _mm256_mul_ps(
            _mm256_mul_ps(inv_32768, _mm256_cvtepi32_ps(tmp_kv)),
            _mm256_mul_ps(inv_64, _mm256_cvtepi32_ps(ov)),
        );
        let rst_d_f = _mm256_mul_ps(
            _mm256_mul_ps(inv_32768, _mm256_cvtepi32_ps(tmp_kd)),
            _mm256_mul_ps(inv_64, _mm256_cvtepi32_ps(od)),
        );

        let gt0_h = _mm256_castps_si256(_mm256_cmp_ps::<14>(rst_h_f, zero_ps));
        let lt0_h = _mm256_castps_si256(_mm256_cmp_ps::<1>(rst_h_f, zero_ps));
        let gt0_v = _mm256_castps_si256(_mm256_cmp_ps::<14>(rst_v_f, zero_ps));
        let lt0_v = _mm256_castps_si256(_mm256_cmp_ps::<1>(rst_v_f, zero_ps));
        let gt0_d = _mm256_castps_si256(_mm256_cmp_ps::<14>(rst_d_f, zero_ps));
        let lt0_d = _mm256_castps_si256(_mm256_cmp_ps::<1>(rst_d_f, zero_ps));

        let mask_h = _mm256_and_si256(_mm256_or_si256(gt0_h, lt0_h), angle_mask);
        let mask_v = _mm256_and_si256(_mm256_or_si256(gt0_v, lt0_v), angle_mask);
        let mask_d = _mm256_and_si256(_mm256_or_si256(gt0_d, lt0_d), angle_mask);

        let gh_lo = _mm256_mul_pd(
            _mm256_cvtepi32_pd(_mm256_extractf128_si256::<0>(rst_h)),
            gain_d,
        );
        let gh_hi = _mm256_mul_pd(
            _mm256_cvtepi32_pd(_mm256_extractf128_si256::<1>(rst_h)),
            gain_d,
        );
        let rst_h_gain = _mm256_insertf128_si256::<1>(
            _mm256_castsi128_si256(_mm256_cvtpd_epi32(gh_lo)),
            _mm256_cvtpd_epi32(gh_hi),
        );
        let gv_lo = _mm256_mul_pd(
            _mm256_cvtepi32_pd(_mm256_extractf128_si256::<0>(rst_v)),
            gain_d,
        );
        let gv_hi = _mm256_mul_pd(
            _mm256_cvtepi32_pd(_mm256_extractf128_si256::<1>(rst_v)),
            gain_d,
        );
        let rst_v_gain = _mm256_insertf128_si256::<1>(
            _mm256_castsi128_si256(_mm256_cvtpd_epi32(gv_lo)),
            _mm256_cvtpd_epi32(gv_hi),
        );
        let gd_lo = _mm256_mul_pd(
            _mm256_cvtepi32_pd(_mm256_extractf128_si256::<0>(rst_d)),
            gain_d,
        );
        let gd_hi = _mm256_mul_pd(
            _mm256_cvtepi32_pd(_mm256_extractf128_si256::<1>(rst_d)),
            gain_d,
        );
        let rst_d_gain = _mm256_insertf128_si256::<1>(
            _mm256_castsi128_si256(_mm256_cvtpd_epi32(gd_lo)),
            _mm256_cvtpd_epi32(gd_hi),
        );

        let h_sel = _mm256_or_si256(
            _mm256_and_si256(_mm256_min_epi32(rst_h_gain, th), gt0_h),
            _mm256_and_si256(_mm256_max_epi32(rst_h_gain, th), lt0_h),
        );
        let v_sel = _mm256_or_si256(
            _mm256_and_si256(_mm256_min_epi32(rst_v_gain, tv), gt0_v),
            _mm256_and_si256(_mm256_max_epi32(rst_v_gain, tv), lt0_v),
        );
        let d_sel = _mm256_or_si256(
            _mm256_and_si256(_mm256_min_epi32(rst_d_gain, td), gt0_d),
            _mm256_and_si256(_mm256_max_epi32(rst_d_gain, td), lt0_d),
        );

        rst_h = _mm256_or_si256(
            _mm256_and_si256(h_sel, mask_h),
            _mm256_andnot_si256(mask_h, rst_h),
        );
        rst_v = _mm256_or_si256(
            _mm256_and_si256(v_sel, mask_v),
            _mm256_andnot_si256(mask_v, rst_v),
        );
        rst_d = _mm256_or_si256(
            _mm256_and_si256(d_sel, mask_d),
            _mm256_andnot_si256(mask_d, rst_d),
        );

        let ah = _mm256_sub_epi32(th, rst_h);
        let av = _mm256_sub_epi32(tv, rst_v);
        let ad = _mm256_sub_epi32(td, rst_d);

        // packs_epi32(v, permute4x64(v, 0x0E)) yields the 8 i16 lanes in order
        // in the low 128 bits.
        let ph = _mm256_packs_epi32(rst_h, _mm256_permute4x64_epi64(rst_h, 0x0E));
        let pv = _mm256_packs_epi32(rst_v, _mm256_permute4x64_epi64(rst_v, 0x0E));
        let pd = _mm256_packs_epi32(rst_d, _mm256_permute4x64_epi64(rst_d, 0x0E));
        let pah = _mm256_packs_epi32(ah, _mm256_permute4x64_epi64(ah, 0x0E));
        let pav = _mm256_packs_epi32(av, _mm256_permute4x64_epi64(av, 0x0E));
        let pad = _mm256_packs_epi32(ad, _mm256_permute4x64_epi64(ad, 0x0E));

        _mm_storeu_si128(
            a8m::<i16, 8>(&mut r_h[idx..idx + 8]),
            _mm256_castsi256_si128(ph),
        );
        _mm_storeu_si128(
            a8m::<i16, 8>(&mut r_v[idx..idx + 8]),
            _mm256_castsi256_si128(pv),
        );
        _mm_storeu_si128(
            a8m::<i16, 8>(&mut r_d[idx..idx + 8]),
            _mm256_castsi256_si128(pd),
        );
        _mm_storeu_si128(
            a8m::<i16, 8>(&mut a_h[idx..idx + 8]),
            _mm256_castsi256_si128(pah),
        );
        _mm_storeu_si128(
            a8m::<i16, 8>(&mut a_v[idx..idx + 8]),
            _mm256_castsi256_si128(pav),
        );
        _mm_storeu_si128(
            a8m::<i16, 8>(&mut a_d[idx..idx + 8]),
            _mm256_castsi256_si128(pad),
        );
        j += 8;
    }
}

#[inline(always)]
fn get_best15_from32(temp: u32) -> (u16, i32) {
    let k = 17 - temp.leading_zeros() as i32;
    let v = ((temp as u64 + (1u64 << (k - 1))) >> k) as u16;
    (v, k)
}

/// Direct port of `sra_epi64`: variable arithmetic >> on i64 lanes —
/// logical srlv plus the sign mask shifted into place.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[rite]
fn sra_epi64_v3(_token: X64V3Token, a: __m256i, mask: __m256i) -> __m256i {
    let rl_shift = _mm256_srlv_epi64(a, mask);
    let signmask = _mm256_cmpgt_epi64(_mm256_setzero_si256(), a);
    let newmask = _mm256_sub_epi64(_mm256_set1_epi64x(64), mask);
    _mm256_or_si256(rl_shift, _mm256_sllv_epi64(signmask, newmask))
}

/// Direct port of `blend(a, b, mask)`: select a where mask is set.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[rite]
fn blend_v3(_token: X64V3Token, a: __m256i, b: __m256i, mask: __m256i) -> __m256i {
    _mm256_or_si256(_mm256_and_si256(mask, a), _mm256_andnot_si256(mask, b))
}

/// Narrow each i64 lane to its low 32 bits; C stores to an i64 array and
/// casts each element to int — identical to keeping the even dwords.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[rite]
fn pack_i64x4x2_i32_v3(_token: X64V3Token, lo: __m256i, hi: __m256i) -> __m256i {
    let idx = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
    let l = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(lo, idx));
    let h = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(hi, idx));
    _mm256_inserti128_si256(_mm256_castsi128_si256(l), h, 1)
}

/// Direct port of `adm_decouple_s123_avx2`'s inner kernel for one row range.
/// i64 lanes throughout: mul_epi32 on sign-widened inputs, sra_epi64 for the
/// shift-dependent division rounding, per-lane get_best15_from32 (no SIMD
/// lzcnt in AVX2 — C extracts to scalar the same way). Vector loop bounded by
/// `j + 8 <= j_end`; scalar tail identical to C's `right_mod8..right`.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane(import_intrinsics)]
#[allow(clippy::too_many_arguments)]
fn adm_decouple_s123_row_v3(
    _token: X64V3Token,
    ref_h: &[i32],
    ref_v: &[i32],
    ref_d: &[i32],
    dis_h: &[i32],
    dis_v: &[i32],
    dis_d: &[i32],
    r_h: &mut [i32],
    r_v: &mut [i32],
    r_d: &mut [i32],
    a_h: &mut [i32],
    a_v: &mut [i32],
    a_d: &mut [i32],
    row: usize,
    stride: usize,
    j_start: usize,
    j_end: usize,
    div: &[i32; 65537],
    enhn_gain_limit: f64,
) {
    let cos_1deg_sq = (std::f64::consts::PI / 180.0).cos().powi(2);
    let base = row * stride;
    let const_0_epi64 = _mm256_set1_epi64x(0);
    let const_0_pd = _mm256_set1_pd(0.0);
    let const_1_epi32 = _mm256_set1_epi32(1);
    let const_14_epi32 = _mm256_set1_epi32(14);
    let const_15_epi32 = _mm256_set1_epi32(15);
    let const_16384_epi64 = _mm256_set1_epi64x(16384);
    let const_32768_epi32 = _mm256_set1_epi32(32768);
    let const_32768_epi64 = _mm256_set1_epi64x(32768);
    let inv_32768_f = _mm256_set1_ps(1.0 / 32768.0);
    let inv_64_f = _mm256_set1_ps(1.0 / 64.0);
    let gain_d = _mm256_set1_pd(enhn_gain_limit);

    let mut j = j_start;
    while j + 8 <= j_end {
        let idx = base + j;
        let oh_epi32 = _mm256_loadu_si256(a8::<i32, 8>(&ref_h[idx..idx + 8]));
        let ov_epi32 = _mm256_loadu_si256(a8::<i32, 8>(&ref_v[idx..idx + 8]));
        let od_epi32 = _mm256_loadu_si256(a8::<i32, 8>(&ref_d[idx..idx + 8]));
        let th_epi32 = _mm256_loadu_si256(a8::<i32, 8>(&dis_h[idx..idx + 8]));
        let tv_epi32 = _mm256_loadu_si256(a8::<i32, 8>(&dis_v[idx..idx + 8]));
        let td_epi32 = _mm256_loadu_si256(a8::<i32, 8>(&dis_d[idx..idx + 8]));

        let oh_lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(oh_epi32));
        let oh_hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(oh_epi32, 1));
        let ov_lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(ov_epi32));
        let ov_hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(ov_epi32, 1));
        let od_lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(od_epi32));
        let od_hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(od_epi32, 1));
        let th_lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(th_epi32));
        let th_hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(th_epi32, 1));
        let tv_lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(tv_epi32));
        let tv_hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(tv_epi32, 1));
        let td_lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(td_epi32));
        let td_hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(td_epi32, 1));

        let dp_lo = _mm256_add_epi64(
            _mm256_mul_epi32(oh_lo, th_lo),
            _mm256_mul_epi32(ov_lo, tv_lo),
        );
        let dp_hi = _mm256_add_epi64(
            _mm256_mul_epi32(oh_hi, th_hi),
            _mm256_mul_epi32(ov_hi, tv_hi),
        );
        let oms_lo = _mm256_add_epi64(
            _mm256_mul_epi32(oh_lo, oh_lo),
            _mm256_mul_epi32(ov_lo, ov_lo),
        );
        let oms_hi = _mm256_add_epi64(
            _mm256_mul_epi32(oh_hi, oh_hi),
            _mm256_mul_epi32(ov_hi, ov_hi),
        );
        let tms_lo = _mm256_add_epi64(
            _mm256_mul_epi32(th_lo, th_lo),
            _mm256_mul_epi32(tv_lo, tv_lo),
        );
        let tms_hi = _mm256_add_epi64(
            _mm256_mul_epi32(th_hi, th_hi),
            _mm256_mul_epi32(tv_hi, tv_hi),
        );

        let mut dp_arr = [0i64; 8];
        let mut oms_arr = [0i64; 8];
        let mut tms_arr = [0i64; 8];
        _mm256_storeu_si256(a8m::<i64, 4>(&mut dp_arr[0..4]), dp_lo);
        _mm256_storeu_si256(a8m::<i64, 4>(&mut dp_arr[4..8]), dp_hi);
        _mm256_storeu_si256(a8m::<i64, 4>(&mut oms_arr[0..4]), oms_lo);
        _mm256_storeu_si256(a8m::<i64, 4>(&mut oms_arr[4..8]), oms_hi);
        _mm256_storeu_si256(a8m::<i64, 4>(&mut tms_arr[0..4]), tms_lo);
        _mm256_storeu_si256(a8m::<i64, 4>(&mut tms_arr[4..8]), tms_hi);
        let mut angle_flag = [0i64; 8];
        for lane in 0..8 {
            let dp_f = dp_arr[lane] as f32 as f64 / 4096.0;
            angle_flag[lane] = (dp_f >= 0.0
                && dp_f * dp_f
                    >= cos_1deg_sq
                        * (oms_arr[lane] as f32 as f64 / 4096.0)
                        * (tms_arr[lane] as f32 as f64 / 4096.0))
                as i64;
        }
        let angle_lo = _mm256_loadu_si256(a8::<i64, 4>(&angle_flag[0..4]));
        let angle_hi = _mm256_loadu_si256(a8::<i64, 4>(&angle_flag[4..8]));
        let angle_nz_lo = _mm256_xor_si256(
            _mm256_cmpeq_epi64(angle_lo, const_0_epi64),
            _mm256_set1_epi64x(-1),
        );
        let angle_nz_hi = _mm256_xor_si256(
            _mm256_cmpeq_epi64(angle_hi, const_0_epi64),
            _mm256_set1_epi64x(-1),
        );

        let abs_oh = _mm256_abs_epi32(oh_epi32);
        let abs_ov = _mm256_abs_epi32(ov_epi32);
        let abs_od = _mm256_abs_epi32(od_epi32);
        let kh_sign = _mm256_or_si256(
            _mm256_cmpgt_epi32(_mm256_setzero_si256(), oh_epi32),
            const_1_epi32,
        );
        let kv_sign = _mm256_or_si256(
            _mm256_cmpgt_epi32(_mm256_setzero_si256(), ov_epi32),
            const_1_epi32,
        );
        let kd_sign = _mm256_or_si256(
            _mm256_cmpgt_epi32(_mm256_setzero_si256(), od_epi32),
            const_1_epi32,
        );

        // get_best15_from32 has no SIMD clz — extract lanes like C.
        let mut abs_h = [0i32; 8];
        let mut abs_v = [0i32; 8];
        let mut abs_d = [0i32; 8];
        _mm256_storeu_si256(a8m::<i32, 8>(&mut abs_h), abs_oh);
        _mm256_storeu_si256(a8m::<i32, 8>(&mut abs_v), abs_ov);
        _mm256_storeu_si256(a8m::<i32, 8>(&mut abs_d), abs_od);
        let mut kh_msb = [0i32; 8];
        let mut kh_sh = [0i32; 8];
        let mut kv_msb = [0i32; 8];
        let mut kv_sh = [0i32; 8];
        let mut kd_msb = [0i32; 8];
        let mut kd_sh = [0i32; 8];
        for lane in 0..8 {
            // get_best15_from32 is only meaningful for abs >= 32768; smaller
            // lanes use abs_o directly via the msb blend (C calls clz(0)
            // unconditionally — UB there, but its result is blended out).
            let (m, s) = if abs_h[lane] >= 32768 {
                let (m, s) = get_best15_from32(abs_h[lane] as u32);
                (m as i32, s)
            } else {
                (0, 0)
            };
            kh_msb[lane] = m;
            kh_sh[lane] = s;
            let (m, s) = if abs_v[lane] >= 32768 {
                let (m, s) = get_best15_from32(abs_v[lane] as u32);
                (m as i32, s)
            } else {
                (0, 0)
            };
            kv_msb[lane] = m;
            kv_sh[lane] = s;
            let (m, s) = if abs_d[lane] >= 32768 {
                let (m, s) = get_best15_from32(abs_d[lane] as u32);
                (m as i32, s)
            } else {
                (0, 0)
            };
            kd_msb[lane] = m;
            kd_sh[lane] = s;
        }
        let kh_shift = _mm256_loadu_si256(a8::<i32, 8>(&kh_sh));
        let kv_shift = _mm256_loadu_si256(a8::<i32, 8>(&kv_sh));
        let kd_shift = _mm256_loadu_si256(a8::<i32, 8>(&kd_sh));
        let mut kh_msb_v = _mm256_loadu_si256(a8::<i32, 8>(&kh_msb));
        let mut kv_msb_v = _mm256_loadu_si256(a8::<i32, 8>(&kv_msb));
        let mut kd_msb_v = _mm256_loadu_si256(a8::<i32, 8>(&kd_msb));
        let mask_kh = _mm256_cmpgt_epi32(const_32768_epi32, abs_oh);
        let mask_kv = _mm256_cmpgt_epi32(const_32768_epi32, abs_ov);
        let mask_kd = _mm256_cmpgt_epi32(const_32768_epi32, abs_od);
        kh_msb_v = blend_v3(_token, abs_oh, kh_msb_v, mask_kh);
        kv_msb_v = blend_v3(_token, abs_ov, kv_msb_v, mask_kv);
        kd_msb_v = blend_v3(_token, abs_od, kd_msb_v, mask_kd);
        let kh_shift = blend_v3(_token, _mm256_setzero_si256(), kh_shift, mask_kh);
        let kv_shift = blend_v3(_token, _mm256_setzero_si256(), kv_shift, mask_kv);
        let kd_shift = blend_v3(_token, _mm256_setzero_si256(), kd_shift, mask_kd);

        // tmp_k = (div[msb+32768] * t * sign + (1<<(14+shift))) >> (15+shift),
        // per i64 half; o == 0 lanes take 32768.
        let mut tmp_k = [_mm256_setzero_si256(); 6];
        for (k, (div_idx, t_lo, t_hi, sign, o_lo, o_hi, shift)) in [
            (
                _mm256_add_epi32(kh_msb_v, const_32768_epi32),
                th_lo,
                th_hi,
                kh_sign,
                oh_lo,
                oh_hi,
                kh_shift,
            ),
            (
                _mm256_add_epi32(kv_msb_v, const_32768_epi32),
                tv_lo,
                tv_hi,
                kv_sign,
                ov_lo,
                ov_hi,
                kv_shift,
            ),
            (
                _mm256_add_epi32(kd_msb_v, const_32768_epi32),
                td_lo,
                td_hi,
                kd_sign,
                od_lo,
                od_hi,
                kd_shift,
            ),
        ]
        .iter()
        .enumerate()
        {
            let div_g = gather_i32_v3(_token, &div[..], *div_idx);
            let one_shift =
                _mm256_sllv_epi32(const_1_epi32, _mm256_add_epi32(const_14_epi32, *shift));
            let fifteen = _mm256_add_epi32(const_15_epi32, *shift);
            let mut cond_lo = _mm256_mul_epi32(
                _mm256_cvtepi32_epi64(_mm256_castsi256_si128(div_g)),
                _mm256_mul_epi32(*t_lo, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(*sign))),
            );
            cond_lo = sra_epi64_v3(
                _token,
                _mm256_add_epi64(
                    cond_lo,
                    _mm256_cvtepi32_epi64(_mm256_castsi256_si128(one_shift)),
                ),
                _mm256_cvtepi32_epi64(_mm256_castsi256_si128(fifteen)),
            );
            let mask_lo = _mm256_cmpeq_epi64(*o_lo, const_0_epi64);
            tmp_k[k] = blend_v3(_token, const_32768_epi64, cond_lo, mask_lo);
            let mut cond_hi = _mm256_mul_epi32(
                _mm256_cvtepi32_epi64(_mm256_extracti128_si256(div_g, 1)),
                _mm256_mul_epi32(
                    *t_hi,
                    _mm256_cvtepi32_epi64(_mm256_extracti128_si256(*sign, 1)),
                ),
            );
            cond_hi = sra_epi64_v3(
                _token,
                _mm256_add_epi64(
                    cond_hi,
                    _mm256_cvtepi32_epi64(_mm256_extracti128_si256(one_shift, 1)),
                ),
                _mm256_cvtepi32_epi64(_mm256_extracti128_si256(fifteen, 1)),
            );
            let mask_hi = _mm256_cmpeq_epi64(*o_hi, const_0_epi64);
            tmp_k[k + 3] = blend_v3(_token, const_32768_epi64, cond_hi, mask_hi);
        }

        // kh/kv/kd clamped to [0, 32768].
        let clamp = |_token: X64V3Token, tmp: __m256i| {
            let t = blend_v3(
                _token,
                const_32768_epi64,
                tmp,
                _mm256_cmpgt_epi64(tmp, const_32768_epi64),
            );
            blend_v3(
                _token,
                const_0_epi64,
                t,
                _mm256_cmpgt_epi64(const_0_epi64, tmp),
            )
        };
        let kh_lo = clamp(_token, tmp_k[0]);
        let kv_lo = clamp(_token, tmp_k[1]);
        let kd_lo = clamp(_token, tmp_k[2]);
        let kh_hi = clamp(_token, tmp_k[3]);
        let kv_hi = clamp(_token, tmp_k[4]);
        let kd_hi = clamp(_token, tmp_k[5]);

        // rst = (k*o + 16384) >> 15; only the low 32 bits are kept, so the
        // logical srli (2^49 + arithmetic>>15) agrees with C's i64->int cast.
        let mut rst_h_lo = _mm256_srli_epi64(
            _mm256_add_epi64(_mm256_mul_epi32(kh_lo, oh_lo), const_16384_epi64),
            15,
        );
        let mut rst_h_hi = _mm256_srli_epi64(
            _mm256_add_epi64(_mm256_mul_epi32(kh_hi, oh_hi), const_16384_epi64),
            15,
        );
        let mut rst_v_lo = _mm256_srli_epi64(
            _mm256_add_epi64(_mm256_mul_epi32(kv_lo, ov_lo), const_16384_epi64),
            15,
        );
        let mut rst_v_hi = _mm256_srli_epi64(
            _mm256_add_epi64(_mm256_mul_epi32(kv_hi, ov_hi), const_16384_epi64),
            15,
        );
        let mut rst_d_lo = _mm256_srli_epi64(
            _mm256_add_epi64(_mm256_mul_epi32(kd_lo, od_lo), const_16384_epi64),
            15,
        );
        let mut rst_d_hi = _mm256_srli_epi64(
            _mm256_add_epi64(_mm256_mul_epi32(kd_hi, od_hi), const_16384_epi64),
            15,
        );
        let mut rst_h32 = pack_i64x4x2_i32_v3(_token, rst_h_lo, rst_h_hi);
        let mut rst_v32 = pack_i64x4x2_i32_v3(_token, rst_v_lo, rst_v_hi);
        let mut rst_d32 = pack_i64x4x2_i32_v3(_token, rst_d_lo, rst_d_hi);

        let kh_f = _mm256_cvtepi32_ps(pack_i64x4x2_i32_v3(_token, kh_lo, kh_hi));
        let kv_f = _mm256_cvtepi32_ps(pack_i64x4x2_i32_v3(_token, kv_lo, kv_hi));
        let kd_f = _mm256_cvtepi32_ps(pack_i64x4x2_i32_v3(_token, kd_lo, kd_hi));
        let rst_h_f = _mm256_mul_ps(
            _mm256_mul_ps(kh_f, inv_32768_f),
            _mm256_mul_ps(_mm256_cvtepi32_ps(oh_epi32), inv_64_f),
        );
        let rst_v_f = _mm256_mul_ps(
            _mm256_mul_ps(kv_f, inv_32768_f),
            _mm256_mul_ps(_mm256_cvtepi32_ps(ov_epi32), inv_64_f),
        );
        let rst_d_f = _mm256_mul_ps(
            _mm256_mul_ps(kd_f, inv_32768_f),
            _mm256_mul_ps(_mm256_cvtepi32_ps(od_epi32), inv_64_f),
        );

        // gain products as f64 -> i64 via lane extraction, like C.
        macro_rules! gain_i64 {
            ($rst32:expr) => {{
                let pd_lo =
                    _mm256_mul_pd(_mm256_cvtepi32_pd(_mm256_castsi256_si128($rst32)), gain_d);
                let pd_hi = _mm256_mul_pd(
                    _mm256_cvtepi32_pd(_mm256_extracti128_si256($rst32, 1)),
                    gain_d,
                );
                let mut ga = [0.0f64; 4];
                let mut gb = [0.0f64; 4];
                _mm256_storeu_pd(&mut ga, pd_lo);
                _mm256_storeu_pd(&mut gb, pd_hi);
                (
                    _mm256_setr_epi64x(ga[0] as i64, ga[1] as i64, ga[2] as i64, ga[3] as i64),
                    _mm256_setr_epi64x(gb[0] as i64, gb[1] as i64, gb[2] as i64, gb[3] as i64),
                )
            }};
        }
        macro_rules! apply_gain {
            ($rst_lo:ident, $rst_hi:ident, $rst32:ident, $t_lo:ident, $t_hi:ident, $rst_f:ident) => {{
                let (g_lo, g_hi) = gain_i64!($rst32);
                let min_lo = blend_v3(_token, g_lo, $t_lo, _mm256_cmpgt_epi64($t_lo, g_lo));
                let min_hi = blend_v3(_token, g_hi, $t_hi, _mm256_cmpgt_epi64($t_hi, g_hi));
                let max_lo = blend_v3(_token, g_lo, $t_lo, _mm256_cmpgt_epi64(g_lo, $t_lo));
                let max_hi = blend_v3(_token, g_hi, $t_hi, _mm256_cmpgt_epi64(g_hi, $t_hi));
                let f_lo = _mm256_cvtps_pd(_mm256_castps256_ps128($rst_f));
                let f_hi = _mm256_cvtps_pd(_mm256_extractf128_ps($rst_f, 1));
                let mask_gt_lo = _mm256_and_si256(
                    angle_nz_lo,
                    _mm256_castpd_si256(_mm256_cmp_pd::<_CMP_GT_OS>(f_lo, const_0_pd)),
                );
                let mask_gt_hi = _mm256_and_si256(
                    angle_nz_hi,
                    _mm256_castpd_si256(_mm256_cmp_pd::<_CMP_GT_OS>(f_hi, const_0_pd)),
                );
                let mask_lt_lo = _mm256_and_si256(
                    angle_nz_lo,
                    _mm256_castpd_si256(_mm256_cmp_pd::<_CMP_LT_OS>(f_lo, const_0_pd)),
                );
                let mask_lt_hi = _mm256_and_si256(
                    angle_nz_hi,
                    _mm256_castpd_si256(_mm256_cmp_pd::<_CMP_LT_OS>(f_hi, const_0_pd)),
                );
                $rst_lo = blend_v3(_token, min_lo, $rst_lo, mask_gt_lo);
                $rst_hi = blend_v3(_token, min_hi, $rst_hi, mask_gt_hi);
                $rst_lo = blend_v3(_token, max_lo, $rst_lo, mask_lt_lo);
                $rst_hi = blend_v3(_token, max_hi, $rst_hi, mask_lt_hi);
                $rst32 = pack_i64x4x2_i32_v3(_token, $rst_lo, $rst_hi);
            }};
        }
        apply_gain!(rst_h_lo, rst_h_hi, rst_h32, th_lo, th_hi, rst_h_f);
        apply_gain!(rst_v_lo, rst_v_hi, rst_v32, tv_lo, tv_hi, rst_v_f);
        apply_gain!(rst_d_lo, rst_d_hi, rst_d32, td_lo, td_hi, rst_d_f);

        let ah = _mm256_sub_epi32(th_epi32, rst_h32);
        let av = _mm256_sub_epi32(tv_epi32, rst_v32);
        let ad = _mm256_sub_epi32(td_epi32, rst_d32);

        _mm256_storeu_si256(a8m::<i32, 8>(&mut r_h[idx..idx + 8]), rst_h32);
        _mm256_storeu_si256(a8m::<i32, 8>(&mut r_v[idx..idx + 8]), rst_v32);
        _mm256_storeu_si256(a8m::<i32, 8>(&mut r_d[idx..idx + 8]), rst_d32);
        _mm256_storeu_si256(a8m::<i32, 8>(&mut a_h[idx..idx + 8]), ah);
        _mm256_storeu_si256(a8m::<i32, 8>(&mut a_v[idx..idx + 8]), av);
        _mm256_storeu_si256(a8m::<i32, 8>(&mut a_d[idx..idx + 8]), ad);
        j += 8;
    }
}

#[cfg_attr(feature = "simd", autoversion)]
fn adm_decouple_s123(
    ref_b: &BandI32,
    dis_b: &BandI32,
    r: &mut BandI32,
    a: &mut BandI32,
    w: usize,
    h: usize,
    stride: usize,
    div: &[i32; 65537],
    enhn_gain_limit: f64,
) {
    let cos_1deg_sq = (std::f64::consts::PI / 180.0).cos().powi(2);
    let (left, top, right, bottom) = border_region(w, h, 1);
    for i in top..bottom {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        let j_start = if let Some(token) = v3_token() {
            adm_decouple_s123_row_v3(
                token,
                &ref_b.h,
                &ref_b.v,
                &ref_b.d,
                &dis_b.h,
                &dis_b.v,
                &dis_b.d,
                &mut r.h,
                &mut r.v,
                &mut r.d,
                &mut a.h,
                &mut a.v,
                &mut a.d,
                i as usize,
                stride,
                left as usize,
                right as usize,
                div,
                enhn_gain_limit,
            );
            left + ((right - left) / 8) * 8
        } else {
            left
        };
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        let j_start = left;
        for j in j_start..right {
            let idx = i as usize * stride + j as usize;
            let oh = ref_b.h[idx];
            let ov = ref_b.v[idx];
            let od = ref_b.d[idx];
            let th = dis_b.h[idx];
            let tv = dis_b.v[idx];
            let td = dis_b.d[idx];
            let ot_dp = oh as i64 * th as i64 + ov as i64 * tv as i64;
            let o_mag_sq = oh as i64 * oh as i64 + ov as i64 * ov as i64;
            let t_mag_sq = th as i64 * th as i64 + tv as i64 * tv as i64;
            let dp_f = ot_dp as f32 as f64 / 4096.0;
            let angle_flag = dp_f >= 0.0
                && (dp_f * dp_f
                    >= cos_1deg_sq
                        * (o_mag_sq as f32 as f64 / 4096.0)
                        * (t_mag_sq as f32 as f64 / 4096.0));

            let k = |o: i32, t: i32| -> i64 {
                if o == 0 {
                    return 32768;
                }
                let abs_o = o.unsigned_abs();
                let sign = if o < 0 { -1i64 } else { 1i64 };
                let (msb, shift) = if abs_o < 32768 {
                    (abs_o as u16, 0)
                } else {
                    get_best15_from32(abs_o)
                };
                (div[msb as usize + 32768] as i64 * t as i64 * sign + (1i64 << (14 + shift)))
                    >> (15 + shift)
            };
            let tmp_kh = k(oh, th);
            let tmp_kv = k(ov, tv);
            let tmp_kd = k(od, td);
            let kh = tmp_kh.clamp(0, 32768);
            let kv = tmp_kv.clamp(0, 32768);
            let kd = tmp_kd.clamp(0, 32768);

            let mut rst_h = (((kh * oh as i64) + 16384) >> 15) as i32;
            let mut rst_v = (((kv * ov as i64) + 16384) >> 15) as i32;
            let mut rst_d = (((kd * od as i64) + 16384) >> 15) as i32;

            let rst_h_f = (kh as f32 / 32768.0) * (oh as f32 / 64.0);
            let rst_v_f = (kv as f32 / 32768.0) * (ov as f32 / 64.0);
            let rst_d_f = (kd as f32 / 32768.0) * (od as f32 / 64.0);

            if angle_flag && rst_h_f > 0.0 {
                rst_h = ((rst_h as f64 * enhn_gain_limit).min(th as f64)) as i32;
            }
            if angle_flag && rst_h_f < 0.0 {
                rst_h = ((rst_h as f64 * enhn_gain_limit).max(th as f64)) as i32;
            }
            if angle_flag && rst_v_f > 0.0 {
                rst_v = ((rst_v as f64 * enhn_gain_limit).min(tv as f64)) as i32;
            }
            if angle_flag && rst_v_f < 0.0 {
                rst_v = ((rst_v as f64 * enhn_gain_limit).max(tv as f64)) as i32;
            }
            if angle_flag && rst_d_f > 0.0 {
                rst_d = ((rst_d as f64 * enhn_gain_limit).min(td as f64)) as i32;
            }
            if angle_flag && rst_d_f < 0.0 {
                rst_d = ((rst_d as f64 * enhn_gain_limit).max(td as f64)) as i32;
            }

            r.h[idx] = rst_h;
            r.v[idx] = rst_v;
            r.d[idx] = rst_d;
            a.h[idx] = th - rst_h;
            a.v[idx] = tv - rst_v;
            a.d[idx] = td - rst_d;
        }
    }
}

fn adm_csf_i16(
    src: &BandI16,
    dst: &mut BandI16,
    flt: &mut BandI16,
    w: usize,
    h: usize,
    stride: usize,
    rf: [f32; 3],
    watson_fixed: bool,
) {
    let i_rfactor = if watson_fixed {
        [36453, 36453, 49417]
    } else {
        [
            (rf[0] as f64 * 2f64.powi(21)) as u64 as u16,
            (rf[1] as f64 * 2f64.powi(21)) as u64 as u16,
            (rf[2] as f64 * 2f64.powi(23)) as u64 as u16,
        ]
    };
    let i_shifts = [15u32, 15, 17];
    let i_shiftsadd = [16384i32, 16384, 65535];
    let fix_one_by_30 = 4369i32;
    let (left, top, right, bottom) = border_region(w, h, 1);
    for theta in 0..3 {
        let (src_p, dst_p, flt_p) = match theta {
            0 => (&src.h, &mut dst.h, &mut flt.h),
            1 => (&src.v, &mut dst.v, &mut flt.v),
            _ => (&src.d, &mut dst.d, &mut flt.d),
        };
        for i in top..bottom {
            let mut j = left;
            #[cfg(feature = "simd")]
            while j + 16 <= right {
                let idx = i as usize * stride + j as usize;
                let mut d16 = [0i16; 16];
                let mut f16 = [0i16; 16];
                archmage::incant!(
                    adm_csf_i16_simd(
                        &src_p[idx..idx + 16],
                        i_rfactor[theta] as i32,
                        i_shiftsadd[theta],
                        i_shifts[theta],
                        &mut d16,
                        &mut f16
                    ),
                    [v3, neon, wasm128, scalar]
                );
                dst_p[idx..idx + 16].copy_from_slice(&d16);
                flt_p[idx..idx + 16].copy_from_slice(&f16);
                j += 16;
            }
            while j < right {
                let idx = i as usize * stride + j as usize;
                let dst_val = i_rfactor[theta] as i32 * src_p[idx] as i32;
                let v = ((dst_val + i_shiftsadd[theta]) >> i_shifts[theta]) as i16;
                dst_p[idx] = v;
                flt_p[idx] = (((fix_one_by_30 * (v as i32).abs()) + 2048) >> 12) as i16;
                j += 1;
            }
        }
    }
}

#[cfg(feature = "simd")]
#[magetypes(define(i16x16, i32x8), v3, neon, wasm128, scalar)]
fn adm_csf_i16_simd(
    token: Token,
    src: &[i16],
    rfactor: i32,
    shiftsadd: i32,
    shift: u32,
    dst: &mut [i16; 16],
    flt: &mut [i16; 16],
) {
    let s = i16x16::load(token, src[..16].try_into().unwrap());
    let sl = s.widen_low();
    let sh = s.widen_high();
    let rf = i32x8::splat(token, rfactor);
    let add = i32x8::splat(token, shiftsadd);
    let c4369 = i32x8::splat(token, 4369);
    let a2048 = i32x8::splat(token, 2048);
    let vtl = (rf * sl + add)
        .shr_arithmetic_uniform(shift)
        .shl_uniform(16)
        .shr_arithmetic_uniform(16);
    let vth = (rf * sh + add)
        .shr_arithmetic_uniform(shift)
        .shl_uniform(16)
        .shr_arithmetic_uniform(16);
    let fl = (c4369 * vtl.abs() + a2048).shr_arithmetic_uniform(12);
    let fh = (c4369 * vth.abs() + a2048).shr_arithmetic_uniform(12);
    let (dla, dha) = (vtl.to_array(), vth.to_array());
    let (fla, fha) = (fl.to_array(), fh.to_array());
    for k in 0..8 {
        dst[k] = dla[k] as i16;
        dst[8 + k] = dha[k] as i16;
        flt[k] = fla[k] as i16;
        flt[8 + k] = fha[k] as i16;
    }
}

#[cfg_attr(feature = "simd", autoversion)]
fn adm_csf_i32(
    src: &BandI32,
    dst: &mut BandI32,
    flt: &mut BandI32,
    w: usize,
    h: usize,
    stride: usize,
    rf: [f32; 3],
) {
    let i_rfactor = [
        (rf[0] as f64 * 2f64.powi(32)) as u64 as u32,
        (rf[1] as f64 * 2f64.powi(32)) as u64 as u32,
        (rf[2] as f64 * 2f64.powi(32)) as u64 as u32,
    ];
    let fix_one_by_30 = 143165577i64;
    let add_dst = 1i64 << 27;
    let add_flt = -(1i64 << 31);
    let (left, top, right, bottom) = border_region(w, h, 1);
    for (theta, &rf) in i_rfactor.iter().enumerate() {
        let (src_p, dst_p, flt_p) = match theta {
            0 => (&src.h, &mut dst.h, &mut flt.h),
            1 => (&src.v, &mut dst.v, &mut flt.v),
            _ => (&src.d, &mut dst.d, &mut flt.d),
        };
        for i in top..bottom {
            for j in left..right {
                let idx = i as usize * stride + j as usize;
                let dst_val = ((rf as i64 * src_p[idx] as i64 + add_dst) >> 28) as i32;
                dst_p[idx] = dst_val;
                flt_p[idx] = ((fix_one_by_30 * (dst_val as i64).abs() + add_flt) >> 32) as i32;
            }
        }
    }
}

#[cfg_attr(feature = "simd", autoversion)]
fn adm_csf_den_scale(
    src: &BandI16,
    w: usize,
    h: usize,
    stride: usize,
    rf: [f32; 3],
    noise_weight: f64,
) -> f32 {
    let (left, top, right, bottom) = border_region_inner(w, h);
    let mut accum_h = 0u64;
    let mut accum_v = 0u64;
    let mut accum_d = 0u64;
    let npix = ((bottom - top) * (right - left)) as f64;
    let mut shift_accum = (npix.log2() - 20.0).ceil() as i32;
    if shift_accum < 0 {
        shift_accum = 0;
    }
    let add_shift_accum = if shift_accum > 0 {
        1u64 << (shift_accum - 1)
    } else {
        0
    };
    let bands = [&src.h, &src.v, &src.d];
    let accums = [&mut accum_h, &mut accum_v, &mut accum_d];
    for band_idx in 0..3 {
        for i in top..bottom {
            let mut inner = 0u64;
            for j in left..right {
                let v = bands[band_idx][i as usize * stride + j as usize].unsigned_abs() as u64;
                inner += v * v * v;
            }
            *accums[band_idx] += (inner + add_shift_accum) >> shift_accum;
        }
    }
    let shift_csf = 2f64.powi(18 - shift_accum);
    let csf_h = accum_h as f64 / shift_csf * (rf[0] as f64).powi(3);
    let csf_v = accum_v as f64 / shift_csf * (rf[1] as f64).powi(3);
    let csf_d = accum_d as f64 / shift_csf * (rf[2] as f64).powi(3);
    let powf_add = (((bottom - top) * (right - left)) as f64 * noise_weight) as f32;
    let powf_add = powf_add.powf(1.0 / 3.0);
    (csf_h as f32).powf(1.0 / 3.0)
        + powf_add
        + (csf_v as f32).powf(1.0 / 3.0)
        + powf_add
        + (csf_d as f32).powf(1.0 / 3.0)
        + powf_add
}

#[cfg_attr(feature = "simd", autoversion)]
fn adm_csf_den_s123(
    src: &BandI32,
    scale: usize,
    w: usize,
    h: usize,
    stride: usize,
    rf: [f32; 3],
    noise_weight: f64,
) -> f32 {
    let shift_sq = [31u32, 30, 31];
    let accum_convert_float = [32i32, 27, 23];
    let add_shift_sq = [1u64 << 31, 1u64 << 30, 1u64 << 31];
    let (left, top, right, bottom) = border_region_inner(w, h);
    let shift_cub = ((right - left) as f64).log2().ceil() as u32;
    let add_shift_cub = 1u64 << (shift_cub - 1);
    let shift_accum = ((bottom - top) as f64).log2().ceil() as u32;
    let add_shift_accum = 1u64 << (shift_accum - 1);
    let mut accums = [0u64; 3];
    let bands = [&src.h, &src.v, &src.d];
    for band_idx in 0..3 {
        for i in top..bottom {
            let mut inner = 0u64;
            for j in left..right {
                let v = bands[band_idx][i as usize * stride + j as usize].unsigned_abs() as u64;
                inner += (((v * v + add_shift_sq[scale - 1]) >> shift_sq[scale - 1]) * v
                    + add_shift_cub)
                    >> shift_cub;
            }
            accums[band_idx] += (inner + add_shift_accum) >> shift_accum;
        }
    }
    let shift_csf =
        2f64.powi(accum_convert_float[scale - 1] - shift_accum as i32 - shift_cub as i32);
    let csf_h = accums[0] as f64 / shift_csf * (rf[0] as f64).powi(3);
    let csf_v = accums[1] as f64 / shift_csf * (rf[1] as f64).powi(3);
    let csf_d = accums[2] as f64 / shift_csf * (rf[2] as f64).powi(3);
    let powf_add = (((bottom - top) * (right - left)) as f64 * noise_weight) as f32;
    let powf_add = powf_add.powf(1.0 / 3.0);
    (csf_h as f32).powf(1.0 / 3.0)
        + powf_add
        + (csf_v as f32).powf(1.0 / 3.0)
        + powf_add
        + (csf_d as f32).powf(1.0 / 3.0)
        + powf_add
}

fn edge_x(v: i32, w: usize) -> usize {
    if v < 0 {
        (-v) as usize
    } else {
        (v as usize).min(w - 1)
    }
}

fn edge_y(v: i32, h: usize) -> usize {
    if v < 0 {
        (-v) as usize
    } else {
        (v as usize).min(h - 1)
    }
}

fn adm_cm_i16(
    src: &BandI16,
    csf_f: &BandI16,
    csf_a: &BandI16,
    w: usize,
    h: usize,
    stride: usize,
    rf: [f32; 3],
    noise_weight: f64,
    watson_fixed: bool,
) -> f32 {
    let i_rfactor = if watson_fixed {
        [36453, 36453, 49417]
    } else {
        [
            (rf[0] as f64 * 2f64.powi(21)) as u64 as u16 as i32,
            (rf[1] as f64 * 2f64.powi(21)) as u64 as u16 as i32,
            (rf[2] as f64 * 2f64.powi(23)) as u64 as u16 as i32,
        ]
    };
    let shift_xsub = [10i32, 10, 12];
    let shift_xsq = [29u32, 29, 30];
    let add_shift_xsq = [1i64 << 28, 1i64 << 28, 1i64 << 29];
    let shift_xcub = [
        ((w as f64).log2() - 4.0).ceil() as u32,
        ((w as f64).log2() - 4.0).ceil() as u32,
        ((w as f64).log2() - 3.0).ceil() as u32,
    ];
    let add_shift_xcub = [
        1i64 << (shift_xcub[0] - 1),
        1i64 << (shift_xcub[1] - 1),
        1i64 << (shift_xcub[2] - 1),
    ];
    let shift_inner_accum = (h as f64).log2().ceil() as u32;
    let add_shift_inner_accum = 1i64 << (shift_inner_accum - 1);
    let (left, top, right, bottom) = border_region_inner(w, h);
    let src_b = [&src.h, &src.v, &src.d];
    let angles = [&csf_a.h, &csf_a.v, &csf_a.d];
    let flt = [&csf_f.h, &csf_f.v, &csf_f.d];
    let mut accum = [0i64; 3];
    let i0 = top.max(0);
    let i1 = bottom.min(h as i32);
    let j0 = left.max(0);
    let j1 = right.min(w as i32);
    if i0 >= 1 && j0 >= 1 && i1 < h as i32 && j1 < w as i32 && j0 < j1 {
        let (j0, j1) = (j0 as usize, j1 as usize);
        let len = j1 - j0;
        let mut cs = vec![0i32; 3 * (w + 2)];
        let mut win = vec![0i32; w];
        let (cs_a, cs_vd) = cs.split_at_mut(w + 2);
        let (cs_v, cs_d) = cs_vd.split_at_mut(w + 2);
        let mut cs_bands = [cs_a, cs_v, cs_d];
        for i in i0..i1 {
            let i = i as usize;
            for (band, cs) in flt.iter().zip(cs_bands.iter_mut()) {
                let r0 = &band[(i - 1) * stride..];
                let r1 = &band[i * stride..];
                let r2 = &band[(i + 1) * stride..];
                for (((o, &a), &b), &c) in cs[j0 - 1..=j1]
                    .iter_mut()
                    .zip(&r0[j0 - 1..=j1])
                    .zip(&r1[j0 - 1..=j1])
                    .zip(&r2[j0 - 1..=j1])
                {
                    *o = a as i32 + b as i32 + c as i32;
                }
            }
            win[..len].fill(0);
            for cs in &cs_bands {
                for (((o, &a), &b), &c) in win[..len]
                    .iter_mut()
                    .zip(&cs[j0 - 1..j1 - 1])
                    .zip(&cs[j0..j1])
                    .zip(&cs[j0 + 1..=j1])
                {
                    *o += a + b + c;
                }
            }
            let row = i * stride + j0;
            let ang_rows = [
                &angles[0][row..row + len],
                &angles[1][row..row + len],
                &angles[2][row..row + len],
            ];
            let flt_rows = [
                &flt[0][row..row + len],
                &flt[1][row..row + len],
                &flt[2][row..row + len],
            ];
            let src_rows = [
                &src_b[0][row..row + len],
                &src_b[1][row..row + len],
                &src_b[2][row..row + len],
            ];
            let mut inner = [0i64; 3];
            #[cfg(feature = "simd")]
            let mut jj = 0usize;
            #[cfg(not(feature = "simd"))]
            let jj = 0usize;
            #[cfg(feature = "simd")]
            while jj + 16 <= len {
                let mut xbuf = [[0i32; 16]; 3];
                archmage::incant!(
                    adm_cm_i16_front(
                        &win[jj..jj + 16],
                        &[
                            &ang_rows[0][jj..jj + 16],
                            &ang_rows[1][jj..jj + 16],
                            &ang_rows[2][jj..jj + 16],
                        ],
                        &[
                            &flt_rows[0][jj..jj + 16],
                            &flt_rows[1][jj..jj + 16],
                            &flt_rows[2][jj..jj + 16],
                        ],
                        &[
                            &src_rows[0][jj..jj + 16],
                            &src_rows[1][jj..jj + 16],
                            &src_rows[2][jj..jj + 16],
                        ],
                        i_rfactor,
                        shift_xsub,
                        &mut xbuf
                    ),
                    [v3, neon, wasm128, scalar]
                );
                #[cfg(target_arch = "x86_64")]
                let mut used_v3 = false;
                #[cfg(target_arch = "x86_64")]
                if let Some(token) = v3_token() {
                    cm_accum16_v3(
                        token,
                        &xbuf,
                        add_shift_xsq,
                        shift_xsq,
                        add_shift_xcub,
                        shift_xcub,
                        &mut inner,
                    );
                    used_v3 = true;
                }
                #[cfg(not(target_arch = "x86_64"))]
                let used_v3 = false;
                if !used_v3 {
                    for ((&x0, &x1), &x2) in xbuf[0].iter().zip(&xbuf[1]).zip(&xbuf[2]) {
                        for (t, x) in [x0, x1, x2].into_iter().enumerate() {
                            let x = x as i64;
                            let x_sq = (((x * x) + add_shift_xsq[t]) >> shift_xsq[t]) as i32;
                            inner[t] += ((x_sq as i64 * x) + add_shift_xcub[t]) >> shift_xcub[t];
                        }
                    }
                }
                jj += 16;
            }
            for jj in jj..len {
                let wv = win[jj];
                let mut thr = wv;
                for t in 0..3 {
                    thr += (((ONE_BY_15 * (ang_rows[t][jj] as i32).abs()) + 2048) >> 12) as i16
                        as i32
                        - flt_rows[t][jj] as i32;
                }
                for t in 0..3 {
                    let mut x =
                        (src_rows[t][jj] as i32 * i_rfactor[t]).abs() - (thr << shift_xsub[t]);
                    if x < 0 {
                        x = 0;
                    }
                    let x_sq = (((x as i64 * x as i64) + add_shift_xsq[t]) >> shift_xsq[t]) as i32;
                    let val = ((x_sq as i64 * x as i64) + add_shift_xcub[t]) >> shift_xcub[t];
                    inner[t] += val;
                }
            }
            for t in 0..3 {
                accum[t] += (inner[t] + add_shift_inner_accum) >> shift_inner_accum;
            }
        }
    } else {
        for i in i0..i1 {
            let mut inner = [0i64; 3];
            for j in j0..j1 {
                let idx = i as usize * stride + j as usize;
                let mut thr = 0i32;
                for t in 0..3 {
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let yy = edge_y(i + dy, h);
                            let xx = edge_x(j + dx, w);
                            if dy == 0 && dx == 0 {
                                thr += (((ONE_BY_15 * (angles[t][idx] as i32).abs()) + 2048) >> 12)
                                    as i16 as i32;
                            } else {
                                thr += flt[t][yy * stride + xx] as i32;
                            }
                        }
                    }
                }
                for t in 0..3 {
                    let mut x =
                        (src_b[t][idx] as i32 * i_rfactor[t]).abs() - (thr << shift_xsub[t]);
                    if x < 0 {
                        x = 0;
                    }
                    let x_sq = (((x as i64 * x as i64) + add_shift_xsq[t]) >> shift_xsq[t]) as i32;
                    let val = ((x_sq as i64 * x as i64) + add_shift_xcub[t]) >> shift_xcub[t];
                    inner[t] += val;
                }
            }
            for t in 0..3 {
                accum[t] += (inner[t] + add_shift_inner_accum) >> shift_inner_accum;
            }
        }
    }
    let f_accum = [
        accum[0] as f64 / 2f64.powi(52 - shift_xcub[0] as i32 - shift_inner_accum as i32),
        accum[1] as f64 / 2f64.powi(52 - shift_xcub[1] as i32 - shift_inner_accum as i32),
        accum[2] as f64 / 2f64.powi(57 - shift_xcub[2] as i32 - shift_inner_accum as i32),
    ];
    let powf_add = (((bottom - top) * (right - left)) as f64 * noise_weight) as f32;
    let powf_add = powf_add.powf(1.0 / 3.0);
    (f_accum[0] as f32).powf(1.0 / 3.0)
        + powf_add
        + (f_accum[1] as f32).powf(1.0 / 3.0)
        + powf_add
        + (f_accum[2] as f32).powf(1.0 / 3.0)
        + powf_add
}

/// Direct port of `ADM_CM_ACCUM_ROUND_avx256`'s i64 accumulation chain:
/// x_sq = (x*x + add) >> sq on real 64-bit lanes, then (x_sq*x + add2) >> cub.
/// x already has the abs/sub/max front half applied (see adm_cm_i16_front).
/// mul_epi32 uses only the low 32 bits of each operand, so the i32-wrapping
/// of x_sq matches the scalar `as i32` cast exactly. Returns (lo, hi) qword
/// accumulators covering x lanes {0,2,4,6} and {1,3,5,7}.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[rite]
fn cm_accum_v3(
    _token: X64V3Token,
    x: __m256i,
    add_sq: i64,
    sq: u32,
    add_cub: i64,
    cub: u32,
) -> (__m256i, __m256i) {
    let x_odd = _mm256_srli_epi64(x, 32);
    let sq_cnt = _mm_cvtsi32_si128(sq as i32);
    let cub_cnt = _mm_cvtsi32_si128(cub as i32);
    let xsq_lo = _mm256_srl_epi64(
        _mm256_add_epi64(_mm256_mul_epi32(x, x), _mm256_set1_epi64x(add_sq)),
        sq_cnt,
    );
    let xsq_hi = _mm256_srl_epi64(
        _mm256_add_epi64(_mm256_mul_epi32(x_odd, x_odd), _mm256_set1_epi64x(add_sq)),
        sq_cnt,
    );
    let cub_lo = _mm256_srl_epi64(
        _mm256_add_epi64(_mm256_mul_epi32(xsq_lo, x), _mm256_set1_epi64x(add_cub)),
        cub_cnt,
    );
    let cub_hi = _mm256_srl_epi64(
        _mm256_add_epi64(_mm256_mul_epi32(xsq_hi, x_odd), _mm256_set1_epi64x(add_cub)),
        cub_cnt,
    );
    (cub_lo, cub_hi)
}

/// 16-wide driver for `cm_accum_v3` over the xbuf produced by
/// `adm_cm_i16_front`; reduces both qword accumulators into `inner`.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane(import_intrinsics)]
fn cm_accum16_v3(
    token: X64V3Token,
    xbuf: &[[i32; 16]; 3],
    add_shift_xsq: [i64; 3],
    shift_xsq: [u32; 3],
    add_shift_xcub: [i64; 3],
    shift_xcub: [u32; 3],
    inner: &mut [i64; 3],
) {
    for t in 0..3 {
        let mut acc_lo = _mm256_setzero_si256();
        let mut acc_hi = _mm256_setzero_si256();
        for xc in xbuf[t].chunks_exact(8) {
            let x = _mm256_loadu_si256(a8::<i32, 8>(xc));
            let (lo, hi) = cm_accum_v3(
                token,
                x,
                add_shift_xsq[t],
                shift_xsq[t],
                add_shift_xcub[t],
                shift_xcub[t],
            );
            acc_lo = _mm256_add_epi64(acc_lo, lo);
            acc_hi = _mm256_add_epi64(acc_hi, hi);
        }
        let mut lo = [0i64; 4];
        let mut hi = [0i64; 4];
        _mm256_storeu_si256(a8m::<i64, 4>(&mut lo), acc_lo);
        _mm256_storeu_si256(a8m::<i64, 4>(&mut hi), acc_hi);
        inner[t] += lo.iter().sum::<i64>() + hi.iter().sum::<i64>();
    }
}

/// Direct port of `i4_adm_cm_avx2`'s interior row kernel. Unlike C's 2-pixel
/// sliding-window thresh, our win[] colsum row supplies thr for all lanes, so
/// each i64 quad processes 4 pixels. The signed-shift emulation
/// (srli + msb-mask on negative lanes) and mul_epi32(-1) absolute value
/// mirror I4_ADM_CM_THRESH_S_I_J_avx256 / I4_ADM_CM_ACCUM_ROUND_avx256.
/// Returns the per-band row contribution and the count of pixels processed.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane(import_intrinsics)]
#[allow(clippy::too_many_arguments)]
fn i4_adm_cm_row_v3(
    _token: X64V3Token,
    win: &[i32],
    ang_rows: &[&[i32]; 3],
    flt_rows: &[&[i32]; 3],
    src_rows: &[&[i32]; 3],
    rfactor: [u32; 3],
    add_bef_shift_dst: i64,
    add_bef_shift_flt: i64,
    add_shift_sq: i64,
    add_shift_cub: i64,
    shift_cub: u32,
) -> ([i64; 3], usize) {
    let i4_15 = _mm256_set1_epi64x(I4_ONE_BY_15);
    let neg1 = _mm256_set1_epi32(-1);
    let add_dst = _mm256_set1_epi64x(add_bef_shift_dst);
    let add_flt = _mm256_set1_epi64x(add_bef_shift_flt);
    let add_sq = _mm256_set1_epi64x(add_shift_sq);
    let add_cub = _mm256_set1_epi64x(add_shift_cub);
    let mask_dst = _mm256_set1_epi64x(0xFFFFFFF000000000u64 as i64);
    let mask_flt = _mm256_set1_epi64x(0xFFFFFFFF00000000u64 as i64);
    let zero = _mm256_setzero_si256();

    let mut acc = [_mm256_setzero_si256(); 3];
    let mut j = 0usize;
    while j + 4 <= win.len() {
        // thr = win[j] + sum_t(comp(ang_t[j]) - flt_t[j]), i64 lanes.
        let mut thr = _mm256_cvtepi32_epi64(_mm_loadu_si128(a8::<i32, 4>(&win[j..j + 4])));
        for t in 0..3 {
            let ang = _mm256_cvtepi32_epi64(_mm_loadu_si128(a8::<i32, 4>(&ang_rows[t][j..j + 4])));
            let ltz = _mm256_cmpgt_epi64(zero, ang);
            let a_us = _mm256_and_si256(_mm256_mul_epi32(ang, neg1), ltz);
            let a_abs = _mm256_or_si256(_mm256_andnot_si256(ltz, ang), a_us);
            let comp = _mm256_add_epi64(_mm256_mul_epi32(a_abs, i4_15), add_flt);
            let sgn = _mm256_and_si256(mask_flt, _mm256_cmpgt_epi64(zero, comp));
            let comp = _mm256_or_si256(_mm256_srli_epi64(comp, 32), sgn);
            let flt = _mm256_cvtepi32_epi64(_mm_loadu_si128(a8::<i32, 4>(&flt_rows[t][j..j + 4])));
            thr = _mm256_add_epi64(thr, _mm256_sub_epi64(comp, flt));
        }
        for t in 0..3 {
            let src = _mm256_cvtepi32_epi64(_mm_loadu_si128(a8::<i32, 4>(&src_rows[t][j..j + 4])));
            let rf = _mm256_set1_epi32(rfactor[t] as i32);
            let xf = _mm256_add_epi64(_mm256_mul_epi32(src, rf), add_dst);
            let sgn = _mm256_and_si256(mask_dst, _mm256_cmpgt_epi64(zero, xf));
            let xf = _mm256_or_si256(_mm256_srli_epi64(xf, 28), sgn);
            // abs on i64 lanes then thr subtract (shift_sub = 0), clamp > 0.
            let ltz = _mm256_cmpgt_epi64(zero, xf);
            let x_us = _mm256_and_si256(_mm256_mul_epi32(xf, neg1), ltz);
            let x_abs = _mm256_or_si256(_mm256_andnot_si256(ltz, xf), x_us);
            let mut x = _mm256_sub_epi64(x_abs, thr);
            x = _mm256_and_si256(x, _mm256_cmpgt_epi64(x, zero));
            let xsq = _mm256_srli_epi64(_mm256_add_epi64(_mm256_mul_epi32(x, x), add_sq), 30);
            let val = _mm256_srl_epi64(
                _mm256_add_epi64(_mm256_mul_epi32(xsq, x), add_cub),
                _mm_cvtsi32_si128(shift_cub as i32),
            );
            acc[t] = _mm256_add_epi64(acc[t], val);
        }
        j += 4;
    }
    let mut inner = [0i64; 3];
    for t in 0..3 {
        let mut v = [0i64; 4];
        _mm256_storeu_si256(a8m::<i64, 4>(&mut v), acc[t]);
        inner[t] = v.iter().sum();
    }
    (inner, j)
}

#[cfg(feature = "simd")]
#[magetypes(define(i16x16, i32x8), v3, neon, wasm128, scalar)]
fn adm_cm_i16_front(
    token: Token,
    win: &[i32],
    ang: &[&[i16]; 3],
    flt: &[&[i16]; 3],
    src: &[&[i16]; 3],
    i_rfactor: [i32; 3],
    shift_xsub: [i32; 3],
    x_out: &mut [[i32; 16]; 3],
) {
    let mut thr_l = i32x8::load(token, win[..8].try_into().unwrap());
    let mut thr_h = i32x8::load(token, win[8..16].try_into().unwrap());
    let c15 = i32x8::splat(token, ONE_BY_15);
    let a2048 = i32x8::splat(token, 2048);
    for t in 0..3 {
        let a = i16x16::load(token, ang[t][..16].try_into().unwrap());
        let f = i16x16::load(token, flt[t][..16].try_into().unwrap());
        let (al, ah) = (a.widen_low(), a.widen_high());
        let (fl, fh) = (f.widen_low(), f.widen_high());
        let cl = (c15 * al.abs() + a2048)
            .shr_arithmetic_uniform(12)
            .shl_uniform(16)
            .shr_arithmetic_uniform(16);
        let ch = (c15 * ah.abs() + a2048)
            .shr_arithmetic_uniform(12)
            .shl_uniform(16)
            .shr_arithmetic_uniform(16);
        thr_l += cl - fl;
        thr_h += ch - fh;
    }
    for t in 0..3 {
        let s = i16x16::load(token, src[t][..16].try_into().unwrap());
        let (sl, sh) = (s.widen_low(), s.widen_high());
        let rf = i32x8::splat(token, i_rfactor[t]);
        let sub = shift_xsub[t] as u32;
        let xl = (rf * sl).abs() - thr_l.shl_uniform(sub);
        let xh = (rf * sh).abs() - thr_h.shl_uniform(sub);
        let xl = xl.max(i32x8::zero(token));
        let xh = xh.max(i32x8::zero(token));
        x_out[t][..8].copy_from_slice(&xl.to_array());
        x_out[t][8..16].copy_from_slice(&xh.to_array());
    }
}

#[cfg_attr(feature = "simd", autoversion)]
fn adm_cm_i32(
    src: &BandI32,
    csf_f: &BandI32,
    csf_a: &BandI32,
    scale: usize,
    w: usize,
    h: usize,
    stride: usize,
    rf: [f32; 3],
    noise_weight: f64,
) -> f32 {
    let rfactor = [
        (rf[0] as f64 * 2f64.powi(32)) as u64 as u32,
        (rf[1] as f64 * 2f64.powi(32)) as u64 as u32,
        (rf[2] as f64 * 2f64.powi(32)) as u64 as u32,
    ];
    let add_bef_shift_dst = 1i64 << 27;
    let add_bef_shift_flt = -(1i64 << 31);
    let shift_flt = 32u32;
    let shift_cub = (w as f64).log2().ceil() as u32;
    let add_shift_cub = 1i64 << (shift_cub - 1);
    let shift_inner_accum = (h as f64).log2().ceil() as u32;
    let add_shift_inner_accum = 1i64 << (shift_inner_accum - 1);
    let final_shift = [
        2f64.powi(45 - shift_cub as i32 - shift_inner_accum as i32),
        2f64.powi(39 - shift_cub as i32 - shift_inner_accum as i32),
        2f64.powi(36 - shift_cub as i32 - shift_inner_accum as i32),
    ];
    let add_shift_sq = 1i64 << 29;
    let shift_sq = 30u32;
    let (left, top, right, bottom) = border_region_inner(w, h);
    let src_b = [&src.h, &src.v, &src.d];
    let angles = [&csf_a.h, &csf_a.v, &csf_a.d];
    let flt = [&csf_f.h, &csf_f.v, &csf_f.d];
    let mut accum = [0i64; 3];
    let i0 = top.max(0);
    let i1 = bottom.min(h as i32);
    let j0 = left.max(0);
    let j1 = right.min(w as i32);
    if i0 >= 1 && j0 >= 1 && i1 < h as i32 && j1 < w as i32 && j0 < j1 {
        let (j0, j1) = (j0 as usize, j1 as usize);
        let len = j1 - j0;
        let mut cs = vec![0i32; 3 * (w + 2)];
        let mut win = vec![0i32; w];
        let (cs_a, cs_vd) = cs.split_at_mut(w + 2);
        let (cs_v, cs_d) = cs_vd.split_at_mut(w + 2);
        let mut cs_bands = [cs_a, cs_v, cs_d];
        for i in i0..i1 {
            let i = i as usize;
            for (band, cs) in flt.iter().zip(cs_bands.iter_mut()) {
                let r0 = &band[(i - 1) * stride..];
                let r1 = &band[i * stride..];
                let r2 = &band[(i + 1) * stride..];
                for (((o, &a), &b), &c) in cs[j0 - 1..=j1]
                    .iter_mut()
                    .zip(&r0[j0 - 1..=j1])
                    .zip(&r1[j0 - 1..=j1])
                    .zip(&r2[j0 - 1..=j1])
                {
                    *o = a + b + c;
                }
            }
            win[..len].fill(0);
            for cs in &cs_bands {
                for (((o, &a), &b), &c) in win[..len]
                    .iter_mut()
                    .zip(&cs[j0 - 1..j1 - 1])
                    .zip(&cs[j0..j1])
                    .zip(&cs[j0 + 1..=j1])
                {
                    *o += a + b + c;
                }
            }
            let row = i * stride + j0;
            let ang_rows = [
                &angles[0][row..row + len],
                &angles[1][row..row + len],
                &angles[2][row..row + len],
            ];
            let flt_rows = [
                &flt[0][row..row + len],
                &flt[1][row..row + len],
                &flt[2][row..row + len],
            ];
            let src_rows = [
                &src_b[0][row..row + len],
                &src_b[1][row..row + len],
                &src_b[2][row..row + len],
            ];
            let mut inner = [0i64; 3];
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            let mut jj = 0usize;
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            let jj = 0usize;
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            if let Some(token) = v3_token() {
                let (vec_inner, done) = i4_adm_cm_row_v3(
                    token,
                    &win[..len],
                    &ang_rows,
                    &flt_rows,
                    &src_rows,
                    rfactor,
                    add_bef_shift_dst,
                    add_bef_shift_flt,
                    add_shift_sq,
                    add_shift_cub,
                    shift_cub,
                );
                for t in 0..3 {
                    inner[t] += vec_inner[t];
                }
                jj = done;
            }
            for (off, &wv) in win[jj..len].iter().enumerate() {
                let jj = jj + off;
                let mut thr = wv;
                for t in 0..3 {
                    thr += ((I4_ONE_BY_15 * (ang_rows[t][jj] as i64).abs() + add_bef_shift_flt)
                        >> shift_flt) as i32
                        - flt_rows[t][jj];
                }
                for t in 0..3 {
                    let x_full = ((src_rows[t][jj] as i64 * rfactor[t] as i64 + add_bef_shift_dst)
                        >> 28) as i32;
                    let mut x = x_full.abs() - thr;
                    if x < 0 {
                        x = 0;
                    }
                    let x_sq = (((x as i64 * x as i64) + add_shift_sq) >> shift_sq) as i32;
                    let val = ((x_sq as i64 * x as i64) + add_shift_cub) >> shift_cub;
                    inner[t] += val;
                }
            }
            for t in 0..3 {
                accum[t] += (inner[t] + add_shift_inner_accum) >> shift_inner_accum;
            }
        }
    } else {
        for i in i0..i1 {
            let mut inner = [0i64; 3];
            for j in j0..j1 {
                let idx = i as usize * stride + j as usize;
                let mut thr = 0i32;
                for t in 0..3 {
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let yy = edge_y(i + dy, h);
                            let xx = edge_x(j + dx, w);
                            if dy == 0 && dx == 0 {
                                thr += ((I4_ONE_BY_15 * (angles[t][idx] as i64).abs()
                                    + add_bef_shift_flt)
                                    >> shift_flt) as i32;
                            } else {
                                thr += flt[t][yy * stride + xx];
                            }
                        }
                    }
                }
                for t in 0..3 {
                    let x_full = ((src_b[t][idx] as i64 * rfactor[t] as i64 + add_bef_shift_dst)
                        >> 28) as i32;
                    let mut x = x_full.abs() - thr;
                    if x < 0 {
                        x = 0;
                    }
                    let x_sq = (((x as i64 * x as i64) + add_shift_sq) >> shift_sq) as i32;
                    let val = ((x_sq as i64 * x as i64) + add_shift_cub) >> shift_cub;
                    inner[t] += val;
                }
            }
            for t in 0..3 {
                accum[t] += (inner[t] + add_shift_inner_accum) >> shift_inner_accum;
            }
        }
    }
    let powf_add = (((bottom - top) * (right - left)) as f64 * noise_weight) as f32;
    let powf_add = powf_add.powf(1.0 / 3.0);
    ((accum[0] as f64 / final_shift[scale - 1]) as f32).powf(1.0 / 3.0)
        + powf_add
        + ((accum[1] as f64 / final_shift[scale - 1]) as f32).powf(1.0 / 3.0)
        + powf_add
        + ((accum[2] as f64 / final_shift[scale - 1]) as f32).powf(1.0 / 3.0)
        + powf_add
}

#[cfg_attr(feature = "simd", autoversion)]
fn adm_dwt2_s123_combined(
    i4_ref_scale: &[i32],
    i4_dis_scale: &[i32],
    ref_stride: usize,
    dis_stride: usize,
    ref_out: &mut BandI32,
    dis_out: &mut BandI32,
    a_ref_out: &mut [i32],
    a_dis_out: &mut [i32],
    w: usize,
    h: usize,
    dst_stride: usize,
    scale: usize,
    ind_y: &[[i32; 4]],
    ind_x: &[[i32; 4]],
) {
    let add_vp = [0i64, 32768, 32768][scale - 1];
    let add_hp = [16384i64, 32768, 16384][scale - 1];
    let shift_vp = [0u32, 16, 16][scale - 1];
    let shift_hp = [15u32, 16, 15][scale - 1];
    let mut tmplo_ref = vec![0i32; w];
    let mut tmphi_ref = vec![0i32; w];
    let mut tmplo_dis = vec![0i32; w];
    let mut tmphi_dis = vec![0i32; w];
    for i in 0..h.div_ceil(2) {
        for j in 0..w {
            let mut acc = 0i64;
            for k in 0..4 {
                acc +=
                    DWT2_LO[k] as i64 * i4_ref_scale[ind_y[i][k] as usize * ref_stride + j] as i64;
            }
            tmplo_ref[j] = ((acc + add_vp) >> shift_vp) as i32;
            let mut acc = 0i64;
            for k in 0..4 {
                acc +=
                    DWT2_HI[k] as i64 * i4_ref_scale[ind_y[i][k] as usize * ref_stride + j] as i64;
            }
            tmphi_ref[j] = ((acc + add_vp) >> shift_vp) as i32;

            let mut acc = 0i64;
            for k in 0..4 {
                acc +=
                    DWT2_LO[k] as i64 * i4_dis_scale[ind_y[i][k] as usize * dis_stride + j] as i64;
            }
            tmplo_dis[j] = ((acc + add_vp) >> shift_vp) as i32;
            let mut acc = 0i64;
            for k in 0..4 {
                acc +=
                    DWT2_HI[k] as i64 * i4_dis_scale[ind_y[i][k] as usize * dis_stride + j] as i64;
            }
            tmphi_dis[j] = ((acc + add_vp) >> shift_vp) as i32;
        }
        for (j, ix) in ind_x.iter().enumerate() {
            let (j0, j1, j2, j3) = (
                ix[0] as usize,
                ix[1] as usize,
                ix[2] as usize,
                ix[3] as usize,
            );
            let idx = i * dst_stride + j;
            for (lo, hi, out, a_out) in [
                (&tmplo_ref, &tmphi_ref, &mut *ref_out, &mut *a_ref_out),
                (&tmplo_dis, &tmphi_dis, &mut *dis_out, &mut *a_dis_out),
            ] {
                let s = [lo[j0], lo[j1], lo[j2], lo[j3]];
                let mut acc = 0i64;
                for k in 0..4 {
                    acc += DWT2_LO[k] as i64 * s[k] as i64;
                }
                a_out[idx] = ((acc + add_hp) >> shift_hp) as i32;
                let mut acc = 0i64;
                for k in 0..4 {
                    acc += DWT2_HI[k] as i64 * s[k] as i64;
                }
                out.v[idx] = ((acc + add_hp) >> shift_hp) as i32;
                let s = [hi[j0], hi[j1], hi[j2], hi[j3]];
                let mut acc = 0i64;
                for k in 0..4 {
                    acc += DWT2_LO[k] as i64 * s[k] as i64;
                }
                out.h[idx] = ((acc + add_hp) >> shift_hp) as i32;
                let mut acc = 0i64;
                for k in 0..4 {
                    acc += DWT2_HI[k] as i64 * s[k] as i64;
                }
                out.d[idx] = ((acc + add_hp) >> shift_hp) as i32;
            }
        }
    }
}

fn compute_adm(
    reference_y: &[u16],
    distorted_y: &[u16],
    width: usize,
    height: usize,
    bit_depth: u8,
    settings: &AdmSettings,
) -> Result<(f64, f64), Error> {
    if !matches!(bit_depth, 8 | 10) {
        return Err(Error::InvalidInput("unsupported bit depth"));
    }
    if width < ADM_MIN_DIM || height < ADM_MIN_DIM {
        return Err(Error::InvalidInput("dimensions too small"));
    }
    let npix = width
        .checked_mul(height)
        .ok_or(Error::InvalidInput("dimension overflow"))?;
    if reference_y.len() != npix || distorted_y.len() != npix {
        return Err(Error::InvalidInput("plane length mismatch"));
    }
    let max_sample = (1u32 << bit_depth) - 1;
    if reference_y
        .iter()
        .chain(distorted_y.iter())
        .any(|&v| v as u32 > max_sample)
    {
        return Err(Error::InvalidInput("sample exceeds bit depth"));
    }
    if settings.norm_view_dist * (settings.ref_display_height as f64) < 3.0 * 1080.0 {
        return Err(Error::InvalidInput("viewing distance unsupported"));
    }

    let numden_limit = 1e-10 * (width * height) as f64 / (1920.0 * 1080.0);
    let buf_stride = (width.div_ceil(2) * 4).div_ceil(32) * 8;
    let band_elems = buf_stride * height.div_ceil(2);
    let div = div_lookup();

    let mut num = 0.0f64;
    let mut den = 0.0f64;
    let mut aim_num = 0.0f64;
    let mut w = width;
    let mut h = height;
    let mut i4_ref_scale: Vec<i32> = Vec::new();
    let mut i4_dis_scale: Vec<i32> = Vec::new();
    let mut i4_ref_next = vec![0i32; band_elems];
    let mut i4_dis_next = vec![0i32; band_elems];
    // Reused across scales 1-3: every element read in an iteration is
    // written earlier in that same iteration, so stale contents never
    // leak into results.
    let new_i32_band = || BandI32 {
        h: vec![0; band_elems],
        v: vec![0; band_elems],
        d: vec![0; band_elems],
    };
    let mut ref_b32 = new_i32_band();
    let mut dis_b32 = new_i32_band();
    let mut r32 = new_i32_band();
    let mut a32 = new_i32_band();
    let mut csf_a32 = new_i32_band();
    let mut csf_f32 = new_i32_band();

    for scale in 0..4usize {
        let (ind_y, ind_x) = dwt2_indices(w, h);
        let rf = rfactor(settings, scale);
        let (num_scale, den_scale, aim_num_scale);
        if scale == 0 {
            let mut ref_b = BandI16 {
                h: vec![0; band_elems],
                v: vec![0; band_elems],
                d: vec![0; band_elems],
            };
            let mut dis_b = BandI16 {
                h: vec![0; band_elems],
                v: vec![0; band_elems],
                d: vec![0; band_elems],
            };
            i4_ref_scale = vec![0i32; band_elems];
            i4_dis_scale = vec![0i32; band_elems];
            adm_dwt2(
                reference_y,
                width,
                &mut ref_b,
                w,
                h,
                buf_stride,
                bit_depth,
                &ind_y,
                &ind_x,
                &mut i4_ref_scale,
            );
            adm_dwt2(
                distorted_y,
                width,
                &mut dis_b,
                w,
                h,
                buf_stride,
                bit_depth,
                &ind_y,
                &ind_x,
                &mut i4_dis_scale,
            );
            let mut r = BandI16 {
                h: vec![0; band_elems],
                v: vec![0; band_elems],
                d: vec![0; band_elems],
            };
            let mut a = BandI16 {
                h: vec![0; band_elems],
                v: vec![0; band_elems],
                d: vec![0; band_elems],
            };
            w = w.div_ceil(2);
            h = h.div_ceil(2);
            adm_decouple(
                &ref_b,
                &dis_b,
                &mut r,
                &mut a,
                w,
                h,
                buf_stride,
                div,
                settings.enhn_gain_limit,
            );
            den_scale = adm_csf_den_scale(&ref_b, w, h, buf_stride, rf, settings.noise_weight);
            let mut csf_a = BandI16 {
                h: vec![0; band_elems],
                v: vec![0; band_elems],
                d: vec![0; band_elems],
            };
            let mut csf_f = BandI16 {
                h: vec![0; band_elems],
                v: vec![0; band_elems],
                d: vec![0; band_elems],
            };
            adm_csf_i16(
                &a,
                &mut csf_a,
                &mut csf_f,
                w,
                h,
                buf_stride,
                rf,
                settings.watson_fixed,
            );
            num_scale = adm_cm_i16(
                &r,
                &csf_f,
                &csf_a,
                w,
                h,
                buf_stride,
                rf,
                settings.noise_weight,
                settings.watson_fixed,
            );
            adm_csf_i16(
                &r,
                &mut csf_f,
                &mut csf_a,
                w,
                h,
                buf_stride,
                rf,
                settings.watson_fixed,
            );
            aim_num_scale = adm_cm_i16(
                &a,
                &csf_a,
                &csf_f,
                w,
                h,
                buf_stride,
                rf,
                0.0,
                settings.watson_fixed,
            );
        } else {
            adm_dwt2_s123_combined(
                &i4_ref_scale,
                &i4_dis_scale,
                buf_stride,
                buf_stride,
                &mut ref_b32,
                &mut dis_b32,
                &mut i4_ref_next,
                &mut i4_dis_next,
                w,
                h,
                buf_stride,
                scale,
                &ind_y,
                &ind_x,
            );
            std::mem::swap(&mut i4_ref_scale, &mut i4_ref_next);
            std::mem::swap(&mut i4_dis_scale, &mut i4_dis_next);
            w = w.div_ceil(2);
            h = h.div_ceil(2);
            adm_decouple_s123(
                &ref_b32,
                &dis_b32,
                &mut r32,
                &mut a32,
                w,
                h,
                buf_stride,
                div,
                settings.enhn_gain_limit,
            );
            den_scale =
                adm_csf_den_s123(&ref_b32, scale, w, h, buf_stride, rf, settings.noise_weight);
            adm_csf_i32(&a32, &mut csf_a32, &mut csf_f32, w, h, buf_stride, rf);
            num_scale = adm_cm_i32(
                &r32,
                &csf_f32,
                &csf_a32,
                scale,
                w,
                h,
                buf_stride,
                rf,
                settings.noise_weight,
            );
            adm_csf_i32(&r32, &mut csf_f32, &mut csf_a32, w, h, buf_stride, rf);
            aim_num_scale = adm_cm_i32(&a32, &csf_a32, &csf_f32, scale, w, h, buf_stride, rf, 0.0);
        }
        num += num_scale as f64;
        den += den_scale as f64;
        aim_num += aim_num_scale as f64;
    }

    num = if num < numden_limit { 0.0 } else { num };
    den = if den < numden_limit { 0.0 } else { den };
    let (score, score_aim) = if den == 0.0 {
        (1.0, 0.0)
    } else {
        (num / den, aim_num / den)
    };
    Ok((score, score_aim))
}

pub fn adm3_v1_from_luma(
    reference_y: &[u16],
    distorted_y: &[u16],
    width: usize,
    height: usize,
    bit_depth: u8,
    variant: ModelVariant,
) -> Result<f64, Error> {
    let (norm_view_dist, ref_display_height) = match variant {
        ModelVariant::Standard1080p | ModelVariant::HfrStandard1080p => (3.0f64, 1080),
        ModelVariant::Phone | ModelVariant::HfrPhone => (5.0, 1080),
        ModelVariant::Default4k | ModelVariant::HfrDefault4k => (1.5, 2160),
        ModelVariant::Consumer4k | ModelVariant::HfrConsumer4k => (3.0, 2160),
    };
    let settings = AdmSettings {
        factors: csf_table(variant),
        enhn_gain_limit: ENHN_GAIN_LIMIT,
        noise_weight: NOISE_WEIGHT,
        watson_fixed: false,
        norm_view_dist,
        ref_display_height,
    };
    let (score, aim) = compute_adm(
        reference_y,
        distorted_y,
        width,
        height,
        bit_depth,
        &settings,
    )?;
    Ok((score * DLM_WEIGHT + (1.0 - aim) * (1.0 - DLM_WEIGHT)).max(MIN_VAL))
}

pub fn adm2_v0_from_luma(
    reference_y: &[u16],
    distorted_y: &[u16],
    width: usize,
    height: usize,
    bit_depth: u8,
    variant: VmafV0Variant,
) -> Result<f64, Error> {
    let settings = AdmSettings {
        factors: &WATSON97_CSF_1080_3H,
        enhn_gain_limit: if variant.no_enhancement_gain() {
            1.0
        } else {
            100.0
        },
        noise_weight: 0.03125,
        watson_fixed: true,
        norm_view_dist: 3.0,
        ref_display_height: 1080,
    };
    compute_adm(
        reference_y,
        distorted_y,
        width,
        height,
        bit_depth,
        &settings,
    )
    .map(|(score, _)| score)
}

#[cfg(all(test, feature = "simd"))]
#[test]
fn simd_dwt2_matches_scalar_for_edges_tails_and_bit_depths() {
    for (w, h) in [(17, 17), (18, 26), (19, 30), (32, 19), (65, 29), (128, 45)] {
        for bit_depth in [8, 10] {
            let src: Vec<_> = (0..h)
                .flat_map(|y| {
                    (0..w).map(move |x| ((x * 17 + y * 31 + x * y * 3) % (1 << bit_depth)) as u16)
                })
                .collect();
            let (ind_y, ind_x) = dwt2_indices(w, h);
            let dst_stride = w.div_ceil(2);
            let len = dst_stride * h.div_ceil(2);
            let make_band = || BandI16 {
                h: vec![0; len],
                v: vec![0; len],
                d: vec![0; len],
            };
            let mut scalar = make_band();
            let mut simd = make_band();
            let mut scalar_a = vec![0i32; len];
            let mut simd_a = vec![0i32; len];
            adm_dwt2_scalar(
                &src,
                w,
                &mut scalar,
                w,
                h,
                dst_stride,
                bit_depth,
                &ind_y,
                &ind_x,
                &mut scalar_a,
            );
            adm_dwt2(
                &src,
                w,
                &mut simd,
                w,
                h,
                dst_stride,
                bit_depth,
                &ind_y,
                &ind_x,
                &mut simd_a,
            );
            assert_eq!(scalar_a, simd_a, "{w}x{h}, {bit_depth} bit, band a");
            assert_eq!(scalar.h, simd.h, "{w}x{h}, {bit_depth} bit, band h");
            assert_eq!(scalar.v, simd.v, "{w}x{h}, {bit_depth} bit, band v");
            assert_eq!(scalar.d, simd.d, "{w}x{h}, {bit_depth} bit, band d");
        }
    }
}

/// `adm_decouple_s123_row_v3` must bit-match the scalar loop on every interior
/// pixel, including the get_best15_from32 branch (|v| >= 32768) and zero refs.
#[cfg(all(test, feature = "simd", target_arch = "x86_64"))]
#[test]
fn simd_decouple_s123_matches_scalar_for_tails_zeros_and_gains() {
    use std::sync::atomic::Ordering;
    let div = div_lookup();
    for (w, h) in [(44, 13), (57, 21), (33, 33), (101, 17)] {
        for gain in [1.0f64, 100.0] {
            let n = w * h;
            let mut x = 0x9E3779B97F4A7C15u64;
            let mut rng = move || {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x % 200001) as i32 - 100000
            };
            let make_band = |rng: &mut dyn FnMut() -> i32| BandI32 {
                h: (0..n).map(|_| rng()).collect(),
                v: (0..n).map(|_| rng()).collect(),
                d: (0..n).map(|_| rng()).collect(),
            };
            let mut ref_b = make_band(&mut rng);
            let dis_b = make_band(&mut rng);
            for i in 0..h {
                ref_b.h[i * w + (i % w)] = 0;
                ref_b.v[i * w + ((i * 3) % w)] = 0;
                ref_b.d[i * w + ((i * 7) % w)] = 0;
            }
            let run = |force: bool| {
                FORCE_SCALAR.store(force, Ordering::Relaxed);
                let mut r = BandI32 {
                    h: vec![0; n],
                    v: vec![0; n],
                    d: vec![0; n],
                };
                let mut a = BandI32 {
                    h: vec![0; n],
                    v: vec![0; n],
                    d: vec![0; n],
                };
                adm_decouple_s123(&ref_b, &dis_b, &mut r, &mut a, w, h, w, div, gain);
                FORCE_SCALAR.store(false, Ordering::Relaxed);
                (r, a)
            };
            let (rs, as_) = run(true);
            let (rv, av) = run(false);
            for (name, s, v) in [
                ("r.h", &rs.h, &rv.h),
                ("r.v", &rs.v, &rv.v),
                ("r.d", &rs.d, &rv.d),
                ("a.h", &as_.h, &av.h),
                ("a.v", &as_.v, &av.v),
                ("a.d", &as_.d, &av.d),
            ] {
                assert_eq!(s, v, "{w}x{h} gain={gain} band {name}");
            }
        }
    }
}

/// `adm_decouple_row_v3` must bit-match the scalar loop on every interior
/// pixel. Realistic DWT2 magnitudes (|v| <= 16000) keep th - rst in i16 range
/// so scalar truncation and packs_epi32 saturation agree — the same bound
/// libvmaf's own scalar/AVX2 paths rely on.
#[cfg(all(test, feature = "simd", target_arch = "x86_64"))]
#[test]
fn simd_decouple_matches_scalar_for_tails_zeros_and_gains() {
    use std::sync::atomic::Ordering;
    let div = div_lookup();
    for (w, h) in [(44, 13), (57, 21), (33, 33), (101, 17)] {
        for gain in [1.0f64, 100.0] {
            let n = w * h;
            let mut x = 0x9E3779B97F4A7C15u64;
            let mut rng = move || {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x % 32001) as i32 - 16000
            };
            let make_band = |rng: &mut dyn FnMut() -> i32| BandI16 {
                h: (0..n).map(|_| rng() as i16).collect(),
                v: (0..n).map(|_| rng() as i16).collect(),
                d: (0..n).map(|_| rng() as i16).collect(),
            };
            let mut ref_b = make_band(&mut rng);
            let dis_b = make_band(&mut rng);
            // Exercise the eqz (zero-ref -> k=32768) and orthogonal-angle paths.
            for i in 0..h {
                ref_b.h[i * w + (i % w)] = 0;
                ref_b.v[i * w + ((i * 3) % w)] = 0;
                ref_b.d[i * w + ((i * 7) % w)] = 0;
            }
            let elems = n;
            let run = |force: bool| {
                FORCE_SCALAR.store(force, Ordering::Relaxed);
                let mut r = BandI16 {
                    h: vec![0; elems],
                    v: vec![0; elems],
                    d: vec![0; elems],
                };
                let mut a = BandI16 {
                    h: vec![0; elems],
                    v: vec![0; elems],
                    d: vec![0; elems],
                };
                adm_decouple(&ref_b, &dis_b, &mut r, &mut a, w, h, w, div, gain);
                FORCE_SCALAR.store(false, Ordering::Relaxed);
                (r, a)
            };
            let (rs, as_) = run(true);
            let (rv, av) = run(false);
            for (name, s, v) in [
                ("r.h", &rs.h, &rv.h),
                ("r.v", &rs.v, &rv.v),
                ("r.d", &rs.d, &rv.d),
                ("a.h", &as_.h, &av.h),
                ("a.v", &as_.v, &av.v),
                ("a.d", &as_.d, &av.d),
            ] {
                assert_eq!(s, v, "{w}x{h} gain={gain} band {name}");
            }
        }
    }
}

/// `cm_accum16_v3` must bit-match the scalar x_sq/x_cub chain on every lane.
/// |v| <= 16000 keeps x_sq < 2^31 so scalar's i32 wrap and the AVX2 path
/// agree — the same bound the decouple parity test uses.
#[cfg(all(test, feature = "simd", target_arch = "x86_64"))]
#[test]
fn simd_cm_i16_matches_scalar_for_tails_and_edges() {
    use std::sync::atomic::Ordering;
    for (w, h) in [(44, 13), (57, 21), (33, 33), (101, 17), (67, 9)] {
        let n = w * h;
        let mut x = 0x9E3779B97F4A7C15u64;
        let mut rng = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % 32001) as i32 - 16000
        };
        let make_band = |rng: &mut dyn FnMut() -> i32| BandI16 {
            h: (0..n).map(|_| rng() as i16).collect(),
            v: (0..n).map(|_| rng() as i16).collect(),
            d: (0..n).map(|_| rng() as i16).collect(),
        };
        let src = make_band(&mut rng);
        let csf_f = make_band(&mut rng);
        let csf_a = make_band(&mut rng);
        let run = |force: bool| {
            FORCE_SCALAR.store(force, Ordering::Relaxed);
            let v = adm_cm_i16(&src, &csf_f, &csf_a, w, h, w, [1.0; 3], 0.0, true);
            FORCE_SCALAR.store(false, Ordering::Relaxed);
            v
        };
        let s = run(true);
        let v = run(false);
        assert_eq!(s.to_bits(), v.to_bits(), "{w}x{h}");
    }
}

/// `i4_adm_cm_row_v3` must bit-match the scalar thr/x/accum chain. src bounded
/// to +-2^20 keeps x_full in i32 range and x_sq positive, where libvmaf's own
/// scalar/AVX2 paths agree. rf fractions keep rfactor < 2^31 (mul_epi32 sign).
#[cfg(all(test, feature = "simd", target_arch = "x86_64"))]
#[test]
fn simd_cm_i32_matches_scalar_for_tails_and_edges() {
    use std::sync::atomic::Ordering;
    for scale in 1..=3usize {
        for (w, h) in [(44, 13), (57, 21), (33, 33), (101, 17), (67, 9)] {
            let n = w * h;
            let mut x = 0x9E3779B97F4A7C15u64;
            let mut rng = move || {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x % 2097153) as i32 - 1048576
            };
            let make_band = |rng: &mut dyn FnMut() -> i32| BandI32 {
                h: (0..n).map(|_| rng()).collect(),
                v: (0..n).map(|_| rng()).collect(),
                d: (0..n).map(|_| rng()).collect(),
            };
            let src = make_band(&mut rng);
            let csf_f = make_band(&mut rng);
            let csf_a = make_band(&mut rng);
            let run = |force: bool| {
                FORCE_SCALAR.store(force, Ordering::Relaxed);
                let v = adm_cm_i32(
                    &src,
                    &csf_f,
                    &csf_a,
                    scale,
                    w,
                    h,
                    w,
                    [17.25, 9.4, 63.125],
                    0.0,
                );
                FORCE_SCALAR.store(false, Ordering::Relaxed);
                v
            };
            let s = run(true);
            let v = run(false);
            assert_eq!(s.to_bits(), v.to_bits(), "scale={scale} {w}x{h}");
        }
    }
}
