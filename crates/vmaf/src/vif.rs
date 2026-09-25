#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use archmage::intrinsics::x86_64::*;
#[cfg(feature = "simd")]
use archmage::magetypes;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use archmage::{SimdToken, X64V3Token, arcane, rite};
#[cfg(all(feature = "simd", feature = "avx512", target_arch = "x86_64"))]
use archmage::X64V4Token;
use std::borrow::Cow;
use std::sync::OnceLock;

use crate::{Error, VmafV0Variant};
use crate::pool;

const FILTERS: [&[u16]; 4] = [
    &[
        489, 935, 1640, 2640, 3896, 5274, 6547, 7455, 7784, 7455, 6547, 5274, 3896, 2640, 1640,
        935, 489,
    ],
    &[1244, 3663, 7925, 12590, 14692, 12590, 7925, 3663, 1244],
    &[3571, 16004, 26386, 16004, 3571],
    &[10904, 43728, 10904],
];
const SIGMA_NSQ: i32 = 131072;

struct VifImage<'a> {
    reference: Cow<'a, [u16]>,
    distorted: Cow<'a, [u16]>,
    width: usize,
    height: usize,
}

fn mirror(index: isize, size: usize) -> usize {
    if index < 0 {
        -index as usize
    } else if index >= size as isize {
        (2 * size as isize - index - 2) as usize
    } else {
        index as usize
    }
}

fn log_table() -> &'static [u16; 65536] {
    static TABLE: OnceLock<[u16; 65536]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut values = [0; 65536];
        for (i, value) in values.iter_mut().enumerate().skip(32767) {
            *value = ((i as f32).log2() * 2048.0).round() as u16;
        }
        values
    })
}

fn log2_32(table: &[u16; 65536], value: u32) -> i32 {
    let k = 16 - value.leading_zeros() as i32;
    table[(value >> k) as usize] as i32 + 2048 * k
}

fn log2_64(table: &[u16; 65536], value: u64) -> i32 {
    let k = 48 - value.leading_zeros() as i32;
    table[(value >> k) as usize] as i32 + 2048 * k
}

fn pad_reflected(row: &mut [u32], width: usize, half: usize) {
    for offset in 0..half {
        row[half - offset - 1] = row[half + offset + 1];
        row[half + width + offset] = row[half + width - offset - 2];
    }
}

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

fn subsample<'a>(image: &VifImage<'a>, bit_depth: u8, scale: usize) -> VifImage<'a> {
    let filter = FILTERS[scale + 1];
    let half = (filter.len() / 2) as isize;
    let (width, height) = (image.width, image.height);
    let (out_width, out_height) = (width / 2, height / 2);
    let mut out_reference = pool::take_u16(out_width * out_height);
    let mut out_distorted = pool::take_u16(out_width * out_height);
    let mut vertical_reference = pool::take_u32(width + 2 * half as usize);
    let mut vertical_distorted = pool::take_u32(width + 2 * half as usize);
    let (shift, round) = if bit_depth == 8 && scale == 0 {
        (8, 128)
    } else if scale == 0 {
        (bit_depth as u32, 1u32 << (bit_depth - 1))
    } else {
        (16, 32768)
    };
    for row in (0..height / 2 * 2).step_by(2) {
        let mut row_offsets = [0usize; 9];
        for (tap, slot) in row_offsets.iter_mut().take(filter.len()).enumerate() {
            *slot = mirror(row as isize + tap as isize - half, height) * width;
        }
        #[allow(unused_mut)]
        let mut processed = 0;
        #[cfg(all(feature = "avx512", feature = "simd", target_arch = "x86_64"))]
        if bit_depth == 8 && scale == 0 {
            // Scale-0 8-bit planes hold v ≤ 255 — the i16 madd-pair kernel is
            // exact; higher-scale planes reach ~65280 and stay on v3.
            if let Some(token) = v4_token() {
                processed = vif_subsample_vertical_v4(
                    token,
                    &image.reference,
                    &image.distorted,
                    &row_offsets,
                    width,
                    filter,
                    shift,
                    round,
                    &mut vertical_reference,
                    &mut vertical_distorted,
                );
            }
        }
        let processed = {
            if processed > 0 {
                processed
            } else {
            #[cfg(feature = "simd")]
            {
                #[cfg(target_arch = "x86_64")]
                if let Some(token) = v3_token() {
                    vif_subsample_vertical_v3(
                        token,
                        &image.reference,
                        &image.distorted,
                        &row_offsets,
                        width,
                        filter,
                        shift,
                        round,
                        &mut vertical_reference,
                        &mut vertical_distorted,
                    )
                } else {
                    archmage::incant!(
                        vif_subsample_vertical_simd(
                            &image.reference,
                            &image.distorted,
                            &row_offsets,
                            width,
                            filter,
                            shift,
                            round,
                            &mut vertical_reference,
                            &mut vertical_distorted
                        ),
                        [v3, neon, wasm128, scalar]
                    )
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    archmage::incant!(
                        vif_subsample_vertical_simd(
                            &image.reference,
                            &image.distorted,
                            &row_offsets,
                            width,
                            filter,
                            shift,
                            round,
                            &mut vertical_reference,
                            &mut vertical_distorted
                        ),
                        [v3, neon, wasm128, scalar]
                    )
                }
            }
            #[cfg(not(feature = "simd"))]
            {
                0
            }
            }
        };
        for col in processed..width {
            let center = half as usize;
            let mut ref_sum =
                filter[center] as u32 * image.reference[row_offsets[center] + col] as u32;
            let mut dis_sum =
                filter[center] as u32 * image.distorted[row_offsets[center] + col] as u32;
            for offset in 1..=center {
                let weight = filter[center - offset] as u32;
                let left = row_offsets[center - offset] + col;
                let right = row_offsets[center + offset] + col;
                ref_sum += weight * (image.reference[left] as u32 + image.reference[right] as u32);
                dis_sum += weight * (image.distorted[left] as u32 + image.distorted[right] as u32);
            }
            if bit_depth == 8 && scale == 0 {
                vertical_reference[col + half as usize] = (ref_sum + 128) >> 8;
                vertical_distorted[col + half as usize] = (dis_sum + 128) >> 8;
            } else {
                vertical_reference[col + half as usize] =
                    ((ref_sum + round) >> shift) as u16 as u32;
                vertical_distorted[col + half as usize] =
                    ((dis_sum + round) >> shift) as u16 as u32;
            }
        }
        pad_reflected(&mut vertical_reference, width, half as usize);
        pad_reflected(&mut vertical_distorted, width, half as usize);
        #[allow(unused_mut)]
        let mut hcol = 0usize;
        #[cfg(feature = "simd")]
        while hcol + 16 <= width {
            #[cfg(target_arch = "x86_64")]
            let (out_ref, out_dis) = if let Some(token) = v3_token() {
                vif_subsample_horizontal_v3(
                    token,
                    &vertical_reference,
                    &vertical_distorted,
                    filter,
                    hcol,
                )
            } else {
                archmage::incant!(
                    vif_subsample_horizontal_simd(
                        &vertical_reference,
                        &vertical_distorted,
                        filter,
                        hcol
                    ),
                    [v3, neon, wasm128, scalar]
                )
            };
            #[cfg(not(target_arch = "x86_64"))]
            let (out_ref, out_dis) = archmage::incant!(
                vif_subsample_horizontal_simd(
                    &vertical_reference,
                    &vertical_distorted,
                    filter,
                    hcol
                ),
                [v3, neon, wasm128, scalar]
            );
            let slot = (row / 2) * out_width + hcol / 2;
            out_reference[slot..slot + 8].copy_from_slice(&out_ref);
            out_distorted[slot..slot + 8].copy_from_slice(&out_dis);
            hcol += 16;
        }
        for col in (hcol..width / 2 * 2).step_by(2) {
            let center = col + half as usize;
            let mut ref_sum = filter[half as usize] as u32 * vertical_reference[center];
            let mut dis_sum = filter[half as usize] as u32 * vertical_distorted[center];
            for offset in 1..=half as usize {
                let weight = filter[half as usize - offset] as u32;
                ref_sum += weight
                    * (vertical_reference[center - offset] + vertical_reference[center + offset]);
                dis_sum += weight
                    * (vertical_distorted[center - offset] + vertical_distorted[center + offset]);
            }
            let slot = (row / 2) * out_width + col / 2;
            out_reference[slot] = ((ref_sum + 32768) >> 16) as u16;
            out_distorted[slot] = ((dis_sum + 32768) >> 16) as u16;
        }
    }
    pool::give_u32(vertical_reference);
    pool::give_u32(vertical_distorted);
    VifImage {
        reference: Cow::Owned(out_reference),
        distorted: Cow::Owned(out_distorted),
        width: out_width,
        height: out_height,
    }
}

#[cfg(feature = "simd")]
#[magetypes(define(u16x16, u32x8), v3, neon, wasm128, scalar)]
fn vif_vertical_u8_simd(
    token: Token,
    reference: &[u16],
    distorted: &[u16],
    row_offsets: &[usize; 17],
    width: usize,
    vertical_ref_mean: &mut [u32],
    vertical_dis_mean: &mut [u32],
    vertical_ref_sq: &mut [u32],
    vertical_dis_sq: &mut [u32],
    vertical_ref_dis: &mut [u32],
) -> usize {
    let filter = FILTERS[0];
    let chunks = width / 16;
    for chunk in 0..chunks {
        let col = chunk * 16;
        let mut ref_mean_lo = u32x8::zero(token);
        let mut ref_mean_hi = u32x8::zero(token);
        let mut dis_mean_lo = u32x8::zero(token);
        let mut dis_mean_hi = u32x8::zero(token);
        let mut ref_sq_lo = u32x8::zero(token);
        let mut ref_sq_hi = u32x8::zero(token);
        let mut dis_sq_lo = u32x8::zero(token);
        let mut dis_sq_hi = u32x8::zero(token);
        let mut ref_dis_lo = u32x8::zero(token);
        let mut ref_dis_hi = u32x8::zero(token);
        let weight = u32x8::splat(token, filter[8] as u32);
        let center_ref = u16x16::load(
            token,
            reference[row_offsets[8] + col..row_offsets[8] + col + 16]
                .try_into()
                .unwrap(),
        );
        let center_dis = u16x16::load(
            token,
            distorted[row_offsets[8] + col..row_offsets[8] + col + 16]
                .try_into()
                .unwrap(),
        );
        let ref_lo = center_ref.widen_low();
        let ref_hi = center_ref.widen_high();
        let dis_lo = center_dis.widen_low();
        let dis_hi = center_dis.widen_high();
        ref_mean_lo += ref_lo * weight;
        ref_mean_hi += ref_hi * weight;
        dis_mean_lo += dis_lo * weight;
        dis_mean_hi += dis_hi * weight;
        ref_sq_lo += ref_lo * weight * ref_lo;
        ref_sq_hi += ref_hi * weight * ref_hi;
        dis_sq_lo += dis_lo * weight * dis_lo;
        dis_sq_hi += dis_hi * weight * dis_hi;
        ref_dis_lo += ref_lo * weight * dis_lo;
        ref_dis_hi += ref_hi * weight * dis_hi;
        for offset in 1..=8usize {
            let weight = u32x8::splat(token, filter[8 - offset] as u32);
            let left_ref = u16x16::load(
                token,
                reference[row_offsets[8 - offset] + col..row_offsets[8 - offset] + col + 16]
                    .try_into()
                    .unwrap(),
            );
            let right_ref = u16x16::load(
                token,
                reference[row_offsets[8 + offset] + col..row_offsets[8 + offset] + col + 16]
                    .try_into()
                    .unwrap(),
            );
            let left_dis = u16x16::load(
                token,
                distorted[row_offsets[8 - offset] + col..row_offsets[8 - offset] + col + 16]
                    .try_into()
                    .unwrap(),
            );
            let right_dis = u16x16::load(
                token,
                distorted[row_offsets[8 + offset] + col..row_offsets[8 + offset] + col + 16]
                    .try_into()
                    .unwrap(),
            );
            let lref_lo = left_ref.widen_low();
            let lref_hi = left_ref.widen_high();
            let rref_lo = right_ref.widen_low();
            let rref_hi = right_ref.widen_high();
            let ldis_lo = left_dis.widen_low();
            let ldis_hi = left_dis.widen_high();
            let rdis_lo = right_dis.widen_low();
            let rdis_hi = right_dis.widen_high();
            ref_mean_lo += weight * (lref_lo + rref_lo);
            ref_mean_hi += weight * (lref_hi + rref_hi);
            dis_mean_lo += weight * (ldis_lo + rdis_lo);
            dis_mean_hi += weight * (ldis_hi + rdis_hi);
            ref_sq_lo += weight * (lref_lo * lref_lo + rref_lo * rref_lo);
            ref_sq_hi += weight * (lref_hi * lref_hi + rref_hi * rref_hi);
            dis_sq_lo += weight * (ldis_lo * ldis_lo + rdis_lo * rdis_lo);
            dis_sq_hi += weight * (ldis_hi * ldis_hi + rdis_hi * rdis_hi);
            ref_dis_lo += weight * (lref_lo * ldis_lo + rref_lo * rdis_lo);
            ref_dis_hi += weight * (lref_hi * ldis_hi + rref_hi * rdis_hi);
        }
        let rounding = u32x8::splat(token, 128);
        let mean_lo_out = (ref_mean_lo + rounding).shr_logical_uniform(8).to_array();
        let mean_hi_out = (ref_mean_hi + rounding).shr_logical_uniform(8).to_array();
        let dis_lo_out = (dis_mean_lo + rounding).shr_logical_uniform(8).to_array();
        let dis_hi_out = (dis_mean_hi + rounding).shr_logical_uniform(8).to_array();
        let ref_sq_lo_out = ref_sq_lo.to_array();
        let ref_sq_hi_out = ref_sq_hi.to_array();
        let dis_sq_lo_out = dis_sq_lo.to_array();
        let dis_sq_hi_out = dis_sq_hi.to_array();
        let ref_dis_lo_out = ref_dis_lo.to_array();
        let ref_dis_hi_out = ref_dis_hi.to_array();
        for lane in 0..8 {
            let slot = col + 8 + lane;
            vertical_ref_mean[slot] = mean_lo_out[lane] as u16 as u32;
            vertical_dis_mean[slot] = dis_lo_out[lane] as u16 as u32;
            vertical_ref_sq[slot] = ref_sq_lo_out[lane];
            vertical_dis_sq[slot] = dis_sq_lo_out[lane];
            vertical_ref_dis[slot] = ref_dis_lo_out[lane];
        }
        for lane in 0..8 {
            let slot = col + 16 + lane;
            vertical_ref_mean[slot] = mean_hi_out[lane] as u16 as u32;
            vertical_dis_mean[slot] = dis_hi_out[lane] as u16 as u32;
            vertical_ref_sq[slot] = ref_sq_hi_out[lane];
            vertical_dis_sq[slot] = dis_sq_hi_out[lane];
            vertical_ref_dis[slot] = ref_dis_hi_out[lane];
        }
    }
    chunks * 16
}

#[cfg(feature = "simd")]
#[magetypes(define(u16x16, u32x8), v3, neon, wasm128, scalar)]
fn vif_vertical_u10_simd(
    token: Token,
    reference: &[u16],
    distorted: &[u16],
    row_offsets: &[usize; 17],
    width: usize,
    vertical_ref_mean: &mut [u32],
    vertical_dis_mean: &mut [u32],
    vertical_ref_sq: &mut [u32],
    vertical_dis_sq: &mut [u32],
    vertical_ref_dis: &mut [u32],
) -> usize {
    let filter = FILTERS[0];
    let chunks = width / 16;
    let mask16 = u32x8::splat(token, 0xffff);
    for chunk in 0..chunks {
        let col = chunk * 16;
        let mut ref_mean_lo = u32x8::zero(token);
        let mut ref_mean_hi = u32x8::zero(token);
        let mut dis_mean_lo = u32x8::zero(token);
        let mut dis_mean_hi = u32x8::zero(token);
        let mut ref_sq_lo16 = u32x8::zero(token);
        let mut ref_sq_hi16 = u32x8::zero(token);
        let mut ref_sq_lo8 = u32x8::zero(token);
        let mut ref_sq_hi8 = u32x8::zero(token);
        let mut dis_sq_lo16 = u32x8::zero(token);
        let mut dis_sq_hi16 = u32x8::zero(token);
        let mut dis_sq_lo8 = u32x8::zero(token);
        let mut dis_sq_hi8 = u32x8::zero(token);
        let mut ref_dis_lo16 = u32x8::zero(token);
        let mut ref_dis_hi16 = u32x8::zero(token);
        let mut ref_dis_lo8 = u32x8::zero(token);
        let mut ref_dis_hi8 = u32x8::zero(token);
        let weight = u32x8::splat(token, filter[8] as u32);
        let center_ref = u16x16::load(
            token,
            reference[row_offsets[8] + col..row_offsets[8] + col + 16]
                .try_into()
                .unwrap(),
        );
        let center_dis = u16x16::load(
            token,
            distorted[row_offsets[8] + col..row_offsets[8] + col + 16]
                .try_into()
                .unwrap(),
        );
        let crlo = center_ref.widen_low();
        let crhi = center_ref.widen_high();
        let cdlo = center_dis.widen_low();
        let cdhi = center_dis.widen_high();
        ref_mean_lo += weight * crlo;
        ref_mean_hi += weight * crhi;
        dis_mean_lo += weight * cdlo;
        dis_mean_hi += weight * cdhi;
        let prod_sq_lo = crlo * crlo;
        let prod_sq_hi = crhi * crhi;
        ref_sq_lo16 += weight * (prod_sq_lo & mask16);
        ref_sq_hi16 += weight * prod_sq_lo.shr_logical_uniform(16);
        ref_sq_lo8 += weight * (prod_sq_hi & mask16);
        ref_sq_hi8 += weight * prod_sq_hi.shr_logical_uniform(16);
        let prod_dsq_lo = cdlo * cdlo;
        let prod_dsq_hi = cdhi * cdhi;
        dis_sq_lo16 += weight * (prod_dsq_lo & mask16);
        dis_sq_hi16 += weight * prod_dsq_lo.shr_logical_uniform(16);
        dis_sq_lo8 += weight * (prod_dsq_hi & mask16);
        dis_sq_hi8 += weight * prod_dsq_hi.shr_logical_uniform(16);
        let prod_rd_lo = crlo * cdlo;
        let prod_rd_hi = crhi * cdhi;
        ref_dis_lo16 += weight * (prod_rd_lo & mask16);
        ref_dis_hi16 += weight * prod_rd_lo.shr_logical_uniform(16);
        ref_dis_lo8 += weight * (prod_rd_hi & mask16);
        ref_dis_hi8 += weight * prod_rd_hi.shr_logical_uniform(16);
        for offset in 1..=8usize {
            let weight = u32x8::splat(token, filter[8 - offset] as u32);
            let left_ref = u16x16::load(
                token,
                reference[row_offsets[8 - offset] + col..row_offsets[8 - offset] + col + 16]
                    .try_into()
                    .unwrap(),
            );
            let right_ref = u16x16::load(
                token,
                reference[row_offsets[8 + offset] + col..row_offsets[8 + offset] + col + 16]
                    .try_into()
                    .unwrap(),
            );
            let left_dis = u16x16::load(
                token,
                distorted[row_offsets[8 - offset] + col..row_offsets[8 - offset] + col + 16]
                    .try_into()
                    .unwrap(),
            );
            let right_dis = u16x16::load(
                token,
                distorted[row_offsets[8 + offset] + col..row_offsets[8 + offset] + col + 16]
                    .try_into()
                    .unwrap(),
            );
            let lrlo = left_ref.widen_low();
            let lrhi = left_ref.widen_high();
            let rrlo = right_ref.widen_low();
            let rrhi = right_ref.widen_high();
            let ldlo = left_dis.widen_low();
            let ldhi = left_dis.widen_high();
            let rdlo = right_dis.widen_low();
            let rdhi = right_dis.widen_high();
            ref_mean_lo += weight * (lrlo + rrlo);
            ref_mean_hi += weight * (lrhi + rrhi);
            dis_mean_lo += weight * (ldlo + rdlo);
            dis_mean_hi += weight * (ldhi + rdhi);
            let prod_sq_lo = lrlo * lrlo + rrlo * rrlo;
            let prod_sq_hi = lrhi * lrhi + rrhi * rrhi;
            ref_sq_lo16 += weight * (prod_sq_lo & mask16);
            ref_sq_hi16 += weight * prod_sq_lo.shr_logical_uniform(16);
            ref_sq_lo8 += weight * (prod_sq_hi & mask16);
            ref_sq_hi8 += weight * prod_sq_hi.shr_logical_uniform(16);
            let prod_dsq_lo = ldlo * ldlo + rdlo * rdlo;
            let prod_dsq_hi = ldhi * ldhi + rdhi * rdhi;
            dis_sq_lo16 += weight * (prod_dsq_lo & mask16);
            dis_sq_hi16 += weight * prod_dsq_lo.shr_logical_uniform(16);
            dis_sq_lo8 += weight * (prod_dsq_hi & mask16);
            dis_sq_hi8 += weight * prod_dsq_hi.shr_logical_uniform(16);
            let prod_rd_lo = lrlo * ldlo + rrlo * rdlo;
            let prod_rd_hi = lrhi * ldhi + rrhi * rdhi;
            ref_dis_lo16 += weight * (prod_rd_lo & mask16);
            ref_dis_hi16 += weight * prod_rd_lo.shr_logical_uniform(16);
            ref_dis_lo8 += weight * (prod_rd_hi & mask16);
            ref_dis_hi8 += weight * prod_rd_hi.shr_logical_uniform(16);
        }
        let rounding = u32x8::splat(token, 512);
        let mean_lo_out = (ref_mean_lo + rounding).shr_logical_uniform(10).to_array();
        let mean_hi_out = (ref_mean_hi + rounding).shr_logical_uniform(10).to_array();
        let dis_lo_out = (dis_mean_lo + rounding).shr_logical_uniform(10).to_array();
        let dis_hi_out = (dis_mean_hi + rounding).shr_logical_uniform(10).to_array();
        let sq_lo16 = ref_sq_lo16.to_array();
        let sq_hi16 = ref_sq_hi16.to_array();
        let sq_lo8 = ref_sq_lo8.to_array();
        let sq_hi8 = ref_sq_hi8.to_array();
        let dsq_lo16 = dis_sq_lo16.to_array();
        let dsq_hi16 = dis_sq_hi16.to_array();
        let dsq_lo8 = dis_sq_lo8.to_array();
        let dsq_hi8 = dis_sq_hi8.to_array();
        let rd_lo16 = ref_dis_lo16.to_array();
        let rd_hi16 = ref_dis_hi16.to_array();
        let rd_lo8 = ref_dis_lo8.to_array();
        let rd_hi8 = ref_dis_hi8.to_array();
        for lane in 0..8 {
            let slot = col + 8 + lane;
            vertical_ref_mean[slot] = mean_lo_out[lane] as u16 as u32;
            vertical_dis_mean[slot] = dis_lo_out[lane] as u16 as u32;
            let sq = sq_lo16[lane] as u64 + ((sq_hi16[lane] as u64) << 16);
            vertical_ref_sq[slot] = ((sq + 8) >> 4) as u32;
            let dsq = dsq_lo16[lane] as u64 + ((dsq_hi16[lane] as u64) << 16);
            vertical_dis_sq[slot] = ((dsq + 8) >> 4) as u32;
            let rd = rd_lo16[lane] as u64 + ((rd_hi16[lane] as u64) << 16);
            vertical_ref_dis[slot] = ((rd + 8) >> 4) as u32;
        }
        for lane in 0..8 {
            let slot = col + 16 + lane;
            vertical_ref_mean[slot] = mean_hi_out[lane] as u16 as u32;
            vertical_dis_mean[slot] = dis_hi_out[lane] as u16 as u32;
            let sq = sq_lo8[lane] as u64 + ((sq_hi8[lane] as u64) << 16);
            vertical_ref_sq[slot] = ((sq + 8) >> 4) as u32;
            let dsq = dsq_lo8[lane] as u64 + ((dsq_hi8[lane] as u64) << 16);
            vertical_dis_sq[slot] = ((dsq + 8) >> 4) as u32;
            let rd = rd_lo8[lane] as u64 + ((rd_hi8[lane] as u64) << 16);
            vertical_ref_dis[slot] = ((rd + 8) >> 4) as u32;
        }
    }
    chunks * 16
}

#[cfg(feature = "simd")]
#[magetypes(define(u16x16, u32x8), v3, neon, wasm128, scalar)]
fn vif_vertical_hiscale_simd(
    token: Token,
    reference: &[u16],
    distorted: &[u16],
    row_offsets: &[usize; 17],
    width: usize,
    scale: usize,
    vertical_ref_mean: &mut [u32],
    vertical_dis_mean: &mut [u32],
    vertical_ref_sq: &mut [u32],
    vertical_dis_sq: &mut [u32],
    vertical_ref_dis: &mut [u32],
) -> usize {
    let filter = FILTERS[scale];
    let half = filter.len() / 2;
    let chunks = width / 16;
    let mask16 = u32x8::splat(token, 0xffff);
    for chunk in 0..chunks {
        let col = chunk * 16;
        let mut ref_mean_lo = u32x8::zero(token);
        let mut ref_mean_hi = u32x8::zero(token);
        let mut dis_mean_lo = u32x8::zero(token);
        let mut dis_mean_hi = u32x8::zero(token);
        let mut ref_sq_l_lo16 = u32x8::zero(token);
        let mut ref_sq_r_lo16 = u32x8::zero(token);
        let mut ref_sq_l_hi16 = u32x8::zero(token);
        let mut ref_sq_r_hi16 = u32x8::zero(token);
        let mut ref_sq_l_lo8 = u32x8::zero(token);
        let mut ref_sq_r_lo8 = u32x8::zero(token);
        let mut ref_sq_l_hi8 = u32x8::zero(token);
        let mut ref_sq_r_hi8 = u32x8::zero(token);
        let mut dis_sq_l_lo16 = u32x8::zero(token);
        let mut dis_sq_r_lo16 = u32x8::zero(token);
        let mut dis_sq_l_hi16 = u32x8::zero(token);
        let mut dis_sq_r_hi16 = u32x8::zero(token);
        let mut dis_sq_l_lo8 = u32x8::zero(token);
        let mut dis_sq_r_lo8 = u32x8::zero(token);
        let mut dis_sq_l_hi8 = u32x8::zero(token);
        let mut dis_sq_r_hi8 = u32x8::zero(token);
        let mut ref_dis_l_lo16 = u32x8::zero(token);
        let mut ref_dis_r_lo16 = u32x8::zero(token);
        let mut ref_dis_l_hi16 = u32x8::zero(token);
        let mut ref_dis_r_hi16 = u32x8::zero(token);
        let mut ref_dis_l_lo8 = u32x8::zero(token);
        let mut ref_dis_r_lo8 = u32x8::zero(token);
        let mut ref_dis_l_hi8 = u32x8::zero(token);
        let mut ref_dis_r_hi8 = u32x8::zero(token);
        let weight = u32x8::splat(token, filter[half] as u32);
        let center_ref = u16x16::load(
            token,
            reference[row_offsets[half] + col..row_offsets[half] + col + 16]
                .try_into()
                .unwrap(),
        );
        let center_dis = u16x16::load(
            token,
            distorted[row_offsets[half] + col..row_offsets[half] + col + 16]
                .try_into()
                .unwrap(),
        );
        let crlo = center_ref.widen_low();
        let crhi = center_ref.widen_high();
        let cdlo = center_dis.widen_low();
        let cdhi = center_dis.widen_high();
        ref_mean_lo += weight * crlo;
        ref_mean_hi += weight * crhi;
        dis_mean_lo += weight * cdlo;
        dis_mean_hi += weight * cdhi;
        let prod_sq_lo = crlo * crlo;
        let prod_sq_hi = crhi * crhi;
        ref_sq_l_lo16 += weight * (prod_sq_lo & mask16);
        ref_sq_l_hi16 += weight * prod_sq_lo.shr_logical_uniform(16);
        ref_sq_l_lo8 += weight * (prod_sq_hi & mask16);
        ref_sq_l_hi8 += weight * prod_sq_hi.shr_logical_uniform(16);
        let prod_dsq_lo = cdlo * cdlo;
        let prod_dsq_hi = cdhi * cdhi;
        dis_sq_l_lo16 += weight * (prod_dsq_lo & mask16);
        dis_sq_l_hi16 += weight * prod_dsq_lo.shr_logical_uniform(16);
        dis_sq_l_lo8 += weight * (prod_dsq_hi & mask16);
        dis_sq_l_hi8 += weight * prod_dsq_hi.shr_logical_uniform(16);
        let prod_rd_lo = crlo * cdlo;
        let prod_rd_hi = crhi * cdhi;
        ref_dis_l_lo16 += weight * (prod_rd_lo & mask16);
        ref_dis_l_hi16 += weight * prod_rd_lo.shr_logical_uniform(16);
        ref_dis_l_lo8 += weight * (prod_rd_hi & mask16);
        ref_dis_l_hi8 += weight * prod_rd_hi.shr_logical_uniform(16);
        for offset in 1..=half {
            let weight = u32x8::splat(token, filter[half - offset] as u32);
            let left_ref = u16x16::load(
                token,
                reference[row_offsets[half - offset] + col..row_offsets[half - offset] + col + 16]
                    .try_into()
                    .unwrap(),
            );
            let right_ref = u16x16::load(
                token,
                reference[row_offsets[half + offset] + col..row_offsets[half + offset] + col + 16]
                    .try_into()
                    .unwrap(),
            );
            let left_dis = u16x16::load(
                token,
                distorted[row_offsets[half - offset] + col..row_offsets[half - offset] + col + 16]
                    .try_into()
                    .unwrap(),
            );
            let right_dis = u16x16::load(
                token,
                distorted[row_offsets[half + offset] + col..row_offsets[half + offset] + col + 16]
                    .try_into()
                    .unwrap(),
            );
            let lrlo = left_ref.widen_low();
            let lrhi = left_ref.widen_high();
            let rrlo = right_ref.widen_low();
            let rrhi = right_ref.widen_high();
            let ldlo = left_dis.widen_low();
            let ldhi = left_dis.widen_high();
            let rdlo = right_dis.widen_low();
            let rdhi = right_dis.widen_high();
            ref_mean_lo += weight * (lrlo + rrlo);
            ref_mean_hi += weight * (lrhi + rrhi);
            dis_mean_lo += weight * (ldlo + rdlo);
            dis_mean_hi += weight * (ldhi + rdhi);
            let sq_lo = lrlo * lrlo;
            let sq_hi = lrhi * lrhi;
            let sq2_lo = rrlo * rrlo;
            let sq2_hi = rrhi * rrhi;
            ref_sq_l_lo16 += weight * (sq_lo & mask16);
            ref_sq_l_hi16 += weight * sq_lo.shr_logical_uniform(16);
            ref_sq_r_lo16 += weight * (sq2_lo & mask16);
            ref_sq_r_hi16 += weight * sq2_lo.shr_logical_uniform(16);
            ref_sq_l_lo8 += weight * (sq_hi & mask16);
            ref_sq_l_hi8 += weight * sq_hi.shr_logical_uniform(16);
            ref_sq_r_lo8 += weight * (sq2_hi & mask16);
            ref_sq_r_hi8 += weight * sq2_hi.shr_logical_uniform(16);
            let ds_lo = ldlo * ldlo;
            let ds_hi = ldhi * ldhi;
            let ds2_lo = rdlo * rdlo;
            let ds2_hi = rdhi * rdhi;
            dis_sq_l_lo16 += weight * (ds_lo & mask16);
            dis_sq_l_hi16 += weight * ds_lo.shr_logical_uniform(16);
            dis_sq_r_lo16 += weight * (ds2_lo & mask16);
            dis_sq_r_hi16 += weight * ds2_lo.shr_logical_uniform(16);
            dis_sq_l_lo8 += weight * (ds_hi & mask16);
            dis_sq_l_hi8 += weight * ds_hi.shr_logical_uniform(16);
            dis_sq_r_lo8 += weight * (ds2_hi & mask16);
            dis_sq_r_hi8 += weight * ds2_hi.shr_logical_uniform(16);
            let rd_lo = lrlo * ldlo;
            let rd_hi = lrhi * ldhi;
            let rd2_lo = rrlo * rdlo;
            let rd2_hi = rrhi * rdhi;
            ref_dis_l_lo16 += weight * (rd_lo & mask16);
            ref_dis_l_hi16 += weight * rd_lo.shr_logical_uniform(16);
            ref_dis_r_lo16 += weight * (rd2_lo & mask16);
            ref_dis_r_hi16 += weight * rd2_lo.shr_logical_uniform(16);
            ref_dis_l_lo8 += weight * (rd_hi & mask16);
            ref_dis_l_hi8 += weight * rd_hi.shr_logical_uniform(16);
            ref_dis_r_lo8 += weight * (rd2_hi & mask16);
            ref_dis_r_hi8 += weight * rd2_hi.shr_logical_uniform(16);
        }
        let rounding = u32x8::splat(token, 32768);
        let mean_lo_out = (ref_mean_lo + rounding).shr_logical_uniform(16).to_array();
        let mean_hi_out = (ref_mean_hi + rounding).shr_logical_uniform(16).to_array();
        let dis_lo_out = (dis_mean_lo + rounding).shr_logical_uniform(16).to_array();
        let dis_hi_out = (dis_mean_hi + rounding).shr_logical_uniform(16).to_array();
        let sq_l_lo16 = ref_sq_l_lo16.to_array();
        let sq_r_lo16 = ref_sq_r_lo16.to_array();
        let sq_l_hi16 = ref_sq_l_hi16.to_array();
        let sq_r_hi16 = ref_sq_r_hi16.to_array();
        let sq_l_lo8 = ref_sq_l_lo8.to_array();
        let sq_r_lo8 = ref_sq_r_lo8.to_array();
        let sq_l_hi8 = ref_sq_l_hi8.to_array();
        let sq_r_hi8 = ref_sq_r_hi8.to_array();
        let dsq_l_lo16 = dis_sq_l_lo16.to_array();
        let dsq_r_lo16 = dis_sq_r_lo16.to_array();
        let dsq_l_hi16 = dis_sq_l_hi16.to_array();
        let dsq_r_hi16 = dis_sq_r_hi16.to_array();
        let dsq_l_lo8 = dis_sq_l_lo8.to_array();
        let dsq_r_lo8 = dis_sq_r_lo8.to_array();
        let dsq_l_hi8 = dis_sq_l_hi8.to_array();
        let dsq_r_hi8 = dis_sq_r_hi8.to_array();
        let rd_l_lo16 = ref_dis_l_lo16.to_array();
        let rd_r_lo16 = ref_dis_r_lo16.to_array();
        let rd_l_hi16 = ref_dis_l_hi16.to_array();
        let rd_r_hi16 = ref_dis_r_hi16.to_array();
        let rd_l_lo8 = ref_dis_l_lo8.to_array();
        let rd_r_lo8 = ref_dis_r_lo8.to_array();
        let rd_l_hi8 = ref_dis_l_hi8.to_array();
        let rd_r_hi8 = ref_dis_r_hi8.to_array();
        for lane in 0..8 {
            let slot = col + half + lane;
            vertical_ref_mean[slot] = mean_lo_out[lane] as u16 as u32;
            vertical_dis_mean[slot] = dis_lo_out[lane] as u16 as u32;
            let sq = sq_l_lo16[lane] as u64
                + sq_r_lo16[lane] as u64
                + (((sq_l_hi16[lane] as u64) + (sq_r_hi16[lane] as u64)) << 16);
            vertical_ref_sq[slot] = ((sq + 32768) >> 16) as u32;
            let dsq = dsq_l_lo16[lane] as u64
                + dsq_r_lo16[lane] as u64
                + (((dsq_l_hi16[lane] as u64) + (dsq_r_hi16[lane] as u64)) << 16);
            vertical_dis_sq[slot] = ((dsq + 32768) >> 16) as u32;
            let rd = rd_l_lo16[lane] as u64
                + rd_r_lo16[lane] as u64
                + (((rd_l_hi16[lane] as u64) + (rd_r_hi16[lane] as u64)) << 16);
            vertical_ref_dis[slot] = ((rd + 32768) >> 16) as u32;
        }
        for lane in 0..8 {
            let slot = col + half + 8 + lane;
            vertical_ref_mean[slot] = mean_hi_out[lane] as u16 as u32;
            vertical_dis_mean[slot] = dis_hi_out[lane] as u16 as u32;
            let sq = sq_l_lo8[lane] as u64
                + sq_r_lo8[lane] as u64
                + (((sq_l_hi8[lane] as u64) + (sq_r_hi8[lane] as u64)) << 16);
            vertical_ref_sq[slot] = ((sq + 32768) >> 16) as u32;
            let dsq = dsq_l_lo8[lane] as u64
                + dsq_r_lo8[lane] as u64
                + (((dsq_l_hi8[lane] as u64) + (dsq_r_hi8[lane] as u64)) << 16);
            vertical_dis_sq[slot] = ((dsq + 32768) >> 16) as u32;
            let rd = rd_l_lo8[lane] as u64
                + rd_r_lo8[lane] as u64
                + (((rd_l_hi8[lane] as u64) + (rd_r_hi8[lane] as u64)) << 16);
            vertical_ref_dis[slot] = ((rd + 32768) >> 16) as u32;
        }
    }
    chunks * 16
}

#[cfg(feature = "simd")]
type VifHorizontalSums = ([u32; 16], [u32; 16], [u32; 16], [u32; 16], [u32; 16]);

#[cfg(feature = "simd")]
#[magetypes(define(u32x8), v3, neon, wasm128, scalar)]
fn vif_horizontal_sums(
    token: Token,
    vertical_ref_mean: &[u32],
    vertical_dis_mean: &[u32],
    vertical_ref_sq: &[u32],
    vertical_dis_sq: &[u32],
    vertical_ref_dis: &[u32],
    filter: &[u16],
    col: usize,
) -> VifHorizontalSums {
    let mask16 = u32x8::splat(token, 0xffff);
    let mut ref_mean_lo = u32x8::zero(token);
    let mut ref_mean_hi = u32x8::zero(token);
    let mut dis_mean_lo = u32x8::zero(token);
    let mut dis_mean_hi = u32x8::zero(token);
    let mut ref_sq_lo16 = u32x8::zero(token);
    let mut ref_sq_hi16 = u32x8::zero(token);
    let mut ref_sq_lo8 = u32x8::zero(token);
    let mut ref_sq_hi8 = u32x8::zero(token);
    let mut dis_sq_lo16 = u32x8::zero(token);
    let mut dis_sq_hi16 = u32x8::zero(token);
    let mut dis_sq_lo8 = u32x8::zero(token);
    let mut dis_sq_hi8 = u32x8::zero(token);
    let mut ref_dis_lo16 = u32x8::zero(token);
    let mut ref_dis_hi16 = u32x8::zero(token);
    let mut ref_dis_lo8 = u32x8::zero(token);
    let mut ref_dis_hi8 = u32x8::zero(token);
    for (tap, &w) in filter.iter().enumerate() {
        let weight = u32x8::splat(token, w as u32);
        let rlo = u32x8::load(
            token,
            vertical_ref_mean[col + tap..col + tap + 8]
                .try_into()
                .unwrap(),
        );
        let rhi = u32x8::load(
            token,
            vertical_ref_mean[col + 8 + tap..col + 8 + tap + 8]
                .try_into()
                .unwrap(),
        );
        let dlo = u32x8::load(
            token,
            vertical_dis_mean[col + tap..col + tap + 8]
                .try_into()
                .unwrap(),
        );
        let dhi = u32x8::load(
            token,
            vertical_dis_mean[col + 8 + tap..col + 8 + tap + 8]
                .try_into()
                .unwrap(),
        );
        ref_mean_lo += weight * rlo;
        ref_mean_hi += weight * rhi;
        dis_mean_lo += weight * dlo;
        dis_mean_hi += weight * dhi;
        let sq_lo = u32x8::load(
            token,
            vertical_ref_sq[col + tap..col + tap + 8]
                .try_into()
                .unwrap(),
        );
        let sq_hi = u32x8::load(
            token,
            vertical_ref_sq[col + 8 + tap..col + 8 + tap + 8]
                .try_into()
                .unwrap(),
        );
        ref_sq_lo16 += weight * (sq_lo & mask16);
        ref_sq_hi16 += weight * sq_lo.shr_logical_uniform(16);
        ref_sq_lo8 += weight * (sq_hi & mask16);
        ref_sq_hi8 += weight * sq_hi.shr_logical_uniform(16);
        let ds_lo = u32x8::load(
            token,
            vertical_dis_sq[col + tap..col + tap + 8]
                .try_into()
                .unwrap(),
        );
        let ds_hi = u32x8::load(
            token,
            vertical_dis_sq[col + 8 + tap..col + 8 + tap + 8]
                .try_into()
                .unwrap(),
        );
        dis_sq_lo16 += weight * (ds_lo & mask16);
        dis_sq_hi16 += weight * ds_lo.shr_logical_uniform(16);
        dis_sq_lo8 += weight * (ds_hi & mask16);
        dis_sq_hi8 += weight * ds_hi.shr_logical_uniform(16);
        let rd_lo = u32x8::load(
            token,
            vertical_ref_dis[col + tap..col + tap + 8]
                .try_into()
                .unwrap(),
        );
        let rd_hi = u32x8::load(
            token,
            vertical_ref_dis[col + 8 + tap..col + 8 + tap + 8]
                .try_into()
                .unwrap(),
        );
        ref_dis_lo16 += weight * (rd_lo & mask16);
        ref_dis_hi16 += weight * rd_lo.shr_logical_uniform(16);
        ref_dis_lo8 += weight * (rd_hi & mask16);
        ref_dis_hi8 += weight * rd_hi.shr_logical_uniform(16);
    }
    let mut ref_mean_out = [0u32; 16];
    let mut dis_mean_out = [0u32; 16];
    let mut ref_sq_out = [0u32; 16];
    let mut dis_sq_out = [0u32; 16];
    let mut ref_dis_out = [0u32; 16];
    let mean_lo = ref_mean_lo.to_array();
    let mean_hi = ref_mean_hi.to_array();
    let dmean_lo = dis_mean_lo.to_array();
    let dmean_hi = dis_mean_hi.to_array();
    let rounding = u32x8::splat(token, 32768);
    let sq_lo16 = (ref_sq_lo16 + rounding).shr_logical_uniform(16).to_array();
    let sq_hi16 = ref_sq_hi16.to_array();
    let sq_lo8 = (ref_sq_lo8 + rounding).shr_logical_uniform(16).to_array();
    let sq_hi8 = ref_sq_hi8.to_array();
    let dsq_lo16 = (dis_sq_lo16 + rounding).shr_logical_uniform(16).to_array();
    let dsq_hi16 = dis_sq_hi16.to_array();
    let dsq_lo8 = (dis_sq_lo8 + rounding).shr_logical_uniform(16).to_array();
    let dsq_hi8 = dis_sq_hi8.to_array();
    let rd_lo16 = (ref_dis_lo16 + rounding).shr_logical_uniform(16).to_array();
    let rd_hi16 = ref_dis_hi16.to_array();
    let rd_lo8 = (ref_dis_lo8 + rounding).shr_logical_uniform(16).to_array();
    let rd_hi8 = ref_dis_hi8.to_array();
    for lane in 0..8 {
        ref_mean_out[lane] = mean_lo[lane];
        ref_mean_out[8 + lane] = mean_hi[lane];
        dis_mean_out[lane] = dmean_lo[lane];
        dis_mean_out[8 + lane] = dmean_hi[lane];
        ref_sq_out[lane] = sq_lo16[lane] + sq_hi16[lane];
        ref_sq_out[8 + lane] = sq_lo8[lane] + sq_hi8[lane];
        dis_sq_out[lane] = dsq_lo16[lane] + dsq_hi16[lane];
        dis_sq_out[8 + lane] = dsq_lo8[lane] + dsq_hi8[lane];
        ref_dis_out[lane] = rd_lo16[lane] + rd_hi16[lane];
        ref_dis_out[8 + lane] = rd_lo8[lane] + rd_hi8[lane];
    }
    (
        ref_mean_out,
        dis_mean_out,
        ref_sq_out,
        dis_sq_out,
        ref_dis_out,
    )
}

#[cfg(feature = "simd")]
#[magetypes(define(u16x16, u32x8), v3, neon, wasm128, scalar)]
fn vif_subsample_vertical_simd(
    token: Token,
    reference: &[u16],
    distorted: &[u16],
    row_offsets: &[usize],
    width: usize,
    filter: &[u16],
    shift: u32,
    round: u32,
    vertical_reference: &mut [u32],
    vertical_distorted: &mut [u32],
) -> usize {
    let half = filter.len() / 2;
    let chunks = width / 16;
    let rounding = u32x8::splat(token, round);
    for chunk in 0..chunks {
        let col = chunk * 16;
        let mut ref_lo = u32x8::zero(token);
        let mut ref_hi = u32x8::zero(token);
        let mut dis_lo = u32x8::zero(token);
        let mut dis_hi = u32x8::zero(token);
        let weight = u32x8::splat(token, filter[half] as u32);
        let center_ref = u16x16::load(
            token,
            reference[row_offsets[half] + col..row_offsets[half] + col + 16]
                .try_into()
                .unwrap(),
        );
        let center_dis = u16x16::load(
            token,
            distorted[row_offsets[half] + col..row_offsets[half] + col + 16]
                .try_into()
                .unwrap(),
        );
        ref_lo += weight * center_ref.widen_low();
        ref_hi += weight * center_ref.widen_high();
        dis_lo += weight * center_dis.widen_low();
        dis_hi += weight * center_dis.widen_high();
        for offset in 1..=half {
            let weight = u32x8::splat(token, filter[half - offset] as u32);
            let left_ref = u16x16::load(
                token,
                reference[row_offsets[half - offset] + col..row_offsets[half - offset] + col + 16]
                    .try_into()
                    .unwrap(),
            );
            let right_ref = u16x16::load(
                token,
                reference[row_offsets[half + offset] + col..row_offsets[half + offset] + col + 16]
                    .try_into()
                    .unwrap(),
            );
            let left_dis = u16x16::load(
                token,
                distorted[row_offsets[half - offset] + col..row_offsets[half - offset] + col + 16]
                    .try_into()
                    .unwrap(),
            );
            let right_dis = u16x16::load(
                token,
                distorted[row_offsets[half + offset] + col..row_offsets[half + offset] + col + 16]
                    .try_into()
                    .unwrap(),
            );
            ref_lo += weight * (left_ref.widen_low() + right_ref.widen_low());
            ref_hi += weight * (left_ref.widen_high() + right_ref.widen_high());
            dis_lo += weight * (left_dis.widen_low() + right_dis.widen_low());
            dis_hi += weight * (left_dis.widen_high() + right_dis.widen_high());
        }
        let ref_lo_out = (ref_lo + rounding).shr_logical_uniform(shift).to_array();
        let ref_hi_out = (ref_hi + rounding).shr_logical_uniform(shift).to_array();
        let dis_lo_out = (dis_lo + rounding).shr_logical_uniform(shift).to_array();
        let dis_hi_out = (dis_hi + rounding).shr_logical_uniform(shift).to_array();
        for lane in 0..8 {
            vertical_reference[col + half + lane] = ref_lo_out[lane];
            vertical_distorted[col + half + lane] = dis_lo_out[lane];
        }
        for lane in 0..8 {
            vertical_reference[col + half + 8 + lane] = ref_hi_out[lane];
            vertical_distorted[col + half + 8 + lane] = dis_hi_out[lane];
        }
    }
    chunks * 16
}

/// Direct port of `vif_subsample_rd_16_avx2`'s vertical pass, covering the
/// `vif_subsample_rd_8_avx2` case as well: loading the u16 plane directly is
/// equivalent to `cvtepu8_epi16` for 8-bit inputs, and the `(shift, round)`
/// parameters reproduce both rounding tables. Processes 16 columns per
/// `vif_subsample_rd_8_avx512` vertical port: 32 u16 lanes/iter with the
/// filter folded into tap *pairs* — `madd(unpacklo(v_t, v_t+1), f_pair)` adds
/// both taps' products in one op. Only safe when inputs fit i16 (8-bit scale-0
/// guarantees v ≤ 255; scale≥1 planes reach ~65280 and must use the u16 path).
/// Odd `fwidth` is padded with a zero coefficient against a safe mirrored row.
#[cfg(all(feature = "simd", feature = "avx512", target_arch = "x86_64"))]
#[arcane(import_intrinsics)]
#[allow(clippy::too_many_arguments)]
fn vif_subsample_vertical_v4(
    _token: X64V4Token,
    reference: &[u16],
    distorted: &[u16],
    row_offsets: &[usize],
    width: usize,
    filter: &[u16],
    shift: u32,
    round: u32,
    vertical_reference: &mut [u32],
    vertical_distorted: &mut [u32],
) -> usize {
    let half = filter.len() / 2;
    let n = width >> 5;
    let addnum = _mm512_set1_epi32(round as i32);
    let shift_xmm = _mm_cvtsi32_si128(shift as i32);
    let perm_lo = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
    let perm_hi = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
    // Padded pair coefficients: f_pair[t/2] = f[t] | f[t+1]<<16 (zero when the
    // partner tap doesn't exist — its row is read but multiplied by 0).
    let npairs = filter.len().div_ceil(2);
    let mut fpairs = [0i32; 9];
    for (pi, slot) in fpairs.iter_mut().enumerate().take(npairs) {
        let lo = filter[2 * pi] as i32;
        let hi = if 2 * pi + 1 < filter.len() {
            filter[2 * pi + 1] as i32
        } else {
            0
        };
        *slot = lo | (hi << 16);
    }
    for chunk in 0..n {
        let j = chunk * 32;
        let mut accumr_lo = _mm512_setzero_si512();
        let mut accumr_hi = _mm512_setzero_si512();
        let mut accumd_lo = _mm512_setzero_si512();
        let mut accumd_hi = _mm512_setzero_si512();
        for (pi, &fc) in fpairs.iter().enumerate().take(npairs) {
            let fp = _mm512_set1_epi32(fc);
            let off0 = row_offsets[2 * pi];
            let off1 = row_offsets[(2 * pi + 1).min(row_offsets.len() - 1)];
            let r0 = _mm512_loadu_si512(a8::<u16, 32>(&reference[off0 + j..off0 + j + 32]));
            let r1 = _mm512_loadu_si512(a8::<u16, 32>(&reference[off1 + j..off1 + j + 32]));
            let d0 = _mm512_loadu_si512(a8::<u16, 32>(&distorted[off0 + j..off0 + j + 32]));
            let d1 = _mm512_loadu_si512(a8::<u16, 32>(&distorted[off1 + j..off1 + j + 32]));
            accumr_lo = _mm512_add_epi32(
                accumr_lo,
                _mm512_madd_epi16(_mm512_unpacklo_epi16(r0, r1), fp),
            );
            accumr_hi = _mm512_add_epi32(
                accumr_hi,
                _mm512_madd_epi16(_mm512_unpackhi_epi16(r0, r1), fp),
            );
            accumd_lo = _mm512_add_epi32(
                accumd_lo,
                _mm512_madd_epi16(_mm512_unpacklo_epi16(d0, d1), fp),
            );
            accumd_hi = _mm512_add_epi32(
                accumd_hi,
                _mm512_madd_epi16(_mm512_unpackhi_epi16(d0, d1), fp),
            );
        }
        macro_rules! store_out {
            ($dst:expr, $lo:expr, $hi:expr) => {
                let lo = _mm512_srl_epi32(_mm512_add_epi32($lo, addnum), shift_xmm);
                let hi = _mm512_srl_epi32(_mm512_add_epi32($hi, addnum), shift_xmm);
                _mm512_storeu_si512(
                    a8m::<u32, 16>(&mut $dst[half + j..half + j + 16]),
                    _mm512_permutex2var_epi64(lo, perm_lo, hi),
                );
                _mm512_storeu_si512(
                    a8m::<u32, 16>(&mut $dst[half + j + 16..half + j + 32]),
                    _mm512_permutex2var_epi64(lo, perm_hi, hi),
                );
            };
        }
        store_out!(vertical_reference, accumr_lo, accumr_hi);
        store_out!(vertical_distorted, accumd_lo, accumd_hi);
    }
    n * 32
}

/// iteration, writes the convolved row at `half + j` in the padded buffers.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane(import_intrinsics)]
#[allow(clippy::too_many_arguments)]
fn vif_subsample_vertical_v3(
    _token: X64V3Token,
    reference: &[u16],
    distorted: &[u16],
    row_offsets: &[usize],
    width: usize,
    filter: &[u16],
    shift: u32,
    round: u32,
    vertical_reference: &mut [u32],
    vertical_distorted: &mut [u32],
) -> usize {
    let half = filter.len() / 2;
    let n = width >> 4;
    let addnum = _mm256_set1_epi32(round as i32);
    let shift_xmm = _mm_cvtsi32_si128(shift as i32);
    for chunk in 0..n {
        let j = chunk * 16;
        let mut accumr_lo = _mm256_setzero_si256();
        let mut accumr_hi = _mm256_setzero_si256();
        let mut accumd_lo = _mm256_setzero_si256();
        let mut accumd_hi = _mm256_setzero_si256();
        for (tap, &fc) in filter.iter().enumerate() {
            let f1 = _mm256_set1_epi16(fc as i16);
            let off = row_offsets[tap] + j;
            let ref1 = _mm256_loadu_si256(a8::<u16, 16>(&reference[off..off + 16]));
            let dis1 = _mm256_loadu_si256(a8::<u16, 16>(&distorted[off..off + 16]));
            let rlo = _mm256_mullo_epi16(ref1, f1);
            let rhi = _mm256_mulhi_epu16(ref1, f1);
            accumr_lo = _mm256_add_epi32(accumr_lo, _mm256_unpacklo_epi16(rlo, rhi));
            accumr_hi = _mm256_add_epi32(accumr_hi, _mm256_unpackhi_epi16(rlo, rhi));
            let dlo = _mm256_mullo_epi16(dis1, f1);
            let dhi = _mm256_mulhi_epu16(dis1, f1);
            accumd_lo = _mm256_add_epi32(accumd_lo, _mm256_unpacklo_epi16(dlo, dhi));
            accumd_hi = _mm256_add_epi32(accumd_hi, _mm256_unpackhi_epi16(dlo, dhi));
        }
        accumr_lo = _mm256_srl_epi32(_mm256_add_epi32(accumr_lo, addnum), shift_xmm);
        accumr_hi = _mm256_srl_epi32(_mm256_add_epi32(accumr_hi, addnum), shift_xmm);
        accumd_lo = _mm256_srl_epi32(_mm256_add_epi32(accumd_lo, addnum), shift_xmm);
        accumd_hi = _mm256_srl_epi32(_mm256_add_epi32(accumd_hi, addnum), shift_xmm);
        _mm256_storeu_si256(
            a8m::<u32, 8>(&mut vertical_reference[half + j..half + j + 8]),
            _mm256_permute2x128_si256(accumr_lo, accumr_hi, 0x20),
        );
        _mm256_storeu_si256(
            a8m::<u32, 8>(&mut vertical_reference[half + j + 8..half + j + 16]),
            _mm256_permute2x128_si256(accumr_lo, accumr_hi, 0x31),
        );
        _mm256_storeu_si256(
            a8m::<u32, 8>(&mut vertical_distorted[half + j..half + j + 8]),
            _mm256_permute2x128_si256(accumd_lo, accumd_hi, 0x20),
        );
        _mm256_storeu_si256(
            a8m::<u32, 8>(&mut vertical_distorted[half + j + 8..half + j + 16]),
            _mm256_permute2x128_si256(accumd_lo, accumd_hi, 0x31),
        );
    }
    n * 16
}

/// Direct port of the `vif_subsample_rd_*_avx2` horizontal passes. The
/// vertical buffers hold u32 columns whose high u16 half is always zero, so
/// the mullo/mulhi_epi16 + unpack products land each column's 32-bit product
/// in an even epi32 lane — the rd_8 `packus`/`permutevar8x32`/`packus`
/// sequence then extracts the even-column (decimated) results. Produces 8
/// decimated u16 outputs per call, covering input columns `col..col + 16`
/// at even positions, matching `vif_subsample_horizontal_simd`.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane(import_intrinsics)]
fn vif_subsample_horizontal_v3(
    _token: X64V3Token,
    vertical_reference: &[u32],
    vertical_distorted: &[u32],
    filter: &[u16],
    col: usize,
) -> ([u16; 8], [u16; 8]) {
    let mask1 = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
    let addnum = _mm256_set1_epi32(32768);
    let mut ref_parts = [_mm_setzero_si128(); 2];
    let mut dis_parts = [_mm_setzero_si128(); 2];
    for (part, j) in [col, col + 8].iter().enumerate() {
        let j = *j;
        let mut accumrlo = _mm256_setzero_si256();
        let mut accumrhi = _mm256_setzero_si256();
        let mut accumdlo = _mm256_setzero_si256();
        let mut accumdhi = _mm256_setzero_si256();
        for (tap, &fc) in filter.iter().enumerate() {
            let fcoeff = _mm256_set1_epi16(fc as i16);
            let idx = j + tap;
            let refconvol = _mm256_loadu_si256(a8::<u32, 8>(&vertical_reference[idx..idx + 8]));
            let rlo = _mm256_mullo_epi16(refconvol, fcoeff);
            let rhi = _mm256_mulhi_epu16(refconvol, fcoeff);
            accumrlo = _mm256_add_epi32(accumrlo, _mm256_unpacklo_epi16(rlo, rhi));
            accumrhi = _mm256_add_epi32(accumrhi, _mm256_unpackhi_epi16(rlo, rhi));
            let disconvol = _mm256_loadu_si256(a8::<u32, 8>(&vertical_distorted[idx..idx + 8]));
            let dlo = _mm256_mullo_epi16(disconvol, fcoeff);
            let dhi = _mm256_mulhi_epu16(disconvol, fcoeff);
            accumdlo = _mm256_add_epi32(accumdlo, _mm256_unpacklo_epi16(dlo, dhi));
            accumdhi = _mm256_add_epi32(accumdhi, _mm256_unpackhi_epi16(dlo, dhi));
        }
        accumrlo = _mm256_srli_epi32(_mm256_add_epi32(accumrlo, addnum), 16);
        accumrhi = _mm256_srli_epi32(_mm256_add_epi32(accumrhi, addnum), 16);
        accumdlo = _mm256_srli_epi32(_mm256_add_epi32(accumdlo, addnum), 16);
        accumdhi = _mm256_srli_epi32(_mm256_add_epi32(accumdhi, addnum), 16);
        let result_r = _mm256_packus_epi32(
            _mm256_permutevar8x32_epi32(_mm256_packus_epi32(accumrlo, accumrhi), mask1),
            _mm256_permutevar8x32_epi32(_mm256_packus_epi32(accumrlo, accumrhi), mask1),
        );
        let result_d = _mm256_packus_epi32(
            _mm256_permutevar8x32_epi32(_mm256_packus_epi32(accumdlo, accumdhi), mask1),
            _mm256_permutevar8x32_epi32(_mm256_packus_epi32(accumdlo, accumdhi), mask1),
        );
        ref_parts[part] = _mm256_castsi256_si128(result_r);
        dis_parts[part] = _mm256_castsi256_si128(result_d);
    }
    let mut out_ref = [0u16; 8];
    let mut out_dis = [0u16; 8];
    _mm_storeu_si128(&mut out_ref, _mm_unpacklo_epi64(ref_parts[0], ref_parts[1]));
    _mm_storeu_si128(&mut out_dis, _mm_unpacklo_epi64(dis_parts[0], dis_parts[1]));
    (out_ref, out_dis)
}

#[cfg(feature = "simd")]
#[magetypes(define(u32x8), v3, neon, wasm128, scalar)]
fn vif_subsample_horizontal_simd(
    token: Token,
    vertical_reference: &[u32],
    vertical_distorted: &[u32],
    filter: &[u16],
    col: usize,
) -> ([u16; 8], [u16; 8]) {
    let mut ref_lo = u32x8::zero(token);
    let mut ref_hi = u32x8::zero(token);
    let mut dis_lo = u32x8::zero(token);
    let mut dis_hi = u32x8::zero(token);
    for (tap, &w) in filter.iter().enumerate() {
        let weight = u32x8::splat(token, w as u32);
        ref_lo += weight
            * u32x8::load(
                token,
                vertical_reference[col + tap..col + tap + 8]
                    .try_into()
                    .unwrap(),
            );
        ref_hi += weight
            * u32x8::load(
                token,
                vertical_reference[col + 8 + tap..col + 8 + tap + 8]
                    .try_into()
                    .unwrap(),
            );
        dis_lo += weight
            * u32x8::load(
                token,
                vertical_distorted[col + tap..col + tap + 8]
                    .try_into()
                    .unwrap(),
            );
        dis_hi += weight
            * u32x8::load(
                token,
                vertical_distorted[col + 8 + tap..col + 8 + tap + 8]
                    .try_into()
                    .unwrap(),
            );
    }
    let rounding = u32x8::splat(token, 32768);
    let ref_lo = (ref_lo + rounding).shr_logical_uniform(16).to_array();
    let ref_hi = (ref_hi + rounding).shr_logical_uniform(16).to_array();
    let dis_lo = (dis_lo + rounding).shr_logical_uniform(16).to_array();
    let dis_hi = (dis_hi + rounding).shr_logical_uniform(16).to_array();
    let mut out_ref = [0u16; 8];
    let mut out_dis = [0u16; 8];
    for lane in 0..4 {
        out_ref[lane] = ref_lo[2 * lane] as u16;
        out_ref[4 + lane] = ref_hi[2 * lane] as u16;
        out_dis[lane] = dis_lo[2 * lane] as u16;
        out_dis[4 + lane] = dis_hi[2 * lane] as u16;
    }
    (out_ref, out_dis)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[rite]
fn vif_mul2(_token: X64V3Token, r0: __m256i, f: __m256i) -> (__m256i, __m256i) {
    let zero = _mm256_setzero_si256();
    (
        _mm256_madd_epi16(_mm256_unpacklo_epi16(r0, zero), f),
        _mm256_madd_epi16(_mm256_unpackhi_epi16(r0, zero), f),
    )
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[rite]
fn vif_mul2_acc(
    _token: X64V3Token,
    acc: (__m256i, __m256i),
    r0: __m256i,
    r1: __m256i,
    f: __m256i,
) -> (__m256i, __m256i) {
    (
        _mm256_add_epi32(acc.0, _mm256_madd_epi16(_mm256_unpacklo_epi16(r0, r1), f)),
        _mm256_add_epi32(acc.1, _mm256_madd_epi16(_mm256_unpackhi_epi16(r0, r1), f)),
    )
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[rite]
fn vif_mul3(_token: X64V3Token, r0: __m256i, r1: __m256i, f: __m256i) -> (__m256i, __m256i) {
    let mul = _mm256_mullo_epi16(r0, r1);
    let lo = _mm256_mullo_epi16(mul, f);
    let hi = _mm256_mulhi_epu16(mul, f);
    (_mm256_unpacklo_epi16(lo, hi), _mm256_unpackhi_epi16(lo, hi))
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[rite]
fn vif_mul3_acc(
    _token: X64V3Token,
    acc: (__m256i, __m256i),
    r0: __m256i,
    r1: __m256i,
    f: __m256i,
) -> (__m256i, __m256i) {
    let mul = _mm256_mullo_epi16(r0, r1);
    let lo = _mm256_mullo_epi16(mul, f);
    let hi = _mm256_mulhi_epu16(mul, f);
    (
        _mm256_add_epi32(acc.0, _mm256_unpacklo_epi16(lo, hi)),
        _mm256_add_epi32(acc.1, _mm256_unpackhi_epi16(lo, hi)),
    )
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[rite]
fn vif_shuffle_save(_token: X64V3Token, addr: &mut [u32; 16], x: __m256i, y: __m256i) {
    let left = _mm256_permute2x128_si256(x, y, 0x20);
    let right = _mm256_permute2x128_si256(x, y, 0x31);
    _mm256_storeu_si256(a8m::<u32, 8>(&mut addr[..8]), left);
    _mm256_storeu_si256(a8m::<u32, 8>(&mut addr[8..16]), right);
}

/// Port of libvmaf's `vif_statistic_8_avx512` vertical pass: 32 u16 lanes per
/// iteration (our planes are u16, not C's u8), tap *pairs* folded into one
/// `madd_epi16` each — symmetric sums `f0*(r0+r16)+f1*(r1+r15)` via unpacked
/// pair vectors, and squares via `madd(v,v)` which emits both taps' squares in
/// a single op. Requires `half` even (FILTERS[0] ⇒ 8) and bit_depth == 8 so
/// pair sums ≤ 510 fit i16 and sq products fit i32.
#[cfg(all(feature = "simd", feature = "avx512", target_arch = "x86_64"))]
#[arcane(import_intrinsics)]
#[allow(clippy::too_many_arguments)]
fn vif_stat8_vertical_v4(
    _token: X64V4Token,
    reference: &[u16],
    distorted: &[u16],
    row_offsets: &[usize; 17],
    filt: &[u16],
    half: usize,
    width: usize,
    tmp_mu1: &mut [u32],
    tmp_mu2: &mut [u32],
    tmp_ref: &mut [u32],
    tmp_dis: &mut [u32],
    tmp_ref_dis: &mut [u32],
) {
    let fwidth = 2 * half + 1;
    let n32 = width & !31;
    let zero = _mm512_setzero_si512();
    let round = _mm512_set1_epi32(128);
    let perm_lo = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
    let perm_hi = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);

    for j in (0..n32).step_by(32) {
        let f0 = _mm512_set1_epi32(filt[half] as i32);
        let r0 = _mm512_loadu_si512(a8::<u16, 32>(
            &reference[row_offsets[half] + j..row_offsets[half] + j + 32],
        ));
        let d0 = _mm512_loadu_si512(a8::<u16, 32>(
            &distorted[row_offsets[half] + j..row_offsets[half] + j + 32],
        ));
        let r0_lo = _mm512_unpacklo_epi16(r0, zero);
        let r0_hi = _mm512_unpackhi_epi16(r0, zero);
        let d0_lo = _mm512_unpacklo_epi16(d0, zero);
        let d0_hi = _mm512_unpackhi_epi16(d0, zero);

        let mut mu1_lo = _mm512_mullo_epi32(r0_lo, f0);
        let mut mu1_hi = _mm512_mullo_epi32(r0_hi, f0);
        let mut mu2_lo = _mm512_mullo_epi32(d0_lo, f0);
        let mut mu2_hi = _mm512_mullo_epi32(d0_hi, f0);
        let mut ref_lo = _mm512_mullo_epi32(f0, _mm512_mullo_epi32(r0_lo, r0_lo));
        let mut ref_hi = _mm512_mullo_epi32(f0, _mm512_mullo_epi32(r0_hi, r0_hi));
        let mut dis_lo = _mm512_mullo_epi32(f0, _mm512_mullo_epi32(d0_lo, d0_lo));
        let mut dis_hi = _mm512_mullo_epi32(f0, _mm512_mullo_epi32(d0_hi, d0_hi));
        let mut rd_lo = _mm512_mullo_epi32(f0, _mm512_mullo_epi32(r0_lo, d0_lo));
        let mut rd_hi = _mm512_mullo_epi32(f0, _mm512_mullo_epi32(r0_hi, d0_hi));

        for tap in (0..half).step_by(2) {
            let f0v = _mm512_set1_epi32(filt[tap] as i32);
            let f1v = _mm512_set1_epi32(filt[tap + 1] as i32);
            let f01 = _mm512_set1_epi32(
                (filt[tap] as i32) | ((filt[tap + 1] as i32) << 16),
            );
            let lo0 = row_offsets[tap];
            let hi0 = row_offsets[fwidth - 1 - tap];
            let lo1 = row_offsets[tap + 1];
            let hi1 = row_offsets[fwidth - 2 - tap];
            let r0 = _mm512_loadu_si512(a8::<u16, 32>(&reference[lo0 + j..lo0 + j + 32]));
            let r16 = _mm512_loadu_si512(a8::<u16, 32>(&reference[hi0 + j..hi0 + j + 32]));
            let r1 = _mm512_loadu_si512(a8::<u16, 32>(&reference[lo1 + j..lo1 + j + 32]));
            let r15 = _mm512_loadu_si512(a8::<u16, 32>(&reference[hi1 + j..hi1 + j + 32]));
            let d0 = _mm512_loadu_si512(a8::<u16, 32>(&distorted[lo0 + j..lo0 + j + 32]));
            let d16 = _mm512_loadu_si512(a8::<u16, 32>(&distorted[hi0 + j..hi0 + j + 32]));
            let d1 = _mm512_loadu_si512(a8::<u16, 32>(&distorted[lo1 + j..lo1 + j + 32]));
            let d15 = _mm512_loadu_si512(a8::<u16, 32>(&distorted[hi1 + j..hi1 + j + 32]));

            let r0p16 = _mm512_add_epi16(r0, r16);
            let r1p15 = _mm512_add_epi16(r1, r15);
            let d0p16 = _mm512_add_epi16(d0, d16);
            let d1p15 = _mm512_add_epi16(d1, d15);

            mu1_lo = _mm512_add_epi32(
                mu1_lo,
                _mm512_madd_epi16(_mm512_unpacklo_epi16(r0p16, r1p15), f01),
            );
            mu1_hi = _mm512_add_epi32(
                mu1_hi,
                _mm512_madd_epi16(_mm512_unpackhi_epi16(r0p16, r1p15), f01),
            );
            mu2_lo = _mm512_add_epi32(
                mu2_lo,
                _mm512_madd_epi16(_mm512_unpacklo_epi16(d0p16, d1p15), f01),
            );
            mu2_hi = _mm512_add_epi32(
                mu2_hi,
                _mm512_madd_epi16(_mm512_unpackhi_epi16(d0p16, d1p15), f01),
            );

            let rr_lo = _mm512_unpacklo_epi16(r0, r16);
            let rr_hi = _mm512_unpackhi_epi16(r0, r16);
            let rr1_lo = _mm512_unpacklo_epi16(r1, r15);
            let rr1_hi = _mm512_unpackhi_epi16(r1, r15);
            let dd_lo = _mm512_unpacklo_epi16(d0, d16);
            let dd_hi = _mm512_unpackhi_epi16(d0, d16);
            let dd1_lo = _mm512_unpacklo_epi16(d1, d15);
            let dd1_hi = _mm512_unpackhi_epi16(d1, d15);

            ref_lo = _mm512_add_epi32(
                ref_lo,
                _mm512_mullo_epi32(_mm512_madd_epi16(rr_lo, rr_lo), f0v),
            );
            ref_hi = _mm512_add_epi32(
                ref_hi,
                _mm512_mullo_epi32(_mm512_madd_epi16(rr_hi, rr_hi), f0v),
            );
            dis_lo = _mm512_add_epi32(
                dis_lo,
                _mm512_mullo_epi32(_mm512_madd_epi16(dd_lo, dd_lo), f0v),
            );
            dis_hi = _mm512_add_epi32(
                dis_hi,
                _mm512_mullo_epi32(_mm512_madd_epi16(dd_hi, dd_hi), f0v),
            );
            rd_lo = _mm512_add_epi32(
                rd_lo,
                _mm512_mullo_epi32(_mm512_madd_epi16(dd_lo, rr_lo), f0v),
            );
            rd_hi = _mm512_add_epi32(
                rd_hi,
                _mm512_mullo_epi32(_mm512_madd_epi16(dd_hi, rr_hi), f0v),
            );
            ref_lo = _mm512_add_epi32(
                ref_lo,
                _mm512_mullo_epi32(_mm512_madd_epi16(rr1_lo, rr1_lo), f1v),
            );
            ref_hi = _mm512_add_epi32(
                ref_hi,
                _mm512_mullo_epi32(_mm512_madd_epi16(rr1_hi, rr1_hi), f1v),
            );
            dis_lo = _mm512_add_epi32(
                dis_lo,
                _mm512_mullo_epi32(_mm512_madd_epi16(dd1_lo, dd1_lo), f1v),
            );
            dis_hi = _mm512_add_epi32(
                dis_hi,
                _mm512_mullo_epi32(_mm512_madd_epi16(dd1_hi, dd1_hi), f1v),
            );
            rd_lo = _mm512_add_epi32(
                rd_lo,
                _mm512_mullo_epi32(_mm512_madd_epi16(dd1_lo, rr1_lo), f1v),
            );
            rd_hi = _mm512_add_epi32(
                rd_hi,
                _mm512_mullo_epi32(_mm512_madd_epi16(dd1_hi, rr1_hi), f1v),
            );
        }

        mu1_lo = _mm512_srli_epi32::<8>(_mm512_add_epi32(mu1_lo, round));
        mu1_hi = _mm512_srli_epi32::<8>(_mm512_add_epi32(mu1_hi, round));
        mu2_lo = _mm512_srli_epi32::<8>(_mm512_add_epi32(mu2_lo, round));
        mu2_hi = _mm512_srli_epi32::<8>(_mm512_add_epi32(mu2_hi, round));

        macro_rules! store2 {
            ($dst:expr, $lo:expr, $hi:expr) => {
                let t = $lo;
                _mm512_storeu_si512(
                    a8m::<u32, 16>(&mut $dst[half + j..half + j + 16]),
                    _mm512_permutex2var_epi64(t, perm_lo, $hi),
                );
                _mm512_storeu_si512(
                    a8m::<u32, 16>(&mut $dst[half + j + 16..half + j + 32]),
                    _mm512_permutex2var_epi64(t, perm_hi, $hi),
                );
            };
        }
        store2!(tmp_mu1, mu1_lo, mu1_hi);
        store2!(tmp_mu2, mu2_lo, mu2_hi);
        store2!(tmp_ref, ref_lo, ref_hi);
        store2!(tmp_dis, dis_lo, dis_hi);
        store2!(tmp_ref_dis, rd_lo, rd_hi);
    }

    for j in n32..width {
        let mut accum_mu1 = 0u32;
        let mut accum_mu2 = 0u32;
        let mut accum_ref = 0u64;
        let mut accum_dis = 0u64;
        let mut accum_rd = 0u64;
        for (fi, &fcoeff) in filt.iter().enumerate().take(fwidth) {
            let off = row_offsets[fi];
            let ref_v = reference[off + j] as u32;
            let dis_v = distorted[off + j] as u32;
            let wref = fcoeff as u32 * ref_v;
            let wdis = fcoeff as u32 * dis_v;
            accum_mu1 = accum_mu1.wrapping_add(wref);
            accum_mu2 = accum_mu2.wrapping_add(wdis);
            accum_ref += wref as u64 * ref_v as u64;
            accum_dis += wdis as u64 * dis_v as u64;
            accum_rd += wref as u64 * dis_v as u64;
        }
        tmp_mu1[half + j] = (accum_mu1 + 128) >> 8;
        tmp_mu2[half + j] = (accum_mu2 + 128) >> 8;
        tmp_ref[half + j] = accum_ref as u32;
        tmp_dis[half + j] = accum_dis as u32;
        tmp_ref_dis[half + j] = accum_rd as u32;
    }
}

/// Faithful port of libvmaf's `vif_statistic_8_avx2` vertical pass for one row:
/// fills the five padded tmp arrays for all `width` columns.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane(import_intrinsics)]
fn vif_stat8_vertical_v3(
    token: X64V3Token,
    reference: &[u16],
    distorted: &[u16],
    row_offsets: &[usize; 17],
    filt: &[u16],
    half: usize,
    width: usize,
    tmp_mu1: &mut [u32],
    tmp_mu2: &mut [u32],
    tmp_ref: &mut [u32],
    tmp_dis: &mut [u32],
    tmp_ref_dis: &mut [u32],
) {
    let fwidth = 2 * half + 1;
    let n16 = width & !15;
    for j in (0..n16).step_by(16) {
        let f0 = _mm256_set1_epi16(filt[half] as i16);
        let r0 = _mm256_loadu_si256(a8::<u16, 16>(
            &reference[row_offsets[half] + j..row_offsets[half] + j + 16],
        ));
        let d0 = _mm256_loadu_si256(a8::<u16, 16>(
            &distorted[row_offsets[half] + j..row_offsets[half] + j + 16],
        ));

        let mut mu1 = vif_mul2(token, r0, f0);
        let mut mu2 = vif_mul2(token, d0, f0);
        let mut acc_ref = vif_mul3(token, r0, r0, f0);
        let mut acc_dis = vif_mul3(token, d0, d0, f0);
        let mut acc_rd = vif_mul3(token, d0, r0, f0);

        for tap in 0..half {
            let f0 = _mm256_set1_epi16(filt[tap] as i16);
            let lo = row_offsets[tap];
            let hi = row_offsets[fwidth - 1 - tap];
            let r0 = _mm256_loadu_si256(a8::<u16, 16>(&reference[lo + j..lo + j + 16]));
            let r1 = _mm256_loadu_si256(a8::<u16, 16>(&reference[hi + j..hi + j + 16]));
            let d0 = _mm256_loadu_si256(a8::<u16, 16>(&distorted[lo + j..lo + j + 16]));
            let d1 = _mm256_loadu_si256(a8::<u16, 16>(&distorted[hi + j..hi + j + 16]));

            mu1 = vif_mul2_acc(token, mu1, r0, r1, f0);
            mu2 = vif_mul2_acc(token, mu2, d0, d1, f0);
            acc_ref = vif_mul3_acc(token, acc_ref, r0, r0, f0);
            acc_ref = vif_mul3_acc(token, acc_ref, r1, r1, f0);
            acc_dis = vif_mul3_acc(token, acc_dis, d0, d0, f0);
            acc_dis = vif_mul3_acc(token, acc_dis, d1, d1, f0);
            acc_rd = vif_mul3_acc(token, acc_rd, d0, r0, f0);
            acc_rd = vif_mul3_acc(token, acc_rd, d1, r1, f0);
        }

        let x = _mm256_set1_epi32(128);
        mu1.0 = _mm256_srli_epi32(_mm256_add_epi32(mu1.0, x), 8);
        mu1.1 = _mm256_srli_epi32(_mm256_add_epi32(mu1.1, x), 8);
        mu2.0 = _mm256_srli_epi32(_mm256_add_epi32(mu2.0, x), 8);
        mu2.1 = _mm256_srli_epi32(_mm256_add_epi32(mu2.1, x), 8);

        vif_shuffle_save(
            token,
            a8m::<u32, 16>(&mut tmp_mu1[half + j..half + j + 16]),
            mu1.0,
            mu1.1,
        );
        vif_shuffle_save(
            token,
            a8m::<u32, 16>(&mut tmp_mu2[half + j..half + j + 16]),
            mu2.0,
            mu2.1,
        );
        vif_shuffle_save(
            token,
            a8m::<u32, 16>(&mut tmp_ref[half + j..half + j + 16]),
            acc_ref.0,
            acc_ref.1,
        );
        vif_shuffle_save(
            token,
            a8m::<u32, 16>(&mut tmp_dis[half + j..half + j + 16]),
            acc_dis.0,
            acc_dis.1,
        );
        vif_shuffle_save(
            token,
            a8m::<u32, 16>(&mut tmp_ref_dis[half + j..half + j + 16]),
            acc_rd.0,
            acc_rd.1,
        );
    }

    for j in n16..width {
        let mut accum_mu1 = 0u32;
        let mut accum_mu2 = 0u32;
        let mut accum_ref = 0u64;
        let mut accum_dis = 0u64;
        let mut accum_rd = 0u64;
        for (fi, &fcoeff) in filt.iter().enumerate().take(fwidth) {
            let off = row_offsets[fi];
            let ref_v = reference[off + j] as u32;
            let dis_v = distorted[off + j] as u32;
            let wref = fcoeff as u32 * ref_v;
            let wdis = fcoeff as u32 * dis_v;
            accum_mu1 = accum_mu1.wrapping_add(wref);
            accum_mu2 = accum_mu2.wrapping_add(wdis);
            accum_ref += wref as u64 * ref_v as u64;
            accum_dis += wdis as u64 * dis_v as u64;
            accum_rd += wref as u64 * dis_v as u64;
        }
        tmp_mu1[half + j] = (accum_mu1 + 128) >> 8;
        tmp_mu2[half + j] = (accum_mu2 + 128) >> 8;
        tmp_ref[half + j] = accum_ref as u32;
        tmp_dis[half + j] = accum_dis as u32;
        tmp_ref_dis[half + j] = accum_rd as u32;
    }
}

/// Faithful port of libvmaf's `vif_statistic_8_avx2` horizontal pass for one row:
/// consumes the padded tmp arrays and accumulates num/den for 16-column blocks.
/// Returns the number of columns processed (a multiple of 16).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane(import_intrinsics)]
fn vif_stat8_horizontal_v3(
    token: X64V3Token,
    tmp_mu1: &[u32],
    tmp_mu2: &[u32],
    tmp_ref: &[u32],
    tmp_dis: &[u32],
    tmp_ref_dis: &[u32],
    filt: &[u16],
    half: usize,
    width: usize,
    table: &[u16; 65536],
    gain_limit: f64,
    accums: (&mut i64, &mut i64, &mut i64, &mut i64),
) -> usize {
    let _ = token;
    let fwidth = 2 * half + 1;
    let n16 = width & !15;
    let mut xx = [0u32; 16];
    let mut yy = [0u32; 16];
    let mut xy = [0u32; 16];
    let (num_log, den_log, num_non_log, den_non_log) = accums;

    for j in (0..n16).step_by(16) {
        // mu1 horizontal sum (epi32 products accumulated pairwise in epi64 lanes)
        let fq = _mm256_set1_epi32(filt[half] as i32);
        let mut mu1_lo = _mm256_mullo_epi32(
            _mm256_loadu_si256(a8::<u32, 8>(&tmp_mu1[half + j..half + j + 8])),
            fq,
        );
        let mut mu1_hi = _mm256_mullo_epi32(
            _mm256_loadu_si256(a8::<u32, 8>(&tmp_mu1[half + j + 8..half + j + 16])),
            fq,
        );
        for (fj, &coeff) in filt.iter().enumerate().take(half) {
            let fq = _mm256_set1_epi32(coeff as i32);
            let l = j + fj;
            let r = j + fwidth - 1 - fj;
            mu1_lo = _mm256_add_epi64(
                mu1_lo,
                _mm256_mullo_epi32(_mm256_loadu_si256(a8::<u32, 8>(&tmp_mu1[l..l + 8])), fq),
            );
            mu1_hi = _mm256_add_epi64(
                mu1_hi,
                _mm256_mullo_epi32(
                    _mm256_loadu_si256(a8::<u32, 8>(&tmp_mu1[l + 8..l + 16])),
                    fq,
                ),
            );
            mu1_lo = _mm256_add_epi64(
                mu1_lo,
                _mm256_mullo_epi32(_mm256_loadu_si256(a8::<u32, 8>(&tmp_mu1[r..r + 8])), fq),
            );
            mu1_hi = _mm256_add_epi64(
                mu1_hi,
                _mm256_mullo_epi32(
                    _mm256_loadu_si256(a8::<u32, 8>(&tmp_mu1[r + 8..r + 16])),
                    fq,
                ),
            );
        }

        let zero = _mm256_setzero_si256();
        let round32 = _mm256_set1_epi64x(0x80000000);
        let acc0_lo = _mm256_mul_epu32(
            _mm256_unpacklo_epi32(mu1_lo, zero),
            _mm256_unpacklo_epi32(mu1_lo, zero),
        );
        let acc0_lo = _mm256_srli_epi64(_mm256_add_epi64(acc0_lo, round32), 32);
        let acc0_hi = _mm256_mul_epu32(
            _mm256_unpackhi_epi32(mu1_lo, zero),
            _mm256_unpackhi_epi32(mu1_lo, zero),
        );
        let acc0_hi = _mm256_srli_epi64(_mm256_add_epi64(acc0_hi, round32), 32);
        let acc1_lo = _mm256_mul_epu32(
            _mm256_unpacklo_epi32(mu1_hi, zero),
            _mm256_unpacklo_epi32(mu1_hi, zero),
        );
        let acc1_lo = _mm256_srli_epi64(_mm256_add_epi64(acc1_lo, round32), 32);
        let acc1_hi = _mm256_mul_epu32(
            _mm256_unpackhi_epi32(mu1_hi, zero),
            _mm256_unpackhi_epi32(mu1_hi, zero),
        );
        let acc1_hi = _mm256_srli_epi64(_mm256_add_epi64(acc1_hi, round32), 32);
        let mu1sq_lo = _mm256_blend_epi32(acc0_lo, _mm256_slli_si256(acc0_hi, 4), 0xAA);
        let mu1sq_hi = _mm256_blend_epi32(acc1_lo, _mm256_slli_si256(acc1_hi, 4), 0xAA);

        // mu2 horizontal sum + mu1mu2/mu2sq products
        let fq = _mm256_set1_epi32(filt[half] as i32);
        let mut acc0 = _mm256_mullo_epi32(
            _mm256_loadu_si256(a8::<u32, 8>(&tmp_mu2[half + j..half + j + 8])),
            fq,
        );
        let mut acc1 = _mm256_mullo_epi32(
            _mm256_loadu_si256(a8::<u32, 8>(&tmp_mu2[half + j + 8..half + j + 16])),
            fq,
        );
        for (fj, &coeff) in filt.iter().enumerate().take(half) {
            let fq = _mm256_set1_epi32(coeff as i32);
            let l = j + fj;
            let r = j + fwidth - 1 - fj;
            acc0 = _mm256_add_epi64(
                acc0,
                _mm256_mullo_epi32(_mm256_loadu_si256(a8::<u32, 8>(&tmp_mu2[l..l + 8])), fq),
            );
            acc1 = _mm256_add_epi64(
                acc1,
                _mm256_mullo_epi32(
                    _mm256_loadu_si256(a8::<u32, 8>(&tmp_mu2[l + 8..l + 16])),
                    fq,
                ),
            );
            acc0 = _mm256_add_epi64(
                acc0,
                _mm256_mullo_epi32(_mm256_loadu_si256(a8::<u32, 8>(&tmp_mu2[r..r + 8])), fq),
            );
            acc1 = _mm256_add_epi64(
                acc1,
                _mm256_mullo_epi32(
                    _mm256_loadu_si256(a8::<u32, 8>(&tmp_mu2[r + 8..r + 16])),
                    fq,
                ),
            );
        }

        let mut acc0_lo = _mm256_unpacklo_epi32(acc0, zero);
        let mut acc0_hi = _mm256_unpackhi_epi32(acc0, zero);
        let mut mu1lo_lo = _mm256_unpacklo_epi32(mu1_lo, zero);
        let mut mu1lo_hi = _mm256_unpackhi_epi32(mu1_lo, zero);
        let mut mu1hi_lo = _mm256_unpacklo_epi32(mu1_hi, zero);
        let mut mu1hi_hi = _mm256_unpackhi_epi32(mu1_hi, zero);

        mu1lo_lo = _mm256_srli_epi64(
            _mm256_add_epi64(_mm256_mul_epu32(mu1lo_lo, acc0_lo), round32),
            32,
        );
        mu1lo_hi = _mm256_srli_epi64(
            _mm256_add_epi64(_mm256_mul_epu32(mu1lo_hi, acc0_hi), round32),
            32,
        );
        acc0_lo = _mm256_srli_epi64(
            _mm256_add_epi64(_mm256_mul_epu32(acc0_lo, acc0_lo), round32),
            32,
        );
        acc0_hi = _mm256_srli_epi64(
            _mm256_add_epi64(_mm256_mul_epu32(acc0_hi, acc0_hi), round32),
            32,
        );

        let mut acc1_lo = _mm256_unpacklo_epi32(acc1, zero);
        let mut acc1_hi = _mm256_unpackhi_epi32(acc1, zero);
        mu1hi_lo = _mm256_srli_epi64(
            _mm256_add_epi64(_mm256_mul_epu32(mu1hi_lo, acc1_lo), round32),
            32,
        );
        mu1hi_hi = _mm256_srli_epi64(
            _mm256_add_epi64(_mm256_mul_epu32(mu1hi_hi, acc1_hi), round32),
            32,
        );
        acc1_lo = _mm256_srli_epi64(
            _mm256_add_epi64(_mm256_mul_epu32(acc1_lo, acc1_lo), round32),
            32,
        );
        acc1_hi = _mm256_srli_epi64(
            _mm256_add_epi64(_mm256_mul_epu32(acc1_hi, acc1_hi), round32),
            32,
        );

        let mu2sq_lo = _mm256_blend_epi32(acc0_lo, _mm256_slli_si256(acc0_hi, 4), 0xAA);
        let mu2sq_hi = _mm256_blend_epi32(acc1_lo, _mm256_slli_si256(acc1_hi, 4), 0xAA);
        let mu1mu2_lo = _mm256_blend_epi32(mu1lo_lo, _mm256_slli_si256(mu1lo_hi, 4), 0xAA);
        let mu1mu2_hi = _mm256_blend_epi32(mu1hi_lo, _mm256_slli_si256(mu1hi_hi, 4), 0xAA);

        // filtered ref^2 - mu1^2
        {
            let rounder = _mm256_set1_epi64x(0x8000);
            let fq = _mm256_set1_epi64x(filt[half] as i64);
            let m0 = _mm256_loadu_si256(a8::<u32, 8>(&tmp_ref[half + j..half + j + 8]));
            let m1 = _mm256_loadu_si256(a8::<u32, 8>(&tmp_ref[half + j + 8..half + j + 16]));
            let mut acc0 = _mm256_add_epi64(
                rounder,
                _mm256_mul_epu32(_mm256_unpacklo_epi32(m0, zero), fq),
            );
            let mut acc1 = _mm256_add_epi64(
                rounder,
                _mm256_mul_epu32(_mm256_unpackhi_epi32(m0, zero), fq),
            );
            let mut acc2 = _mm256_add_epi64(
                rounder,
                _mm256_mul_epu32(_mm256_unpacklo_epi32(m1, zero), fq),
            );
            let mut acc3 = _mm256_add_epi64(
                rounder,
                _mm256_mul_epu32(_mm256_unpackhi_epi32(m1, zero), fq),
            );
            for (fj, &coeff) in filt.iter().enumerate().take(half) {
                let fq = _mm256_set1_epi64x(coeff as i64);
                let l = j + fj;
                let r = j + fwidth - 1 - fj;
                let m0 = _mm256_loadu_si256(a8::<u32, 8>(&tmp_ref[l..l + 8]));
                let m1 = _mm256_loadu_si256(a8::<u32, 8>(&tmp_ref[l + 8..l + 16]));
                let m2 = _mm256_loadu_si256(a8::<u32, 8>(&tmp_ref[r..r + 8]));
                let m3 = _mm256_loadu_si256(a8::<u32, 8>(&tmp_ref[r + 8..r + 16]));
                acc0 =
                    _mm256_add_epi64(acc0, _mm256_mul_epu32(_mm256_unpacklo_epi32(m0, zero), fq));
                acc1 =
                    _mm256_add_epi64(acc1, _mm256_mul_epu32(_mm256_unpackhi_epi32(m0, zero), fq));
                acc2 =
                    _mm256_add_epi64(acc2, _mm256_mul_epu32(_mm256_unpacklo_epi32(m1, zero), fq));
                acc3 =
                    _mm256_add_epi64(acc3, _mm256_mul_epu32(_mm256_unpackhi_epi32(m1, zero), fq));
                acc0 =
                    _mm256_add_epi64(acc0, _mm256_mul_epu32(_mm256_unpacklo_epi32(m2, zero), fq));
                acc1 =
                    _mm256_add_epi64(acc1, _mm256_mul_epu32(_mm256_unpackhi_epi32(m2, zero), fq));
                acc2 =
                    _mm256_add_epi64(acc2, _mm256_mul_epu32(_mm256_unpacklo_epi32(m3, zero), fq));
                acc3 =
                    _mm256_add_epi64(acc3, _mm256_mul_epu32(_mm256_unpackhi_epi32(m3, zero), fq));
            }
            acc0 = _mm256_srli_epi64(acc0, 16);
            acc1 = _mm256_srli_epi64(acc1, 16);
            acc2 = _mm256_srli_epi64(acc2, 16);
            acc3 = _mm256_srli_epi64(acc3, 16);
            acc0 = _mm256_blend_epi32(acc0, _mm256_slli_si256(acc1, 4), 0xAA);
            acc1 = _mm256_blend_epi32(acc2, _mm256_slli_si256(acc3, 4), 0xAA);
            acc0 = _mm256_sub_epi32(acc0, mu1sq_lo);
            acc1 = _mm256_sub_epi32(acc1, mu1sq_hi);
            acc0 = _mm256_shuffle_epi32(acc0, 0xD8);
            acc1 = _mm256_shuffle_epi32(acc1, 0xD8);
            _mm256_storeu_si256(a8m::<u32, 8>(&mut xx[..8]), acc0);
            _mm256_storeu_si256(a8m::<u32, 8>(&mut xx[8..16]), acc1);
        }

        // filtered dis^2 - mu2^2
        {
            let rounder = _mm256_set1_epi64x(0x8000);
            let fq = _mm256_set1_epi64x(filt[half] as i64);
            let m0 = _mm256_loadu_si256(a8::<u32, 8>(&tmp_dis[half + j..half + j + 8]));
            let m1 = _mm256_loadu_si256(a8::<u32, 8>(&tmp_dis[half + j + 8..half + j + 16]));
            let mut acc0 = _mm256_add_epi64(
                rounder,
                _mm256_mul_epu32(_mm256_unpacklo_epi32(m0, zero), fq),
            );
            let mut acc1 = _mm256_add_epi64(
                rounder,
                _mm256_mul_epu32(_mm256_unpackhi_epi32(m0, zero), fq),
            );
            let mut acc2 = _mm256_add_epi64(
                rounder,
                _mm256_mul_epu32(_mm256_unpacklo_epi32(m1, zero), fq),
            );
            let mut acc3 = _mm256_add_epi64(
                rounder,
                _mm256_mul_epu32(_mm256_unpackhi_epi32(m1, zero), fq),
            );
            for (fj, &coeff) in filt.iter().enumerate().take(half) {
                let fq = _mm256_set1_epi64x(coeff as i64);
                let l = j + fj;
                let r = j + fwidth - 1 - fj;
                let m0 = _mm256_loadu_si256(a8::<u32, 8>(&tmp_dis[l..l + 8]));
                let m1 = _mm256_loadu_si256(a8::<u32, 8>(&tmp_dis[l + 8..l + 16]));
                let m2 = _mm256_loadu_si256(a8::<u32, 8>(&tmp_dis[r..r + 8]));
                let m3 = _mm256_loadu_si256(a8::<u32, 8>(&tmp_dis[r + 8..r + 16]));
                acc0 =
                    _mm256_add_epi64(acc0, _mm256_mul_epu32(_mm256_unpacklo_epi32(m0, zero), fq));
                acc1 =
                    _mm256_add_epi64(acc1, _mm256_mul_epu32(_mm256_unpackhi_epi32(m0, zero), fq));
                acc2 =
                    _mm256_add_epi64(acc2, _mm256_mul_epu32(_mm256_unpacklo_epi32(m1, zero), fq));
                acc3 =
                    _mm256_add_epi64(acc3, _mm256_mul_epu32(_mm256_unpackhi_epi32(m1, zero), fq));
                acc0 =
                    _mm256_add_epi64(acc0, _mm256_mul_epu32(_mm256_unpacklo_epi32(m2, zero), fq));
                acc1 =
                    _mm256_add_epi64(acc1, _mm256_mul_epu32(_mm256_unpackhi_epi32(m2, zero), fq));
                acc2 =
                    _mm256_add_epi64(acc2, _mm256_mul_epu32(_mm256_unpacklo_epi32(m3, zero), fq));
                acc3 =
                    _mm256_add_epi64(acc3, _mm256_mul_epu32(_mm256_unpackhi_epi32(m3, zero), fq));
            }
            acc0 = _mm256_srli_epi64(acc0, 16);
            acc1 = _mm256_srli_epi64(acc1, 16);
            acc2 = _mm256_srli_epi64(acc2, 16);
            acc3 = _mm256_srli_epi64(acc3, 16);
            acc0 = _mm256_blend_epi32(acc0, _mm256_slli_si256(acc1, 4), 0xAA);
            acc1 = _mm256_blend_epi32(acc2, _mm256_slli_si256(acc3, 4), 0xAA);
            acc0 = _mm256_sub_epi32(acc0, mu2sq_lo);
            acc1 = _mm256_sub_epi32(acc1, mu2sq_hi);
            acc0 = _mm256_shuffle_epi32(acc0, 0xD8);
            acc1 = _mm256_shuffle_epi32(acc1, 0xD8);
            _mm256_storeu_si256(a8m::<u32, 8>(&mut yy[..8]), _mm256_max_epi32(acc0, zero));
            _mm256_storeu_si256(a8m::<u32, 8>(&mut yy[8..16]), _mm256_max_epi32(acc1, zero));
        }

        // filtered ref*dis - mu1*mu2
        {
            let rounder = _mm256_set1_epi64x(0x8000);
            let fq = _mm256_set1_epi64x(filt[half] as i64);
            let m0 = _mm256_loadu_si256(a8::<u32, 8>(&tmp_ref_dis[half + j..half + j + 8]));
            let m1 = _mm256_loadu_si256(a8::<u32, 8>(&tmp_ref_dis[half + j + 8..half + j + 16]));
            let mut acc0 = _mm256_add_epi64(
                rounder,
                _mm256_mul_epu32(_mm256_unpacklo_epi32(m0, zero), fq),
            );
            let mut acc1 = _mm256_add_epi64(
                rounder,
                _mm256_mul_epu32(_mm256_unpackhi_epi32(m0, zero), fq),
            );
            let mut acc2 = _mm256_add_epi64(
                rounder,
                _mm256_mul_epu32(_mm256_unpacklo_epi32(m1, zero), fq),
            );
            let mut acc3 = _mm256_add_epi64(
                rounder,
                _mm256_mul_epu32(_mm256_unpackhi_epi32(m1, zero), fq),
            );
            for (fj, &coeff) in filt.iter().enumerate().take(half) {
                let fq = _mm256_set1_epi64x(coeff as i64);
                let l = j + fj;
                let r = j + fwidth - 1 - fj;
                let m0 = _mm256_loadu_si256(a8::<u32, 8>(&tmp_ref_dis[l..l + 8]));
                let m1 = _mm256_loadu_si256(a8::<u32, 8>(&tmp_ref_dis[l + 8..l + 16]));
                let m2 = _mm256_loadu_si256(a8::<u32, 8>(&tmp_ref_dis[r..r + 8]));
                let m3 = _mm256_loadu_si256(a8::<u32, 8>(&tmp_ref_dis[r + 8..r + 16]));
                acc0 =
                    _mm256_add_epi64(acc0, _mm256_mul_epu32(_mm256_unpacklo_epi32(m0, zero), fq));
                acc1 =
                    _mm256_add_epi64(acc1, _mm256_mul_epu32(_mm256_unpackhi_epi32(m0, zero), fq));
                acc2 =
                    _mm256_add_epi64(acc2, _mm256_mul_epu32(_mm256_unpacklo_epi32(m1, zero), fq));
                acc3 =
                    _mm256_add_epi64(acc3, _mm256_mul_epu32(_mm256_unpackhi_epi32(m1, zero), fq));
                acc0 =
                    _mm256_add_epi64(acc0, _mm256_mul_epu32(_mm256_unpacklo_epi32(m2, zero), fq));
                acc1 =
                    _mm256_add_epi64(acc1, _mm256_mul_epu32(_mm256_unpackhi_epi32(m2, zero), fq));
                acc2 =
                    _mm256_add_epi64(acc2, _mm256_mul_epu32(_mm256_unpacklo_epi32(m3, zero), fq));
                acc3 =
                    _mm256_add_epi64(acc3, _mm256_mul_epu32(_mm256_unpackhi_epi32(m3, zero), fq));
            }
            acc0 = _mm256_srli_epi64(acc0, 16);
            acc1 = _mm256_srli_epi64(acc1, 16);
            acc2 = _mm256_srli_epi64(acc2, 16);
            acc3 = _mm256_srli_epi64(acc3, 16);
            acc0 = _mm256_blend_epi32(acc0, _mm256_slli_si256(acc1, 4), 0xAA);
            acc1 = _mm256_blend_epi32(acc2, _mm256_slli_si256(acc3, 4), 0xAA);
            acc0 = _mm256_sub_epi32(acc0, mu1mu2_lo);
            acc1 = _mm256_sub_epi32(acc1, mu1mu2_hi);
            acc0 = _mm256_shuffle_epi32(acc0, 0xD8);
            acc1 = _mm256_shuffle_epi32(acc1, 0xD8);
            _mm256_storeu_si256(a8m::<u32, 8>(&mut xy[..8]), acc0);
            _mm256_storeu_si256(a8m::<u32, 8>(&mut xy[8..16]), acc1);
        }

        for b in 0..16 {
            vif_finalize_sigma(
                table,
                gain_limit,
                xx[b] as i32,
                yy[b] as i32,
                xy[b] as i32,
                &mut *num_log,
                &mut *den_log,
                &mut *num_non_log,
                &mut *den_non_log,
            );
        }
    }
    n16
}

/// Faithful port of libvmaf's `vif_statistic_16_avx2` vertical pass: covers
/// every case except 8-bit scale 0 (10-bit scale 0 and all depth scales 1-3).
/// Loads 16 u16 lanes per tap; products widen through mullo/mulhi_epu16 +
/// unpack_epi16 for the means and cvtepu32/16_epi64 + mul_epu32 for the
/// 64-bit square sums. Shift/round constants are runtime parameters, matching
/// the C which selects them from (bpc, scale). Returns columns completed
/// (a multiple of 16); the caller's scalar loop finishes the tail.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane(import_intrinsics)]
#[allow(clippy::too_many_arguments)]
fn vif_stat16_vertical_v3(
    token: X64V3Token,
    reference: &[u16],
    distorted: &[u16],
    row_offsets: &[usize; 17],
    filt: &[u16],
    half: usize,
    width: usize,
    shift_vp: u32,
    round_vp: u32,
    shift_vp_sq: u32,
    round_vp_sq: u64,
    tmp_mu1: &mut [u32],
    tmp_mu2: &mut [u32],
    tmp_ref: &mut [u32],
    tmp_dis: &mut [u32],
    tmp_ref_dis: &mut [u32],
) -> usize {
    let _ = token;
    let fwidth = 2 * half + 1;
    let n16 = width & !15;
    let mask2 = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
    let vp_cnt = _mm_cvtsi32_si128(shift_vp as i32);
    let sq_cnt = _mm_cvtsi32_si128(shift_vp_sq as i32);
    for j in (0..n16).step_by(16) {
        let mut accumr_lo = _mm256_setzero_si256();
        let mut accumr_hi = _mm256_setzero_si256();
        let mut accumd_lo = _mm256_setzero_si256();
        let mut accumd_hi = _mm256_setzero_si256();
        let mut accumref1 = _mm256_setzero_si256();
        let mut accumref2 = _mm256_setzero_si256();
        let mut accumref3 = _mm256_setzero_si256();
        let mut accumref4 = _mm256_setzero_si256();
        let mut accumrefdis1 = _mm256_setzero_si256();
        let mut accumrefdis2 = _mm256_setzero_si256();
        let mut accumrefdis3 = _mm256_setzero_si256();
        let mut accumrefdis4 = _mm256_setzero_si256();
        let mut accumdis1 = _mm256_setzero_si256();
        let mut accumdis2 = _mm256_setzero_si256();
        let mut accumdis3 = _mm256_setzero_si256();
        let mut accumdis4 = _mm256_setzero_si256();
        let addnum = _mm256_set1_epi32(round_vp as i32);
        for (fi, &coef) in filt.iter().enumerate().take(fwidth) {
            let f1 = _mm256_set1_epi16(coef as i16);
            let off = row_offsets[fi];
            let ref1 = _mm256_loadu_si256(a8::<u16, 16>(&reference[off + j..off + j + 16]));
            let dis1 = _mm256_loadu_si256(a8::<u16, 16>(&distorted[off + j..off + j + 16]));
            let result2 = _mm256_mulhi_epu16(ref1, f1);
            let result2lo = _mm256_mullo_epi16(ref1, f1);
            let rmul1 = _mm256_unpacklo_epi16(result2lo, result2);
            let rmul2 = _mm256_unpackhi_epi16(result2lo, result2);
            accumr_lo = _mm256_add_epi32(accumr_lo, rmul1);
            accumr_hi = _mm256_add_epi32(accumr_hi, rmul2);
            let d0 = _mm256_mulhi_epu16(dis1, f1);
            let d0lo = _mm256_mullo_epi16(dis1, f1);
            let dmul1 = _mm256_unpacklo_epi16(d0lo, d0);
            let dmul2 = _mm256_unpackhi_epi16(d0lo, d0);
            accumd_lo = _mm256_add_epi32(accumd_lo, dmul1);
            accumd_hi = _mm256_add_epi32(accumd_hi, dmul2);

            let sg0 = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(rmul1));
            let sg1 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(rmul1, 1));
            let sg2 = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(rmul2));
            let sg3 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(rmul2, 1));
            let l0 = _mm256_castsi256_si128(ref1);
            let l1 = _mm256_extracti128_si256(ref1, 1);
            accumref1 =
                _mm256_add_epi64(accumref1, _mm256_mul_epu32(sg0, _mm256_cvtepu16_epi64(l0)));
            accumref2 = _mm256_add_epi64(
                accumref2,
                _mm256_mul_epu32(sg2, _mm256_cvtepu16_epi64(_mm_bsrli_si128(l0, 8))),
            );
            accumref3 =
                _mm256_add_epi64(accumref3, _mm256_mul_epu32(sg1, _mm256_cvtepu16_epi64(l1)));
            accumref4 = _mm256_add_epi64(
                accumref4,
                _mm256_mul_epu32(sg3, _mm256_cvtepu16_epi64(_mm_bsrli_si128(l1, 8))),
            );
            let d_l0 = _mm256_castsi256_si128(dis1);
            let d_l1 = _mm256_extracti128_si256(dis1, 1);
            accumrefdis1 = _mm256_add_epi64(
                accumrefdis1,
                _mm256_mul_epu32(sg0, _mm256_cvtepu16_epi64(d_l0)),
            );
            accumrefdis2 = _mm256_add_epi64(
                accumrefdis2,
                _mm256_mul_epu32(sg2, _mm256_cvtepu16_epi64(_mm_bsrli_si128(d_l0, 8))),
            );
            accumrefdis3 = _mm256_add_epi64(
                accumrefdis3,
                _mm256_mul_epu32(sg1, _mm256_cvtepu16_epi64(d_l1)),
            );
            accumrefdis4 = _mm256_add_epi64(
                accumrefdis4,
                _mm256_mul_epu32(sg3, _mm256_cvtepu16_epi64(_mm_bsrli_si128(d_l1, 8))),
            );
            let sd0 = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(dmul1));
            let sd1 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(dmul1, 1));
            let sd2 = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(dmul2));
            let sd3 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(dmul2, 1));
            accumdis1 = _mm256_add_epi64(
                accumdis1,
                _mm256_mul_epu32(sd0, _mm256_cvtepu16_epi64(d_l0)),
            );
            accumdis2 = _mm256_add_epi64(
                accumdis2,
                _mm256_mul_epu32(sd2, _mm256_cvtepu16_epi64(_mm_bsrli_si128(d_l0, 8))),
            );
            accumdis3 = _mm256_add_epi64(
                accumdis3,
                _mm256_mul_epu32(sd1, _mm256_cvtepu16_epi64(d_l1)),
            );
            accumdis4 = _mm256_add_epi64(
                accumdis4,
                _mm256_mul_epu32(sd3, _mm256_cvtepu16_epi64(_mm_bsrli_si128(d_l1, 8))),
            );
        }

        accumr_lo = _mm256_srl_epi32(_mm256_add_epi32(accumr_lo, addnum), vp_cnt);
        accumr_hi = _mm256_srl_epi32(_mm256_add_epi32(accumr_hi, addnum), vp_cnt);
        let accu2_lo = _mm256_permute2x128_si256(accumr_lo, accumr_hi, 0x20);
        let accu2_hi = _mm256_permute2x128_si256(accumr_lo, accumr_hi, 0x31);
        _mm256_storeu_si256(
            a8m::<u32, 8>(&mut tmp_mu1[half + j..half + j + 8]),
            accu2_lo,
        );
        _mm256_storeu_si256(
            a8m::<u32, 8>(&mut tmp_mu1[half + j + 8..half + j + 16]),
            accu2_hi,
        );

        accumd_lo = _mm256_srl_epi32(_mm256_add_epi32(accumd_lo, addnum), vp_cnt);
        accumd_hi = _mm256_srl_epi32(_mm256_add_epi32(accumd_hi, addnum), vp_cnt);
        let accu3_lo = _mm256_permute2x128_si256(accumd_lo, accumd_hi, 0x20);
        let accu3_hi = _mm256_permute2x128_si256(accumd_lo, accumd_hi, 0x31);
        _mm256_storeu_si256(
            a8m::<u32, 8>(&mut tmp_mu2[half + j..half + j + 8]),
            accu3_lo,
        );
        _mm256_storeu_si256(
            a8m::<u32, 8>(&mut tmp_mu2[half + j + 8..half + j + 16]),
            accu3_hi,
        );

        let addnum64 = _mm256_set1_epi64x(round_vp_sq as i64);
        accumref1 = _mm256_srl_epi64(_mm256_add_epi64(accumref1, addnum64), sq_cnt);
        accumref2 = _mm256_srl_epi64(_mm256_add_epi64(accumref2, addnum64), sq_cnt);
        accumref3 = _mm256_srl_epi64(_mm256_add_epi64(accumref3, addnum64), sq_cnt);
        accumref4 = _mm256_srl_epi64(_mm256_add_epi64(accumref4, addnum64), sq_cnt);
        accumref2 = _mm256_slli_si256(accumref2, 4);
        accumref1 = _mm256_blend_epi32(accumref1, accumref2, 0xAA);
        accumref1 = _mm256_permutevar8x32_epi32(accumref1, mask2);
        _mm256_storeu_si256(
            a8m::<u32, 8>(&mut tmp_ref[half + j..half + j + 8]),
            accumref1,
        );
        accumref4 = _mm256_slli_si256(accumref4, 4);
        accumref3 = _mm256_blend_epi32(accumref3, accumref4, 0xAA);
        accumref3 = _mm256_permutevar8x32_epi32(accumref3, mask2);
        _mm256_storeu_si256(
            a8m::<u32, 8>(&mut tmp_ref[half + j + 8..half + j + 16]),
            accumref3,
        );

        accumrefdis1 = _mm256_srl_epi64(_mm256_add_epi64(accumrefdis1, addnum64), sq_cnt);
        accumrefdis2 = _mm256_srl_epi64(_mm256_add_epi64(accumrefdis2, addnum64), sq_cnt);
        accumrefdis3 = _mm256_srl_epi64(_mm256_add_epi64(accumrefdis3, addnum64), sq_cnt);
        accumrefdis4 = _mm256_srl_epi64(_mm256_add_epi64(accumrefdis4, addnum64), sq_cnt);
        accumrefdis2 = _mm256_slli_si256(accumrefdis2, 4);
        accumrefdis1 = _mm256_blend_epi32(accumrefdis1, accumrefdis2, 0xAA);
        accumrefdis1 = _mm256_permutevar8x32_epi32(accumrefdis1, mask2);
        _mm256_storeu_si256(
            a8m::<u32, 8>(&mut tmp_ref_dis[half + j..half + j + 8]),
            accumrefdis1,
        );
        accumrefdis4 = _mm256_slli_si256(accumrefdis4, 4);
        accumrefdis3 = _mm256_blend_epi32(accumrefdis3, accumrefdis4, 0xAA);
        accumrefdis3 = _mm256_permutevar8x32_epi32(accumrefdis3, mask2);
        _mm256_storeu_si256(
            a8m::<u32, 8>(&mut tmp_ref_dis[half + j + 8..half + j + 16]),
            accumrefdis3,
        );

        accumdis1 = _mm256_srl_epi64(_mm256_add_epi64(accumdis1, addnum64), sq_cnt);
        accumdis2 = _mm256_srl_epi64(_mm256_add_epi64(accumdis2, addnum64), sq_cnt);
        accumdis3 = _mm256_srl_epi64(_mm256_add_epi64(accumdis3, addnum64), sq_cnt);
        accumdis4 = _mm256_srl_epi64(_mm256_add_epi64(accumdis4, addnum64), sq_cnt);
        accumdis2 = _mm256_slli_si256(accumdis2, 4);
        accumdis1 = _mm256_blend_epi32(accumdis1, accumdis2, 0xAA);
        accumdis1 = _mm256_permutevar8x32_epi32(accumdis1, mask2);
        _mm256_storeu_si256(
            a8m::<u32, 8>(&mut tmp_dis[half + j..half + j + 8]),
            accumdis1,
        );
        accumdis4 = _mm256_slli_si256(accumdis4, 4);
        accumdis3 = _mm256_blend_epi32(accumdis3, accumdis4, 0xAA);
        accumdis3 = _mm256_permutevar8x32_epi32(accumdis3, mask2);
        _mm256_storeu_si256(
            a8m::<u32, 8>(&mut tmp_dis[half + j + 8..half + j + 16]),
            accumdis3,
        );
    }
    n16
}

/// Faithful port of libvmaf's `vif_statistic_16_avx2` horizontal pass for one
/// row. Mean sums share the mullo_epi32/add_epi64 pair-packing of the 8-bit
/// path; square terms accumulate via 4x u32 -> epi64 loads and mul_epu32 in
/// true column groups (0-3,4-7,8-11,12-15) then re-linearize through
/// slli/blend/permutevar8x32. Mean-square terms are also stored linearly here
/// (blend + shuffle_epi32(0xD8)). Returns columns processed (multiple of 16).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane(import_intrinsics)]
#[allow(clippy::too_many_arguments)]
fn vif_stat16_horizontal_v3(
    token: X64V3Token,
    tmp_mu1: &[u32],
    tmp_mu2: &[u32],
    tmp_ref: &[u32],
    tmp_dis: &[u32],
    tmp_ref_dis: &[u32],
    filt: &[u16],
    half: usize,
    width: usize,
    table: &[u16; 65536],
    gain_limit: f64,
    accums: (&mut i64, &mut i64, &mut i64, &mut i64),
) -> usize {
    let _ = token;
    let fwidth = 2 * half + 1;
    let n16 = width & !15;
    let mut xx = [0u32; 16];
    let mut yy = [0u32; 16];
    let mut xy = [0u32; 16];
    let mask1 = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
    let (num_log, den_log, num_non_log, den_non_log) = accums;

    for j in (0..n16).step_by(16) {
        // mu1 filtered sum (pair-packed in epi64 lanes; exact since the
        // accumulated 32-bit products cannot carry past bit 31)
        let fq = _mm256_set1_epi32(filt[half] as i32);
        let mut mu1_lo = _mm256_mullo_epi32(
            _mm256_loadu_si256(a8::<u32, 8>(&tmp_mu1[half + j..half + j + 8])),
            fq,
        );
        let mut mu1_hi = _mm256_mullo_epi32(
            _mm256_loadu_si256(a8::<u32, 8>(&tmp_mu1[half + j + 8..half + j + 16])),
            fq,
        );
        for (fj, &coeff) in filt.iter().enumerate().take(half) {
            let fq = _mm256_set1_epi32(coeff as i32);
            let l = j + fj;
            let r = j + fwidth - 1 - fj;
            mu1_lo = _mm256_add_epi64(
                mu1_lo,
                _mm256_mullo_epi32(_mm256_loadu_si256(a8::<u32, 8>(&tmp_mu1[l..l + 8])), fq),
            );
            mu1_hi = _mm256_add_epi64(
                mu1_hi,
                _mm256_mullo_epi32(
                    _mm256_loadu_si256(a8::<u32, 8>(&tmp_mu1[l + 8..l + 16])),
                    fq,
                ),
            );
            mu1_lo = _mm256_add_epi64(
                mu1_lo,
                _mm256_mullo_epi32(_mm256_loadu_si256(a8::<u32, 8>(&tmp_mu1[r..r + 8])), fq),
            );
            mu1_hi = _mm256_add_epi64(
                mu1_hi,
                _mm256_mullo_epi32(
                    _mm256_loadu_si256(a8::<u32, 8>(&tmp_mu1[r + 8..r + 16])),
                    fq,
                ),
            );
        }

        let zero = _mm256_setzero_si256();
        let round32 = _mm256_set1_epi64x(0x80000000);
        let acc0_lo = _mm256_srli_epi64(
            _mm256_add_epi64(
                _mm256_mul_epu32(
                    _mm256_unpacklo_epi32(mu1_lo, zero),
                    _mm256_unpacklo_epi32(mu1_lo, zero),
                ),
                round32,
            ),
            32,
        );
        let acc0_hi = _mm256_srli_epi64(
            _mm256_add_epi64(
                _mm256_mul_epu32(
                    _mm256_unpackhi_epi32(mu1_lo, zero),
                    _mm256_unpackhi_epi32(mu1_lo, zero),
                ),
                round32,
            ),
            32,
        );
        let acc1_lo = _mm256_srli_epi64(
            _mm256_add_epi64(
                _mm256_mul_epu32(
                    _mm256_unpacklo_epi32(mu1_hi, zero),
                    _mm256_unpacklo_epi32(mu1_hi, zero),
                ),
                round32,
            ),
            32,
        );
        let acc1_hi = _mm256_srli_epi64(
            _mm256_add_epi64(
                _mm256_mul_epu32(
                    _mm256_unpackhi_epi32(mu1_hi, zero),
                    _mm256_unpackhi_epi32(mu1_hi, zero),
                ),
                round32,
            ),
            32,
        );
        let mu1sq_lo = _mm256_shuffle_epi32(
            _mm256_blend_epi32(acc0_lo, _mm256_slli_si256(acc0_hi, 4), 0xAA),
            0xD8,
        );
        let mu1sq_hi = _mm256_shuffle_epi32(
            _mm256_blend_epi32(acc1_lo, _mm256_slli_si256(acc1_hi, 4), 0xAA),
            0xD8,
        );

        // mu2 filtered sum + mu2*mu2 and mu1*mu2 products
        let fq = _mm256_set1_epi32(filt[half] as i32);
        let mut acc0 = _mm256_mullo_epi32(
            _mm256_loadu_si256(a8::<u32, 8>(&tmp_mu2[half + j..half + j + 8])),
            fq,
        );
        let mut acc1 = _mm256_mullo_epi32(
            _mm256_loadu_si256(a8::<u32, 8>(&tmp_mu2[half + j + 8..half + j + 16])),
            fq,
        );
        for (fj, &coeff) in filt.iter().enumerate().take(half) {
            let fq = _mm256_set1_epi32(coeff as i32);
            let l = j + fj;
            let r = j + fwidth - 1 - fj;
            acc0 = _mm256_add_epi64(
                acc0,
                _mm256_mullo_epi32(_mm256_loadu_si256(a8::<u32, 8>(&tmp_mu2[l..l + 8])), fq),
            );
            acc1 = _mm256_add_epi64(
                acc1,
                _mm256_mullo_epi32(
                    _mm256_loadu_si256(a8::<u32, 8>(&tmp_mu2[l + 8..l + 16])),
                    fq,
                ),
            );
            acc0 = _mm256_add_epi64(
                acc0,
                _mm256_mullo_epi32(_mm256_loadu_si256(a8::<u32, 8>(&tmp_mu2[r..r + 8])), fq),
            );
            acc1 = _mm256_add_epi64(
                acc1,
                _mm256_mullo_epi32(
                    _mm256_loadu_si256(a8::<u32, 8>(&tmp_mu2[r + 8..r + 16])),
                    fq,
                ),
            );
        }

        let acc0_lo = _mm256_unpacklo_epi32(acc0, zero);
        let acc0_hi = _mm256_unpackhi_epi32(acc0, zero);
        let mut mu1lo_lo = _mm256_unpacklo_epi32(mu1_lo, zero);
        let mut mu1lo_hi = _mm256_unpackhi_epi32(mu1_lo, zero);
        let mut mu1hi_lo = _mm256_unpacklo_epi32(mu1_hi, zero);
        let mut mu1hi_hi = _mm256_unpackhi_epi32(mu1_hi, zero);

        mu1lo_lo = _mm256_srli_epi64(
            _mm256_add_epi64(_mm256_mul_epu32(mu1lo_lo, acc0_lo), round32),
            32,
        );
        mu1lo_hi = _mm256_srli_epi64(
            _mm256_add_epi64(_mm256_mul_epu32(mu1lo_hi, acc0_hi), round32),
            32,
        );
        let acc0_lo = _mm256_srli_epi64(
            _mm256_add_epi64(_mm256_mul_epu32(acc0_lo, acc0_lo), round32),
            32,
        );
        let acc0_hi = _mm256_srli_epi64(
            _mm256_add_epi64(_mm256_mul_epu32(acc0_hi, acc0_hi), round32),
            32,
        );

        let acc1_lo = _mm256_unpacklo_epi32(acc1, zero);
        let acc1_hi = _mm256_unpackhi_epi32(acc1, zero);
        mu1hi_lo = _mm256_srli_epi64(
            _mm256_add_epi64(_mm256_mul_epu32(mu1hi_lo, acc1_lo), round32),
            32,
        );
        mu1hi_hi = _mm256_srli_epi64(
            _mm256_add_epi64(_mm256_mul_epu32(mu1hi_hi, acc1_hi), round32),
            32,
        );
        let acc1_lo = _mm256_srli_epi64(
            _mm256_add_epi64(_mm256_mul_epu32(acc1_lo, acc1_lo), round32),
            32,
        );
        let acc1_hi = _mm256_srli_epi64(
            _mm256_add_epi64(_mm256_mul_epu32(acc1_hi, acc1_hi), round32),
            32,
        );

        let mu2sq_lo = _mm256_shuffle_epi32(
            _mm256_blend_epi32(acc0_lo, _mm256_slli_si256(acc0_hi, 4), 0xAA),
            0xD8,
        );
        let mu2sq_hi = _mm256_shuffle_epi32(
            _mm256_blend_epi32(acc1_lo, _mm256_slli_si256(acc1_hi, 4), 0xAA),
            0xD8,
        );
        let mu1mu2_lo = _mm256_shuffle_epi32(
            _mm256_blend_epi32(mu1lo_lo, _mm256_slli_si256(mu1lo_hi, 4), 0xAA),
            0xD8,
        );
        let mu1mu2_hi = _mm256_shuffle_epi32(
            _mm256_blend_epi32(mu1hi_lo, _mm256_slli_si256(mu1hi_hi, 4), 0xAA),
            0xD8,
        );

        // filtered ref^2 minus mu1^2
        {
            let rounder = _mm256_set1_epi64x(0x8000);
            let fq = _mm256_set1_epi64x(filt[half] as i64);
            let mut acc0 = _mm256_add_epi64(
                rounder,
                _mm256_mul_epu32(
                    _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                        &tmp_ref[half + j..half + j + 4],
                    ))),
                    fq,
                ),
            );
            let mut acc1 = _mm256_add_epi64(
                rounder,
                _mm256_mul_epu32(
                    _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                        &tmp_ref[half + j + 4..half + j + 8],
                    ))),
                    fq,
                ),
            );
            let mut acc2 = _mm256_add_epi64(
                rounder,
                _mm256_mul_epu32(
                    _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                        &tmp_ref[half + j + 8..half + j + 12],
                    ))),
                    fq,
                ),
            );
            let mut acc3 = _mm256_add_epi64(
                rounder,
                _mm256_mul_epu32(
                    _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                        &tmp_ref[half + j + 12..half + j + 16],
                    ))),
                    fq,
                ),
            );
            for (fj, &coeff) in filt.iter().enumerate().take(half) {
                let fq = _mm256_set1_epi64x(coeff as i64);
                let l = j + fj;
                let r = j + fwidth - 1 - fj;
                acc0 = _mm256_add_epi64(
                    acc0,
                    _mm256_mul_epu32(
                        _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(&tmp_ref[l..l + 4]))),
                        fq,
                    ),
                );
                acc1 = _mm256_add_epi64(
                    acc1,
                    _mm256_mul_epu32(
                        _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                            &tmp_ref[l + 4..l + 8],
                        ))),
                        fq,
                    ),
                );
                acc2 = _mm256_add_epi64(
                    acc2,
                    _mm256_mul_epu32(
                        _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                            &tmp_ref[l + 8..l + 12],
                        ))),
                        fq,
                    ),
                );
                acc3 = _mm256_add_epi64(
                    acc3,
                    _mm256_mul_epu32(
                        _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                            &tmp_ref[l + 12..l + 16],
                        ))),
                        fq,
                    ),
                );
                acc0 = _mm256_add_epi64(
                    acc0,
                    _mm256_mul_epu32(
                        _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(&tmp_ref[r..r + 4]))),
                        fq,
                    ),
                );
                acc1 = _mm256_add_epi64(
                    acc1,
                    _mm256_mul_epu32(
                        _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                            &tmp_ref[r + 4..r + 8],
                        ))),
                        fq,
                    ),
                );
                acc2 = _mm256_add_epi64(
                    acc2,
                    _mm256_mul_epu32(
                        _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                            &tmp_ref[r + 8..r + 12],
                        ))),
                        fq,
                    ),
                );
                acc3 = _mm256_add_epi64(
                    acc3,
                    _mm256_mul_epu32(
                        _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                            &tmp_ref[r + 12..r + 16],
                        ))),
                        fq,
                    ),
                );
            }
            acc0 = _mm256_srli_epi64(acc0, 16);
            acc1 = _mm256_srli_epi64(acc1, 16);
            acc2 = _mm256_srli_epi64(acc2, 16);
            acc3 = _mm256_srli_epi64(acc3, 16);

            acc1 = _mm256_slli_si256(acc1, 4);
            acc1 = _mm256_blend_epi32(acc0, acc1, 0xAA);
            acc0 = _mm256_permutevar8x32_epi32(acc1, mask1);
            acc3 = _mm256_slli_si256(acc3, 4);
            acc3 = _mm256_blend_epi32(acc2, acc3, 0xAA);
            acc1 = _mm256_permutevar8x32_epi32(acc3, mask1);

            acc0 = _mm256_sub_epi32(acc0, mu1sq_lo);
            acc1 = _mm256_sub_epi32(acc1, mu1sq_hi);
            _mm256_storeu_si256(a8m::<u32, 8>(&mut xx[..8]), acc0);
            _mm256_storeu_si256(a8m::<u32, 8>(&mut xx[8..16]), acc1);
        }

        // filtered dis^2 minus mu2^2
        {
            let rounder = _mm256_set1_epi64x(0x8000);
            let fq = _mm256_set1_epi64x(filt[half] as i64);
            let mut acc0 = _mm256_add_epi64(
                rounder,
                _mm256_mul_epu32(
                    _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                        &tmp_dis[half + j..half + j + 4],
                    ))),
                    fq,
                ),
            );
            let mut acc1 = _mm256_add_epi64(
                rounder,
                _mm256_mul_epu32(
                    _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                        &tmp_dis[half + j + 4..half + j + 8],
                    ))),
                    fq,
                ),
            );
            let mut acc2 = _mm256_add_epi64(
                rounder,
                _mm256_mul_epu32(
                    _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                        &tmp_dis[half + j + 8..half + j + 12],
                    ))),
                    fq,
                ),
            );
            let mut acc3 = _mm256_add_epi64(
                rounder,
                _mm256_mul_epu32(
                    _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                        &tmp_dis[half + j + 12..half + j + 16],
                    ))),
                    fq,
                ),
            );
            for (fj, &coeff) in filt.iter().enumerate().take(half) {
                let fq = _mm256_set1_epi64x(coeff as i64);
                let l = j + fj;
                let r = j + fwidth - 1 - fj;
                acc0 = _mm256_add_epi64(
                    acc0,
                    _mm256_mul_epu32(
                        _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(&tmp_dis[l..l + 4]))),
                        fq,
                    ),
                );
                acc1 = _mm256_add_epi64(
                    acc1,
                    _mm256_mul_epu32(
                        _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                            &tmp_dis[l + 4..l + 8],
                        ))),
                        fq,
                    ),
                );
                acc2 = _mm256_add_epi64(
                    acc2,
                    _mm256_mul_epu32(
                        _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                            &tmp_dis[l + 8..l + 12],
                        ))),
                        fq,
                    ),
                );
                acc3 = _mm256_add_epi64(
                    acc3,
                    _mm256_mul_epu32(
                        _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                            &tmp_dis[l + 12..l + 16],
                        ))),
                        fq,
                    ),
                );
                acc0 = _mm256_add_epi64(
                    acc0,
                    _mm256_mul_epu32(
                        _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(&tmp_dis[r..r + 4]))),
                        fq,
                    ),
                );
                acc1 = _mm256_add_epi64(
                    acc1,
                    _mm256_mul_epu32(
                        _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                            &tmp_dis[r + 4..r + 8],
                        ))),
                        fq,
                    ),
                );
                acc2 = _mm256_add_epi64(
                    acc2,
                    _mm256_mul_epu32(
                        _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                            &tmp_dis[r + 8..r + 12],
                        ))),
                        fq,
                    ),
                );
                acc3 = _mm256_add_epi64(
                    acc3,
                    _mm256_mul_epu32(
                        _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                            &tmp_dis[r + 12..r + 16],
                        ))),
                        fq,
                    ),
                );
            }
            acc0 = _mm256_srli_epi64(acc0, 16);
            acc1 = _mm256_srli_epi64(acc1, 16);
            acc2 = _mm256_srli_epi64(acc2, 16);
            acc3 = _mm256_srli_epi64(acc3, 16);

            acc1 = _mm256_slli_si256(acc1, 4);
            acc1 = _mm256_blend_epi32(acc0, acc1, 0xAA);
            acc0 = _mm256_permutevar8x32_epi32(acc1, mask1);
            acc3 = _mm256_slli_si256(acc3, 4);
            acc3 = _mm256_blend_epi32(acc2, acc3, 0xAA);
            acc1 = _mm256_permutevar8x32_epi32(acc3, mask1);

            acc0 = _mm256_sub_epi32(acc0, mu2sq_lo);
            acc1 = _mm256_sub_epi32(acc1, mu2sq_hi);
            _mm256_storeu_si256(a8m::<u32, 8>(&mut yy[..8]), _mm256_max_epi32(acc0, zero));
            _mm256_storeu_si256(a8m::<u32, 8>(&mut yy[8..16]), _mm256_max_epi32(acc1, zero));
        }

        // filtered ref*dis minus mu1*mu2
        {
            let rounder = _mm256_set1_epi64x(0x8000);
            let fq = _mm256_set1_epi64x(filt[half] as i64);
            let mut acc0 = _mm256_add_epi64(
                rounder,
                _mm256_mul_epu32(
                    _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                        &tmp_ref_dis[half + j..half + j + 4],
                    ))),
                    fq,
                ),
            );
            let mut acc1 = _mm256_add_epi64(
                rounder,
                _mm256_mul_epu32(
                    _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                        &tmp_ref_dis[half + j + 4..half + j + 8],
                    ))),
                    fq,
                ),
            );
            let mut acc2 = _mm256_add_epi64(
                rounder,
                _mm256_mul_epu32(
                    _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                        &tmp_ref_dis[half + j + 8..half + j + 12],
                    ))),
                    fq,
                ),
            );
            let mut acc3 = _mm256_add_epi64(
                rounder,
                _mm256_mul_epu32(
                    _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                        &tmp_ref_dis[half + j + 12..half + j + 16],
                    ))),
                    fq,
                ),
            );
            for (fj, &coeff) in filt.iter().enumerate().take(half) {
                let fq = _mm256_set1_epi64x(coeff as i64);
                let l = j + fj;
                let r = j + fwidth - 1 - fj;
                acc0 = _mm256_add_epi64(
                    acc0,
                    _mm256_mul_epu32(
                        _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                            &tmp_ref_dis[l..l + 4],
                        ))),
                        fq,
                    ),
                );
                acc1 = _mm256_add_epi64(
                    acc1,
                    _mm256_mul_epu32(
                        _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                            &tmp_ref_dis[l + 4..l + 8],
                        ))),
                        fq,
                    ),
                );
                acc2 = _mm256_add_epi64(
                    acc2,
                    _mm256_mul_epu32(
                        _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                            &tmp_ref_dis[l + 8..l + 12],
                        ))),
                        fq,
                    ),
                );
                acc3 = _mm256_add_epi64(
                    acc3,
                    _mm256_mul_epu32(
                        _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                            &tmp_ref_dis[l + 12..l + 16],
                        ))),
                        fq,
                    ),
                );
                acc0 = _mm256_add_epi64(
                    acc0,
                    _mm256_mul_epu32(
                        _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                            &tmp_ref_dis[r..r + 4],
                        ))),
                        fq,
                    ),
                );
                acc1 = _mm256_add_epi64(
                    acc1,
                    _mm256_mul_epu32(
                        _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                            &tmp_ref_dis[r + 4..r + 8],
                        ))),
                        fq,
                    ),
                );
                acc2 = _mm256_add_epi64(
                    acc2,
                    _mm256_mul_epu32(
                        _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                            &tmp_ref_dis[r + 8..r + 12],
                        ))),
                        fq,
                    ),
                );
                acc3 = _mm256_add_epi64(
                    acc3,
                    _mm256_mul_epu32(
                        _mm256_cvtepu32_epi64(_mm_loadu_si128(a8::<u32, 4>(
                            &tmp_ref_dis[r + 12..r + 16],
                        ))),
                        fq,
                    ),
                );
            }
            acc0 = _mm256_srli_epi64(acc0, 16);
            acc1 = _mm256_srli_epi64(acc1, 16);
            acc2 = _mm256_srli_epi64(acc2, 16);
            acc3 = _mm256_srli_epi64(acc3, 16);

            acc1 = _mm256_slli_si256(acc1, 4);
            acc1 = _mm256_blend_epi32(acc0, acc1, 0xAA);
            acc0 = _mm256_permutevar8x32_epi32(acc1, mask1);
            acc3 = _mm256_slli_si256(acc3, 4);
            acc3 = _mm256_blend_epi32(acc2, acc3, 0xAA);
            acc1 = _mm256_permutevar8x32_epi32(acc3, mask1);

            acc0 = _mm256_sub_epi32(acc0, mu1mu2_lo);
            acc1 = _mm256_sub_epi32(acc1, mu1mu2_hi);
            _mm256_storeu_si256(a8m::<u32, 8>(&mut xy[..8]), acc0);
            _mm256_storeu_si256(a8m::<u32, 8>(&mut xy[8..16]), acc1);
        }

        for b in 0..16 {
            vif_finalize_sigma(
                table,
                gain_limit,
                xx[b] as i32,
                yy[b] as i32,
                xy[b] as i32,
                &mut *num_log,
                &mut *den_log,
                &mut *num_non_log,
                &mut *den_non_log,
            );
        }
    }
    n16
}

#[inline(always)]
fn vif_finalize_sigma(
    table: &[u16; 65536],
    gain_limit: f64,
    sigma_ref: i32,
    sigma_dis: i32,
    sigma_ref_dis: i32,
    num_log: &mut i64,
    den_log: &mut i64,
    num_non_log: &mut i64,
    den_non_log: &mut i64,
) {
    let sigma_dis = sigma_dis.max(0);
    if sigma_ref >= SIGMA_NSQ {
        *den_log += (log2_32(table, (SIGMA_NSQ + sigma_ref) as u32) - 2048 * 17) as i64;
        if sigma_ref_dis > 0 && sigma_dis > 0 {
            let gain = sigma_ref_dis as f64 / (sigma_ref as f64 + 65536.0 * 1.0e-10);
            let residual = (sigma_dis as f64 - gain * sigma_ref_dis as f64) as i32;
            let residual = residual.max(0) as u32;
            let gain = gain.min(gain_limit);
            let first = residual + SIGMA_NSQ as u32;
            let second = (gain * gain * sigma_ref as f64) as i64 + first as i64;
            *num_log += (log2_64(table, second as u64) - log2_64(table, first as u64)) as i64;
        }
    } else {
        *num_non_log += sigma_dis as i64;
        *den_non_log += 1;
    }
}

fn vif_pixel_finalize(
    table: &[u16; 65536],
    gain_limit: f64,
    ref_mean: u32,
    dis_mean: u32,
    ref_sq_shifted: u32,
    dis_sq_shifted: u32,
    ref_dis_shifted: u32,
    num_log: &mut i64,
    den_log: &mut i64,
    num_non_log: &mut i64,
    den_non_log: &mut i64,
) {
    let ref_mean_sq = ((ref_mean as u64 * ref_mean as u64 + 2147483648) >> 32) as u32;
    let dis_mean_sq = ((dis_mean as u64 * dis_mean as u64 + 2147483648) >> 32) as u32;
    let mean_product = ((ref_mean as u64 * dis_mean as u64 + 2147483648) >> 32) as u32;
    let sigma_ref = ref_sq_shifted.wrapping_sub(ref_mean_sq) as i32;
    let sigma_dis = dis_sq_shifted.wrapping_sub(dis_mean_sq) as i32;
    let sigma_ref_dis = ref_dis_shifted.wrapping_sub(mean_product) as i32;
    vif_finalize_sigma(
        table,
        gain_limit,
        sigma_ref,
        sigma_dis,
        sigma_ref_dis,
        num_log,
        den_log,
        num_non_log,
        den_non_log,
    );
}

#[cfg(all(feature = "simd", feature = "avx512", target_arch = "x86_64"))]
#[inline(always)]
fn v4_token() -> Option<X64V4Token> {
    #[cfg(test)]
    if V3_DISABLED_FOR_TEST.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    X64V4Token::summon()
}

/// Lane gather into the u16 log table: identical to C's
/// `i32gather_epi64(..., log2_table, 2) & 0xffff` but bounds-checked — indices
/// are already in `[32768, 65536)` by construction (mantissa after srlv).
/// Scalar-index gather of 8 table entries whose indices sit in the low u32 of
/// each i64 lane — expands inline (a `#[rite]` fn can't force-inline through
/// `#[target_feature]` on stable, and per-iteration calls marshal zmm results
/// through the stack).
#[cfg(all(feature = "simd", feature = "avx512", target_arch = "x86_64"))]
macro_rules! gather_log_u16_v4 {
    ($table:expr, $idx:expr) => {{
        let idx = $idx;
        let mut a = [0i64; 8];
        _mm512_storeu_si512(a8m::<i64, 8>(&mut a), idx);
        // Mantissa indices are always < 65536 by construction; the &0xffff
        // mask is a no-op that makes LLVM prove it (no bounds checks).
        _mm512_setr_epi64(
            $table[(a[0] & 0xffff) as usize] as i64,
            $table[(a[1] & 0xffff) as usize] as i64,
            $table[(a[2] & 0xffff) as usize] as i64,
            $table[(a[3] & 0xffff) as usize] as i64,
            $table[(a[4] & 0xffff) as usize] as i64,
            $table[(a[5] & 0xffff) as usize] as i64,
            $table[(a[6] & 0xffff) as usize] as i64,
            $table[(a[7] & 0xffff) as usize] as i64,
        )
    }};
}

/// Vectorized port of libvmaf's `vif_statistic_avx512`: the per-16-block
/// finalize for `vif_statistic_8`. Matches `vif_finalize_sigma` bit-for-bit —
/// same log2 table approximation (k = 48 - lzcnt, `v >> k` mantissa index,
/// `2048*k` exponent), same truncating f64 divides, same masks. A macro so it
/// expands inside the caller's `#[arcane]` body (see `gather_log_u16_v4`).
#[cfg(all(feature = "simd", feature = "avx512", target_arch = "x86_64"))]
macro_rules! vif_statistic_finalize_v4 {
    ($xx:expr, $xy:expr, $yy:expr, $table:expr, $gain_limit:expr, $num_log:expr, $den_log:expr, $num_non_log:expr, $den_non_log:expr) => {{
        let xx = $xx;
        let xy = $xy;
        let yy = $yy;
        let table: &[u16; 65536] = $table;
        let gain_limit: f64 = $gain_limit;
        let num_log: &mut __m512i = $num_log;
        let den_log: &mut __m512i = $den_log;
        let num_non_log: &mut __m512i = $num_non_log;
        let den_non_log: &mut __m512i = $den_non_log;
        const SIGMA_NSQ_I64: i64 = 65536 << 1;
        let eps = 65536.0 * 1.0e-10;
        for iter in 0..2 {
        // Each pass consumes 8 i32 lanes; the second takes the high half.
        let take = |v: __m512i| -> __m256i {
            if iter == 0 {
                _mm512_castsi512_si256(v)
            } else {
                _mm512_extracti64x4_epi64(v, 1)
            }
        };
        let msigma1 = _mm512_cvtepi32_epi64(take(xx));
        let msigma2 = _mm512_cvtepi32_epi64(take(yy));
        let msigma12 = _mm512_cvtepi32_epi64(take(xy));
        let msigma2 = _mm512_max_epi64(msigma2, _mm512_setzero_si512());
        let msigma12 = _mm512_max_epi64(msigma12, _mm512_setzero_si512());

        // den = log2(sigma1 + sigma_nsq) - 2048*17
        let stage1 = _mm512_add_epi64(msigma1, _mm512_set1_epi64(SIGMA_NSQ_I64));
        let mnorm = _mm512_sub_epi64(_mm512_set1_epi64(48), _mm512_lzcnt_epi64(stage1));
        let mant = _mm512_srlv_epi64(stage1, mnorm);
        let mut mden_val = gather_log_u16_v4!(table, mant);
        mden_val = _mm512_add_epi64(mden_val, _mm512_slli_epi64(mnorm, 11));
        mden_val = _mm512_sub_epi64(mden_val, _mm512_set1_epi64(2048 * 17));

        let sigma1_small = _mm512_cmpgt_epi64_mask(_mm512_set1_epi64(SIGMA_NSQ_I64), msigma1);
        let sigma2_pos = _mm512_cmpgt_epi64_mask(msigma2, _mm512_setzero_si512());
        let sigma12_pos = _mm512_cmpgt_epi64_mask(msigma12, _mm512_setzero_si512());

        let msigma1_d = _mm512_cvtepu64_pd(msigma1);
        let mut mg = _mm512_div_pd(
            _mm512_cvtepu64_pd(msigma12),
            _mm512_add_pd(msigma1_d, _mm512_set1_pd(eps)),
        );
        let mut msv_sq = _mm512_cvttpd_epi64(_mm512_sub_pd(
            _mm512_cvtepi64_pd(msigma2),
            _mm512_mul_pd(mg, _mm512_cvtepi64_pd(msigma12)),
        ));
        msv_sq = _mm512_max_epi64(msv_sq, _mm512_setzero_si512());
        mg = _mm512_min_pd(mg, _mm512_set1_pd(gain_limit));

        // log2(residual + sigma_nsq)
        let numer1 = _mm512_add_epi64(msv_sq, _mm512_set1_epi64(SIGMA_NSQ_I64));
        let numer1_lz = _mm512_sub_epi64(_mm512_set1_epi64(48), _mm512_lzcnt_epi64(numer1));
        let numer1_log = _mm512_add_epi64(
            gather_log_u16_v4!(table, _mm512_srlv_epi64(numer1, numer1_lz)),
            _mm512_slli_epi64(numer1_lz, 11),
        );

        // log2(residual + sigma_nsq + trunc(gain^2 * sigma1))
        let numer1_tmp = _mm512_add_epi64(
            numer1,
            _mm512_cvttpd_epi64(_mm512_mul_pd(_mm512_mul_pd(mg, mg), msigma1_d)),
        );
        let numer1_tmp_lz =
            _mm512_sub_epi64(_mm512_set1_epi64(48), _mm512_lzcnt_epi64(numer1_tmp));
        let numer1_tmp_log = _mm512_add_epi64(
            gather_log_u16_v4!(table, _mm512_srlv_epi64(numer1_tmp, numer1_tmp_lz)),
            _mm512_slli_epi64(numer1_tmp_lz, 11),
        );

        let mnum_val = _mm512_sub_epi64(numer1_tmp_log, numer1_log);
        *num_log = _mm512_mask_add_epi64(
            *num_log,
            (!sigma1_small) & sigma12_pos & sigma2_pos,
            *num_log,
            mnum_val,
        );
        *den_log = _mm512_mask_add_epi64(*den_log, !sigma1_small, *den_log, mden_val);
        *num_non_log =
            _mm512_mask_add_epi64(*num_non_log, sigma1_small, *num_non_log, msigma2);
        *den_non_log = _mm512_mask_add_epi64(
            *den_non_log,
            sigma1_small,
            *den_non_log,
            _mm512_set1_epi64(1),
        );
        }
    }};
}

/// `vif_stat_horizontal_v4` filtered-square accumulator: `sum_t filt[t] *
/// tmp[..]` over the symmetric fwidth window, widened through cvtepu32_epi64
/// and accumulated in two i64×8 halves, then `(acc + 0x8000) >> 16` merged to
/// 16 u32 lanes via mask2. Mirrors C's `refsq/dissq/refdis` blocks. A macro so
/// it expands inside the caller's `#[arcane]` body (see `gather_log_u16_v4`).
#[cfg(all(feature = "simd", feature = "avx512", target_arch = "x86_64"))]
macro_rules! vif_stat_fsq_v4 {
    ($w:expr, $fqv:expr, $fwidth:expr, $rounder16:expr, $mask2:expr) => {{
        let w: &[u32; 33] = $w;
        let fqv: &[__m512i; 17] = $fqv;
        let fwidth: usize = $fwidth;
        let rounder16 = $rounder16;
        let mask2 = $mask2;
        let mut acc_lo = _mm512_set1_epi64(0x8000);
        let mut acc_hi = acc_lo;
        let _ = rounder16;
        for k in 0..fwidth.min(17) {
            let fq = fqv[k];
            let s0 = _mm512_cvtepu32_epi64(_mm256_loadu_si256(a8::<u32, 8>(&w[k..k + 8])));
            let s1 = _mm512_cvtepu32_epi64(_mm256_loadu_si256(a8::<u32, 8>(&w[k + 8..k + 16])));
            acc_lo = _mm512_add_epi64(acc_lo, _mm512_mul_epu32(s0, fq));
            acc_hi = _mm512_add_epi64(acc_hi, _mm512_mul_epu32(s1, fq));
        }
        let acc_lo = _mm512_srli_epi64(acc_lo, 16);
        let acc_hi = _mm512_srli_epi64(acc_hi, 16);
        _mm512_permutex2var_epi32(acc_lo, mask2, acc_hi)
    }};
}

/// AVX-512 port of `vif_statistic_8_avx512`'s horizontal pass for one row:
/// same 16-column block shape as the `_v3` kernels but zmm throughout, with
/// the per-block finalize vectorized (lzcnt + table gather + f64 divide)
/// instead of scalar `vif_finalize_sigma` calls. The horizontal math is
/// identical between the 8- and 16-bit v3 kernels (u32 tmp lanes either way),
/// so this one kernel serves both `vif_stat8` and `vif_stat16` rows. Returns
/// columns processed (a multiple of 16).
#[cfg(all(feature = "simd", feature = "avx512", target_arch = "x86_64"))]
#[arcane(import_intrinsics)]
#[allow(clippy::too_many_arguments)]
fn vif_stat_horizontal_v4(
    _token: X64V4Token,
    tmp_mu1: &[u32],
    tmp_mu2: &[u32],
    tmp_ref: &[u32],
    tmp_dis: &[u32],
    tmp_ref_dis: &[u32],
    filt: &[u16],
    half: usize,
    width: usize,
    table: &[u16; 65536],
    gain_limit: f64,
    accums: (&mut i64, &mut i64, &mut i64, &mut i64),
) -> usize {
    let fwidth = 2 * half + 1;
    let n16 = width & !15;
    let zero = _mm512_setzero_si512();
    let round32 = _mm512_set1_epi64(0x80000000);
    let rounder16 = _mm512_set1_epi64(0x8000);
    let mask5 = _mm512_set_epi32(30, 28, 14, 12, 26, 24, 10, 8, 22, 20, 6, 4, 18, 16, 2, 0);
    let mask2 = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    let mut m_num_log = _mm512_setzero_si512();
    let mut m_den_log = _mm512_setzero_si512();
    let mut m_num_non_log = _mm512_setzero_si512();
    let mut m_den_non_log = _mm512_setzero_si512();
    let (num_log, den_log, num_non_log_out, den_non_log_out) = accums;

    // Padded filter arrays: index by k < fwidth ≤ 17 — provably in bounds.
    let mut fqv32 = [_mm512_setzero_si512(); 17];
    let mut fqv64 = [_mm512_setzero_si512(); 17];
    for (k, &c) in filt.iter().enumerate() {
        fqv32[k] = _mm512_set1_epi32(c as i32);
        fqv64[k] = _mm512_set1_epi64(c as i64);
    }
    assert!(fwidth <= 17 && half <= 8);
    // For j <= n16-16 the window [j, j+33) fits: j+33 <= n16+17 <= padded.
    // Hoisting the fact once lets LLVM drop the five per-iter checks.
    assert!(n16 + 17 <= tmp_mu1.len());
    assert!(n16 + 17 <= tmp_mu2.len());
    assert!(n16 + 17 <= tmp_ref.len());
    assert!(n16 + 17 <= tmp_dis.len());
    assert!(n16 + 17 <= tmp_ref_dis.len());

    for j in (0..n16).step_by(16) {
        // 33-wide windows per block: every tap index k..k+16 is provably in
        // bounds, so the inner loops carry no bounds checks.
        let w_mu1: &[u32; 33] = tmp_mu1[j..j + 33].try_into().unwrap();
        let w_mu2: &[u32; 33] = tmp_mu2[j..j + 33].try_into().unwrap();
        let w_ref: &[u32; 33] = tmp_ref[j..j + 33].try_into().unwrap();
        let w_dis: &[u32; 33] = tmp_dis[j..j + 33].try_into().unwrap();
        let w_rd: &[u32; 33] = tmp_ref_dis[j..j + 33].try_into().unwrap();

        // mu1/mu2 horizontal sums — i32 products accumulate pairwise inside
        // the i64 lanes, identical to the AVX2 path (sums stay < 2^32 per
        // half). The filter is symmetric, so the straight convolution over
        // k ∈ [0,fwidth) is bit-identical to the paired l/r form.
        let fq = fqv32[half];
        let mut mu1 = _mm512_mullo_epi32(
            _mm512_loadu_si512(a8::<u32, 16>(&w_mu1[half..half + 16])),
            fq,
        );
        let mut mu2 = _mm512_mullo_epi32(
            _mm512_loadu_si512(a8::<u32, 16>(&w_mu2[half..half + 16])),
            fq,
        );
        for k in 0..fwidth.min(17) {
            if k == half {
                continue;
            }
            let fq = fqv32[k];
            mu1 = _mm512_add_epi64(
                mu1,
                _mm512_mullo_epi32(_mm512_loadu_si512(a8::<u32, 16>(&w_mu1[k..k + 16])), fq),
            );
            mu2 = _mm512_add_epi64(
                mu2,
                _mm512_mullo_epi32(_mm512_loadu_si512(a8::<u32, 16>(&w_mu2[k..k + 16])), fq),
            );
        }

        // mu1sq / mu2sq / mu1mu2: widen each i32 pair to i64, square/multiply,
        // round-shift back, then even-lane merge via permutex2var (mask5).
        let mu1_lo = _mm512_unpacklo_epi32(mu1, zero);
        let mu1_hi = _mm512_unpackhi_epi32(mu1, zero);
        let mu1sq = {
            let lo = _mm512_srli_epi64(
                _mm512_add_epi64(_mm512_mul_epu32(mu1_lo, mu1_lo), round32),
                32,
            );
            let hi = _mm512_srli_epi64(
                _mm512_add_epi64(_mm512_mul_epu32(mu1_hi, mu1_hi), round32),
                32,
            );
            _mm512_permutex2var_epi32(lo, mask5, hi)
        };
        let mu2_lo = _mm512_unpacklo_epi32(mu2, zero);
        let mu2_hi = _mm512_unpackhi_epi32(mu2, zero);
        let mu1mu2 = {
            let lo = _mm512_srli_epi64(
                _mm512_add_epi64(_mm512_mul_epu32(mu1_lo, mu2_lo), round32),
                32,
            );
            let hi = _mm512_srli_epi64(
                _mm512_add_epi64(_mm512_mul_epu32(mu1_hi, mu2_hi), round32),
                32,
            );
            _mm512_permutex2var_epi32(lo, mask5, hi)
        };
        let mu2sq = {
            let lo = _mm512_srli_epi64(
                _mm512_add_epi64(_mm512_mul_epu32(mu2_lo, mu2_lo), round32),
                32,
            );
            let hi = _mm512_srli_epi64(
                _mm512_add_epi64(_mm512_mul_epu32(mu2_hi, mu2_hi), round32),
                32,
            );
            _mm512_permutex2var_epi32(lo, mask5, hi)
        };

        // filtered ref²/dis²/ref·dis (u32 lanes widened to i64 accumulators,
        // srli 16, even-lane merge via mask2), minus the matching mu term.
        let xx = _mm512_sub_epi32(
            vif_stat_fsq_v4!(w_ref, &fqv64, fwidth, rounder16, mask2),
            mu1sq,
        );
        let yy = _mm512_max_epi32(
            _mm512_sub_epi32(
                vif_stat_fsq_v4!(w_dis, &fqv64, fwidth, rounder16, mask2),
                mu2sq,
            ),
            zero,
        );
        let xy = _mm512_sub_epi32(
            vif_stat_fsq_v4!(w_rd, &fqv64, fwidth, rounder16, mask2),
            mu1mu2,
        );
        vif_statistic_finalize_v4!(
            xx,
            xy,
            yy,
            table,
            gain_limit,
            &mut m_num_log,
            &mut m_den_log,
            &mut m_num_non_log,
            &mut m_den_non_log
        );
    }
    *num_log += _mm512_reduce_add_epi64(m_num_log);
    *den_log += _mm512_reduce_add_epi64(m_den_log);
    *num_non_log_out += _mm512_reduce_add_epi64(m_num_non_log);
    *den_non_log_out += _mm512_reduce_add_epi64(m_den_non_log);
    n16
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline(always)]
fn v3_token() -> Option<X64V3Token> {
    #[cfg(test)]
    if V3_DISABLED_FOR_TEST.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    X64V3Token::summon()
}

#[cfg(all(test, feature = "simd", target_arch = "x86_64"))]
static V3_DISABLED_FOR_TEST: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn statistics(
    image: &VifImage<'_>,
    bit_depth: u8,
    scale: usize,
    gain_limit: f64,
    table: &[u16; 65536],
) -> (f32, f32) {
    let filter = FILTERS[scale];
    let half = (filter.len() / 2) as isize;
    let (width, height) = (image.width, image.height);
    let (shift_mean, round_mean, shift_square, round_square) = if bit_depth == 8 && scale == 0 {
        (8, 128u64, 0, 0u64)
    } else if scale == 0 {
        (
            bit_depth as u32,
            1u64 << (bit_depth - 1),
            2 * (bit_depth as u32 - 8),
            1u64 << (2 * (bit_depth - 8) - 1),
        )
    } else {
        (16, 32768, 16, 32768)
    };
    // +17 tail: the v4 horizontal reads a fixed [u32;33] window per 16-col
    // block so every tap index is compile-time provable (no per-load bounds
    // checks); indices past `padded` are read but never used.
    let padded = width + 2 * half as usize + 17;
    let mut vertical_ref_mean = pool::take_u32(padded);
    let mut vertical_dis_mean = pool::take_u32(padded);
    let mut vertical_ref_sq = pool::take_u32(padded);
    let mut vertical_dis_sq = pool::take_u32(padded);
    let mut vertical_ref_dis = pool::take_u32(padded);
    let mut num_log = 0i64;
    let mut den_log = 0i64;
    let mut num_non_log = 0i64;
    let mut den_non_log = 0i64;
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    let x64v3 = v3_token();
    for row in 0..height {
        let mut row_offsets = [0usize; 17];
        for (tap, slot) in row_offsets.iter_mut().take(filter.len()).enumerate() {
            *slot = mirror(row as isize + tap as isize - half, height) * width;
        }
        #[allow(unused_mut)]
        let mut processed = 0;
        #[cfg(feature = "simd")]
        if !(bit_depth == 8 && scale == 0) {
            #[cfg(target_arch = "x86_64")]
            let avx2 = if let Some(token) = x64v3 {
                processed = vif_stat16_vertical_v3(
                    token,
                    &image.reference,
                    &image.distorted,
                    &row_offsets,
                    filter,
                    half as usize,
                    width,
                    shift_mean,
                    round_mean as u32,
                    shift_square,
                    round_square,
                    &mut vertical_ref_mean,
                    &mut vertical_dis_mean,
                    &mut vertical_ref_sq,
                    &mut vertical_dis_sq,
                    &mut vertical_ref_dis,
                );
                true
            } else {
                false
            };
            #[cfg(not(target_arch = "x86_64"))]
            let avx2 = false;
            if !avx2 {
                if scale > 0 {
                    processed = archmage::incant!(
                        vif_vertical_hiscale_simd(
                            &image.reference,
                            &image.distorted,
                            &row_offsets,
                            width,
                            scale,
                            &mut vertical_ref_mean,
                            &mut vertical_dis_mean,
                            &mut vertical_ref_sq,
                            &mut vertical_dis_sq,
                            &mut vertical_ref_dis
                        ),
                        [v3, neon, wasm128, scalar]
                    );
                }
                #[cfg(feature = "simd")]
                if bit_depth == 10 && scale == 0 {
                    processed = archmage::incant!(
                        vif_vertical_u10_simd(
                            &image.reference,
                            &image.distorted,
                            &row_offsets,
                            width,
                            &mut vertical_ref_mean,
                            &mut vertical_dis_mean,
                            &mut vertical_ref_sq,
                            &mut vertical_dis_sq,
                            &mut vertical_ref_dis
                        ),
                        [v3, neon, wasm128, scalar]
                    );
                }
            }
        }
        #[cfg(all(feature = "avx512", feature = "simd", target_arch = "x86_64"))]
        if bit_depth == 8 && scale == 0 && half % 2 == 0 {
            if let Some(token) = v4_token() {
                vif_stat8_vertical_v4(
                    token,
                    &image.reference,
                    &image.distorted,
                    &row_offsets,
                    filter,
                    half as usize,
                    width,
                    &mut vertical_ref_mean,
                    &mut vertical_dis_mean,
                    &mut vertical_ref_sq,
                    &mut vertical_dis_sq,
                    &mut vertical_ref_dis,
                );
                processed = width;
            }
        }
        #[cfg(feature = "simd")]
        if bit_depth == 8 && scale == 0 && processed < width {
            #[cfg(target_arch = "x86_64")]
            let avx2 = if let Some(token) = x64v3 {
                vif_stat8_vertical_v3(
                    token,
                    &image.reference,
                    &image.distorted,
                    &row_offsets,
                    filter,
                    half as usize,
                    width,
                    &mut vertical_ref_mean,
                    &mut vertical_dis_mean,
                    &mut vertical_ref_sq,
                    &mut vertical_dis_sq,
                    &mut vertical_ref_dis,
                );
                processed = width;
                true
            } else {
                false
            };
            #[cfg(not(target_arch = "x86_64"))]
            let avx2 = false;
            if !avx2 {
                processed = archmage::incant!(
                    vif_vertical_u8_simd(
                        &image.reference,
                        &image.distorted,
                        &row_offsets,
                        width,
                        &mut vertical_ref_mean,
                        &mut vertical_dis_mean,
                        &mut vertical_ref_sq,
                        &mut vertical_dis_sq,
                        &mut vertical_ref_dis
                    ),
                    [v3, neon, wasm128, scalar]
                );
            }
        }
        for col in processed..width {
            let center = half as usize;
            let ref_value = image.reference[row_offsets[center] + col] as u32;
            let dis_value = image.distorted[row_offsets[center] + col] as u32;
            let weight = filter[center] as u32;
            let mut ref_mean = weight * ref_value;
            let mut dis_mean = weight * dis_value;
            let mut ref_sq = ref_mean as u64 * ref_value as u64;
            let mut dis_sq = dis_mean as u64 * dis_value as u64;
            let mut ref_dis = ref_mean as u64 * dis_value as u64;
            for offset in 1..=center {
                let weight = filter[center - offset] as u32;
                let left = row_offsets[center - offset] + col;
                let right = row_offsets[center + offset] + col;
                let left_ref = image.reference[left] as u32;
                let right_ref = image.reference[right] as u32;
                let left_dis = image.distorted[left] as u32;
                let right_dis = image.distorted[right] as u32;
                let ref_value = left_ref + right_ref;
                let dis_value = left_dis + right_dis;
                let weighted_ref = weight * ref_value;
                let weighted_dis = weight * dis_value;
                ref_mean += weighted_ref;
                dis_mean += weighted_dis;
                ref_sq += weight as u64
                    * (left_ref as u64 * left_ref as u64 + right_ref as u64 * right_ref as u64);
                dis_sq += weight as u64
                    * (left_dis as u64 * left_dis as u64 + right_dis as u64 * right_dis as u64);
                ref_dis += weight as u64
                    * (left_ref as u64 * left_dis as u64 + right_ref as u64 * right_dis as u64);
            }
            let slot = col + half as usize;
            vertical_ref_mean[slot] = ((ref_mean as u64 + round_mean) >> shift_mean) as u16 as u32;
            vertical_dis_mean[slot] = ((dis_mean as u64 + round_mean) >> shift_mean) as u16 as u32;
            vertical_ref_sq[slot] = ((ref_sq + round_square) >> shift_square) as u32;
            vertical_dis_sq[slot] = ((dis_sq + round_square) >> shift_square) as u32;
            vertical_ref_dis[slot] = ((ref_dis + round_square) >> shift_square) as u32;
        }
        for scratch in [
            &mut vertical_ref_mean,
            &mut vertical_dis_mean,
            &mut vertical_ref_sq,
            &mut vertical_dis_sq,
            &mut vertical_ref_dis,
        ] {
            pad_reflected(scratch, width, half as usize);
        }
        #[allow(unused_mut)]
        let mut hcol = 0;
        #[cfg(feature = "simd")]
        {
            #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
            if bit_depth == 8
                && scale == 0
                && let Some(token) = v4_token()
            {
                hcol = vif_stat_horizontal_v4(
                    token,
                    &vertical_ref_mean,
                    &vertical_dis_mean,
                    &vertical_ref_sq,
                    &vertical_dis_sq,
                    &vertical_ref_dis,
                    filter,
                    half as usize,
                    width,
                    table,
                    gain_limit,
                    (
                        &mut num_log,
                        &mut den_log,
                        &mut num_non_log,
                        &mut den_non_log,
                    ),
                );
            }
            #[cfg(target_arch = "x86_64")]
            if hcol == 0
                && bit_depth == 8
                && scale == 0
                && let Some(token) = x64v3
            {
                hcol = vif_stat8_horizontal_v3(
                    token,
                    &vertical_ref_mean,
                    &vertical_dis_mean,
                    &vertical_ref_sq,
                    &vertical_dis_sq,
                    &vertical_ref_dis,
                    filter,
                    half as usize,
                    width,
                    table,
                    gain_limit,
                    (
                        &mut num_log,
                        &mut den_log,
                        &mut num_non_log,
                        &mut den_non_log,
                    ),
                );
            }
            #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
            if hcol == 0
                && !(bit_depth == 8 && scale == 0)
                && let Some(token) = v4_token()
            {
                // Same horizontal math as the 8-bit path; the v4 kernel is
                // width/bit-depth agnostic on the u32 tmp planes.
                hcol = vif_stat_horizontal_v4(
                    token,
                    &vertical_ref_mean,
                    &vertical_dis_mean,
                    &vertical_ref_sq,
                    &vertical_dis_sq,
                    &vertical_ref_dis,
                    filter,
                    half as usize,
                    width,
                    table,
                    gain_limit,
                    (
                        &mut num_log,
                        &mut den_log,
                        &mut num_non_log,
                        &mut den_non_log,
                    ),
                );
            }
            #[cfg(target_arch = "x86_64")]
            if hcol == 0
                && !(bit_depth == 8 && scale == 0)
                && let Some(token) = x64v3
            {
                hcol = vif_stat16_horizontal_v3(
                    token,
                    &vertical_ref_mean,
                    &vertical_dis_mean,
                    &vertical_ref_sq,
                    &vertical_dis_sq,
                    &vertical_ref_dis,
                    filter,
                    half as usize,
                    width,
                    table,
                    gain_limit,
                    (
                        &mut num_log,
                        &mut den_log,
                        &mut num_non_log,
                        &mut den_non_log,
                    ),
                );
            }
            while hcol + 16 <= width {
                let (ref_means, dis_means, ref_sqs, dis_sqs, ref_diss) = archmage::incant!(
                    vif_horizontal_sums(
                        &vertical_ref_mean,
                        &vertical_dis_mean,
                        &vertical_ref_sq,
                        &vertical_dis_sq,
                        &vertical_ref_dis,
                        filter,
                        hcol
                    ),
                    [v3, neon, wasm128, scalar]
                );
                for lane in 0..16 {
                    vif_pixel_finalize(
                        table,
                        gain_limit,
                        ref_means[lane],
                        dis_means[lane],
                        ref_sqs[lane],
                        dis_sqs[lane],
                        ref_diss[lane],
                        &mut num_log,
                        &mut den_log,
                        &mut num_non_log,
                        &mut den_non_log,
                    );
                }
                hcol += 16;
            }
        }
        for col in hcol..width {
            let center = col + half as usize;
            let weight = filter[half as usize] as u32;
            let mut ref_mean = weight * vertical_ref_mean[center];
            let mut dis_mean = weight * vertical_dis_mean[center];
            let mut ref_sq = weight as u64 * vertical_ref_sq[center] as u64;
            let mut dis_sq = weight as u64 * vertical_dis_sq[center] as u64;
            let mut ref_dis = weight as u64 * vertical_ref_dis[center] as u64;
            for offset in 1..=half as usize {
                let weight = filter[half as usize - offset] as u32;
                let left = center - offset;
                let right = center + offset;
                ref_mean += weight * (vertical_ref_mean[left] + vertical_ref_mean[right]);
                dis_mean += weight * (vertical_dis_mean[left] + vertical_dis_mean[right]);
                ref_sq +=
                    weight as u64 * (vertical_ref_sq[left] as u64 + vertical_ref_sq[right] as u64);
                dis_sq +=
                    weight as u64 * (vertical_dis_sq[left] as u64 + vertical_dis_sq[right] as u64);
                ref_dis += weight as u64
                    * (vertical_ref_dis[left] as u64 + vertical_ref_dis[right] as u64);
            }
            vif_pixel_finalize(
                table,
                gain_limit,
                ref_mean,
                dis_mean,
                ((ref_sq + 32768) >> 16) as u32,
                ((dis_sq + 32768) >> 16) as u32,
                ((ref_dis + 32768) >> 16) as u32,
                &mut num_log,
                &mut den_log,
                &mut num_non_log,
                &mut den_non_log,
            );
        }
    }
    let num = (num_log as f64 / 2048.0
        + (den_non_log as f64 - num_non_log as f64 / 16384.0 / 65025.0)) as f32;
    let den = (den_log as f64 / 2048.0 + den_non_log as f64) as f32;
    pool::give_u32(vertical_ref_mean);
    pool::give_u32(vertical_dis_mean);
    pool::give_u32(vertical_ref_sq);
    pool::give_u32(vertical_dis_sq);
    pool::give_u32(vertical_ref_dis);
    (num, den)
}

pub fn vif_v0_from_luma(
    reference_y: &[u16],
    distorted_y: &[u16],
    width: usize,
    height: usize,
    bit_depth: u8,
    variant: VmafV0Variant,
) -> Result<[f64; 4], Error> {
    if !matches!(bit_depth, 8 | 10) {
        return Err(Error::InvalidInput("unsupported bit depth"));
    }
    if width < 17 || height < 17 {
        return Err(Error::InvalidInput("dimensions too small"));
    }
    let pixels = width
        .checked_mul(height)
        .filter(|&value| value <= isize::MAX as usize)
        .ok_or(Error::InvalidInput("dimension overflow"))?;
    if reference_y.len() != pixels || distorted_y.len() != pixels {
        return Err(Error::InvalidInput("plane length mismatch"));
    }
    let max_sample = (1u16 << bit_depth) - 1;
    if reference_y
        .iter()
        .chain(distorted_y.iter())
        .any(|&value| value > max_sample)
    {
        return Err(Error::InvalidInput("sample exceeds bit depth"));
    }
    let mut image = VifImage {
        reference: Cow::Borrowed(reference_y),
        distorted: Cow::Borrowed(distorted_y),
        width,
        height,
    };
    let table = log_table();
    let limit = if variant.no_enhancement_gain() {
        1.0
    } else {
        100.0
    };
    let mut scores = [0.0; 4];
    for (scale, score) in scores.iter_mut().enumerate() {
        let (num, den) = statistics(&image, bit_depth, scale, limit, table);
        *score = (num / den) as f64;
        if scale != 3 {
            let next = subsample(&image, bit_depth, scale);
            if let Cow::Owned(v) = image.reference {
                pool::give_u16(v);
            }
            if let Cow::Owned(v) = image.distorted {
                pool::give_u16(v);
            }
            image = next;
        }
    }
    if let Cow::Owned(v) = image.reference {
        pool::give_u16(v);
    }
    if let Cow::Owned(v) = image.distorted {
        pool::give_u16(v);
    }
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "simd")]
    fn scalar_vertical_u8(
        reference: &[u16],
        distorted: &[u16],
        row_offsets: &[usize; 17],
        buffers: &mut [Vec<u32>; 5],
        from: usize,
        to: usize,
    ) {
        let filter = FILTERS[0];
        for col in from..to {
            let weight = filter[8] as u32;
            let ref_value = reference[row_offsets[8] + col] as u32;
            let dis_value = distorted[row_offsets[8] + col] as u32;
            let mut ref_mean = weight * ref_value;
            let mut dis_mean = weight * dis_value;
            let mut ref_sq = ref_mean * ref_value;
            let mut dis_sq = dis_mean * dis_value;
            let mut ref_dis = ref_mean * dis_value;
            for offset in 1..=8usize {
                let weight = filter[8 - offset] as u32;
                let left = row_offsets[8 - offset] + col;
                let right = row_offsets[8 + offset] + col;
                let left_ref = reference[left] as u32;
                let right_ref = reference[right] as u32;
                let left_dis = distorted[left] as u32;
                let right_dis = distorted[right] as u32;
                ref_mean += weight * (left_ref + right_ref);
                dis_mean += weight * (left_dis + right_dis);
                ref_sq += weight * (left_ref * left_ref + right_ref * right_ref);
                dis_sq += weight * (left_dis * left_dis + right_dis * right_dis);
                ref_dis += weight * (left_ref * left_dis + right_ref * right_dis);
            }
            let slot = col + 8;
            buffers[0][slot] = ((ref_mean + 128) >> 8) as u16 as u32;
            buffers[1][slot] = ((dis_mean + 128) >> 8) as u16 as u32;
            buffers[2][slot] = ref_sq;
            buffers[3][slot] = dis_sq;
            buffers[4][slot] = ref_dis;
        }
    }

    #[cfg(feature = "simd")]
    fn scalar_vertical_u10(
        reference: &[u16],
        distorted: &[u16],
        row_offsets: &[usize; 17],
        buffers: &mut [Vec<u32>; 5],
        from: usize,
        to: usize,
    ) {
        let filter = FILTERS[0];
        for col in from..to {
            let weight = filter[8] as u32;
            let ref_value = reference[row_offsets[8] + col] as u32;
            let dis_value = distorted[row_offsets[8] + col] as u32;
            let mut ref_mean = weight * ref_value;
            let mut dis_mean = weight * dis_value;
            let mut ref_sq = ref_mean as u64 * ref_value as u64;
            let mut dis_sq = dis_mean as u64 * dis_value as u64;
            let mut ref_dis = ref_mean as u64 * dis_value as u64;
            for offset in 1..=8usize {
                let weight = filter[8 - offset] as u32;
                let left = row_offsets[8 - offset] + col;
                let right = row_offsets[8 + offset] + col;
                let left_ref = reference[left] as u32;
                let right_ref = reference[right] as u32;
                let left_dis = distorted[left] as u32;
                let right_dis = distorted[right] as u32;
                ref_mean += weight * (left_ref + right_ref);
                dis_mean += weight * (left_dis + right_dis);
                ref_sq += weight as u64
                    * (left_ref as u64 * left_ref as u64 + right_ref as u64 * right_ref as u64);
                dis_sq += weight as u64
                    * (left_dis as u64 * left_dis as u64 + right_dis as u64 * right_dis as u64);
                ref_dis += weight as u64
                    * (left_ref as u64 * left_dis as u64 + right_ref as u64 * right_dis as u64);
            }
            let slot = col + 8;
            buffers[0][slot] = ((ref_mean as u64 + 512) >> 10) as u16 as u32;
            buffers[1][slot] = ((dis_mean as u64 + 512) >> 10) as u16 as u32;
            buffers[2][slot] = ((ref_sq + 8) >> 4) as u32;
            buffers[3][slot] = ((dis_sq + 8) >> 4) as u32;
            buffers[4][slot] = ((ref_dis + 8) >> 4) as u32;
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn simd_vertical_u10_matches_scalar_formula() {
        for (width, height) in [
            (17, 17),
            (31, 17),
            (32, 17),
            (33, 17),
            (17, 33),
            (31, 33),
            (32, 33),
            (33, 33),
        ] {
            for case in 0..3 {
                let reference: Vec<u16> = (0..height)
                    .flat_map(|y| {
                        (0..width).map(move |x| {
                            (match case {
                                0 => (x * 37 + y * 61 + x * y * 5) % 1024,
                                1 => 1023,
                                _ => (x * 211 + y * 149 + 17) % 1024,
                            }) as u16
                        })
                    })
                    .collect();
                let distorted: Vec<u16> = (0..height)
                    .flat_map(|y| {
                        (0..width).map(move |x| {
                            (match case {
                                0 => (x * 23 + y * 41 + 9) % 1024,
                                1 => {
                                    if (x + y) % 2 == 0 {
                                        1023
                                    } else {
                                        0
                                    }
                                }
                                _ => (x * 157 + y * 89 + 31) % 1024,
                            }) as u16
                        })
                    })
                    .collect();
                for row in [0usize, 1, 8, height - 1] {
                    let mut row_offsets = [0usize; 17];
                    for (tap, slot) in row_offsets.iter_mut().enumerate() {
                        *slot = mirror(row as isize + tap as isize - 8, height) * width;
                    }
                    let mut expected: [Vec<u32>; 5] = [
                        vec![0; width + 16],
                        vec![0; width + 16],
                        vec![0; width + 16],
                        vec![0; width + 16],
                        vec![0; width + 16],
                    ];
                    scalar_vertical_u10(
                        &reference,
                        &distorted,
                        &row_offsets,
                        &mut expected,
                        0,
                        width,
                    );
                    for scalar_tier in [false, true] {
                        let mut actual: [Vec<u32>; 5] = [
                            vec![0; width + 16],
                            vec![0; width + 16],
                            vec![0; width + 16],
                            vec![0; width + 16],
                            vec![0; width + 16],
                        ];
                        let [rm, dm, rsq, dsq, rd] = actual.each_mut();
                        let processed = if scalar_tier {
                            vif_vertical_u10_simd_scalar(
                                archmage::ScalarToken,
                                &reference,
                                &distorted,
                                &row_offsets,
                                width,
                                rm,
                                dm,
                                rsq,
                                dsq,
                                rd,
                            )
                        } else {
                            archmage::incant!(
                                vif_vertical_u10_simd(
                                    &reference,
                                    &distorted,
                                    &row_offsets,
                                    width,
                                    rm,
                                    dm,
                                    rsq,
                                    dsq,
                                    rd
                                ),
                                [v3, neon, wasm128, scalar]
                            )
                        };
                        assert_eq!(processed, width / 16 * 16);
                        scalar_vertical_u10(
                            &reference,
                            &distorted,
                            &row_offsets,
                            &mut actual,
                            processed,
                            width,
                        );
                        for (band, (expected, actual)) in
                            expected.iter().zip(actual.iter()).enumerate()
                        {
                            assert_eq!(
                                expected, actual,
                                "{width}x{height}, case {case}, row {row}, scalar_tier {scalar_tier}, buffer {band}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[cfg(feature = "simd")]
    fn scalar_vertical_hiscale(
        reference: &[u16],
        distorted: &[u16],
        row_offsets: &[usize; 17],
        scale: usize,
        buffers: &mut [Vec<u32>; 5],
        from: usize,
        to: usize,
    ) {
        let filter = FILTERS[scale];
        let half = filter.len() / 2;
        for col in from..to {
            let weight = filter[half] as u32;
            let ref_value = reference[row_offsets[half] + col] as u32;
            let dis_value = distorted[row_offsets[half] + col] as u32;
            let mut ref_mean = weight * ref_value;
            let mut dis_mean = weight * dis_value;
            let mut ref_sq = ref_mean as u64 * ref_value as u64;
            let mut dis_sq = dis_mean as u64 * dis_value as u64;
            let mut ref_dis = ref_mean as u64 * dis_value as u64;
            for offset in 1..=half {
                let weight = filter[half - offset] as u32;
                let left = row_offsets[half - offset] + col;
                let right = row_offsets[half + offset] + col;
                let left_ref = reference[left] as u32;
                let right_ref = reference[right] as u32;
                let left_dis = distorted[left] as u32;
                let right_dis = distorted[right] as u32;
                ref_mean += weight * (left_ref + right_ref);
                dis_mean += weight * (left_dis + right_dis);
                ref_sq += weight as u64
                    * (left_ref as u64 * left_ref as u64 + right_ref as u64 * right_ref as u64);
                dis_sq += weight as u64
                    * (left_dis as u64 * left_dis as u64 + right_dis as u64 * right_dis as u64);
                ref_dis += weight as u64
                    * (left_ref as u64 * left_dis as u64 + right_ref as u64 * right_dis as u64);
            }
            let slot = col + half;
            buffers[0][slot] = ((ref_mean as u64 + 32768) >> 16) as u16 as u32;
            buffers[1][slot] = ((dis_mean as u64 + 32768) >> 16) as u16 as u32;
            buffers[2][slot] = ((ref_sq + 32768) >> 16) as u32;
            buffers[3][slot] = ((dis_sq + 32768) >> 16) as u32;
            buffers[4][slot] = ((ref_dis + 32768) >> 16) as u32;
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn simd_vertical_hiscale_matches_scalar_formula() {
        for (scale, filter) in FILTERS.iter().enumerate().skip(1) {
            let half = filter.len() / 2;
            let padded_half = 8;
            for (width, height) in [
                (17, 17),
                (31, 17),
                (32, 17),
                (33, 17),
                (17, 33),
                (31, 33),
                (32, 33),
                (33, 33),
            ] {
                for case in 0..4 {
                    let reference: Vec<u16> = (0..height)
                        .flat_map(|y| {
                            (0..width).map(move |x| {
                                (match case {
                                    0 => (x * 501 + y * 977 + x * y * 13) % 65536,
                                    1 => 65535,
                                    2 => (x * 3001 + y * 1999 + 77) % 65536,
                                    _ => 1023,
                                }) as u16
                            })
                        })
                        .collect();
                    let distorted: Vec<u16> = (0..height)
                        .flat_map(|y| {
                            (0..width).map(move |x| {
                                (match case {
                                    0 => (x * 401 + y * 733 + 19) % 65536,
                                    1 => {
                                        if (x + y) % 2 == 0 {
                                            65535
                                        } else {
                                            0
                                        }
                                    }
                                    2 => (x * 2503 + y * 1543 + 91) % 65536,
                                    _ => 1023,
                                }) as u16
                            })
                        })
                        .collect();
                    for row in [0usize, 1, half, height - 1] {
                        let mut row_offsets = [0usize; 17];
                        for (tap, slot) in row_offsets.iter_mut().take(filter.len()).enumerate() {
                            *slot =
                                mirror(row as isize + tap as isize - half as isize, height) * width;
                        }
                        let mut expected: [Vec<u32>; 5] = [
                            vec![0; width + 2 * padded_half],
                            vec![0; width + 2 * padded_half],
                            vec![0; width + 2 * padded_half],
                            vec![0; width + 2 * padded_half],
                            vec![0; width + 2 * padded_half],
                        ];
                        scalar_vertical_hiscale(
                            &reference,
                            &distorted,
                            &row_offsets,
                            scale,
                            &mut expected,
                            0,
                            width,
                        );
                        for scalar_tier in [false, true] {
                            let mut actual: [Vec<u32>; 5] = [
                                vec![0; width + 2 * padded_half],
                                vec![0; width + 2 * padded_half],
                                vec![0; width + 2 * padded_half],
                                vec![0; width + 2 * padded_half],
                                vec![0; width + 2 * padded_half],
                            ];
                            let [rm, dm, rsq, dsq, rd] = actual.each_mut();
                            let processed = if scalar_tier {
                                vif_vertical_hiscale_simd_scalar(
                                    archmage::ScalarToken,
                                    &reference,
                                    &distorted,
                                    &row_offsets,
                                    width,
                                    scale,
                                    rm,
                                    dm,
                                    rsq,
                                    dsq,
                                    rd,
                                )
                            } else {
                                archmage::incant!(
                                    vif_vertical_hiscale_simd(
                                        &reference,
                                        &distorted,
                                        &row_offsets,
                                        width,
                                        scale,
                                        rm,
                                        dm,
                                        rsq,
                                        dsq,
                                        rd
                                    ),
                                    [v3, neon, wasm128, scalar]
                                )
                            };
                            assert_eq!(processed, width / 16 * 16);
                            scalar_vertical_hiscale(
                                &reference,
                                &distorted,
                                &row_offsets,
                                scale,
                                &mut actual,
                                processed,
                                width,
                            );
                            for (band, (expected, actual)) in
                                expected.iter().zip(actual.iter()).enumerate()
                            {
                                assert_eq!(
                                    expected, actual,
                                    "scale {scale}, {width}x{height}, case {case}, row {row}, scalar_tier {scalar_tier}, buffer {band}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn simd_vertical_u8_matches_scalar_formula() {
        for (width, height) in [
            (17, 17),
            (31, 17),
            (32, 17),
            (33, 17),
            (17, 33),
            (31, 33),
            (32, 33),
            (33, 33),
        ] {
            for case in 0..3 {
                let reference: Vec<u16> = (0..height)
                    .flat_map(|y| {
                        (0..width).map(move |x| {
                            (match case {
                                0 => (x * 13 + y * 29 + x * y) % 256,
                                1 => 255,
                                _ => (x * 97 + y * 53 + 11) % 256,
                            }) as u16
                        })
                    })
                    .collect();
                let distorted: Vec<u16> = (0..height)
                    .flat_map(|y| {
                        (0..width).map(move |x| {
                            (match case {
                                0 => (x * 7 + y * 19 + 3) % 256,
                                1 => {
                                    if (x + y) % 3 == 0 {
                                        255
                                    } else {
                                        0
                                    }
                                }
                                _ => (x * 31 + y * 71 + 5) % 256,
                            }) as u16
                        })
                    })
                    .collect();
                for row in [0usize, 1, 8, height - 1] {
                    let mut row_offsets = [0usize; 17];
                    for (tap, slot) in row_offsets.iter_mut().enumerate() {
                        *slot = mirror(row as isize + tap as isize - 8, height) * width;
                    }
                    let mut expected: [Vec<u32>; 5] = [
                        vec![0; width + 16],
                        vec![0; width + 16],
                        vec![0; width + 16],
                        vec![0; width + 16],
                        vec![0; width + 16],
                    ];
                    scalar_vertical_u8(
                        &reference,
                        &distorted,
                        &row_offsets,
                        &mut expected,
                        0,
                        width,
                    );
                    for scalar_tier in [false, true] {
                        let mut actual: [Vec<u32>; 5] = [
                            vec![0; width + 16],
                            vec![0; width + 16],
                            vec![0; width + 16],
                            vec![0; width + 16],
                            vec![0; width + 16],
                        ];
                        let [rm, dm, rsq, dsq, rd] = actual.each_mut();
                        let processed = if scalar_tier {
                            vif_vertical_u8_simd_scalar(
                                archmage::ScalarToken,
                                &reference,
                                &distorted,
                                &row_offsets,
                                width,
                                rm,
                                dm,
                                rsq,
                                dsq,
                                rd,
                            )
                        } else {
                            archmage::incant!(
                                vif_vertical_u8_simd(
                                    &reference,
                                    &distorted,
                                    &row_offsets,
                                    width,
                                    rm,
                                    dm,
                                    rsq,
                                    dsq,
                                    rd
                                ),
                                [v3, neon, wasm128, scalar]
                            )
                        };
                        assert_eq!(processed, width / 16 * 16);
                        scalar_vertical_u8(
                            &reference,
                            &distorted,
                            &row_offsets,
                            &mut actual,
                            processed,
                            width,
                        );
                        for (band, (expected, actual)) in
                            expected.iter().zip(actual.iter()).enumerate()
                        {
                            assert_eq!(
                                expected, actual,
                                "{width}x{height}, case {case}, row {row}, scalar_tier {scalar_tier}, buffer {band}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn reflected_padding_matches_border_indices() {
        assert!(
            FILTERS
                .iter()
                .all(|filter| filter.iter().eq(filter.iter().rev()))
        );
        for (width, half) in [(2, 1), (4, 2), (8, 4), (17, 8)] {
            let mut padded = vec![0u32; width + 2 * half];
            for col in 0..width {
                padded[col + half] = (col * 41 + 7) as u32;
            }
            pad_reflected(&mut padded, width, half);
            for col in 0..width {
                for tap in 0..=2 * half {
                    assert_eq!(
                        padded[col + tap],
                        padded[half + mirror(col as isize + tap as isize - half as isize, width)]
                    );
                }
            }
        }
    }

    #[test]
    fn odd_frame_dimensions_remain_supported() {
        for (width, height) in [(17, 17), (35, 33)] {
            for bit_depth in [8, 10] {
                let reference: Vec<_> = (0..width * height)
                    .map(|index| ((index * 41 + index / width * 7) % (1 << bit_depth)) as u16)
                    .collect();
                let scores = vif_v0_from_luma(
                    &reference,
                    &reference,
                    width,
                    height,
                    bit_depth,
                    VmafV0Variant::Standard,
                )
                .unwrap();
                assert!(scores.iter().all(|score| score.is_finite() && *score > 0.0));
            }
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn v3_stat8_matches_simd_fallback_at_odd_widths() {
        for (width, height) in [
            (16, 17),
            (17, 17),
            (31, 9),
            (33, 17),
            (47, 33),
            (64, 9),
            (19, 9),
        ] {
            for case in 0..3 {
                let reference: Vec<u16> = (0..height)
                    .flat_map(|y| {
                        (0..width).map(move |x| {
                            (match case {
                                0 => (x * 37 + y * 61 + x * y * 5) % 256,
                                1 => 255,
                                _ => ((x * 211 + y * 149 + 17) % 251) + (x % 3),
                            }) as u16
                        })
                    })
                    .collect();
                let distorted: Vec<u16> = (0..height)
                    .flat_map(|y| {
                        (0..width).map(move |x| {
                            (match case {
                                0 => (x * 23 + y * 41 + 9) % 256,
                                1 => {
                                    if (x + y) % 2 == 0 {
                                        255
                                    } else {
                                        0
                                    }
                                }
                                _ => (x * 157 + y * 89 + 31) % 256,
                            }) as u16
                        })
                    })
                    .collect();
                let image = VifImage {
                    reference: Cow::Borrowed(reference.as_slice()),
                    distorted: Cow::Borrowed(distorted.as_slice()),
                    width,
                    height,
                };
                let table = log_table();
                V3_DISABLED_FOR_TEST.store(true, std::sync::atomic::Ordering::Relaxed);
                let fallback = statistics(&image, 8, 0, 100.0, table);
                V3_DISABLED_FOR_TEST.store(false, std::sync::atomic::Ordering::Relaxed);
                let avx2 = statistics(&image, 8, 0, 100.0, table);
                assert_eq!(
                    (fallback.0.to_bits(), fallback.1.to_bits()),
                    (avx2.0.to_bits(), avx2.1.to_bits()),
                    "width={width} height={height} case={case}"
                );
            }
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn v3_stat16_matches_simd_fallback_at_odd_widths() {
        // The _16 kernel covers every (bit_depth, scale) pair except 8-bit
        // scale-0: 10-bit scale-0 plus scales 1-3 at any depth. Heights start
        // at 2*half+1 (the smallest legal mirror radius for each filter) to
        // stress edge-dominated rows, plus interior heights.
        for (bit_depth, scale) in [(8, 1), (8, 2), (8, 3), (10, 0), (10, 1), (10, 2), (10, 3)] {
            let half = FILTERS[scale].len() / 2;
            let max_val = 1usize << bit_depth;
            for width in [16, 17, 31, 33, 47, 64, 19] {
                for height in [2 * half + 1, 9, 17] {
                    for case in 0..3 {
                        let reference: Vec<u16> = (0..height)
                            .flat_map(|y| {
                                (0..width).map(move |x| {
                                    (match case {
                                        0 => (x * 37 + y * 61 + x * y * 5) % max_val,
                                        1 => max_val - 1,
                                        _ => ((x * 211 + y * 149 + 17) % (max_val - 5)) + (x % 3),
                                    }) as u16
                                })
                            })
                            .collect();
                        let distorted: Vec<u16> = (0..height)
                            .flat_map(|y| {
                                (0..width).map(move |x| {
                                    (match case {
                                        0 => (x * 23 + y * 41 + 9) % max_val,
                                        1 => {
                                            if (x + y) % 2 == 0 {
                                                max_val - 1
                                            } else {
                                                0
                                            }
                                        }
                                        _ => (x * 157 + y * 89 + 31) % max_val,
                                    }) as u16
                                })
                            })
                            .collect();
                        let image = VifImage {
                            reference: Cow::Borrowed(reference.as_slice()),
                            distorted: Cow::Borrowed(distorted.as_slice()),
                            width,
                            height,
                        };
                        let table = log_table();
                        V3_DISABLED_FOR_TEST.store(true, std::sync::atomic::Ordering::Relaxed);
                        let fallback = statistics(&image, bit_depth, scale, 100.0, table);
                        V3_DISABLED_FOR_TEST.store(false, std::sync::atomic::Ordering::Relaxed);
                        let avx2 = statistics(&image, bit_depth, scale, 100.0, table);
                        assert_eq!(
                            (fallback.0.to_bits(), fallback.1.to_bits()),
                            (avx2.0.to_bits(), avx2.1.to_bits()),
                            "bpc={bit_depth} scale={scale} width={width} height={height} case={case}"
                        );
                    }
                }
            }
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn v3_subsample_matches_simd_fallback_at_odd_widths() {
        // subsample() handles the scale transitions 0->1 (9-tap), 1->2 (5-tap),
        // 2->3 (3-tap) at every bit depth. Heights start at 2*half+1 (smallest
        // legal mirror radius for each filter).
        for (bit_depth, scale) in [(8, 0), (8, 1), (8, 2), (10, 0), (10, 1), (10, 2)] {
            let half = FILTERS[scale + 1].len() / 2;
            let max_val = 1usize << bit_depth;
            for width in [16, 17, 31, 33, 47, 64, 19] {
                for height in [4 * half + 2, 18, 34] {
                    for case in 0..3 {
                        let reference: Vec<u16> = (0..height)
                            .flat_map(|y| {
                                (0..width).map(move |x| {
                                    (match case {
                                        0 => (x * 37 + y * 61 + x * y * 5) % max_val,
                                        1 => max_val - 1,
                                        _ => ((x * 211 + y * 149 + 17) % (max_val - 5)) + (x % 3),
                                    }) as u16
                                })
                            })
                            .collect();
                        let distorted: Vec<u16> = (0..height)
                            .flat_map(|y| {
                                (0..width).map(move |x| {
                                    (match case {
                                        0 => (x * 23 + y * 41 + 9) % max_val,
                                        1 => {
                                            if (x + y) % 2 == 0 {
                                                max_val - 1
                                            } else {
                                                0
                                            }
                                        }
                                        _ => (x * 157 + y * 89 + 31) % max_val,
                                    }) as u16
                                })
                            })
                            .collect();
                        let image = VifImage {
                            reference: Cow::Borrowed(reference.as_slice()),
                            distorted: Cow::Borrowed(distorted.as_slice()),
                            width,
                            height,
                        };
                        V3_DISABLED_FOR_TEST.store(true, std::sync::atomic::Ordering::Relaxed);
                        let fallback = subsample(&image, bit_depth, scale);
                        V3_DISABLED_FOR_TEST.store(false, std::sync::atomic::Ordering::Relaxed);
                        let avx2 = subsample(&image, bit_depth, scale);
                        assert_eq!(
                            (
                                fallback.width,
                                fallback.height,
                                fallback.reference.as_ref(),
                                fallback.distorted.as_ref()
                            ),
                            (
                                avx2.width,
                                avx2.height,
                                avx2.reference.as_ref(),
                                avx2.distorted.as_ref()
                            ),
                            "bpc={bit_depth} scale={scale} width={width} height={height} case={case}"
                        );
                    }
                }
            }
        }
    }
}
