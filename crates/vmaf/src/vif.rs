#[cfg(feature = "simd")]
use archmage::magetypes;
use std::borrow::Cow;
use std::sync::OnceLock;

use crate::{Error, VmafV0Variant};

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

fn subsample<'a>(image: &VifImage<'a>, bit_depth: u8, scale: usize) -> VifImage<'a> {
    let filter = FILTERS[scale + 1];
    let half = (filter.len() / 2) as isize;
    let (width, height) = (image.width, image.height);
    let (out_width, out_height) = (width / 2, height / 2);
    let mut out_reference = vec![0; out_width * out_height];
    let mut out_distorted = vec![0; out_width * out_height];
    let mut vertical_reference = vec![0u32; width + 2 * half as usize];
    let mut vertical_distorted = vec![0u32; width + 2 * half as usize];
    for row in (0..height / 2 * 2).step_by(2) {
        let mut row_offsets = [0usize; 9];
        for (tap, slot) in row_offsets.iter_mut().take(filter.len()).enumerate() {
            *slot = mirror(row as isize + tap as isize - half, height) * width;
        }
        for col in 0..width {
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
                let shift = if scale == 0 { bit_depth } else { 16 };
                let round = 1u32 << (shift - 1);
                vertical_reference[col + half as usize] =
                    ((ref_sum + round) >> shift) as u16 as u32;
                vertical_distorted[col + half as usize] =
                    ((dis_sum + round) >> shift) as u16 as u32;
            }
        }
        pad_reflected(&mut vertical_reference, width, half as usize);
        pad_reflected(&mut vertical_distorted, width, half as usize);
        for col in (0..width / 2 * 2).step_by(2) {
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
        let center_ref = u16x16::from_slice(token, &reference[row_offsets[8] + col..]);
        let center_dis = u16x16::from_slice(token, &distorted[row_offsets[8] + col..]);
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
            let left_ref = u16x16::from_slice(token, &reference[row_offsets[8 - offset] + col..]);
            let right_ref = u16x16::from_slice(token, &reference[row_offsets[8 + offset] + col..]);
            let left_dis = u16x16::from_slice(token, &distorted[row_offsets[8 - offset] + col..]);
            let right_dis = u16x16::from_slice(token, &distorted[row_offsets[8 + offset] + col..]);
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
        let center_ref = u16x16::from_slice(token, &reference[row_offsets[8] + col..]);
        let center_dis = u16x16::from_slice(token, &distorted[row_offsets[8] + col..]);
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
            let left_ref = u16x16::from_slice(token, &reference[row_offsets[8 - offset] + col..]);
            let right_ref = u16x16::from_slice(token, &reference[row_offsets[8 + offset] + col..]);
            let left_dis = u16x16::from_slice(token, &distorted[row_offsets[8 - offset] + col..]);
            let right_dis = u16x16::from_slice(token, &distorted[row_offsets[8 + offset] + col..]);
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
        let center_ref = u16x16::from_slice(token, &reference[row_offsets[half] + col..]);
        let center_dis = u16x16::from_slice(token, &distorted[row_offsets[half] + col..]);
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
            let left_ref =
                u16x16::from_slice(token, &reference[row_offsets[half - offset] + col..]);
            let right_ref =
                u16x16::from_slice(token, &reference[row_offsets[half + offset] + col..]);
            let left_dis =
                u16x16::from_slice(token, &distorted[row_offsets[half - offset] + col..]);
            let right_dis =
                u16x16::from_slice(token, &distorted[row_offsets[half + offset] + col..]);
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
type VifHorizontalSums = ([u32; 16], [u32; 16], [u64; 16], [u64; 16], [u64; 16]);

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
        let rlo = u32x8::from_slice(token, &vertical_ref_mean[col + tap..]);
        let rhi = u32x8::from_slice(token, &vertical_ref_mean[col + 8 + tap..]);
        let dlo = u32x8::from_slice(token, &vertical_dis_mean[col + tap..]);
        let dhi = u32x8::from_slice(token, &vertical_dis_mean[col + 8 + tap..]);
        ref_mean_lo += weight * rlo;
        ref_mean_hi += weight * rhi;
        dis_mean_lo += weight * dlo;
        dis_mean_hi += weight * dhi;
        let sq_lo = u32x8::from_slice(token, &vertical_ref_sq[col + tap..]);
        let sq_hi = u32x8::from_slice(token, &vertical_ref_sq[col + 8 + tap..]);
        ref_sq_lo16 += weight * (sq_lo & mask16);
        ref_sq_hi16 += weight * sq_lo.shr_logical_uniform(16);
        ref_sq_lo8 += weight * (sq_hi & mask16);
        ref_sq_hi8 += weight * sq_hi.shr_logical_uniform(16);
        let ds_lo = u32x8::from_slice(token, &vertical_dis_sq[col + tap..]);
        let ds_hi = u32x8::from_slice(token, &vertical_dis_sq[col + 8 + tap..]);
        dis_sq_lo16 += weight * (ds_lo & mask16);
        dis_sq_hi16 += weight * ds_lo.shr_logical_uniform(16);
        dis_sq_lo8 += weight * (ds_hi & mask16);
        dis_sq_hi8 += weight * ds_hi.shr_logical_uniform(16);
        let rd_lo = u32x8::from_slice(token, &vertical_ref_dis[col + tap..]);
        let rd_hi = u32x8::from_slice(token, &vertical_ref_dis[col + 8 + tap..]);
        ref_dis_lo16 += weight * (rd_lo & mask16);
        ref_dis_hi16 += weight * rd_lo.shr_logical_uniform(16);
        ref_dis_lo8 += weight * (rd_hi & mask16);
        ref_dis_hi8 += weight * rd_hi.shr_logical_uniform(16);
    }
    let mut ref_mean_out = [0u32; 16];
    let mut dis_mean_out = [0u32; 16];
    let mut ref_sq_out = [0u64; 16];
    let mut dis_sq_out = [0u64; 16];
    let mut ref_dis_out = [0u64; 16];
    let mean_lo = ref_mean_lo.to_array();
    let mean_hi = ref_mean_hi.to_array();
    let dmean_lo = dis_mean_lo.to_array();
    let dmean_hi = dis_mean_hi.to_array();
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
        ref_mean_out[lane] = mean_lo[lane];
        ref_mean_out[8 + lane] = mean_hi[lane];
        dis_mean_out[lane] = dmean_lo[lane];
        dis_mean_out[8 + lane] = dmean_hi[lane];
        ref_sq_out[lane] = sq_lo16[lane] as u64 + ((sq_hi16[lane] as u64) << 16);
        ref_sq_out[8 + lane] = sq_lo8[lane] as u64 + ((sq_hi8[lane] as u64) << 16);
        dis_sq_out[lane] = dsq_lo16[lane] as u64 + ((dsq_hi16[lane] as u64) << 16);
        dis_sq_out[8 + lane] = dsq_lo8[lane] as u64 + ((dsq_hi8[lane] as u64) << 16);
        ref_dis_out[lane] = rd_lo16[lane] as u64 + ((rd_hi16[lane] as u64) << 16);
        ref_dis_out[8 + lane] = rd_lo8[lane] as u64 + ((rd_hi8[lane] as u64) << 16);
    }
    (
        ref_mean_out,
        dis_mean_out,
        ref_sq_out,
        dis_sq_out,
        ref_dis_out,
    )
}

fn vif_pixel_finalize(
    table: &[u16; 65536],
    gain_limit: f64,
    ref_mean: u32,
    dis_mean: u32,
    ref_sq: u64,
    dis_sq: u64,
    ref_dis: u64,
    num_log: &mut i64,
    den_log: &mut i64,
    num_non_log: &mut i64,
    den_non_log: &mut i64,
) {
    let ref_mean_sq = ((ref_mean as u64 * ref_mean as u64 + 2147483648) >> 32) as u32;
    let dis_mean_sq = ((dis_mean as u64 * dis_mean as u64 + 2147483648) >> 32) as u32;
    let mean_product = ((ref_mean as u64 * dis_mean as u64 + 2147483648) >> 32) as u32;
    let sigma_ref = (((ref_sq + 32768) >> 16) as u32).wrapping_sub(ref_mean_sq) as i32;
    let sigma_dis = (((dis_sq + 32768) >> 16) as u32).wrapping_sub(dis_mean_sq) as i32;
    let sigma_ref_dis = (((ref_dis + 32768) >> 16) as u32).wrapping_sub(mean_product) as i32;
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
    let padded = width + 2 * half as usize;
    let mut vertical_ref_mean = vec![0u32; padded];
    let mut vertical_dis_mean = vec![0u32; padded];
    let mut vertical_ref_sq = vec![0u32; padded];
    let mut vertical_dis_sq = vec![0u32; padded];
    let mut vertical_ref_dis = vec![0u32; padded];
    let mut num_log = 0i64;
    let mut den_log = 0i64;
    let mut num_non_log = 0i64;
    let mut den_non_log = 0i64;
    for row in 0..height {
        let mut row_offsets = [0usize; 17];
        for (tap, slot) in row_offsets.iter_mut().take(filter.len()).enumerate() {
            *slot = mirror(row as isize + tap as isize - half, height) * width;
        }
        #[allow(unused_mut)]
        let mut processed = 0;
        #[cfg(feature = "simd")]
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
        #[cfg(feature = "simd")]
        if bit_depth == 8 && scale == 0 {
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
                ref_sq,
                dis_sq,
                ref_dis,
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
            image = subsample(&image, bit_depth, scale);
        }
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
}
