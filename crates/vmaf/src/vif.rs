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

struct VifImage {
    reference: Vec<u16>,
    distorted: Vec<u16>,
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

fn subsample(image: &VifImage, bit_depth: u8, scale: usize) -> VifImage {
    let filter = FILTERS[scale + 1];
    let half = (filter.len() / 2) as isize;
    let (width, height) = (image.width, image.height);
    let mut output = VifImage {
        reference: vec![0; (width / 2) * (height / 2)],
        distorted: vec![0; (width / 2) * (height / 2)],
        width: width / 2,
        height: height / 2,
    };
    let mut vertical_reference = vec![0u32; width];
    let mut vertical_distorted = vec![0u32; width];
    for row in (0..height).step_by(2) {
        for col in 0..width {
            let mut ref_sum = 0u32;
            let mut dis_sum = 0u32;
            for (tap, &weight) in filter.iter().enumerate() {
                let y = mirror(row as isize + tap as isize - half, height);
                ref_sum += weight as u32 * image.reference[y * width + col] as u32;
                dis_sum += weight as u32 * image.distorted[y * width + col] as u32;
            }
            if bit_depth == 8 && scale == 0 {
                vertical_reference[col] = (ref_sum + 128) >> 8;
                vertical_distorted[col] = (dis_sum + 128) >> 8;
            } else {
                let shift = if scale == 0 { bit_depth } else { 16 };
                let round = 1u32 << (shift - 1);
                vertical_reference[col] = ((ref_sum + round) >> shift) as u16 as u32;
                vertical_distorted[col] = ((dis_sum + round) >> shift) as u16 as u32;
            }
        }
        for col in (0..width).step_by(2) {
            let mut ref_sum = 0u32;
            let mut dis_sum = 0u32;
            for (tap, &weight) in filter.iter().enumerate() {
                let x = mirror(col as isize + tap as isize - half, width);
                ref_sum += weight as u32 * vertical_reference[x];
                dis_sum += weight as u32 * vertical_distorted[x];
            }
            let slot = (row / 2) * output.width + col / 2;
            if row / 2 < output.height && col / 2 < output.width {
                output.reference[slot] = ((ref_sum + 32768) >> 16) as u16;
                output.distorted[slot] = ((dis_sum + 32768) >> 16) as u16;
            }
        }
    }
    output
}

fn statistics(
    image: &VifImage,
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
    let mut vertical_ref_mean = vec![0u32; width];
    let mut vertical_dis_mean = vec![0u32; width];
    let mut vertical_ref_sq = vec![0u32; width];
    let mut vertical_dis_sq = vec![0u32; width];
    let mut vertical_ref_dis = vec![0u32; width];
    let mut num_log = 0i64;
    let mut den_log = 0i64;
    let mut num_non_log = 0i64;
    let mut den_non_log = 0i64;
    for row in 0..height {
        for col in 0..width {
            let mut ref_mean = 0u32;
            let mut dis_mean = 0u32;
            let mut ref_sq = 0u64;
            let mut dis_sq = 0u64;
            let mut ref_dis = 0u64;
            for (tap, &weight) in filter.iter().enumerate() {
                let y = mirror(row as isize + tap as isize - half, height);
                let ref_value = image.reference[y * width + col] as u32;
                let dis_value = image.distorted[y * width + col] as u32;
                let weighted_ref = weight as u32 * ref_value;
                let weighted_dis = weight as u32 * dis_value;
                ref_mean += weighted_ref;
                dis_mean += weighted_dis;
                ref_sq += weighted_ref as u64 * ref_value as u64;
                dis_sq += weighted_dis as u64 * dis_value as u64;
                ref_dis += weighted_ref as u64 * dis_value as u64;
            }
            vertical_ref_mean[col] = ((ref_mean as u64 + round_mean) >> shift_mean) as u16 as u32;
            vertical_dis_mean[col] = ((dis_mean as u64 + round_mean) >> shift_mean) as u16 as u32;
            vertical_ref_sq[col] = ((ref_sq + round_square) >> shift_square) as u32;
            vertical_dis_sq[col] = ((dis_sq + round_square) >> shift_square) as u32;
            vertical_ref_dis[col] = ((ref_dis + round_square) >> shift_square) as u32;
        }
        for col in 0..width {
            let mut ref_mean = 0u32;
            let mut dis_mean = 0u32;
            let mut ref_sq = 0u64;
            let mut dis_sq = 0u64;
            let mut ref_dis = 0u64;
            for (tap, &weight) in filter.iter().enumerate() {
                let x = mirror(col as isize + tap as isize - half, width);
                ref_mean += weight as u32 * vertical_ref_mean[x];
                dis_mean += weight as u32 * vertical_dis_mean[x];
                ref_sq += weight as u64 * vertical_ref_sq[x] as u64;
                dis_sq += weight as u64 * vertical_dis_sq[x] as u64;
                ref_dis += weight as u64 * vertical_ref_dis[x] as u64;
            }
            let ref_mean_sq = ((ref_mean as u64 * ref_mean as u64 + 2147483648) >> 32) as u32;
            let dis_mean_sq = ((dis_mean as u64 * dis_mean as u64 + 2147483648) >> 32) as u32;
            let mean_product = ((ref_mean as u64 * dis_mean as u64 + 2147483648) >> 32) as u32;
            let sigma_ref = (((ref_sq + 32768) >> 16) as u32).wrapping_sub(ref_mean_sq) as i32;
            let sigma_dis = (((dis_sq + 32768) >> 16) as u32).wrapping_sub(dis_mean_sq) as i32;
            let sigma_ref_dis =
                (((ref_dis + 32768) >> 16) as u32).wrapping_sub(mean_product) as i32;
            let sigma_dis = sigma_dis.max(0);
            if sigma_ref >= SIGMA_NSQ {
                den_log += (log2_32(table, (SIGMA_NSQ + sigma_ref) as u32) - 2048 * 17) as i64;
                if sigma_ref_dis > 0 && sigma_dis > 0 {
                    let gain = sigma_ref_dis as f64 / (sigma_ref as f64 + 65536.0 * 1.0e-10);
                    let residual = (sigma_dis as f64 - gain * sigma_ref_dis as f64) as i32;
                    let residual = residual.max(0) as u32;
                    let gain = gain.min(gain_limit);
                    let first = residual + SIGMA_NSQ as u32;
                    let second = (gain * gain * sigma_ref as f64) as i64 + first as i64;
                    num_log +=
                        (log2_64(table, second as u64) - log2_64(table, first as u64)) as i64;
                }
            } else {
                num_non_log += sigma_dis as i64;
                den_non_log += 1;
            }
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
        reference: reference_y.to_vec(),
        distorted: distorted_y.to_vec(),
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
