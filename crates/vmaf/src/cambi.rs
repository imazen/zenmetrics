use crate::Error;

const NUM_SCALES: usize = 5;
const SCALE_WEIGHTS: [i32; NUM_SCALES] = [16, 8, 4, 2, 1];
const CONTRAST_WEIGHTS: [i32; 32] = [
    1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
];
const MASK_FILTER_SIZE: usize = 7;
const WINDOW_SIZE: usize = 65;
const TOPK: f64 = 0.6;
const TVI_THRESHOLD: f64 = 0.019;
const VIS_LUM_THRESHOLD: f64 = 0.06;
const MAX_LOG_CONTRAST: u32 = 2;
const CAMBI_MAX_VAL: f64 = 17.0;
const CAMBI_MIN_WIDTH_HEIGHT: usize = 216;
const SPEEDUP_THRESHOLD_1080P: usize = 1920 * 1080;

const BT1886_GAMMA: f64 = 2.4;
const BT1886_LW: f64 = 300.0;
const BT1886_LB: f64 = 0.01;

const LUMA_FOOT_10B: i32 = 64;
const LUMA_HEAD_10B: i32 = 940;

fn bt1886_eotf(v: f64) -> f64 {
    let a = (BT1886_LW.powf(1.0 / BT1886_GAMMA) - BT1886_LB.powf(1.0 / BT1886_GAMMA))
        .powf(BT1886_GAMMA);
    let b = BT1886_LB.powf(1.0 / BT1886_GAMMA)
        / (BT1886_LW.powf(1.0 / BT1886_GAMMA) - BT1886_LB.powf(1.0 / BT1886_GAMMA));
    a * (v + b).max(0.0).powf(BT1886_GAMMA)
}

fn luminance(sample: i32) -> f64 {
    let clipped = sample.clamp(LUMA_FOOT_10B, LUMA_HEAD_10B);
    let normalized = (clipped - LUMA_FOOT_10B) as f64 / (LUMA_HEAD_10B - LUMA_FOOT_10B) as f64;
    bt1886_eotf(normalized)
}

fn tvi_condition(sample: i32, diff: i32, tvi_threshold: f64) -> bool {
    let mean_luminance = luminance(sample);
    let diff_luminance = luminance(sample + diff);
    diff_luminance - mean_luminance > tvi_threshold * mean_luminance
}

const BISECT_TOO_SMALL: i32 = 0;
const BISECT_CORRECT: i32 = 1;
const BISECT_TOO_BIG: i32 = 2;

fn tvi_hard_threshold_condition(sample: i32, diff: i32, tvi_threshold: f64) -> i32 {
    if !tvi_condition(sample, diff, tvi_threshold) {
        return BISECT_TOO_BIG;
    }
    if tvi_condition(sample + 1, diff, tvi_threshold) {
        return BISECT_TOO_SMALL;
    }
    BISECT_CORRECT
}

fn get_tvi_for_diff(diff: i32, tvi_threshold: f64) -> i32 {
    const MAX_VAL: i32 = (1 << 10) - 1;
    let mut foot = LUMA_FOOT_10B;
    let mut head = LUMA_HEAD_10B - diff - 1;

    let b = tvi_hard_threshold_condition(foot, diff, tvi_threshold);
    if b == BISECT_TOO_BIG {
        return 0;
    }
    if b == BISECT_CORRECT {
        return foot;
    }
    let b = tvi_hard_threshold_condition(head, diff, tvi_threshold);
    if b == BISECT_TOO_SMALL {
        return MAX_VAL;
    }
    if b == BISECT_CORRECT {
        return head;
    }
    loop {
        let mid = foot + (head - foot) / 2;
        match tvi_hard_threshold_condition(mid, diff, tvi_threshold) {
            BISECT_TOO_BIG => head = mid,
            BISECT_TOO_SMALL => foot = mid,
            _ => return mid,
        }
    }
}

fn get_vlt_luma(visibility_luminance_threshold: f64) -> u16 {
    let mut sample: i32 = LUMA_FOOT_10B;
    while luminance(sample) < visibility_luminance_threshold {
        sample += 1;
    }
    if sample == LUMA_FOOT_10B {
        0
    } else {
        sample as u16
    }
}

fn adjust_window_size(window_size: usize, w: usize, h: usize, speedup: bool) -> usize {
    let mut ws = ((window_size * (w + h)) / 375) >> 4;
    if speedup {
        ws = (ws + 1) >> 1;
    }
    ws | 1
}

fn anti_dithering_filter(data: &mut [u16], width: usize, height: usize) {
    for i in 0..height - 1 {
        for j in 0..width - 1 {
            data[i * width + j] = (data[i * width + j]
                + data[i * width + j + 1]
                + data[(i + 1) * width + j]
                + data[(i + 1) * width + j + 1])
                >> 2;
        }
        let j = width - 1;
        data[i * width + j] = (data[i * width + j] + data[(i + 1) * width + j]) >> 1;
    }
    let i = height - 1;
    for j in 0..width - 1 {
        data[i * width + j] = (data[i * width + j] + data[i * width + j + 1]) >> 1;
    }
}

fn decimate(data: &mut [u16], stride: usize, width: usize, height: usize) {
    for i in 0..height {
        for j in 0..width {
            data[i * stride + j] = data[(i * 2) * stride + j * 2];
        }
    }
}

fn min3(a: u16, b: u16, c: u16) -> u16 {
    if a <= b && a <= c {
        return a;
    }
    if b <= c {
        return b;
    }
    c
}

fn mode3(a: u16, b: u16, c: u16) -> u16 {
    if a == b || a == c {
        return a;
    }
    if b == c {
        return b;
    }
    min3(a, b, c)
}

fn filter_mode(data: &mut [u16], stride: usize, width: usize, height: usize, buffer: &mut [u16]) {
    let mut curr_line = 0usize;
    for i in 0..height {
        buffer[curr_line * width] = data[i * stride];
        for j in 1..width - 1 {
            buffer[curr_line * width + j] = mode3(
                data[i * stride + j - 1],
                data[i * stride + j],
                data[i * stride + j + 1],
            );
        }
        buffer[curr_line * width + width - 1] = data[i * stride + width - 1];

        if i > 1 {
            for j in 0..width {
                data[(i - 1) * stride + j] =
                    mode3(buffer[j], buffer[width + j], buffer[2 * width + j]);
            }
        }
        curr_line = if curr_line + 1 == 3 { 0 } else { curr_line + 1 };
    }
}

fn ceil_log2(num: u32) -> i32 {
    if num == 0 {
        return 0;
    }
    let mut tmp = num - 1;
    let mut shift = 0;
    while tmp > 0 {
        tmp >>= 1;
        shift += 1;
    }
    shift
}

fn get_mask_index(input_width: usize, input_height: usize, filter_size: usize) -> u32 {
    let shifted_wh = (input_width >> 6) as u32 * (input_height >> 6) as u32;
    (((filter_size * filter_size) as i32 + 3 * (ceil_log2(shifted_wh) - 11) - 1) >> 1) as u32
}

fn get_spatial_mask(
    image: &[u16],
    mask: &mut [u16],
    width: usize,
    height: usize,
) -> Result<(), Error> {
    let pad = MASK_FILTER_SIZE / 2;
    let mask_index = get_mask_index(width, height, MASK_FILTER_SIZE);

    let sat_w = width
        .checked_add(1)
        .ok_or(Error::InvalidInput("dimension overflow"))?;
    let sat_h = height
        .checked_add(1)
        .ok_or(Error::InvalidInput("dimension overflow"))?;
    let mut sat = vec![
        0u32;
        sat_w
            .checked_mul(sat_h)
            .ok_or(Error::InvalidInput("dimension overflow"))?
    ];
    for i in 0..height {
        let mut row_sum = 0u32;
        for j in 0..width {
            let horizontal = j == width - 1 || image[i * width + j] == image[i * width + j + 1];
            let vertical = i == height - 1 || image[i * width + j] == image[(i + 1) * width + j];
            row_sum += (horizontal && vertical) as u32;
            sat[(i + 1) * sat_w + (j + 1)] = sat[i * sat_w + (j + 1)] + row_sum;
        }
    }

    for i in 0..height {
        let r_lo = i.saturating_sub(pad);
        let r_hi = (i + pad).min(height - 1);
        for j in 0..width {
            let c_lo = j.saturating_sub(pad);
            let c_hi = (j + pad).min(width - 1);
            let sum = sat[(r_hi + 1) * sat_w + (c_hi + 1)] + sat[r_lo * sat_w + c_lo]
                - sat[(r_hi + 1) * sat_w + c_lo]
                - sat[r_lo * sat_w + (c_hi + 1)];
            mask[i * width + j] = (sum > mask_index) as u16;
        }
    }
    Ok(())
}

fn c_value_pixel(
    histograms: &[u16],
    hist_width: usize,
    value: u16,
    diff_weights: &[i32],
    num_diffs: usize,
    tvi_thresholds: &[u16],
    vlt_luma: u16,
    v_band_offset_val: u16,
    v_band_size: u16,
    col: usize,
) -> f32 {
    let compact_v_signed = value as i32 - v_band_offset_val as i32;
    if compact_v_signed < 0 || compact_v_signed >= v_band_size as i32 {
        return 0.0;
    }
    let compact_v = compact_v_signed as usize;
    let p_0 = histograms[compact_v * hist_width + col] as i64;
    let mut c_value: f32 = 0.0;
    for d in 0..num_diffs {
        let dp = (d + 1) as i32;
        if value <= tvi_thresholds[d] && (value as i32 + dp) > vlt_luma as i32 {
            let idx1 = compact_v_signed + dp;
            let idx2 = compact_v_signed - dp;
            let p_1 = histograms[idx1 as usize * hist_width + col] as i64;
            let p_2 = if idx2 >= 0 {
                histograms[idx2 as usize * hist_width + col] as i64
            } else {
                0
            };
            let (num, den) = if p_1 > p_2 {
                (diff_weights[d] as i64 * p_0 * p_1, p_1 + p_0)
            } else {
                (diff_weights[d] as i64 * p_0 * p_2, p_2 + p_0)
            };
            let val = if den == 0 {
                0.0
            } else {
                num as f32 / den as f32
            };
            if val > c_value {
                c_value = val;
            }
        }
    }
    c_value
}

fn calculate_c_values(
    image: &[u16],
    mask: &[u16],
    c_values: &mut [f32],
    stride: usize,
    width: usize,
    height: usize,
    window_size: usize,
    num_diffs: usize,
    tvi_for_diff: &[u16],
    vlt_luma: u16,
    v_band_base: u16,
    v_band_size: u16,
    diff_weights: &[i32],
) -> Result<(), Error> {
    let pad = window_size / 2;
    let v_band_offset_val = v_band_base + num_diffs as u16;
    let mut histograms_acc = vec![
        0u16;
        (v_band_size as usize)
            .checked_mul(width)
            .ok_or(Error::InvalidInput("dimension overflow"))?
    ];
    c_values[..width * height].fill(0.0);

    let update_row = |histograms: &mut [u16], r: usize, delta: i16| {
        for j in 0..width {
            if mask[r * stride + j] == 0 {
                continue;
            }
            let v = image[r * stride + j];
            let rel = v.wrapping_sub(v_band_base);
            if rel >= v_band_size {
                continue;
            }
            let row = &mut histograms[rel as usize * width..(rel as usize + 1) * width];
            let c_lo = j.saturating_sub(pad);
            let c_hi = (j + pad).min(width - 1);
            for v in row[c_lo..=c_hi].iter_mut() {
                *v = (*v as i16 + delta) as u16;
            }
        }
    };

    for r in 0..=pad.min(height - 1) {
        update_row(&mut histograms_acc, r, 1);
    }
    for i in 0..height {
        if i > 0 {
            if i + pad < height {
                update_row(&mut histograms_acc, i + pad, 1);
            }
            if i > pad {
                update_row(&mut histograms_acc, i - pad - 1, -1);
            }
        }
        let histograms = &histograms_acc[..];
        for col in 0..width {
            if mask[i * stride + col] != 0 {
                c_values[i * width + col] = c_value_pixel(
                    histograms,
                    width,
                    image[i * stride + col] + num_diffs as u16,
                    diff_weights,
                    num_diffs,
                    tvi_for_diff,
                    vlt_luma,
                    v_band_offset_val,
                    v_band_size,
                    col,
                );
            }
        }
    }
    Ok(())
}

fn spatial_pooling(c_values: &mut [f32], topk: f64, width: usize, height: usize) -> f64 {
    let num_elements = width * height;
    let topk_num = ((topk * num_elements as f64) as usize).clamp(1, num_elements);
    if topk_num < num_elements {
        c_values[..num_elements].select_nth_unstable_by(topk_num - 1, |a, b| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    let sum: f64 = c_values[..topk_num].iter().map(|&v| v as f64).sum();
    sum / topk_num as f64
}

pub fn cambi_v1_from_luma(
    dist_y: &[u16],
    width: usize,
    height: usize,
    bit_depth: u8,
) -> Result<f64, Error> {
    if !(8..=16).contains(&bit_depth) {
        return Err(Error::InvalidInput("unsupported bit depth"));
    }
    if width == 0 || height == 0 {
        return Err(Error::InvalidInput("zero dimension"));
    }
    let npix = width
        .checked_mul(height)
        .filter(|&n| n <= usize::MAX / 4)
        .ok_or(Error::InvalidInput("dimension overflow"))?;
    if width < CAMBI_MIN_WIDTH_HEIGHT && height < CAMBI_MIN_WIDTH_HEIGHT {
        return Err(Error::InvalidInput("both dimensions below CAMBI minimum"));
    }
    if dist_y.len() != npix {
        return Err(Error::InvalidInput("frame length mismatch"));
    }
    let max_sample = (1u32 << bit_depth) - 1;
    if dist_y.iter().any(|&v| v as u32 > max_sample) {
        return Err(Error::InvalidInput("sample exceeds bit depth"));
    }

    let speedup = width * height >= SPEEDUP_THRESHOLD_1080P;
    let window_size = adjust_window_size(WINDOW_SIZE, width, height, speedup);
    if window_size * window_size >= 4226 {
        return Err(Error::InvalidInput("window size too large"));
    }

    let mut image = vec![0u16; npix];
    if bit_depth < 10 {
        let shift = 10 - bit_depth;
        for (o, &v) in image.iter_mut().zip(dist_y.iter()) {
            *o = v << shift;
        }
    } else if bit_depth == 10 {
        image.copy_from_slice(dist_y);
    } else {
        let shift = bit_depth - 10;
        let rounding = 1u32 << (shift - 1);
        for (o, &v) in image.iter_mut().zip(dist_y.iter()) {
            *o = ((v as u32 + rounding) >> shift) as u16;
        }
    }
    if bit_depth < 10 {
        anti_dithering_filter(&mut image, width, height);
    }

    let num_diffs: usize = 1 << MAX_LOG_CONTRAST;
    let mut tvi_for_diff = [0u16; 32];
    for (d, slot) in tvi_for_diff.iter_mut().enumerate().take(num_diffs) {
        *slot = (get_tvi_for_diff((d + 1) as i32, TVI_THRESHOLD) + num_diffs as i32) as u16;
    }
    let vlt_luma = get_vlt_luma(VIS_LUM_THRESHOLD);

    let v_lo_signed = vlt_luma as i32 - 3 * num_diffs as i32 + 1;
    let v_band_base: u16 = if v_lo_signed > 0 {
        v_lo_signed as u16
    } else {
        0
    };
    let v_band_size: u16 = tvi_for_diff[num_diffs - 1] + 1 - v_band_base;

    let mut mask = vec![0u16; npix];
    get_spatial_mask(&image, &mut mask, width, height)?;

    let mut c_values = vec![0f32; npix];
    let mut filter_buffer = vec![
        0u16;
        width
            .checked_mul(3)
            .ok_or(Error::InvalidInput("dimension overflow"))?
    ];
    let mut scores_per_scale = [0.0f64; NUM_SCALES];

    let mut scaled_width = width;
    let mut scaled_height = height;
    for (scale, slot) in scores_per_scale.iter_mut().enumerate() {
        if scale > 0 || speedup {
            scaled_width = (scaled_width + 1) >> 1;
            scaled_height = (scaled_height + 1) >> 1;
            decimate(&mut image, width, scaled_width, scaled_height);
            decimate(&mut mask, width, scaled_width, scaled_height);
        }
        filter_mode(
            &mut image,
            width,
            scaled_width,
            scaled_height,
            &mut filter_buffer,
        );
        calculate_c_values(
            &image,
            &mask,
            &mut c_values,
            width,
            scaled_width,
            scaled_height,
            window_size,
            num_diffs,
            &tvi_for_diff,
            vlt_luma,
            v_band_base,
            v_band_size,
            &CONTRAST_WEIGHTS,
        )?;
        *slot = spatial_pooling(&mut c_values, TOPK, scaled_width, scaled_height);
    }

    let odd = 2 * (window_size >> 1) + 1;
    let pixels_in_window = (odd * odd) as f64;
    let mut score = 0.0;
    for (scale, &s) in scores_per_scale.iter().enumerate() {
        score += s * SCALE_WEIGHTS[scale] as f64;
    }
    score /= pixels_in_window;

    Ok(score.min(CAMBI_MAX_VAL))
}
