// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! 3-channel fused Gaussian blur.
//!
//! Same separable Gaussian as `blur::{horizontal,vertical}_blur_kernel`
//! but processes X / Y / B channels in one launch each direction.
//! Per pixel, the kernel weights are identical across channels (same
//! sigma, same radius), so we compute weights once and accumulate 3
//! sums in parallel. Saves 2 launches per blur call.
//!
//! Used by butteraugli's apply_opsin and separate_frequencies inner
//! loops where the same blur runs on all 3 channels back to back.

#![allow(clippy::assign_op_pattern)]

use cubecl::prelude::*;

const M: f32 = 2.25;
const LOG2_E: f32 = std::f32::consts::LOG2_E;

#[cube]
fn exp_f32(x: f32) -> f32 {
    f32::powf(2.0, x * LOG2_E)
}

#[cube]
fn gauss(d: f32, s: f32) -> f32 {
    let inv = 1.0 / s;
    let z = d * inv;
    exp_f32(-0.5 * z * z)
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn horizontal_blur_3ch_kernel(
    src_x: &Array<f32>,
    src_y: &Array<f32>,
    src_b: &Array<f32>,
    dst_x: &mut Array<f32>,
    dst_y: &mut Array<f32>,
    dst_b: &mut Array<f32>,
    width: u32,
    height: u32,
    sigma: f32,
) {
    let idx = ABSOLUTE_POS;
    let total = (width * height) as usize;
    if idx >= total {
        terminate!();
    }
    let w = width as usize;
    let row = idx / w;
    let x = idx % w;

    let raw = u32::cast_from(M * sigma);
    let radius_us = u32::max(raw, 1u32) as usize;
    let begin = usize::saturating_sub(x, radius_us);
    let end = u32::min((x + radius_us) as u32, (w - 1) as u32) as usize;

    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    let mut sum_b = 0.0f32;
    let mut wsum = 0.0f32;
    let row_off = row * w;

    let mut i = begin;
    while i <= end {
        let i32_ = i as u32;
        let x32 = x as u32;
        let dist = (u32::saturating_sub(i32_, x32) + u32::saturating_sub(x32, i32_)) as f32;
        let weight = gauss(dist, sigma);
        sum_x += src_x[row_off + i] * weight;
        sum_y += src_y[row_off + i] * weight;
        sum_b += src_b[row_off + i] * weight;
        wsum += weight;
        i += 1;
    }
    // sum/wsum (NOT sum*(1/wsum)) — bit-exact with the per-channel
    // kernel. FMA differences here would shift rounding-tied
    // butteraugli scores and flip the smart-gate's Tie/Refine path
    // selection on edge cases.
    dst_x[idx] = sum_x / wsum;
    dst_y[idx] = sum_y / wsum;
    dst_b[idx] = sum_b / wsum;
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn vertical_blur_3ch_kernel(
    src_x: &Array<f32>,
    src_y: &Array<f32>,
    src_b: &Array<f32>,
    dst_x: &mut Array<f32>,
    dst_y: &mut Array<f32>,
    dst_b: &mut Array<f32>,
    width: u32,
    height: u32,
    sigma: f32,
) {
    let idx = ABSOLUTE_POS;
    let total = (width * height) as usize;
    if idx >= total {
        terminate!();
    }
    let w = width as usize;
    let h = height as usize;
    let y = idx / w;
    let x = idx - y * w;

    let raw = u32::cast_from(M * sigma);
    let radius_us = u32::max(raw, 1u32) as usize;
    let begin = usize::saturating_sub(y, radius_us);
    let end = u32::min((y + radius_us) as u32, (h - 1) as u32) as usize;

    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    let mut sum_b = 0.0f32;
    let mut wsum = 0.0f32;
    let mut i = begin;
    while i <= end {
        let i32_ = i as u32;
        let y32 = y as u32;
        let dist = (u32::saturating_sub(i32_, y32) + u32::saturating_sub(y32, i32_)) as f32;
        let weight = gauss(dist, sigma);
        let off = i * w + x;
        sum_x += src_x[off] * weight;
        sum_y += src_y[off] * weight;
        sum_b += src_b[off] * weight;
        wsum += weight;
        i += 1;
    }
    dst_x[idx] = sum_x / wsum;
    dst_y[idx] = sum_y / wsum;
    dst_b[idx] = sum_b / wsum;
}
