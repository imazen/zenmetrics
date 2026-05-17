//! Masked + IW (information-content-weighted) feature kernel.
//!
//! Computes the **extended** (`228..300`) and **IW** (`300..372`)
//! feature blocks on the GPU, matching CPU `zensim` semantics:
//!
//! - Extended (masked) features per (scale, channel) — 6 values:
//!   - `masked_ssim_mean`  (mean weighted by `mask`)
//!   - `masked_ssim_4th`   (L4 weighted by `mask`)
//!   - `masked_ssim_2nd`   (L2 weighted by `mask`)
//!   - `masked_art_4th`    (edge artifact L4 weighted by `mask`)
//!   - `masked_det_4th`    (edge detail-lost L4 weighted by `mask`)
//!   - `masked_mse`        (mean of `(src-dst)² · mask`)
//!
//! - IW features per (scale, channel) — 6 values:
//!   - `iw_ssim_mean / iw_ssim_4th / iw_ssim_2nd`
//!   - `iw_art_4th / iw_det_4th`
//!   - `iw_mse`
//!
//! ## Weight derivation (matches CPU `streaming::process_strip`)
//!
//! 1. `activity_raw[y,x] = |src[y,x] - mu1[y,x]|`
//! 2. `activity[y,x] = blur_1pass(activity_raw)` — box-blur radius `R=5`
//! 3. `mask[y,x]  = 1 / (1 + k_mask * activity[y,x])` with `k_mask = 4.0`
//! 4. `iw[y,x]    = 1 + k_iw   * activity[y,x]`     with `k_iw   = 4.0`
//!
//! The masked SSIM and edge math is:
//! - `d  = ((1 - num_m * num_s / denom_s) * mask).max(0)` for masked SSIM
//! - `art = max((1+|dv-mu2|)/(1+|sv-mu1|) - 1, 0) * <weight>`
//!   (CPU edge does `mask` multiply pre-split; same formula here, post-split)
//!   We follow the CPU's exact ordering: `d1 = (ratio - 1) * mask`, then
//!   `artifact = max(d1, 0)`, `detail_lost = max(-d1, 0)`.
//!
//! ## Block layout
//!
//! Grid: `(ceil(padded_w / TX), n_strips, 3)`. Same as `fused_features_kernel`.
//! `TX = 64`. Shared memory for the activity 2D blur:
//! - `src_row[TX + 2R]`, `dst_row[TX + 2R]`, `mu1_row[TX + 2R]`
//!   — reloaded for each row entering the V-blur window.
//! - `buf_activity[DIAM × TX]` — circular buffer holding the H-blurred
//!   `|src - mu1|` for the current sliding window (per column).
//!
//! ## Inputs
//!
//! The kernel reads the per-(scale, channel) `mu1`, `mu2`, `sigma_sq`,
//! `sigma12` planes emitted by the optional **persist-planes path** in
//! `fused_features_kernel_persist` (gated by the `WithIw` / `Extended`
//! regime). Without those planes the kernel can't produce the masked
//! SSIM math identically to CPU.

#![allow(clippy::doc_overindented_list_items)]

use cubecl::prelude::*;

// Mirrors `fused.rs` constants.
const TX: u32 = 64;
const R: u32 = 5;
const DIAM: u32 = 11;
const TILE_COLS: u32 = TX + 2u32 * R;
const TILE_COLS_US: usize = (TX + 2u32 * R) as usize;
const BUF_LEN_US: usize = (DIAM * TX) as usize;

// Same C2 used by fused.rs (no C1; SSIMULACRA2-style).
const C2: f32 = 0.0009;
const INV_DIAM: f32 = 1.0 / 11.0;

// k_mask = k_iw = 4.0 — matches `ZensimConfig::extended_masking_strength`
// and `ZensimConfig::iw_strength` defaults set in
// `metric.rs::config_from_params`.
const K_MASK: f32 = 4.0;
const K_IW: f32 = 4.0;

/// Slots per (channel, strip, col) for the masked + IW path.
///
/// Layout (12 f64 per slot):
/// - `0` — masked_ssim_d  (Σ d_m)
/// - `1` — masked_ssim_d4 (Σ d_m^4)
/// - `2` — masked_ssim_d2 (Σ d_m^2)
/// - `3` — masked_art_4   (Σ art_m^4)
/// - `4` — masked_det_4   (Σ det_m^4)
/// - `5` — masked_mse     (Σ (src-dst)² · mask)
/// - `6` — iw_ssim_d
/// - `7` — iw_ssim_d4
/// - `8` — iw_ssim_d2
/// - `9` — iw_art_4
/// - `10` — iw_det_4
/// - `11` — iw_mse
pub const SLOTS_PER_COL: usize = 12;

/// Masked + IW pooling kernel. Runs after `fused_features_kernel_persist`
/// has populated the `mu1 / mu2 / sigma_sq / sigma12` planes for the
/// current `(scale, channel)` block.
///
/// Grid is `(ceil(padded_w / TX), n_strips, 3)`; one block emits one
/// `(channel, strip, col)` row in the partials buffer of length
/// `SLOTS_PER_COL`.
///
/// `do_ext` / `do_iw` are scalar flags read uniformly across the block;
/// the kernel branches to skip work for the disabled path. Both flags
/// off is a valid no-op (cheap launch is fine; the host only launches
/// this kernel when at least one flag is on).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn masked_iw_kernel(
    src_a: &Array<f32>,
    dst_a: &Array<f32>,
    src_b: &Array<f32>,
    dst_b: &Array<f32>,
    src_c: &Array<f32>,
    dst_c: &Array<f32>,
    // Per-channel mu1/mu2/ssq/s12 planes emitted by the persist kernel.
    // Each is `padded_w × height × 3` channels, contiguous per channel.
    // We index `mu1_ch[ch * pad_total + offset]` to fetch per-pixel.
    mu1_all: &Array<f32>,
    mu2_all: &Array<f32>,
    ssq_all: &Array<f32>,
    s12_all: &Array<f32>,
    partials_ext_f64: &mut Array<f64>,
    width: u32,
    height: u32,
    n_strips: u32,
    pad_total: u32,
    slot_off_ext_f64: u32,
    do_ext: u32, // 0 or 1
    do_iw: u32,  // 0 or 1
) {
    let tx = UNIT_POS_X;
    let col_block = CUBE_POS_X;
    let strip = CUBE_POS_Y;
    let channel = CUBE_POS_Z;
    let col_base = col_block * TX;
    let col = col_base + tx;
    let in_bounds = col < width;

    let w = width as usize;
    let n_strips_us = n_strips as usize;
    let pw = width as usize;
    let pt = pad_total as usize;
    let ch_base = (channel as usize) * pt;
    let period_x = 2u32 * (width - 1u32);
    let period_y = 2u32 * (height - 1u32);

    // Strip range — mirror fused.rs's split.
    let strip_h_base = height / n_strips;
    let strip_rem = height - strip_h_base * n_strips;
    let y_start = strip * strip_h_base + u32::min(strip, strip_rem);
    let y_end_unclamp = y_start + strip_h_base + (if strip < strip_rem { 1u32 } else { 0u32 });
    let y_end = u32::min(y_end_unclamp, height);

    // Shared memory: rows for the activity 2D blur, plus the circular
    // H-blur buffer.
    let mut src_row = SharedMemory::<f32>::new(TILE_COLS_US);
    let mut dst_row = SharedMemory::<f32>::new(TILE_COLS_US);
    let mut mu1_row = SharedMemory::<f32>::new(TILE_COLS_US);
    let mut buf_act = SharedMemory::<f32>::new(BUF_LEN_US);

    let mut sum_act = 0.0_f32;

    // Per-thread accumulators (12 slots).
    let mut s0 = 0.0_f64; // masked_ssim_d
    let mut s1 = 0.0_f64; // masked_ssim_d4
    let mut s2 = 0.0_f64; // masked_ssim_d2
    let mut s3 = 0.0_f64; // masked_art_4
    let mut s4 = 0.0_f64; // masked_det_4
    let mut s5 = 0.0_f64; // masked_mse
    let mut s6 = 0.0_f64; // iw_ssim_d
    let mut s7 = 0.0_f64; // iw_ssim_d4
    let mut s8 = 0.0_f64; // iw_ssim_d2
    let mut s9 = 0.0_f64; // iw_art_4
    let mut s10 = 0.0_f64; // iw_det_4
    let mut s11 = 0.0_f64; // iw_mse

    // ============================ PREFIX INIT ============================
    // For each prefix row k in [0, DIAM), load src/dst/mu1 for one full
    // tile row at `mirror(y_start + k - R)`, compute per-column H-sum of
    // `|src - mu1|`, and accumulate into the sliding V-sum.
    let mut k: u32 = 0u32;
    while k < DIAM {
        let raw_y = (y_start + k + period_y - R) % period_y;
        let y_in = if raw_y < height {
            raw_y
        } else {
            period_y - raw_y
        };

        sync_cube();
        let mut i: u32 = 0u32;
        while i * TX + tx < TILE_COLS {
            let load_x = i * TX + tx;
            let raw_x = (col_base + load_x + period_x - R) % period_x;
            let gx = if raw_x < width {
                raw_x
            } else {
                period_x - raw_x
            };
            let off = (y_in as usize) * w + (gx as usize);
            let s_val = if channel == 0u32 {
                src_a[off]
            } else if channel == 1u32 {
                src_b[off]
            } else {
                src_c[off]
            };
            let d_val = if channel == 0u32 {
                dst_a[off]
            } else if channel == 1u32 {
                dst_b[off]
            } else {
                dst_c[off]
            };
            let mu1_val = mu1_all[ch_base + off];
            src_row[load_x as usize] = s_val;
            dst_row[load_x as usize] = d_val;
            mu1_row[load_x as usize] = mu1_val;
            i += 1u32;
        }
        sync_cube();

        // Per-column H-sum of |src - mu1| over the DIAM-wide kernel.
        let mut h_sum = 0.0_f32;
        let mut j: u32 = 0u32;
        while j < DIAM {
            let s = src_row[(tx + j) as usize];
            let m = mu1_row[(tx + j) as usize];
            h_sum += f32::abs(s - m);
            j += 1u32;
        }
        let h_avg = h_sum * INV_DIAM;
        let buf_idx = (k * TX + tx) as usize;
        buf_act[buf_idx] = h_avg;
        sum_act += h_avg;

        k += 1u32;
    }

    // ============================ WALK Y ============================
    let mut slot: u32 = 0u32;
    let mut y: u32 = y_start;
    while y < y_end {
        // V-blur output: activity at (y, col).
        let activity = sum_act * INV_DIAM;
        let mask_w = 1.0 / (1.0 + K_MASK * activity);
        let iw_w = 1.0 + K_IW * activity;

        // Per-pixel reads (one element each).
        let off = (y as usize) * w + (col as usize);
        let mut sv: f32 = 0.0;
        let mut dv: f32 = 0.0;
        let mut mu1: f32 = 0.0;
        let mut mu2: f32 = 0.0;
        let mut ssq: f32 = 0.0;
        let mut s12_v: f32 = 0.0;
        if in_bounds {
            if channel == 0u32 {
                sv = src_a[off];
                dv = dst_a[off];
            } else {
                if channel == 1u32 {
                    sv = src_b[off];
                    dv = dst_b[off];
                } else {
                    sv = src_c[off];
                    dv = dst_c[off];
                }
            }
            mu1 = mu1_all[ch_base + off];
            mu2 = mu2_all[ch_base + off];
            ssq = ssq_all[ch_base + off];
            s12_v = s12_all[ch_base + off];
        }

        // SSIMULACRA2-style SSIM (no C1). Same math as fused.rs but
        // applied with mask BEFORE max-clamp (matches CPU
        // `ssim_channel_masked_inner`).
        let mu_diff = mu1 - mu2;
        let num_m = fma(mu_diff, -mu_diff, 1.0);
        let inner_ns = fma(-mu1, mu2, s12_v);
        let num_s = fma(2.0, inner_ns, C2);
        let inner_ds_inner = fma(-mu1, mu1, ssq);
        let denom_s = fma(-mu2, mu2, inner_ds_inner) + C2;
        let raw_sd = 1.0 - (num_m * num_s) / denom_s;

        // Per-CPU `ssim_channel_masked_inner`: `((1 - q) * mask).max(0)`.
        if do_ext == 1u32 {
            let d_m = (raw_sd * mask_w).max(f32::new(0.0));
            let d_m_sq = d_m * d_m;
            let d_m_4 = d_m_sq * d_m_sq;
            s0 += d_m as f64;
            s1 += d_m_4 as f64;
            s2 += d_m_sq as f64;
        }
        if do_iw == 1u32 {
            let d_i = (raw_sd * iw_w).max(f32::new(0.0));
            let d_i_sq = d_i * d_i;
            let d_i_4 = d_i_sq * d_i_sq;
            s6 += d_i as f64;
            s7 += d_i_4 as f64;
            s8 += d_i_sq as f64;
        }

        // Masked edge — matches `edge_diff_masked_inner`:
        //   diff1 = |sv - mu1|; diff2 = |dv - mu2|
        //   d1 = ((1 + diff2) / (1 + diff1) - 1) * mask
        //   artifact    = max(d1, 0); detail_lost = max(-d1, 0)
        let diff1 = f32::abs(sv - mu1);
        let diff2 = f32::abs(dv - mu2);
        let ed_raw = (1.0 + diff2) / (1.0 + diff1) - 1.0;

        if do_ext == 1u32 {
            let ed_m = ed_raw * mask_w;
            let art_m = ed_m.max(f32::new(0.0));
            let det_m = (-ed_m).max(f32::new(0.0));
            let am2 = art_m * art_m;
            let dm2 = det_m * det_m;
            s3 += (am2 * am2) as f64;
            s4 += (dm2 * dm2) as f64;

            let p = sv - dv;
            s5 += (p * p * mask_w) as f64;
        }
        if do_iw == 1u32 {
            let ed_i = ed_raw * iw_w;
            let art_i = ed_i.max(f32::new(0.0));
            let det_i = (-ed_i).max(f32::new(0.0));
            let ai2 = art_i * art_i;
            let di2 = det_i * det_i;
            s9 += (ai2 * ai2) as f64;
            s10 += (di2 * di2) as f64;

            let p = sv - dv;
            s11 += (p * p * iw_w) as f64;
        }

        // Slide the activity V-window down by one row.
        let buf_idx = (slot * TX + tx) as usize;
        let old_act = buf_act[buf_idx];

        let raw_y2 = (y + R + 1u32 + period_y) % period_y;
        let y_in2 = if raw_y2 < height {
            raw_y2
        } else {
            period_y - raw_y2
        };

        sync_cube();
        let mut i2: u32 = 0u32;
        while i2 * TX + tx < TILE_COLS {
            let load_x = i2 * TX + tx;
            let raw_x = (col_base + load_x + period_x - R) % period_x;
            let gx = if raw_x < width {
                raw_x
            } else {
                period_x - raw_x
            };
            let off2 = (y_in2 as usize) * w + (gx as usize);
            let s_val = if channel == 0u32 {
                src_a[off2]
            } else if channel == 1u32 {
                src_b[off2]
            } else {
                src_c[off2]
            };
            let m_val = mu1_all[ch_base + off2];
            // Only need src and mu1 for the activity blur.
            src_row[load_x as usize] = s_val;
            mu1_row[load_x as usize] = m_val;
            // dst_row is unused beyond prefix init; reuse it as scratch
            // to keep the load-loop balanced.
            dst_row[load_x as usize] = 0.0_f32;
            i2 += 1u32;
        }
        sync_cube();

        let mut nh = 0.0_f32;
        let mut j2: u32 = 0u32;
        while j2 < DIAM {
            let s = src_row[(tx + j2) as usize];
            let m = mu1_row[(tx + j2) as usize];
            nh += f32::abs(s - m);
            j2 += 1u32;
        }
        let new_h_avg = nh * INV_DIAM;
        sum_act = sum_act + new_h_avg - old_act;
        buf_act[buf_idx] = new_h_avg;

        slot = (slot + 1u32) % DIAM;
        y += 1u32;
    }

    // ============================ WRITE PARTIALS ============================
    if !in_bounds {
        terminate!();
    }
    let slot_idx_us =
        (channel as usize) * n_strips_us * pw + (strip as usize) * pw + (col as usize);
    let f64_base = (slot_off_ext_f64 as usize) + slot_idx_us * SLOTS_PER_COL;
    partials_ext_f64[f64_base] = s0;
    partials_ext_f64[f64_base + 1] = s1;
    partials_ext_f64[f64_base + 2] = s2;
    partials_ext_f64[f64_base + 3] = s3;
    partials_ext_f64[f64_base + 4] = s4;
    partials_ext_f64[f64_base + 5] = s5;
    partials_ext_f64[f64_base + 6] = s6;
    partials_ext_f64[f64_base + 7] = s7;
    partials_ext_f64[f64_base + 8] = s8;
    partials_ext_f64[f64_base + 9] = s9;
    partials_ext_f64[f64_base + 10] = s10;
    partials_ext_f64[f64_base + 11] = s11;
}
