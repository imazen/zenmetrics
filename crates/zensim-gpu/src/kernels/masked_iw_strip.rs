//! Strip-local masked + IW kernel (matches CPU's
//! [`zensim::streaming::process_strip_channel`] semantics).
//!
//! ## Principled per-channel H-blur activity (2026-05-17)
//!
//! The CPU's masked + IW activity computation was redesigned on
//! `feat/principled-activity` to use a strip-local per-channel
//! `H_blur(src)` as the "local mean" reference for the activity map:
//!
//! ```text
//! activity[c] = box_blur(|src[c] - H_blur(src[c])|)
//! ```
//!
//! at ALL strip rows (inner + overlap). This decouples channels
//! entirely; no cross-channel buffer reuse. See `docs/PRINCIPLED_ACTIVITY.md`
//! in the zensim repo for the rationale.
//!
//! The GPU kernel mirrors this: per (strip, channel), `mu1_row` is the
//! H-blur of the CURRENT channel's source at the current strip row.
//! The host-side carry-plane simulator and cross-channel cascade branches
//! that were here before are deleted.
//!
//! ## Grid layout
//!
//! `(ceil(padded_w / TX), num_cpu_strips, 3)` where
//! `num_cpu_strips = ceil(height / STRIP_INNER)`. This DIFFERS from
//! the basic-kernel strip count (`pick_n_strips`) which is tuned for
//! GPU occupancy; the strip count here is fixed to CPU semantics.
//!
//! ## Shared memory
//!
//! - `wide_src[TILE_COLS_WIDE]` — DIAM-wider per-row source window so
//!   each thread can compute H-blurred source at its column.
//! - `mu1_row[TILE_COLS]` — H-blurred source per column (per-channel,
//!   per-row). Replaces the prior cross-channel cascade.
//! - `buf_activity[DIAM × TX]` — sliding H-blurred-activity circular
//!   buffer for the strip-local V-blur of the activity map.
//!
//! Total ~4 KB per cube; slightly larger than the prior 3.4 KB but
//! the carry plane (≤23 MB at 12 MP per scale) is gone, and the kernel
//! has no per-channel branches in its hot path.
//!
//! ## Inner-row math
//!
//! Per-pixel mu1/mu2/ssq/s12 reads at the inner pixel come from the
//! persist planes (image-wide V-blur-of-H-blur values). These are
//! used by the masked-SSIM and masked-edge math, NOT by the activity
//! computation. Activity uses the per-row H-blur exclusively.

#![allow(clippy::doc_overindented_list_items)]

use cubecl::prelude::*;

// Mirrors `fused.rs` constants.
const TX: u32 = 64;
/// Blur radius (DIAM = 2R + 1 = 11). Public so `pipeline.rs` can size
/// allocations.
pub const R: u32 = 5;
const DIAM: u32 = 11;
const TILE_COLS: u32 = TX + 2u32 * R;
const TILE_COLS_US: usize = (TX + 2u32 * R) as usize;
const TILE_COLS_WIDE: u32 = TX + 4u32 * R;
const TILE_COLS_WIDE_US: usize = (TX + 4u32 * R) as usize;
const BUF_LEN_US: usize = (DIAM * TX) as usize;

// Same C2 used by fused.rs (no C1; SSIMULACRA2-style).
const C2: f32 = 0.0009;
const INV_DIAM: f32 = 1.0 / 11.0;

// k_mask = k_iw = 4.0 — matches `ZensimConfig::extended_masking_strength`
// and `ZensimConfig::iw_strength` defaults set in
// `metric.rs::config_from_params`.
const K_MASK: f32 = 4.0;
const K_IW: f32 = 4.0;

/// Inner rows per CPU strip (matches `zensim::streaming::STRIP_INNER`).
/// **Do not change without re-deriving the CPU/GPU strip-overlap
/// parity story** — this constant ties the GPU launch grid to CPU's
/// per-strip mu1-overlap behavior.
pub const STRIP_INNER: u32 = 32;

/// Slots per (channel, strip, col) for the masked + IW path. Same as
/// [`super::masked_iw::SLOTS_PER_COL`].
pub const SLOTS_PER_COL: usize = 12;

/// Compute the number of CPU-shaped strips for a given image height.
/// Mirrors `streaming::process_scale_bands`'s `total_strips`.
#[inline]
pub const fn cpu_strip_count(height: u32) -> u32 {
    (height + STRIP_INNER - 1) / STRIP_INNER
}

/// Strip-local masked + IW pooling kernel with principled per-channel
/// H-blur activity reference. Replaces the pre-2026-05-17 cross-channel
/// cascade variant.
///
/// Grid is `(ceil(padded_w / TX), cpu_strip_count(height), 3)`; one
/// block emits one `(channel, strip, col)` row in the partials buffer
/// of length `SLOTS_PER_COL`.
///
/// `do_ext` / `do_iw` are scalar flags read uniformly across the
/// block; the kernel branches to skip work for the disabled path.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn masked_iw_strip_kernel(
    src_a: &Array<f32>,
    dst_a: &Array<f32>,
    src_b: &Array<f32>,
    dst_b: &Array<f32>,
    src_c: &Array<f32>,
    dst_c: &Array<f32>,
    // Per-channel mu1/mu2/ssq/s12 planes emitted by the persist kernel.
    // Each is `padded_w × height × 3` channels, contiguous per channel.
    // Used by masked-SSIM / masked-edge math at inner rows. NOT used by
    // the activity computation (which uses on-the-fly H_blur(src) below).
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
    // Body-row range gate (relative to this kernel's buffer y-coord
    // system, i.e. `[0, height)` where height = strip_alloc_h in
    // strip mode or image_h in full mode). Halo rows still drive the
    // strip-local mu1/activity computation; their per-pixel feature
    // contributions are not accumulated. Full-image callers pass
    // `(0, height)`.
    y_body_start: u32,
    y_body_end: u32,
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

    // ============================ STRIP LAYOUT ============================
    // CPU: STRIP_INNER=32 + overlap=R=5. Strip k spans inner rows
    // [k*STRIP_INNER, min((k+1)*STRIP_INNER, height)). Strip buffer
    // spans [strip_top, strip_bot) where:
    //   strip_top  = max(0, inner_start - R)             (saturating_sub)
    //   strip_bot  = min(height, inner_end   + R)
    //   strip_h    = strip_bot - strip_top
    //   inner_off  = inner_start - strip_top              (0 for top strip, R for middle/bot)
    //   inner_h    = inner_end   - inner_start            (≤ STRIP_INNER)
    //
    // Mirror for the activity V-blur is "strip-local" — relative to
    // [0, strip_h) — not image-wide.
    let inner_start = strip * STRIP_INNER;
    let inner_end_unc = inner_start + STRIP_INNER;
    let inner_end = u32::min(inner_end_unc, height);
    // saturating_sub(R) for u32 — cubecl rejects mixing a `u32` literal
    // with an expanded `u32` value in if-else branches, so we use the
    // `inner_start + R - R` mirror trick instead.
    let strip_top = if inner_start >= R {
        inner_start - R
    } else {
        inner_start - inner_start
    };
    let strip_bot = u32::min(inner_end + R, height);
    let strip_h = strip_bot - strip_top;
    let inner_off = inner_start - strip_top;
    let inner_h = inner_end - inner_start;
    let period_sy = 2u32 * (strip_h - 1u32);

    // Shared memory.
    let mut wide_src = SharedMemory::<f32>::new(TILE_COLS_WIDE_US);
    let mut mu1_row = SharedMemory::<f32>::new(TILE_COLS_US);
    let mut buf_act = SharedMemory::<f32>::new(BUF_LEN_US);

    let mut sum_act = 0.0_f32;

    // Per-thread accumulators (12 slots, mirrors masked_iw::SLOTS_PER_COL).
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
    // Compute the H-blurred activity for the DIAM rows centred on
    // strip-row 0, using strip-local mirror. For each prefix row k,
    // the corresponding strip-local row is mirror(k - R) using
    // period_sy = 2*(strip_h - 1).
    let mut k: u32 = 0u32;
    while k < DIAM {
        // strip-local mirror: raw_sy = (k + period_sy - R) % period_sy
        // (positive); then fold via period_sy - raw_sy when raw_sy >= strip_h.
        let raw_sy = (k + period_sy - R) % period_sy;
        let sy_eff = if raw_sy < strip_h {
            raw_sy
        } else {
            period_sy - raw_sy
        };
        let gy = strip_top + sy_eff;

        sync_cube();
        // Cooperative load of WIDE src into TILE_COLS_WIDE (covers
        // col_base - 2R .. col_base + TX + 2R - 1 with horizontal mirror).
        let mut i: u32 = 0u32;
        while i * TX + tx < TILE_COLS_WIDE {
            let load_x = i * TX + tx;
            let period_x = 2u32 * (width - 1u32);
            // wide window starts at col_base - 2R (offset load_x - 2R from col_base).
            let raw_x = (col_base + load_x + period_x - 2u32 * R) % period_x;
            let gx = if raw_x < width {
                raw_x
            } else {
                period_x - raw_x
            };
            let off = (gy as usize) * w + (gx as usize);
            let s_val = if channel == 0u32 {
                src_a[off]
            } else if channel == 1u32 {
                src_b[off]
            } else {
                src_c[off]
            };
            wide_src[load_x as usize] = s_val;
            i += 1u32;
        }
        sync_cube();

        // Compute H_blur(src) into mu1_row at TILE_COLS positions.
        // mu1_row[load_x] corresponds to gx = col_base - R + load_x; its
        // H-blur reads wide_src[load_x .. load_x + DIAM] (= src around
        // gx - R .. gx + R, i.e. centered DIAM samples).
        let mut i2: u32 = 0u32;
        while i2 * TX + tx < TILE_COLS {
            let load_x = i2 * TX + tx;
            let mut hsum = 0.0_f32;
            let mut jj: u32 = 0u32;
            while jj < DIAM {
                hsum += wide_src[(load_x + jj) as usize];
                jj += 1u32;
            }
            mu1_row[load_x as usize] = hsum * INV_DIAM;
            i2 += 1u32;
        }
        sync_cube();

        // Per-column H-sum of |src - mu1| over the DIAM-wide kernel.
        // src at this column comes from wide_src[2R + tx + j] (the
        // src_row[tx+j] equivalent in the inner TILE_COLS window
        // sits at wide_src offset R + tx + j; we want the inner
        // src_row equivalent at index tx + j, which is wide_src[R + tx + j]
        // ... wait, let me re-derive: wide_src[idx] holds src at
        // gx = col_base - 2R + idx. The narrow src_row would have
        // held src at gx = col_base - R + load_x = col_base - 2R + (load_x + R)
        // → wide_src[load_x + R]. The H-sum uses src at tx + j for
        // j in 0..DIAM (i.e., load_x = tx + j), so wide_src[tx + j + R].
        let mut h_sum = 0.0_f32;
        let mut j: u32 = 0u32;
        while j < DIAM {
            let s = wide_src[(tx + j + R) as usize];
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

    // ============================ WALK STRIP ============================
    let mut slot: u32 = 0u32;
    let mut sy: u32 = 0u32;
    while sy < strip_h {
        // V-blur output: activity at strip-row sy.
        let activity = sum_act * INV_DIAM;
        let mask_w = 1.0 / (1.0 + K_MASK * activity);
        let iw_w = 1.0 + K_IW * activity;

        // Only the strip's INNER rows contribute features. Per-pixel
        // mu1/mu2/ssq/s12 reads at the inner pixel come from the persist
        // planes (image-wide-correct values). Skip everything else for
        // overlap rows — they only matter as inputs to the activity
        // V-blur slide.
        //
        // Strip-mode walker: also gate on the body row range. The
        // image-strip-level body row range is in the SAME coordinate
        // system as the kernel's `height` axis (the strip allocation
        // is `strip_alloc_h` rows; body is `[y_body_start, y_body_end)`
        // within that).
        let is_inner_row = sy >= inner_off && sy < inner_off + inner_h;
        let gy_check = strip_top + sy;
        let is_body_row = gy_check >= y_body_start && gy_check < y_body_end;
        if is_inner_row && is_body_row && in_bounds {
            let gy = strip_top + sy;
            let off = (gy as usize) * w + (col as usize);

            let sv = if channel == 0u32 {
                src_a[off]
            } else if channel == 1u32 {
                src_b[off]
            } else {
                src_c[off]
            };
            let dv = if channel == 0u32 {
                dst_a[off]
            } else if channel == 1u32 {
                dst_b[off]
            } else {
                dst_c[off]
            };
            let mu1 = mu1_all[ch_base + off];
            let mu2 = mu2_all[ch_base + off];
            let ssq = ssq_all[ch_base + off];
            let s12_v = s12_all[ch_base + off];

            // Masked SSIM math — same as fused.rs but applied with
            // mask BEFORE max-clamp (matches CPU
            // `ssim_channel_masked_inner`).
            let mu_diff = mu1 - mu2;
            let num_m = fma(mu_diff, -mu_diff, 1.0);
            let inner_ns = fma(-mu1, mu2, s12_v);
            let num_s = fma(2.0, inner_ns, C2);
            let inner_ds_inner = fma(-mu1, mu1, ssq);
            let denom_s = fma(-mu2, mu2, inner_ds_inner) + C2;
            let raw_sd = 1.0 - (num_m * num_s) / denom_s;

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
            //   artifact = max(d1, 0); detail_lost = max(-d1, 0)
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
        }

        // ============================ SLIDE ACTIVITY ============================
        // Compute next-row H-blurred activity at strip-local row sy+R+1
        // (strip-local mirror). Remove the oldest entry in the circular
        // buffer.
        let buf_idx = (slot * TX + tx) as usize;
        let old_act = buf_act[buf_idx];

        // Next strip-local row to load: sy + R + 1, with strip-local mirror.
        let next_raw = sy + R + 1u32;
        let next_sy = if next_raw < strip_h {
            next_raw
        } else {
            // Reflect around (strip_h - 1).
            let folded = period_sy - (next_raw % period_sy);
            // After the % the result is in [1, period_sy]; the fold above
            // can land at strip_h-1+1 = strip_h when next_raw == strip_h.
            // Clamp to strip_h - 1 to be safe.
            u32::min(folded, strip_h - 1u32)
        };
        let next_gy = strip_top + next_sy;

        sync_cube();
        // Load WIDE src for next row.
        let mut i2w: u32 = 0u32;
        while i2w * TX + tx < TILE_COLS_WIDE {
            let load_x = i2w * TX + tx;
            let period_x = 2u32 * (width - 1u32);
            let raw_x = (col_base + load_x + period_x - 2u32 * R) % period_x;
            let gx = if raw_x < width {
                raw_x
            } else {
                period_x - raw_x
            };
            let off2 = (next_gy as usize) * w + (gx as usize);
            let s_val = if channel == 0u32 {
                src_a[off2]
            } else if channel == 1u32 {
                src_b[off2]
            } else {
                src_c[off2]
            };
            wide_src[load_x as usize] = s_val;
            i2w += 1u32;
        }
        sync_cube();

        // Compute H_blur(src) into mu1_row for next row.
        let mut i2d: u32 = 0u32;
        while i2d * TX + tx < TILE_COLS {
            let load_x = i2d * TX + tx;
            let mut hsum2 = 0.0_f32;
            let mut jj2: u32 = 0u32;
            while jj2 < DIAM {
                hsum2 += wide_src[(load_x + jj2) as usize];
                jj2 += 1u32;
            }
            mu1_row[load_x as usize] = hsum2 * INV_DIAM;
            i2d += 1u32;
        }
        sync_cube();

        let mut nh = 0.0_f32;
        let mut j2: u32 = 0u32;
        while j2 < DIAM {
            let s = wide_src[(tx + j2 + R) as usize];
            let m = mu1_row[(tx + j2) as usize];
            nh += f32::abs(s - m);
            j2 += 1u32;
        }
        let new_h_avg = nh * INV_DIAM;
        sum_act = sum_act + new_h_avg - old_act;
        buf_act[buf_idx] = new_h_avg;

        slot = (slot + 1u32) % DIAM;
        sy += 1u32;
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
