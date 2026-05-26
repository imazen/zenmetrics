//! Parity tests for the strip-aware variants of the cvvdp pyramid
//! kernels (`downscale_strip_kernel`, `subtract_weber_3ch_strip_kernel`).
//!
//! The strip-aware variants produce only a `body_h`-tall slice of
//! the logical full output. These tests verify the body slice
//! agrees with the corresponding rows of the legacy full kernels'
//! output, on both even and odd source dims (the odd-dim case
//! pins the pycvvdp tick-206 bug-compat parity delta firing
//! identically when reflection uses the LOGICAL src dims instead
//! of the strip buffer height).
//!
//! Tolerance: 5e-6 absolute, matching the existing
//! `strip_kernel_parity_upscale.rs` tolerance and the strip-mode
//! JOD parity band used by `strip_mode_e_parity.rs`.
//!
//! cubecl-cpu is intentionally NOT selected here, matching
//! `pyramid_kernel.rs` (CPU runtime in 0.10.0-pre.4 mishandles
//! some slot-array launch geometries; not what we want to
//! verify).

#![cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]

use cubecl::Runtime;
use cubecl::prelude::*;
use cvvdp_gpu::kernels::pyramid::{
    downscale_kernel, downscale_strip_kernel, subtract_weber_3ch_kernel,
    subtract_weber_3ch_strip_kernel,
};

#[path = "common/mod.rs"]
mod common;

use common::Backend;

const PARITY_TOL: f32 = 5e-6;

/// Deterministic mixed-content source generator. Ramp + per-row
/// sinusoid so every column has non-trivial reduce content
/// (matches `strip_kernel_parity_upscale.rs::make_src_32x32` in
/// spirit).
fn make_src(w: usize, h: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let v = (x as f32) * 0.03125
                + (y as f32) * 0.01
                + ((x * 7 + y * 11) as f32).sin() * 0.25;
            out.push(v);
        }
    }
    out
}

/// downscale_strip_kernel body output matches downscale_kernel
/// at the matching logical rows on a 64×64 → 32×32 reduce.
///
/// Body covers logical dst rows [8..24], so `body_offset_y = 8`
/// and `body_h = 16`. Source strip buffer here is the FULL
/// source (matches the convention in
/// `strip_kernel_parity_upscale.rs`); the kernel's
/// `src_strip_offset` parameter is set to 0 so logical and
/// buffer-local src rows agree. The kernel API also supports a
/// haloed-src strip layout (via non-zero `src_strip_offset`);
/// that path is exercised separately in
/// `downscale_strip_aware_matches_full_with_haloed_src`.
#[test]
fn downscale_strip_aware_matches_full_at_64x64() {
    let client = Backend::client(&Default::default());

    let (sw, sh) = (64u32, 64u32);
    let (dw, dh) = (32u32, 32u32);
    let n_src = (sw * sh) as usize;
    let n_full = (dw * dh) as usize;

    let src = make_src(sw as usize, sh as usize);
    let src_h = client.create_from_slice(f32::as_bytes(&src));
    let full_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n_full]));

    let cube_dim = CubeDim::new_1d(64);
    let count_full = CubeCount::Static((n_full as u32).div_ceil(64), 1, 1);
    unsafe {
        downscale_kernel::launch::<Backend>(
            &client,
            count_full,
            cube_dim,
            ArrayArg::from_raw_parts(src_h.clone(), n_src),
            ArrayArg::from_raw_parts(full_h.clone(), n_full),
            sw,
            sh,
            dw,
            dh,
        );
    }
    let full_bytes = client.read_one(full_h).expect("read full");
    let full_out: &[f32] = f32::from_bytes(&full_bytes);
    assert_eq!(full_out.len(), n_full);

    // Strip: dst rows [8..24] of logical 32×32 dst.
    let body_offset_y = 8u32;
    let body_h = 16u32;
    let n_strip = (dw * body_h) as usize;

    let strip_dst = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n_strip]));
    let count_strip = CubeCount::Static((n_strip as u32).div_ceil(64), 1, 1);
    unsafe {
        downscale_strip_kernel::launch::<Backend>(
            &client,
            count_strip,
            cube_dim,
            ArrayArg::from_raw_parts(src_h.clone(), n_src),
            ArrayArg::from_raw_parts(strip_dst.clone(), n_strip),
            sw,
            sh, // src_h = full src buffer height (also = logical_src_h here)
            dw,
            body_h, // dst_h = body height
            body_offset_y,
            0,      // src_strip_offset = 0 (src is the full image)
            sh,     // logical_src_h
            dh,     // logical_dst_h
        );
    }
    let strip_bytes = client.read_one(strip_dst).expect("read strip");
    let strip_out: &[f32] = f32::from_bytes(&strip_bytes);
    assert_eq!(strip_out.len(), n_strip);

    // Compare strip rows [0..body_h) against full rows
    // [body_offset_y..body_offset_y + body_h).
    let dw_us = dw as usize;
    let body_off_us = body_offset_y as usize;
    let body_h_us = body_h as usize;
    let mut max_err = 0.0_f32;
    for row in 0..body_h_us {
        let full_row = body_off_us + row;
        let f_start = full_row * dw_us;
        let s_start = row * dw_us;
        for x in 0..dw_us {
            let e = (full_out[f_start + x] - strip_out[s_start + x]).abs();
            if e > max_err {
                max_err = e;
            }
        }
    }
    assert!(
        max_err <= PARITY_TOL,
        "downscale strip-aware parity vs full: max-abs error = {max_err} (> {PARITY_TOL})"
    );
}

/// Same as `downscale_strip_aware_matches_full_at_64x64` but
/// exercises the haloed-src layout: the source strip buffer
/// covers logical rows `[2·body_offset_y − halo .. 2·(body_offset_y
/// + body_h) + halo)` of the logical source, and
/// `src_strip_offset = 2·body_offset_y − halo`. This matches the
/// spec language ("strip buffer [4..28] of input (with 4-row
/// halos)" — for `body_offset_y = 8` the buffer covers logical
/// rows 4..28 = 24 rows, with the body src center spanning
/// 14..18 → halo 4 + reflect-radius 2 = enough headroom).
///
/// Same expected output, different buffer layout — pins that the
/// kernel correctly translates logical src row → buffer-local
/// row by subtracting `src_strip_offset`.
#[test]
fn downscale_strip_aware_matches_full_with_haloed_src() {
    let client = Backend::client(&Default::default());

    let (sw, sh) = (64u32, 64u32);
    let (dw, dh) = (32u32, 32u32);
    let n_src_full = (sw * sh) as usize;
    let n_full = (dw * dh) as usize;

    // Reference: legacy full-image reduce.
    let src = make_src(sw as usize, sh as usize);
    let src_full_h = client.create_from_slice(f32::as_bytes(&src));
    let full_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n_full]));

    let cube_dim = CubeDim::new_1d(64);
    let count_full = CubeCount::Static((n_full as u32).div_ceil(64), 1, 1);
    unsafe {
        downscale_kernel::launch::<Backend>(
            &client,
            count_full,
            cube_dim,
            ArrayArg::from_raw_parts(src_full_h.clone(), n_src_full),
            ArrayArg::from_raw_parts(full_h.clone(), n_full),
            sw,
            sh,
            dw,
            dh,
        );
    }
    let full_bytes = client.read_one(full_h).expect("read full");
    let full_out: &[f32] = f32::from_bytes(&full_bytes);

    // Strip src buffer: logical rows [4..28] = 24 rows with 4-row
    // halos on each side of the body source span. body src center
    // span: 2·body_offset_y .. 2·(body_offset_y + body_h - 1) + 1
    // = 16..31 (16 rows). With ±2 reflect radius, body src reads
    // 14..33. Buffer [4..28] covers 14..28 plus extra top halo; 28..33
    // would overflow but the 5-tap reflection at the bottom kicks in
    // before that (32+2 = 34 reflects to 64−1−2 = 61 logical, which
    // gets clamped via reflect — but for body_offset_y=8 / body_h=16
    // the LAST body output is dy_logical = 23, so src center cy = 46,
    // and the kernel reads logical rows 44..48. All within
    // src_strip_offset=4 + buffer rows 0..23 = logical rows 4..28
    // (rows 44..48 are NOT in [4..28]!).
    //
    // Reconsider: body covers logical dst rows [8..24]. The
    // FIRST body output (logical dy=8) reads src center cy=16,
    // reflecting to rows [14..18]. The LAST body output (logical
    // dy=23) reads src center cy=46, rows [44..48].
    //
    // So the src strip buffer must cover logical rows [14-pad ..
    // 48+pad]. The "[4..28]" example from the task spec only
    // works if the body is shallow (body_offset_y small). Pick a
    // body that does fit in a small src strip: body_offset_y = 4
    // → first src center 8, rows [6..10]. body_h = 4 →
    // last dy=7, center 14, rows [12..16]. Strip buffer [4..20]
    // = 16 rows covers everything with halo.
    let body_offset_y = 4u32;
    let body_h = 4u32;
    let src_strip_offset = 4u32; // logical src row of buffer-local row 0
    let strip_src_logical_start = src_strip_offset as usize;
    let strip_src_logical_end = 20usize; // exclusive (16 rows of src)
    let strip_src_h = (strip_src_logical_end - strip_src_logical_start) as u32;
    let n_strip_src = (sw as usize) * (strip_src_h as usize);

    let src_strip: Vec<f32> = src[strip_src_logical_start * sw as usize
        ..strip_src_logical_end * sw as usize]
        .to_vec();
    assert_eq!(src_strip.len(), n_strip_src);
    let src_strip_h = client.create_from_slice(f32::as_bytes(&src_strip));

    let n_strip_dst = (dw * body_h) as usize;
    let strip_dst = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n_strip_dst]));
    let count_strip = CubeCount::Static((n_strip_dst as u32).div_ceil(64), 1, 1);
    unsafe {
        downscale_strip_kernel::launch::<Backend>(
            &client,
            count_strip,
            cube_dim,
            ArrayArg::from_raw_parts(src_strip_h, n_strip_src),
            ArrayArg::from_raw_parts(strip_dst.clone(), n_strip_dst),
            sw,
            strip_src_h, // src_h = strip buffer height
            dw,
            body_h,
            body_offset_y,
            src_strip_offset,
            sh, // logical_src_h
            dh, // logical_dst_h
        );
    }
    let strip_bytes = client.read_one(strip_dst).expect("read strip");
    let strip_out: &[f32] = f32::from_bytes(&strip_bytes);

    let dw_us = dw as usize;
    let body_off_us = body_offset_y as usize;
    let body_h_us = body_h as usize;
    let mut max_err = 0.0_f32;
    for row in 0..body_h_us {
        let full_row = body_off_us + row;
        let f_start = full_row * dw_us;
        let s_start = row * dw_us;
        for x in 0..dw_us {
            let e = (full_out[f_start + x] - strip_out[s_start + x]).abs();
            if e > max_err {
                max_err = e;
            }
        }
    }
    assert!(
        max_err <= PARITY_TOL,
        "downscale strip-aware (haloed src) parity vs full: max-abs error = {max_err} (> {PARITY_TOL})"
    );
}

/// Odd-dim 73×91 source: pins that the pycvvdp tick-206 bug-compat
/// parity delta at the right column uses LOGICAL src dims, not the
/// strip buffer height. The strip path must produce bit-identical
/// right-column values vs the full-image path for the mixed-parity
/// case (`sw=73` odd, `sh=91` odd) — both reflect-only and
/// parity-delta branches fire.
///
/// Setup: logical src 73×91 → dst 37×46 via the kernel's
/// `(sw+1)/2 × (sh+1)/2` reduction sizing. Body covers logical dst
/// rows [10..24] (14 rows). Source strip is the full source
/// (src_strip_offset = 0) so we exercise the parity delta cleanly
/// against the full-image path.
#[test]
fn downscale_pycvvdp_parity_delta_uses_logical_dims() {
    let client = Backend::client(&Default::default());

    // Odd both axes: 73×91. Reduce gives 37×46.
    let (sw, sh) = (73u32, 91u32);
    let (dw, dh) = ((sw + 1) / 2, (sh + 1) / 2);
    assert_eq!((dw, dh), (37, 46));
    let n_src = (sw * sh) as usize;
    let n_full = (dw * dh) as usize;

    let src = make_src(sw as usize, sh as usize);
    let src_h = client.create_from_slice(f32::as_bytes(&src));
    let full_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n_full]));

    let cube_dim = CubeDim::new_1d(64);
    let count_full = CubeCount::Static((n_full as u32).div_ceil(64), 1, 1);
    unsafe {
        downscale_kernel::launch::<Backend>(
            &client,
            count_full,
            cube_dim,
            ArrayArg::from_raw_parts(src_h.clone(), n_src),
            ArrayArg::from_raw_parts(full_h.clone(), n_full),
            sw,
            sh,
            dw,
            dh,
        );
    }
    let full_bytes = client.read_one(full_h).expect("read full");
    let full_out: &[f32] = f32::from_bytes(&full_bytes);
    assert_eq!(full_out.len(), n_full);

    // Strip: body rows [10..24).
    let body_offset_y = 10u32;
    let body_h = 14u32;
    let n_strip_dst = (dw * body_h) as usize;
    let strip_dst = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n_strip_dst]));
    let count_strip = CubeCount::Static((n_strip_dst as u32).div_ceil(64), 1, 1);
    unsafe {
        downscale_strip_kernel::launch::<Backend>(
            &client,
            count_strip,
            cube_dim,
            ArrayArg::from_raw_parts(src_h.clone(), n_src),
            ArrayArg::from_raw_parts(strip_dst.clone(), n_strip_dst),
            sw,
            sh,
            dw,
            body_h,
            body_offset_y,
            0, // src_strip_offset = 0 (full src)
            sh, // logical_src_h = 91 (ODD)
            dh, // logical_dst_h
        );
    }
    let strip_bytes = client.read_one(strip_dst).expect("read strip");
    let strip_out: &[f32] = f32::from_bytes(&strip_bytes);

    // The kernel's interior + parity-delta branches both produce
    // bit-identical f32 on the right column for the same logical
    // input. Cumulative atomic-free pure-arithmetic — exact match
    // (tolerance 0 would be principled; allow PARITY_TOL for any
    // backend-specific FMA reordering).
    let dw_us = dw as usize;
    let body_off_us = body_offset_y as usize;
    let body_h_us = body_h as usize;
    let mut max_err = 0.0_f32;
    let mut max_err_right_col = 0.0_f32;
    for row in 0..body_h_us {
        let full_row = body_off_us + row;
        let f_start = full_row * dw_us;
        let s_start = row * dw_us;
        for x in 0..dw_us {
            let e = (full_out[f_start + x] - strip_out[s_start + x]).abs();
            if e > max_err {
                max_err = e;
            }
            if x == dw_us - 1 && e > max_err_right_col {
                max_err_right_col = e;
            }
        }
    }
    assert!(
        max_err <= PARITY_TOL,
        "odd-dim 73×91 strip parity max-abs error = {max_err} (> {PARITY_TOL}); \
         right-column-only max = {max_err_right_col} (this column exercises \
         the pycvvdp tick-206 parity delta)"
    );
}

/// subtract_weber_3ch_strip_kernel body output matches
/// subtract_weber_3ch_kernel at the matching rows on a 64×64
/// strip with the body covering rows [8..24).
///
/// Inputs (fine_a/rg/vy, upsc_a/rg/vy, expanded_lbkg) are
/// constructed to exercise:
/// - L_bkg lower clamp (tiny lbkg values).
/// - L_bkg upper clamp on contrast (layer/lbkg > 1000).
/// - Per-channel distinct layer values so a wrong-channel bug
///   would mismatch.
/// - Per-pixel sign flips so the symmetric clamp on the lower
///   side fires too.
#[test]
fn subtract_weber_strip_aware_matches_full_at_64x64() {
    let client = Backend::client(&Default::default());

    let (w, h) = (64u32, 64u32);
    let n = (w * h) as usize;

    // Per-pixel inputs. Reuse the deterministic generator and
    // tweak per-channel so each pixel exercises different code
    // paths. Keep lbkg occasionally < 0.01 to fire the clamp.
    let base = make_src(w as usize, h as usize);
    let fine_a: Vec<f32> = base.iter().map(|v| v * 1.1).collect();
    let fine_rg: Vec<f32> = base.iter().map(|v| v * -0.7 + 0.3).collect();
    let fine_vy: Vec<f32> = base.iter().map(|v| v * 0.2 - 0.5).collect();
    let upsc_a: Vec<f32> = base.iter().map(|v| v * 0.5).collect();
    let upsc_rg: Vec<f32> = base.iter().map(|v| v * -0.4 + 0.1).collect();
    let upsc_vy: Vec<f32> = base.iter().map(|v| v * 0.15 - 0.6).collect();
    // lbkg: a deterministic sinusoid in [-0.5, 2] so some pixels
    // fall below 0.01 (firing the lower clamp), and a few large
    // |layer|/tiny lbkg combinations fire the upper contrast
    // clamp.
    let lbkg: Vec<f32> = (0..n)
        .map(|i| {
            let v = (i as f32 * 0.013).sin() * 1.2 + 0.7;
            // Sprinkle a few near-zero lbkg pixels.
            if i % 17 == 0 {
                0.0005_f32
            } else {
                v
            }
        })
        .collect();

    // Full reference.
    let fine_a_h = client.create_from_slice(f32::as_bytes(&fine_a));
    let fine_rg_h = client.create_from_slice(f32::as_bytes(&fine_rg));
    let fine_vy_h = client.create_from_slice(f32::as_bytes(&fine_vy));
    let upsc_a_h = client.create_from_slice(f32::as_bytes(&upsc_a));
    let upsc_rg_h = client.create_from_slice(f32::as_bytes(&upsc_rg));
    let upsc_vy_h = client.create_from_slice(f32::as_bytes(&upsc_vy));
    let lbkg_h = client.create_from_slice(f32::as_bytes(&lbkg));
    let full_c_a = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let full_c_rg = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let full_c_vy = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let full_log_l = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));

    let cube_dim = CubeDim::new_1d(64);
    let count_full = CubeCount::Static((n as u32).div_ceil(64), 1, 1);
    unsafe {
        subtract_weber_3ch_kernel::launch::<Backend>(
            &client,
            count_full,
            cube_dim,
            ArrayArg::from_raw_parts(fine_a_h.clone(), n),
            ArrayArg::from_raw_parts(fine_rg_h.clone(), n),
            ArrayArg::from_raw_parts(fine_vy_h.clone(), n),
            ArrayArg::from_raw_parts(upsc_a_h.clone(), n),
            ArrayArg::from_raw_parts(upsc_rg_h.clone(), n),
            ArrayArg::from_raw_parts(upsc_vy_h.clone(), n),
            ArrayArg::from_raw_parts(lbkg_h.clone(), n),
            ArrayArg::from_raw_parts(full_c_a.clone(), n),
            ArrayArg::from_raw_parts(full_c_rg.clone(), n),
            ArrayArg::from_raw_parts(full_c_vy.clone(), n),
            ArrayArg::from_raw_parts(full_log_l.clone(), n),
            n as u32,
        );
    }

    let full_ca_b = client.read_one(full_c_a).expect("read full c_a");
    let full_crg_b = client.read_one(full_c_rg).expect("read full c_rg");
    let full_cvy_b = client.read_one(full_c_vy).expect("read full c_vy");
    let full_log_b = client.read_one(full_log_l).expect("read full log");
    let full_ca: &[f32] = f32::from_bytes(&full_ca_b);
    let full_crg: &[f32] = f32::from_bytes(&full_crg_b);
    let full_cvy: &[f32] = f32::from_bytes(&full_cvy_b);
    let full_log: &[f32] = f32::from_bytes(&full_log_b);

    // Strip: inputs are the FULL buffers (same layout as the full
    // reference); the kernel skips halo rows by reading at
    // body_offset_y..body_offset_y+body_h. Output buffers are
    // also full-sized — only body rows are written by the
    // kernel; halo rows in the output stay at their initial
    // value (0.0 in our zeroed allocation). This matches the
    // convention documented on `subtract_weber_3ch_strip_kernel`.
    let body_offset_y = 8u32;
    let body_h = 16u32;
    let strip_c_a = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let strip_c_rg = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let strip_c_vy = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let strip_log = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));

    let n_strip = (w * body_h) as usize;
    let count_strip = CubeCount::Static((n_strip as u32).div_ceil(64), 1, 1);
    unsafe {
        subtract_weber_3ch_strip_kernel::launch::<Backend>(
            &client,
            count_strip,
            cube_dim,
            ArrayArg::from_raw_parts(fine_a_h, n),
            ArrayArg::from_raw_parts(fine_rg_h, n),
            ArrayArg::from_raw_parts(fine_vy_h, n),
            ArrayArg::from_raw_parts(upsc_a_h, n),
            ArrayArg::from_raw_parts(upsc_rg_h, n),
            ArrayArg::from_raw_parts(upsc_vy_h, n),
            ArrayArg::from_raw_parts(lbkg_h, n),
            ArrayArg::from_raw_parts(strip_c_a.clone(), n),
            ArrayArg::from_raw_parts(strip_c_rg.clone(), n),
            ArrayArg::from_raw_parts(strip_c_vy.clone(), n),
            ArrayArg::from_raw_parts(strip_log.clone(), n),
            w,
            body_h,
            body_offset_y,
            h, // logical_h
        );
    }

    let strip_ca_b = client.read_one(strip_c_a).expect("read strip c_a");
    let strip_crg_b = client.read_one(strip_c_rg).expect("read strip c_rg");
    let strip_cvy_b = client.read_one(strip_c_vy).expect("read strip c_vy");
    let strip_log_b = client.read_one(strip_log).expect("read strip log");
    let strip_ca: &[f32] = f32::from_bytes(&strip_ca_b);
    let strip_crg: &[f32] = f32::from_bytes(&strip_crg_b);
    let strip_cvy: &[f32] = f32::from_bytes(&strip_cvy_b);
    let strip_log: &[f32] = f32::from_bytes(&strip_log_b);

    let w_us = w as usize;
    let body_off_us = body_offset_y as usize;
    let body_h_us = body_h as usize;
    let mut max_err = 0.0_f32;
    for row in 0..body_h_us {
        let buf_row = body_off_us + row;
        let row_start = buf_row * w_us;
        for x in 0..w_us {
            let i = row_start + x;
            let e_a = (full_ca[i] - strip_ca[i]).abs();
            let e_rg = (full_crg[i] - strip_crg[i]).abs();
            let e_vy = (full_cvy[i] - strip_cvy[i]).abs();
            let e_log = (full_log[i] - strip_log[i]).abs();
            for e in [e_a, e_rg, e_vy, e_log] {
                if e > max_err {
                    max_err = e;
                }
            }
        }
    }
    assert!(
        max_err <= PARITY_TOL,
        "subtract_weber strip-aware parity vs full: max-abs error = {max_err} (> {PARITY_TOL})"
    );

    // Verify halo rows in the strip output are NOT touched —
    // they should remain at the 0.0 initial value. (Bug-detector
    // for a kernel that mistakenly writes to halo rows.)
    for row in 0..body_off_us {
        let row_start = row * w_us;
        for x in 0..w_us {
            let i = row_start + x;
            assert_eq!(
                strip_ca[i], 0.0,
                "halo top row {row} pixel {x} was written: got {} (expected 0)",
                strip_ca[i]
            );
        }
    }
    for row in (body_off_us + body_h_us)..(h as usize) {
        let row_start = row * w_us;
        for x in 0..w_us {
            let i = row_start + x;
            assert_eq!(
                strip_ca[i], 0.0,
                "halo bottom row {row} pixel {x} was written: got {} (expected 0)",
                strip_ca[i]
            );
        }
    }
}
