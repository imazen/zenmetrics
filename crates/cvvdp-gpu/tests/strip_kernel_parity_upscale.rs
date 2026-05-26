//! Parity tests for the strip-aware variants of the cvvdp expand
//! kernels (`upscale_v_strip_kernel`, `upscale_h_strip_kernel`).
//!
//! The strip-aware variants produce only a `body_h`-tall slice of
//! the logical full expand output. These tests verify the body
//! slice agrees with the corresponding rows of the legacy full
//! kernels' output at 32×32 → 64×64.
//!
//! Tolerance: 5e-6 absolute, matching the existing
//! `pyramid_kernel.rs` upscale tolerances and the strip-mode JOD
//! parity band used by `strip_mode_e_parity.rs`.

#![cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]

use cubecl::Runtime;
use cubecl::prelude::*;
use cvvdp_gpu::kernels::pyramid::{
    upscale_h_kernel, upscale_h_strip_kernel, upscale_v_kernel, upscale_v_strip_kernel,
};

#[path = "common/mod.rs"]
mod common;

use common::Backend;

const PARITY_TOL: f32 = 5e-6;

/// Deterministic mixed-content 32×32 source: a small ramp plus a
/// per-row sinusoid so the V expand has non-trivial content in
/// every column.
fn make_src_32x32() -> Vec<f32> {
    let (w, h) = (32usize, 32usize);
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            // Mix a ramp with a deterministic high-frequency
            // component so the expand exercises all 5 taps.
            let v = (x as f32) * 0.03125 + (y as f32) * 0.01
                + ((x * 7 + y * 11) as f32).sin() * 0.25;
            out.push(v);
        }
    }
    out
}

#[test]
fn upscale_v_strip_aware_matches_full_at_32x32_to_64x64() {
    // Full V expand: 32×32 → 32×64.
    let client = Backend::client(&Default::default());

    let src = make_src_32x32();
    let (sw, sh) = (32u32, 32u32);
    let dst_h = 64u32;
    let n_src = (sw * sh) as usize;
    let n_full = (sw * dst_h) as usize;

    let src_h = client.create_from_slice(f32::as_bytes(&src));
    let full_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n_full]));

    let cube_dim = CubeDim::new_1d(64);
    let count_full = CubeCount::Static((n_full as u32).div_ceil(64), 1, 1);

    unsafe {
        upscale_v_kernel::launch::<Backend>(
            &client,
            count_full,
            cube_dim,
            ArrayArg::from_raw_parts(src_h.clone(), n_src),
            ArrayArg::from_raw_parts(full_h.clone(), n_full),
            sw,
            sh,
            dst_h,
        );
    }

    let full_bytes = client.read_one(full_h.clone()).expect("read full");
    let full_out: &[f32] = f32::from_bytes(&full_bytes);
    assert_eq!(full_out.len(), n_full);

    // Strip V expand: same logical 32×32 → 32×64, body covers rows
    // [16..48], so body_offset_y = 16, body_h = 32. Strip buffer is
    // 32×32 (sw × body_h).
    let body_offset_y = 16u32;
    let body_h = 32u32;
    let n_strip = (sw * body_h) as usize;

    let strip_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n_strip]));
    let count_strip = CubeCount::Static((n_strip as u32).div_ceil(64), 1, 1);

    unsafe {
        upscale_v_strip_kernel::launch::<Backend>(
            &client,
            count_strip,
            cube_dim,
            ArrayArg::from_raw_parts(src_h.clone(), n_src),
            ArrayArg::from_raw_parts(strip_h.clone(), n_strip),
            sw,
            sh,        // logical_src_h
            dst_h,     // logical_dst_h
            body_offset_y,
            body_h,
        );
    }

    let strip_bytes = client.read_one(strip_h.clone()).expect("read strip");
    let strip_out: &[f32] = f32::from_bytes(&strip_bytes);
    assert_eq!(strip_out.len(), n_strip);

    // Compare strip rows [0..body_h) against full rows
    // [body_offset_y..body_offset_y + body_h).
    let sw_us = sw as usize;
    let body_off_us = body_offset_y as usize;
    let body_h_us = body_h as usize;
    let mut max_err = 0.0_f32;
    for row in 0..body_h_us {
        let full_row = body_off_us + row;
        let f_start = full_row * sw_us;
        let s_start = row * sw_us;
        for x in 0..sw_us {
            let e = (full_out[f_start + x] - strip_out[s_start + x]).abs();
            if e > max_err {
                max_err = e;
            }
        }
    }
    assert!(
        max_err <= PARITY_TOL,
        "V strip-aware parity vs full: max-abs error = {max_err} (> {PARITY_TOL})"
    );
}

#[test]
fn upscale_h_strip_aware_matches_full_at_32x32_to_64x64() {
    // Full pipeline: V expand 32×32 → 32×64, then H expand 32×64 → 64×64.
    let client = Backend::client(&Default::default());

    let src = make_src_32x32();
    let (sw, sh) = (32u32, 32u32);
    let (dw, dh) = (64u32, 64u32);
    let n_src = (sw * sh) as usize;
    let n_v = (sw * dh) as usize;
    let n_dst = (dw * dh) as usize;

    let src_h = client.create_from_slice(f32::as_bytes(&src));
    let vscratch_full = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n_v]));
    let full_h_out = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n_dst]));

    let cube_dim = CubeDim::new_1d(64);
    let count_v = CubeCount::Static((n_v as u32).div_ceil(64), 1, 1);
    let count_h = CubeCount::Static((n_dst as u32).div_ceil(64), 1, 1);

    unsafe {
        upscale_v_kernel::launch::<Backend>(
            &client,
            count_v,
            cube_dim,
            ArrayArg::from_raw_parts(src_h.clone(), n_src),
            ArrayArg::from_raw_parts(vscratch_full.clone(), n_v),
            sw,
            sh,
            dh,
        );
        upscale_h_kernel::launch::<Backend>(
            &client,
            count_h,
            cube_dim,
            ArrayArg::from_raw_parts(vscratch_full.clone(), n_v),
            ArrayArg::from_raw_parts(full_h_out.clone(), n_dst),
            sw,
            dw,
            dh,
        );
    }

    let full_bytes = client.read_one(full_h_out.clone()).expect("read full h");
    let full_out: &[f32] = f32::from_bytes(&full_bytes);
    assert_eq!(full_out.len(), n_dst);

    // Strip-aware: V-strip 32×32 → 32×32 body, then H-strip 32×32 → 64×32 body.
    // Body covers full output rows [16..48].
    let body_offset_y = 16u32;
    let body_h = 32u32;
    let n_v_strip = (sw * body_h) as usize;
    let n_dst_strip = (dw * body_h) as usize;

    let vscratch_strip = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n_v_strip]));
    let strip_h_out = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n_dst_strip]));

    let count_v_strip = CubeCount::Static((n_v_strip as u32).div_ceil(64), 1, 1);
    let count_h_strip = CubeCount::Static((n_dst_strip as u32).div_ceil(64), 1, 1);

    unsafe {
        upscale_v_strip_kernel::launch::<Backend>(
            &client,
            count_v_strip,
            cube_dim,
            ArrayArg::from_raw_parts(src_h.clone(), n_src),
            ArrayArg::from_raw_parts(vscratch_strip.clone(), n_v_strip),
            sw,
            sh,
            dh,
            body_offset_y,
            body_h,
        );
        upscale_h_strip_kernel::launch::<Backend>(
            &client,
            count_h_strip,
            cube_dim,
            ArrayArg::from_raw_parts(vscratch_strip.clone(), n_v_strip),
            ArrayArg::from_raw_parts(strip_h_out.clone(), n_dst_strip),
            sw,
            dw,
            body_h,         // in_h = body height (rows actually populated)
            dh,             // logical_dst_h (documentary, ignored inside)
            body_offset_y,  // documentary, ignored inside
        );
    }

    let strip_bytes = client.read_one(strip_h_out.clone()).expect("read strip h");
    let strip_out: &[f32] = f32::from_bytes(&strip_bytes);
    assert_eq!(strip_out.len(), n_dst_strip);

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
        "H strip-aware parity vs full: max-abs error = {max_err} (> {PARITY_TOL})"
    );
}
