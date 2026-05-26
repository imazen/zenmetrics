//! Strip-aware kernel parity tests (Mode E Phase 3 follow-on,
//! chunk 1/6 — masking PU blur).
//!
//! Pins the bit-exact (modulo f32 add-reorder noise) parity claim
//! between the strip-aware sibling kernels added in this chunk and
//! the existing full-image kernels they shadow. The methodology is
//! the same one used by the prior `pool_band_3ch_offset_kernel`
//! parity tests in `strip_mode_e_phase3.rs`:
//!
//! 1. Build a synthetic 64×64 input plane per channel.
//! 2. Dispatch the **legacy** full-image kernel; capture its output
//!    rows `[BODY_TOP..BODY_BOT)` as the bit-exact reference.
//! 3. Build a **strip buffer** containing global rows
//!    `[BODY_TOP - HALO_TOP .. BODY_BOT + HALO_BOT)` — a contiguous
//!    sub-slab of the full plane. The buffer's row 0 corresponds
//!    to global row `BODY_TOP - HALO_TOP` (which is what
//!    `body_offset_y` encodes).
//! 4. Dispatch the **strip-aware** kernel on the strip buffer.
//! 5. Assert the strip kernel's body rows match the legacy
//!    kernel's reference body rows within `STRIP_PARITY_TOL`.
//!
//! Halo sizing: the V-blur taps reach `±6` rows, so a `HALO_TOP =
//! HALO_BOT = 6` row halo is the **minimum** for body rows to read
//! within-buffer. Tests use `HALO = 6` to make the strip footprint
//! as tight as possible; the strip-aware kernel reflects past the
//! buffer edge correctly via `logical_h = 64` reflection target.
//!
//! Tolerance: `STRIP_PARITY_TOL = 5e-6` absolute f32. The kernel
//! math is the same multiply-add sequence in both cases (single
//! pass per output pixel, no atomics or workgroup reductions), so
//! deterministic-bit-exact is the expectation — the tolerance is
//! a safety margin against any unforeseen f32 add-reorder LLVM
//! might emit at the kernel codegen layer.
//!
//! Skips when no compatible cubecl runtime feature is enabled.

#![cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]

use cubecl::Runtime;
use cubecl::prelude::*;
use cvvdp_gpu::kernels::masking::{
    pu_blur_h_3ch_kernel, pu_blur_h_3ch_strip_aware_kernel, pu_blur_v_3ch_scaled_kernel,
    pu_blur_v_3ch_scaled_strip_aware_kernel,
};

#[path = "common/mod.rs"]
mod common;

use common::Backend;

/// Absolute f32 tolerance for the parity assertion. See module
/// docstring — the strip-aware kernel performs the same arithmetic
/// in the same order as the legacy kernel; this tolerance is a
/// safety margin against backend-codegen-level add-reorder.
const STRIP_PARITY_TOL: f32 = 5e-6;

/// Build a deterministic 64×64×3 synthetic input where each
/// channel has a distinct spatial signature — so a per-channel
/// cross-wiring bug in the strip kernel would mismatch.
fn synth_3ch_64x64() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (w, h) = (64usize, 64usize);
    let n = w * h;
    let mut a = Vec::with_capacity(n);
    let mut rg = Vec::with_capacity(n);
    let mut vy = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i % w) as f32;
        let y = (i / w) as f32;
        a.push((x * 0.31 + y * 0.47).sin() * 3.5 + 2.0);
        rg.push((x * 0.17 - y * 0.29).cos() * 2.25 - 0.5);
        vy.push((x - y).abs() * 0.04 + 1.25);
    }
    (a, rg, vy)
}

#[test]
fn pu_blur_v_3ch_scaled_strip_aware_matches_full_at_64x64() {
    // GPU parity: dispatch the legacy V kernel on a 64×64 plane,
    // dispatch the strip-aware V kernel on a halo-padded strip
    // covering global rows [BODY_TOP..BODY_BOT), and assert the
    // strip kernel's body rows reproduce the legacy kernel's
    // [BODY_TOP..BODY_BOT) output rows bit-exact (within
    // STRIP_PARITY_TOL).
    let client = Backend::client(&Default::default());

    const W: usize = 64;
    const H: usize = 64;
    const HALO: usize = 6;
    const BODY_TOP: usize = 16;
    const BODY_BOT: usize = 48;
    const BODY_H: usize = BODY_BOT - BODY_TOP;
    const STRIP_H: usize = HALO + BODY_H + HALO; // 6 + 32 + 6 = 44
    const STRIP_TOP: usize = BODY_TOP - HALO; // global y of strip-buffer row 0
    const N_FULL: usize = W * H;
    const N_STRIP: usize = W * STRIP_H;

    let pu_scale = 10.0_f32.powf(-0.7955); // canonical MASK_C order

    let (src_a, src_rg, src_vy) = synth_3ch_64x64();

    // --- Legacy full-image dispatch ---
    let src_a_h = client.create_from_slice(f32::as_bytes(&src_a));
    let src_rg_h = client.create_from_slice(f32::as_bytes(&src_rg));
    let src_vy_h = client.create_from_slice(f32::as_bytes(&src_vy));
    let dst_a_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; N_FULL]));
    let dst_rg_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; N_FULL]));
    let dst_vy_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; N_FULL]));

    let cube_dim = CubeDim::new_1d(64);
    let cube_count_full = CubeCount::Static((N_FULL as u32).div_ceil(64), 1, 1);
    unsafe {
        pu_blur_v_3ch_scaled_kernel::launch::<Backend>(
            &client,
            cube_count_full,
            cube_dim,
            ArrayArg::from_raw_parts(src_a_h, N_FULL),
            ArrayArg::from_raw_parts(src_rg_h, N_FULL),
            ArrayArg::from_raw_parts(src_vy_h, N_FULL),
            ArrayArg::from_raw_parts(dst_a_h.clone(), N_FULL),
            ArrayArg::from_raw_parts(dst_rg_h.clone(), N_FULL),
            ArrayArg::from_raw_parts(dst_vy_h.clone(), N_FULL),
            pu_scale,
            W as u32,
            H as u32,
        );
    }
    let legacy_a_bytes = client.read_one(dst_a_h).expect("read legacy A");
    let legacy_rg_bytes = client.read_one(dst_rg_h).expect("read legacy RG");
    let legacy_vy_bytes = client.read_one(dst_vy_h).expect("read legacy VY");
    let legacy_a: &[f32] = f32::from_bytes(&legacy_a_bytes);
    let legacy_rg: &[f32] = f32::from_bytes(&legacy_rg_bytes);
    let legacy_vy: &[f32] = f32::from_bytes(&legacy_vy_bytes);

    // --- Strip dispatch (rows [STRIP_TOP..STRIP_TOP+STRIP_H) of full) ---
    let strip_a: Vec<f32> = src_a[STRIP_TOP * W..(STRIP_TOP + STRIP_H) * W].to_vec();
    let strip_rg: Vec<f32> = src_rg[STRIP_TOP * W..(STRIP_TOP + STRIP_H) * W].to_vec();
    let strip_vy: Vec<f32> = src_vy[STRIP_TOP * W..(STRIP_TOP + STRIP_H) * W].to_vec();

    let strip_src_a = client.create_from_slice(f32::as_bytes(&strip_a));
    let strip_src_rg = client.create_from_slice(f32::as_bytes(&strip_rg));
    let strip_src_vy = client.create_from_slice(f32::as_bytes(&strip_vy));
    let strip_dst_a = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; N_STRIP]));
    let strip_dst_rg = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; N_STRIP]));
    let strip_dst_vy = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; N_STRIP]));

    let cube_count_strip = CubeCount::Static((N_STRIP as u32).div_ceil(64), 1, 1);
    unsafe {
        pu_blur_v_3ch_scaled_strip_aware_kernel::launch::<Backend>(
            &client,
            cube_count_strip,
            cube_dim,
            ArrayArg::from_raw_parts(strip_src_a, N_STRIP),
            ArrayArg::from_raw_parts(strip_src_rg, N_STRIP),
            ArrayArg::from_raw_parts(strip_src_vy, N_STRIP),
            ArrayArg::from_raw_parts(strip_dst_a.clone(), N_STRIP),
            ArrayArg::from_raw_parts(strip_dst_rg.clone(), N_STRIP),
            ArrayArg::from_raw_parts(strip_dst_vy.clone(), N_STRIP),
            pu_scale,
            W as u32,
            STRIP_H as u32,
            STRIP_TOP as u32, // body_offset_y = global y of buffer row 0
            H as u32,         // logical_h = full image height (reflection target)
        );
    }
    let strip_a_bytes = client.read_one(strip_dst_a).expect("read strip A");
    let strip_rg_bytes = client.read_one(strip_dst_rg).expect("read strip RG");
    let strip_vy_bytes = client.read_one(strip_dst_vy).expect("read strip VY");
    let strip_a_out: &[f32] = f32::from_bytes(&strip_a_bytes);
    let strip_rg_out: &[f32] = f32::from_bytes(&strip_rg_bytes);
    let strip_vy_out: &[f32] = f32::from_bytes(&strip_vy_bytes);

    // --- Parity: strip-buffer body rows [HALO..HALO+BODY_H) must
    //     match legacy rows [BODY_TOP..BODY_BOT). ---
    let mut max_err = 0.0_f32;
    let mut max_at: (usize, usize, &'static str) = (0, 0, "?");
    for by in 0..BODY_H {
        let strip_row = HALO + by;
        let legacy_row = BODY_TOP + by;
        for x in 0..W {
            for (strip_plane, legacy_plane, tag) in [
                (strip_a_out, legacy_a, "A"),
                (strip_rg_out, legacy_rg, "RG"),
                (strip_vy_out, legacy_vy, "VY"),
            ] {
                let s = strip_plane[strip_row * W + x];
                let l = legacy_plane[legacy_row * W + x];
                let err = (s - l).abs();
                if err > max_err {
                    max_err = err;
                    max_at = (by, x, tag);
                }
                assert!(
                    err <= STRIP_PARITY_TOL,
                    "V-blur strip parity miss at body_row={by} (strip_row={strip_row}, legacy_row={legacy_row}) x={x} ch={tag}: strip={s} legacy={l} err={err:.3e} (tol={STRIP_PARITY_TOL:.3e})"
                );
            }
        }
    }
    eprintln!(
        "V-blur strip parity: max_err={max_err:.3e} at body_row={} x={} ch={} (tol={STRIP_PARITY_TOL:.3e})",
        max_at.0, max_at.1, max_at.2
    );
}

#[test]
fn pu_blur_h_3ch_strip_aware_matches_full_at_64x64() {
    // GPU parity: H-blur is X-axis only — strip-aware version
    // ignores body_offset_y/logical_h (they're accepted purely for
    // API uniformity with the V-blur sibling). At identical
    // (w, h) the output should reproduce the legacy kernel
    // bit-exact for the BODY rows of a halo-padded strip buffer.
    //
    // Why test it at all: locking in the API-uniformity contract
    // — any change that smuggles body_offset_y/logical_h into the
    // X-axis math would break this gate.
    let client = Backend::client(&Default::default());

    const W: usize = 64;
    const H: usize = 64;
    const HALO: usize = 6;
    const BODY_TOP: usize = 16;
    const BODY_BOT: usize = 48;
    const BODY_H: usize = BODY_BOT - BODY_TOP;
    const STRIP_H: usize = HALO + BODY_H + HALO;
    const STRIP_TOP: usize = BODY_TOP - HALO;
    const N_FULL: usize = W * H;
    const N_STRIP: usize = W * STRIP_H;

    let (src_a, src_rg, src_vy) = synth_3ch_64x64();

    // --- Legacy full-image dispatch ---
    let src_a_h = client.create_from_slice(f32::as_bytes(&src_a));
    let src_rg_h = client.create_from_slice(f32::as_bytes(&src_rg));
    let src_vy_h = client.create_from_slice(f32::as_bytes(&src_vy));
    let dst_a_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; N_FULL]));
    let dst_rg_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; N_FULL]));
    let dst_vy_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; N_FULL]));

    let cube_dim = CubeDim::new_1d(64);
    let cube_count_full = CubeCount::Static((N_FULL as u32).div_ceil(64), 1, 1);
    unsafe {
        pu_blur_h_3ch_kernel::launch::<Backend>(
            &client,
            cube_count_full,
            cube_dim,
            ArrayArg::from_raw_parts(src_a_h, N_FULL),
            ArrayArg::from_raw_parts(src_rg_h, N_FULL),
            ArrayArg::from_raw_parts(src_vy_h, N_FULL),
            ArrayArg::from_raw_parts(dst_a_h.clone(), N_FULL),
            ArrayArg::from_raw_parts(dst_rg_h.clone(), N_FULL),
            ArrayArg::from_raw_parts(dst_vy_h.clone(), N_FULL),
            W as u32,
            H as u32,
        );
    }
    let legacy_a_bytes = client.read_one(dst_a_h).expect("read legacy A");
    let legacy_rg_bytes = client.read_one(dst_rg_h).expect("read legacy RG");
    let legacy_vy_bytes = client.read_one(dst_vy_h).expect("read legacy VY");
    let legacy_a: &[f32] = f32::from_bytes(&legacy_a_bytes);
    let legacy_rg: &[f32] = f32::from_bytes(&legacy_rg_bytes);
    let legacy_vy: &[f32] = f32::from_bytes(&legacy_vy_bytes);

    // --- Strip dispatch ---
    let strip_a: Vec<f32> = src_a[STRIP_TOP * W..(STRIP_TOP + STRIP_H) * W].to_vec();
    let strip_rg: Vec<f32> = src_rg[STRIP_TOP * W..(STRIP_TOP + STRIP_H) * W].to_vec();
    let strip_vy: Vec<f32> = src_vy[STRIP_TOP * W..(STRIP_TOP + STRIP_H) * W].to_vec();

    let strip_src_a = client.create_from_slice(f32::as_bytes(&strip_a));
    let strip_src_rg = client.create_from_slice(f32::as_bytes(&strip_rg));
    let strip_src_vy = client.create_from_slice(f32::as_bytes(&strip_vy));
    let strip_dst_a = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; N_STRIP]));
    let strip_dst_rg = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; N_STRIP]));
    let strip_dst_vy = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; N_STRIP]));

    let cube_count_strip = CubeCount::Static((N_STRIP as u32).div_ceil(64), 1, 1);
    unsafe {
        pu_blur_h_3ch_strip_aware_kernel::launch::<Backend>(
            &client,
            cube_count_strip,
            cube_dim,
            ArrayArg::from_raw_parts(strip_src_a, N_STRIP),
            ArrayArg::from_raw_parts(strip_src_rg, N_STRIP),
            ArrayArg::from_raw_parts(strip_src_vy, N_STRIP),
            ArrayArg::from_raw_parts(strip_dst_a.clone(), N_STRIP),
            ArrayArg::from_raw_parts(strip_dst_rg.clone(), N_STRIP),
            ArrayArg::from_raw_parts(strip_dst_vy.clone(), N_STRIP),
            W as u32,
            STRIP_H as u32,
            STRIP_TOP as u32, // body_offset_y — unused but threaded through
            H as u32,         // logical_h — unused but threaded through
        );
    }
    let strip_a_bytes = client.read_one(strip_dst_a).expect("read strip A");
    let strip_rg_bytes = client.read_one(strip_dst_rg).expect("read strip RG");
    let strip_vy_bytes = client.read_one(strip_dst_vy).expect("read strip VY");
    let strip_a_out: &[f32] = f32::from_bytes(&strip_a_bytes);
    let strip_rg_out: &[f32] = f32::from_bytes(&strip_rg_bytes);
    let strip_vy_out: &[f32] = f32::from_bytes(&strip_vy_bytes);

    // --- Parity ---
    let mut max_err = 0.0_f32;
    let mut max_at: (usize, usize, &'static str) = (0, 0, "?");
    for by in 0..BODY_H {
        let strip_row = HALO + by;
        let legacy_row = BODY_TOP + by;
        for x in 0..W {
            for (strip_plane, legacy_plane, tag) in [
                (strip_a_out, legacy_a, "A"),
                (strip_rg_out, legacy_rg, "RG"),
                (strip_vy_out, legacy_vy, "VY"),
            ] {
                let s = strip_plane[strip_row * W + x];
                let l = legacy_plane[legacy_row * W + x];
                let err = (s - l).abs();
                if err > max_err {
                    max_err = err;
                    max_at = (by, x, tag);
                }
                assert!(
                    err <= STRIP_PARITY_TOL,
                    "H-blur strip parity miss at body_row={by} (strip_row={strip_row}, legacy_row={legacy_row}) x={x} ch={tag}: strip={s} legacy={l} err={err:.3e} (tol={STRIP_PARITY_TOL:.3e})"
                );
            }
        }
    }
    eprintln!(
        "H-blur strip parity: max_err={max_err:.3e} at body_row={} x={} ch={} (tol={STRIP_PARITY_TOL:.3e})",
        max_at.0, max_at.1, max_at.2
    );
}

#[test]
fn pu_blur_v_3ch_scaled_strip_aware_full_image_dispatch_matches_legacy() {
    // Degenerate-strip parity: when body_offset_y = 0 and
    // logical_h = h, the strip-aware kernel reduces to the legacy
    // kernel by construction (reflection target collapses to the
    // buffer height, offset is a no-op). This test pins that
    // identity at every pixel — a future refactor that introduces
    // an arithmetic asymmetry in the offset-zero case would trip
    // this gate.
    let client = Backend::client(&Default::default());

    const W: usize = 64;
    const H: usize = 64;
    const N: usize = W * H;
    let pu_scale = 10.0_f32.powf(-0.7955);

    let (src_a, src_rg, src_vy) = synth_3ch_64x64();

    let mk_legacy = || {
        let src_a_h = client.create_from_slice(f32::as_bytes(&src_a));
        let src_rg_h = client.create_from_slice(f32::as_bytes(&src_rg));
        let src_vy_h = client.create_from_slice(f32::as_bytes(&src_vy));
        let dst_a_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; N]));
        let dst_rg_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; N]));
        let dst_vy_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; N]));
        let cube_dim = CubeDim::new_1d(64);
        let cube_count = CubeCount::Static((N as u32).div_ceil(64), 1, 1);
        unsafe {
            pu_blur_v_3ch_scaled_kernel::launch::<Backend>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(src_a_h, N),
                ArrayArg::from_raw_parts(src_rg_h, N),
                ArrayArg::from_raw_parts(src_vy_h, N),
                ArrayArg::from_raw_parts(dst_a_h.clone(), N),
                ArrayArg::from_raw_parts(dst_rg_h.clone(), N),
                ArrayArg::from_raw_parts(dst_vy_h.clone(), N),
                pu_scale,
                W as u32,
                H as u32,
            );
        }
        (dst_a_h, dst_rg_h, dst_vy_h)
    };
    let (la, lrg, lvy) = mk_legacy();

    let src_a_h = client.create_from_slice(f32::as_bytes(&src_a));
    let src_rg_h = client.create_from_slice(f32::as_bytes(&src_rg));
    let src_vy_h = client.create_from_slice(f32::as_bytes(&src_vy));
    let dst_a_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; N]));
    let dst_rg_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; N]));
    let dst_vy_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; N]));
    let cube_dim = CubeDim::new_1d(64);
    let cube_count = CubeCount::Static((N as u32).div_ceil(64), 1, 1);
    unsafe {
        pu_blur_v_3ch_scaled_strip_aware_kernel::launch::<Backend>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(src_a_h, N),
            ArrayArg::from_raw_parts(src_rg_h, N),
            ArrayArg::from_raw_parts(src_vy_h, N),
            ArrayArg::from_raw_parts(dst_a_h.clone(), N),
            ArrayArg::from_raw_parts(dst_rg_h.clone(), N),
            ArrayArg::from_raw_parts(dst_vy_h.clone(), N),
            pu_scale,
            W as u32,
            H as u32,
            0,        // body_offset_y = 0 (degenerate strip = full image)
            H as u32, // logical_h = strip buffer height
        );
    }

    let lab = client.read_one(la).expect("read legacy A");
    let lrgb = client.read_one(lrg).expect("read legacy RG");
    let lvyb = client.read_one(lvy).expect("read legacy VY");
    let sab = client.read_one(dst_a_h).expect("read strip A");
    let srgb = client.read_one(dst_rg_h).expect("read strip RG");
    let svyb = client.read_one(dst_vy_h).expect("read strip VY");
    let la_p: &[f32] = f32::from_bytes(&lab);
    let lrg_p: &[f32] = f32::from_bytes(&lrgb);
    let lvy_p: &[f32] = f32::from_bytes(&lvyb);
    let sa_p: &[f32] = f32::from_bytes(&sab);
    let srg_p: &[f32] = f32::from_bytes(&srgb);
    let svy_p: &[f32] = f32::from_bytes(&svyb);

    let mut max_err = 0.0_f32;
    for i in 0..N {
        for (s, l, tag) in [
            (sa_p[i], la_p[i], "A"),
            (srg_p[i], lrg_p[i], "RG"),
            (svy_p[i], lvy_p[i], "VY"),
        ] {
            let err = (s - l).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err <= STRIP_PARITY_TOL,
                "V-blur degenerate-strip mismatch at i={i} ch={tag}: strip={s} legacy={l} err={err:.3e}"
            );
        }
    }
    eprintln!("V-blur degenerate-strip max_err={max_err:.3e}");
}
