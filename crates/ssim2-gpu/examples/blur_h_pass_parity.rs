//! Parity gate for the new `blur_h_pass_kernel`.
//!
//! Verifies that `v_pass + h_pass` produces output bit-equivalent to
//! the existing `v_pass + transpose + v_pass` chain. The IIR is
//! separable so the two chains compute the same mathematical 2D
//! Gaussian — just with different storage orderings and a different
//! number of launches.
//!
//! Tolerance is **0** (bit-identical). Both paths use the same
//! `blur_pass_kernel` for the first v-pass; the only difference is
//! whether the second pass is `transpose + v_pass` or `h_pass`. The
//! IIR coefficients are identical, the boundary handling is identical
//! (state seeded to zero, zero-padding outside the row/column), and
//! the per-pixel summation order is identical (the IIR adds taps in
//! the same `prev_1 + prev_3 + prev_5` order in both kernels). So the
//! emitted pixel values must match bit-for-bit.

#![cfg(feature = "cuda")]

use cubecl::prelude::*;
use ssim2_gpu::kernels::{blur, transpose};

type Backend = cubecl::cuda::CudaRuntime;

fn run_pair(width: u32, height: u32, src: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = (width as usize) * (height as usize);
    assert_eq!(src.len(), n);

    let client = Backend::client(&Default::default());
    let src_h = client.create_from_slice(f32::as_bytes(src));

    // ── Path A: v + transpose + v ──
    let v_a = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let t_a = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let out_a = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));

    let blur_cubes_w = (width + blur::BLOCK_WIDTH - 1) / blur::BLOCK_WIDTH;
    let blur_cubes_h = (height + blur::BLOCK_WIDTH - 1) / blur::BLOCK_WIDTH;
    let tx_cubes_w = width.div_ceil(transpose::TILE_DIM).max(1);
    let tx_cubes_h = height.div_ceil(transpose::TILE_DIM).max(1);

    unsafe {
        blur::blur_pass_kernel::launch_unchecked::<Backend>(
            &client,
            CubeCount::Static(blur_cubes_w.max(1), 1, 1),
            CubeDim::new_1d(blur::BLOCK_WIDTH),
            ArrayArg::from_raw_parts(src_h.clone(), n),
            ArrayArg::from_raw_parts(v_a.clone(), n),
            width,
            height,
        );
        transpose::transpose_kernel::launch_unchecked::<Backend>(
            &client,
            CubeCount::Static(tx_cubes_w, tx_cubes_h, 1),
            CubeDim::new_2d(transpose::TPB_X, transpose::TPB_Y),
            ArrayArg::from_raw_parts(v_a.clone(), n),
            ArrayArg::from_raw_parts(t_a.clone(), n),
            width,
            height,
        );
        blur::blur_pass_kernel::launch_unchecked::<Backend>(
            &client,
            CubeCount::Static(blur_cubes_h.max(1), 1, 1),
            CubeDim::new_1d(blur::BLOCK_WIDTH),
            ArrayArg::from_raw_parts(t_a, n),
            ArrayArg::from_raw_parts(out_a.clone(), n),
            height,
            width,
        );
    }
    // out_a is in TRANSPOSED orientation (height × width).
    let raw_a = f32::from_bytes(&client.read_one(out_a).unwrap()).to_vec();
    let mut a = vec![0.0_f32; n];
    let w_us = width as usize;
    let h_us = height as usize;
    for y in 0..h_us {
        for x in 0..w_us {
            a[y * w_us + x] = raw_a[x * h_us + y];
        }
    }

    // ── Path B: v + h ──
    let v_b = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let out_b = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    unsafe {
        blur::blur_pass_kernel::launch_unchecked::<Backend>(
            &client,
            CubeCount::Static(blur_cubes_w.max(1), 1, 1),
            CubeDim::new_1d(blur::BLOCK_WIDTH),
            ArrayArg::from_raw_parts(src_h, n),
            ArrayArg::from_raw_parts(v_b.clone(), n),
            width,
            height,
        );
        blur::blur_h_pass_kernel::launch_unchecked::<Backend>(
            &client,
            CubeCount::Static(blur_cubes_h.max(1), 1, 1),
            CubeDim::new_1d(blur::BLOCK_WIDTH),
            ArrayArg::from_raw_parts(v_b, n),
            ArrayArg::from_raw_parts(out_b.clone(), n),
            width,
            height,
        );
    }
    let b = f32::from_bytes(&client.read_one(out_b).unwrap()).to_vec();
    // b is already in untransposed orientation.

    (a, b)
}

fn report(name: &str, width: u32, height: u32, src: &[f32]) -> bool {
    let (a, b) = run_pair(width, height, src);
    let n = a.len();
    let mut max_abs = 0.0_f32;
    let mut max_idx = 0;
    let mut mismatches = 0;
    for i in 0..n {
        let d = (a[i] - b[i]).abs();
        if d > 0.0 {
            mismatches += 1;
        }
        if d > max_abs {
            max_abs = d;
            max_idx = i;
        }
    }
    let pass = max_abs == 0.0;
    println!(
        "[{}] {}×{}  max|a-b|={:.3e} mismatches={}  ({})",
        if pass { "ok " } else { "FAIL" },
        width,
        height,
        max_abs,
        mismatches,
        name,
    );
    if !pass {
        println!(
            "  worst-case at idx={}  a[{}]={}  b[{}]={}",
            max_idx, max_idx, a[max_idx], max_idx, b[max_idx],
        );
    }
    pass
}

fn build(width: u32, height: u32, mut f: impl FnMut(usize, usize) -> f32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let mut v = vec![0.0_f32; w * h];
    for y in 0..h {
        for x in 0..w {
            v[y * w + x] = f(x, y);
        }
    }
    v
}

fn main() {
    let mut all_pass = true;

    all_pass &= report("constant 0.5", 32, 32, &build(32, 32, |_, _| 0.5));
    all_pass &= report(
        "delta at (8, 12)",
        32,
        32,
        &build(32, 32, |x, y| if x == 8 && y == 12 { 1.0 } else { 0.0 }),
    );
    all_pass &= report("x-gradient", 64, 32, &build(64, 32, |x, _| x as f32 / 63.0));
    all_pass &= report("y-gradient", 32, 64, &build(32, 64, |_, y| y as f32 / 63.0));

    let mut state = 0x5EEDu32;
    let img256 = build(256, 256, |_, _| {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32) / (u32::MAX as f32)
    });
    all_pass &= report("random 256x256", 256, 256, &img256);

    let mut state2 = 0xCAFEu32;
    let img1k = build(1024, 768, |_, _| {
        state2 ^= state2 << 13;
        state2 ^= state2 >> 17;
        state2 ^= state2 << 5;
        (state2 as f32) / (u32::MAX as f32)
    });
    all_pass &= report("random 1024x768", 1024, 768, &img1k);

    let mut state3 = 0xBEEFu32;
    let img2k = build(2048, 2048, |_, _| {
        state3 ^= state3 << 13;
        state3 ^= state3 >> 17;
        state3 ^= state3 << 5;
        (state3 as f32) / (u32::MAX as f32)
    });
    all_pass &= report("random 2048x2048", 2048, 2048, &img2k);

    if !all_pass {
        eprintln!("FAIL: h-pass IIR does not match transpose+v-pass bit-identically");
        std::process::exit(1);
    }
    println!("All h-pass parity cases pass (bit-identical to v+t+v)");
}
