//! Per-thread-accumulator cov_accum kernels — partials version.
//!
//! Each thread maintains a 9x9 (no-parent) or 10x10 (with-parent) local
//! f32 register file, accumulates outer products over its grid-strided
//! pixel range, and writes its 81 or 100 partials to a global
//! `partials` array. The layout is transposed so that thread `tid` writes
//! cell `i` to `partials[i * n_threads + tid]`, giving coalesced writes
//! on GPU and contiguous per-cell strips for the finalize reducer.
//!
//! This replaces an earlier `Atomic<f32>::fetch_add` accumulator: the
//! atomic kernel hit a hard wall on `cubecl-cpu` (no `atomic<f32>`
//! lowering in the MLIR backend, fall-through panics silently, all
//! cells stay at 0). Per-thread partials + a separate finalize kernel
//! works on every backend.
//!
//! Layout:
//! - `partials[i * n_threads + tid]`  — partial for cell `i` from thread `tid`
//! - `n_threads = cube_count.x * cube_dim.x` for the cov launch
//! - `i ∈ 0..81` for no-parent, `i ∈ 0..100` for with-parent
//!
//! Finalize via `cov_finalize_kernel` in this same module.

use cubecl::prelude::*;

/// `py_start` / `py_end` define an inclusive / exclusive iw row range
/// to sum over. Whole-image: `(0, h − 2)`. Strip mode passes the
/// strip's body iw row range so per-strip cov contributions don't
/// overlap across strips.
#[cube(launch_unchecked)]
pub fn cov_accum_no_parent_kernel(
    lp: &Array<f32>,
    partials: &mut Array<f32>,
    h: u32,
    w: u32,
    n_threads: u32,
    py_start: u32,
    py_end: u32,
) {
    let tid = ABSOLUTE_POS;
    let stride = ((CUBE_COUNT_X * CUBE_COUNT_Y * CUBE_COUNT_Z) as usize) * (CUBE_DIM_X as usize);
    let _ = h;
    let nblh = w - 2;
    let py_lo = py_start as usize;
    let py_hi = py_end as usize;
    let rows = py_hi - py_lo;
    let nexp = rows * (nblh as usize);
    let w_us = w as usize;
    let n_threads_us = n_threads as usize;

    let mut a00 = 0.0_f32;
    let mut a01 = 0.0_f32;
    let mut a02 = 0.0_f32;
    let mut a03 = 0.0_f32;
    let mut a04 = 0.0_f32;
    let mut a05 = 0.0_f32;
    let mut a06 = 0.0_f32;
    let mut a07 = 0.0_f32;
    let mut a08 = 0.0_f32;
    let mut a10 = 0.0_f32;
    let mut a11 = 0.0_f32;
    let mut a12 = 0.0_f32;
    let mut a13 = 0.0_f32;
    let mut a14 = 0.0_f32;
    let mut a15 = 0.0_f32;
    let mut a16 = 0.0_f32;
    let mut a17 = 0.0_f32;
    let mut a18 = 0.0_f32;
    let mut a20 = 0.0_f32;
    let mut a21 = 0.0_f32;
    let mut a22 = 0.0_f32;
    let mut a23 = 0.0_f32;
    let mut a24 = 0.0_f32;
    let mut a25 = 0.0_f32;
    let mut a26 = 0.0_f32;
    let mut a27 = 0.0_f32;
    let mut a28 = 0.0_f32;
    let mut a30 = 0.0_f32;
    let mut a31 = 0.0_f32;
    let mut a32 = 0.0_f32;
    let mut a33 = 0.0_f32;
    let mut a34 = 0.0_f32;
    let mut a35 = 0.0_f32;
    let mut a36 = 0.0_f32;
    let mut a37 = 0.0_f32;
    let mut a38 = 0.0_f32;
    let mut a40 = 0.0_f32;
    let mut a41 = 0.0_f32;
    let mut a42 = 0.0_f32;
    let mut a43 = 0.0_f32;
    let mut a44 = 0.0_f32;
    let mut a45 = 0.0_f32;
    let mut a46 = 0.0_f32;
    let mut a47 = 0.0_f32;
    let mut a48 = 0.0_f32;
    let mut a50 = 0.0_f32;
    let mut a51 = 0.0_f32;
    let mut a52 = 0.0_f32;
    let mut a53 = 0.0_f32;
    let mut a54 = 0.0_f32;
    let mut a55 = 0.0_f32;
    let mut a56 = 0.0_f32;
    let mut a57 = 0.0_f32;
    let mut a58 = 0.0_f32;
    let mut a60 = 0.0_f32;
    let mut a61 = 0.0_f32;
    let mut a62 = 0.0_f32;
    let mut a63 = 0.0_f32;
    let mut a64 = 0.0_f32;
    let mut a65 = 0.0_f32;
    let mut a66 = 0.0_f32;
    let mut a67 = 0.0_f32;
    let mut a68 = 0.0_f32;
    let mut a70 = 0.0_f32;
    let mut a71 = 0.0_f32;
    let mut a72 = 0.0_f32;
    let mut a73 = 0.0_f32;
    let mut a74 = 0.0_f32;
    let mut a75 = 0.0_f32;
    let mut a76 = 0.0_f32;
    let mut a77 = 0.0_f32;
    let mut a78 = 0.0_f32;
    let mut a80 = 0.0_f32;
    let mut a81 = 0.0_f32;
    let mut a82 = 0.0_f32;
    let mut a83 = 0.0_f32;
    let mut a84 = 0.0_f32;
    let mut a85 = 0.0_f32;
    let mut a86 = 0.0_f32;
    let mut a87 = 0.0_f32;
    let mut a88 = 0.0_f32;

    let mut p = tid;
    while p < nexp {
        let local_py = (p as u32) / nblh;
        let px = (p as u32) - local_py * nblh;
        let py_us = (local_py as usize) + py_lo;
        let px_us = px as usize;
        let v0 = lp[(py_us + 2) * w_us + (px_us + 2)];
        let v1 = lp[(py_us + 2) * w_us + (px_us + 1)];
        let v2 = lp[(py_us + 2) * w_us + px_us];
        let v3 = lp[(py_us + 1) * w_us + (px_us + 2)];
        let v4 = lp[(py_us + 1) * w_us + (px_us + 1)];
        let v5 = lp[(py_us + 1) * w_us + px_us];
        let v6 = lp[py_us * w_us + (px_us + 2)];
        let v7 = lp[py_us * w_us + (px_us + 1)];
        let v8 = lp[py_us * w_us + px_us];
        a00 += v0 * v0;
        a01 += v0 * v1;
        a02 += v0 * v2;
        a03 += v0 * v3;
        a04 += v0 * v4;
        a05 += v0 * v5;
        a06 += v0 * v6;
        a07 += v0 * v7;
        a08 += v0 * v8;
        a10 += v1 * v0;
        a11 += v1 * v1;
        a12 += v1 * v2;
        a13 += v1 * v3;
        a14 += v1 * v4;
        a15 += v1 * v5;
        a16 += v1 * v6;
        a17 += v1 * v7;
        a18 += v1 * v8;
        a20 += v2 * v0;
        a21 += v2 * v1;
        a22 += v2 * v2;
        a23 += v2 * v3;
        a24 += v2 * v4;
        a25 += v2 * v5;
        a26 += v2 * v6;
        a27 += v2 * v7;
        a28 += v2 * v8;
        a30 += v3 * v0;
        a31 += v3 * v1;
        a32 += v3 * v2;
        a33 += v3 * v3;
        a34 += v3 * v4;
        a35 += v3 * v5;
        a36 += v3 * v6;
        a37 += v3 * v7;
        a38 += v3 * v8;
        a40 += v4 * v0;
        a41 += v4 * v1;
        a42 += v4 * v2;
        a43 += v4 * v3;
        a44 += v4 * v4;
        a45 += v4 * v5;
        a46 += v4 * v6;
        a47 += v4 * v7;
        a48 += v4 * v8;
        a50 += v5 * v0;
        a51 += v5 * v1;
        a52 += v5 * v2;
        a53 += v5 * v3;
        a54 += v5 * v4;
        a55 += v5 * v5;
        a56 += v5 * v6;
        a57 += v5 * v7;
        a58 += v5 * v8;
        a60 += v6 * v0;
        a61 += v6 * v1;
        a62 += v6 * v2;
        a63 += v6 * v3;
        a64 += v6 * v4;
        a65 += v6 * v5;
        a66 += v6 * v6;
        a67 += v6 * v7;
        a68 += v6 * v8;
        a70 += v7 * v0;
        a71 += v7 * v1;
        a72 += v7 * v2;
        a73 += v7 * v3;
        a74 += v7 * v4;
        a75 += v7 * v5;
        a76 += v7 * v6;
        a77 += v7 * v7;
        a78 += v7 * v8;
        a80 += v8 * v0;
        a81 += v8 * v1;
        a82 += v8 * v2;
        a83 += v8 * v3;
        a84 += v8 * v4;
        a85 += v8 * v5;
        a86 += v8 * v6;
        a87 += v8 * v7;
        a88 += v8 * v8;
        p += stride;
    }

    // Layout: partials[i * n_threads + tid] = a_i
    // 81 cells (9×9), written contiguously per i to match
    // cov_finalize_kernel's grid-strided reducer.
    partials[0 * n_threads_us + tid] = a00;
    partials[1 * n_threads_us + tid] = a01;
    partials[2 * n_threads_us + tid] = a02;
    partials[3 * n_threads_us + tid] = a03;
    partials[4 * n_threads_us + tid] = a04;
    partials[5 * n_threads_us + tid] = a05;
    partials[6 * n_threads_us + tid] = a06;
    partials[7 * n_threads_us + tid] = a07;
    partials[8 * n_threads_us + tid] = a08;
    partials[9 * n_threads_us + tid] = a10;
    partials[10 * n_threads_us + tid] = a11;
    partials[11 * n_threads_us + tid] = a12;
    partials[12 * n_threads_us + tid] = a13;
    partials[13 * n_threads_us + tid] = a14;
    partials[14 * n_threads_us + tid] = a15;
    partials[15 * n_threads_us + tid] = a16;
    partials[16 * n_threads_us + tid] = a17;
    partials[17 * n_threads_us + tid] = a18;
    partials[18 * n_threads_us + tid] = a20;
    partials[19 * n_threads_us + tid] = a21;
    partials[20 * n_threads_us + tid] = a22;
    partials[21 * n_threads_us + tid] = a23;
    partials[22 * n_threads_us + tid] = a24;
    partials[23 * n_threads_us + tid] = a25;
    partials[24 * n_threads_us + tid] = a26;
    partials[25 * n_threads_us + tid] = a27;
    partials[26 * n_threads_us + tid] = a28;
    partials[27 * n_threads_us + tid] = a30;
    partials[28 * n_threads_us + tid] = a31;
    partials[29 * n_threads_us + tid] = a32;
    partials[30 * n_threads_us + tid] = a33;
    partials[31 * n_threads_us + tid] = a34;
    partials[32 * n_threads_us + tid] = a35;
    partials[33 * n_threads_us + tid] = a36;
    partials[34 * n_threads_us + tid] = a37;
    partials[35 * n_threads_us + tid] = a38;
    partials[36 * n_threads_us + tid] = a40;
    partials[37 * n_threads_us + tid] = a41;
    partials[38 * n_threads_us + tid] = a42;
    partials[39 * n_threads_us + tid] = a43;
    partials[40 * n_threads_us + tid] = a44;
    partials[41 * n_threads_us + tid] = a45;
    partials[42 * n_threads_us + tid] = a46;
    partials[43 * n_threads_us + tid] = a47;
    partials[44 * n_threads_us + tid] = a48;
    partials[45 * n_threads_us + tid] = a50;
    partials[46 * n_threads_us + tid] = a51;
    partials[47 * n_threads_us + tid] = a52;
    partials[48 * n_threads_us + tid] = a53;
    partials[49 * n_threads_us + tid] = a54;
    partials[50 * n_threads_us + tid] = a55;
    partials[51 * n_threads_us + tid] = a56;
    partials[52 * n_threads_us + tid] = a57;
    partials[53 * n_threads_us + tid] = a58;
    partials[54 * n_threads_us + tid] = a60;
    partials[55 * n_threads_us + tid] = a61;
    partials[56 * n_threads_us + tid] = a62;
    partials[57 * n_threads_us + tid] = a63;
    partials[58 * n_threads_us + tid] = a64;
    partials[59 * n_threads_us + tid] = a65;
    partials[60 * n_threads_us + tid] = a66;
    partials[61 * n_threads_us + tid] = a67;
    partials[62 * n_threads_us + tid] = a68;
    partials[63 * n_threads_us + tid] = a70;
    partials[64 * n_threads_us + tid] = a71;
    partials[65 * n_threads_us + tid] = a72;
    partials[66 * n_threads_us + tid] = a73;
    partials[67 * n_threads_us + tid] = a74;
    partials[68 * n_threads_us + tid] = a75;
    partials[69 * n_threads_us + tid] = a76;
    partials[70 * n_threads_us + tid] = a77;
    partials[71 * n_threads_us + tid] = a78;
    partials[72 * n_threads_us + tid] = a80;
    partials[73 * n_threads_us + tid] = a81;
    partials[74 * n_threads_us + tid] = a82;
    partials[75 * n_threads_us + tid] = a83;
    partials[76 * n_threads_us + tid] = a84;
    partials[77 * n_threads_us + tid] = a85;
    partials[78 * n_threads_us + tid] = a86;
    partials[79 * n_threads_us + tid] = a87;
    partials[80 * n_threads_us + tid] = a88;
}

/// `py_start` / `py_end` define an inclusive / exclusive iw row range
/// to sum over. Same semantics as [`cov_accum_no_parent_kernel`].
#[cube(launch_unchecked)]
pub fn cov_accum_with_parent_kernel(
    lp: &Array<f32>,
    parent: &Array<f32>,
    partials: &mut Array<f32>,
    h: u32,
    w: u32,
    n_threads: u32,
    py_start: u32,
    py_end: u32,
) {
    let tid = ABSOLUTE_POS;
    let stride = ((CUBE_COUNT_X * CUBE_COUNT_Y * CUBE_COUNT_Z) as usize) * (CUBE_DIM_X as usize);
    let _ = h;
    let nblh = w - 2;
    let py_lo = py_start as usize;
    let py_hi = py_end as usize;
    let rows = py_hi - py_lo;
    let nexp = rows * (nblh as usize);
    let w_us = w as usize;
    let n_threads_us = n_threads as usize;

    let mut a00 = 0.0_f32;
    let mut a01 = 0.0_f32;
    let mut a02 = 0.0_f32;
    let mut a03 = 0.0_f32;
    let mut a04 = 0.0_f32;
    let mut a05 = 0.0_f32;
    let mut a06 = 0.0_f32;
    let mut a07 = 0.0_f32;
    let mut a08 = 0.0_f32;
    let mut a09 = 0.0_f32;
    let mut a10 = 0.0_f32;
    let mut a11 = 0.0_f32;
    let mut a12 = 0.0_f32;
    let mut a13 = 0.0_f32;
    let mut a14 = 0.0_f32;
    let mut a15 = 0.0_f32;
    let mut a16 = 0.0_f32;
    let mut a17 = 0.0_f32;
    let mut a18 = 0.0_f32;
    let mut a19 = 0.0_f32;
    let mut a20 = 0.0_f32;
    let mut a21 = 0.0_f32;
    let mut a22 = 0.0_f32;
    let mut a23 = 0.0_f32;
    let mut a24 = 0.0_f32;
    let mut a25 = 0.0_f32;
    let mut a26 = 0.0_f32;
    let mut a27 = 0.0_f32;
    let mut a28 = 0.0_f32;
    let mut a29 = 0.0_f32;
    let mut a30 = 0.0_f32;
    let mut a31 = 0.0_f32;
    let mut a32 = 0.0_f32;
    let mut a33 = 0.0_f32;
    let mut a34 = 0.0_f32;
    let mut a35 = 0.0_f32;
    let mut a36 = 0.0_f32;
    let mut a37 = 0.0_f32;
    let mut a38 = 0.0_f32;
    let mut a39 = 0.0_f32;
    let mut a40 = 0.0_f32;
    let mut a41 = 0.0_f32;
    let mut a42 = 0.0_f32;
    let mut a43 = 0.0_f32;
    let mut a44 = 0.0_f32;
    let mut a45 = 0.0_f32;
    let mut a46 = 0.0_f32;
    let mut a47 = 0.0_f32;
    let mut a48 = 0.0_f32;
    let mut a49 = 0.0_f32;
    let mut a50 = 0.0_f32;
    let mut a51 = 0.0_f32;
    let mut a52 = 0.0_f32;
    let mut a53 = 0.0_f32;
    let mut a54 = 0.0_f32;
    let mut a55 = 0.0_f32;
    let mut a56 = 0.0_f32;
    let mut a57 = 0.0_f32;
    let mut a58 = 0.0_f32;
    let mut a59 = 0.0_f32;
    let mut a60 = 0.0_f32;
    let mut a61 = 0.0_f32;
    let mut a62 = 0.0_f32;
    let mut a63 = 0.0_f32;
    let mut a64 = 0.0_f32;
    let mut a65 = 0.0_f32;
    let mut a66 = 0.0_f32;
    let mut a67 = 0.0_f32;
    let mut a68 = 0.0_f32;
    let mut a69 = 0.0_f32;
    let mut a70 = 0.0_f32;
    let mut a71 = 0.0_f32;
    let mut a72 = 0.0_f32;
    let mut a73 = 0.0_f32;
    let mut a74 = 0.0_f32;
    let mut a75 = 0.0_f32;
    let mut a76 = 0.0_f32;
    let mut a77 = 0.0_f32;
    let mut a78 = 0.0_f32;
    let mut a79 = 0.0_f32;
    let mut a80 = 0.0_f32;
    let mut a81 = 0.0_f32;
    let mut a82 = 0.0_f32;
    let mut a83 = 0.0_f32;
    let mut a84 = 0.0_f32;
    let mut a85 = 0.0_f32;
    let mut a86 = 0.0_f32;
    let mut a87 = 0.0_f32;
    let mut a88 = 0.0_f32;
    let mut a89 = 0.0_f32;
    let mut a90 = 0.0_f32;
    let mut a91 = 0.0_f32;
    let mut a92 = 0.0_f32;
    let mut a93 = 0.0_f32;
    let mut a94 = 0.0_f32;
    let mut a95 = 0.0_f32;
    let mut a96 = 0.0_f32;
    let mut a97 = 0.0_f32;
    let mut a98 = 0.0_f32;
    let mut a99 = 0.0_f32;

    let mut p = tid;
    while p < nexp {
        let local_py = (p as u32) / nblh;
        let px = (p as u32) - local_py * nblh;
        let py_us = (local_py as usize) + py_lo;
        let px_us = px as usize;
        let v0 = lp[(py_us + 2) * w_us + (px_us + 2)];
        let v1 = lp[(py_us + 2) * w_us + (px_us + 1)];
        let v2 = lp[(py_us + 2) * w_us + px_us];
        let v3 = lp[(py_us + 1) * w_us + (px_us + 2)];
        let v4 = lp[(py_us + 1) * w_us + (px_us + 1)];
        let v5 = lp[(py_us + 1) * w_us + px_us];
        let v6 = lp[py_us * w_us + (px_us + 2)];
        let v7 = lp[py_us * w_us + (px_us + 1)];
        let v8 = lp[py_us * w_us + px_us];
        let v9 = parent[(py_us + 1) * w_us + (px_us + 1)];
        a00 += v0 * v0;
        a01 += v0 * v1;
        a02 += v0 * v2;
        a03 += v0 * v3;
        a04 += v0 * v4;
        a05 += v0 * v5;
        a06 += v0 * v6;
        a07 += v0 * v7;
        a08 += v0 * v8;
        a09 += v0 * v9;
        a10 += v1 * v0;
        a11 += v1 * v1;
        a12 += v1 * v2;
        a13 += v1 * v3;
        a14 += v1 * v4;
        a15 += v1 * v5;
        a16 += v1 * v6;
        a17 += v1 * v7;
        a18 += v1 * v8;
        a19 += v1 * v9;
        a20 += v2 * v0;
        a21 += v2 * v1;
        a22 += v2 * v2;
        a23 += v2 * v3;
        a24 += v2 * v4;
        a25 += v2 * v5;
        a26 += v2 * v6;
        a27 += v2 * v7;
        a28 += v2 * v8;
        a29 += v2 * v9;
        a30 += v3 * v0;
        a31 += v3 * v1;
        a32 += v3 * v2;
        a33 += v3 * v3;
        a34 += v3 * v4;
        a35 += v3 * v5;
        a36 += v3 * v6;
        a37 += v3 * v7;
        a38 += v3 * v8;
        a39 += v3 * v9;
        a40 += v4 * v0;
        a41 += v4 * v1;
        a42 += v4 * v2;
        a43 += v4 * v3;
        a44 += v4 * v4;
        a45 += v4 * v5;
        a46 += v4 * v6;
        a47 += v4 * v7;
        a48 += v4 * v8;
        a49 += v4 * v9;
        a50 += v5 * v0;
        a51 += v5 * v1;
        a52 += v5 * v2;
        a53 += v5 * v3;
        a54 += v5 * v4;
        a55 += v5 * v5;
        a56 += v5 * v6;
        a57 += v5 * v7;
        a58 += v5 * v8;
        a59 += v5 * v9;
        a60 += v6 * v0;
        a61 += v6 * v1;
        a62 += v6 * v2;
        a63 += v6 * v3;
        a64 += v6 * v4;
        a65 += v6 * v5;
        a66 += v6 * v6;
        a67 += v6 * v7;
        a68 += v6 * v8;
        a69 += v6 * v9;
        a70 += v7 * v0;
        a71 += v7 * v1;
        a72 += v7 * v2;
        a73 += v7 * v3;
        a74 += v7 * v4;
        a75 += v7 * v5;
        a76 += v7 * v6;
        a77 += v7 * v7;
        a78 += v7 * v8;
        a79 += v7 * v9;
        a80 += v8 * v0;
        a81 += v8 * v1;
        a82 += v8 * v2;
        a83 += v8 * v3;
        a84 += v8 * v4;
        a85 += v8 * v5;
        a86 += v8 * v6;
        a87 += v8 * v7;
        a88 += v8 * v8;
        a89 += v8 * v9;
        a90 += v9 * v0;
        a91 += v9 * v1;
        a92 += v9 * v2;
        a93 += v9 * v3;
        a94 += v9 * v4;
        a95 += v9 * v5;
        a96 += v9 * v6;
        a97 += v9 * v7;
        a98 += v9 * v8;
        a99 += v9 * v9;
        p += stride;
    }

    // Layout: partials[i * n_threads + tid] = a_i. 100 cells (10×10).
    partials[0 * n_threads_us + tid] = a00;
    partials[1 * n_threads_us + tid] = a01;
    partials[2 * n_threads_us + tid] = a02;
    partials[3 * n_threads_us + tid] = a03;
    partials[4 * n_threads_us + tid] = a04;
    partials[5 * n_threads_us + tid] = a05;
    partials[6 * n_threads_us + tid] = a06;
    partials[7 * n_threads_us + tid] = a07;
    partials[8 * n_threads_us + tid] = a08;
    partials[9 * n_threads_us + tid] = a09;
    partials[10 * n_threads_us + tid] = a10;
    partials[11 * n_threads_us + tid] = a11;
    partials[12 * n_threads_us + tid] = a12;
    partials[13 * n_threads_us + tid] = a13;
    partials[14 * n_threads_us + tid] = a14;
    partials[15 * n_threads_us + tid] = a15;
    partials[16 * n_threads_us + tid] = a16;
    partials[17 * n_threads_us + tid] = a17;
    partials[18 * n_threads_us + tid] = a18;
    partials[19 * n_threads_us + tid] = a19;
    partials[20 * n_threads_us + tid] = a20;
    partials[21 * n_threads_us + tid] = a21;
    partials[22 * n_threads_us + tid] = a22;
    partials[23 * n_threads_us + tid] = a23;
    partials[24 * n_threads_us + tid] = a24;
    partials[25 * n_threads_us + tid] = a25;
    partials[26 * n_threads_us + tid] = a26;
    partials[27 * n_threads_us + tid] = a27;
    partials[28 * n_threads_us + tid] = a28;
    partials[29 * n_threads_us + tid] = a29;
    partials[30 * n_threads_us + tid] = a30;
    partials[31 * n_threads_us + tid] = a31;
    partials[32 * n_threads_us + tid] = a32;
    partials[33 * n_threads_us + tid] = a33;
    partials[34 * n_threads_us + tid] = a34;
    partials[35 * n_threads_us + tid] = a35;
    partials[36 * n_threads_us + tid] = a36;
    partials[37 * n_threads_us + tid] = a37;
    partials[38 * n_threads_us + tid] = a38;
    partials[39 * n_threads_us + tid] = a39;
    partials[40 * n_threads_us + tid] = a40;
    partials[41 * n_threads_us + tid] = a41;
    partials[42 * n_threads_us + tid] = a42;
    partials[43 * n_threads_us + tid] = a43;
    partials[44 * n_threads_us + tid] = a44;
    partials[45 * n_threads_us + tid] = a45;
    partials[46 * n_threads_us + tid] = a46;
    partials[47 * n_threads_us + tid] = a47;
    partials[48 * n_threads_us + tid] = a48;
    partials[49 * n_threads_us + tid] = a49;
    partials[50 * n_threads_us + tid] = a50;
    partials[51 * n_threads_us + tid] = a51;
    partials[52 * n_threads_us + tid] = a52;
    partials[53 * n_threads_us + tid] = a53;
    partials[54 * n_threads_us + tid] = a54;
    partials[55 * n_threads_us + tid] = a55;
    partials[56 * n_threads_us + tid] = a56;
    partials[57 * n_threads_us + tid] = a57;
    partials[58 * n_threads_us + tid] = a58;
    partials[59 * n_threads_us + tid] = a59;
    partials[60 * n_threads_us + tid] = a60;
    partials[61 * n_threads_us + tid] = a61;
    partials[62 * n_threads_us + tid] = a62;
    partials[63 * n_threads_us + tid] = a63;
    partials[64 * n_threads_us + tid] = a64;
    partials[65 * n_threads_us + tid] = a65;
    partials[66 * n_threads_us + tid] = a66;
    partials[67 * n_threads_us + tid] = a67;
    partials[68 * n_threads_us + tid] = a68;
    partials[69 * n_threads_us + tid] = a69;
    partials[70 * n_threads_us + tid] = a70;
    partials[71 * n_threads_us + tid] = a71;
    partials[72 * n_threads_us + tid] = a72;
    partials[73 * n_threads_us + tid] = a73;
    partials[74 * n_threads_us + tid] = a74;
    partials[75 * n_threads_us + tid] = a75;
    partials[76 * n_threads_us + tid] = a76;
    partials[77 * n_threads_us + tid] = a77;
    partials[78 * n_threads_us + tid] = a78;
    partials[79 * n_threads_us + tid] = a79;
    partials[80 * n_threads_us + tid] = a80;
    partials[81 * n_threads_us + tid] = a81;
    partials[82 * n_threads_us + tid] = a82;
    partials[83 * n_threads_us + tid] = a83;
    partials[84 * n_threads_us + tid] = a84;
    partials[85 * n_threads_us + tid] = a85;
    partials[86 * n_threads_us + tid] = a86;
    partials[87 * n_threads_us + tid] = a87;
    partials[88 * n_threads_us + tid] = a88;
    partials[89 * n_threads_us + tid] = a89;
    partials[90 * n_threads_us + tid] = a90;
    partials[91 * n_threads_us + tid] = a91;
    partials[92 * n_threads_us + tid] = a92;
    partials[93 * n_threads_us + tid] = a93;
    partials[94 * n_threads_us + tid] = a94;
    partials[95 * n_threads_us + tid] = a95;
    partials[96 * n_threads_us + tid] = a96;
    partials[97 * n_threads_us + tid] = a97;
    partials[98 * n_threads_us + tid] = a98;
    partials[99 * n_threads_us + tid] = a99;
}

/// Finalize: one cube per cell `i`, sum `partials[i * n_threads..(i+1) * n_threads]`
/// into `cu[i]`. Launch with `CubeCount::Static(n_cells, 1, 1)` and
/// `CubeDim::new_1d(1)` (one thread per cube — minimal but matches the
/// shape of reduction.rs's `finalize_kernel`; could parallelize per-cube
/// later if profiling shows it matters).
#[cube(launch_unchecked)]
pub fn cov_finalize_kernel(partials: &Array<f32>, cu: &mut Array<f32>, n_threads: u32) {
    let cell = CUBE_POS_X;
    let n_t = n_threads as usize;
    let base = (cell as usize) * n_t;
    let mut s = 0.0_f32;
    let mut k: u32 = 0;
    while k < n_threads {
        s += partials[base + (k as usize)];
        k += 1;
    }
    cu[cell as usize] = s;
}
