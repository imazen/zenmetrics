//! Covariance accumulator for the IW-SSIM neighborhood matrix.
//!
//! Per scale `s`, the paper builds a `nexp × N` matrix `Y` where each
//! row is a 3×3 neighborhood from `LP[s]` (plus, if `parent = true`
//! and `s < Nsc − 2`, a 10ᵗʰ column from the parent band). The
//! covariance is `C_u = (1 / nexp) · Yᵀ Y` — host divides by `nexp`
//! after readback, so the kernel produces the unscaled sum.
//!
//! Accumulation strategy: `Atomic<f32>::fetch_add` on an `N²`-element
//! buffer. ≤ 100 atomic targets, contention is bounded by `nblv ·
//! nblh` pixels — empirically fine on CUDA. The metal codegen path
//! silently no-ops `Atomic<f32>::add`; same caveat as `ssim2-gpu`'s
//! reduction. Until a portable path matters here, CUDA / wgpu-DX12
//! is the supported deployment surface.

use cubecl::prelude::*;

/// No-parent variant: 9 columns of `Y`. Reads `lp` at the 9 spatial
/// neighborhood offsets and accumulates the full `9×9` outer product
/// into `cu`.
#[cube(launch_unchecked)]
pub fn cov_accum_no_parent_kernel(
    lp: &Array<f32>,
    cu: &mut Array<Atomic<f32>>,
    h: u32,
    w: u32,
) {
    let tid = ABSOLUTE_POS;
    let stride = CUBE_COUNT * (CUBE_DIM_X as usize);
    let nblv = h - 2;
    let nblh = w - 2;
    let nexp = (nblv * nblh) as usize;
    let w_us = w as usize;

    let mut p = tid;
    while p < nexp {
        let py = (p as u32) / nblh;
        let px = (p as u32) - py * nblh;
        let py_us = py as usize;
        let px_us = px as usize;
        // 9 neighborhood reads (offsets fixed by reference code order).
        let v0 = lp[(py_us + 2) * w_us + (px_us + 2)];
        let v1 = lp[(py_us + 2) * w_us + (px_us + 1)];
        let v2 = lp[(py_us + 2) * w_us + px_us];
        let v3 = lp[(py_us + 1) * w_us + (px_us + 2)];
        let v4 = lp[(py_us + 1) * w_us + (px_us + 1)];
        let v5 = lp[(py_us + 1) * w_us + px_us];
        let v6 = lp[py_us * w_us + (px_us + 2)];
        let v7 = lp[py_us * w_us + (px_us + 1)];
        let v8 = lp[py_us * w_us + px_us];

        // 9×9 = 81 atomic fetch_adds. Stride 9 = 9·N for N=9 layout.
        cu[0].fetch_add(v0 * v0);
        cu[1].fetch_add(v0 * v1);
        cu[2].fetch_add(v0 * v2);
        cu[3].fetch_add(v0 * v3);
        cu[4].fetch_add(v0 * v4);
        cu[5].fetch_add(v0 * v5);
        cu[6].fetch_add(v0 * v6);
        cu[7].fetch_add(v0 * v7);
        cu[8].fetch_add(v0 * v8);
        cu[9].fetch_add(v1 * v0);
        cu[10].fetch_add(v1 * v1);
        cu[11].fetch_add(v1 * v2);
        cu[12].fetch_add(v1 * v3);
        cu[13].fetch_add(v1 * v4);
        cu[14].fetch_add(v1 * v5);
        cu[15].fetch_add(v1 * v6);
        cu[16].fetch_add(v1 * v7);
        cu[17].fetch_add(v1 * v8);
        cu[18].fetch_add(v2 * v0);
        cu[19].fetch_add(v2 * v1);
        cu[20].fetch_add(v2 * v2);
        cu[21].fetch_add(v2 * v3);
        cu[22].fetch_add(v2 * v4);
        cu[23].fetch_add(v2 * v5);
        cu[24].fetch_add(v2 * v6);
        cu[25].fetch_add(v2 * v7);
        cu[26].fetch_add(v2 * v8);
        cu[27].fetch_add(v3 * v0);
        cu[28].fetch_add(v3 * v1);
        cu[29].fetch_add(v3 * v2);
        cu[30].fetch_add(v3 * v3);
        cu[31].fetch_add(v3 * v4);
        cu[32].fetch_add(v3 * v5);
        cu[33].fetch_add(v3 * v6);
        cu[34].fetch_add(v3 * v7);
        cu[35].fetch_add(v3 * v8);
        cu[36].fetch_add(v4 * v0);
        cu[37].fetch_add(v4 * v1);
        cu[38].fetch_add(v4 * v2);
        cu[39].fetch_add(v4 * v3);
        cu[40].fetch_add(v4 * v4);
        cu[41].fetch_add(v4 * v5);
        cu[42].fetch_add(v4 * v6);
        cu[43].fetch_add(v4 * v7);
        cu[44].fetch_add(v4 * v8);
        cu[45].fetch_add(v5 * v0);
        cu[46].fetch_add(v5 * v1);
        cu[47].fetch_add(v5 * v2);
        cu[48].fetch_add(v5 * v3);
        cu[49].fetch_add(v5 * v4);
        cu[50].fetch_add(v5 * v5);
        cu[51].fetch_add(v5 * v6);
        cu[52].fetch_add(v5 * v7);
        cu[53].fetch_add(v5 * v8);
        cu[54].fetch_add(v6 * v0);
        cu[55].fetch_add(v6 * v1);
        cu[56].fetch_add(v6 * v2);
        cu[57].fetch_add(v6 * v3);
        cu[58].fetch_add(v6 * v4);
        cu[59].fetch_add(v6 * v5);
        cu[60].fetch_add(v6 * v6);
        cu[61].fetch_add(v6 * v7);
        cu[62].fetch_add(v6 * v8);
        cu[63].fetch_add(v7 * v0);
        cu[64].fetch_add(v7 * v1);
        cu[65].fetch_add(v7 * v2);
        cu[66].fetch_add(v7 * v3);
        cu[67].fetch_add(v7 * v4);
        cu[68].fetch_add(v7 * v5);
        cu[69].fetch_add(v7 * v6);
        cu[70].fetch_add(v7 * v7);
        cu[71].fetch_add(v7 * v8);
        cu[72].fetch_add(v8 * v0);
        cu[73].fetch_add(v8 * v1);
        cu[74].fetch_add(v8 * v2);
        cu[75].fetch_add(v8 * v3);
        cu[76].fetch_add(v8 * v4);
        cu[77].fetch_add(v8 * v5);
        cu[78].fetch_add(v8 * v6);
        cu[79].fetch_add(v8 * v7);
        cu[80].fetch_add(v8 * v8);

        p += stride;
    }
}

/// With-parent variant: 10 columns of `Y` (9 spatial + 1 parent band).
/// Accumulates the full `10×10` outer product into `cu`.
#[cube(launch_unchecked)]
pub fn cov_accum_with_parent_kernel(
    lp: &Array<f32>,
    parent: &Array<f32>,
    cu: &mut Array<Atomic<f32>>,
    h: u32,
    w: u32,
) {
    let tid = ABSOLUTE_POS;
    let stride = CUBE_COUNT * (CUBE_DIM_X as usize);
    let nblv = h - 2;
    let nblh = w - 2;
    let nexp = (nblv * nblh) as usize;
    let w_us = w as usize;

    let mut p = tid;
    while p < nexp {
        let py = (p as u32) / nblh;
        let px = (p as u32) - py * nblh;
        let py_us = py as usize;
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

        // 10×10 = 100 atomic fetch_adds — index = i*10 + j.
        cu[0].fetch_add(v0 * v0);
        cu[1].fetch_add(v0 * v1);
        cu[2].fetch_add(v0 * v2);
        cu[3].fetch_add(v0 * v3);
        cu[4].fetch_add(v0 * v4);
        cu[5].fetch_add(v0 * v5);
        cu[6].fetch_add(v0 * v6);
        cu[7].fetch_add(v0 * v7);
        cu[8].fetch_add(v0 * v8);
        cu[9].fetch_add(v0 * v9);
        cu[10].fetch_add(v1 * v0);
        cu[11].fetch_add(v1 * v1);
        cu[12].fetch_add(v1 * v2);
        cu[13].fetch_add(v1 * v3);
        cu[14].fetch_add(v1 * v4);
        cu[15].fetch_add(v1 * v5);
        cu[16].fetch_add(v1 * v6);
        cu[17].fetch_add(v1 * v7);
        cu[18].fetch_add(v1 * v8);
        cu[19].fetch_add(v1 * v9);
        cu[20].fetch_add(v2 * v0);
        cu[21].fetch_add(v2 * v1);
        cu[22].fetch_add(v2 * v2);
        cu[23].fetch_add(v2 * v3);
        cu[24].fetch_add(v2 * v4);
        cu[25].fetch_add(v2 * v5);
        cu[26].fetch_add(v2 * v6);
        cu[27].fetch_add(v2 * v7);
        cu[28].fetch_add(v2 * v8);
        cu[29].fetch_add(v2 * v9);
        cu[30].fetch_add(v3 * v0);
        cu[31].fetch_add(v3 * v1);
        cu[32].fetch_add(v3 * v2);
        cu[33].fetch_add(v3 * v3);
        cu[34].fetch_add(v3 * v4);
        cu[35].fetch_add(v3 * v5);
        cu[36].fetch_add(v3 * v6);
        cu[37].fetch_add(v3 * v7);
        cu[38].fetch_add(v3 * v8);
        cu[39].fetch_add(v3 * v9);
        cu[40].fetch_add(v4 * v0);
        cu[41].fetch_add(v4 * v1);
        cu[42].fetch_add(v4 * v2);
        cu[43].fetch_add(v4 * v3);
        cu[44].fetch_add(v4 * v4);
        cu[45].fetch_add(v4 * v5);
        cu[46].fetch_add(v4 * v6);
        cu[47].fetch_add(v4 * v7);
        cu[48].fetch_add(v4 * v8);
        cu[49].fetch_add(v4 * v9);
        cu[50].fetch_add(v5 * v0);
        cu[51].fetch_add(v5 * v1);
        cu[52].fetch_add(v5 * v2);
        cu[53].fetch_add(v5 * v3);
        cu[54].fetch_add(v5 * v4);
        cu[55].fetch_add(v5 * v5);
        cu[56].fetch_add(v5 * v6);
        cu[57].fetch_add(v5 * v7);
        cu[58].fetch_add(v5 * v8);
        cu[59].fetch_add(v5 * v9);
        cu[60].fetch_add(v6 * v0);
        cu[61].fetch_add(v6 * v1);
        cu[62].fetch_add(v6 * v2);
        cu[63].fetch_add(v6 * v3);
        cu[64].fetch_add(v6 * v4);
        cu[65].fetch_add(v6 * v5);
        cu[66].fetch_add(v6 * v6);
        cu[67].fetch_add(v6 * v7);
        cu[68].fetch_add(v6 * v8);
        cu[69].fetch_add(v6 * v9);
        cu[70].fetch_add(v7 * v0);
        cu[71].fetch_add(v7 * v1);
        cu[72].fetch_add(v7 * v2);
        cu[73].fetch_add(v7 * v3);
        cu[74].fetch_add(v7 * v4);
        cu[75].fetch_add(v7 * v5);
        cu[76].fetch_add(v7 * v6);
        cu[77].fetch_add(v7 * v7);
        cu[78].fetch_add(v7 * v8);
        cu[79].fetch_add(v7 * v9);
        cu[80].fetch_add(v8 * v0);
        cu[81].fetch_add(v8 * v1);
        cu[82].fetch_add(v8 * v2);
        cu[83].fetch_add(v8 * v3);
        cu[84].fetch_add(v8 * v4);
        cu[85].fetch_add(v8 * v5);
        cu[86].fetch_add(v8 * v6);
        cu[87].fetch_add(v8 * v7);
        cu[88].fetch_add(v8 * v8);
        cu[89].fetch_add(v8 * v9);
        cu[90].fetch_add(v9 * v0);
        cu[91].fetch_add(v9 * v1);
        cu[92].fetch_add(v9 * v2);
        cu[93].fetch_add(v9 * v3);
        cu[94].fetch_add(v9 * v4);
        cu[95].fetch_add(v9 * v5);
        cu[96].fetch_add(v9 * v6);
        cu[97].fetch_add(v9 * v7);
        cu[98].fetch_add(v9 * v8);
        cu[99].fetch_add(v9 * v9);

        p += stride;
    }
}
