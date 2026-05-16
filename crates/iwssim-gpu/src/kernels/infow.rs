//! Per-pixel quadratic form (`ss`) and mutual-information weight
//! (`infow`) — the final per-scale operation in the IW path.
//!
//! Reference (`IW_SSIM_PyTorch.py`):
//!
//! ```python
//! ss = (Y @ C_u_inv) * Y / N           # row-wise quadratic form
//! ss = sum(ss, axis=1).reshape(nblv, nblh)
//! infow = sum_j log2(1 + ((vv + (1 + g²)·σ²) · ss · λ_j + σ²·vv) / σ⁴)
//! infow[infow < tol] = 0
//! ```
//!
//! Numerics: the Python writes the quadratic form as
//! `(Y @ C_u_inv) * Y → sum(axis=1)` — i.e. for each column `k` it
//! computes `inner_k = Σ_j y[j]·C_u_inv[j,k]`, multiplies by `y[k]`,
//! and sums over `k`. We replicate that order so the result matches
//! to within ULPs.

use cubecl::prelude::*;

const TOL: f32 = 1.0e-15_f32;

/// 9-neighbor case (no parent band). C_u_inv is row-major 9×9 (length 81).
#[cube(launch_unchecked)]
pub fn infow_no_parent_kernel(
    lp: &Array<f32>,
    g_buf: &Array<f32>,
    vv_buf: &Array<f32>,
    cu_inv: &Array<f32>,
    eig_lambda: &Array<f32>,
    iw_out: &mut Array<f32>,
    h: u32,
    w: u32,
    sigma_nsq: f32,
) {
    let _ = h;
    let idx = ABSOLUTE_POS;
    let nblh = w - 2;
    let n = iw_out.len();
    if idx >= n {
        terminate!();
    }
    let py = (idx as u32) / nblh;
    let px = (idx as u32) - py * nblh;
    let w_us = w as usize;
    let py_us = py as usize;
    let px_us = px as usize;

    // 9 neighborhood reads (same offsets as cov_accum).
    let v0 = lp[(py_us + 2) * w_us + (px_us + 2)];
    let v1 = lp[(py_us + 2) * w_us + (px_us + 1)];
    let v2 = lp[(py_us + 2) * w_us + px_us];
    let v3 = lp[(py_us + 1) * w_us + (px_us + 2)];
    let v4 = lp[(py_us + 1) * w_us + (px_us + 1)];
    let v5 = lp[(py_us + 1) * w_us + px_us];
    let v6 = lp[py_us * w_us + (px_us + 2)];
    let v7 = lp[py_us * w_us + (px_us + 1)];
    let v8 = lp[py_us * w_us + px_us];

    // inner_k = Σ_j y[j] · C_u_inv[j*9 + k]
    let i0 = cu_inv[0] * v0
        + cu_inv[9] * v1
        + cu_inv[18] * v2
        + cu_inv[27] * v3
        + cu_inv[36] * v4
        + cu_inv[45] * v5
        + cu_inv[54] * v6
        + cu_inv[63] * v7
        + cu_inv[72] * v8;
    let i1 = cu_inv[1] * v0
        + cu_inv[10] * v1
        + cu_inv[19] * v2
        + cu_inv[28] * v3
        + cu_inv[37] * v4
        + cu_inv[46] * v5
        + cu_inv[55] * v6
        + cu_inv[64] * v7
        + cu_inv[73] * v8;
    let i2 = cu_inv[2] * v0
        + cu_inv[11] * v1
        + cu_inv[20] * v2
        + cu_inv[29] * v3
        + cu_inv[38] * v4
        + cu_inv[47] * v5
        + cu_inv[56] * v6
        + cu_inv[65] * v7
        + cu_inv[74] * v8;
    let i3 = cu_inv[3] * v0
        + cu_inv[12] * v1
        + cu_inv[21] * v2
        + cu_inv[30] * v3
        + cu_inv[39] * v4
        + cu_inv[48] * v5
        + cu_inv[57] * v6
        + cu_inv[66] * v7
        + cu_inv[75] * v8;
    let i4 = cu_inv[4] * v0
        + cu_inv[13] * v1
        + cu_inv[22] * v2
        + cu_inv[31] * v3
        + cu_inv[40] * v4
        + cu_inv[49] * v5
        + cu_inv[58] * v6
        + cu_inv[67] * v7
        + cu_inv[76] * v8;
    let i5 = cu_inv[5] * v0
        + cu_inv[14] * v1
        + cu_inv[23] * v2
        + cu_inv[32] * v3
        + cu_inv[41] * v4
        + cu_inv[50] * v5
        + cu_inv[59] * v6
        + cu_inv[68] * v7
        + cu_inv[77] * v8;
    let i6 = cu_inv[6] * v0
        + cu_inv[15] * v1
        + cu_inv[24] * v2
        + cu_inv[33] * v3
        + cu_inv[42] * v4
        + cu_inv[51] * v5
        + cu_inv[60] * v6
        + cu_inv[69] * v7
        + cu_inv[78] * v8;
    let i7 = cu_inv[7] * v0
        + cu_inv[16] * v1
        + cu_inv[25] * v2
        + cu_inv[34] * v3
        + cu_inv[43] * v4
        + cu_inv[52] * v5
        + cu_inv[61] * v6
        + cu_inv[70] * v7
        + cu_inv[79] * v8;
    let i8 = cu_inv[8] * v0
        + cu_inv[17] * v1
        + cu_inv[26] * v2
        + cu_inv[35] * v3
        + cu_inv[44] * v4
        + cu_inv[53] * v5
        + cu_inv[62] * v6
        + cu_inv[71] * v7
        + cu_inv[80] * v8;

    // outer sum: Σ_k inner_k · y[k], then divide by N (= 9).
    let ss = (i0 * v0 + i1 * v1 + i2 * v2 + i3 * v3 + i4 * v4 + i5 * v5 + i6 * v6 + i7 * v7
        + i8 * v8)
        * (1.0_f32 / 9.0_f32);

    let g = g_buf[(py_us + 1) * w_us + (px_us + 1)];
    let vv = vv_buf[(py_us + 1) * w_us + (px_us + 1)];

    let a = (vv + (1.0_f32 + g * g) * sigma_nsq) * ss;
    let b = sigma_nsq * vv;
    let sig4 = sigma_nsq * sigma_nsq;

    let l0 = (1.0_f32 + (a * eig_lambda[0] + b) / sig4).ln() * (1.0_f32 / 0.693_147_18_f32);
    let l1 = (1.0_f32 + (a * eig_lambda[1] + b) / sig4).ln() * (1.0_f32 / 0.693_147_18_f32);
    let l2 = (1.0_f32 + (a * eig_lambda[2] + b) / sig4).ln() * (1.0_f32 / 0.693_147_18_f32);
    let l3 = (1.0_f32 + (a * eig_lambda[3] + b) / sig4).ln() * (1.0_f32 / 0.693_147_18_f32);
    let l4 = (1.0_f32 + (a * eig_lambda[4] + b) / sig4).ln() * (1.0_f32 / 0.693_147_18_f32);
    let l5 = (1.0_f32 + (a * eig_lambda[5] + b) / sig4).ln() * (1.0_f32 / 0.693_147_18_f32);
    let l6 = (1.0_f32 + (a * eig_lambda[6] + b) / sig4).ln() * (1.0_f32 / 0.693_147_18_f32);
    let l7 = (1.0_f32 + (a * eig_lambda[7] + b) / sig4).ln() * (1.0_f32 / 0.693_147_18_f32);
    let l8 = (1.0_f32 + (a * eig_lambda[8] + b) / sig4).ln() * (1.0_f32 / 0.693_147_18_f32);
    let mut infow = l0 + l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8;
    if infow < TOL {
        infow = 0.0_f32;
    }
    iw_out[idx] = infow;
}

/// 10-neighbor case (parent band). C_u_inv row-major 10×10 (length 100).
#[cube(launch_unchecked)]
pub fn infow_with_parent_kernel(
    lp: &Array<f32>,
    parent: &Array<f32>,
    g_buf: &Array<f32>,
    vv_buf: &Array<f32>,
    cu_inv: &Array<f32>,
    eig_lambda: &Array<f32>,
    iw_out: &mut Array<f32>,
    h: u32,
    w: u32,
    sigma_nsq: f32,
) {
    let _ = h;
    let idx = ABSOLUTE_POS;
    let nblh = w - 2;
    let n = iw_out.len();
    if idx >= n {
        terminate!();
    }
    let py = (idx as u32) / nblh;
    let px = (idx as u32) - py * nblh;
    let w_us = w as usize;
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

    // inner_k = Σ_j y[j] · C_u_inv[j*10 + k]; k ∈ 0..10.
    let i0 = cu_inv[0] * v0
        + cu_inv[10] * v1
        + cu_inv[20] * v2
        + cu_inv[30] * v3
        + cu_inv[40] * v4
        + cu_inv[50] * v5
        + cu_inv[60] * v6
        + cu_inv[70] * v7
        + cu_inv[80] * v8
        + cu_inv[90] * v9;
    let i1 = cu_inv[1] * v0
        + cu_inv[11] * v1
        + cu_inv[21] * v2
        + cu_inv[31] * v3
        + cu_inv[41] * v4
        + cu_inv[51] * v5
        + cu_inv[61] * v6
        + cu_inv[71] * v7
        + cu_inv[81] * v8
        + cu_inv[91] * v9;
    let i2 = cu_inv[2] * v0
        + cu_inv[12] * v1
        + cu_inv[22] * v2
        + cu_inv[32] * v3
        + cu_inv[42] * v4
        + cu_inv[52] * v5
        + cu_inv[62] * v6
        + cu_inv[72] * v7
        + cu_inv[82] * v8
        + cu_inv[92] * v9;
    let i3 = cu_inv[3] * v0
        + cu_inv[13] * v1
        + cu_inv[23] * v2
        + cu_inv[33] * v3
        + cu_inv[43] * v4
        + cu_inv[53] * v5
        + cu_inv[63] * v6
        + cu_inv[73] * v7
        + cu_inv[83] * v8
        + cu_inv[93] * v9;
    let i4 = cu_inv[4] * v0
        + cu_inv[14] * v1
        + cu_inv[24] * v2
        + cu_inv[34] * v3
        + cu_inv[44] * v4
        + cu_inv[54] * v5
        + cu_inv[64] * v6
        + cu_inv[74] * v7
        + cu_inv[84] * v8
        + cu_inv[94] * v9;
    let i5 = cu_inv[5] * v0
        + cu_inv[15] * v1
        + cu_inv[25] * v2
        + cu_inv[35] * v3
        + cu_inv[45] * v4
        + cu_inv[55] * v5
        + cu_inv[65] * v6
        + cu_inv[75] * v7
        + cu_inv[85] * v8
        + cu_inv[95] * v9;
    let i6 = cu_inv[6] * v0
        + cu_inv[16] * v1
        + cu_inv[26] * v2
        + cu_inv[36] * v3
        + cu_inv[46] * v4
        + cu_inv[56] * v5
        + cu_inv[66] * v6
        + cu_inv[76] * v7
        + cu_inv[86] * v8
        + cu_inv[96] * v9;
    let i7 = cu_inv[7] * v0
        + cu_inv[17] * v1
        + cu_inv[27] * v2
        + cu_inv[37] * v3
        + cu_inv[47] * v4
        + cu_inv[57] * v5
        + cu_inv[67] * v6
        + cu_inv[77] * v7
        + cu_inv[87] * v8
        + cu_inv[97] * v9;
    let i8 = cu_inv[8] * v0
        + cu_inv[18] * v1
        + cu_inv[28] * v2
        + cu_inv[38] * v3
        + cu_inv[48] * v4
        + cu_inv[58] * v5
        + cu_inv[68] * v6
        + cu_inv[78] * v7
        + cu_inv[88] * v8
        + cu_inv[98] * v9;
    let i9 = cu_inv[9] * v0
        + cu_inv[19] * v1
        + cu_inv[29] * v2
        + cu_inv[39] * v3
        + cu_inv[49] * v4
        + cu_inv[59] * v5
        + cu_inv[69] * v6
        + cu_inv[79] * v7
        + cu_inv[89] * v8
        + cu_inv[99] * v9;

    let ss = (i0 * v0
        + i1 * v1
        + i2 * v2
        + i3 * v3
        + i4 * v4
        + i5 * v5
        + i6 * v6
        + i7 * v7
        + i8 * v8
        + i9 * v9)
        * (1.0_f32 / 10.0_f32);

    let g = g_buf[(py_us + 1) * w_us + (px_us + 1)];
    let vv = vv_buf[(py_us + 1) * w_us + (px_us + 1)];

    let a = (vv + (1.0_f32 + g * g) * sigma_nsq) * ss;
    let b = sigma_nsq * vv;
    let sig4 = sigma_nsq * sigma_nsq;

    let l0 = (1.0_f32 + (a * eig_lambda[0] + b) / sig4).ln() * (1.0_f32 / 0.693_147_18_f32);
    let l1 = (1.0_f32 + (a * eig_lambda[1] + b) / sig4).ln() * (1.0_f32 / 0.693_147_18_f32);
    let l2 = (1.0_f32 + (a * eig_lambda[2] + b) / sig4).ln() * (1.0_f32 / 0.693_147_18_f32);
    let l3 = (1.0_f32 + (a * eig_lambda[3] + b) / sig4).ln() * (1.0_f32 / 0.693_147_18_f32);
    let l4 = (1.0_f32 + (a * eig_lambda[4] + b) / sig4).ln() * (1.0_f32 / 0.693_147_18_f32);
    let l5 = (1.0_f32 + (a * eig_lambda[5] + b) / sig4).ln() * (1.0_f32 / 0.693_147_18_f32);
    let l6 = (1.0_f32 + (a * eig_lambda[6] + b) / sig4).ln() * (1.0_f32 / 0.693_147_18_f32);
    let l7 = (1.0_f32 + (a * eig_lambda[7] + b) / sig4).ln() * (1.0_f32 / 0.693_147_18_f32);
    let l8 = (1.0_f32 + (a * eig_lambda[8] + b) / sig4).ln() * (1.0_f32 / 0.693_147_18_f32);
    let l9 = (1.0_f32 + (a * eig_lambda[9] + b) / sig4).ln() * (1.0_f32 / 0.693_147_18_f32);
    let mut infow = l0 + l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8 + l9;
    if infow < TOL {
        infow = 0.0_f32;
    }
    iw_out[idx] = infow;
}
