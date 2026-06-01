//! Small symmetric eigendecomposition + PSD-cleaning + matrix inverse,
//! sized for the IW-SSIM neighborhood covariance (`N ≤ 10`).
//!
//! Bit-identical algorithm to `iwssim-gpu::eig` — the GPU port already
//! runs this on the host, so reusing the same numerical recipe gives
//! us free atomic-tolerance parity vs the GPU path. (Same Jacobi
//! sweep, same PSD-cleaning rescale, same `EIG_FLOOR` regularization.)

use alloc::vec::Vec;

/// Symmetric Jacobi eigendecomposition. Converges quadratically once
/// off-diagonal terms are small. `MAX_ITERS` is generous for `N ≤ 10`.
const MAX_ITERS: usize = 100;
const TOL_OFF: f64 = 1.0e-14;
/// Floor for eigenvalues during PSD-cleaning. Anything below this is
/// treated as zero in the matrix-inverse step (regularized to a large
/// inverse).
const EIG_FLOOR: f64 = 1.0e-30;

/// Result of `decompose_and_invert`. `lambda` and `c_u_inv` are sized
/// for the caller's `n` (≤ 10), with trailing entries unused.
pub(crate) struct EigResult {
    /// PSD-cleaned eigenvalues — `λ_k` in the paper's eq (28). Stored
    /// as `f32` to match downstream per-pixel arithmetic precision.
    pub lambda: [f32; 10],
    /// `N²` inverse of the PSD-cleaned `C_u`, row-major.
    pub c_u_inv: [f32; 100],
    /// Active dimension. `1..=10`.
    pub n: usize,
}

impl EigResult {
    /// Iterator over the active eigenvalues (length `n`).
    pub(crate) fn lambdas(&self) -> &[f32] {
        &self.lambda[..self.n]
    }

    /// `c_u_inv` view as a flat slice of length `n*n`.
    pub(crate) fn c_u_inv_slice(&self) -> &[f32] {
        &self.c_u_inv[..self.n * self.n]
    }
}

/// `c_u` is row-major `n × n`, with `n ∈ {1, ..., 10}` for IW-SSIM. The
/// matrix must be symmetric (we read both halves but assume
/// `c_u[i, j] == c_u[j, i]`). Output is suitable for direct use in
/// `f32` arithmetic downstream.
pub(crate) fn decompose_and_invert(c_u: &[f64], n: usize) -> EigResult {
    assert!(n <= 10);
    assert_eq!(c_u.len(), n * n);

    // Working copies; Jacobi modifies in place.
    let mut a = [[0.0_f64; 10]; 10];
    let mut q = [[0.0_f64; 10]; 10];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = c_u[i * n + j];
        }
        q[i][i] = 1.0;
    }
    // Symmetrize to defend against tiny rounding asymmetry (the
    // upstream uses Yᵀ Y which is symmetric in exact arithmetic; f32
    // sums in non-deterministic order can leave |Δ| ~ ULP).
    for i in 0..n {
        for j in i + 1..n {
            let avg = 0.5 * (a[i][j] + a[j][i]);
            a[i][j] = avg;
            a[j][i] = avg;
        }
    }

    // Classical cyclic Jacobi: sweep all super-diagonal (p, r) pairs
    // for `MAX_ITERS` iterations or until off-diagonal sum < TOL.
    for _ in 0..MAX_ITERS {
        let mut off = 0.0;
        for p in 0..n {
            for r_idx in p + 1..n {
                off += a[p][r_idx].abs();
            }
        }
        if off < TOL_OFF {
            break;
        }
        for p in 0..n {
            for r in p + 1..n {
                let app = a[p][p];
                let arr = a[r][r];
                let apr = a[p][r];
                if apr.abs() < TOL_OFF {
                    continue;
                }
                // Rotation angle θ: tan(2θ) = 2 apr / (app − arr).
                let theta = (arr - app) / (2.0 * apr);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update A
                a[p][p] = app - t * apr;
                a[r][r] = arr + t * apr;
                a[p][r] = 0.0;
                a[r][p] = 0.0;
                for k in 0..n {
                    if k != p && k != r {
                        let akp = a[k][p];
                        let akr = a[k][r];
                        a[k][p] = c * akp - s * akr;
                        a[p][k] = a[k][p];
                        a[k][r] = s * akp + c * akr;
                        a[r][k] = a[k][r];
                    }
                }
                // Update Q (eigenvector matrix)
                for k in 0..n {
                    let qkp = q[k][p];
                    let qkr = q[k][r];
                    q[k][p] = c * qkp - s * qkr;
                    q[k][r] = s * qkp + c * qkr;
                }
            }
        }
    }

    // Eigenvalues live on the diagonal of A.
    let mut lambda_raw = [0.0_f64; 10];
    for i in 0..n {
        lambda_raw[i] = a[i][i];
    }

    // PSD-cleaning matches the MATLAB / Python step exactly:
    //   keep λ⁺ = max(0, λ), rescale so trace(L_new) = trace(L_orig).
    let trace_orig: f64 = lambda_raw.iter().take(n).sum();
    let sum_pos: f64 = lambda_raw.iter().take(n).filter(|&&v| v > 0.0).sum();
    let scale = if sum_pos > 0.0 {
        trace_orig / sum_pos
    } else {
        trace_orig // unused; L_new is zero matrix.
    };

    let mut lambda = [0.0_f64; 10];
    for i in 0..n {
        if lambda_raw[i] > 0.0 {
            lambda[i] = lambda_raw[i] * scale;
        }
    }

    // C_u_inv = Q · diag(1/lambda) · Qᵀ.
    let mut inv_lambda = [0.0_f64; 10];
    for i in 0..n {
        if lambda[i] > EIG_FLOOR {
            inv_lambda[i] = 1.0 / lambda[i];
        } else {
            // Singular — return a large but finite inverse. Matches the
            // GPU port's behavior on degenerate inputs (flat patches).
            inv_lambda[i] = 1.0 / EIG_FLOOR;
        }
    }

    // C_u_inv[i, j] = Σ_k Q[i, k] · (1/λ_k) · Q[j, k].
    let mut c_u_inv = [0.0_f32; 100];
    for i in 0..n {
        for j in 0..n {
            let mut v = 0.0_f64;
            for k in 0..n {
                v += q[i][k] * inv_lambda[k] * q[j][k];
            }
            c_u_inv[i * n + j] = v as f32;
        }
    }

    let mut lambda_f32 = [0.0_f32; 10];
    for i in 0..n {
        lambda_f32[i] = lambda[i] as f32;
    }

    EigResult {
        lambda: lambda_f32,
        c_u_inv,
        n,
    }
}

/// `Y` is `nexp × N` row-major. Returns `Yᵀ · Y / nexp` as a `N × N`
/// row-major `f64` matrix — feed straight into [`decompose_and_invert`].
///
/// `nexp = nblv * nblh`. The accumulation runs in `f64` to match the
/// Python's `torch.mm(Yᵀ, Y) / nexp` precision.
pub(crate) fn cov_from_neighborhood(y: &[f32], nexp: usize, big_n: usize) -> Vec<f64> {
    assert_eq!(y.len(), nexp * big_n);
    let mut c = alloc::vec![0.0_f64; big_n * big_n];
    // Y is row-major (nexp, N). Cᵤ[i, j] = Σ_k Y[k, i] · Y[k, j] / nexp.
    // Compute via two nested loops over (i, j) with an inner reduction
    // over k — simpler than blocked dgemm and faster than per-element
    // atomic-style indexing on cache-cold matrices.
    for i in 0..big_n {
        for j in i..big_n {
            let mut acc = 0.0_f64;
            for k in 0..nexp {
                let a = y[k * big_n + i] as f64;
                let b = y[k * big_n + j] as f64;
                acc += a * b;
            }
            let v = acc / (nexp as f64);
            c[i * big_n + j] = v;
            c[j * big_n + i] = v;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_3x3() {
        let c = alloc::vec![4.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 16.0];
        let r = decompose_and_invert(&c, 3);
        let mut sorted: Vec<f32> = r.lambda.iter().take(3).copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 4.0).abs() < 1e-5);
        assert!((sorted[1] - 9.0).abs() < 1e-5);
        assert!((sorted[2] - 16.0).abs() < 1e-5);
        assert!((r.c_u_inv[0] - 0.25).abs() < 1e-5);
        assert!((r.c_u_inv[4] - 1.0 / 9.0).abs() < 1e-5);
        assert!((r.c_u_inv[8] - 1.0 / 16.0).abs() < 1e-5);
    }

    #[test]
    fn negative_eigenvalue_clipped() {
        let c = alloc::vec![2.0, 1.0, 1.0, -1.0];
        let r = decompose_and_invert(&c, 2);
        let lam: Vec<f32> = r.lambda.iter().take(2).copied().collect();
        let positives: Vec<f32> = lam.iter().filter(|&&v| v > 0.0).copied().collect();
        assert_eq!(positives.len(), 1);
        // Single positive equals trace(original) = 1.0 after rescale.
        assert!((positives[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn cov_symmetric() {
        // 4 samples × 3 dimensions
        let y: Vec<f32> = alloc::vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0, //
            10.0, 11.0, 12.0
        ];
        let c = cov_from_neighborhood(&y, 4, 3);
        for i in 0..3 {
            for j in 0..3 {
                assert!((c[i * 3 + j] - c[j * 3 + i]).abs() < 1e-9);
            }
        }
        // Cᵤ[0,0] = (1²+4²+7²+10²)/4 = 166/4 = 41.5
        assert!((c[0] - 41.5).abs() < 1e-9);
    }
}
