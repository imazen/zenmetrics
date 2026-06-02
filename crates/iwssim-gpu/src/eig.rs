//! Small symmetric eigendecomposition + PSD-cleaning + matrix inverse,
//! sized for the IW-SSIM neighborhood covariance (`N ≤ 10`).
//!
//! The Python / MATLAB reference does exactly this sequence on the CPU
//! via NumPy / `eig` — we replicate it host-side for two reasons:
//!
//! 1. `N ≤ 10` ⇒ even a textbook Jacobi sweep converges in ≤ 20
//!    iterations with O(N³) cost ≈ 1000 fma — negligible next to the
//!    per-pixel work.
//! 2. Avoiding a per-scale device → host → device round-trip for tiny
//!    matrices would not save real time and would add a great deal of
//!    cubecl code.
//!
//! Two outputs the caller pushes back to the GPU:
//! - `C_u_inv`: the `N²` matrix used in the per-pixel quadratic form.
//! - `lambda`: the `N` positive eigenvalues used in the `infow` sum.

/// Symmetric Jacobi eigendecomposition. Converges quadratically once
/// off-diagonal terms are small. `MAX_ITERS` is generous for `N ≤ 10`.
const MAX_ITERS: usize = 100;
const TOL_OFF: f64 = 1.0e-14;
/// Floor for eigenvalues during PSD-cleaning. Anything below this is
/// treated as zero in the matrix-inverse step (regularized to a large
/// inverse). Matches the spirit of the reference's `max(0, λ)` step;
/// the floor is needed only when the cleaned matrix is exactly
/// singular, which is rare in real images.
const EIG_FLOOR: f64 = 1.0e-30;

/// Result of `decompose_and_invert`. `lambda` and `c_u_inv` are sized
/// for the caller's `n` (≤ 10), with trailing entries unused.
pub struct EigResult {
    /// PSD-cleaned eigenvalues — `λ_k` in the paper's eq (28).
    pub lambda: [f32; 10],
    /// `N²` inverse of the PSD-cleaned `C_u`, row-major.
    pub c_u_inv: [f32; 100],
    /// Active dimension. `1..=10`. Carried for callers / parity checks;
    /// the GPU upload path reads the fixed-size arrays directly.
    #[allow(dead_code)]
    pub n: usize,
}

/// `c_u` is row-major `n × n`, with `n ∈ {9, 10}` for IW-SSIM. The
/// matrix must be symmetric (we read both halves but assume
/// `c_u[i, j] == c_u[j, i]`). Output is suitable for direct upload as
/// `f32` arrays.
pub fn decompose_and_invert(c_u: &[f64], n: usize) -> EigResult {
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
    // Symmetrize to defend against tiny atomic-add rounding asymmetry
    // (the upstream uses Yᵀ Y which is symmetric in exact arithmetic;
    // f32 atomic adds in non-deterministic order can leave |Δ| ~ ULP).
    for i in 0..n {
        for j in i + 1..n {
            let avg = 0.5 * (a[i][j] + a[j][i]);
            a[i][j] = avg;
            a[j][i] = avg;
        }
    }

    // Classical cyclic Jacobi: sweep all super-diagonal (p, q) pairs
    // for `MAX_ITERS` iterations or until off-diagonal sum < TOL.
    for _ in 0..MAX_ITERS {
        let mut off = 0.0;
        for p in 0..n {
            for q_idx in p + 1..n {
                off += a[p][q_idx].abs();
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
    // The MATLAB form `+(sum_pos==0)` adds 1 to the denominator iff
    // sum_pos is exactly 0 — equivalent to "no rescale" in that case,
    // but it also forces L_new = 0 matrix which is singular. We
    // tolerate that via EIG_FLOOR below.
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

    // C_u_new = Q · diag(lambda) · Qᵀ — but we never materialize it;
    // we go straight to C_u_inv = Q · diag(1/lambda) · Qᵀ.
    let mut inv_lambda = [0.0_f64; 10];
    for i in 0..n {
        if lambda[i] > EIG_FLOOR {
            inv_lambda[i] = 1.0 / lambda[i];
        } else {
            // Singular — return a large but finite inverse. The Python
            // reference would have crashed here; clamping gives a
            // deterministic output for degenerate inputs (flat patches
            // etc.) without hiding the underlying degeneracy.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_inverse_roundtrip() {
        // Diagonal matrix with positive eigenvalues.
        let c = vec![4.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 16.0];
        let r = decompose_and_invert(&c, 3);
        // Expect lambda = [4, 9, 16] in some order (Jacobi doesn't sort).
        let mut sorted: Vec<f32> = r.lambda.iter().take(3).copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 4.0).abs() < 1e-5);
        assert!((sorted[1] - 9.0).abs() < 1e-5);
        assert!((sorted[2] - 16.0).abs() < 1e-5);
        // Inverse diagonal = [1/4, 1/9, 1/16] (some permutation; but
        // for diagonal input Q = I and inverse is diagonal too).
        assert!((r.c_u_inv[0] - 0.25).abs() < 1e-5);
        assert!((r.c_u_inv[4] - 1.0 / 9.0).abs() < 1e-5);
        assert!((r.c_u_inv[8] - 1.0 / 16.0).abs() < 1e-5);
    }

    #[test]
    fn diagonal_3x3() {
        identity_inverse_roundtrip();
    }

    #[test]
    fn negative_eigenvalue_clipped() {
        // [[2, 1], [1, -1]] — eigenvalues (1 ± √5)/2 + 1/2 = ~2.3 and ~-1.3.
        // After PSD cleaning, only the positive is kept and rescaled
        // so trace == trace(original) == 1.0.
        let c = vec![2.0, 1.0, 1.0, -1.0];
        let r = decompose_and_invert(&c, 2);
        let lam: Vec<f32> = r.lambda.iter().take(2).copied().collect();
        let positives: Vec<f32> = lam.iter().filter(|&&v| v > 0.0).copied().collect();
        assert_eq!(positives.len(), 1);
        // The single positive equals trace(original) = 1.0 after rescale.
        assert!((positives[0] - 1.0).abs() < 1e-4);
    }
}
