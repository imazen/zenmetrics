//! GPU kernels for IW-SSIM.
//!
//! Per-scale data flow:
//!
//! ```text
//! gray u8 ──► f32 gray
//!         └─► Laplacian pyramid (binom5, reflect1) ──► LP[0..4]_ref, LP[0..4]_dist
//!
//!  For each scale j ∈ 0..5:
//!    LP[j]_{ref,dist} ──► 11×11 Gaussian (separable, VALID)
//!                    ├──► μ₁, μ₂
//!                    ├──► σ₁², σ₂², σ₁₂   (= E[x²] − µ₁² etc.)
//!                    ──► cs_map (j ∈ 0..5)  and  l_map (j = 4)
//!
//!  For each scale j ∈ 0..4 (IW path):
//!    LP[j]_{ref,dist} ──► 3×3 box stats → g, vv
//!    LP[j+1]_ref       ──► imenlarge2 → parent band (cropped to LP[j] shape)
//!    LP[j]_ref + parent ──► Y rows → C_u accumulate (NxN, host eigendecomp)
//!    Y + C_u_inv       ──► per-pixel ss
//!    g, vv, ss, λ_k    ──► per-pixel infow
//!
//!  Reduction (per scale):
//!    j < 4: wmcs_j = Σ(cs_j · iw_j) / Σ(iw_j)
//!    j = 4: wmcs_4 = mean(cs_4 · l_4)
//!
//!  Host final:  score = Π |wmcs_j|^{β_j}
//! ```

pub(crate) mod box3;
pub(crate) mod cov;
pub(crate) mod gauss11;
pub(crate) mod imenlarge2;
pub(crate) mod infow;
pub(crate) mod lap_pyramid;
pub(crate) mod neighborhood;
pub(crate) mod reduction;
pub(crate) mod rgb2gray;
pub(crate) mod ssim_combine;
pub(crate) mod util;
