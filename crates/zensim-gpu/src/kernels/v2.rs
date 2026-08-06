//! v2 (`Folded720Append`) per-pixel primitives — stage 1 of the 924 port.
//!
//! See `docs/F924_PORT_SPEC.md`. The key fact these functions exist to encode:
//! **v2's per-pixel SSIM is not the one the v1 kernels compute.** `fused.rs`
//! evaluates the SSIMULACRA2 form with no C1 term; v2 evaluates the standard
//! SSIM form WITH C1. The blur planes (`mu1`, `mu2`, `s12`, `ssq`) are shared,
//! the SSIM evaluation is not, so the v2 blocks cannot pool the existing
//! per-pixel `sd` map no matter how convenient that would be.

use cubecl::prelude::*;

/// SSIM luminance stability constant, v2 — CPU `feature_v2::C1_V2`.
/// `(K1*L)^2` with `K1 = 0.01`, `L = 1`. The v1 kernels have no equivalent:
/// their luminance term is the SSIMULACRA2 `1 - (mu1-mu2)^2`.
pub const C1_V2: f32 = 0.0001;

/// SSIM contrast stability constant, v2 — CPU `feature_v2::C2_V2`.
/// Numerically equal to the v1 `C2`, kept as its own name so the two cannot
/// drift silently if either side is retuned.
pub const C2_V2: f32 = 0.0009;

/// Per-pixel v2 SSIM deviation `d`, a line-for-line transcription of CPU
/// `feature_v2::ssim_d_local`:
///
/// ```text
/// a     = 2*mu1*mu2 + C1
/// b     = mu1^2 + mu2^2 + C1
/// c     = 2*(s12 - mu1*mu2) + C2
/// d     = ssq - mu1^2 - mu2^2 + C2
/// out   = max(0, 1 - (a*c)/(b*d))
/// ```
///
/// Written as the plain expression rather than an FMA-fused rearrangement:
/// the CPU reference is scalar `f64` arithmetic in this exact order, and the
/// port's acceptance bar (1.38e-4, what the already-correct v1-basic block
/// achieves) is set by f32-vs-f64 width, not by contraction order. Matching
/// the source's shape keeps the comparison honest; if a later stage needs the
/// FMA form for speed, it should be introduced against a passing gate, not
/// before one exists.
#[cube]
pub fn v2_ssim_d(mu1: f32, mu2: f32, s12: f32, ssq: f32) -> f32 {
    let a = 2.0 * mu1 * mu2 + C1_V2;
    let b = mu1 * mu1 + mu2 * mu2 + C1_V2;
    let cov = s12 - mu1 * mu2;
    let c = 2.0 * cov + C2_V2;
    let d = ssq - mu1 * mu1 - mu2 * mu2 + C2_V2;
    let local = (a * c) / (b * d);
    let out = 1.0 - local;
    if out > 0.0 { out } else { f32::new(0.0_f32) }
}
