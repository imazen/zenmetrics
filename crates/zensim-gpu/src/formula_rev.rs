//! **Which arithmetic revision this GPU pipeline computes** — a MIRROR of
//! `zensim`'s own revision owner, kept honest by the parity gates.
//!
//! # Why a mirror and not a call
//!
//! `zensim-gpu` is the **fourth hand-copy** of two per-pixel expressions that
//! revision 2 changes (`benchmarks/f4_arm_decision_2026-09-05.md` §10). It
//! cannot call zensim's CPU kernels — the whole point of this crate is that
//! the arithmetic runs on a device — so under NO DUPLICATE IMPLEMENTATIONS it
//! is a **sanctioned gated mirror**: a second implementation that exists for a
//! measured engineering reason, held exact against the owner by
//! `tests/it/cpu_parity.rs`, `tests/it/extended_parity.rs` and
//! `tests/it/cpu_gpu_feature_sweep.rs`.
//!
//! Three facts make this module the mirror rather than a stray duplicate:
//!
//! 1. **It mirrors `zensim::ssim_form::active_revision`**, whose module is
//!    `pub(crate)` in that crate and therefore cannot be called from here.
//!    Same environment variable, same two accepted spellings, same
//!    `OnceLock`-per-process read, same fall-through.
//! 2. **The fallback is hard-coded to [`FormulaRevision::Rev1`]**, which is
//!    zensim's `SHIPPED_REVISION` today. It is a literal here, not a lookup,
//!    because there is no lookup available — so if zensim's shipped revision
//!    ever moves, this line does not move with it.
//! 3. **The parity tests are what catch (2).** A zensim whose default became
//!    revision 2 while this module still defaulted to revision 1 would make
//!    the CPU-vs-GPU feature sweep disagree on the twelve `hf_energy_gain`
//!    slots at every (scale, channel) — which is exactly the comparison those
//!    gates already run. The mirror is allowed to be a copy precisely because
//!    a divergence is a test failure and not a silent wrong number.
//!
//! # What revision 2 changes on this side
//!
//! Two of revision 2's three registered eras have a hand-copy here; the third
//! does not exist in this crate at all:
//!
//! | era | defect | arm | where it lives here |
//! |---|---|---|---|
//! | `v1ssimcap` | F4 — unbounded SSIM luminance `1 − D²` | `Clamp` = `max(0, 1 − D²)` | 4 v1 feature kernels + the diffmap kernel + its host-scalar reference |
//! | `v1hfgain` | F17 — unbounded `contrast_inc` | `SaturatingExcess` = `g/(g+1)` | host-side finalize in [`crate::pipeline`] |
//! | `freecomp` | F5 — free-40 raw-moment route parity | reassociation | **structurally absent** — see below |
//!
//! **F5 is not reachable from this crate.** It lives in the free-40
//! raw-moment route, which only exists at a feature width carrying an append
//! block; [`crate::ZensimFeatureRegime`] tops out at
//! [`WithIw`](crate::ZensimFeatureRegime::WithIw) = 372 features (Basic 228 /
//! Extended 300 / WithIw 372) and this crate has no raw-moment accumulator,
//! no `GLOBAL_*` slots and no append kernel. Pinned by
//! [`tests::f5_free_route_is_structurally_absent`] so a future regime bump
//! cannot quietly acquire the defect without tripping a gate.

/// The arithmetic revision the pipeline computes.
///
/// Mirrors `zensim::feature_defs::FormulaRevision`. Public because
/// [`crate::Zensim::with_formula_revision`] takes it: a test cannot select a
/// revision through the environment (`std::env::set_var` is `unsafe` in
/// edition 2024 and this crate is `#![forbid(unsafe_code)]`-adjacent for
/// everything but the CubeCL launch calls), so the revision has to be
/// settable through the API for the rev2 gates to be runnable at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum FormulaRevision {
    /// The shipped revision: SSIMULACRA2's unbounded `1 − D²` luminance term
    /// and the unbounded `max(0, var_dst/var_src − 1)` HF gain.
    #[default]
    Rev1,
    /// F4's `Clamp` luminance arm + F17's `SaturatingExcess` gain arm.
    Rev2,
}

impl FormulaRevision {
    /// The `luma_clamp` kernel scalar: `1` when the per-pixel SSIM luminance
    /// term is clamped at zero, `0` for the shipped unbounded form.
    ///
    /// Passed to every SSIM kernel as an ordinary `u32` parameter. It is a
    /// plain runtime value on purpose — this crate has never used CubeCL's
    /// `#[comptime]`, and a codegen-affecting construct cannot be validated on
    /// a box with no GPU.
    #[must_use]
    pub const fn luma_clamp(self) -> u32 {
        match self {
            Self::Rev1 => 0,
            Self::Rev2 => 1,
        }
    }

    /// Whether the HF-energy **gain** member uses F17's bounded
    /// `g/(g+1)` arm. Host-side only — the gain is computed in
    /// [`crate::pipeline`]'s finalize from already-reduced sums, not in a
    /// kernel.
    #[must_use]
    pub const fn saturating_hf_gain(self) -> bool {
        matches!(self, Self::Rev2)
    }

    /// Stable lower-case token, matching the `ZENSIM_FORMULA_REV` spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Rev1 => "1",
            Self::Rev2 => "2",
        }
    }
}

/// **The revision this build ships**, mirroring `zensim`'s
/// `ssim_form::SHIPPED_REVISION`.
///
/// A literal, because zensim's constant is `pub(crate)` and cannot be read
/// from here. The mirror's honesty rests on the CPU/GPU parity gates, not on
/// this line staying in sync by inspection — see the module docs.
pub(crate) const SHIPPED_REVISION: FormulaRevision = FormulaRevision::Rev1;

/// **The revision switch — one owner, read once per process.**
///
/// A transcription of `zensim::ssim_form::active_revision`:
/// `ZENSIM_FORMULA_REV=1` pins revision 1, `=2` pins revision 2, and anything
/// else — including unset — is [`SHIPPED_REVISION`]. The two accepted values
/// are the SAME BYTE LENGTH on purpose; zensim has measured an environment
/// block's size shifting a binary's layout by ~10 % at 2304²
/// (`benchmarks/era2_perf_break_2026-08-31.md` §22.5), so an A/B that varies
/// the value must not vary the length.
///
/// Read ONCE at [`crate::Zensim`] construction, never per launch and never
/// per pixel.
#[must_use]
pub fn active_revision() -> FormulaRevision {
    use std::sync::OnceLock;
    static REV: OnceLock<FormulaRevision> = OnceLock::new();
    *REV.get_or_init(|| match std::env::var("ZENSIM_FORMULA_REV").as_deref() {
        Ok("1") => FormulaRevision::Rev1,
        Ok("2") => FormulaRevision::Rev2,
        _ => SHIPPED_REVISION,
    })
}

/// **F17's `contrast_inc` / `hf_energy_gain`** — the one member with an arm.
///
/// Transcribes `zensim::hf_gain_form::hf_energy_gain` for the two arms a
/// registered revision selects. The `var_src` gate, the `.max(0.0)` floor and
/// the `f64` width are the shipped expression's own; only the post-floor map
/// differs.
///
/// * revision 1 — `g = max(0, var_dst/var_src − 1)`, unbounded above.
/// * revision 2 — `g/(g+1)`, i.e. the crate's `saturate(g, 1)` idiom, bounded
///   `[0, 1)`, strictly increasing, and first-order-identical to revision 1.
///
/// `g/(g+1)` is an EXACT closed-form function of the revision-1 value, which
/// is what lets [`tests`](self) and `tests/it/formula_rev_parity.rs` assert
/// the rev2 output against the rev1 output slot-for-slot rather than against a
/// second transcription.
#[must_use]
#[inline]
pub fn hf_energy_gain(rev: FormulaRevision, var_src_sum: f64, var_dst_sum: f64) -> f64 {
    let g = (var_dst_sum / var_src_sum - 1.0).max(0.0);
    if rev.saturating_hf_gain() {
        g / (g + 1.0)
    } else {
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ZensimFeatureRegime;

    /// The expression the host-side finalize carried before F17 got an arm,
    /// transcribed verbatim — the control for the extraction.
    fn legacy_gain(sum_dst: f64, sum_src: f64) -> f64 {
        let r = sum_dst / sum_src;
        (r - 1.0).max(0.0)
    }

    fn ratio_cases() -> Vec<(f64, f64)> {
        let mut v = Vec::new();
        for &src in &[1e-9_f64, 1e-4, 0.25, 1.0, 7.5, 1e4] {
            for &mult in &[0.0_f64, 0.5, 1.0, 1.000001, 2.0, 11.0, 36_466.0] {
                v.push((src, src * mult));
            }
        }
        v
    }

    /// **The extraction gate.** Revision 1 through the new helper must
    /// reproduce the replaced expression BIT-for-BIT — this is what lets the
    /// host-side rewrite claim to be inert.
    #[test]
    fn rev1_gain_is_bit_identical_to_the_expression_it_replaced() {
        for (src, dst) in ratio_cases() {
            let got = hf_energy_gain(FormulaRevision::Rev1, src, dst);
            let want = legacy_gain(dst, src);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "rev1 gain diverged at src={src} dst={dst}: {got} vs {want}"
            );
        }
    }

    /// **F17, stated as a property.** Revision 1 is unbounded; revision 2 is
    /// not, and it is the exact map `g → g/(g+1)` of revision 1.
    #[test]
    fn rev2_gain_is_the_saturating_map_of_rev1_and_is_bounded() {
        let mut saw_positive = false;
        for (src, dst) in ratio_cases() {
            let g1 = hf_energy_gain(FormulaRevision::Rev1, src, dst);
            let g2 = hf_energy_gain(FormulaRevision::Rev2, src, dst);
            assert_eq!(
                g2.to_bits(),
                (g1 / (g1 + 1.0)).to_bits(),
                "rev2 is not the saturating map of rev1 at src={src} dst={dst}"
            );
            assert!(
                (0.0..1.0).contains(&g2),
                "rev2 gain {g2} escaped [0, 1) at src={src} dst={dst}"
            );
            if g1 > 0.0 {
                saw_positive = true;
            }
        }
        // Anti-vacuity: a sweep that never leaves g == 0 would pass every
        // assertion above and prove nothing.
        assert!(saw_positive, "the sweep never entered the changed regime");
        // The headline number from the F17 record: max 36,465.74 over
        // 216,756 real pairs.
        let g1 = hf_energy_gain(FormulaRevision::Rev1, 1.0, 36_466.0);
        assert!(
            g1 > 3.6e4,
            "the F17 pathology should reproduce here, got {g1}"
        );
        assert!(hf_energy_gain(FormulaRevision::Rev2, 1.0, 36_466.0) < 1.0);
    }

    /// Zero-preservation (zensim's gate H4): both arms return exactly 0 when
    /// the distorted term does not exceed the source term, so the identity
    /// vector stays zero on both revisions.
    #[test]
    fn both_arms_preserve_zero() {
        for &(src, dst) in &[(1.0_f64, 1.0_f64), (1.0, 0.5), (7.5, 0.0)] {
            for rev in [FormulaRevision::Rev1, FormulaRevision::Rev2] {
                assert_eq!(
                    hf_energy_gain(rev, src, dst),
                    0.0,
                    "{rev:?} at ({src},{dst})"
                );
            }
        }
    }

    /// The kernel scalar is the only thing the SSIM kernels see, so pin its
    /// two values rather than letting a future edit renumber them.
    #[test]
    fn luma_clamp_scalar_is_pinned() {
        assert_eq!(FormulaRevision::Rev1.luma_clamp(), 0);
        assert_eq!(FormulaRevision::Rev2.luma_clamp(), 1);
        assert!(!FormulaRevision::Rev1.saturating_hf_gain());
        assert!(FormulaRevision::Rev2.saturating_hf_gain());
    }

    /// **G3.2's mirror** — pinning the shipped revision is the same as not
    /// pinning it. Fails loudly if the test process has `ZENSIM_FORMULA_REV`
    /// set, which would silently re-grade every other test in the binary.
    #[test]
    fn selecting_the_shipped_revision_is_a_no_op() {
        assert_eq!(
            active_revision(),
            SHIPPED_REVISION,
            "the active revision is not the shipped one (is ZENSIM_FORMULA_REV set?)"
        );
    }

    /// **F5's structural absence, checked rather than asserted in prose.**
    ///
    /// The `freecomp` era lives in the free-40 raw-moment route, which only
    /// exists at a feature width carrying an append block. This crate's widest
    /// regime is 372, so there is no append block for it to live in. If a
    /// future regime bump raises this ceiling, F5 becomes reachable and this
    /// gate says so before a wave trusts the GPU oracle at that width.
    #[test]
    fn f5_free_route_is_structurally_absent() {
        for regime in [
            ZensimFeatureRegime::Basic,
            ZensimFeatureRegime::Extended,
            ZensimFeatureRegime::WithIw,
        ] {
            assert!(
                regime.total_features() <= crate::TOTAL_FEATURES_WITH_IW,
                "{regime:?} exceeds the 372 ceiling — F5's raw-moment route may now be reachable"
            );
        }
        assert_eq!(crate::TOTAL_FEATURES_WITH_IW, 372);
        assert_eq!(
            ZensimFeatureRegime::default().total_features(),
            crate::TOTAL_FEATURES
        );
    }
}
