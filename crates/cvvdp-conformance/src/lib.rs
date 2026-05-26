//! cvvdp conformance harness — validates BOTH `cvvdp-cpu` and
//! `cvvdp-gpu` against the canonical pycvvdp v0.5.4 reference across a
//! matrix of display models × content/distortion situations.
//!
//! This crate is the authoritative "are our cvvdp impls correct?"
//! gate. The thin end-to-end `1e-4 JOD` check on a single 4K image
//! that previously stood in for conformance could MASK a per-display
//! or per-content divergence (the metric's pooling/masking can absorb
//! a localized error without moving the final JOD). The conformance
//! matrix scores every `(impl × display × situation)` cell against the
//! pycvvdp reference and quantifies the deltas.
//!
//! # Layout
//!
//! - [`situations`] — the deterministic `(ref, dist)` corpus.
//! - [`displays`] — the upstream display-model selection.
//! - `bin/emit_situations` — writes the situation PNGs + a manifest
//!   the Python golden builder consumes.
//! - `tests/conformance.rs` (feature `conformance-goldens`) — fetches
//!   the pycvvdp goldens from R2, scores both impls, writes the
//!   result TSV, and asserts the per-cell tolerance.
//!
//! # Why a shared situation generator
//!
//! Both the Rust scorers and the Python golden builder must see the
//! EXACT SAME bytes. Rather than re-implement the synthetic patterns
//! in Python (and risk a subtle divergence), `emit_situations` writes
//! every situation's reference + distorted halves to PNG, and the
//! Python builder scores those files verbatim. The Rust harness loads
//! the same situations in-process (the generator is deterministic, so
//! the in-process bytes are bit-identical to what was emitted).

pub mod displays;
pub mod situations;

pub use displays::{ConformanceDisplay, conformance_displays};
pub use situations::{Situation, SituationClass, all_situations};

/// The pinned pycvvdp reference version this conformance matrix
/// validates against. Sourced transitively from the cvvdp crates.
pub const PYCVVDP_REFERENCE_VERSION: &str = cvvdp_cpu::PYCVVDP_REFERENCE_VERSION;

/// Per-cell JOD-delta tolerance. A cell PASSES when
/// `|jod_impl - jod_ref| <= TOLERANCE_JOD` for both impls. Matches
/// the documented cvvdp parity tolerance (`docs/CVVDP_CONFORMANCE.md`
/// records the rationale + any per-cell exceptions).
pub const TOLERANCE_JOD: f64 = 1e-3;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_meets_acceptance_cardinality() {
        // Acceptance gate (b): >= 8 display models x >= 15 situations.
        let displays = conformance_displays();
        let situations = all_situations();
        assert!(
            displays.len() >= 8,
            "need >= 8 display models, have {}",
            displays.len()
        );
        assert!(
            situations.len() >= 15,
            "need >= 15 situations, have {}",
            situations.len()
        );
    }

    #[test]
    fn situations_are_well_formed() {
        for s in all_situations() {
            assert!(s.width >= 8 && s.height >= 8, "{} too small", s.name);
            let expect = (s.width * s.height * 3) as usize;
            assert_eq!(s.reference.len(), expect, "{} ref len", s.name);
            assert_eq!(s.distorted.len(), expect, "{} dist len", s.name);
        }
    }

    #[test]
    fn situation_names_unique() {
        let mut names: Vec<&str> = all_situations().iter().map(|s| s.name).collect();
        let n = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), n, "duplicate situation name");
    }

    #[test]
    fn display_names_unique_and_known_upstream() {
        let mut names: Vec<&str> = conformance_displays()
            .iter()
            .map(|d| d.upstream_name)
            .collect();
        let n = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), n, "duplicate display name");

        // Every selected display MUST resolve in our by_name registry
        // (so cvvdp-cpu/gpu can be configured identically to pycvvdp).
        for d in conformance_displays() {
            assert!(
                cvvdp_gpu::params::DisplayModel::by_name(d.upstream_name).is_some(),
                "display {} not in by_name registry",
                d.upstream_name
            );
            assert!(
                cvvdp_gpu::params::DisplayGeometry::by_name(d.upstream_name).is_some(),
                "display geometry {} not in by_name registry",
                d.upstream_name
            );
        }
    }
}
