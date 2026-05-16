//! Invariant pins on [`PerfMode`]'s trait contract + default
//! behavior. The existing `params_placeholder.rs` test pins that
//! `CvvdpParams::PLACEHOLDER.perf_mode == PerfMode::Strict`, but
//! doesn't pin:
//!
//! - The `#[derive(Default)]` value is `Strict` (the parity-
//!   calibrated baseline). A refactor that swapped the `#[default]`
//!   attribute to `Fast` would silently shift `Default::default()`
//!   but the existing test only checks `PLACEHOLDER.perf_mode`.
//! - Copy/Clone/PartialEq/Eq/Debug derives behave correctly.
//! - Strict != Fast (catches a refactor that accidentally collapses
//!   the variants).
//! - Exhaustive match visits exactly 2 variants.

use cvvdp_gpu::PerfMode;

#[test]
fn default_is_strict() {
    // The #[default] attribute on the Strict variant means
    // PerfMode::default() == Strict. Pin this explicitly so a
    // refactor that moves #[default] to Fast trips here even when
    // PLACEHOLDER also gets updated.
    assert_eq!(PerfMode::default(), PerfMode::Strict);
}

#[test]
fn copy_semantics_work() {
    // Copy means a value can be used after being passed by value.
    let mode = PerfMode::Strict;
    let _moved = mode; // Copy, not move
    let _still_usable = mode; // would fail compile if Copy were dropped
    assert_eq!(mode, PerfMode::Strict);
}

#[test]
fn clone_yields_equal_value() {
    let mode = PerfMode::Fast;
    let cloned = mode.clone();
    assert_eq!(cloned, mode);
}

#[test]
fn strict_and_fast_are_distinct() {
    // The two variants must not collapse — a refactor that maps
    // both to the same discriminant or removes one would be
    // silent without this pin.
    assert_ne!(PerfMode::Strict, PerfMode::Fast);
    assert_eq!(PerfMode::Strict, PerfMode::Strict);
    assert_eq!(PerfMode::Fast, PerfMode::Fast);
}

#[test]
fn debug_output_is_non_empty_and_distinct() {
    let strict = format!("{:?}", PerfMode::Strict);
    let fast = format!("{:?}", PerfMode::Fast);
    assert!(strict.contains("Strict"), "Debug Strict = {strict:?}");
    assert!(fast.contains("Fast"), "Debug Fast = {fast:?}");
    assert_ne!(strict, fast);
}

#[test]
fn match_exhaustiveness_covers_both_variants() {
    // Compile-time + runtime check: a match on PerfMode has
    // exactly 2 branches. Both variants must be constructible.
    let mut seen = 0;
    for mode in [PerfMode::Strict, PerfMode::Fast] {
        match mode {
            PerfMode::Strict => seen += 1,
            PerfMode::Fast => seen += 1,
        }
    }
    assert_eq!(seen, 2, "exhaustive match did not visit 2 variants");
}
