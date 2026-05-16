//! Invariant pins on the [`CsfChannel`] enum's discriminants and
//! trait contract. The kernel code in `csf.rs` and many call sites in
//! `pipeline.rs` index per-channel buffers as `channel as usize`,
//! relying on:
//!
//! - `A = 0`, `Rg = 1`, `Vy = 2` discriminants (matches the [A, Rg, Vy]
//!   ordering used by every other [f32; 3] in the crate).
//! - Copy + Clone (for ergonomic value-passing without lifetime fuss).
//! - PartialEq + Eq (used by tests + the `match cc` patterns).
//! - Debug (printable in error messages).
//!
//! No prior test pins these — a refactor that, for example, reorders
//! the variants (`Rg = 0, A = 1, Vy = 2`) would silently shift every
//! per-channel buffer index and corrupt the CSF stage.

use cvvdp_gpu::kernels::csf::CsfChannel;

#[test]
fn discriminants_are_zero_one_two() {
    // Pin via `as u32`. Reordering the variants would shift these.
    assert_eq!(CsfChannel::A as u32, 0, "A must be 0");
    assert_eq!(CsfChannel::Rg as u32, 1, "Rg must be 1");
    assert_eq!(CsfChannel::Vy as u32, 2, "Vy must be 2");
}

#[test]
fn discriminants_fit_in_usize_for_array_indexing() {
    // Some call sites use `channel as usize`. Pin that the values
    // fit in [0, N_CHANNELS) so indexing into [_; 3] arrays is sound.
    let n_channels = cvvdp_gpu::N_CHANNELS;
    for cc in [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy] {
        let idx = cc as usize;
        assert!(
            idx < n_channels,
            "{cc:?} as usize = {idx} ≥ N_CHANNELS = {n_channels}"
        );
    }
}

#[test]
fn copy_semantics_work() {
    // Copy means a value can be used after being passed by value.
    // Pin so a refactor that drops the `Copy` derive trips here.
    let cc = CsfChannel::A;
    let _moved = cc; // Copy, not move
    let _still_usable = cc; // would fail compile if Copy were dropped
    assert_eq!(cc, CsfChannel::A);
}

#[test]
fn clone_yields_equal_value() {
    // Clone produces an Eq-equal value. Catches a refactor that
    // accidentally derives PartialEq but not Clone.
    let cc = CsfChannel::Rg;
    let cloned = cc.clone();
    assert_eq!(cloned, cc);
}

#[test]
fn partial_eq_holds_across_variants() {
    // Self-equality and inequality with other variants.
    assert_eq!(CsfChannel::A, CsfChannel::A);
    assert_eq!(CsfChannel::Rg, CsfChannel::Rg);
    assert_eq!(CsfChannel::Vy, CsfChannel::Vy);
    assert_ne!(CsfChannel::A, CsfChannel::Rg);
    assert_ne!(CsfChannel::A, CsfChannel::Vy);
    assert_ne!(CsfChannel::Rg, CsfChannel::Vy);
}

#[test]
fn debug_output_is_non_empty_and_unique() {
    // Debug derives include the variant name — used by every test's
    // assertion message. Pin so the output contains the variant name
    // string and is distinct across variants.
    let a_dbg = format!("{:?}", CsfChannel::A);
    let rg_dbg = format!("{:?}", CsfChannel::Rg);
    let vy_dbg = format!("{:?}", CsfChannel::Vy);

    assert!(a_dbg.contains("A"), "Debug A = {a_dbg:?}");
    assert!(rg_dbg.contains("Rg"), "Debug Rg = {rg_dbg:?}");
    assert!(vy_dbg.contains("Vy"), "Debug Vy = {vy_dbg:?}");
    assert_ne!(a_dbg, rg_dbg);
    assert_ne!(a_dbg, vy_dbg);
    assert_ne!(rg_dbg, vy_dbg);
}

#[test]
fn match_exhaustiveness_covers_all_three_variants() {
    // A compile-time + runtime check: a match on CsfChannel hits
    // exactly 3 branches. The runtime counter pins that all 3
    // variants are constructible.
    let mut seen = 0;
    for cc in [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy] {
        match cc {
            CsfChannel::A => seen += 1,
            CsfChannel::Rg => seen += 1,
            CsfChannel::Vy => seen += 1,
        }
    }
    assert_eq!(seen, 3, "exhaustive match did not visit 3 variants");
}
