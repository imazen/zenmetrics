//! Additional pins on `CvvdpParams::PLACEHOLDER`'s csf / masking /
//! pooling / jod sub-bundles. The existing `params_placeholder.rs`
//! covers the display + perf_mode fields; this file pins the
//! remaining scaffolding-but-public fields.
//!
//! Per the source docstring, these fields are **unused scaffolding**
//! — the production code reads from `kernels::*` consts. Pinning
//! their current PLACEHOLDER values matters because:
//!
//! 1. They are public constructor inputs — a caller who passes a
//!    custom `CvvdpParams { ..PLACEHOLDER }` relies on these
//!    defaults staying stable.
//! 2. Future work to actually wire them through must replace these
//!    values with real cvvdp v0.5.4 numbers; a silent change before
//!    that wire-through would mislead callers.
//! 3. The fields have to compile correctly even though they're
//!    unused — this also serves as a build-shape regression.

use cvvdp_gpu::CvvdpParams;

#[test]
fn placeholder_csf_fields_are_zero() {
    // CsfParams placeholder values are all 0.0 — the production
    // CSF stage reads from the per-channel LUTs in csf_lut/, not
    // from this struct. Pin via `to_bits()` so any future change
    // (which should also wire the fields through in the same PR)
    // surfaces explicitly.
    let p = CvvdpParams::PLACEHOLDER;
    assert_eq!(
        p.csf.a_peak.to_bits(),
        0.0_f32.to_bits(),
        "PLACEHOLDER.csf.a_peak = {}, expected 0.0",
        p.csf.a_peak
    );
    assert_eq!(
        p.csf.rg_peak.to_bits(),
        0.0_f32.to_bits(),
        "PLACEHOLDER.csf.rg_peak = {}, expected 0.0",
        p.csf.rg_peak
    );
    assert_eq!(
        p.csf.vy_peak.to_bits(),
        0.0_f32.to_bits(),
        "PLACEHOLDER.csf.vy_peak = {}, expected 0.0",
        p.csf.vy_peak
    );
}

#[test]
fn placeholder_masking_fields_match_scaffolded_values() {
    // The MaskingParams placeholder values (p=2.4, q=2.2, k=0.04)
    // do NOT match the production kernels::masking constants
    // (MASK_P=2.264, MASK_Q=[1.30, 2.89, 3.68], no single k). They
    // are scaffolding placeholders — but stable scaffolding pins
    // are still valuable.
    let p = CvvdpParams::PLACEHOLDER;
    assert_eq!(
        p.masking.p.to_bits(),
        2.4_f32.to_bits(),
        "PLACEHOLDER.masking.p = {}, expected 2.4 (scaffolding)",
        p.masking.p
    );
    assert_eq!(
        p.masking.q.to_bits(),
        2.2_f32.to_bits(),
        "PLACEHOLDER.masking.q = {}, expected 2.2 (scaffolding)",
        p.masking.q
    );
    assert_eq!(
        p.masking.k.to_bits(),
        0.04_f32.to_bits(),
        "PLACEHOLDER.masking.k = {}, expected 0.04 (scaffolding)",
        p.masking.k
    );
}

#[test]
fn placeholder_pooling_fields_match_scaffolded_values() {
    // PoolingParams placeholder is uniform 4.0 for all three betas.
    // The production kernels::pool has BETA_SPATIAL=2.0,
    // BETA_BAND=4.0, BETA_CH=4.0 — they don't all agree, so the
    // placeholder is intentionally a scaffolding placeholder.
    let p = CvvdpParams::PLACEHOLDER;
    assert_eq!(
        p.pooling.beta_spatial.to_bits(),
        4.0_f32.to_bits(),
        "PLACEHOLDER.pooling.beta_spatial = {}, expected 4.0",
        p.pooling.beta_spatial
    );
    assert_eq!(
        p.pooling.beta_band.to_bits(),
        4.0_f32.to_bits(),
        "PLACEHOLDER.pooling.beta_band = {}, expected 4.0",
        p.pooling.beta_band
    );
    assert_eq!(
        p.pooling.beta_channel.to_bits(),
        4.0_f32.to_bits(),
        "PLACEHOLDER.pooling.beta_channel = {}, expected 4.0",
        p.pooling.beta_channel
    );
}

#[test]
fn placeholder_jod_fields_match_scaffolded_values() {
    // JodParams placeholder (jod_a=10.0, jod_b=1.0, jod_c=0.30) does
    // NOT match the production kernels::pool met2jod constants
    // (JOD_A=0.044, JOD_EXP=0.93, no jod_b/jod_c). Again pure
    // scaffolding — pin so a refactor that flips wiring on (which
    // would then need real values) trips here as a flag.
    let p = CvvdpParams::PLACEHOLDER;
    assert_eq!(
        p.jod.jod_a.to_bits(),
        10.0_f32.to_bits(),
        "PLACEHOLDER.jod.jod_a = {}, expected 10.0",
        p.jod.jod_a
    );
    assert_eq!(
        p.jod.jod_b.to_bits(),
        1.0_f32.to_bits(),
        "PLACEHOLDER.jod.jod_b = {}, expected 1.0",
        p.jod.jod_b
    );
    assert_eq!(
        p.jod.jod_c.to_bits(),
        0.30_f32.to_bits(),
        "PLACEHOLDER.jod.jod_c = {}, expected 0.30",
        p.jod.jod_c
    );
}

#[test]
fn placeholder_can_be_used_in_struct_update_syntax() {
    // The struct must support `CvvdpParams { ..PLACEHOLDER }` —
    // i.e. all fields are accessible AND the struct is Copy. This
    // is a compile-time check; the runtime body just exercises the
    // pattern.
    let custom = CvvdpParams {
        perf_mode: cvvdp_gpu::PerfMode::Fast,
        ..CvvdpParams::PLACEHOLDER
    };
    assert_eq!(custom.perf_mode, cvvdp_gpu::PerfMode::Fast);
    assert_eq!(
        custom.display.y_peak.to_bits(),
        CvvdpParams::PLACEHOLDER.display.y_peak.to_bits()
    );
}
