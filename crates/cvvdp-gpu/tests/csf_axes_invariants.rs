//! Invariant pins on the public CSF LUT axis arrays
//! [`LOG_L_BKG_AXIS`] and [`LOG_RHO_AXIS`]. These 32-entry arrays
//! are the axes against which `interp1_uniform` and `interp1_clamped`
//! query the per-channel `LOG_S_O0_C*` sensitivity LUTs. A refactor
//! that shifts an endpoint, breaks monotonicity, or changes uniform
//! spacing on `LOG_L_BKG_AXIS` (which `interp1_uniform` requires)
//! would corrupt every CSF query.
//!
//! Existing csf_constants_match_pycvvdp_v0_5_4 pins individual
//! entries (length, scalar constants); this file pins the AXES'
//! structural properties separately.

#![allow(clippy::excessive_precision)]
// Intentional: f32 literals carry their pycvvdp f64 source values
// verbatim so the digits document the source even though LLVM
// rounds to f32 at compile time.

use cvvdp_gpu::kernels::csf::{LOG_L_BKG_AXIS, LOG_RHO_AXIS, N_L_BKG, N_RHO};

// Tick 548: compile-time enforcement of the CSF LUT axis dimensions.
// These mirror the runtime test fns below — promoting the most
// load-bearing invariants (length match + canonical value) to
// `const _: () = assert!(...)` so a refactor that changes either
// constant fails to compile rather than just failing at test time.
// Same pattern as ticks 522-524 (lib_constants.rs PYRAMID_MIN_DIM
// and CsfChannel discriminants).
const _: () = assert!(
    LOG_L_BKG_AXIS.len() == N_L_BKG,
    "LOG_L_BKG_AXIS.len() must equal N_L_BKG (CSF luminance-axis LUT size)",
);
const _: () = assert!(N_L_BKG == 32, "N_L_BKG drifted from canonical pycvvdp value 32");
const _: () = assert!(
    LOG_RHO_AXIS.len() == N_RHO,
    "LOG_RHO_AXIS.len() must equal N_RHO (CSF frequency-axis LUT size)",
);
const _: () = assert!(N_RHO == 32, "N_RHO drifted from canonical pycvvdp value 32");

// Tick 562: compile-time bit-pins for the LUT-axis endpoints.
// pycvvdp's `csf_lut_weber_fixed_size.json` defines the queryable
// range of the CSF LUT — drifting either endpoint silently shifts
// every CSF lookup. Pin endpoints at compile time so the change
// surfaces before any per-pixel kernel runs.
//
// LOG_L_BKG_AXIS: log10(0.005) = -2.3010299957 → log10(1e4) = 4.0
// LOG_RHO_AXIS:   log10(0.1) = -1.0 → log10(64) = 1.8061799740
const _: () = assert!(
    LOG_L_BKG_AXIS[0].to_bits() == (-2.3010299957_f32).to_bits(),
    "LOG_L_BKG_AXIS[0] drifted from cvvdp v0.5.4 log10(0.005) = -2.301",
);
const _: () = assert!(
    LOG_L_BKG_AXIS[31].to_bits() == 4.0_f32.to_bits(),
    "LOG_L_BKG_AXIS[31] drifted from cvvdp v0.5.4 log10(1e4) = 4.0",
);
const _: () = assert!(
    LOG_RHO_AXIS[0].to_bits() == (-1.0_f32).to_bits(),
    "LOG_RHO_AXIS[0] drifted from cvvdp v0.5.4 log10(0.1) = -1.0",
);
const _: () = assert!(
    LOG_RHO_AXIS[31].to_bits() == 1.8061799740_f32.to_bits(),
    "LOG_RHO_AXIS[31] drifted from cvvdp v0.5.4 log10(64) ≈ 1.806_180",
);

// Tick 571: per-channel sensitivity LUTs (LOG_S_O0_C1/C2/C3) must
// have exactly N_L_BKG × N_RHO = 32 × 32 = 1024 entries. The CSF
// kernel indexes these via `idx = l_bkg_i * N_RHO + rho_i` so the
// length is load-bearing — a size mismatch silently corrupts every
// per-pixel CSF query. Plus a cross-check pin that all 3 channels
// have matching lengths (refactor that drops a row from one
// channel-LUT but not the others would surface here).
const _: () = {
    use cvvdp_gpu::kernels::csf::{LOG_S_O0_C1, LOG_S_O0_C2, LOG_S_O0_C3};
    assert!(
        LOG_S_O0_C1.len() == N_L_BKG * N_RHO,
        "LOG_S_O0_C1 length must equal N_L_BKG × N_RHO (32 × 32 = 1024)",
    );
    assert!(
        LOG_S_O0_C2.len() == N_L_BKG * N_RHO,
        "LOG_S_O0_C2 length must equal N_L_BKG × N_RHO (32 × 32 = 1024)",
    );
    assert!(
        LOG_S_O0_C3.len() == N_L_BKG * N_RHO,
        "LOG_S_O0_C3 length must equal N_L_BKG × N_RHO (32 × 32 = 1024)",
    );
    // Cross-channel length consistency (catches a row drop from one
    // channel's LUT even if the absolute length somehow stayed 1024
    // on another via unrelated additions).
    assert!(
        LOG_S_O0_C1.len() == LOG_S_O0_C2.len()
            && LOG_S_O0_C2.len() == LOG_S_O0_C3.len(),
        "All 3 CSF sensitivity LUTs must have matching length",
    );
};

#[test]
fn log_l_bkg_axis_length_matches_n_l_bkg() {
    assert_eq!(LOG_L_BKG_AXIS.len(), N_L_BKG, "axis length != N_L_BKG");
    assert_eq!(N_L_BKG, 32, "N_L_BKG drifted from canonical 32");
}

#[test]
fn log_rho_axis_length_matches_n_rho() {
    assert_eq!(LOG_RHO_AXIS.len(), N_RHO, "axis length != N_RHO");
    assert_eq!(N_RHO, 32, "N_RHO drifted from canonical 32");
}

#[test]
fn log_l_bkg_axis_strictly_monotonic() {
    for i in 1..LOG_L_BKG_AXIS.len() {
        assert!(
            LOG_L_BKG_AXIS[i] > LOG_L_BKG_AXIS[i - 1],
            "non-monotonic at [{i}]: {} <= {}",
            LOG_L_BKG_AXIS[i],
            LOG_L_BKG_AXIS[i - 1]
        );
    }
}

#[test]
fn log_rho_axis_strictly_monotonic() {
    for i in 1..LOG_RHO_AXIS.len() {
        assert!(
            LOG_RHO_AXIS[i] > LOG_RHO_AXIS[i - 1],
            "non-monotonic at [{i}]: {} <= {}",
            LOG_RHO_AXIS[i],
            LOG_RHO_AXIS[i - 1]
        );
    }
}

#[test]
fn log_l_bkg_axis_endpoints_bit_pinned() {
    // Endpoints define the LUT's queryable range. pycvvdp's
    // csf_lut_weber_fixed_size.json sets these to log10(0.005) =
    // -2.301 and 4.0 (= log10(1e4 cd/m²)).
    let first_expected: f32 = -2.3010299957e+00;
    let last_expected: f32 = 4.0;
    assert_eq!(
        LOG_L_BKG_AXIS[0].to_bits(),
        first_expected.to_bits(),
        "LOG_L_BKG_AXIS[0] = {} != {}",
        LOG_L_BKG_AXIS[0],
        first_expected
    );
    assert_eq!(
        LOG_L_BKG_AXIS[31].to_bits(),
        last_expected.to_bits(),
        "LOG_L_BKG_AXIS[31] = {} != {}",
        LOG_L_BKG_AXIS[31],
        last_expected
    );
}

#[test]
fn log_rho_axis_endpoints_bit_pinned() {
    // pycvvdp's rho axis ranges log10(0.1) = -1.0 to log10(64) ≈ 1.806.
    let first_expected: f32 = -1.0;
    let last_expected: f32 = 1.8061799740e+00;
    assert_eq!(
        LOG_RHO_AXIS[0].to_bits(),
        first_expected.to_bits(),
        "LOG_RHO_AXIS[0] = {} != {}",
        LOG_RHO_AXIS[0],
        first_expected
    );
    assert_eq!(
        LOG_RHO_AXIS[31].to_bits(),
        last_expected.to_bits(),
        "LOG_RHO_AXIS[31] = {} != {}",
        LOG_RHO_AXIS[31],
        last_expected
    );
}

#[test]
fn log_l_bkg_axis_uniformly_spaced() {
    // `interp1_uniform` (used by `sensitivity_scalar` for the L_bkg
    // axis) assumes equal spacing. Pin all 31 inter-sample gaps
    // are equal within f32 noise. Without uniform spacing, the
    // L_bkg interpolation produces wrong sensitivities silently.
    let step = LOG_L_BKG_AXIS[1] - LOG_L_BKG_AXIS[0];
    for i in 2..LOG_L_BKG_AXIS.len() {
        let gap = LOG_L_BKG_AXIS[i] - LOG_L_BKG_AXIS[i - 1];
        let rel = ((gap - step) / step).abs();
        assert!(
            rel < 1e-5,
            "LOG_L_BKG_AXIS non-uniform at [{i}]: gap = {gap} vs step = {step} (rel = {rel})"
        );
    }
}

#[test]
fn log_rho_axis_is_uniformly_spaced_in_log10() {
    // The log10-space axis IS uniformly spaced (step ≈ 0.0905). The
    // source comment "first interval has a different ratio (0.3228 vs
    // 0.5)" in sensitivity_scalar refers to LINEAR rho values (= 10^log_rho),
    // not the log10 axis itself: 10^(-1.0)=0.1, 10^(-0.91)=0.123,
    // ratio = 0.1/0.123 ≈ 0.812 vs subsequent equal-log-spaced
    // points all having ratio ~10^(-step) ≈ 0.812. The comment is
    // about why interp1_uniform isn't used on rho — but the AXIS
    // itself is still uniform in log10.
    let step = LOG_RHO_AXIS[1] - LOG_RHO_AXIS[0];
    for i in 2..LOG_RHO_AXIS.len() {
        let gap = LOG_RHO_AXIS[i] - LOG_RHO_AXIS[i - 1];
        let rel = ((gap - step) / step).abs();
        assert!(
            rel < 1e-5,
            "LOG_RHO_AXIS non-uniform at [{i}]: gap = {gap} vs step = {step} (rel = {rel})"
        );
    }
}

#[test]
fn log_l_bkg_axis_step_size_matches_pycvvdp() {
    // The pycvvdp source axis spans -2.3010 to 4.0 with 32 samples,
    // giving step = (4.0 - (-2.3010)) / 31 ≈ 0.2032. Pin this so a
    // refactor that changes axis bounds (without realising it
    // breaks step alignment) trips here. Use a loose tolerance
    // since the source uses excessive precision but f32 representation
    // rounds.
    let step = LOG_L_BKG_AXIS[1] - LOG_L_BKG_AXIS[0];
    let expected_step = (4.0_f32 - (-2.3010299957)) / 31.0_f32;
    let rel = ((step - expected_step) / expected_step).abs();
    assert!(
        rel < 1e-5,
        "LOG_L_BKG_AXIS step = {step}, expected ≈ {expected_step} (rel = {rel})"
    );
}
