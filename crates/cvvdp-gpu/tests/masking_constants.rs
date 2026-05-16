//! Pin the exact f32 bit patterns of the cvvdp v0.5.4 masking
//! constants. Sibling to ticks 393 (pool) / 394 (csf) / 395
//! (pyramid) / 396 (display) / 397 (color matrix). A silent edit
//! (typo, sign flip, decimal-point shift) cascades into JOD drift
//! across every parity gate; this pin trips with a specific
//! constant name + expected value.
//!
//! Lives in a dedicated file (not `masking_scalar.rs`) because the
//! latter has historically been edge-case sensitive to linter
//! reverts — keeping the consts pin standalone is more durable.
#![allow(clippy::excessive_precision)]

use cvvdp_gpu::kernels::masking::{
    CH_GAIN, D_MAX, MASK_C, MASK_P, MASK_Q, PU_BLUR_KERNEL_1D, XCM_3X3,
};

// Tick 549: compile-time pin of the PU blur kernel's tap count.
// pycvvdp's σ=3 blur uses 13 taps (precomputed from 3σ truncation
// rounded up + symmetric padding); the per-element expected[] array
// below assumes that exact count. A refactor changing the truncation
// rule (e.g. 4σ → 25 taps) would silently desynchronise the
// expected[] table; this assert forces a compile-time stop instead.
// Same pattern as ticks 522-524 + 548.
const _: () = assert!(
    PU_BLUR_KERNEL_1D.len() == 13,
    "PU_BLUR_KERNEL_1D length drifted from canonical 13 taps (σ=3, 3σ-trunc)",
);

// Tick 556: compile-time bit-pins for the 3 scalar masking
// constants. `f32::to_bits` is const fn since Rust 1.83 (workspace
// MSRV 1.93). A silent drift on any of these would cascade into
// JOD differences across every parity gate — compile-time
// enforcement catches the drift before any test runs.
//
// Tick 558: extended to cover the 3-entry arrays CH_GAIN (the
// per-channel masking gain — RG gets 1.45× boost vs A/VY) and
// MASK_Q (per-channel masking exponents — three distinct values
// where a typo swapping any pair would visibly shift chroma
// masking). Each entry pinned independently.
const _: () = assert!(
    MASK_P.to_bits() == 2.264_355_2_f32.to_bits(),
    "MASK_P drifted from cvvdp v0.5.4 = 2.264_355_2",
);
const _: () = assert!(
    MASK_C.to_bits() == (-0.795_497_12_f32).to_bits(),
    "MASK_C drifted from cvvdp v0.5.4 = -0.795_497_12 (sign flip would amplify masking 6×)",
);
const _: () = assert!(
    D_MAX.to_bits() == 2.564_245_5_f32.to_bits(),
    "D_MAX drifted from cvvdp v0.5.4 = 2.564_245_5 (50% clamp at 10^D_MAX ≈ 366)",
);
const _: () = assert!(
    CH_GAIN[0].to_bits() == 1.0_f32.to_bits(),
    "CH_GAIN[A] drifted from 1.0",
);
const _: () = assert!(
    CH_GAIN[1].to_bits() == 1.45_f32.to_bits(),
    "CH_GAIN[Rg] drifted from 1.45 (cvvdp v0.5.4 RG masking-gain boost)",
);
const _: () = assert!(
    CH_GAIN[2].to_bits() == 1.0_f32.to_bits(),
    "CH_GAIN[Vy] drifted from 1.0",
);
const _: () = assert!(
    MASK_Q[0].to_bits() == 1.302_622_7_f32.to_bits(),
    "MASK_Q[A] drifted from cvvdp v0.5.4 = 1.302_622_7",
);
const _: () = assert!(
    MASK_Q[1].to_bits() == 2.888_590_8_f32.to_bits(),
    "MASK_Q[Rg] drifted from cvvdp v0.5.4 = 2.888_590_8",
);
const _: () = assert!(
    MASK_Q[2].to_bits() == 3.680_771_3_f32.to_bits(),
    "MASK_Q[Vy] drifted from cvvdp v0.5.4 = 3.680_771_3",
);

// Tick 559: compile-time bit-pins for the XCM_3X3 cross-channel
// masking matrix (3×3 = 9 entries). Each entry is independently
// derived from a published log2-space coefficient via 2^x; a
// re-derivation that rounds differently would surface here at
// compile time rather than during a parity-test run.
const _: () = assert!(
    XCM_3X3[0][0].to_bits() == 0.876_968_f32.to_bits(),
    "XCM_3X3[0][0] drifted from cvvdp v0.5.4 = 0.876_968",
);
const _: () = assert!(
    XCM_3X3[0][1].to_bits() == 0.016_103_15_f32.to_bits(),
    "XCM_3X3[0][1] drifted from cvvdp v0.5.4 = 0.016_103_15",
);
const _: () = assert!(
    XCM_3X3[0][2].to_bits() == 0.050_159_38_f32.to_bits(),
    "XCM_3X3[0][2] drifted from cvvdp v0.5.4 = 0.050_159_38",
);
const _: () = assert!(
    XCM_3X3[1][0].to_bits() == 5.918_792_f32.to_bits(),
    "XCM_3X3[1][0] drifted from cvvdp v0.5.4 = 5.918_792",
);
const _: () = assert!(
    XCM_3X3[1][1].to_bits() == 1.269_323_f32.to_bits(),
    "XCM_3X3[1][1] drifted from cvvdp v0.5.4 = 1.269_323",
);
const _: () = assert!(
    XCM_3X3[1][2].to_bits() == 0.152_080_92_f32.to_bits(),
    "XCM_3X3[1][2] drifted from cvvdp v0.5.4 = 0.152_080_92",
);
const _: () = assert!(
    XCM_3X3[2][0].to_bits() == 14.041_055_f32.to_bits(),
    "XCM_3X3[2][0] drifted from cvvdp v0.5.4 = 14.041_055",
);
const _: () = assert!(
    XCM_3X3[2][1].to_bits() == 0.498_209_6_f32.to_bits(),
    "XCM_3X3[2][1] drifted from cvvdp v0.5.4 = 0.498_209_6",
);
const _: () = assert!(
    XCM_3X3[2][2].to_bits() == 0.697_756_55_f32.to_bits(),
    "XCM_3X3[2][2] drifted from cvvdp v0.5.4 = 0.697_756_55",
);

#[test]
fn masking_constants_match_pycvvdp_v0_5_4() {
    // CH_GAIN: per-channel gain multiplier inside the masking
    // pipeline. cvvdp v0.5.4 weights chroma differently — RG
    // gets 1.45x relative to A and VY. A swap with A or VY
    // would mute the chrominance contribution to D.
    let ch_gain_expected: [f32; 3] = [1.0, 1.45, 1.0];
    for (i, (got, exp)) in CH_GAIN.iter().zip(ch_gain_expected.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            exp.to_bits(),
            "CH_GAIN[{i}] = {got}, expected {exp} (cvvdp v0.5.4)",
        );
    }

    // MASK_P: exponent in the masking transducer.
    assert_eq!(
        MASK_P.to_bits(),
        2.264_355_2_f32.to_bits(),
        "MASK_P = {MASK_P}, expected 2.264_355_2 (cvvdp v0.5.4)",
    );

    // MASK_Q: per-channel masking exponent. Three distinct
    // values — a typo that copies index 0 into 1 or 2 would
    // visibly shift chrominance masking.
    let mask_q_expected: [f32; 3] = [1.302_622_7, 2.888_590_8, 3.680_771_3];
    for (i, (got, exp)) in MASK_Q.iter().zip(mask_q_expected.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            exp.to_bits(),
            "MASK_Q[{i}] = {got}, expected {exp} (cvvdp v0.5.4)",
        );
    }

    // MASK_C: phase-uncertainty scaling exponent, applied as
    // `10^MASK_C`. Negative dB-style attenuator: 10^-0.7955
    // ≈ 0.16. A sign flip would amplify masking by ~6×.
    assert_eq!(
        MASK_C.to_bits(),
        (-0.795_497_12_f32).to_bits(),
        "MASK_C = {MASK_C}, expected -0.795_497_12 (cvvdp v0.5.4)",
    );

    // D_MAX: soft-clamp ceiling exponent, applied as `10^D_MAX`.
    // The clamp's 50%-point sits at d = 10^D_MAX (≈ 366).
    assert_eq!(
        D_MAX.to_bits(),
        2.564_245_5_f32.to_bits(),
        "D_MAX = {D_MAX}, expected 2.564_245_5 (cvvdp v0.5.4)",
    );
}

#[test]
fn xcm_3x3_matches_pycvvdp_v0_5_4() {
    // The 3×3 cross-channel masking matrix from cvvdp v0.5.4
    // `xcm_weights` (16 values reshaped 4×4, first 3 rows ×
    // 3 cols, elementwise `2^x`). Each entry is independently
    // derived from a published log2-space coefficient — a
    // refactor that re-derives but rounds differently would
    // surface here.
    let xcm_expected: [[f32; 3]; 3] = [
        [0.876_968, 0.016_103_15, 0.050_159_38],
        [5.918_792, 1.269_323, 0.152_080_92],
        [14.041_055, 0.498_209_6, 0.697_756_55],
    ];
    for row in 0..3 {
        for col in 0..3 {
            assert_eq!(
                XCM_3X3[row][col].to_bits(),
                xcm_expected[row][col].to_bits(),
                "XCM_3X3[{row}][{col}] = {}, expected {}",
                XCM_3X3[row][col],
                xcm_expected[row][col],
            );
        }
    }
}

#[test]
fn pu_blur_kernel_matches_torchvision_gaussianblur_13_3() {
    // 1D phase-uncertainty blur kernel — 13 taps, σ=3 px,
    // matches `torchvision.transforms.GaussianBlur(13, 3.0)`.
    // Pin each tap by bit pattern AND check structural
    // invariants (sum to 1.0, symmetric around the centre).
    let expected: [f32; 13] = [
        1.854_402_2e-2,
        3.416_694_2e-2,
        5.633_176_4e-2,
        8.310_854e-2,
        1.097_193e-1,
        1.296_180_3e-1,
        1.370_228_2e-1,
        1.296_180_3e-1,
        1.097_193e-1,
        8.310_854e-2,
        5.633_176_4e-2,
        3.416_694_2e-2,
        1.854_402_2e-2,
    ];
    assert_eq!(PU_BLUR_KERNEL_1D.len(), 13);
    for (i, (got, exp)) in PU_BLUR_KERNEL_1D.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            exp.to_bits(),
            "PU_BLUR_KERNEL_1D[{i}] = {got}, expected {exp}",
        );
    }
    // Sum-to-1 DC preservation invariant. Use abs-diff < 1e-6
    // because the 13-tap float sum carries rounding noise.
    let sum: f32 = PU_BLUR_KERNEL_1D.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "PU_BLUR_KERNEL_1D sum = {sum}, expected ≈ 1.0 (DC preservation)",
    );
    // Symmetry around the centre tap (index 6).
    for offset in 1..=6 {
        let left = PU_BLUR_KERNEL_1D[6 - offset];
        let right = PU_BLUR_KERNEL_1D[6 + offset];
        assert_eq!(
            left.to_bits(),
            right.to_bits(),
            "PU_BLUR_KERNEL_1D not symmetric at offset ±{offset}: [{}]={} vs [{}]={}",
            6 - offset,
            left,
            6 + offset,
            right,
        );
    }
}
