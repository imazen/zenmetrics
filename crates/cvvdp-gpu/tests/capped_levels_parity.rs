//! Capped-pyramid-depth parity gates.
//!
//! Validates two contracts for `MemoryMode::Strip { capped_levels:
//! Some(k) }`:
//!
//! 1. **Host-scalar parity at the natural depth.** Calling
//!    `predict_jod_still_3ch_capped` with `cap_levels = None` (or
//!    `Some(k) >= natural_n_levels`) must reproduce
//!    `predict_jod_still_3ch` byte-for-byte. The capped variant is
//!    the same code path with a clamp; without this gate, refactors
//!    to the cap-handling could silently shift the canonical
//!    parity-tested code.
//!
//! 2. **Cap=8 fits the 0.005 JOD gate vs pycvvdp on every measured
//!    fixture.** Per the sweep at
//!    `benchmarks/cvvdp_capped_levels_2026-05-22.csv`, the cap=8
//!    setting keeps the worst-case |jod - pycvvdp_golden| below
//!    0.001 on all 5 fixtures with natural_n_levels = 9. Cap=7 fails
//!    on the 720×1280 fixture (drift 0.0117), so cap=8 is the
//!    deepest cap that ships per the canonical gate.
//!
//! Run with:
//!
//!     cargo test -p cvvdp-gpu --features cubecl-types \
//!         --test capped_levels_parity -- --nocapture
//!
//! This test runs purely on host_scalar so no GPU runtime is needed.
//! A future companion test in `pipeline_color.rs` will pin the GPU
//! path against the same fixtures.

mod common;

use cvvdp_gpu::host_scalar::{predict_jod_still_3ch, predict_jod_still_3ch_capped};
use cvvdp_gpu::kernels::pyramid::band_frequencies;
use cvvdp_gpu::params::{DisplayGeometry, DisplayModel};

const TOLERANCE: f32 = 0.005;

/// Build the canonical (ref, dist) pair from `common::synth_pair_ref`
/// + `common::apply_offset_dist`. Returns (ref_bytes, dist_bytes,
/// w, h, name, pycvvdp_golden_jod).
fn fixtures() -> Vec<(Vec<u8>, Vec<u8>, usize, usize, &'static str, f32)> {
    let mut out = Vec::new();
    let cases = [
        ("synth_128x128_offset", 128usize, 128usize, false),
        ("synth_1024x1024_offset", 1024, 1024, false),
        ("synth_1280x720_offset", 1280, 720, false),
        ("synth_720x1280_offset", 720, 1280, false),
        ("synth_73x91_odd", 73, 91, true),
    ];
    for &(name, w, h, odd_dim) in &cases {
        let r = if odd_dim {
            common::synth_pair_odd_dim_ref(w, h)
        } else {
            common::synth_pair_ref(w, h)
        };
        let d = common::apply_offset_dist(&r);
        let golden = common::pycvvdp_synth_golden_jod(name);
        out.push((r, d, w, h, name, golden));
    }
    out
}

#[test]
fn capped_none_matches_uncapped() {
    // cap_levels = None must produce exactly the uncapped JOD. The
    // capped variant is supposed to be a strict superset — no
    // unconditional changes to the natural-depth code path.
    let display = DisplayModel::STANDARD_4K;
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();

    for (r, d, w, h, name, _golden) in fixtures() {
        let uncapped = predict_jod_still_3ch(&r, &d, w, h, display, ppd);
        let capped_none = predict_jod_still_3ch_capped(&r, &d, w, h, display, ppd, None);
        assert_eq!(
            uncapped, capped_none,
            "{name}: cap=None must equal uncapped, got {capped_none} vs {uncapped}"
        );
    }
}

#[test]
fn capped_above_natural_matches_uncapped() {
    // cap >= natural_n_levels is identical to natural — the clamp
    // `cap.min(natural_n_levels)` collapses to natural.
    let display = DisplayModel::STANDARD_4K;
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();

    for (r, d, w, h, name, _golden) in fixtures() {
        let natural_n = band_frequencies(ppd, w, h).len();
        let uncapped = predict_jod_still_3ch(&r, &d, w, h, display, ppd);
        // Try cap = natural_n + 5; should clamp to natural_n.
        let above = predict_jod_still_3ch_capped(
            &r,
            &d,
            w,
            h,
            display,
            ppd,
            Some(natural_n + 5),
        );
        assert_eq!(
            uncapped, above,
            "{name}: cap above natural ({}) must clamp to natural, got {above} vs {uncapped}",
            natural_n + 5,
        );
    }
}

#[test]
fn cap_8_host_scalar_meets_pycvvdp_gate() {
    // Cap=8 is the production-ready cap depth per the
    // capped_levels_sweep_2026-05-22 data: all 4 fixtures with
    // natural_n_levels = 9 (4000×3000, 1024×1024, 1280×720, 720×1280)
    // stay under 0.001 JOD drift vs pycvvdp golden at cap=8.
    //
    // This is the canonical gate for shipping `MemoryMode::Strip {
    // capped_levels: Some(8) }`. If this fails, capped mode does NOT
    // ship at cap=8 — surface the failure and document a higher cap.
    let display = DisplayModel::STANDARD_4K;
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();

    for (r, d, w, h, name, golden) in fixtures() {
        let natural_n = band_frequencies(ppd, w, h).len();
        // Skip fixtures whose natural depth is already <= 8 — the cap
        // is a no-op there and parity is already covered by the
        // existing 73×91 odd-dim test.
        if natural_n <= 8 {
            continue;
        }
        let jod = predict_jod_still_3ch_capped(&r, &d, w, h, display, ppd, Some(8));
        let diff = (jod - golden).abs();
        eprintln!(
            "{name}: natural_n={natural_n}, cap=8, jod={jod:.6}, golden={golden:.6}, diff={diff:.6}"
        );
        assert!(
            diff < TOLERANCE,
            "{name}: cap=8 JOD {jod:.6} drifts from pycvvdp golden {golden:.6} by {diff:.6} > {TOLERANCE:.4}"
        );
    }
}

#[test]
fn cap_7_drift_exceeds_gate_on_720x1280() {
    // Pin the known cap=7 failure on the 720×1280 fixture so a future
    // change that lifts the cap to 7 must explicitly acknowledge the
    // drift increase. Per the sweep:
    //   synth_720x1280_offset cap=7 → diff = 0.0117 (> 0.005 gate)
    // If a refactor closes this drift the test will fail and a human
    // can document the cap-7 ship decision.
    let display = DisplayModel::STANDARD_4K;
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();

    let r = common::synth_pair_ref(720, 1280);
    let d = common::apply_offset_dist(&r);
    let golden = common::pycvvdp_synth_golden_jod("synth_720x1280_offset");
    let jod = predict_jod_still_3ch_capped(&r, &d, 720, 1280, display, ppd, Some(7));
    let diff = (jod - golden).abs();
    eprintln!(
        "720x1280 cap=7: jod={jod:.6}, golden={golden:.6}, diff={diff:.6}"
    );
    // The 0.0117 measurement gives us margin around 0.005 to detect
    // both improvements and regressions.
    assert!(
        diff > TOLERANCE,
        "720x1280 cap=7 drift {diff:.6} unexpectedly closed below {TOLERANCE:.4} — \
         re-evaluate cap-7 ship decision and update STRIP_PROCESSING.md"
    );
    assert!(
        diff < 0.020,
        "720x1280 cap=7 drift {diff:.6} regressed unexpectedly past 0.020 — \
         investigate which earlier-band contribution shifted"
    );
}
