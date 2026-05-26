//! Extended- + IW-regime per-feature parity tests against the
//! path-pinned CPU `zensim`.
//!
//! Coverage:
//!
//! - `ZensimFeatureRegime::Extended` (300) vs CPU
//!   `Zensim::compute_extended_features` (300) under the same
//!   f32-vs-f64 budget as `cpu_parity.rs`.
//! - `ZensimFeatureRegime::WithIw` (372): first 300 slots match
//!   Extended (turning on IW doesn't disturb earlier work), IW
//!   block (300..372) is bit-tested against CPU per slot.
//!
//! The CPU IW block is reached via the same
//! `Zensim::compute_extended_features` entry: the `latest()` profile
//! (`PreviewV0_3` / `A`) carries `compute_iw_features: true` in its
//! `ProfileParams`, so `config_from_params` keeps the IW pass on and
//! `combine_scores` Pass 4 appends the 72 IW slots to the returned
//! feature vector. No `compute_zensim_with_ref_and_config`, no
//! `training` feature flag needed.

use cubecl::Runtime;
use zensim::{RgbSlice, Zensim as ZensimCpu, ZensimProfile};
use zensim_gpu::{
    TOTAL_FEATURES, TOTAL_FEATURES_EXTENDED, TOTAL_FEATURES_WITH_IW, Zensim, ZensimFeatureRegime,
};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
compile_error!(
    "zensim-gpu extended_parity test requires either the `cuda` or `wgpu` feature to select a runtime"
);

macro_rules! make_client {
    () => {
        Backend::client(&Default::default())
    };
}

// ───────────────────────── helpers ─────────────────────────

fn gradient(w: usize, h: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 255) / w) as u8;
            let g = ((y * 255) / h) as u8;
            let b = (((x + y) * 255) / (w + h)) as u8;
            v.push(r);
            v.push(g);
            v.push(b);
        }
    }
    v
}

fn add_noise(data: &[u8], amount: i16) -> Vec<u8> {
    use std::num::Wrapping;
    let mut out = Vec::with_capacity(data.len());
    let mut seed = Wrapping(12345_u32);
    for &v in data {
        seed = seed * Wrapping(1103515245_u32) + Wrapping(12345_u32);
        let noise = ((seed.0 >> 16) as i16 % (amount * 2 + 1)) - amount;
        out.push((v as i16 + noise).clamp(0, 255) as u8);
    }
    out
}

fn cpu_extended_features(rgb_ref: &[u8], rgb_dis: &[u8], w: usize, h: usize) -> Vec<f64> {
    let z = ZensimCpu::new(ZensimProfile::latest());
    let to_pix =
        |buf: &[u8]| -> Vec<[u8; 3]> { buf.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect() };
    let src = to_pix(rgb_ref);
    let dst = to_pix(rgb_dis);
    let s = RgbSlice::new(&src, w, h);
    let d = RgbSlice::new(&dst, w, h);
    let r = z
        .compute_extended_features(&s, &d)
        .expect("zensim cpu compute_extended_features");
    r.into_features()
}

/// Decode a flat extended (300) index into (scale, channel, slot_kind, slot_offset).
/// slot_kind 0 = basic (0..13), 1 = peak (13..19), 2 = masked (19..25).
fn decode_extended_idx(idx: usize) -> (usize, usize, usize, usize) {
    const SCALES: usize = 4;
    let basic_total = SCALES * 3 * 13;
    let peaks_total = SCALES * 3 * 6;
    if idx < basic_total {
        let s = idx / (3 * 13);
        let rem = idx - s * 3 * 13;
        let c = rem / 13;
        let off = rem - c * 13;
        (s, c, 0, off)
    } else if idx < basic_total + peaks_total {
        let pidx = idx - basic_total;
        let s = pidx / (3 * 6);
        let rem = pidx - s * 3 * 6;
        let c = rem / 6;
        let off = rem - c * 6;
        (s, c, 1, off)
    } else {
        let midx = idx - basic_total - peaks_total;
        let s = midx / (3 * 6);
        let rem = midx - s * 3 * 6;
        let c = rem / 6;
        let off = rem - c * 6;
        (s, c, 2, off)
    }
}

fn slot_label(kind: usize, off: usize) -> &'static str {
    match (kind, off) {
        (0, 0) => "ssim_mean",
        (0, 1) => "ssim_4th",
        (0, 2) => "ssim_2nd",
        (0, 3) => "art_mean",
        (0, 4) => "art_4th",
        (0, 5) => "art_2nd",
        (0, 6) => "det_mean",
        (0, 7) => "det_4th",
        (0, 8) => "det_2nd",
        (0, 9) => "mse",
        (0, 10) => "hf_energy_loss",
        (0, 11) => "hf_mag_loss",
        (0, 12) => "hf_energy_gain",
        (1, 0) => "ssim_max",
        (1, 1) => "art_max",
        (1, 2) => "det_max",
        (1, 3) => "ssim_l8",
        (1, 4) => "art_l8",
        (1, 5) => "det_l8",
        (2, 0) => "masked_ssim_mean",
        (2, 1) => "masked_ssim_4th",
        (2, 2) => "masked_ssim_2nd",
        (2, 3) => "masked_art_4th",
        (2, 4) => "masked_det_4th",
        (2, 5) => "masked_mse",
        _ => "?",
    }
}

// ───────────────────────── tests ─────────────────────────

/// Identical input → 300-feature vector all zeros (CPU short-circuits,
/// GPU runs the full kernel and rounds to ~ULP noise).
#[test]
fn extended_identical_zeros() {
    let w = 64;
    let h = 64;
    let img = gradient(w, h);

    let mut z = Zensim::<Backend>::new_with_regime(
        make_client!(),
        w as u32,
        h as u32,
        ZensimFeatureRegime::Extended,
    )
    .unwrap();
    let gpu = z.compute_features_vec(&img, &img).unwrap();
    assert_eq!(gpu.len(), TOTAL_FEATURES_EXTENDED);

    let cpu = cpu_extended_features(&img, &img, w, h);
    assert!(cpu.len() >= TOTAL_FEATURES_EXTENDED);

    // Identical inputs → expect all values near zero. The HF terms can
    // pick up sub-ULP noise from the f32 σ² division. Tightened
    // 2026-05-22 from 5e-2 → 2e-3 (measured max 6.8e-4 → 3× margin).
    let mut max_abs = 0.0_f64;
    for i in 0..TOTAL_FEATURES_EXTENDED {
        let a = (gpu[i] - cpu[i]).abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    eprintln!("ext identical: max |gpu - cpu| = {max_abs:.4e}");
    assert!(max_abs < 2e-3, "ext identical max diff {max_abs}");
}

/// Per-feature parity on a noisy gradient at 64×64. All 300 slots must
/// agree with the published-zensim `compute_extended_features` output
/// within the same f32-vs-f64 budget the `cpu_parity.rs` basic+peaks
/// test uses, plus a small loosening for the masked SSIM mean (the
/// fourth/squared pools self-correct via the powf(0.25) / .sqrt()).
#[test]
fn extended_noisy_gradient_64() {
    let w = 64;
    let h = 64;
    let r = gradient(w, h);
    let d = add_noise(&r, 8);

    let mut z = Zensim::<Backend>::new_with_regime(
        make_client!(),
        w as u32,
        h as u32,
        ZensimFeatureRegime::Extended,
    )
    .unwrap();
    let gpu = z.compute_features_vec(&r, &d).unwrap();
    let cpu = cpu_extended_features(&r, &d, w, h);
    assert_eq!(gpu.len(), TOTAL_FEATURES_EXTENDED);
    assert!(cpu.len() >= TOTAL_FEATURES_EXTENDED);

    let mut failed = Vec::new();
    let mut max_abs_basic = 0.0_f64;
    let mut max_abs_peak = 0.0_f64;
    let mut max_abs_l8 = 0.0_f64;
    let mut max_abs_masked = 0.0_f64;
    for i in 0..TOTAL_FEATURES_EXTENDED {
        let (s, c, kind, off) = decode_extended_idx(i);
        let cv = cpu[i];
        let gv = gpu[i];
        let abs = (cv - gv).abs();
        let rel = abs / cv.abs().max(1e-6);
        // Budget by slot kind. Tightened 2026-05-22 to reflect actual
        // measured drift on this fixture (see drift summary printed
        // below). Bands set to (measured_max × ~2–3) for headroom.
        //
        // ## Principled per-channel H-blur activity (2026-05-17)
        //
        // CPU now computes activity as `box_blur(|src - H_blur(src)|)`
        // per channel at all strip rows (inner + overlap). GPU mirrors
        // this exactly — no cross-channel cascade, no carry plane.
        let (abs_budget, rel_budget) = match (kind, off, s, c) {
            // peak / max-pooled (kind 1, off 0..3)
            (1, 0, _, _) | (1, 1, _, _) | (1, 2, _, _) => (2e-3, 3e-2),
            // L8 pool (kind 1, off 3..6)
            (1, _, _, _) => (1e-3, 5e-3),
            // masked block — principled per-channel H-blur
            (2, _, _, _) => (2e-3, 5e-3),
            // basic (kind 0)
            _ => (1e-3, 2e-3),
        };
        // skip clamped-to-zero slots
        if cv.abs() < 1e-6 && gv.abs() < abs_budget {
            continue;
        }
        match (kind, off) {
            (1, 0) | (1, 1) | (1, 2) => max_abs_peak = max_abs_peak.max(abs),
            (1, _) => max_abs_l8 = max_abs_l8.max(abs),
            (2, _) => max_abs_masked = max_abs_masked.max(abs),
            _ => max_abs_basic = max_abs_basic.max(abs),
        }
        if abs > abs_budget && rel > rel_budget {
            failed.push((i, kind, off, cv, gv, abs, rel));
        }
    }
    eprintln!(
        "extended_parity drift summary:\n  basic : max_abs={max_abs_basic:.3e}\n  peak  : max_abs={max_abs_peak:.3e}\n  l8    : max_abs={max_abs_l8:.3e}\n  masked: max_abs={max_abs_masked:.3e}"
    );
    if !failed.is_empty() {
        for &(idx, k, o, cv, gv, abs, rel) in failed.iter().take(20) {
            eprintln!(
                "FAIL idx={idx:3} {:14}: cpu={cv:+.6e} gpu={gv:+.6e} \
                 abs={abs:.3e} rel={rel:.3e}",
                slot_label(k, o)
            );
        }
        eprintln!("({} failed in total)", failed.len());
        panic!(
            "extended per-feature parity failed on {} slots",
            failed.len()
        );
    }
}

/// 128×128 with checkerboard + noise — same parity budget as the
/// 64×64 case, larger N tightens the mean-pool slot drift but the
/// per-slot budget already accounts for the loosest case.
#[test]
fn extended_checkerboard_128() {
    let w = 128;
    let h = 128;
    let mut r = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let on = ((x / 8) + (y / 8)) % 2 == 0;
            let v = if on { 220 } else { 64 };
            r.push(v);
            r.push(v);
            r.push(v);
        }
    }
    let d = add_noise(&r, 12);

    let mut z = Zensim::<Backend>::new_with_regime(
        make_client!(),
        w as u32,
        h as u32,
        ZensimFeatureRegime::Extended,
    )
    .unwrap();
    let gpu = z.compute_features_vec(&r, &d).unwrap();
    let cpu = cpu_extended_features(&r, &d, w, h);

    let mut failed = Vec::new();
    for i in 0..TOTAL_FEATURES_EXTENDED {
        let (s, c, kind, off) = decode_extended_idx(i);
        let cv = cpu[i];
        let gv = gpu[i];
        let abs = (cv - gv).abs();
        let rel = abs / cv.abs().max(1e-6);
        // Principled per-channel H-blur activity (2026-05-17): all
        // masked slots match CPU within 5e-3 rel at every scale,
        // including multi-strip scales 0 and 1 on 128². No
        // strip-boundary cross-channel cascade and no persist-plane vs
        // strip-local-V-blur mismatch — every channel sees its own
        // H_blur(src) at every strip row.
        let (abs_budget, rel_budget) = match (kind, off, s, c) {
            (1, 0, _, _) | (1, 1, _, _) | (1, 2, _, _) => (5e-3, 3e-2),
            (1, _, _, _) => (3e-3, 5e-3),
            (2, _, _, _) => (5e-3, 5e-3),
            _ => (2e-3, 2e-3),
        };
        if cv.abs() < 1e-6 && gv.abs() < abs_budget {
            continue;
        }
        if abs > abs_budget && rel > rel_budget {
            failed.push((i, kind, off, cv, gv, abs, rel));
        }
    }
    if !failed.is_empty() {
        for &(idx, k, o, cv, gv, abs, rel) in failed.iter().take(20) {
            eprintln!(
                "FAIL idx={idx:3} {:14}: cpu={cv:+.6e} gpu={gv:+.6e} \
                 abs={abs:.3e} rel={rel:.3e}",
                slot_label(k, o)
            );
        }
        eprintln!("({} failed in total)", failed.len());
        panic!(
            "extended per-feature parity failed on {} slots at 128²",
            failed.len()
        );
    }
}

/// WithIw regime: 372 features. First 300 must match Extended exactly
/// (same fixture, two pipelines). IW block (300..372) must be finite
/// and mostly non-zero on noisy input.
#[test]
fn with_iw_structural_noisy() {
    let w = 64;
    let h = 64;
    let r = gradient(w, h);
    let d = add_noise(&r, 8);

    let mut z_ext = Zensim::<Backend>::new_with_regime(
        make_client!(),
        w as u32,
        h as u32,
        ZensimFeatureRegime::Extended,
    )
    .unwrap();
    let ext = z_ext.compute_features_vec(&r, &d).unwrap();
    assert_eq!(ext.len(), TOTAL_FEATURES_EXTENDED);

    let mut z_iw = Zensim::<Backend>::new_with_regime(
        make_client!(),
        w as u32,
        h as u32,
        ZensimFeatureRegime::WithIw,
    )
    .unwrap();
    let iw = z_iw.compute_features_vec(&r, &d).unwrap();
    assert_eq!(iw.len(), TOTAL_FEATURES_WITH_IW);

    // First 300 slots: WithIw should match Extended at the same
    // fixture (within the f32 noise budget — different kernel launches
    // mean different scheduling, but the math is identical).
    let mut max_300 = 0.0_f64;
    for i in 0..TOTAL_FEATURES_EXTENDED {
        let a = (ext[i] - iw[i]).abs();
        if a > max_300 {
            max_300 = a;
        }
    }
    eprintln!("with_iw[0..300] vs extended max |diff| = {max_300:.4e}");
    // Tightened 2026-05-22: WithIw runs the same kernels as Extended
    // and the measured drift is 0.0 (bit-identical across launches).
    // Gate at 1e-9 to catch any future divergence.
    assert!(
        max_300 < 1e-9,
        "WithIw[0..300] diverged from Extended (max diff {max_300})"
    );

    // IW block: finite, magnitude range looks reasonable, AT LEAST half
    // of the 72 slots non-zero (noisy input should hit most of them).
    let mut n_nonzero = 0;
    let mut max_iw = 0.0_f64;
    for i in TOTAL_FEATURES_EXTENDED..TOTAL_FEATURES_WITH_IW {
        let v = iw[i];
        assert!(v.is_finite(), "IW slot {i} is non-finite: {v}");
        if v.abs() > 1e-9 {
            n_nonzero += 1;
        }
        if v.abs() > max_iw {
            max_iw = v.abs();
        }
    }
    eprintln!(
        "with_iw IW block: {n_nonzero}/{} non-zero, max |val| = {max_iw:.4e}",
        TOTAL_FEATURES_WITH_IW - TOTAL_FEATURES_EXTENDED
    );
    assert!(
        n_nonzero >= 36,
        "IW block should have ≥ half non-zero on noisy input ({n_nonzero}/72)"
    );
    assert!(
        max_iw < 1e3,
        "IW slot magnitude looks runaway (max {max_iw})"
    );
}

/// Identical input → WithIw vector is all zeros (CPU short-circuit
/// behavior; GPU runs the full kernel but the math collapses).
#[test]
fn with_iw_identical_zeros() {
    let w = 64;
    let h = 64;
    let img = gradient(w, h);

    let mut z = Zensim::<Backend>::new_with_regime(
        make_client!(),
        w as u32,
        h as u32,
        ZensimFeatureRegime::WithIw,
    )
    .unwrap();
    let gpu = z.compute_features_vec(&img, &img).unwrap();
    assert_eq!(gpu.len(), TOTAL_FEATURES_WITH_IW);

    let mut max_abs = 0.0_f64;
    for &v in gpu.iter() {
        let a = v.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    eprintln!("with_iw identical: max |val| = {max_abs:.4e}");
    // Tightened to match Extended identical: 2e-3 absolute
    // (measured 6.8e-4 → 3× margin).
    assert!(max_abs < 2e-3, "with_iw identical max abs {max_abs}");
}

/// Decode an IW-block flat index (relative to `TOTAL_FEATURES_EXTENDED`,
/// i.e. `gpu[300 + iw_off]`) into (scale, channel, slot_offset). The IW
/// block is 4 scales × 3 channels × 6 features = 72 slots, ordered as
/// (scale-major, channel-medial, feature-minor):
///   `iw[s][c][off]` = `iw_offset` 0..72 where
///   `iw_offset = s * 18 + c * 6 + off`.
fn decode_iw_idx(iw_off: usize) -> (usize, usize, usize) {
    let s = iw_off / 18;
    let rem = iw_off - s * 18;
    let c = rem / 6;
    let off = rem - c * 6;
    (s, c, off)
}

fn iw_slot_label(off: usize) -> &'static str {
    match off {
        0 => "iw_ssim_mean",
        1 => "iw_ssim_4th",
        2 => "iw_ssim_2nd",
        3 => "iw_art_4th",
        4 => "iw_det_4th",
        5 => "iw_mse",
        _ => "?",
    }
}

/// Per-slot IW parity on a 64×64 noisy gradient. Mirrors
/// `extended_noisy_gradient_64`'s contract but for slots 300..372.
///
/// CPU path: `ZensimCpu::new(latest()).compute_extended_features(...)`.
/// The `latest()` profile (`PreviewV0_3` aka `A`) carries
/// `compute_iw_features: true` in its `ProfileParams`, so the CPU
/// `config_from_params` keeps `compute_iw_features = true` and the
/// resulting `ZensimResult.features()` already contains the IW block
/// at offsets 300..372 (CPU `combine_scores` Pass 4). No separate
/// `compute_zensim_with_ref_and_config` call needed — and no
/// `training` feature gate either.
#[test]
fn iw_slot_parity_noisy_gradient_64() {
    let w = 64;
    let h = 64;
    let r = gradient(w, h);
    let d = add_noise(&r, 8);

    let mut z = Zensim::<Backend>::new_with_regime(
        make_client!(),
        w as u32,
        h as u32,
        ZensimFeatureRegime::WithIw,
    )
    .unwrap();
    let gpu = z.compute_features_vec(&r, &d).unwrap();
    assert_eq!(gpu.len(), TOTAL_FEATURES_WITH_IW);

    let cpu = cpu_extended_features(&r, &d, w, h);
    assert_eq!(
        cpu.len(),
        TOTAL_FEATURES_WITH_IW,
        "latest() CPU profile must emit 372 features (extended + IW)"
    );

    // Tolerance budget — mirror `extended_noisy_gradient_64`'s masked
    // budget for symmetry. The IW kernel is the same as the masked
    // kernel with `1 + k·a` vs `1 / (1 + k·a)`; mean/L4/L2/MSE pools
    // behave identically. Use `5e-3 rel` consistently — the
    // brief's spec.
    const ABS_BUDGET: f64 = 2e-3;
    const REL_BUDGET: f64 = 5e-3;

    let mut failed = Vec::new();
    let mut max_abs_iw = 0.0_f64;
    let mut max_rel_iw = 0.0_f64;
    for iw_off in 0..(TOTAL_FEATURES_WITH_IW - TOTAL_FEATURES_EXTENDED) {
        let i = TOTAL_FEATURES_EXTENDED + iw_off;
        let (s, c, off) = decode_iw_idx(iw_off);
        let cv = cpu[i];
        let gv = gpu[i];
        let abs = (cv - gv).abs();
        let rel = abs / cv.abs().max(1e-6);
        // Skip clamped-to-zero slots.
        if cv.abs() < 1e-6 && gv.abs() < ABS_BUDGET {
            continue;
        }
        if abs > max_abs_iw {
            max_abs_iw = abs;
        }
        if rel > max_rel_iw {
            max_rel_iw = rel;
        }
        if abs > ABS_BUDGET && rel > REL_BUDGET {
            failed.push((iw_off, s, c, off, cv, gv, abs, rel));
        }
    }
    eprintln!(
        "iw parity drift summary (64x64):\n  max_abs={max_abs_iw:.3e}\n  max_rel={max_rel_iw:.3e}"
    );
    if !failed.is_empty() {
        for &(iw_off, s, c, off, cv, gv, abs, rel) in failed.iter().take(20) {
            eprintln!(
                "FAIL iw_off={iw_off:3} (s={s},c={c}) {:14}: cpu={cv:+.6e} \
                 gpu={gv:+.6e} abs={abs:.3e} rel={rel:.3e}",
                iw_slot_label(off)
            );
        }
        eprintln!("({} failed in total)", failed.len());
        panic!(
            "IW per-feature parity failed on {} of 72 slots at 64x64",
            failed.len()
        );
    }
}

/// Per-slot IW parity on a 128×128 checkerboard + noise — same budget
/// as `extended_checkerboard_128` for the masked block.
#[test]
fn iw_slot_parity_checkerboard_128() {
    let w = 128;
    let h = 128;
    let mut r = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let on = ((x / 8) + (y / 8)) % 2 == 0;
            let v = if on { 220 } else { 64 };
            r.push(v);
            r.push(v);
            r.push(v);
        }
    }
    let d = add_noise(&r, 12);

    let mut z = Zensim::<Backend>::new_with_regime(
        make_client!(),
        w as u32,
        h as u32,
        ZensimFeatureRegime::WithIw,
    )
    .unwrap();
    let gpu = z.compute_features_vec(&r, &d).unwrap();
    assert_eq!(gpu.len(), TOTAL_FEATURES_WITH_IW);

    let cpu = cpu_extended_features(&r, &d, w, h);
    assert_eq!(
        cpu.len(),
        TOTAL_FEATURES_WITH_IW,
        "latest() CPU profile must emit 372 features (extended + IW)"
    );

    // Same `5e-3 rel` budget as `iw_slot_parity_noisy_gradient_64`.
    const ABS_BUDGET: f64 = 5e-3;
    const REL_BUDGET: f64 = 5e-3;

    let mut failed = Vec::new();
    let mut max_abs_iw = 0.0_f64;
    let mut max_rel_iw = 0.0_f64;
    for iw_off in 0..(TOTAL_FEATURES_WITH_IW - TOTAL_FEATURES_EXTENDED) {
        let i = TOTAL_FEATURES_EXTENDED + iw_off;
        let (s, c, off) = decode_iw_idx(iw_off);
        let cv = cpu[i];
        let gv = gpu[i];
        let abs = (cv - gv).abs();
        let rel = abs / cv.abs().max(1e-6);
        if cv.abs() < 1e-6 && gv.abs() < ABS_BUDGET {
            continue;
        }
        if abs > max_abs_iw {
            max_abs_iw = abs;
        }
        if rel > max_rel_iw {
            max_rel_iw = rel;
        }
        if abs > ABS_BUDGET && rel > REL_BUDGET {
            failed.push((iw_off, s, c, off, cv, gv, abs, rel));
        }
    }
    eprintln!(
        "iw parity drift summary (128x128):\n  max_abs={max_abs_iw:.3e}\n  max_rel={max_rel_iw:.3e}"
    );
    if !failed.is_empty() {
        for &(iw_off, s, c, off, cv, gv, abs, rel) in failed.iter().take(20) {
            eprintln!(
                "FAIL iw_off={iw_off:3} (s={s},c={c}) {:14}: cpu={cv:+.6e} \
                 gpu={gv:+.6e} abs={abs:.3e} rel={rel:.3e}",
                iw_slot_label(off)
            );
        }
        eprintln!("({} failed in total)", failed.len());
        panic!(
            "IW per-feature parity failed on {} of 72 slots at 128x128",
            failed.len()
        );
    }
}

/// Basic regime still emits the canonical 228-vector with bit-for-bit
/// identical numbers to the original (pre-372) pipeline. Used as a
/// regression guard: the new conditional paths must not perturb the
/// fast path.
#[test]
fn basic_regime_unchanged() {
    let w = 64;
    let h = 64;
    let r = gradient(w, h);
    let d = add_noise(&r, 8);

    let mut z = Zensim::<Backend>::new(make_client!(), w as u32, h as u32).unwrap();
    let v228 = z.compute_features(&r, &d).unwrap();
    assert_eq!(v228.len(), TOTAL_FEATURES);

    // Same construct via the explicit-regime constructor — must be
    // identical bit-for-bit.
    let mut z2 = Zensim::<Backend>::new_with_regime(
        make_client!(),
        w as u32,
        h as u32,
        ZensimFeatureRegime::Basic,
    )
    .unwrap();
    let v228b = z2.compute_features(&r, &d).unwrap();
    assert_eq!(v228, v228b, "Basic-regime constructor diverged from new()");
}
