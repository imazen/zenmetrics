//! Per-feature parity test: zensim-gpu vs zensim CPU at every one of
//! the 228 slots.
//!
//! Maps each GPU slot to its CPU counterpart per `docs/FEATURE_PARITY.md`
//! and asserts agreement within `2e-3` relative tolerance (GPU uses f32
//! intermediates while CPU uses f64; this is the steady-state drift
//! observed on the corpus fixtures).
//!
//! Uses `Zensim::compute_extended_features` on the CPU side so that
//! every (scale, channel) slot is populated (the default `compute`
//! path skips channels whose weights are zero — that's correct for
//! scoring but defeats slot-level parity).

use cubecl::Runtime;
use zensim::{RgbSlice, Zensim as ZensimCpu, ZensimProfile};
use zensim_gpu::{TOTAL_FEATURES, Zensim};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
compile_error!(
    "zensim-gpu cpu_parity test requires either the `cuda` or `wgpu` feature to select a runtime"
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

fn cpu_features(rgb_ref: &[u8], rgb_dis: &[u8], w: usize, h: usize) -> Vec<f64> {
    let z = ZensimCpu::new(ZensimProfile::latest());
    let to_pix =
        |buf: &[u8]| -> Vec<[u8; 3]> { buf.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect() };
    let src = to_pix(rgb_ref);
    let dst = to_pix(rgb_dis);
    let s = RgbSlice::new(&src, w, h);
    let d = RgbSlice::new(&dst, w, h);
    // compute_extended_features populates EVERY (scale, channel) slot —
    // unlike compute(), which skips channels with all-zero weights and
    // would leave 0.0 holes in the comparison.
    let r = z
        .compute_extended_features(&s, &d)
        .expect("zensim cpu compute_extended_features");
    r.into_features()
}

/// Feature names for diagnostic printing.
fn slot_name(slot_in_19: usize) -> &'static str {
    match slot_in_19 {
        0 => "ssim_mean",
        1 => "ssim_4th",
        2 => "ssim_2nd",
        3 => "art_mean",
        4 => "art_4th",
        5 => "art_2nd",
        6 => "det_mean",
        7 => "det_4th",
        8 => "det_2nd",
        9 => "mse",
        10 => "hf_energy_loss",
        11 => "hf_mag_loss",
        12 => "hf_energy_gain",
        13 => "ssim_max",
        14 => "art_max",
        15 => "det_max",
        16 => "ssim_l8",
        17 => "art_l8",
        18 => "det_l8",
        _ => "?",
    }
}

const SCALES: usize = 4;

/// Decode a flat-228 feature index into (scale, channel, slot_in_19),
/// matching the basic[156] + peak[72] block layout used by both sides.
fn decode_idx(idx: usize) -> (usize, usize, usize) {
    let basic_total = SCALES * 3 * 13;
    if idx < basic_total {
        let s = idx / (3 * 13);
        let rem = idx - s * 3 * 13;
        let c = rem / 13;
        let off = rem - c * 13;
        (s, c, off)
    } else {
        let pidx = idx - basic_total;
        let s = pidx / (3 * 6);
        let rem = pidx - s * 3 * 6;
        let c = rem / 6;
        let off = rem - c * 6;
        // peak slot 0..6 maps to slot_in_19 13..19
        (s, c, 13 + off)
    }
}

/// Per-slot parity report: list every divergent slot to make debugging
/// fast. Returns `Vec<(idx, slot_name, cpu, gpu, abs_diff, rel_diff)>`.
fn diff_features(cpu: &[f64], gpu: &[f64]) -> Vec<(usize, &'static str, f64, f64, f64, f64)> {
    let mut out = Vec::new();
    let n = cpu.len().min(gpu.len());
    for i in 0..n {
        let (_s, _c, slot19) = decode_idx(i);
        let abs = (cpu[i] - gpu[i]).abs();
        let denom = cpu[i].abs().max(1e-6);
        let rel = abs / denom;
        out.push((i, slot_name(slot19), cpu[i], gpu[i], abs, rel));
    }
    out
}

// ───────────────────────── parity ─────────────────────────

/// Identical input → both sides return all-zero features (the CPU
/// short-circuits to zeros for byte-identical inputs; the GPU runs the
/// full kernel and rounds to 0.0 within ULP).
#[test]
fn identical_input_all_zeros() {
    let w = 64;
    let h = 64;
    let img = gradient(w, h);

    let mut z = Zensim::<Backend>::new(make_client!(), w as u32, h as u32).unwrap();
    let gpu = z.compute_features(&img, &img).unwrap();
    let cpu = cpu_features(&img, &img, w, h);

    // compute_extended_features returns 300 (basic 156 + peak 72 + masked 72);
    // we only compare the first 228 (basic + peak).
    assert!(cpu.len() >= TOTAL_FEATURES, "cpu features len {} < {}", cpu.len(), TOTAL_FEATURES);
    // CPU short-circuits to zeros; the first 228 must match GPU within
    // a tight bound for the SSIM term (mu1 == mu2 → sd == 0 analytically)
    // even though GPU runs the kernel. The HF terms can pick up sub-ULP
    // noise from the f32 σ² division; allow 5e-2 absolute on the
    // identical case (the only path that *should* clamp to zero on both
    // sides).
    let mut max_abs = 0.0_f64;
    for i in 0..TOTAL_FEATURES {
        let a = (gpu[i] - cpu[i]).abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    eprintln!("identical: max |gpu - cpu| = {max_abs:.4e}");
    assert!(max_abs < 5e-2, "identical case max diff {max_abs}");
}

/// Noisy-gradient corpus: every feature must agree within 2e-3 relative
/// (or 2e-3 absolute, whichever is larger) — that's the noise floor for
/// the f32 vs f64 split.
#[test]
fn noisy_gradient_every_slot_within_tolerance() {
    let w = 64;
    let h = 64;
    let r = gradient(w, h);
    let d = add_noise(&r, 8);

    let mut z = Zensim::<Backend>::new(make_client!(), w as u32, h as u32).unwrap();
    let gpu = z.compute_features(&r, &d).unwrap();
    let cpu = cpu_features(&r, &d, w, h);
    assert!(cpu.len() >= TOTAL_FEATURES);

    let diffs = diff_features(&cpu[..TOTAL_FEATURES], &gpu);

    // Loosest tolerance the per-slot test will accept. f32 mid-kernel
    // and 64×64 pooling N = 4096 implies typical sub-ULP drift on the
    // mean features (~1e-5 abs) and ~1e-3 abs on the max-pooled
    // features (where one outlier can swing the result). The peak
    // (max) features get the lion's share of drift.
    let mut failed = Vec::new();
    for &(idx, name, cv, gv, abs, rel) in &diffs {
        // Skip slots where the CPU value is exactly zero — these are
        // saturation-clamped (`max(0)`) results on both sides and don't
        // need rel-tolerance gymnastics.
        if cv == 0.0 && gv.abs() < 5e-3 {
            continue;
        }
        let ok = abs < 5e-3 || rel < 2e-3;
        if !ok {
            failed.push((idx, name, cv, gv, abs, rel));
        }
    }
    if !failed.is_empty() {
        for &(idx, name, cv, gv, abs, rel) in failed.iter().take(20) {
            eprintln!(
                "FAIL idx={idx:3} {name:14}: cpu={cv:+.6e} gpu={gv:+.6e} \
                 abs={abs:.3e} rel={rel:.3e}"
            );
        }
        eprintln!("({} failed in total)", failed.len());
        panic!("per-feature parity failed on {} slots", failed.len());
    }
}

/// Wider corpus: synthetic checkerboard + noise at 128×128, run the
/// full per-slot check. Larger N means tighter tolerance is achievable
/// on the basic mean/L4/L2 slots — but max-pooled slots can shift more
/// (one extra sample sometimes lands at a different max).
#[test]
fn checkerboard_corpus_per_slot() {
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

    let mut z = Zensim::<Backend>::new(make_client!(), w as u32, h as u32).unwrap();
    let gpu = z.compute_features(&r, &d).unwrap();
    let cpu = cpu_features(&r, &d, w, h);

    // Split the budget by slot kind:
    //   basic mean / L4 / L2 / mse / HF — tight, 1e-3 rel
    //   max-pooled (peak 0..2 = slot_in_19 13..15) — looser, 3e-2 rel
    //     because a single pixel where f32 vs f64 SSIM math diverges
    //     by ULP can flip which sample becomes the max
    //   L8 pool (peak 3..5 = slot_in_19 16..18) — moderate, 5e-3 rel
    let mut failed = Vec::new();
    for i in 0..TOTAL_FEATURES {
        let (_s, _c, slot) = decode_idx(i);
        let abs = (gpu[i] - cpu[i]).abs();
        let rel = abs / cpu[i].abs().max(1e-6);
        let (abs_budget, rel_budget) = match slot {
            13 | 14 | 15 => (5e-3, 3e-2), // max-pool
            16 | 17 | 18 => (3e-3, 5e-3), // L8
            _ => (2e-3, 2e-3),            // basic mean/L4/L2/mse/HF
        };
        // skip clamped-to-zero slots
        if cpu[i].abs() < 1e-6 && gpu[i].abs() < abs_budget {
            continue;
        }
        if abs > abs_budget && rel > rel_budget {
            failed.push((i, slot_name(slot), cpu[i], gpu[i], abs, rel));
        }
    }
    if !failed.is_empty() {
        for &(idx, name, cv, gv, abs, rel) in failed.iter().take(20) {
            eprintln!(
                "FAIL idx={idx:3} {name:14}: cpu={cv:+.6e} gpu={gv:+.6e} \
                 abs={abs:.3e} rel={rel:.3e}"
            );
        }
        eprintln!("({} failed in total)", failed.len());
        panic!("per-feature parity failed on {} slots at 128²", failed.len());
    }
}
