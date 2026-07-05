//! Reproduction harness for the odd-dimension zensim-gpu feature
//! corruption reported against `zenmetrics sweep --metric zensim-gpu`
//! (zensim repo: `benchmarks/linear_projections_2026-07-03.md` §w11 +
//! `benchmarks/provenance_best_results_2026-07-04.md` §f155).
//!
//! Symptom: on images with a non-16-aligned width somewhere in the
//! scale pyramid, specific masked/IW feature slots come back
//! BIT-CONSTANT across an entire quality ladder (many distorted
//! variants of the same reference), at values wildly outside the
//! feature's normal range — instead of NaN (which the existing
//! "~25% of cells NaN on odd-dim images" pathology already produces
//! and the sweep drops).
//!
//! This example runs a WARM loop (one `Zensim` instance, one
//! `set_reference`, many `compute_with_reference_vec` calls against
//! DIFFERENT distorted images) — the exact call shape
//! `zenmetrics-cli`'s `MetricCache` uses for a sweep ladder — and
//! compares every call's GPU 372-vector against a fresh CPU
//! `compute_extended_features` call on the SAME pixels. It also runs
//! a COLD variant (fresh `Zensim` + fresh `set_reference` per distorted
//! image) so we can tell whether the corruption is warm-state-carry
//! specific or present even without any state reuse.
//!
//! Run: `cargo run --release --features cuda,cubecl-types -p zensim-gpu --example odd_dim_repro -- 769 513`

use cubecl::Runtime;
#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime as Backend;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
use cubecl::wgpu::WgpuRuntime as Backend;
use zensim::{RgbSlice, Zensim as ZensimCpu, ZensimProfile};
use zensim_gpu::{TOTAL_FEATURES_WITH_IW, Zensim, ZensimFeatureRegime};

fn pattern_photo_wash(w: usize, h: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let fx = (x as f32) / (w as f32);
            let fy = (y as f32) / (h as f32);
            let r = 127.5 + 80.0 * (4.0 * fx + 1.7 * fy).cos() + 25.0 * (11.0 * fx).sin();
            let g = 127.5 + 70.0 * (3.0 * fx - 2.5 * fy).sin() + 30.0 * (7.0 * fy).cos();
            let b = 127.5 + 65.0 * (2.0 * fx + 3.0 * fy).cos() + 20.0 * (9.0 * fx + fy).sin();
            v.push(r.clamp(0.0, 255.0) as u8);
            v.push(g.clamp(0.0, 255.0) as u8);
            v.push(b.clamp(0.0, 255.0) as u8);
        }
    }
    v
}

/// Deterministic xorshift-ish noise, one distinct variant per `seed` —
/// stands in for "40 different quality levels of the same source".
fn add_noise(data: &[u8], amount: i16, seed: u32) -> Vec<u8> {
    use std::num::Wrapping;
    let mut out = Vec::with_capacity(data.len());
    let mut s = Wrapping(seed);
    for &v in data {
        s = s * Wrapping(1103515245_u32) + Wrapping(12345_u32);
        let n = ((s.0 >> 16) as i16 % (amount * 2 + 1)) - amount;
        out.push((v as i16 + n).clamp(0, 255) as u8);
    }
    out
}

fn cpu_372(rgb_ref: &[u8], rgb_dis: &[u8], w: usize, h: usize) -> Vec<f64> {
    let z = ZensimCpu::new(ZensimProfile::A);
    let to_pix =
        |buf: &[u8]| -> Vec<[u8; 3]> { buf.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect() };
    let src = to_pix(rgb_ref);
    let dst = to_pix(rgb_dis);
    let s = RgbSlice::new(&src, w, h);
    let d = RgbSlice::new(&dst, w, h);
    z.compute_extended_features(&s, &d)
        .expect("cpu compute_extended_features")
        .into_features()
}

const NAMES_BASIC: [&str; 13] = [
    "ssim_mean",
    "ssim_4th",
    "ssim_2nd",
    "art_mean",
    "art_4th",
    "art_2nd",
    "det_mean",
    "det_4th",
    "det_2nd",
    "mse",
    "hf_energy_loss",
    "hf_mag_loss",
    "hf_energy_gain",
];
const NAMES_PEAK: [&str; 6] = [
    "ssim_max", "art_max", "det_max", "ssim_l8", "art_l8", "det_l8",
];
const NAMES_MASKED: [&str; 6] = [
    "masked_ssim_mean",
    "masked_ssim_4th",
    "masked_ssim_2nd",
    "masked_art_4th",
    "masked_det_4th",
    "masked_mse",
];
const NAMES_IW: [&str; 6] = [
    "iw_ssim_mean",
    "iw_ssim_4th",
    "iw_ssim_2nd",
    "iw_art_4th",
    "iw_det_4th",
    "iw_mse",
];

fn decode_372_idx(idx: usize) -> (&'static str, usize, usize, usize) {
    const SCALES: usize = 4;
    let basic_total = SCALES * 3 * 13;
    let peak_total = SCALES * 3 * 6;
    let masked_total = SCALES * 3 * 6;
    let (kind, s, c, off) = if idx < basic_total {
        let s = idx / (3 * 13);
        let rem = idx - s * 3 * 13;
        let c = rem / 13;
        let off = rem - c * 13;
        ("basic", s, c, off)
    } else if idx < basic_total + peak_total {
        let pidx = idx - basic_total;
        let s = pidx / (3 * 6);
        let rem = pidx - s * 3 * 6;
        let c = rem / 6;
        let off = rem - c * 6;
        ("peak", s, c, off)
    } else if idx < basic_total + peak_total + masked_total {
        let midx = idx - basic_total - peak_total;
        let s = midx / (3 * 6);
        let rem = midx - s * 3 * 6;
        let c = rem / 6;
        let off = rem - c * 6;
        ("masked", s, c, off)
    } else {
        let iwidx = idx - basic_total - peak_total - masked_total;
        let s = iwidx / (3 * 6);
        let rem = iwidx - s * 3 * 6;
        let c = rem / 6;
        let off = rem - c * 6;
        ("iw", s, c, off)
    };
    let name = match kind {
        "basic" => NAMES_BASIC[off],
        "peak" => NAMES_PEAK[off],
        "masked" => NAMES_MASKED[off],
        _ => NAMES_IW[off],
    };
    (name, s, c, off)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let w: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(769);
    let h: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(513);
    let n_variants: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);

    println!(
        "=== odd_dim_repro: {w}x{h}, padded_w0={} (pad_count={}), {n_variants} distorted variants ===",
        zensim_gpu::simd_padded_width(w as usize),
        zensim_gpu::simd_padded_width(w as usize) - w as usize,
    );

    let ref_buf = pattern_photo_wash(w as usize, h as usize);
    let dist_bufs: Vec<Vec<u8>> = (0..n_variants)
        .map(|i| add_noise(&ref_buf, 4 + (i as i16) * 3, 0xCAFE_0000 + i as u32))
        .collect();

    // ---- WARM loop: one Zensim, one set_reference, many computes ----
    println!("\n--- WARM loop (set_reference once, compute_with_reference_vec per variant) ---");
    let client = Backend::client(&Default::default());
    let mut z_warm = Zensim::<Backend>::new_with_regime(client, w, h, ZensimFeatureRegime::WithIw)
        .expect("gpu construct (warm)");
    z_warm.set_reference(&ref_buf).expect("set_reference");

    let mut warm_gpu_vectors: Vec<Vec<f64>> = Vec::new();
    for (i, dist) in dist_bufs.iter().enumerate() {
        let gpu = z_warm
            .compute_with_reference_vec(dist)
            .expect("compute_with_reference_vec");
        let cpu = cpu_372(&ref_buf, dist, w as usize, h as usize);
        report_variant(i, &gpu, &cpu);
        warm_gpu_vectors.push(gpu);
    }

    // Cross-variant bit-constancy screen on the WARM run: any feature
    // slot that is EXACTLY identical across every distorted variant is
    // suspicious (a real feature depends on the distorted pixels, which
    // differ every variant).
    println!("\n--- warm-loop bit-constant-across-variants screen ---");
    let mut n_constant = 0;
    for idx in 0..TOTAL_FEATURES_WITH_IW {
        let first = warm_gpu_vectors[0][idx];
        let all_same = warm_gpu_vectors
            .iter()
            .all(|v| v[idx].to_bits() == first.to_bits());
        if all_same && n_variants > 1 {
            n_constant += 1;
            let (name, s, c, off) = decode_372_idx(idx);
            println!(
                "  CONSTANT idx={idx:3} ({name} s={s} c={c} off={off}) = {first:+.6e} across all {n_variants} variants"
            );
        }
    }
    println!("total constant-across-variants slots: {n_constant} / {TOTAL_FEATURES_WITH_IW}");

    // ---- COLD variant: fresh Zensim + fresh set_reference per distorted image ----
    println!("\n--- COLD (fresh Zensim + fresh set_reference per variant) ---");
    let mut cold_gpu_vectors: Vec<Vec<f64>> = Vec::new();
    for (i, dist) in dist_bufs.iter().enumerate() {
        let client = Backend::client(&Default::default());
        let mut z_cold =
            Zensim::<Backend>::new_with_regime(client, w, h, ZensimFeatureRegime::WithIw)
                .expect("gpu construct (cold)");
        let gpu = z_cold
            .compute_features_vec(&ref_buf, dist)
            .expect("compute_features_vec");
        let cpu = cpu_372(&ref_buf, dist, w as usize, h as usize);
        report_variant(i, &gpu, &cpu);
        cold_gpu_vectors.push(gpu);
    }
    println!("\n--- cold bit-constant-across-variants screen ---");
    let mut n_constant_cold = 0;
    for idx in 0..TOTAL_FEATURES_WITH_IW {
        let first = cold_gpu_vectors[0][idx];
        let all_same = cold_gpu_vectors
            .iter()
            .all(|v| v[idx].to_bits() == first.to_bits());
        if all_same && n_variants > 1 {
            n_constant_cold += 1;
            let (name, s, c, off) = decode_372_idx(idx);
            println!(
                "  CONSTANT idx={idx:3} ({name} s={s} c={c} off={off}) = {first:+.6e} across all {n_variants} variants"
            );
        }
    }
    println!(
        "total constant-across-variants slots (cold): {n_constant_cold} / {TOTAL_FEATURES_WITH_IW}"
    );
}

/// Per-slot budget mirroring `tests/it/cpu_gpu_feature_sweep.rs::slot_budget`
/// (the crate's own claimed GPU/CPU parity contract).
fn real_budget(kind: &str) -> (f64, f64) {
    match kind {
        "masked" | "iw" => (5e-3, 5e-3),
        "peak" => (3e-3, 5e-3),
        _ => (2e-3, 2e-3),
    }
}

fn report_variant(i: usize, gpu: &[f64], cpu: &[f64]) {
    let mut max_abs = 0.0f64;
    let mut max_idx = 0usize;
    let mut n_bad = 0usize;
    let mut n_real_budget_fail = 0usize;
    let mut real_fails: Vec<(usize, f64, f64, f64)> = Vec::new();
    for idx in 0..cpu.len() {
        let abs = (gpu[idx] - cpu[idx]).abs();
        let rel = abs / cpu[idx].abs().max(1e-6);
        if abs > max_abs {
            max_abs = abs;
            max_idx = idx;
        }
        let (kind, _s, _c, _off) = decode_372_idx(idx);
        let (abs_budget, rel_budget) = real_budget(kind);
        if abs > abs_budget && rel > rel_budget {
            n_real_budget_fail += 1;
            real_fails.push((idx, cpu[idx], gpu[idx], abs));
        }
        // Loose budget just for the report — matches the sweep's own
        // "wildly outside range" symptom, not a strict pass/fail gate.
        if abs > 1e-1 && rel > 1e-1 {
            n_bad += 1;
        }
    }
    let (name, s, c, off) = decode_372_idx(max_idx);
    println!(
        "variant {i:2}: worst idx={max_idx:3} ({name} s={s} c={c} off={off}) cpu={:+.6e} gpu={:+.6e} abs={max_abs:.4e}  [{n_bad} slots > loose budget] [{n_real_budget_fail} slots FAIL the crate's own claimed parity budget]",
        cpu[max_idx], gpu[max_idx]
    );
    if std::env::var("ODD_DIM_REPRO_VERBOSE").is_ok() {
        for (idx, cpu_v, gpu_v, abs) in &real_fails {
            let (name, s, c, off) = decode_372_idx(*idx);
            let rel = abs / cpu_v.abs().max(1e-6);
            println!(
                "    FAIL idx={idx:3} ({name} s={s} c={c} off={off}) cpu={cpu_v:+.6e} gpu={gpu_v:+.6e} abs={abs:.4e} rel={rel:.4e}"
            );
        }
    }
}
