//! Native CPU cvvdp HDR via the faithful linear-light path (no cubeclcpu).
//! `HdrScorer` on `Backend::Cpu` routes cvvdp through
//! `cpu_dispatch::compute_from_linear_interleaved` →
//! `cvvdp::Cvvdp::score_from_linear_planes` (pure-Rust SIMD, archmage), with the
//! DisplayModel from the cvvdp params — mirroring the GPU `Backend::Cuda` linear
//! path. CUDA-gated (the comparison side needs the GPU); NO graceful skips.
#![cfg(all(feature = "cuda", feature = "cpu-cvvdp", feature = "cvvdp"))]

use zenmetrics_api::Backend;
use zenmetrics_api::hdr::{HDR_PEAK_NITS, HdrScorer};

fn srgb_eotf(c: u8) -> f32 {
    let v = c as f32 / 255.0;
    if v <= 0.040_449_936 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Interleaved nits gradient (sRGB-decode → linear → ×peak).
fn nits_gradient(w: u32, h: u32, seed: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            out.push(srgb_eotf(((x.wrapping_add(seed)) & 0xff) as u8) * HDR_PEAK_NITS);
            out.push(srgb_eotf(((y.wrapping_add(seed * 3)) & 0xff) as u8) * HDR_PEAK_NITS);
            out.push(srgb_eotf(((x ^ y ^ seed) & 0xff) as u8) * HDR_PEAK_NITS);
        }
    }
    out
}

#[test]
fn cpu_cvvdp_hdr_linear_identity_and_discrimination() {
    let (w, h) = (128_u32, 128_u32);
    let reference = nits_gradient(w, h, 0);
    let distorted = nits_gradient(w, h, 9);

    let mut cpu = HdrScorer::new(
        zenmetrics_api::MetricKind::Cvvdp,
        Backend::Cpu,
        w,
        h,
        HDR_PEAK_NITS,
    )
    .expect("cpu cvvdp scorer");

    // Identity: cvvdp == JOD max (10, higher is better).
    let id = cpu
        .compute(&reference, &reference)
        .expect("cpu identity")
        .value;
    assert!(
        id > 9.9,
        "CPU cvvdp HDR identity should be ~10 JOD, got {id}"
    );

    // Distortion discriminates (drops below the max).
    let dist = cpu
        .compute(&reference, &distorted)
        .expect("cpu distorted")
        .value;
    assert!(
        dist < 9.5,
        "CPU cvvdp HDR should detect distortion (<9.5), got {dist}"
    );
}

#[test]
fn cpu_cvvdp_hdr_linear_tracks_gpu() {
    let (w, h) = (128_u32, 128_u32);
    let reference = nits_gradient(w, h, 0);
    let distorted = nits_gradient(w, h, 9);

    let mut cpu = HdrScorer::new(
        zenmetrics_api::MetricKind::Cvvdp,
        Backend::Cpu,
        w,
        h,
        HDR_PEAK_NITS,
    )
    .expect("cpu");
    let mut gpu = HdrScorer::new(
        zenmetrics_api::MetricKind::Cvvdp,
        Backend::Cuda,
        w,
        h,
        HDR_PEAK_NITS,
    )
    .expect("gpu");

    let cpu_d = cpu.compute(&reference, &distorted).expect("cpu").value;
    let gpu_d = gpu.compute(&reference, &distorted).expect("gpu").value;

    // cvvdp's CPU port and GPU (cubecl) implement the same algorithm; on the
    // SAME faithful linear feeding they should agree closely. Report + assert.
    eprintln!("cvvdp HDR linear: cpu={cpu_d}  gpu={gpu_d}");
    let rel = (cpu_d - gpu_d).abs() / gpu_d.abs().max(1e-6);
    assert!(
        rel < 0.05,
        "CPU vs GPU cvvdp HDR linear should track within 5%: cpu={cpu_d} gpu={gpu_d} (rel {rel})"
    );
}
