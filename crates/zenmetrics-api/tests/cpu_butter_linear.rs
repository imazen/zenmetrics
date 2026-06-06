//! Native CPU butter HDR via the faithful linear-light path (no cubeclcpu).
//! `HdrScorer` on `Backend::Cpu` routes butter through
//! `cpu_dispatch::compute_from_linear_interleaved` → `butteraugli_linear`
//! with `intensity_target = peak`, mirroring the GPU `Backend::Cuda` linear
//! path. CUDA-gated (the comparison side needs the GPU); NO graceful skips.
#![cfg(all(
    feature = "cuda",
    feature = "cpu-butter",
    feature = "butter",
    feature = "pixels"
))]

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

/// Interleaved nits gradient: sRGB-decode a gradient to linear, scale to nits.
fn nits_gradient(w: u32, h: u32, seed: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x.wrapping_add(seed)) & 0xff) as u8;
            let g = ((y.wrapping_add(seed * 3)) & 0xff) as u8;
            let b = ((x ^ y ^ seed) & 0xff) as u8;
            out.push(srgb_eotf(r) * HDR_PEAK_NITS);
            out.push(srgb_eotf(g) * HDR_PEAK_NITS);
            out.push(srgb_eotf(b) * HDR_PEAK_NITS);
        }
    }
    out
}

#[test]
fn cpu_butter_hdr_linear_identity_and_discrimination() {
    let (w, h) = (64_u32, 64_u32);
    let reference = nits_gradient(w, h, 0);
    let distorted = nits_gradient(w, h, 9);

    let mut cpu = HdrScorer::new(
        zenmetrics_api::MetricKind::Butter,
        Backend::Cpu,
        w,
        h,
        HDR_PEAK_NITS,
    )
    .expect("cpu butter scorer");

    // Identity: butter == 0 (lower is better).
    let id = cpu
        .compute_multi(&reference, &reference)
        .expect("cpu identity");
    let id_max = id.primary();
    assert!(
        id_max < 0.05,
        "CPU butter HDR identity should be ~0, got {id_max}"
    );
    // pnorm_3 preserved through the native linear path.
    assert!(
        id.get("pnorm_3").is_some(),
        "CPU butter multi must keep pnorm_3"
    );

    // Distortion discriminates clearly.
    let dist = cpu
        .compute_multi(&reference, &distorted)
        .expect("cpu distorted");
    assert!(
        dist.primary() > 1.0,
        "CPU butter HDR should detect distortion (>1), got {}",
        dist.primary()
    );
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_ne_bytes());
    }
    out
}

/// The zenpixels descriptor-driven entry on `Backend::Cpu`: a display-relative
/// `RGBF32_LINEAR` slice through `compute_pixels_multi` routes to the native CPU
/// butter linear path and matches `compute_multi(nits)` — same native feeding,
/// one descriptor-driven call. (zenpixels is the primary entry; CPU is native.)
#[test]
fn cpu_butter_hdr_via_zenpixels_slice_matches_nits() {
    use zenpixels::{PixelDescriptor, PixelSlice};
    let (w, h) = (64_u32, 64_u32);
    // Display-relative [0,1] linear, and the same content as nits (×peak).
    let to_lin = |seed: u32| -> Vec<f32> {
        let mut out = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                out.push(srgb_eotf(((x.wrapping_add(seed)) & 0xff) as u8));
                out.push(srgb_eotf(((y.wrapping_add(seed * 3)) & 0xff) as u8));
                out.push(srgb_eotf(((x ^ y ^ seed) & 0xff) as u8));
            }
        }
        out
    };
    let ref_lin = to_lin(0);
    let dis_lin = to_lin(9);
    let ref_nits: Vec<f32> = ref_lin.iter().map(|&v| v * HDR_PEAK_NITS).collect();
    let dis_nits: Vec<f32> = dis_lin.iter().map(|&v| v * HDR_PEAK_NITS).collect();
    let row = (w * 3 * 4) as usize;

    let mut cpu = HdrScorer::new(
        zenmetrics_api::MetricKind::Butter,
        Backend::Cpu,
        w,
        h,
        HDR_PEAK_NITS,
    )
    .expect("cpu");

    let via_nits = cpu
        .compute_multi(&ref_nits, &dis_nits)
        .expect("nits")
        .primary();
    let via_pixels = cpu
        .compute_pixels_multi(
            PixelSlice::new(
                &f32_bytes(&ref_lin),
                w,
                h,
                row,
                PixelDescriptor::RGBF32_LINEAR,
            )
            .unwrap(),
            PixelSlice::new(
                &f32_bytes(&dis_lin),
                w,
                h,
                row,
                PixelDescriptor::RGBF32_LINEAR,
            )
            .unwrap(),
        )
        .expect("pixels")
        .primary();
    let rel = (via_nits - via_pixels).abs() / via_nits.abs().max(1e-6);
    assert!(
        rel < 1e-4,
        "CPU butter zenpixels slice ({via_pixels}) must match nits path ({via_nits}); rel {rel}"
    );
}

#[test]
fn cpu_butter_hdr_linear_tracks_gpu() {
    let (w, h) = (64_u32, 64_u32);
    let reference = nits_gradient(w, h, 0);
    let distorted = nits_gradient(w, h, 9);

    let mut cpu = HdrScorer::new(
        zenmetrics_api::MetricKind::Butter,
        Backend::Cpu,
        w,
        h,
        HDR_PEAK_NITS,
    )
    .expect("cpu");
    let mut gpu = HdrScorer::new(
        zenmetrics_api::MetricKind::Butter,
        Backend::Cuda,
        w,
        h,
        HDR_PEAK_NITS,
    )
    .expect("gpu");

    let cpu_d = cpu
        .compute_multi(&reference, &distorted)
        .expect("cpu")
        .primary();
    let gpu_d = gpu
        .compute_multi(&reference, &distorted)
        .expect("gpu")
        .primary();

    // Native CPU (butteraugli C-port) and GPU (cubecl) are independent
    // implementations of the same metric on the same faithful linear feeding —
    // they won't be bit-identical, but must agree closely on the SAME pair.
    // Report the gap; assert both detect and are within a cross-impl band.
    eprintln!("butter HDR linear: cpu={cpu_d}  gpu={gpu_d}");
    assert!(
        cpu_d > 1.0 && gpu_d > 1.0,
        "both must detect: cpu={cpu_d} gpu={gpu_d}"
    );
    let rel = (cpu_d - gpu_d).abs() / gpu_d.abs().max(1e-6);
    assert!(
        rel < 0.10,
        "CPU vs GPU butter HDR linear should track within 10%: cpu={cpu_d} gpu={gpu_d} (rel {rel})"
    );
}
