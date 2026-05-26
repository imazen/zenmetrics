//! Cross-repo GPU SSIM2 parity test — zenmetrics CubeCL vs
//! coefficient cudarse/turbo-metrics.
//!
//! # Why this exists
//!
//! Master dedup audit
//! (`zensim/benchmarks/dedup_inventory_master_2026-05-26.md`,
//! Tier-0 #2 / §A.6 #1 / Class 7) flagged that we ship TWO independent
//! GPU SSIMULACRA2 implementations of the *same* metric:
//!
//! 1. **zenmetrics CubeCL** (this crate, `ssim2-gpu`) — multi-vendor via
//!    the lilith/cubecl fork; runs on CUDA, wgpu (incl. DX12/Metal/Vulkan),
//!    HIP, and CPU.
//! 2. **coefficient cudarse** (`coefficient::gpu::GpuMetrics`, 299 LOC,
//!    `src/gpu.rs`) — NVIDIA-only via `ssimulacra2-cuda` /
//!    `cudarse-driver` / `cudarse-npp` from `../turbo-metrics`.
//!
//! Same metric → two backends → no published parity test → silent
//! risk that they score differently and no one notices until a join
//! mixes their outputs.  This test is the cheapest first step in the
//! audit's recommendation chain:
//!
//! > parity test gate first; then pick one backend or a shared
//! > `zen-gpu-metrics`
//!
//! # Methodology
//!
//! * Both backends accept the **same input shape**: packed sRGB-RGB8
//!   `&[u8]` of length `width * height * 3`, row-major, no stride
//!   padding.
//! * Both return a scalar SSIM2 score where **higher = better, 100 =
//!   identical**.  zenmetrics returns `f64`; coefficient returns `f64`.
//! * For each of three fixtures (1 ref + 1 distorted variant), score
//!   the pair on both backends and assert the absolute delta is below
//!   the tolerance documented in `docs/GPU_METRIC_PARITY.md`.
//!
//! # Tolerance
//!
//! Initial gate: **abs(zenmetrics - coefficient) < 0.5 SSIM2 points**.
//!
//! Two independent GPU implementations of the same multi-scale
//! perceptual metric, with different reduction orders, different
//! Gaussian-blur kernels (CubeCL vs NPP), different XYB linearization
//! sites, etc. should agree to within a small fraction of a SSIM2
//! point on the same input — but they will never agree to 1e-6 the
//! way two scalar libm calls do.  Start loose, tighten in a follow-on
//! once the measured agreement is in the table.  The expected band per
//! the audit is "small but nonzero"; anything over 1 SSIM2 point is a
//! **finding** that needs documentation in the parity doc.
//!
//! # Why `#[ignore]`
//!
//! coefficient's `gpu` feature has path-deps on `../turbo-metrics`
//! which, as of 2026-05-26, was archived to
//! `~/work/turbo-metrics--archived-2026-05-06`.  The path resolution
//! depends on operator workspace layout AND on the presence of the
//! NVIDIA CUDA driver and toolkit.  Marking the test `#[ignore]` lets
//! `cargo check -p ssim2-gpu --features cudarse-parity` validate the
//! signature surface without forcing all CI matrix legs to depend on
//! CUDA, while leaving the test runnable on machines that have the
//! stack:
//!
//! ```text
//! cargo test -p ssim2-gpu --features cudarse-parity --test cudarse_parity -- --ignored
//! ```
//!
//! When the test runs, record the measured deltas in
//! `docs/GPU_METRIC_PARITY.md` and tighten the tolerance constant
//! below as appropriate.
//!
//! # NOT in scope (follow-on candidates)
//!
//! * Butteraugli parity (`butteraugli-gpu` vs `butteraugli-cuda`).
//! * DSSIM parity (`dssim-gpu` vs `dssim-cuda`).
//! * Per-octave reduction parity (cubecl's
//!   `reduction_determinism` test covers single-backend stability;
//!   cross-backend agreement on the per-octave intermediates is a
//!   tighter gate worth landing AFTER the scalar score gate
//!   stabilizes).

#![cfg(all(feature = "cudarse-parity", feature = "cuda", feature = "cubecl-types"))]

use ssim2_gpu::{Backend, Ssim2Opaque, Ssim2Params};

/// Max permitted abs(zenmetrics_score - coefficient_score).
///
/// Initial loose gate.  Tighten after `docs/GPU_METRIC_PARITY.md`
/// records measured agreement.
const PARITY_TOLERANCE_SSIM2: f64 = 0.5;

/// 3 CID22 fixtures, one source + one distorted variant each.
///
/// Variant pattern: light JPEG / heavy JPEG / a 3rd photo at moderate
/// distortion, to cover the SSIM2 range from ~95 (near-identical) down
/// to ~70 (visibly degraded) where the two backends are most likely
/// to disagree.
///
/// Distorted variants are generated on the fly inside the test (cheap
/// JPEG encode+decode through the `image` crate) so the test stays
/// self-contained — no fixture pre-staging step.
const FIXTURES: &[FixtureSpec] = &[
    FixtureSpec {
        source_rel: "CID22/CID22-512/training/1001682.png",
        jpeg_quality: 90, // ~near-identical, score should be high 90s
        label: "1001682_q90",
    },
    FixtureSpec {
        source_rel: "CID22/CID22-512/training/1028637.png",
        jpeg_quality: 50, // mid-range, score should be ~80
        label: "1028637_q50",
    },
    FixtureSpec {
        source_rel: "CID22/CID22-512/training/1029604.png",
        jpeg_quality: 20, // heavy distortion, score should be ~65 or lower
        label: "1029604_q20",
    },
];

struct FixtureSpec {
    /// Path under `~/work/codec-corpus/` to the reference PNG.
    source_rel: &'static str,
    /// JPEG quality (1..=100) used to synthesize the distorted
    /// variant via `image` crate roundtrip.
    jpeg_quality: u8,
    /// Short label used in the per-fixture assertion message.
    label: &'static str,
}

/// Resolve `~/work/codec-corpus/{rel}` → loaded sRGB-RGB8 buffer + dims.
fn load_corpus_rgb8(rel: &str) -> (Vec<u8>, u32, u32) {
    let home = std::env::var("HOME").expect("HOME env var");
    let path = format!("{}/work/codec-corpus/{}", home, rel);
    let img = image::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let rgb8 = img.to_rgb8();
    let (w, h) = (rgb8.width(), rgb8.height());
    (rgb8.into_raw(), w, h)
}

/// Round-trip the source through JPEG at the given quality and return
/// the decoded RGB8 buffer.
fn synth_distorted_jpeg(source_rgb8: &[u8], w: u32, h: u32, quality: u8) -> Vec<u8> {
    use image::ImageEncoder;
    use std::io::Cursor;

    let mut jpeg_bytes: Vec<u8> = Vec::new();
    {
        let mut cursor = Cursor::new(&mut jpeg_bytes);
        let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, quality);
        encoder
            .write_image(source_rgb8, w, h, image::ExtendedColorType::Rgb8)
            .expect("jpeg encode");
    }
    let decoded = image::load_from_memory_with_format(&jpeg_bytes, image::ImageFormat::Jpeg)
        .expect("jpeg decode")
        .to_rgb8();
    assert_eq!(decoded.width(), w);
    assert_eq!(decoded.height(), h);
    decoded.into_raw()
}

#[test]
#[ignore = "requires CUDA driver + coefficient's gpu feature (path-dep on \
            ../turbo-metrics which may be archived); see test header for \
            run instructions"]
fn cudarse_vs_cubecl_ssim2_within_tolerance() {
    // Initialize CUDA once before either backend is touched.
    coefficient::gpu::init_cuda().expect("init_cuda");

    let mut max_delta: f64 = 0.0;
    let mut findings: Vec<String> = Vec::new();

    for fx in FIXTURES {
        let (ref_rgb, w, h) = load_corpus_rgb8(fx.source_rel);
        let dis_rgb = synth_distorted_jpeg(&ref_rgb, w, h, fx.jpeg_quality);
        assert_eq!(ref_rgb.len(), dis_rgb.len());
        assert_eq!(ref_rgb.len(), (w * h * 3) as usize);

        // CubeCL CUDA path (this crate).
        let mut cubecl = Ssim2Opaque::new(Backend::Cuda, w, h, Ssim2Params::DEFAULT)
            .expect("Ssim2Opaque::new");
        let cubecl_score = cubecl
            .compute_srgb_u8(&ref_rgb, &dis_rgb)
            .expect("cubecl compute_srgb_u8")
            .value;
        drop(cubecl); // release the cubecl client/handle before cudarse warms up

        // cudarse/turbo-metrics path (coefficient).
        let mut cudarse_ctx = coefficient::gpu::GpuMetrics::new(w, h)
            .expect("GpuMetrics::new");
        let cudarse_score = cudarse_ctx
            .ssimulacra2(&ref_rgb, &dis_rgb)
            .expect("cudarse ssimulacra2");

        let delta = (cubecl_score - cudarse_score).abs();
        if delta > max_delta {
            max_delta = delta;
        }

        eprintln!(
            "fixture={} dims={}x{} cubecl={:.6} cudarse={:.6} delta={:.6}",
            fx.label, w, h, cubecl_score, cudarse_score, delta
        );

        if delta >= PARITY_TOLERANCE_SSIM2 {
            findings.push(format!(
                "{}: cubecl={:.6} cudarse={:.6} delta={:.6} >= tolerance={:.6} (source: {})",
                fx.label,
                cubecl_score,
                cudarse_score,
                delta,
                PARITY_TOLERANCE_SSIM2,
                fx.source_rel,
            ));
        }
    }

    eprintln!("max_delta across fixtures = {:.6}", max_delta);

    assert!(
        findings.is_empty(),
        "GPU SSIM2 backends disagree beyond tolerance on {} of {} fixtures:\n  {}\n\
         Update docs/GPU_METRIC_PARITY.md with the measured deltas.",
        findings.len(),
        FIXTURES.len(),
        findings.join("\n  "),
    );
}
