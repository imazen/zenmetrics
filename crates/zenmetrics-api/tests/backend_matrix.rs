//! Brute-force backend × metric × size matrix (task #159 phase 5).
//!
//! Two always-on layers (run in CI via the cpu-metrics job) plus a
//! GPU-parity layer that compiles in only when the `cuda` feature is
//! present (mirrors the `session_parity` convention — a real GPU host
//! runs it, CI's CPU job compiles it out, never a runtime graceful skip):
//!
//! 1. **CPU breadth** — every metric on [`Backend::Cpu`] at 256 / 512 /
//!    1024, asserting a finite, *discriminating* score (identical vs a
//!    real distortion differ by a per-metric floor). Proves the optimized
//!    native path runs the metric, not a constant, across sizes.
//! 2. **Auto → Cpu** — with `ZENMETRICS_FORCE_NO_GPU=1`, [`Backend::Auto`]
//!    must resolve to [`Backend::Cpu`] for every metric AND score
//!    byte-identically to constructing `Backend::Cpu` explicitly (Auto
//!    resolution never changes the score).
//! 3. **CPU vs CUDA parity** (`#[cfg(feature = "cuda")]`) — every metric,
//!    every size, `|cpu − gpu|` within a measured, documented tolerance.
//!
//! Gated via the Cargo.toml `[[test]] required-features = ["cpu-metrics"]`
//! entry: the skip decision lives in the CI→test chain, not the test body.

use zenmetrics_api::{Backend, Metric, MetricKind, MetricParams};

/// Sizes swept by the matrix. All ≥ 176 so IW-SSIM's minimum side holds.
const SIZES: [u32; 3] = [256, 512, 1024];

/// Every metric the umbrella exposes.
fn all_kinds() -> [MetricKind; 6] {
    [
        MetricKind::Ssim2,
        MetricKind::Cvvdp,
        MetricKind::Dssim,
        MetricKind::Butter,
        MetricKind::Iwssim,
        MetricKind::Zensim,
    ]
}

/// Minimum `|identical − distorted|` gap that proves the metric ran (not a
/// constant). Per-metric because the scales differ wildly (SSIM2 ~0..100,
/// CVVDP ~3..10 JOD, DSSIM/Butter near 0, IW-SSIM 0..1, Zensim ~0..1).
/// Values are the conservative floors already used by `cpu_dispatch.rs`.
fn discrimination_floor(kind: MetricKind) -> f64 {
    match kind {
        MetricKind::Ssim2 => 10.0,
        MetricKind::Cvvdp => 0.5,
        MetricKind::Dssim => 0.01,
        MetricKind::Butter => 0.1,
        MetricKind::Iwssim => 0.1,
        MetricKind::Zensim => 1e-4,
    }
}

/// Deterministic `w×h` packed sRGB (`R, G, B, …`) image from `f`.
fn img(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> Vec<u8> {
    let mut v = Vec::with_capacity((w as usize) * (h as usize) * 3);
    for y in 0..h {
        for x in 0..w {
            v.extend_from_slice(&f(x, y));
        }
    }
    v
}

/// Reference image: gradient + xor texture (high-frequency detail at every
/// size so the metrics have real structure to compare).
fn ref_img(w: u32, h: u32) -> Vec<u8> {
    img(w, h, |x, y| {
        [
            x.wrapping_mul(4) as u8,
            y.wrapping_mul(4) as u8,
            (x ^ y).wrapping_mul(3) as u8,
        ]
    })
}

/// Distorted image: channel-inverted vs the reference — a large, real
/// perceptual difference every metric must register.
fn dist_img(w: u32, h: u32) -> Vec<u8> {
    img(w, h, |x, y| {
        [
            255 - x.wrapping_mul(4) as u8,
            255 - y.wrapping_mul(4) as u8,
            128,
        ]
    })
}

/// Construct + score a `(reference, distorted)` pair on `backend`.
fn score_pair(kind: MetricKind, backend: Backend, w: u32, h: u32, r: &[u8], d: &[u8]) -> f64 {
    let mut m = Metric::new(kind, backend, w, h, MetricParams::default_for(kind))
        .unwrap_or_else(|e| panic!("{kind:?} on {backend:?} at {w}x{h} must construct: {e}"));
    m.compute_srgb_u8(r, d)
        .unwrap_or_else(|e| panic!("{kind:?} on {backend:?} at {w}x{h} must score: {e}"))
        .value
}

/// Layer 1: every metric on `Backend::Cpu`, every size — finite +
/// discriminating.
#[test]
fn cpu_matrix_all_metrics_all_sizes() {
    for kind in all_kinds() {
        for &s in &SIZES {
            let r = ref_img(s, s);
            let d = dist_img(s, s);
            let identical = score_pair(kind, Backend::Cpu, s, s, &r, &r);
            let distorted = score_pair(kind, Backend::Cpu, s, s, &r, &d);
            assert!(
                identical.is_finite() && distorted.is_finite(),
                "{kind:?} @ {s}x{s}: non-finite (identical={identical}, distorted={distorted})"
            );
            let gap = (identical - distorted).abs();
            assert!(
                gap >= discrimination_floor(kind),
                "{kind:?} @ {s}x{s}: gap {gap} below floor {} (identical={identical}, distorted={distorted})",
                discrimination_floor(kind)
            );
        }
    }
}

/// Layer 2: `Backend::Auto` must resolve to the optimized native
/// `Backend::Cpu` when no GPU is available, and score byte-identically to
/// an explicit `Backend::Cpu` for every metric.
///
/// All `ZENMETRICS_FORCE_NO_GPU` mutation lives in this single `#[test]`
/// fn (set → assert → restore) so the process-global env var can't race a
/// sibling test — the same discipline `backend_resolve.rs` uses. No other
/// fn in this binary reads the variable (the CPU/CUDA layers use explicit
/// backends), so even the cargo multi-thread interleaving is benign here.
#[test]
fn auto_force_no_gpu_resolves_to_cpu_and_matches() {
    // SAFETY: edition-2024 marks env mutation `unsafe`; this integration
    // test is a separate compilation unit (not `#![forbid(unsafe_code)]`),
    // and the set/restore is confined to this single fn (see doc above).
    unsafe {
        std::env::set_var("ZENMETRICS_FORCE_NO_GPU", "1");
    }

    assert_eq!(
        Backend::resolve_auto(),
        Backend::Cpu,
        "with cpu-metrics built and no GPU, Auto must resolve to optimized Cpu"
    );

    // Use a small size so all six metrics (incl. iwssim's 176 minimum) run
    // quickly; the equivalence is size-independent.
    let (w, h) = (256u32, 256u32);
    let r = ref_img(w, h);
    let d = dist_img(w, h);
    for kind in all_kinds() {
        let via_auto = score_pair(kind, Backend::Auto, w, h, &r, &d);
        let via_cpu = score_pair(kind, Backend::Cpu, w, h, &r, &d);
        assert_eq!(
            via_auto, via_cpu,
            "{kind:?}: Auto (→Cpu) must score identically to explicit Cpu"
        );
    }

    // SAFETY: same as above — restore the environment for sibling tests.
    unsafe {
        std::env::remove_var("ZENMETRICS_FORCE_NO_GPU");
    }
}

// ---------------------------------------------------------------------------
// Layer 3: CPU vs CUDA parity (real-GPU host only — compiles in only with the
// `cuda` feature, mirroring `session_parity`'s gate). The CPU crates and the
// `-gpu` cubecl kernels are INDEPENDENT implementations of each metric, so the
// cross-backend deltas are measured (not assumed) and documented as the
// per-metric tolerance below.
// ---------------------------------------------------------------------------

/// CPU-vs-CUDA parity: every metric, every size, both backends produce a
/// finite, *discriminating* score AND agree within a per-metric tolerance.
///
/// The CPU crates and the cubecl `-gpu` kernels are INDEPENDENT
/// implementations, so the tolerances are MEASURED, not assumed. Measured
/// `max|Δ|` on an RTX 5070 over the fixed fixtures below
/// (`benchmarks/backend_parity_cpu_vs_cuda_2026-06-01.tsv`), identical and
/// distorted pairs:
///
/// | metric | max\|Δident\| | max\|Δdist\| | tolerance |
/// |--------|--------------|-------------|-----------|
/// | ssim2  | 0.0081       | 0.0762      | 0.5       |
/// | cvvdp  | 0.0000       | 0.0011      | 0.05      |
/// | dssim  | 0.0000       | 0.0000      | 0.01      |
/// | butter | 0.0000       | 3.0031      | 4.0       |
/// | iwssim | 0.0000       | 0.0466      | 0.15      |
/// | zensim | 0.0000       | 0.0014      | 0.05      |
///
/// butteraugli's larger delta is a *stable* cross-impl offset (CPU
/// `butteraugli` vs `butteraugli-gpu` settle on slightly different max-norm
/// for the channel-inverted, norm-saturating fixture — ~1.5% of the ~200
/// magnitude, byte-identical across all three sizes), not run-to-run noise.
/// Each tolerance guards against *regression* past the established cross-impl
/// delta — not bit-equality the independent impls never had.
#[cfg(feature = "cuda")]
#[test]
fn cpu_vs_cuda_parity() {
    // Per-metric CPU-vs-CUDA tolerance: measured max|Δ| (see table above) with
    // a safety margin so legitimate float-order variance never flakes while a
    // real backend regression still trips it.
    fn tolerance(kind: MetricKind) -> f64 {
        match kind {
            MetricKind::Ssim2 => 0.5,
            MetricKind::Cvvdp => 0.05,
            MetricKind::Dssim => 0.01,
            MetricKind::Butter => 4.0,
            MetricKind::Iwssim => 0.15,
            MetricKind::Zensim => 0.05,
        }
    }
    eprintln!("METRIC      SIZE     cpu_ident  cpu_dist   gpu_ident  gpu_dist   |Δident|  |Δdist|");
    for kind in all_kinds() {
        let tol = tolerance(kind);
        for &s in &SIZES {
            let r = ref_img(s, s);
            let d = dist_img(s, s);
            let cpu_id = score_pair(kind, Backend::Cpu, s, s, &r, &r);
            let cpu_di = score_pair(kind, Backend::Cpu, s, s, &r, &d);
            let gpu_id = score_pair(kind, Backend::Cuda, s, s, &r, &r);
            let gpu_di = score_pair(kind, Backend::Cuda, s, s, &r, &d);
            // Both backends must run the metric (finite + discriminating).
            for (be, id, di) in [("cpu", cpu_id, cpu_di), ("gpu", gpu_id, gpu_di)] {
                assert!(
                    id.is_finite() && di.is_finite(),
                    "{kind:?} @ {s}x{s} {be}: non-finite (ident={id}, dist={di})"
                );
                assert!(
                    (id - di).abs() >= discrimination_floor(kind),
                    "{kind:?} @ {s}x{s} {be}: gap {} below floor {}",
                    (id - di).abs(),
                    discrimination_floor(kind)
                );
            }
            let d_id = (cpu_id - gpu_id).abs();
            let d_di = (cpu_di - gpu_di).abs();
            eprintln!(
                "{:<11} {:>4}   {:>9.4} {:>9.4}   {:>9.4} {:>9.4}   {:>8.4} {:>8.4}",
                format!("{kind:?}"),
                s,
                cpu_id,
                cpu_di,
                gpu_id,
                gpu_di,
                d_id,
                d_di,
            );
            assert!(
                d_id <= tol,
                "{kind:?} @ {s}x{s}: identical-pair CPU {cpu_id} vs CUDA {gpu_id} differ by {d_id} > tol {tol}"
            );
            assert!(
                d_di <= tol,
                "{kind:?} @ {s}x{s}: distorted-pair CPU {cpu_di} vs CUDA {gpu_di} differ by {d_di} > tol {tol}"
            );
        }
    }
}
