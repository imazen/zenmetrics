//! Regression: GPU zensim scoring must stay correct after a
//! *different-size* pipeline was built and dropped on the same
//! (process-global) cubecl client first.
//!
//! Bug (2026-06-22): `zenmetrics sweep --metric zensim-gpu` produced a
//! ~7-9 JOD divergence at 1448×1448 ONLY when a differently-sized
//! pipeline (e.g. 1024²) had been constructed, used, and dropped on the
//! same client earlier in the run. 1448 in isolation, and 1448 via the
//! standalone `score` subcommand, were both correct; only after a
//! foreign-size rebuild did the next Full-mode score diverge. Root
//! cause: a freed cubecl pool page from the previous pipeline was
//! handed back to the new pipeline's `client.empty()` allocation still
//! holding the previous pipeline's data, and read before being
//! overwritten — a stale-page read.
//!
//! The fix (`zenmetrics-cli` `MetricCache::get_or_build_umbrella`)
//! reclaims the dropped pipeline's pooled VRAM before building the new
//! one, so the new allocation gets clean memory. This test reproduces
//! the hazard at the crate layer: build + score a 1024² Full pipeline,
//! DROP it, then build a 1448² Full pipeline and check both GPU scoring
//! paths against the CPU `zensim` ground truth. Forcing `MemoryMode::Full`
//! makes the test deterministic regardless of the host's live free-VRAM
//! (the Auto resolver would otherwise pick Strip on a busy GPU and dodge
//! the Full-path hazard).
//!
//! NOTE: this test exercises the *crate-level* hazard. The CLI fix in
//! `MetricCache` is what protects the `sweep` / `score-pairs` workers;
//! this gate proves the underlying pipeline produces a correct score
//! once clean pages are guaranteed (and documents the failure mode so a
//! future cubecl bump that changes pool reuse can't silently regress it).

use zensim::{RgbSlice, Zensim as ZensimCpu, ZensimProfile};
use zensim_gpu::{Backend, MemoryMode, ZensimOpaque, ZensimParams};

#[cfg(feature = "cuda")]
fn backend() -> Backend {
    Backend::Cuda
}
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
fn backend() -> Backend {
    Backend::Wgpu
}
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
compile_error!("cached_ref_slot_rebuild test requires the `cuda` or `wgpu` feature");

/// Deterministic textured RGB image — varied enough that the masked/IW
/// activity terms are non-trivial (a flat image would hide the bug).
fn textured(w: usize, h: usize, seed_off: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 3);
    let mut s = std::num::Wrapping(0x1234_5678u32 ^ seed_off);
    for y in 0..h {
        for x in 0..w {
            s = s * std::num::Wrapping(1664525) + std::num::Wrapping(1013904223);
            let n = (s.0 >> 16) as u8;
            let r = (((x * 255) / w) as u8).wrapping_add(n / 8);
            let g = (((y * 255) / h) as u8).wrapping_add(n / 8);
            let b = ((((x + y) * 255) / (w + h)) as u8) ^ (n / 16);
            v.push(r);
            v.push(g);
            v.push(b);
        }
    }
    v
}

/// Coarse-quantize distortion (non-identity, JPEG-ish).
fn distort(src: &[u8], step: u8) -> Vec<u8> {
    src.iter()
        .map(|&p| (p / step) as u16 * step as u16)
        .map(|q| q.min(255) as u8)
        .collect()
}

fn to_pix(b: &[u8]) -> Vec<[u8; 3]> {
    b.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

fn cpu_score(w: u32, h: u32, r: &[u8], d: &[u8]) -> f64 {
    let z = ZensimCpu::new(ZensimProfile::A);
    let rp = to_pix(r);
    let dp = to_pix(d);
    let rs = RgbSlice::new(&rp, w as usize, h as usize);
    let ds = RgbSlice::new(&dp, w as usize, h as usize);
    z.compute(&rs, &ds).expect("cpu compute").score()
}

fn full(w: u32, h: u32) -> ZensimOpaque {
    ZensimOpaque::new_with_memory_mode(
        backend(),
        w,
        h,
        ZensimParams::default_weights(),
        MemoryMode::Full,
    )
    .expect("build opaque (Full)")
}

fn gpu_oneshot(w: u32, h: u32, r: &[u8], d: &[u8]) -> f64 {
    full(w, h).compute_srgb_u8(r, d).expect("one-shot score").value
}

fn gpu_cached(w: u32, h: u32, r: &[u8], d: &[u8]) -> f64 {
    let mut m = full(w, h);
    m.set_reference_srgb_u8(r).expect("set reference");
    m.compute_with_reference_srgb_u8(d)
        .expect("cached-ref score")
        .value
}

#[test]
fn gpu_score_correct_after_foreign_size_rebuild() {
    // 1. Build + score a 1024² Full pipeline, then DROP it — seeds the
    //    cubecl pool with freed pages of a different geometry, exactly
    //    like the sweep scoring image N-1 of a different size.
    {
        let w = 1024u32;
        let h = 1024u32;
        let r = textured(w as usize, h as usize, 0);
        let d = distort(&r, 24);
        let _ = gpu_cached(w, h, &r, &d); // value unused; side effect = pool churn
    }

    // 2. Now 1448² Full (padded width 1456 → 8 mirror-pad columns, the
    //    geometry that tripped the original bug). Both GPU scoring paths
    //    must match the CPU ground truth.
    let w = 1448u32;
    let h = 1448u32;
    let r = textured(w as usize, h as usize, 99);
    let d = distort(&r, 24);

    let cpu = cpu_score(w, h, &r, &d);
    let g_one = gpu_oneshot(w, h, &r, &d);
    let g_cached = gpu_cached(w, h, &r, &d);
    eprintln!("1448 after 1024-rebuild: cpu={cpu:.4} gpu_oneshot={g_one:.4} gpu_cached={g_cached:.4}");

    for (label, g) in [("one-shot", g_one), ("cached-ref", g_cached)] {
        assert!(g.is_finite(), "{label} GPU score must be finite, got {g}");
        let diff = (g - cpu).abs();
        assert!(
            diff <= 0.10,
            "{label} GPU score {g} diverges from CPU {cpu} by {diff} JOD (> 0.10) \
             after a foreign-size pipeline rebuild — stale cubecl pool page read"
        );
    }
}
