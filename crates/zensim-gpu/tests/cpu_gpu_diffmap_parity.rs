//! Phase 1b CPU↔GPU diffmap pointwise parity test.
//!
//! Compares the **pure-GPU diffmap path** (chunk-3 wiring:
//! `Zensim::score_from_linear_planes_with_diffmap`, which runs the
//! WithIw GPU feature pipeline + the chunk-1/2 diffmap kernels) against
//! the **CPU canonical** `zensim::Zensim::compute_with_ref_and_diffmap_linear_planar`
//! for the default `DiffmapOptions` path.
//!
//! For each of 5 fixtures × 4 distortion levels:
//!   • pointwise diffmap parity within [`DIFFMAP_ABS_TOL`] absolute
//!     (GPU uses f32 in the per-cell SSIM map arithmetic; CPU uses f32
//!     too in the per-pixel `sd0`, but the multi-scale fusion +
//!     downscale pyramid differ in padding-column geometry — only the
//!     LOGICAL columns are compared after trim, so the envelope is the
//!     steady-state f32-roundoff drift between the GPU FMA-fused
//!     opsin/SSIM chain and the CPU SIMD chain);
//!   • scalar score parity within [`SCORE_ABS_TOL`] (the GPU score is
//!     the CPU V0_3 MLP run on GPU-extracted features; the only drift
//!     is the f32 feature-extraction divergence the `cpu_parity` test
//!     already bounds).
//!
//! Tolerance provenance: see `docs/DIFFMAP_DIVERGENCES.md` §11. The
//! measured envelope at land time is recorded there; the constants
//! below carry a safety margin above the measured max.
//!
//! Requires a real GPU runtime (CUDA or wgpu) — the CpuRuntime JIT is
//! exercised by `tests/diffmap_invariants.rs` instead.

#![cfg(feature = "cubecl-types")]

use cubecl::Runtime;
use zensim::{DiffmapOptions, Zensim as ZensimCpu, ZensimProfile};
use zensim_gpu::Zensim;

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
compile_error!(
    "cpu_gpu_diffmap_parity requires either the `cuda` or `wgpu` feature to select a GPU runtime"
);

/// Pointwise diffmap absolute tolerance. The GPU per-scale SSIM map +
/// upsample-fuse runs in f32 with the same FMA fusion order as the CPU
/// scalar `sd0`; the residual envelope comes from the opsin-matrix +
/// cbrt + downscale pyramid f32 drift, plus the GPU vs CPU diffmap
/// pyramid having different padding-column widths per scale (logical
/// columns only are compared after trim).
///
/// **Measured** max over the 5 fixtures × 4 distortions in this test at
/// land time: `2.08e-4` (CUDA, RTX 5070). The `1e-3` tolerance carries
/// a ~5× safety margin — see `docs/DIFFMAP_DIVERGENCES.md` §11.
///
/// Tight `1e-3` for **all** backends. cubecl emits the same kernel
/// source everywhere; CUDA matches the CPU canonical to `2.08e-4`, so
/// `1e-3` is a genuine regression gate there.
///
/// **Known Metal bug — NOT f32 fast-math, NOT our kernel.** On Apple
/// Metal this test fails with a max pointwise diffmap error of **`1.098`**
/// at fixture 1 (96×80) under the *lightest* distortion (`delta=3`) —
/// measured on Metal CI 2026-06-02/03. A >1.0 per-pixel divergence at the
/// lightest delta is far outside any fast-math envelope. The backend
/// matrix isolates it precisely (all on RTX 5070 except Metal CI):
///   • CUDA (cubecl-cuda):                       ✅ 2.08e-4
///   • Vulkan (cubecl-wgpu → naga → SPIR-V):     ✅ 2.2e-4  (local)
///   • Metal  (cubecl-wgpu → naga → MSL):        ❌ 1.098
/// Vulkan runs the *identical* cubecl-wgpu WGSL through naga and is
/// correct, so the defect is NOT in our kernel logic and NOT in cubecl's
/// WGSL generation — it lives in the WGSL→MSL (naga) translation or
/// Apple's Metal shader compiler. An earlier revision widened the wgpu
/// tolerance to `1.5e-1` assuming fast-math; that masked the bug and has
/// been reverted. The test is left RED on Metal so the bug stays visible.
/// Tracked in #20 (needs Metal hardware to debug; likely a naga/wgpu
/// upstream report once the per-pixel pattern is captured).
const DIFFMAP_ABS_TOL: f32 = 1e-3;

/// Scalar score absolute tolerance (butteraugli-direction, 0..100
/// scale). The GPU score is the CPU V0_3 MLP evaluated on GPU-extracted
/// 372 features; the drift is the f32 feature-extraction divergence
/// (bounded by `cpu_parity`). 0.1 score units (== the existing
/// linear-vs-srgb invariant tolerance).
const SCORE_ABS_TOL: f32 = 0.1;

macro_rules! make_zensim {
    ($w:expr, $h:expr) => {{
        let client = Backend::client(&Default::default());
        Zensim::<Backend>::new(client, $w, $h).expect("Zensim::new")
    }};
}

/// sRGB EOTF, single channel.
fn srgb_to_lin(b: u8) -> f32 {
    let v = b as f32 / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Build a `width × height` sRGB-u8 image via a pseudo-random LCG with
/// a structured gradient overlay (exercises all 3 channels + edges).
fn make_fixture(seed: u32, w: usize, h: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(w * h * 3);
    let mut s = seed.wrapping_add(1);
    for y in 0..h {
        for x in 0..w {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = (s >> 16) & 0x3f; // 0..63
            let r = (((x * 255) / w.max(1)) as u32 + noise).min(255) as u8;
            let g = (((y * 255) / h.max(1)) as u32 + (noise >> 1)).min(255) as u8;
            let b = ((((x + y) * 255) / (w + h).max(1)) as u32 + (noise >> 2)).min(255) as u8;
            out.push(r);
            out.push(g);
            out.push(b);
        }
    }
    out
}

/// Apply additive distortion (clamped).
fn distort(src: &[u8], delta: i32) -> Vec<u8> {
    src.iter()
        .map(|&v| (v as i32 + delta).clamp(0, 255) as u8)
        .collect()
}

/// Decode packed sRGB-u8 → 3 tight linear-f32 planes.
fn to_linear_planes(rgb: &[u8], n: usize) -> [Vec<f32>; 3] {
    let mut r = vec![0.0f32; n];
    let mut g = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    for (i, p) in rgb.chunks_exact(3).enumerate() {
        r[i] = srgb_to_lin(p[0]);
        g[i] = srgb_to_lin(p[1]);
        b[i] = srgb_to_lin(p[2]);
    }
    [r, g, b]
}

/// Force the opt-in GPU diffmap path on for this test process. The
/// production default is CPU (see `pipeline::gpu_diffmap_enabled`); the
/// parity test exists precisely to validate the GPU kernel chain, so it
/// flips the env gate. SAFETY: set once at the top of each test before
/// any pipeline call; tests in this file run single-threaded relative
/// to each other only insofar as they share the gate — both set it.
fn force_gpu_diffmap() {
    // SAFETY: setting a process env var. The cubecl runtime + the
    // pipeline read it lazily per-call; no other thread races it in
    // this test binary (both tests set the same value).
    unsafe {
        std::env::set_var("ZENSIM_GPU_DIFFMAP", "1");
    }
}

/// Build a human-readable spatial report of where a GPU diffmap diverges
/// from the CPU canonical, for CI-log diagnosis of the Metal-only #20 bug
/// without Metal hardware. Reports the divergent-pixel bounding box, edge
/// clustering, the rows/columns that contain ANY divergence (so a
/// scale-boundary pattern is obvious), and the worst 12 pixels.
fn localize_diffmap_divergence(
    gpu_dm: &[f32],
    cpu_dm: &[f32],
    w: usize,
    h: usize,
    tol: f32,
) -> String {
    use std::fmt::Write as _;
    let mut div: Vec<(usize, usize, f32, f32, f32)> = Vec::new(); // x,y,gpu,cpu,err
    let mut rows = vec![0u32; h];
    let mut cols = vec![0u32; w];
    for (i, (&g, &c)) in gpu_dm.iter().zip(cpu_dm.iter()).enumerate() {
        let e = (g - c).abs();
        if e > tol {
            let x = i % w;
            let y = i / w;
            div.push((x, y, g, c, e));
            rows[y] += 1;
            cols[x] += 1;
        }
    }
    if div.is_empty() {
        return "  (no pixels exceed tol)".to_string();
    }
    let xmin = div.iter().map(|d| d.0).min().unwrap();
    let xmax = div.iter().map(|d| d.0).max().unwrap();
    let ymin = div.iter().map(|d| d.1).min().unwrap();
    let ymax = div.iter().map(|d| d.1).max().unwrap();
    let on_last_col = div.iter().filter(|d| d.0 == w - 1).count();
    let on_last_row = div.iter().filter(|d| d.1 == h - 1).count();
    let on_first_col = div.iter().filter(|d| d.0 == 0).count();
    let on_first_row = div.iter().filter(|d| d.1 == 0).count();

    let mut s = String::new();
    let _ = writeln!(
        s,
        "  bbox x[{xmin}..={xmax}] y[{ymin}..={ymax}] of {w}x{h}; \
         edges: x0={on_first_col} x{}={on_last_col} y0={on_first_row} y{}={on_last_row}",
        w - 1,
        h - 1
    );
    // Rows / columns touched (concise if few; capped list otherwise) — a
    // scale-boundary defect concentrates divergence on a small set.
    let touched_rows: Vec<usize> = (0..h).filter(|&y| rows[y] > 0).collect();
    let touched_cols: Vec<usize> = (0..w).filter(|&x| cols[x] > 0).collect();
    let fmt_set = |v: &[usize]| -> String {
        if v.len() <= 24 {
            format!("{v:?}")
        } else {
            format!("{} distinct (first 24: {:?})", v.len(), &v[..24])
        }
    };
    let _ = writeln!(s, "  rows touched: {}", fmt_set(&touched_rows));
    let _ = writeln!(s, "  cols touched: {}", fmt_set(&touched_cols));
    // Worst 12 pixels by error.
    div.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());
    let _ = writeln!(s, "  worst {} px (x,y gpu cpu err):", div.len().min(12));
    for &(x, y, g, c, e) in div.iter().take(12) {
        let _ = writeln!(s, "    ({x},{y}) {g:.5} {c:.5} {e:.5}");
    }
    s
}

#[test]
fn gpu_diffmap_matches_cpu_canonical_pointwise() {
    force_gpu_diffmap();
    // 5 fixtures × 4 distortion levels. Sizes vary to exercise both
    // padded (width not multiple of 16) and tight cases.
    let cases: &[(u32, u32)] = &[(64, 64), (96, 80), (128, 128), (160, 120), (200, 160)];
    let deltas: &[i32] = &[3, 8, 16, 30];

    let mut max_dm_err_overall = 0.0f32;
    let mut max_score_err_overall = 0.0f32;
    // Soft-assert the per-pixel diffmap bound: collect every fixture's
    // divergence report, then assert ONCE after the loop. The tight 1e-3 gate
    // would otherwise abort at fixture 0 (64×64) and hide the larger 96×80
    // (1.098) case from a single (scarce) Metal CI run. The score + finite
    // invariants below stay hard inline (they hold on every backend).
    let mut diffmap_fail_cells = 0usize;
    let mut worst_fail: (usize, u32, u32, i32, f32) = (0, 0, 0, 0, 0.0); // fi,w,h,delta,err

    for (fi, &(w, h)) in cases.iter().enumerate() {
        let wu = w as usize;
        let hu = h as usize;
        let n = wu * hu;
        let refimg = make_fixture(fi as u32 * 7 + 1, wu, hu);
        let [rr, rg, rb] = to_linear_planes(&refimg, n);

        // GPU pipeline (one per fixture; warm the reference once).
        let mut gpu = make_zensim!(w, h);

        // CPU canonical scorer + precomputed reference (V0_3 == A).
        let cpu = ZensimCpu::new(ZensimProfile::A);
        let pre = cpu
            .precompute_reference_linear_planar([&rr, &rg, &rb], wu, hu, wu)
            .expect("precompute_reference");

        for &delta in deltas {
            let dist = distort(&refimg, delta);
            let [dr, dg, db] = to_linear_planes(&dist, n);

            // GPU path: one-shot linear-planes-with-diffmap.
            let mut gpu_dm = Vec::new();
            let gpu_score = gpu
                .score_from_linear_planes_with_diffmap(&rr, &rg, &rb, &dr, &dg, &db, &mut gpu_dm)
                .expect("gpu score_from_linear_planes_with_diffmap");

            // CPU canonical.
            let cpu_res = cpu
                .compute_with_ref_and_diffmap_linear_planar(
                    &pre,
                    [&dr, &dg, &db],
                    wu,
                    hu,
                    wu,
                    DiffmapOptions::default(),
                )
                .expect("cpu compute_with_ref_and_diffmap");
            let cpu_dm = cpu_res.diffmap();
            // CPU score is higher-is-better 0..100; normalize to the
            // butteraugli direction the GPU path returns.
            let cpu_score = (100.0 - cpu_res.score()).clamp(0.0, 100.0) as f32;

            assert_eq!(
                gpu_dm.len(),
                n,
                "fixture {fi} {w}x{h} d={delta}: gpu diffmap len {} != {n}",
                gpu_dm.len()
            );
            assert_eq!(cpu_dm.len(), n, "cpu diffmap len mismatch");

            let mut max_dm_err = 0.0f32;
            let mut argmax_i = 0usize;
            let mut n_div = 0usize;
            for (i, (&g, &c)) in gpu_dm.iter().zip(cpu_dm.iter()).enumerate() {
                assert!(
                    g.is_finite(),
                    "fixture {fi} {w}x{h} d={delta}: gpu diffmap[{i}] not finite ({g})"
                );
                let e = (g - c).abs();
                if e > DIFFMAP_ABS_TOL {
                    n_div += 1;
                }
                if e > max_dm_err {
                    max_dm_err = e;
                    argmax_i = i;
                }
            }
            max_dm_err_overall = max_dm_err_overall.max(max_dm_err);

            // On failure, localize the divergence to stderr so the Metal-only
            // bug (#20) can be characterized from CI logs without Metal
            // hardware. Measured on Metal CI 2026-06-03: fixture 0 (64×64) at
            // delta=8 diverges at ~49% of pixels but only up to 7e-3 — every
            // worst pixel is `gpu=0` vs `cpu=tiny-positive` (the `sd = max(0,·)`
            // clamp: Metal's contracted FMAs push `sd_raw` just negative → 0
            // where CPU keeps it just positive), edge-amplified (the whole last
            // row/col diverges). That small 64×64 case IS f32-clamp behaviour.
            // The large 96×80 → 1.098 case is separate and not yet localized;
            // the loop now SOFT-asserts (collect every fixture's report, assert
            // once after the loop) so one Metal run shows fixture 1's pattern
            // too instead of aborting here at fixture 0.
            if max_dm_err > DIFFMAP_ABS_TOL {
                let report = localize_diffmap_divergence(&gpu_dm, &cpu_dm, wu, hu, DIFFMAP_ABS_TOL);
                eprintln!(
                    "DIFFMAP DIVERGENCE fixture {fi} {w}x{h} d={delta}: \
                     {n_div}/{n} px exceed tol {DIFFMAP_ABS_TOL}; argmax (x={}, y={}) \
                     gpu={} cpu={}\n{report}",
                    argmax_i % wu,
                    argmax_i / wu,
                    gpu_dm[argmax_i],
                    cpu_dm[argmax_i],
                );
                diffmap_fail_cells += 1;
                if max_dm_err > worst_fail.4 {
                    worst_fail = (fi, w, h, delta, max_dm_err);
                }
            }

            let score_err = (gpu_score - cpu_score).abs();
            max_score_err_overall = max_score_err_overall.max(score_err);

            // Diffmap bound is soft-asserted after the loop (see above) so all
            // fixtures' reports land in one CI run. Score parity stays hard.
            assert!(
                score_err <= SCORE_ABS_TOL,
                "fixture {fi} {w}x{h} d={delta}: score err {score_err} exceeds {SCORE_ABS_TOL} \
                 (gpu {gpu_score} vs cpu {cpu_score})"
            );
        }
    }

    eprintln!(
        "cpu_gpu_diffmap_parity: max pointwise diffmap err = {max_dm_err_overall:.6} \
         (tol {DIFFMAP_ABS_TOL}), max score err = {max_score_err_overall:.6} (tol {SCORE_ABS_TOL})"
    );

    // Soft-asserted diffmap bound — fires once, after every fixture's
    // DIFFMAP DIVERGENCE report has been printed above (see #20).
    assert!(
        diffmap_fail_cells == 0,
        "diffmap diverged on {diffmap_fail_cells} (fixture,delta) cell(s); worst: fixture {} \
         {}x{} d={} err {} > tol {DIFFMAP_ABS_TOL}. See the per-cell DIFFMAP DIVERGENCE reports above.",
        worst_fail.0,
        worst_fail.1,
        worst_fail.2,
        worst_fail.3,
        worst_fail.4,
    );
}

/// Warm-ref path produces the same diffmap as the cold one-shot path
/// (within 1e-6 — same GPU kernels, same reference pyramid).
#[test]
fn gpu_warm_ref_diffmap_matches_one_shot() {
    force_gpu_diffmap();
    let w = 96u32;
    let h = 96u32;
    let n = (w * h) as usize;
    let refimg = make_fixture(99, w as usize, h as usize);
    let dist = distort(&refimg, 14);
    let [rr, rg, rb] = to_linear_planes(&refimg, n);
    let [dr, dg, db] = to_linear_planes(&dist, n);

    let mut z = make_zensim!(w, h);

    // Cold one-shot.
    let mut cold_dm = Vec::new();
    let cold_score = z
        .score_from_linear_planes_with_diffmap(&rr, &rg, &rb, &dr, &dg, &db, &mut cold_dm)
        .expect("cold");

    // Warm-ref.
    z.warm_reference_from_linear_planes(&rr, &rg, &rb)
        .expect("warm_reference");
    let mut warm_dm = Vec::new();
    let warm_score = z
        .score_from_linear_planes_with_warm_ref_diffmap(&dr, &dg, &db, &mut warm_dm)
        .expect("warm");

    assert_eq!(cold_dm.len(), warm_dm.len());
    let max_err = cold_dm
        .iter()
        .zip(warm_dm.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 1e-6,
        "warm vs cold GPU diffmap diverge: max_err = {max_err}"
    );
    assert!(
        (cold_score - warm_score).abs() < 1e-4,
        "warm vs cold GPU score diverge: cold={cold_score}, warm={warm_score}"
    );
}
