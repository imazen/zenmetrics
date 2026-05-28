//! Phase 9x — single-call CPU metric profiler driver.
//!
//! Invocation:
//!   cpu-profile <metric> <mode> <width> <height>
//!
//! `<metric>` ∈ { cvvdp ssim2 dssim butter iwssim zensim }
//! `<mode>`   ∈ { full warm_ref strip warm_ref_strip }
//! `<width>`/`<height>` in pixels.
//!
//! Drives the CPU metric API exactly once per process so heaptrack /
//! callgrind output is attributable to a single (metric, mode, size)
//! cell. The reference + distorted buffers are built from
//! `synth_pair_offset_dist` (mirrored from zenmetrics-orchestrator) so
//! every cell sees deterministic input shape.
//!
//! Modes:
//! - `full`            — `new()` + `compute(ref, dist)`
//! - `warm_ref`        — `new()` + `warm_reference(ref)` + `score_with_warm_ref(dist)`
//! - `strip`           — zensim-only: one-off strip score with internal
//!                        full-ref reuse: `precompute_reference(ref)` +
//!                        `compute_with_ref_streaming_strips(precomp, dist, 256, 128)`.
//!                        Note (2026-05-27 Phase 9.Y, finding #5 fix):
//!                        the previous `strip` wiring called
//!                        `compute_streaming_strips_default` which
//!                        rebuilds a `PrecomputedReference` PER strip
//!                        (the heaptrack report measured +36% peak heap
//!                        @ 40 MP vs. `full`). Hoisting the
//!                        `PrecomputedReference` construction out of
//!                        the strip loop drops the strip peak from
//!                        +36% to ≈+13%, matching `warm_ref_strip`.
//!                        See [`run_zensim`] for the helper and
//!                        `docs/ZENSIM_STRIP_WARM_REF_HOIST.md` for the
//!                        production-caller pattern.
//! - `warm_ref_strip`  — zensim-only: same call path as `strip` since
//!                        the hoisted-ref pattern subsumes the prior
//!                        difference. Retained for matrix-symmetry —
//!                        any heap delta between `strip` and
//!                        `warm_ref_strip` now reflects only call-site
//!                        differences (here: identical), not the
//!                        per-strip re-precompute overhead.
//!
//! For metrics that do not implement a mode (everything except zensim
//! for the two strip modes), the driver prints `GAP:<metric>:<mode>`
//! to stdout and exits 2 so the calling harness records the gap
//! without running heaptrack on a degenerate call.

use std::env;
use std::process::ExitCode;
use std::time::Instant;

fn synth_pair(width: u32, height: u32) -> (Vec<u8>, Vec<u8>) {
    let w = width as usize;
    let h = height as usize;
    let n = w * h * 3;
    let mut r = vec![0u8; n];
    for y in 0..h {
        for x in 0..w {
            let rr = (((x * 17 + y * 5) % 251) as u8).wrapping_add(40);
            let gg = (((x * 11 + y * 13) % 247) as u8).wrapping_add(40);
            let bb = (((x * 7 + y * 19) % 241) as u8).wrapping_add(40);
            let i = (y * w + x) * 3;
            r[i] = rr;
            r[i + 1] = gg;
            r[i + 2] = bb;
        }
    }
    let d: Vec<u8> = r
        .chunks_exact(3)
        .flat_map(|p| {
            [
                p[0].saturating_sub(8),
                p[1].saturating_sub(4),
                p[2].saturating_add(12),
            ]
        })
        .collect();
    (r, d)
}

/// Borrow the interleaved sRGB-u8 byte buffer as `&[[u8; 3]]`.
///
/// Phase 9.Y (2026-05-27): replaces the prior `chunks_exact(3).collect()`
/// path that allocated a 120 MB Vec per side at 40 MP. `[u8; 3]` is
/// `bytemuck::Pod`, so the reinterpret is zero-copy and safe. Mirrors
/// the change in `crates/zenmetrics-orchestrator/src/cpu_adapter.rs` so
/// the heaptrack driver's accounting reflects the production adapter
/// allocation pattern, not the obsolete `chunks_exact` overhead.
fn rgb_pix(bytes: &[u8]) -> &[[u8; 3]] {
    bytemuck::cast_slice(bytes)
}

// ---------------------------------------------------------------------------
// Per-metric runners
// ---------------------------------------------------------------------------

fn run_cvvdp(mode: &str, w: u32, h: u32, r: &[u8], d: &[u8]) -> Result<f64, String> {
    use cvvdp::{Cvvdp, CvvdpParams};
    // Strip body height — picks the canonical GPU default (512) for
    // strip-mode runs. Aligns with `cvvdp::strip::STRIP_H_BODY_DEFAULT`.
    const STRIP_H_BODY: u32 = 512;
    match mode {
        "full" => {
            let mut c = Cvvdp::new(w, h, CvvdpParams::default()).map_err(|e| e.to_string())?;
            let v = c.score(r, d).map_err(|e| e.to_string())?;
            Ok(v as f64)
        }
        "warm_ref" => {
            let mut c = Cvvdp::new(w, h, CvvdpParams::default()).map_err(|e| e.to_string())?;
            c.warm_reference(r).map_err(|e| e.to_string())?;
            let v = c.score_with_warm_ref(d).map_err(|e| e.to_string())?;
            Ok(v as f64)
        }
        "strip" => {
            // Phase 9.Z.B: real strip walker (was GAP through 2026-05-27).
            // Memory impact today: ZERO vs `full` — only the pool stage
            // iterates in strips; weber pyramid + masking + d_scratch
            // remain full-image-sized. Matches the GPU's currently-
            // shipped strip walker. Heaptrack will report ~same peak as
            // `full`; the parity gate (bit-identical JOD) is the real
            // contract this run validates.
            let mut c = Cvvdp::new(w, h, CvvdpParams::default()).map_err(|e| e.to_string())?;
            let v = c.score_strip(r, d, STRIP_H_BODY).map_err(|e| e.to_string())?;
            Ok(v as f64)
        }
        "warm_ref_strip" => {
            // Phase 9.Z.B: Mode E (cached-ref) strip variant. Same
            // ZERO-memory-impact caveat as `strip`.
            let mut c = Cvvdp::new(w, h, CvvdpParams::default()).map_err(|e| e.to_string())?;
            c.warm_reference(r).map_err(|e| e.to_string())?;
            let v = c
                .score_with_warm_ref_strip(d, STRIP_H_BODY)
                .map_err(|e| e.to_string())?;
            Ok(v as f64)
        }
        _ => Err(format!("bad-mode:{mode}")),
    }
}

fn run_ssim2(mode: &str, w: u32, h: u32, r: &[u8], d: &[u8]) -> Result<f64, String> {
    use fast_ssim2::Ssimulacra2Reference;
    use imgref::ImgRef;
    let wu = w as usize;
    let hu = h as usize;
    match mode {
        "full" => {
            let ri = ImgRef::new(rgb_pix(r), wu, hu);
            let di = ImgRef::new(rgb_pix(d), wu, hu);
            let v = fast_ssim2::compute_ssimulacra2(ri, di)
                .map_err(|e| format!("{e:?}"))?;
            Ok(v)
        }
        "warm_ref" => {
            let ri = ImgRef::new(rgb_pix(r), wu, hu);
            let di = ImgRef::new(rgb_pix(d), wu, hu);
            let pre = Ssimulacra2Reference::new(ri).map_err(|e| format!("{e:?}"))?;
            let v = pre.compare(di).map_err(|e| format!("{e:?}"))?;
            Ok(v)
        }
        "strip" | "warm_ref_strip" => Err(format!("GAP:ssim2:{mode}")),
        _ => Err(format!("bad-mode:{mode}")),
    }
}

fn run_dssim(mode: &str, w: u32, h: u32, r: &[u8], d: &[u8]) -> Result<f64, String> {
    use dssim_core::Dssim;
    let wu = w as usize;
    let hu = h as usize;
    let dssim = Dssim::new();
    let to_img = |bytes: &[u8]| -> dssim_core::DssimImage<f32> {
        // Phase 9.Y: zero-copy reinterpret via bytemuck — mirrors the
        // adapter's `make_dssim_image`. Uses `create_image_rgb`, the
        // dssim-core 3.4 shortcut that runs `to_rgblu()` internally.
        let rgb: &[rgb::RGB<u8>] = bytemuck::cast_slice(bytes);
        dssim
            .create_image_rgb(rgb, wu, hu)
            .expect("dssim create_image_rgb")
    };
    match mode {
        "full" => {
            let ri = to_img(r);
            let di = to_img(d);
            let (score, _maps) = dssim.compare(&ri, di);
            Ok(f64::from(score))
        }
        "warm_ref" => {
            let ri = to_img(r);
            let di = to_img(d);
            let (score, _maps) = dssim.compare(&ri, di);
            Ok(f64::from(score))
        }
        "strip" | "warm_ref_strip" => Err(format!("GAP:dssim:{mode}")),
        _ => Err(format!("bad-mode:{mode}")),
    }
}

fn run_butter(mode: &str, w: u32, h: u32, r: &[u8], d: &[u8]) -> Result<f64, String> {
    use butteraugli::{ButteraugliParams, ButteraugliReference};
    use imgref::ImgRef;
    let wu = w as usize;
    let hu = h as usize;
    let p = ButteraugliParams::new();
    match mode {
        "full" => {
            // Phase 9.Y: zero-copy reinterpret via bytemuck — mirrors the
            // adapter's `compute_butter`.
            let rb: &[rgb::RGB<u8>] = bytemuck::cast_slice(r);
            let db: &[rgb::RGB<u8>] = bytemuck::cast_slice(d);
            let ri = ImgRef::new(rb, wu, hu);
            let di = ImgRef::new(db, wu, hu);
            let result = butteraugli::butteraugli(ri, di, &p).map_err(|e| format!("{e:?}"))?;
            Ok(result.score)
        }
        "warm_ref" => {
            // Phase 9.Y: butteraugli 0.9.2 has a proper warm-ref API
            // (`ButteraugliReference::new(&[u8], ...)` + `.compare(&[u8])`).
            // The original heaptrack driver compared `full` to itself
            // here — fixed now so warm_ref measures the cached path the
            // adapter actually uses.
            let pre = ButteraugliReference::new(r, wu, hu, p.clone())
                .map_err(|e| format!("ButteraugliReference::new: {e:?}"))?;
            let result = pre.compare(d).map_err(|e| format!("compare: {e:?}"))?;
            Ok(result.score)
        }
        "strip" | "warm_ref_strip" => Err(format!("GAP:butter:{mode}")),
        _ => Err(format!("bad-mode:{mode}")),
    }
}

fn run_iwssim(mode: &str, w: u32, h: u32, r: &[u8], d: &[u8]) -> Result<f64, String> {
    use iwssim::{Iwssim, STRIP_BODY_DEFAULT};
    match mode {
        "full" => {
            let mut c = Iwssim::new(w, h).map_err(|e| e.to_string())?;
            let v = c.score(r, d).map_err(|e| e.to_string())?;
            Ok(v.score)
        }
        "warm_ref" => {
            let mut c = Iwssim::new(w, h).map_err(|e| e.to_string())?;
            c.warm_reference(r).map_err(|e| e.to_string())?;
            let v = c.score_with_warm_ref(d).map_err(|e| e.to_string())?;
            Ok(v.score)
        }
        "strip" => {
            // Phase 9.Z.A: real strip walker. Uses STRIP_BODY_DEFAULT
            // = 512 rows + STRIP_HALO_ROWS = 320 per side.
            let mut c = Iwssim::new(w, h).map_err(|e| e.to_string())?;
            let v = c
                .score_strip(r, d, STRIP_BODY_DEFAULT)
                .map_err(|e| e.to_string())?;
            Ok(v.score)
        }
        "warm_ref_strip" => {
            // Phase 9.Z.A: warm_ref + strip dist. Cached ref pyramid
            // + eigendecomp in WarmState; per-strip dist working set.
            let mut c = Iwssim::new(w, h).map_err(|e| e.to_string())?;
            c.warm_reference(r).map_err(|e| e.to_string())?;
            let v = c
                .score_with_warm_ref_strip(d, STRIP_BODY_DEFAULT)
                .map_err(|e| e.to_string())?;
            Ok(v.score)
        }
        _ => Err(format!("bad-mode:{mode}")),
    }
}

fn run_zensim(mode: &str, w: u32, h: u32, r: &[u8], d: &[u8]) -> Result<f64, String> {
    use zensim::{RgbSlice, Zensim, ZensimProfile};
    let z = Zensim::new(ZensimProfile::latest_preview());
    // Phase 9.Y: zero-copy reinterpret via bytemuck — mirrors the adapter.
    let ri = RgbSlice::new(rgb_pix(r), w as usize, h as usize);
    let di = RgbSlice::new(rgb_pix(d), w as usize, h as usize);
    match mode {
        "full" | "warm_ref" => {
            // zensim has no public per-instance warm-reference; warm_ref
            // falls back to full compute. (Note: precompute_reference
            // returns a value-type ref; treated as separate "warm_ref" by
            // computing once then scoring once. We mirror the
            // cpu_adapter's no-cache behavior for `warm_ref` to keep
            // numbers comparable.)
            let v = z.compute(&ri, &di).map_err(|e| format!("{e:?}"))?;
            Ok(v.score())
        }
        // Phase 9.Y finding #5 fix (2026-05-27): the original `strip`
        // wiring called `compute_streaming_strips_default(&ri, &di)`
        // which rebuilds a fresh `PrecomputedReference` per strip — at
        // 40 MP this measured +36% peak heap vs. full mode (3.53 GB
        // vs. 2.64 GB). The waste is per-strip ref XYB conversion +
        // pyramid downscale that's recomputed for every strip even
        // though the source image is the same.
        //
        // The fix: build a single `PrecomputedReference` over the full
        // image up front, then call `compute_with_ref_streaming_strips_default`
        // which slices that ref per strip (zero-copy) — bit-identical
        // score, ~+13% peak heap (matching the prior `warm_ref_strip`
        // baseline). `warm_ref_strip` keeps the same call path so the
        // matrix still reports both cells; future heaptracks will show
        // them at parity.
        "strip" | "warm_ref_strip" => {
            let pre = z.precompute_reference(&ri).map_err(|e| format!("{e:?}"))?;
            let v = z
                .compute_with_ref_streaming_strips_default(&pre, &di)
                .map_err(|e| format!("{e:?}"))?;
            Ok(v.score())
        }
        _ => Err(format!("bad-mode:{mode}")),
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        eprintln!(
            "usage: cpu-profile <metric> <mode> <width> <height>\n  metrics: cvvdp ssim2 dssim butter iwssim zensim\n  modes:   full warm_ref strip warm_ref_strip"
        );
        return ExitCode::from(64);
    }
    let metric = args[1].as_str();
    let mode = args[2].as_str();
    let w: u32 = args[3].parse().expect("width");
    let h: u32 = args[4].parse().expect("height");

    let t0 = Instant::now();
    let (r, d) = synth_pair(w, h);
    let t_synth = t0.elapsed();

    let t1 = Instant::now();
    let result = match metric {
        "cvvdp" => run_cvvdp(mode, w, h, &r, &d),
        "ssim2" => run_ssim2(mode, w, h, &r, &d),
        "dssim" => run_dssim(mode, w, h, &r, &d),
        "butter" => run_butter(mode, w, h, &r, &d),
        "iwssim" => run_iwssim(mode, w, h, &r, &d),
        "zensim" => run_zensim(mode, w, h, &r, &d),
        _ => {
            eprintln!("unknown metric: {metric}");
            return ExitCode::from(64);
        }
    };
    let t_score = t1.elapsed();

    match result {
        Ok(score) => {
            println!(
                "OK metric={metric} mode={mode} w={w} h={h} score={score} t_synth_ms={:.2} t_score_ms={:.2}",
                t_synth.as_secs_f64() * 1000.0,
                t_score.as_secs_f64() * 1000.0
            );
            ExitCode::SUCCESS
        }
        Err(e) if e.starts_with("GAP:") => {
            println!("{e}");
            ExitCode::from(2)
        }
        Err(e) => {
            eprintln!("FAIL metric={metric} mode={mode} w={w} h={h} err={e}");
            ExitCode::from(1)
        }
    }
}
