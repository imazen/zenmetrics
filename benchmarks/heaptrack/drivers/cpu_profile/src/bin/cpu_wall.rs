//! Task #139 — zenbench WALL-time harness for the 6 CPU metrics.
//!
//! Times each (metric, mode, size) per-call using zenbench's interleaved
//! round-robin execution (kills thermal/turbo bias — the reason we use
//! zenbench over criterion or the heaptrack-instrumented runtime). All
//! metric/mode cells for a single size run in one zenbench group so they
//! interleave under identical machine conditions.
//!
//! cold vs warm semantics:
//! - `full` / `strip`            : inherently cold (each call builds the
//!   reference). One bench → `per_call_ms`.
//! - `warm_ref` / `warm_ref_strip`: cached-ref. TWO benches —
//!     `<mode>__cold`: builds metric + warms reference + one warm score
//!                     per iteration → mean = cold_first_call (incl the
//!                     one-time precompute).
//!     `<mode>__warm`: metric warmed ONCE outside the loop, each iter is
//!                     one `score_with_warm_ref` → mean = warm_amortized.
//!   The cold/warm delta is the precompute one-time cost.
//!
//! dssim strip / warm_ref_strip are NOT_SUPPORTED (dssim-core 3.4 has no
//! strip walker) — they are not registered, mirroring the heaptrack
//! driver's GAP.
//!
//! Usage:
//!   cpu-wall <size_label> <out_tsv>
//!   size_label ∈ { 512 1024 2K 4096 12MP 30MP }
//!
//! Writes/appends rows to <out_tsv>:
//!   size_label  metric  mode  cold_or_warm  w  h  mean_ns  mean_ms  n_rounds  score
//!
//! Built release, NO `-C target-cpu=native` (runtime SIMD dispatch is
//! what users get — per CLAUDE.md).

use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use std::time::Duration;

use zenbench::prelude::*;

// ---------------------------------------------------------------------------
// Synthetic inputs — identical pattern to the heaptrack driver so wall and
// memory measurements use the same input shape.
// ---------------------------------------------------------------------------
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

fn rgb_pix(bytes: &[u8]) -> &[[u8; 3]] {
    bytemuck::cast_slice(bytes)
}

const STRIP_H: u32 = 512;

fn size_dims(label: &str) -> Option<(u32, u32)> {
    match label {
        "512" => Some((512, 512)),
        "1024" => Some((1024, 1024)),
        "2K" => Some((2048, 2048)),
        // Task #141 (2026-05-29): 16 MP (4096²) added so the CPU wall join
        // lands on the SAME sizes as the GPU cold TSV
        // (benchmarks/gpu_coldstart_2026-05-29.tsv: 512²/1024²/2048²/4096²).
        // Exact join → exact one-shot CPU-vs-GPU crossover, no extrapolation.
        "4096" => Some((4096, 4096)),
        "12MP" => Some((4000, 3000)),
        "30MP" => Some((6000, 5000)),
        _ => None,
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 || args.len() > 4 {
        eprintln!(
            "usage: cpu-wall <size_label> <out_tsv> [metric_filter]\n  size_label: 512 1024 2K 4096 12MP 30MP\n  metric_filter (optional): cvvdp ssim2 dssim butter iwssim zensim\n    — when set, only that metric's cells register (bounds peak harness RAM\n      at large sizes where holding all 6 warmed refs would be heavy).\n      Cells still interleave WITHIN the metric's modes (the comparison that\n      matters: full vs strip vs warm)."
        );
        std::process::exit(64);
    }
    let label = args[1].clone();
    let out_tsv = args[2].clone();
    let metric_filter: Option<String> = args.get(3).cloned();
    let want = |m: &str| metric_filter.as_deref().map(|f| f == m).unwrap_or(true);
    let (w, h) = match size_dims(&label) {
        Some(d) => d,
        None => {
            eprintln!("bad size label: {label}");
            std::process::exit(64);
        }
    };

    let (r, d) = synth_pair(w, h);
    let wu = w as usize;
    let hu = h as usize;

    // Score sentinels (printed for provenance — confirms benches ran real
    // calls, not stubs). Filled per cell as a side-channel.
    let mut scores: Vec<(String, f64)> = Vec::new();

    // One zenbench group per size — all 32 metric/mode cells interleave
    // under identical machine conditions (the whole point of zenbench).
    //
    // The binding budget for a group is `max_wall_time` (shared across all
    // benches in the group). With 32 cells we must raise it well above the
    // 120s default so each cell gathers enough rounds for stable means.
    // `auto_rounds` converges each cell to `target_precision` (2%); we set
    // `min_rounds` so even the slowest cells get a robust sample, and a
    // generous per-bench `max_time` so fast cells aren't starved.
    //
    // At 30MP a single cvvdp full call is ~seconds — there we accept fewer
    // rounds (min_rounds floor) within a large wall budget rather than
    // pinning a fixed high round count (which would run for an hour).
    let (group_wall, per_cell_max_time, min_rounds) = match label.as_str() {
        "512" => (Duration::from_secs(600), Duration::from_secs(10), 16usize),
        "1024" => (Duration::from_secs(700), Duration::from_secs(12), 16),
        "2K" => (Duration::from_secs(900), Duration::from_secs(16), 14),
        // 16 MP (4096²): a single cvvdp full call is ~4.6 s on the 7950X
        // (cpu_path_a_recovered_2026-05-29.tsv), so per-cell max_time must
        // exceed it; min_rounds floored at 10 keeps the slowest cells
        // sampled while a 1800 s group wall bounds total runtime.
        "4096" => (Duration::from_secs(1800), Duration::from_secs(40), 10),
        "12MP" => (Duration::from_secs(1500), Duration::from_secs(30), 12),
        "30MP" => (Duration::from_secs(2400), Duration::from_secs(60), 10),
        _ => (Duration::from_secs(600), Duration::from_secs(10), 12),
    };

    // Task #141 (2026-05-29): the default zenbench resource gate does a full
    // `sysinfo` process enumeration (`System::new(); refresh_processes(All)`)
    // every round, per cell, to detect concurrent benchmark harnesses and
    // system noise. On the 7950X workstation with ~1000 live processes (many
    // idle `claude` sessions) that scan dominates each round's wall time —
    // the metric thread sits in `hrtimer_nanosleep` while the scan runs, so
    // measurement crawls (observed: 2 s CPU in 23 s wall for dssim@512).
    // Setting CPU_WALL_NO_GATE=1 swaps in `GateConfig::disabled()` (skips the
    // per-round scan) — the interleaved round-robin + paired stats that make
    // zenbench better than criterion are unaffected; only the auto-pause is
    // dropped. The CALLER guarantees a quiet machine (CLAUDE.md sweep
    // discipline), which is the contract the gate would otherwise enforce.
    let build = |suite: &mut zenbench::prelude::Suite| {
        suite.group(format!("cpu_wall_{label}"), |g| {
            g.config().max_wall_time(group_wall);
            g.config().max_time(per_cell_max_time);
            g.config().min_rounds(min_rounds);

            // ---- cvvdp ----
            if want("cvvdp") {
                use cvvdp::{Cvvdp, CvvdpParams};
                let (r, d) = (r.clone(), d.clone());
                g.bench("cvvdp__full", {
                    let (r, d) = (r.clone(), d.clone());
                    move |b| {
                        b.iter(|| {
                            let mut c = Cvvdp::new(w, h, CvvdpParams::default()).unwrap();
                            zenbench::black_box(c.score(&r, &d).unwrap())
                        })
                    }
                });
                g.bench("cvvdp__strip", {
                    let (r, d) = (r.clone(), d.clone());
                    move |b| {
                        b.iter(|| {
                            let mut c = Cvvdp::new(w, h, CvvdpParams::default()).unwrap();
                            zenbench::black_box(c.score_strip(&r, &d, STRIP_H).unwrap())
                        })
                    }
                });
                // warm_ref cold: new + warm + 1 score per iter
                g.bench("cvvdp__warm_ref__cold", {
                    let (r, d) = (r.clone(), d.clone());
                    move |b| {
                        b.iter(|| {
                            let mut c = Cvvdp::new(w, h, CvvdpParams::default()).unwrap();
                            c.warm_reference(&r).unwrap();
                            zenbench::black_box(c.score_with_warm_ref(&d).unwrap())
                        })
                    }
                });
                // warm_ref warm: warm once outside, time only the score
                g.bench("cvvdp__warm_ref__warm", {
                    let d = d.clone();
                    let mut c = Cvvdp::new(w, h, CvvdpParams::default()).unwrap();
                    c.warm_reference(&r).unwrap();
                    move |b| b.iter(|| zenbench::black_box(c.score_with_warm_ref(&d).unwrap()))
                });
                // warm_ref_strip cold
                g.bench("cvvdp__warm_ref_strip__cold", {
                    let (r, d) = (r.clone(), d.clone());
                    move |b| {
                        b.iter(|| {
                            let mut c = Cvvdp::new(w, h, CvvdpParams::default()).unwrap();
                            c.warm_reference(&r).unwrap();
                            zenbench::black_box(c.score_with_warm_ref_strip(&d, STRIP_H).unwrap())
                        })
                    }
                });
                // warm_ref_strip warm
                g.bench("cvvdp__warm_ref_strip__warm", {
                    let d = d.clone();
                    let mut c = Cvvdp::new(w, h, CvvdpParams::default()).unwrap();
                    c.warm_reference(&r).unwrap();
                    move |b| {
                        b.iter(|| {
                            zenbench::black_box(c.score_with_warm_ref_strip(&d, STRIP_H).unwrap())
                        })
                    }
                });
            }

            // ---- ssim2 ----
            if want("ssim2") {
                use fast_ssim2::Ssimulacra2Reference;
                use imgref::ImgRef;
                g.bench("ssim2__full", {
                    let (r, d) = (r.clone(), d.clone());
                    move |b| {
                        b.iter(|| {
                            let ri = ImgRef::new(rgb_pix(&r), wu, hu);
                            let di = ImgRef::new(rgb_pix(&d), wu, hu);
                            zenbench::black_box(fast_ssim2::compute_ssimulacra2(ri, di).unwrap())
                        })
                    }
                });
                g.bench("ssim2__strip", {
                    let (r, d) = (r.clone(), d.clone());
                    move |b| {
                        b.iter(|| {
                            let ri = ImgRef::new(rgb_pix(&r), wu, hu);
                            let di = ImgRef::new(rgb_pix(&d), wu, hu);
                            zenbench::black_box(
                                fast_ssim2::compute_ssimulacra2_strip(ri, di, STRIP_H).unwrap(),
                            )
                        })
                    }
                });
                g.bench("ssim2__warm_ref__cold", {
                    let (r, d) = (r.clone(), d.clone());
                    move |b| {
                        b.iter(|| {
                            let ri = ImgRef::new(rgb_pix(&r), wu, hu);
                            let pre = Ssimulacra2Reference::new(ri).unwrap();
                            let di = ImgRef::new(rgb_pix(&d), wu, hu);
                            zenbench::black_box(pre.compare(di).unwrap())
                        })
                    }
                });
                g.bench("ssim2__warm_ref__warm", {
                    let (r, d) = (r.clone(), d.clone());
                    let ri = ImgRef::new(rgb_pix(&r), wu, hu);
                    let pre = Ssimulacra2Reference::new(ri).unwrap();
                    move |b| {
                        b.iter(|| {
                            let di = ImgRef::new(rgb_pix(&d), wu, hu);
                            zenbench::black_box(pre.compare(di).unwrap())
                        })
                    }
                });
                g.bench("ssim2__warm_ref_strip__cold", {
                    let (r, d) = (r.clone(), d.clone());
                    move |b| {
                        b.iter(|| {
                            let ri = ImgRef::new(rgb_pix(&r), wu, hu);
                            let pre = Ssimulacra2Reference::new(ri).unwrap();
                            let di = ImgRef::new(rgb_pix(&d), wu, hu);
                            zenbench::black_box(pre.compare_strip(di, STRIP_H).unwrap())
                        })
                    }
                });
                g.bench("ssim2__warm_ref_strip__warm", {
                    let (r, d) = (r.clone(), d.clone());
                    let ri = ImgRef::new(rgb_pix(&r), wu, hu);
                    let pre = Ssimulacra2Reference::new(ri).unwrap();
                    move |b| {
                        b.iter(|| {
                            let di = ImgRef::new(rgb_pix(&d), wu, hu);
                            zenbench::black_box(pre.compare_strip(di, STRIP_H).unwrap())
                        })
                    }
                });
            }

            // ---- dssim (full + warm_ref only) ----
            if want("dssim") {
                use dssim_core::Dssim;
                g.bench("dssim__full", {
                    let (r, d) = (r.clone(), d.clone());
                    move |b| {
                        let dssim = Dssim::new();
                        b.iter(|| {
                            let rr: &[rgb::RGB<u8>] = bytemuck::cast_slice(&r);
                            let dd: &[rgb::RGB<u8>] = bytemuck::cast_slice(&d);
                            let ri = dssim.create_image_rgb(rr, wu, hu).unwrap();
                            let di = dssim.create_image_rgb(dd, wu, hu).unwrap();
                            let (s, _m) = dssim.compare(&ri, di);
                            zenbench::black_box(f64::from(s))
                        })
                    }
                });
                // warm_ref cold: build ref image + dist image + compare per iter
                g.bench("dssim__warm_ref__cold", {
                    let (r, d) = (r.clone(), d.clone());
                    move |b| {
                        let dssim = Dssim::new();
                        b.iter(|| {
                            let rr: &[rgb::RGB<u8>] = bytemuck::cast_slice(&r);
                            let dd: &[rgb::RGB<u8>] = bytemuck::cast_slice(&d);
                            let ri = dssim.create_image_rgb(rr, wu, hu).unwrap();
                            let di = dssim.create_image_rgb(dd, wu, hu).unwrap();
                            let (s, _m) = dssim.compare(&ri, di);
                            zenbench::black_box(f64::from(s))
                        })
                    }
                });
                // warm_ref warm: reference DssimImage built ONCE, reused
                g.bench("dssim__warm_ref__warm", {
                    let (r, d) = (r.clone(), d.clone());
                    let dssim = Dssim::new();
                    let rr: &[rgb::RGB<u8>] = bytemuck::cast_slice(&r);
                    let ri = dssim.create_image_rgb(rr, wu, hu).unwrap();
                    move |b| {
                        b.iter(|| {
                            let dd: &[rgb::RGB<u8>] = bytemuck::cast_slice(&d);
                            let di = dssim.create_image_rgb(dd, wu, hu).unwrap();
                            let (s, _m) = dssim.compare(&ri, di);
                            zenbench::black_box(f64::from(s))
                        })
                    }
                });
            }

            // ---- butter ----
            if want("butter") {
                use butteraugli::{ButteraugliParams, ButteraugliReference};
                use imgref::ImgRef;
                let p = ButteraugliParams::new();
                g.bench("butter__full", {
                    let (r, d, p) = (r.clone(), d.clone(), p.clone());
                    move |b| {
                        b.iter(|| {
                            let rb: &[rgb::RGB<u8>] = bytemuck::cast_slice(&r);
                            let db: &[rgb::RGB<u8>] = bytemuck::cast_slice(&d);
                            let ri = ImgRef::new(rb, wu, hu);
                            let di = ImgRef::new(db, wu, hu);
                            zenbench::black_box(butteraugli::butteraugli(ri, di, &p).unwrap().score)
                        })
                    }
                });
                g.bench("butter__strip", {
                    let (r, d, p) = (r.clone(), d.clone(), p.clone());
                    move |b| {
                        b.iter(|| {
                            let rb: &[rgb::RGB<u8>] = bytemuck::cast_slice(&r);
                            let db: &[rgb::RGB<u8>] = bytemuck::cast_slice(&d);
                            let ri = ImgRef::new(rb, wu, hu);
                            let di = ImgRef::new(db, wu, hu);
                            zenbench::black_box(
                                butteraugli::butteraugli_strip(ri, di, &p, STRIP_H)
                                    .unwrap()
                                    .score,
                            )
                        })
                    }
                });
                g.bench("butter__warm_ref__cold", {
                    let (r, d, p) = (r.clone(), d.clone(), p.clone());
                    move |b| {
                        b.iter(|| {
                            let pre = ButteraugliReference::new(&r, wu, hu, p.clone()).unwrap();
                            zenbench::black_box(pre.compare(&d).unwrap().score)
                        })
                    }
                });
                g.bench("butter__warm_ref__warm", {
                    let (r, d, p) = (r.clone(), d.clone(), p.clone());
                    let pre = ButteraugliReference::new(&r, wu, hu, p).unwrap();
                    move |b| b.iter(|| zenbench::black_box(pre.compare(&d).unwrap().score))
                });
                g.bench("butter__warm_ref_strip__cold", {
                    let (r, d, p) = (r.clone(), d.clone(), p.clone());
                    move |b| {
                        b.iter(|| {
                            let pre = ButteraugliReference::new(&r, wu, hu, p.clone()).unwrap();
                            let db: &[rgb::RGB<u8>] = bytemuck::cast_slice(&d);
                            let di = ImgRef::new(db, wu, hu);
                            zenbench::black_box(pre.compare_strip_srgb(di, STRIP_H).unwrap().score)
                        })
                    }
                });
                g.bench("butter__warm_ref_strip__warm", {
                    let (r, d, p) = (r.clone(), d.clone(), p.clone());
                    let pre = ButteraugliReference::new(&r, wu, hu, p).unwrap();
                    move |b| {
                        b.iter(|| {
                            let db: &[rgb::RGB<u8>] = bytemuck::cast_slice(&d);
                            let di = ImgRef::new(db, wu, hu);
                            zenbench::black_box(pre.compare_strip_srgb(di, STRIP_H).unwrap().score)
                        })
                    }
                });
            }

            // ---- iwssim ----
            if want("iwssim") {
                use iwssim::{Iwssim, STRIP_BODY_DEFAULT};
                g.bench("iwssim__full", {
                    let (r, d) = (r.clone(), d.clone());
                    move |b| {
                        b.iter(|| {
                            let mut c = Iwssim::new(w, h).unwrap();
                            zenbench::black_box(c.score(&r, &d).unwrap().score)
                        })
                    }
                });
                g.bench("iwssim__strip", {
                    let (r, d) = (r.clone(), d.clone());
                    move |b| {
                        b.iter(|| {
                            let mut c = Iwssim::new(w, h).unwrap();
                            zenbench::black_box(
                                c.score_strip(&r, &d, STRIP_BODY_DEFAULT).unwrap().score,
                            )
                        })
                    }
                });
                g.bench("iwssim__warm_ref__cold", {
                    let (r, d) = (r.clone(), d.clone());
                    move |b| {
                        b.iter(|| {
                            let mut c = Iwssim::new(w, h).unwrap();
                            c.warm_reference(&r).unwrap();
                            zenbench::black_box(c.score_with_warm_ref(&d).unwrap().score)
                        })
                    }
                });
                g.bench("iwssim__warm_ref__warm", {
                    let d = d.clone();
                    let mut c = Iwssim::new(w, h).unwrap();
                    c.warm_reference(&r).unwrap();
                    move |b| {
                        b.iter(|| zenbench::black_box(c.score_with_warm_ref(&d).unwrap().score))
                    }
                });
                g.bench("iwssim__warm_ref_strip__cold", {
                    let (r, d) = (r.clone(), d.clone());
                    move |b| {
                        b.iter(|| {
                            let mut c = Iwssim::new(w, h).unwrap();
                            c.warm_reference(&r).unwrap();
                            zenbench::black_box(
                                c.score_with_warm_ref_strip(&d, STRIP_BODY_DEFAULT)
                                    .unwrap()
                                    .score,
                            )
                        })
                    }
                });
                g.bench("iwssim__warm_ref_strip__warm", {
                    let d = d.clone();
                    let mut c = Iwssim::new(w, h).unwrap();
                    c.warm_reference(&r).unwrap();
                    move |b| {
                        b.iter(|| {
                            zenbench::black_box(
                                c.score_with_warm_ref_strip(&d, STRIP_BODY_DEFAULT)
                                    .unwrap()
                                    .score,
                            )
                        })
                    }
                });
            }

            // ---- zensim ----
            if want("zensim") {
                use zensim::{RgbSlice, Zensim, ZensimProfile};
                g.bench("zensim__full", {
                    let (r, d) = (r.clone(), d.clone());
                    move |b| {
                        let z = Zensim::new(ZensimProfile::latest_preview());
                        b.iter(|| {
                            let ri = RgbSlice::new(rgb_pix(&r), wu, hu);
                            let di = RgbSlice::new(rgb_pix(&d), wu, hu);
                            zenbench::black_box(z.compute(&ri, &di).unwrap().score())
                        })
                    }
                });
                g.bench("zensim__strip", {
                    let (r, d) = (r.clone(), d.clone());
                    move |b| {
                        let z = Zensim::new(ZensimProfile::latest_preview());
                        b.iter(|| {
                            let ri = RgbSlice::new(rgb_pix(&r), wu, hu);
                            let di = RgbSlice::new(rgb_pix(&d), wu, hu);
                            let pre = z.precompute_reference(&ri).unwrap();
                            zenbench::black_box(
                                z.compute_with_ref_streaming_strips_default(&pre, &di)
                                    .unwrap()
                                    .score(),
                            )
                        })
                    }
                });
                g.bench("zensim__warm_ref__cold", {
                    let (r, d) = (r.clone(), d.clone());
                    move |b| {
                        let z = Zensim::new(ZensimProfile::latest_preview());
                        b.iter(|| {
                            let ri = RgbSlice::new(rgb_pix(&r), wu, hu);
                            let di = RgbSlice::new(rgb_pix(&d), wu, hu);
                            let pre = z.precompute_reference(&ri).unwrap();
                            zenbench::black_box(z.compute_with_ref(&pre, &di).unwrap().score())
                        })
                    }
                });
                g.bench("zensim__warm_ref__warm", {
                    let (r, d) = (r.clone(), d.clone());
                    let z = Zensim::new(ZensimProfile::latest_preview());
                    let ri = RgbSlice::new(rgb_pix(&r), wu, hu);
                    let pre = z.precompute_reference(&ri).unwrap();
                    move |b| {
                        b.iter(|| {
                            let di = RgbSlice::new(rgb_pix(&d), wu, hu);
                            zenbench::black_box(z.compute_with_ref(&pre, &di).unwrap().score())
                        })
                    }
                });
                g.bench("zensim__warm_ref_strip__cold", {
                    let (r, d) = (r.clone(), d.clone());
                    move |b| {
                        let z = Zensim::new(ZensimProfile::latest_preview());
                        b.iter(|| {
                            let ri = RgbSlice::new(rgb_pix(&r), wu, hu);
                            let di = RgbSlice::new(rgb_pix(&d), wu, hu);
                            let pre = z.precompute_reference(&ri).unwrap();
                            zenbench::black_box(
                                z.compute_with_ref_streaming_strips_default(&pre, &di)
                                    .unwrap()
                                    .score(),
                            )
                        })
                    }
                });
                g.bench("zensim__warm_ref_strip__warm", {
                    let (r, d) = (r.clone(), d.clone());
                    let z = Zensim::new(ZensimProfile::latest_preview());
                    let ri = RgbSlice::new(rgb_pix(&r), wu, hu);
                    let pre = z.precompute_reference(&ri).unwrap();
                    move |b| {
                        b.iter(|| {
                            let di = RgbSlice::new(rgb_pix(&d), wu, hu);
                            zenbench::black_box(
                                z.compute_with_ref_streaming_strips_default(&pre, &di)
                                    .unwrap()
                                    .score(),
                            )
                        })
                    }
                });
            }
        });
    };

    let no_gate = std::env::var("CPU_WALL_NO_GATE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let result = if no_gate {
        eprintln!(
            "[cpu-wall] CPU_WALL_NO_GATE=1 — resource gate DISABLED (caller guarantees quiet machine)"
        );
        zenbench::run_gated(zenbench::GateConfig::disabled(), build)
    } else {
        zenbench::run(build)
    };

    // Separately compute a representative score per metric for provenance
    // (the bench closures black_box the score but don't surface it).
    record_scores(&mut scores, w, h, wu, hu, &r, &d, metric_filter.as_deref());

    // Extract per-bench means from the zenbench SuiteResult and write TSV.
    write_tsv(&out_tsv, &label, w, h, &result, &scores);
}

/// Compute one score per (metric, mode) to attach to the TSV as a
/// provenance sentinel (confirms real calls, parity full vs strip).
/// Honors the metric filter so large-size per-metric runs don't recompute
/// the other (heavy) metrics' scores.
#[allow(clippy::too_many_arguments)]
fn record_scores(
    scores: &mut Vec<(String, f64)>,
    w: u32,
    h: u32,
    wu: usize,
    hu: usize,
    r: &[u8],
    d: &[u8],
    metric_filter: Option<&str>,
) {
    use imgref::ImgRef;
    let want = |m: &str| metric_filter.map(|f| f == m).unwrap_or(true);
    let mut push = |k: &str, v: f64| scores.push((k.to_string(), v));

    if want("cvvdp") {
        use cvvdp::{Cvvdp, CvvdpParams};
        let mut c = Cvvdp::new(w, h, CvvdpParams::default()).unwrap();
        push("cvvdp__full", c.score(r, d).unwrap() as f64);
        let mut c2 = Cvvdp::new(w, h, CvvdpParams::default()).unwrap();
        push(
            "cvvdp__strip",
            c2.score_strip(r, d, STRIP_H).unwrap() as f64,
        );
    }
    if want("ssim2") {
        let ri = ImgRef::new(rgb_pix(r), wu, hu);
        let di = ImgRef::new(rgb_pix(d), wu, hu);
        push(
            "ssim2__full",
            fast_ssim2::compute_ssimulacra2(ri, di).unwrap(),
        );
        let ri = ImgRef::new(rgb_pix(r), wu, hu);
        let di = ImgRef::new(rgb_pix(d), wu, hu);
        push(
            "ssim2__strip",
            fast_ssim2::compute_ssimulacra2_strip(ri, di, STRIP_H).unwrap(),
        );
    }
    if want("dssim") {
        use dssim_core::Dssim;
        let dssim = Dssim::new();
        let rr: &[rgb::RGB<u8>] = bytemuck::cast_slice(r);
        let dd: &[rgb::RGB<u8>] = bytemuck::cast_slice(d);
        let ri = dssim.create_image_rgb(rr, wu, hu).unwrap();
        let di = dssim.create_image_rgb(dd, wu, hu).unwrap();
        let (s, _m) = dssim.compare(&ri, di);
        push("dssim__full", f64::from(s));
    }
    if want("butter") {
        use butteraugli::{ButteraugliParams, ButteraugliReference};
        let p = ButteraugliParams::new();
        let rb: &[rgb::RGB<u8>] = bytemuck::cast_slice(r);
        let db: &[rgb::RGB<u8>] = bytemuck::cast_slice(d);
        let ri = ImgRef::new(rb, wu, hu);
        let di = ImgRef::new(db, wu, hu);
        push(
            "butter__full",
            butteraugli::butteraugli(ri, di, &p).unwrap().score,
        );
        let pre = ButteraugliReference::new(r, wu, hu, p.clone()).unwrap();
        let di = ImgRef::new(db, wu, hu);
        push(
            "butter__strip",
            pre.compare_strip_srgb(di, STRIP_H).unwrap().score,
        );
    }
    if want("iwssim") {
        use iwssim::{Iwssim, STRIP_BODY_DEFAULT};
        let mut c = Iwssim::new(w, h).unwrap();
        push("iwssim__full", c.score(r, d).unwrap().score);
        let mut c2 = Iwssim::new(w, h).unwrap();
        push(
            "iwssim__strip",
            c2.score_strip(r, d, STRIP_BODY_DEFAULT).unwrap().score,
        );
    }
    if want("zensim") {
        use zensim::{RgbSlice, Zensim, ZensimProfile};
        let z = Zensim::new(ZensimProfile::latest_preview());
        let ri = RgbSlice::new(rgb_pix(r), wu, hu);
        let di = RgbSlice::new(rgb_pix(d), wu, hu);
        push("zensim__full", z.compute(&ri, &di).unwrap().score());
        let ri2 = RgbSlice::new(rgb_pix(r), wu, hu);
        let di2 = RgbSlice::new(rgb_pix(d), wu, hu);
        let pre = z.precompute_reference(&ri2).unwrap();
        push(
            "zensim__strip",
            z.compute_with_ref_streaming_strips_default(&pre, &di2)
                .unwrap()
                .score(),
        );
    }
}

fn lookup_score(scores: &[(String, f64)], key: &str) -> String {
    scores
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| format!("{v}"))
        .unwrap_or_else(|| "-".to_string())
}

fn write_tsv(
    out_tsv: &str,
    label: &str,
    w: u32,
    h: u32,
    result: &SuiteResult,
    scores: &[(String, f64)],
) {
    let mut need_header = !std::path::Path::new(out_tsv).exists();
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(out_tsv)
        .expect("open out tsv");
    if need_header {
        writeln!(
            f,
            "size_label\tmetric\tmode\tcold_or_warm\tw\th\tmean_ns\tmean_ms\tn_rounds\tscore"
        )
        .unwrap();
        need_header = false;
    }
    let _ = need_header;

    for comp in &result.comparisons {
        for bm in &comp.benchmarks {
            // bm.name like "cvvdp__full", "cvvdp__warm_ref__cold", ...
            let parts: Vec<&str> = bm.name.split("__").collect();
            let metric = parts.first().copied().unwrap_or("?");
            let (mode, cw) = match parts.len() {
                2 => (parts[1], "cold"),   // full / strip → cold per-call
                3 => (parts[1], parts[2]), // warm_ref / warm_ref_strip + cold/warm
                _ => (bm.name.as_str(), "?"),
            };
            let mean_ns = bm.summary.mean;
            let mean_ms = mean_ns / 1.0e6;
            // score sentinel keyed on metric__<full|strip> (parity check)
            let score_key = match mode {
                "full" | "warm_ref" => format!("{metric}__full"),
                "strip" | "warm_ref_strip" => format!("{metric}__strip"),
                _ => format!("{metric}__full"),
            };
            let score = lookup_score(scores, &score_key);
            writeln!(
                f,
                "{label}\t{metric}\t{mode}\t{cw}\t{w}\t{h}\t{mean_ns:.1}\t{mean_ms:.4}\t{}\t{score}",
                comp.completed_rounds
            )
            .unwrap();
        }
    }
    eprintln!("wrote wall rows for size {label} to {out_tsv}");
}
