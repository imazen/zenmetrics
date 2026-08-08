//! CUDA vs Vulkan (cubecl-wgpu) A/B for the ssim2 GPU kernel.
//!
//! Answers: how fast is cubecl on Vulkan relative to the CUDA backend, for
//! the SAME kernel on the SAME card? Both backends run the identical typed
//! `Ssim2<R>` pipeline; only the `cubecl::Runtime` differs.
//!
//! Build + run (needs BOTH backends compiled in so one process can
//! interleave them):
//! ```bash
//! cargo run --release -p ssim2-gpu --example backend_ab \
//!     --no-default-features --features cuda,wgpu,cubecl-types,cpu
//! ```
//!
//! ## Two environment gotchas that cost real time (2026-08-07, lianli)
//!
//! 1. **`CUDA_PATH` must point at a toolkit with `include/`, not just libs.**
//!    cubecl-cuda JITs kernels through NVRTC, which needs `cuda_runtime.h` &
//!    co. `LD_LIBRARY_PATH` alone is NOT enough: `cubecl-cuda/src/lib.rs`
//!    calls `cuda_path()`, which reads `$CUDA_PATH` or `/usr/local/cuda`, and
//!    **panics on a cubecl worker thread** when absent. That panic is invisible
//!    to any `Result` on the calling thread, so the run happily prints a
//!    column of garbage CUDA timings. The preflight below exists to catch
//!    exactly that. On a box with only the driver (no toolkit), the CUDA
//!    runtime can be lifted out of the fleet worker image:
//!    ```bash
//!    cid=$(docker create ghcr.io/imazen/zenfleet-worker:exec-gpu)
//!    docker cp "$cid":/usr/local/cuda/. ~/opt/cuda/   # include/ + lib64/
//!    export CUDA_PATH=$HOME/opt/cuda LD_LIBRARY_PATH=$HOME/opt/cuda/lib64:$LD_LIBRARY_PATH
//!    ```
//!
//! 2. **The zenmetrics workspace may not resolve on a secondary box.** The
//!    root `[patch]` table pins sibling paths (e.g. a local `dssim-core ^3.5`
//!    that is unpushed), so `cargo -p ssim2-gpu` can die at resolution over a
//!    crate ssim2-gpu does not even depend on. ssim2-gpu's own deps are just
//!    `cubecl` + `zenmetrics-gpu-core`, so a throwaway out-of-workspace crate
//!    that path-depends on it resolves only that subgraph and builds fine.
//!    Mirror `[profile.release]` from the workspace root or the timings are
//!    not comparable.
//!
//! ## Why the adapter check is load-bearing
//! A Linux box with Mesa installed exposes **two** Vulkan devices: the real
//! discrete GPU and `llvmpipe`, a CPU software rasterizer. If wgpu selects
//! llvmpipe the numbers are meaningless (they measure the CPU). This example
//! therefore selects Vulkan explicitly via `init_setup::<Vulkan>`, prints the
//! chosen `wgpu::AdapterInfo`, and **aborts** unless the adapter is a discrete
//! GPU — set `ALLOW_SOFTWARE_ADAPTER=1` to override (only for a smoke test;
//! never for numbers you intend to report).
//!
//! ## Method
//! - Measures the **warm-reference** path (`set_reference` once, then
//!   `compute_with_reference` per candidate). That is the production scoring
//!   shape: score many distorted images against one reference.
//! - Instance construction (`Ssim2::new`, which allocates the whole pyramid)
//!   is timed separately — it is the per-cell fixed cost `α` that dominates
//!   tiny images, per the sweep discipline in CLAUDE.md.
//! - Rounds alternate **ABBA** so any monotonic thermal/clock drift cancels
//!   rather than being charged to whichever backend ran second.
//! - Only ONE backend holds a pipeline at a time. Two live 12 MP ssim2
//!   pipelines would not fit an 8 GB card (the whole-image path allocates
//!   ~57 f32 planes/scale × 6 scales).
//! - Cross-backend **score agreement** is checked and reported. If the two
//!   backends disagree, a speed comparison is not meaningful — that is a
//!   correctness bug, not a benchmark result.
//!
//! Env knobs: `BENCH_SIZES` (`WxH,WxH,...`), `BENCH_REPS`, `BENCH_ROUNDS`,
//! `BENCH_WARMUP`, `BENCH_CSV`, `ALLOW_SOFTWARE_ADAPTER`.

#[cfg(all(feature = "cuda", feature = "wgpu"))]
mod ab {
    use cubecl::Runtime;
    use cubecl::client::ComputeClient;
    use ssim2_gpu::Ssim2;
    use std::time::Instant;

    /// Deterministic LCG sRGB generator — identical bytes for a given
    /// (seed, w, h) so both backends score the exact same pixels.
    fn make_srgb(w: u32, h: u32, seed: u32) -> Vec<u8> {
        use std::num::Wrapping;
        let n = (w as usize) * (h as usize);
        let mut v = Vec::with_capacity(n * 3);
        let mut s = Wrapping(seed);
        for _ in 0..n {
            s = s * Wrapping(1_664_525_u32) + Wrapping(1_013_904_223_u32);
            v.push(((s.0 >> 16) & 0xFF) as u8);
            v.push(((s.0 >> 8) & 0xFF) as u8);
            v.push((s.0 & 0xFF) as u8);
        }
        v
    }

    fn median(v: &mut [f64]) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if v.is_empty() {
            return f64::NAN;
        }
        let m = v.len() / 2;
        if v.len() % 2 == 0 {
            (v[m - 1] + v[m]) / 2.0
        } else {
            v[m]
        }
    }

    fn pct(v: &mut [f64], p: f64) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if v.is_empty() {
            return f64::NAN;
        }
        let idx = (((v.len() - 1) as f64) * p).round() as usize;
        v[idx]
    }

    /// One backend's measurement for one image size.
    pub struct Run {
        pub create_ms: Vec<f64>,
        pub warm_ms: Vec<f64>,
        pub score: f64,
    }

    /// Build a pipeline, cache the reference, and time `reps`
    /// warm-reference scores. Generic over the cubecl runtime — this is the
    /// whole point: byte-identical work, different backend.
    fn measure<R: Runtime>(
        client: ComputeClient<R>,
        w: u32,
        h: u32,
        refimg: &[u8],
        dist: &[u8],
        warmup: usize,
        reps: usize,
    ) -> Result<Run, String> {
        let t0 = Instant::now();
        let mut s = Ssim2::<R>::new(client, w, h).map_err(|e| format!("Ssim2::new: {e:?}"))?;
        let create_ms = t0.elapsed().as_secs_f64() * 1e3;

        s.set_reference(refimg)
            .map_err(|e| format!("set_reference: {e:?}"))?;

        let mut score = f64::NAN;
        for _ in 0..warmup {
            score = s
                .compute_with_reference(dist)
                .map_err(|e| format!("warmup: {e:?}"))?
                .score as f64;
        }

        let mut warm_ms = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t = Instant::now();
            let r = s
                .compute_with_reference(dist)
                .map_err(|e| format!("compute: {e:?}"))?;
            warm_ms.push(t.elapsed().as_secs_f64() * 1e3);
            score = r.score as f64;
        }

        Ok(Run {
            create_ms: vec![create_ms],
            warm_ms,
            score,
        })
    }

    fn env_usize(k: &str, d: usize) -> usize {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    }

    /// Snapshot of other GPU tenants — a benchmark sharing the card with a
    /// fleet worker is measuring contention, not the backend.
    fn gpu_tenants() -> String {
        let out = std::process::Command::new("nvidia-smi")
            .args([
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ])
            .output();
        match out {
            Ok(o) => {
                let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if s.is_empty() {
                    "none".into()
                } else {
                    s.replace('\n', " | ")
                }
            }
            Err(_) => "nvidia-smi unavailable".into(),
        }
    }

    pub fn main() {
        let sizes: Vec<(u32, u32)> = std::env::var("BENCH_SIZES")
            .unwrap_or_else(|_| "64x64,256x256,1024x1024,2048x2048".into())
            .split(',')
            .filter_map(|t| {
                let (a, b) = t.trim().split_once('x')?;
                Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
            })
            .collect();
        let reps = env_usize("BENCH_REPS", 5);
        let rounds = env_usize("BENCH_ROUNDS", 3);
        let warmup = env_usize("BENCH_WARMUP", 2);

        // ---- Backend setup, with explicit + verified Vulkan selection ----
        println!("== backend setup ==");

        let cuda_client = cubecl::cuda::CudaRuntime::client(&Default::default());
        println!("  cuda   : CudaRuntime (default device)");

        use cubecl::wgpu::{
            RuntimeOptions, Vulkan, WgpuDevice, WgpuRuntime, init_device, init_setup,
        };
        let setup = init_setup::<Vulkan>(&WgpuDevice::default(), RuntimeOptions::default());
        let info = setup.adapter.get_info();
        let dtype = format!("{:?}", info.device_type);
        println!(
            "  vulkan : adapter={:?} type={} backend={:?} driver={:?} {}",
            info.name, dtype, info.backend, info.driver, info.driver_info
        );
        if !dtype.contains("DiscreteGpu") && std::env::var("ALLOW_SOFTWARE_ADAPTER").is_err() {
            eprintln!(
                "\nFATAL: Vulkan adapter is {dtype}, not DiscreteGpu — this would \
                 benchmark a CPU software rasterizer (llvmpipe), not the GPU.\n\
                 Numbers from such a run are meaningless. Set \
                 ALLOW_SOFTWARE_ADAPTER=1 only for a smoke test."
            );
            std::process::exit(3);
        }
        let wgpu_device = init_device(setup, RuntimeOptions::default());
        let wgpu_client = WgpuRuntime::client(&wgpu_device);

        println!("  other GPU tenants at start: {}", gpu_tenants());

        // ---- Preflight -------------------------------------------------
        // A backend that fails to initialise can still print numbers: with
        // CUDA_PATH unset, cubecl-cuda panics on a *worker* thread, which no
        // Result on this thread ever observes, and the run emits a
        // plausible-looking column of garbage (observed 2026-08-07). Prove
        // both backends produce a sane ssim2 score before trusting timings.
        {
            let (pw, ph) = (128u32, 128u32);
            let pr = make_srgb(pw, ph, 1);
            let pd = make_srgb(pw, ph, 2);
            let c = match measure::<cubecl::cuda::CudaRuntime>(
                cuda_client.clone(),
                pw,
                ph,
                &pr,
                &pd,
                0,
                1,
            ) {
                Ok(r) => r.score,
                Err(e) => {
                    eprintln!("\nFATAL: cuda preflight failed: {e}");
                    std::process::exit(4);
                }
            };
            let v = match measure::<WgpuRuntime>(wgpu_client.clone(), pw, ph, &pr, &pd, 0, 1) {
                Ok(r) => r.score,
                Err(e) => {
                    eprintln!("\nFATAL: vulkan preflight failed: {e}");
                    std::process::exit(4);
                }
            };
            // No range assumption: SSIMULACRA2 is 100 for identical images and
            // decreases without a floor — two independent noise images score
            // strongly NEGATIVE, which is correct, not broken. The
            // assumption-free liveness test is cross-backend agreement: a
            // backend that failed to initialise cannot agree with one that did.
            let agree = (c - v).abs() / c.abs().max(1.0);
            if !c.is_finite() || !v.is_finite() || c > 100.001 || v > 100.001 || agree > 1e-3 {
                eprintln!(
                    "\nFATAL: preflight disagreement — cuda={c:.6} vulkan={v:.6} \
                     (rel {agree:.2e}). One backend is not computing the real kernel; \
                     refusing to report timings."
                );
                std::process::exit(4);
            }
            println!("  preflight: ok (cuda {c:.4} vs vulkan {v:.4}, rel {agree:.2e})");
        }

        println!(
            "\n  method: warm-reference path; ABBA round order; {rounds} rounds \
             × {reps} reps (+{warmup} warmup), one live pipeline at a time\n"
        );

        println!(
            "{:>11} {:>10} {:>10} {:>10} {:>10} {:>8}  {:>12} {:>12}  {}",
            "size",
            "cuda_med",
            "vk_med",
            "cuda_min",
            "vk_min",
            "vk/cuda",
            "cuda_new",
            "vk_new",
            "score_delta"
        );

        let mut csv = String::from(
            "size,pixels,cuda_med_ms,vk_med_ms,cuda_min_ms,vk_min_ms,cuda_p90_ms,vk_p90_ms,\
             ratio_vk_over_cuda,cuda_create_ms,vk_create_ms,cuda_score,vk_score,score_abs_delta\n",
        );

        for (w, h) in &sizes {
            let (w, h) = (*w, *h);
            let refimg = make_srgb(w, h, 0x1234_5678);
            let dist = make_srgb(w, h, 0x9ABC_DEF0);

            let mut c_warm = Vec::new();
            let mut v_warm = Vec::new();
            let mut c_new = Vec::new();
            let mut v_new = Vec::new();
            let (mut c_score, mut v_score) = (f64::NAN, f64::NAN);
            let mut failed: Option<String> = None;

            for round in 0..rounds {
                // ABBA: flip order each round so linear drift cancels.
                let cuda_first = round % 2 == 0;
                for turn in 0..2 {
                    let do_cuda = (turn == 0) == cuda_first;
                    let res = if do_cuda {
                        measure::<cubecl::cuda::CudaRuntime>(
                            cuda_client.clone(),
                            w,
                            h,
                            &refimg,
                            &dist,
                            warmup,
                            reps,
                        )
                    } else {
                        measure::<WgpuRuntime>(
                            wgpu_client.clone(),
                            w,
                            h,
                            &refimg,
                            &dist,
                            warmup,
                            reps,
                        )
                    };
                    match res {
                        Ok(r) => {
                            if do_cuda {
                                c_warm.extend(r.warm_ms);
                                c_new.extend(r.create_ms);
                                c_score = r.score;
                            } else {
                                v_warm.extend(r.warm_ms);
                                v_new.extend(r.create_ms);
                                v_score = r.score;
                            }
                        }
                        Err(e) => {
                            failed = Some(format!(
                                "{} @ {w}x{h}: {e}",
                                if do_cuda { "cuda" } else { "vulkan" }
                            ));
                        }
                    }
                }
                if failed.is_some() {
                    break;
                }
            }

            if let Some(e) = failed {
                println!("{:>11} FAILED: {}", format!("{w}x{h}"), e);
                csv.push_str(&format!(
                    "{w}x{h},{},,,,,,,,,,,,FAILED\n",
                    w as u64 * h as u64
                ));
                continue;
            }

            let cm = median(&mut c_warm.clone());
            let vm = median(&mut v_warm.clone());
            let cmin = pct(&mut c_warm.clone(), 0.0);
            let vmin = pct(&mut v_warm.clone(), 0.0);
            let c90 = pct(&mut c_warm.clone(), 0.90);
            let v90 = pct(&mut v_warm.clone(), 0.90);
            let cn = median(&mut c_new.clone());
            let vn = median(&mut v_new.clone());
            let delta = (c_score - v_score).abs();
            // ssim2 is a 0..100 scale, so judge agreement RELATIVE to the
            // score. Cross-backend float reduction order differs (the two
            // runtimes schedule the per-octave reduction differently), which
            // shows up as ~1e-5 relative noise — expected, not divergence.
            // Flag only a departure large enough to change a picker decision.
            let rel = if c_score.abs() > 0.0 {
                delta / c_score.abs()
            } else {
                delta
            };

            println!(
                "{:>11} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>8.2}x {:>12.1} {:>12.1}  {:.3e}{}",
                format!("{w}x{h}"),
                cm,
                vm,
                cmin,
                vmin,
                vm / cm,
                cn,
                vn,
                delta,
                if rel > 1e-4 {
                    "  <-- SCORES DIVERGE"
                } else {
                    ""
                }
            );

            csv.push_str(&format!(
                "{w}x{h},{},{cm:.4},{vm:.4},{cmin:.4},{vmin:.4},{c90:.4},{v90:.4},\
                 {:.4},{cn:.3},{vn:.3},{c_score:.6},{v_score:.6},{delta:.3e}\n",
                w as u64 * h as u64,
                vm / cm
            ));
        }

        println!("\n  other GPU tenants at end: {}", gpu_tenants());

        let path = std::env::var("BENCH_CSV").unwrap_or_else(|_| {
            format!(
                "benchmarks/ssim2_backend_ab_{}.csv",
                std::env::var("BENCH_DATE").unwrap_or_else(|_| "latest".into())
            )
        });
        if let Some(parent) = std::path::Path::new(&path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match std::fs::write(&path, &csv) {
            Ok(()) => println!("  wrote {path}"),
            Err(e) => eprintln!("  WARN: could not write {path}: {e}"),
        }
    }
}

#[cfg(all(feature = "cuda", feature = "wgpu"))]
fn main() {
    ab::main()
}

#[cfg(not(all(feature = "cuda", feature = "wgpu")))]
fn main() {
    eprintln!(
        "backend_ab needs BOTH backends compiled in.\n\
         Rebuild with: --features cuda,wgpu,cubecl-types"
    );
    std::process::exit(2);
}
