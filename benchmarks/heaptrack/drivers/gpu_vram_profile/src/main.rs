//! Phase 9.AA — peak-VRAM measurement harness for cvvdp-gpu memory modes.
//!
//! Invocation (parent):
//!   gpu-vram-profile cvvdp <mode> <width> <height> [--backend cuda|wgpu]
//!
//! `<mode>` ∈ { full strip strip_pair capped_pyramid5 capped_pyramid8 }
//! `<width>`/`<height>` in pixels.
//!
//! Parent process:
//!   1. Sample baseline `nvidia-smi --query-gpu=memory.used` (post-warmup).
//!   2. Fork a child holding the GPU buffers alive for `CHILD_HOLD_MS`.
//!   3. Sample `nvidia-smi memory.used` at ~30 ms intervals.
//!   4. Track peak, report `PEAK_VRAM_BYTES=<N>` and `PEAK_VRAM_DELTA_BYTES=<N>`
//!      on stdout, plus a human-readable summary on stderr.
//!
//! Child process (selected via `WORKER_MODE` env var):
//!   1. Build (ref, dist) synth pair via `cvvdp_gpu::synth_pair_offset`
//!      (replicates the heaptrack CPU driver's deterministic test pair).
//!   2. Construct `Cvvdp::<R>` per mode.
//!   3. Call `compute_dkl_jod(ref, dist, ppd)` exactly once.
//!   4. Print `READY <jod> warm_ms=<n>` on stdout, flush, sleep
//!      `CHILD_HOLD_MS` so the parent can sample steady-state VRAM.
//!
//! Subprocess-per-cell is mandatory because cubecl's GPU memory pool
//! retains buffers across `Drop` for reuse — a single-process before/after
//! measurement would see the Mode B baseline equal to the Full peak. The
//! OS reclaims the pool on child exit; the next child sees a clean
//! baseline.
//!
//! Output (stdout, one line per field, machine-parseable):
//!   PEAK_VRAM_BYTES=<absolute peak nvsmi memory.used in bytes>
//!   PEAK_VRAM_DELTA_BYTES=<peak - baseline in bytes>
//!   BASELINE_VRAM_BYTES=<baseline nvsmi memory.used in bytes>
//!   JOD=<f32 JOD value the child computed>
//!   ESTIMATOR_BYTES=<analytical estimator bytes, when applicable>
//!   WARM_MS=<wall-time for the child's compute_dkl_jod, ms>
//!
//! On error: nonzero exit + diagnostic on stderr. The wrapper script
//! treats an OOM (child panic on cubecl alloc failure) as an "OOM"
//! row rather than aborting the whole matrix.
//!
//! ## Backend selection at build time
//!
//! - `--features cuda` (default) → `cubecl::cuda::CudaRuntime`
//! - `--features wgpu` → `cubecl::wgpu::WgpuRuntime`
//!
//! Only one backend is active per binary. To compare backends, build
//! the harness twice (once per feature flag) and rename the resulting
//! binaries (`gpu-vram-profile-cuda` / `gpu-vram-profile-wgpu`).
//!
//! ## Why this driver lives outside the cvvdp-gpu crate
//!
//! - Mirrors the structure of `benchmarks/heaptrack/drivers/cpu_profile`
//!   for the CPU heaptrack matrix.
//! - Keeps the cvvdp-gpu crate's example surface focused on parity /
//!   in-process measurements (`mem_mode_b_vs_full.rs`, `mem_one_size.rs`).
//! - Lets the build pipeline depend only on a single bin target without
//!   pulling in the rest of the cvvdp-gpu test/parity infrastructure.

#![allow(clippy::needless_collect)]

use std::env;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Backend type alias (build-time feature selection)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
compile_error!("gpu-vram-profile requires either --features cuda or --features wgpu at build time");

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------

/// How long the child holds GPU buffers alive AFTER signalling READY so
/// the parent can sample nvidia-smi at quiescent steady state. Matches
/// `cvvdp-gpu/examples/mem_mode_b_vs_full.rs` const of the same name.
const CHILD_HOLD_MS: u64 = 800;

/// Sample interval for nvidia-smi polling. 30 ms is fast enough to catch
/// the steady-state peak after compute_dkl_jod returns, and slow enough
/// that `nvidia-smi` invocation overhead doesn't perturb the GPU process.
const SAMPLE_INTERVAL_MS: u64 = 30;

/// Strip body height for Mode E (Strip) and Mode B (StripPair) cells.
/// Aligns with the canonical default in `cvvdp_gpu::memory_mode::STRIP_H_BODY_DEFAULT`,
/// scaled down for the 4096² nvsmi number in CHANGELOG.md:1361-1363 which
/// was measured at h_body=256.
///
/// We use h_body=256 to keep the harness comparable to the CHANGELOG
/// reference. Callers wanting a sweep over h_body should script the
/// outer loop and parse PEAK_VRAM_BYTES from stdout per call.
const STRIP_H_BODY: u32 = 256;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Deterministic (ref, dist) synth pair. Mirrors
/// `cvvdp_gpu::tests::common::synth_pair_with_offset_dist` (which the
/// `mem_mode_b_vs_full.rs` example uses), but inlined here so the
/// harness doesn't depend on dev-only test fixtures.
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

/// Query `nvidia-smi --query-gpu=memory.used`. Returns memory in bytes.
/// `None` if `nvidia-smi` is not on PATH (e.g. AMD / no driver) or
/// the query fails.
fn nvidia_smi_memory_used_bytes() -> Option<u64> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "--id=0",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let mib: u64 = s.lines().next()?.trim().parse().ok()?;
    Some(mib.saturating_mul(1024 * 1024))
}

// ---------------------------------------------------------------------------
// Child process body: drive cvvdp + sleep
// ---------------------------------------------------------------------------

fn run_worker_cvvdp(mode: &str, w: u32, h: u32) -> Result<(f32, f64, Option<u64>), String> {
    use cubecl::Runtime;
    use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};
    use cvvdp_gpu::{
        Cvvdp, estimate_gpu_memory_bytes, estimate_gpu_memory_bytes_capped,
        estimate_gpu_memory_bytes_strip_pair,
    };

    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let (r, d) = synth_pair(w, h);
    let client = Backend::client(&Default::default());

    let t0 = Instant::now();
    let (jod, estimator_bytes): (f32, Option<u64>) = match mode {
        "full" => {
            let est = estimate_gpu_memory_bytes(w, h).map(|x| x as u64);
            let mut c = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
                .map_err(|e| format!("Cvvdp::new(Full): {e:?}"))?;
            let j = c
                .compute_dkl_jod(&r, &d, ppd)
                .map_err(|e| format!("compute_dkl_jod: {e:?}"))?;
            (j, est)
        }
        "strip" => {
            // Mode E (Strip): full ref + dist-strip cached-ref path.
            // Single-shot scoring uses `score()` which is Full-only;
            // Mode E is for `warm_reference` + `score_with_warm_ref_strip`.
            // For one-shot peak-VRAM measurement we exercise the
            // construct + warm_ref + score path so the cached-ref buffers
            // are allocated and visible to nvidia-smi.
            let est = None; // No public estimator for Mode E yet.
            let mut c =
                Cvvdp::<Backend>::new_strip(client, w, h, STRIP_H_BODY, CvvdpParams::PLACEHOLDER)
                    .map_err(|e| format!("Cvvdp::new_strip: {e:?}"))?;
            // For Mode E peak measurement we run the warm_reference + score
            // path that the production batch caller hits.
            c.warm_reference(&r)
                .map_err(|e| format!("warm_reference: {e:?}"))?;
            let j = c
                .compute_dkl_jod_with_warm_ref(&d, ppd)
                .map_err(|e| format!("compute_dkl_jod_with_warm_ref: {e:?}"))?;
            (j, est)
        }
        "strip_pair" => {
            // Mode B: one-shot pair stripwise.
            let est = estimate_gpu_memory_bytes_strip_pair(w, h, STRIP_H_BODY).map(|x| x as u64);
            let mut c = Cvvdp::<Backend>::new_strip_pair(
                client,
                w,
                h,
                STRIP_H_BODY,
                CvvdpParams::PLACEHOLDER,
            )
            .map_err(|e| format!("Cvvdp::new_strip_pair: {e:?}"))?;
            let j = c
                .compute_dkl_jod(&r, &d, ppd)
                .map_err(|e| format!("compute_dkl_jod: {e:?}"))?;
            (j, est)
        }
        m if m.starts_with("capped_pyramid") => {
            let levels: u32 = m
                .strip_prefix("capped_pyramid")
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| {
                    format!("bad capped_pyramid mode `{m}` — expected `capped_pyramid<N>`")
                })?;
            let est = estimate_gpu_memory_bytes_capped(w, h, levels).map(|x| x as u64);
            let mut c = Cvvdp::<Backend>::new_capped_pyramid(
                client,
                w,
                h,
                CvvdpParams::PLACEHOLDER,
                levels,
            )
            .map_err(|e| format!("Cvvdp::new_capped_pyramid({levels}): {e:?}"))?;
            let j = c
                .compute_dkl_jod(&r, &d, ppd)
                .map_err(|e| format!("compute_dkl_jod: {e:?}"))?;
            (j, est)
        }
        other => return Err(format!("unknown WORKER_MODE: {other}")),
    };
    let warm_ms = t0.elapsed().as_secs_f64() * 1e3;

    Ok((jod, warm_ms, estimator_bytes))
}

fn run_worker(metric: &str, mode: &str, w: u32, h: u32) {
    let (jod, warm_ms, est_bytes) = match metric {
        "cvvdp" => match run_worker_cvvdp(mode, w, h) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("WORKER_ERROR: {e}");
                std::process::exit(2);
            }
        },
        other => {
            eprintln!("unknown metric `{other}` — only cvvdp implemented");
            std::process::exit(64);
        }
    };

    let est_str = est_bytes
        .map(|b| b.to_string())
        .unwrap_or_else(|| "0".to_string());
    println!("READY jod={jod:.6} warm_ms={warm_ms:.2} estimator_bytes={est_str}");
    std::io::stdout().flush().expect("flush stdout");

    std::thread::sleep(Duration::from_millis(CHILD_HOLD_MS));
    // Child exits, freeing GPU buffers via OS reclaim.
}

// ---------------------------------------------------------------------------
// Parent process body: spawn child, sample nvidia-smi, report peak
// ---------------------------------------------------------------------------

/// Measurement outcome of one parent/child run.
struct Measurement {
    baseline_bytes: u64,
    peak_bytes: u64,
    delta_bytes: i64,
    jod: f32,
    warm_ms: f64,
    estimator_bytes: u64,
}

fn measure_one(
    child_bin: &str,
    metric: &str,
    mode: &str,
    w: u32,
    h: u32,
) -> Result<Measurement, String> {
    // Settle: let any prior GPU work drain. Then sample baseline.
    std::thread::sleep(Duration::from_millis(500));
    let baseline = nvidia_smi_memory_used_bytes().ok_or("nvidia-smi baseline failed")?;

    let mut child = Command::new(child_bin)
        .env("WORKER_METRIC", metric)
        .env("WORKER_MODE", mode)
        .env("WORKER_W", w.to_string())
        .env("WORKER_H", h.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn child: {e}"))?;

    // Sample nvidia-smi in the parent while the child runs. We start
    // sampling immediately so we catch the allocation transient too,
    // not just the steady-state post-readback peak.
    let stdout = child.stdout.take().ok_or("child stdout missing")?;
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();

    let ready_deadline = Instant::now() + Duration::from_secs(600);
    let mut jod = f32::NAN;
    let mut warm_ms = 0.0_f64;
    let mut estimator_bytes = 0_u64;
    let mut peak_bytes = baseline;

    // Concurrent: poll nvidia-smi while waiting for READY. The reader
    // is blocking, but we sample between attempts.
    let sample_thread_running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let peak_shared = std::sync::Arc::new(std::sync::Mutex::new(peak_bytes));
    {
        let running = std::sync::Arc::clone(&sample_thread_running);
        let peak_arc = std::sync::Arc::clone(&peak_shared);
        std::thread::spawn(move || {
            while running.load(std::sync::atomic::Ordering::Relaxed) {
                if let Some(v) = nvidia_smi_memory_used_bytes() {
                    let mut p = peak_arc.lock().unwrap();
                    if v > *p {
                        *p = v;
                    }
                }
                std::thread::sleep(Duration::from_millis(SAMPLE_INTERVAL_MS));
            }
        });
    }

    while Instant::now() < ready_deadline {
        line.clear();
        let n = reader
            .read_line(&mut line)
            .map_err(|e| format!("read child stdout: {e}"))?;
        if n == 0 {
            break;
        }
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("READY ") {
            // Parse "jod=<f> warm_ms=<f> estimator_bytes=<u64>"
            for tok in rest.split_whitespace() {
                if let Some(v) = tok.strip_prefix("jod=") {
                    jod = v.parse().unwrap_or(f32::NAN);
                } else if let Some(v) = tok.strip_prefix("warm_ms=") {
                    warm_ms = v.parse().unwrap_or(0.0);
                } else if let Some(v) = tok.strip_prefix("estimator_bytes=") {
                    estimator_bytes = v.parse().unwrap_or(0);
                }
            }
            break;
        }
    }

    if jod.is_nan() {
        // Child died before sending READY — collect diagnostic.
        sample_thread_running.store(false, std::sync::atomic::Ordering::Relaxed);
        let out = child.wait_with_output().map_err(|e| format!("wait: {e}"))?;
        return Err(format!(
            "child never sent READY: status={:?}\nstdout:\n{}\nstderr:\n{}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        ));
    }

    // Hold sampling open for CHILD_HOLD_MS to catch steady-state peak.
    let sample_until = Instant::now() + Duration::from_millis(CHILD_HOLD_MS);
    while Instant::now() < sample_until {
        std::thread::sleep(Duration::from_millis(SAMPLE_INTERVAL_MS));
    }
    sample_thread_running.store(false, std::sync::atomic::Ordering::Relaxed);

    peak_bytes = *peak_shared.lock().unwrap();

    let out = child.wait_with_output().map_err(|e| format!("wait: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "child failed: status={:?}\nstdout:\n{}\nstderr:\n{}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        ));
    }

    let delta_bytes = (peak_bytes as i64) - (baseline as i64);

    Ok(Measurement {
        baseline_bytes: baseline,
        peak_bytes,
        delta_bytes,
        jod,
        warm_ms,
        estimator_bytes,
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> ExitCode {
    // Worker entry?
    if let Ok(metric) = env::var("WORKER_METRIC") {
        let mode = env::var("WORKER_MODE").expect("WORKER_MODE");
        let w: u32 = env::var("WORKER_W").expect("WORKER_W").parse().unwrap();
        let h: u32 = env::var("WORKER_H").expect("WORKER_H").parse().unwrap();
        run_worker(&metric, &mode, w, h);
        return ExitCode::SUCCESS;
    }

    let args: Vec<String> = env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "usage: gpu-vram-profile <metric> <mode> <width> <height> [--backend cuda|wgpu]\n\
             metrics: cvvdp\n\
             modes:   full strip strip_pair capped_pyramid<N>  (e.g. capped_pyramid5)\n\
             outputs (stdout, one per line):\n  \
                PEAK_VRAM_BYTES=<u64>\n  \
                PEAK_VRAM_DELTA_BYTES=<i64>\n  \
                BASELINE_VRAM_BYTES=<u64>\n  \
                JOD=<f32>\n  \
                ESTIMATOR_BYTES=<u64>\n  \
                WARM_MS=<f64>"
        );
        return ExitCode::from(64);
    }
    let metric = args[1].as_str();
    let mode = args[2].as_str();
    let w: u32 = args[3].parse().expect("width");
    let h: u32 = args[4].parse().expect("height");

    // Best-effort --backend forwarding. The active backend is selected
    // at build time via the cargo --features flag; this parameter only
    // affects diagnostic labelling.
    let backend_label: String = args
        .iter()
        .skip_while(|a| a.as_str() != "--backend")
        .nth(1)
        .cloned()
        .unwrap_or_else(|| {
            if cfg!(feature = "cuda") {
                "cuda".to_string()
            } else {
                "wgpu".to_string()
            }
        });

    let child_bin = env::current_exe()
        .expect("current_exe")
        .to_string_lossy()
        .into_owned();

    let result = measure_one(&child_bin, metric, mode, w, h);
    match result {
        Ok(m) => {
            println!("PEAK_VRAM_BYTES={}", m.peak_bytes);
            println!("PEAK_VRAM_DELTA_BYTES={}", m.delta_bytes);
            println!("BASELINE_VRAM_BYTES={}", m.baseline_bytes);
            println!("JOD={:.6}", m.jod);
            println!("ESTIMATOR_BYTES={}", m.estimator_bytes);
            println!("WARM_MS={:.2}", m.warm_ms);
            eprintln!(
                "# cell {metric}/{mode} {w}x{h} backend={backend_label}: \
                 baseline {:.1} MiB, peak {:.1} MiB, delta {:+.1} MiB, jod {:.4}, warm {:.0} ms",
                m.baseline_bytes as f64 / (1u64 << 20) as f64,
                m.peak_bytes as f64 / (1u64 << 20) as f64,
                m.delta_bytes as f64 / (1u64 << 20) as f64,
                m.jod,
                m.warm_ms,
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("FAIL {metric}/{mode} {w}x{h}: {e}");
            ExitCode::from(1)
        }
    }
}
