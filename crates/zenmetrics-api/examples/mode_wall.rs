//! mode_wall — Full-vs-Strip(-vs-StripPair) WALL-TIME + peak-VRAM wall
//! benchmark across ALL SIX umbrella metrics, BOTH contexts (one-off and
//! warm/batch), four sizes (256² / 512² / 1024² / 4096²). Task #157 Phase A.
//!
//! ## Why this exists
//!
//! Every per-crate `resolve_auto` (`crates/<metric>-gpu/src/memory_mode.rs`)
//! is a memory-only heuristic — "Full when it fits the cap, else Strip" —
//! resting on the UNMEASURED assumption that "Full is fastest when it
//! fits". It never considers StripPair, and it never weighs the fact that
//! Full's huge working set might lose cache locality at large sizes. This
//! harness fills the Full-vs-StripPair wall benchmark that
//! `crates/cvvdp-gpu/docs/MODE_SELECTION.md` mandated and left unfilled,
//! and generalises it to all six metrics so each metric's `resolve_auto`
//! can be grounded in real numbers (Phase B, gated on a design decision —
//! NOT in this harness).
//!
//! ## What is measured, per `(metric, mode, size, context)` cell
//!
//! - `wall_ms_p25 / p50 / p75` — wall time of the timed operation,
//!   synced (host readback or explicit `client.sync()` inside the timed
//!   region, so the wall is real GPU execution, not async submission).
//!   n≥20 timed reps, percentiles reported.
//! - `peak_vram_mib` — peak `nvidia-smi memory.used` delta over the
//!   baseline sampled immediately before the child launched, sampled
//!   during a hold window the child takes after its timed reps complete.
//! - `free_vram_start_mib` — free VRAM (`nvidia-smi memory.free`) the
//!   parent saw right before launching the child. The GPU may be shared
//!   with another session; cells launched under contention are flagged
//!   (`note` column) and their VRAM number is suspect (timing still valid
//!   because the timed region is synced and isolated to this process).
//! - `score` — the metric value for the (ref, dist) pair, carried so the
//!   parity check (each non-Full mode's score == Full within the metric's
//!   Atomic<f32> reduction-noise band) can run from the committed CSV.
//!
//! ## Contexts
//!
//! - **one-off** — construct a fresh `Metric` + `compute_srgb_u8` a single
//!   pair, fresh each rep (mirrors `score_pair` / `Metric::compute_srgb_u8`).
//!   Construction IS inside the timed region for this context (that is what
//!   a one-off CLI caller pays). A cubecl context + kernel warmup pass runs
//!   ONCE before the timed reps so the per-rep wall is steady-state JIT-hot
//!   construction+compute, not first-call kernel-compile.
//! - **warm** — `set_reference_srgb_u8` once, then time
//!   `compute_with_cached_reference_srgb_u8` per dist over many dists
//!   (mirrors the orchestrator's cached-ref path). A warmup score runs
//!   before the timed reps.
//!
//! ## Modes
//!
//! - **Full** / **Strip** — forced via the umbrella
//!   `Metric::new_with_memory_mode(kind, Cuda, w, h, params, mode)`.
//! - **StripPair** (cvvdp only) — the umbrella `MemoryMode` has NO
//!   StripPair variant (it's a cvvdp-specific Mode B), so StripPair is
//!   forced through the typed `cvvdp_gpu::CvvdpOpaque::new_with_memory_mode(
//!   .., cvvdp_gpu::MemoryMode::StripPair { h_body })`. This is the only
//!   reachable path for StripPair.
//!
//! ## IMPORTANT measured caveats (read before interpreting results)
//!
//! - cvvdp `Strip` (Mode E) and `StripPair` (Mode B) are **cached-ref /
//!   strip-walker paths**. cvvdp one-off `compute_srgb_u8` routes through
//!   the Full pipeline for ALL modes (`pipeline.rs::score` for StripPair
//!   explicitly; Strip is cached-ref-only). So a cvvdp `Strip`/`StripPair`
//!   one-off cell measures the Full pipeline wall regardless of the
//!   constructed mode — the harness records the cell anyway (the
//!   `note` column flags `cvvdp_oneoff_routes_full`) so the data shows the
//!   construction-mode-vs-runtime-path divergence explicitly.
//! - At current master HEAD the cvvdp StripPair runtime memory savings are
//!   NOT yet wired (`pipeline.rs::score` comment: "the constructor today
//!   still allocates Full-mode buffers, so runtime savings are zero until
//!   Chunk 2 lands"). Expect StripPair peak VRAM ≈ Full at HEAD.
//!
//! ## Subprocess-per-cell
//!
//! cubecl's memory pool keeps GPU buffers cached across `Drop` for reuse,
//! defeating a single-process before/after VRAM measurement. Each cell
//! runs in a child process: the OS reclaims the pool on child exit, so
//! the next child sees a clean baseline. (Same pattern as
//! `cvvdp-gpu/examples/mem_mode_b_vs_full.rs` and
//! `scripts/memory_audit/audit_gpu_metrics.py`.)
//!
//! ## Build (release, cuda, cubecl-types — NO target-cpu=native)
//! ```sh
//! PATH=/usr/local/cuda/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
//! cargo build --release -p zenmetrics-api \
//!   --no-default-features --features cuda,all-metrics,cubecl-types,pixels \
//!   --example mode_wall
//! ```
//!
//! ## Run (parent enumerates the grid, spawns one child per cell)
//! ```sh
//! MW_REPS=20 MW_SIZES=256,512,1024,4096 \
//!   target/release/examples/mode_wall > benchmarks/mode_wall_<date>.csv
//! ```
//! Restrict the grid for a smoke run: `MW_METRICS=cvvdp MW_SIZES=256,512`.

#![cfg(feature = "cubecl-types")]

use std::env;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::ComputeClient;
use zenmetrics_api::{Backend, MemoryMode, Metric, MetricKind, MetricParams};

type Rt = CudaRuntime;

// --------------------------------------------------------------------
// Grid
// --------------------------------------------------------------------

const ALL_METRICS: &[(&str, MetricKind)] = &[
    ("cvvdp", MetricKind::Cvvdp),
    ("butter", MetricKind::Butter),
    ("ssim2", MetricKind::Ssim2),
    ("dssim", MetricKind::Dssim),
    ("iwssim", MetricKind::Iwssim),
    ("zensim", MetricKind::Zensim),
];

/// Modes per metric. cvvdp gets {Full, Strip, StripPair}; the other five
/// get {Full, Strip}; zensim only {Full} (no strip path of its own — it
/// surfaces a clear error if asked for Strip, so we skip it to avoid a
/// failed cell). Excludes CappedPyramid (not JOD-identical, opt-in only).
fn modes_for(metric: &str) -> &'static [&'static str] {
    match metric {
        "cvvdp" => &["full", "strip", "strippair"],
        "zensim" => &["full"],
        _ => &["full", "strip"],
    }
}

const CONTEXTS: &[&str] = &["oneoff", "warm"];

/// Cells that are unsupported BY DESIGN (would panic / error), with the
/// reason. Recorded as a `SKIP` row in the CSV so the data documents the
/// gap rather than hiding it. cvvdp Strip (Mode E) is a cached-ref-only
/// path: a one-off `compute_srgb_u8` on a Strip-constructed cvvdp hits the
/// Mode-E band loop with no `ref_full_state` cached and panics
/// (`pipeline.rs:4963`). There is no valid cvvdp one-off Strip path.
fn unsupported_reason(metric: &str, mode: &str, ctx: &str) -> Option<&'static str> {
    match (metric, mode, ctx) {
        ("cvvdp", "strip", "oneoff") => {
            Some("cvvdp_strip_mode_e_is_cached_ref_only_no_oneoff_path")
        }
        _ => None,
    }
}

/// Strip body height in rows. Power of two (the strip walkers assume it).
/// 256 matches the gpu_memory_audit reference table so VRAM numbers are
/// comparable, and is the smallest aligned body so the strip path is
/// genuinely exercised (more than one strip) at 512²+.
const STRIP_H_BODY: u32 = 256;

/// How long the child holds GPU buffers alive AFTER its timed reps so the
/// parent can sample nvidia-smi at quiescent steady state.
const CHILD_HOLD_MS: u64 = 400;

// --------------------------------------------------------------------
// Shared helpers
// --------------------------------------------------------------------

fn parse_u32(name: &str, default: u32) -> u32 {
    env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    // Linear-interpolation percentile (type-7, the numpy/Excel default).
    let rank = p / 100.0 * (sorted.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = rank - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Deterministic LCG-filled sRGB bytes, seed-controlled so distinct reps
/// can use distinct pixel content. Same generator shape as
/// `setref_all_timing.rs::synth_srgb`.
fn synth_srgb(w: u32, h: u32, seed: u32) -> Vec<u8> {
    use std::num::Wrapping;
    let n = (w as usize) * (h as usize) * 3;
    let mut v = Vec::with_capacity(n);
    let mut s = Wrapping(seed.wrapping_mul(2_654_435_761).wrapping_add(1));
    for _ in 0..n {
        s = s * Wrapping(1_664_525_u32) + Wrapping(1_013_904_223_u32);
        v.push(((s.0 >> 16) & 0xff) as u8);
    }
    v
}

fn metric_kind_from_tag(s: &str) -> Option<MetricKind> {
    ALL_METRICS.iter().find(|(t, _)| *t == s).map(|(_, k)| *k)
}

fn umbrella_mode(mode: &str) -> Option<MemoryMode> {
    match mode {
        "full" => Some(MemoryMode::Full),
        "strip" => Some(MemoryMode::Strip { h_body: Some(STRIP_H_BODY) }),
        // strippair handled separately (cvvdp typed API)
        _ => None,
    }
}

fn nvidia_smi_query(field: &str) -> Option<u64> {
    let out = Command::new("nvidia-smi")
        .args([
            &format!("--query-gpu={field}"),
            "--format=csv,noheader,nounits",
            "--id=0",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    s.lines().next()?.trim().parse().ok()
}

// --------------------------------------------------------------------
// Worker: construct forced mode, run the context, emit RESULT line, hold.
// --------------------------------------------------------------------

/// One-off worker for the umbrella path (Full / Strip). Each rep
/// constructs a fresh Metric + scores one pair (+sync), individually
/// timed. Construction IS inside the timed region (mirrors `score_pair`).
fn time_oneoff_umbrella(
    kind: MetricKind,
    mode: MemoryMode,
    w: u32,
    h: u32,
    client: &ComputeClient<Rt>,
    reps: usize,
) -> (Vec<f64>, f64) {
    let r = synth_srgb(w, h, 0x0A11_0001);
    let d = synth_srgb(w, h, 0x0A11_0002);
    let params = MetricParams::try_default_for(kind).expect("params");
    // Warmup: one full construct+score (loads kernels, fills pool), synced.
    {
        let mut m = Metric::new_with_memory_mode(kind, Backend::Cuda, w, h, params.clone(), mode)
            .expect("warmup construct");
        let _ = m.compute_srgb_u8(&r, &d).expect("warmup score");
        cubecl::future::block_on(client.sync()).expect("warmup sync");
        // Drop returns buffers to the pool; subsequent reps reuse it.
    }
    let mut times = Vec::with_capacity(reps);
    let mut score = f64::NAN;
    for _ in 0..reps {
        let t = Instant::now();
        let mut m =
            Metric::new_with_memory_mode(kind, Backend::Cuda, w, h, params.clone(), mode)
                .expect("construct");
        let s = m.compute_srgb_u8(&r, &d).expect("score");
        cubecl::future::block_on(client.sync()).expect("sync");
        times.push(t.elapsed().as_secs_f64() * 1e3);
        score = s.value;
        drop(m);
    }
    (times, score)
}

/// Warm/batch worker for the umbrella path. Construct once, set_reference
/// once, then time compute_with_cached_reference per distinct dist.
fn time_warm_umbrella(
    kind: MetricKind,
    mode: MemoryMode,
    w: u32,
    h: u32,
    client: &ComputeClient<Rt>,
    reps: usize,
) -> (Vec<f64>, f64) {
    let r = synth_srgb(w, h, 0x0B22_0001);
    let params = MetricParams::try_default_for(kind).expect("params");
    let mut m = Metric::new_with_memory_mode(kind, Backend::Cuda, w, h, params, mode)
        .expect("construct warm");
    m.set_reference_srgb_u8(&r).expect("set_reference");
    // Warmup score against ref (JIT + pool), synced.
    let warm_d = synth_srgb(w, h, 0x0B22_9999);
    let _ = m
        .compute_with_cached_reference_srgb_u8(&warm_d)
        .expect("warmup cached score");
    cubecl::future::block_on(client.sync()).expect("warm warmup sync");

    let mut times = Vec::with_capacity(reps);
    let mut score = f64::NAN;
    for i in 0..reps {
        let d = synth_srgb(w, h, 0x0B22_0100u32.wrapping_add(i as u32 * 13 + 1));
        let t = Instant::now();
        let s = m
            .compute_with_cached_reference_srgb_u8(&d)
            .expect("cached score");
        cubecl::future::block_on(client.sync()).expect("sync");
        times.push(t.elapsed().as_secs_f64() * 1e3);
        score = s.value;
    }
    (times, score)
}

/// One-off worker for cvvdp StripPair (typed opaque API — umbrella has no
/// StripPair). Each rep constructs a fresh CvvdpOpaque(StripPair) + scores
/// one pair (+sync). NB: cvvdp one-off routes through Full internally for
/// all modes — this measures the Full pipeline wall.
#[cfg(feature = "cvvdp")]
fn time_oneoff_cvvdp_strippair(
    w: u32,
    h: u32,
    client: &ComputeClient<Rt>,
    reps: usize,
) -> (Vec<f64>, f64) {
    use cvvdp_gpu::params::CvvdpParams;
    let r = synth_srgb(w, h, 0x0A11_0001);
    let d = synth_srgb(w, h, 0x0A11_0002);
    let mk = || {
        cvvdp_gpu::CvvdpOpaque::new_with_memory_mode(
            cvvdp_gpu::Backend::Cuda,
            w,
            h,
            CvvdpParams::default(),
            cvvdp_gpu::MemoryMode::StripPair { h_body: Some(STRIP_H_BODY) },
        )
        .expect("construct StripPair")
    };
    // Warmup.
    {
        let mut m = mk();
        let _ = m.compute_srgb_u8(&r, &d).expect("warmup score");
        cubecl::future::block_on(client.sync()).expect("warmup sync");
    }
    let mut times = Vec::with_capacity(reps);
    let mut score = f64::NAN;
    for _ in 0..reps {
        let t = Instant::now();
        let mut m = mk();
        let s = m.compute_srgb_u8(&r, &d).expect("score");
        cubecl::future::block_on(client.sync()).expect("sync");
        times.push(t.elapsed().as_secs_f64() * 1e3);
        score = s.value;
        drop(m);
    }
    (times, score)
}

/// Warm/batch worker for cvvdp StripPair. StripPair is a one-shot-pair
/// mode with NO cached-ref; cvvdp's warm path is Mode E (Strip). When a
/// StripPair-constructed cvvdp is asked to warm, the documented behaviour
/// is fall-back to Mode E. We still construct StripPair and run the warm
/// path so the data shows what StripPair+warm actually costs; the
/// `note` column flags this.
#[cfg(feature = "cvvdp")]
fn time_warm_cvvdp_strippair(
    w: u32,
    h: u32,
    client: &ComputeClient<Rt>,
    reps: usize,
) -> Result<(Vec<f64>, f64), String> {
    use cvvdp_gpu::params::CvvdpParams;
    let r = synth_srgb(w, h, 0x0B22_0001);
    let mut m = cvvdp_gpu::CvvdpOpaque::new_with_memory_mode(
        cvvdp_gpu::Backend::Cuda,
        w,
        h,
        CvvdpParams::default(),
        cvvdp_gpu::MemoryMode::StripPair { h_body: Some(STRIP_H_BODY) },
    )
    .map_err(|e| format!("construct StripPair: {e}"))?;
    m.warm_reference_srgb(&r).map_err(|e| format!("warm_reference: {e}"))?;
    let warm_d = synth_srgb(w, h, 0x0B22_9999);
    let _ = m
        .compute_with_warm_ref_srgb(&warm_d, None)
        .map_err(|e| format!("warmup cached score: {e}"))?;
    cubecl::future::block_on(client.sync()).map_err(|e| format!("sync: {e}"))?;

    let mut times = Vec::with_capacity(reps);
    let mut score = f64::NAN;
    for i in 0..reps {
        let d = synth_srgb(w, h, 0x0B22_0100u32.wrapping_add(i as u32 * 13 + 1));
        let t = Instant::now();
        let s = m
            .compute_with_warm_ref_srgb(&d, None)
            .map_err(|e| format!("cached score: {e}"))?;
        cubecl::future::block_on(client.sync()).map_err(|e| format!("sync: {e}"))?;
        times.push(t.elapsed().as_secs_f64() * 1e3);
        score = s.value;
    }
    Ok((times, score))
}

fn run_worker(metric_tag: &str, mode: &str, ctx: &str, w: u32, h: u32, reps: usize) {
    let kind = metric_kind_from_tag(metric_tag).expect("metric tag");
    let client = CudaRuntime::client(&Default::default());

    let (times, score): (Vec<f64>, f64) = match (mode, ctx) {
        ("strippair", "oneoff") => {
            #[cfg(feature = "cvvdp")]
            {
                time_oneoff_cvvdp_strippair(w, h, &client, reps)
            }
            #[cfg(not(feature = "cvvdp"))]
            {
                eprintln!("strippair requires cvvdp feature");
                std::process::exit(3);
            }
        }
        ("strippair", "warm") => {
            #[cfg(feature = "cvvdp")]
            {
                match time_warm_cvvdp_strippair(w, h, &client, reps) {
                    Ok(v) => v,
                    Err(e) => {
                        // Print a structured FAIL line the parent records,
                        // then exit cleanly so the sweep continues.
                        println!("RESULT_FAIL {e}");
                        std::io::stdout().flush().ok();
                        return;
                    }
                }
            }
            #[cfg(not(feature = "cvvdp"))]
            {
                eprintln!("strippair requires cvvdp feature");
                std::process::exit(3);
            }
        }
        (m, "oneoff") => {
            let mm = umbrella_mode(m).expect("umbrella mode");
            time_oneoff_umbrella(kind, mm, w, h, &client, reps)
        }
        (m, "warm") => {
            let mm = umbrella_mode(m).expect("umbrella mode");
            time_warm_umbrella(kind, mm, w, h, &client, reps)
        }
        _ => {
            eprintln!("unknown context {ctx}");
            std::process::exit(4);
        }
    };

    let mut sorted = times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p25 = percentile(&sorted, 25.0);
    let p50 = percentile(&sorted, 50.0);
    let p75 = percentile(&sorted, 75.0);

    // RESULT <p25> <p50> <p75> <score> <n>
    println!("RESULT {p25:.5} {p50:.5} {p75:.5} {score:.6} {}", times.len());
    std::io::stdout().flush().expect("flush");

    // Hold so the parent samples nvidia-smi at quiescent steady state.
    std::thread::sleep(Duration::from_millis(CHILD_HOLD_MS));
}

// --------------------------------------------------------------------
// Parent: enumerate the grid, spawn one child per cell, sample VRAM.
// --------------------------------------------------------------------

struct CellResult {
    p25: f64,
    p50: f64,
    p75: f64,
    score: f64,
    n: usize,
    peak_vram_mib: i64,
    free_vram_start_mib: u64,
    fail: Option<String>,
}

fn measure_cell(
    child_bin: &str,
    metric: &str,
    mode: &str,
    ctx: &str,
    w: u32,
    h: u32,
    reps: usize,
    sample_interval: Duration,
) -> CellResult {
    // Let the previous cell's pool decommit; sample baseline + free.
    std::thread::sleep(Duration::from_millis(350));
    let baseline_used = nvidia_smi_query("memory.used").unwrap_or(0);
    let free_start = nvidia_smi_query("memory.free").unwrap_or(0);

    let mut child = Command::new(child_bin)
        .env("MW_WORKER", "1")
        .env("MW_METRIC", metric)
        .env("MW_MODE", mode)
        .env("MW_CTX", ctx)
        .env("MW_W", w.to_string())
        .env("MW_H", h.to_string())
        .env("MW_REPS", reps.to_string())
        // Inherit CUDA env from parent.
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn child");

    let stdout = child.stdout.take().expect("child stdout");
    let mut reader = BufReader::new(stdout);
    // Sample VRAM continuously while we wait for RESULT.
    let mut peak_used = baseline_used;
    let mut result_line: Option<String> = None;
    let deadline = Instant::now() + Duration::from_secs(600);

    // Read RESULT line (blocking read, but sample VRAM between lines by
    // using a short loop with a non-blocking-ish strategy: we read line by
    // line; the worker prints RESULT only once at the end, so during the
    // timed reps stdout is silent. To sample during that silent window we
    // spawn a sampler thread.)
    let peak_handle = {
        let interval = sample_interval;
        std::thread::spawn(move || {
            let mut local_peak = baseline_used;
            let start = Instant::now();
            // Sample for up to 600 s; the main thread joins after the
            // child exits, so this thread ends when the child does (we
            // signal via a channel-free deadline + the parent dropping).
            // We cap by checking an Arc<AtomicBool> would be cleaner, but
            // a wall-clock cap + the parent's join after child.wait keeps
            // it simple. Use a generous cap; the parent interrupts by
            // process exit ordering (we join below right after wait).
            while start.elapsed() < Duration::from_secs(600) {
                if let Some(v) = nvidia_smi_query("memory.used") {
                    if v > local_peak {
                        local_peak = v;
                    }
                }
                std::thread::sleep(interval);
                // Stop sampling once the STOP file marker is observed.
                if SAMPLER_STOP.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
            }
            local_peak
        })
    };

    let mut line = String::new();
    while Instant::now() < deadline {
        line.clear();
        let nread = reader.read_line(&mut line).expect("read child");
        if nread == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.starts_with("RESULT") {
            result_line = Some(trimmed.to_string());
            break;
        }
    }

    // Give the sampler a beat to catch the post-RESULT hold-window peak,
    // then signal it to stop and collect.
    std::thread::sleep(Duration::from_millis(CHILD_HOLD_MS + 50));
    SAMPLER_STOP.store(true, std::sync::atomic::Ordering::Relaxed);
    if let Ok(p) = peak_handle.join() {
        peak_used = peak_used.max(p);
    }
    SAMPLER_STOP.store(false, std::sync::atomic::Ordering::Relaxed);

    let out = child.wait_with_output().expect("wait child");
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();

    let peak_vram_mib = (peak_used as i64) - (baseline_used as i64);

    match result_line {
        Some(l) if l.starts_with("RESULT_FAIL") => {
            let msg = l.strip_prefix("RESULT_FAIL ").unwrap_or("unknown").to_string();
            CellResult {
                p25: f64::NAN, p50: f64::NAN, p75: f64::NAN, score: f64::NAN,
                n: 0, peak_vram_mib, free_vram_start_mib: free_start,
                fail: Some(msg),
            }
        }
        Some(l) => {
            // RESULT <p25> <p50> <p75> <score> <n>
            let parts: Vec<&str> = l.split_whitespace().collect();
            let p25 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(f64::NAN);
            let p50 = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(f64::NAN);
            let p75 = parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(f64::NAN);
            let score = parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(f64::NAN);
            let n = parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);
            if !out.status.success() {
                CellResult {
                    p25, p50, p75, score, n, peak_vram_mib,
                    free_vram_start_mib: free_start,
                    fail: Some(format!("child-nonzero-exit: {}", stderr.lines().last().unwrap_or(""))),
                }
            } else {
                CellResult {
                    p25, p50, p75, score, n, peak_vram_mib,
                    free_vram_start_mib: free_start, fail: None,
                }
            }
        }
        None => CellResult {
            p25: f64::NAN, p50: f64::NAN, p75: f64::NAN, score: f64::NAN,
            n: 0, peak_vram_mib, free_vram_start_mib: free_start,
            fail: Some(format!(
                "no-RESULT-line; exit={:?}; stderr-tail={}",
                out.status, stderr.lines().rev().take(3).collect::<Vec<_>>().join(" | ")
            )),
        },
    }
}

static SAMPLER_STOP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn note_for(metric: &str, mode: &str, ctx: &str, free_start_mib: u64, peak_mib: i64) -> String {
    let mut notes = Vec::new();
    // cvvdp one-off routes through Full for ALL modes (Strip/StripPair
    // are cached-ref / strip-walker paths; one-off score() uses Full).
    if metric == "cvvdp" && ctx == "oneoff" && mode != "full" {
        notes.push("cvvdp_oneoff_routes_full");
    }
    // StripPair has no cached-ref of its own; warm falls back to Mode E.
    if metric == "cvvdp" && mode == "strippair" && ctx == "warm" {
        notes.push("strippair_warm_falls_back_mode_e");
    }
    // Contention flag: if free VRAM at start was low or the peak is a
    // big fraction of free, another session may be sharing the GPU and
    // the VRAM number is suspect (timing still valid — synced+isolated).
    if free_start_mib < 4096 {
        notes.push("low_free_vram_at_start");
    }
    let _ = peak_mib;
    if notes.is_empty() {
        "ok".to_string()
    } else {
        notes.join(";")
    }
}

fn main() {
    // Worker entry.
    if env::var("MW_WORKER").is_ok() {
        let metric = env::var("MW_METRIC").unwrap();
        let mode = env::var("MW_MODE").unwrap();
        let ctx = env::var("MW_CTX").unwrap();
        let w: u32 = env::var("MW_W").unwrap().parse().unwrap();
        let h: u32 = env::var("MW_H").unwrap().parse().unwrap();
        let reps: usize = env::var("MW_REPS").unwrap().parse().unwrap();
        run_worker(&metric, &mode, &ctx, w, h, reps);
        return;
    }

    // Parent entry — enumerate the grid.
    let reps = parse_u32("MW_REPS", 20).max(1) as usize;
    let sizes: Vec<u32> = env::var("MW_SIZES")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![256, 512, 1024, 4096]);
    let metric_filter: Option<Vec<String>> = env::var("MW_METRICS")
        .ok()
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect());
    let sample_interval = Duration::from_millis(parse_u32("MW_SAMPLE_MS", 10) as u64);

    let child_bin: String = std::env::current_exe()
        .expect("current_exe")
        .to_string_lossy()
        .into_owned();

    // CSV header.
    println!(
        "metric,mode,context,size,w,h,wall_ms_p25,wall_ms_p50,wall_ms_p75,score,n,peak_vram_mib,free_vram_start_mib,note"
    );
    std::io::stdout().flush().ok();

    eprintln!(
        "# mode_wall: reps={reps} sizes={sizes:?} strip_h_body={STRIP_H_BODY} sample_ms={}ms subprocess-per-cell",
        sample_interval.as_millis()
    );

    for &(metric_tag, _kind) in ALL_METRICS {
        if let Some(ref filt) = metric_filter {
            if !filt.iter().any(|m| m == metric_tag) {
                continue;
            }
        }
        for &mode in modes_for(metric_tag) {
            for &ctx in CONTEXTS {
                for &size in &sizes {
                    let (w, h) = (size, size);
                    if let Some(reason) = unsupported_reason(metric_tag, mode, ctx) {
                        println!(
                            "{metric_tag},{mode},{ctx},{size},{w},{h},NaN,NaN,NaN,NaN,0,0,0,SKIP:{reason}"
                        );
                        std::io::stdout().flush().ok();
                        eprintln!("# cell {metric_tag}/{mode}/{ctx}/{size}² ... SKIP:{reason}");
                        continue;
                    }
                    eprint!("# cell {metric_tag}/{mode}/{ctx}/{size}² ... ");
                    std::io::stderr().flush().ok();
                    let r = measure_cell(
                        &child_bin, metric_tag, mode, ctx, w, h, reps, sample_interval,
                    );
                    let note = match &r.fail {
                        Some(msg) => format!("FAIL:{}", msg.replace(',', ";")),
                        None => note_for(metric_tag, mode, ctx, r.free_vram_start_mib, r.peak_vram_mib),
                    };
                    println!(
                        "{metric_tag},{mode},{ctx},{size},{w},{h},{:.5},{:.5},{:.5},{:.6},{},{},{},{}",
                        r.p25, r.p50, r.p75, r.score, r.n, r.peak_vram_mib, r.free_vram_start_mib, note
                    );
                    std::io::stdout().flush().ok();
                    eprintln!(
                        "p50={:.3}ms peak={}MiB free_start={}MiB {}",
                        r.p50, r.peak_vram_mib, r.free_vram_start_mib, note
                    );
                }
            }
        }
    }

    eprintln!("# done");
}
