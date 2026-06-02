//! inprocess_warmth — MEASURE the in-process GPU warmth transitions a
//! single long-lived warm worker pays (task #144).
//!
//! The orchestrator runs mixed metrics through ONE long-lived warm
//! worker (single-warm-instance pool), so these transitions are
//! real-deployment-relevant. Prior statements about them (e.g. "a
//! second metric in a warm process pays the ~181 ms context init
//! again", "a new warm_ref reference is free") were INFERRED from
//! architecture, never measured. This driver replaces the guesses with
//! committed numbers.
//!
//! ## The four questions
//!
//! - **Q1 — cross-metric context sharing.** In ONE process: time
//!   `CudaRuntime::client()` (context init), then metric A's first
//!   score, then metric B's first score reusing the same process-global
//!   client. Does B skip the ~181 ms context init? Compare B's
//!   same-process-first to B's FRESH-process `cold_total` (#140).
//! - **Q2 — per-metric kernel warmth.** Same process: A.first /
//!   A.warm×5 / B.first / B.warm×5. Confirms kernels are per-metric
//!   while context is shared.
//! - **Q3 — new reference in warm_ref mode.** On a warmed metric:
//!   `set_reference(ref1)` → time it (sync'd); warm calls ×5; then
//!   `set_reference(ref2)` with DIFFERENT pixel content → time it. Is a
//!   new ref ≈ the first ref (re-pays precompute) or ≈ free?
//! - **Q4 — full-mode, different ref every call.** Same warm instance:
//!   `score(refA,distA)`, `score(refB,distB)`, `score(refC,distC)` with
//!   different reference pixels each call. Does changing the reference
//!   cost anything beyond normal per-call work?
//!
//! ## Correctness
//!
//! Every timed *score* call ends in a host readback inside the opaque
//! `compute_*` (it returns a scalar), forcing a GPU sync — the wall is
//! real execution, not async submission (see project CLAUDE.md
//! "Diagnosing Slow GPU Code"). `set_reference_srgb_u8` does NOT read
//! back, so every timed `set_reference` is followed by
//! `block_on(client.sync())` to flush the queue before the timer stops.
//! The driver creates the cubecl client explicitly first; the umbrella
//! `Metric::new` reuses that same process-global client (cubecl caches
//! per device), so the driver's `sync()` flushes the same queue the
//! metric submits to — empirically confirmed by a nonzero, stable
//! `setref` cost.
//!
//! Cold = a FRESH process (new CUDA context). Each scenario/ordering is
//! a separate child process. Driver mode spawns the children and
//! collects their `RESULT` lines.
//!
//! Build (release, cuda, NO target-cpu=native):
//! ```sh
//! cargo build --release -p zenmetrics-api \
//!   --no-default-features --features cuda,all-metrics,cubecl-types,pixels \
//!   --example inprocess_warmth
//! ```
//!
//! Child mode (one process, one scenario):
//! ```sh
//! WARMTH_W=512 WARMTH_H=512 WARMTH_REPS=5 \
//!   inprocess_warmth --child <scenario> <metric_a> <metric_b>
//! ```
//! where scenario ∈ {q1q2, q3, q4}. (q1 and q2 share one process — the
//! A→B sequence — so they are emitted from the same `q1q2` child.)
//!
//! Output: one `RESULT\t<scenario>\t<metric_a>\t<metric_b>\t<phase>\t<ms>\t<n>\t<notes>`
//! line per measured phase.

#![cfg(feature = "cubecl-types")]

use std::env;
use std::time::Instant;

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use zenmetrics_api::{Backend, Metric, MetricKind, MetricParams};

// ===================================================================
// Helpers
// ===================================================================

fn parse_u32(name: &str, default: u32) -> u32 {
    env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn median(mut t: Vec<f64>) -> f64 {
    if t.is_empty() {
        return f64::NAN;
    }
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = t.len();
    if n % 2 == 0 {
        (t[n / 2 - 1] + t[n / 2]) / 2.0
    } else {
        t[n / 2]
    }
}

/// Deterministic XorShift64 image. Distinct `seed` ⇒ distinct pixel
/// content — used so ref1 ≠ ref2 ≠ ref3 are NOT identical inputs (no
/// accidental "same-input" GPU shortcut). Same generator shape as
/// `mem_per_metric.rs::make_image`.
fn make_image(seed: u64, w: u32, h: u32) -> Vec<u8> {
    let mut state = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x1234_5678);
    let n = (w as usize) * (h as usize) * 3;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.push((state & 0xFF) as u8);
    }
    out
}

fn metric_kind_from_tag(s: &str) -> Option<MetricKind> {
    match s {
        "butter" => Some(MetricKind::Butter),
        "ssim2" => Some(MetricKind::Ssim2),
        "dssim" => Some(MetricKind::Dssim),
        "iwssim" => Some(MetricKind::Iwssim),
        "cvvdp" => Some(MetricKind::Cvvdp),
        "zensim" => Some(MetricKind::Zensim),
        _ => None,
    }
}

fn build_metric(kind: MetricKind, w: u32, h: u32) -> Metric {
    let params: MetricParams = MetricParams::try_default_for(kind)
        .unwrap_or_else(|e| panic!("params for {}: {e}", kind.tag()));
    Metric::new(kind, Backend::Cuda, w, h, params)
        .unwrap_or_else(|e| panic!("Metric::new {}: {e}", kind.tag()))
}

/// Emit one structured result line. The Python harness parses these.
fn emit(scenario: &str, a: &str, b: &str, phase: &str, ms: f64, n: usize, notes: &str) {
    println!("RESULT\t{scenario}\t{a}\t{b}\t{phase}\t{ms:.4}\t{n}\t{notes}");
}

// ===================================================================
// Q1 + Q2 — one process, A then B (cross-metric context sharing +
// per-metric kernel warmth). The A→B sequence is WITHIN one process —
// that is the whole point: B is measured with A's context already warm.
// ===================================================================

fn run_q1q2(metric_a: &str, metric_b: &str, w: u32, h: u32, reps: usize) {
    let kind_a = metric_kind_from_tag(metric_a).expect("metric_a");
    let kind_b = metric_kind_from_tag(metric_b).expect("metric_b");

    // Distinct images so neither side hits an identical-input shortcut.
    let r = make_image(0xA5A5, w, h);
    let d = make_image(0x5A5A, w, h);

    // --- Phase: client_init (cold context) ---
    // This explicit client() call IS the CUDA context init. The umbrella
    // Metric::new below calls R::client() internally and gets THIS same
    // cached client (cubecl caches per device), so the context init is
    // paid exactly once here. We keep the handle to drain A's queue
    // before building B (see drop(a) below).
    let t = Instant::now();
    let client = CudaRuntime::client(&Default::default());
    let client_ms = t.elapsed().as_secs_f64() * 1e3;
    emit(
        "Q1_crossmetric",
        metric_a,
        metric_b,
        "client_init",
        client_ms,
        1,
        "cold_context",
    );

    // --- A_first: build A + first score (context warm, A kernels cold) ---
    let t = Instant::now();
    let mut a = build_metric(kind_a, w, h);
    let a_new_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let sa = a.compute_srgb_u8(&r, &d).expect("A first compute");
    let a_first_compute_ms = t.elapsed().as_secs_f64() * 1e3;
    let a_first_ms = a_new_ms + a_first_compute_ms;
    emit(
        "Q2_kernelwarm",
        metric_a,
        "-",
        "A_new",
        a_new_ms,
        1,
        "ctor_alloc",
    );
    emit(
        "Q2_kernelwarm",
        metric_a,
        "-",
        "A_first_compute",
        a_first_compute_ms,
        1,
        "kernel_jit+first_upload+compute",
    );
    emit(
        "Q2_kernelwarm",
        metric_a,
        "-",
        "A_first",
        a_first_ms,
        1,
        &format!("new+first_compute score={:.4}", sa.value),
    );

    // --- A_warm ×reps ---
    let mut a_warm = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        let _ = a.compute_srgb_u8(&r, &d).expect("A warm compute");
        a_warm.push(t.elapsed().as_secs_f64() * 1e3);
    }
    emit(
        "Q2_kernelwarm",
        metric_a,
        "-",
        "A_warm",
        median(a_warm.clone()),
        a_warm.len(),
        &format!(
            "all={}",
            a_warm
                .iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(",")
        ),
    );

    // Drop A's GPU working set before building B. This frees A's buffers
    // back to the cubecl pool but KEEPS the CUDA context alive (the
    // context is process-global, not owned by `a`). Two reasons:
    //  1. Correctness/realism: the real warm worker processes one metric
    //     at a time — it does NOT hold two metrics' full working sets
    //     simultaneously. At 16 MP, cvvdp(~3.4 GB)+ssim2(~5.7 GB) held
    //     together would exceed the free VRAM on a 12 GiB card shared
    //     with the WSL2/Windows desktop.
    //  2. The sync drains A's queue so B's first-call timer is not
    //     polluted by A's still-in-flight work.
    drop(a);
    cubecl::future::block_on(client.sync()).expect("sync after dropping A");

    // --- B_first: build B + first score. Context is WARM (A ran; context
    // survived the drop). A's kernels are loaded but B's are NOT — so this
    // measures B's alloc + B's kernel JIT only, NOT context init. Compare
    // to B's fresh-process cold_total from #140 to quantify the
    // context-sharing saving. ---
    let t = Instant::now();
    let mut bm = build_metric(kind_b, w, h);
    let b_new_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let sb = bm.compute_srgb_u8(&r, &d).expect("B first compute");
    let b_first_compute_ms = t.elapsed().as_secs_f64() * 1e3;
    let b_first_ms = b_new_ms + b_first_compute_ms;
    // Q1 row: B's same-process-first cost (the headline number).
    emit(
        "Q1_crossmetric",
        metric_a,
        metric_b,
        "B_first_same_process",
        b_first_ms,
        1,
        &format!(
            "warm_context_cold_B_kernels new={b_new_ms:.3} compute={b_first_compute_ms:.3} score={:.4}",
            sb.value
        ),
    );
    emit(
        "Q2_kernelwarm",
        "-",
        metric_b,
        "B_new",
        b_new_ms,
        1,
        "ctor_alloc warm_context",
    );
    emit(
        "Q2_kernelwarm",
        "-",
        metric_b,
        "B_first_compute",
        b_first_compute_ms,
        1,
        "kernel_jit+first_upload+compute",
    );
    emit(
        "Q2_kernelwarm",
        "-",
        metric_b,
        "B_first",
        b_first_ms,
        1,
        &format!("new+first_compute score={:.4}", sb.value),
    );

    // --- B_warm ×reps ---
    let mut b_warm = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        let _ = bm.compute_srgb_u8(&r, &d).expect("B warm compute");
        b_warm.push(t.elapsed().as_secs_f64() * 1e3);
    }
    emit(
        "Q2_kernelwarm",
        "-",
        metric_b,
        "B_warm",
        median(b_warm.clone()),
        b_warm.len(),
        &format!(
            "all={}",
            b_warm
                .iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(",")
        ),
    );
}

// ===================================================================
// Q3 — new reference in warm_ref mode. On a metric warmed (context +
// kernels hot via a throwaway full-mode score), measure:
//   setref1  -> set_reference(ref1)            [sync'd]
//   warm_call (×reps) -> compute_with_cached_reference(dist)
//   setref2  -> set_reference(ref2 ≠ ref1)     [sync'd]
//   newref_call (×reps) -> compute_with_cached_reference(dist)
// Question: setref2 ≈ setref1 (each new ref re-pays precompute) or
// ≈ free (machine-wide cache)?  ref1 and ref2 are DIFFERENT pixels.
// ===================================================================

fn run_q3(metric: &str, w: u32, h: u32, reps: usize) {
    let kind = metric_kind_from_tag(metric).expect("metric");

    let ref1 = make_image(0x1111, w, h);
    let ref2 = make_image(0x2222, w, h); // DIFFERENT pixel content
    let dist = make_image(0x3333, w, h);

    // Build client explicitly so we can force a sync after set_reference.
    let client = CudaRuntime::client(&Default::default());
    let mut m = build_metric(kind, w, h);

    // Warm the metric: a throwaway full-mode score loads all kernels +
    // populates the cubecl pool. After this, context + kernels are hot,
    // so the set_reference timings below isolate the REFERENCE precompute
    // cost, not first-run JIT.
    let _ = m.compute_srgb_u8(&ref1, &dist).expect("warmup full score");
    cubecl::future::block_on(client.sync()).expect("sync after warmup");

    // --- setref1: first set_reference on the warm instance ---
    let t = Instant::now();
    m.set_reference_srgb_u8(&ref1).expect("set_reference(ref1)");
    cubecl::future::block_on(client.sync()).expect("sync after setref1");
    let setref1_ms = t.elapsed().as_secs_f64() * 1e3;
    emit(
        "Q3_newref",
        metric,
        "-",
        "setref1",
        setref1_ms,
        1,
        "first_ref_warm_instance sync'd",
    );

    // --- warm_call ×reps against ref1 ---
    let mut warm1 = Vec::with_capacity(reps);
    let mut score1 = 0.0_f64;
    for _ in 0..reps {
        let t = Instant::now();
        let s = m
            .compute_with_reference_srgb_u8(&dist)
            .expect("warm_call ref1");
        warm1.push(t.elapsed().as_secs_f64() * 1e3);
        score1 = s.value;
    }
    emit(
        "Q3_newref",
        metric,
        "-",
        "warm_call",
        median(warm1.clone()),
        warm1.len(),
        &format!(
            "ref1 score={score1:.4} all={}",
            warm1
                .iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(",")
        ),
    );

    // --- setref2: NEW reference (different pixels) on same warm instance.
    // This is THE measured guess to kill: is it ≈ setref1 or ≈ free? ---
    let t = Instant::now();
    m.set_reference_srgb_u8(&ref2).expect("set_reference(ref2)");
    cubecl::future::block_on(client.sync()).expect("sync after setref2");
    let setref2_ms = t.elapsed().as_secs_f64() * 1e3;
    emit(
        "Q3_newref",
        metric,
        "-",
        "setref2",
        setref2_ms,
        1,
        "NEW_ref_different_pixels sync'd",
    );

    // --- newref_call ×reps against ref2 ---
    let mut warm2 = Vec::with_capacity(reps);
    let mut score2 = 0.0_f64;
    for _ in 0..reps {
        let t = Instant::now();
        let s = m
            .compute_with_reference_srgb_u8(&dist)
            .expect("newref_call ref2");
        warm2.push(t.elapsed().as_secs_f64() * 1e3);
        score2 = s.value;
    }
    emit(
        "Q3_newref",
        metric,
        "-",
        "newref_call",
        median(warm2.clone()),
        warm2.len(),
        &format!(
            "ref2 score={score2:.4} all={}",
            warm2
                .iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(",")
        ),
    );
}

// ===================================================================
// Q4 — full-mode, different ref every call. Same warm instance:
// score(refA,distA), score(refB,distB), score(refC,distC) with
// DIFFERENT reference pixels each call. Full mode rebuilds the ref
// side every call anyway — does changing the reference cost anything
// beyond the normal per-call work? Measure the per-call wall when the
// reference is NEVER the same vs. the warm-per-call (same pair) from
// the matching Q2 row.
// ===================================================================

fn run_q4(metric: &str, w: u32, h: u32, reps: usize) {
    let kind = metric_kind_from_tag(metric).expect("metric");

    let client = CudaRuntime::client(&Default::default());
    let mut m = build_metric(kind, w, h);

    // Warm: throwaway score so context + kernels + pool are hot.
    let r0 = make_image(0x9001, w, h);
    let d0 = make_image(0x9002, w, h);
    let _ = m.compute_srgb_u8(&r0, &d0).expect("warmup full score");
    cubecl::future::block_on(client.sync()).expect("sync after warmup");

    // Baseline: warm full-mode per-call with the SAME (r,d) pair repeated
    // — this is the per-call work when the ref does NOT change.
    let mut same_pair = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        let _ = m.compute_srgb_u8(&r0, &d0).expect("same-pair full score");
        same_pair.push(t.elapsed().as_secs_f64() * 1e3);
    }
    emit(
        "Q4_fullmode_newref",
        metric,
        "-",
        "fullmode_same_ref",
        median(same_pair.clone()),
        same_pair.len(),
        &format!(
            "repeated_same_pair all={}",
            same_pair
                .iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(",")
        ),
    );

    // DIFFERENT reference (and distorted) pixels every call. Pre-generate
    // all pairs so image generation is OUT of the timed region.
    let pairs: Vec<(Vec<u8>, Vec<u8>)> = (0..reps)
        .map(|i| {
            let ri = make_image(0xB000 + i as u64 * 7 + 1, w, h);
            let di = make_image(0xC000 + i as u64 * 11 + 1, w, h);
            (ri, di)
        })
        .collect();
    let mut diff_ref = Vec::with_capacity(reps);
    let mut last_score = 0.0_f64;
    for (ri, di) in &pairs {
        let t = Instant::now();
        let s = m.compute_srgb_u8(ri, di).expect("diff-ref full score");
        diff_ref.push(t.elapsed().as_secs_f64() * 1e3);
        last_score = s.value;
    }
    emit(
        "Q4_fullmode_newref",
        metric,
        "-",
        "fullmode_diff_ref",
        median(diff_ref.clone()),
        diff_ref.len(),
        &format!(
            "different_ref_each_call last_score={last_score:.4} all={}",
            diff_ref
                .iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(",")
        ),
    );
}

// ===================================================================
// main — child dispatch
// ===================================================================

fn main() {
    let args: Vec<String> = env::args().collect();
    let w = parse_u32("WARMTH_W", 512);
    let h = parse_u32("WARMTH_H", 512);
    let reps = parse_u32("WARMTH_REPS", 5) as usize;

    if args.len() >= 3 && args[1] == "--child" {
        let scenario = args[2].as_str();
        match scenario {
            "q1q2" => {
                let a = args.get(3).map(String::as_str).expect("metric_a");
                let b = args.get(4).map(String::as_str).expect("metric_b");
                run_q1q2(a, b, w, h, reps);
            }
            "q3" => {
                let m = args.get(3).map(String::as_str).expect("metric");
                run_q3(m, w, h, reps);
            }
            "q4" => {
                let m = args.get(3).map(String::as_str).expect("metric");
                run_q4(m, w, h, reps);
            }
            other => {
                eprintln!("unknown scenario {other}");
                std::process::exit(2);
            }
        }
        return;
    }

    eprintln!(
        "inprocess_warmth: run a single scenario in a fresh process.\n\
         Usage: inprocess_warmth --child <q1q2|q3|q4> <metric_a> [metric_b]\n\
         Env: WARMTH_W WARMTH_H WARMTH_REPS\n\
         The Python harness (scripts/.../sweep_gpu_inprocess_warmth_*.py) \
         orchestrates the full matrix."
    );
    std::process::exit(1);
}
