//! setref_all_timing — clean re-measure of per-reference `set_reference`
//! cost for ALL six umbrella metrics on a WARM instance (task #151).
//!
//! ## Why this exists
//!
//! Task #144 measured (via `inprocess_warmth.rs` scenario Q3, n=1) that
//! the FIRST `set_reference` on a warm metric was much more expensive than
//! a SUBSEQUENT new reference — most dramatically for iwssim, whose 16 MP
//! `setref1` read 196.5 ms vs 67.4 ms for `setref2` (a 3× "first-ref
//! warmup"). Those #144 numbers were single-sample (`n=1`) on a machine
//! where a concurrent zensim eval contaminated some cells. Task #148
//! re-measured the SAME quantity for *butter only* with proper repeated
//! sampling and found butter's per-ref cost is FLAT (setref1 ≈ setref2 ≈
//! …) — the #144 "expensive first ref" was the n=1 contamination, not a
//! real first-ref penalty.
//!
//! This driver does for the OTHER FIVE metrics (cvvdp, ssim2, dssim,
//! zensim, iwssim) — and re-confirms butter for a single consistent table
//! — what #148 did for butter: on a quiet machine, on a fully warm
//! instance, time `set_reference(refN)` for N = 1..K with DISTINCT pixel
//! content each, n≥8 samples per phase, reporting median + min + raw so a
//! single transient outlier can't lie. The headline question is whether
//! iwssim's 196-vs-67 ms 16 MP gap reproduces cleanly (real per-ref scratch
//! alloc on first ref, reused after) or collapses to flat (contamination,
//! same as butter).
//!
//! ## What is measured
//!
//! On a single warm `Metric` instance of the requested kind (context +
//! kernels hot via a throwaway full-mode `compute_srgb_u8`, then a sync):
//!
//! - `setref1` — `set_reference_srgb_u8(ref)`, the FIRST set_reference on
//!   the warm instance. Repeated ≥`REPS` times (a fresh-pixel ref each rep
//!   so no identical-input shortcut), each followed by a full sync, each
//!   individually timed.
//! - `setref2..setrefK` — `set_reference_srgb_u8(refN)` with DISTINCT pixel
//!   content, the reuse path (the #144 headline). Each phase is ≥`REPS`
//!   distinct refs, individually timed + synced. ≥3 distinct new-ref phases
//!   (setref2/3/4) confirm any first-ref penalty is one-off, not per-ref.
//! - `warm_call` — `compute_with_reference_srgb_u8(dist)` warm score
//!   against the last cached reference, ≥`REPS` reps. Sanity check that the
//!   warm-call wall is the per-score cost, separate from the per-ref
//!   precompute. Self-syncs via the host readback inside the call.
//!
//! ## Correctness — synced timers, not async submission
//!
//! `set_reference_srgb_u8` queues GPU work (upload, ref-side precompute)
//! and does NOT read back to the host, so `Instant::elapsed()` around it
//! alone would measure SUBMISSION, not execution (see project CLAUDE.md
//! "Diagnosing Slow GPU Code"). Every timed `set_reference` is therefore
//! immediately followed by `block_on(client.sync())` INSIDE the timed
//! region, so the wall is the real GPU precompute. The driver builds the
//! cubecl client explicitly first; `Metric::new` reuses that same
//! process-global client (cubecl caches per device), so the driver's
//! `sync()` flushes the same queue the metric submits to.
//!
//! Each rep within a phase uses a DISTINCT-pixel reference (seed varies),
//! so no two consecutive `set_reference` calls feed identical bytes — the
//! kernel cannot shortcut on input equality, and we exercise the true
//! upload-and-precompute path every rep.
//!
//! ## Build (release, cuda, NO target-cpu=native)
//! ```sh
//! cargo build --release -p zenmetrics-api \
//!   --no-default-features --features cuda,all-metrics,cubecl-types,pixels \
//!   --example setref_all_timing
//! ```
//!
//! ## Run (one metric per process — keeps VRAM bounded)
//! ```sh
//! SETREF_W=4096 SETREF_H=4096 SETREF_REPS=8 SETREF_NEWREF_PHASES=3 \
//!   setref_all_timing iwssim
//! ```
//! where the metric tag ∈ {butter, cvvdp, ssim2, dssim, iwssim, zensim}.
//!
//! Output: TSV rows on stdout, header first (unless SETREF_NOHEADER=1).
//! Columns:
//!   metric  size_mp  w  h  phase  ms_median  ms_min  ms_max  n  raw_samples  notes

#![cfg(feature = "cubecl-types")]

use std::env;
use std::time::Instant;

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::ComputeClient;
use zenmetrics_api::{Backend, Metric, MetricKind, MetricParams};

type Rt = CudaRuntime;

fn parse_u32(name: &str, default: u32) -> u32 {
    env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn median(t: &[f64]) -> f64 {
    if t.is_empty() {
        return f64::NAN;
    }
    let mut v = t.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n.is_multiple_of(2) {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    } else {
        v[n / 2]
    }
}

fn min_of(t: &[f64]) -> f64 {
    t.iter().cloned().fold(f64::INFINITY, f64::min)
}
fn max_of(t: &[f64]) -> f64 {
    t.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

/// Deterministic LCG-filled sRGB bytes, seed-controlled so each call can
/// produce DISTINCT pixel content (so consecutive set_reference calls never
/// feed identical bytes). Same generator shape as
/// butteraugli-gpu/examples/setref_timing.rs::synth_srgb so the butter
/// confirm row is apples-to-apples.
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

/// Time `set_reference_srgb_u8` `reps` times, a DISTINCT-pixel reference
/// per rep, each call immediately followed by a full sync inside the timed
/// region. `seed_base` distinguishes phases so no two phases share ref
/// bytes.
fn time_setref_phase(
    m: &mut Metric,
    client: &ComputeClient<Rt>,
    w: u32,
    h: u32,
    seed_base: u32,
    reps: usize,
) -> Vec<f64> {
    // Pre-generate refs so host-side image gen is OUT of the timed region.
    let refs: Vec<Vec<u8>> = (0..reps)
        .map(|i| synth_srgb(w, h, seed_base.wrapping_add(i as u32 * 31 + 1)))
        .collect();
    let mut times = Vec::with_capacity(reps);
    for r in &refs {
        let t = Instant::now();
        m.set_reference_srgb_u8(r).expect("set_reference_srgb_u8");
        cubecl::future::block_on(client.sync()).expect("sync after set_reference");
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times
}

fn emit(metric: &str, w: u32, h: u32, phase: &str, samples: &[f64], notes: &str) {
    let mp = (w as f64 * h as f64) / 1_000_000.0;
    let raw = samples
        .iter()
        .map(|v| format!("{v:.4}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "{metric}\t{mp:.3}\t{w}\t{h}\t{phase}\t{med:.4}\t{min:.4}\t{max:.4}\t{n}\t{raw}\t{notes}",
        med = median(samples),
        min = min_of(samples),
        max = max_of(samples),
        n = samples.len(),
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let metric_tag = args.get(1).cloned().unwrap_or_else(|| {
        eprintln!(
            "setref_all_timing: measure per-reference set_reference cost for one metric.\n\
             Usage: setref_all_timing <butter|cvvdp|ssim2|dssim|iwssim|zensim>\n\
             Env: SETREF_W SETREF_H SETREF_REPS SETREF_NEWREF_PHASES SETREF_NOHEADER"
        );
        std::process::exit(1);
    });
    let kind = metric_kind_from_tag(&metric_tag).unwrap_or_else(|| {
        eprintln!("unknown metric tag '{metric_tag}'");
        std::process::exit(2);
    });

    let w = parse_u32("SETREF_W", 512);
    let h = parse_u32("SETREF_H", 512);
    let reps = parse_u32("SETREF_REPS", 8).max(1) as usize;
    // Number of DISTINCT new-reference phases AFTER setref1 (setref2,
    // setref3, ...). Default 3 so the task's "≥3 distinct new refs" gate is
    // met (setref2/3/4).
    let newref_phases = parse_u32("SETREF_NEWREF_PHASES", 3).max(1) as usize;
    let no_header = parse_u32("SETREF_NOHEADER", 0) != 0;

    // CUDA context init (this is the cold context; everything below runs
    // warm against it). Build the client explicitly so we can sync.
    let client = CudaRuntime::client(&Default::default());

    // Full-mode instance (matches the #144 / #148 warm-instance path).
    let mut m = build_metric(kind, w, h);

    // Warm the instance: a throwaway full-mode compute loads ALL kernels
    // (JIT) and populates the cubecl memory pool, then sync to flush. After
    // this, context + kernels + pool are hot, so the set_reference timings
    // below isolate the per-REFERENCE precompute cost, not first-run JIT.
    let warm_r = synth_srgb(w, h, 9_001);
    let warm_d = synth_srgb(w, h, 9_002);
    let _ = m
        .compute_srgb_u8(&warm_r, &warm_d)
        .expect("warmup full compute");
    cubecl::future::block_on(client.sync()).expect("sync after warmup");

    // TSV header.
    if !no_header {
        println!("metric\tsize_mp\tw\th\tphase\tms_median\tms_min\tms_max\tn\traw_samples\tnotes");
    }
    eprintln!(
        "# metric={metric_tag} w={w} h={h} reps={reps} newref_phases={newref_phases} (warm instance, cuda)"
    );

    // --- setref1: FIRST set_reference on the warm instance (reps×). ---
    let s1 = time_setref_phase(&mut m, &client, w, h, 0x1000_0000, reps);
    emit(
        &metric_tag,
        w,
        h,
        "setref1",
        &s1,
        "first_ref_warm_instance synced distinct_pixels_per_rep",
    );

    // --- setref2..setrefK: DISTINCT new references (the reuse path). ---
    for p in 0..newref_phases {
        let seed_base = 0x2000_0000u32.wrapping_add((p as u32) * 0x0010_0000);
        let s = time_setref_phase(&mut m, &client, w, h, seed_base, reps);
        let phase = format!("setref{}", p + 2);
        emit(
            &metric_tag,
            w,
            h,
            &phase,
            &s,
            "new_ref_distinct_pixels reuse_path synced",
        );
    }

    // --- warm_call: warm compute_with_cached_reference against the last
    // cached reference. Distinct distorted image per rep. Self-syncs via
    // the host readback inside the call; we add an explicit sync too. ---
    let mut warm = Vec::with_capacity(reps);
    let mut last_score = 0.0_f64;
    for i in 0..reps {
        let d = synth_srgb(w, h, 0x3000_0000u32.wrapping_add(i as u32 * 13 + 1));
        let t = Instant::now();
        let s = m
            .compute_with_reference_srgb_u8(&d)
            .expect("warm compute_with_cached_reference");
        cubecl::future::block_on(client.sync()).expect("sync after warm_call");
        warm.push(t.elapsed().as_secs_f64() * 1e3);
        last_score = s.value;
    }
    emit(
        &metric_tag,
        w,
        h,
        "warm_call",
        &warm,
        &format!("compute_with_cached_reference last_score={last_score:.4}"),
    );

    eprintln!("# done {metric_tag}");
}
