//! setref_timing — clean re-measure of butteraugli-gpu `set_reference`
//! per-reference cost on a WARM instance (task #148).
//!
//! ## Why this exists
//!
//! Task #144 measured (via the `zenmetrics_api::Metric` umbrella driver
//! `inprocess_warmth.rs`, scenario Q3) that on a warm butter instance the
//! FIRST `set_reference` is expensive but a SUBSEQUENT new reference is
//! nearly free (buffer reuse — the cached-ref planes are reallocated
//! in place, only the per-ref precompute re-runs). The #144 numbers
//! (34.3 ms @ 512², 3990 ms @ 16 MP for the first ref; 0.76 ms @ 512²,
//! 21.6 ms @ 16 MP for a subsequent new ref) were taken on a machine
//! where a concurrent zensim eval contaminated some cells. This driver
//! re-measures the SAME quantity on a quiet machine, with proper
//! repeated sampling (≥7 samples per cell, median + min + raw), so the
//! number can be trusted.
//!
//! ## What is measured
//!
//! On a single warm `Butteraugli` instance (context + kernels hot via a
//! throwaway full-mode `compute`, then a sync):
//!
//! - `setref1` — `set_reference(ref1)`, the FIRST set_reference on the warm
//!   instance. Repeated ≥`REPS` times (a fresh-pixel ref each rep so no
//!   identical-input shortcut), each followed by a full sync, each
//!   individually timed.
//! - `setref2..setrefK` — `set_reference(refN)` with DISTINCT pixel content,
//!   the reuse path (the #144 headline). Each phase is ≥`REPS` distinct refs,
//!   individually timed + synced. ≥3 distinct new-ref phases confirm the
//!   reuse cost is stable, not a one-off for ref2.
//! - `warm_call` — `compute_with_reference(dist)` warm score against the last
//!   cached reference, ≥`REPS` reps. Sanity check that the warm-call wall is
//!   the per-score cost, separate from the set_reference precompute.
//!
//! ## Correctness — synced timers, not async submission
//!
//! `set_reference` queues GPU work (upload, opsin, frequency separation,
//! reference-only mask pipeline) and does NOT read back to the host, so
//! `Instant::elapsed()` around it alone would measure SUBMISSION, not
//! execution (see project CLAUDE.md "Diagnosing Slow GPU Code"). Every
//! timed `set_reference` is therefore immediately followed by
//! `block_on(client.sync())` INSIDE the timed region, so the wall is the
//! real GPU precompute. `compute_with_reference` ends in a host readback
//! (it returns a scalar), so it self-syncs; we still keep the same shape.
//!
//! Each rep within a phase uses a DISTINCT-pixel reference (seed varies),
//! so no two consecutive `set_reference` calls feed identical bytes — the
//! kernel cannot shortcut on input equality, and we exercise the true
//! upload-and-precompute path every rep.
//!
//! ## Build (release, cuda, NO target-cpu=native)
//! ```sh
//! cargo build --release -p butteraugli-gpu --no-default-features \
//!   --features cuda --example setref_timing
//! ```
//!
//! ## Run
//! ```sh
//! SETREF_W=4096 SETREF_H=4096 SETREF_REPS=7 SETREF_NEWREF_PHASES=3 \
//!   setref_timing
//! ```
//!
//! Output: TSV rows on stdout, header first. Columns:
//!   size_mp  w  h  phase  ms_median  ms_min  ms_max  n  raw_samples  notes

#![cfg(feature = "cuda")]

use std::time::Instant;

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime as Backend;
use cubecl::prelude::ComputeClient;

use butteraugli_gpu::Butteraugli;

/// LCG-filled pseudo-random sRGB bytes, seed-controlled so each call can
/// produce DISTINCT pixel content (so consecutive set_reference calls
/// never feed identical bytes). Same generator as vram_leak_check.rs.
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

fn parse_u32(name: &str, default: u32) -> u32 {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn median(t: &[f64]) -> f64 {
    if t.is_empty() {
        return f64::NAN;
    }
    let mut v = t.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n.is_multiple_of(2) { (v[n / 2 - 1] + v[n / 2]) / 2.0 } else { v[n / 2] }
}

fn min_of(t: &[f64]) -> f64 {
    t.iter().cloned().fold(f64::INFINITY, f64::min)
}
fn max_of(t: &[f64]) -> f64 {
    t.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

/// Time `set_reference` `reps` times, a DISTINCT-pixel reference per rep,
/// each call immediately followed by a full sync inside the timed region.
/// `seed_base` distinguishes phases so no two phases share ref bytes.
fn time_setref_phase(
    b: &mut Butteraugli<Backend>,
    client: &ComputeClient<Backend>,
    w: u32,
    h: u32,
    seed_base: u32,
    reps: usize,
) -> Vec<f64> {
    // Pre-generate refs so host-side image gen is OUT of the timed region.
    let refs: Vec<Vec<u8>> =
        (0..reps).map(|i| synth_srgb(w, h, seed_base.wrapping_add(i as u32 * 31 + 1))).collect();
    let mut times = Vec::with_capacity(reps);
    for r in &refs {
        let t = Instant::now();
        b.set_reference(r).expect("set_reference");
        cubecl::future::block_on(client.sync()).expect("sync after set_reference");
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times
}

fn emit(w: u32, h: u32, phase: &str, samples: &[f64], notes: &str) {
    let mp = (w as f64 * h as f64) / 1_000_000.0;
    let raw = samples.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(",");
    println!(
        "{mp:.3}\t{w}\t{h}\t{phase}\t{med:.4}\t{min:.4}\t{max:.4}\t{n}\t{raw}\t{notes}",
        med = median(samples),
        min = min_of(samples),
        max = max_of(samples),
        n = samples.len(),
    );
}

fn main() {
    let w = parse_u32("SETREF_W", 512);
    let h = parse_u32("SETREF_H", 512);
    let reps = parse_u32("SETREF_REPS", 7).max(1) as usize;
    // Number of DISTINCT new-reference phases AFTER setref1 (setref2,
    // setref3, ...). Default 3 so the task's "≥3 distinct new refs" gate
    // is met (setref2/3/4).
    let newref_phases = parse_u32("SETREF_NEWREF_PHASES", 3).max(1) as usize;

    // CUDA context init (this is the cold context; everything below runs
    // warm against it). Build the client explicitly so we can sync.
    let client = Backend::client(&Default::default());

    // Full-mode instance (matches the #144 warm-instance Q3 path, which
    // used the umbrella Metric in full mode — not strip).
    let mut b = Butteraugli::<Backend>::new(client.clone(), w, h);

    // Warm the instance: a throwaway full-mode compute loads ALL kernels
    // (JIT) and populates the cubecl memory pool, then sync to flush. After
    // this, context + kernels + pool are hot, so the set_reference timings
    // below isolate the per-REFERENCE precompute cost, not first-run JIT.
    let warm_r = synth_srgb(w, h, 9_001);
    let warm_d = synth_srgb(w, h, 9_002);
    let _ = b.compute(&warm_r, &warm_d).expect("warmup full compute");
    cubecl::future::block_on(client.sync()).expect("sync after warmup");

    // TSV header.
    println!("size_mp\tw\th\tphase\tms_median\tms_min\tms_max\tn\traw_samples\tnotes");
    eprintln!("# w={w} h={h} reps={reps} newref_phases={newref_phases} (warm instance, cuda)");

    // --- setref1: FIRST set_reference on the warm instance (reps×). ---
    let s1 = time_setref_phase(&mut b, &client, w, h, 0x1000_0000, reps);
    emit(w, h, "setref1", &s1, "first_ref_warm_instance synced distinct_pixels_per_rep");

    // --- setref2..setrefK: DISTINCT new references (the reuse path). ---
    for p in 0..newref_phases {
        let seed_base = 0x2000_0000u32.wrapping_add((p as u32) * 0x0010_0000);
        let s = time_setref_phase(&mut b, &client, w, h, seed_base, reps);
        let phase = format!("setref{}", p + 2);
        emit(w, h, &phase, &s, "new_ref_distinct_pixels reuse_path synced");
    }

    // --- warm_call: warm compute_with_reference against the last cached
    // reference. Distinct distorted image per rep. Self-syncs via readback. ---
    let mut warm = Vec::with_capacity(reps);
    let mut last_score = 0.0_f64;
    for i in 0..reps {
        let d = synth_srgb(w, h, 0x3000_0000u32.wrapping_add(i as u32 * 13 + 1));
        let t = Instant::now();
        let s = b.compute_with_reference(&d).expect("warm compute_with_reference");
        // Defensive: ensure the queue is drained even though the readback
        // inside compute_with_reference already syncs.
        cubecl::future::block_on(client.sync()).expect("sync after warm_call");
        warm.push(t.elapsed().as_secs_f64() * 1e3);
        last_score = s.score as f64;
    }
    emit(w, h, "warm_call", &warm, &format!("compute_with_reference last_score={last_score:.4}"));

    eprintln!("# done");
}
