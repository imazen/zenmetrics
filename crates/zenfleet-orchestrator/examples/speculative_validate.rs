//! Validate the speculative-execution algorithm against synthetic
//! straggler distributions.
//!
//! Two scenarios:
//!
//! 1. `iter2.5` (default) — the actual N=10 sweep shape: stragglers are
//!    only marginally slower than the median tail (380 s vs 280 s, a
//!    ~1.4× ratio). The speculative scheduler with factor=1.5 *cannot*
//!    fire earlier than TTL=360 s on this workload, BECAUSE p95 of the
//!    fast completions already approaches the straggler time. This is
//!    a falsification of "speculative always beats TTL on every
//!    workload" — for tight bimodal distributions, TTL is the right
//!    tool. The honest result.
//!
//! 2. `mapreduce-shape` — Dean & Ghemawat 2004's regime: stragglers
//!    run 5-7× slower than median (800-900 s vs 100-280 s). This is
//!    the canonical speculative-execution win case.
//!
//! Run with no args for iter2.5; `cargo run --example speculative_validate -- mapreduce-shape`
//! for the canonical win case.

use serde_json::json;
use zenfleet_orchestrator::{SpeculativeConfig, SpeculativeState};

fn iter25_shape() -> (f64, Vec<f64>, Vec<f64>) {
    let baseline_t_done = 388.0_f64;
    let fast_completions: Vec<f64> = vec![
        // First chunk per fast worker (boot included): 130-170 s.
        140.0, 150.0, 130.0, 160.0, 145.0, 155.0, 135.0, 165.0,
        // Second chunk per fast worker (~40 s after first):
        185.0, 192.0, 175.0, 198.0, 188.0, 196.0, 180.0, 200.0,
        // Third chunk per fast worker:
        220.0, 230.0, 215.0, 240.0, 225.0, 235.0, 218.0, 245.0,
        // Fourth-chunk overflow from fast workers (mopping up):
        255.0, 268.0, 272.0, 280.0,
    ];
    let straggler_completions: Vec<f64> = vec![380.0, 395.0];
    (baseline_t_done, fast_completions, straggler_completions)
}

fn mapreduce_shape() -> (f64, Vec<f64>, Vec<f64>) {
    let baseline_t_done = 900.0_f64;
    let fast_completions: Vec<f64> = vec![
        100.0, 110.0, 105.0, 115.0, 120.0, 95.0, 102.0, 108.0,
        150.0, 160.0, 145.0, 155.0, 165.0, 140.0, 158.0, 152.0,
        200.0, 210.0, 195.0, 215.0, 220.0, 198.0, 212.0, 205.0,
        250.0, 260.0, 265.0, 275.0,
    ];
    let straggler_completions: Vec<f64> = vec![800.0, 900.0];
    (baseline_t_done, fast_completions, straggler_completions)
}

fn run_scenario(name: &str, baseline_t_done: f64, fast: Vec<f64>, stragglers: Vec<f64>) {
    println!("=== Speculative execution validation — scenario={} ===", name);
    println!("Baseline (no speculative): t_done = {} s", baseline_t_done);
    println!();

    let cfg = SpeculativeConfig::default();
    let mut state = SpeculativeState::new();
    let chunk_ids: Vec<String> = (0..30).map(|i| format!("c{:02}", i)).collect();
    for cid in &chunk_ids {
        state.record_dispatched(cid, 0.0);
    }
    let mut completion_schedule: Vec<(String, f64)> = Vec::new();
    for (i, t) in fast.iter().enumerate() {
        completion_schedule.push((format!("c{:02}", i), *t));
    }
    for (i, t) in stragglers.iter().enumerate() {
        completion_schedule.push((format!("c{:02}", 28 + i), *t));
    }

    let mut speculative_dispatches: Vec<(String, f64, f64, f64)> = Vec::new();
    let mut t_first_speculative_secs: Option<f64> = None;
    let mut tick = 0.0_f64;
    let dt = 10.0_f64;

    while tick <= baseline_t_done + 60.0 {
        // Mark any completion that happened by this tick.
        for (cid, t_done) in &completion_schedule {
            if *t_done <= tick + dt / 2.0 {
                state.record_completed(cid, *t_done);
            }
        }
        // Iterate in-flight, ask the scheduler.
        for cid in &chunk_ids {
            if let Some(elapsed) = state.decide_speculative(cid, tick, &cfg) {
                let p95 = state.p95_completion_secs().unwrap_or(0.0);
                speculative_dispatches.push((cid.clone(), tick, elapsed, p95));
                state.record_speculative_dispatched(cid);
                if t_first_speculative_secs.is_none() {
                    t_first_speculative_secs = Some(tick);
                }
            }
        }
        tick += dt;
    }

    // Simulate the wall-clock benefit. A re-dispatched chunk completes
    // at min(original_t_done, dispatch_tick + 60s) — assuming a fast
    // worker picks it up immediately and finishes within one steady-
    // state chunk-time (~60 s for the iter2.5 fleet; ~50 s for the
    // mapreduce-shape fleet).
    let pickup_secs = 60.0;
    let mut sim_t_done = 0.0_f64;
    for (cid, t_orig) in &completion_schedule {
        let spec_t = speculative_dispatches
            .iter()
            .find(|(c, _, _, _)| c == cid)
            .map(|(_, dispatched_at, _, _)| *dispatched_at + pickup_secs);
        let effective = match spec_t {
            Some(t_spec) => t_orig.min(t_spec),
            None => *t_orig,
        };
        if effective > sim_t_done {
            sim_t_done = effective;
        }
    }

    let reduction_pct = ((baseline_t_done - sim_t_done) / baseline_t_done) * 100.0;
    let n_spec = speculative_dispatches.len();
    let dup_compute_secs = n_spec as f64 * pickup_secs;
    let total_fleet_secs = 10.0_f64 * sim_t_done;
    let dup_overhead_pct = (dup_compute_secs / total_fleet_secs) * 100.0;

    println!("Speculative scheduler decisions:");
    if speculative_dispatches.is_empty() {
        println!("  (none — distribution too tight; TTL would fire first)");
    } else {
        for (cid, t, elapsed, p95) in &speculative_dispatches {
            println!(
                "  t={:>5.1}s  chunk={}  elapsed={:>5.1}s  p95={:>5.1}s  threshold={:>5.1}s",
                t,
                cid,
                elapsed,
                p95,
                p95 * cfg.straggler_factor
            );
        }
    }
    println!();
    println!("End-of-run stats:");
    println!("  p95(completion) = {:?} s", state.p95_completion_secs());
    println!("  n_speculative_dispatches = {}", n_spec);
    println!("  t_first_speculative = {:?} s", t_first_speculative_secs);
    println!("  simulated t_done WITH speculative = {:.1} s", sim_t_done);
    println!("  baseline t_done WITHOUT speculative = {:.1} s", baseline_t_done);
    println!("  reduction = {:+.1}%", reduction_pct);
    println!("  duplicate compute overhead = {:.2}%", dup_overhead_pct);

    let report = json!({
        "scenario": name,
        "baseline_t_done_secs": baseline_t_done,
        "simulated_t_done_with_speculative_secs": sim_t_done,
        "wall_reduction_percent": reduction_pct,
        "speculative_dispatches": speculative_dispatches.iter().map(|(c, t, e, p95)| {
            json!({"chunk_id": c, "dispatched_at_secs": t, "elapsed_secs": e, "p95_at_dispatch_secs": p95})
        }).collect::<Vec<_>>(),
        "p95_completion_secs": state.p95_completion_secs(),
        "n_completed": state.n_completed(),
        "n_speculative_dispatches": n_spec,
        "t_first_speculative_secs": t_first_speculative_secs,
        "duplicate_compute_overhead_percent": dup_overhead_pct,
        "assumed_speculative_pickup_secs": pickup_secs,
        "config": json!({
            "straggler_factor": cfg.straggler_factor,
            "min_completed_for_stats": cfg.min_completed_for_stats,
            "speculation_cap_per_chunk": cfg.speculation_cap_per_chunk,
        }),
    });
    println!();
    println!("--- JSON report ---");
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn main() {
    let scenario = std::env::args().nth(1).unwrap_or_else(|| "iter2.5".to_string());
    let (baseline, fast, straggler) = match scenario.as_str() {
        "mapreduce-shape" => mapreduce_shape(),
        _ => iter25_shape(),
    };
    run_scenario(&scenario, baseline, fast, straggler);
}
