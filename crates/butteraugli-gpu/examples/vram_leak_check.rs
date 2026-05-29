//! In-process VRAM leak check for butteraugli-gpu (task #147).
//!
//! Distinguishes a genuine VRAM leak (monotonic growth per cycle) from
//! healthy cubecl memory-pool reuse (growth during warmup, then a flat
//! plateau). Three checks, each emitting one TSV row per cycle:
//!
//!   1. `many_scores`   — 1 instance, `set_reference` once, then
//!                        `compute_with_reference` (or `compute` /
//!                        `compute_strip`) N times. PASS = plateau after
//!                        warmup; FAIL = monotonic growth.
//!   2. `many_new_refs` — 1 instance, `set_reference` with DISTINCT-pixel
//!                        references N times, score after each (the reuse
//!                        path #144 flagged). PASS = stable (planes
//!                        reused); FAIL = grows per new ref.
//!   3. `create_drop`   — construct → score → DROP the instance, repeat N
//!                        times, sample after each drop+sync. NOTE: cubecl
//!                        pools buffers ACROSS Drop, so the absolute value
//!                        does NOT return to baseline even when healthy —
//!                        PASS = plateau (not strictly increasing every
//!                        cycle); FAIL = strictly increasing every cycle.
//!
//! VRAM probe: `nvidia-smi --query-gpu=memory.used` (global card used MiB).
//! `--query-compute-apps=used_memory` returns empty under WSL2 (the GPU
//! paravirt layer hides per-PID accounting), so we sample GLOBAL used and
//! report the delta from the process-start baseline. This is valid only
//! when no other process perturbs the card; the absolute value is logged
//! every row so a perturbation is visible as a step in the baseline.
//!
//! Every sample is taken AFTER `block_on(client.sync())` + a short settle
//! sleep so cubecl's deferred frees land before we read the card.
//!
//! Environment:
//! - `LEAK_CHECK` — one of `many_scores`, `many_new_refs`, `create_drop`,
//!                  or `all` (default `all`).
//! - `LEAK_MODE`  — `full`, `warm_ref`, `strip`, `warm_ref_strip`
//!                  (default `warm_ref`). For `many_scores`:
//!                    full/strip       → cold-ref `compute`/`compute_strip`
//!                    warm_ref/_strip  → `set_reference` + `compute_with_reference`
//! - `LEAK_W` / `LEAK_H` — image dims (default 1024×1024 = 1 MP).
//! - `LEAK_N`     — cycle count (default 100 for checks 1/2, 30 for check 3).
//! - `LEAK_BODY`  — strip body height (default 256).
//! - `LEAK_SETTLE_MS` — settle sleep after sync before sampling (default 60).
//!
//! Output: TSV rows on stdout with a header. Columns:
//!   check  mode  size_mp  w  h  cycle  vram_used_mib  vram_delta_mib

#![cfg(feature = "cuda")]

use std::process::Command;
use std::time::Duration;

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime as Backend;
use cubecl::prelude::ComputeClient;

use butteraugli_gpu::Butteraugli;

/// LCG-filled pseudo-random sRGB bytes, seed-controlled so each call can
/// produce DISTINCT pixel content (required for check 2).
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

/// Global card used-memory in MiB via nvidia-smi. Returns None if the
/// query fails (no GPU / no nvidia-smi).
fn vram_used_mib() -> Option<u64> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    s.lines().next()?.trim().parse().ok()
}

/// Drain the GPU queue and let deferred frees land, then sample the card.
///
/// Takes the MINIMUM of `samples` nvidia-smi reads. The card's `memory.used`
/// is GLOBAL on this host (per-PID accounting is hidden under WSL2), so a
/// concurrent GPU consumer's allocation rides ADDITIVELY on top of ours.
/// The minimum over a short window filters out those transient spikes —
/// the true "card with only my process" value is the floor. A genuine
/// leak in OUR process pushes that floor up monotonically and survives
/// the min; another process's churn does not.
fn sync_and_sample(client: &ComputeClient<Backend>, settle_ms: u64, samples: u32) -> u64 {
    cubecl::future::block_on(client.sync()).expect("sync");
    std::thread::sleep(Duration::from_millis(settle_ms));
    let mut lo = vram_used_mib().expect("nvidia-smi memory.used query failed");
    for _ in 1..samples.max(1) {
        std::thread::sleep(Duration::from_millis(8));
        if let Some(v) = vram_used_mib() {
            lo = lo.min(v);
        }
    }
    lo
}

struct Ctx {
    mode: String,
    w: u32,
    h: u32,
    body: u32,
    settle_ms: u64,
    samples: u32,
    baseline_mib: u64,
}

impl Ctx {
    fn sample(&self, client: &ComputeClient<Backend>) -> u64 {
        sync_and_sample(client, self.settle_ms, self.samples)
    }
    fn size_mp(&self) -> f64 {
        (self.w as f64 * self.h as f64) / 1_000_000.0
    }
    fn emit(&self, check: &str, cycle: usize, used: u64) {
        let delta = used as i64 - self.baseline_mib as i64;
        let mp = self.size_mp();
        let (mode, w, h) = (&self.mode, self.w, self.h);
        println!("{check}\t{mode}\t{mp:.3}\t{w}\t{h}\t{cycle}\t{used}\t{delta}");
    }
}

/// Check 1: many scores against one cached/cold reference.
fn check_many_scores(client: &ComputeClient<Backend>, ctx: &Ctx, n: usize) {
    let r = synth_srgb(ctx.w, ctx.h, 42);
    // Distinct distorted images per iter so the kernel can't shortcut on
    // identical input (and to exercise the upload staging buffer each call).
    match ctx.mode.as_str() {
        "full" => {
            let mut b = Butteraugli::<Backend>::new(client.clone(), ctx.w, ctx.h);
            for i in 0..n {
                let d = synth_srgb(ctx.w, ctx.h, 1000 + i as u32);
                let _ = b.compute(&r, &d).expect("compute");
                ctx.emit("many_scores", i, ctx.sample(client));
            }
        }
        "strip" => {
            let mut b = Butteraugli::<Backend>::new_strip(client.clone(), ctx.w, ctx.h, ctx.body);
            for i in 0..n {
                let d = synth_srgb(ctx.w, ctx.h, 1000 + i as u32);
                let _ = b.compute_strip(&r, &d).expect("compute_strip");
                ctx.emit("many_scores", i, ctx.sample(client));
            }
        }
        "warm_ref" => {
            let mut b = Butteraugli::<Backend>::new(client.clone(), ctx.w, ctx.h);
            b.set_reference(&r).expect("set_reference");
            for i in 0..n {
                let d = synth_srgb(ctx.w, ctx.h, 1000 + i as u32);
                let _ = b.compute_with_reference(&d).expect("compute_with_reference");
                ctx.emit("many_scores", i, ctx.sample(client));
            }
        }
        "warm_ref_strip" => {
            let mut b = Butteraugli::<Backend>::new_strip(client.clone(), ctx.w, ctx.h, ctx.body);
            b.set_reference(&r).expect("set_reference (strip)");
            for i in 0..n {
                let d = synth_srgb(ctx.w, ctx.h, 1000 + i as u32);
                let _ = b
                    .compute_with_reference(&d)
                    .expect("compute_with_reference (strip)");
                ctx.emit("many_scores", i, ctx.sample(client));
            }
        }
        other => panic!("unknown LEAK_MODE for many_scores: {other}"),
    }
}

/// Check 2: many DISTINCT references on one instance (the reuse path).
fn check_many_new_refs(client: &ComputeClient<Backend>, ctx: &Ctx, n: usize) {
    // Distorted image fixed; the REFERENCE changes every cycle. This
    // exercises set_reference's in-place plane reuse N times.
    let d = synth_srgb(ctx.w, ctx.h, 137);
    let strip_mode = matches!(ctx.mode.as_str(), "strip" | "warm_ref_strip");
    if strip_mode {
        let mut b = Butteraugli::<Backend>::new_strip(client.clone(), ctx.w, ctx.h, ctx.body);
        for i in 0..n {
            let r = synth_srgb(ctx.w, ctx.h, 5000 + i as u32);
            b.set_reference(&r).expect("set_reference (strip) new ref");
            let _ = b
                .compute_with_reference(&d)
                .expect("compute_with_reference (strip) new ref");
            ctx.emit("many_new_refs", i, ctx.sample(client));
        }
    } else {
        let mut b = Butteraugli::<Backend>::new(client.clone(), ctx.w, ctx.h);
        for i in 0..n {
            let r = synth_srgb(ctx.w, ctx.h, 5000 + i as u32);
            b.set_reference(&r).expect("set_reference new ref");
            let _ = b
                .compute_with_reference(&d)
                .expect("compute_with_reference new ref");
            ctx.emit("many_new_refs", i, ctx.sample(client));
        }
    }
}

/// Check 3: construct → score → DROP, repeated. cubecl pools across Drop,
/// so a plateau (not strictly-increasing) is PASS.
fn check_create_drop(client: &ComputeClient<Backend>, ctx: &Ctx, n: usize) {
    let r = synth_srgb(ctx.w, ctx.h, 42);
    let d = synth_srgb(ctx.w, ctx.h, 137);
    let strip_mode = matches!(ctx.mode.as_str(), "strip" | "warm_ref_strip");
    let warm = matches!(ctx.mode.as_str(), "warm_ref" | "warm_ref_strip");
    for i in 0..n {
        {
            if strip_mode {
                let mut b =
                    Butteraugli::<Backend>::new_strip(client.clone(), ctx.w, ctx.h, ctx.body);
                if warm {
                    b.set_reference(&r).expect("set_reference (strip)");
                    let _ = b
                        .compute_with_reference(&d)
                        .expect("compute_with_reference (strip)");
                } else {
                    let _ = b.compute_strip(&r, &d).expect("compute_strip");
                }
            } else {
                let mut b = Butteraugli::<Backend>::new(client.clone(), ctx.w, ctx.h);
                if warm {
                    b.set_reference(&r).expect("set_reference");
                    let _ = b.compute_with_reference(&d).expect("compute_with_reference");
                } else {
                    let _ = b.compute(&r, &d).expect("compute");
                }
            }
            // `b` dropped here at end of scope.
        }
        ctx.emit("create_drop", i, ctx.sample(client));
    }
}

fn main() {
    let check = std::env::var("LEAK_CHECK").unwrap_or_else(|_| "all".into());
    let mode = std::env::var("LEAK_MODE").unwrap_or_else(|_| "warm_ref".into());
    let w = parse_u32("LEAK_W", 1024);
    let h = parse_u32("LEAK_H", 1024);
    let body = parse_u32("LEAK_BODY", 256);
    let settle_ms = parse_u32("LEAK_SETTLE_MS", 60) as u64;
    // Number of nvidia-smi reads per sample; the MIN is kept (transient-
    // spike-robust against a concurrent GPU consumer on the shared card).
    let samples = parse_u32("LEAK_SAMPLES", 3).max(1);

    let client = Backend::client(&Default::default());

    // Baseline BEFORE any GPU allocation by this process. Discard the first
    // read (the first nvidia-smi spawn can perturb scheduling), then take
    // the min of `samples` reads to filter any concurrent transient.
    let _ = vram_used_mib();
    std::thread::sleep(Duration::from_millis(150));
    let mut baseline_mib = vram_used_mib().expect("nvidia-smi baseline query failed");
    for _ in 1..samples {
        std::thread::sleep(Duration::from_millis(10));
        if let Some(v) = vram_used_mib() {
            baseline_mib = baseline_mib.min(v);
        }
    }

    let ctx = Ctx { mode: mode.clone(), w, h, body, settle_ms, samples, baseline_mib };

    // TSV header (only when running a single check; `all` emits one header
    // total). check\tmode\tsize_mp\tw\th\tcycle\tvram_used_mib\tvram_delta_mib
    println!("check\tmode\tsize_mp\tw\th\tcycle\tvram_used_mib\tvram_delta_mib");
    eprintln!(
        "# baseline_mib={baseline_mib} mode={mode} w={w} h={h} body={body} settle_ms={settle_ms} samples={samples}"
    );

    let n1 = parse_u32("LEAK_N", 100) as usize;
    let n3 = parse_u32("LEAK_N", 30) as usize;

    match check.as_str() {
        "many_scores" => check_many_scores(&client, &ctx, n1),
        "many_new_refs" => check_many_new_refs(&client, &ctx, n1),
        "create_drop" => check_create_drop(&client, &ctx, n3),
        "all" => {
            check_many_scores(&client, &ctx, n1);
            check_many_new_refs(&client, &ctx, n1);
            // create_drop default N is smaller; honor LEAK_N if set else 30.
            let n_cd = std::env::var("LEAK_N")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(30usize);
            check_create_drop(&client, &ctx, n_cd);
        }
        other => panic!("unknown LEAK_CHECK: {other}"),
    }

    let _ = n3; // silence unused when not in single-check create_drop path
    eprintln!("# done");
}
