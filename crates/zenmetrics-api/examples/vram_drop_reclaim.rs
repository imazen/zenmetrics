//! vram_drop_reclaim — prove that `reclaim_pooled_vram` returns a
//! dropped metric's GPU memory to the driver (task #150).
//!
//! cubecl pools device pages across `Handle` drop, so dropping a
//! [`Metric`] alone returns its buffers to the pool free list but leaves
//! the pages resident (the ~3830 MiB plateau task #147 measured for
//! butteraugli at 16 MP). [`zenmetrics_api::reclaim_pooled_vram`] issues
//! cubecl's `memory_cleanup` + `sync` to hand those pages back to the
//! driver. This binary measures, **in one process**, the VRAM at each
//! step so the plateau-vs-reclaim difference is visible as real
//! `nvidia-smi memory.used` numbers (no subprocess isolation — that's
//! the whole point: reclaim makes in-process drop actually free).
//!
//! Two scenarios:
//!
//!   1. `drop_reclaim` — one metric: baseline → construct → score →
//!      (resident) → drop → (POOLED PLATEAU, still high) → reclaim →
//!      (≈ baseline). PASS = post-reclaim delta << post-drop delta and
//!      within a small band of baseline.
//!
//!   2. `mixed_chunk` — three metrics in sequence (cvvdp → ssim2 →
//!      butter) at one size, mimicking an orchestrator chunk that swaps
//!      signatures. With reclaim BETWEEN metrics (default) the peak is
//!      ≈ MAX(single metric); with `ZENMETRICS_NO_SWAP_VRAM_CLEANUP=1`
//!      (or `--no-reclaim`) the pooled pages accumulate toward SUM. The
//!      binary runs BOTH variants back-to-back and prints the peak of
//!      each so the MAX-vs-SUM gap is one number.
//!
//! VRAM probe: `nvidia-smi --query-gpu=memory.used` (global card used
//! MiB; per-PID is hidden under WSL2). Each sample is the min of a short
//! read window after a settle delay, filtering transient spikes from any
//! concurrent GPU consumer. Absolute used-MiB is printed every step so a
//! perturbation is visible as a baseline shift.
//!
//! Run:
//! ```sh
//! cargo run -p zenmetrics-api --release \
//!   --features cuda,all-metrics,pixels \
//!   --example vram_drop_reclaim -- --size 4096 --tsv /tmp/vram_dr.tsv
//! ```

#![cfg(feature = "cuda")]

use std::process::Command;
use std::time::Duration;

use zenmetrics_api::{
    reclaim_pooled_vram, Backend, MemoryMode, Metric, MetricKind, MetricParams,
};

fn nvidia_smi_used_mib() -> Option<u64> {
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
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()
}

/// Min-of-window sample after a settle delay. The settle lets cubecl's
/// deferred frees land (a `reclaim` already syncs, but `score` readback
/// also syncs the GPU queue), and the min filters additive transients
/// from another GPU process.
fn sample_mib(settle_ms: u64, reads: u32) -> u64 {
    std::thread::sleep(Duration::from_millis(settle_ms));
    let mut lo = nvidia_smi_used_mib().expect("nvidia-smi memory.used");
    for _ in 1..reads.max(1) {
        std::thread::sleep(Duration::from_millis(10));
        if let Some(v) = nvidia_smi_used_mib() {
            lo = lo.min(v);
        }
    }
    lo
}

fn make_image(seed: u64, w: u32, h: u32) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
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

/// Construct + one score for `kind`, returning the resident used-MiB
/// after the score (pre-drop). Drops the metric on return.
fn construct_score_drop(
    kind: MetricKind,
    w: u32,
    h: u32,
    r: &[u8],
    d: &[u8],
    settle_ms: u64,
    reads: u32,
) -> (f64, u64) {
    let params = MetricParams::default_for(kind);
    let mut m = Metric::new_with_memory_mode(kind, Backend::Cuda, w, h, params, MemoryMode::Full)
        .expect("construct");
    let score = m.compute_srgb_u8(r, d).expect("score").value;
    let resident = sample_mib(settle_ms, reads);
    drop(m);
    (score, resident)
}

struct Cfg {
    w: u32,
    h: u32,
    settle_ms: u64,
    reads: u32,
    tsv: Option<String>,
}

fn emit_tsv(cfg: &Cfg, scenario: &str, step: &str, variant: &str, used_mib: u64, baseline: u64) {
    if let Some(path) = &cfg.tsv {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(path)
            .expect("append tsv");
        let mp = (cfg.w as f64 * cfg.h as f64) / 1_000_000.0;
        writeln!(
            f,
            "{scenario}\t{step}\t{variant}\t{}\t{}\t{:.2}\t{used_mib}\t{}",
            cfg.w,
            cfg.h,
            mp,
            used_mib as i64 - baseline as i64,
        )
        .ok();
    }
}

/// Scenario 1: one metric, prove drop→reclaim returns VRAM.
fn drop_reclaim(cfg: &Cfg) {
    let w = cfg.w;
    let h = cfg.h;
    let r = make_image(0xA5A5, w, h);
    let d = make_image(0x5A5A, w, h);

    // Use butteraugli — it's the #147 reference metric and has a large
    // (~3.8 GiB at 16 MP) Full working set so the plateau is obvious.
    let kind = if cfg!(feature = "butter") {
        MetricKind::Butter
    } else {
        MetricKind::Cvvdp
    };

    let baseline = sample_mib(cfg.settle_ms, cfg.reads);
    println!("\n== drop_reclaim ({}, {w}x{h}) ==", kind.tag());
    println!("  baseline                 used={baseline} MiB  (delta 0)");
    emit_tsv(cfg, "drop_reclaim", "baseline", "reclaim", baseline, baseline);

    let params = MetricParams::default_for(kind);
    let mut m = Metric::new_with_memory_mode(kind, Backend::Cuda, w, h, params, MemoryMode::Full)
        .expect("construct");
    let score = m.compute_srgb_u8(&r, &d).expect("score").value;
    let resident = sample_mib(cfg.settle_ms, cfg.reads);
    println!(
        "  after construct+score    used={resident} MiB  (delta {:+})  score={score:.4}",
        resident as i64 - baseline as i64
    );
    emit_tsv(cfg, "drop_reclaim", "resident", "reclaim", resident, baseline);

    drop(m);
    let after_drop = sample_mib(cfg.settle_ms, cfg.reads);
    println!(
        "  after drop (POOL plateau) used={after_drop} MiB  (delta {:+})  <- still resident: cubecl pool",
        after_drop as i64 - baseline as i64
    );
    emit_tsv(cfg, "drop_reclaim", "after_drop", "reclaim", after_drop, baseline);

    reclaim_pooled_vram(Backend::Cuda);
    let after_reclaim = sample_mib(cfg.settle_ms, cfg.reads);
    println!(
        "  after reclaim_pooled_vram used={after_reclaim} MiB  (delta {:+})  <- returned to driver",
        after_reclaim as i64 - baseline as i64
    );
    emit_tsv(cfg, "drop_reclaim", "after_reclaim", "reclaim", after_reclaim, baseline);

    let drop_delta = after_drop as i64 - baseline as i64;
    let reclaim_delta = after_reclaim as i64 - baseline as i64;
    let returned = after_drop as i64 - after_reclaim as i64;
    println!(
        "  => reclaim returned {returned} MiB to driver; post-reclaim delta {reclaim_delta:+} MiB \
         (post-drop was {drop_delta:+} MiB)"
    );
}

/// Scenario 2: three metrics in sequence, peak with vs without
/// between-metric reclaim. Mirrors the orchestrator swap path.
fn mixed_chunk(cfg: &Cfg, do_reclaim: bool) -> u64 {
    let w = cfg.w;
    let h = cfg.h;
    let r = make_image(0xA5A5, w, h);
    let d = make_image(0x5A5A, w, h);

    let kinds: &[MetricKind] = &[MetricKind::Cvvdp, MetricKind::Ssim2, MetricKind::Butter];

    let baseline = sample_mib(cfg.settle_ms, cfg.reads);
    let variant = if do_reclaim { "reclaim" } else { "no_reclaim" };
    println!(
        "\n== mixed_chunk variant={variant} ({w}x{h}) baseline={baseline} MiB =="
    );
    emit_tsv(cfg, "mixed_chunk", "baseline", variant, baseline, baseline);

    let mut peak = baseline;
    for (i, &kind) in kinds.iter().enumerate() {
        let (score, resident) = construct_score_drop(kind, w, h, &r, &d, cfg.settle_ms, cfg.reads);
        peak = peak.max(resident);
        println!(
            "  [{i}] {:<7} resident={resident} MiB (delta {:+})  peak_so_far={peak}  score={score:.4}",
            kind.tag(),
            resident as i64 - baseline as i64
        );
        emit_tsv(cfg, "mixed_chunk", kind.tag(), variant, resident, baseline);
        // Between-metric reclaim = the orchestrator swap cleanup. Off in
        // the no_reclaim variant so the pool accumulates toward SUM.
        if do_reclaim {
            reclaim_pooled_vram(Backend::Cuda);
            let after = sample_mib(cfg.settle_ms, cfg.reads);
            println!(
                "      reclaim -> used={after} MiB (delta {:+})",
                after as i64 - baseline as i64
            );
        }
    }
    // Final reclaim so the next variant starts from a clean pool.
    reclaim_pooled_vram(Backend::Cuda);
    println!(
        "  => variant={variant} PEAK delta over chunk = {:+} MiB",
        peak as i64 - baseline as i64
    );
    emit_tsv(cfg, "mixed_chunk", "PEAK", variant, peak, baseline);
    peak.saturating_sub(baseline)
}

/// Scenario 3: warm per-call path regression guard. One metric, one
/// signature, N warm scores in a row — NO reclaim between scores (that
/// is the contract: reclaim fires only on metric-swap / drop, never
/// between same-signature scores). Reports per-call wall stats and
/// confirms the VRAM floor is flat across the loop, so the reader can
/// verify the #150 cleanup did not regress the warm path measured in
/// #145.
fn warm_loop(cfg: &Cfg, n: usize) {
    use std::time::Instant;
    let w = cfg.w;
    let h = cfg.h;
    let r = make_image(0xA5A5, w, h);
    let kind = MetricKind::Cvvdp; // cached-ref metric, deep warm path.

    let baseline = sample_mib(cfg.settle_ms, cfg.reads);
    println!("\n== warm_loop ({}, {w}x{h}, n={n}) baseline={baseline} MiB ==", kind.tag());
    let params = MetricParams::default_for(kind);
    let mut m = Metric::new_with_memory_mode(kind, Backend::Cuda, w, h, params, MemoryMode::Full)
        .expect("construct");
    // Warm the reference once (cvvdp warm-ref path).
    let _ = m.set_reference_srgb_u8(&r);

    let mut walls_us: Vec<u128> = Vec::with_capacity(n);
    let mut vram_floor = u64::MAX;
    let mut vram_peak = 0u64;
    for i in 0..n {
        let d = make_image(0x5A5A_u64.wrapping_add(i as u64), w, h);
        let t = Instant::now();
        // Same-signature warm score against the cached reference.
        let _ = m
            .compute_with_cached_reference_srgb_u8(&d)
            .or_else(|_| m.compute_srgb_u8(&r, &d))
            .expect("warm score");
        walls_us.push(t.elapsed().as_micros());
        // Sample VRAM every few iters to confirm the floor stays flat
        // (no per-call alloc growth, no reclaim churn).
        if i % 4 == 0 {
            let used = sample_mib(20, 2);
            vram_floor = vram_floor.min(used);
            vram_peak = vram_peak.max(used);
        }
    }
    drop(m);
    reclaim_pooled_vram(Backend::Cuda);

    walls_us.sort_unstable();
    let p50 = walls_us[walls_us.len() / 2];
    let p10 = walls_us[walls_us.len() / 10.max(1).min(walls_us.len() - 1)];
    let p90 = walls_us[(walls_us.len() * 9 / 10).min(walls_us.len() - 1)];
    let span = vram_peak.saturating_sub(vram_floor);
    println!(
        "  warm per-call wall: p10={:.2} p50={:.2} p90={:.2} ms over {n} scores",
        p10 as f64 / 1000.0,
        p50 as f64 / 1000.0,
        p90 as f64 / 1000.0
    );
    println!(
        "  warm VRAM floor={vram_floor} peak={vram_peak} span={span} MiB \
         (flat span => no per-call growth, no reclaim churn)"
    );
    emit_tsv(cfg, "warm_loop", "vram_floor", "reclaim", vram_floor, baseline);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut w = 4096u32;
    let mut tsv: Option<String> = None;
    let mut settle_ms = 80u64;
    let mut reads = 5u32;
    let mut scenario = "all".to_string();
    let mut iter = args.iter().skip(1);
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--size" => {
                if let Some(v) = iter.next() {
                    w = v.parse().expect("size");
                }
            }
            "--tsv" => {
                tsv = iter.next().cloned();
            }
            "--settle-ms" => {
                if let Some(v) = iter.next() {
                    settle_ms = v.parse().unwrap_or(80);
                }
            }
            "--reads" => {
                if let Some(v) = iter.next() {
                    reads = v.parse().unwrap_or(5);
                }
            }
            "--scenario" => {
                scenario = iter.next().cloned().unwrap_or_else(|| "all".into());
            }
            _ => {}
        }
    }
    let cfg = Cfg {
        w,
        h: w,
        settle_ms,
        reads,
        tsv: tsv.clone(),
    };

    if let Some(path) = &tsv {
        use std::io::Write;
        let mut f = std::fs::File::create(path).expect("create tsv");
        writeln!(
            f,
            "scenario\tstep\tvariant\twidth\theight\tmp\tused_mib\tdelta_mib"
        )
        .unwrap();
    }

    println!("# vram_drop_reclaim — size {w}x{w}, GPU0 used now: {:?} MiB", nvidia_smi_used_mib());

    if scenario == "all" || scenario == "drop_reclaim" {
        drop_reclaim(&cfg);
    }
    if scenario == "all" || scenario == "mixed_chunk" {
        let peak_reclaim = mixed_chunk(&cfg, true);
        let peak_no_reclaim = mixed_chunk(&cfg, false);
        println!("\n== mixed_chunk SUMMARY ({w}x{w}) ==");
        println!("  peak delta WITH between-metric reclaim   : {peak_reclaim} MiB  (~MAX single metric)");
        println!("  peak delta WITHOUT reclaim (pool plateau): {peak_no_reclaim} MiB  (~SUM of metrics)");
        if peak_no_reclaim > peak_reclaim {
            println!(
                "  => reclaim cut peak by {} MiB ({:.1}x lower)",
                peak_no_reclaim - peak_reclaim,
                peak_no_reclaim as f64 / (peak_reclaim.max(1)) as f64
            );
        }
    }
    if scenario == "all" || scenario == "warm_loop" {
        let n: usize = std::env::var("WARM_N").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
        warm_loop(&cfg, n);
    }
    println!("\n# done{}", tsv.map(|p| format!(" — TSV at {p}")).unwrap_or_default());
}
