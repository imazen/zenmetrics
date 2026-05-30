//! vram_swap_peak — measure orchestrator peak VRAM across a mixed-metric
//! chunk, with vs without the swap-time pool reclaim (task #150).
//!
//! The single GPU lane reuses a warm metric instance for consecutive
//! same-signature tasks and rebuilds on a signature change. Before
//! task #150 the rebuild dropped the old metric's `Handle`s to cubecl's
//! pool but the device pages stayed resident, so a chunk that visits
//! several metrics drove peak VRAM toward the SUM of their working sets.
//! The fix calls `reclaim_pooled_vram` at the swap (after dropping the
//! old instance, before constructing the new one), returning the pooled
//! pages to the driver so peak stays at ≈ MAX(single metric).
//!
//! This binary submits a mixed-metric task list (cvvdp → ssim2 → butter,
//! same size) through `Orchestrator::run_all`, polling
//! `nvidia-smi memory.used` in a tight loop to capture the peak. It runs
//! the chunk TWICE in two child subprocesses — one with the swap reclaim
//! ON (default) and one with `ZENMETRICS_NO_SWAP_VRAM_CLEANUP=1` — so the
//! cubecl pool starts cold for each and the peaks are directly
//! comparable. Driver prints both peaks and the MAX-vs-SUM gap.
//!
//! Run:
//! ```sh
//! cargo run -p zenmetrics-orchestrator --release --features cuda \
//!   --example vram_swap_peak -- --size 4096 --tsv /tmp/vram_swap.tsv
//! ```

#![cfg(feature = "cuda")]

use std::env;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

use zenmetrics_api::MetricKind;
use zenmetrics_orchestrator::{
    Orchestrator, OrchestratorConfig, Task, TaskData, swap_vram_reclaim_count,
    synth_pair_offset_dist,
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

/// The mixed-metric chunk. Each metric appears twice so a same-signature
/// warm reuse happens between the swaps (proving reclaim does NOT fire
/// between same-signature tasks). Order forces 3 signature swaps:
/// cvvdp,cvvdp → ssim2,ssim2 → butter,butter.
const CHUNK: &[MetricKind] = &[
    MetricKind::Cvvdp,
    MetricKind::Cvvdp,
    MetricKind::Ssim2,
    MetricKind::Ssim2,
    MetricKind::Butter,
    MetricKind::Butter,
];

// ===================================================================
// Child — runs one chunk, prints peak + swap-reclaim count.
// ===================================================================

fn run_child(size: u32) {
    let mut orch = Orchestrator::new(OrchestratorConfig::default()).expect("orchestrator");
    // Skip the warmup bench so the only allocations are the chunk's.
    // (warm() would run its own bench grid and pollute the pool/peak.)
    let (r, d) = synth_pair_offset_dist(size, size);
    let tasks: Vec<Task> = CHUNK
        .iter()
        .enumerate()
        .map(|(i, &metric)| Task {
            task_id: 150_000 + i as u64,
            ref_data: TaskData::Srgb8(r.clone()),
            dist_data: TaskData::Srgb8(d.clone()),
            width: size,
            height: size,
            metric,
            params: None,
            ref_hash: 0,
        })
        .collect();

    let results: Vec<_> = orch.run_all(tasks).collect();
    let ok = results.iter().filter(|r| r.outcome.is_ok()).count();
    let swaps = swap_vram_reclaim_count();
    println!("CHILD_DONE ok={ok}/{} swap_reclaims={swaps}", results.len());
}

// ===================================================================
// Driver — spawns a child per variant, polls nvidia-smi peak.
// ===================================================================

struct VariantResult {
    variant: String,
    baseline_mib: u64,
    peak_mib: u64,
    peak_delta_mib: i64,
    ok: usize,
    total: usize,
    swap_reclaims: usize,
}

fn run_variant(self_exe: &std::path::Path, size: u32, reclaim_on: bool) -> VariantResult {
    let variant = if reclaim_on { "reclaim" } else { "no_reclaim" }.to_string();

    // Let the card settle, capture baseline.
    thread::sleep(Duration::from_millis(500));
    let baseline = nvidia_smi_used_mib().unwrap_or(0);

    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(baseline));
    let stop2 = stop.clone();
    let peak2 = peak.clone();
    let poller = thread::spawn(move || {
        while !stop2.load(Ordering::Relaxed) {
            if let Some(m) = nvidia_smi_used_mib() {
                peak2.fetch_max(m, Ordering::Relaxed);
            }
            thread::sleep(Duration::from_millis(40));
        }
    });

    let mut cmd = Command::new(self_exe);
    cmd.arg("--child")
        .arg(size.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    if !reclaim_on {
        cmd.env("ZENMETRICS_NO_SWAP_VRAM_CLEANUP", "1");
    }
    let out = cmd.output().expect("spawn child");

    stop.store(true, Ordering::Relaxed);
    let _ = poller.join();

    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut ok = 0;
    let mut total = 0;
    let mut swap_reclaims = 0;
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("CHILD_DONE ") {
            for kv in rest.split_whitespace() {
                if let Some(v) = kv.strip_prefix("ok=") {
                    if let Some((a, b)) = v.split_once('/') {
                        ok = a.parse().unwrap_or(0);
                        total = b.parse().unwrap_or(0);
                    }
                } else if let Some(v) = kv.strip_prefix("swap_reclaims=") {
                    swap_reclaims = v.parse().unwrap_or(0);
                }
            }
        }
    }

    let peak_mib = peak.load(Ordering::Relaxed);
    VariantResult {
        variant,
        baseline_mib: baseline,
        peak_mib,
        peak_delta_mib: peak_mib as i64 - baseline as i64,
        ok,
        total,
        swap_reclaims,
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 3 && args[1] == "--child" {
        let size: u32 = args[2].parse().expect("size");
        run_child(size);
        return;
    }

    let mut size = 4096u32;
    let mut tsv: Option<String> = None;
    let mut iter = args.iter().skip(1);
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--size" => {
                if let Some(v) = iter.next() {
                    size = v.parse().expect("size");
                }
            }
            "--tsv" => tsv = iter.next().cloned(),
            _ => {}
        }
    }

    let self_exe = env::current_exe().expect("current exe");
    println!(
        "# vram_swap_peak — chunk {:?} at {size}x{size}",
        CHUNK.iter().map(|m| m.tag()).collect::<Vec<_>>()
    );
    println!("# GPU0 used now: {:?} MiB", nvidia_smi_used_mib());

    // Run reclaim variant first, then no-reclaim, each in a cold child.
    let with = run_variant(&self_exe, size, true);
    thread::sleep(Duration::from_millis(800));
    let without = run_variant(&self_exe, size, false);

    for v in [&with, &without] {
        println!(
            "  variant={:<11} baseline={} peak={} peak_delta={:+} MiB  ok={}/{}  swap_reclaims={}",
            v.variant, v.baseline_mib, v.peak_mib, v.peak_delta_mib, v.ok, v.total, v.swap_reclaims
        );
    }

    println!("\n== SUMMARY ({size}x{size}) ==");
    println!(
        "  peak WITH swap reclaim    : {:+} MiB  (~MAX single metric; {} reclaims fired)",
        with.peak_delta_mib, with.swap_reclaims
    );
    println!(
        "  peak WITHOUT swap reclaim : {:+} MiB  (~SUM; {} reclaims fired)",
        without.peak_delta_mib, without.swap_reclaims
    );
    if without.peak_delta_mib > with.peak_delta_mib {
        let saved = without.peak_delta_mib - with.peak_delta_mib;
        println!(
            "  => swap reclaim cut peak by {saved} MiB ({:.2}x lower)",
            without.peak_delta_mib as f64 / (with.peak_delta_mib.max(1)) as f64
        );
    }

    if let Some(path) = tsv {
        use std::io::Write;
        let mut f = std::fs::File::create(&path).expect("create tsv");
        writeln!(f, "variant\twidth\theight\tbaseline_mib\tpeak_mib\tpeak_delta_mib\tok\ttotal\tswap_reclaims").unwrap();
        for v in [&with, &without] {
            writeln!(
                f,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                v.variant,
                size,
                size,
                v.baseline_mib,
                v.peak_mib,
                v.peak_delta_mib,
                v.ok,
                v.total,
                v.swap_reclaims
            )
            .unwrap();
        }
        println!("\n# TSV at {path}");
    }
}
