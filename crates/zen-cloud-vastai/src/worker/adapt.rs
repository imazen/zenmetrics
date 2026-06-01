//! Adaptive concurrency controller.
//!
//! ## Why AIMD
//!
//! The fleet runs on heterogeneous hardware (6-vCPU GTX 1660 boxes
//! through 24-vCPU Titan Xp). A fixed PARALLEL_CHUNKS value either
//! starves the big boxes or thrashes the small ones. The 2026-05-19
//! six-box snapshot showed GPU util at 0-6% across all boxes — the
//! workload is CPU-bound on most offers, so the right PC is higher
//! than the fixed default the bash version shipped.
//!
//! Additive-increase, multiplicative-decrease (AIMD) — the same
//! algorithm TCP uses for congestion control — auto-tunes the
//! in-flight count by sampling GPU util. Two thresholds:
//!
//! - **Below `ramp_up`**: increment in-flight count by 1.
//! - **Above `back_off`**: decrement by 1 (we're saturating GPU).
//!
//! The gap between thresholds is a hysteresis band that prevents
//! oscillation. Defaults: ramp_up=30%, back_off=90% — a 60-point
//! gap that's wide enough to be stable but narrow enough to find
//! a useful operating point on most boxes.
//!
//! ## Why semaphore deltas instead of recreating the bound
//!
//! `tokio::sync::Semaphore::add_permits(n)` is the documented way
//! to grow capacity; `forget_permits(n)` permanently consumes
//! permits without releasing them (the inverse of `add_permits`).
//! These are O(1), don't require recreating the semaphore, and
//! don't block currently-in-flight tasks.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use tokio::process::Command;
use tokio::sync::Semaphore;
use tokio::time;
use tracing::{debug, info, warn};

/// Initial PARALLEL_CHUNKS heuristic from host specs.
///
/// 2026-05-22 update: when `ZENSIM_FEATURES_REGIME` is set to a
/// feature-extracting regime (`with-iw` / `withiw` / `iw` —
/// 372 features), default PC=1. The WithIw regime's persist planes
/// can require single allocations up to ~1.5 GB for an 8.7 MPixel
/// source image, and cubecl-cuda's pool cannot satisfy 1.5 GB
/// requests when N concurrent chunks have already reserved their
/// own pool slots. Empirically (v26 smoke #1 on RTX 3090 24 GB),
/// PC=4 with WithIw panicked every cell of every group at
/// cubecl-cuda/src/compute/server.rs:124 with "can't allocate buffer
/// of size: 1581133824".
///
/// For other regimes (Basic 228 / Extended 300) the planes are
/// smaller (~200 MB at 1080p) and the legacy 2 GB-per-chunk
/// budget heuristic still holds.
pub fn auto_parallel_chunks() -> usize {
    if regime_is_withiw() {
        return 1;
    }
    let cores = num_cpus();
    let gpu_mb = nvidia_smi_total_memory_mb().unwrap_or(4096);
    let pc_cpu = cores / 4;
    let pc_gpu = (gpu_mb / 2048) as usize;
    let pc = pc_cpu.min(pc_gpu);
    pc.clamp(1, 4)
}

/// Hard ceiling on PARALLEL_CHUNKS for the AIMD loop. Beyond this
/// the rayon thread pool inside zen-metrics oversubscribes the box
/// (each `--jobs 0` sweep uses all cores; 5+ concurrent = 5×N
/// rayon threads for N cores).
///
/// Same WithIw caveat as [`auto_parallel_chunks`]: when the feature
/// regime requires large persist planes, the ceiling is also 1 so
/// AIMD cannot ramp into the territory where the cubecl pool runs
/// out of contiguous space.
pub fn derive_pc_max() -> usize {
    if regime_is_withiw() {
        return 1;
    }
    let cores = num_cpus();
    (cores / 2).clamp(1, 8)
}

/// Returns true when `ZENSIM_FEATURES_REGIME` selects the WithIw
/// 372-feature regime (or its aliases). Matches the same parsing
/// rules as [`crate::worker::inline::parse_regime_env_or_default`].
fn regime_is_withiw() -> bool {
    match std::env::var("ZENSIM_FEATURES_REGIME") {
        Ok(s) => matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "with-iw" | "with_iw" | "withiw" | "iw"
        ),
        Err(_) => false,
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Parse `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits`.
/// Returns the first GPU's VRAM in MB, or None if nvidia-smi isn't
/// available (e.g. running on a developer laptop).
pub fn nvidia_smi_total_memory_mb() -> Option<u32> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    s.lines().next()?.trim().parse().ok()
}

/// Sample `nvidia-smi` for 5x 1s polls of GPU util, return the average.
/// Returns 0 if nvidia-smi unavailable.
pub async fn sample_gpu_util_avg() -> u32 {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
            "-lms",
            "1000",
            "-c",
            "5",
        ])
        .output()
        .await
        .ok();
    let Some(o) = out else { return 0 };
    if !o.status.success() {
        return 0;
    }
    let s = String::from_utf8_lossy(&o.stdout);
    let (sum, n) = s
        .lines()
        .filter_map(|l| l.trim().parse::<u32>().ok())
        .fold((0u32, 0u32), |(s, n), v| (s + v, n + 1));
    sum.checked_div(n).unwrap_or(0)
}

/// Controller that owns the Semaphore's capacity. The dispatcher
/// `acquire`s permits as normal; the AIMD loop adjusts the total
/// permit count via `bump()` / `drop_one()`.
///
/// The current PC is tracked in an `AtomicUsize` so the dispatcher
/// can also observe it for logging.
pub struct PcController {
    pc_max: usize,
    current: AtomicUsize,
    sem: Arc<Semaphore>,
}

impl PcController {
    pub fn new(initial: usize, pc_max: usize, sem: Arc<Semaphore>) -> Self {
        Self {
            pc_max,
            current: AtomicUsize::new(initial),
            sem,
        }
    }

    pub fn current(&self) -> usize {
        self.current.load(Ordering::Relaxed)
    }

    /// Increase capacity by one if room remains. Returns true on bump.
    fn bump(&self) -> bool {
        let prev = self.current.load(Ordering::Relaxed);
        if prev >= self.pc_max {
            return false;
        }
        self.current.store(prev + 1, Ordering::Relaxed);
        self.sem.add_permits(1);
        true
    }

    /// Decrease capacity by one. Permanently consumes a permit. The
    /// in-flight task currently holding the consumed permit is
    /// unaffected (the back-off applies to FUTURE acquires, not
    /// running tasks — graceful drain).
    async fn drop_one(&self) -> bool {
        let prev = self.current.load(Ordering::Relaxed);
        if prev <= 1 {
            return false;
        }
        // `acquire_owned + forget` permanently consumes one permit.
        match self.sem.clone().try_acquire_owned() {
            Ok(p) => {
                self.current.store(prev - 1, Ordering::Relaxed);
                p.forget();
                true
            }
            Err(_) => {
                // No idle permit — all in-flight. Wait for one.
                match self.sem.clone().acquire_owned().await {
                    Ok(p) => {
                        self.current.store(prev - 1, Ordering::Relaxed);
                        p.forget();
                        true
                    }
                    Err(_) => false,
                }
            }
        }
    }
}

/// Run the AIMD loop forever (until aborted by the dispatcher on
/// shutdown). Samples GPU util every `interval`, applies the
/// ramp_up / back_off rule, updates the controller.
pub async fn run_aimd_loop(
    ctrl: Arc<PcController>,
    interval: Duration,
    ramp_up_below: u32,
    back_off_above: u32,
) {
    let mut tick = time::interval(interval);
    // Skip the initial tick so we don't bump PC before any work has
    // even started — gpu_util reads 0 on a freshly-booted box.
    tick.tick().await;
    loop {
        tick.tick().await;
        let util = sample_gpu_util_avg().await;
        let pc = ctrl.current();
        if util < ramp_up_below && pc < ctrl.pc_max {
            if ctrl.bump() {
                info!(util, pc_old = pc, pc_new = ctrl.current(), "AIMD ramp-up");
            }
        } else if util > back_off_above && pc > 1 {
            if ctrl.drop_one().await {
                info!(util, pc_old = pc, pc_new = ctrl.current(), "AIMD back-off");
            } else {
                warn!(util, "AIMD wanted to back off but couldn't drop a permit");
            }
        } else {
            debug!(util, pc, "AIMD hold");
        }
    }
}
