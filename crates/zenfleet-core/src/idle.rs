//! Idle / underutilized-infrastructure detection — the money-leak guard.
//!
//! ONE canonical detector so "idle" means the same thing in every fleet tool
//! (the dashboard's `FleetStalled`/`Underutilized` notifications, `jobctl`, the
//! python/bash watch scripts). A paid box that claimed two jobs then sat at 0%
//! GPU for an hour is exactly what this flags — the failure the fleet was blind
//! to before (GPU util was sampled on-box and discarded, `FleetStalled` was
//! declared but never emitted).
//!
//! It works on data that already exists in [`WorkerReport`]:
//! - **throughput** from `jobs_done / uptime_secs` (no new plumbing needed), and
//! - **staleness** from `last_report_unix_secs` ("last seen"),
//! and is *enriched* by `gpu_util_pct` once the worker reports the util it
//! already samples. Missing signals are skipped, never guessed.

use serde::{Deserialize, Serialize};

use crate::cost::WorkerReport;
use crate::job::ResourceClass;

/// Tunable thresholds. Defaults are deliberately conservative so a real idle box
/// is caught while a warming-up or briefly-bursty box is not.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct IdleThresholds {
    /// No heartbeat in at least this long ⇒ STALE (frozen/dead). 0 disables.
    pub stale_heartbeat_secs: u64,
    /// GPU utilization at or below this % (when reported) on a GPU box ⇒ underutilized.
    pub min_gpu_util_pct: u8,
    /// Jobs/hour at or below this on a PAID box ⇒ starved (producing ~nothing for money).
    pub min_jobs_per_hr: f64,
    /// Ignore boxes younger than this — they may still be pulling the image / warming up.
    pub grace_secs: u64,
}

impl Default for IdleThresholds {
    fn default() -> Self {
        Self {
            stale_heartbeat_secs: 180,
            min_gpu_util_pct: 10,
            min_jobs_per_hr: 1.0,
            grace_secs: 120,
        }
    }
}

/// How bad. A paid box wasting money is `Critical`; a free/basement box just idling is `Warn`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Warn,
    Critical,
}

/// Why a worker is flagged.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum IdleReason {
    /// No heartbeat for `secs_since_report` — the worker is frozen or dead.
    StaleHeartbeat { secs_since_report: u64 },
    /// GPU box running at `pct`% utilization — paying for a GPU that isn't working.
    LowGpuUtil { pct: u8 },
    /// Paid box completing `jobs_per_hr` jobs/hour — burning money for ~no output.
    Starved { jobs_per_hr: f64 },
}

impl IdleReason {
    pub fn describe(&self) -> String {
        match self {
            IdleReason::StaleHeartbeat { secs_since_report } => {
                format!("no heartbeat for {secs_since_report}s (frozen/dead)")
            }
            IdleReason::LowGpuUtil { pct } => format!("GPU at {pct}% util (idle)"),
            IdleReason::Starved { jobs_per_hr } => {
                format!("{jobs_per_hr:.2} jobs/hr (producing ~nothing)")
            }
        }
    }
}

/// One flagged worker. A paid one is burning `wasted_usd_per_hr` while idle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IdleWarning {
    pub worker: String,
    pub provider: String,
    pub reason: IdleReason,
    /// What this box costs per hour while idle (0.0 for free tiers — still flagged, just not $-bleeding).
    pub wasted_usd_per_hr: f64,
    pub severity: Severity,
}

impl IdleWarning {
    /// A one-line human warning, e.g. `⚠ CRITICAL vast/box-3: GPU at 0% util (idle) — $0.40/hr wasted`.
    pub fn line(&self) -> String {
        let tag = match self.severity {
            Severity::Critical => "CRITICAL",
            Severity::Warn => "warn",
        };
        let cost = if self.wasted_usd_per_hr > 0.0 {
            format!(" — ${:.2}/hr wasted", self.wasted_usd_per_hr)
        } else {
            String::new()
        };
        format!(
            "⚠ {tag} {}/{}: {}{cost}",
            self.provider,
            self.worker,
            self.reason.describe()
        )
    }
}

/// Flag every idle / underutilized worker in `reports`.
///
/// `now_unix` is the current Unix time for the staleness check; pass 0 if unknown
/// (staleness is then skipped, throughput + util still apply). Each worker yields
/// at most one warning, in priority order: stale ▸ low-GPU ▸ starved.
pub fn detect_idle(reports: &[WorkerReport], now_unix: u64, th: &IdleThresholds) -> Vec<IdleWarning> {
    let mut out = Vec::new();
    for r in reports {
        // Warming up — don't cry wolf on a box that just booted.
        if r.uptime_secs < th.grace_secs {
            continue;
        }
        let paid = r.rate_usd_per_hr > 0.0;
        let warn = |reason: IdleReason, severity: Severity| IdleWarning {
            worker: r.worker.clone(),
            provider: r.provider.clone(),
            reason,
            wasted_usd_per_hr: r.rate_usd_per_hr,
            severity,
        };

        // 1) Stale heartbeat — frozen/dead. Dominates (don't also flag throughput).
        if th.stale_heartbeat_secs > 0
            && now_unix > 0
            && let Some(last) = r.last_report_unix_secs
            && last > 0
            && now_unix.saturating_sub(last) > th.stale_heartbeat_secs
        {
            out.push(warn(
                IdleReason::StaleHeartbeat {
                    secs_since_report: now_unix - last,
                },
                Severity::Critical,
            ));
            continue;
        }

        // 2) GPU box with a util reading — that reading is AUTHORITATIVE: low ⇒ idle (flag),
        //    high ⇒ busy (a long job can show few completed jobs/hr, so don't fall through to the
        //    throughput check and false-flag it).
        if r.class == ResourceClass::Gpu
            && let Some(util) = r.gpu_util_pct
        {
            if util <= th.min_gpu_util_pct {
                out.push(warn(
                    IdleReason::LowGpuUtil { pct: util },
                    if paid { Severity::Critical } else { Severity::Warn },
                ));
            }
            continue;
        }

        // 3) Producing ~nothing (throughput from existing jobs_done/uptime). Flags BOTH paid boxes
        //    (money for no output) and free/basement ones (a dead or stuck worker). Warn-level
        //    because throughput is a soft signal (one very long job looks the same as idle); the
        //    `wasted_usd_per_hr` field carries the $ so a paid starved box still stands out.
        if r.jobs_per_hr() <= th.min_jobs_per_hr {
            out.push(warn(
                IdleReason::Starved {
                    jobs_per_hr: r.jobs_per_hr(),
                },
                Severity::Warn,
            ));
        }
    }
    out
}

/// Total $/hr being burned by flagged PAID boxes — the headline "you are wasting $X/hr" number.
pub fn wasted_burn_usd_per_hr(warnings: &[IdleWarning]) -> f64 {
    warnings.iter().map(|w| w.wasted_usd_per_hr).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rep(
        worker: &str,
        class: ResourceClass,
        rate: f64,
        uptime: u64,
        jobs: u64,
        gpu: Option<u8>,
        last: Option<u64>,
    ) -> WorkerReport {
        WorkerReport {
            worker: worker.into(),
            provider: "vast".into(),
            class,
            rate_usd_per_hr: rate,
            uptime_secs: uptime,
            jobs_done: jobs,
            gpu_util_pct: gpu,
            cpu_util_pct: None,
            last_report_unix_secs: last,
        }
    }

    #[test]
    fn paid_gpu_box_at_zero_util_is_critical() {
        // 1hr uptime, 2 jobs, 0% GPU, $0.40/hr.
        let w = rep("box-3", ResourceClass::Gpu, 0.40, 3600, 2, Some(0), None);
        let warns = detect_idle(&[w], 0, &IdleThresholds::default());
        assert_eq!(warns.len(), 1);
        assert!(matches!(warns[0].reason, IdleReason::LowGpuUtil { pct: 0 }));
        assert_eq!(warns[0].severity, Severity::Critical);
        assert!((warns[0].wasted_usd_per_hr - 0.40).abs() < 1e-9);
    }

    #[test]
    fn starved_paid_box_flagged_from_throughput_alone() {
        // No util reported, but 1hr uptime and 0 jobs on a paid box = starved.
        let w = rep("box-1", ResourceClass::CpuHeavy, 0.10, 3600, 0, None, None);
        let warns = detect_idle(&[w], 0, &IdleThresholds::default());
        assert_eq!(warns.len(), 1);
        assert!(matches!(warns[0].reason, IdleReason::Starved { .. }));
    }

    #[test]
    fn stale_heartbeat_dominates_and_is_critical() {
        // Last seen 600s ago (> 180s default), now = 1_000_000.
        let w = rep("box-7", ResourceClass::Gpu, 0.30, 3600, 0, Some(0), Some(1_000_000 - 600));
        let warns = detect_idle(&[w], 1_000_000, &IdleThresholds::default());
        assert_eq!(warns.len(), 1);
        assert!(matches!(warns[0].reason, IdleReason::StaleHeartbeat { .. }));
        assert_eq!(warns[0].severity, Severity::Critical);
    }

    #[test]
    fn busy_box_not_flagged() {
        // 60 jobs/hr, GPU at 85%, fresh heartbeat — healthy.
        let w = rep("box-9", ResourceClass::Gpu, 0.40, 3600, 60, Some(85), Some(1_000_000 - 5));
        assert!(detect_idle(&[w], 1_000_000, &IdleThresholds::default()).is_empty());
    }

    #[test]
    fn warming_up_box_is_spared() {
        // Only 30s uptime (< 120s grace) with 0 jobs — don't cry wolf.
        let w = rep("box-new", ResourceClass::Gpu, 0.40, 30, 0, Some(0), None);
        assert!(detect_idle(&[w], 0, &IdleThresholds::default()).is_empty());
    }

    #[test]
    fn free_tier_idle_is_warn_not_critical_and_wastes_nothing() {
        let w = rep("oracle-arm", ResourceClass::CpuArm, 0.0, 3600, 0, Some(0), None);
        let warns = detect_idle(&[w], 0, &IdleThresholds::default());
        // free GPU-less box: starved-throughput path, Warn, $0 wasted
        assert_eq!(warns.len(), 1);
        assert_eq!(warns[0].severity, Severity::Warn);
        assert_eq!(warns[0].wasted_usd_per_hr, 0.0);
    }

    #[test]
    fn wasted_burn_sums_paid_only() {
        let reports = [
            rep("a", ResourceClass::Gpu, 0.40, 3600, 0, Some(0), None),
            rep("b", ResourceClass::CpuArm, 0.0, 3600, 0, Some(0), None),
            rep("c", ResourceClass::Gpu, 0.25, 3600, 0, Some(2), None),
        ];
        let warns = detect_idle(&reports, 0, &IdleThresholds::default());
        // 0.40 + 0.0 + 0.25 = 0.65/hr being wasted
        assert!((wasted_burn_usd_per_hr(&warns) - 0.65).abs() < 1e-9);
    }
}
