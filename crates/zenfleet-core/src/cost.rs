//! Cost model (goals B & F): each worker self-reports its own rate in its heartbeat, so the
//! dashboard sums `rate × uptime` with no per-provider billing-API integration — Oracle/basement
//! report `0.0`, vast reports its `dph`, Hetzner its hourly. Cost-per-1000-jobs *per tier* is the
//! measured number that says which tier is actually cheapest for the real workload. A budget breach
//! drives stop-spend (auto-teardown of paid tiers).

use serde::{Deserialize, Serialize};

use crate::job::ResourceClass;

/// A worker's self-reported cost + productivity (from its heartbeat).
///
/// The `gpu_util_pct` / `cpu_util_pct` / `last_report_unix_secs` fields are
/// `#[serde(default)]` so heartbeats written by older workers (which omit them)
/// still deserialize — they simply read back as `None`, and the idle detector
/// (`crate::idle`) skips the checks it can't make. New workers SHOULD populate
/// them so the fleet can be flagged for underutilization and staleness.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkerReport {
    pub worker: String,
    pub provider: String,
    pub class: ResourceClass,
    /// 0.0 for free tiers (Oracle always-free; basement amortized as power_kW × $/kWh if desired).
    pub rate_usd_per_hr: f64,
    pub uptime_secs: u64,
    pub jobs_done: u64,
    /// Mean GPU utilization % at last sample (the worker already samples this for AIMD; now it
    /// reports it up). `None` = not a GPU box / not reported.
    #[serde(default)]
    pub gpu_util_pct: Option<u8>,
    /// Mean CPU utilization %. `None` = not reported.
    #[serde(default)]
    pub cpu_util_pct: Option<u8>,
    /// Unix seconds when this report was written — lets the dashboard tell a frozen worker from a
    /// busy one ("last seen"). `None` = not stamped (staleness check skipped).
    #[serde(default)]
    pub last_report_unix_secs: Option<u64>,
}

impl WorkerReport {
    pub fn spent_usd(&self) -> f64 {
        self.rate_usd_per_hr * (self.uptime_secs as f64 / 3600.0)
    }

    /// Jobs completed per hour so far (0 if no uptime yet). The throughput signal the idle detector
    /// uses to flag a paid box that's producing ~nothing — computable from existing data alone.
    pub fn jobs_per_hr(&self) -> f64 {
        if self.uptime_secs == 0 {
            0.0
        } else {
            self.jobs_done as f64 / self.uptime_secs as f64 * 3600.0
        }
    }
}

/// Fleet-wide cost rollup for the dashboard.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FleetCost {
    pub total_spent_usd: f64,
    /// Sum of rates of currently-reporting workers — the live burn rate.
    pub burn_usd_per_hr: f64,
    pub jobs_done: u64,
}

impl FleetCost {
    /// $ per 1000 jobs across the fleet (None until something is done).
    pub fn cost_per_1000_jobs(&self) -> Option<f64> {
        if self.jobs_done == 0 {
            None
        } else {
            Some(self.total_spent_usd / self.jobs_done as f64 * 1000.0)
        }
    }
}

/// Aggregate worker self-reports into a fleet cost view.
pub fn aggregate(reports: &[WorkerReport]) -> FleetCost {
    let mut c = FleetCost::default();
    for r in reports {
        c.total_spent_usd += r.spent_usd();
        c.burn_usd_per_hr += r.rate_usd_per_hr;
        c.jobs_done += r.jobs_done;
    }
    c
}

/// Per-tier cost-per-1000-jobs — surfaces which resource class is actually cheapest for this work.
pub fn cost_per_1000_by_tier(reports: &[WorkerReport]) -> Vec<(ResourceClass, Option<f64>)> {
    [
        ResourceClass::CpuLight,
        ResourceClass::CpuHeavy,
        ResourceClass::CpuArm,
        ResourceClass::Gpu,
        ResourceClass::HighRam,
    ]
    .into_iter()
    .map(|t| {
        let (spent, jobs) = reports
            .iter()
            .filter(|r| r.class == t)
            .fold((0.0_f64, 0u64), |(s, j), r| {
                (s + r.spent_usd(), j + r.jobs_done)
            });
        let per_k = (jobs > 0).then(|| spent / jobs as f64 * 1000.0);
        (t, per_k)
    })
    .collect()
}

/// Stop-spend gate (goals C & F): is cumulative spend at or over the cap?
pub fn over_budget(total_spent_usd: f64, cap_usd: f64) -> bool {
    total_spent_usd >= cap_usd
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn rep(class: ResourceClass, rate: f64, uptime: u64, jobs: u64) -> WorkerReport {
        WorkerReport {
            worker: "w".into(),
            provider: "p".into(),
            class,
            rate_usd_per_hr: rate,
            uptime_secs: uptime,
            jobs_done: jobs,
            gpu_util_pct: None,
            cpu_util_pct: None,
            last_report_unix_secs: None,
        }
    }

    #[test]
    fn spent_is_rate_times_hours() {
        assert!((rep(ResourceClass::Gpu, 0.30, 3600, 0).spent_usd() - 0.30).abs() < EPS);
        assert!((rep(ResourceClass::Gpu, 0.30, 1800, 0).spent_usd() - 0.15).abs() < EPS);
        // free tier costs nothing regardless of uptime
        assert!(
            rep(ResourceClass::CpuArm, 0.0, 999_999, 0)
                .spent_usd()
                .abs()
                < EPS
        );
    }

    #[test]
    fn aggregate_sums() {
        let reports = [
            rep(ResourceClass::Gpu, 0.40, 3600, 100),  // $0.40, 100 jobs
            rep(ResourceClass::CpuArm, 0.0, 3600, 50), // free Oracle, 50 jobs
        ];
        let c = aggregate(&reports);
        assert!((c.total_spent_usd - 0.40).abs() < EPS);
        assert!((c.burn_usd_per_hr - 0.40).abs() < EPS);
        assert_eq!(c.jobs_done, 150);
        // $0.40 / 150 jobs * 1000 = $2.6667 per 1000
        assert!((c.cost_per_1000_jobs().unwrap() - (0.40 / 150.0 * 1000.0)).abs() < EPS);
    }

    #[test]
    fn cost_per_1000_none_when_idle() {
        assert!(aggregate(&[]).cost_per_1000_jobs().is_none());
    }

    #[test]
    fn per_tier_isolates_classes() {
        let reports = [
            rep(ResourceClass::Gpu, 0.50, 3600, 100), // GPU: $0.50/100 → $5/1000
            rep(ResourceClass::CpuArm, 0.0, 3600, 100), // free ARM: $0/1000
        ];
        let by_tier = cost_per_1000_by_tier(&reports);
        let gpu = by_tier
            .iter()
            .find(|(t, _)| *t == ResourceClass::Gpu)
            .unwrap()
            .1;
        let arm = by_tier
            .iter()
            .find(|(t, _)| *t == ResourceClass::CpuArm)
            .unwrap()
            .1;
        let light = by_tier
            .iter()
            .find(|(t, _)| *t == ResourceClass::CpuLight)
            .unwrap()
            .1;
        assert!((gpu.unwrap() - 5.0).abs() < EPS);
        assert!(arm.unwrap().abs() < EPS); // free tier is cheapest
        assert!(
            light.is_none(),
            "a tier with no workers has no cost-per-1000"
        );
    }

    #[test]
    fn budget_gate() {
        assert!(!over_budget(4.99, 5.0));
        assert!(over_budget(5.0, 5.0));
        assert!(over_budget(7.5, 5.0));
    }
}
