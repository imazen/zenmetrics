//! Control intents (goal C) — pure computation of what an action *would* do. The dashboard renders
//! these; the fleet layer actuates them. GC is always a dry-run preview before any delete.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use zen_job_core::{gc_plan, over_budget, BlobIndexEntry, Sha256Hex, WorkerReport};

/// GC dry-run preview (goal C "clean with dry-run preview"): how many blobs and how many bytes each
/// verdict bucket covers, *before* anything is deleted.
#[derive(Serialize, Debug, PartialEq, Eq)]
pub struct GcDryRun {
    pub kept: usize,
    pub evict_cheap: usize,
    pub evict_under_pressure: usize,
    pub refuse_surface: usize,
    pub freed_cheap_bytes: u64,
    pub freed_under_pressure_bytes: u64,
    pub refused_bytes: u64,
}

pub fn gc_dry_run(
    index: &[BlobIndexEntry],
    referenced: &HashSet<Sha256Hex>,
    roots: &HashSet<Sha256Hex>,
) -> GcDryRun {
    let plan = gc_plan(index, referenced, roots);
    let bytes_of = |shas: &[Sha256Hex]| -> u64 {
        let set: HashSet<&Sha256Hex> = shas.iter().collect();
        index.iter().filter(|e| set.contains(&e.sha)).map(|e| e.size).sum()
    };
    GcDryRun {
        kept: plan.keep.len(),
        evict_cheap: plan.evict_cheap.len(),
        evict_under_pressure: plan.evict_under_pressure.len(),
        refuse_surface: plan.refuse_surface.len(),
        freed_cheap_bytes: bytes_of(&plan.evict_cheap),
        freed_under_pressure_bytes: bytes_of(&plan.evict_under_pressure),
        refused_bytes: bytes_of(&plan.refuse_surface),
    }
}

/// Stop-spend decision (goals C & F): if cumulative spend is over the cap, every paid worker is torn
/// down; free-tier workers (rate 0) keep draining.
#[derive(Serialize, Debug, PartialEq, Eq)]
pub struct StopSpendDecision {
    pub over_budget: bool,
    pub tear_down: Vec<String>,
    pub keep_free: Vec<String>,
}

pub fn stop_spend(workers: &[WorkerReport], spent_usd: f64, cap_usd: f64) -> StopSpendDecision {
    let over = over_budget(spent_usd, cap_usd);
    let mut tear_down = Vec::new();
    let mut keep_free = Vec::new();
    for w in workers {
        if w.rate_usd_per_hr > 0.0 {
            if over {
                tear_down.push(w.worker.clone());
            }
        } else {
            keep_free.push(w.worker.clone());
        }
    }
    StopSpendDecision { over_budget: over, tear_down, keep_free }
}

/// The control surface the dashboard exposes (goal C). The fleet layer consumes these to actuate;
/// the dashboard itself only computes/echoes the resulting plan.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ControlIntent {
    KillFleet,
    KillTier { tier: String },
    KillRun { run: String },
    StopSpend { cap_usd: f64 },
    GcDryRun,
    Pause { run: String },
    Drain { run: String },
    Resume { run: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use zen_job_core::{sha256, Regenerability, ResourceClass};

    fn blob(b: &[u8], size: u64, regen: Regenerability) -> BlobIndexEntry {
        BlobIndexEntry { sha: sha256(b), size, regenerability: regen, last_ref_secs: 0 }
    }

    #[test]
    fn gc_dry_run_counts_bytes_per_bucket() {
        let referenced: HashSet<Sha256Hex> = [sha256(b"keep")].into_iter().collect();
        let index = vec![
            blob(b"keep", 10, Regenerability::CheapRegenerable),
            blob(b"jpeg-orphan", 500, Regenerability::CheapRegenerable),
            blob(b"avif-orphan", 9000, Regenerability::ExpensiveRegenerable),
            blob(b"src-orphan", 1_000_000, Regenerability::NotRegenerable),
        ];
        let dr = gc_dry_run(&index, &referenced, &HashSet::new());
        assert_eq!(dr.kept, 1);
        assert_eq!(dr.freed_cheap_bytes, 500, "only the orphan jpeg is freely evictable");
        assert_eq!(dr.freed_under_pressure_bytes, 9000);
        assert_eq!(dr.refused_bytes, 1_000_000, "irreplaceable orphan is surfaced, not freed");
    }

    #[test]
    fn stop_spend_tears_down_paid_keeps_free() {
        let workers = vec![
            WorkerReport { worker: "vast1".into(), provider: "vast".into(), class: ResourceClass::Gpu, rate_usd_per_hr: 0.4, uptime_secs: 3600, jobs_done: 10 },
            WorkerReport { worker: "oracle1".into(), provider: "oracle".into(), class: ResourceClass::CpuArm, rate_usd_per_hr: 0.0, uptime_secs: 3600, jobs_done: 5 },
        ];
        let over = stop_spend(&workers, 10.0, 5.0);
        assert!(over.over_budget);
        assert_eq!(over.tear_down, vec!["vast1".to_string()]);
        assert_eq!(over.keep_free, vec!["oracle1".to_string()]);

        let under = stop_spend(&workers, 1.0, 5.0);
        assert!(!under.over_budget);
        assert!(under.tear_down.is_empty(), "under budget → tear down nothing");
        assert_eq!(under.keep_free, vec!["oracle1".to_string()]);
    }

    #[test]
    fn control_intent_serde() {
        let i = ControlIntent::StopSpend { cap_usd: 25.0 };
        let j = serde_json::to_string(&i).unwrap();
        assert!(j.contains("\"action\":\"stop_spend\""));
        assert_eq!(serde_json::from_str::<ControlIntent>(&j).unwrap(), i);
    }
}
