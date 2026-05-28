//! # zenfleet-orchestrator
//!
//! Provider-generic fleet orchestration logic, hoisted out of the
//! Salad-specific launcher in `zen-cloud-salad/src/bin/zen-salad-sweep.rs`.
//!
//! This crate owns the **launcher-side** decisions that DO NOT depend on
//! which compute provider (Salad / Vast / RunPod / local) is hosting the
//! fleet:
//!
//! * **Chunk TTL re-dispatch** — when a chunk's sidecar has not landed by
//!   the configured TTL, push it back onto the job queue. Worker-side
//!   idempotency (sidecar HEAD pre-check) ensures a re-dispatched chunk
//!   that is actually in-flight on a slow worker never double-writes —
//!   the second worker exits early. See `inline.rs::process_chunk_inline`
//!   in `zen-cloud-vastai` for the worker side.
//!
//! * **Replicas overshoot** — when the user requests N replicas and the
//!   provider has a quota, provision `min(quota, ceil(N × overshoot))`
//!   so a single dead worker does not gate the sweep. Default 1.7×.
//!
//! * **Class-aware filtering** — load a prior `fleet_summary.json` and
//!   filter GPU classes by observed median boot warmup + mean productive
//!   chunks before re-using them.
//!
//! * **Speculative execution** — once the launcher has observed enough
//!   completed chunks to estimate the per-chunk completion distribution,
//!   identify in-flight chunks whose elapsed time exceeds a multiplier of
//!   the p95 of that distribution and re-dispatch them. This is the
//!   classic MapReduce straggler-mitigation pattern (Dean & Ghemawat
//!   2004 §3.6) adapted for our chunk model. Worker-side idempotency
//!   reconciles duplicates by keeping the OLDEST `worker_chunk_start_unix`.
//!
//! ## What this crate does NOT do
//!
//! * Mint provider credentials (Salad has its own scoped-R2 minter;
//!   Vast.ai has its own ephemeral creds).
//! * Build provider-specific request bodies (container groups, vast
//!   `asks`, etc.).
//! * Read or write R2 directly — the calling launcher does that and
//!   passes us the resulting state.
//! * Run the worker side. Worker code lives in `zen-cloud-vastai/src/worker/`.
//!
//! The Salad bin retains responsibility for provisioning, credential
//! minting, R2 polling, and teardown; it calls into this crate for the
//! launcher-side decisions listed above.
//!
//! ## Public types (skeletal — subject to provider-trait extraction)
//!
//! * `SweepConfig` — all launcher knobs, with the defaults that the
//!   2026-05-28 iter2.5 tuning landed (TTL=360, overshoot=1.7).
//! * `SpeculativeConfig` — speculative-execution tuning.
//! * `SpeculativeState` — running per-chunk first-dispatch timestamps +
//!   the completion-time distribution + already-speculated set.
//! * `DispatchAction` — the decision a poll-tick produces: either
//!   "do nothing", "TTL re-dispatch chunk X", or "speculatively dispatch
//!   chunk X (because it has been in-flight too long)".

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod driver;
pub mod provider;

pub use driver::FleetSweep;
pub use provider::{
    FleetSummary, GroupId, GroupStatus, InstanceStatus, PollResult, ProviderHandle, ProvisionSpec,
    QueueJob, R2Operator, r2_layout,
};

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Top-level configuration for one fleet sweep.
///
/// Defaults reflect the iter2.5 tuning that landed on 2026-05-28: TTL
/// 360 s + replicas overshoot 1.7. Override via the CLI; the launcher's
/// Args struct mirrors these fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepConfig {
    /// Number of worker replicas requested by the user (pre-overshoot).
    pub replicas: u32,
    /// Replicas overshoot multiplier. The provisioned count is
    /// `ceil(replicas × replicas_overshoot)`, then clamped to
    /// `provider_replica_quota`.
    pub replicas_overshoot: f64,
    /// Provider replica quota (e.g. Salad org quota = 10). The launcher
    /// MUST pass the provider's real quota; this crate just clamps.
    pub provider_replica_quota: u32,
    /// Per-chunk TTL in seconds. After this, the launcher re-pushes any
    /// chunk whose sidecar is still missing (capped to one re-push per
    /// chunk via the TTL path).
    pub chunk_ttl_secs: u64,
    /// Number of cells per chunk. Used downstream when stitching the
    /// fleet summary; this crate only carries the value through.
    pub cells_per_chunk: u32,
    /// Filter: drop classes whose median warmup exceeds this many seconds.
    pub max_warmup_secs: u32,
    /// Filter: drop classes whose mean productive chunks fell below this.
    pub min_productive_chunks: f32,
    /// Speculative-execution tuning.
    pub speculative: SpeculativeConfig,
}

impl Default for SweepConfig {
    fn default() -> Self {
        Self {
            replicas: 10,
            replicas_overshoot: 1.7,
            provider_replica_quota: 10,
            chunk_ttl_secs: 360,
            cells_per_chunk: 12,
            max_warmup_secs: 60,
            min_productive_chunks: 2.0,
            speculative: SpeculativeConfig::default(),
        }
    }
}

/// Speculative-execution tuning. Independently disable-able from TTL
/// re-dispatch via `enabled = false`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeConfig {
    /// Master switch — when false, `SpeculativeState::decide_speculative`
    /// always returns `None`. The CLI off-switch `--no-speculative` flips
    /// this.
    pub enabled: bool,
    /// A chunk's elapsed in-flight time must exceed
    /// `p95(completion_secs) × straggler_factor` before we speculatively
    /// re-dispatch it. Default 1.5 (Dean & Ghemawat 2004 used 1.5–2.0
    /// in production at Google MapReduce).
    pub straggler_factor: f64,
    /// Do not produce speculative decisions until at least this many
    /// chunks have completed (i.e. we have enough samples for a stable
    /// p95). Default 3.
    pub min_completed_for_stats: u32,
    /// Maximum number of speculative dispatches per chunk. Default 1.
    /// Worker idempotency makes higher values safe, but they waste
    /// compute.
    pub speculation_cap_per_chunk: u32,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            straggler_factor: 1.5,
            min_completed_for_stats: 3,
            speculation_cap_per_chunk: 1,
        }
    }
}

/// Per-chunk state for the speculative-execution scheduler.
///
/// The launcher creates one `SpeculativeState` at sweep start, calls
/// `record_dispatched` for every chunk push (both initial and TTL
/// re-dispatch), and calls `record_completed` when a sidecar lands.
/// Each poll tick the launcher iterates in-flight chunks and asks
/// `decide_speculative` whether to re-push.
///
/// All times are in seconds since some launcher-chosen epoch
/// (typically `t_post`).
#[derive(Debug, Default)]
pub struct SpeculativeState {
    /// chunk_id → earliest dispatch time (seconds relative to launcher
    /// epoch).
    first_dispatched_at: HashMap<String, f64>,
    /// chunk_id → number of times we have speculatively re-dispatched.
    /// Initial dispatch + TTL dispatch don't count; only speculative
    /// counts go here.
    speculative_count: HashMap<String, u32>,
    /// Completed chunk_ids → completion duration (in seconds, from
    /// first_dispatched_at).
    completed_secs: Vec<f64>,
    /// Set of chunk_ids the launcher has marked completed (so the
    /// scheduler can skip them quickly).
    completed_set: HashSet<String>,
}

impl SpeculativeState {
    /// Construct an empty scheduler state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Note that `chunk_id` was dispatched at `now_secs`. Idempotent on
    /// the chunk_id key — the FIRST dispatch wins.
    pub fn record_dispatched(&mut self, chunk_id: &str, now_secs: f64) {
        self.first_dispatched_at
            .entry(chunk_id.to_string())
            .or_insert(now_secs);
    }

    /// Record that a sidecar for `chunk_id` landed at `now_secs`. The
    /// duration recorded is `now_secs - first_dispatched_at[chunk_id]`,
    /// or zero if we never saw a dispatch (shouldn't happen, but be
    /// defensive). Idempotent on the chunk_id key.
    pub fn record_completed(&mut self, chunk_id: &str, now_secs: f64) {
        if !self.completed_set.insert(chunk_id.to_string()) {
            return;
        }
        let started = self
            .first_dispatched_at
            .get(chunk_id)
            .copied()
            .unwrap_or(now_secs);
        let dur = (now_secs - started).max(0.0);
        self.completed_secs.push(dur);
    }

    /// Number of completed chunks so far.
    pub fn n_completed(&self) -> u32 {
        self.completed_secs.len() as u32
    }

    /// Compute p95 of the completion-time distribution. Returns
    /// `None` until we have at least one sample.
    pub fn p95_completion_secs(&self) -> Option<f64> {
        if self.completed_secs.is_empty() {
            return None;
        }
        let mut sorted = self.completed_secs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Nearest-rank p95. With small n this is generous; that's fine —
        // speculative execution should err on the side of NOT speculating
        // when we have few samples.
        let idx = ((sorted.len() as f64 * 0.95).ceil() as usize).saturating_sub(1);
        Some(sorted[idx.min(sorted.len() - 1)])
    }

    /// Decide whether to speculatively re-dispatch `chunk_id`.
    ///
    /// Returns `Some(elapsed_secs)` to indicate the launcher SHOULD
    /// re-push, where `elapsed_secs` is the chunk's current in-flight
    /// time (the launcher can log it). Returns `None` to do nothing.
    ///
    /// Gates:
    /// 1. `cfg.enabled` must be true.
    /// 2. Chunk must be in-flight (not in `completed_set`).
    /// 3. We have at least `cfg.min_completed_for_stats` samples.
    /// 4. `elapsed > p95 × cfg.straggler_factor`.
    /// 5. `speculative_count[chunk_id] < cfg.speculation_cap_per_chunk`.
    pub fn decide_speculative(
        &self,
        chunk_id: &str,
        now_secs: f64,
        cfg: &SpeculativeConfig,
    ) -> Option<f64> {
        if !cfg.enabled {
            return None;
        }
        if self.completed_set.contains(chunk_id) {
            return None;
        }
        let already = self
            .speculative_count
            .get(chunk_id)
            .copied()
            .unwrap_or(0);
        if already >= cfg.speculation_cap_per_chunk {
            return None;
        }
        if self.n_completed() < cfg.min_completed_for_stats {
            return None;
        }
        let p95 = self.p95_completion_secs()?;
        let started = self.first_dispatched_at.get(chunk_id).copied()?;
        let elapsed = (now_secs - started).max(0.0);
        if elapsed > p95 * cfg.straggler_factor {
            Some(elapsed)
        } else {
            None
        }
    }

    /// Note that the launcher acted on a speculative decision for
    /// `chunk_id`. Must be called AFTER `decide_speculative` returns
    /// `Some` AND the re-push succeeded — otherwise the cap counter
    /// drifts.
    pub fn record_speculative_dispatched(&mut self, chunk_id: &str) {
        *self
            .speculative_count
            .entry(chunk_id.to_string())
            .or_insert(0) += 1;
    }

    /// Total speculative dispatches across all chunks. The launcher
    /// stitches this into the final summary.
    pub fn total_speculative_dispatches(&self) -> u32 {
        self.speculative_count.values().sum()
    }
}

/// Compute the provisioned replica count from the user's request +
/// overshoot multiplier + provider quota.
///
/// `min` is always 1 — even with overshoot=0 we provision at least one
/// worker. The launcher caps at `quota` separately.
pub fn compute_provisioned_replicas(replicas: u32, overshoot: f64, quota: u32) -> u32 {
    let overshoot = overshoot.max(1.0);
    let raw = (replicas as f64 * overshoot).ceil() as u32;
    raw.clamp(1, quota.max(1))
}

/// Per-class statistics from a prior `fleet_summary.json` used for the
/// class-aware filter.
///
/// The launcher loads the fleet_summary file, derives one
/// `PriorClassStats` per GPU class, then calls `filter_classes` to
/// produce the kept + dropped lists. Mirrors `ClassStats` in the Salad
/// bin (which is the only producer of fleet_summary today).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorClassStats {
    /// GPU class name (e.g. "RTX 3060", "RTX 5090").
    pub name: String,
    /// Median warmup seconds across replicas of this class in the
    /// prior sweep. `None` when the prior sweep had no completed boot
    /// for this class.
    pub median_warmup_secs: Option<u32>,
    /// Mean productive chunks (omni sidecars) per replica.
    pub mean_chunks_processed: f32,
}

/// Outcome of the class-aware filter: which classes to KEEP and which
/// were DROPPED. Manual `--gpu-classes` override bypasses this
/// (launcher decides not to call filter_classes at all).
#[derive(Debug, Default)]
pub struct ClassFilterOutcome {
    /// Class names that survived the filter (the launcher provisions
    /// these).
    pub keep: Vec<String>,
    /// Class names dropped — both the name and the reason (one-line).
    pub dropped: Vec<(String, String)>,
}

/// Apply the warmup + productive-chunks filter to a candidate class
/// list, given a snapshot of prior stats keyed by class name.
///
/// Classes with no prior signal are KEPT by default (the filter only
/// drops classes for which we have evidence they underperformed).
///
/// `cfg.max_warmup_secs` and `cfg.min_productive_chunks` drive the
/// per-class verdict.
pub fn filter_classes(
    candidates: &[String],
    prior: &HashMap<String, PriorClassStats>,
    cfg: &SweepConfig,
) -> ClassFilterOutcome {
    let mut out = ClassFilterOutcome::default();
    for name in candidates {
        match prior.get(name) {
            None => out.keep.push(name.clone()),
            Some(stats) => {
                let warmup_ok = stats
                    .median_warmup_secs
                    .map(|w| w <= cfg.max_warmup_secs)
                    .unwrap_or(true);
                let prod_ok = stats.mean_chunks_processed >= cfg.min_productive_chunks;
                if warmup_ok && prod_ok {
                    out.keep.push(name.clone());
                } else {
                    let reason = format!(
                        "median_warmup={:?} mean_chunks={:.2} (warmup_ok={} prod_ok={})",
                        stats.median_warmup_secs,
                        stats.mean_chunks_processed,
                        warmup_ok,
                        prod_ok,
                    );
                    out.dropped.push((name.clone(), reason));
                }
            }
        }
    }
    // If the filter dropped EVERY class, fall back to the unfiltered
    // list — better to provision SOMETHING with stale priors than to
    // crash the sweep on an empty class list.
    if out.keep.is_empty() && !candidates.is_empty() {
        out.keep = candidates.to_vec();
        out.dropped.clear();
    }
    out
}

/// TTL re-dispatch decision for one poll tick.
///
/// The launcher tracks the set of "completed chunk_ids" (parsed from
/// R2 omni sidecar list) and the "already TTL-redispatched" set, and
/// asks this helper for the list of chunk_ids to re-push.
///
/// Returns the chunk_ids that should be re-pushed AND records them in
/// `already_redispatched` so the next poll tick skips them.
pub fn ttl_redispatch_decisions(
    elapsed_secs: f64,
    cfg: &SweepConfig,
    all_chunk_ids: &[String],
    completed: &HashSet<String>,
    already_redispatched: &mut HashSet<String>,
) -> Vec<String> {
    if elapsed_secs < cfg.chunk_ttl_secs as f64 {
        return Vec::new();
    }
    let mut to_push = Vec::new();
    for cid in all_chunk_ids {
        if completed.contains(cid) {
            continue;
        }
        if already_redispatched.contains(cid) {
            continue;
        }
        to_push.push(cid.clone());
        already_redispatched.insert(cid.clone());
    }
    to_push
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provisioned_replicas_overshoots_then_clamps() {
        // 6 × 1.7 = 10.2 → ceil = 11 → clamp to quota 10.
        assert_eq!(compute_provisioned_replicas(6, 1.7, 10), 10);
        // 3 × 1.5 = 4.5 → ceil = 5; under quota.
        assert_eq!(compute_provisioned_replicas(3, 1.5, 10), 5);
        // overshoot < 1.0 is normalised to 1.0.
        assert_eq!(compute_provisioned_replicas(4, 0.5, 10), 4);
        // quota=0 is clamped to 1 (always provision SOMETHING).
        assert_eq!(compute_provisioned_replicas(2, 1.0, 0), 1);
    }

    #[test]
    fn speculative_disabled_returns_none() {
        let mut s = SpeculativeState::new();
        s.record_dispatched("c1", 0.0);
        s.record_completed("c2", 10.0);
        s.record_completed("c3", 12.0);
        s.record_completed("c4", 14.0);
        let cfg = SpeculativeConfig {
            enabled: false,
            ..SpeculativeConfig::default()
        };
        assert_eq!(s.decide_speculative("c1", 100.0, &cfg), None);
    }

    /// Helper: dispatch then complete with explicit duration so the
    /// per-chunk dispatch row is set before the completion record
    /// references it.
    fn dispatch_then_complete(s: &mut SpeculativeState, chunk_id: &str, duration_secs: f64) {
        s.record_dispatched(chunk_id, 0.0);
        s.record_completed(chunk_id, duration_secs);
    }

    #[test]
    fn speculative_waits_for_min_samples() {
        let mut s = SpeculativeState::new();
        s.record_dispatched("c1", 0.0);
        dispatch_then_complete(&mut s, "c2", 10.0); // n=1, < min 3
        let cfg = SpeculativeConfig::default();
        // Even at huge elapsed time, no speculative dispatch with n<3.
        assert_eq!(s.decide_speculative("c1", 1000.0, &cfg), None);
    }

    #[test]
    fn speculative_fires_after_threshold() {
        let mut s = SpeculativeState::new();
        s.record_dispatched("straggler", 0.0);
        dispatch_then_complete(&mut s, "c2", 10.0);
        dispatch_then_complete(&mut s, "c3", 12.0);
        dispatch_then_complete(&mut s, "c4", 14.0);
        // n=3 >= min 3, p95 (nearest-rank with n=3) = 14, factor 1.5
        // → threshold = 21.
        let cfg = SpeculativeConfig::default();
        // elapsed = 15 → below threshold (21), no speculative dispatch.
        assert_eq!(s.decide_speculative("straggler", 15.0, &cfg), None);
        // elapsed = 25 → above 21, speculative dispatch fires.
        let res = s.decide_speculative("straggler", 25.0, &cfg);
        assert!(res.is_some(), "expected speculative dispatch at elapsed=25");
        assert!((res.unwrap() - 25.0).abs() < 0.001);
    }

    #[test]
    fn speculative_respects_per_chunk_cap() {
        let mut s = SpeculativeState::new();
        s.record_dispatched("straggler", 0.0);
        for (i, t) in [10.0, 12.0, 14.0].iter().enumerate() {
            dispatch_then_complete(&mut s, &format!("c{}", i), *t);
        }
        let cfg = SpeculativeConfig::default();
        // First decision fires.
        assert!(s.decide_speculative("straggler", 100.0, &cfg).is_some());
        // After recording one speculative dispatch (cap=1), no more.
        s.record_speculative_dispatched("straggler");
        assert_eq!(s.decide_speculative("straggler", 200.0, &cfg), None);
    }

    #[test]
    fn ttl_redispatch_returns_missing_chunks_once() {
        let cfg = SweepConfig {
            chunk_ttl_secs: 60,
            ..SweepConfig::default()
        };
        let ids = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let completed: HashSet<String> = ["b".to_string()].into_iter().collect();
        let mut already = HashSet::new();
        // Before TTL → nothing.
        let r = ttl_redispatch_decisions(30.0, &cfg, &ids, &completed, &mut already);
        assert!(r.is_empty());
        // After TTL → a, c (b is completed).
        let r = ttl_redispatch_decisions(90.0, &cfg, &ids, &completed, &mut already);
        assert_eq!(r.len(), 2);
        assert!(r.contains(&"a".to_string()));
        assert!(r.contains(&"c".to_string()));
        // Second call → empty (each chunk re-pushed at most once via TTL).
        let r2 = ttl_redispatch_decisions(120.0, &cfg, &ids, &completed, &mut already);
        assert!(r2.is_empty());
    }

    #[test]
    fn filter_drops_slow_warmup_keeps_unknown() {
        let mut prior = HashMap::new();
        prior.insert(
            "FAST".to_string(),
            PriorClassStats {
                name: "FAST".into(),
                median_warmup_secs: Some(40),
                mean_chunks_processed: 3.0,
            },
        );
        prior.insert(
            "SLOW".to_string(),
            PriorClassStats {
                name: "SLOW".into(),
                median_warmup_secs: Some(120),
                mean_chunks_processed: 3.0,
            },
        );
        prior.insert(
            "STARVED".to_string(),
            PriorClassStats {
                name: "STARVED".into(),
                median_warmup_secs: Some(40),
                mean_chunks_processed: 0.5,
            },
        );
        let cfg = SweepConfig::default(); // max_warmup=60, min_productive=2.0
        let candidates = vec![
            "FAST".into(),
            "SLOW".into(),
            "STARVED".into(),
            "UNKNOWN".into(),
        ];
        let out = filter_classes(&candidates, &prior, &cfg);
        assert!(out.keep.contains(&"FAST".to_string()));
        assert!(out.keep.contains(&"UNKNOWN".to_string())); // no prior signal → keep
        assert!(out.dropped.iter().any(|(n, _)| n == "SLOW"));
        assert!(out.dropped.iter().any(|(n, _)| n == "STARVED"));
    }

    #[test]
    fn filter_falls_back_when_everything_drops() {
        let mut prior = HashMap::new();
        prior.insert(
            "SLOW".to_string(),
            PriorClassStats {
                name: "SLOW".into(),
                median_warmup_secs: Some(120),
                mean_chunks_processed: 3.0,
            },
        );
        let cfg = SweepConfig::default();
        let candidates = vec!["SLOW".into()];
        let out = filter_classes(&candidates, &prior, &cfg);
        // Fallback: rather than crash with empty class list, keep all.
        assert_eq!(out.keep, candidates);
        assert!(out.dropped.is_empty());
    }
}
