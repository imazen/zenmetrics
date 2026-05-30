//! Phase B (task #155) — bounded multi-warm LRU session pool for the
//! GPU worker.
//!
//! ## What this replaces
//!
//! Before #155 the GPU worker ([`crate::pool::gpu_worker_main`]) was
//! **single-warm**: it held ONE `ExecMetric` for the current
//! `(metric, w, h, backend)` signature and ONE cached `ref_hash`. An
//! *interleaved-reference* workload — ref0, ref1, …, refR-1, ref0, … —
//! thrashes that single slot: every ref switch re-runs `set_reference`
//! (the per-metric reference precompute), even when the same handful of
//! references recur. cvvdp's `set_reference` is the heaviest precompute
//! of the wired metrics, so the thrash cost is largest there.
//!
//! ## What this is
//!
//! A bounded LRU of [`OwnedSessionMetric`] entries keyed by
//! `(MetricKind, width, height, params_hash, ref_hash)`. On a task for
//! `(metric, ref)`:
//!
//! - **warm hit** (an entry with the matching key exists): reuse it via
//!   `score_with_warm_ref` — the reference precompute is skipped, the
//!   device-resident reference state is reused, only the distorted side
//!   runs. This is the perf unlock for interleaved-reference workloads.
//! - **miss**: build a new entry (`MetricSession::acquire` →
//!   `into_metric` → `set_reference`) and insert it. Before inserting,
//!   if the predicted total warm footprint would exceed the VRAM budget,
//!   evict LRU entries (drop them → exact per-entry reclaim, the property
//!   proven in `zenmetrics-api`'s `OwnedSessionMetric` isolation test)
//!   until it fits.
//!
//! ## Soundness invariants (preserved by construction)
//!
//! - **(a) no eviction mid-score.** A [`WarmSessionPool`] is owned by a
//!   single GPU lane thread (it lives on that thread's stack in
//!   `gpu_worker_main`); every method takes `&mut self`. So an entry is
//!   never dropped while a score on it is in flight — the lane is
//!   single-threaded and a score runs to completion before the next
//!   `get_or_build` can evict anything. Each entry belongs to exactly
//!   one lane/thread.
//! - **(b) CPU + cvvdp-StripPair paths untouched.** This pool only
//!   serves the umbrella session metrics (cvvdp Full / butter / ssim2 /
//!   dssim / iwssim / zensim on a GPU backend). The CPU worker and the
//!   `GpuStripPair` dispatch keep their existing non-session paths.
//! - **(c) parity.** An [`OwnedSessionMetric`] changes only WHERE
//!   buffers allocate (its private stream), not the kernel math — proven
//!   in `zenmetrics-api`'s owned-vs-borrowed-vs-plain parity tests. So
//!   the multi-warm pool returns the same scores as the single-warm
//!   path, within the metric's `Atomic<f32>` reduction-noise band.
//!
//! ## Degenerate single-warm
//!
//! If even one entry doesn't fit the budget (`max_entries == 0` after
//! the budget check, or `acquire` keeps returning `TooManyContexts`
//! after evicting everything), the pool falls back to holding exactly
//! one entry — i.e. the pre-#155 single-warm behaviour. It never
//! refuses to score.

#![cfg(all(feature = "bench", feature = "cuda"))]

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

use zenmetrics_api::{
    Backend as ApiBackend, MemoryMode, MetricKind, MetricParams, MetricSession, OwnedSessionMetric,
    Score,
};

// ---------------------------------------------------------------------------
// Process-wide observability counters (test/measurement surface).
//
// The per-lane `WarmSessionPool` lives on its lane thread's stack, so the
// Orchestrator can't read its fields directly. These atomics aggregate
// across lanes the same way `pool::WARM_INSTANCE_CONSTRUCTIONS` does, so
// the soundness/perf tests can confirm the multi-warm pool is firing
// (reference-precompute reuse) and that eviction bounds peak VRAM —
// without instrumenting cubecl. Production code MUST NOT depend on these.
// ---------------------------------------------------------------------------

/// Total `set_reference` (reference precompute) calls across all lanes'
/// warm pools. The interleaved-reference perf unlock is exactly "fewer of
/// these than the single-warm path would run".
static MW_SET_REFERENCE_CALLS: AtomicU64 = AtomicU64::new(0);
/// Warm hits (entry reused → reference precompute skipped).
static MW_HITS: AtomicU64 = AtomicU64::new(0);
/// Entries built (cache misses).
static MW_BUILDS: AtomicU64 = AtomicU64::new(0);
/// Entries evicted (dropped → reclaim) to satisfy budget/cap.
static MW_EVICTIONS: AtomicU64 = AtomicU64::new(0);

/// Snapshot of the multi-warm session pool's cross-lane counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MultiWarmStats {
    /// Total reference-precompute (`set_reference`) calls run.
    pub set_reference_calls: u64,
    /// Warm hits (reference precompute reused).
    pub hits: u64,
    /// Cache misses that built a fresh entry.
    pub builds: u64,
    /// Entries evicted (dropped → VRAM reclaimed).
    pub evictions: u64,
}

/// Read the current cross-lane multi-warm counters.
pub fn multiwarm_stats() -> MultiWarmStats {
    MultiWarmStats {
        set_reference_calls: MW_SET_REFERENCE_CALLS.load(Ordering::Relaxed),
        hits: MW_HITS.load(Ordering::Relaxed),
        builds: MW_BUILDS.load(Ordering::Relaxed),
        evictions: MW_EVICTIONS.load(Ordering::Relaxed),
    }
}

/// Reset the cross-lane multi-warm counters to zero. Test helper.
pub fn reset_multiwarm_stats() {
    MW_SET_REFERENCE_CALLS.store(0, Ordering::Relaxed);
    MW_HITS.store(0, Ordering::Relaxed);
    MW_BUILDS.store(0, Ordering::Relaxed);
    MW_EVICTIONS.store(0, Ordering::Relaxed);
}

/// Identity of a warm entry. Two tasks with the same key can share one
/// device-resident reference precompute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WarmKey {
    pub metric: MetricKind,
    pub width: u32,
    pub height: u32,
    /// Hash of the `Option<MetricParams>` debug rendering. `MetricParams`
    /// is not `Hash`/`Eq` (it carries floats), so we fold its `Debug`
    /// form — deterministic and sufficient for dedup. Distinct params →
    /// distinct precompute, so they must NOT alias an entry.
    pub params_hash: u64,
    /// xxhash3_64 of the reference bytes (the pool's `ref_hash`). The
    /// reference precompute is keyed on this — the warm hit reuses it.
    pub ref_hash: u64,
}

impl WarmKey {
    fn new(
        metric: MetricKind,
        width: u32,
        height: u32,
        params: &Option<MetricParams>,
        ref_hash: u64,
    ) -> Self {
        let mut h = DefaultHasher::new();
        // Debug-render is stable for a given params value within a build
        // (the enum + its fields all derive Debug). This is a dedup key,
        // not a security hash — collisions only cost a redundant rebuild.
        format!("{params:?}").hash(&mut h);
        Self {
            metric,
            width,
            height,
            params_hash: h.finish(),
            ref_hash,
        }
    }
}

/// One warm entry: an `OwnedSessionMetric` welded to its private stream,
/// plus the bookkeeping the LRU + budget need.
struct WarmEntry {
    key: WarmKey,
    metric: OwnedSessionMetric,
    /// Estimated device footprint in MiB (from the per-metric estimator)
    /// — used for the budget arithmetic. Conservative (estimator + a
    /// margin baked into the estimators themselves).
    est_mib: usize,
    /// Monotonic last-use tick for LRU eviction (higher = more recent).
    last_used: u64,
    /// Whether a reference is currently installed on this entry. Set on
    /// build (we `set_reference` immediately); a future caller could
    /// clear it, but the pool always builds with the ref installed.
    has_ref: bool,
}

/// Outcome of a pool dispatch attempt — mirrors the worker's existing
/// `CallErrPub` shape so the caller can keep its OOM/Other branching.
pub(crate) enum PoolScoreErr {
    /// A runtime OOM bubbled from the score (cubecl
    /// `cudaErrorMemoryAllocation` etc.). The caller drops the offending
    /// entry (already done inside the pool) and surfaces FullyExhausted.
    Oom,
    /// Any other error (dim mismatch, non-OOM dispatch failure, a
    /// build/acquire failure that the pool couldn't recover from).
    Other(String),
}

/// A bounded multi-warm LRU of [`OwnedSessionMetric`] entries for one GPU
/// lane. See the module docs.
pub(crate) struct WarmSessionPool {
    backend: ApiBackend,
    entries: Vec<WarmEntry>,
    /// VRAM budget for the warm set, in MiB. Eviction keeps the predicted
    /// total `est_mib` sum at-or-under this. `0` is treated as "single
    /// entry only" (degenerate single-warm).
    budget_mib: usize,
    /// Hard cap on entry count (a backstop independent of the byte
    /// budget — also bounds the session-slot consumption). Range 1..=N.
    max_entries: usize,
    /// Monotonic tick for LRU ordering.
    tick: u64,
    // --- observability counters (test surface) ---
    /// Warm hits (entry reused, reference precompute skipped).
    pub(crate) hits: u64,
    /// Misses that built a fresh entry.
    pub(crate) builds: u64,
    /// Entries evicted (dropped → reclaim) to satisfy the budget/cap.
    pub(crate) evictions: u64,
    /// Count of `set_reference` (reference precompute) calls actually
    /// run. The single-warm path runs one per ref switch; the multi-warm
    /// path runs one per distinct `(metric,dims,params,ref)` first-seen.
    pub(crate) set_reference_calls: u64,
}

impl WarmSessionPool {
    /// Build an empty pool. `budget_mib` is the VRAM ceiling for the warm
    /// set; `max_entries` is the hard entry-count backstop (clamped to
    /// `>= 1`).
    pub(crate) fn new(backend: ApiBackend, budget_mib: usize, max_entries: usize) -> Self {
        Self {
            backend,
            entries: Vec::new(),
            budget_mib,
            max_entries: max_entries.max(1),
            tick: 0,
            hits: 0,
            builds: 0,
            evictions: 0,
            set_reference_calls: 0,
        }
    }

    /// Number of currently-resident warm entries.
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Sum of the resident entries' estimated footprints (MiB).
    pub(crate) fn resident_mib(&self) -> usize {
        self.entries.iter().map(|e| e.est_mib).sum()
    }

    fn next_tick(&mut self) -> u64 {
        self.tick = self.tick.wrapping_add(1);
        self.tick
    }

    /// Find the index of the entry matching `key`, if resident.
    fn find(&self, key: &WarmKey) -> Option<usize> {
        self.entries.iter().position(|e| e.key == *key)
    }

    /// Drop the least-recently-used entry (lowest `last_used`). Returns
    /// `true` if one was evicted. The drop reclaims exactly that entry's
    /// VRAM (the `OwnedSessionMetric` field-order drop: scorer handles →
    /// free-list, then the session's `Drop` cleans its private stream).
    fn evict_lru(&mut self) -> bool {
        if self.entries.is_empty() {
            return false;
        }
        let mut lru_idx = 0usize;
        let mut lru_tick = self.entries[0].last_used;
        for (i, e) in self.entries.iter().enumerate().skip(1) {
            if e.last_used < lru_tick {
                lru_tick = e.last_used;
                lru_idx = i;
            }
        }
        let evicted = self.entries.swap_remove(lru_idx);
        // Explicit drop documents the reclaim point; the
        // OwnedSessionMetric Drop runs memory_cleanup()+sync() on its
        // private stream here.
        drop(evicted);
        self.evictions = self.evictions.wrapping_add(1);
        MW_EVICTIONS.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Evict LRU entries until adding a new entry of `incoming_mib` would
    /// keep the resident total at-or-under the budget AND the entry count
    /// strictly below `max_entries`. Never evicts below zero resident.
    ///
    /// Degenerate single-warm: if `budget_mib` is smaller than one
    /// entry's footprint, this evicts everything (resident → 0) so the
    /// new entry can take the lane alone.
    fn make_room_for(&mut self, incoming_mib: usize) {
        // Count backstop: keep room for one more entry.
        while self.entries.len() >= self.max_entries {
            if !self.evict_lru() {
                break;
            }
        }
        // Byte budget: evict until incoming fits (or nothing left).
        while !self.entries.is_empty()
            && self.resident_mib().saturating_add(incoming_mib) > self.budget_mib
        {
            if !self.evict_lru() {
                break;
            }
        }
    }

    /// Score `(ref_bytes, dist_bytes)` for `(metric, w, h, params)` with
    /// the reference identified by `ref_hash`, reusing a warm entry when
    /// one matches and building (with eviction under budget) otherwise.
    ///
    /// Returns the score plus the metric-specific extras the worker needs
    /// for `output_columns` (currently always empty for the session path
    /// — butter pnorm3 over the cached-ref path is a future addition,
    /// matching the pre-#155 `compute_with_cached_reference_with_extras`
    /// contract).
    ///
    /// On a runtime OOM the offending entry is dropped (reclaiming its
    /// VRAM) before returning `Err(PoolScoreErr::Oom)` so the caller's
    /// retry has the freed pages available.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn score(
        &mut self,
        metric: MetricKind,
        width: u32,
        height: u32,
        params: &Option<MetricParams>,
        ref_hash: u64,
        ref_bytes: &[u8],
        dist_bytes: &[u8],
    ) -> Result<Score, PoolScoreErr> {
        let key = WarmKey::new(metric, width, height, params, ref_hash);

        // -- Warm hit: reuse the cached reference precompute -----------
        if let Some(idx) = self.find(&key) {
            let tick = self.next_tick();
            self.entries[idx].last_used = tick;
            self.hits = self.hits.wrapping_add(1);
            MW_HITS.fetch_add(1, Ordering::Relaxed);
            let entry = &mut self.entries[idx];
            // The reference is already resident — score the dist side
            // against it (or a one-shot score if the metric had no cached
            // ref, though build always installs one).
            let res = if entry.has_ref {
                entry.metric.score_with_warm_ref(dist_bytes)
            } else {
                entry.metric.score(ref_bytes, dist_bytes)
            };
            return match res {
                Ok(s) => Ok(s),
                Err(e) => self.handle_score_err(idx, e),
            };
        }

        // -- Miss: build a fresh entry (evicting under budget first) ---
        let est_mib = estimate_entry_mib(metric, width, height);
        self.make_room_for(est_mib);

        // Acquire a session, retrying with LRU eviction on
        // TooManyContexts (slot pressure independent of the byte budget).
        let session = match self.acquire_with_eviction() {
            Ok(s) => s,
            Err(e) => return Err(PoolScoreErr::Other(e)),
        };

        // Build the owned metric on the session's private stream
        // (MemoryMode::Auto — same policy as the single-warm path's
        // construct_via_umbrella first attempt).
        let mut owned = match session.into_metric_with_memory_mode(
            metric,
            width,
            height,
            metric_params_or_default(metric, params),
            MemoryMode::Auto,
        ) {
            Ok(m) => m,
            Err(e) => return Err(classify_build_err(&e.to_string())),
        };

        // Install the reference precompute once. If the metric has no
        // separate cached-ref path this can fail — fall back to one-shot
        // scoring (has_ref = false), matching the worker's set_reference
        // fallback.
        let has_ref = match owned.set_reference_srgb_u8(ref_bytes) {
            Ok(()) => {
                self.set_reference_calls = self.set_reference_calls.wrapping_add(1);
                MW_SET_REFERENCE_CALLS.fetch_add(1, Ordering::Relaxed);
                true
            }
            Err(_) => false,
        };
        self.builds = self.builds.wrapping_add(1);
        MW_BUILDS.fetch_add(1, Ordering::Relaxed);
        // A warm-entry build IS a metric-instance construction — record
        // it on the unified counter so warm-instance-churn accounting
        // (and the reorder churn test) stays valid across both warm paths.
        crate::pool::record_warm_instance_construction();

        // Score the dist side.
        let res = if has_ref {
            owned.score_with_warm_ref(dist_bytes)
        } else {
            owned.score(ref_bytes, dist_bytes)
        };
        let score = match res {
            Ok(s) => s,
            Err(e) => {
                // Don't insert a failed entry; classify + reclaim by drop.
                drop(owned);
                return Err(classify_score_err(&e.to_string()));
            }
        };

        // Insert the warm entry.
        let tick = self.next_tick();
        self.entries.push(WarmEntry {
            key,
            metric: owned,
            est_mib,
            last_used: tick,
            has_ref,
        });
        Ok(score)
    }

    /// `acquire` a session, evicting LRU entries on `TooManyContexts`
    /// until it succeeds or there's nothing left to evict. Returns the
    /// session, or an error string if even an empty pool can't acquire.
    fn acquire_with_eviction(&mut self) -> Result<MetricSession, String> {
        loop {
            match MetricSession::acquire(self.backend) {
                Ok(s) => return Ok(s),
                Err(zenmetrics_api::Error::TooManyContexts { .. }) => {
                    // Slot pressure — drop our LRU entry to free a slot.
                    if !self.evict_lru() {
                        // We hold no entries and still can't acquire: all
                        // 128 slots are held by OTHER lanes / leaked
                        // sessions. Surface the error; the worker will
                        // fall back to the non-session path.
                        return Err(
                            "MetricSession::acquire: all slots held elsewhere (TooManyContexts) \
                             and this lane has no entries to evict"
                                .to_string(),
                        );
                    }
                }
                Err(e) => return Err(e.to_string()),
            }
        }
    }

    /// Handle a score error from a resident entry: classify it, and on
    /// OOM drop the entry (reclaiming its VRAM) so the retry has the
    /// pages. The entry index is into `self.entries`.
    fn handle_score_err(
        &mut self,
        idx: usize,
        err: zenmetrics_api::Error,
    ) -> Result<Score, PoolScoreErr> {
        let msg = err.to_string();
        let classified = classify_score_err(&msg);
        if matches!(classified, PoolScoreErr::Oom) {
            // Drop the offending entry → reclaim its private stream's pool.
            let evicted = self.entries.swap_remove(idx);
            drop(evicted);
            self.evictions = self.evictions.wrapping_add(1);
            MW_EVICTIONS.fetch_add(1, Ordering::Relaxed);
        }
        Err(classified)
    }

    /// Drop every warm entry, reclaiming all their VRAM. Called when the
    /// worker shuts down or when the caller wants a hard reset.
    #[allow(dead_code)]
    pub(crate) fn clear(&mut self) {
        // Each drop reclaims its own private stream's pool.
        self.entries.clear();
    }
}

/// Resolve `params` to a concrete `MetricParams` (defaulting per-kind)
/// for the owned-metric constructor, which takes a non-`Option` value.
fn metric_params_or_default(kind: MetricKind, params: &Option<MetricParams>) -> MetricParams {
    match params {
        Some(p) => p.clone(),
        None => MetricParams::default_for(kind),
    }
}

/// Per-metric device-footprint estimate in MiB, used for the warm-set
/// VRAM budget. Reaches into each metric crate's `estimate_gpu_memory_*`
/// (the same estimators the chooser's profile is calibrated against).
/// Conservative: the estimators bake in a margin, and for zensim we pass
/// the largest regime (`WithIw`).
fn estimate_entry_mib(metric: MetricKind, width: u32, height: u32) -> usize {
    let bytes: usize = match metric {
        MetricKind::Cvvdp => {
            zenmetrics_api::cvvdp::memory_mode::estimate_gpu_memory_bytes_usize(width, height)
        }
        MetricKind::Butter => {
            zenmetrics_api::butter::memory_mode::estimate_gpu_memory_bytes(width, height)
        }
        MetricKind::Ssim2 => {
            zenmetrics_api::ssim2::memory_mode::estimate_gpu_memory_bytes(width, height)
        }
        MetricKind::Dssim => {
            zenmetrics_api::dssim::memory_mode::estimate_gpu_memory_bytes(width, height)
        }
        MetricKind::Iwssim => {
            zenmetrics_api::iwssim::memory_mode::estimate_gpu_memory_bytes(width, height)
        }
        MetricKind::Zensim => zenmetrics_api::zensim::memory_mode::estimate_gpu_memory_bytes(
            width,
            height,
            // Largest regime → most conservative budget.
            zenmetrics_api::zensim::ZensimFeatureRegime::WithIw,
        ),
    };
    // Round up to MiB, floor of 1 so a tiny image still counts as one
    // entry against the budget.
    (bytes / (1024 * 1024)).max(1)
}

/// Classify a build/construction error string into the pool's OOM vs
/// other shape. Mirrors `executor::classify_construct_err`.
fn classify_build_err(msg: &str) -> PoolScoreErr {
    let lowered = msg.to_ascii_lowercase();
    if lowered.contains("toobigforfull")
        || lowered.contains("out of memory")
        || lowered.contains("oom")
    {
        PoolScoreErr::Oom
    } else {
        PoolScoreErr::Other(msg.to_string())
    }
}

/// Classify a compute/score error string. Mirrors
/// `executor::classify_call_err`.
fn classify_score_err(msg: &str) -> PoolScoreErr {
    let lowered = msg.to_ascii_lowercase();
    if lowered.contains("oom")
        || lowered.contains("out of memory")
        || lowered.contains("toobigforfull")
        || lowered.contains("cuda_error_out_of_memory")
        || lowered.contains("cudaerrormemoryallocation")
    {
        PoolScoreErr::Oom
    } else {
        PoolScoreErr::Other(msg.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pure-logic tests of the LRU + budget arithmetic. These don't touch
    // the GPU — they exercise the eviction policy by constructing the
    // pool's bookkeeping directly is not possible (entries hold real
    // OwnedSessionMetrics), so instead we test the small pure helpers.

    #[test]
    fn warm_key_distinguishes_ref_and_params() {
        let p = Some(MetricParams::default_for(MetricKind::Cvvdp));
        let k1 = WarmKey::new(MetricKind::Cvvdp, 256, 256, &p, 0xAAAA);
        let k2 = WarmKey::new(MetricKind::Cvvdp, 256, 256, &p, 0xBBBB);
        assert_ne!(k1, k2, "distinct ref_hash → distinct key");
        let k3 = WarmKey::new(MetricKind::Cvvdp, 512, 256, &p, 0xAAAA);
        assert_ne!(k1, k3, "distinct dims → distinct key");
        let k4 = WarmKey::new(MetricKind::Ssim2, 256, 256, &p, 0xAAAA);
        assert_ne!(k1, k4, "distinct metric → distinct key");
        let k5 = WarmKey::new(MetricKind::Cvvdp, 256, 256, &p, 0xAAAA);
        assert_eq!(k1, k5, "same tuple → same key (warm hit)");
    }

    #[test]
    fn estimate_entry_mib_is_positive_and_scales() {
        let small = estimate_entry_mib(MetricKind::Cvvdp, 256, 256);
        let large = estimate_entry_mib(MetricKind::Cvvdp, 4096, 4096);
        assert!(small >= 1, "tiny estimate floored to >=1 MiB");
        assert!(large > small, "larger image → larger estimate");
    }

    #[test]
    fn classify_errs() {
        assert!(matches!(
            classify_score_err("cudaErrorMemoryAllocation"),
            PoolScoreErr::Oom
        ));
        assert!(matches!(
            classify_score_err("dim mismatch"),
            PoolScoreErr::Other(_)
        ));
        assert!(matches!(
            classify_build_err("TooBigForFull { .. }"),
            PoolScoreErr::Oom
        ));
    }
}
