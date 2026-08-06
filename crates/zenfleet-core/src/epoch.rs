//! Epoch-sharded claiming — deterministic work partitioning on wall-clock epochs.
//!
//! The lease claim path (one R2 conditional-PUT lease per chunk/cell) is correct but pays a
//! round-trip per claim *attempt*, and every worker attempts most claims: with N workers and M
//! chunks the fleet spends ≈ N×M claim round-trips to win M chunks. Worse, when a worker's
//! reconcile view is empty or stale, dedup degrades to those leases alone — the avifgen encode
//! campaign measured **1,237,361 ledger rows for 343,465 distinct DONE cells (3.6×)** before the
//! snapshot fix (`92432e37`). Epoch sharding removes the contention itself:
//!
//! - **Epochs come from the wall clock** ([`epoch_index`]): boundary = unix seconds rounded down
//!   to `epoch_len_secs`. No coordinator, no election — every worker computes the same epoch from
//!   its own clock (NTP-sane skew ≪ epoch length is assumed; skew only shifts *when* a worker
//!   re-shards, never *what* it owns).
//! - **At each boundary every worker independently**: pulls the latest ledger snapshot (the
//!   existing `--ledger-in` machinery — this module never re-implements it), computes
//!   `remaining = declared − done`, reads the **alive roster** (workers that heartbeated during
//!   the previous epoch), and partitions `remaining` by **rendezvous hashing** ([`hrw_owner`]:
//!   owner = argmax over workers of `hash(cell, worker)`). It then works *only its own shard*
//!   until the next boundary — **zero claim traffic in steady state**.
//! - **Leases remain the safety net, not the primary path** ([`shard_decision`]): a cell owned in
//!   this epoch *and* under the previous epoch's roster view runs lease-free; a cell that
//!   ownership *moved to* this worker (roster churn — a joined/died peer) is executed only behind
//!   the ordinary per-cell lease, which is what fences the boundary seam where the old owner may
//!   still be in flight or two workers may briefly hold divergent roster views. Exactly-once
//!   stays **ledger-enforced** (content-addressed outputs + latest-wins done-set), exactly as in
//!   lease mode — a rare seam duplicate is bounded waste, never wrong data.
//! - **HRW is stable under churn**: adding worker W moves only the cells whose new argmax is W;
//!   removing W re-distributes only W's cells. No global reshuffle, so a stable fleet keeps a
//!   stable partition across every epoch and the lease-free fast path covers ~everything.
//!
//! All functions here are pure (no clock, no I/O): the worker crate wires the heartbeat writes,
//! roster listing, and lease calls around them, and `zenfleet-sim` drives them deterministically.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::job::JobKind;

/// How a run's workers avoid executing the same cell twice while the campaign is live.
/// Campaign-level: **every worker on a run must use the same mode** — a lease-mode worker forms
/// full-gap chunks that overlap every sharded worker's cells (still ledger-correct, but it
/// re-introduces the duplicate-work tax this mode exists to kill). The [`crate::RunControl`]
/// object can carry the mode so a whole fleet converges without per-box env edits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimMode {
    /// R2 conditional-PUT leases per chunk/cell (the original mechanism). The default.
    #[default]
    Lease,
    /// Wall-clock epochs + rendezvous-hash ownership; leases only at ownership seams.
    EpochSharded,
}

impl ClaimMode {
    /// Parse the CLI/env spelling (`lease` / `epoch-sharded`; underscores accepted).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "lease" => Some(Self::Lease),
            "epoch-sharded" | "epochsharded" | "epoch" => Some(Self::EpochSharded),
            _ => None,
        }
    }
}

/// Epoch-sharded claiming knobs (campaign-configurable; the defaults are the registered design).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpochShardCfg {
    /// Epoch length in seconds. Boundary = `unix_time - unix_time % epoch_len_secs`. Keep it
    /// ≥ 2× the chunk wall target so an in-flight chunk can't span a whole epoch.
    #[serde(default = "default_epoch_len_secs")]
    pub epoch_len_secs: u64,
    /// How often a working pass refreshes its heartbeat (between chunks). Any beat inside an
    /// epoch keeps the worker in the next epoch's roster, so this only needs to be ≤ epoch_len;
    /// smaller values tolerate more missed beats.
    #[serde(default = "default_heartbeat_interval_secs")]
    pub heartbeat_interval_secs: u64,
    /// Straggler-tail stealing: once a worker exhausts its own shard inside the tail fraction of
    /// an epoch, it may work *other* workers' remaining cells in a deterministic per-worker order
    /// — always lease-guarded, so stealers never collide with each other. A steal can still
    /// duplicate the owner's own late execution of that cell (the owner runs lease-free); that is
    /// the same harmless-duplicate contract as speculative execution, and it only arises in the
    /// campaign endgame where the alternative is an idle paid box.
    #[serde(default = "default_tail_steal")]
    pub tail_steal: bool,
}

fn default_epoch_len_secs() -> u64 {
    600
}
fn default_heartbeat_interval_secs() -> u64 {
    120
}
fn default_tail_steal() -> bool {
    true
}

impl Default for EpochShardCfg {
    fn default() -> Self {
        Self {
            epoch_len_secs: default_epoch_len_secs(),
            heartbeat_interval_secs: default_heartbeat_interval_secs(),
            tail_steal: default_tail_steal(),
        }
    }
}

/// Fraction of an epoch after which an exhausted worker may start tail-stealing (see
/// [`EpochShardCfg::tail_steal`]). Fixed, not configurable: stealing earlier trades duplicate
/// work for very little tail latency, because a healthy owner is still mid-shard.
pub const TAIL_STEAL_FRACTION: f64 = 0.75;

/// The epoch containing unix-seconds `now`: `now / epoch_len_secs` (length floored to 1).
pub fn epoch_index(now: u64, epoch_len_secs: u64) -> u64 {
    now / epoch_len_secs.max(1)
}

/// Start (unix seconds) of epoch `idx`.
pub fn epoch_start(idx: u64, epoch_len_secs: u64) -> u64 {
    idx.saturating_mul(epoch_len_secs.max(1))
}

/// Seconds from `now` until the next epoch boundary (in `(0, epoch_len]`).
pub fn secs_to_next_boundary(now: u64, epoch_len_secs: u64) -> u64 {
    let len = epoch_len_secs.max(1);
    len - (now % len)
}

/// True once `now` is inside the tail-steal window of its epoch (the last
/// `1 − TAIL_STEAL_FRACTION` of it).
pub fn in_tail(now: u64, epoch_len_secs: u64) -> bool {
    let len = epoch_len_secs.max(1);
    let into = now % len;
    (into as f64) >= TAIL_STEAL_FRACTION * (len as f64)
}

/// Seconds from `now` until this epoch's tail-steal window opens (0 once inside it).
pub fn secs_to_tail(now: u64, epoch_len_secs: u64) -> u64 {
    let len = epoch_len_secs.max(1);
    let tail_at = (TAIL_STEAL_FRACTION * len as f64).ceil() as u64;
    tail_at.saturating_sub(now % len)
}

// ───────────────────────────── stable hashing (HRW scores) ─────────────────────────────
//
// Rendezvous hashing only works if every worker computes the *same* scores, so the hash must be
// stable across builds, platforms, and Rust releases — `DefaultHasher` explicitly is not. FNV-1a
// (64-bit, spec constants) over `cell ∥ 0x1F ∥ worker` gives a spec-frozen digest; a splitmix64
// finalizer fixes FNV's weak low-bit avalanche so shard sizes stay balanced. Both algorithms are
// published constants — pinned by test vectors below, never change them.

const FNV64_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV64_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a64_step(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV64_PRIME);
    }
    h
}

/// FNV-1a 64 of `bytes` (spec offset/prime; see the module vectors).
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    fnv1a64_step(FNV64_OFFSET, bytes)
}

/// splitmix64 finalizer — a bijective mixer with full avalanche.
fn mix64(mut z: u64) -> u64 {
    z ^= z >> 30;
    z = z.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The HRW score of `(cell, worker)`: `mix64(fnv1a64(cell ∥ 0x1F ∥ worker))`. The 0x1F separator
/// (also used by [`crate::CellId::tuple_key`]) makes the pair encoding unambiguous.
pub fn hrw_score(cell_key: &str, worker: &str) -> u64 {
    let h = fnv1a64_step(FNV64_OFFSET, cell_key.as_bytes());
    let h = fnv1a64_step(h, &[0x1f]);
    mix64(fnv1a64_step(h, worker.as_bytes()))
}

// ───────────────────────────────────── roster ─────────────────────────────────────

/// The alive-worker set an epoch's sharding is computed against: the workers that heartbeated
/// during the *previous* epoch. Members are sorted + deduped so every reader derives the same
/// roster from the same heartbeat listing regardless of listing order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Roster {
    /// The epoch this roster governs (NOT the epoch the heartbeats were written in).
    pub epoch: u64,
    workers: Vec<String>,
}

impl Roster {
    pub fn new(epoch: u64, mut workers: Vec<String>) -> Self {
        workers.sort_unstable();
        workers.dedup();
        Self { epoch, workers }
    }

    pub fn workers(&self) -> &[String] {
        &self.workers
    }

    pub fn contains(&self, worker: &str) -> bool {
        self.workers
            .binary_search_by(|w| w.as_str().cmp(worker))
            .is_ok()
    }

    pub fn is_empty(&self) -> bool {
        self.workers.is_empty()
    }

    pub fn len(&self) -> usize {
        self.workers.len()
    }

    /// Same member set (epoch ignored). When true, the owner function is *identical* across the
    /// two epochs — the cheap test that lets a stable fleet skip per-cell previous-owner checks.
    pub fn same_members(&self, other: &Roster) -> bool {
        self.workers == other.workers
    }
}

/// Rendezvous (highest-random-weight) owner of `cell_key` under `roster`: the worker with the
/// maximum [`hrw_score`], ties broken by the lexicographically greatest worker id (total order ⇒
/// exactly one owner). `None` on an empty roster.
///
/// Stability property (the reason HRW over consistent-hash rings): changing the roster by one
/// worker moves only that worker's cells — every other cell's argmax is unchanged.
pub fn hrw_owner<'a>(roster: &'a Roster, cell_key: &str) -> Option<&'a str> {
    roster
        .workers
        .iter()
        .map(|w| (hrw_score(cell_key, w), w.as_str()))
        .max_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)))
        .map(|(_, w)| w)
}

/// True iff `me` is the [`hrw_owner`] of `cell_key`.
pub fn owns(roster: &Roster, me: &str, cell_key: &str) -> bool {
    hrw_owner(roster, cell_key) == Some(me)
}

/// Per-cell claim decision for one epoch (see the module docs for the seam rationale).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShardDecision {
    /// Mine now, and mine under the previous epoch's roster too → execute lease-free.
    OwnedFast,
    /// Mine now, but ownership just moved here (roster churn, or no previous roster to compare —
    /// the epoch right after bootstrap) → execute only behind the per-cell lease.
    OwnedGuarded,
    /// Another worker's cell. Touch only via the lease-guarded tail-steal path.
    Other,
}

/// Decide how this worker may execute `cell_key` in the epoch governed by `roster_now`.
/// `roster_prev` is the roster that governed the previous epoch (`None` when unknown — first
/// sharded epoch, or the listing failed); unknown always guards, never fast-paths.
pub fn shard_decision(
    roster_now: &Roster,
    roster_prev: Option<&Roster>,
    me: &str,
    cell_key: &str,
) -> ShardDecision {
    if !owns(roster_now, me, cell_key) {
        return ShardDecision::Other;
    }
    match roster_prev {
        // Identical member set ⇒ identical owner function ⇒ it was mine last epoch too.
        Some(prev) if prev.same_members(roster_now) => ShardDecision::OwnedFast,
        Some(prev) if !prev.is_empty() => {
            if owns(prev, me, cell_key) {
                ShardDecision::OwnedFast
            } else {
                ShardDecision::OwnedGuarded
            }
        }
        _ => ShardDecision::OwnedGuarded,
    }
}

/// Deterministic per-worker order for tail-stealing other workers' cells: descending
/// [`hrw_score`]`(cell, me)`. Different stealers walk near-uncorrelated permutations (their
/// scores are keyed on their own ids), so concurrent stealers rarely contend for the same lease;
/// the same stealer re-derives the same order every pass. Returns indices into `cell_keys`.
pub fn steal_order(me: &str, cell_keys: &[&str]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..cell_keys.len()).collect();
    idx.sort_unstable_by_key(|&i| std::cmp::Reverse(hrw_score(cell_keys[i], me)));
    idx
}

// ───────────────────── weighted rendezvous hashing (registered speed handicaps) ─────────────────────
//
// Boxes are not interchangeable: the avifgen encode ledger measured a ~2.5× per-thread and ~5×
// per-box spread (attribution audit, `0f9fa073`). Uniform shards make the fast box idle in every
// epoch's tail while the slow box straggles. Weighted HRW gives each worker a cell share
// proportional to a REGISTERED (measured, committed — never auto-tuned) weight, via the canonical
// logarithm method: owner = argmax over workers of `−w_i / ln(u_i)` where `u_i` maps the frozen
// pair hash into (0,1). Properties: shares ∝ w_i; changing one worker's weight moves only the
// share delta (every other pair score is untouched); w = 0 excludes a worker from a mode's
// sharding entirely (role specialization). Weights are per (worker, workload mode) — and encode
// weights are per ENCODER TYPE, because per-encode concurrency differs (process-parallel
// single-threaded encoders scale with core count; internally-threaded encoders like svt compress
// the many-core advantage; memory-bound encoders re-rank boxes again), so relative box throughput
// is a function of (box, type) and cross-type extrapolation is wrong.
//
// Determinism note: `ln` comes from the pure-Rust `libm` crate, NOT the platform libm — vendor
// libms round differently in the last ulp, and this fleet mixes x86 and aarch64. `libm::log` is
// the same IEEE arithmetic everywhere, so every worker computes bit-identical scores. The
// uniform case (every weight exactly 1.0, or no registry at all) never enters the float path:
// it delegates verbatim to [`hrw_owner`], pinning absent-weights == old behavior bit-for-bit.

/// The workload axis a cell's handicap weight is read from. Derived from the MANIFEST (every
/// worker derives the same mode for the same cell — never from box-local env):
/// - `Encode { codec }` — [`JobKind::Encode`], keyed per encoder type.
/// - `GpuMetric` — metric/score/diffmap jobs whose metric names carry the `_gpu` suffix (the
///   wave convention: sf-gpu manifests declare `butteraugli_max_gpu`, …). The card does the
///   work, so CPU-derived weights would mis-shard.
/// - `Metric` — every other (CPU-bound) job: plain/`_cpu` metric names, features, bakes, ….
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShardMode<'a> {
    Encode { codec: &'a str },
    Metric,
    GpuMetric,
}

fn any_gpu_metric<'a, I: IntoIterator<Item = &'a str>>(names: I) -> bool {
    names.into_iter().any(|m| m.ends_with("_gpu"))
}

/// Map a job to its handicap mode (see [`ShardMode`]).
pub fn shard_mode(kind: &JobKind) -> ShardMode<'_> {
    match kind {
        JobKind::Encode { codec, .. } => ShardMode::Encode { codec },
        JobKind::Metric { metric } => {
            if any_gpu_metric([metric.as_str()]) {
                ShardMode::GpuMetric
            } else {
                ShardMode::Metric
            }
        }
        JobKind::ScoreFile { metrics, .. } => {
            if any_gpu_metric(metrics.iter().map(String::as_str)) {
                ShardMode::GpuMetric
            } else {
                ShardMode::Metric
            }
        }
        JobKind::Diffmap { metric, .. } => {
            if any_gpu_metric([metric.as_str()]) {
                ShardMode::GpuMetric
            } else {
                ShardMode::Metric
            }
        }
        _ => ShardMode::Metric,
    }
}

/// One worker's registered handicap row. `1.0` = the reference throughput; `0.0` = excluded
/// from that mode's sharding. Encode is keyed per encoder type with a `"default"` fallback
/// (absent type AND absent default → 1.0).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WorkerHandicap {
    /// Per-encoder-type multipliers, e.g. `{ "zenavif": 0.72, "default": 1.0 }`.
    #[serde(default)]
    pub encode: BTreeMap<String, f64>,
    /// CPU-metric multiplier (memory/vector-bound work — NOT the same ranking as encode).
    #[serde(default = "one")]
    pub metric: f64,
    /// GPU-metric multiplier (the card, not the CPU).
    #[serde(default = "one")]
    pub gpu_metric: f64,
}

fn one() -> f64 {
    1.0
}

impl WorkerHandicap {
    /// The multiplier this row assigns to `mode`. Negative/NaN registry values are clamped to
    /// 0.0 (excluded) — a nonsense weight must never un-determinize the argmax.
    pub fn weight(&self, mode: &ShardMode<'_>) -> f64 {
        let w = match mode {
            ShardMode::Encode { codec } => self
                .encode
                .get(*codec)
                .or_else(|| self.encode.get("default"))
                .copied()
                .unwrap_or(1.0),
            ShardMode::Metric => self.metric,
            ShardMode::GpuMetric => self.gpu_metric,
        };
        if w.is_finite() && w > 0.0 { w } else { 0.0 }
    }
}

/// The registered handicap table: worker key → [`WorkerHandicap`]. Workers absent from the
/// table weigh 1.0 in every mode. Precedence at the worker: campaign `RunControl.worker_weights`
/// override > the committed registry (`fleet/handicaps.toml`, baked into the binary) > 1.0.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Handicaps(pub BTreeMap<String, WorkerHandicap>);

impl Handicaps {
    /// The weight of `worker` for `mode` (1.0 when the worker has no row).
    pub fn weight(&self, worker: &str, mode: &ShardMode<'_>) -> f64 {
        self.0.get(worker).map(|h| h.weight(mode)).unwrap_or(1.0)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Map the frozen 64-bit pair hash into (0,1) via its top 52 bits: `((h >> 12) + 0.5) × 2⁻⁵²`.
/// Every step is EXACT f64 arithmetic — `k + 0.5` is exactly representable for every
/// `k < 2⁵²` (the 0.5-spacing binade starts at 2⁵¹, and 2⁵² − 0.5 still sits inside it), and
/// the final scale is a power of two — so `u ∈ [2⁻⁵³, 1 − 2⁻⁵³]`, never 0 or 1, and `ln(u)` is
/// always finite and negative. Two tempting alternatives are WRONG at the boundaries:
/// `(h as f64 + 0.5) × 2⁻⁶⁴` rounds a whole band of top-end hashes to exactly 1.0 (`as f64`
/// rounds-to-nearest above 2⁵³), and 53-bit `((h >> 11) + 0.5) × 2⁻⁵³` still hits 1.0 at the
/// very top (`(2⁵³ − 1) + 0.5` ties-to-even UP to 2⁵³) — either way `ln(1) = 0` flips the score
/// to −∞ and excludes precisely the workers the hash most preferred. Dropping the 12 low bits
/// costs nothing (the hash is finalizer-mixed; a tie collapses ~2¹² hash values onto one u and
/// breaks deterministically by worker id in the argmax).
fn u01(h: u64) -> f64 {
    const INV52: f64 = 1.0 / 4_503_599_627_370_496.0; // 2^-52
    ((h >> 12) as f64 + 0.5) * INV52
}

/// The weighted rendezvous score of `(cell, worker)` under weight `w`: `−w / ln(u)` with
/// `u = u01(hrw_score(cell, worker))`. Monotone in u, share ∝ w over the fleet. `w ≤ 0` (or
/// non-finite) returns `None` — the worker is excluded for this cell's mode.
pub fn hrw_score_weighted(cell_key: &str, worker: &str, w: f64) -> Option<f64> {
    if !w.is_finite() || w <= 0.0 {
        return None;
    }
    // libm::log, NOT f64::ln — cross-platform bit-determinism (see the section comment).
    Some(-w / libm::log(u01(hrw_score(cell_key, worker))))
}

/// Weighted rendezvous owner: argmax over the roster of [`hrw_score_weighted`], ties broken by
/// the lexicographically greatest worker id. When every roster member weighs exactly 1.0 this
/// DELEGATES to the unweighted [`hrw_owner`] — bit-identical old behavior (the float path is
/// never entered), which is what lets a weightless fleet upgrade without any cell moving.
/// `None` when the roster is empty or every member is excluded (weight 0) for this cell.
pub fn hrw_owner_weighted<'a>(
    roster: &'a Roster,
    weight_of: &dyn Fn(&str) -> f64,
    cell_key: &str,
) -> Option<&'a str> {
    if roster.workers().iter().all(|w| weight_of(w) == 1.0) {
        return hrw_owner(roster, cell_key);
    }
    roster
        .workers()
        .iter()
        .filter_map(|w| hrw_score_weighted(cell_key, w, weight_of(w)).map(|s| (s, w.as_str())))
        .max_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(b.1)))
        .map(|(_, w)| w)
}

/// [`shard_decision`] with weighted ownership on both rosters. The previous epoch's ownership is
/// evaluated with the CURRENT weights: weight edits are rare registered events (a commit or a
/// RunControl write), and treating them like un-guarded roster churn bounds the cost to at most
/// one in-flight chunk of duplicate work per moved cell, once per edit — ledger-deduped, same
/// contract as the other seam windows. (Changing weights via RunControl converges the whole
/// fleet on the next pass; prefer it over image-embedded registry edits for live campaigns.)
pub fn shard_decision_weighted(
    roster_now: &Roster,
    roster_prev: Option<&Roster>,
    me: &str,
    cell_key: &str,
    weight_of: &dyn Fn(&str) -> f64,
) -> ShardDecision {
    if hrw_owner_weighted(roster_now, weight_of, cell_key) != Some(me) {
        return ShardDecision::Other;
    }
    match roster_prev {
        Some(prev) if prev.same_members(roster_now) => ShardDecision::OwnedFast,
        Some(prev) if !prev.is_empty() => {
            if hrw_owner_weighted(prev, weight_of, cell_key) == Some(me) {
                ShardDecision::OwnedFast
            } else {
                ShardDecision::OwnedGuarded
            }
        }
        _ => ShardDecision::OwnedGuarded,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn roster(epoch: u64, ws: &[&str]) -> Roster {
        Roster::new(epoch, ws.iter().map(|s| s.to_string()).collect())
    }

    fn cells(n: usize) -> Vec<String> {
        // Shaped like real job ids (hex sha) so balance numbers are representative.
        (0..n)
            .map(|i| format!("{:064x}", (i as u128) * 0x9e37_79b9_7f4a_7c15))
            .collect()
    }

    // ── stable-hash pinning ──────────────────────────────────────────────────────────

    #[test]
    fn fnv1a64_matches_published_vectors() {
        // Canonical FNV-1a 64 test vectors (Fowler/Noll/Vo reference).
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn hrw_score_is_pinned_forever() {
        // Regression pins: if any of these change, every deployed fleet's shard assignment
        // changes under it — mixed-build fleets would double/miss work. NEVER update these
        // values; a new hash needs a new claim-mode name.
        // Values verified by an independent Python implementation of the same published
        // algorithms (FNV-1a 64 + splitmix64 finalizer), 2026-08-06.
        assert_eq!(hrw_score("", ""), mix64(fnv1a64(&[0x1f])));
        assert_eq!(hrw_score("cell-a", "worker-1"), 0xe8ef_f34b_5054_7aed);
        assert_eq!(hrw_score("cell-a", "worker-2"), 0xd0e7_374c_9d9e_0eb6);
        assert_eq!(hrw_score("cell-b", "worker-1"), 0x3f2e_9d23_aa97_daf2);
    }

    #[test]
    fn hrw_score_separator_disambiguates() {
        // ("ab","c") must not collide with ("a","bc") — the 0x1F separator guarantees it.
        assert_ne!(hrw_score("ab", "c"), hrw_score("a", "bc"));
    }

    // ── epoch math / skew ────────────────────────────────────────────────────────────

    #[test]
    fn epoch_boundaries_round_down() {
        assert_eq!(epoch_index(0, 600), 0);
        assert_eq!(epoch_index(599, 600), 0);
        assert_eq!(epoch_index(600, 600), 1);
        assert_eq!(epoch_start(1, 600), 600);
        assert_eq!(secs_to_next_boundary(600, 600), 600);
        assert_eq!(secs_to_next_boundary(1199, 600), 1);
        // Degenerate length floors to 1 rather than dividing by zero.
        assert_eq!(epoch_index(42, 0), 42);
    }

    #[test]
    fn skewed_start_times_inside_one_epoch_agree() {
        // A worker starting mid-epoch (any now in the window) computes the same epoch index,
        // hence the same roster prefix and the same shard.
        let len = 600;
        for now in [1_754_500_800, 1_754_500_801, 1_754_501_399] {
            assert_eq!(epoch_index(now, len), 1_754_500_800 / 600);
        }
        assert_eq!(epoch_index(1_754_501_400, len), 1_754_500_800 / 600 + 1);
    }

    #[test]
    fn tail_window_is_the_last_quarter_by_default() {
        assert!(!in_tail(0, 600));
        assert!(!in_tail(449, 600));
        assert!(in_tail(450, 600)); // 0.75 × 600
        assert!(in_tail(599, 600));
        assert!(!in_tail(600, 600)); // next epoch starts fresh
        assert_eq!(secs_to_tail(0, 600), 450);
        assert_eq!(secs_to_tail(449, 600), 1);
        assert_eq!(secs_to_tail(450, 600), 0);
        assert_eq!(secs_to_tail(599, 600), 0);
    }

    // ── roster + owner determinism ───────────────────────────────────────────────────

    #[test]
    fn roster_is_order_and_dup_insensitive() {
        let a = Roster::new(7, vec!["w2".into(), "w1".into(), "w2".into()]);
        let b = Roster::new(7, vec!["w1".into(), "w2".into()]);
        assert_eq!(a, b);
        assert!(a.contains("w1") && a.contains("w2") && !a.contains("w3"));
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn owner_is_deterministic_and_exhaustive() {
        // Two workers computing the same epoch from the same roster produce disjoint,
        // exhaustive shards — per cell there is exactly one owner, and every worker's
        // independent computation agrees on it.
        let r1 = roster(5, &["node-2", "node-3", "tower"]);
        let r2 = roster(5, &["tower", "node-2", "node-3"]); // same set, different input order
        for c in cells(500) {
            let o = hrw_owner(&r1, &c).unwrap();
            assert_eq!(Some(o), hrw_owner(&r2, &c));
            let owners: Vec<&str> = r1
                .workers()
                .iter()
                .map(|s| s.as_str())
                .filter(|w| owns(&r1, w, &c))
                .collect();
            assert_eq!(owners, vec![o], "exactly one owner per cell");
        }
    }

    #[test]
    fn hrw_add_moves_only_the_joiners_cells() {
        let before = roster(5, &["a", "b", "c"]);
        let after = roster(6, &["a", "b", "c", "d"]);
        let mut moved = 0usize;
        let cs = cells(2000);
        for c in &cs {
            let o0 = hrw_owner(&before, c).unwrap().to_string();
            let o1 = hrw_owner(&after, c).unwrap().to_string();
            if o0 != o1 {
                assert_eq!(o1, "d", "an added worker may only pull cells to ITSELF");
                moved += 1;
            }
        }
        // d should own ≈ 1/4 of the cells; sanity-band the move volume.
        assert!(
            (300..=700).contains(&moved),
            "expected ≈500 of 2000 cells to move to d, got {moved}"
        );
    }

    #[test]
    fn hrw_remove_moves_only_the_departed_workers_cells() {
        let before = roster(5, &["a", "b", "c", "d"]);
        let after = roster(6, &["a", "b", "c"]);
        for c in cells(2000) {
            let o0 = hrw_owner(&before, &c).unwrap().to_string();
            let o1 = hrw_owner(&after, &c).unwrap().to_string();
            if o0 != o1 {
                assert_eq!(o0, "d", "only the departed worker's cells may move");
            } else {
                assert_ne!(o0, "d");
            }
        }
    }

    #[test]
    fn shards_are_reasonably_balanced() {
        let r = roster(1, &["w-a", "w-b", "w-c", "w-d", "w-e"]);
        let cs = cells(10_000);
        let mut count: HashMap<&str, usize> = HashMap::new();
        for c in &cs {
            *count.entry(hrw_owner(&r, c).unwrap()).or_default() += 1;
        }
        for w in r.workers() {
            let n = count[w.as_str()];
            // Mean 2000; a deterministic outcome, banded generously (±20%) so the test
            // documents the balance property without depending on exact hash values.
            assert!(
                (1600..=2400).contains(&n),
                "worker {w} owns {n} of 10000 (expected ≈2000)"
            );
        }
    }

    // ── seam decisions ───────────────────────────────────────────────────────────────

    #[test]
    fn stable_roster_fast_paths_everything_owned() {
        let now = roster(9, &["a", "b", "c"]);
        let prev = roster(8, &["c", "b", "a"]);
        for c in cells(300) {
            let d = shard_decision(&now, Some(&prev), "b", &c);
            if owns(&now, "b", &c) {
                assert_eq!(d, ShardDecision::OwnedFast);
            } else {
                assert_eq!(d, ShardDecision::Other);
            }
        }
    }

    #[test]
    fn churn_guards_exactly_the_moved_to_me_cells() {
        let prev = roster(8, &["a", "b", "c", "d"]);
        let now = roster(9, &["a", "b", "c"]); // d died
        let mut guarded = 0usize;
        for c in cells(2000) {
            match shard_decision(&now, Some(&prev), "a", &c) {
                ShardDecision::OwnedFast => {
                    assert!(owns(&prev, "a", &c), "fast requires prior ownership");
                }
                ShardDecision::OwnedGuarded => {
                    // Newly mine — must have been the departed worker's (only d's cells move).
                    assert_eq!(hrw_owner(&prev, &c), Some("d"));
                    guarded += 1;
                }
                ShardDecision::Other => assert!(!owns(&now, "a", &c)),
            }
        }
        assert!(guarded > 0, "some of d's cells must land on a");
    }

    #[test]
    fn unknown_previous_roster_always_guards() {
        let now = roster(3, &["a", "b"]);
        let mine = cells(200)
            .into_iter()
            .find(|c| owns(&now, "a", c))
            .expect("a owns something");
        assert_eq!(
            shard_decision(&now, None, "a", &mine),
            ShardDecision::OwnedGuarded
        );
        let empty = roster(2, &[]);
        assert_eq!(
            shard_decision(&now, Some(&empty), "a", &mine),
            ShardDecision::OwnedGuarded
        );
    }

    #[test]
    fn joiner_owns_nothing_under_a_roster_excluding_it() {
        let now = roster(4, &["a", "b"]);
        for c in cells(100) {
            assert_ne!(hrw_owner(&now, &c), Some("z"));
            assert_eq!(shard_decision(&now, None, "z", &c), ShardDecision::Other);
        }
        assert_eq!(hrw_owner(&roster(1, &[]), "x"), None);
    }

    // ── steal order ──────────────────────────────────────────────────────────────────

    #[test]
    fn steal_order_is_a_deterministic_permutation_and_differs_per_worker() {
        let owned: Vec<String> = cells(64);
        let keys: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
        let a1 = steal_order("stealer-a", &keys);
        let a2 = steal_order("stealer-a", &keys);
        let b = steal_order("stealer-b", &keys);
        assert_eq!(a1, a2, "same worker, same order");
        let mut sorted = a1.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..64).collect::<Vec<_>>(), "a permutation");
        assert_ne!(a1, b, "different workers walk different orders");
    }

    // ── weighted rendezvous (registered speed handicaps) ────────────────────────────

    fn wmap<'a>(pairs: &'a [(&'a str, f64)]) -> impl Fn(&str) -> f64 + 'a {
        move |w: &str| {
            pairs
                .iter()
                .find(|(n, _)| *n == w)
                .map(|(_, v)| *v)
                .unwrap_or(1.0)
        }
    }

    fn owned_counts<'a>(
        roster: &'a Roster,
        weight_of: &dyn Fn(&str) -> f64,
        cs: &[String],
    ) -> HashMap<&'a str, usize> {
        let mut count: HashMap<&str, usize> = HashMap::new();
        for c in cs {
            if let Some(o) = hrw_owner_weighted(roster, weight_of, c) {
                *count.entry(o).or_default() += 1;
            }
        }
        count
    }

    #[test]
    fn weighted_absent_or_all_ones_is_bit_identical_to_unweighted() {
        // The uniform case DELEGATES to hrw_owner — old ownership, bit-for-bit, for every cell.
        let r = roster(1, &["w-a", "w-b", "w-c"]);
        let ones = wmap(&[]);
        let explicit_ones = wmap(&[("w-a", 1.0), ("w-b", 1.0), ("w-c", 1.0)]);
        for c in cells(3000) {
            let old = hrw_owner(&r, &c);
            assert_eq!(hrw_owner_weighted(&r, &ones, &c), old);
            assert_eq!(hrw_owner_weighted(&r, &explicit_ones, &c), old);
        }
    }

    #[test]
    fn weighted_scores_are_pinned_forever() {
        // Same discipline as the unweighted pins: these bytes decide ownership fleet-wide.
        // u01 mapping first (exact IEEE arithmetic — verified by an independent Python impl),
        // then −w/ln(u) via the pure-Rust libm (NEVER the platform libm — vendor rounding
        // would split the fleet's argmax).
        assert_eq!(u01(0).to_bits(), 0x3ca0_0000_0000_0000); // 2⁻⁵³, never 0
        assert_eq!(u01(u64::MAX).to_bits(), 0x3fef_ffff_ffff_ffff); // 1 − 2⁻⁵³, never 1
        let s = hrw_score_weighted("cell-a", "worker-1", 1.0).unwrap();
        assert_eq!(s.to_bits(), 0x4025_2f43_747a_28cb);
        let s2 = hrw_score_weighted("cell-a", "worker-1", 2.0).unwrap();
        assert_eq!(s2.to_bits(), 0x4035_2f43_747a_28cb); // exactly 2× (mul by a power of two)
        assert_eq!(hrw_score_weighted("cell-a", "worker-1", 0.0), None);
        assert_eq!(hrw_score_weighted("cell-a", "worker-1", f64::NAN), None);
    }

    #[test]
    fn weighted_share_is_proportional() {
        // w = 2.0 draws ~2× the cells of a w = 1.0 peer. Deterministic outcome, banded at
        // ±8% relative of the expected share (n = 12000; binomial 3σ ≈ ±1.5%, band is slack).
        let r = roster(1, &["w-a", "w-b", "w-c"]);
        let w = wmap(&[("w-a", 2.0)]);
        let cs = cells(12_000);
        let count = owned_counts(&r, &w, &cs);
        let expect_a = 12_000.0 * 2.0 / 4.0; // 6000
        let expect_bc = 12_000.0 / 4.0; // 3000
        assert!(
            (count["w-a"] as f64 - expect_a).abs() < expect_a * 0.08,
            "w-a owns {} (expected ≈{expect_a})",
            count["w-a"]
        );
        for wk in ["w-b", "w-c"] {
            assert!(
                (count[wk] as f64 - expect_bc).abs() < expect_bc * 0.08,
                "{wk} owns {} (expected ≈{expect_bc})",
                count[wk]
            );
        }
    }

    #[test]
    fn weight_change_moves_only_the_delta_share() {
        // Raising one worker's weight may only PULL cells to that worker; every other pair
        // score is untouched, so no cell may move between the unchanged workers.
        let r = roster(1, &["w-a", "w-b", "w-c"]);
        let before = wmap(&[("w-a", 1.0)]);
        let after = wmap(&[("w-a", 1.2)]);
        let cs = cells(12_000);
        let mut moved = 0usize;
        for c in &cs {
            let o0 = hrw_owner_weighted(&r, &before, c).unwrap();
            let o1 = hrw_owner_weighted(&r, &after, c).unwrap();
            if o0 != o1 {
                assert_eq!(o1, "w-a", "a raised weight may only pull cells to ITSELF");
                moved += 1;
            }
        }
        // Share goes 1/3 → 1.2/3.2 = 0.375: expected movement ≈ 4.17% of cells (500).
        assert!(
            (250..=750).contains(&moved),
            "expected ≈500 of 12000 cells to move to w-a, got {moved}"
        );
    }

    #[test]
    fn zero_weight_excludes_a_worker_entirely() {
        let r = roster(1, &["w-a", "w-b", "w-c"]);
        let w = wmap(&[("w-b", 0.0)]);
        let cs = cells(4000);
        let count = owned_counts(&r, &w, &cs);
        assert_eq!(count.get("w-b"), None, "weight 0 owns nothing");
        assert_eq!(
            count["w-a"] + count["w-c"],
            4000,
            "others absorb the whole set"
        );
        // Everyone excluded → no owner at all.
        let none = wmap(&[("w-a", 0.0), ("w-b", 0.0), ("w-c", 0.0)]);
        assert_eq!(hrw_owner_weighted(&r, &none, &cs[0]), None);
    }

    #[test]
    fn weighted_owner_is_deterministic_across_computers() {
        // Two workers computing the same epoch from the same roster + registry agree on every
        // owner (input order and who asks are irrelevant).
        let r1 = roster(5, &["node-2", "tower", "lianli"]);
        let r2 = roster(5, &["lianli", "node-2", "tower"]);
        let w = wmap(&[("lianli", 1.0), ("tower", 0.67), ("node-2", 0.55)]);
        for c in cells(2000) {
            let o = hrw_owner_weighted(&r1, &w, &c);
            assert_eq!(o, hrw_owner_weighted(&r2, &w, &c));
            assert!(o.is_some());
        }
    }

    #[test]
    fn mixed_codec_wave_shards_each_codec_by_its_own_weights() {
        // One wave, two codecs; a box weighs 2.0 for codec A and 0.5 for codec B. Per-codec
        // shares must be independently proportional WITHIN the same wave.
        let r = roster(1, &["box-x", "box-y", "box-z"]);
        let mut h = Handicaps::default();
        h.0.insert(
            "box-x".into(),
            WorkerHandicap {
                encode: [("codec-a".to_string(), 2.0), ("codec-b".to_string(), 0.5)]
                    .into_iter()
                    .collect(),
                ..Default::default()
            },
        );
        let cs = cells(12_000);
        let mode_a = ShardMode::Encode { codec: "codec-a" };
        let mode_b = ShardMode::Encode { codec: "codec-b" };
        let wa = |wk: &str| h.weight(wk, &mode_a);
        let wb = |wk: &str| h.weight(wk, &mode_b);
        let ca = owned_counts(&r, &wa, &cs);
        let cb = owned_counts(&r, &wb, &cs);
        let ea = 12_000.0 * 2.0 / 4.0; // codec A: x share 2/(2+1+1)
        let eb = 12_000.0 * 0.5 / 2.5; // codec B: x share 0.5/(0.5+1+1)
        assert!(
            (ca["box-x"] as f64 - ea).abs() < ea * 0.08,
            "codec-a: box-x owns {} (expected ≈{ea})",
            ca["box-x"]
        );
        assert!(
            (cb["box-x"] as f64 - eb).abs() < eb * 0.08,
            "codec-b: box-x owns {} (expected ≈{eb})",
            cb["box-x"]
        );
    }

    #[test]
    fn handicap_lookup_defaults_and_mode_mapping() {
        let mut h = Handicaps::default();
        h.0.insert(
            "b1".into(),
            WorkerHandicap {
                encode: [("zenavif".to_string(), 0.72), ("default".to_string(), 0.9)]
                    .into_iter()
                    .collect(),
                metric: 0.6,
                gpu_metric: 0.0,
            },
        );
        // Typed key > default key > 1.0; absent worker → 1.0 everywhere.
        assert_eq!(
            h.weight("b1", &ShardMode::Encode { codec: "zenavif" }),
            0.72
        );
        assert_eq!(h.weight("b1", &ShardMode::Encode { codec: "zenjxl" }), 0.9);
        assert_eq!(h.weight("b1", &ShardMode::Metric), 0.6);
        assert_eq!(h.weight("b1", &ShardMode::GpuMetric), 0.0);
        assert_eq!(
            h.weight("ghost", &ShardMode::Encode { codec: "zenavif" }),
            1.0
        );
        // Negative / NaN registry values clamp to excluded, never poison the argmax.
        let bad = WorkerHandicap {
            metric: -3.0,
            ..Default::default()
        };
        assert_eq!(bad.weight(&ShardMode::Metric), 0.0);
        // Mode mapping: encode by type; the _gpu metric-name suffix selects GpuMetric.
        let enc = JobKind::Encode {
            codec: "zenavif".into(),
            q: 50,
            knobs: "{}".into(),
            hdr: false,
        };
        assert_eq!(shard_mode(&enc), ShardMode::Encode { codec: "zenavif" });
        let gm = JobKind::Metric {
            metric: "butteraugli_max_gpu".into(),
        };
        assert_eq!(shard_mode(&gm), ShardMode::GpuMetric);
        let cm = JobKind::Metric {
            metric: "butteraugli_max".into(),
        };
        assert_eq!(shard_mode(&cm), ShardMode::Metric);
    }

    #[test]
    fn weighted_seam_guards_moved_cells_on_weight_or_roster_change() {
        let now = roster(9, &["a", "b", "c"]);
        let prev = roster(8, &["a", "b", "c", "d"]); // d died
        let w = wmap(&[("a", 2.0), ("d", 1.5)]);
        let mut guarded = 0;
        for c in cells(2000) {
            match shard_decision_weighted(&now, Some(&prev), "a", &c, &w) {
                ShardDecision::OwnedFast => {
                    assert_eq!(hrw_owner_weighted(&prev, &w, &c), Some("a"));
                }
                ShardDecision::OwnedGuarded => {
                    assert_eq!(hrw_owner_weighted(&prev, &w, &c), Some("d"));
                    guarded += 1;
                }
                ShardDecision::Other => {
                    assert_ne!(hrw_owner_weighted(&now, &w, &c), Some("a"));
                }
            }
        }
        assert!(guarded > 0, "d's cells must guard on takeover");
    }

    #[test]
    fn handicaps_serde_shape_matches_the_registry() {
        // The RunControl override and the TOML registry share this serde shape.
        let json = r#"{"lianli":{"encode":{"zenavif":1.0,"default":1.0},"metric":1.0,"gpu_metric":1.0},
                       "tower":{"encode":{"zenavif":0.67},"gpu_metric":0.0}}"#;
        let h: Handicaps = serde_json::from_str(json).unwrap();
        assert_eq!(
            h.weight("tower", &ShardMode::Encode { codec: "zenavif" }),
            0.67
        );
        assert_eq!(
            h.weight("tower", &ShardMode::Encode { codec: "zenjxl" }),
            1.0
        ); // no default key → 1.0
        assert_eq!(h.weight("tower", &ShardMode::Metric), 1.0); // absent field → serde default 1.0
        assert_eq!(h.weight("tower", &ShardMode::GpuMetric), 0.0);
        let rt: Handicaps = serde_json::from_str(&serde_json::to_string(&h).unwrap()).unwrap();
        assert_eq!(rt, h);
    }

    #[test]
    fn claim_mode_parses_and_serdes() {
        assert_eq!(ClaimMode::parse("lease"), Some(ClaimMode::Lease));
        assert_eq!(
            ClaimMode::parse("epoch-sharded"),
            Some(ClaimMode::EpochSharded)
        );
        assert_eq!(
            ClaimMode::parse("EPOCH_SHARDED"),
            Some(ClaimMode::EpochSharded)
        );
        assert_eq!(ClaimMode::parse("nope"), None);
        assert_eq!(ClaimMode::default(), ClaimMode::Lease);
        assert_eq!(
            serde_json::to_string(&ClaimMode::EpochSharded).unwrap(),
            "\"epoch_sharded\""
        );
        let c: EpochShardCfg = serde_json::from_str("{}").unwrap();
        assert_eq!(c, EpochShardCfg::default());
        assert_eq!(c.epoch_len_secs, 600);
        assert_eq!(c.heartbeat_interval_secs, 120);
        assert!(c.tail_steal);
    }
}
