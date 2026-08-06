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

use serde::{Deserialize, Serialize};

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
        self.workers.binary_search_by(|w| w.as_str().cmp(worker)).is_ok()
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn roster(epoch: u64, ws: &[&str]) -> Roster {
        Roster::new(epoch, ws.iter().map(|s| s.to_string()).collect())
    }

    fn cells(n: usize) -> Vec<String> {
        // Shaped like real job ids (hex sha) so balance numbers are representative.
        (0..n).map(|i| format!("{:064x}", (i as u128) * 0x9e37_79b9_7f4a_7c15)).collect()
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
        assert_eq!(shard_decision(&now, None, "a", &mine), ShardDecision::OwnedGuarded);
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

    #[test]
    fn claim_mode_parses_and_serdes() {
        assert_eq!(ClaimMode::parse("lease"), Some(ClaimMode::Lease));
        assert_eq!(ClaimMode::parse("epoch-sharded"), Some(ClaimMode::EpochSharded));
        assert_eq!(ClaimMode::parse("EPOCH_SHARDED"), Some(ClaimMode::EpochSharded));
        assert_eq!(ClaimMode::parse("nope"), None);
        assert_eq!(ClaimMode::default(), ClaimMode::Lease);
        assert_eq!(serde_json::to_string(&ClaimMode::EpochSharded).unwrap(), "\"epoch_sharded\"");
        let c: EpochShardCfg = serde_json::from_str("{}").unwrap();
        assert_eq!(c, EpochShardCfg::default());
        assert_eq!(c.epoch_len_secs, 600);
        assert_eq!(c.heartbeat_interval_secs, 120);
        assert!(c.tail_steal);
    }
}
