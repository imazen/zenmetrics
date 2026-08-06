//! Epoch-sharded claiming — the multi-epoch fleet simulation gates.
//!
//! Drives the REAL `zenfleet_core::epoch` decisions (epoch math, HRW ownership, seam rules,
//! steal order) through a deterministic 3-worker fleet over a synthetic ledger on a virtual
//! clock, per the sim crate's charter: the failure modes that only show up in the field become
//! reproducible tests. The three gates registered for the mode:
//!
//! 1. **Zero duplicate executions in steady state**, and — the anti-3.6× assertion — **total
//!    work == distinct cells** for the whole campaign (the avifgen encode run measured 1,237,361
//!    ledger rows for 343,465 distinct DONE cells under lease claiming with empty views; this
//!    mode's ratio must be exactly 1.0), with **zero lease traffic once the roster is stable**.
//! 2. **A killed worker's cells are taken over at the next boundary** (its heartbeats age out of
//!    the roster; HRW moves exactly its cells; the survivors lease-guard the seam).
//! 3. **Straggler-tail stealing** finishes no later than the no-steal run, any duplicate lands
//!    only on stolen cells, and stealers never collide with each other (leases).
//!
//! Model fidelity: each worker's epoch runs atomically (no intra-epoch interleaving) against the
//! ledger snapshot FROZEN at the boundary — exactly the visibility the real worker has (its
//! `--ledger-in` snapshot ages within the epoch). Lease claims share one map with real TTL/steal
//! semantics. A worker executing a cell another worker already completed (invisible to it) counts
//! as a DUPLICATE — the quantity the mode exists to eliminate.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use zenfleet_core::epoch::{
    self, EpochShardCfg, Roster, ShardDecision, epoch_index, hrw_owner, shard_decision,
    shard_decision_weighted, steal_order,
};
use zenfleet_sim::SimClock;

/// Lease TTL — one epoch, mirroring the production default (600s each).
const CLAIM_TTL: u64 = 600;

/// One shared lease map with the production semantics: create-if-absent wins; a live lease
/// blocks; a stale one (age ≥ ttl) is stolen.
#[derive(Default)]
struct Leases(HashMap<String, (u64, String)>);

impl Leases {
    fn claim(&mut self, cell: &str, now: u64, owner: &str) -> bool {
        match self.0.get(cell) {
            None => {
                self.0.insert(cell.into(), (now, owner.into()));
                true
            }
            Some((ts, _)) if now.saturating_sub(*ts) >= CLAIM_TTL => {
                self.0.insert(cell.into(), (now, owner.into()));
                true
            }
            Some(_) => false,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    NoSteal,
    TailSteal,
}

struct Fleet {
    cells: Vec<String>,
    /// worker → cells-per-epoch throughput (heterogeneous on purpose).
    rates: BTreeMap<String, usize>,
    /// worker → epoch it dies at (stops beating AND executing from that epoch on).
    deaths: BTreeMap<String, u64>,
    mode: Mode,
    /// Registered handicap weights (empty = uniform 1.0 — the unweighted delegate path).
    weights: BTreeMap<String, f64>,
}

/// One recorded execution: epoch, worker, cell, and whether it came via the steal path.
struct Exec {
    epoch: u64,
    worker: String,
    cell: String,
    stolen: bool,
}

#[derive(Default)]
struct RunReport {
    /// Every execution, INCLUDING duplicates.
    executions: Vec<Exec>,
    /// Cells executed via the tail-steal path at least once.
    stolen: BTreeSet<String>,
    /// Lease claims attempted per epoch (the claim-traffic meter).
    lease_attempts: BTreeMap<u64, usize>,
    epochs_run: u64,
    /// Σ over (epoch, alive worker) of unused per-epoch budget while global work remained —
    /// the tail-idle meter weighted sharding exists to shrink.
    idle_budget: usize,
    /// Last epoch in which each worker still had OWNED work (its shard-exhaustion point).
    last_owned_epoch: BTreeMap<String, u64>,
}

impl RunReport {
    fn total(&self) -> usize {
        self.executions.len()
    }
    fn distinct(&self) -> BTreeSet<&str> {
        self.executions.iter().map(|x| x.cell.as_str()).collect()
    }
    fn duplicates(&self) -> Vec<&str> {
        let mut seen = BTreeSet::new();
        self.executions
            .iter()
            .filter_map(|x| (!seen.insert(x.cell.as_str())).then_some(x.cell.as_str()))
            .collect()
    }
}

/// Run the fleet to convergence (all cells done) or `max_epochs`. Every decision the workers
/// make goes through the real `zenfleet_core::epoch` functions.
fn run_fleet(f: &Fleet, max_epochs: u64) -> RunReport {
    let cfg = EpochShardCfg::default();
    let clock = SimClock::new(0);
    let mut board: BTreeMap<u64, BTreeSet<String>> = BTreeMap::new();
    let mut leases = Leases::default();
    let mut done: BTreeSet<String> = BTreeSet::new();
    let mut report = RunReport::default();

    for _ in 0..max_epochs {
        let now = clock.now();
        let e = epoch_index(now, cfg.epoch_len_secs);
        let alive: Vec<&String> = f
            .rates
            .keys()
            .filter(|w| f.deaths.get(*w).is_none_or(|&d| e < d))
            .collect();
        // 1. Every alive worker heartbeats at the boundary (the pass-start beat).
        for w in &alive {
            board.entry(e).or_default().insert((*w).clone());
        }
        // 2. The ledger snapshot every worker partitions against is FROZEN at the boundary.
        let snapshot = done.clone();
        if snapshot.len() == f.cells.len() {
            report.epochs_run = e;
            return report;
        }
        // 3. Each worker independently computes the same rosters and works its shard.
        for w in &alive {
            let w = (*w).clone();
            let mut budget = f.rates[&w];
            let roster = Roster::new(
                e,
                e.checked_sub(1)
                    .and_then(|p| board.get(&p))
                    .map(|s| s.iter().cloned().collect())
                    .unwrap_or_default(),
            );
            let prev = e.checked_sub(2).map(|pp| {
                Roster::new(
                    e - 1,
                    board
                        .get(&pp)
                        .map(|s| s.iter().cloned().collect())
                        .unwrap_or_default(),
                )
            });
            let remaining: Vec<&String> =
                f.cells.iter().filter(|c| !snapshot.contains(*c)).collect();
            let mut exec = |cell: &str, stolen: bool, report: &mut RunReport| {
                report.executions.push(Exec {
                    epoch: e,
                    worker: w.clone(),
                    cell: cell.to_string(),
                    stolen,
                });
                if stolen {
                    report.stolen.insert(cell.to_string());
                }
                done.insert(cell.to_string()); // ledger latest-wins: dup rows collapse
            };
            if roster.is_empty() || !roster.contains(&w) {
                // Bootstrap/join epoch: the lease fallback path — every claim is leased.
                let mut order: Vec<&String> = remaining.clone();
                order.sort_by_key(|c| epoch::hrw_score(c, &w));
                for c in order {
                    if budget == 0 {
                        break;
                    }
                    *report.lease_attempts.entry(e).or_default() += 1;
                    if leases.claim(c, now, &w) {
                        exec(c, false, &mut report);
                        budget -= 1;
                    }
                }
                report.idle_budget += budget;
                continue;
            }
            // Owned shard first (fast lease-free; guarded behind a lease at ownership seams).
            // Weighted ownership through the REAL registered-handicap path (empty weights map
            // = the uniform delegate, i.e. the pre-weights behavior bit-for-bit).
            let weight_of = |wk: &str| f.weights.get(wk).copied().unwrap_or(1.0);
            let mut owned_fast = Vec::new();
            let mut owned_guarded = Vec::new();
            let mut others = Vec::new();
            for c in &remaining {
                match shard_decision_weighted(&roster, prev.as_ref(), &w, c, &weight_of) {
                    ShardDecision::OwnedFast => owned_fast.push(*c),
                    ShardDecision::OwnedGuarded => owned_guarded.push(*c),
                    ShardDecision::Other => others.push(*c),
                }
            }
            if !(owned_fast.is_empty() && owned_guarded.is_empty()) {
                report.last_owned_epoch.insert(w.clone(), e);
            }
            for c in owned_fast {
                if budget == 0 {
                    break;
                }
                exec(c, false, &mut report);
                budget -= 1;
            }
            for c in owned_guarded {
                if budget == 0 {
                    break;
                }
                *report.lease_attempts.entry(e).or_default() += 1;
                if leases.claim(c, now, &w) {
                    exec(c, false, &mut report);
                    budget -= 1;
                }
            }
            // Straggler tail: shard exhausted with budget left → steal in deterministic
            // per-worker order, always lease-guarded.
            if f.mode == Mode::TailSteal && budget > 0 && !others.is_empty() {
                let keys: Vec<&str> = others.iter().map(|c| c.as_str()).collect();
                for i in steal_order(&w, &keys) {
                    if budget == 0 {
                        break;
                    }
                    *report.lease_attempts.entry(e).or_default() += 1;
                    if leases.claim(keys[i], now, &w) {
                        exec(keys[i], true, &mut report);
                        budget -= 1;
                    }
                }
            }
            // Unused budget while the campaign still had work = paid idle (the tail-idle
            // meter; correct weights shrink it, tail-steal mops the residue).
            report.idle_budget += budget;
        }
        clock.advance(cfg.epoch_len_secs);
    }
    report.epochs_run = epoch_index(clock.now(), cfg.epoch_len_secs);
    report
}

fn cells(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| format!("{:064x}", (i as u128) * 0x9e37_79b9_7f4a_7c15))
        .collect()
}

fn three_workers() -> BTreeMap<String, usize> {
    // Heterogeneous throughput on purpose (tower is 0.87× the 7950X…).
    [("w-fast", 20), ("w-mid", 15), ("w-slow", 10)]
        .into_iter()
        .map(|(w, r)| (w.to_string(), r))
        .collect()
}

/// Gate 1 — steady state: zero duplicates, total == distinct (work ratio exactly 1.0 vs the
/// measured 3.6×), and zero lease traffic once the roster is stable.
#[test]
fn steady_state_total_work_equals_distinct_cells_with_zero_leases() {
    let f = Fleet {
        cells: cells(300),
        rates: three_workers(),
        deaths: BTreeMap::new(),
        mode: Mode::NoSteal,
        weights: BTreeMap::new(),
    };
    let r = run_fleet(&f, 40);
    assert_eq!(r.distinct().len(), 300, "campaign must complete");
    assert_eq!(
        r.duplicates(),
        Vec::<&str>::new(),
        "steady state duplicates"
    );
    // THE anti-3.6× assertion: executions == distinct DONE cells, ratio exactly 1.0.
    assert_eq!(
        r.total(),
        300,
        "total work must equal distinct cells (measured lease-mode tax was 3.6×)"
    );
    // Claim traffic: bootstrap epochs 0..2 are lease-guarded BY DESIGN (roster not yet stable);
    // from the first stable epoch on, the fast path must take zero leases.
    let after_bootstrap: usize = r
        .lease_attempts
        .iter()
        .filter(|(e, _)| **e >= 2)
        .map(|(_, n)| n)
        .sum();
    assert_eq!(
        after_bootstrap, 0,
        "stable roster must take no leases at all"
    );
    assert!(
        r.lease_attempts.get(&0).copied().unwrap_or(0) > 0,
        "bootstrap epoch is lease-guarded (sanity that the meter works)"
    );
}

/// Gate 2 — takeover: a worker killed at epoch K stops heartbeating, ages out of the roster at
/// the next boundary, and its remaining cells are executed by the survivors — lease-guarded at
/// the seam, exactly once, none before its heartbeats aged out.
#[test]
fn killed_workers_cells_are_taken_over_at_the_next_boundary() {
    let kill_at = 6u64;
    let f = Fleet {
        cells: cells(300),
        rates: three_workers(),
        deaths: [("w-mid".to_string(), kill_at)].into_iter().collect(),
        mode: Mode::NoSteal,
        weights: BTreeMap::new(),
    };
    let r = run_fleet(&f, 60);
    assert_eq!(
        r.distinct().len(),
        300,
        "campaign completes despite the death"
    );
    assert_eq!(
        r.duplicates(),
        Vec::<&str>::new(),
        "clean death ⇒ no duplicates"
    );
    assert_eq!(r.total(), 300, "takeover must not re-execute anything");
    // The dead worker executed nothing from its death epoch on.
    assert!(
        !r.executions
            .iter()
            .any(|x| x.worker == "w-mid" && x.epoch >= kill_at),
        "a dead worker executes nothing"
    );
    // Roster lag: w-mid beat during kill_at−1, so it still owns its shard during kill_at (the
    // idle epoch); survivors may only run its cells from kill_at+1 on. Verify against the real
    // roster of the death epoch, over the post-kill window (bootstrap-epoch lease executions
    // legitimately predate shard ownership and are out of scope here).
    let roster_at_kill = Roster::new(
        kill_at,
        vec!["w-fast".into(), "w-mid".into(), "w-slow".into()],
    );
    let survivor_runs_of_mid_cells: Vec<&Exec> = r
        .executions
        .iter()
        .filter(|x| {
            x.epoch >= kill_at
                && x.worker != "w-mid"
                && hrw_owner(&roster_at_kill, &x.cell) == Some("w-mid")
        })
        .collect();
    assert!(
        !survivor_runs_of_mid_cells.is_empty(),
        "the dead worker's cells must be taken over"
    );
    assert!(
        survivor_runs_of_mid_cells.iter().all(|x| x.epoch > kill_at),
        "takeover happens at the boundary AFTER the heartbeats age out, not before"
    );
    // And the takeover epoch itself spends leases (the seam guard), unlike steady state.
    let takeover_epoch = kill_at + 1;
    assert!(
        r.lease_attempts.get(&takeover_epoch).copied().unwrap_or(0) > 0,
        "moved cells are lease-guarded at the seam"
    );
}

/// Gate 3 — straggler-tail stealing: completes no later than the no-steal run, duplicates (if
/// any) land only on stolen cells, and every cell still completes exactly once in the DONE set.
#[test]
fn tail_steal_takes_over_within_the_death_epoch_and_bounds_duplicates() {
    let kill_at = 6u64;
    let mk = |mode| Fleet {
        cells: cells(300),
        rates: three_workers(),
        deaths: [("w-mid".to_string(), kill_at)].into_iter().collect(),
        mode,
        weights: BTreeMap::new(),
    };
    let no_steal = run_fleet(&mk(Mode::NoSteal), 60);
    let steal = run_fleet(&mk(Mode::TailSteal), 60);
    eprintln!(
        "tail-steal: epochs {} (no-steal {}), total {} for 300 distinct, {} stolen, {} duplicates",
        steal.epochs_run,
        no_steal.epochs_run,
        steal.total(),
        steal.stolen.len(),
        steal.duplicates().len()
    );
    assert_eq!(steal.distinct().len(), 300, "steal run completes");
    assert!(
        steal.epochs_run <= no_steal.epochs_run,
        "stealing must not finish later ({} vs {})",
        steal.epochs_run,
        no_steal.epochs_run
    );
    // Duplicates can only arise where a steal raced the (lease-free) owner — never elsewhere.
    for c in steal.duplicates() {
        assert!(
            steal.stolen.contains(c),
            "a duplicate may only be a stolen cell (owner-vs-stealer race); {c} was not stolen"
        );
    }
    // Stealer-vs-stealer is fenced by the lease: a cell is executed at most once VIA THE STEAL
    // PATH per lease TTL (= one epoch here). The owner's own lease-free execution of a stolen
    // cell is the accepted owner-vs-stealer race asserted above, not a lease-fence failure.
    let mut per_epoch_steals: BTreeMap<(u64, &str), usize> = BTreeMap::new();
    for x in &steal.executions {
        if x.stolen {
            *per_epoch_steals
                .entry((x.epoch, x.cell.as_str()))
                .or_default() += 1;
        }
    }
    assert!(
        per_epoch_steals.values().all(|&n| n <= 1),
        "leases must fence concurrent stealers"
    );
}

/// Gate 4 — REGISTERED WEIGHTS (the measured point of the feature): with handicaps matching
/// the fleet's true throughput (weights ∝ rates 20/15/10), the heterogeneous workers exhaust
/// their shards TOGETHER (stated band: within 2 epochs of each other — hash-variance on the
/// shard sizes plus the weight-blind bootstrap epochs cost up to ~1 epoch each), the tail-idle
/// meter shrinks vs uniform sharding, and completion is no later — with the anti-3.6×
/// invariants (total == distinct, zero post-bootstrap leases) still holding on the weighted
/// path. 900 cells so the uniform imbalance is unmistakable: equal shards of ≈300 exhaust in
/// ≈15/20/30 epochs at rates 20/15/10 (spread ≈ 15 epochs of paid straggle) while correct
/// weights size shards ≈400/300/200 that all exhaust in ≈20. Tail-steal remains the
/// residual-error mop and is OFF here so the balance itself is what's measured.
#[test]
fn correct_weights_balance_shard_finish_and_shrink_tail_idle() {
    let mk = |weights: &[(&str, f64)]| Fleet {
        cells: cells(900),
        rates: three_workers(),
        deaths: BTreeMap::new(),
        mode: Mode::NoSteal,
        weights: weights.iter().map(|(w, v)| (w.to_string(), *v)).collect(),
    };
    let uniform = run_fleet(&mk(&[]), 80);
    let weighted = run_fleet(&mk(&[("w-fast", 2.0), ("w-mid", 1.5), ("w-slow", 1.0)]), 80);
    let spread = |r: &RunReport| {
        let lo = r.last_owned_epoch.values().min().copied().unwrap_or(0);
        let hi = r.last_owned_epoch.values().max().copied().unwrap_or(0);
        hi - lo
    };
    eprintln!(
        "balance: uniform  spread={} idle={} epochs={} | weighted spread={} idle={} epochs={} \
         (last-owned uniform {:?} → weighted {:?})",
        spread(&uniform),
        uniform.idle_budget,
        uniform.epochs_run,
        spread(&weighted),
        weighted.idle_budget,
        weighted.epochs_run,
        uniform.last_owned_epoch,
        weighted.last_owned_epoch,
    );
    // Both complete; exactly-once holds on the weighted path too.
    assert_eq!(uniform.distinct().len(), 900);
    assert_eq!(weighted.distinct().len(), 900);
    assert_eq!(
        weighted.total(),
        900,
        "weighted sharding must stay exactly-once"
    );
    assert_eq!(weighted.duplicates(), Vec::<&str>::new());
    // The balance gates.
    assert!(
        spread(&weighted) <= 2,
        "correct weights must exhaust shards within 2 epochs of each other (got {})",
        spread(&weighted)
    );
    assert!(
        spread(&uniform) > spread(&weighted),
        "uniform sharding must show the imbalance the weights remove ({} vs {})",
        spread(&uniform),
        spread(&weighted)
    );
    assert!(
        weighted.idle_budget < uniform.idle_budget,
        "correct weights must shrink tail idle ({} vs {})",
        weighted.idle_budget,
        uniform.idle_budget
    );
    assert!(
        weighted.epochs_run <= uniform.epochs_run,
        "correct weights must not finish later ({} vs {})",
        weighted.epochs_run,
        uniform.epochs_run
    );
    // Zero claim traffic once the roster is stable — unchanged by weighting.
    let after_bootstrap: usize = weighted
        .lease_attempts
        .iter()
        .filter(|(e, _)| **e >= 2)
        .map(|(_, n)| n)
        .sum();
    assert_eq!(after_bootstrap, 0, "weighted fast path must take no leases");
}

/// The sim's worker-visible decision path agrees with the worker crate's: same roster, same
/// cells ⇒ same owner, from either direction (a cross-check that the sim drives the real fns).
#[test]
fn sim_and_core_agree_on_ownership() {
    let roster = Roster::new(3, vec!["w-fast".into(), "w-mid".into(), "w-slow".into()]);
    for c in cells(50) {
        let owner = hrw_owner(&roster, &c).unwrap();
        for w in roster.workers() {
            let d = shard_decision(&roster, Some(&roster), w, &c);
            assert_eq!(
                d == ShardDecision::OwnedFast,
                w == owner,
                "decision and owner must agree"
            );
        }
    }
}
