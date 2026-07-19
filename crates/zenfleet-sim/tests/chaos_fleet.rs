//! Chaos tests for box lifecycle + teardown, driven through the REAL
//! `zenfleet_core::detect_idle`.
//!
//! These cover the "infra goes wrong" modes: a box that dies mid-run, one that
//! never boots, one that heartbeats but does no GPU work, and — the one the user
//! named — a box that **refuses to tear down** and keeps burning money. Each
//! asserts the canonical detector flags it and a reaper built on it bounds spend.

use zenfleet_sim::{BoxFault, BoxState, SimBox, SimClock, SimFleet};
use zenfleet_core::{IdleReason, IdleThresholds, ResourceClass, Severity, detect_idle};

// A realistic epoch so `booted_at > 0` (detect_idle ignores a last-seen of 0).
const EPOCH: u64 = 1_000_000;

fn busy_gpu(id: &str, booted_at: u64, rate: f64) -> SimBox {
    SimBox::new(
        id,
        "vast",
        ResourceClass::Gpu,
        rate,
        booted_at,
        BoxFault {
            gpu_util_pct: Some(88),
            jobs_per_sec: 0.05,
            ..BoxFault::default()
        },
    )
}

/// A box that crashes mid-run (goes silent) is flagged StaleHeartbeat/Critical
/// once its heartbeat ages past the threshold; a healthy peer is left alone.
#[test]
fn dead_box_is_flagged_stale_and_healthy_one_is_not() {
    let clock = SimClock::new(EPOCH);
    let mut fleet = SimFleet::new(clock.clone());
    fleet.add(busy_gpu("healthy", EPOCH - 3600, 0.40));
    fleet.add(SimBox::new(
        "crashed",
        "vast",
        ResourceClass::Gpu,
        0.40,
        EPOCH - 3600,
        BoxFault {
            gpu_util_pct: Some(88),
            jobs_per_sec: 0.05,
            dies_at: Some(EPOCH - 300), // went silent 300s ago (> 180s stale)
            ..BoxFault::default()
        },
    ));

    let warns = detect_idle(&fleet.reports(), clock.now(), &IdleThresholds::default());
    assert_eq!(warns.len(), 1, "only the crashed box is flagged");
    assert_eq!(warns[0].worker, "crashed");
    assert!(matches!(warns[0].reason, IdleReason::StaleHeartbeat { .. }));
    assert_eq!(warns[0].severity, Severity::Critical);
}

/// A box that never boots (image-pull hang / onstart crash) bills from creation
/// but its "last seen" is stuck at boot — so it flags StaleHeartbeat once past
/// the stale window. The startup-failure case.
#[test]
fn never_booted_box_is_caught_as_a_startup_failure() {
    let clock = SimClock::new(EPOCH);
    let mut fleet = SimFleet::new(clock.clone());
    fleet.add(SimBox::new(
        "stillborn",
        "hetzner",
        ResourceClass::Gpu,
        0.35,
        EPOCH - 300, // created 300s ago, never beat since
        BoxFault {
            never_boots: true,
            gpu_util_pct: Some(0),
            ..BoxFault::default()
        },
    ));

    let warns = detect_idle(&fleet.reports(), clock.now(), &IdleThresholds::default());
    assert_eq!(warns.len(), 1);
    assert!(matches!(warns[0].reason, IdleReason::StaleHeartbeat { .. }));
    assert_eq!(warns[0].severity, Severity::Critical);
}

/// A live box that heartbeats but sits at ~0% GPU is flagged LowGpuUtil; a reaper
/// tears it down, and its billing stops — while the busy box keeps running.
#[test]
fn idle_gpu_box_is_reaped_and_its_billing_stops() {
    let clock = SimClock::new(EPOCH);
    let mut fleet = SimFleet::new(clock.clone());
    fleet.add(busy_gpu("worker", EPOCH - 3600, 0.40));
    fleet.add(SimBox::new(
        "stuck",
        "vast",
        ResourceClass::Gpu,
        0.50,
        EPOCH - 3600,
        BoxFault {
            gpu_util_pct: Some(0), // alive (fresh beat) but doing no GPU work
            jobs_per_sec: 0.0,
            ..BoxFault::default()
        },
    ));

    let out = fleet.reap(&IdleThresholds::default());
    assert_eq!(out.destroyed, 1, "the stuck box is torn down");
    assert_eq!(out.teardown_failed, 0);
    assert_eq!(fleet.live_count(), 1, "the busy box keeps running");

    let stuck_spend_at_reap = fleet
        .boxes
        .iter()
        .find(|b| b.id == "stuck")
        .unwrap()
        .spend_usd(clock.now());

    // An hour later the stuck box has NOT billed more (billing froze at teardown);
    // the busy box has.
    clock.advance(3600);
    let stuck = fleet.boxes.iter().find(|b| b.id == "stuck").unwrap();
    assert_eq!(stuck.state(), BoxState::Destroyed);
    assert!(
        (stuck.spend_usd(clock.now()) - stuck_spend_at_reap).abs() < 1e-9,
        "a reaped box stops burning money"
    );
}

/// The money-leak the user named: a dead box whose `destroy()` FAILS. The reaper
/// keeps flagging + retrying; the box keeps billing until a retry finally
/// succeeds. Asserts the failed-teardown window costs real money and that a
/// retrying reaper eventually stops the bleed.
#[test]
fn failed_teardown_keeps_billing_until_a_retry_succeeds() {
    let clock = SimClock::new(EPOCH);
    let mut fleet = SimFleet::new(clock.clone());
    fleet.add(SimBox::new(
        "wont-die",
        "vast",
        ResourceClass::Gpu,
        0.60,
        EPOCH - 3600,
        BoxFault {
            gpu_util_pct: Some(80),
            jobs_per_sec: 0.05,
            dies_at: Some(EPOCH - 400), // crashed (stale), so it will be flagged...
            teardown_failures: 2,       // ...but the first two teardowns fail
            ..BoxFault::default()
        },
    ));

    let spend_before = fleet.total_spend_usd();

    // Pass 1: flagged + teardown attempted, but it fails — still billing.
    let p1 = fleet.reap(&IdleThresholds::default());
    assert_eq!(p1.destroyed, 0);
    assert_eq!(p1.teardown_failed, 1, "first teardown fails — the leak");
    assert_eq!(fleet.live_count(), 1, "box is still up, still burning");

    // Time passes before the reaper retries — money leaks.
    clock.advance(120);
    let p2 = fleet.reap(&IdleThresholds::default());
    assert_eq!(p2.teardown_failed, 1, "second teardown also fails");

    clock.advance(120);
    let p3 = fleet.reap(&IdleThresholds::default());
    assert_eq!(p3.destroyed, 1, "third teardown finally succeeds");
    assert_eq!(fleet.live_count(), 0, "the bleed is stopped");

    let spend_after = fleet.total_spend_usd();
    assert!(
        spend_after > spend_before,
        "the failed-teardown window billed real money ({spend_before:.4} -> {spend_after:.4})"
    );
}

/// The headline: a reaper built on `detect_idle` bounds fleet spend, vs. a
/// launch-and-forget fleet that burns for the whole run. Same two boxes, same
/// timeline; the only difference is whether idle boxes get reaped.
#[test]
fn a_reaper_bounds_spend_versus_launch_and_forget() {
    // Two boxes that go silent a minute after boot and never recover.
    let mk = |clock: &SimClock| {
        let mut f = SimFleet::new(clock.clone());
        for i in 0..2 {
            f.add(SimBox::new(
                format!("z{i}"),
                "vast",
                ResourceClass::Gpu,
                1.00, // $1/hr each
                EPOCH,
                BoxFault {
                    gpu_util_pct: Some(70),
                    jobs_per_sec: 0.05,
                    dies_at: Some(EPOCH + 60), // silent after 1 min
                    ..BoxFault::default()
                },
            ));
        }
        f
    };

    // Launch-and-forget: nobody reaps; run a full hour.
    let clock_a = SimClock::new(EPOCH);
    let fleet_a = mk(&clock_a);
    clock_a.advance(3600);
    let burned_no_reaper = fleet_a.total_spend_usd(); // ~2 * $1 = $2

    // With a reaper that checks a few minutes in.
    let clock_b = SimClock::new(EPOCH);
    let mut fleet_b = mk(&clock_b);
    clock_b.advance(300); // past the stale window
    let out = fleet_b.reap(&IdleThresholds::default());
    assert_eq!(out.destroyed, 2, "both dead boxes reaped");
    clock_b.advance(3600); // rest of the hour — but they're already gone
    let burned_with_reaper = fleet_b.total_spend_usd(); // ~2 * $1 * (300/3600)

    assert!(
        burned_with_reaper < burned_no_reaper * 0.25,
        "reaping cut spend by >4x: reaper ${burned_with_reaper:.3} vs forget ${burned_no_reaper:.3}"
    );
}
