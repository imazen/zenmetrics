//! TDD contract: the scheduler must keep the box's compute units saturated.
//!
//! These tests define the target behavior; the implementation in `sched.rs` is
//! grown to satisfy them. Written red-first: `schedule_max_util` starts as the
//! naive serial policy (which fails the utilization floor) and is then
//! implemented to pass.

use zenfleet_sim::{BoxCap, Task, schedule_max_util};

const MB: u64 = 1 << 20;
const GB: u64 = 1 << 30;

/// CYCLE 1 — a box fed plenty of work must run its cores near-fully, even though
/// every task spends real time fetching inputs and uploading outputs. The naive
/// serial worker fails this (I/O idles the cores; only one core is ever used).
#[test]
fn cpu_stays_saturated_despite_per_task_io_latency() {
    let cap = BoxCap::new(8, 24 * GB, 0);
    // 80 light tasks: 2s fetch, 3s compute (1 core), 1s upload. I/O is 50% of
    // each task's wall time — exactly what starves a serial worker's cores.
    let tasks = vec![Task::light(2, 3, 1, 100 * MB); 80];

    let s = schedule_max_util(&cap, &tasks);

    assert_eq!(s.tasks_done, 80, "every task completes");
    assert!(
        s.cpu_util(8) >= 0.90,
        "the scheduler must keep 8 cores >=90% busy; got {:.3} (wall={}s)",
        s.cpu_util(8),
        s.wall_secs
    );
}
