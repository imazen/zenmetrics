//! TDD contract: the scheduler must keep the box's compute units saturated.
//!
//! These tests define the target behavior; the implementation in `sched.rs` is
//! grown to satisfy them. Written red-first: `schedule_max_util` starts as the
//! naive serial policy (which fails the utilization floor) and is then
//! implemented to pass.

use zenfleet_sim::{BoxCap, Task, schedule_fleet, schedule_max_util};

const MB: u64 = 1 << 20;
const GB: u64 = 1 << 30;

fn gpu_task(fetch: u64, compute: u64, upload: u64, mem: u64) -> Task {
    Task {
        fetch_secs: fetch,
        compute_secs: compute,
        upload_secs: upload,
        mem_bytes: mem,
        threads: 1,
        gpu: true,
    }
}

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

/// CYCLE 2 — the GPU lane must stay fed. A GPU task's inputs must be uploaded to
/// the device (fetch) before compute; without prefetch the lane idles on every
/// H2D transfer (the measured "GPU starves on uploads" case). The scheduler must
/// keep the single lane near-fully busy by fetching the next task ahead.
#[test]
fn gpu_lane_stays_fed_despite_h2d_upload_latency() {
    let cap = BoxCap::new(8, 24 * GB, 1); // 1 GPU lane
    // 80 GPU tasks: 2s H2D fetch, 3s GPU compute, 1s readback.
    let tasks = vec![gpu_task(2, 3, 1, 200 * MB); 80];

    let s = schedule_max_util(&cap, &tasks);

    assert_eq!(s.tasks_done, 80);
    assert!(
        s.gpu_util(1) >= 0.90,
        "the GPU lane must stay >=90% busy; got {:.3} (wall={}s)",
        s.gpu_util(1),
        s.wall_secs
    );
}

/// CYCLE 3 — work-stealing keeps every box busy under an imbalanced workload. A
/// static contiguous split hands one box all the heavy tasks; it grinds while the
/// others finish early and sit idle (wasted, paid-for capacity). A shared pull
/// queue balances the load and keeps the whole fleet's cores busy.
#[test]
fn work_stealing_keeps_the_whole_fleet_busy_under_imbalance() {
    let boxes = vec![BoxCap::new(8, 24 * GB, 0); 3];
    // 120 tasks; the last third are 10x heavier — a contiguous split dumps them
    // all on box 2.
    let mut tasks: Vec<Task> = (0..80).map(|_| Task::light(2, 3, 1, 100 * MB)).collect();
    tasks.extend((0..40).map(|_| Task::light(2, 30, 1, 100 * MB)));

    let s = schedule_fleet(&boxes, &tasks, &[]);

    assert_eq!(s.tasks_done, 120, "every task completes");
    assert!(
        s.cpu_util() >= 0.85,
        "the fleet must stay >=85% utilized under imbalance; got {:.3} (wall={}s, per-box done={:?})",
        s.cpu_util(),
        s.wall_secs,
        s.per_box_done
    );
}

/// CYCLE 4 — resilience: a box that dies mid-run must not strand its work. The
/// scheduler must reclaim the dead box's in-flight tasks and complete them
/// elsewhere, so EVERY task still finishes. A static split loses them.
#[test]
fn a_dead_box_does_not_strand_its_work() {
    let boxes = vec![BoxCap::new(8, 24 * GB, 0); 3];
    let tasks = vec![Task::light(2, 4, 1, 100 * MB); 150];

    // Box 1 dies 20 seconds in — mid-run, with work in flight and queued.
    let s = schedule_fleet(&boxes, &tasks, &[(1, 20)]);

    assert_eq!(
        s.tasks_done, 150,
        "all work completes despite a mid-run box death (per-box done={:?})",
        s.per_box_done
    );
}
