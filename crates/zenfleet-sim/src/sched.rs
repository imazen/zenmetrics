//! A utilization-maximizing scheduler, developed TDD against the simulator.
//!
//! The goal the tests pin down: **a box must keep its compute units (cores, GPU
//! lanes) saturated whenever there is queued work**, regardless of per-task I/O
//! latency (fetch inputs / upload outputs). A naive serial worker — fetch, then
//! compute, then upload, one task at a time — leaves the CPU idle most of the
//! time (the cores sit dark during every fetch and upload, and only one core is
//! ever used). The max-util scheduler overlaps I/O with compute (prefetch) and
//! admits as many tasks to compute as the box's resource envelope allows.
//!
//! Compute admission uses the REAL [`zenfleet_core::BoxBudget::can_admit`] — the
//! dormant core primitive from the fleet inventory — so these tests double as
//! proof that turning it on is what maximizes utilization.
//!
//! The engine is a deterministic 1-second-tick discrete-event simulation: each
//! task moves Fetch → Ready → Compute → Upload, compute occupies `threads` cores
//! (and a GPU lane if `gpu`), and I/O phases overlap compute freely. Utilization
//! is measured as busy compute-unit-seconds over wall-seconds × units.

use std::collections::VecDeque;

use zenfleet_core::{BoxBudget, InFlight};

/// One unit of work with its timing + resource profile (whole seconds, bytes).
#[derive(Clone, Copy, Debug)]
pub struct Task {
    /// I/O to fetch inputs before compute can start (does not use a core).
    pub fetch_secs: u64,
    /// Compute time on `threads` cores (or a GPU lane if `gpu`).
    pub compute_secs: u64,
    /// I/O to persist the output after compute (does not use a core).
    pub upload_secs: u64,
    /// Peak memory while resident.
    pub mem_bytes: u64,
    /// Cores this task's compute occupies.
    pub threads: u32,
    /// Whether compute occupies a GPU lane instead of being pure CPU.
    pub gpu: bool,
}

impl Task {
    /// A light single-thread task with the given phase times and memory.
    pub fn light(fetch: u64, compute: u64, upload: u64, mem_bytes: u64) -> Self {
        Self {
            fetch_secs: fetch,
            compute_secs: compute,
            upload_secs: upload,
            mem_bytes,
            threads: 1,
            gpu: false,
        }
    }
}

/// A box's compute envelope.
#[derive(Clone, Copy, Debug)]
pub struct BoxCap {
    pub budget: BoxBudget,
    pub gpu_lanes: u32,
}

impl BoxCap {
    pub fn new(cores: u32, ram_bytes: u64, gpu_lanes: u32) -> Self {
        Self {
            budget: BoxBudget::new(ram_bytes, cores),
            gpu_lanes,
        }
    }
    pub fn cores(&self) -> u32 {
        self.budget.cores
    }
}

/// Admission policy the engine runs under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// One task through the whole pipeline at a time — the naive baseline.
    Serial,
    /// Prefetch ahead and admit to compute up to the resource envelope.
    MaxUtil,
}

/// Outcome of a scheduled run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RunStats {
    pub wall_secs: u64,
    /// Σ core-seconds actually spent in compute.
    pub cpu_busy_secs: u64,
    /// Σ GPU-lane-seconds actually spent in compute.
    pub gpu_busy_secs: u64,
    pub tasks_done: usize,
}

impl RunStats {
    /// CPU utilization: busy core-seconds / (wall × cores), in `[0, 1]`.
    pub fn cpu_util(&self, cores: u32) -> f64 {
        if self.wall_secs == 0 || cores == 0 {
            return 0.0;
        }
        self.cpu_busy_secs as f64 / (self.wall_secs * cores as u64) as f64
    }
    /// GPU utilization: busy lane-seconds / (wall × lanes), in `[0, 1]`.
    pub fn gpu_util(&self, lanes: u32) -> f64 {
        if self.wall_secs == 0 || lanes == 0 {
            return 0.0;
        }
        self.gpu_busy_secs as f64 / (self.wall_secs * lanes as u64) as f64
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Fetch,
    Ready,
    Compute,
    Upload,
}

#[derive(Clone)]
struct Slot {
    task: Task,
    phase: Phase,
    remaining: u64,
}

/// Run `tasks` on `cap` under `policy` and report utilization. Deterministic.
pub fn run(cap: &BoxCap, tasks: &[Task], policy: Policy) -> RunStats {
    let mut queue: VecDeque<Task> = tasks.iter().copied().collect();
    let mut slots: Vec<Slot> = Vec::new();
    let (mut cpu_busy, mut gpu_busy, mut done) = (0u64, 0u64, 0usize);
    let mut wall = 0u64;
    let cap_wall = 100_000_000u64; // safety valve against a scheduling bug

    let prefetch_target = match policy {
        Policy::Serial => 1,
        // Enough ready-ahead tasks that a core is never left waiting on a fetch.
        Policy::MaxUtil => cap.cores() as usize + 4,
    };

    loop {
        match policy {
            Policy::Serial => {
                // Nothing new starts until the current task is fully done.
                if slots.is_empty()
                    && let Some(t) = queue.pop_front()
                {
                    slots.push(Slot {
                        task: t,
                        phase: Phase::Fetch,
                        remaining: t.fetch_secs,
                    });
                }
                if let Some(i) = slots.iter().position(|s| s.phase == Phase::Ready) {
                    slots[i].phase = Phase::Compute;
                    slots[i].remaining = slots[i].task.compute_secs.max(1);
                }
            }
            Policy::MaxUtil => {
                // Admit ready tasks to compute while the envelope allows.
                loop {
                    let (mut running, mut gpu_used) = (InFlight::default(), 0u32);
                    for s in &slots {
                        if s.phase == Phase::Compute {
                            running.add(s.task.mem_bytes, s.task.threads);
                            if s.task.gpu {
                                gpu_used += 1;
                            }
                        }
                    }
                    let pick = slots.iter().position(|s| {
                        s.phase == Phase::Ready
                            && cap.budget.can_admit(&running, s.task.mem_bytes, s.task.threads)
                            && (!s.task.gpu || gpu_used < cap.gpu_lanes)
                    });
                    match pick {
                        Some(i) => {
                            slots[i].phase = Phase::Compute;
                            slots[i].remaining = slots[i].task.compute_secs.max(1);
                        }
                        None => break,
                    }
                }
                // Prefetch: keep `prefetch_target` tasks fetching/ready ahead.
                while slots
                    .iter()
                    .filter(|s| s.phase == Phase::Fetch || s.phase == Phase::Ready)
                    .count()
                    < prefetch_target
                {
                    match queue.pop_front() {
                        Some(t) => slots.push(Slot {
                            task: t,
                            phase: Phase::Fetch,
                            remaining: t.fetch_secs,
                        }),
                        None => break,
                    }
                }
            }
        }

        if slots.is_empty() && queue.is_empty() {
            break;
        }
        if wall >= cap_wall {
            break;
        }

        // Account this tick (compute occupancy, capped at capacity).
        let (mut cores_used, mut lanes_used) = (0u32, 0u32);
        for s in &slots {
            if s.phase == Phase::Compute {
                cores_used += s.task.threads;
                if s.task.gpu {
                    lanes_used += 1;
                }
            }
        }
        cpu_busy += cores_used.min(cap.cores()) as u64;
        gpu_busy += lanes_used.min(cap.gpu_lanes) as u64;

        // Advance one second.
        for s in &mut slots {
            if matches!(s.phase, Phase::Fetch | Phase::Compute | Phase::Upload) && s.remaining > 0 {
                s.remaining -= 1;
            }
        }
        // Phase transitions for anything that just finished.
        let mut i = 0;
        while i < slots.len() {
            if slots[i].remaining == 0 {
                match slots[i].phase {
                    Phase::Fetch => {
                        slots[i].phase = Phase::Ready;
                        i += 1;
                    }
                    Phase::Ready => i += 1, // waits for a compute slot next tick
                    Phase::Compute => {
                        slots[i].phase = Phase::Upload;
                        slots[i].remaining = slots[i].task.upload_secs;
                        i += 1;
                    }
                    Phase::Upload => {
                        slots.remove(i);
                        done += 1;
                    }
                }
            } else {
                i += 1;
            }
        }
        wall += 1;
    }

    RunStats {
        wall_secs: wall,
        cpu_busy_secs: cpu_busy,
        gpu_busy_secs: gpu_busy,
        tasks_done: done,
    }
}

/// Schedule `tasks` on `cap` to MAXIMIZE compute utilization: prefetch inputs
/// ahead so a core is never idle waiting on I/O, and admit as many tasks to
/// compute as [`BoxBudget::can_admit`] allows (fill the cores / GPU lanes).
pub fn schedule_max_util(cap: &BoxCap, tasks: &[Task]) -> RunStats {
    run(cap, tasks, Policy::MaxUtil)
}

// ─────────────────────────── fleet-level scheduling ───────────────────────────

/// How a fleet of boxes divides the work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FleetPolicy {
    /// Split the task list into a fixed contiguous slice per box up front. A box
    /// that drains its slice goes idle even while others are backlogged, and a box
    /// that dies takes its slice's unfinished work to the grave. The naive baseline.
    StaticSplit,
    /// One shared queue; every box pulls (steals) whenever it has spare capacity,
    /// and a dead box's in-flight work returns to the queue to be re-run elsewhere.
    WorkSteal,
}

/// Outcome of a fleet run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FleetRun {
    pub wall_secs: u64,
    pub tasks_done: usize,
    /// Aggregate core-seconds actually spent computing across all boxes.
    pub cpu_busy_secs: u64,
    /// Aggregate core-seconds of capacity that existed (alive boxes × cores × ticks)
    /// — the denominator for fleet utilization.
    pub core_seconds_alive: u64,
    pub per_box_done: Vec<usize>,
}

impl FleetRun {
    /// Fleet CPU utilization: busy core-seconds / capacity core-seconds, in `[0, 1]`.
    pub fn cpu_util(&self) -> f64 {
        if self.core_seconds_alive == 0 {
            return 0.0;
        }
        self.cpu_busy_secs as f64 / self.core_seconds_alive as f64
    }
}

fn compute_footprint(slots: &[Slot]) -> (InFlight, u32) {
    let (mut running, mut gpu) = (InFlight::default(), 0u32);
    for s in slots {
        if s.phase == Phase::Compute {
            running.add(s.task.mem_bytes, s.task.threads);
            if s.task.gpu {
                gpu += 1;
            }
        }
    }
    (running, gpu)
}

fn admit_compute(cap: &BoxCap, slots: &mut [Slot]) {
    loop {
        let (running, gpu_used) = compute_footprint(slots);
        let pick = slots.iter().position(|s| {
            s.phase == Phase::Ready
                && cap.budget.can_admit(&running, s.task.mem_bytes, s.task.threads)
                && (!s.task.gpu || gpu_used < cap.gpu_lanes)
        });
        match pick {
            Some(i) => {
                slots[i].phase = Phase::Compute;
                slots[i].remaining = slots[i].task.compute_secs.max(1);
            }
            None => break,
        }
    }
}

fn prefetch(source: &mut VecDeque<Task>, slots: &mut Vec<Slot>, target: usize) {
    while slots
        .iter()
        .filter(|s| s.phase == Phase::Fetch || s.phase == Phase::Ready)
        .count()
        < target
    {
        match source.pop_front() {
            Some(t) => slots.push(Slot {
                task: t,
                phase: Phase::Fetch,
                remaining: t.fetch_secs,
            }),
            None => break,
        }
    }
}

/// Advance every slot one second and apply phase transitions; returns how many
/// tasks completed this tick.
fn advance_and_transition(slots: &mut Vec<Slot>) -> usize {
    for s in slots.iter_mut() {
        if matches!(s.phase, Phase::Fetch | Phase::Compute | Phase::Upload) && s.remaining > 0 {
            s.remaining -= 1;
        }
    }
    let mut done = 0;
    let mut i = 0;
    while i < slots.len() {
        if slots[i].remaining == 0 {
            match slots[i].phase {
                Phase::Fetch => {
                    slots[i].phase = Phase::Ready;
                    i += 1;
                }
                Phase::Ready => i += 1,
                Phase::Compute => {
                    slots[i].phase = Phase::Upload;
                    slots[i].remaining = slots[i].task.upload_secs;
                    i += 1;
                }
                Phase::Upload => {
                    slots.remove(i);
                    done += 1;
                }
            }
        } else {
            i += 1;
        }
    }
    done
}

/// Run `tasks` across `boxes` under `policy`, with `deaths` = `(box_index,
/// tick)` events. Deterministic tick simulation. Reports whether all tasks
/// completed and how well the fleet's cores were utilized.
pub fn run_fleet(
    boxes: &[BoxCap],
    tasks: &[Task],
    deaths: &[(usize, u64)],
    policy: FleetPolicy,
) -> FleetRun {
    let n = boxes.len();
    let mut slots: Vec<Vec<Slot>> = vec![Vec::new(); n];
    let mut alive = vec![true; n];
    let mut per_box_done = vec![0usize; n];

    // Work source(s).
    let mut shared: VecDeque<Task> = VecDeque::new();
    let mut per_box_q: Vec<VecDeque<Task>> = vec![VecDeque::new(); n];
    match policy {
        FleetPolicy::WorkSteal => {
            // Longest-processing-time-first: dispatch the heaviest tasks first so
            // the drain tail is cheap small tasks, not one grinding giant — the
            // difference between ~80% and ~97% utilization under imbalance.
            // Production has the same signal (the encode time/resource estimate).
            let mut v: Vec<Task> = tasks.to_vec();
            v.sort_by(|a, b| b.compute_secs.cmp(&a.compute_secs));
            shared.extend(v);
        }
        FleetPolicy::StaticSplit => {
            // Contiguous slice per box.
            let per = tasks.len().div_ceil(n.max(1));
            for (b, chunk) in tasks.chunks(per.max(1)).enumerate() {
                if b < n {
                    per_box_q[b].extend(chunk.iter().copied());
                }
            }
        }
    }

    let (mut wall, mut cpu_busy, mut core_secs, mut done) = (0u64, 0u64, 0u64, 0usize);
    let cap_wall = 100_000_000u64;

    loop {
        // 1. Deaths at this tick.
        for &(b, at) in deaths {
            if b < n && alive[b] && at == wall {
                alive[b] = false;
                if policy == FleetPolicy::WorkSteal {
                    // Reclaim: unfinished work goes back to the shared queue.
                    for s in slots[b].drain(..) {
                        shared.push_front(s.task);
                    }
                } else {
                    slots[b].clear(); // StaticSplit: the slice's work is lost.
                    per_box_q[b].clear();
                }
            }
        }

        // 2. Admit + prefetch on every alive box.
        for b in 0..n {
            if !alive[b] {
                continue;
            }
            admit_compute(&boxes[b], &mut slots[b]);
            let target = boxes[b].cores() as usize + 4;
            match policy {
                FleetPolicy::WorkSteal => prefetch(&mut shared, &mut slots[b], target),
                FleetPolicy::StaticSplit => prefetch(&mut per_box_q[b], &mut slots[b], target),
            }
        }

        // 3. Termination: nothing running anywhere and no source has work.
        let any_slots = slots.iter().enumerate().any(|(b, s)| alive[b] && !s.is_empty());
        let any_queued = !shared.is_empty() || per_box_q.iter().enumerate().any(|(b, q)| alive[b] && !q.is_empty());
        if !any_slots && !any_queued {
            break;
        }
        if wall >= cap_wall {
            break;
        }

        // 4. Account this tick.
        for b in 0..n {
            if !alive[b] {
                continue;
            }
            core_secs += boxes[b].cores() as u64;
            let mut used = 0u32;
            for s in &slots[b] {
                if s.phase == Phase::Compute {
                    used += s.task.threads;
                }
            }
            cpu_busy += used.min(boxes[b].cores()) as u64;
        }

        // 5. Advance.
        for b in 0..n {
            if !alive[b] {
                continue;
            }
            let d = advance_and_transition(&mut slots[b]);
            per_box_done[b] += d;
            done += d;
        }
        wall += 1;
    }

    FleetRun {
        wall_secs: wall,
        tasks_done: done,
        cpu_busy_secs: cpu_busy,
        core_seconds_alive: core_secs,
        per_box_done,
    }
}

/// Schedule `tasks` across a fleet, tolerating `deaths`, to maximize utilization
/// and complete every task.
///
/// Uses `WorkSteal`: one shared queue every box pulls from (so imbalance can't
/// idle a box while work remains), and a dead box's in-flight work is reclaimed
/// to the queue and re-run elsewhere (so no task is stranded).
pub fn schedule_fleet(boxes: &[BoxCap], tasks: &[Task], deaths: &[(usize, u64)]) -> FleetRun {
    run_fleet(boxes, tasks, deaths, FleetPolicy::WorkSteal)
}

#[cfg(test)]
mod tests {
    use super::*;
    const MB: u64 = 1 << 20;
    const GB: u64 = 1 << 30;

    #[test]
    fn serial_leaves_the_cpu_mostly_idle() {
        let cap = BoxCap::new(8, 24 * GB, 0);
        let tasks = vec![Task::light(2, 3, 1, 100 * MB); 80];
        let s = run(&cap, &tasks, Policy::Serial);
        assert_eq!(s.tasks_done, 80);
        assert!(
            s.cpu_util(8) < 0.15,
            "serial wastes the box: util {:.3}",
            s.cpu_util(8)
        );
    }
}
