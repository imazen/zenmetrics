//! Resource-aware admission control for a single box.
//!
//! Encoders expose per-encode estimates (peak memory + useful thread count;
//! `zencodec::estimate::ResourceEstimate`, surfaced uniformly via
//! `estimate_encode_resources`). Within one codec these vary by orders of
//! magnitude — an 80 MB small JPEG vs a multi-GB JXL-modular-e9 — and some
//! encoders are single-threaded while others self-thread to dozens of cores.
//! A fixed N-per-core fan-out therefore either OOMs the box (N heavy encodes ×
//! peak_mem > RAM) or starves it (a few heavy encodes when it could run
//! hundreds of light ones).
//!
//! This module packs concurrent jobs under two constraints simultaneously:
//!   Σ peak_mem ≤ ram_budget   AND   Σ threads ≤ cores.
//!
//! It is generic (no codec dependency) — the worker computes each job's
//! `(peak_mem, threads)` from the codec estimate and feeds the numbers here.

/// A box's admission budget. `ram_budget_bytes` should sit *below* physical
/// RAM — leave headroom for the OS, page cache, GPU readback buffers, and the
/// estimate's own slop (use the estimate's `peak_memory_bytes_max`). `cores`
/// is the usable CPU thread count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoxBudget {
    pub ram_budget_bytes: u64,
    pub cores: u32,
}

/// What is currently running on the box — the running sum the packer checks
/// against. Update with [`InFlight::add`] on admit, [`InFlight::remove`] on
/// completion.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InFlight {
    pub mem_bytes: u64,
    pub threads: u32,
    pub count: u32,
}

impl InFlight {
    pub fn add(&mut self, mem_bytes: u64, threads: u32) {
        self.mem_bytes = self.mem_bytes.saturating_add(mem_bytes);
        self.threads = self.threads.saturating_add(threads);
        self.count = self.count.saturating_add(1);
    }
    pub fn remove(&mut self, mem_bytes: u64, threads: u32) {
        self.mem_bytes = self.mem_bytes.saturating_sub(mem_bytes);
        self.threads = self.threads.saturating_sub(threads);
        self.count = self.count.saturating_sub(1);
    }
}

/// A single job's cost for chunk packing: its estimated *serial* wall-time plus
/// the resource footprint that determines how many such jobs a box runs at once.
/// `cost_sec` is the time one job takes run alone (encode + score); the box then
/// runs a chunk's jobs at the concurrency its mem+core envelope allows, so a
/// chunk's wall-time ≈ `Σ cost_sec / concurrency`. The worker fills these from
/// the codec estimate (`peak_mem`, `threads`) plus a per-cell time estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JobCost {
    /// Estimated wall-time of this job run alone, in seconds (encode + score).
    pub cost_sec: f64,
    /// Conservative peak memory, bytes (same source as [`InFlight`]/admission).
    pub peak_mem_bytes: u64,
    /// Useful threads at the box's core count (`1` for serial).
    pub threads: u32,
}

impl BoxBudget {
    pub fn new(ram_budget_bytes: u64, cores: u32) -> Self {
        Self {
            ram_budget_bytes,
            cores,
        }
    }

    /// Can a candidate encode — its estimated `(cand_mem, cand_threads)` — start
    /// now without pushing the box past its RAM or core budget, given what is
    /// already in flight?
    ///
    /// When nothing is running this **always admits**, so a single job whose
    /// footprint exceeds the whole budget still makes progress (it runs alone
    /// rather than deadlocking the queue). Once anything is running, a
    /// candidate that would breach either limit waits.
    pub fn can_admit(&self, running: &InFlight, cand_mem: u64, cand_threads: u32) -> bool {
        if running.count == 0 {
            return true;
        }
        let mem_ok = running.mem_bytes.saturating_add(cand_mem) <= self.ram_budget_bytes;
        let thr_ok = running.threads.saturating_add(cand_threads) <= self.cores;
        mem_ok && thr_ok
    }

    /// Greedy maximum concurrency for a homogeneous batch of `(mem, threads)`
    /// jobs: how many identical jobs fit, bounded by whichever of memory or
    /// cores binds first (≥ 1). This is the segmentation lever — size a chunk
    /// so a box can saturate it: light single-threaded encodes pack to ~cores
    /// (or hundreds if cores allow), heavy multi-threaded encodes pack to a
    /// handful.
    pub fn max_concurrent(&self, mem_each: u64, threads_each: u32) -> u32 {
        // checked_div → None on a zero divisor (a zero footprint doesn't bind
        // that axis, so it admits unboundedly): map None to u32::MAX.
        let by_mem = self
            .ram_budget_bytes
            .checked_div(mem_each)
            .map_or(u32::MAX, |v| v.min(u32::MAX as u64) as u32);
        let by_thr = self.cores.checked_div(threads_each).unwrap_or(u32::MAX);
        by_mem.min(by_thr).max(1)
    }

    /// The safe fixed concurrency for a whole (possibly heterogeneous) manifest:
    /// the most jobs that can run at once assuming the *heaviest* job's
    /// footprint, so any selection of that many is admissible. This is the
    /// onstart/launcher lever — size worker fan-out (or a chunk) to the batch's
    /// resource envelope instead of a blind N-per-core, the difference between a
    /// 64×64-JPEG batch saturating all cores and a 4K-JXL-modular batch packing
    /// to a handful without OOM. Jobs carrying no [`crate::ledger::ResourceHint`]
    /// use `fallback`. Equivalent to [`max_concurrent`](Self::max_concurrent) at
    /// the per-axis maxima; always ≥ 1.
    ///
    /// A worker that admits dynamically should prefer [`can_admit`](Self::can_admit)
    /// per candidate (it packs tighter); this is for callers that need one fixed
    /// number up front.
    pub fn recommend_concurrency(
        &self,
        hints: &[Option<crate::ledger::ResourceHint>],
        fallback: crate::ledger::ResourceHint,
    ) -> u32 {
        let (max_mem, max_threads) = hints.iter().fold((0u64, 1u32), |(m, t), h| {
            let h = h.unwrap_or(fallback);
            (m.max(h.peak_mem_bytes), t.max(h.threads))
        });
        self.max_concurrent(max_mem, max_threads)
    }

    /// Group `jobs` (in order) into chunks each estimated to take ≈
    /// `target_wall_sec` on *this* box, so a chunk is one work-stealing claim
    /// unit instead of one-claim-per-cell (the per-cell R2-lease round-trip is
    /// pure overhead for sub-second cells). The box runs a chunk's cells at the
    /// concurrency its mem+core envelope allows — the heaviest cell in the chunk
    /// binds, per [`max_concurrent`](Self::max_concurrent) — so the chunk's
    /// wall-time ≈ `Σ cost_sec / concurrency`. Packing recomputes that estimate
    /// as each cell is added and closes the chunk once it reaches the target.
    ///
    /// Properties: order-preserving, single greedy pass, every job appears in
    /// exactly one chunk. A cell whose own serial cost already ≥ target becomes
    /// its own chunk (never split a cell). `target_wall_sec` is clamped to ≥ 1.0.
    /// Memory safety is unchanged from per-cell execution: chunking only batches
    /// the *claim* — cells still execute under [`can_admit`](Self::can_admit), so
    /// concurrent peak memory stays ≤ `ram_budget_bytes` (set that to ~75% of
    /// physical RAM). Per the modes_full OOM note, cells run as fresh processes;
    /// the chunk does not accumulate their memory.
    pub fn pack_chunks(&self, jobs: &[JobCost], target_wall_sec: f64) -> Vec<Vec<usize>> {
        let target = target_wall_sec.max(1.0);
        let mut chunks: Vec<Vec<usize>> = Vec::new();
        let mut cur: Vec<usize> = Vec::new();
        let (mut sum_cost, mut max_mem, mut max_thr) = (0.0f64, 0u64, 1u32);
        for (i, j) in jobs.iter().enumerate() {
            sum_cost += j.cost_sec.max(0.0);
            max_mem = max_mem.max(j.peak_mem_bytes);
            max_thr = max_thr.max(j.threads.max(1));
            cur.push(i);
            // Effective concurrency can't exceed the cells actually in the chunk:
            // a lone heavy cell runs alone (wall = its cost), not at the box's
            // full fan-out — without this a single 400 s cell would mis-estimate
            // to cost/cores and never close its own chunk.
            let conc = self
                .max_concurrent(max_mem, max_thr)
                .min(cur.len() as u32)
                .max(1);
            let wall = sum_cost / conc as f64;
            if wall >= target {
                chunks.push(std::mem::take(&mut cur));
                sum_cost = 0.0;
                max_mem = 0;
                max_thr = 1;
            }
        }
        if !cur.is_empty() {
            chunks.push(cur);
        }
        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const GB: u64 = 1 << 30;
    const MB: u64 = 1 << 20;

    #[test]
    fn light_single_thread_is_core_bound() {
        // 80 MB single-threaded JPEG on a 24 GB / 16-core box: 24 GB/80 MB ≈ 307
        // by memory, but only 16 cores → core-bound at 16.
        let b = BoxBudget::new(24 * GB, 16);
        assert_eq!(b.max_concurrent(80 * MB, 1), 16);
    }

    #[test]
    fn heavy_multi_thread_is_mem_bound() {
        // 8 GB JXL-modular-e9 using 4 threads on a 24 GB / 16-core box:
        // 24/8 = 3 by memory, 16/4 = 4 by threads → mem-bound at 3.
        let b = BoxBudget::new(24 * GB, 16);
        assert_eq!(b.max_concurrent(8 * GB, 4), 3);
    }

    #[test]
    fn admission_respects_both_limits() {
        let b = BoxBudget::new(24 * GB, 16);
        let mut run = InFlight::default();
        run.add(8 * GB, 4);
        run.add(8 * GB, 4); // 16 GB, 8 threads in flight
        // a 3rd 8 GB job would be 24 GB ≤ 24 GB (mem ok) and 12 ≤ 16 (thr ok) → admit
        assert!(b.can_admit(&run, 8 * GB, 4));
        run.add(8 * GB, 4); // 24 GB, 12 threads
        // a 4th would be 32 GB > 24 GB → memory blocks it
        assert!(!b.can_admit(&run, 8 * GB, 4));
        // but a tiny job is still thread-blocked? 24 GB + 80 MB > 24 GB → no.
        assert!(!b.can_admit(&run, 80 * MB, 1));
    }

    #[test]
    fn over_budget_singleton_runs_alone() {
        // A 64 GB JXL on a 24 GB box: deadlock-free — admitted when idle.
        let b = BoxBudget::new(24 * GB, 16);
        let idle = InFlight::default();
        assert!(b.can_admit(&idle, 64 * GB, 16));
        // but not alongside anything.
        let mut run = InFlight::default();
        run.add(MB, 1);
        assert!(!b.can_admit(&run, 64 * GB, 16));
    }

    #[test]
    fn recommend_concurrency_is_bound_by_the_heaviest_job() {
        use crate::ledger::ResourceHint;
        let b = BoxBudget::new(24 * GB, 16);
        let light = ResourceHint {
            peak_mem_bytes: 80 * MB,
            threads: 1,
        };
        // A manifest of mostly-light jobs with one 8 GB / 4-thread JXL: the heavy
        // one binds → 24/8 = 3 (mem-bound), NOT the 16 the light jobs alone allow.
        let mixed = vec![
            Some(light),
            Some(ResourceHint {
                peak_mem_bytes: 8 * GB,
                threads: 4,
            }),
            Some(ResourceHint {
                peak_mem_bytes: 120 * MB,
                threads: 1,
            }),
            None, // no hint → fallback (light)
        ];
        assert_eq!(b.recommend_concurrency(&mixed, light), 3);
        // An all-light batch is core-bound at 16 (24 GB / 80 MB ≈ 307 by memory).
        let all_light = vec![Some(light); 100];
        assert_eq!(b.recommend_concurrency(&all_light, light), 16);
        // Empty manifest: nothing binds → default to cores.
        assert_eq!(b.recommend_concurrency(&[], light), 16);
    }

    #[test]
    fn pack_light_cells_into_five_minute_chunks() {
        // 3 s, 80 MB, single-thread cells on a 24 GB / 16-core box: core-bound at
        // 16-way, so a 300 s chunk holds 300*16/3 = 1600 cells (vs 1600 separate
        // R2-lease claims). 3200 cells → two chunks.
        let b = BoxBudget::new(24 * GB, 16);
        let jobs = vec![
            JobCost {
                cost_sec: 3.0,
                peak_mem_bytes: 80 * MB,
                threads: 1,
            };
            3200
        ];
        let chunks = b.pack_chunks(&jobs, 300.0);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].len(), 1600);
        assert_eq!(chunks[1].len(), 1600);
    }

    #[test]
    fn pack_heavy_cells_is_memory_bound() {
        // 120 s, 8 GB, 4-thread cells on a 24 GB / 16-core box: mem-bound at 3-way
        // → a 300 s chunk needs k with 120*k/3 ≥ 300, i.e. k = 8 (≈320 s). Far
        // fewer cells per chunk than the light case — the chunk auto-sizes to the
        // box's resource envelope, so a heavy chunk never OOMs.
        let b = BoxBudget::new(24 * GB, 16);
        let jobs = vec![
            JobCost {
                cost_sec: 120.0,
                peak_mem_bytes: 8 * GB,
                threads: 4,
            };
            16
        ];
        let chunks = b.pack_chunks(&jobs, 300.0);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].len(), 8);
    }

    #[test]
    fn oversized_cell_gets_its_own_chunk() {
        // A cell whose serial cost alone exceeds the target is never split: it
        // runs alone (wall = its own cost), one cell per chunk.
        let b = BoxBudget::new(24 * GB, 16);
        let jobs = vec![
            JobCost {
                cost_sec: 400.0,
                peak_mem_bytes: 80 * MB,
                threads: 1,
            };
            3
        ];
        let chunks = b.pack_chunks(&jobs, 300.0);
        assert_eq!(chunks.len(), 3);
        assert!(chunks.iter().all(|c| c.len() == 1));
    }

    #[test]
    fn pack_covers_every_job_in_order_once() {
        // Mixed heavy/light: every job lands in exactly one chunk, order preserved.
        let b = BoxBudget::new(24 * GB, 16);
        let jobs: Vec<JobCost> = (0..1000)
            .map(|i| {
                if i % 7 == 0 {
                    JobCost {
                        cost_sec: 50.0,
                        peak_mem_bytes: 2 * GB,
                        threads: 2,
                    }
                } else {
                    JobCost {
                        cost_sec: 2.0,
                        peak_mem_bytes: 100 * MB,
                        threads: 1,
                    }
                }
            })
            .collect();
        let chunks = b.pack_chunks(&jobs, 300.0);
        let flat: Vec<usize> = chunks.iter().flatten().copied().collect();
        assert_eq!(flat, (0..1000).collect::<Vec<_>>());
    }

    #[test]
    fn pack_empty_is_empty() {
        let b = BoxBudget::new(24 * GB, 16);
        assert!(b.pack_chunks(&[], 300.0).is_empty());
    }
}
