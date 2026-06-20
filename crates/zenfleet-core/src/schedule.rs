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

impl BoxBudget {
    pub fn new(ram_budget_bytes: u64, cores: u32) -> Self {
        Self { ram_budget_bytes, cores }
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
        let by_mem = if mem_each == 0 {
            u32::MAX
        } else {
            (self.ram_budget_bytes / mem_each).min(u32::MAX as u64) as u32
        };
        let by_thr = if threads_each == 0 {
            u32::MAX
        } else {
            self.cores / threads_each
        };
        by_mem.min(by_thr).max(1)
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
        run.add(1 * MB, 1);
        assert!(!b.can_admit(&run, 64 * GB, 16));
    }
}
