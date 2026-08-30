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
///
/// `vram_budget_bytes` is the **third, optional** admission axis (landed as a Nomad-migration P0
/// precondition — the ADR's "VRAM admission dimension missing in BoxBudget" defect, the avifgen
/// OOM-storm follow-up): `None` means "don't gate on VRAM" — a CPU-only box, or a GPU box whose
/// VRAM capacity couldn't be probed — and admission behaves exactly as before (RAM + cores only).
/// `Some(bytes)` adds `Σ vram ≤ vram_budget_bytes` to [`can_admit`](Self::can_admit) alongside the
/// existing two axes, so concurrent GPU-metric jobs can't silently overrun a card's memory. Set it
/// with [`Self::with_vram_budget`] — `new()` stays 2-arg and defaults to `None` so every existing
/// call site keeps compiling unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoxBudget {
    pub ram_budget_bytes: u64,
    pub cores: u32,
    pub vram_budget_bytes: Option<u64>,
}

/// What is currently running on the box — the running sum the packer checks
/// against. Update with [`InFlight::add`] on admit, [`InFlight::remove`] on
/// completion.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InFlight {
    pub mem_bytes: u64,
    pub threads: u32,
    /// Running VRAM sum, bytes. `0` for every non-GPU job (pass `0` for `vram_bytes` on
    /// [`Self::add`]/[`Self::remove`] when a job has no VRAM footprint — harmless no-op on this
    /// field either way).
    pub vram_bytes: u64,
    pub count: u32,
}

impl InFlight {
    pub fn add(&mut self, mem_bytes: u64, threads: u32, vram_bytes: u64) {
        self.mem_bytes = self.mem_bytes.saturating_add(mem_bytes);
        self.threads = self.threads.saturating_add(threads);
        self.vram_bytes = self.vram_bytes.saturating_add(vram_bytes);
        self.count = self.count.saturating_add(1);
    }
    pub fn remove(&mut self, mem_bytes: u64, threads: u32, vram_bytes: u64) {
        self.mem_bytes = self.mem_bytes.saturating_sub(mem_bytes);
        self.threads = self.threads.saturating_sub(threads);
        self.vram_bytes = self.vram_bytes.saturating_sub(vram_bytes);
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
            vram_budget_bytes: None,
        }
    }

    /// Attach a VRAM budget (builder style) — see the [`BoxBudget`] doc for what this gates.
    pub fn with_vram_budget(mut self, vram_budget_bytes: u64) -> Self {
        self.vram_budget_bytes = Some(vram_budget_bytes);
        self
    }

    /// Can a candidate encode — its estimated `(cand_mem, cand_threads, cand_vram)` — start now
    /// without pushing the box past its RAM, core, or (if set) VRAM budget, given what is already
    /// in flight? Pass `cand_vram: 0` for a job with no GPU-memory footprint.
    ///
    /// When nothing is running this **always admits**, so a single job whose
    /// footprint exceeds the whole budget still makes progress (it runs alone
    /// rather than deadlocking the queue). Once anything is running, a
    /// candidate that would breach any set limit waits. The VRAM axis only
    /// binds when [`Self::vram_budget_bytes`] is `Some` — a box with no probed
    /// GPU never gates on it, identical to the pre-VRAM behavior.
    pub fn can_admit(
        &self,
        running: &InFlight,
        cand_mem: u64,
        cand_threads: u32,
        cand_vram: u64,
    ) -> bool {
        if running.count == 0 {
            return true;
        }
        let mem_ok = running.mem_bytes.saturating_add(cand_mem) <= self.ram_budget_bytes;
        let thr_ok = running.threads.saturating_add(cand_threads) <= self.cores;
        let vram_ok = match self.vram_budget_bytes {
            Some(budget) => running.vram_bytes.saturating_add(cand_vram) <= budget,
            None => true,
        };
        mem_ok && thr_ok && vram_ok
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
        let order: Vec<usize> = (0..jobs.len()).collect();
        self.pack_in_order(jobs, &order, target_wall_sec)
    }

    /// Longest-processing-time-first packing: order the gap by descending per-cell
    /// cost before packing, so the heaviest cells land in the earliest chunks and
    /// the drain tail is cheap cells instead of one grinding giant. On an
    /// imbalanced workload this is the difference between ~80% and ~97% fleet
    /// utilization (validated by `zenfleet-sim`'s TDD scheduler, cycle 3 — a
    /// contiguous split idles boxes on the heavy tail; LPT keeps them full). The
    /// sort is stable on `(−cost, original index)`, so it stays deterministic —
    /// every worker forms identical chunks and the per-chunk claim remains
    /// exclusive. Returned indices are into the ORIGINAL `jobs` slice.
    pub fn pack_chunks_lpt(&self, jobs: &[JobCost], target_wall_sec: f64) -> Vec<Vec<usize>> {
        self.pack_in_order(jobs, &lpt_order(jobs), target_wall_sec)
    }

    /// [`pack_chunks_lpt`](Self::pack_chunks_lpt) with **fleet-uniform boundaries** — the
    /// packing every worker on a HETEROGENEOUS fleet must use for the per-chunk claim to
    /// mean anything.
    ///
    /// `pack_chunks_lpt` closes a chunk at `Σcost / self.max_concurrent(..) >= target`, and
    /// `max_concurrent` is a property of THIS BOX (its cores and RAM budget). Its doc claim
    /// that "every worker forms identical chunks and the per-chunk claim remains exclusive"
    /// therefore held only on a homogeneous fleet: the LPT *order* is deterministic, but the
    /// *boundaries* are not. MEASURED on `avifaom-enc-20260830` (2026-08-30), three boxes of
    /// different core counts on one run: chunk sizes 19 / 41 / 82 cells, so no two workers
    /// ever computed the same `chunk_id`, every claim succeeded, and **36,068 of 184,302
    /// executions (19.6%) were a second box redundantly re-encoding a cell another box was
    /// encoding at that moment** — 24,695 distinct cells hit twice or more.
    ///
    /// Two changes, both deterministic given only `jobs` (so all workers agree):
    ///
    /// 1. **`ref_concurrency`** replaces `self.max_concurrent(..)` in the close test. It is a
    ///    fleet-wide constant ([`FLEET_REF_CONCURRENCY`]), not a box property. Execution
    ///    concurrency is untouched — cells still run under `can_admit`, so each box still
    ///    saturates its own cores within its own RAM envelope. Only the CLAIM granularity
    ///    becomes uniform.
    /// 2. **A gap-relative cell cap**: no chunk holds more than `gap_len / spread` cells
    ///    (`spread` = [`TAIL_SPREAD`], floored at [`MIN_CHUNK_CELLS`]). On a big gap the cost target closes chunks
    ///    long before the cap binds, so nothing changes; as the gap DRAINS the cap shrinks
    ///    the chunks, which is the straggler-tail lever — the same wave ended with one box
    ///    grinding while two sat idle at `rows=0` for 1.07 h because every remaining cell was
    ///    inside one already-claimed chunk. A cap of `gap/64` puts ≥ 64 claimable chunks in
    ///    front of the fleet no matter how small the remainder gets.
    ///
    /// Exactly-once is unaffected: this changes only how the gap is grouped, never the
    /// claim/ledger reduce. Every job still appears in exactly one chunk.
    pub fn pack_chunks_lpt_uniform(
        &self,
        jobs: &[JobCost],
        target_wall_sec: f64,
        ref_concurrency: u32,
        spread: u32,
    ) -> Vec<Vec<usize>> {
        let order = lpt_order(jobs);
        let target = target_wall_sec.max(1.0);
        let conc = ref_concurrency.max(1) as f64;
        // Floor the cap at MIN_CHUNK_CELLS: below it the "chunk" stops batching claims at
        // all and every cheap cell pays a full claim round-trip, which is the overhead
        // chunking exists to remove. Genuinely expensive tail cells are already isolated by
        // the cost target (a cell whose own cost >= target becomes its own chunk), so the
        // floor only ever groups CHEAP cells.
        let cap = (jobs.len() / spread.max(1) as usize).max(MIN_CHUNK_CELLS);
        let mut chunks: Vec<Vec<usize>> = Vec::new();
        let mut cur: Vec<usize> = Vec::new();
        let mut sum_cost = 0.0f64;
        for &i in &order {
            sum_cost += jobs[i].cost_sec.max(0.0);
            cur.push(i);
            // Effective concurrency can't exceed the cells actually in the chunk (same
            // guard as `pack_in_order`: a lone heavy cell runs alone, not at full fan-out).
            let eff = conc.min(cur.len() as f64).max(1.0);
            if sum_cost / eff >= target || cur.len() >= cap {
                chunks.push(std::mem::take(&mut cur));
                sum_cost = 0.0;
            }
        }
        if !cur.is_empty() {
            chunks.push(cur);
        }
        chunks
    }

    /// Greedy packer over a given visitation `order` of `jobs` (indices into
    /// `jobs`). Shared by [`pack_chunks`](Self::pack_chunks) (manifest order) and
    /// [`pack_chunks_lpt`](Self::pack_chunks_lpt) (cost-descending). Closes a
    /// chunk once its estimated wall-time (`Σcost / in-chunk concurrency`) reaches
    /// `target`; a lone over-target cell becomes its own chunk. Emits the original
    /// `jobs` indices in visitation order.
    fn pack_in_order(
        &self,
        jobs: &[JobCost],
        order: &[usize],
        target_wall_sec: f64,
    ) -> Vec<Vec<usize>> {
        let target = target_wall_sec.max(1.0);
        let mut chunks: Vec<Vec<usize>> = Vec::new();
        let mut cur: Vec<usize> = Vec::new();
        let (mut sum_cost, mut max_mem, mut max_thr) = (0.0f64, 0u64, 1u32);
        for &i in order {
            let j = &jobs[i];
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

/// The fleet-wide reference concurrency used to close chunks in
/// [`BoxBudget::pack_chunks_lpt_uniform`]. It is deliberately a CONSTANT and not any box's
/// real fan-out: its only job is to make every worker draw the same chunk boundaries so the
/// per-chunk claim is exclusive. Execution concurrency is still per-box (`can_admit`).
///
/// 8 is the middle of the observed LAN fleet's admitted fan-out (the 2026-08-30 aom wave
/// packed at 19/41/82 cells per chunk on three boxes); a value in that range keeps chunk
/// wall-times near `chunk_wall_sec` on every box rather than optimal on one and wrong on
/// the rest. Override fleet-WIDE (never per box) if a fleet's shape changes.
pub const FLEET_REF_CONCURRENCY: u32 = 8;

/// Minimum number of claimable chunks the packer keeps in front of the fleet, via the
/// gap-relative cell cap in [`BoxBudget::pack_chunks_lpt_uniform`]. Sized so a fleet several
/// times larger than today's three boxes still has spare chunks to claim as the gap drains.
pub const TAIL_SPREAD: u32 = 64;

/// Floor for the gap-relative cell cap in [`BoxBudget::pack_chunks_lpt_uniform`]. Keeps a
/// small gap from degenerating into one-claim-per-cell — the per-claim store round-trip is
/// exactly the overhead chunk claiming was introduced to amortize. Expensive cells are
/// isolated by the cost target regardless, so this only groups cheap ones.
pub const MIN_CHUNK_CELLS: usize = 4;

/// Longest-processing-time-first visitation order: descending `cost_sec`, ties broken by
/// original index so it is deterministic (identical on every worker).
fn lpt_order(jobs: &[JobCost]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..jobs.len()).collect();
    order.sort_by(|&a, &b| {
        jobs[b]
            .cost_sec
            .partial_cmp(&jobs[a].cost_sec)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order
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
        run.add(8 * GB, 4, 0);
        run.add(8 * GB, 4, 0); // 16 GB, 8 threads in flight
        // a 3rd 8 GB job would be 24 GB ≤ 24 GB (mem ok) and 12 ≤ 16 (thr ok) → admit
        assert!(b.can_admit(&run, 8 * GB, 4, 0));
        run.add(8 * GB, 4, 0); // 24 GB, 12 threads
        // a 4th would be 32 GB > 24 GB → memory blocks it
        assert!(!b.can_admit(&run, 8 * GB, 4, 0));
        // but a tiny job is still thread-blocked? 24 GB + 80 MB > 24 GB → no.
        assert!(!b.can_admit(&run, 80 * MB, 1, 0));
    }

    #[test]
    fn over_budget_singleton_runs_alone() {
        // A 64 GB JXL on a 24 GB box: deadlock-free — admitted when idle.
        let b = BoxBudget::new(24 * GB, 16);
        let idle = InFlight::default();
        assert!(b.can_admit(&idle, 64 * GB, 16, 0));
        // but not alongside anything.
        let mut run = InFlight::default();
        run.add(MB, 1, 0);
        assert!(!b.can_admit(&run, 64 * GB, 16, 0));
    }

    #[test]
    fn vram_budget_none_never_gates() {
        // A box with no probed GPU (vram_budget_bytes: None, the BoxBudget::new default) admits
        // purely on RAM/cores regardless of how much VRAM a candidate claims — identical to the
        // pre-VRAM-axis behavior. This is the compatibility case: every existing caller that never
        // learned about VRAM keeps working unchanged.
        let b = BoxBudget::new(24 * GB, 16);
        assert_eq!(b.vram_budget_bytes, None);
        let mut run = InFlight::default();
        run.add(MB, 1, 100 * GB); // absurd VRAM claim
        assert!(b.can_admit(&run, MB, 1, 100 * GB));
    }

    #[test]
    fn vram_admission_gates_gpu_jobs() {
        // A 12 GB-VRAM box (e.g. an RTX 2080/3070-class card in this fleet): two 8 GB-VRAM GPU
        // metric jobs can't run concurrently even though RAM/cores have plenty of headroom.
        let b = BoxBudget::new(64 * GB, 32).with_vram_budget(12 * GB);
        let mut run = InFlight::default();
        run.add(GB, 2, 8 * GB); // one job in flight: 1 GB RAM, 2 threads, 8 GB VRAM
        // A 2nd 8 GB-VRAM job: RAM (2 GB ≤ 64) and cores (4 ≤ 32) are fine, but VRAM
        // (16 GB > 12 GB) blocks it — this is the axis that didn't exist before.
        assert!(!b.can_admit(&run, GB, 2, 8 * GB));
        // A small 2 GB-VRAM job fits (1+2=3 ≤ 12).
        assert!(b.can_admit(&run, MB, 1, 2 * GB));
    }

    #[test]
    fn vram_over_budget_singleton_runs_alone() {
        // Same deadlock-freedom guarantee as RAM/cores: a single job whose VRAM need exceeds the
        // whole card still runs when the box is idle (CappedPyramid/Strip modes exist precisely so
        // a metric can shrink its footprint, but admission must never wedge the queue either way).
        let b = BoxBudget::new(64 * GB, 32).with_vram_budget(12 * GB);
        let idle = InFlight::default();
        assert!(b.can_admit(&idle, GB, 2, 20 * GB));
        let mut run = InFlight::default();
        run.add(MB, 1, GB);
        assert!(!b.can_admit(&run, GB, 2, 20 * GB));
    }

    #[test]
    fn recommend_concurrency_is_bound_by_the_heaviest_job() {
        use crate::ledger::ResourceHint;
        let b = BoxBudget::new(24 * GB, 16);
        let light = ResourceHint {
            peak_mem_bytes: 80 * MB,
            threads: 1,
            vram_bytes: None,
        };
        // A manifest of mostly-light jobs with one 8 GB / 4-thread JXL: the heavy
        // one binds → 24/8 = 3 (mem-bound), NOT the 16 the light jobs alone allow.
        let mixed = vec![
            Some(light),
            Some(ResourceHint {
                peak_mem_bytes: 8 * GB,
                threads: 4,
                vram_bytes: None,
            }),
            Some(ResourceHint {
                peak_mem_bytes: 120 * MB,
                threads: 1,
                vram_bytes: None,
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
        assert!(b.pack_chunks_lpt(&[], 300.0).is_empty());
    }

    #[test]
    fn lpt_dispatches_the_heaviest_cells_first() {
        // 60 light (3s) then 4 heavy (100s) on a 2-core box, 30 s target: the
        // light cells fill several chunks before the heavy ones appear, so
        // manifest order buries the heavy cells in the tail. LPT hoists them to
        // the front so a fleet's boxes don't finish the light work and idle on a
        // heavy tail.
        let b = BoxBudget::new(24 * GB, 2);
        let mut jobs = vec![
            JobCost {
                cost_sec: 3.0,
                peak_mem_bytes: 100 * MB,
                threads: 1,
            };
            60
        ];
        jobs.extend(std::iter::repeat_n(
            JobCost {
                cost_sec: 100.0,
                peak_mem_bytes: 100 * MB,
                threads: 1,
            },
            4,
        ));
        let heavy: std::collections::HashSet<usize> = (60..64).collect();

        let manifest = b.pack_chunks(&jobs, 30.0);
        let lpt = b.pack_chunks_lpt(&jobs, 30.0);

        assert!(
            lpt[0].iter().all(|i| heavy.contains(i)),
            "LPT's first chunk is heavy work: {:?}",
            lpt[0]
        );
        assert!(
            !manifest[0].iter().any(|i| heavy.contains(i)),
            "manifest order leaves the heavy cells for the tail: {:?}",
            manifest[0]
        );
    }

    #[test]
    fn lpt_is_deterministic_and_covers_every_cell_once() {
        let b = BoxBudget::new(24 * GB, 8);
        let jobs: Vec<JobCost> = (0..500)
            .map(|i| JobCost {
                cost_sec: ((i * 7) % 53) as f64 + 1.0, // varied, deterministic
                peak_mem_bytes: 100 * MB,
                threads: 1,
            })
            .collect();

        let a = b.pack_chunks_lpt(&jobs, 120.0);
        let c = b.pack_chunks_lpt(&jobs, 120.0);
        assert_eq!(a, c, "stable sort → identical chunks on every worker");

        let mut all: Vec<usize> = a.iter().flatten().copied().collect();
        all.sort_unstable();
        assert_eq!(
            all,
            (0..500).collect::<Vec<_>>(),
            "every original cell appears exactly once"
        );
    }

    /// THE 2026-08-30 heterogeneous-fleet claim collapse, as a gate.
    ///
    /// `pack_chunks_lpt` closes chunks on `self.max_concurrent(..)`, a per-BOX quantity, so
    /// three boxes of different core counts drew three different sets of chunk boundaries,
    /// produced three disjoint sets of `chunk_id`s, and every claim succeeded — 19.6% of the
    /// wave's executions were redundant. `pack_chunks_lpt_uniform` must be byte-identical
    /// across boxes.
    #[test]
    fn uniform_packing_is_identical_across_heterogeneous_boxes() {
        let jobs: Vec<JobCost> = (0..500)
            .map(|i| JobCost {
                cost_sec: 1.0 + (i % 37) as f64 * 3.0,
                peak_mem_bytes: (64 << 20) * (1 + (i % 5) as u64),
                threads: 1 + (i % 4) as u32,
            })
            .collect();
        let small = BoxBudget {
            cores: 12,
            ram_budget_bytes: 8 * GB,
            vram_budget_bytes: None,
        };
        let big = BoxBudget {
            cores: 48,
            ram_budget_bytes: 45 * GB,
            vram_budget_bytes: None,
        };
        // The OLD packer disagrees — this is the defect, pinned so a regression is loud.
        assert_ne!(
            small.pack_chunks_lpt(&jobs, 300.0),
            big.pack_chunks_lpt(&jobs, 300.0),
            "per-box packing is expected to differ; that is the bug uniform packing fixes"
        );
        // The NEW packer agrees, exactly.
        let a = small.pack_chunks_lpt_uniform(&jobs, 300.0, FLEET_REF_CONCURRENCY, TAIL_SPREAD);
        let b = big.pack_chunks_lpt_uniform(&jobs, 300.0, FLEET_REF_CONCURRENCY, TAIL_SPREAD);
        assert_eq!(a, b, "chunk boundaries must not depend on the box");
        // Coverage: every job in exactly one chunk (exactly-once is downstream of this).
        let mut seen: Vec<usize> = a.iter().flatten().copied().collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..jobs.len()).collect::<Vec<_>>());
    }

    /// The straggler-tail lever: as the gap drains, chunks must shrink so the remainder is
    /// claimable by more than one box. The same wave ended with one box grinding while two
    /// sat idle for 1.07 h because every remaining cell was inside one claimed chunk.
    #[test]
    fn gap_relative_cap_spreads_the_tail_and_is_inert_on_a_big_gap() {
        let bb = BoxBudget {
            cores: 32,
            ram_budget_bytes: 32 * GB,
            vram_budget_bytes: None,
        };
        // A small remainder of cheap cells: cost alone would put them all in ONE chunk.
        let tail: Vec<JobCost> = (0..312)
            .map(|_| JobCost {
                cost_sec: 0.5,
                peak_mem_bytes: 64 << 20,
                threads: 1,
            })
            .collect();
        // spread=1 -> cap = max(312/1, MIN) = 312, i.e. the cost target alone applies.
        let one = bb.pack_chunks_lpt_uniform(&tail, 300.0, FLEET_REF_CONCURRENCY, 1);
        assert_eq!(one.len(), 1, "with spread=1 the cost target alone applies");
        let spread = bb.pack_chunks_lpt_uniform(&tail, 300.0, FLEET_REF_CONCURRENCY, TAIL_SPREAD);
        assert!(
            spread.len() >= TAIL_SPREAD as usize,
            "a {}-cell tail must offer >= {} claimable chunks, got {}",
            tail.len(),
            TAIL_SPREAD,
            spread.len()
        );
        assert!(
            spread
                .iter()
                .all(|c| c.len() <= (312 / TAIL_SPREAD as usize).max(MIN_CHUNK_CELLS))
        );
        let mut seen: Vec<usize> = spread.iter().flatten().copied().collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..tail.len()).collect::<Vec<_>>());

        // Inert on a big gap: 100k cells at 30 s each close on cost long before gap/64.
        let big: Vec<JobCost> = (0..100_000)
            .map(|_| JobCost {
                cost_sec: 30.0,
                peak_mem_bytes: 64 << 20,
                threads: 1,
            })
            .collect();
        let c = bb.pack_chunks_lpt_uniform(&big, 300.0, FLEET_REF_CONCURRENCY, TAIL_SPREAD);
        let cap = 100_000 / TAIL_SPREAD as usize;
        assert!(
            c.iter().all(|k| k.len() < cap),
            "on a big gap the cost target must bind, not the cap"
        );
    }

    /// LPT ordering is preserved by the uniform packer: the heaviest cells land in the
    /// earliest chunks, so the drain tail is cheap cells rather than one grinding giant.
    #[test]
    fn uniform_packing_keeps_lpt_heaviest_first() {
        let bb = BoxBudget {
            cores: 16,
            ram_budget_bytes: 16 * GB,
            vram_budget_bytes: None,
        };
        let jobs: Vec<JobCost> = (0..200)
            .map(|i| JobCost {
                cost_sec: i as f64,
                peak_mem_bytes: 64 << 20,
                threads: 1,
            })
            .collect();
        let c = bb.pack_chunks_lpt_uniform(&jobs, 100.0, FLEET_REF_CONCURRENCY, TAIL_SPREAD);
        let first: f64 = c[0].iter().map(|&i| jobs[i].cost_sec).sum::<f64>() / c[0].len() as f64;
        let last: f64 = c[c.len() - 1]
            .iter()
            .map(|&i| jobs[i].cost_sec)
            .sum::<f64>()
            / c[c.len() - 1].len() as f64;
        assert!(
            first > last,
            "heaviest cells must land first ({first} vs {last})"
        );
    }
}
