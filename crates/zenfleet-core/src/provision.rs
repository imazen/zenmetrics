//! Manifest → instance recommendation (the inverse of [`crate::schedule`]).
//!
//! [`crate::schedule::BoxBudget`] answers "given a box, how many of these
//! jobs fit?" — admission control on a **fixed** machine. This module answers
//! the provisioning question: "given a manifest of work and a target
//! wall-clock, what box should I rent and how many?" It sizes the host RAM,
//! GPU VRAM, and box count for a sweep, then hands back an
//! [`InstanceRecommendation`].
//!
//! It adds the **GPU/VRAM dimension** that `schedule` (host RAM + cores only)
//! does not model. The execution model it sizes for is the one the unified
//! worker (`scripts/sweep/onstart_unified.sh`) actually runs: **N CPU encoders
//! sharing one GPU** — many encodes run concurrently across the box's cores, while metric
//! scoring is **serialized on the single GPU device**. That asymmetry drives
//! every rule below.
//!
//! Like [`crate::schedule`] this crate is **generic**: it has no codec or
//! metric dependency. The caller computes each cell's [`CellCost`] from the
//! codec's `estimate_encode_resources` and the metrics' VRAM + time
//! estimators (which live in the codec / `*-gpu` crates), then passes the
//! plain numbers here — exactly the seam `schedule` uses with
//! [`crate::ledger::ResourceHint`].

use crate::ledger::ResourceHint;
use crate::schedule::BoxBudget;

/// One unit of sweep work: encode a source variant, then score it with one or
/// more metrics. All figures are **per cell** (one image × one codec-config ×
/// the cell's metric set).
///
/// Build these from:
/// - encode side: `zencodec::estimate::ResourceEstimate` — `peak_memory_bytes_max`
///   → [`encode_peak_ram_bytes`](Self::encode_peak_ram_bytes),
///   `threading().effective_threads(cores)` →
///   [`encode_threads`](Self::encode_threads), `at_cores(n).wall_ms()` →
///   [`encode_ms`](Self::encode_ms).
/// - score side: per metric, `<metric>_gpu::estimate_gpu_memory_bytes` (take
///   the **max** over the cell's metrics) →
///   [`score_vram_bytes`](Self::score_vram_bytes), and
///   `<metric>_gpu::estimate_score_time_ms` (take the **sum**) →
///   [`score_ms`](Self::score_ms).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellCost {
    /// Worst-case host RAM the encode needs, bytes
    /// (`ResourceEstimate::peak_memory_bytes_max`). Drives the per-box RAM
    /// sizing.
    pub encode_peak_ram_bytes: u64,
    /// CPU threads the encode actually uses on this box
    /// (`ResourceEstimate::threading().effective_threads(cores)`). Drives the
    /// per-box concurrency packing.
    pub encode_threads: u32,
    /// Encode wall time in milliseconds. Single-thread, or already re-scaled
    /// via `ResourceEstimate::at_cores(n)` — be consistent across cells.
    pub encode_ms: f32,
    /// Peak GPU working-set this cell's scoring needs, bytes — the **max**
    /// over the cell's metrics (they run one at a time on the device, so the
    /// card must hold the single heaviest, not the sum).
    pub score_vram_bytes: u64,
    /// Total score wall time in milliseconds — the **sum** over the cell's
    /// metrics (GPU scoring is serialized on the one device).
    pub score_ms: f32,
}

impl CellCost {
    /// A cell with the given encode + score costs.
    #[must_use]
    pub fn new(
        encode_peak_ram_bytes: u64,
        encode_threads: u32,
        encode_ms: f32,
        score_vram_bytes: u64,
        score_ms: f32,
    ) -> Self {
        Self {
            encode_peak_ram_bytes,
            encode_threads,
            encode_ms,
            score_vram_bytes,
            score_ms,
        }
    }

    /// The encode side as a [`ResourceHint`] for the `schedule` packer.
    #[must_use]
    pub fn encode_hint(&self) -> ResourceHint {
        ResourceHint {
            peak_mem_bytes: self.encode_peak_ram_bytes,
            threads: self.encode_threads.max(1),
        }
    }
}

/// A recommended instance spec + fleet size for a sweep manifest.
///
/// Produced by [`recommend_instance`]. The fields split cleanly into "what
/// one box must be" (RAM / cores / VRAM / `needs_gpu` /
/// `recommended_concurrency`) and "how many of them + how long"
/// (`box_count` / `est_wall_clock_s` and the total-work breakdown).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InstanceRecommendation {
    /// Host RAM the box should have, bytes:
    /// `recommended_concurrency × max(encode_peak_ram)` plus a headroom
    /// factor (OS, page cache, GPU readback, estimate slop). Size the rented
    /// box at or above this.
    pub host_ram_bytes: u64,
    /// CPU cores the recommendation assumes per box (echoes the
    /// `cores_per_box` argument).
    pub cores: u32,
    /// GPU VRAM the card must hold, bytes: `max(score_vram_bytes)` over all
    /// cells — GPU scoring is serialized, so the card holds the single
    /// heaviest metric working set, not the sum.
    pub gpu_vram_bytes: u64,
    /// Whether any cell needs the GPU at all (any `score_vram_bytes > 0` or
    /// `score_ms > 0`). A pure-encode sweep needs no GPU box.
    pub needs_gpu: bool,
    /// Concurrent encodes to run per box — from
    /// [`BoxBudget::recommend_concurrency`] over the cells' encode hints,
    /// bound by whichever of RAM or cores binds first. The "N" in "N CPU
    /// encoders share one GPU".
    pub recommended_concurrency: u32,
    /// Boxes needed to finish within `target_wall_clock_s`:
    /// `ceil(total_wall_ms / (target × 1000))`. At least 1.
    pub box_count: u32,
    /// Estimated wall-clock of the whole sweep on `box_count` boxes, seconds.
    pub est_wall_clock_s: f64,
    /// Sum of all cells' encode milliseconds (single-box, pre-parallelism).
    pub total_encode_ms: f64,
    /// Sum of all cells' score milliseconds (single-box, serialized GPU).
    pub total_score_ms: f64,
    /// `true` when even `max(score_vram_bytes)` exceeds [`SANE_CARD_VRAM_BYTES`]
    /// — Full-mode scoring won't fit a common card and the metric's
    /// `resolve_auto` would fall back to Strip mode. Surfaced so the caller
    /// can rent a bigger card or accept Strip-mode scoring.
    pub vram_exceeds_sane_card: bool,
}

/// Headroom multiplier on the packed encode RAM when sizing the host. The box
/// must hold `recommended_concurrency` concurrent encodes at their worst-case
/// peak; this margin covers the OS, page cache, GPU readback staging, and the
/// estimate's own slop. 1.25 = +25%.
pub const RAM_HEADROOM_FACTOR: f64 = 1.25;

/// VRAM above which Full-mode scoring is assumed not to fit a "common" rented
/// GPU (24 GiB — an RTX 4090 / A10G / L4-class card). When `max(score_vram)`
/// exceeds this, [`recommend_instance`] sets
/// [`vram_exceeds_sane_card`](InstanceRecommendation::vram_exceeds_sane_card)
/// so the caller knows the metric's `resolve_auto` would pick Strip mode (or
/// a bigger card is needed).
pub const SANE_CARD_VRAM_BYTES: u64 = 24 * 1024 * 1024 * 1024;

/// Recommend an instance spec + box count for a sweep manifest.
///
/// `cells` is the per-cell cost list; `target_wall_clock_s` is the wall-clock
/// budget the fleet should finish within; `cores_per_box` is the CPU core
/// count of the box class being sized.
///
/// # Aggregation rules
///
/// - **`recommended_concurrency`** = [`BoxBudget::recommend_concurrency`] over
///   the cells' encode hints, on a budget of `(host_ram, cores_per_box)`. To
///   break the RAM↔concurrency circularity (RAM depends on concurrency, which
///   the packer derives from RAM), the budget uses a generous RAM ceiling =
///   `cores_per_box × max(encode_peak_ram)` (enough for one heavy encode per
///   core), so concurrency is **core-bound for light work and RAM-bound for
///   heavy work**, exactly like a real box.
/// - **`host_ram`** = `recommended_concurrency × max(encode_peak_ram) ×`
///   [`RAM_HEADROOM_FACTOR`]. The box must hold that many concurrent encodes
///   plus headroom.
/// - **`gpu_vram`** = `max(score_vram_bytes)` over all cells. GPU scoring is
///   serialized on the device (the onstart model), so the card holds the
///   single heaviest metric working set, **not** the sum.
/// - **`box_count`** = `ceil(total_wall_ms / (target × 1000))`, where
///   `total_wall_ms = total_encode_ms / recommended_concurrency +
///   total_score_ms`. Encodes parallelize across the box's cores
///   (`/concurrency`); GPU scoring serializes (added whole). This is the
///   "N CPU encoders share one GPU" wall model — work splits evenly across
///   `box_count` boxes, each running the same shape.
///
/// An empty manifest yields a minimal 1-box, no-GPU recommendation.
#[must_use]
pub fn recommend_instance(
    cells: &[CellCost],
    target_wall_clock_s: f64,
    cores_per_box: u32,
) -> InstanceRecommendation {
    let cores = cores_per_box.max(1);

    // --- per-cell extrema + totals -------------------------------------
    let mut max_encode_ram: u64 = 0;
    let mut max_score_vram: u64 = 0;
    let mut total_encode_ms: f64 = 0.0;
    let mut total_score_ms: f64 = 0.0;
    let mut needs_gpu = false;
    let mut hints: Vec<Option<ResourceHint>> = Vec::with_capacity(cells.len());

    for c in cells {
        max_encode_ram = max_encode_ram.max(c.encode_peak_ram_bytes);
        max_score_vram = max_score_vram.max(c.score_vram_bytes);
        total_encode_ms += c.encode_ms.max(0.0) as f64;
        total_score_ms += c.score_ms.max(0.0) as f64;
        if c.score_vram_bytes > 0 || c.score_ms > 0.0 {
            needs_gpu = true;
        }
        hints.push(Some(c.encode_hint()));
    }

    // --- per-box concurrency (reuse schedule's packer) -----------------
    // Break the RAM↔concurrency circularity: pack against a RAM ceiling
    // that allows one heaviest encode per core. Then recommend_concurrency
    // returns the core-bound count for light work and the RAM-bound count
    // for heavy work — the same number a real box of this size would admit.
    let pack_ram_ceiling = (max_encode_ram.max(1)).saturating_mul(cores as u64);
    let budget = BoxBudget::new(pack_ram_ceiling, cores);
    // Fallback hint for cells with no encode cost (shouldn't occur, but the
    // signature requires one): a 1-thread, ~0-mem job.
    let fallback = ResourceHint {
        peak_mem_bytes: 1,
        threads: 1,
    };
    let recommended_concurrency = if cells.is_empty() {
        1
    } else {
        budget.recommend_concurrency(&hints, fallback).max(1)
    };

    // --- host RAM sizing ------------------------------------------------
    let packed_ram = max_encode_ram.saturating_mul(recommended_concurrency as u64);
    let host_ram_bytes = ((packed_ram as f64) * RAM_HEADROOM_FACTOR).ceil() as u64;

    // --- wall-clock + box count ----------------------------------------
    // Encodes overlap across cores (÷ concurrency); GPU scoring serializes
    // (whole). This is the single-box wall for the entire manifest.
    let single_box_wall_ms =
        total_encode_ms / (recommended_concurrency as f64).max(1.0) + total_score_ms;
    let target_ms = (target_wall_clock_s.max(0.0)) * 1000.0;
    let box_count = if single_box_wall_ms <= 0.0 || target_ms <= 0.0 {
        1
    } else {
        (single_box_wall_ms / target_ms).ceil().max(1.0) as u32
    };
    // Work splits evenly across the fleet: each box runs ~1/box_count of it.
    let est_wall_clock_s = if box_count == 0 {
        0.0
    } else {
        (single_box_wall_ms / box_count as f64) / 1000.0
    };

    InstanceRecommendation {
        host_ram_bytes,
        cores,
        gpu_vram_bytes: max_score_vram,
        needs_gpu,
        recommended_concurrency,
        box_count,
        est_wall_clock_s,
        total_encode_ms,
        total_score_ms,
        vram_exceeds_sane_card: max_score_vram > SANE_CARD_VRAM_BYTES,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const GB: u64 = 1 << 30;
    const MB: u64 = 1 << 20;

    /// Mirror schedule.rs's style: an all-light manifest is core-bound, so
    /// the box is small and concurrency saturates the cores.
    #[test]
    fn all_light_cells_are_core_bound_with_a_small_box() {
        // 80 MB / 1-thread JPEG encodes, ~6 ms encode, tiny score.
        let cells = vec![CellCost::new(80 * MB, 1, 6.0, 256 * MB, 4.0); 100];
        let r = recommend_instance(&cells, 60.0, 16);
        // 100 light jobs, 16 cores → concurrency 16 (core-bound, like
        // schedule::recommend_concurrency_is_bound_by_the_heaviest_job).
        assert_eq!(r.recommended_concurrency, 16);
        // host RAM = 16 × 80 MB × 1.25 = 1600 MB.
        assert_eq!(r.host_ram_bytes, (16 * 80 * MB) as f64 as u64 * 5 / 4);
        assert!(r.needs_gpu);
        // gpu_vram = max score vram = 256 MB; well under a sane card.
        assert_eq!(r.gpu_vram_bytes, 256 * MB);
        assert!(!r.vram_exceeds_sane_card);
    }

    /// One heavy-RAM encode in the mix binds concurrency by memory, not cores
    /// — the same heaviest-job logic schedule.rs packs by.
    #[test]
    fn heavy_encode_binds_concurrency_by_memory() {
        // Mostly light, one 8 GB / 4-thread JXL-modular. On a 24-core box
        // sized to 24 × 8 GB ceiling, the heavy job's 8 GB binds:
        // recommend_concurrency packs to ceiling/8GB but the ceiling is
        // 24×8GB so by-mem = 24; by-thread = 24/4 = 6 → 6 (thread-bound on
        // the heavy job's threads). Either way it's < 24.
        let mut cells = vec![CellCost::new(80 * MB, 1, 5.0, 512 * MB, 3.0); 10];
        cells.push(CellCost::new(8 * GB, 4, 200.0, GB, 5.0));
        let r = recommend_instance(&cells, 60.0, 24);
        assert!(
            r.recommended_concurrency < 24,
            "heavy job should pull concurrency below cores, got {}",
            r.recommended_concurrency
        );
        // host RAM sized to that concurrency × the 8 GB heaviest encode.
        let expect_ram =
            ((8 * GB * r.recommended_concurrency as u64) as f64 * RAM_HEADROOM_FACTOR) as u64;
        assert_eq!(r.host_ram_bytes, expect_ram);
    }

    /// gpu_vram is the MAX over metrics (serialized GPU), never the sum.
    #[test]
    fn gpu_vram_is_max_not_sum() {
        // Two cells, different score VRAM. The card must hold the larger.
        let cells = vec![
            CellCost::new(100 * MB, 1, 5.0, 2 * GB, 10.0),
            CellCost::new(100 * MB, 1, 5.0, 6 * GB, 20.0),
        ];
        let r = recommend_instance(&cells, 60.0, 8);
        assert_eq!(r.gpu_vram_bytes, 6 * GB); // max, not 8 GB sum
    }

    /// A metric working set bigger than a sane card flags Strip-mode fallback.
    #[test]
    fn oversize_vram_flags_strip_fallback() {
        let cells = vec![CellCost::new(100 * MB, 1, 5.0, 32 * GB, 50.0)];
        let r = recommend_instance(&cells, 60.0, 8);
        assert_eq!(r.gpu_vram_bytes, 32 * GB);
        assert!(r.vram_exceeds_sane_card);
    }

    /// box_count scales with total work / target wall-clock.
    #[test]
    fn box_count_scales_with_total_time_over_target() {
        // 1000 cells: encode 100 ms each (parallel ÷ concurrency), score
        // 50 ms each (serialized). On 10 cores, light RAM → concurrency 10.
        // total_encode = 100 s, total_score = 50 s.
        // single-box wall = 100/10 + 50 = 60 s.
        let cells = vec![CellCost::new(50 * MB, 1, 100.0, 512 * MB, 50.0); 1000];
        let r = recommend_instance(&cells, 60.0, 10);
        assert_eq!(r.recommended_concurrency, 10);
        assert!((r.total_encode_ms - 100_000.0).abs() < 1.0);
        assert!((r.total_score_ms - 50_000.0).abs() < 1.0);
        // single-box wall ≈ 60 s; target 60 s → 1 box.
        assert_eq!(r.box_count, 1);
        // Halve the target → need 2 boxes.
        let r2 = recommend_instance(&cells, 30.0, 10);
        assert_eq!(r2.box_count, 2);
        // est wall on 2 boxes ≈ 30 s.
        assert!((r2.est_wall_clock_s - 30.0).abs() < 1.0);
    }

    /// A pure-encode manifest (no scoring) needs no GPU.
    #[test]
    fn pure_encode_manifest_needs_no_gpu() {
        let cells = vec![CellCost::new(200 * MB, 2, 20.0, 0, 0.0); 50];
        let r = recommend_instance(&cells, 60.0, 16);
        assert!(!r.needs_gpu);
        assert_eq!(r.gpu_vram_bytes, 0);
        assert!(!r.vram_exceeds_sane_card);
    }

    /// Empty manifest → a minimal, valid recommendation (no panics, ≥ 1 box).
    #[test]
    fn empty_manifest_is_minimal() {
        let r = recommend_instance(&[], 60.0, 16);
        assert_eq!(r.recommended_concurrency, 1);
        assert_eq!(r.box_count, 1);
        assert_eq!(r.host_ram_bytes, 0);
        assert!(!r.needs_gpu);
        assert_eq!(r.total_encode_ms, 0.0);
        assert_eq!(r.est_wall_clock_s, 0.0);
    }

    /// The score side dominating wall does not get divided by concurrency —
    /// the serialized GPU is the bottleneck and box_count reflects it.
    #[test]
    fn serialized_score_dominates_box_count() {
        // Tiny encodes (1 ms), heavy scoring (500 ms each), 100 cells.
        // total_encode = 100 ms, total_score = 50_000 ms.
        // On 16 cores concurrency 16: wall = 100/16 + 50000 ≈ 50006 ms.
        // ceil(50006 / 25000) = 3 boxes (the tiny encode term tips it past 2).
        let cells = vec![CellCost::new(40 * MB, 1, 1.0, GB, 500.0); 100];
        let r = recommend_instance(&cells, 25.0, 16);
        assert_eq!(r.box_count, 3);
        // Adding cores can't meaningfully help: scoring is serialized, so the
        // wall (and thus box_count) is driven by total_score, which does NOT
        // divide by concurrency — still 3 even at 64 cores.
        let r_more_cores = recommend_instance(&cells, 25.0, 64);
        assert_eq!(r_more_cores.box_count, 3);
        // The wall is dominated by the serialized score term (50 s) — the
        // encode contribution (≤ 6 ms) is negligible.
        assert!(r.total_score_ms > 100.0 * r.total_encode_ms);
    }
}
