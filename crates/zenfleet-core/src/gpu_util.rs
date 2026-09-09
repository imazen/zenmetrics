//! GPU-utilization classification — the ONE registered definition of "ideal"
//! for a GPU metric fleet box (zenmetrics#48).
//!
//! "GPU usage is bad" was anecdotal-per-wave: dmon snippets pasted into status
//! docs, with no shared definition of what good would look like. The efficiency
//! program (warm exec #45, shared-resource sessions #47, native SPIR-V #44)
//! needs before/after claims to be *measurements against a registered target*
//! rather than impressions, and that requires the target to exist once, here,
//! rather than being re-argued per wave.
//!
//! # The definition, taken verbatim from the issue
//!
//! A GPU metric fleet box is **ideal** when it is **compute-bound** — kernel
//! execution time dominates the CUDA API's memory time (H2D + allocation) —
//! **or** when it sits at a **quantified upload-bound floor**: if the workload
//! is structurally upload-bound even after pinned staging and warm references,
//! then state the floor (bytes uploaded ÷ pinned bandwidth) and measure the
//! distance from *that*, not from 100% SM.
//!
//! **`sm%` alone is not the target.** The discipline is nsys's
//! `cuda_api_sum`-vs-`cuda_gpu_kern_sum` comparison, and the workspace GPU
//! notes are explicit that `api_sum` is read FIRST — a box can sit at a
//! respectable `sm%` while more than half its CUDA API time is
//! `cuMemcpyHtoDAsync`, which is the measured baseline this issue was filed
//! against (54% H2D, `sm` ~10%).
//!
//! # What this module is and is not
//!
//! It classifies **already-extracted** numbers. It deliberately does NOT parse
//! `nsys` or `nvidia-smi dmon` output: that collection half is a thin shell-out
//! that must be written against real tool output on a real box, and writing a
//! parser against a *remembered* format is how a harness ends up confidently
//! reporting nonsense. Feed it from whichever collector you build.

use serde::{Deserialize, Serialize};

/// Thresholds for [`classify`]. Defaults follow the issue's framing; they are
/// deliberately not "sm% > N".
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GpuUtilThresholds {
    /// Kernel time must exceed memory time by at least this factor to count as
    /// compute-bound. `1.0` = "kernels merely dominate"; the default demands a
    /// clear margin so a marginal box is not declared ideal.
    pub compute_bound_ratio: f64,
    /// Within this factor of the computed upload floor still counts as "at the
    /// floor" — an upload-bound workload cannot do better, so this is success,
    /// not failure.
    pub at_floor_tolerance: f64,
}

impl Default for GpuUtilThresholds {
    fn default() -> Self {
        Self {
            compute_bound_ratio: 1.5,
            at_floor_tolerance: 1.25,
        }
    }
}

/// One profiling window's already-extracted totals.
///
/// Times are nanoseconds, matching nsys's `--stats=true` tables. Fields that
/// were not collected are `None` and are skipped rather than guessed — the same
/// rule [`crate::idle`] follows.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GpuProfile {
    /// Total GPU kernel execution time (nsys `cuda_gpu_kern_sum`).
    pub kernel_ns: u64,
    /// Host→device copy time (nsys `cuda_api_sum`, `cuMemcpy*HtoD*`).
    pub h2d_ns: u64,
    /// Device allocation time (`cuMemAlloc*`), which per-call buffer churn
    /// inflates — 5–15 ms/iter at 12 MP when a caller re-allocates per call.
    pub alloc_ns: u64,
    /// Mean SM occupancy %, when sampled. Reported, never the criterion.
    pub sm_pct: Option<f64>,
    /// Bytes uploaded during the window — the numerator of the upload floor.
    pub uploaded_bytes: Option<u64>,
    /// Achievable host→device bandwidth in GB/s. Pageable `Vec` staging runs
    /// ~5–6 GB/s because the driver bounces through a hidden pinned buffer;
    /// genuinely pinned memory reaches 12–25 GB/s on PCIe 4.0. Pass the one
    /// that matches how the workload actually uploads.
    pub upload_bandwidth_gbps: Option<f64>,
}

/// What a window was bound by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuBound {
    /// Kernels dominate memory time by at least the configured ratio. **Ideal.**
    Compute,
    /// Upload-dominated, but at (or within tolerance of) the structural floor
    /// for the bytes moved. **Also ideal** — the workload cannot do better
    /// without moving fewer bytes.
    UploadAtFloor,
    /// Upload-dominated AND materially above the floor: bytes are moving slower
    /// than the link allows. Pageable staging instead of pinned, or redundant
    /// re-uploads. Actionable.
    UploadAboveFloor,
    /// Neither kernels nor uploads dominate — the GPU is waiting on the host.
    /// The measured 2026-08-07 baseline (sm alternating 85→0→49→0, FB down to
    /// 17 MB) is this.
    HostStalled,
    /// Not enough signal to classify (e.g. an all-zero window).
    Unknown,
}

/// The verdict, with the numbers that produced it so a report can show its work.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GpuVerdict {
    pub bound: GpuBound,
    /// `kernel_ns / (h2d_ns + alloc_ns)`. `None` when there is no memory time.
    pub compute_to_memory_ratio: Option<f64>,
    /// Fraction of (kernel + memory) time spent on H2D — the "54%" of the
    /// baseline.
    pub h2d_fraction: f64,
    /// The structural floor for the bytes moved, in nanoseconds.
    pub upload_floor_ns: Option<u64>,
    /// Measured H2D ÷ floor. `1.0` = at the floor; `3.0` = three times slower
    /// than the link allows. **This is the number to track over time**, not
    /// `sm%`.
    pub floor_excess: Option<f64>,
}

impl GpuVerdict {
    /// True when the box is meeting the registered definition of ideal.
    pub fn is_ideal(&self) -> bool {
        matches!(self.bound, GpuBound::Compute | GpuBound::UploadAtFloor)
    }

    /// One reportable line.
    pub fn line(&self) -> String {
        let ratio = self
            .compute_to_memory_ratio
            .map_or_else(|| "n/a".to_string(), |r| format!("{r:.2}x"));
        let floor = self
            .floor_excess
            .map_or_else(|| "no floor".to_string(), |e| format!("{e:.2}x floor"));
        format!(
            "{:?}: kern/mem {ratio}, h2d {:.0}% of busy time, {floor}{}",
            self.bound,
            self.h2d_fraction * 100.0,
            if self.is_ideal() { " [IDEAL]" } else { "" }
        )
    }
}

/// Classify one profiling window against the registered definition.
pub fn classify(p: &GpuProfile, t: &GpuUtilThresholds) -> GpuVerdict {
    let mem_ns = p.h2d_ns.saturating_add(p.alloc_ns);
    let busy = p.kernel_ns.saturating_add(mem_ns);

    let h2d_fraction = if busy == 0 {
        0.0
    } else {
        p.h2d_ns as f64 / busy as f64
    };
    let compute_to_memory_ratio = if mem_ns == 0 {
        None
    } else {
        Some(p.kernel_ns as f64 / mem_ns as f64)
    };

    // The upload floor: bytes ÷ bandwidth. GB/s here means 1e9 bytes/s, which
    // is how PCIe bandwidth is quoted; ns = bytes / (GB/s) exactly.
    let upload_floor_ns = match (p.uploaded_bytes, p.upload_bandwidth_gbps) {
        (Some(b), Some(bw)) if bw > 0.0 => Some((b as f64 / bw) as u64),
        _ => None,
    };
    let floor_excess = upload_floor_ns.and_then(|f| {
        if f == 0 {
            None
        } else {
            Some(p.h2d_ns as f64 / f as f64)
        }
    });

    let bound = if busy == 0 {
        GpuBound::Unknown
    } else if compute_to_memory_ratio.is_none_or(|r| r >= t.compute_bound_ratio) {
        // No memory time at all, or kernels clearly dominate it.
        GpuBound::Compute
    } else if h2d_fraction >= 0.5 {
        // Upload-dominated. Whether that is a problem depends entirely on
        // whether it is at the structural floor -- which is the whole point of
        // the issue's "measure distance from the floor, not from 100% sm".
        match floor_excess {
            Some(e) if e <= t.at_floor_tolerance => GpuBound::UploadAtFloor,
            Some(_) => GpuBound::UploadAboveFloor,
            // Upload-dominated with no floor supplied: we cannot say whether it
            // is achievable. Report it as above-floor so it is investigated
            // rather than silently blessed.
            None => GpuBound::UploadAboveFloor,
        }
    } else {
        GpuBound::HostStalled
    };

    GpuVerdict {
        bound,
        compute_to_memory_ratio,
        h2d_fraction,
        upload_floor_ns,
        floor_excess,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(kernel_ns: u64, h2d_ns: u64, alloc_ns: u64) -> GpuProfile {
        GpuProfile {
            kernel_ns,
            h2d_ns,
            alloc_ns,
            sm_pct: None,
            uploaded_bytes: None,
            upload_bandwidth_gbps: None,
        }
    }

    #[test]
    fn kernels_dominating_memory_is_ideal() {
        let v = classify(&p(900, 100, 0), &GpuUtilThresholds::default());
        assert_eq!(v.bound, GpuBound::Compute);
        assert!(v.is_ideal());
    }

    #[test]
    fn the_measured_2026_08_baseline_is_not_ideal() {
        // nsys on the ScoreFile shape: cuMemcpyHtoDAsync 54% of CUDA API time,
        // sm ~10%. Whatever else is true, this must not classify as ideal.
        let mut prof = p(100, 540, 60);
        prof.sm_pct = Some(10.0);
        let v = classify(&prof, &GpuUtilThresholds::default());
        assert!(
            !v.is_ideal(),
            "the baseline this issue was filed against classified as ideal"
        );
        assert!(v.h2d_fraction > 0.5);
    }

    #[test]
    fn upload_bound_at_the_structural_floor_is_ideal() {
        // 2 GB at 20 GB/s = 100 ms floor; measured 105 ms is at it.
        let prof = GpuProfile {
            kernel_ns: 10_000_000,
            h2d_ns: 105_000_000,
            alloc_ns: 0,
            sm_pct: Some(8.0),
            uploaded_bytes: Some(2_000_000_000),
            upload_bandwidth_gbps: Some(20.0),
        };
        let v = classify(&prof, &GpuUtilThresholds::default());
        assert_eq!(v.bound, GpuBound::UploadAtFloor);
        assert!(
            v.is_ideal(),
            "a workload at its structural floor cannot do better"
        );
        assert!((v.floor_excess.unwrap() - 1.05).abs() < 0.01);
    }

    #[test]
    fn pageable_staging_shows_up_as_above_floor() {
        // Same 2 GB, but running at pageable ~5.5 GB/s against a pinned-capable
        // 20 GB/s link: ~3.6x the floor. This is the pinned-upload win.
        let prof = GpuProfile {
            kernel_ns: 10_000_000,
            h2d_ns: 364_000_000,
            alloc_ns: 0,
            sm_pct: Some(8.0),
            uploaded_bytes: Some(2_000_000_000),
            upload_bandwidth_gbps: Some(20.0),
        };
        let v = classify(&prof, &GpuUtilThresholds::default());
        assert_eq!(v.bound, GpuBound::UploadAboveFloor);
        assert!(!v.is_ideal());
        assert!(v.floor_excess.unwrap() > 3.0);
    }

    #[test]
    fn upload_dominated_without_a_floor_is_flagged_not_blessed() {
        // No bytes/bandwidth supplied: we cannot prove it is achievable, so it
        // must be investigated rather than silently pass.
        let v = classify(&p(10, 900, 0), &GpuUtilThresholds::default());
        assert_eq!(v.bound, GpuBound::UploadAboveFloor);
        assert!(!v.is_ideal());
    }

    #[test]
    fn host_stalled_when_neither_dominates() {
        // Kernels below the compute ratio, H2D below half: the GPU is waiting.
        let v = classify(&p(100, 40, 60), &GpuUtilThresholds::default());
        assert_eq!(v.bound, GpuBound::HostStalled);
        assert!(!v.is_ideal());
    }

    #[test]
    fn sm_pct_alone_never_decides() {
        // Identical timings, opposite sm%: the verdict must not move, because
        // sm% is explicitly NOT the criterion.
        let mut hi = p(900, 100, 0);
        hi.sm_pct = Some(95.0);
        let mut lo = p(900, 100, 0);
        lo.sm_pct = Some(3.0);
        let t = GpuUtilThresholds::default();
        assert_eq!(classify(&hi, &t).bound, classify(&lo, &t).bound);
    }

    #[test]
    fn an_empty_window_is_unknown_not_ideal() {
        let v = classify(&p(0, 0, 0), &GpuUtilThresholds::default());
        assert_eq!(v.bound, GpuBound::Unknown);
        assert!(!v.is_ideal());
    }
}
