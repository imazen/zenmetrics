//! Job taxonomy: what kinds of work exist, and the per-kind profile (resource class, batching
//! group key, GC regenerability). The cost asymmetry the user flagged — JPEG encode ≈ 1/100th of a
//! metric (cheap), AVIF > a metric (expensive), JXL ≈ a metric (balanced); metrics are
//! reference-local — lives here as profile *data*, so the engine (queue, reconciler, GC, dashboard)
//! never special-cases a job kind. A new job kind is one variant + one `profile()` arm.

use serde::{Deserialize, Serialize};

/// Where a job runs — maps to a queue subject so workers pull only what their hardware serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceClass {
    CpuLight,
    CpuHeavy,
    CpuArm,
    Gpu,
    HighRam,
}

impl ResourceClass {
    /// Queue subject segment for capability routing (e.g. NATS `job.<subject>.*`).
    pub fn subject(self) -> &'static str {
        match self {
            ResourceClass::CpuLight => "cpu.light",
            ResourceClass::CpuHeavy => "cpu.heavy",
            ResourceClass::CpuArm => "cpu.arm",
            ResourceClass::Gpu => "gpu",
            ResourceClass::HighRam => "highram",
        }
    }

    /// Parse a worker-declared capability token (the serde snake_case name) — used by
    /// `zenfleet-worker --capability`. Accepts e.g. `cpu_light`/`cpu-light`/`gpu`/`high_ram`.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "cpu_light" | "cpulight" => Some(ResourceClass::CpuLight),
            "cpu_heavy" | "cpuheavy" => Some(ResourceClass::CpuHeavy),
            "cpu_arm" | "cpuarm" | "arm" => Some(ResourceClass::CpuArm),
            "gpu" => Some(ResourceClass::Gpu),
            "high_ram" | "highram" => Some(ResourceClass::HighRam),
            _ => None,
        }
    }

    /// Derive the routing class from an encode's *estimated peak memory*
    /// (`zencodec::estimate::ResourceEstimate::peak_memory_bytes_est`). This refines
    /// the static per-codec class (`encode_cost`) with the real per-encode
    /// footprint — a 64×64 AVIF is `CpuLight`, a 100 MP 10-bit AVIF or a
    /// JXL-modular-e9 is `HighRam`. Thresholds are deliberately coarse: they
    /// pick a *queue* (capability routing); the per-box [`crate::schedule`]
    /// admission control does the fine packing.
    pub fn from_peak_mem(peak_mem_bytes: u64) -> ResourceClass {
        const MB: u64 = 1 << 20;
        if peak_mem_bytes >= 4096 * MB {
            ResourceClass::HighRam // ≥ 4 GB → big-RAM boxes
        } else if peak_mem_bytes >= 512 * MB {
            ResourceClass::CpuHeavy // ≥ 512 MB
        } else {
            ResourceClass::CpuLight // < 512 MB
        }
    }
}

/// Capability routing (goal H "capability-routed (GPU/CPU/ARM)"): a worker advertising the resource
/// classes it serves handles a job **iff** the job's class is in that set. An empty set means "serve
/// everything" (a general worker), preserving prior behaviour. This is what lets a GPU box pull only
/// metric/diffmap jobs while an ARM box pulls only `cpu_arm`/`cpu_light` work — off one shared queue.
pub fn worker_serves(served: &[ResourceClass], kind: &JobKind) -> bool {
    served.is_empty() || served.contains(&kind.profile().class)
}

/// How items batch into a chunk — the locality lever. `SourceSha` lets the metric handler decode the
/// reference once and score many distorted variants against it (the orchestrator `run_all` win).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroupBy {
    None,
    SourceSha,
    Codec,
}

/// GC tier: can this output be cheaply rebuilt from recorded inputs, expensively rebuilt, or not at
/// all? Drives the reachability-GC eviction policy (cheap → LRU cache; expensive → keep under
/// pressure; not-regenerable → never auto-delete).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Regenerability {
    CheapRegenerable,
    ExpensiveRegenerable,
    NotRegenerable,
}

/// The declared profile of a job kind — everything the generic engine needs to route, batch, and GC.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobProfile {
    pub class: ResourceClass,
    pub group_by: GroupBy,
    pub output_regenerability: Regenerability,
}

/// The kinds of work the system can run. Strings for codec/metric keep this crate decoupled from the
/// codec/metric enums in sibling crates (the keystone depends on nothing heavy).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum JobKind {
    Encode {
        codec: String,
        q: i64,
        knobs: String,
    },
    Metric {
        metric: String,
    },
    /// Score ALL `metrics` for one source file in a single pass: decode the reference once, then for
    /// each input variant (a content sha the executor resolves to bytes) decode it once and score
    /// every metric against the shared reference. Makes the `SourceSha` grouping concrete — collapsing
    /// the per-(cell,metric) fan-out so a 24 MP source's decode and each variant's decode happen ONCE,
    /// never re-encoded or re-decoded per metric. `inputs` are the variant content shas.
    ScoreFile {
        metrics: Vec<String>,
    },
    Feature {
        regime: String,
    },
    Diffmap {
        metric: String,
    },
    Resample {
        kernel: String,
        w: u32,
        h: u32,
    },
    Bake {
        view: String,
    },
}

impl JobKind {
    /// The routing/batching/GC profile for this kind. This is the single place asymmetries live.
    pub fn profile(&self) -> JobProfile {
        match self {
            JobKind::Encode { codec, .. } => {
                let (class, output_regenerability) = encode_cost(codec);
                JobProfile {
                    class,
                    group_by: GroupBy::Codec,
                    output_regenerability,
                }
            }
            JobKind::Metric { metric } => JobProfile {
                class: metric_class(metric),
                // Decode the reference once, score many variants → group a source's encodes together.
                group_by: GroupBy::SourceSha,
                // Re-scoring is cheap *given the encode already exists*.
                output_regenerability: Regenerability::CheapRegenerable,
            },
            // Whole-file scoring: the SourceSha grouping made concrete — one GPU job decodes the ref
            // once and scores every metric for every variant (no per-(cell,metric) re-encode/re-decode).
            JobKind::ScoreFile { .. } => JobProfile {
                class: ResourceClass::Gpu,
                group_by: GroupBy::SourceSha,
                output_regenerability: Regenerability::CheapRegenerable,
            },
            JobKind::Feature { .. } => JobProfile {
                class: ResourceClass::CpuHeavy,
                group_by: GroupBy::SourceSha,
                output_regenerability: Regenerability::CheapRegenerable,
            },
            JobKind::Diffmap { metric } => JobProfile {
                class: metric_class(metric),
                group_by: GroupBy::SourceSha,
                // A GPU pass to rebuild — keep unless under storage pressure.
                output_regenerability: Regenerability::ExpensiveRegenerable,
            },
            JobKind::Resample { .. } => JobProfile {
                class: ResourceClass::CpuLight,
                group_by: GroupBy::None,
                output_regenerability: Regenerability::CheapRegenerable,
            },
            JobKind::Bake { .. } => JobProfile {
                class: ResourceClass::HighRam,
                group_by: GroupBy::None,
                output_regenerability: Regenerability::CheapRegenerable,
            },
        }
    }
}

/// Per-codec encode cost asymmetry (user-stated): JPEG/PNG are trivially cheap to re-encode → the
/// stored blob is effectively a cache. WebP/JXL/AVIF cost ≈ or ≫ a metric pass → treat as expensive
/// (persist; evict only under pressure). Unknown codecs default to expensive (safer for GC — keep).
fn encode_cost(codec: &str) -> (ResourceClass, Regenerability) {
    let c = codec.to_ascii_lowercase();
    if c.contains("jpeg") || c.contains("jpg") || c.contains("png") {
        (ResourceClass::CpuLight, Regenerability::CheapRegenerable)
    } else {
        // webp / jxl / avif / unknown
        (
            ResourceClass::CpuHeavy,
            Regenerability::ExpensiveRegenerable,
        )
    }
}

/// The production metric set (cvvdp/butter/ssim2/dssim/iwssim/zensim) is GPU-backed; route metrics to
/// GPU workers by default. A future pure-CPU metric can override to `CpuArm`/`CpuLight` here.
fn metric_class(_metric: &str) -> ResourceClass {
    ResourceClass::Gpu
}

impl JobKind {
    /// A **rough** per-cell *serial* wall-time estimate (seconds), used only to SIZE work-stealing
    /// chunks (it fills [`crate::schedule::JobCost::cost_sec`], which feeds
    /// [`crate::schedule::BoxBudget::pack_chunks`]). `zencodec`'s `ResourceEstimate` carries no time
    /// field yet, so we proxy from two cheap signals:
    ///   1. the kind's [`ResourceClass`] — the same cheap/expensive codec asymmetry [`encode_cost`]
    ///      encodes (JPEG/PNG re-encode ≈ 1/100th of a metric; WebP/JXL/AVIF ≈ or ≫ a metric), and
    ///   2. `peak_mem_bytes` (the job's [`crate::ledger::ResourceHint`] footprint) as an image-size
    ///      proxy — a bigger working set ≈ a bigger image ≈ more encode/score time.
    ///
    /// **DELIBERATELY COARSE — these are NOT measured numbers.** Chunk sizing is self-correcting: a
    /// 2× error only changes how often a box re-claims (claim cadence), never correctness or the
    /// per-cell memory bound (that is [`crate::schedule::BoxBudget::can_admit`]). Refine the constants
    /// from measured omni `encode_ms` telemetry when it lands.
    pub fn estimate_cost_sec(&self, peak_mem_bytes: u64) -> f64 {
        const GIB: f64 = (1u64 << 30) as f64;
        // Floor: even a 64×64 cell pays process spawn + source fetch + IO — no cell is free.
        const FLOOR_SEC: f64 = 1.0;
        // Seconds per GiB of working set, by class (rough proxies, NOT measurements).
        let per_gib = match self.profile().class {
            ResourceClass::CpuLight => 3.0, // jpeg/png/resample — cheap re-encode
            ResourceClass::CpuArm => 8.0,
            ResourceClass::Gpu => 15.0, // metric/diffmap/scorefile — reference-local
            ResourceClass::CpuHeavy | ResourceClass::HighRam => 40.0, // webp/jxl/avif/feature/bake
        };
        FLOOR_SEC + per_gib * (peak_mem_bytes as f64 / GIB)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_cost_asymmetry() {
        let jpeg = JobKind::Encode {
            codec: "zenjpeg".into(),
            q: 80,
            knobs: "{}".into(),
        }
        .profile();
        let avif = JobKind::Encode {
            codec: "zenavif".into(),
            q: 50,
            knobs: "{}".into(),
        }
        .profile();
        // JPEG: cheap to regenerate, light CPU.
        assert_eq!(jpeg.output_regenerability, Regenerability::CheapRegenerable);
        assert_eq!(jpeg.class, ResourceClass::CpuLight);
        // AVIF: expensive — must persist, heavy CPU.
        assert_eq!(
            avif.output_regenerability,
            Regenerability::ExpensiveRegenerable
        );
        assert_eq!(avif.class, ResourceClass::CpuHeavy);
    }

    #[test]
    fn metric_groups_by_source_on_gpu() {
        let p = JobKind::Metric {
            metric: "cvvdp".into(),
        }
        .profile();
        assert_eq!(p.group_by, GroupBy::SourceSha);
        assert_eq!(p.class, ResourceClass::Gpu);
    }

    #[test]
    fn cost_sec_floor_and_codec_ordering() {
        const MB: u64 = 1 << 20;
        let jpeg = JobKind::Encode {
            codec: "zenjpeg".into(),
            q: 80,
            knobs: "{}".into(),
        };
        let avif = JobKind::Encode {
            codec: "zenavif".into(),
            q: 50,
            knobs: "{}".into(),
        };
        // A ~zero-footprint cell costs ≈ the floor (no cell is free: spawn + fetch + IO).
        assert!((jpeg.estimate_cost_sec(0) - 1.0).abs() < 1e-9);
        // At the SAME footprint, the expensive codec (AVIF, CpuHeavy) is estimated dearer than the
        // cheap one (JPEG, CpuLight) — the encode_cost asymmetry flows through.
        assert!(
            avif.estimate_cost_sec(256 * MB) > jpeg.estimate_cost_sec(256 * MB),
            "AVIF should size dearer than JPEG at equal working set"
        );
    }

    #[test]
    fn cost_sec_is_monotonic_in_memory() {
        const MB: u64 = 1 << 20;
        let k = JobKind::Encode {
            codec: "zenjxl".into(),
            q: 50,
            knobs: "{}".into(),
        };
        // Bigger working set (≈ bigger image) → larger time estimate.
        assert!(k.estimate_cost_sec(64 * MB) < k.estimate_cost_sec(2048 * MB));
    }

    #[test]
    fn subjects_are_distinct() {
        let subs = [
            ResourceClass::CpuLight,
            ResourceClass::CpuHeavy,
            ResourceClass::CpuArm,
            ResourceClass::Gpu,
            ResourceClass::HighRam,
        ]
        .map(ResourceClass::subject);
        let mut uniq = subs.to_vec();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(
            uniq.len(),
            subs.len(),
            "every class must route to a distinct subject"
        );
    }

    #[test]
    fn class_from_peak_mem_buckets() {
        use ResourceClass::*;
        const MB: u64 = 1 << 20;
        assert_eq!(ResourceClass::from_peak_mem(80 * MB), CpuLight); // small JPEG
        assert_eq!(ResourceClass::from_peak_mem(600 * MB), CpuHeavy); // mid AVIF
        assert_eq!(ResourceClass::from_peak_mem(8192 * MB), HighRam); // JXL-modular-e9
    }

    #[test]
    fn capability_routing_matches_class() {
        let metric = JobKind::Metric {
            metric: "cvvdp".into(),
        }; // Gpu
        let jpeg = JobKind::Encode {
            codec: "zenjpeg".into(),
            q: 80,
            knobs: "{}".into(),
        }; // CpuLight
        let avif = JobKind::Encode {
            codec: "zenavif".into(),
            q: 50,
            knobs: "{}".into(),
        }; // CpuHeavy
        let gpu = [ResourceClass::Gpu];
        let cpu = [ResourceClass::CpuLight, ResourceClass::CpuHeavy];
        // GPU worker serves the metric, not the encodes.
        assert!(worker_serves(&gpu, &metric));
        assert!(!worker_serves(&gpu, &jpeg));
        // CPU worker serves both encodes, not the GPU metric.
        assert!(worker_serves(&cpu, &jpeg));
        assert!(worker_serves(&cpu, &avif));
        assert!(!worker_serves(&cpu, &metric));
        // empty set = general worker, serves everything.
        assert!(worker_serves(&[], &metric) && worker_serves(&[], &jpeg));
    }

    #[test]
    fn resource_class_parse() {
        assert_eq!(
            ResourceClass::parse("cpu_light"),
            Some(ResourceClass::CpuLight)
        );
        assert_eq!(ResourceClass::parse("GPU"), Some(ResourceClass::Gpu));
        assert_eq!(ResourceClass::parse("cpu-arm"), Some(ResourceClass::CpuArm));
        assert_eq!(ResourceClass::parse("nonsense"), None);
    }

    #[test]
    fn jobkind_serde_roundtrip() {
        let k = JobKind::Encode {
            codec: "zenjxl".into(),
            q: 90,
            knobs: "{\"effort\":7}".into(),
        };
        let j = serde_json::to_string(&k).unwrap();
        assert_eq!(serde_json::from_str::<JobKind>(&j).unwrap(), k);
    }
}
