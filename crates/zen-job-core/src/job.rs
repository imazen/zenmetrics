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
    Encode { codec: String, q: i64, knobs: String },
    Metric { metric: String },
    Feature { regime: String },
    Diffmap { metric: String },
    Resample { kernel: String, w: u32, h: u32 },
    Bake { view: String },
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
        (ResourceClass::CpuHeavy, Regenerability::ExpensiveRegenerable)
    }
}

/// The production metric set (cvvdp/butter/ssim2/dssim/iwssim/zensim) is GPU-backed; route metrics to
/// GPU workers by default. A future pure-CPU metric can override to `CpuArm`/`CpuLight` here.
fn metric_class(_metric: &str) -> ResourceClass {
    ResourceClass::Gpu
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_cost_asymmetry() {
        let jpeg = JobKind::Encode { codec: "zenjpeg".into(), q: 80, knobs: "{}".into() }.profile();
        let avif = JobKind::Encode { codec: "zenavif".into(), q: 50, knobs: "{}".into() }.profile();
        // JPEG: cheap to regenerate, light CPU.
        assert_eq!(jpeg.output_regenerability, Regenerability::CheapRegenerable);
        assert_eq!(jpeg.class, ResourceClass::CpuLight);
        // AVIF: expensive — must persist, heavy CPU.
        assert_eq!(avif.output_regenerability, Regenerability::ExpensiveRegenerable);
        assert_eq!(avif.class, ResourceClass::CpuHeavy);
    }

    #[test]
    fn metric_groups_by_source_on_gpu() {
        let p = JobKind::Metric { metric: "cvvdp".into() }.profile();
        assert_eq!(p.group_by, GroupBy::SourceSha);
        assert_eq!(p.class, ResourceClass::Gpu);
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
        assert_eq!(uniq.len(), subs.len(), "every class must route to a distinct subject");
    }

    #[test]
    fn jobkind_serde_roundtrip() {
        let k = JobKind::Encode { codec: "zenjxl".into(), q: 90, knobs: "{\"effort\":7}".into() };
        let j = serde_json::to_string(&k).unwrap();
        assert_eq!(serde_json::from_str::<JobKind>(&j).unwrap(), k);
    }
}
