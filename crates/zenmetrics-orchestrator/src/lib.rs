//! zenmetrics orchestrator — Phase 1: capability detection + persistent cache.
//!
//! This crate is the foundation for the higher-level orchestrator described
//! in `crates/zenmetrics-api/docs/ORCHESTRATOR_DESIGN.md`. Phase 1 ships
//! only the parts that every later phase depends on:
//!
//! 1. Hardware/driver detection (GPU via `nvidia-smi`, CPU via `raw-cpuid`,
//!    RAM via `sysinfo`).
//! 2. A deterministic `machine_hash` so the cache file name is stable.
//! 3. TOML serialization to `~/.cache/zenmetrics/capability_<short>.toml`.
//! 4. Stale-detection (time-based + hardware/driver-change).
//!
//! No scheduling, no benchmark runner, no worker pool. Those land in
//! Phases 2–7.
//!
//! ## Stability
//!
//! All public structs keep their fields `pub` so later phases can add
//! fields without going through accessor refactors. The metric-profile
//! table is reserved as an empty `BTreeMap` in Phase 1 and populated
//! in Phase 2.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

mod bench;
#[cfg(feature = "bench")]
mod chooser;
mod cpu;
#[cfg(feature = "bench")]
mod cpu_adapter;
#[cfg(all(feature = "bench", feature = "cuda"))]
mod executor;
mod gpu;
#[cfg(all(feature = "bench", feature = "cuda"))]
mod pool;

pub use bench::{locate_bench_worker, synth_pair_offset_dist, BenchPlan, BenchReport};
#[cfg(feature = "bench")]
pub use chooser::{
    BackendChoice, CandidateStatus, ChooserConfig, ChooserError, ConsideredCandidate,
    RejectReason, TaskShape,
};
pub use cpu::{detect_cpu, detect_wsl2_host_ram_mib_hint};
#[cfg(all(feature = "bench", feature = "cuda"))]
pub use executor::{
    AttemptOutcome, OrchestratorError as ExecutorError, Task, TaskData, TaskResult,
};
pub use gpu::detect_gpu;
#[cfg(all(feature = "bench", feature = "cuda"))]
pub use pool::{CachedRefStats, PoolConfig, RunAllIter, TaskHandle, TaskRefHandle};

/// Error type for orchestrator operations. Variants will be extended in
/// later phases (benchmark failures, scheduler errors, etc.) — callers
/// should match exhaustively only when they want the compiler to flag
/// future additions.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum OrchestratorError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("toml decode: {0}")]
    TomlDecode(String),
    #[error("toml encode: {0}")]
    TomlEncode(String),
    #[error("cache dir could not be resolved (HOME unset and no XDG_CACHE_HOME)")]
    NoCacheDir,
    #[error("system time is before UNIX_EPOCH")]
    BadSystemTime,
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, OrchestratorError>;

/// Static configuration for an [`Orchestrator`] instance. Phase 1 only
/// uses `cache_dir` + `cache_validity`; later phases will add scheduling
/// knobs (parallelism caps, OOM-retry strategy, etc.). The struct is
/// `#[non_exhaustive]` so adding fields is non-breaking; callers must
/// use [`OrchestratorConfig::default`] + struct-update syntax or the
/// builder pattern that Phase 2 introduces.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OrchestratorConfig {
    /// Where to store the persistent capability profile. Defaults to
    /// `$XDG_CACHE_HOME/zenmetrics/` or `~/.cache/zenmetrics/`.
    pub cache_dir: PathBuf,
    /// How long a cached profile is considered fresh. Defaults to 7
    /// days. After this elapses the orchestrator re-runs detection +
    /// rewrites the file (Phase 2 will also re-bench).
    pub cache_validity: Duration,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            cache_dir: default_cache_dir().unwrap_or_else(|| PathBuf::from(".cache/zenmetrics")),
            cache_validity: Duration::from_secs(7 * 24 * 60 * 60),
        }
    }
}

/// Resolve the cache directory using `dirs::cache_dir()` (honours
/// `XDG_CACHE_HOME` on Linux + `LOCALAPPDATA` on Windows + the macOS
/// equivalent). Returns `None` only on truly homeless systems — caller
/// can fall back to a relative path.
pub fn default_cache_dir() -> Option<PathBuf> {
    dirs::cache_dir().map(|p| p.join("zenmetrics"))
}

/// GPU capability snapshot. `present = false` means we couldn't talk to
/// `nvidia-smi`; the other fields hold default/empty values in that
/// case so the struct round-trips cleanly through TOML.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct GpuCapability {
    pub present: bool,
    pub model: String,
    pub total_vram_mib: usize,
    pub driver_version: String,
    pub cuda_runtime: Option<String>,
    pub compute_capability: Option<String>,
}

/// CPU capability snapshot — brand string, core count, SIMD feature
/// set, total physical RAM.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct CpuCapability {
    pub brand: String,
    pub logical_cores: usize,
    pub features: Vec<String>,
    pub ram_mib: usize,
}

/// Backend variant the bench runner measured against. One row per
/// backend in [`MetricProfile::ns_per_px_at`] / [`MetricProfile::vram_mib_at`].
///
/// `GpuStripPair` is cvvdp-only — it surfaces from
/// [`zenmetrics_api::cvvdp::CvvdpOpaque::new_with_memory_mode`] with
/// `cvvdp_gpu::MemoryMode::StripPair`. Other metrics omit that backend.
///
/// `Cpu` is reserved for Phase 6; Phase 2 leaves it `None` everywhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    /// `MemoryMode::Full` against the GPU's CUDA backend.
    GpuFull,
    /// `MemoryMode::Strip { h_body: None }` against the GPU's CUDA backend.
    GpuStrip,
    /// cvvdp-only: `cvvdp_gpu::MemoryMode::StripPair { h_body: 256 }`.
    GpuStripPair,
    /// Reference CPU backend (wired in Phase 6).
    Cpu,
}

impl Backend {
    /// Short stable tag, used in error messages and TOML keys.
    pub fn tag(self) -> &'static str {
        match self {
            Backend::GpuFull => "gpu_full",
            Backend::GpuStrip => "gpu_strip",
            Backend::GpuStripPair => "gpu_strip_pair",
            Backend::Cpu => "cpu",
        }
    }
}

/// Wall-time p50 (steady-state) in nanoseconds per pixel, one entry per
/// backend at a fixed image size. `None` means "backend not measured at
/// this size" — either feature-gated out, surfaced an OOM, or skipped
/// because a prior size already exhausted the budget.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct BackendBench {
    /// Full-image working-set on GPU.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_full: Option<f64>,
    /// Strip-walker on GPU.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_strip: Option<f64>,
    /// cvvdp-only one-shot strip-pair walker.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_strip_pair: Option<f64>,
    /// CPU reference path (Phase 6).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu: Option<f64>,
}

/// VRAM peak in MiB during steady-state compute, one entry per backend
/// at a fixed image size. Same `None` semantics as [`BackendBench`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendVram {
    /// Full-image working-set on GPU.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_full: Option<usize>,
    /// Strip-walker on GPU.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_strip: Option<usize>,
    /// cvvdp-only one-shot strip-pair walker.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_strip_pair: Option<usize>,
    /// CPU resident-set MiB (Phase 6 — proxy for working-set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu: Option<usize>,
}

impl BackendBench {
    /// Set the entry for a given backend.
    pub fn set(&mut self, backend: Backend, ns_per_px: f64) {
        match backend {
            Backend::GpuFull => self.gpu_full = Some(ns_per_px),
            Backend::GpuStrip => self.gpu_strip = Some(ns_per_px),
            Backend::GpuStripPair => self.gpu_strip_pair = Some(ns_per_px),
            Backend::Cpu => self.cpu = Some(ns_per_px),
        }
    }

    /// Read the entry for a given backend (returns `None` if not measured).
    pub fn get(&self, backend: Backend) -> Option<f64> {
        match backend {
            Backend::GpuFull => self.gpu_full,
            Backend::GpuStrip => self.gpu_strip,
            Backend::GpuStripPair => self.gpu_strip_pair,
            Backend::Cpu => self.cpu,
        }
    }
}

impl BackendVram {
    /// Set the entry for a given backend.
    pub fn set(&mut self, backend: Backend, mib: usize) {
        match backend {
            Backend::GpuFull => self.gpu_full = Some(mib),
            Backend::GpuStrip => self.gpu_strip = Some(mib),
            Backend::GpuStripPair => self.gpu_strip_pair = Some(mib),
            Backend::Cpu => self.cpu = Some(mib),
        }
    }

    /// Read the entry for a given backend (returns `None` if not measured).
    pub fn get(&self, backend: Backend) -> Option<usize> {
        match backend {
            Backend::GpuFull => self.gpu_full,
            Backend::GpuStrip => self.gpu_strip,
            Backend::GpuStripPair => self.gpu_strip_pair,
            Backend::Cpu => self.cpu,
        }
    }
}

/// Per-metric profile populated by [`Orchestrator::bench`] /
/// [`Orchestrator::warm`]. Phase 1 left this empty; Phase 2 fills the
/// `ns_per_px_at` + `vram_mib_at` measured points + the OOM-cell log.
///
/// Keys on the inner maps are `u64` "size pixels" (`width × height`),
/// e.g. `1024 * 1024 = 1048576`. Phase 3's backend chooser interpolates
/// between measured sizes.
///
/// TOML representation: the inner `BTreeMap<u64, _>` is serialised with
/// stringified integer keys (TOML maps don't support integer keys
/// directly). The orchestrator's TOML round-trip handles the conversion
/// transparently via the `u64_keyed_map` serde helper.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct MetricProfile {
    /// Wall-time p50 (steady-state, post-warmup) keyed by image size
    /// (`width × height` in pixels).
    #[serde(default, with = "u64_keyed_map_bench")]
    pub ns_per_px_at: BTreeMap<u64, BackendBench>,
    /// Peak VRAM during compute, keyed by same size.
    #[serde(default, with = "u64_keyed_map_vram")]
    pub vram_mib_at: BTreeMap<u64, BackendVram>,
    /// Wall-clock timestamp of the last benchmark for this metric.
    #[serde(default, with = "systime_opt")]
    pub last_measured: Option<SystemTime>,
    /// `(backend, size_pixels)` cells that surfaced OOM during the
    /// bench. Phase 3's chooser treats these as a hard "do not retry"
    /// list per the cached snapshot.
    #[serde(default)]
    pub cells_failed_oom: Vec<(Backend, u64)>,
}

/// The full persistent profile. Round-trips through TOML; the file
/// lives at `<cache_dir>/capability_<short_machine_hash>.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityProfile {
    /// `sha256(gpu.model + "|" + gpu.driver_version + "|" + cpu.brand
    /// + "|" + cpu.features.join(","))` — the full 64-char hex string.
    /// The cache filename uses the first 16 chars (see [`short_hash`]).
    pub machine_hash: String,
    #[serde(with = "systime")]
    pub detected_at: SystemTime,
    #[serde(with = "systime")]
    pub last_validated: SystemTime,
    pub gpu: GpuCapability,
    pub cpu: CpuCapability,
    /// Reserved for Phase 2: keyed by `"<metric>.<backend>"`, e.g.
    /// `"cvvdp.gpu_full"`, `"ssim2.cpu"`. Empty in Phase 1.
    #[serde(default)]
    pub metrics: BTreeMap<String, MetricProfile>,
}

impl CapabilityProfile {
    /// Build a fresh profile from the currently-detected hardware. The
    /// `detected_at` and `last_validated` stamps are both set to `now`.
    pub fn detect_now() -> Self {
        let gpu = detect_gpu();
        let cpu = detect_cpu();
        let machine_hash = compute_machine_hash(&gpu, &cpu);
        let now = SystemTime::now();
        Self {
            machine_hash,
            detected_at: now,
            last_validated: now,
            gpu,
            cpu,
            metrics: BTreeMap::new(),
        }
    }

    /// First 16 hex chars of `machine_hash` — used for the cache
    /// filename so multiple machines can share `~/.cache/zenmetrics/`
    /// (rare locally but useful in tests / shared dotfiles).
    pub fn short_hash(&self) -> &str {
        short_hash_str(&self.machine_hash)
    }
}

/// Compute `sha256(gpu.model + "|" + gpu.driver_version + "|" +
/// cpu.brand + "|" + cpu.features.join(","))` as a lowercase 64-char
/// hex string. Deterministic across runs on the same machine — the
/// inputs are all detection outputs that don't fluctuate.
pub fn compute_machine_hash(gpu: &GpuCapability, cpu: &CpuCapability) -> String {
    let mut hasher = Sha256::new();
    hasher.update(gpu.model.as_bytes());
    hasher.update(b"|");
    hasher.update(gpu.driver_version.as_bytes());
    hasher.update(b"|");
    hasher.update(cpu.brand.as_bytes());
    hasher.update(b"|");
    hasher.update(cpu.features.join(",").as_bytes());
    let digest = hasher.finalize();
    let mut s = String::with_capacity(64);
    for b in digest.iter() {
        use std::fmt::Write;
        let _ = write!(&mut s, "{:02x}", b);
    }
    s
}

fn short_hash_str(full: &str) -> &str {
    if full.len() >= 16 { &full[..16] } else { full }
}

/// Cache file path for a given `(cache_dir, machine_hash)`. The
/// `machine_hash` may be either the full 64-char hex or the short
/// 16-char form — both produce the same path.
pub fn cache_file_path(cache_dir: &Path, machine_hash: &str) -> PathBuf {
    let short = short_hash_str(machine_hash);
    cache_dir.join(format!("capability_{}.toml", short))
}

/// Read + deserialize a profile from disk. Returns `None` on any
/// failure (missing file, parse error, truncated file). Callers treat
/// `None` as "no cache, regenerate" — never as a fatal error.
pub fn load_cached_profile(path: &Path) -> Option<CapabilityProfile> {
    let text = fs::read_to_string(path).ok()?;
    toml::from_str::<CapabilityProfile>(&text).ok()
}

/// Serialize + write a profile to disk, pretty-printed for human
/// debugging. Creates the parent directory recursively if needed. The
/// write is atomic-ish: we write to `<path>.tmp` then rename, so a
/// crash mid-write doesn't corrupt the cache.
pub fn save_profile(path: &Path, profile: &CapabilityProfile) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let text = toml::to_string_pretty(profile)
        .map_err(|e| OrchestratorError::TomlEncode(e.to_string()))?;
    let tmp = path.with_extension("toml.tmp");
    fs::write(&tmp, text)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Stale-check: returns `true` when the cached profile should be
/// regenerated. Two triggers:
///
/// 1. Time: `now - profile.last_validated > validity`.
/// 2. Hardware/driver: a fresh `detect_gpu()` would produce a different
///    GPU model or driver version than the cached snapshot.
///
/// We re-run `detect_gpu()` here (cheap — one `nvidia-smi` call) but
/// not `detect_cpu()` (the CPU doesn't change mid-session, and a CPU
/// swap also produces a different `machine_hash` → different file).
pub fn is_profile_stale(
    profile: &CapabilityProfile,
    validity: Duration,
    now: SystemTime,
) -> bool {
    // Time check.
    if let Ok(elapsed) = now.duration_since(profile.last_validated) {
        if elapsed > validity {
            return true;
        }
    } else {
        // Clock went backwards more than `validity` — treat as stale to
        // avoid trusting an apparently-future timestamp.
        return true;
    }
    // Hardware / driver check. Only meaningful when a GPU is present
    // in the cached profile; absent-GPU profiles only stale on time.
    if profile.gpu.present {
        let fresh = detect_gpu();
        if !fresh.present
            || fresh.model != profile.gpu.model
            || fresh.driver_version != profile.gpu.driver_version
        {
            return true;
        }
    }
    false
}

/// Top-level orchestrator. Phase 1 held only configuration + the
/// cached capability profile; Phase 5 adds the worker pool, pool
/// configuration, and cached-ref auto-detect state.
///
/// The worker pool is lazily initialized on first `submit` /
/// `run_all` / `upload_reference` — `Orchestrator::new` stays cheap
/// (no thread spawn, no VRAM probe thread). Callers that just want
/// to inspect the capability profile pay nothing for the pool.
///
/// **Not `Clone`** — the pool's `mpsc::Receiver` and `JoinHandle`s are
/// single-owner. Multi-threaded callers wrap the orchestrator in
/// `Arc<Mutex<Orchestrator>>` themselves.
///
/// **Not `Sync`** for the same reason. `Send` *is* implemented, so
/// the orchestrator can move between threads as long as only one
/// thread touches it at a time.
pub struct Orchestrator {
    config: OrchestratorConfig,
    capability: CapabilityProfile,
    /// Pool configuration. Set via [`Self::set_pool_config`] before
    /// the first `submit`; frozen once workers spawn.
    #[cfg(all(feature = "bench", feature = "cuda"))]
    pool_config: crate::pool::PoolConfig,
    /// Worker pool — lazily initialised on first `submit` / `run_all` /
    /// `upload_reference`. `None` until then.
    #[cfg(all(feature = "bench", feature = "cuda"))]
    pool: Option<Box<crate::pool::PoolState>>,
}

impl std::fmt::Debug for Orchestrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("Orchestrator");
        s.field("config", &self.config)
            .field("capability_machine_hash", &self.capability.machine_hash);
        #[cfg(all(feature = "bench", feature = "cuda"))]
        {
            s.field("pool_initialised", &self.pool.is_some());
        }
        s.finish()
    }
}

impl Orchestrator {
    /// Build an orchestrator with the given config. Side effects:
    ///
    /// 1. Detects current hardware (cheap — < 100 ms typically).
    /// 2. Computes the machine hash and resolves the cache file path.
    /// 3. Tries to load an existing cached profile. If present AND not
    ///    stale, uses it; otherwise builds fresh from detection.
    /// 4. Updates `last_validated` to `now` and rewrites the cache.
    ///
    /// `new` is idempotent — calling it twice on the same machine
    /// produces the same `capability_hash` and rewrites the same file.
    pub fn new(config: OrchestratorConfig) -> Result<Self> {
        let now = SystemTime::now();

        // Detect first; we need the machine_hash to know where the
        // cache file lives.
        let mut fresh = CapabilityProfile::detect_now();
        let path = cache_file_path(&config.cache_dir, &fresh.machine_hash);

        let capability = match load_cached_profile(&path) {
            Some(cached)
                if cached.machine_hash == fresh.machine_hash
                    && !is_profile_stale(&cached, config.cache_validity, now) =>
            {
                // Cache hit. Bump last_validated and re-save.
                let mut updated = cached;
                updated.last_validated = now;
                save_profile(&path, &updated)?;
                updated
            }
            _ => {
                // Cache miss / stale / hash mismatch. Persist fresh.
                fresh.last_validated = now;
                save_profile(&path, &fresh)?;
                fresh
            }
        };

        Ok(Self {
            config,
            capability,
            #[cfg(all(feature = "bench", feature = "cuda"))]
            pool_config: crate::pool::PoolConfig::default(),
            #[cfg(all(feature = "bench", feature = "cuda"))]
            pool: None,
        })
    }

    /// Borrow the active capability profile.
    pub fn capability(&self) -> &CapabilityProfile {
        &self.capability
    }

    /// Construct an orchestrator from a pre-built `CapabilityProfile`
    /// + config without running detection or touching the cache file.
    ///
    /// Use cases:
    /// - Tests that need to feed a deterministic profile into the
    ///   chooser without depending on real hardware.
    /// - Fleet deployments that load a curated profile from a shared
    ///   store (S3, NFS) instead of running the local bench.
    /// - Re-creating an orchestrator from a profile produced by a
    ///   separate detection pass (e.g., a privileged daemon).
    ///
    /// Phase 3 onwards relies on this constructor for the chooser
    /// test suite (`tests/chooser.rs`). Production code paths should
    /// still prefer [`Self::new`].
    pub fn from_capability(config: OrchestratorConfig, capability: CapabilityProfile) -> Self {
        Self {
            config,
            capability,
            #[cfg(all(feature = "bench", feature = "cuda"))]
            pool_config: crate::pool::PoolConfig::default(),
            #[cfg(all(feature = "bench", feature = "cuda"))]
            pool: None,
        }
    }

    /// Borrow the active config.
    pub fn config(&self) -> &OrchestratorConfig {
        &self.config
    }

    /// Compute the cache file path this orchestrator writes to. Useful
    /// for tests + the `print_capability` example.
    pub fn cache_path(&self) -> PathBuf {
        cache_file_path(&self.config.cache_dir, &self.capability.machine_hash)
    }
}

// ---------------------------------------------------------------------------
// Bench runner — populates `capability.metrics`. Defined inline so the
// Orchestrator impl block keeps all the public methods in one file.
// ---------------------------------------------------------------------------

impl Orchestrator {
    /// Run the full quick-bench across every metric × backend × size
    /// the build supports. Unconditional — overwrites any prior
    /// measurements in `capability.metrics`, refreshes `last_validated`,
    /// and re-saves the cache file.
    ///
    /// Total runtime budget: < 60s on an RTX 5070 / 7950X workstation
    /// at the default sizes (1024², 2048², 4096²). Cells that exceed
    /// 5s mark a "likely too large" warning + skip the remaining sizes
    /// for that backend (see [`BenchPlan::soft_timeout_per_cell`]).
    pub fn bench(&mut self) -> Result<()> {
        self.bench_with_plan(BenchPlan::default())
    }

    /// Same as [`Self::bench`] but with an explicit [`BenchPlan`] (test
    /// suites use this to override sizes / iterations / timeouts).
    pub fn bench_with_plan(&mut self, plan: BenchPlan) -> Result<()> {
        let report = bench::run(&plan);
        self.capability.metrics = report.metrics;
        self.capability.last_validated = SystemTime::now();
        save_profile(&self.cache_path(), &self.capability)?;
        Ok(())
    }

    /// Bench-on-demand: only run [`Self::bench`] if `capability.metrics`
    /// is empty OR any metric profile's `last_measured` is older than
    /// `self.config.cache_validity`. Cache-hit otherwise.
    ///
    /// Returns `true` if the bench actually ran, `false` on cache-hit.
    pub fn warm(&mut self) -> Result<bool> {
        let now = SystemTime::now();
        let validity = self.config.cache_validity;
        let needs_bench = self.capability.metrics.is_empty()
            || self.capability.metrics.values().any(|m| match m.last_measured {
                Some(t) => match now.duration_since(t) {
                    Ok(elapsed) => elapsed > validity,
                    Err(_) => true,
                },
                None => true,
            });
        if needs_bench {
            self.bench()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

// ---------------------------------------------------------------------------
// u64-keyed map serde helpers — TOML doesn't support non-string map keys
// so we round-trip BTreeMap<u64, _> via stringified integer keys.
// ---------------------------------------------------------------------------

mod u64_keyed_map_bench {
    use std::collections::BTreeMap;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use crate::BackendBench;

    pub fn serialize<S: Serializer>(
        m: &BTreeMap<u64, BackendBench>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        let stringified: BTreeMap<String, &BackendBench> =
            m.iter().map(|(k, v)| (k.to_string(), v)).collect();
        stringified.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<BTreeMap<u64, BackendBench>, D::Error> {
        let stringified: BTreeMap<String, BackendBench> = BTreeMap::deserialize(d)?;
        let mut out: BTreeMap<u64, BackendBench> = BTreeMap::new();
        for (k, v) in stringified {
            let parsed: u64 = k.parse().map_err(serde::de::Error::custom)?;
            out.insert(parsed, v);
        }
        Ok(out)
    }
}

mod u64_keyed_map_vram {
    use std::collections::BTreeMap;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use crate::BackendVram;

    pub fn serialize<S: Serializer>(
        m: &BTreeMap<u64, BackendVram>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        let stringified: BTreeMap<String, &BackendVram> =
            m.iter().map(|(k, v)| (k.to_string(), v)).collect();
        stringified.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<BTreeMap<u64, BackendVram>, D::Error> {
        let stringified: BTreeMap<String, BackendVram> = BTreeMap::deserialize(d)?;
        let mut out: BTreeMap<u64, BackendVram> = BTreeMap::new();
        for (k, v) in stringified {
            let parsed: u64 = k.parse().map_err(serde::de::Error::custom)?;
            out.insert(parsed, v);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// SystemTime serde helpers — TOML doesn't natively understand SystemTime,
// so we round-trip via RFC-3339-ish epoch seconds.
// ---------------------------------------------------------------------------

mod systime {
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(t: &SystemTime, s: S) -> Result<S::Ok, S::Error> {
        let secs = t
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        secs.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<SystemTime, D::Error> {
        let secs = u64::deserialize(d)?;
        Ok(UNIX_EPOCH + Duration::from_secs(secs))
    }
}

mod systime_opt {
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(
        t: &Option<SystemTime>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        match t {
            Some(t) => {
                let secs = t
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                Some(secs).serialize(s)
            }
            None => None::<u64>.serialize(s),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<SystemTime>, D::Error> {
        let opt = Option::<u64>::deserialize(d)?;
        Ok(opt.map(|s| UNIX_EPOCH + Duration::from_secs(s)))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_gpu(model: &str, driver: &str) -> GpuCapability {
        GpuCapability {
            present: true,
            model: model.into(),
            total_vram_mib: 12288,
            driver_version: driver.into(),
            cuda_runtime: Some("13.2.1".into()),
            compute_capability: Some("8.9".into()),
        }
    }

    fn fake_cpu() -> CpuCapability {
        CpuCapability {
            brand: "AMD Ryzen 9 7950X".into(),
            logical_cores: 32,
            features: vec!["avx2".into(), "avx512f".into(), "sse4.2".into()],
            ram_mib: 131072,
        }
    }

    fn fake_profile() -> CapabilityProfile {
        let gpu = fake_gpu("NVIDIA GeForce RTX 5070", "596.21");
        let cpu = fake_cpu();
        let machine_hash = compute_machine_hash(&gpu, &cpu);
        let now = SystemTime::now();
        CapabilityProfile {
            machine_hash,
            detected_at: now,
            last_validated: now,
            gpu,
            cpu,
            metrics: BTreeMap::new(),
        }
    }

    #[test]
    fn machine_hash_is_deterministic_and_64_hex() {
        let gpu = fake_gpu("NVIDIA GeForce RTX 5070", "596.21");
        let cpu = fake_cpu();
        let h1 = compute_machine_hash(&gpu, &cpu);
        let h2 = compute_machine_hash(&gpu, &cpu);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64);
        assert!(h1.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn machine_hash_changes_on_driver_bump() {
        let gpu_a = fake_gpu("NVIDIA GeForce RTX 5070", "596.21");
        let gpu_b = fake_gpu("NVIDIA GeForce RTX 5070", "600.00");
        let cpu = fake_cpu();
        assert_ne!(
            compute_machine_hash(&gpu_a, &cpu),
            compute_machine_hash(&gpu_b, &cpu)
        );
    }

    #[test]
    fn machine_hash_changes_on_cpu_feature_set() {
        let gpu = fake_gpu("NVIDIA GeForce RTX 5070", "596.21");
        let mut cpu = fake_cpu();
        let h1 = compute_machine_hash(&gpu, &cpu);
        cpu.features.push("aes".into());
        let h2 = compute_machine_hash(&gpu, &cpu);
        assert_ne!(h1, h2);
    }

    #[test]
    fn cache_file_path_uses_short_hash() {
        let p = cache_file_path(
            Path::new("/tmp/zm"),
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        );
        assert_eq!(
            p,
            PathBuf::from("/tmp/zm/capability_0123456789abcdef.toml")
        );
    }

    #[test]
    fn save_then_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let profile = fake_profile();
        let path = cache_file_path(dir.path(), &profile.machine_hash);
        save_profile(&path, &profile).unwrap();
        let loaded = load_cached_profile(&path).expect("load");
        assert_eq!(loaded.machine_hash, profile.machine_hash);
        assert_eq!(loaded.gpu, profile.gpu);
        assert_eq!(loaded.cpu, profile.cpu);
    }

    #[test]
    fn load_returns_none_for_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does-not-exist.toml");
        assert!(load_cached_profile(&path).is_none());
    }

    #[test]
    fn load_returns_none_for_garbage_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("junk.toml");
        fs::write(&path, b"\xff\xfeNOT TOML\x00").unwrap();
        assert!(load_cached_profile(&path).is_none());
    }

    #[test]
    fn save_creates_parent_dir() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("a/b/c");
        let profile = fake_profile();
        let path = cache_file_path(&nested, &profile.machine_hash);
        save_profile(&path, &profile).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn fresh_profile_is_not_stale_by_time() {
        let profile = fake_profile();
        let now = profile.last_validated + Duration::from_secs(10);
        assert!(!is_profile_stale(&profile, Duration::from_secs(86400), now));
    }

    #[test]
    fn old_profile_is_stale_by_time() {
        let profile = fake_profile();
        let now = profile.last_validated + Duration::from_secs(8 * 86400);
        assert!(is_profile_stale(&profile, Duration::from_secs(7 * 86400), now));
    }

    #[test]
    fn clock_skew_backwards_is_stale() {
        let profile = fake_profile();
        // `now` earlier than last_validated by more than validity.
        let now = profile.last_validated - Duration::from_secs(8 * 86400);
        assert!(is_profile_stale(&profile, Duration::from_secs(7 * 86400), now));
    }

    #[test]
    fn absent_gpu_profile_only_stales_on_time() {
        // GPU absent → hardware-change check is skipped, only time matters.
        let mut profile = fake_profile();
        profile.gpu = GpuCapability::default();
        let now = profile.last_validated + Duration::from_secs(10);
        assert!(!is_profile_stale(&profile, Duration::from_secs(86400), now));
    }

    #[test]
    fn short_hash_works_on_short_and_long() {
        assert_eq!(short_hash_str("0123456789abcdef0123"), "0123456789abcdef");
        assert_eq!(short_hash_str("abc"), "abc");
        assert_eq!(short_hash_str(""), "");
    }

    #[test]
    fn detect_cpu_returns_nonempty_brand() {
        let cpu = detect_cpu();
        assert!(!cpu.brand.is_empty(), "CPU brand should be detectable");
        assert!(cpu.logical_cores >= 1);
    }

    #[test]
    fn orchestrator_writes_cache_file() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = OrchestratorConfig {
            cache_dir: dir.path().to_path_buf(),
            cache_validity: Duration::from_secs(60),
        };
        let orch = Orchestrator::new(cfg).expect("Orchestrator::new");
        let path = orch.cache_path();
        assert!(path.exists(), "cache file should exist after new()");
        let loaded = load_cached_profile(&path).expect("load");
        assert_eq!(loaded.machine_hash, orch.capability().machine_hash);
    }

    #[test]
    fn orchestrator_second_call_loads_cache() {
        use std::time::UNIX_EPOCH;
        let dir = tempfile::tempdir().unwrap();
        let cfg = OrchestratorConfig {
            cache_dir: dir.path().to_path_buf(),
            cache_validity: Duration::from_secs(60),
        };
        let orch1 = Orchestrator::new(cfg.clone()).unwrap();
        let detected_at_1_secs = orch1
            .capability()
            .detected_at
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        // Second call should not change `detected_at` (only
        // `last_validated`), proving the cache was used. Compare at
        // second-resolution because the on-disk TOML format stores
        // SystemTime as UNIX seconds (sub-second nanos are dropped
        // on serialize/deserialize).
        let orch2 = Orchestrator::new(cfg).unwrap();
        let detected_at_2_secs = orch2
            .capability()
            .detected_at
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        assert_eq!(orch1.capability().machine_hash, orch2.capability().machine_hash);
        assert_eq!(detected_at_1_secs, detected_at_2_secs);
    }

    #[test]
    fn orchestrator_send_sync() {
        fn _assert_send_sync<T: Send + Sync>() {}
        // Phase 5: Orchestrator gained a worker pool whose mpsc::Receiver
        // is !Sync, so the orchestrator dropped Sync. It remains Send
        // (and the cap/config remain Send + Sync).
        fn _assert_send<T: Send>() {}
        _assert_send::<Orchestrator>();
        _assert_send_sync::<CapabilityProfile>();
        _assert_send_sync::<OrchestratorConfig>();
    }

    // ----------- MetricProfile / Backend round-trip tests (Phase 2) -----------

    #[test]
    fn backend_tag_round_trip() {
        for b in [
            Backend::GpuFull,
            Backend::GpuStrip,
            Backend::GpuStripPair,
            Backend::Cpu,
        ] {
            // Tag is what serializes into TOML keys + report headers —
            // assert each variant has a distinct stable tag.
            let s = b.tag();
            assert!(!s.is_empty());
        }
        // Spot check: all four tags must be distinct.
        let tags: std::collections::HashSet<_> =
            [Backend::GpuFull, Backend::GpuStrip, Backend::GpuStripPair, Backend::Cpu]
                .iter()
                .map(|b| b.tag())
                .collect();
        assert_eq!(tags.len(), 4);
    }

    #[test]
    fn backend_bench_set_get() {
        let mut b = BackendBench::default();
        assert_eq!(b.get(Backend::GpuFull), None);
        b.set(Backend::GpuFull, 2.71);
        assert_eq!(b.get(Backend::GpuFull), Some(2.71));
        b.set(Backend::GpuStripPair, 2.62);
        assert_eq!(b.get(Backend::GpuStripPair), Some(2.62));
        assert_eq!(b.get(Backend::GpuStrip), None);
        assert_eq!(b.get(Backend::Cpu), None);
    }

    #[test]
    fn backend_vram_set_get() {
        let mut v = BackendVram::default();
        assert_eq!(v.get(Backend::GpuFull), None);
        v.set(Backend::GpuFull, 3970);
        assert_eq!(v.get(Backend::GpuFull), Some(3970));
    }

    #[test]
    fn metric_profile_toml_round_trip() {
        // Build a populated profile, save through the cache helpers,
        // load back and check structural equality at second-resolution.
        let mut profile = fake_profile();
        let mut cvvdp = MetricProfile::default();
        let size_4k_px: u64 = 4096 * 4096;
        let size_2k_px: u64 = 2048 * 2048;
        let size_1k_px: u64 = 1024 * 1024;

        let mut bench_4k = BackendBench::default();
        bench_4k.set(Backend::GpuFull, 2.71);
        bench_4k.set(Backend::GpuStripPair, 2.62);
        let mut bench_2k = BackendBench::default();
        bench_2k.set(Backend::GpuFull, 3.10);
        let mut bench_1k = BackendBench::default();
        bench_1k.set(Backend::GpuFull, 5.34);

        cvvdp.ns_per_px_at.insert(size_4k_px, bench_4k);
        cvvdp.ns_per_px_at.insert(size_2k_px, bench_2k);
        cvvdp.ns_per_px_at.insert(size_1k_px, bench_1k);

        let mut vram_4k = BackendVram::default();
        vram_4k.set(Backend::GpuFull, 3970);
        vram_4k.set(Backend::GpuStripPair, 2272);
        cvvdp.vram_mib_at.insert(size_4k_px, vram_4k);

        cvvdp.last_measured = Some(SystemTime::now());
        cvvdp
            .cells_failed_oom
            .push((Backend::GpuFull, 4096u64 * 4096u64 * 4));

        profile.metrics.insert("cvvdp".into(), cvvdp);

        let dir = tempfile::tempdir().unwrap();
        let path = cache_file_path(dir.path(), &profile.machine_hash);
        save_profile(&path, &profile).expect("save");
        let loaded = load_cached_profile(&path).expect("load");

        let cv = loaded.metrics.get("cvvdp").expect("cvvdp metric");
        assert_eq!(cv.ns_per_px_at.len(), 3);
        let b4k = cv.ns_per_px_at.get(&size_4k_px).unwrap();
        assert_eq!(b4k.get(Backend::GpuFull), Some(2.71));
        assert_eq!(b4k.get(Backend::GpuStripPair), Some(2.62));
        assert_eq!(b4k.get(Backend::GpuStrip), None);
        let v4k = cv.vram_mib_at.get(&size_4k_px).unwrap();
        assert_eq!(v4k.get(Backend::GpuFull), Some(3970));
        assert_eq!(v4k.get(Backend::GpuStripPair), Some(2272));
        assert_eq!(cv.cells_failed_oom.len(), 1);
        let (oom_b, oom_px) = cv.cells_failed_oom[0];
        assert_eq!(oom_b, Backend::GpuFull);
        assert_eq!(oom_px, 4096u64 * 4096u64 * 4);
    }

    #[test]
    fn metric_profile_toml_emits_string_size_keys() {
        // Confirm the TOML serializer stringifies u64 keys (the format
        // mandated by the design doc — see ORCHESTRATOR_DESIGN.md).
        let mut profile = fake_profile();
        let mut cvvdp = MetricProfile::default();
        let size_1k_px: u64 = 1024 * 1024;
        let mut bench = BackendBench::default();
        bench.set(Backend::GpuFull, 5.34);
        cvvdp.ns_per_px_at.insert(size_1k_px, bench);
        let mut vram = BackendVram::default();
        vram.set(Backend::GpuFull, 385);
        cvvdp.vram_mib_at.insert(size_1k_px, vram);
        cvvdp.last_measured = Some(SystemTime::now());
        profile.metrics.insert("cvvdp".into(), cvvdp);
        let toml_text = toml::to_string_pretty(&profile).unwrap();
        // The key 1048576 (= 1024 * 1024) must appear somewhere in
        // the output. TOML's dotted-key format emits integer-looking
        // keys unquoted (`[metrics.cvvdp.ns_per_px_at.1048576]`); our
        // serde helper stringifies u64 -> "1048576" but the TOML
        // writer is free to drop the quotes since the lexer parses
        // unquoted integers + bare-keys identically. Either form is
        // a valid round-trip — assert just on the substring.
        assert!(
            toml_text.contains("1048576"),
            "expected stringified u64 key '1048576' somewhere in TOML:\n{toml_text}"
        );
        // And the metric tag heading must be present.
        assert!(toml_text.contains("[metrics.cvvdp"));
    }

    #[test]
    fn metric_profile_default_is_empty() {
        let p = MetricProfile::default();
        assert!(p.ns_per_px_at.is_empty());
        assert!(p.vram_mib_at.is_empty());
        assert!(p.cells_failed_oom.is_empty());
        assert!(p.last_measured.is_none());
    }

    #[test]
    #[cfg(not(feature = "bench"))]
    fn bench_report_empty_without_feature() {
        // Without the `bench` feature, run() returns an empty report —
        // this keeps the public API stable across feature mixes.
        // Gated to default builds because under `--features bench` this
        // would actually run the full 30-cell bench (>>60s).
        let plan = BenchPlan::default();
        let report = bench::run(&plan);
        assert!(report.metrics.is_empty());
    }

    #[test]
    fn bench_plan_default_round_trip_stable() {
        // Smoke check that BenchPlan::default() compiles and the
        // public field shape is what callers expect.
        let p = BenchPlan::default();
        assert_eq!(p.sizes, vec![1024u32, 2048, 4096]);
    }
}
