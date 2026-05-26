//! Unified memory-mode API. See `butteraugli-gpu/src/memory_mode.rs`
//! for shared design rationale.
//!
//! cvvdp-gpu supports two memory modes:
//!
//! - **Full** — whole-image working set on device. Bit-stable with
//!   the host-scalar reference. Default; preferred when the image
//!   fits the VRAM cap.
//! - **Strip { h_body }** — **Mode E only** (ref-full + dist-strip
//!   cached-ref). Task #79 reintroduces a Strip variant that is
//!   **JOD-preserving**: the reference-side state stays at full
//!   image resolution on device (so per-band masking has the
//!   correct neighbour pixels at every level); the dist side walks
//!   the image in vertical strips. Per-band atomic-pool sums are
//!   associative across strips, so the final JOD equals Full-mode
//!   JOD within the documented Atomic<f32> reduction-order noise
//!   band.
//!
//! The earlier capped-pyramid Strip variant that lived here before
//! task #77 was rolled back because **capping the pyramid depth
//! changes the JOD value** at any k < 9. See `docs/STRIP_PROCESSING.md`
//! for the full rationale on what was rolled back vs what mode E
//! preserves.
//!
//! Strip mode is **only valid for the cached-ref code path**
//! ([`crate::pipeline::Cvvdp::warm_reference`] +
//! [`crate::pipeline::Cvvdp::compute_dkl_jod_with_warm_ref`] and the
//! umbrella `MetricCache` surface). One-shot scoring
//! ([`crate::pipeline::Cvvdp::score`]) is still Full-only because
//! its memory profile is the dist working set that mode E aims to
//! shrink anyway.
//!
//! `MemoryMode::Auto` picks Full when it fits the cap, else Strip
//! with a crate-default `h_body`. Callers can override `h_body`
//! explicitly via `MemoryMode::Strip { h_body: Some(N) }`.

fn env_cap_bytes() -> Option<usize> {
    std::env::var("ZENMETRICS_VRAM_CAP_BYTES")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
}

/// Cache for the live nvidia-smi probe result. Process-wide.
static LIVE_PROBE_CACHE: std::sync::OnceLock<Option<usize>> =
    std::sync::OnceLock::new();

/// Probe live free-VRAM. See `iwssim_gpu::memory_mode::live_vram_probe_bytes`.
pub fn live_vram_probe_bytes() -> Option<usize> {
    *LIVE_PROBE_CACHE.get_or_init(query_nvidia_smi_memory_free)
}

fn query_nvidia_smi_memory_free() -> Option<usize> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let mb: u64 = s.lines().next()?.trim().parse().ok()?;
    let bytes = (mb as usize).saturating_mul(1024 * 1024);
    Some(bytes.saturating_sub(bytes / 10))
}

/// Effective VRAM cap in bytes (task #51 cap policy):
/// 1. `ZENMETRICS_VRAM_CAP_BYTES` env var if set
/// 2. Live `nvidia-smi` probe (cached, 10% headroom)
/// 3. 8 GB default
pub fn vram_cap_bytes() -> usize {
    if let Some(cap) = env_cap_bytes() {
        return cap;
    }
    if let Some(probed) = live_vram_probe_bytes() {
        return probed;
    }
    8 * 1024 * 1024 * 1024
}

/// Crate-default Strip body height. Picked at the small side so the
/// dist-side strip working set stays modest even on tiny VRAM caps.
/// The Auto resolver can pick a larger value when the cap permits.
///
/// Must be a multiple of `2^(MAX_LEVELS - 1) = 256` so the per-level
/// halving in the Weber pyramid doesn't drift through the strip
/// boundary. 512 = 2 * 256 — small enough to fit in 1 GB at 12 MP
/// (`512 × 4096 × 9 levels × N_CHANNELS × 4 bytes ≈ 224 MB` for the
/// dist-pyramid working set, plus halo).
pub const STRIP_H_BODY_DEFAULT: u32 = 512;

/// Strip body height alignment factor. Multiples of `2^(MAX_LEVELS - 1)`
/// — body and halo must both be divisible by this so the strip body
/// region maps cleanly through every Weber pyramid level (ceil-div
/// halving truncates exact body boundaries when they're not aligned).
pub const STRIP_ALIGN: u32 = 1 << (crate::MAX_LEVELS as u32 - 1);

/// How the GPU pipeline should partition its working set.
///
/// cvvdp-gpu supports two variants:
///
/// - [`Self::Full`] — whole-image working set. Default.
/// - [`Self::Strip`] — Mode E (ref-full + dist-strip cached-ref).
///   Only valid for the cached-ref code path
///   ([`crate::pipeline::Cvvdp::warm_reference`] +
///   [`crate::pipeline::Cvvdp::compute_dkl_jod_with_warm_ref`]).
///
/// See module-level docs for the JOD-preservation rationale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryMode {
    /// Pick Full when it fits the cap, else Strip with a crate-
    /// default `h_body`. See [`vram_cap_bytes()`] for the cap source.
    Auto,
    /// Allocate the whole-image working set.
    Full,
    /// Mode E strip walker: full ref state on device + per-strip dist
    /// working set. `h_body` is the dist-side strip body height in
    /// rows. `None` lets the [`Self::Auto`] policy pick a default
    /// (= [`STRIP_H_BODY_DEFAULT`]).
    ///
    /// `h_body` must be a positive multiple of [`STRIP_ALIGN`].
    Strip {
        /// Dist-side strip body height in rows. `None` → crate-default.
        h_body: Option<u32>,
    },
}

/// Outcome of resolving [`MemoryMode::Auto`]. cvvdp-gpu can resolve
/// to either Full or Strip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedMode {
    /// Whole-image allocation.
    Full,
    /// Mode E with the picked `h_body`.
    Strip {
        /// Resolved dist-side strip body height in rows.
        h_body: u32,
    },
}

/// Auto-resolve policy: prefer Full when it fits the cap, else Strip
/// with a crate-default `h_body`. See module-level docs.
pub fn resolve_auto(
    width: u32,
    height: u32,
    cap: usize,
) -> crate::Result<ResolvedMode> {
    let Some(full_bytes) = crate::pipeline::estimate_gpu_memory_bytes(width, height) else {
        // Below the pyramid minimum — `Cvvdp::new` would reject too;
        // surface a TooBigForFull-shaped error with `needed: 0` to
        // signal "image too small to allocate at all".
        return Err(crate::Error::TooBigForFull { needed: 0, cap });
    };
    if full_bytes <= cap {
        return Ok(ResolvedMode::Full);
    }
    // Try Strip with the crate-default h_body. The strip-mode
    // estimator returns the ref-full footprint + one-strip dist
    // working set; if even that doesn't fit, surface TooBigForFull.
    let strip_bytes =
        crate::pipeline::estimate_gpu_memory_bytes_strip(width, height, STRIP_H_BODY_DEFAULT)
            .unwrap_or(usize::MAX);
    if strip_bytes <= cap {
        return Ok(ResolvedMode::Strip {
            h_body: STRIP_H_BODY_DEFAULT,
        });
    }
    Err(crate::Error::TooBigForFull {
        needed: strip_bytes,
        cap,
    })
}

/// Unified-API wrapper around
/// [`crate::pipeline::estimate_gpu_memory_bytes`]. Returns a `usize`
/// (matching the other metric crates' signature); below-pyramid-minimum
/// images surface `usize::MAX` so the resolver classifies them as
/// "too big for the cap". cvvdp's own pipeline-level helper still
/// returns `Option<usize>` for callers that prefer the explicit
/// invalid-size encoding.
#[must_use]
pub fn estimate_gpu_memory_bytes_usize(width: u32, height: u32) -> usize {
    crate::pipeline::estimate_gpu_memory_bytes(width, height).unwrap_or(usize::MAX)
}

/// Unified-API wrapper that selects between Full / Strip estimates
/// based on the supplied [`MemoryMode`]. For [`MemoryMode::Auto`] this
/// returns the Full estimate (mirroring the umbrella `Auto`-prefers-
/// Full behavior at small sizes); the resolver consults
/// [`resolve_auto`] for the actual pick.
#[must_use]
pub fn estimate_gpu_memory_bytes_for_mode(width: u32, height: u32, mode: MemoryMode) -> usize {
    match mode {
        MemoryMode::Full | MemoryMode::Auto => {
            crate::pipeline::estimate_gpu_memory_bytes(width, height).unwrap_or(usize::MAX)
        }
        MemoryMode::Strip { h_body } => {
            let body = h_body.unwrap_or(STRIP_H_BODY_DEFAULT);
            crate::pipeline::estimate_gpu_memory_bytes_strip(width, height, body)
                .unwrap_or(usize::MAX)
        }
    }
}
