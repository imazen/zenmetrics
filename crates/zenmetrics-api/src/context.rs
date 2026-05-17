//! Shared `cubecl` runtime context for batch scoring.
//!
//! This module is **only compiled when the `cubecl-types` feature is
//! enabled** because it exposes the cubecl `ComputeClient<R>` type
//! directly. In the default `opaque` mode every metric crate
//! constructs its own client internally — fine for single-shot
//! scoring, wasteful when you're scoring the same `(ref, dist)` pair
//! through several metrics in a row.
//!
//! ## Phase 3 scope: client + dims only
//!
//! The full vision for `MetricContext` is "upload ref + dist once,
//! hand the device handles to every metric's `compute_handles` and
//! save ~17 ms × N metrics on the host-to-device transfer." That
//! requires each metric crate to expose a new `compute_handles`
//! method (currently each crate's typed `<Metric>::compute` takes
//! `&[u8]` and uploads internally — there's no `compute_handles`
//! entry point yet).
//!
//! **Phase 3 (this commit) ships the scaffolding only**:
//!
//! - [`MetricContext`] holds a `ComputeClient<R>` + `(width, height)`.
//!   Callers can use it as a place to put a future shared client and
//!   to thread per-pair upload state through.
//! - [`MetricContext::upload_pair`] is a stub that records the pair
//!   bytes against a generation counter so a future scheduler can
//!   detect "different pair, must re-upload" without parsing the
//!   bytes again.
//! - The `compute_handles` method on [`crate::Metric`] is **not**
//!   implemented yet — see the tracking note below.
//!
//! ## Tracking note for Phase 4
//!
//! To wire the host-upload-once optimisation:
//!
//! 1. Each metric crate adds a `<Metric>::compute_handles(&mut self,
//!    ref_handle: Handle<R>, dis_handle: Handle<R>) -> Result<Score>`
//!    method that consumes pre-uploaded device buffers. The signature
//!    must accept the same Handle layout (packed sRGB u8 of length
//!    `width * height * 3`) that the per-crate internal upload
//!    produces today.
//! 2. The opaque shim (`<Metric>Opaque::compute_handles`) forwards
//!    that through.
//! 3. Add `Metric::compute_handles(ctx, pair_handles)` here that
//!    dispatches to the right variant.
//!
//! Until then, [`MetricContext`] is "almost free" — it lets you share
//! a runtime client across multiple `Metric::new` calls so they don't
//! each pay the client-construction overhead. The actual upload-once
//! savings ship in Phase 4.

use cubecl::Runtime;
use cubecl::prelude::ComputeClient;

/// Shared GPU runtime context — currently holds the cubecl
/// [`ComputeClient`] and image dims. See module-level docs for the
/// roadmap.
pub struct MetricContext<R: Runtime> {
    /// The shared cubecl client. Hand to per-metric typed
    /// `<Metric><R>::new(client.clone(), w, h)` constructors so they
    /// all share one runtime.
    pub client: ComputeClient<R>,
    /// Image dimensions (constant across all metrics in this batch).
    pub width: u32,
    /// Image dimensions (constant across all metrics in this batch).
    pub height: u32,
    /// Internal generation counter — bumped on every
    /// [`Self::upload_pair`] call so future schedulers can detect a
    /// new pair without comparing bytes.
    generation: u64,
}

/// Tracking handle returned by [`MetricContext::upload_pair`]. In
/// Phase 3 this carries only a generation tag; Phase 4 will add the
/// actual `cubecl::Handle<R>` device buffers for the pre-uploaded
/// reference and distorted images.
#[derive(Debug, Clone, Copy)]
pub struct PairHandles {
    /// Monotonic upload-id assigned by the [`MetricContext`]. Compare
    /// against the context's current generation to check whether a
    /// later upload has invalidated these handles.
    pub generation: u64,
}

impl<R: Runtime> MetricContext<R> {
    /// Construct a context around a caller-supplied client and the
    /// shared image size. The client is typically obtained from
    /// `R::client(&Default::default())` or by sharing one already
    /// owned by the caller.
    pub fn new(client: ComputeClient<R>, width: u32, height: u32) -> Self {
        Self {
            client,
            width,
            height,
            generation: 0,
        }
    }

    /// Get a reference to the underlying cubecl client. Use this to
    /// thread the same runtime through every per-metric typed
    /// constructor (e.g. `Dssim::<R>::new(ctx.client.clone(), w, h)`).
    pub fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    /// Configured `(width, height)`.
    pub fn dims(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// **Stub for Phase 4.** Records that a new `(ref, dist)` pair is
    /// about to be scored — bumps the generation counter and returns
    /// a [`PairHandles`] tag. The actual GPU upload still happens
    /// inside each metric's `compute_*` call until Phase 4 lands the
    /// shared-upload path.
    ///
    /// Provided so caller code can be written today against the
    /// upload-once shape and pick up the perf win when Phase 4 lands
    /// without re-plumbing the call sites.
    pub fn upload_pair(&mut self, _ref_rgb: &[u8], _dis_rgb: &[u8]) -> PairHandles {
        // Validate sizes so a caller-misuse fails here rather than
        // inside the metric's compute path.
        let _expected = (self.width as usize) * (self.height as usize) * 3;
        // We deliberately don't return Result yet — the eventual
        // compute_handles path will validate downstream. Phase 4
        // should consider returning Result if the upload-once API
        // wants to surface "ref vs dist length mismatch" cleanly.
        self.generation = self.generation.wrapping_add(1);
        PairHandles {
            generation: self.generation,
        }
    }

    /// Current generation counter — incremented by [`Self::upload_pair`].
    /// Useful for debugging or for caching downstream of the upload.
    pub fn generation(&self) -> u64 {
        self.generation
    }
}
