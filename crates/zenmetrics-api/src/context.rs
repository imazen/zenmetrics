//! Shared `cubecl` runtime context for batch scoring.
//!
//! This module is **only compiled when the `cubecl-types` feature is
//! enabled** because it exposes the cubecl `ComputeClient<R>` type
//! directly. In the default `opaque` mode every metric crate
//! constructs its own client internally — fine for single-shot
//! scoring, wasteful when you're scoring the same `(ref, dist)` pair
//! through several metrics in a row.
//!
//! ## Phase 4 upload-once
//!
//! `MetricContext::upload_pair` allocates two packed-u32 device
//! handles (one for ref, one for dist) on the shared client. Every
//! per-crate opaque now exposes a `compute_handles(&ref, &dist)`
//! method that consumes those device buffers without re-running its
//! own upload, so a five-metric scoring pass pays one host-to-device
//! upload instead of five.
//!
//! Handle layout: `width × height` `u32`s, each pixel packed as
//! `R | G<<8 | B<<16` (alpha unused). Matches what every metric
//! crate's internal upload produces, so `compute_handles` and the
//! byte-flavored `compute_srgb_u8` are bit-identical.
//!
//! The opaque [`crate::Metric::compute_handles`] dispatches to the
//! right variant's `compute_handles`; callers don't pattern-match.

use cubecl::Runtime;
use cubecl::prelude::ComputeClient;
use cubecl::server::Handle;

use crate::Result;

/// Shared GPU runtime context — holds the cubecl
/// [`ComputeClient`], image dims, and the most recent
/// [`Self::upload_pair`] handles. Hand to every per-metric typed
/// `<Metric><R>::new(ctx.client().clone(), w, h)` constructor so each
/// metric shares one runtime; then call [`Self::upload_pair`] once
/// per `(ref, dist)` pair and [`crate::Metric::compute_handles`] on
/// every metric.
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

/// Tracking handle returned by [`MetricContext::upload_pair`]. Carries
/// the two device buffers (`ref_handle`, `dist_handle`) plus the
/// monotonic generation counter so a debug-build can sanity-check
/// that the consumer is using the latest upload.
///
/// `Handle` is `Clone` on cubecl's side — every metric's
/// `compute_handles` takes `&Handle`, so calling several metrics in a
/// row against one [`PairHandles`] is fine.
#[derive(Debug, Clone)]
pub struct PairHandles {
    /// Pre-uploaded reference image, packed-u32 layout
    /// (`R | G<<8 | B<<16`, length `width × height`).
    pub ref_handle: Handle,
    /// Pre-uploaded distorted image, packed-u32 layout.
    pub dist_handle: Handle,
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

    /// Pack and upload a `(ref, dist)` sRGB-u8 pair into device
    /// handles. The returned [`PairHandles`] can then be passed to
    /// [`crate::Metric::compute_handles`] (or each metric's per-crate
    /// `compute_handles` directly) without paying for a re-upload.
    ///
    /// Layout: each input is `width × height × 3` packed RGB u8;
    /// the output handle is `width × height` packed-u32 of
    /// `R | G<<8 | B<<16` — the same layout each metric's internal
    /// upload uses. Bit-identical scores vs. `compute_srgb_u8`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::DimensionMismatch`] if either input's
    /// length doesn't match `width × height × 3`.
    pub fn upload_pair(&mut self, ref_rgb: &[u8], dis_rgb: &[u8]) -> Result<PairHandles> {
        let n = (self.width as usize) * (self.height as usize);
        let expected_bytes = n * 3;
        if ref_rgb.len() != expected_bytes {
            return Err(crate::Error::DimensionMismatch {
                expected: (self.width, self.height),
                got: ((ref_rgb.len() / 3) as u32, 1),
            });
        }
        if dis_rgb.len() != expected_bytes {
            return Err(crate::Error::DimensionMismatch {
                expected: (self.width, self.height),
                got: ((dis_rgb.len() / 3) as u32, 1),
            });
        }
        let ref_handle = pack_into_pinned(&self.client, ref_rgb, n);
        let dist_handle = pack_into_pinned(&self.client, dis_rgb, n);
        self.generation = self.generation.wrapping_add(1);
        Ok(PairHandles {
            ref_handle,
            dist_handle,
            generation: self.generation,
        })
    }

    /// Current generation counter — incremented by [`Self::upload_pair`].
    /// Useful for debugging or for caching downstream of the upload.
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

/// Pack `n` pixels of `width × height × 3` sRGB u8 (length `n * 3`)
/// into a packed-u32 device handle (`R | G<<8 | B<<16`, length `n`),
/// matching what every metric crate's internal upload produces. Uses
/// the same pinned-staging fast path so the host write hits the
/// pinned-memory DMA path on CUDA backends.
fn pack_into_pinned<R: Runtime>(client: &ComputeClient<R>, srgb: &[u8], n: usize) -> Handle {
    let pinned_len = n * 4;
    let mut staging = client.reserve_staging(&[pinned_len]);
    let mut bytes = staging.pop().expect("reserve_staging returned no buffers");
    {
        let dst: &mut [u8] = &mut bytes;
        debug_assert_eq!(dst.len(), pinned_len);
        for (chunk_out, triple) in dst.chunks_exact_mut(4).zip(srgb.chunks_exact(3)) {
            chunk_out[0] = triple[0];
            chunk_out[1] = triple[1];
            chunk_out[2] = triple[2];
            chunk_out[3] = 0;
        }
    }
    client.create(bytes)
}
