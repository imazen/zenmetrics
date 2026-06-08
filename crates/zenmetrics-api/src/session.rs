//! [`MetricSession`] — an opt-in, isolated GPU execution context whose
//! `Drop` reclaims **exactly its own VRAM** back to the driver,
//! ironclad-correctly.
//!
//! See `docs/VRAM_LIFECYCLE_DESIGN_2026-05-30.md` (Option B) and issue
//! imazen/zenmetrics#17 for the full rationale and the measured CUDA +
//! wgpu isolation spikes. The short version:
//!
//! - Dropping a [`crate::Metric`] returns its device buffers to
//!   cubecl's *shared, thread-local* pool free-list — not to the
//!   driver. The only reclaim path on that shared stream
//!   ([`crate::reclaim_pooled_vram`]) is best-effort and thread-scoped,
//!   and it can't free pages a co-resident sibling metric still
//!   partially occupies.
//! - A [`MetricSession`] instead owns a **private cubecl stream + its
//!   own memory pool**. Every metric built through the session
//!   allocates on that stream; nothing else does. So when the session
//!   drops, its metric handles are the *only* occupants of its pool
//!   pages — `memory_cleanup()` + `sync()` on the session's stream
//!   returns all of it to the driver, independent of every other
//!   session, from any thread. The CUDA spike measured exactly this
//!   (stream 101 freed 2272 MiB while stream 202 stayed resident).
//!
//! ## Opaque surface
//!
//! [`MetricSession`] and [`SessionMetric`] hide all cubecl types behind
//! [`Backend`] / [`MetricKind`] / [`MetricParams`], so they live in the
//! **default** API surface — no `cubecl-types` feature required to
//! call them. The stream binding itself (a `cubecl` `unsafe set_stream`
//! call) is confined to each metric crate's internal `session` module;
//! `zenmetrics-api` stays `#![forbid(unsafe_code)]`.
//!
//! ## What is reserved vs. freed
//!
//! Reserved for the process lifetime (NOT freed by dropping a session):
//! the GPU device context (~181 ms one-time init), the compiled-kernel
//! (PTX) cache, and the stream *shell* (a slot in cubecl's fixed
//! `max_streams` table — bytes, not GiBs; cubecl has no remove-stream
//! API). Freed by dropping a session: the session's entire device
//! working-set pool (the GiBs). See the crate-level "VRAM lifecycle"
//! docs and `docs/VRAM_LIFECYCLE_DESIGN_2026-05-30.md` §3.

use crate::Result;
use crate::error::Error;
use crate::memory_mode::MemoryMode;
use crate::metric::{Backend, MetricKind, MetricParams};

use std::sync::Mutex;

/// Maximum number of simultaneously-live isolated sessions per backend.
///
/// cubecl stores streams in a fixed table indexed by
/// `stream_id.value % max_streams`, default `max_streams = 128`
/// (cubecl `config/streaming.rs`, `stream/base.rs`). Two distinct
/// `StreamId`s whose `value` collide modulo this bound alias the same
/// physical stream + pool — which would re-introduce the shared-pool
/// reclaim hazard a session exists to eliminate. So the allocator hands
/// out at most this many distinct slots; the 129th
/// [`MetricSession::acquire`] returns [`Error::TooManyContexts`] rather
/// than silently alias.
pub const MAX_SESSIONS_PER_BACKEND: u32 = 128;

/// Base offset added to a slot index to form the cubecl
/// `StreamId.value`. A large multiple of [`MAX_SESSIONS_PER_BACKEND`]
/// so that `value % MAX_SESSIONS_PER_BACKEND == slot` (each slot maps
/// to a distinct physical stream, deterministically) while keeping the
/// `value`s well clear of the small ids cubecl's per-thread
/// `StreamId::current()` allocator hands out from its global counter
/// (so a session's stream doesn't collide with an ambient thread-local
/// stream's pool that happens to share a low `value`).
///
/// NB: there are only `MAX_SESSIONS_PER_BACKEND` *physical* pool slots
/// in total, shared with thread-local + GC streams. Reserving slots for
/// ambient thread-locals is not possible (their `value`s aren't known
/// here); the offset minimizes the practical collision risk by avoiding
/// the low range. The orchestrator's warm-set is far below 128, so this
/// is comfortable in practice.
const STREAM_VALUE_BASE: u64 = (MAX_SESSIONS_PER_BACKEND as u64) * 0x1_0000;

/// Per-backend 128-bit allocator bitmask. Bit `i` set ⇒ slot `i` is
/// currently held by a live session. One mask per backend so sessions
/// on different backends never contend for the same slot space.
struct SlotAllocator {
    cuda: Mutex<u128>,
    wgpu: Mutex<u128>,
    hip: Mutex<u128>,
    /// cubecl-cpu reference backend (`Backend::CubeclCpu`).
    cpu: Mutex<u128>,
    /// optimized native-CPU backend (`Backend::Cpu`, task #159 phase 2).
    /// Separate slot space from `cpu` — they are distinct backends.
    cpu_native: Mutex<u128>,
}

impl SlotAllocator {
    const fn new() -> Self {
        Self {
            cuda: Mutex::new(0),
            wgpu: Mutex::new(0),
            hip: Mutex::new(0),
            cpu: Mutex::new(0),
            cpu_native: Mutex::new(0),
        }
    }

    fn mask_for(&self, backend: Backend) -> &Mutex<u128> {
        match backend {
            // `MetricSession::acquire` resolves `Auto` before any slot
            // work, so this arm is normally unreachable — resolve again
            // defensively rather than panic, since `resolve` never
            // returns `Auto`.
            Backend::Auto => self.mask_for(backend.resolve()),
            Backend::Cuda => &self.cuda,
            Backend::Wgpu => &self.wgpu,
            Backend::Hip => &self.hip,
            Backend::Cpu => &self.cpu_native,
            Backend::CubeclCpu => &self.cpu,
        }
    }

    /// Claim the lowest free slot for `backend`, returning its index, or
    /// `None` if all [`MAX_SESSIONS_PER_BACKEND`] slots are held.
    fn claim(&self, backend: Backend) -> Option<u32> {
        let m = self.mask_for(backend);
        let mut guard = m.lock().unwrap_or_else(|p| p.into_inner());
        for i in 0..MAX_SESSIONS_PER_BACKEND {
            let bit = 1u128 << i;
            if (*guard & bit) == 0 {
                *guard |= bit;
                return Some(i);
            }
        }
        None
    }

    /// Release a previously-claimed slot for `backend`.
    fn release(&self, backend: Backend, slot: u32) {
        debug_assert!(slot < MAX_SESSIONS_PER_BACKEND);
        let m = self.mask_for(backend);
        let mut guard = m.lock().unwrap_or_else(|p| p.into_inner());
        *guard &= !(1u128 << slot);
    }

    /// Number of slots currently held for `backend` (live session
    /// count). Test/diagnostic helper.
    fn live_count(&self, backend: Backend) -> u32 {
        let m = self.mask_for(backend);
        let guard = m.lock().unwrap_or_else(|p| p.into_inner());
        guard.count_ones()
    }
}

static ALLOCATOR: SlotAllocator = SlotAllocator::new();

/// An isolated GPU execution context. Owns a private cubecl stream and
/// its memory pool. Every metric built through this session
/// ([`Self::metric`]) allocates on that stream; dropping the session
/// reclaims **exactly** this session's device VRAM back to the driver,
/// independent of every other session, from any thread.
///
/// `MetricSession: Send + Sync` — it is plain slot/stream bookkeeping
/// (no scorer state of its own), so it may be moved or dropped from a
/// thread other than the one that built its metrics: the explicit stream
/// id overrides cubecl's thread-local stream selection (measured
/// cross-thread in the CUDA spike). The single-threaded scorer state
/// lives in [`SessionMetric`] / [`OwnedSessionMetric`], not here. A
/// compile-time assertion at the bottom of this module enforces the
/// `Send + Sync` contract (and `Send` on the metric handles — `!Send`
/// would break the orchestrator's hand-a-warm-metric-to-a-lane dispatch).
///
/// # Cap
///
/// At most [`MAX_SESSIONS_PER_BACKEND`] (128) sessions may be live per
/// backend at once; [`Self::acquire`] returns [`Error::TooManyContexts`]
/// beyond that rather than silently aliasing a stream.
///
/// # Stream-aliasing caveat
///
/// A session's stream id is `STREAM_VALUE_BASE + slot`, chosen so live
/// sessions never collide with *each other*. But cubecl indexes its stream
/// table by `value % 128`, and we cannot reserve slots against *ambient*
/// thread-local streams the process may already be using elsewhere — so a
/// session's `value % 128` could in principle collide with an unrelated
/// ambient cubecl stream in the same process. Exposure is low in practice
/// (it scales with the count of live sessions, far below 128 at
/// orchestrator lane scale, and the orchestrator's default path uses no
/// sessions at all), but a consumer that holds many sessions alongside
/// other direct cubecl usage on the same device should be aware. This is a
/// documented caveat, not a closed hazard.
///
/// # Example
///
/// ```no_run
/// use zenmetrics_api::{Backend, MetricKind, MetricParams, MetricSession};
///
/// let ctx = MetricSession::acquire(Backend::Cuda)?;
/// let mut m = ctx.metric(
///     MetricKind::Cvvdp,
///     256,
///     256,
///     MetricParams::default_for(MetricKind::Cvvdp),
/// )?;
/// m.set_reference_srgb_u8(&vec![128u8; 256 * 256 * 3])?;
/// let s = m.score_with_warm_ref(&vec![100u8; 256 * 256 * 3])?;
/// println!("{} = {:.4}", s.metric_name, s.value);
/// drop(m);
/// drop(ctx); // memory_cleanup + sync on the private stream → driver
/// # Ok::<(), zenmetrics_api::Error>(())
/// ```
pub struct MetricSession {
    backend: Backend,
    /// Slot index in the per-backend allocator (0..128).
    slot: u32,
    /// The cubecl `StreamId.value` this session's metrics allocate on.
    stream_value: u64,
    /// When `true`, `Drop` skips reclaim and does NOT recycle the slot
    /// (set by [`Self::leak`]). The pool stays resident and the slot
    /// keeps counting against the cap.
    leaked: bool,
}

impl MetricSession {
    /// Acquire a fresh isolated session on `backend`.
    ///
    /// Allocates a collision-free slot from the process-global
    /// per-backend [`MAX_SESSIONS_PER_BACKEND`]-slot allocator and binds
    /// this session's metrics to a private cubecl stream derived from
    /// that slot.
    ///
    /// # Errors
    ///
    /// - [`Error::BackendNotEnabled`] if `backend`'s Cargo feature is
    ///   disabled in this build.
    /// - [`Error::TooManyContexts`] if [`MAX_SESSIONS_PER_BACKEND`]
    ///   sessions are already live on `backend` (refusing avoids silent
    ///   stream aliasing → the shared-pool reclaim hazard).
    pub fn acquire(backend: Backend) -> Result<Self> {
        // Resolve `Backend::Auto` to a concrete backend up front so the
        // session's whole lifecycle (slot allocation, error tags, stream
        // cleanup) operates on a real backend rather than the `Auto`
        // request. Non-`Auto` backends pass through unchanged.
        let backend = backend.resolve();
        // Surface a disabled backend before touching the allocator so
        // the slot isn't consumed by a session that can't build a
        // metric anyway. We probe via the per-metric backend mapping —
        // if NO enabled metric supports this backend, it's not usable.
        if !backend_is_usable(backend) {
            return Err(Error::BackendNotEnabled {
                backend: backend.tag(),
            });
        }
        let slot = ALLOCATOR.claim(backend).ok_or(Error::TooManyContexts {
            backend: backend.tag(),
        })?;
        Ok(Self {
            backend,
            slot,
            stream_value: STREAM_VALUE_BASE + slot as u64,
            leaked: false,
        })
    }

    /// The [`Backend`] this session is bound to.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Build a metric on this session's isolated stream. The returned
    /// [`SessionMetric`] borrows the session, so the borrow checker
    /// forbids it from outliving the session (which would leave a live
    /// binding into a stream the session's `Drop` is about to clean —
    /// the use-after-cleanup hazard).
    ///
    /// Mirrors [`crate::Metric::new`] but pins the metric to this
    /// session's stream. Uses [`MemoryMode::Auto`].
    ///
    /// # Errors
    ///
    /// - [`Error::MetricNotEnabled`] if `kind`'s Cargo feature is
    ///   disabled in this build.
    /// - [`Error::BackendNotEnabled`] if this session's backend isn't
    ///   supported by `kind`'s metric crate.
    /// - [`Error::Metric`] if the underlying constructor fails.
    ///
    /// # Panics
    ///
    /// Panics if `params` does not match `kind` (same contract as
    /// [`crate::Metric::new`]).
    pub fn metric(
        &self,
        kind: MetricKind,
        width: u32,
        height: u32,
        params: MetricParams,
    ) -> Result<SessionMetric<'_>> {
        self.metric_with_memory_mode(kind, width, height, params, MemoryMode::Auto)
    }

    /// [`MemoryMode`]-explicit variant of [`Self::metric`].
    ///
    /// # Errors
    ///
    /// Same as [`Self::metric`], plus per-crate [`Error::Metric`] when
    /// the requested mode isn't supported for that metric.
    #[allow(unused_variables)]
    pub fn metric_with_memory_mode(
        &self,
        kind: MetricKind,
        width: u32,
        height: u32,
        params: MetricParams,
        mode: MemoryMode,
    ) -> Result<SessionMetric<'_>> {
        let scorer = build_session_scorer(
            self.backend,
            self.stream_value,
            kind,
            width,
            height,
            params,
            mode,
        )?;
        Ok(SessionMetric {
            scorer,
            _session: core::marker::PhantomData,
        })
    }

    /// Consume this session into an **owned** [`OwnedSessionMetric`] — a
    /// single warm metric welded to the session's private stream, with
    /// no borrow leash. Use this (over [`Self::metric`]) when the metric
    /// must outlive the lexical scope that built it — e.g. a warm pool
    /// entry stored in a map. Uses [`MemoryMode::Auto`].
    ///
    /// One warm metric per isolated stream is the clean reclaim model:
    /// the returned bundle owns both the scorer and the session, so
    /// dropping it reclaims **exactly** this session's VRAM (see
    /// [`OwnedSessionMetric`] for the drop-order soundness argument).
    ///
    /// # Errors
    ///
    /// Same as [`Self::metric`].
    ///
    /// # Panics
    ///
    /// Panics if `params` does not match `kind` (same contract as
    /// [`Self::metric`]).
    pub fn into_metric(
        self,
        kind: MetricKind,
        width: u32,
        height: u32,
        params: MetricParams,
    ) -> Result<OwnedSessionMetric> {
        self.into_metric_with_memory_mode(kind, width, height, params, MemoryMode::Auto)
    }

    /// [`MemoryMode`]-explicit variant of [`Self::into_metric`].
    ///
    /// # Errors
    ///
    /// Same as [`Self::into_metric`], plus per-crate [`Error::Metric`]
    /// when the requested mode isn't supported for that metric.
    pub fn into_metric_with_memory_mode(
        self,
        kind: MetricKind,
        width: u32,
        height: u32,
        params: MetricParams,
        mode: MemoryMode,
    ) -> Result<OwnedSessionMetric> {
        // Build the scorer BEFORE moving `self` into the bundle so a
        // construction failure drops the session (→ reclaim of any
        // partial pool) without leaving a half-built owned metric.
        let scorer = build_session_scorer(
            self.backend,
            self.stream_value,
            kind,
            width,
            height,
            params,
            mode,
        )?;
        Ok(OwnedSessionMetric {
            scorer,
            session: self,
        })
    }

    /// Reclaim this session's pooled VRAM to the driver **without**
    /// dropping the session (an idle hook — e.g. between scoring
    /// batches when no [`SessionMetric`] is alive).
    ///
    /// Safe only when no [`SessionMetric`] on this session is currently
    /// alive (all handles dropped): a live binding into a relocated
    /// page would panic the cubecl allocator on its next dispatch. The
    /// borrow checker enforces this — `reclaim` takes `&self`, and a
    /// live `SessionMetric` borrows `&self`, but the cleanup runs on a
    /// *fresh* client bound to this session's stream, so there is no
    /// aliasing of an in-flight handle. Prefer calling it only when the
    /// session is idle.
    pub fn reclaim(&self) {
        cleanup_session_stream(self.backend, self.stream_value);
    }

    /// Opt-out: consume the session but DO NOT reclaim — leave its pool
    /// resident (e.g. handing the warm pool to a successor, or a
    /// deliberate leak in a short-lived process). The stream slot is
    /// **not** recycled (its pool is non-empty), so it keeps counting
    /// against the [`MAX_SESSIONS_PER_BACKEND`] cap until the process
    /// exits.
    pub fn leak(mut self) {
        self.leaked = true;
        // `self` drops here; Drop sees `leaked == true` and skips both
        // cleanup and slot recycling.
    }

    /// Current number of live sessions on `backend` (slots held in the
    /// process-global allocator). Diagnostic / test helper.
    pub fn live_count(backend: Backend) -> u32 {
        ALLOCATOR.live_count(backend)
    }

    /// The cubecl `StreamId.value` this session's metrics allocate on.
    /// `#[doc(hidden)]` — exposed only so the VRAM-isolation integration
    /// test can probe this exact session's pool (`memory_usage`) without
    /// guessing the slot assignment. Not a supported API.
    #[doc(hidden)]
    pub fn __stream_value(&self) -> u64 {
        self.stream_value
    }
}

impl Drop for MetricSession {
    /// Reclaim exactly this session's VRAM: any [`SessionMetric`]
    /// borrowing the session has already been dropped (the borrow
    /// checker guarantees it), so the session's metric handles are gone
    /// and its pool pages are fully-free. Then `memory_cleanup()` +
    /// `sync()` run on this session's explicit stream (→ driver) and
    /// the slot is recycled. Ironclad because nothing else ever
    /// allocated on this stream.
    ///
    /// Skipped entirely when [`Self::leak`] consumed the session.
    fn drop(&mut self) {
        if self.leaked {
            return;
        }
        cleanup_session_stream(self.backend, self.stream_value);
        ALLOCATOR.release(self.backend, self.slot);
    }
}

/// The borrowed scorer handle returned by [`MetricSession::metric`].
///
/// A distinct type that borrows `&'ctx MetricSession`, so the borrow
/// checker forbids it from outliving the session. Forwards the scoring
/// surface of [`crate::Metric`] (the scorer it wraps is built bound to
/// the session's private stream).
///
/// Holding a `SessionMetric` keeps the session's warm working set
/// resident; that is the **opt-out / warm-batch** path — keep the
/// session (and a metric on it) alive across many `(ref, dist)` pairs
/// to avoid per-score reclaim, then drop the session to reclaim.
///
/// # The borrow is compiler-enforced
///
/// A `SessionMetric` **cannot outlive its `MetricSession`**. If it
/// could, the session's `Drop` would run `memory_cleanup()` on the
/// session's stream while a live binding still pointed into it — the
/// use-after-cleanup panic the design exists to prevent. The borrow
/// makes that a compile error:
///
/// ```compile_fail
/// use zenmetrics_api::{Backend, MetricKind, MetricParams, MetricSession};
///
/// let ctx = MetricSession::acquire(Backend::Cuda).unwrap();
/// let m = ctx.metric(
///     MetricKind::Cvvdp, 64, 64,
///     MetricParams::default_for(MetricKind::Cvvdp),
/// ).unwrap();
/// drop(ctx);          // session gone...
/// let _ = m.dims();   // ...but `m` still borrows it → E0505: cannot move out of `ctx`
/// ```
///
/// Likewise a `SessionMetric` cannot be returned past the scope that
/// owns its session:
///
/// ```compile_fail
/// use zenmetrics_api::{Backend, MetricKind, MetricParams, MetricSession, SessionMetric};
///
/// fn escape<'a>() -> SessionMetric<'a> {
///     let ctx = MetricSession::acquire(Backend::Cuda).unwrap();
///     ctx.metric(
///         MetricKind::Cvvdp, 64, 64,
///         MetricParams::default_for(MetricKind::Cvvdp),
///     ).unwrap() // E0515: returns a value referencing local `ctx`
/// }
/// ```
pub struct SessionMetric<'ctx> {
    scorer: crate::metric::Metric,
    _session: core::marker::PhantomData<&'ctx MetricSession>,
}

/// Generate the shared scoring-surface forwarders on a type whose
/// `self.scorer` field is a [`crate::metric::Metric`]. Both
/// [`SessionMetric`] and [`OwnedSessionMetric`] expose the identical
/// surface; rather than copy-paste the bodies (an API eyesore the user
/// dislikes), this macro emits them once per type. Each forwarder is a
/// thin delegate to the inner [`crate::Metric`] — the scorer the type
/// owns, built bound to a session's private stream.
macro_rules! impl_session_scoring_surface {
    ($ty:ty) => {
        impl $ty {
            /// The [`MetricKind`] this scorer dispatches.
            pub fn kind(&self) -> MetricKind {
                self.scorer.kind()
            }

            /// The configured `(width, height)`.
            pub fn dims(&self) -> (u32, u32) {
                self.scorer.dims()
            }

            /// Score one reference / distorted pair of packed sRGB
            /// buffers. See [`crate::Metric::compute_srgb_u8`].
            pub fn score(&mut self, r: &[u8], d: &[u8]) -> Result<crate::Score> {
                self.scorer.compute_srgb_u8(r, d)
            }

            /// Cache the reference image's metric-side state on this
            /// session's stream. See
            /// [`crate::Metric::set_reference_srgb_u8`].
            pub fn set_reference_srgb_u8(&mut self, r: &[u8]) -> Result<()> {
                self.scorer.set_reference_srgb_u8(r)
            }

            /// Score a distorted candidate against the cached reference.
            /// See [`crate::Metric::compute_with_reference_srgb_u8`].
            pub fn score_with_warm_ref(&mut self, d: &[u8]) -> Result<crate::Score> {
                self.scorer.compute_with_reference_srgb_u8(d)
            }

            /// Drop cached reference state. See
            /// [`crate::Metric::clear_reference`].
            pub fn clear_reference(&mut self) {
                self.scorer.clear_reference();
            }

            /// Returns `true` if a cached reference is currently valid.
            /// See [`crate::Metric::has_reference`].
            pub fn has_reference(&self) -> bool {
                self.scorer.has_reference()
            }

            /// Score one reference / distorted pair from
            /// [`zenpixels::PixelSlice`] inputs. See
            /// [`crate::Metric::compute_pixels`].
            #[cfg(feature = "pixels")]
            pub fn score_pixels(
                &mut self,
                r: zenpixels::PixelSlice<'_>,
                d: zenpixels::PixelSlice<'_>,
            ) -> Result<crate::Score> {
                self.scorer.compute_pixels(r, d)
            }
        }
    };
}

impl_session_scoring_surface!(SessionMetric<'_>);

/// An **owned** isolated-stream metric: bundles a warm scorer with the
/// [`MetricSession`] whose private cubecl stream it allocates on, with
/// no borrow leash. Built via [`MetricSession::into_metric`].
///
/// Unlike [`SessionMetric`] (which borrows `&'ctx MetricSession`),
/// `OwnedSessionMetric` owns its session outright, so it can be stored
/// past the scope that built it — e.g. as an entry in a warm session
/// pool keyed by `(metric, dims, params, ref)`. Dropping it reclaims
/// **exactly** this session's VRAM back to the driver, independent of
/// every other session, from any thread (the session's `Drop` runs
/// `memory_cleanup()` + `sync()` on its private stream).
///
/// # Drop order is the soundness lever
///
/// The fields are declared **scorer first, session second**. Rust drops
/// struct fields in declaration order, so on drop the `scorer`'s cubecl
/// device handles are returned to the pool's free-list *before*
/// [`MetricSession`]'s `Drop` runs `memory_cleanup()` + `sync()` on the
/// stream. That ordering is what makes the owned shape ironclad: at the
/// moment cleanup runs, **no live handle points into the stream's pool**
/// — every page is fully-free — so cleanup returns the entire pool to
/// the driver and there is no use-after-cleanup hazard (the exact hazard
/// the borrowed `'ctx` leash guarded against by construction). Do **not**
/// reorder the fields, and do **not** add a manual `Drop` that moves
/// fields out: the field-order default drop + [`MetricSession`]'s
/// existing `Drop` is correct and sufficient.
///
/// # No in-place reclaim
///
/// [`MetricSession::reclaim`] is intentionally **not** forwarded here.
/// Reclaiming in place would run `memory_cleanup()` on the stream while
/// the welded scorer's handles are still live — exactly the
/// use-after-cleanup hazard the drop ordering above avoids. For an
/// `OwnedSessionMetric`, **eviction is a full drop** (which reclaims
/// correctly because the scorer drops first). Use [`Self::leak`] only to
/// *skip* reclaim for a short-lived process.
pub struct OwnedSessionMetric {
    // FIELD ORDER IS LOAD-BEARING — see the type docs. `scorer` MUST
    // come before `session` so the scorer's device handles drop to the
    // cubecl free-list before the session's Drop cleans the stream.
    scorer: crate::metric::Metric,
    session: MetricSession,
}

impl_session_scoring_surface!(OwnedSessionMetric);

impl OwnedSessionMetric {
    /// The [`Backend`] this metric's session is bound to.
    pub fn backend(&self) -> Backend {
        self.session.backend()
    }

    /// Consume the bundle but DO NOT reclaim — leave the session's pool
    /// resident (a deliberate leak for a short-lived process where the
    /// reclaim sync is pure overhead before exit). The scorer is still
    /// dropped (its handles return to the pool free-list); only the
    /// session's `memory_cleanup()` + `sync()` and slot recycling are
    /// skipped. The slot keeps counting against
    /// [`MAX_SESSIONS_PER_BACKEND`] until the process exits.
    pub fn leak(self) {
        // Move the scorer and session out, drop the scorer (handles →
        // free-list), then `MetricSession::leak` consumes the session
        // without reclaim. Destructuring is sound here: the scorer drops
        // first (no live handle), exactly as the field-order Drop would
        // sequence it.
        let OwnedSessionMetric { scorer, session } = self;
        drop(scorer);
        session.leak();
    }

    /// The cubecl `StreamId.value` this metric's session allocates on.
    /// `#[doc(hidden)]` — exposed only so the pool VRAM-isolation
    /// integration test can probe this exact session's pool
    /// (`memory_usage`) without guessing the slot assignment. Not a
    /// supported API.
    #[doc(hidden)]
    pub fn __stream_value(&self) -> u64 {
        self.session.__stream_value()
    }
}

// ---------------------------------------------------------------
// Backend usability probe + per-crate stream-bound construction.
//
// These bridge the opaque `MetricSession` to each metric crate's
// internal `session` plumbing (the `unsafe set_stream` lives there,
// keeping this crate `#![forbid(unsafe_code)]`). Only cvvdp is wired
// end-to-end in the foundation; the other five surface a clear
// `Error::Metric { message: "MetricSession not yet wired ..." }` so a
// caller can detect-and-fallback to a plain `Metric` on the shared
// stream. Replicating the hook per crate is mechanical (#17 follow-up).
// ---------------------------------------------------------------

/// True if at least one enabled metric crate can build on `backend`.
/// Used by [`MetricSession::acquire`] to fail fast on a disabled
/// backend before consuming an allocator slot.
#[allow(unused_variables)]
fn backend_is_usable(backend: Backend) -> bool {
    #[cfg(feature = "cvvdp")]
    {
        if crate::metric::cvvdp_backend(backend).is_ok() {
            return true;
        }
    }
    #[cfg(feature = "butter")]
    {
        if crate::metric::butter_backend(backend).is_ok() {
            return true;
        }
    }
    #[cfg(feature = "ssim2")]
    {
        if crate::metric::ssim2_backend(backend).is_ok() {
            return true;
        }
    }
    #[cfg(feature = "dssim")]
    {
        if crate::metric::dssim_backend(backend).is_ok() {
            return true;
        }
    }
    #[cfg(feature = "iwssim")]
    {
        if crate::metric::iwssim_backend(backend).is_ok() {
            return true;
        }
    }
    #[cfg(feature = "zensim")]
    {
        if crate::metric::zensim_backend(backend).is_ok() {
            return true;
        }
    }
    false
}

/// Build a stream-bound scorer for `kind` on `backend`'s stream
/// `stream_value`, wrapped in the umbrella [`crate::Metric`] enum.
///
/// The no-geometry metrics (ssim2/butter/dssim/iwssim/zensim) share the
/// uniform `session::new_opaque_on_stream(backend, stream, w, h, params,
/// mode)` shape; cvvdp additionally takes a `DisplayGeometry`. The arms
/// are written out (rather than macro-generated) because `macro_rules!`
/// cannot expand to a match arm in stable Rust.
#[allow(unused_variables)]
fn build_session_scorer(
    backend: Backend,
    stream_value: u64,
    kind: MetricKind,
    width: u32,
    height: u32,
    params: MetricParams,
    mode: MemoryMode,
) -> Result<crate::metric::Metric> {
    match kind {
        #[cfg(all(
            feature = "cvvdp",
            any(
                feature = "cuda",
                feature = "wgpu",
                feature = "hip",
                feature = "cpu",
                feature = "cubecl-types"
            )
        ))]
        MetricKind::Cvvdp => {
            let p = match params {
                MetricParams::Cvvdp(p) => p,
                _ => panic!("MetricParams variant mismatch (expected Cvvdp)"),
            };
            let b = crate::metric::cvvdp_backend(backend)?;
            let opaque = cvvdp_gpu::session::new_opaque_on_stream(
                b,
                stream_value,
                width,
                height,
                p,
                cvvdp_gpu::params::DisplayGeometry::STANDARD_4K,
                mode.into(),
            )
            .map_err(|e| Error::Metric {
                kind: "cvvdp",
                message: e.to_string(),
            })?;
            Ok(crate::metric::Metric::from_inner_with_peak(
                crate::metric::MetricInner::Cvvdp(opaque),
                crate::metric::SDR_REFERENCE_NITS,
            ))
        }
        #[cfg(all(
            feature = "ssim2",
            any(
                feature = "cuda",
                feature = "wgpu",
                feature = "hip",
                feature = "cpu",
                feature = "cubecl-types"
            )
        ))]
        MetricKind::Ssim2 => {
            let p = match params {
                MetricParams::Ssim2(p) => p,
                _ => panic!("MetricParams variant mismatch (expected Ssim2)"),
            };
            let b = crate::metric::ssim2_backend(backend)?;
            let opaque = ssim2_gpu::session::new_opaque_on_stream(
                b,
                stream_value,
                width,
                height,
                p,
                mode.into(),
            )
            .map_err(|e| Error::Metric {
                kind: "ssim2",
                message: e.to_string(),
            })?;
            Ok(crate::metric::Metric::from_inner_with_peak(
                crate::metric::MetricInner::Ssim2(opaque),
                crate::metric::SDR_REFERENCE_NITS,
            ))
        }
        #[cfg(all(
            feature = "butter",
            any(
                feature = "cuda",
                feature = "wgpu",
                feature = "hip",
                feature = "cpu",
                feature = "cubecl-types"
            )
        ))]
        MetricKind::Butter => {
            let p = match params {
                MetricParams::Butter(p) => p,
                _ => panic!("MetricParams variant mismatch (expected Butter)"),
            };
            let b = crate::metric::butter_backend(backend)?;
            let opaque = butteraugli_gpu::session::new_opaque_on_stream(
                b,
                stream_value,
                width,
                height,
                p,
                mode.into(),
            )
            .map_err(|e| Error::Metric {
                kind: "butter",
                message: e.to_string(),
            })?;
            Ok(crate::metric::Metric::from_inner_with_peak(
                crate::metric::MetricInner::Butter(opaque),
                crate::metric::SDR_REFERENCE_NITS,
            ))
        }
        #[cfg(all(
            feature = "dssim",
            any(
                feature = "cuda",
                feature = "wgpu",
                feature = "hip",
                feature = "cpu",
                feature = "cubecl-types"
            )
        ))]
        MetricKind::Dssim => {
            let p = match params {
                MetricParams::Dssim(p) => p,
                _ => panic!("MetricParams variant mismatch (expected Dssim)"),
            };
            let b = crate::metric::dssim_backend(backend)?;
            let opaque = dssim_gpu::session::new_opaque_on_stream(
                b,
                stream_value,
                width,
                height,
                p,
                mode.into(),
            )
            .map_err(|e| Error::Metric {
                kind: "dssim",
                message: e.to_string(),
            })?;
            Ok(crate::metric::Metric::from_inner_with_peak(
                crate::metric::MetricInner::Dssim(opaque),
                crate::metric::SDR_REFERENCE_NITS,
            ))
        }
        #[cfg(all(
            feature = "iwssim",
            any(
                feature = "cuda",
                feature = "wgpu",
                feature = "hip",
                feature = "cpu",
                feature = "cubecl-types"
            )
        ))]
        MetricKind::Iwssim => {
            let p = match params {
                MetricParams::Iwssim(p) => p,
                _ => panic!("MetricParams variant mismatch (expected Iwssim)"),
            };
            let b = crate::metric::iwssim_backend(backend)?;
            let opaque = iwssim_gpu::session::new_opaque_on_stream(
                b,
                stream_value,
                width,
                height,
                p,
                mode.into(),
            )
            .map_err(|e| Error::Metric {
                kind: "iwssim",
                message: e.to_string(),
            })?;
            Ok(crate::metric::Metric::from_inner_with_peak(
                crate::metric::MetricInner::Iwssim(opaque),
                crate::metric::SDR_REFERENCE_NITS,
            ))
        }
        #[cfg(all(
            feature = "zensim",
            any(
                feature = "cuda",
                feature = "wgpu",
                feature = "hip",
                feature = "cpu",
                feature = "cubecl-types"
            )
        ))]
        MetricKind::Zensim => {
            let p = match params {
                MetricParams::Zensim(p) => p,
                _ => panic!("MetricParams variant mismatch (expected Zensim)"),
            };
            let b = crate::metric::zensim_backend(backend)?;
            let opaque = zensim_gpu::session::new_opaque_on_stream(
                b,
                stream_value,
                width,
                height,
                p,
                mode.into(),
            )
            .map_err(|e| Error::Metric {
                kind: "zensim",
                message: e.to_string(),
            })?;
            Ok(crate::metric::Metric::from_inner_with_peak(
                crate::metric::MetricInner::Zensim(opaque),
                crate::metric::SDR_REFERENCE_NITS,
            ))
        }
        #[allow(unreachable_patterns)]
        other => Err(Error::Metric {
            kind: other.tag(),
            message: format!(
                "MetricSession not yet wired for '{}' (issue #17 follow-up). Use Metric::new on \
                 the shared default stream, or wait for the per-crate session hook.",
                other.tag()
            ),
        }),
    }
}

/// Run `memory_cleanup()` + `sync()` on `backend`'s pool for the
/// session's `stream_value`. Routes to the wired metric crate's stream
/// helper (cvvdp). For backends/builds where no wired crate supports
/// the backend, this is a best-effort no-op.
#[allow(unused_variables)]
fn cleanup_session_stream(backend: Backend, stream_value: u64) {
    // cubecl's `memory_cleanup` is keyed by `stream_id.value % max_streams`
    // (allocator- and metric-independent), and every enabled metric crate
    // resolves the SAME per-(device, thread) cubecl client for a backend, so
    // ANY enabled crate's `session::cleanup_stream` cleans this session's
    // stream pool — not just cvvdp's. Try each enabled crate in a fixed order
    // (mirrors `metric::reclaim_pooled_vram`) so a build that compiles OUT
    // cvvdp but IN another metric still reclaims a dropped session's VRAM
    // instead of silently no-op'ing (the cvvdp-only routing was correct only
    // when cvvdp happened to be in the build).
    #[cfg(all(
        feature = "cvvdp",
        any(
            feature = "cuda",
            feature = "wgpu",
            feature = "hip",
            feature = "cpu",
            feature = "cubecl-types"
        )
    ))]
    if let Ok(b) = crate::metric::cvvdp_backend(backend) {
        cvvdp_gpu::session::cleanup_stream(b, stream_value);
        return;
    }
    #[cfg(all(
        feature = "butter",
        any(
            feature = "cuda",
            feature = "wgpu",
            feature = "hip",
            feature = "cpu",
            feature = "cubecl-types"
        )
    ))]
    if let Ok(b) = crate::metric::butter_backend(backend) {
        butteraugli_gpu::session::cleanup_stream(b, stream_value);
        return;
    }
    #[cfg(all(
        feature = "ssim2",
        any(
            feature = "cuda",
            feature = "wgpu",
            feature = "hip",
            feature = "cpu",
            feature = "cubecl-types"
        )
    ))]
    if let Ok(b) = crate::metric::ssim2_backend(backend) {
        ssim2_gpu::session::cleanup_stream(b, stream_value);
        return;
    }
    #[cfg(all(
        feature = "dssim",
        any(
            feature = "cuda",
            feature = "wgpu",
            feature = "hip",
            feature = "cpu",
            feature = "cubecl-types"
        )
    ))]
    if let Ok(b) = crate::metric::dssim_backend(backend) {
        dssim_gpu::session::cleanup_stream(b, stream_value);
        return;
    }
    #[cfg(all(
        feature = "iwssim",
        any(
            feature = "cuda",
            feature = "wgpu",
            feature = "hip",
            feature = "cpu",
            feature = "cubecl-types"
        )
    ))]
    if let Ok(b) = crate::metric::iwssim_backend(backend) {
        iwssim_gpu::session::cleanup_stream(b, stream_value);
        return;
    }
    #[cfg(all(
        feature = "zensim",
        any(
            feature = "cuda",
            feature = "wgpu",
            feature = "hip",
            feature = "cpu",
            feature = "cubecl-types"
        )
    ))]
    if let Ok(b) = crate::metric::zensim_backend(backend) {
        zensim_gpu::session::cleanup_stream(b, stream_value);
        return;
    }
    // No enabled metric crate supports this backend in this build — nothing
    // to clean.
    let _ = (backend, stream_value);
}

/// Read `bytes_reserved` for `backend`'s pool on the session's
/// `stream_value`, via the wired metric crate (cvvdp). Returns `None`
/// if no wired crate supports the backend or the query fails. Exposed
/// for the VRAM-isolation test.
#[doc(hidden)]
#[allow(unused_variables)]
pub fn stream_reserved_bytes(backend: Backend, stream_value: u64) -> Option<u64> {
    // Same metric-agnostic stream keying as `cleanup_session_stream` — any
    // enabled crate reports the same pool's `bytes_reserved` for the stream.
    #[cfg(all(
        feature = "cvvdp",
        any(
            feature = "cuda",
            feature = "wgpu",
            feature = "hip",
            feature = "cpu",
            feature = "cubecl-types"
        )
    ))]
    if let Ok(b) = crate::metric::cvvdp_backend(backend) {
        return cvvdp_gpu::session::stream_reserved_bytes(b, stream_value);
    }
    #[cfg(all(
        feature = "butter",
        any(
            feature = "cuda",
            feature = "wgpu",
            feature = "hip",
            feature = "cpu",
            feature = "cubecl-types"
        )
    ))]
    if let Ok(b) = crate::metric::butter_backend(backend) {
        return butteraugli_gpu::session::stream_reserved_bytes(b, stream_value);
    }
    #[cfg(all(
        feature = "ssim2",
        any(
            feature = "cuda",
            feature = "wgpu",
            feature = "hip",
            feature = "cpu",
            feature = "cubecl-types"
        )
    ))]
    if let Ok(b) = crate::metric::ssim2_backend(backend) {
        return ssim2_gpu::session::stream_reserved_bytes(b, stream_value);
    }
    #[cfg(all(
        feature = "dssim",
        any(
            feature = "cuda",
            feature = "wgpu",
            feature = "hip",
            feature = "cpu",
            feature = "cubecl-types"
        )
    ))]
    if let Ok(b) = crate::metric::dssim_backend(backend) {
        return dssim_gpu::session::stream_reserved_bytes(b, stream_value);
    }
    #[cfg(all(
        feature = "iwssim",
        any(
            feature = "cuda",
            feature = "wgpu",
            feature = "hip",
            feature = "cpu",
            feature = "cubecl-types"
        )
    ))]
    if let Ok(b) = crate::metric::iwssim_backend(backend) {
        return iwssim_gpu::session::stream_reserved_bytes(b, stream_value);
    }
    #[cfg(all(
        feature = "zensim",
        any(
            feature = "cuda",
            feature = "wgpu",
            feature = "hip",
            feature = "cpu",
            feature = "cubecl-types"
        )
    ))]
    if let Ok(b) = crate::metric::zensim_backend(backend) {
        return zensim_gpu::session::stream_reserved_bytes(b, stream_value);
    }
    let _ = (backend, stream_value);
    None
}

/// The `StreamId.value` a session with the given backend-local `slot`
/// allocates on. Exposed `#[doc(hidden)]` for the VRAM-isolation test
/// so it can probe a session's stream without reaching into private
/// fields. (Sessions hand slots out lowest-first, so the first session
/// on a fresh backend gets slot 0 → this base value.)
#[doc(hidden)]
pub fn stream_value_for_slot(slot: u32) -> u64 {
    STREAM_VALUE_BASE + slot as u64
}

// Compile-time enforcement of the threading contract documented on
// `MetricSession`: the session itself is plain slot/stream bookkeeping, so it
// is `Send + Sync`. The metric handles stay `Send` so the orchestrator can
// hand a warm metric to a worker lane — `!Send` would be a regression (it
// closes strictly fewer hazards than the private-stream design already does).
// If any of these stops holding this block fails to compile: adjust the
// design, not the assertion.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<MetricSession>();
    assert_sync::<MetricSession>();
    assert_send::<SessionMetric<'static>>();
    assert_send::<OwnedSessionMetric>();
};
