//! # zenfleet-sim — deterministic fault-injection simulator for the fleet job system
//!
//! The fleet's load-bearing reliability code — the R2 claim/steal race, the
//! reconcile gap loop, lease staleness, idle detection, teardown — is exactly
//! the code that is hardest to test, because in production it is coupled to a
//! real object store (Cloudflare R2 via s5cmd, with all its eventual-consistency
//! and partial-failure quirks) and to real paid cloud boxes that die, fail to
//! start, and fail to tear down. So today it is tested only by live bash demos
//! that never run in CI (see `docs/RUNNING_JOBS.md` and the fleet inventory).
//!
//! This crate closes that gap. It provides a **breakable in-memory cloud
//! substrate** and drives the **real** [`zenfleet_core`] logic against it, so
//! the failure modes that only show up in the field become deterministic,
//! reproducible unit/integration tests:
//!
//! - [`FaultStore`] — an in-memory [`zenfleet_cloud::BlobStorage`] that injects
//!   the things s5cmd/R2 actually do to us: **eventual consistency**
//!   (read-after-write and list-after-write staleness — the reason the vast
//!   claim carries a 1.5 s read-back delay), **403s** (bad or expired scoped
//!   credentials), **partial uploads** (a truncated object that "succeeded"),
//!   **transient 5xx**, per-op **latency**, and **delete failures** (a claim or
//!   box that refuses to be cleaned up → the "fail to teardown" money leak).
//! - [`claim`] — the two claim algorithms side by side over that store: the
//!   production token-race ([`claim::claim_token_race`], mirroring
//!   `zenfleet-vastai::worker::claim::try_claim`) and the atomic conditional-PUT
//!   claim ([`claim::claim_conditional`], modeling R2 `If-None-Match: *`, which
//!   `zenfleet-worker::claim_or_steal_r2` already uses). The chaos tests show
//!   the token-race can double-acquire (or stall) under aggressive consistency
//!   delay while the conditional claim cannot — the test behind the
//!   "port the claim to conditional PUT" recommendation.
//! - [`SimClock`] — virtual time. Every time-dependent decision in the fleet
//!   ([`zenfleet_core::Lease::can_steal`], [`zenfleet_core::idle::detect_idle`],
//!   claim staleness) takes `now` as a parameter, so the sim drives them off one
//!   deterministic clock — no `sleep`, no wall-clock flake.
//! - [`Rng`] — a 3-line deterministic xorshift so probabilistic fault schedules
//!   are seed-reproducible.
//!
//! Later chunks add a box-lifecycle simulator ([`fleet`]) — boxes that fail to
//! start, die mid-job, go silent, or refuse to tear down — plus a full
//! [`zenfleet_cloud::run_worker`]-over-[`FaultStore`] convergence harness and the
//! invariant checkers ([`invariant`]) that assert exactly-once execution,
//! convergence, claim exclusivity, idle-fires, and bounded cost.
//!
//! Nothing here spawns a thread or touches the network: "concurrency" is modeled
//! as a controlled interleaving over the shared [`FaultStore`] handle, which is
//! what makes a race a *reproducible* test instead of a flaky one.

#![forbid(unsafe_code)]

pub mod claim;
pub mod clock;
pub mod fault;
pub mod rng;
pub mod store;

pub use claim::{ClaimOutcome, claim_conditional, claim_token_race};
pub use clock::SimClock;
pub use fault::FaultSpec;
pub use rng::Rng;
pub use store::{FaultStore, OpCounts};
