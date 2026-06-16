#![forbid(unsafe_code)]
//! # zenfleet-dash
//!
//! The Railway-hosted **control plane** for the zen job system (goals B/C/D). It reads the columnar
//! Parquet **ledger** / **blob index** / worker heartbeats — the single source of truth — and serves:
//!
//! - **Monitoring** ([`views`]): live fleet, progress per job kind, cost (incl. cost-per-1000-jobs
//!   per tier), failure drill-down (failures are rows), storage per tier.
//! - **Control** ([`control`]): GC dry-run preview, stop-spend decision, kill/pause/drain *intents*.
//!
//! Per the Foundations, the dashboard **never runs workers** — it observes the ledger and emits
//! control intents. Actuation (actually tearing down boxes) is performed by the fleet layer
//! (`zenfleet-orchestrator` / `zen-cloud-*`) consuming these intents; that wiring is the integration
//! step gated on a clean tree. The pure view/control logic here is fully testable offline.

pub mod control;
pub mod data;
pub mod fleet;
pub mod notify;
pub mod views;

pub use control::{ControlIntent, GcDryRun, StopSpendDecision, gc_dry_run, stop_spend};
pub use data::{DashData, DashError};
pub use fleet::{
    FleetBox, KillResult, fleet_label_key, fleet_token, idle_boxes, kill_fleet, kill_named,
    list_fleet, selector_for,
};
pub use notify::{NotifyEvent, NotifyPayload, detect, format_event};
pub use views::{
    CatalogRow, CostView, FailureCell, KindProgress, QueryRow, ResultRow, RunSummary, TierStorage,
    WorkerStat, catalog_view, cost_view, failures, kind_label, progress, query_view, results_view,
    run_summary, storage, workers_view,
};
