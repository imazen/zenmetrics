# zenfleet-orchestrator trait hoist (iter 2 — provider-generic driver)

**Date:** 2026-05-28
**Companion to:** `docs/zenfleet_orchestrator_hoist_2026-05-28.md` (iter 1, issue #68).

## What this iter completes

Iter 1 (#68, commits `f080564d` + `9eb7107e`) hoisted the four
pure algorithms — `compute_provisioned_replicas`,
`ttl_redispatch_decisions`, `filter_classes`, `SpeculativeState` —
into `zenfleet-orchestrator`. The Salad bin called them but still
owned the full lifecycle (provision, queue create, poll loop,
fleet_summary stitch, teardown) at ~2,100 LOC.

This iter extracts the lifecycle behind two traits and a generic
driver:

| Trait              | Implementors do                                                |
| ------------------ | -------------------------------------------------------------- |
| `ProviderHandle`   | Provision / poll / teardown / push jobs on a compute provider  |
| `R2Operator`       | Operator-side blob list / upload / get_bytes                   |

The generic driver `FleetSweep<P: ProviderHandle, R: R2Operator>`
(at `crates/zenfleet-orchestrator/src/driver.rs`) owns the full
lifecycle:

```rust
let summary = FleetSweep::new(provider, storage, sweep_cfg, ...)
    .run(queue_jobs)
    .await?;
```

`run()` does: provision → push initial jobs → poll loop
(TTL + speculative re-dispatch, R2 snapshot upload, group-state
read, exit conditions) → fleet_summary stitch + upload →
retry-bounded teardown.

## The Salad bin

`crates/zen-cloud-salad/src/bin/zen-salad-sweep.rs`:
**357 LOC** (down from 2096). What remains:

- 35-flag clap `Args` struct (110 LOC — back-compat with iter 1's CLI).
- `LauncherSummary` stdout JSON struct (28 LOC).
- `main()` (~200 LOC) — wire `R2OperatorImpl`, `SaladProviderHandle`,
  `FleetSweep` together; emit summary.

Everything else moved into reusable `zen-cloud-salad` lib modules
(all behind `feature = "launcher"`):

| Module                                  | What it owns                                                          |
| --------------------------------------- | --------------------------------------------------------------------- |
| `zen_cloud_salad::r2_ops`               | Operator-side R2 SigV4 (HEAD / PUT / GET / LIST); `R2OperatorImpl`    |
| `zen_cloud_salad::provider`             | `SaladProviderHandle` + `SaladProviderConfig`                         |
| `zen_cloud_salad::launcher_support`     | GPU class resolve, prior-fleet class filter, chunk synth, R2 cred load |

## Why this matters for follow-ups

`zencloud-hetzner` (next iter) implements `ProviderHandle` and
reuses `FleetSweep::run` verbatim. Same path for any future
RunPod / Vast.ai / GCP-Batch launcher. The poll-loop algorithms
(TTL + speculative + class filter + replica overshoot) are now
single-source-of-truth in `zenfleet-orchestrator`.

## Verification (no Salad spend)

```sh
# Build (lib + bin, launcher feature)
cargo build --release -p zenfleet-orchestrator -p zen-cloud-salad \
    --features zen-cloud-salad/launcher

# Unit tests
cargo test -p zenfleet-orchestrator          # 8 algorithm tests pass
cargo test -p zen-cloud-salad --features launcher --lib   # 24 salad lib tests pass

# Dry-run: synthesises the container-group request body WITHOUT any
# provisioning / R2 mint / queue create.
./target/release/zen-salad-sweep --dry-run --replicas 10 --max-price-per-hour 0.10
./target/release/zen-salad-sweep --dry-run --replicas 5 --max-price-per-hour 0.05 \
    --no-speculative --chunk-ttl-secs 240 --cells-per-chunk 8
```

All build + test + dry-run combinations green. Dry-run prints the
expected `gpu_class_names` list (auto-enumerated at the named price
tier) plus `replicas_overshoot`, `cells_per_chunk`, and the resolved
GPU class id Vec.

## Behavior parity

Every CLI flag from iter 1's bin still parses and round-trips into
the right `SweepConfig` / `SpeculativeConfig` / `ProvisionSpec` /
`SaladProviderConfig` field:

- `--replicas`, `--replicas-overshoot` → `SweepConfig` (driver applies overshoot internally via `compute_provisioned_replicas`)
- `--chunk-ttl-secs` → `SweepConfig.chunk_ttl_secs`
- `--cells-per-chunk` → `SweepConfig.cells_per_chunk`
- `--no-speculative` + `--speculative-*` → `SpeculativeConfig`
- `--max-warmup-secs`, `--min-productive-chunks` → class filter (`apply_class_filter`)
- `--prior-fleet-summary` → class filter via `R2Operator::get_bytes`
- `--gpu-class` / `--gpu-classes` / `--max-price-per-hour` → `resolve_gpu_classes`
- `--cpu`, `--memory-mib`, `--registry-username/password` → `SaladProviderConfig`
- `--max-wall-secs`, `--poll-secs` → `FleetSweep::new(..., max_wall_secs, poll_secs, ...)`
- `--keep-running` → `FleetSweep::new(..., keep_running)` (skips teardown)
- `--dry-run` short-circuits before any R2 / Salad call

## Files touched

| Path                                                | Change                          |
| --------------------------------------------------- | ------------------------------- |
| `crates/zenfleet-orchestrator/Cargo.toml`           | add tokio + anyhow              |
| `crates/zenfleet-orchestrator/src/lib.rs`           | re-export driver + provider     |
| `crates/zenfleet-orchestrator/src/provider.rs`      | NEW (264 LOC) — traits + types  |
| `crates/zenfleet-orchestrator/src/driver.rs`        | NEW (630 LOC) — FleetSweep      |
| `crates/zen-cloud-salad/src/lib.rs`                 | re-export new modules           |
| `crates/zen-cloud-salad/src/launch.rs`              | `#[derive(Clone)]` for RegistryAuth |
| `crates/zen-cloud-salad/src/provider.rs`            | NEW (245 LOC) — SaladProviderHandle |
| `crates/zen-cloud-salad/src/r2_ops.rs`              | NEW (398 LOC) — operator-side R2 + SigV4 |
| `crates/zen-cloud-salad/src/launcher_support.rs`    | NEW (333 LOC) — class resolve / class filter / chunk synth / cred load |
| `crates/zen-cloud-salad/src/bin/zen-salad-sweep.rs` | REWRITE (357 LOC, was 2096)     |

## Open items

- Bin is 357 LOC, not 150. ~110 LOC of that is the 35-flag clap
  `Args` struct (iter-1 back-compat). A config-file approach would
  let it shrink further but would break existing operator scripts;
  out of scope here.
- `ProvisionSpec.extra` carries Salad's `gpu_class_ids` via
  `serde_json::Value`. Works but opaque. When Hetzner lands we'll
  see whether per-provider config (`ProvisionSpec<SaladExtras>`)
  is the right shape or whether the Value blob is fine in practice.
- `R2Operator` is the only blob-store trait the driver consumes.
  If a future provider stores sidecars in non-S3-compat storage
  (e.g., Hetzner Object Storage's native API) the trait stays
  valid; only the implementor changes.
