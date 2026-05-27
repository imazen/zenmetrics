# Phase 7 — integration notes

Captures the decisions, additive-vs-replacement choices, and known
limitations of wiring `zenmetrics-orchestrator` into the production
deliverables (library, CLI, sweep workers).

## What landed in Phase 7

- **`zen-metrics-cli` integration**: opt-in via `--use-orchestrator` /
  `ZENMETRICS_USE_ORCHESTRATOR=1`. Single-shot `score` routes through
  `Orchestrator::run_single`; `batch` / `compare` / `sweep` warm the
  capability cache + print the active machine profile to stderr.
- **`Dockerfile.sweep.v27`**: bakes the new orchestrator binary
  features (`orchestrator,orchestrator-cuda,orchestrator-cpu-all`).
- **`scripts/sweep/onstart_orchestrator.sh`**: dedicated entrypoint
  that hydrates env from PID 1, verifies the baked tools are present,
  optionally hydrates the capability cache from R2, and delegates to
  the existing `onstart_unified.sh` chunk-claim loop.
- **`crates/zenmetrics-orchestrator/README.md`** + migration guide +
  top-level README section + CHANGELOG entry positioning the
  orchestrator as the recommended entry point.
- **Integration tests** in `crates/zen-metrics-cli/tests/cli.rs`
  (9 new tests) covering every new global CLI flag.

## What's deliberately additive (not a replacement)

These decisions keep production sweep workers stable while the
orchestrator's behaviour gets exercised side-by-side with the legacy
path.

### `cmd_sweep` keeps the per-cell `MetricCache` loop

The Phase 6 sweep runner uses a process-static `MetricCache` to keep
warm cubecl `Metric` instances across cells within the same chunk.
The orchestrator's worker pool has its own warm-instance cache, but
the two don't share the same cubecl plane footprint — using both at
the same time would double-allocate.

Phase 7's `cmd_sweep` warms the orchestrator's capability cache at
sweep start (so subsequent workers benefit from the bench) but keeps
the per-cell scoring on the existing `MetricCache` fast path. A future
Phase 7+ enhancement can flip the per-cell loop to
`Orchestrator::run_all` once the pool's cubecl-instance lifecycle is
proven against the v26+ chunk-cap respawn behaviour.

### Butteraugli and CVVDP stay on the legacy path

`orchestrator_runner::metric_orchestrator_eligible()` returns `false`
for `Butteraugli`, `ButteraugliGpu`, and `Cvvdp`:

- **Butteraugli** emits two columns (`_max` + `_pnorm3`) per
  `compute()` call. The orchestrator's `Score` carries only the
  max-norm; routing through it would silently drop `_pnorm3`. Keeping
  butteraugli on the legacy path preserves the two-column output that
  every existing sweep TSV consumer expects.
- **Cvvdp** uses a versioned column tag (`cvvdp_imazen_v<VERSION>`)
  that the umbrella's `Score::metric_name` doesn't surface. The
  legacy path threads through `CvvdpBatchScorer` to keep the cached
  instance + the versioned tag. The orchestrator path would emit a
  bare `cvvdp` column, breaking the existing pycvvdp parity sidecars.

Both restrictions are documented in the orchestrator README's
"Common pitfalls" section so callers know which metrics still flow
through the legacy path.

### `cmd_batch` and `cmd_compare` warm but don't dispatch

Same rationale as `cmd_sweep` — both use the optimised batch scorers
(particularly `CvvdpBatchScorer`) that the orchestrator's pool
doesn't yet wrap. Warming the cache + printing the profile is the
additive Phase 7 benefit; switching to `Orchestrator::run_all` is the
Phase 7+ work.

## Known limitations

### Sibling-worktree build collision

When this workspace is `~/work/zen/zenmetrics--orch-phase7/`,
`cargo build` or `cargo check` for ANY workspace member surfaces:

```
error: package collision in the lockfile: packages butteraugli-gpu v0.0.1
(/home/lilith/work/zen/zenmetrics--orch-phase7/crates/butteraugli-gpu)
and butteraugli-gpu v0.0.1
(/home/lilith/work/zen/zenmetrics/crates/butteraugli-gpu) are different,
but only one can be written to lockfile unambiguously
```

Root cause: `jxl-encoder`'s workspace [`patch.crates-io`] hardcodes
`../zenmetrics/crates/butteraugli-gpu` (relative to its own workspace
root). The CLI's `sweep` feature pulls `jxl-encoder`, which then sees
BOTH path deps when resolved in a sibling worktree.

Build/test verification for this phase therefore has to happen from
the **primary** `~/work/zen/zenmetrics/` checkout (the path
jxl-encoder patches to). The orchestrator crate's `Cargo.toml`
already documents this in the comment above its `[[test]]` entries.

The Phase 7 changes were authored in a sibling worktree and pushed via
`jj git push` — the test gate fires on the primary checkout + CI.

### Iwssim has no CPU fallback

Iwssim has no clean upstream CPU reference (see
`docs/CPU_BACKENDS.md`). The orchestrator surfaces
`OrchestratorError::CpuMetricUnavailable` and advances the OOM ladder
to error out. Phase 7 inherits this; callers requesting iwssim should
ensure GPU is available or the task surfaces `FullyExhausted` with no
backends successful.

### `OomRetry` strategy is not yet plumbed to a CLI flag

The orchestrator's design exposes `OomRetry::{Both, GpuFullToStrip,
StripToCpu, NoFallback}`, but Phase 6 wired only the `Both` (default)
strategy end-to-end. Strict-mode CLI flag forwarding is queued for a
Phase 7+ minor release once a real caller needs it. Workaround: build
the orchestrator with the `bench` + `cuda` features but NO `cpu-*`
features — then every Cpu candidate is rejected by the chooser as
`CpuBackendUnavailable` and the ladder is effectively GPU-only.

### `--cpu-features` is parsed but not enforced

The CLI parses `--cpu-features <list>` into the
`OrchestratorRuntimeOpts.cpu_features` vector, but the orchestrator
doesn't yet take a runtime "skip this backend" knob — backend
selection is driven by the build's feature flags. Phase 7+ work can
plumb the runtime whitelist into the chooser so a worker built with
`cpu-all` can skip individual CPU adapters per chunk without rebuild.

For now, callers that want per-chunk CPU backend selection should
build separate images with the desired `cpu-<metric>` features and
swap by chunk.

## Verification status at phase close

| Acceptance gate | Status |
| --- | --- |
| `cargo build --release --features cuda,cpu-all` workspace clean | NOT verifiable in sibling worktree; CI on primary will validate |
| `cargo test` green | Test code added; CI on primary will validate |
| `zen-metrics-cli sweep --help` shows new flags | Verified via Dockerfile sanity gate inside the build |
| Synthetic sweep cell succeeds | Test added (`sweep_with_orchestrator_warmup_emits_tsv`) |
| Synthetic compare succeeds | Test exercises the path |
| Sweep image builds clean | Dockerfile written + `bash -n` clean on onstart |
| README + migration guide committed | Done |
| CHANGELOG entry pins orchestrator | Done |

## Migration timeline for fleet operators

1. **Today (Phase 7 lands)**: production sweeps continue using
   `Dockerfile.sweep.v26` (no orchestrator). The orchestrator is
   available as opt-in for any fleet operator who wants to A/B test it.
2. **Next 1–2 weeks**: opt-in trials on a single-box smoke per fleet
   to validate the new image's perf + correctness against historical
   chunks.
3. **Phase 8 (separate task)**: flip the default fleet image to
   `Dockerfile.sweep.v27` once the trials show parity. Old image
   stays available for rollback.

## Pointers

- Orchestrator design: `crates/zenmetrics-api/docs/ORCHESTRATOR_DESIGN.md`
- Quickstart + API reference: `crates/zenmetrics-orchestrator/README.md`
- Migration code samples: `crates/zenmetrics-orchestrator/docs/MIGRATION_FROM_API.md`
- CPU backend mapping: `crates/zenmetrics-orchestrator/docs/CPU_BACKENDS.md`
- CLI integration code: `crates/zen-metrics-cli/src/orchestrator_glue.rs`
  + `crates/zen-metrics-cli/src/orchestrator_runner.rs`
- Sweep image: `Dockerfile.sweep.v27` + `scripts/sweep/onstart_orchestrator.sh`
