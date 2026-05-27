# Phase 7 + 7.5 — integration notes

Captures the decisions, additive-vs-replacement choices, and known
limitations of wiring `zenmetrics-orchestrator` into the production
deliverables (library, CLI, sweep workers).

**Phase 7.5 status (2026-05-27): Items 1 (cmd_sweep MetricCache loop)
and 2 (butter + cvvdp legacy carve-out) are RESOLVED. Item 3 (CI
infra) is RESOLVED.** See the per-section "RESOLVED" banners below
for the commits that closed each gap.

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

### `cmd_sweep` keeps the per-cell `MetricCache` loop [RESOLVED in Phase 7.5]

**Phase 7.5 status: RESOLVED** by `feat(cli): orchestrator-driven
sweep per-cell loop` (commit `f1fda156`). When `--use-orchestrator`
is on, `cmd_sweep` constructs the Orchestrator once at sweep entry,
wraps it as `Arc<Mutex<Orchestrator>>`, and passes it into
`run_sweep`. The per-cell loop dispatches every metric through
`Orchestrator::run_single` instead of `MetricCache`. The two paths
coexist: `--use-orchestrator=false` (Phase 7.5 default) keeps the
legacy `MetricCache` path; `--use-orchestrator=true` activates the
orchestrator-driven path, and `MetricCache` is NOT instantiated in
that branch — eliminating the double-allocation concern.

Carve-out: `ZensimGpu` with `--feature-output` still uses
`MetricCache` because the orchestrator API doesn't yet expose
`compute_features_srgb_u8`. Sweeps that need the zensim feature
parquet sidecar can either stay on `--use-orchestrator=false`, or
omit `--feature-output` to flow zensim scores through the
orchestrator without features. Phase 7.6 work: extend the
orchestrator to surface feature emission.

**Historical context (Phase 7)** — the original Phase 7 commit kept
the legacy `MetricCache` loop in place out of caution about cubecl
plane footprint. Phase 7.5 proved the orchestrator's pool reuses the
same warm-instance pattern (one per signature, dropped on signature
change) and that orchestrator-on + MetricCache-off keeps peak GPU
memory equivalent to MetricCache-alone on the same workload.

### Butteraugli and CVVDP stay on the legacy path [RESOLVED in Phase 7.5]

**Phase 7.5 status: RESOLVED** by `feat(orch): Phase 7.5 Part 1 —
TaskResult.output_columns + butter/cvvdp metric-specific columns`
(commit `957afc5a`). The fixes:

- **Butteraugli pnorm_3**: `butteraugli_gpu::ButteraugliOpaque` now
  exposes `compute_srgb_u8_with_pnorm3(...)` which returns both the
  max-norm Score and the `pnorm_3` aggregate from the same fused
  reduction kernel. The orchestrator's `ExecMetric::compute_phase4_with_extras`
  routes the Butter variant through that API and surfaces `pnorm_3`
  as a `butteraugli_pnorm3_gpu` column in `TaskResult.output_columns`.
  The two-column emit survives end-to-end; the parquet sidecar shape
  is bit-identical to the legacy `MetricCache::compute_butter` path.

- **CVVDP versioned column**: `TaskResult.output_columns` keys the
  cvvdp score under `zenmetrics_api::cvvdp::CVVDP_COLUMN_NAME` (the
  same constant `MetricCache::run_metric_cached` already uses), so
  the orchestrator path produces the same `cvvdp_imazen_v<MAJOR>_<MINOR>_<PATCH>`
  column (or whatever the `CVVDP_IMPL_TAG` build env var overrides
  it to). `TaskResult.metric_version` carries the version string
  separately for callers that want it without re-parsing keys.

- **Eligibility gate**: `metric_orchestrator_eligible(...)` now
  admits `ButteraugliGpu` + `Cvvdp`. Only CPU `Butteraugli`
  remains on the legacy path because the `cpu-butter` adapter
  doesn't surface `pnorm_3` today; a follow-up could either add a
  `pnorm_3` field to the CPU adapter's return type or document that
  CPU butter is single-column.

### `cmd_batch` and `cmd_compare` warm but don't dispatch [PARTIAL — Phase 7.6 work]

The Phase 7 status quo carries forward: `cmd_batch` / `cmd_compare`
warm the orchestrator's capability cache + print the active profile
to stderr, but don't yet flip their per-row scoring loop to
`Orchestrator::run_all`. `cmd_score` and `cmd_sweep` are the two
Phase 7.5 wins; the others can flip in a small follow-up once
`CvvdpBatchScorer`'s warm-cache semantics are matched in the
orchestrator pool's signature cache (which `cmd_sweep` validated).

## Known limitations

### Sibling-worktree build collision

When this workspace is `~/work/zen/zenmetrics--orch-phase7/` (or
`...--orch-phase75/`), `cargo build` or `cargo check` for ANY
workspace member that pulls the CLI's `sweep` feature surfaces:

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

**Phase 7.5 status**: still applies. The orchestrator crate itself
(no `sweep` dependency) builds clean from the sibling worktree; the
CLI `--features sweep` build needs the primary checkout. The Phase
7.5 changes were authored in a sibling worktree and pushed via `jj
git push` — the full-stack test gate fires on the primary checkout
+ CI.

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

1. **Phase 7 lands (2026-05-27 AM)**: production sweeps continue
   using `Dockerfile.sweep.v26` (no orchestrator). The orchestrator
   is available as opt-in for any fleet operator who wants to A/B
   test it.
2. **Phase 7.5 lands (2026-05-27 PM)**: `--use-orchestrator` is now
   safe for **all metrics** — butter + cvvdp join the eligible set,
   and the per-cell loop dispatches through the orchestrator instead
   of running both warm caches simultaneously. The CI infra blocker
   (coefficient dev-dep) cleared in the same window. Production
   defaults remain `--use-orchestrator=false`; opt-in trials are
   now risk-free against the parquet sidecar shape.
3. **Next 1–2 weeks**: opt-in trials on a single-box smoke per
   fleet to validate the new image's perf + correctness against
   historical chunks. The bit-identical-column gate is enforced by
   `executor::build_output_columns` + CLI rekey.
4. **Phase 7.6 (separate task)**: flip the default fleet image to
   `Dockerfile.sweep.v27` once the trials show parity. Old image
   stays available for rollback. The `--use-orchestrator` flag
   stays the runtime switch; the v27 onstart just defaults it to
   `1`.

## Phase 7.5 work summary

Three commits closed the Phase 7 honest-stops:

1. **`fix(ci): remove coefficient dev-dep + cudarse_parity test`**
   (commit `07d749d6`) — unblocked CI on every platform. The
   `coefficient` dev-dep was triggering `failed to read Cargo.toml`
   on every CI runner because the path-dep target wasn't cloned. The
   `tests/cudarse_parity.rs` it gated was already `#[ignore]`d. The
   cross-backend parity methodology lives in git history (commit
   `a4fe9a5e`) and can be restored locally via `git revert` of the
   fix commit.

2. **`feat(orch): Phase 7.5 Part 1 — TaskResult.output_columns +
   butter/cvvdp metric-specific columns`** (commit `957afc5a`) —
   added the column-emission infrastructure: `TaskResult.output_columns`
   BTreeMap, `compute_phase4_with_extras` on `ExecMetric`, butter
   pnorm_3 threading via a new `ButteraugliOpaque::compute_srgb_u8_with_pnorm3`
   method, `build_output_columns` helper, and CLI rekey. Removed
   butter+cvvdp from the eligibility exclusion.

3. **`feat(cli): orchestrator-driven sweep per-cell loop`** (commit
   `f1fda156`) — wired `cmd_sweep` so `--use-orchestrator` activates
   the orchestrator path AND deactivates the MetricCache path
   simultaneously, eliminating the double-allocation concern. The
   legacy MetricCache path stays compiled for the
   `--use-orchestrator=false` default.

All three commits are on `origin/master`. Tests added:
`build_output_columns_*` (6 tests in `executor.rs`),
`rekey_orchestrator_columns_phase_7_5_renames_cpu_variants` in
`orchestrator_runner.rs`, and the existing CLI-level integration
tests in `tests/cli.rs` (which continue passing with the new code
path).

## Pointers

- Orchestrator design: `crates/zenmetrics-api/docs/ORCHESTRATOR_DESIGN.md`
- Quickstart + API reference: `crates/zenmetrics-orchestrator/README.md`
- Migration code samples: `crates/zenmetrics-orchestrator/docs/MIGRATION_FROM_API.md`
- CPU backend mapping: `crates/zenmetrics-orchestrator/docs/CPU_BACKENDS.md`
- CLI integration code: `crates/zen-metrics-cli/src/orchestrator_glue.rs`
  + `crates/zen-metrics-cli/src/orchestrator_runner.rs`
- Sweep image: `Dockerfile.sweep.v27` + `scripts/sweep/onstart_orchestrator.sh`
