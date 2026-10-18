# dvifmish durable IQA supplements

dvifmish is a benchmark-only optional dependency of the unpublished
`cpu-profile` driver. Enable it with `compete-dvifmish`, or set
`IQA_DVIFMISH=1` when using `just iqa-compete-build durable`. The dependency
is pinned to `74ec5eeae118ead266c7b66b56e56cb5f3bef372`. Its manifest retains
`publish = false`; this work does not publish the crate, binaries or weights.
No metric-library or public CLI API is added.

The six routes are the four `port-min-v0.2`, `port-2d-v0.2`,
`port-lap-v0.2`, `port-lap-v0.3` presets, prefixed `dvifmish-`, and
`dvifmish-talk-faithful-luma-float` / `dvifmish-talk-faithful-luma-int`.
The latter use the default talk-inspired preset. Other fitted presets and
the streaming API are outside this supplement's coverage.

Port scalars are JND values. Talk-inspired scalars are raw distortion E.
Every native map is the library's per-plane/per-level block contribution
field in raw distortion units. It is not a full-resolution JND map. The
inventory and each timing row retain these units, the preset name and its
SHA-256. Timed map calls request and consume the actual native field; they
do not paint, interpolate or substitute scalar-only work.

Preset parsing and validation happen inside the existing constructor-inclusive
cold clock. First-call integer table initialization is also cold work.
Each cold sample has a fresh process. Warm calls reuse parsed constants and
run the same ten zenbench rounds with 100-ms targets as the base study.
RGB loading remains outside the compute clock. Both library APIs receive
the explicit one/eight-worker budget; small levels may use fewer workers.
Linux uses the existing checked P-core boundary and shared heavy lock.

Adapter controls compare these six routes against direct library calls with
exact scalar/map/thread bits, including odd dimensions. They do not loosen
the crate's existing numerical controls or establish new perceptual parity.
The pinned lock resolves through the existing isolated-snapshot check.

`compete-dvifmish` is a private-snapshot feature. The normal workspace
manifest and lock do not resolve this private unpublished dependency, even
for unrelated browser builds. With `IQA_DVIFMISH=1`, the builder overlays the
pinned dependency and feature in its owned snapshot, resolves only that
snapshot lock, and records both original/adapted hashes in
`dvifmish-patch.meta.json`. It never writes the overlay back to the source
checkout. The driver cfg is enabled only by this explicit build path.

## Prepare and queue without changing the original sequence

Build a native worker under the host's existing heavy lock and wrapper:

```sh
IQA_DVIFMISH=1 LOCK_SNAP_DIR="$HOME/tmp/zenmetrics-lock-snap" just iqa-compete-build durable
```

Record the actual outer build command as JSON, then bind the native binary,
inventory, compiler, committed driver sources and snapshot patch receipt:

```sh
just iqa-durable-build-receipt BINARY NEW_BUILD_JSON COMMAND_JSON \
  --source-commit COMMIT --features compete,compete-nn,metrics-parallel,compete-dvifmish \
  --snapshot "$HOME/tmp/zenmetrics-lock-snap"
```

Preparation uses an existing frozen seven-size non-NN bundle as its input
template, retaining its RGB hashes and provenance. It creates new source,
512 and larger bundles, without editing the original template or results:

```sh
just iqa-dvifmish-prepare TEMPLATE BINARY BUILD_JSON NEW_BUNDLE_ROOT NEW_RESULTS_ROOT \
  --lock HOST_HEAVY_LOCK \
  --after-512 ORIGINAL_512_NN_GPU_BUNDLE ORIGINAL_512_NN_GPU_RESULTS \
  --after-larger ORIGINAL_LARGER_NN_GPU_BUNDLE ORIGINAL_LARGER_NN_GPU_RESULTS
just iqa-dvifmish-timing-check NEW_BUNDLE_ROOT/512-dvifmish NEW_CONTROL_OUTPUT \
  --lock HOST_HEAVY_LOCK
just iqa-dvifmish-install NEW_BUNDLE_ROOT/sequence.json
```

The real control requires all 96 cells, exact within-route score bits across
cold/warm/scalar/map/one/eight workers, and complete clean warm rounds.
Refused or contention-flagged controls retain their attempts and do not pass.
The control does not replace the original library's parity gates.

The 96-cell 512 supplement waits for the original NN512 GPU stage. The
576-cell larger supplement waits for both its own 512 stage and the original
final larger GPU stage. On Mac, “GPU” here means the existing NN Metal stage;
dvifmish itself is CPU work. Each service uses the same original host heavy
lock, 12-GiB wrapper budget, eight-job wrapper cap, per-cell one/eight-worker
budgets, watchdog and dead-ticket recovery. Linux records external waits in
the supplement's `.service/queue-wait.json`; Mac uses the existing LaunchAgent
dependency mechanism. Neither supplement becomes an NN512 predecessor.

## i265 qualification observed 2026-10-18

Source `121ecf3ff3f7536e2b12499302a2bb8b7875a1f3` produced the frozen
worker SHA-256 `8d4204a6b70321b70caf1bb70e2133af4fe1f7adddc993c903997f28b6eef228`.
The real 512 control completed 96/96 cells and 240 clean warm rounds, with
exact within-route bits across all phases, modes and worker budgets. Its
retained report is `~/benchmark-results/iqa-i265-20261018-dvifmish-control/control.json`,
SHA-256 `e37d2efac512fb7429caeb82b47a998cce182a5229ccd28324ed1ac39797f729`.

The private dependency overlay implementation passes all nine driver tests
and strict driver clippy. The resource wrapper recorded test rc=0, 34 s,
peak RSS 0.93 GiB, minimum available 24734 MiB and peak load 3.95; clippy
rc=0, 8 s, peak RSS 0.54 GiB, minimum available 25462 MiB and peak load 3.51.
These are build/test resource receipts, not metric memory measurements.

Resolve current scheduling with `systemctl --user status iqa-durable-4018449423e7.service`
and `python3 benchmarks/iqa_compete_2026-10-16/linux_service.py status
~/benchmark-bundles/iqa-i265-20261018-dvifmish/sequence.json`. At 11:59 UTC
the 512 supplement had five successful cells and 91 pending; larger was
queued. The sequence freezes 96 and 576 cells respectively. Original coverage
inputs and attempt ledgers remain separate.

The first Mac control completed all 96 cells and matched all six i265 score
bit patterns, but nine of 240 warm rounds carried CPU-load flags with zero
heavy processes. It remains unqualified and retained. A fresh control can use
`--cooldown-seconds 2`: an explicitly recorded wrapper waits after acquiring
the shared lock, before each attempt and outside every metric clock. It does
not change admission thresholds, sampled-round gates or required clean rounds.
The wrapper hash and cadence are bound in the resulting control receipt.
This addresses prior-holder activity as an experiment; it does not establish
the cause of the original flags or guarantee a clean retry.

## Resource-accounting diagnostic

`resource-accounting/` beside the harness is a standalone, unpublished binary
using sysinfo 0.36.1. It repeats the pinned zenbench refresh order and records
raw per-core global use, owner percentage, cumulative owner CPU milliseconds,
interval deltas, nonzero process observations and independent native `ps`
cumulative CPU times. Idle, persistent-eight-worker and short-lived-worker
panels distinguish startup, steady activity and thread churn. Mac runs retain
native `iostat` user/system/idle and BSD `time` user/system totals as independent
observations. These native samples have their own displayed cadence; they are
not claimed to have exactly the same endpoints as each 200-ms sysinfo interval.
No diagnostic output qualifies timings or changes the benchmark gate.

Build with `just iqa-resource-accounting-build`. Under the existing shared
heavy lock and `run-heavy --mem 12G --jobs 8`, run
`just iqa-resource-accounting-run BINARY NEW_OUTPUT`. The source and standalone
lock belong to this repository; compiled binaries and raw observations remain
outside git. The diagnostic never replaces an existing competition binary.

The native Mac diagnostic completed all 120 synthetic observations without a
residual above 0.20: idle 0.11227–0.18625, stable eight-worker
0.09798–0.15114, churn eight-worker 0.09637–0.15834. This does not reproduce
the metric flags or prove an accounting defect. Raw Mac observations and
native receipts remain under `~/tmp/iqa-dvifmish-mac/accounting-v1`.

For an actual frozen warm worker, use `just iqa-resource-accounting-external
DIAGNOSTIC WORKER CONFIG NEW_OUTPUT` under the same lock/resource wrapper.
It copies the supplied config unchanged, binds executable/config hashes,
launches that worker and samples its PID until exit (at most 300 observations).
Worker output and Mac native `iostat` are retained separately. External
sampling adds diagnostic activity; none of these rows qualifies benchmark
performance. The original binary, config and results are untouched.

Local Cargo configuration can inject unrelated sibling patches into this
standalone build. The Mac `--locked` root invocation refused that inherited
resolution. An exact source/lock copy outside the checkout built unchanged
under the shared lock; keep a hash receipt for that copy. The original IQA
binary and snapshot remain independent. Linux external-worker diagnostics
also require the established P-core process boundary; an unrestricted
invocation is expected to refuse rather than qualify any observation.

An internal raw-counter diagnostic can copy pinned zenbench with
`patch_resource_trace.py PINNED_ZENBENCH NEW_COPY --snapshot OWNED_SNAPSHOT
--original SOURCE_CHECKOUT`. First build that dedicated snapshot with the
normal explicit dvifmish builder; then apply this override and rebuild its
worker with the same feature set and `--cfg iqa_compete_dvifmish`. The patch
records original/adapted source, manifest and lock hashes. Setting
`ZENBENCH_RAW_CPU_TRACE` to a new output path records each internal refresh
before foreign-load subtraction, including process identity, name, raw CPU,
cumulative CPU and owner classification. No threshold or gate logic changes.
Trace I/O adds diagnostic activity; this binary cannot qualify timings or
replace any frozen worker. The normal checkout and pinned zenbench source
are read-only inputs, with changes confined to the new copies.

The native internal-trace run identifies real foreign Spotlight work in that
diagnostic: `mdworker_shared` accounts for two to six heavy processes. One
refresh reports global 0.5589434767 and owner 265.5517273% on 12 logical CPUs,
yielding residual 0.3376503706; the next reports global 0.8280303160 and owner
595.0332031%, yielding 0.3321693134. Both match the worker's recorded warm-round
gate observations exactly. All ten rounds are flagged in that diagnostic.
This establishes foreign contention in this run, not an owner-accounting bug
or the cause of every earlier flag. Thresholds and original attempts remain
unchanged. The diagnostic binary hash is
`9ea9b1ec685cc0d9f21017adbf23e0fa7ad048b7c065d9865213472d96819c89`;
its frozen config still names the original worker, so the separate actual-binary
receipt is mandatory. Native raw evidence lives at
`~/tmp/iqa-dvifmish-mac/accounting-internal-1`.

A separate passive quiet check resolved a foreign high-CPU PID to the original
NN CPU controller, rather than its metric worker. That frozen controller
hashes active model assets before acquiring the shared heavy lock, so its
pre-attempt work can overlap another admitted benchmark. Maintained
`durable.py` now places per-attempt asset, manifest, binary and pair rehashing
inside admission and outside metric clocks. An OS-backed control requires
hashing to wait while another file descriptor holds the lock; the historical
controller fails that control. Frozen bundles are not edited or silently
upgraded. Startup manifest verification remains distinct from this per-attempt
fix; original running controllers retain their frozen behavior.

`--protected-window --window-settle-seconds 5` holds the existing shared host
lock across preparation, all 96 control cells and final verification. A
private inner attempt lock avoids acquiring the same host lock twice.
Requested/acquired/settled/releasing timestamps are recorded; quiet settling
and optional per-attempt cadence stay outside metric clocks. The child
controller inherits the held outer descriptor. This finite control window
preserves the original NN controller and dependency graph, while preventing
its repeated pre-lock hashes from interleaving between control cells. The
original gate still rejects any foreign activity that remains during rounds.
Use a new output directory and retain every earlier flagged control.

## Captured Mac owner undercount and private correction

A later protected internal-only diagnostic reproduces a distinct accounting
fault without an external observer. Owner PID 35262 advances cumulative CPU
1490→2353 ms over 202.086 ms (427.045911%), but sysinfo 0.36.1 reports
10.355022%. Global 0.4857978185 then leaves false residual 0.4771686331,
matching flagged warm rounds exactly. The next interval advances 889 CPU ms
across 208.526 ms while reporting 10.665134%, again matching the false residual
0.4883450445. These observations do not erase the separately measured genuine
Spotlight and controller contention.

Zenbench's private correction is committed at `3b1d0f71`. It computes Mac
process percentages from cumulative CPU milliseconds over monotonic intervals,
binds history to PID/start time, subtracts zero for missing/reset/reused owner
history and retains unknown foreign percentages plus global load. Counter
rates also preserve foreign heavy-process detection. Gates, Linux arithmetic
and public API are unchanged. All 34 platform controls, strict clippy and ARM
macOS cross-check pass; corrected native IQA qualification is still separate.

`IQA_MAC_CPU_COUNTERS=1 IQA_DVIFMISH=1` enables this correction only in the
owned benchmark snapshot. `patch_mac_cpu_counters.py` requires the exact frozen
`ac675d3` platform hash and copies the private counter module from zenbench
`3b1d0f71`; it does not import unrelated later zenbench changes. The source
module here is an exact copy from that commit. The overlay retains original /
adapted platform, module, snapshot manifest and lock hashes. Build receipts
validate the full committed-source → private dvifmish dependency → counter
correction hash chain. New binary hashes require new build receipts, frozen
bundles, controls and supplemental service plans. Old failures and original
manifests cannot be relabeled or overwritten.

## Qualified supplements and observed queues

The corrected native Mac worker is SHA-256
`c4d49bd53dbdef9d88dc664b547af49221c18e0e6b337d5a84d0bacf71a3bd19`,
source `3ca65f64d42cea44a30f4468092004f44bfa91fb`, snapshot export
`ffe384996876d2cd988ea3bd124f0c5b3100a337`. Its build receipt verifies the
committed-source / dvifmish / private-counter overlay chain. Native build
rc=0, 102 s, peak RSS 1.33 GiB, minimum available proxy 15922 MiB, peak load
8.85. The Mac guard is monitored RSS, not a cgroup memory-cap claim.

Its protected 512 control passes 96/96 cells and 240/240 clean warm rounds.
Report SHA-256 is
`33d974ce5407120d5fcfd5892406218c18b04f7ec80daefed4e0b3f4e46c1b18`.
All six score-bit patterns exactly match the clean i265 control; that fixture
agreement does not establish general cross-host perceptual parity. The i265
control also retains 96/96 `enforced_uniform` P-core cpuset proofs, CPUs 0–7,
12-GiB memory maximum and swap maximum zero. Its production 512 supplement
passes 96/96 with 240/240 clean warm rounds. A later separate i265 protected
control completed 96 cells but flagged one `HeavyProcesses(2)` round; that
attempt remains unqualified and does not replace the clean control.

Both hosts have separate frozen 96-cell 512 and 576-cell larger supplements.
On i265, resolve `iqa-durable-4018449423e7.service`; 512 is complete and larger
currently waits for the original final larger NN CUDA stage. On Mac the
corrected bundle root is `~/benchmark-bundles/iqa-mac-20261018-dvifmish-counters.noindex`.
LaunchAgents `io.imazen.iqa-durable.0ebeb356ba7cc6fb` and
`io.imazen.iqa-durable.d214eb85bb2ce49e` are installed and observed running.
The former records a wait for original NN512 Metal; the latter records waits
for original final larger NN Metal and corrected own 512. Both actual durable
commands retain 12G, eight jobs, 1100-second attempts, two attempts and the
same `~/tmp/mac-heavy.lock`. Original NN dependencies remain unchanged.

Resolve Mac state with `launchctl print user/$(id -u)/LABEL` and each output's
adjacent `.service/queue-wait.json`. These are dated observations, not promises
that counts remain unchanged. Retained Mac qualification/service receipts live
under `~/tmp/iqa-dvifmish-mac/`, and the control under
`~/benchmark-results/iqa-mac-20261018-dvifmish-counter-control.noindex`.

The maintained installer omits unsupported install-time `--memory`; the frozen
Mac service already supplies `--memory 12G` to durable execution. The regression
now exercises the actual service CLI and generated commands for both phases,
including memory, jobs, lock and dependencies. No frozen bundle edit is needed.

### Corrected Mac 512 production completion — 2026-10-18

An independent read of the actual production `report.json` verifies
`complete=true`, `qualified=true`, all 96 cells `ok`, and all 240 recorded
warm samples `gate_clean=true` (72 cold rows, 24 ten-round warm rows).
This is the production supplement, separate from the earlier strict control.
The report remains at
`~/benchmark-results/iqa-mac-20261018-dvifmish-counters.noindex/512-dvifmish/report.json`
on mac, with coordinator copy
`/home/lilith/benchmark-artifacts/iqa-dvifmish-mac-20261018/corrected-production-512-report.json`.
SHA256: `06c5f1635c5834eb91b65fcb4b97ccc4b33d61556a5d22afa2bbc1dd1c242725`.
The larger supplement retains its original dependencies; no service changed.

### Operational priority update — 2026-10-18

Owner direction now makes all dvifmish supplements eligible alongside the
non-NN work. This supersedes the earlier larger-supplement final-NN wait;
it does not change the original NN sequence, frozen workloads or shared locks.

On i265, the original supplemental service `iqa-durable-4018449423e7.service`
was verified waiting only for larger NN CUDA, with no child and no larger
measurement output. It was stopped cleanly (inactive, MainPID 0). The original
plan and frozen 576-cell manifest remain byte-identical. Fresh plan
`~/benchmark-bundles/iqa-i265-20261018-dvifmish/larger-priority-sequence.json`
installs `iqa-durable-05438c23f86b.service`, writing fresh
`~/benchmark-results/iqa-i265-20261018-dvifmish/larger-dvifmish-priority`.
Its sole predecessor is its completed, qualified own 512 stage. Actual
`queue-wait.json` records no pending predecessors and the service launches the
larger controller with the same shared `i265-heavy.lock`, 12G, 8 jobs,
1100-second attempts and two-attempt limit. Frozen P-core placement is unchanged.
Resolve current status with `systemctl --user status iqa-durable-05438c23f86b.service`
and the fresh output's `progress.json`; service activity alone is not a score.
Installation receipt:
`/home/lilith/benchmark-artifacts/iqa-dvifmish-i265-20261018/larger-priority-install.json`.

The Mac coordinator likewise replaced the never-started old larger service
`io.imazen.iqa-durable.d214eb85bb2ce49e` with
`io.imazen.iqa-durable.4160e2ff4d8706e3`, same frozen corrected 576-cell bundle,
fresh `larger-dvifmish-priority` output, sole completed-own-512 predecessor,
and the same shared Mac lock and 12G/8-job policy. Its supplied installation
receipt is `/home/lilith/benchmark-artifacts/iqa-dvifmish-mac-20261018/larger-priority-install.json`;
Mac measurement progress is verified separately by that coordinator.

NN data already exists on i265: both completed 512 CPU and CUDA reports have
128 successful and 56 failed cells, with 96 cold rows and 32 warm rows each.
Failed cells cover A-FINE, AHIQ, CKDN, DreamSim, HyperIQA, LIQE and PaQ-2-PiQ;
they remain failed. At priority installation the live larger NN CPU progress
had 138 successful cells and 966 pending. Its initial `report.json` still said
1104 pending, so live progress must come from `progress.json` and cell outputs
until the controller writes its final report. The original larger non-NN
report is complete: 5354 successful, 2688 unsupported, 118 failed cells.
A hashed availability snapshot is retained in
`/home/lilith/benchmark-artifacts/iqa-dvifmish-i265-20261018/priority-nn-availability.json`.

### Scalar-only scheduling supersedes the larger-priority map queue

The subsequent owner direction schedules no new map scopes. Completed scalar
cells and map attempts stay in their original outputs. The [scalar priority
contract and installation receipts](IQA_SCALAR_PRIORITY.md) replace the two
larger-priority services with immutable scalar-only sequences on both hosts:
fastest-ten baseline, all six dvifmish routes, additional named metric families,
remaining scalar work, then NN CPU/GPU. Retained failure budgets and the shared
locks are unchanged. The original bundles and predecessor files are historical
inputs; the active NN waits follow the new scalar outputs.
