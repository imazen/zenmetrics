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
