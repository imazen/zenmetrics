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
