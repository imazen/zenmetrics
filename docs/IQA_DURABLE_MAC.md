# Durable Mac CPU timing

The durable runner separates process recovery from measurement acceptance. A
failed, interrupted, timed-out or incomplete attempt remains in the record and
cannot become a timing sample. Completed cells survive controller restart;
remaining cells run before failed cells receive their bounded retry.

## Start on a fresh Mac

Copy a prepared ARM64 bundle (including its `tools/` directory) onto the Mac.
With this repository available, start it with:

```
just bench-iqa-mac "$HOME/benchmark-bundles/iqa-mac-20261018-core" "$HOME/benchmark-results/iqa-core"
```

Newly prepared bundles also contain their own justfile, so a repository checkout
is unnecessary:

```
just --justfile "$HOME/benchmark-bundles/BUNDLE/justfile" run "$HOME/benchmark-results/iqa-core"
```

Use an absolute output path. Installation starts the launchd service immediately;
it continues after the terminal disconnects. In the repository command, append
`status`, `stop`, or `start` to inspect, stop, or resume an installed service.
With the bundle justfile, use `status`, `stop`, or `resume` instead of `run`.
Reuse the same bundle and output for resume. `run` installs a new service and
deliberately refuses an already installed one.

### Dependencies and reproducibility

| Path | Requirements |
| --- | --- |
| Run a prepared bundle | Compatible macOS ARM64; Python 3.10+ standard library; `just`; macOS `launchctl`, `caffeinate`, shell and system resource utilities; a logged-in user session; writable results and scratch directories. |
| Resource protection | The bundle must include the explicitly supplied `tools/run-heavy` and its macOS companion. This wrapper is separately maintained and is not downloaded by this command. Its process-tree RSS guard is not a kernel hard memory cap. |
| Prepare from source | Repository checkout, `jj`, Git, Rust/Cargo and a working native linker/toolchain, plus access to fetch the Cargo.lock dependencies and pinned sibling repositories. Supply the built worker, exact RGB pairs, input provenance and build receipt to `iqa-durable-prepare`. |

No pip packages, neural model weights, Cargo, or network access are needed to
execute the copied bundle. The runtime was exercised on macOS ARM64 with Python
3.14.7; the syntax floor is not a claim that every older macOS/Python combination
has been tested. The executable's macOS deployment compatibility still applies.

The manifest freezes input bytes, worker binary, metric selection, dimensions,
thread counts and cold/warm policy. Bundle checksums cover the scripts and copied
resource wrapper; attempts record the actual machine and execution command.
Resume refuses changed identities. Preserve `build.json`, `SHA256SUMS.json`, the
manifest and all result attempts together. Checksums detect changes against the
supplied inventory; authenticate the bundle's origin separately when copying it.

This reproduces the workload and executable, not identical timings across Macs.
OS version, CPU, power/thermal state and background load remain experimental
variables. A source rebuild is pinned through Cargo.lock and sibling revisions,
but is not certified bit-for-bit reproducible across compiler/linker versions.
Cold means fresh scorer construction, not flushed OS caches. The frozen S67
panel measures scaling on one pair, not perceptual ranking accuracy.

## Timing contract

Each cell fixes metric, input hashes, dimensions, thread budget, mode and phase.
Scalar and scalar-plus-map calls use separate processes. Unsupported maps are
explicit terminal observations, not zero-duration measurements.

- **Cold:** a fresh worker process loads both RGB inputs, then times scorer
  construction and its first evaluation. No scoring or map warmup precedes the
  timer. Input loading is recorded separately. Scorer teardown, process startup
  and OS file-cache flushing are outside this timer. Three independent cold
  processes are scheduled by the bundle default.
- **Warm:** initialization and warmup precede ten zenbench rounds, with a
  100 ms target per round. Raw round samples are retained. A cell that cannot
  finish all ten rounds within its scope has no accepted warm result.
- Controller elapsed time covers process orchestration and resource-wrapper
  overhead. It must not be presented as compute time or pure process startup.

The default thread budgets are explicitly 1 and 8, leaving capacity for the
system on the validation Mac. They are not called “all cores.” Choose other
budgets explicitly when preparing a bundle and retain that manifest.

A successful worker protocol observation is not independent oracle parity or
proof of an uncontended benchmark. Preserve load observations and resource logs;
qualify metric correctness and environmental conditions before publication. The
existing VMAF v1 macOS chroma discrepancy remains a separate correctness issue.
Do not promote that route by virtue of completing a timing run.

## Prepare and run

Build from the pinned snapshot, with the host resource wrapper and shared heavy
lock. Do not use `target-cpu=native`. The worker needs these features:

```
just snap-cargo @ build --release -p cpu-profile --features compete,metrics-parallel --bin iqa-compete
```

`just iqa-durable-prepare BINARY PAIRS OUTPUT REVISION --build-receipt BUILD_JSON`
freezes the binary, controller, launch helper, exact RGB input bytes, source/build
receipt and hash manifest. `BUILD_JSON` must identify the binary SHA-256, source
commit and actual feature set; retain compiler, dependency pins and build command
there too. `--resource-wrapper /absolute/path/run-heavy` also freezes a local
copy of the resource wrapper and its macOS implementation under `tools/`.
Optional `--metrics psnr,gmsd` and `--sizes cid22,aic-small` select a
subset. `--threads 1,8` and `--cold-repeats 3` are explicit defaults.

The default input provenance is the existing S67 scaling panel. The 4K and 8K
inputs are mirrored derivatives, explicitly labeled. This is a performance
scaling study of one reference/distortion pair. It does not measure perceptual
ranking accuracy or provide a diverse-content meeting benchmark. A future
corpus needs its own frozen provenance and unchanged correctness gates.

For a different subset after copying a bundle, no build tools are needed:

```
python3 BUNDLE/select_durable.py BUNDLE SUBSET --metrics psnr,gmsd --sizes cid22,aic-small --threads 1
```

This makes a separate hash-verified bundle and retains both timing phases. It
refuses unknown names and leaves the source bundle and existing results intact.

Copy the bundle as a directory. It requires a compatible ARM64 Mac, Python 3,
and the explicit native `run-heavy` resource wrapper. It does not require Cargo,
model downloads or a repository checkout. Keep all outputs outside the immutable
bundle. For foreground execution and exact resume, run the same command:

```
python3 BUNDLE/durable.py run BUNDLE/manifest.json RESULTS \
  --run-heavy /absolute/path/run-heavy --lock "$HOME/tmp/mac-heavy.lock" \
  --memory 12G --jobs 8 --timeout-seconds 1100 --max-attempts 2
```

For unattended controller restart on macOS:

```
python3 BUNDLE/mac_service.py install BUNDLE RESULTS \
  --run-heavy /absolute/path/run-heavy --lock "$HOME/tmp/mac-heavy.lock"
python3 BUNDLE/mac_service.py status BUNDLE RESULTS
python3 BUNDLE/mac_service.py stop BUNDLE RESULTS
python3 BUNDLE/mac_service.py start BUNDLE RESULTS
```

Service configuration and stdout/stderr live in `RESULTS.service`, beside the
measurement output. `progress.json` is the live compact cell count; `report.json`
contains full phase rows after completion, interruption, or a status command.

The default launchd domain is `user`, which supports SSH-only sessions; `--domain
gui` is explicit. The service holds an idle-sleep assertion while running and
throttles restarts. A missing controller heartbeat triggers an owned-child kill
and automatic restart; worker deadlines are supervised independently. Completion and permanent integrity/configuration failure stop
restarts; inspect the saved report and service stderr to distinguish them. A
LaunchAgent does not execute before the user session exists after a reboot.
Hardware failure, disk failure and forced power loss can interrupt the current
attempt; they cannot be promised away by a software watchdog.

Never edit a manifest in place to resume a different experiment. Retain original
attempt directories and logs, including failures. Bundle integrity failures
require fixing the input/setup, not accepting a new hash into an old run.

## Sequential families and neural exports

Install the next bundle with an explicit predecessor:

```sh
just bench-iqa-mac NEXT_BUNDLE NEXT_RESULTS install --after PREVIOUS_BUNDLE PREVIOUS_RESULTS
```

Repeat this for remaining non-neural methods, then NN CPU, then NN Metal.
The next service waits for the predecessor's frozen manifest identity, complete
terminal cell inventory and released controller lock. Exhausted failures remain
failed in that report and do not block the next family. An incomplete report or
integrity refusal does block it. Pending dependencies are visible in
`NEXT_RESULTS.service/queue-wait.json`. Every family still uses the same heavy
lock and per-attempt watchdog; waiting services perform no measurements.

`just iqa-durable-nn-prepare TEMPLATE BINARY BUILD INVENTORY MODELS OUTPUT ENGINE`
freezes NN scalar cells using an existing input bundle. `MODELS` is a JSON map
from metric names to caller-supplied export directories. The native-host build
receipt must bind both binary and inventory SHA-256. `ENGINE` is `rten` for CPU
or `ort-webgpu` for Metal; the latter requires `--ort-library` and
`--webgpu-library`. Restricted weights retain the scorer's explicit
`--accept-weights` contract. This preparation step requires the Python `onnx`
package to discover external tensors, including Constant-node tensors whose
filenames have no extension. The resulting bundle requires only Python stdlib
and the included native files at execution time.

Model graphs, metadata, heads, external tensors and supplied runtime libraries
are hashed into the manifest. Each attempt rechecks its active model/runtime
files before and after execution. Its resolved engine/environment and asset
hashes travel with the retained config and row. NN routes are scheduled for
scalar scores because this adapter has no qualified NN native-map route.
Metal uses the existing ORT WebGPU engine, which disables CPU fallback; runtime
or resource refusals remain failed attempts, never CPU timings labelled Metal.
These records qualify the timing protocol, not numerical parity for every model.
The normal NN group uses the shared slow-metric selection policy: PieAPP is
excluded unless named explicitly with `--metrics`. Its implementation remains
available; a routine group does not spend the remaining machine window on it.

The controller records a worker's PID, process group and start time before exec.
After verifying that its entire group has exited, it can archive that worker's
orphaned zenbench admission ticket and preserve a backup beside the original.
Live owners, live groups, other owners, old PID records and locked ticket guards
are retained. Both Linux process inventory and macOS `ps` process inventory are
covered by the admission-recovery controls. This cleanup changes scheduling
state; interrupted samples remain unqualified.

Controls: `just iqa-durable-check` and, with a preparation interpreter containing
ONNX, `just iqa-durable-nn-check /absolute/path/to/python`.
`just iqa-nn-timing-check BUNDLE NEW_OUTPUT` runs a real LPIPS-Alex cold/warm
control at 512x512 through the same supervisor and requires identical score bits
within the selected engine. It retains both rows and all attempt diagnostics.

`just bench-iqa-mac-progress BUNDLE OUTPUT --other-output NEXT_OUTPUT` reads
compact progress, separately counts failed attempts and exhausted cells, lists
successful cold/warm rows and reports queued predecessors. It does not start a
measurement. This avoids mistaking zero exhausted cells for zero failed attempts.

For an operator check-in, `just bench-iqa-mac-check-in PLAN LOG_DIRECTORY
--agent-pane PANE_ID` reads a local JSON plan with `machine` (an SSH alias),
`script` (the remote `mac_service.py`) and `phases` (objects containing absolute
`bundle` and `output` paths). It saves each SSH observation, including timeouts,
and shows a Herdr notification before attempting to prompt the existing agent.
A blocked agent prompt therefore does not suppress the machine status record or
notification. A half-hour user timer can invoke this command; once every phase
is terminal the check-in stops `iqa-mac-checkin.timer`.
