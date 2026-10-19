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

### Coverage-first scheduling

To run every method at 512×512 before expanding sizes, prepare six fresh
bundles from the frozen core, remaining non-NN, NN CPU and NN Metal bundles:

```sh
just iqa-mac-coverage-prepare CORE REMAINING NN_CPU NN_METAL CORE_REPORT NEW_DIRECTORY
```

Install the phases listed in `NEW_DIRECTORY/sequence.json` in order, each
with the previous bundle/output as its `--after` gate. The order is 512 non-NN,
512 NN CPU, 512 NN Metal, then the three larger-size families. Larger sizes
are ordered by pixel count. Retained successful and unsupported core cells
are excluded only after validating their original attempt identities, configs,
log hashes and parsed rows. A stale pending report does not erase validated
attempt evidence. Old observations retain their original executable identity;
the new bundles bind the retained report hash and both executable hashes in
the sequence receipt. Failed attempts remain in their original output tree.
`just iqa-mac-coverage-check` checks retention and corruption refusal.

### Frozen WebGPU refusal classification (2026-10-18)

The strict 512 NN Metal queue's A-FINE, ARNIQA, CKDN and DeepDC failures are
reproduced session capability refusals. A read-only matching-runtime diagnostic
separates registry misses from supported nodes deliberately preferred on CPU;
A-FINE and DeepDC show both. The [node audit](../benchmarks/mac_validation_2026-10-18/WEBGPU_NODE_AUDIT.md)
retains exact observations and provenance. No fallback or frozen manifest was
changed. Original attempts remain unqualified and the queue advances.

The subsequent MUSIQ session audit reproduces the fifth strict Metal refusal:
92 registry-miss observations (Gather/Unsqueeze/Concat), zero explicit
CPU-preference messages. The [MUSIQ extension](../benchmarks/mac_validation_2026-10-18/WEBGPU_NODE_AUDIT.md#musiq-extension--same-frozen-runtime)
records exact names, unchanged graph/runtime hashes and separate receipts.
The four-model audit alone did not cover MUSIQ.

### Scalar-only priority replacement (2026-10-18)

The active graph now follows the [scalar-only priority contract](IQA_SCALAR_PRIORITY.md).
Its frozen manifests preserve completed cells/old maps and remaining attempt
budgets, and add no upcoming map scopes. NN CPU and Metal follow the replacement
scalar outputs; disabled original service configs remain unchanged. The existing
30-minute check-in plan reads the new frozen scheduler's `status` output and
retains a backup of the earlier eight-phase plan.

## Retained STLPIPS-Alex numerical refusal, 2026-10-19

The GPU-first Metal 2K warm attempts at one and eight workers each retain a
`variant_mismatch` row with three differing call results despite ten clean
resource-gate samples. The controller correctly leaves them unqualified; a
clean resource gate alone does not qualify score repeatability. The call
counter spans warmup/calibration/timed calls, so the exact varying timed rounds
are not identified. Original two-attempt caps and failed packets are retained.
[Sealed diagnostic evidence and scope](../benchmarks/mac_validation_2026-10-18/STLPIPS_VARIATION_2026-10-19.md).

### Finite Metal timing acceptance (owner direction, 2026-10-19)

The opt-in `mac-metal-finite-logged-v1` policy accepts finite score variation
for warm scalar ORT WebGPU Metal timings. It has no numerical error bound.
Accepted rows retain their raw status, observation count, baseline, finite
extrema, maximum absolute/relative deltas and an explicit numerical-instability
flag. This qualifies timing only; it does not establish native-oracle parity.
Nonfinite observations, incomplete rounds and resource-gate failures still
refuse qualification. CPU, CUDA, spatial maps and native parity keep their
existing numerical contracts.

The new worker counts every warmup, calibration and timed result in deferred
output destruction after the measurement clock. Baseline-inclusive extrema
and overflow-marked deltas are logged. Historical workers recorded only the
first alternate and final score, which cannot exclude a hidden intermediate
nonfinite value. Their failed attempts remain unchanged. A replacement worker
therefore requires a new binary hash, frozen bundle, protocol controls and
successor lineage with the original remaining attempt budgets.

The [native protocol/deployment receipt](../benchmarks/mac_validation_2026-10-18/finite-metal-policy-2026-10-19.json)
records six passing cells and 40 clean warm rounds. Both STLPIPS-Alex 2K warm
controls varied in three of twelve observed calls, including calibration;
maximum absolute delta was 0.0014443499967455864 and relative delta was
0.15962169409729945. Both timings were accepted and flagged. LPIPS-Alex's
512 controls remained stable. This does not identify the cause of variation.

The replacement executable is `c3dc4ddb5962549beff06ab419c63f296a85a5722f074e84155a3ddd21d43373`:
base source `3ca65f64`, tested snapshot `ffe384996876`, telemetry overlay
`5461e05e`. Metric crate sources and sibling pins are unchanged from the
original Metal worker source `7310d24c`; the base adds private DVIF routing and
corrected process accounting. The build explicitly enables
`zenmetrics-cli/nn-webgpu`; the reusable builder selects it with
`IQA_NN_ENGINE=ort-webgpu`. The earlier feature-omission build and its six
refused controls remain retained and consumed no production attempts.

The immutable `iqa-mac-20261019-finite-metal.noindex` sequence retains 298
qualified prior cells, then schedules 122 Metal cells with one remaining
attempt, 500 Metal cells with two remaining attempts, and the existing 694
CPU cells. Its LaunchAgent is `io.imazen.iqa-scalar.7ea9de0d2f40af9c`;
`io.imazen.iqa-window.372afe5c8da4e851` preserves the
2026-10-19T12:56:43Z endpoint. Both were observed running. CPU waits behind
both Metal phases; pending maps and 8K geometries are absent. The previous
coordinator and guard were disabled after the worker completed under the
shared lock. The 30-minute check-in plan points at the new sequence. Resolve
live progress with the frozen coordinator's `status sequence.json` command;
the receipt records the exact paths, hashes and actual dependency waits.

Both original STLPIPS-Alex 2K numerical refusals subsequently used their one
remaining production attempt in the new bundle. Both timings passed all ten
resource gates, were accepted with instability flags, and reproduced the
control maximum deltas. Their original failed packets remain unchanged;
the new packets explicitly report `score_repeatability_qualified=false`.

### Due-time correction and uninterrupted authorization (2026-10-19)

The owner clarified that the 15-hour window meant results were due, **not**
authorization to stop. The previous Mac endpoint guard stopped the CPU
coordinator at a worker boundary at 13:01:10Z. Its original plans, guard and
results remain unchanged. The immutable `iqa-mac-20261019-resume-no-stop.noindex`
successor resumed at 14:44:56Z: four CPU cells retain one remaining attempt,
then 544 retain two. GPU is already terminal; its finite-score policy and all
prior qualified lineage remain preserved. The new service is
`io.imazen.iqa-scalar.e5cedae3de4c5d74`. It has no endpoint condition or window
guard; the check-in plan now follows this sequence.

Maintained scheduler plans treat due times as reporting metadata. A cutoff or
new window guard requires explicit `stop_at_deadline=true`; absent/false does
not authorize a stop. Historical frozen schedulers keep their original bytes.

The i265 frontier was already terminal, including its two CPU groups, so
removing a cutoff alone could not create runnable attempts. The owner explicitly
authorized **two new attempts per consent-blocked cell** in a separate campaign.
`iqa-i265-20261019-consent-retry-no-stop` schedules 336 GPU cells then 256 CPU
cells, including affected 512 cells. Original attempts, original caps and
exhausted non-consent faults remain preserved. Neither successor schedules maps
or pending 8K geometries. Receipts and current blockers are linked from the
[resumption receipt](../benchmarks/mac_validation_2026-10-18/no-stop-resumption-2026-10-19.json).

### Explicit weight-consent propagation

The frozen i265 configs requested `ZENMETRICS_NN_ACCEPT_WEIGHTS=all`, but the
native CLI accepts named score columns and ignores unknown names. This was
benchmark preparation plumbing, not an unsupported implementation. Preparation
now expands an explicit `--accept-weights all` to the selected gated models'
recognized names and records requested/resolved consent in the manifest.
Empty acceptance still grants nothing; unknown explicit names refuse.
The native consent gate, metric code, binaries, weights and numerical/resource
gates are unchanged. The corrected campaign uses the existing binaries with
explicit names, and preserves the original missing-consent packets.

### Updated chart evidence, 2026-10-19

The [October 19 chart snapshot](../benchmarks/iqa_chart_snapshot_2026-10-19/README.md)
adds the later NN, finite Metal and consent-corrected observations. Schema 2 can
include completed sealed attempts before a running phase rewrites its report;
seals, experiment/cell identity, config/intent/log hashes and qualified rows are
verified without recovery or mutation. It retains source report and extra
result hashes, numerical-instability flags, native error text, engine and
explicit weight-consent names. Counts remain attempt counts, not distinct
coverage. The October 18 snapshot remains unchanged.

### Canonical CPU refresh, 2026-10-19

`just iqa-canonical-cpu-prepare` freezes existing native workers against the
committed RGB derivative hashes. It emits scalar scopes at one and eight
workers for six non-8K geometries: three fresh constructor-inclusive cold
processes plus ten warm zenbench rounds targeting 100 ms. RGB loading precedes
the cold clock. Portable 5/3-round observations remain supplemental. A changed
worker/source remains a separate series; preparing a bundle does not establish
compatibility with previous implementations or numerical oracle parity.

The i265 four-scope SSIMULACRA2 control passed. Mac’s initial four
rows had controller status `ok`, but its eight-worker warm scope contained
four flagged rounds. That is a strict qualification failure. The original
Mac campaign and binary remain preserved; `rebuild_canonical_mac.py` stops
its coordinator at the shared-lock boundary, rebuilds in a new snapshot
with the existing private cumulative-counter correction, and requires all
control rounds clean before installing a successor. No thresholds change. The r7900x campaign contains 95 compiled non-NN
routes, including six private DVIF variants. All use the existing durable
controller and per-host shared locks with 12G/8-job wrappers; i265 retains
checked zenbench performance-core placement. Mac's wrapper is a monitored
process-tree guard, not a Linux hard memory-cap claim. Native logs retain the
resource measurements. No NN CPU or GPU campaign was started. The due time is
reporting metadata and does not stop unfinished cells.

Immutable bundles and live output paths are recorded outside the repository in
`~/benchmark-artifacts/iqa-unified-20261019/STATUS.json`. Full coverage remains
pending while those controllers run. Previous attempts and frozen bundles are
retained, including unsupported input-contract refusals and flagged timings.

The chart exporter also projects the finalized worker-wrapper peak RSS from
its hash-bound log. Rounded GiB retain 0.01 GiB precision; native macOS BSD
time bytes are separate. Missing, ambiguous or changed logs produce null
memory fields, never zeros. These include worker input/setup and are not
metric-only heap or GPU VRAM measurements.
