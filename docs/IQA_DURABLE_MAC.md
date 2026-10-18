# Durable Mac CPU timing

The durable runner separates process recovery from measurement acceptance. A
failed, interrupted, timed-out or incomplete attempt remains in the record and
cannot become a timing sample. Completed cells survive controller restart;
remaining cells run before failed cells receive their bounded retry.

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
