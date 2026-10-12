# Benchmark observations through zenfleet

Use the existing declare → gap → worker → ledger flow. The `benchmark` job kind
binds the bundle, installed runner SHA-256, experiment, machine/testbed, CPU
domain and planned replicate. The same code on a different machine is a different
job. A retry has a new attempt ID; never select its fastest result.

Build a native zenbench executable per target and a separate latency or memory
driver. [Zenbench's bundle/driver contract](https://github.com/imazen/zenbench/blob/main/docs/fleet.md)
specifies paths, correctness, memory quantities and result files. Pin the runner
and bundle hashes; a version string alone does not identify executable bytes.

## Stage and declare

Set these on the staging host and worker, using deployment-specific values:

```sh
export ZENBENCH_BIN=/opt/zenbench
export ZENBENCH_WORK_DIR=/scratch/benchmark
export ZEN_BLOBS_DIR=/scratch/blobs
export ZENBENCH_BLOB_URI=s3://BUCKET/RUN/blobs
# Use the same scoped credentials and endpoint as the existing fleet store.
export ZENBENCH_MACHINE=cpu-testbed-a
export ZENBENCH_DOMAIN=performance
export ZENBENCH_COORDINATION_DIR=/shared-local/zenbench
scripts/jobsys/benchmark_exec.py stage deployed-bundle > identities.json
scripts/jobsys/benchmark_exec.py capabilities
```

The domain must agree with the runner's verified CPU class. `uniform` accepts a
verified uniform machine. `unrestricted` requires explicitly setting
`ZENBENCH_CPU_POLICY=unrestricted`; its observations retain that limitation.
Containers on one host must share the coordination bind mount. CPU restrictions
must cover the worker and descendants before pools initialize. Machine identity
is an operator-supplied testbed ID, not a detected serial number. The runner also
records its process and reported machine architecture; `native_arch_verified:false`
is not proof of native execution. Check the deployed binary architecture against
the host before enrollment, especially on Windows ARM64 and emulated containers.

Use the two hashes from `identities.json` in this declaration:

```json
{"schema":1,"observations":[{"job":{
 "kind":"benchmark","bundle_sha":"BUNDLE_SHA256","runner_sha":"RUNNER_SHA256",
 "experiment":"study-1","machine":"cpu-testbed-a","domain":"performance",
 "replicate":0,"backend":"heaptrack","timeout_seconds":300,"admission_seconds":600
},"peak_mem_bytes":4294967296,"threads":8}]}
```

```sh
zenfleet-ctl declare-benchmarks --spec observations.json --out manifest.jsonl
```

Declare every machine/domain/replicate explicitly. Duplicate job IDs are refused.
The backend and execution timeout must match the bundle's plan. Resource hints
feed the worker's existing admission logic; they are not OS memory limits. Apply
container/cgroup memory and CPU limits in the existing deployment configuration.
No benchmark cell is eligible for a persistent warm executor, even with `all`.
Long-running benchmark chunks renew their lease while the worker lives.

Set `ZEN_EXEC` to the absolute path of `benchmark_exec.py` in the existing worker
configuration. The adapter uses the fleet's content store; it adds no scheduler,
lease service or retry loop. `stage` verifies and uploads the manifest, listed
files and runner. Workers verify every blob before materializing a fresh bundle.
With `ZENBENCH_BLOB_URI` absent, the same protocol uses only `ZEN_BLOBS_DIR` for
local controls. Remote transfer uses the baked `s5cmd`.

## Outcomes and recovery

A ledger `Done` row means an observation was durably captured. It does **not** mean
valid performance evidence: inspect `status` in the observation packet. Busy,
wrong output, cancellation and timeout are retained outcomes, not regressions.
This prevents automatic execution retries from selecting a lucky measurement.
A transfer/worker failure is still a fleet failure and can be retried, with a new
attempt. The adapter preserves local bundles, streams, packets and receipts.

Signals delivered to the adapter request runner cancellation. If its worker
parent dies, the adapter also requests cancellation, reaps through the runner and
stores the receipt when the OS leaves it alive. An entire container kill cannot
guarantee a final receipt: mount scratch durably and allow a shutdown grace
period. Abnormal supervisor death leaves a fail-closed reservation; inspect it
before clearing it. A worker watchdog is a final backstop, not successful cleanup.

Fetch the worker output blob through the existing ledger tooling, then:

```sh
scripts/jobsys/benchmark_exec.py collect RECEIPT_OR_SHA recovered-attempt
zenbench fleet-read recovered-attempt
zenbench fleet-diff recovered-before recovered-after
zenbench fleet-export study.jsonl recovered-before recovered-after
```

The adapter also stores its compact receipt in
`ZENBENCH_WORK_DIR/receipts/ATTEMPT.json`, prints its blob hash to stderr and uploads
it. `collect` accepts that file or blob hash, verifies artifact hashes and lengths,
and refuses to overwrite a destination. Keep raw traces in the content store;
commit one compact JSONL study file when a version-controlled record is needed.
Fleet comparisons are descriptive; use `zenbench pair` for latency inference.

## Validation

```sh
just benchmark-contract-check
just benchmark-executor-check /path/to/zenbench NEW_CONTROL_DIRECTORY
```

The first checks Rust/Python identity agreement, routing and fresh-worker policy.
The second runs actual bundle staging, child execution, correctness mismatch,
timeout, cancellation, wrong-machine refusal and hash-verified recovery. It uses
small synthetic controls and makes no IQA performance claim. Heap accuracy is
separately checked by zenbench's `just heap-oracle NEW_DIRECTORY` against heaptrack.

## Baked Linux worker image

Use the existing image builder's benchmark variant; no boot-time installs:

```sh
BENCHMARK_IMAGE=1 BENCHMARK_BASE=BASE_IMAGE_DIGEST \
ZENBENCH_BIN=/path/to/prebuilt/zenbench \
ZEN_WORKER_BIN=/path/to/prebuilt/zenfleet-worker \
bash scripts/jobsys/build_executor_image.sh ghcr.io/imazen/zenfleet-worker:bench-REVISION
```

The base must match the target architecture and runtime libraries of both
prebuilt binaries. Build-time executable checks reject an incompatible libc.
Build the benchmarked driver for that same runtime. Dynamically linked workload
allocations can be intercepted by heaptrack; a static allocator is outside that
interception coverage. The image includes Python, blob tools, heaptrack, its
analyzer and OS time. Set `PUSH=1` only when shipping the verified image. The
builder retains its context and SHA-256 provenance. Mount scratch and the local
coordination directory, set CPU and memory limits, and use the existing fleet
launcher to deploy the image by digest. Run the tiny observation control inside
the actual image before enrolling it for a study.

Benchmark jobs use lease claiming with positive chunk duration (the worker
currently defaults to 300 seconds). Epoch-sharded claiming and serial R2 claims
are refused for this job kind because they do not provide its required renewal
contract. Local execution also has a bounded worker watchdog. Receipts label
all nested evidence as `not_regenerable` and list its hash references; include
those references when constructing campaign archive/GC roots.

## IQA driver phases

`new-metrics-wall fleet CASE.json` runs one SDR CPU metric through the existing
CLI dispatch, without timing calibration or batching. Build a normal binary for
latency/heaptrack/OS counters, and a separate `--features heap-phases` binary for
Rust heap phases. The instrumented build refuses the old timing entry points;
its output says `timing_usable:false`.

```json
{"schema":1,"metric":"ssim2","width":65,"height":67,"stride_bytes":202,
 "reference":"reference.rgb","distorted":"distorted.rgb",
 "pixel_format":"rgb8_srgb","reuse":"warm_reference","threads":2}
```

Raw files contain exactly `stride_bytes * height` bytes. Paths resolve beside
the case file. Padded rows are supported; packing for the CLI's packed RGB8
dispatch belongs to `input_and_pool_preparation`, as does pool initialization.
This driver measures decoded sRGB8 inputs; decoding, other transfer functions,
HDR and GPU execution require another driver and must not reuse this case label.
Include both inputs and the case in the bundle so hashes bind their contents.
Set `RAYON_NUM_THREADS` to the declared value and supply `ZENBENCH_OPERATIONS`
(1..1024). The driver refuses mismatches. Metric runtime SIMD dispatch remains
active; its effective tier is not independently observed.

`cold` supports the driver's existing `METRICS` list, including reference work
inside every score call. `warm_reference` currently supports **4 metrics**:
SSIMULACRA2, butteraugli, DSSIM and zensim. It calls the umbrella's actual cached
reference API, retains that state, performs one declared warmup score, then
measures the requested calls. Other warm cases are refused. CVVDP cold uses the
existing `standard_fhd` display. The output is the primary score, without a
retained diffmap. Case-specific correctness is an exact array of primary-score
float bits; supply the expected array from the matching validated latency driver
in the bundle's `expected_correctness` contract.

Rust memory records expose entry/peak/exit requested live bytes and allocation
traffic for preparation, reference preparation, score warmup, scoring and
reference release, where applicable. Peaks are absolute live-byte watermarks,
never divided by operations. Earlier inputs, retained references and harness
records remain part of the process ledger; differences are not ownership-based
scratch attribution. Native allocations, children, direct mappings and GPU
memory remain outside this allocator's coverage. Use whole-process OS/heaptrack
observations as separate views.

```sh
just iqa-fleet-build NEW_BINARIES_DIRECTORY
just iqa-fleet-check NEW_BINARIES_DIRECTORY NEW_CONTROLS_DIRECTORY
```

The build recipe retains distinct debug artifacts for protocol controls; build
release equivalents with debug symbols for a real study. Local controls on
2026-10-12 exercised all four warm-capable metrics, cold and warm reuse, one and
two Rayon threads, and packed/padded 65×67 inputs. Instrumented and ordinary
primary scores matched exactly, as did the packed/padded inputs. These controls
validate phase plumbing and output preservation; they are not representative
performance measurements or an allocator-overhead estimate.

The image builder requires Docker Buildx/BuildKit for executable-preserving COPY.
`just benchmark-image-check IMAGE ABSOLUTE_NEW_SCRATCH_DIRECTORY` exercises the
baked adapter and runner, then the baked worker's actual Parquet ledger and blob
recovery, including `ZEN_PERSISTENT_EXEC=1` with `ZEN_WARM_KINDS=all`.
A local Linux x86-64 image passed this control on 2026-10-12. This does not certify
an ARM image or the separate Windows/macOS worker deployment.

For a Windows worker, install Python and put `benchmark-exec.cmd` beside
`benchmark_exec.py`; point `ZEN_EXEC` at the absolute `.cmd` path. The shim
preserves the worker's JSON stdin contract. Use
`benchmark_exec.py cancel ATTEMPT` (with the same work directory) for portable
file-based cancellation; Windows process termination does not provide Unix
signal-handler grace. This packaging path still requires a native worker smoke
run on the target before enrollment.

`just iqa-heap-oracle MEMORY_BINARY CASE.json NEW_DIRECTORY` runs a real IQA case
under heaptrack and checks its global Rust-allocation contribution against the
ledger's lifetime maximum. The fixture needs debug symbols containing
`HeapProfiler`. This is a whole-process control, not an external validation of
individual phase peaks. Use the normal driver with the `heaptrack` backend to
observe intercepted native allocations as a separately labeled quantity.

Fleet IQA cases require relative input paths contained by the case directory.
When `ZENBENCH_RESULT_FILE` is set, the case and both inputs must also appear
in the current bundle manifest. Standalone cases omit that variable.
`just iqa-bundle-check BINARIES RUNNER CASE NEW_DIRECTORY` checks both real
IQA builds through the runner and refuses escaped or unlisted inputs.

## Validation and rollout

The control recipes are correctness checks, not a performance study. The real
IQA parity matrix covers four metrics, cold and cached-reference execution,
one and two Rayon threads, and packed and padded 65×67 RGB8 input. Each of the
32 cases requires exact score bits between latency and memory builds; each
padded case must equal its packed counterpart. This does not establish parity
for untested metrics, formats, sizes or target-specific SIMD paths.

The native CI workflow runs the adapter and a real worker with the persistent
executor option enabled, reads its Parquet ledger, recovers every artifact by
hash, and checks success, wrong output, timeout, cancellation and wrong-machine
routing. Linux image controls additionally exercise the baked executables.
These checks use explicitly unrestricted CPU placement and cannot produce
confirmatory benchmark claims.

Before enrolling a production worker, verify its native architecture, selected
CPU/cache domain, installed runner hash and workload outputs on that actual
host. Publish a target-compatible image through the existing image builder,
then run a small declared campaign and collect its receipts before scaling.
A CI result or local image control does not verify production credentials,
object-store permissions, target host configuration or image deployment.
