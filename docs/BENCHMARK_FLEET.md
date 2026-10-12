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
