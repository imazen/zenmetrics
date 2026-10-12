# Declaring metric cost studies

`just benchmark-study inventory` resolves CLI names and the fleet driver's
current cold/warm coverage from source. It does not infer platform support from
a feature name. `just benchmark-study plan SPEC.json` validates inputs and
prints exact ready/unsupported/unqualified counts without measuring anything.
`just benchmark-study write SPEC.json NEW_DIRECTORY` retains every cell and
writes bundle specs only for qualified cases. Existing directories are refused.

The study schema is 1. All fields below are required except testbed `artifacts`
and `qualification`. Supply precomputed, hash-verified image derivatives; this
tool does not resize, distort, decode or choose a corpus.

```json
{
  "schema": 1, "study": "cpu-size-pilot",
  "metrics": ["ssim2", "iwssim"], "reuse": ["cold"], "threads": [1],
  "passes": ["wall", "os_peak", "driver"], "replicates": 3, "operations": 1,
  "timeout_seconds": 300, "admission_seconds": 600, "peak_mem_bytes": 4294967296,
  "pairs": [{"id":"photo-256", "origin":"photo-1", "content":"photo",
    "width":256, "height":256, "stride_bytes":768, "pixel_format":"rgb8_srgb",
    "reference":{"path":"reference.rgb", "sha256":"REPLACE_WITH_HASH"},
    "distorted":{"path":"distorted.rgb", "sha256":"REPLACE_WITH_HASH"}}],
  "testbeds": [{"id":"cpu-linux", "machine":"cpu-linux", "domain":"uniform",
    "target":"x86_64-unknown-linux-gnu"}]
}
```

This example becomes 18 declared cells after inserting real input hashes. It
schedules zero launches until release artifacts and matching qualification are
provided. `wall`, `os_peak` and `heaptrack` use the ordinary latency artifact;
`driver` uses the instrumented memory artifact. Fleet wall is supervisor elapsed
time, not internal metric latency. Hashes and relative file paths are required;
the generated local packaging hints use resolved paths, while study identity
excludes those paths so relocating unchanged content preserves identity.

Each testbed may add:

```json
"artifacts": {
  "latency": {"binary":{"path":"latency", "sha256":"SHA"},
              "build":{"path":"latency-build.json", "sha256":"SHA"}},
  "memory":  {"binary":{"path":"memory", "sha256":"SHA"},
              "build":{"path":"memory-build.json", "sha256":"SHA"}}
},
"qualification": {"path":"qualification.json", "sha256":"SHA"}
```

Build metadata has exact fields `schema`, `target`, `binary_sha256`,
`source_revision`, `dependency_lock_sha256`, `compiler`, `features`, `rustflags`,
`profile` and `instrumented`. Use `schema:1`, `profile:"release"`; instrumented
is true only for the memory build. The tool verifies bytes/target/profile but
does not prove a source-to-binary build recipe: retain the exact command and
compiler/dependency provenance in the reviewed build process.

Qualification is an **operator-reviewed record**, not generated ground truth:
`schema:1`, matching `target`, `native_arch_verified:true`, current
`inventory_sha`, and `cases`. The native claim must follow an independent
on-target architecture check. A successful emulated process is insufficient.
Correctness must already have been qualified against the metric's oracle and
between ordinary/instrumented artifacts. Do not generate expected output from
an unreviewed candidate and treat it as accuracy evidence.

`cases` maps a SHA-256 proof ID to `{"score_bits":[...]}` with one u64 primary
score bit pattern per declared operation. Compute the proof ID with this tool's
canonical sorted compact JSON and SHA-256, over:

```json
{"case":{"schema":1,"metric":"NAME","width":256,"height":256,
 "stride_bytes":768,"reference":"reference.rgb","distorted":"distorted.rgb",
 "pixel_format":"rgb8_srgb","reuse":"cold","threads":1},
 "inputs":["REFERENCE_SHA","DISTORTED_SHA"],
 "binary_sha256":"DRIVER_SHA","operations":1}
```

Different builds, inputs, threading, dimensions, operations or reuse require a
new proof. The proof omits measurement pass/replicate intentionally: those do
not change expected output. An absent proof is visible `unqualified`, not
silently executed or skipped. Unsupported combinations include missing driver
routes, invalid layout, too-small IW-SSIM, odd VMAF and non-Linux heaptrack.
No implicit crop, backend fallback or reduced test expectation occurs.

Retain `study.json`, `inventory.json`, `cells.jsonl` and each ready cell's
`case.json`/`bundle-spec.json`. Package ready specs with the installed native
runner, stage through the [benchmark adapter](BENCHMARK_FLEET.md), and declare
the resulting bundle/runner hashes through `zenfleet-ctl declare-benchmarks`.
Keep raw artifacts in the content store; selected small review records belong
in Git. Tests: `just benchmark-study-check`.

See [the study design](MULTIDIMENSIONAL_BENCHMARKING.md) for timing boundaries,
peak-memory coverage, independent process blocks, hardware confounding and
the remaining platform/analysis qualification gates.

## SDR driver coverage and scalar products

The release fleet driver exposes **38 of 46 CLI variants**: all SDR CPU
variants. The remaining eight are the six GPU variants and the two native HDR
variants; they require separately qualified workload contracts. Warm-reference
coverage remains four variants. These counts describe driver routes, not
cross-platform build or accuracy certification.

`just iqa-study-build DIRECTORY` retains separate release `latency` and `memory`
executables from the pinned sibling snapshot. `just iqa-study-check DIRECTORY
NEW_CONTROLS_DIRECTORY` checks all 38 SDR variants on a textured 256×256 pair,
two operations per invocation, packed and padded rows, and both executables.
The 2026-10-12 Linux x86-64 controls preserved every scalar product exactly
across instrumentation and row layouts. The existing schema-1 matrix also
passed its 32 cold/warm/thread/layout cases. These are functional controls;
they establish neither metric accuracy nor admissible benchmark conditions.

Standalone fleet cases may select schema 2, adding required fields
`products:"primary"|"all"` and `warmup_operations:0..1024`. Schema 1 retains
its previous contract: no added fields, zero cold warmup, one warm-reference
warmup, and finite primary scores. Schema 2 executes the declared warmup count.
`all` is available for cold scoring and preserves ordered CLI scalar columns
as `{"products":[[{"name":"COLUMN","bits":U64},...],...]}`, one array per
operation. This retains secondary outputs such as butteraugli norms, FSIMc and
VMAF feature columns. It does not retain spatial maps. Schema 2 encodes IEEE
bits, including mathematically defined infinity, without a JSON nonfinite
number; the independent qualification must approve the exact expected values.
An unexpected NaN cannot be accepted merely because it can be encoded.

The study generator currently emits schema-1 primary-score cases. Schema-2
bundles require their matching explicit correctness contract; schema-1 proofs
cannot qualify them. Internal timing covers the metric call and its returned
scalar-column collection. Supervisor `wall` covers the whole invocation,
including preparation, warmup, measurement and serialization. Heap phase
records include the harness allocations made inside each phase. Keep these
boundaries labeled when comparing results.

## Materialize and stage a qualified study

```sh
just benchmark-study package SPEC.json GENERATED_DIRECTORY NATIVE_RUNNER RUNNER_SHA NEW_BUNDLE_DIRECTORY
```

Packaging re-expands the original spec, compares the inventory, summary and all
cells, verifies the generated case and plan, then uses the hash-bound native
runner's `bundle` command. It verifies each copied file against the declared
input/build/binary hashes and checks sources again before writing a declaration.
A refusal retains materialization logs and partial bundles, but writes no final
`observations.json`. A new output directory is mandatory. This command performs
no workload execution, upload or worker enrollment.

The output contains `identities.json`, a bundle directory per ready cell and
`observations.json` in the existing controller schema. Each job experiment is
the cell hash, preserving the selected study/pass/testbed/replicate identity.
Use the exact hash-bound runner as `ZENBENCH_BIN` when staging each listed bundle
with `scripts/jobsys/benchmark_exec.py stage BUNDLE`; require the returned bundle
and runner hashes to match `identities.json`. Then pass `observations.json` to
`zenfleet-ctl declare-benchmarks --spec ... --out ...`. Deployment and execution
continue through the existing fleet flow. Packaging supplies no retry policy
and does not certify production admission or installation.

`just benchmark-study-package-check RUNNER NEW_DIRECTORY` exercises actual
native bundle materialization with fixture bytes that are never executed. On
2026-10-12 the Linux x86-64 runner materialized 12 qualified fixture cells and
preserved all declaration hashes. The unit controls also refuse a changed
runner or case before creating output. This evidence concerns protocol
materialization, not metric accuracy, resource admission or performance.

## Native HDR-VDP cases

Schema-3 fleet cases support `hdrvdp` and `hdrvdp3` on the native CPU, cold
reuse only. Inputs use `pixel_format:"rgb_f32le_bt709_linear_nits"`: interleaved
little-endian f32 RGB channels in absolute cd/m², with BT.709 primaries. Every
channel must be finite and nonnegative. Row padding is supported and excluded
when unpacking during input preparation. No transfer-function conversion,
normalization, tone mapping or reinterpretation as sRGB8 occurs.

Schema 3 requires `products:"primary"|"all"`, `warmup_operations:0..1024` and
an explicit `viewing` object. For HDR-VDP-2 use
`{"model":"hdrvdp2-defaults-bt709","pixels_per_degree":45}`. This selects the
source-pinned HDR-VDP-2 default calibration with the stated geometry. For
HDR-VDP-3 use
`{"model":"hdrvdp3-quality-bt709-age24-nosurround-reference","pixels_per_degree":45}`.
This selects quality task, BT.709 input/default emission, observer age 24, no
surround and reference options. Both require positive finite geometry of at
least four pixels per degree. These are different metric models and output
scales. The metric constructor and reference work occur inside each cold
score; input unpacking and pool initialization occur before scoring.

A study pair may include the same `viewing` object. The generator emits
schema-3 primary-score cases for the matching HDR metric/preset. Missing
geometry, a mismatched preset, sRGB8 inputs or warm reuse remain explicit
unsupported cells. Exact input and parameter identities enter qualification.
The inventory now exposes all 40 CPU routes across their two input contracts;
GPU routes still require separate implementation and on-target qualification.

`just iqa-hdr-check BINARIES NEW_DIRECTORY` checks both HDR routes, two
operations, padded and packed rows, and ordinary/instrumented builds. Private
Rust controls compare the routed outputs exactly with each metric's direct
absolute-nits entry point and refuse nonfinite/negative channels and missing
geometry. These controls verify routing and input preservation; the metric
crates' author/reference gates remain the accuracy oracles.
