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
