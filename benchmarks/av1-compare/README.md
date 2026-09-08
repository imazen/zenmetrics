# Static AV1 comparison executable

One executable links C libaom, C SVT-AV1, zenav1-aom, zenav1-svt and zenrav1e.
The C adapters use public encoder APIs and return owned OBU bytes through a
checked Rust boundary. This is benchmark tooling, with no change to the
production Rust codecs' zero-C dependency contract.

The `av1-api-planar-v3` protocol accepts packed planar 8/10/12-bit
420/422/444/mono according to each backend's validated capabilities. High-depth
samples are little-endian u16. C SVT accepts 8/10-bit 420; Rust SVT also accepts
mono. The three AOM/rav1e arms accept all twelve depth/chroma combinations.
Dimensions are 64..16384 (C SVT requires even dimensions). Threads are 1..16;
Rust AOM currently requires 1. For C SVT threads means **lp**, not a promise of
exactly that many OS threads.

Quantizers and speeds are backend-native; equal numbers are not equal effort
or achieved quality. The signed preset field additionally supports C SVT's
public research mode -1 in both C and Rust SVT. Rust uses its checked signed
`NativePreset` and `new_with_preset` constructor. C enum
entries -2/-3 are rejected by this reference build's public validator and by
this adapter. No negative value is cast into a Rust unsigned preset.

AOM uses ALL_INTRA with explicit SB64 (optional SB128). SVT uses still-picture
CQP (AQ0), with optional tune and screen-content controls. Unsupported requests
are errors before FFI. The defaults retain each library's coding-tool policy.

C source is taken from the pinned sibling submodules, built separately under
Cargo OUT_DIR with multithreading enabled, native tuning off and FP contraction
off. The differential oracles and their caches are untouched. The small nested
workspace allows building this executor without unrelated GPU and HDR corpus
sibling repositories; the normal zenmetrics workspace currently requires those.

On Linux x86-64, build and test with `benchmarks/av1-compare/build-static.sh`.
The helper serializes tests and builds through the shared resource wrapper,
enables `crt-static`, and rejects dynamic `NEEDED` entries in the comparator,
worker and controller. Static codec archives alone do not prove static linkage.
Run the smoke in the deployment environment before distributing the binaries:

```sh
~/work/claudehints/scripts/run-heavy --mem 16G --jobs 1 -- python3 \
  benchmarks/av1-compare/smoke-jobexec.py \
  benchmarks/av1-compare/target/x86_64-unknown-linux-gnu/release/zenmetrics-av1-compare \
  /path/to/canonical/source.png /path/to/new-smoke-output
```

The smoke checks actual declaration/execution, native 8/10-bit reconstruction
with Zen intra-edge off/on, artifact hashes, invalid-setting evidence retention,
rejection of another executable's job, and canonical worker/ledger success and
failure with retained artifacts. It is a correctness check, not
routing calibration.

Input is a JSON request on stdin, for example:

```json
{"config":{"backend":"libaom","width":64,"height":64,"quantizer":40,"speed":6,"threads":1},"input":"source.i420","output":"libaom.obu"}
```

Backend names are `libaom`, `c-svt-av1`, `zenav1-aom`, `zenav1-svt`, `zenrav1e`.
The output row records settings, source/output hashes, revision, byte count,
and fresh-lifecycle API elapsed nanoseconds, including setup, owned output
copying and teardown. Plane preparation is included. Input file I/O, hashing,
independent decode and output file I/O are outside the timer. This is not a
codec-kernel-only timing. Every output must pass libaom decode and dimensions,
bit-depth and chroma checks before a success row is emitted.

The unit-test checkerboard is solely a correctness witness. It is not a speed
or quality calibration corpus. Real comparisons must use the registered
imazen-26 sources, repeated interleaved trials on the same worker,
independent scores, and matched-quality comparisons.

`measure` reads a JSON grid of `inputs`, `max_edges`, `arms`, `repeats` and
`output_dir`; it scores full decoded RGB with fast-ssim2 and retains every OBU.
After all timed rounds, every unique Rust SVT cell is replayed: the input and
measured OBU must match exactly, and every reconstructed sample must match the
independent libaom decode. `reconstruction-verification.jsonl` records these
checks. Only a successful check writes `validation.json` with `complete=true`;
non-SVT-only grids explicitly record that reconstruction is not applicable.
Standalone `verify-measurement` remains available for older saved measurements.

All arms share BT.709 limited-range conversion. Conversion-only ceilings and
codec-only scores accompany end-to-end scores. High-depth conversion of RGB8
sources is not native HDR coverage. Repeated trials rotate arm order.

`declare` consumes canonical zenfleet EncodeDeclareItem JSONL for codec
`av1-compare`, q=0. Each job bundles a source/size/settings comparison with
interleaved repeats. `jobexec` consumes DesiredJob JSON and emits a tar containing
rows, reference PNGs and encoded artifacts. It checks the source SHA and exact
executable SHA. `capabilities` self-reports the required executor token; this
is distinct from the worker's hardware resource-class `--capability` flag.

The executor does not schedule work. Zenfleet owns claims, retries, Parquet
ledgers and content-addressed output storage; Nomad manages worker lifetimes.
Set `ZEN_CORPUS_DIR` and a disk-backed `TMPDIR`. The worker discards stdout
on executor failure, so failed measurements preserve their scratch directory
and write a separate content-addressed failure tar using zenfleet's blob store.
The executor error records its location. Configure the launcher environment
`ZEN_AV1_FAILURE_STORE` for remote workers, for example:

```json
{"kind":"s3","endpoint":"https://your-store.example","bucket":"zentrain","prefix":"benchmarks/your-run/failures"}
```

The S3 store uses launcher-injected scoped `AWS_*` credentials. A local target
uses `{"kind":"local","path":"/path/to/failure-blobs"}`; the default is
`$TMPDIR/av1-compare-failure-blobs`. Remote launches must choose durable storage
before workers are drained. Job/corpus settings cannot choose this destination.
 The 2026-09-08 remote supplement
completed 20 jobs / 960 encodes through three Nomad-managed zenfleet workers.

`analyze.py` produces measured cells and bracketed quality estimates.
`time_budget.py` selects the smallest payload at each target quality under a
common time budget across tested presets. Do not combine timings from different
CPUs, interpolate preset numbers, or interpret estimated medians as deadlines.
See [the measured results](../av1_compare_2026-09-08/README.md).

## Validation

The native lossless matrix checks exact decoded samples for all 42 supported
backend/depth/chroma combinations. Decode and validation tests cover all five
backends, metadata mismatches, out-of-range samples and research-preset limits.
The local zenfleet smoke verifies real output bundles and ledger completion.

The Linux x86-64 executable has been verified as static PIE with no dynamic
`NEEDED` entries. Existing codec dependency warnings and their broader workspace
gates are separate from checks of this benchmark crate. No CI has been launched.

The v3 static build passed six library tests and four measurement/failure-store
tests. Real photo and screenshot job bundles each checked 15 encodes and four
native SVT reconstruction witnesses; the screenshot smoke additionally checked
the serial canonical worker, ledger and both success/failure blob stores.
The S3 failure path was downloaded and SHA-verified independently. See
[recorded validation](../av1_compare_2026-09-08/automatic_reconstruction_validation.json).
These checks do not establish full corpus coverage or completion of the codec's
remaining real-image parity, HDR, policy and routing work.
