# IQA compute-only competition

Work in progress. No completed matrix or performance claim yet.

The driver extends `cpu-profile`, reusing zenbench and the CLI metric inventory.
The initial VMAF CLI timings include embedded-model parsing and are diagnostic,
not qualified final compute-only observations. The benchmark snapshot adapter
caches only immutable model parameters after warmup; owned-model clones stay
inside the call. No shipped metric source is changed. Its source hashes and
score/map parity gate must accompany the final rows.

Input preparation decodes the selected real AIC pair once, then uses zenresize
Lanczos for targets within both source dimensions and mirrored source pixels
for larger targets. Raw RGB files are loaded before timing. Their hashes belong
in each cell's input manifest. Learned model initialization runs before timing. VMAF uses the separately
qualified snapshot adapter described above to exclude embedded SVM parsing.

Each Rust group uses warmup and at least ten interleaved zenbench rounds.
Percentiles describe per-round batch means; raw batches and iteration counts
must be retained so these are not confused with individual-call latencies.
Scores and native-map calls are distinct arms. Unsupported routes and refused
sizes remain explicit rows. Every competitor must use the same pair and record
its score difference from the corresponding Rust implementation.

Raw data is external: `~/tmp/iqa-compete/<host>.jsonl`. Committed summaries
will be split below 30 KB. All heavy steps use the shared host resource wrapper;
individual lock holds have a hard bound below twenty minutes. No codec or file
read belongs inside a measured region. Third-party dependencies remain confined
to this benchmark harness.


## Reproduce on the current host

From a clean jj checkout, install Rust, just, Python 3.12+, uv, CMake, Ninja,
Meson, pkg-config, GNU coreutils for a macOS resource wrapper, a C++17 compiler and Highway/Brotli/LittleCMS2 development
packages (Homebrew on macOS). The command below builds this commit and pinned
native competitors, verifies the named AIC2026 S67 pair and every derivative
against [the checksum manifest](inputs.meta.json), then measures 1/all threads,
both modes and 5/3/1 samples. Python competitors use [uv.lock](python-env/uv.lock);
only explicit `--skip-python` omits them. CPU is the default; select GPU routes
with `--gpu-runtime cuda|wgpu` and supply licensed learned weights with
`--model-root`. `--cache`, `--build-cache` and `--output` choose owned cache and
result directories. Content-keyed completed cells resume; differing provenance
in an explicit output directory fails. Raw receipts and `run.meta` retain source,
binary, host, toolchain and competitor identities; ignored `repro-results/`
shards pass the registered dashboard adapter. Installed host resource caps are
used, otherwise execution is plain with a portable timeout and lock, explicitly
recorded in metadata. Linux pins 1T; macOS uses its documented unrestricted
placement. Windows still needs the [Job-object supervisor](PORTABILITY.md).

```sh
just bench-iqa cid22,aic-small ssim2,gmsd --host i265 --corpus /path/to/AIC2026/img --skip-python
```

## Initial qualification

The pinned release build and driver clippy pass. The initial CID22/2K
qualification covers PSNR, SSIMULACRA2, GMSD and FSIM at budgets 1 and 20;
CVVDP also has a CID22 one-worker check. All 30 timed score/map arms match
scores exactly; four PSNR map requests explicitly report unsupported.
Compact observations: [CID22](native-smoke-cid22.json), [2K](native-smoke-2k.json).
The full 112-name CPU inventory pass is still in progress; these smoke
checks do not establish full coverage.

The hybrid all-core panel is descriptive. Zenbench's uniform-core class
certification cannot describe that required configuration; the harness
records unrestricted placement plus explicit OS affinity (one performance
core for one-thread cells; all available CPUs for all-core cells). No
baseline certification or promotion is claimed. The ordinary load/resource
admission gate stays enabled.

The newer SSIMULACRA2 native-map adapter uses the same pinned sibling as
the CLI through a benchmark-only alias. The legacy profiling driver's
registry dependency remains unchanged. CVVDP allocates its scorer before
timing; both paths consume the same named standard_fhd display.

The benchmark-only model cache passes 24 bitwise comparisons against the
original parser (six VMAF names, CID22/2K, one/all 20 workers). Each probe
checks the cold and warm scalar outputs. VMAF-v1 also compares every one
of its 43 native maps by geometry and raw-byte SHA-256. The
[receipt](model-adapter-parity.json) binds both executable hashes.
The cached build uses `--cfg iqa_compete_ingress --cfg iqa_compete_cached`;
the reference uses only the ingress flag. These flags do not select SIMD
or change metric arithmetic. The initial VMAF timing rows remain diagnostic.

[DSSIM/Zensim qualification](extra-map-smoke.json) adds 16 score/map arms
at CID22/2K and one/all workers, with zero score difference.
C/C++ competitor adapters are buildable scaffolding; their timing and
same-score qualification are not yet complete.

## Native competitor correctness probes

The [CID22 one-call smoke](native-probe.json) verifies shim invocation and
compares every headline with a separate Rust probe. It is not a timing panel.
libvmaf reports version 3.2.0 from vmaf-head-sys 0.2.0's vendored source;
this is recorded rather than relabeled as the port's 3.2.1 parity reference.
VMAF, SSIM/MS-SSIM, PSNR, PSNR-HVS-Y and CAMBI fall within their existing
parity gates on that pair. The initial exact comparison flags VIF and Butteraugli. The final C++
Butteraugli decision uses its pinned reference test's 0.001 relative bound;
the measured CID22 residual is within that existing gate. VIF remains a
variant comparison. The initial SSIMULACRA2
probe accidentally reused Butteraugli's 80-nit metadata. The corrected
[probe](native-probe-intensity.json) uses SSIMULACRA2's standard 255-nit
metadata and matches Rust exactly on CID22. The earlier receipt remains
as evidence of the adapter correction; its 2.4278133457799242-point gap
was not an implementation difference.

Native input exports reuse the CLI's BT.709/BT.601 4:2:0 conversion,
including even cropping. SSIM uses the full-size f32 studio-601 luma
formula, PSNR the full-size f64 BT.709 formula. Raw buffer hashes are
recorded in each competitor row. Input conversion to the native buffer
type is untimed; conversions performed inside the metric remain timed.
The libvmaf timer covers picture submission, feature extraction, flush and
score prediction. Context registration, picture allocation/fill and model
loading precede it. The libjxl timer covers ComputeScore or
ComputeSSIMULACRA2, including conversion within those calls and map release.
Raw RGB unpacking and pool/metadata creation precede it.

The native runner saves every timed sample and repeated score, the
corresponding Rust reference probe, input hashes, admission load and the
resource-wrapper receipt. Public APIs without a native buffer map route
produce explicit unsupported rows.

## GPU qualification

[CID22 CUDA observations](gpu-qualified-cid22.json) cover five score paths
and the CVVDP map at one worker. Four score paths and CVVDP's map stay
within their selected CPU parity bounds. Butteraugli differs by
0.008256673812866211 on both independent attempts and is plotted as a
variant mismatch. Both attempts and logs are retained externally. Zensim
GPU scoring is disabled by the product and has explicit refusal rows.
This smoke does not establish the larger-size or all-worker matrix.

GPU instances persist outside timing, with full memory mode and no
cached reference features; each call recomputes both images. Readback
finishes before the call returns. CVVDP names standard_fhd photometry
and geometry, matching the CPU panel. CUDA mismatches against a supplied
CPU ledger trigger a second complete bounded cell; the first attempt
is preserved and linked from the selected row.

Adapter boundaries and source links: [protocol](PROTOCOL.md).

## CID22 CPU progress

Missing: PieAPP, the remaining 2K cells and size ladder, competitor timing
panels and other hosts. The [CID22 panel](cid22-cpu-progress.meta.json)
covers 112 of the required 113 names: 448 cells at one/all workers, with
260 timed results, 144 unsupported native-map routes and 44 explicit
input/build refusals. [Split summaries](cid22-cpu-progress/) retain each
cell's statistics, score, load and resource receipt. These are observations
from the named raw ledgers, not a completed full-matrix claim.

Source freeze: zenmetrics `205efe2e` (master at this study's start), plus
PieAPP `9b5f704b` and the benchmark commits. The separate Round 14 performance
branch is outside this freeze. Every executable has its own SHA; later master
changes do not silently alter an existing observation.

## CID22 progress including PieAPP

The [second progress snapshot](cid22-cpu-progress-v2.meta.json) covers
452 explicit CPU cells across all 113 names: 261 qualified timed cells,
146 unsupported-map rows and 45 other refusals. PieAPP single-thread
sampling still lacks ten rounds and needs its chunked rerun. This inventory
coverage does not complete the timing panel or the larger-size matrix.
Split compact observations: [CID22 progress v2](cid22-cpu-progress-v2/).

## Owner coverage pass (2026-10-17)

For one source revision across hosts and preserved historical results, follow
the [fresh-run procedure](RESTART.md). Set `IQA_RUN_HEAVY` to the resource wrapper
before launching controllers or landing gates.

The coverage-first pass supersedes the earlier depth requirements for new cells.
It uses the existing single pair per resolution: five actual samples at CID22,
AIC-small and AIC-medium; three at AIC-large/2K/4K; and one at 8K. Rows record
`coverage_first` and `required_samples`; the legacy controller default stays ten.
The explicit first score/map call warms each route, with no additional zenbench
warmup interval in this pass. Partial cells do not continue in another scope.
A cell below its sample target records `skipped:scope_sample_limit` with its
actual samples and retained log. Each scope retains the existing 1100-second
limit and 16 GiB Linux memory cap. The coverage pass retains the first completed attempt with its contamination
label. Automatic contamination reruns are disabled until coverage is complete;
the owner allows at most one rerun per cell.

Score comparisons run at CID22 once per metric, using the first thread budget.
Larger sizes and the other thread budget carry timings without another parity
qualification or mismatch rerun. Existing completed observations are retained.
The final coverage audit reports requested, present and timed cells per size;
unsupported routes and bounded refusals are counted separately from timings.

The CPU, CUDA, native and Python coverage controllers can group short cells into an existing
1100-second lock hold. They release it before starting another cell whenever
less than 1000 seconds remain. Each row records its actual scope, between
1000 and 1100 seconds; partial cells still do not continue. This reduces lock
queue overhead between short cells. PieAPP keeps its independent per-cell lock.

The permitted cgroup CPU set in zenbench's metadata is separate from process
CPU affinity. The [live one-thread snapshot](coverage-active-pinning-control.json)
records every thread restricted to CPU 2, without another metric invocation.

The optional GPU `--isolate-modes` path gives score and score+map separate
processes and resource scopes. It avoids holding independent GPU scorers at
the same time. The missing 8K Butteraugli cells use this path after the grouped
scorers hit device allocation failures; completed observations are retained.
For coverage rows without a repeated CPU comparison, collection clears
`ref_score` and deltas; `self_warmup_score` remains separate.

libvmaf CAMBI refuses 8K with `window_size 131 too large for reciprocal LUT`.
That exact public-entry-point error is an explicit input refusal; the remaining
native cells continue. Other unexpected native errors still stop that stage.

The [isolated Butteraugli 8K receipt](butter8k-isolated-v17-control.json)
records all four missing mode/thread cells with n=1 and clean contention
receipts. The [build receipt](cuda-v17-build-control.json) binds the executable.
No completed timing or CID22 comparison was repeated. Chart summaries retain
`coverage_first`, `required_samples` and `parity_checked`; bar labels show n.

After releasing the shared i265 lock, controllers sleep 45 seconds whenever
`pgrep -x flock` reports more than one process. This applies to batch and
per-cell releases. [Controls](lock-cooldown-control.json) verify that the wait
happens after unlocking and is omitted without multiple flock processes.

The [master merge validation](landing-master-20261017.json) retains each gate
log hash and resource receipt. The ORT snapshot adapter covers both three
independent builders and master's shared builder across three constructors.
Its tests recover the source byte for byte after removing the adapter edits.
Existing timing rows retain their frozen binary and source provenance across
master merges; merged-source validation has its own receipt.

[Current landing validation](landing-master-20261017-current.json) includes
master's NN WebGPU changes, both provider builders across three constructors,
14 feature-matrix rows, CLI/driver clippy and the full harness controls.
At each round end, including data updates, merge current master and provide
the verified branch landing SHA to the coordinator.

## Remote coverage snapshot (2026-10-17)

The retained historical cohort has [1,684/2,996 r7900x rows](task4-r7900x-progress-20261017-1012.json)
and [2,258/3,164 Mac rows](task4-mac-progress-20261017-1012.json). These are
recorded attempts, including unsupported routes and refusals. Per-size receipts
separate timings meeting the requested minimum from short samples. In particular,
the external controller requested three AIC-medium samples; the policy requires
five. Actual observations remain unchanged and count as short. Known native-map
adapter omissions are counted separately from metric map unavailability.

The queues continued after that snapshot and were then stopped by the owner.
These observations retain frozen source `d6b18bbf` and
their actual executable hashes; merging the harness does not relabel them as
measurements of current master. The immutable raw prefixes remain external,
with SHA256 pointers in the compact receipts.

The [Mac containment control](task4-mac-supervision-control-20261017.json)
records a real child terminated by the resource guard with `gtimeout --foreground`.
The tracked launcher now retains that option. This is process containment
evidence, not a metric or score-parity qualification.

## Owner stop (2026-10-17)

Missing rows: 1,216 on r7900x and 876 on Mac. Neither host reached 8K.
AIC-medium's three-sample observations remain short against the five-sample
requirement; the planned repair was not started. Further measurements wait for
the separately developed reproducible launcher.

The queues stopped after their active cells, with no subsequent cell started.
The [closing receipt](task4-owner-stop-20261017.meta.json) binds all
4,068 saved observations: [1,780/2,996 on r7900x](task4-r7900x-owner-stop-20261017.json)
and [2,288/3,164 on Mac](task4-mac-owner-stop-20261017.json).
[Compact cell receipts](task4-owner-stop-cells-20261017/) retain every row,
including refusals and short samples. Their n, timing and score fields match
the archived raw values bit for bit. Each generated JSON file is below 30 KB.

Both remote integrity checks and the copied log/config hash checks pass.
Four older Mac logs acquired late output during the earlier guard escape.
The exact originally hashed prefixes and complete logs are both retained;
late output remains diagnostic. No additional metric evaluation, parity
qualification or reboot was performed during closing collection.

The closing checks pass 102 Python harness controls and 10 inventory controls.
The inherited formatter findings were corrected and the six affected files
pass rustfmt check. Remaining queued build gates were cancelled before
admission at the owner's stop directive; they are not claimed as passing.
## Resident NIQE reference parameters

`canonical_niqe_resident_cell.py` uses the unchanged BasicSR 1.4.2 NIQE
callable with its parameter archive loaded before timing. Each callable still
receives owned parameter-array copies with their original memory order.
Before recording samples, every process checks both images against the original
archive loader and requires identical score bits. Only `SAMPLE` observations
are warm timings; its initial `COLD` token follows those controls and does not
qualify a cold-call measurement.

```sh
just bench-iqa-canonical-niqe-resident PYTHON niqe WIDTH HEIGHT \
  REFERENCE_RGB DISTORTED_RGB THREADS SAMPLES --extra-path BASICSR_PARENT
```

`bench-iqa-canonical-niqe-plan` requires a successful control receipt, preserves
the caller-identified future plan in a backup, and refuses a plan whose output
directory already exists. It retains native settings and numerical bounds.
Earlier archive-loading observations remain diagnostic evidence for that
different timing boundary.

## Frozen Butteraugli map comparison

`bench-iqa-canonical-butteraugli-maps` builds accuracy-only probes against a
captured CPU Butteraugli rlib and the captured libjxl comparator. Both consume
the same hashed RGB8 inputs at intensity target 80. Each native map call checks
its scalar score bits against the scalar-only call. The existing FIR scalar
policy applies; map differences are observations without a new tolerance.

```sh
just bench-iqa-canonical-butteraugli-maps PYTHON \
  --native-build NATIVE_BUILD --reference-build LIBJXL_BUILD \
  --pairs PAIRS_JSON --output NEW_DIRECTORY
```

The caller must provide heavy-job admission. The tool accepts captured CMake
Makefiles or Ninja link commands, records library/probe/input hashes, retains
raw maps outside git, and refuses an existing output directory. This accuracy
probe does not supply comparative timing measurements.

## No-reference auxiliary scores

`bench-iqa-canonical-nr-auxiliary` compares a native metric's `_ref` column
against every retained canonical reference-image observation. The native probe
also checks all returned columns for bit stability across two evaluations.
It requires a passed headline comparison, validates the original source/input
bindings before and after each probe, and uses that case's unchanged bound.
It records accuracy observations and process maximum RSS, without timing ratios.

```sh
just bench-iqa-canonical-nr-auxiliary PYTHON MANIFEST NEW_OUTPUT
just bench-iqa-canonical-nr-auxiliary-check
```

The manifest supplies `cases` with `results` and `case` entries. An explicit
`binding_relocations` mapping may point nonexecuted preparation provenance to
retained historical bytes with the original hash. Runtime commands cannot be
relocated this way. Missing reference-image observations fail the check; the
caller must select actual no-reference routes. The caller supplies heavy-job
admission for the probes.

`bench-iqa-canonical-cpu-revision-build REVISION NEW_BUILD` exports the pinned
source and dependency snapshot before compiling a fresh CPU worker. The
existing `bench-iqa-canonical-cpu-build SOURCE TARGET` builds an already
prepared source tree; both require caller-supplied heavy-job admission.

## Ordered comparative selection after further metric changes

`bench-iqa-canonical-changed MANIFEST NEW_OUTPUT` accepts an optional manifest
`order` path. The declared routes must match that table exactly; reference
bindings, compiled native coverage, numerical bounds and output preservation
are checked before acquisition. `bench-iqa-canonical-ordered-check` exercises
those controls. A manifest without `order` retains the historical 27-route
table, and `bench-iqa-canonical-changed-plan` remains the frozen legacy preparer.

The current native `changed` selection includes the eight PSNR routes revised
on October 20, for 35 routes total. Its separate scheduling table preserves
[the original table](../rebenchmark_since_f7668c7/order.tsv) for retained
comparison plans. Both tables use recorded historical durations as scheduling
hints; their values are not current speed measurements.

## Resident first evaluation and live regression auditing

`just bench-iqa-canonical-nn-first-build PLAN NN_TARGET NEW_OUTPUT` links a
probe to the plan's frozen original-NN and PyIQA libraries. The builder checks
the worker and reference bindings, retains source and library hashes, and
primes the existing model cache before timing the first evaluation. RGB inputs
and weights are resident. The probe uses the stock scoring body and existing
`cold_call` timer, then checks a second score's bits outside the clock.
This measures first evaluation after construction; constructor time and outer
CLI dispatch are excluded. It does not qualify full cold startup.

Real 512×512 one-thread controls passed for DISTS (absolute difference 0) and
TOPIQ-FR (1.1920928955078125e-7), within their existing bounds. Both probes
matched their retained native warm scores bit for bit. These are setup and
accuracy controls, not coverage of the requested cold matrix.

`just bench-iqa-canonical-watch MANIFEST FLAGS REGRESSIONS PROGRESS_LOG PID`
audits an existing queue on every scored-case completion, including an initial
and final catch-up. It records cells above 1.08 immediately without launching
additional inference. Follow-up measurements and profiles remain explicit
queued jobs.

The subsequent FLIP/FLIP-v17 and WS-HV revisions expand the current native
`changed` selection to 38 routes. Their separate
[scheduling table](../rebenchmark_since_f7668c7/order_current_after_flip_2026-10-20.tsv)
retains the earlier 35-route table for existing comparison manifests.
