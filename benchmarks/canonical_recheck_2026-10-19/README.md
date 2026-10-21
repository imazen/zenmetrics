# Local canonical rechecks

The [DSSIM full-call reference controls](dssim-fullcall-controls.json) passed
24/24 pair/budget comparisons and 144 exact-bit old/new scalar observations.
The new reference times context and image construction, comparison and release
inside each sample; its C source, header, author archive, compiler and ELF are
bound explicitly. Accuracy controls do not supply a comparative timing ratio.
The 47 s run used peak-RSS 1.42 GiB, minimum available 13,980 MiB and peak load
5.61. Older context-staged timings and their source-proof limit remain retained.

The [three-route neural scope review](../iqa_compete_2026-10-16/nn-flag-scope-review-2026-10-20.md)
finds no concrete clock-scope defect in the selected DeepDC/VGG first-call and
PaQ2PiQ warm rows. Image conversion and both required evaluations are timed;
model loading is staged. This is source review of those three rows, not runtime
contamination, effective worker-count or transitive rebuild qualification.

The [latest four CPU profiles](grid-flag-profiles-2026-10-20/index.json) cover
three revised FLIP cells and one DSSIM-GPU submission path. Their binary,
configuration, raw recording and report hashes were verified before publication.
FLIP profiles contain the AVX-512 `conv_band_t_v4x` entry. CPU samples do not
measure GPU kernel time or establish the cause of a cross-implementation gap;
the DSSIM comparison retains its older-fork alternative disposition. Each profile
has its original resource receipt. Export verification used peak-RSS 0.02 GiB,
minimum available 14,903 MiB and peak load 7.43.

`just bench-iqa-canonical-profiles-publish INDEX NEW_DIRECTORY` reproduces the
projection under caller-held heavy admission and refuses existing output paths.

The [40-route timing-source review](../iqa_compete_2026-10-16/current40-timing-audit/matrix.md)
records exact selected commands, source hashes, timed preparation and thread
controls. It finds parameter-preparation differences for four HDR routes and
six VMAF model routes, plus DSSIM context/provenance limits. These are scope
qualifications without human-approved exceptions or measured overhead estimates.
SSIMULACRA2 and FLIP retain explicit wrapper/package provenance and thread-control
limits. Reviewing 40/40 entry points does not qualify 40/40 timing comparisons.

The [latest scoped command](local40-scoped-current40worker-command.json) keeps
all 80 numerical cases and explicitly withholds comparable timing ratios on
22 cases with source-review caveats. Preparation passed in 3 s, peak-RSS
0.06 GiB, minimum available 15,482 MiB and peak load 3.60. It does not report
80 completed comparisons. The [VMAF worker lineage](vmaf-worker-scope-lineage.json)
distinguishes the older prepared-model benchmark snapshot from the current
production-call worker. Both model-library hashes match their original source;
the CLI's later parallel ingress is a separate source change. The current40
preparation finding must not be applied automatically to the older cached-model
timings. No timing estimate is derived from that source comparison.

Missing: the full multi-size per-name grid, remaining learned comparisons,
map comparisons and qualified regression acceptance. The initial classical
RGB8 cohorts are complete; explicit route/input exceptions remain below. These are scalar checks on
Ryzen 9 7950X / WSL2. The first six-name panel uses one 512×512 photo pair.
Additional original MATLAB, corrected VIF and libvmaf cohorts are indexed in
[cohorts.json](cohorts.json); its observed/expected counts expose partial runs.
The per-case TSVs retain exact dimensions, threads, bounds, deltas and ratios.
They do not replace multi-size or content validation.

The [declared broader classical grid](classical-grid-plan.json) has started:
84 CPU routes × four geometries × three content pairs × threads 1/28 = 2,016
warm scalar cases. Its raw queue and frozen input/plan hashes are indexed there.
Declaration is not completion; existing README ratios remain representative
completed cohorts. One historical preparer was preserved byte-for-byte from git
before moving its nonexecuted provenance binding to a retained path. Runtime
commands, case contracts and numerical bounds remain unchanged.

The frozen Rust build is upstream master `efae6b8fed9e57346157d92dffc3cdb57c44560e`
(clean exported tree `f82b3614b385`). Binary SHA256:
`7859f91207fceeb45243dff3955ef4b276a885e35d5e1ca93ca49bd983a02cc0`.
It enables `compete,metrics-parallel,zenmetrics-cli/nr`, runtime SIMD dispatch,
and the existing immutable-model preload/map benchmark adapters. No native-CPU
compiler flag was used. The build receipt is rc=0, 126s, peak-RSS 0.87GiB,
minimum available 10651MiB, peak load 3.42.

[results.json](results.json) supplies the README's exact-name observations.
`pass` uses the existing NR JND/1000 bound. A displayed delta without `pass`
means no external acceptance bound has been assigned. No golden or threshold
was changed. The first attempt retained only canonical observations because a
verified orphan zenbench reservation blocked the Rust arms. Its process and
former process groups had exited, its guard was exclusively acquired, and the
reservation was preserved under a `.bak` name before the successful new attempt.

Speed ratios are Rust/canonical (lower is faster), from the median of three
alternating acquisition-block ratios, with five warm observations per arm.
Both sides begin with resident RGB8 and include their callable conversions;
immutable model loading is excluded, and owned parameter clones remain timed.
The NR references evaluate both images. Ratios are descriptive approximations:
there is no confidence interval or 1.08 regression acceptance claim. The initial
SSIMULACRA1 adapter preconverted RGB to BGR and its ratio remains withheld.
The separate `ssim1-rgb` cohort times conversion with the original metric body;
its one-thread 512² ratio is 1.12 and is flagged for confirmation.
[Resource receipts](resource-receipts.txt) retain every acquisition block's
wrapper peak-RSS/minimum-available/load line. Process `/usr/bin/time -v`
receipts and all samples remain in the raw evidence.

Raw evidence on the local WSL2 machine:
`~/tmp/local-canonical-20261019T2330/retry-run/`.
The adjacent `retry-run-SHA256SUMS.json` binds all files; SHA256
`949f92cd45e2c421cad703a01a06b91b3d3fbb80c553227c345d0e06c49c70f5`.
The plan binds inputs, binaries, explicit helper sources and reference parameters.
This is exploratory executable evidence, not a complete runtime-closure audit.

## Reproduce

Use Python 3.12 for the installed scientific references. First build a current
frozen Rust worker with the existing snapshot build recipe. The ordinary
`just bench-iqa` follows the shared cold/warm CPU protocol; the command below
adds explicit original-implementation comparisons.
Prepare a plan with explicit installed reference paths:

```sh
just bench-iqa-canonical-plan \
  --binary /path/to/iqa-compete --pairs /path/to/pairs.json \
  --output /path/to/new-plan.json --source COMMIT \
  --python /path/to/reference-venv/bin/python \
  --extras /path/to/basicsr-and-pypiqe --brisque-weights /path/to/weights.pt \
  --ssim1 /path/to/ssimulacra-canonical --ssim2 /path/to/iqa-ssim2-primary \
  --sizes cid22 --threads 1
just bench-iqa-canonical /path/to/new-plan.json /path/to/new-results
```

The initial registry includes BRISQUE/PIQ 0.8.0, NIQE/BasicSR 1.4.2,
PIQE/pypiqe 1.2, original SSIMULACRA1, primary SSIMULACRA2 v2.1 and NVIDIA
FLIP 1.7. Missing adapters remain explicit in the plan. `--metrics` on either
command restricts the named cohort. The comparison runner serializes heavy
work under both existing local admission locks and the 12GiB resource wrapper.
Existing outputs are refused rather than overwritten. Run
`just bench-iqa-canonical-check` for parser, failure and scope controls.

To extend an existing reference contract onto a size/content grid, supply a
`pairs.json` with a `pairs` list: each entry names `pair`, `content`, `width`,
`height`, `reference`, `distorted`, and the two `*_sha256` values. Paths are
relative to that manifest; inputs must be packed RGB8. The preparer verifies
the original bindings and every input hash, retains numerical bounds and
canonical commands, and refuses existing output files:

```sh
just bench-iqa-canonical-grid-plan SOURCE_PLAN PAIRS_JSON NEW_PLAN \
  --metrics EXACT_NAME,OTHER_NAME --threads 1,28
just bench-iqa-canonical NEW_PLAN NEW_RESULTS
```

This expands warm scalar coverage. Refusals remain recorded process findings;
it does not establish matching refusal semantics, cold timing or map parity.

## Additional original implementations

The explicit registry interface adds original MATLAB functions through GNU
Octave 6.4, original DSSIM through its C API, and libvmaf 3.2.0 with named RGB
ingress. Every conversion and per-call context/picture lifetime is timed;
immutable model loading stays outside. The libvmaf version is the reference
actually built, rather than the Rust metadata version. Assigned bounds remain
unchanged; missing bounds produce a delta, never an implied pass.

The first Octave VIF cohort used 0.299 luma where the port contract specifies
0.2989. Those six observations remain in its TSV with an explicit exclusion;
the separate corrected cohort checks the documented ingress. The corrected
six cases pass 1e-9, with maximum absolute difference 5.379585665821196e-13.

Add `--registry /path/to/explicit-registry.json --metrics NAME,...` when
preparing these plans. A registry supplies `metrics` (canonical command,
absolute bound, scope and whether scope matches) and `bindings` (absolute
artifact path to SHA256). The runner rejects changed artifacts. Build helpers
are `bench-iqa-canonical-dssim-build` and `bench-iqa-canonical-vmaf-build`.

Project retained results with `just bench-iqa-canonical-publish OUTPUT COHORT...`;
pass `--dispositions FILE` for explicit invalid-ingress exclusions. Raw evidence
is retained; this command writes only the small public projection.

## Local NR golden recheck

All 170 BRISQUE, 170 NIQE and 170 PIQE golden checks pass on this frozen
build, using their unchanged JND/1000 bounds. NIQE has 158 finite scores and
12 oracle-NaN inputs with the required matching errors. Printed worst absolute
differences are 8.301e-3, 7.317e-12 and 2.888e-12 respectively (three
significant digits, as emitted by the tests). [The receipt](nr-goldens.json)
binds the tests, original golden tables, input manifest and full log.
The wrapper reports rc=0, 23s, peak-RSS 0.73GiB, minimum available 11270MiB,
peak load 1.13. These are additional validation checks, separate from the
paired timing cases. Add `--goldens nr-goldens.json` to the publication command
to preserve this evidence in the compact README.

## SSIMULACRA1 matched-call follow-up

The new `bench-iqa-canonical-ssim1-build` adapter preserves the original metric
body byte-for-byte in a separately generated reference binary. RGB-to-BGR,
color conversion, scoring and temporary destruction are all inside the call;
OpenCV threads are explicitly set. The initial 512×512, one-thread three-block
result is Rust/canonical 1.121579214567383, with scalar difference
1.0480505352461478e-13. Its block ratios are 1.121579214567383,
1.1245052764711023 and 1.024168640307484. This is a provisional flag above
1.08, with a separate fixed 20-block confirmation and a completed perf profile retained below.
It is not a qualified regression acceptance decision. No metric code changes.
The three wrapper receipts report peak-RSS 0.09GiB each:

```text
run-heavy: done rc=0 2s | peak-RSS 0.09GiB | min-avail 12244MiB | peak-load 0.42
run-heavy: done rc=0 3s | peak-RSS 0.09GiB | min-avail 12081MiB | peak-load 1.25
run-heavy: done rc=0 2s | peak-RSS 0.09GiB | min-avail 12190MiB | peak-load 0.79
```

Current authoritative per-cohort TSVs and their hashes are listed in
`cohorts.json`; older unindexed TSV snapshots remain historical. Each current
row also records maximum measured process RSS for both arms. Large projections
use hash-checked JSON shards so no generated evidence file exceeds 30 KB.

## Expanded local checks

The index now includes original IW-SSIM, MAD, CW-SSIM, PSNR-HVS-M,
PyIQA SSIM/MS-SSIM variants, original Butteraugli and CPU/CUDA CVVDP.
The projection retains process failures separately from numerical findings; a
completed replacement can supply its own ratio while the earlier error stays
visible. Interrupted controllers retain every attempt and resume missing blocks.
The 1152² PyIQA inputs are exact crops of the retained RGB pairs; they exercise
all corrected fixed factors without the approximately 9 GiB reference process
seen at 4K. Results remain tied to their actual geometry.

All five corrected MS-SSIM k2–k6 first-photo cases pass the approved 1e-6 bound;
the largest difference there is 1.2287708517355611e-8. The larger grid remains
visible through observed/expected counts. CPU/CUDA CVVDP at 512² pass 0.001
with differences 2.86102294921875e-6 and 9.5367431640625e-7.

FLIP at eight threads measured a Rust/canonical ratio of 1.9754296991678477
over 20 alternating blocks, compared with 0.3344310291418364 at one thread.
FLIP-v17 independently measured 1.891036050578694 at eight threads over three
blocks. The README marks any measured case above 1.08 even when its displayed
representative one-thread ratio is faster. These are flags, not accepted
regression intervals. Linux perf sampled the FLIP convolution at 39.51% of
cycles and the SSIMULACRA1 AVX-512 blur at 25.69%; these identify optimization
work, not a causal attribution for the ratios. Both profiles completed with
peak-RSS 0.11 GiB; FLIP minimum available 12070 MiB, load 0.61; SSIMULACRA1
12127 MiB, load 0.76. Profiler artifacts remain beside the raw cohorts.

The frozen source uses DSSIM revision 331b56cf. The later master dependency
pin is 22e9ab033d5e5ac41ee961de00e7ae9bd959b634. Historical measurements
are not relabeled as the new pin. Its local snapshot lock check passes against
23 pinned siblings (51s, peak-RSS 0.15 GiB, minimum available 12047 MiB,
load 0.73). Neither this check nor the comparisons launch or wait for CI.

## Revised sources and matched ingress

The current DSSIM pin `22e9ab033d5e5ac41ee961de00e7ae9bd959b634` now has
six completed scalar comparisons (three 255×257 content pairs, threads 1/8).
Its snapshot source is `5fa7be7d42b51af3dd27b42dd50159972132ef12`;
[source equivalence](dssim-current-source-equivalence.json) verifies the named
production inputs against the synchronized master snapshot. The largest scalar
difference is 1.4643110879397625e-7. No external scalar bound is assigned.
Rust/canonical ratios range from 0.3477999740191757 to 0.47852966037580164;
the representative one-thread three-content median is 0.47528047145506236.
The earlier process error remains historical evidence, not a failed new-pin
comparison. Original standalone DSSIM's 30 core tests also pass; that control
reports 9s, peak-RSS 0.39 GiB, minimum available 11749 MiB, load 0.82.

The new PSNR Cb/Cr/611 fixtures follow the actual float64 studio-BT.601 ingress
and all six cases match exactly. Original results with the initial float32
fixture stay retained with an explicit invalid-contract disposition. Corrected
CIEDE2000 and SSIM adapters likewise declare their D65 matrix and whole-sample
mirror boundary; their original scoring bodies are unchanged. Their maximum
absolute differences are 9.2148511043888e-15 and 6.661338147750939e-16.
No production metric or numerical bound was changed.

Original Clunie libgmsd at even 512×512, threads 1/8, passes its existing
relative 1e-12 bound; maximum difference is 1.734723475976807e-18.
The original single-thread library remains single threaded under both budgets.
Its resident-RGB builder reports 1s, peak-RSS 0.04 GiB, minimum available
12448 MiB, load 1.01. Odd-size upstream refusals are not timed as valid scores.
Per-case process RSS and wrapper peak-RSS receipts remain in the indexed TSVs.

The initial 20-block [SSIMULACRA1 interval](ssim1-initial20-interval.json) is
1.0716442661769214, 95% interval [1.0317827326283462, 1.098290558074084].
It straddles 1.08 and misses the requested ±3% precision. The initial
[FLIP interval](flip-initial20-interval.json) is 1.9754296991678477,
95% interval [1.9486232227532867, 1.9950047813325684]. Native resource flags
occur in both panels (49/200 and 17/200 observations). A fixed additional
40 blocks per case are being acquired; every old observation remains visible.
These intervals remain descriptive and do not qualify a regression decision.

## Explicit gaps in the local RGB8 comparison

The compact README still lists every name. Eight routes have explicit reasons
for withholding a fresh external-canonical ratio, recorded in
[unmeasured.json](unmeasured.json): both PieAPP routes are owner-excluded;
`hdrvdp` and `hdrvdp3` require calibrated luminance/viewing inputs outside this
frozen RGB8 campaign; `zensim` is the original implementation; `zensim-gpu`
retains its explicit disabled-route refusal. `ms-gmsd` and `ms-gmsdc` are declared
paper-derived variants whose exact conventions have no qualified author callable
in this campaign. This does not claim that author code cannot exist. Their
independent qualification remains missing; no ratio or parity pass is invented.
AIMOS is excluded (proprietary, unreleased) and is not a CLI metric row.

The resident original mDCTpsnr caller now matches exactly at 512×512 with
threads 1 and 8. Its ratios are 0.8043324527167578 and 0.8625549003477649.
The unchanged original color conversion and pooling are timed, using a resident
P6 view of identical RGB8 bytes; file I/O and process launch are excluded.
The compiler retains the documented fast-math contract with portable Highway
dispatch. Build receipt: 10s, peak-RSS 0.36 GiB, minimum available 12209 MiB,
peak load 0.96. Scalar/map and broader-size coverage remain separate questions.

## Fixed-stage interval results

The [combined 60-block SSIMULACRA1 result](ssim1-combined60-interval.json)
is 1.0716442661769214, 95% interval [1.0618754401254606, 1.0960947055896795]
(±2.28%). It remains borderline against 1.08. All 600 native observations stay
in the primary result; 105 carry resource flags. The 40 entirely clean blocks
have a separate diagnostic median 1.0841294481188646, interval
[1.0620697994564496, 1.101185716322878], also borderline.

The [combined 60-block FLIP result](flip-combined60-interval.json) at eight
threads is 1.957927961720583, 95% interval
[1.917466421937279, 1.973160062353411] (±2.07%). There are 49 resource-flagged
observations among 600. Its 40 entirely clean blocks have diagnostic median
1.950251670583271 and interval [1.9129967018465255, 1.9761492467288064].
The slowdown flag persists in that subset. These are descriptive intervals;
neither canonical contamination nor complete runtime closure is independently
qualified. Every original and follow-up block remains retained.

An independent FLIP-v17 perf sample places 38.48% of cycles in the same
convolution function. Receipt: 9s, peak-RSS 0.11 GiB, minimum available
12421 MiB, load 1.26. [Profile hashes](profiles.json) preserve the raw
approximately 65 MB recordings outside public git. No optimization is merged.

The explicit SDR HDR-VDP aliases use the current native relaxed arithmetic.
On this 512×512 photo, v2 Q is 72.71885993948507 versus the adapted original
Octave value 72.6269108964876; v3 Q is 9.780021042500268 versus
9.780312182285215. Their observed deltas remain visible without an assigned
route-specific bound. They must not be presented as the strict-mode golden
validation results or as reproducing a dataset's unpublished invocation.

The full 60-case revised PyIQA SSIM/MS-SSIM grid is complete: 20 names on three
1152×1152 content pairs, all passing their assigned bounds. The corrected
fixed-factor names are included; the maximum across the whole grid is
2.0900847584881532e-6 (bounds differ by named convention). These are headline
scalar checks; auxiliary output columns and maps require separate validation.

The [independent FLIP-v17 20-block result](flip-v17-confirm20-interval.json)
is 1.9073448386822038, interval [1.873063545529841, 1.9386291310864587],
within ±3% precision. Its 23 flagged native observations among 200 remain
recorded. The eight-thread slowdown remains a descriptive flag.

Initial neural plans with a 1200-second budget were refused by the unchanged
1100-second supervisor before either arm executed. These setup refusals remain
in the raw cohort and explicit dispositions. Replacement plans use 1100 seconds
and new output paths; they do not relax a numerical or native sampling gate.

## Neural engine availability and reference setup

The original learned-metric worker accepts RTEN and optional ORT CUDA/WebGPU;
it has no ORT CPU route. The [native control](original-engine-refusal.json)
records its explicit `ort-cpu` refusal with zero completed rounds. These are
configuration refusals, excluded from score/speed projections. The
[retained experimental preparer](unsupported-ort-preparer.py.txt) preserves the
attempt's source; it is not an active benchmark command. The separate PyIQA
ports expose ORT CPU and keep their own native comparison plan.

The first original ST-LPIPS reference imports failed because its isolated
runtime lacked `torchsummary`. Version 1.5.1 is now installed. A new explicit
three-case plan replaces only those import-refusal cases; all earlier attempts
remain. Completed native ST-LPIPS scores and later valid comparisons are not
discarded. This setup correction changes no metric body or numerical bound.

Neural ratios now identify their executed engine (R = RTEN, O = ORT CPU).
Different source/engine cohorts retain separate artifacts even when their raw
directories share a basename. The index records each raw directory and every
artifact hash; publication refuses a remaining label collision.

## First completed original neural panel

All seven original learned methods (LPIPS Alex/VGG/Squeeze, ST-LPIPS Alex/VGG,
DISTS and DeepDC) pass their unchanged bounds at threads 1/8 on the retained
512×512 photo pair, counting the three successful ST-LPIPS runtime replacements.
Their largest scalar difference is 6.3964012003703274e-9; DISTS and DeepDC
match the canonical scalar exactly on both thread settings. This is one local
content/geometry, not a replacement for the broader frozen golden ladders.

Initial RTEN ratios exceed 1.08 for LPIPS-Alex, ST-LPIPS-Alex and DeepDC at both
thread settings, plus LPIPS-VGG and DISTS at one thread. LPIPS-Squeeze is faster
at both settings; VGG/DISTS are faster at eight threads. Each flagged case is
logged with both measured arm times and prepares a separate fixed follow-up.
The [first 20-block LPIPS-Alex one-thread interval](lpips-alex-t1-initial20-interval.json)
is 1.162649413650009, 95% interval [1.1455797183886958, 1.1843805843799495].
Its 19 flagged observations among 200 remain recorded; this is descriptive,
not an accepted regression decision. Warm-only perf captures exclude model
construction through the worker's existing completed-warmup event.

The eight-thread LPIPS-Alex follow-up is 1.3014597690629857, with
[95% interval](lpips-alex-t8-initial20-interval.json)
[1.2544969542532667, 1.3354180677802032]. It misses the requested ±3%
precision (±3.61%); a fixed additional 40 blocks are being acquired. All
33 flagged native observations among 200 remain in the initial result.

TOPIQ-FR and TOPIQ-FR-PIPAL pass their existing bounds at threads 1/8 on this
512×512 pair. Their maximum differences are 1.1920928955078125e-7 and
5.960464477539063e-8. Rust ORT CPU/canonical PyTorch CPU ratios are
1.046245452605012 / 0.8645151973914056 for TOPIQ-FR and
0.9099312717602236 / 0.911848700081464 for TOPIQ-FR-PIPAL. These remain
three-block descriptive comparisons rather than regression intervals.

All eight initially flagged neural cases have completed warm-call profiles.
[Their receipts and artifact hashes](neural-profiles.json) retain the exact
recording boundaries and resource measurements. The LPIPS-Alex samples show
RTEN's AVX-512 GEMM kernels executing at both thread settings; a sampled
function share does not explain the complete cross-implementation timing gap.
The DeepDC eight-thread profile reports 13s, peak-RSS 0.58 GiB, minimum
available 11689 MiB and peak load 2.15. Profiling does not change metric code.

LPIPS-VGG at one thread is borderline after the fixed 20-block follow-up:
1.082037733147578, [95% interval](lpips-vgg-pair-cid22-t1-initial20-interval.json)
[1.0688039346866853, 1.094678059488856]. The interval meets ±3% precision
and straddles 1.08. Its 22 flagged native observations among 200 remain
recorded; regression qualification is still absent.

The [DISTS one-thread fixed 20-block result](dists-pair-cid22-t1-initial20-interval.json)
is 1.1078520655203383, with 95% interval
[1.0988772441757937, 1.1152821768564318]. It meets ±3% precision
and remains above 1.08. All 200 native observations remain included,
including ten with resource flags; regression qualification remains absent.

For learned metrics, "warm" means resident immutable weights. Reference-image
features are recomputed inside each complete pair call. In particular, the
PyIQA CLI calls `Metric::evaluate`, whose port implementation uses `score_cold`;
the separate warm-reference API is not used by these comparisons. The
canonical PyTorch calls likewise evaluate both images inside the timed call.

The initial ST-LPIPS-Alex one-thread flag does not persist in the fixed
[20-block follow-up](stlpips-alex-pair-cid22-t1-initial20-interval.json):
1.0565326061418554, 95% interval [1.047118389726684, 1.0648778619171533].
The interval meets ±3% precision and lies below 1.08. All 13 flagged native
observations among 200 remain recorded. This is descriptive evidence; the
eight-thread follow-up remains separate. The last block's resource receipt is:

```text
run-heavy: done rc=0 26s | peak-RSS 0.97GiB | min-avail 12094MiB | peak-load 1.77
```

## Original WaDIQaM reference setup

The four original Chainer references now use an explicit inference wrapper.
Its NumPy device aliases live only in the author modules; Chainer's CUDA
backend remains intact. The initial control exposed a global-CuPy-slot shim
failure before producing scalar results or replacing plans. That failed
control and its source remain in the raw evidence.

The corrected control evaluates all four unchanged author models on the
512×512 pair. All six scalar outputs (two full-reference scores and distorted/
reference scores for both no-reference models) are bit-identical with gradient
recording enabled and disabled. [Control provenance](wadiqam-inference-control.json)
binds both policies, weights and source helpers. This validates the reference
fixture, not Rust parity; the actual Rust comparisons remain separate jobs.

Only four still-unexecuted plan files were replaced. Their author-default
versions remain as backups, and every existing input, bound, native setting
and budget is preserved. The queue reads their new identities when those jobs
start. Resource receipt:

```text
run-heavy: done rc=0 37s | peak-RSS 1.20GiB | min-avail 11974MiB | peak-load 3.14
```

## Completed PyIQA neural panel

TOPIQ-FR, TOPIQ-FR-PIPAL, WaDIQaM-PyIQA, AHIQ-PyIQA and DMM-PyIQA pass all
ten declared local scalar cases on the 512×512 pair (threads 1/8). AHIQ's
maximum difference is 1.1920928910669204e-8, with Rust/canonical ratios
0.45325707462175846 / 0.4066629018859161. DMM-PyIQA's maximum difference
is 1.6204631279276597e-5, with ratios 0.5208965994033374 /
0.13267680880383784, within its existing 1.78438e-4 bound.

These native calls use ORT CPU. DMM-PyIQA's separate documented RTEN ladder
misses its bound, so the compact README retains a partial-parity warning.
That warning, and original DMM's distinct broader partial-parity record, are
not erased by a local pass. The new ORT CPU observations do not qualify its
unmeasured size/content grid. Final-block resource receipts:

```text
AHIQ-PyIQA, 8 threads: run-heavy: done rc=0 242s | peak-RSS 5.34GiB | min-avail 7832MiB | peak-load 8.47
DMM-PyIQA, 1 thread: run-heavy: done rc=0 51s | peak-RSS 1.93GiB | min-avail 10930MiB | peak-load 2.78
DMM-PyIQA, 8 threads: run-heavy: done rc=0 27s | peak-RSS 1.93GiB | min-avail 11212MiB | peak-load 3.42
```

DeepDC at eight threads has completed its initial
[20-block follow-up](deepdc-pair-cid22-t8-initial20-interval.json):
1.1435249287058062, 95% interval [1.0973602070974315, 1.1685640565931696].
It remains above 1.08, but the ±4.04% interval misses the precision target.
A fixed additional 40 blocks are queued, retaining all 26 flagged native
observations among the initial 200. These intervals remain descriptive.

The [native-round audit](native-warm-round-audit.json) checks the retained
successful native arms against all three representations of the ten-round
contract: row count, timing observations and recorded round means. Its raw
manifest and hashes remain outside git. This is a structural check of the
acquired prefix, not an assertion of complete matrix coverage.

## Current HaarPSI native maps

The [current frozen-source recheck](haarpsi-current-maps/haarpsi-maps.json) completes all 24 map cases: four sizes, three content pairs and threads 1/28. On the 3840×2160 illustration, 41 native positions per thread setting are NaN where the original weighted map is finite. The other 22 cases have no nonfinite mismatches. The largest finite-pixel difference across all cases is 1.1233741833871491e-5; no external map bound is assigned. Each native map call also checks its scalar score bits against the scalar-only API. Scalar agreement does not establish pointwise map parity.

The probe links the current frozen HaarPSI rlib from source efae6b8fed9e57346157d92dffc3cdb57c44560e, verifying original crate files and the recorded luma-map adapter. Its 38s wrapper receipt reports peak-RSS 1.12 GiB, minimum available 11755 MiB and peak load 3.34. Raw planes remain outside public git, with hashes in the linked index.
## NIQE resident-parameter amendment, 2026-10-20

The original BasicSR callable opens its parameter archive on each score. Those
earlier numerical observations remain retained; its timing ratio is withheld
because parameter-file I/O differs from the declared resident-parameter scope.
The [new control and plan amendment](niqe-resident-amendment.json) compares both
images bit-for-bit against the original loader before warm acquisition. Its
2s receipt reports peak-RSS 0.28 GiB, minimum available 12697 MiB, load 1.61.

Only the 24 unexecuted NIQE commands in grid group 05 change. The other 96
cases, every numerical bound and all native sampling settings are preserved;
the original plan remains beside the amended plan with a backup suffix. Fresh
512×512 thread-1/8 comparisons are running. The existing 170-case native golden
check is independent of this reference-timing correction.

ARNIQA's initial one-thread three-block comparison passes with maximum scalar
difference 1.1920928955078125e-7, and has a descriptive ratio
1.1888128573459844. Its fixed 20-block interval is running. The warm profile
places 52.51% of sampled cycles in RTEN's AVX-512 GEMM; a sampled share does not
identify the cause of the complete cross-implementation gap. Profiling receipt:
25s, peak-RSS 0.19 GiB, minimum available 12365 MiB, load 1.94.
ARNIQA at eight threads also passes, with an exact scalar match on this pair,
and a descriptive initial ratio 1.4371044692018131. Its fixed 20-block
confirmation is running; both thread settings now have warm profiles. The
eight-thread profile receipt reports 9s, peak-RSS 0.21 GiB, minimum available
12098 MiB and load 2.81.

The [combined LPIPS-Alex eight-thread 60-block interval](lpips-alex-pair-cid22-t8-combined60-interval.json)
is 1.2666206178082384, 95% [1.248051466404187, 1.2913325106393114],
meeting the requested ±3% precision. All 600 native observations remain,
including 102 resource flags. Its slowdown flag persists; full regression
qualification remains absent. The last acquisition block's wrapper reports
8s, peak-RSS 1.12 GiB, minimum available 11991 MiB and load 1.34.
The [additional classical grids](additional-classical-grids.json) add five GPU
routes and the weightless `cw-ssim-pyiqa` Rust route. Together with the main
84-route CPU grid, this declares 90 runnable classical names and 2,160 warm
scalar cases. The GPU comparisons use the stated original CPU references;
they do not establish a comparison with an alternative GPU implementation.

[CVVDP map source inspection](cvvdp-map-semantics.json) identifies different
native-map and reference-heatmap definitions: the port sums upsampled bands per
channel before its norm; pycvvdp folds channels per band, reconstructs a
pyramid, transforms to `1 - JOD/10`, and stores float16. This is a semantic
extension recorded in DIVERGENCES.md, with source hashes. No new numerical map
parity result is claimed from that inspection.

## Current Butteraugli scalar and map acquisition

The [frozen-library probe](butteraugli-current-maps/butteraugli-maps.json) completes 24 cases: four sizes, three content pairs and threads 1/28. All scalar comparisons meet the existing FIR policy (relative 0.001 above canonical 0.001, absolute 0.001 otherwise); the largest absolute scalar difference is 8.344650268554688e-6. Every native map call also matches its scalar-only score bits.

The largest finite-pixel map difference is 0.0002071857452392578, with no nonfinite mismatches. No map bound is assigned. This accuracy-only probe supplies no comparative timing ratio. Its wrapper reports 91s, peak-RSS 1.85 GiB, minimum available 10316 MiB and peak load 4.04. Raw planes and per-arm maximum-RSS records remain outside git; the linked shards preserve their hashes.

## Prepared learned-metric size/content grid

The [28-route learned grid](neural-grid-plan.json) declares 672 warm scalar cases on the same twelve retained pairs at threads 1/28. It is prepared, not acquired; the full classical grids precede it. Existing model/reference contracts and numerical bounds are unchanged. Preparation took 16s with peak-RSS 0.02 GiB, minimum available 12328 MiB and peak load 3.05.

## Auxiliary-score and follow-up update, 2026-10-20

The [six auxiliary comparisons](nr-auxiliary.json) check reference-image columns independently of headline distorted-image scores. BRISQUE differs by 0.00537109375, PIQE by 2.842170943040401e-14, NIQE by 0.00013621654076612089 at both thread settings, and ARNIQA by 2.384185791015625e-7 at both settings. All satisfy their unchanged case bounds. This covers these acquired cases, not every NR input or auxiliary column. The 6s probe reports peak-RSS 0.13 GiB, minimum available 12459 MiB and peak load 3.92.

[DeepDC eight-thread combined 60-block results](deepdc-pair-cid22-t8-combined60-interval.json) give 1.114711566363791, 95% [1.0977631953028684, 1.1274934746291763], meeting the precision target while retaining all 69 resource flags among 600 native observations. Last block: 21s, peak-RSS 1.72 GiB, minimum available 11502 MiB, load 3.97.

[ARNIQA one-thread initial 20 blocks](arniqa-original-cid22-t1-initial20-interval.json) give 1.1507064694296374, 95% [1.1374999626952738, 1.1772551172274746], meeting precision with 19 resource flags among 200 observations. Last block: 35s, peak-RSS 0.90 GiB, minimum available 11854 MiB, load 2.28. Its eight-thread 20-block interval misses precision; a fixed additional 40 blocks retain that original stage. These intervals remain descriptive.

The [GPU-grid profiles](gpu-grid-profiles.json) retain the two 64×64 DSSIM CUDA recordings. Host-side CubeCL CUDA enqueue accounts for 30.21% / 27.81% of sampled CPU cycles at threads 1/28; this is not GPU-kernel timing or proof of the complete slowdown's cause. Profiling receipts: 8s / 7s, peak-RSS 0.33 GiB each, minimum available 11971 / 12147 MiB, peak load 3.85 / 4.71.

## Later learned and revised-source update, 2026-10-20

The summary retains subsequent MANIQA, MUSIQ, PaQ2PiQ and WaDIQaM observations and prefers the revised PSNR source when that route has completed. Every older observation keeps its source identity. [Runtime controls](runtime-controls.json) check resident first-evaluation probes in both NN families, author CUDA references for CVVDP and IW-SSIM, and a separate compatible NumPy 1.25.2 MUSIQ runtime. These are setup/accuracy controls, not comparative timing claims or full cold-matrix coverage.

The [GPU reference amendment](gpu-reference-fixture-amendment.json) retains the first IW-SSIM setup failures and replaces the incomplete reference environment with the previously controlled author runner. Its 64×64 refusal is a genuine size-domain observation; the 255×257 scores are acquired without assigning a new bound.

## DSSIM reference lineage finding, 2026-10-20

The earlier DSSIM C reference was built from `lilith/dssim` at `eec33f55`,
an older fork baseline. These results remain alternative comparisons; they
do not establish agreement with the author repository. The
[lineage record](dssim-reference-lineage.json) identifies all affected plans.
Primary canonical comparisons are being replayed against author DSSIM 3.5.1
at `kornelski/dssim` `0e44c9b7`, retaining the same native code and bounds.
The older-fork map results are retained with that identity too.

## Completed initial learned pass, 2026-10-20

The initial learned pass has observations for all 28 requested non-PieAPP names. WD passes at both thread settings: one-thread absolute difference 0, Rust/canonical time ratio 0.11094338612430271; eight-thread difference 1.862645149230957e-9, ratio 0.07819718004981498. Last blocks: 293s / 84s, peak-RSS 2.34 / 2.35 GiB, minimum available 10029 / 11760 MiB, peak load 5.57 / 4.89. These are representative 512×512 scalar comparisons, not completion of the multi-size, map or cold matrix. The three original STLPIPS import failures remain retained with their already completed replacements.

## Author DSSIM and revised FLIP maps, 2026-10-20

[Author DSSIM maps](dssim-author-maps/index.json) match native scalar scores and all 120 maps bit for bit across 24 cases (four sizes × three pairs × threads 1/28). Both sides also retain identical map/plain scalar bits. Author source is 3.5.1 at `0e44c9b7`; only its dependency lock was resolved in an owned export. The wrapper reports 143s, peak-RSS 1.36 GiB, minimum available 11942 MiB, peak load 2.07, including the one-case public FLIP control.

[Revised FLIP maps](flip-current-maps/index.json) are bit-identical in all 24 cases. Their f64 and row-major f32 scalar means remain separate. Wrapper: 116s, peak-RSS 1.22 GiB, minimum available 11447 MiB, load 4.82. These are accuracy observations, without new map bounds or speed claims.

The [current 38-route worker](current38-worker.json) verifies the published Rust inputs, runtime AVX-512 and NR feature coverage, with the declared existing benchmark adapters. Its manifest uses the corrected author DSSIM reference; both earlier scheduling manifests remain retained.

## Completed CUDA author-reference controls, 2026-10-20

[Reference-device selection](cuda-reference-headline-selection.json) retains both measured devices. At the same 512² photo pair and one host thread, canonical medians were 33.176356 ms CUDA versus 226.122096 ms CPU for CVVDP, and 59.358956 ms CUDA versus 109.837588 ms CPU for IW-SSIM. The headline uses the measured CUDA cohort: Rust/canonical 0.12406038505253561 for CVVDP and 0.1429021679559189 for IW-SSIM. These are descriptive acquisition medians, without a device-selection confidence claim or extrapolation to other cells. Both 1/28-thread CUDA cases completed for both methods. CVVDP maximum score difference is 9.5367431640625e-7 against its unchanged 0.001 bound; IW-SSIM difference is 1.4385154963880709e-6 without an assigned bound. Final block resource receipt: rc=0, 8s, peak-RSS 1.58 GiB, minimum available 12387 MiB, peak load 2.37.

The complete CW-SSIM/PyIQA grid passes all 24 cases; maximum absolute difference is 5.551115123125783e-15. Final block resource receipt: rc=0, 67s, peak-RSS 5.80 GiB, minimum available 7927 MiB, peak load 10.64. The declared broader matrices remain incomplete.

## Separate first-evaluation evidence and boundary review

The [resident-model first-evaluation ledger](nn-first-evaluation/index.json) retains new-process first calls separately from all warm timing rows. It excludes immutable-model construction and input/model file I/O, and does not measure full process startup. Numerical decisions are recomputed from retained arms; failures withhold public ratios. Three alternating blocks per acquired case remain descriptive.

The [classical boundary review](cold-boundary-review/index.json) classifies all 84 warm-grid names: 65 first-evaluation candidates, 13 with resident setup excluded, two missing first-call output, two with parameter-I/O scope differences, NIQE with an earlier validation evaluation, and one with a precreated thread runner. Static eligibility is not acquired cold coverage. Base `hdrvdp`/`hdrvdp3` typed-input limitations remain distinct from the two included `*-aic2026` SDR routes.

[Revised FLIP profiles](revised-flip-profiles.json) identify the current native binary, sampled stacks and resource receipts. They confirm AVX-512 convolution and batched power-function dispatch; they do not establish the complete cause of the eight-thread timing candidate.

## Original Google Butteraugli maps

[All 24 original-Google map checks](butter-original-maps/index.json) match scalar scores and every finite map pixel bit for bit: four sizes, three contents and requested worker budgets 1/28. Both implementations use one effective worker. This uses original Google revision `71b18b6`, separately from libjxl Butteraugli. The accuracy wrapper reports 63s, peak-RSS 1.43 GiB, minimum available 12704 MiB and peak load 2.60. No map or scalar bound was assigned.

[Additional CUDA map-probe development](extra-gpu-map-scope.json) was canceled at owner request before compilation or inference. That draft is unpushed; the existing scalar comparisons continue.

## Revised base HDR reference controls (2026-10-20)

The [current 40-route worker](current40-worker.json) includes master’s revised
`hdrvdp`/`hdrvdp3` sRGB-display entry points and AVX-512 features, without benchmark
code overlays. These two names still need paired canonical comparisons.

The [retained reference controls](hdrvdp-reference-controls/controls.json) verify
14/14 original raw fixtures and preserve both runtime variants. Official
HDR-VDP-2 sRGB goldens at the unchanged 0.005 bound produce 20/24 failures with
Octave 6.4/image 2.12, versus 0/24 with unchanged image 2.20.1 `imresize.m` and
`private/interp_method.m` overlaid. The selected variant’s largest absolute
difference is 2.842170943040401e-14. The full newer image package is not installed.
All captured runtime/source/dependency snapshots agree before and after.

Nine resident-parameter controls also pass: exact arrays, copy-on-write,
sealed-cache and missing-key refusals. These checks do not establish
original/resident metric-score identity, native parity, an official HDR-VDP-3
bound or comparative timing qualification. Those remain separate pending work.
The preflight wrapper reports rc=0, 901s, peak-RSS 0.12GiB, minimum available
14632MiB and peak load 1.69; much of its elapsed time is outside metric scoring.

## Headless HDR photo controls (2026-10-20)

All four actual warm cell protocols and four default/resident ABBA controls
pass at 512² and threads 1/8. [The receipt](hdrvdp-photo-controls.json) retains
native resource flags and the source-bound observations. HDR-VDP-2 differs from
the official original by 0.0007070822066168603, within its unchanged 0.005 bound.
HDR-VDP-3 differs by 1.7763568394002505e-15; no official external acceptance bound
is assigned. These are photo controls, not a full-grid claim or a paired ratio.

An independent syscall trace identified X11 initialization outside the metric
clock. The new controls use Octave’s `--no-window-system`; every old observation
remains retained. The four-case score-control batch reports rc=0, 235s,
peak-RSS 0.20GiB, minimum available 15026MiB and peak load 1.82. The actual cell
protocol batch reports rc=0, 40s, peak-RSS 0.19GiB, minimum available 15331MiB
and peak load 6.88. Both retain per-child process memory receipts.

## RTEN configured thread budgets, 2026-10-20

The [source review](../iqa_compete_2026-10-16/rten-thread-budget-review/README.md)
binds RTEN 0.27.0, Rayon 1.12.0 and rayon-core 1.13.0 to retained dependency
fingerprints. The coordinator independently verified all four registry archives,
the nine inspected dependency files and every recorded local helper hash.
The warm wrapper and resident first-evaluation comparator set RTEN_NUM_THREADS
before model construction. PaQ2PiQ uses that default pool; paired DeepDC and
LPIPS use two explicit half-width pools when more than one worker is requested.
No configured-cap mismatch was found. Actual utilization, pool creation success
and canonical runtime cleanliness remain unmeasured by this source review;
existing timing flags and numerical decisions retain their qualifications.

## Full-call DSSIM comparative command, 2026-10-20

The [new command receipt](local40-scoped-fullcall-dssim-command.json) selects the source-bound author DSSIM reference with context construction and destruction inside each call. It preserves the forty-route ascending scheduling order and every numerical bound; four HDR and six VMAF model routes still withhold comparable timing ratios. Earlier plans and results remain unchanged.

[Three further CPU profiles](grid-latest-profiles-2026-10-20/index.json) retain FLIP at 4K/28 workers and the small screen-pair DSSIM/SSIMULACRA2 GPU flags. CPU samples describe host execution and do not measure GPU kernel time or establish the full cause of a timing gap. Resource receipts accompany every profile.

## Canonical runtime evidence limits, 2026-10-20

The [bounded three-route source review](runtime-cleanliness-review.md) finds no independent canonical foreign-CPU observations in the selected retained PaQ2PiQ warm and DeepDC/VGG first-call arms. Whole-process memory/time receipts and admission locks do not establish call-local cleanliness. Numerical observations remain unchanged; timing intervals retain their descriptive qualification.

## VMAF full-call reference source, 2026-10-20

The [new reference source and control driver](../iqa_compete_2026-10-16/vmaf-fullcall-reference.md)
move embedded model loading and destruction into each original scoring clock.
The coordinator independently reviewed the C++ relocation, Python syntax,
frozen bundle hashes and original compiler flags. Actual linking and twelve
old/new exact-score controls are queued separately; this source review does
not promote the six model routes or change any existing plan, result or bound.

[Two additional 4K CPU profiles](grid-4k-profiles-2026-10-20/index.json) retain the FLIP-v17/28-worker and WS-HV/one-worker flags. Each projection binds its unchanged binary/configuration and raw recording, and includes the resource-wrapper receipt. CPU samples do not establish the complete cause of the Rust/canonical timing gaps. Fixed twenty-block confirmations remain separate.

## VMAF full-call controls completed, 2026-10-20

The previously queued [reference controls](vmaf-fullcall-controls.json) now pass
all twelve old/new comparisons: 24 fresh children and 72 finite, identical
binary64 observations. The coordinator independently verified every case hash,
arm log hash and score-bit identity. Build/control wrapper: 4s, peak-RSS
0.13 GiB, minimum available15054MiB, peak load2.16. This qualifies the reference
fixture only; comparisons against the normal current40 Rust worker are
prepared separately and preserve all original numerical bounds.

## Current forty-route comparison command, 2026-10-20

The [current full-call command](local40-scoped-fullcall-command.json) selects the
controlled DSSIM and six VMAF model references. Their per-call setup/destruction
now falls inside both clocks. Four HDR routes retain spectral-parameter timing
caveats; all original numerical bounds remain unchanged. The source-reviewed
plan has eighty declared cases, with the remaining current-worker measurements
acquiring separately from historical sources. Preparation does not establish
eighty completed comparisons.

## Strict VMAF plan verification completed, 2026-10-20

The [receipt-verifier controls](vmaf-plan-verifier-controls.json) pass 22 synthetic
cases in normal and optimized Python and verify the actual retained reference
evidence. The new twelve-case plan preserves every active case, argument, input
and numerical bound while binding the fuller retained evidence. The combined
wrapper reports 6s, peak-RSS0.03GiB, minimum available14924MiB, peak load2.87.
The earlier source-JSON parsing failure remains recorded.

The [frozen original preparer](../iqa_compete_2026-10-16/frozen_vmaf_fullcall_plan_20261020.py)
is an exact historical source copy for the active cohort, whose actual records
passed independent full protocol review. It is not the general strict verifier;
new preparation uses [the strict successor](../iqa_compete_2026-10-16/vmaf-fullcall-plan.md).
No active plan or measured result is replaced.

## Complete changed-set command, 2026-10-20

The [41-route command](local41-scoped-fullcall-command.json) adds the previously
omitted Daala preprocessing path, uses the stricter VMAF retained-evidence
verifier, and preserves the measured ascending scheduling order. It declares
82 one/eight-thread cases, with eight HDR scope caveats. Preparation passed
in 4s, peak-RSS0.08GiB, minimum available14955MiB, peak load1.47; these are
not82completed measurements. Every previous manifest and result remains retained.

## Additional first-call and original HDR diagnostics, 2026-10-20

Missing: the new cold cohorts and fixed-stage confirmations remain incomplete;
the original HDR runtime/direct-call and parameter-storage gates remain open.
The [initial classical cold flags](classical-first-initial-flags.json) retain
eleven cells above 1.08 across GMSD, FLIP and FLIP v1.7. These three-block ratios
are separate from warm observations. Fixed twenty-block confirmations have been
queued for all eleven cases; no regression is accepted from these initial data.

The [partial original HDR grid](hdr-original-partial12.json) records 12 of 48
declared cases. HDR-VDP v2's 1024×1024 photo difference is 0.009351851113351017
at both worker budgets, above the retained 0.005 golden-check level. V3's largest
completed difference is 7.105427357601002e-15, with no assigned bound. The local
Octave 6.4/image 2.12 runtime carries two reviewed upstream resize M-file overlays.
Native parity and every timing ratio remain withheld; this does not identify
the cause of the v2 difference. Per-case resource receipts and raw artifact
hashes are retained in both projections.

The [ten-route constructor reference controls](classical-first-ten-reference-controls.json)
retain 20 odd-size photo/screen cases and 80 fresh original/candidate processes.
Every corresponding scalar bit pattern agrees. These controls cover BRISQUE,
NIQE, both NLPD routes, both IW-SSIM routes, CVVDP, both libjxl Butteraugli
routes and SSIMULACRA2; they supply no native timing or numerical acceptance.
Each route retains its source and resource receipt. The IW-SSIM/PIQ control
used peak-RSS 0.85 GiB, minimum available 14,571 MiB and peak load 3.43.

ARNIQA's [one-thread fixed20 confirmation](arniqa-first-original-cid22-t1-first-confirmation20.json)
reports first-evaluation Rust/canonical ratio 1.1343200009907863 with 95% interval
[1.1228163255741586, 1.151777720297966]. Numerical parity passes and interval
precision meets the three-percent criterion. Its [eight-thread initial20](arniqa-first-original-cid22-t8-first-confirmation20.json)
reports 1.1764430166284257 [1.1123383245857097, 1.2463601494859644]; precision
fails, so a fixed additional40 is queued and all60 will be retained. These are
first evaluations after resident model construction; canonical runtime cleanliness
is not independently qualified and neither interval is an accepted regression.

## Cold-call and supported-runtime update, 2026-10-21

The [acquired cold-call panels](acquired-cold-panels-2026-10-21/index.json) retain
completed CPU and CUDA cases with per-arm maximum RSS and wrapper receipts.
These initial three-block ratios remain separate from warm results and fixed
confirmation stages. Unassigned numerical bounds remain unqualified.

[Completed CPU recordings](cpu-first-profiles-2026-10-20/index.json) and
[IW-SSIM CUDA t1](iwssim-first-profile.json),
[IW-SSIM CUDA t28](iwssim-t28-first-profile.json) and
[Butteraugli CUDA 64²](butteraugli64-first-profile.json) retain source and
recording hashes. Whole-process stacks and CUDA traces do not isolate the cold
clock or establish a causal share of a comparison gap. Two capped GMSD captures
retain [13/100](gmsd-t28-first-cap-adopt-v3.json) and
[19/30](gmsd-t28-capped-adopt-v3.json) completed workers as partial diagnostics.
[DSSIM lifetime controls](dssim-first-lifetime-controls.json) separate actual
original-only bit checks from synthetic clock-order controls; no timing ratio
is inferred from either control.

The [supported Octave 11.1/image 2.20.1 HDR-v2 controls](hdr-octave111-v2-controls.json)
reproduce the 1 MP photo difference: original 61.71098098074218, retained native
61.701629129628834, absolute difference 0.009351851113343912. The original agrees
with its legacy-runtime result to 7.105427357601002e-15 on both retained controls.
The legacy runtime therefore does not explain this case. The existing 0.005
golden-fixture level remains a fixture check; no universal bound or HDR timing
ratio is assigned. The original 1 MP process recorded maximum RSS 628,252 KiB.
The two-control wrapper used 14 s / 0.09 GiB peak RSS / minimum available
14,118 MiB / peak load 2.40 and exited 1 because its separate newly built v3 MEX
fixture failed. Completed v2 controls are retained; v3-only followups restore
the original M-file dispatch without changing author sources.

The [two completed v3 controls](hdr-octave111-v3-controls.json) use that original
M-file dispatch on the same modern runtime. The largest difference from retained
native initialization scores is 1.0658141036401503e-14 across the 64² and 1 MP
photos. The 1 MP original maximum RSS is 667,644 KiB. The successful wrapper
recorded 13 s / 0.09 GiB peak RSS / minimum available 13,439 MiB / peak load 2.13.
No universal v3 bound or timing ratio is assigned.

The [BRISQUE 64² photo finding](brisque-64-numerical-finding.json) repeats a
0.0865478515625 difference against PIQ 0.8.0/torch 2.14.1+cpu at both one and
28 workers, above the unchanged 0.015138428 bound. Both timing ratios remain
withheld. Separate existing-wheel/CPU-dispatch diagnostics are pending; the
cause is not assigned and no oracle environment exception is accepted.
[Historical runtime/control harness sources](supported-hdr-runtime-sources-2026-10-20/index.json)
retain each exact setup and wrapper snapshot; frozen assets remain outside git.

### BRISQUE runtime controls completed (2026-10-21)

[Eight original-only controls](brisque-runtime-controls-2026-10-21.json) use both installed torch 2.14.1 wheels
(+cpu and +cu130 running on CPU), with default AVX-512 and effective ATen AVX2.
Every configuration gives the same 64² score 75.73614501953125 and 255×257 score
89.107421875. Those tested wheel/dispatch choices do not explain the 64² native
difference. The computed R_GGD and R_AGGD grids differ from the frozen native
tables by one ULP in 329 and 327 entries respectively; this is not established
as the cause. The numerical failure and unchanged 0.015138428 bound remain
retained. No timing ratio or exception is accepted.

The CPU-wheel wrapper recorded 7 s / 0.33 GiB peak RSS / minimum available 14,227 MiB
/ peak load 2.79. The CUDA-wheel-on-CPU wrapper recorded 8 s / 0.80 GiB / 14,055 MiB
/ 2.52. Per-process GNU time receipts and exact source snapshots are retained.

The GMSD 64² photo first-call [one-thread initial20 interval](gmsd-photo64-t1-first-confirmation20.json)
is 1.207 [1.174, 1.228]; its precision target is met. The [28-thread interval](gmsd-photo64-t28-first-confirmation20.json)
is 16.737 [16.130, 17.618] and misses the precision target. A fixed additional
40-block stage is queued; every original block remains retained. Numerical
checks pass, while runtime cleanliness and regression acceptance remain
unqualified. Final acquisition wrappers both recorded 1 s / 0.02 GiB peak RSS /
peak load 1.52; minimum available was 14,435 MiB at one thread and 14,426 MiB at28.

The FLIP v1.7 255×257/t28 [first-call initial20 interval](flip17-photo-odd-t28-first-confirmation20.json)
is 1.459 [1.369, 1.556], missing the precision target. The fixed additional
40-block stage retains the full initial stage; numerical acceptance remains
unassigned. Its final acquisition wrapper recorded 2 s / 0.22 GiB peak RSS /
minimum available 14,294 MiB / peak load 1.41.

### Backend and NIQE numerical follow-up (2026-10-21)

The [oneDNN controls](brisque-backend-controls-2026-10-21.json) narrow
BRISQUE's 64² photo miss. With oneDNN's effective ISA limited to AVX2, the
unchanged PIQ source returns 75.65032958984375, versus Rust 75.64959716796875
(difference 0.000732421875). The retained default original is 75.73614501953125.
ATen-only AVX2 and the separate MKL AVX2 request do not change that default
score; the latter request does not prove MKL's effective ISA. Disabling oneDNN
gives 75.47998046875. These counterfactual diagnostics do not replace the
default canonical failure or its 0.015138428 bound. The wrapper recorded 7 s,
0.33 GiB peak RSS, minimum available 13,998 MiB and peak load 2.28.

The [native stage controls](brisque-native-original-stages.json)
reproduce both retained native scalars exactly. Actual native and original
grayscale planes are bit-identical on both photos. The default original differs
in 20 of 36 statistical features at 64² and 26 of 36 at 255×257. The oneDNN
AVX2 control changes two 64² quantized shape parameters into exact agreement
with Rust. This narrows numerical attribution to statistical feature computation;
it does not clear the default-backend failure. The native diagnostic wrapper
recorded 1 s, 0.22 GiB peak RSS, minimum available 14,262 MiB and peak load 1.87.

The [NIQE odd-photo finding](niqe-odd-numerical-finding.json) retains
three failing blocks at both 1 and 28 workers: Rust 18.807175290717275 versus
BasicSR 18.799147370663427, difference 0.008027920053848447 above the unchanged
0.000602329 bound. This is separate from the documented 64² size refusal.
Root cause and a code correction remain unverified. Earlier golden checks and
all original observations remain retained.

### Remaining cold-flag profiles (2026-10-21)

The [four additional complete CPU profiles](cpu-first-profiles-supplemental-2026-10-21/index.json)
bring completed distinct CPU cases to 19. Together with the retained partial
GMSD captures and [5-of-30 FSIM capture](fsim64-t28-capped-adopt-v1.json),
these 21 CPU flags have profiling evidence. Partial captures
remain partial; no completed scorer is repeated for publication. The five
additional CUDA profiles bring distinct GPU flag profiles to eight. These are
whole fresh-worker diagnostics; their instrumented times and sampled shares
do not qualify comparative timings or a causal fraction of a gap. Each linked
projection retains its complete resource-wrapper receipt.

The [SSIMULACRA1 64²/t1 initial20 interval](ssim1-photo64-t1-first-confirmation20.json)
is 1.250 [1.241, 1.290], missing the unchanged precision target. Its fixed
additional 40-block stage is queued and retains every original observation.
No timing exception or regression acceptance is assigned.

### NIQE feature attribution (2026-10-21)

The [stage cross-rescore](niqe-native-original-stages.json) reproduces
both retained scalars before comparing intermediates. All 65,535 native grayscale
samples equal the original on the 255×257 photo. BasicSR's final MVG operations
on native block features yield 18.80717529071772, only 4.440892098500626e-13
from native's 18.807175290717275. Original features reproduce BasicSR's
18.799147370663427 exactly. The observed score gap is therefore attributable
to block-feature differences in this control, rather than RGB ingress or the
final native covariance calculation. Of 144 feature entries, 61 differ: 7 at
scale one and 54 at scale two. Two second-scale shape parameters differ by
one 0.001 grid step. The precise operation responsible remains unverified;
the numerical failure and original 0.000602329 bound remain unchanged.

Resource receipts: native diagnostic build/control 11 s, peak RSS 0.42 GiB,
minimum available 13,658 MiB, peak load 1.26; original feature control 2 s,
0.27 GiB, 13,840 MiB, 1.22; cross-rescore under one second, 0.04 GiB,
14,108 MiB, 1.18. These are diagnostic-process resources, not metric timings.

### Additional acquired cold cases (2026-10-21)

The [immutable batch manifest](cold-panel-publications.json) now retains 249
initial cold cases, including [this batch of 57 additions](acquired-cold-additions-2026-10-21-v1/index.json).
Every previously published record was revalidated against its retained values and remains
in its original shard. New cases are exported once; the README checks combine
the batches and refuse duplicate observations or changed index/shard hashes.
The exporter caps each new batch at 150 cases to keep generated indexes and
shards below 30 KB. Any further acquired cases remain explicitly outside this
publication cut. Initial ratios remain descriptive and separate from fixed-stage
intervals, warm measurements and numerical controls.
