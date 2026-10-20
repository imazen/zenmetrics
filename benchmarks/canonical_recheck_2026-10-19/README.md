# Local canonical rechecks

Missing: the full per-name grid, remaining classical and learned references,
map comparisons and qualified regression intervals. These are scalar checks on
Ryzen 9 7950X / WSL2. The first six-name panel uses one 512×512 photo pair.
Additional original MATLAB, corrected VIF and libvmaf cohorts are indexed in
[cohorts.json](cohorts.json); its observed/expected counts expose partial runs.
The per-case TSVs retain exact dimensions, threads, bounds, deltas and ratios.
They do not replace multi-size or content validation.

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
