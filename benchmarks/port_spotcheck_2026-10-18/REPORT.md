# Canonical port checks: evidence status, 2026-10-19

Existing metric-path change on master: commit
[`b52fadb6e996e89b3d28fa1bcb45c2070afcf9cd`](https://github.com/imazen/zenmetrics/commit/b52fadb6e996e89b3d28fa1bcb45c2070afcf9cd)
changes BRISQUE convolution, luma, normalization, resize and reductions,
including f64 `LaneSums8` in place of f32 reduction trees. Its commit timestamp
is 2026-10-19 03:31:53 UTC; remote-tracking ancestry was verified on October 19.
It is already on master and has not been reverted by this review. Its
[committed study](../brisque_simd_2026-10-19/README.md) has separate provenance;
this review has not independently remeasured its score or timing effects.
The older unmerged `ef9763f2` BRISQUE candidate uses different source and cannot
stand in for this master implementation. Retain that distinction when deciding
which code and benchmark cohort to use.

Missing: complete executable/source/model bindings for the inventory, the classical size/content/thread matrix, the
learned-method matrix, qualified ratio intervals, and the final site-readable
`summary.tsv`. The replacement timing panel has **0/19 qualified metric
configurations**. This denominator describes that panel, not the complete CLI
inventory. No canonical speed or parity conclusion follows from this note.

The 19 configurations are `brisque`, `butteraugli`, `butteraugli-original`,
`ssim2`, `vsi`, `vsi-k2`, `vsi-k3`, `vsi-k4`, `vsi-nodownscale`, `mdctpsnr`,
`cambi`, `cambi-aic2026`, `psnr-y-libvmaf`, `vmaf`, `vmaf-4k`, `vmaf-neg`,
`vmaf-neg-studio601`, `vmaf-studio601`, and `vmaf-v1`.

## Parity findings

The [VIFp RGB grid](primary-vif.md) has 22 numerical findings out of 24
completed cases against its unchanged 1e-9 absolute bound. Maximum difference
is 4.093188010756421e-9. The candidate private f64 luma-coefficient correction
remains unmerged and unqualified; regression and `vifvec` controls are pending.


The initial [64×64 S67/CCM15 photo crop](vsi-photo-finding.md) fails against
the image-2.12.0 reference runner:
Rust `vsi` and `vsi-nodownscale` return 0.9567964257437082 versus
0.94679144074028154 from the unmodified primary reference. The absolute
difference, 0.010004985003426614, exceeds the unchanged 1e-4 bound.
Independent pixel-ingress and original-source controls reproduce the failure.
A [resize-dependency control](vsi-resize-runtime.md) explains this local
mismatch. The same author source with unchanged image-2.20.1 resize M-files
returns 0.95679610895892431, for an absolute Rust difference of
3.1678478384922215e-7, within the unchanged 1e-4 bound. Three procedural
low-contrast cases also pass with that dependency. No metric code changed.
This is an explicitly identified two-M-file overlay on Octave 6.4, not a
complete newer package installation; its full panel is a separate experiment.
The earlier 2.12 observations and synthetic-input partial passes are retained.

## Why the replacement panel remains unqualified

The local October 19 port-check remeasurement batches pooled samples from
separate batch windows and paired them by index. Those indices do not identify
paired experimental rounds. Their reported confidence intervals cannot qualify
the 1.08 regression threshold. Contention, incomplete executable/source binding,
and differing timed work also require resolution. Preserve these raw batches
as diagnostic evidence; their earlier CONFIRMED/CLEARED labels do not establish
qualification. This disposition concerns those port-check batches, not every
historical result in the repository.

Source review identified these concrete scope gaps:

| Family | Difference requiring a matched experiment |
|---|---|
| BRISQUE | The inspected Rust driver evaluates reference and distorted images; the inspected canonical adapter evaluates only the distorted image. The crate's PIQ lineage also needs an explicit contract when comparing with PyIQA. |
| libvmaf routes | Canonical input planes and picture preparation precede timing; Rust's RGB ingress and scorer lifecycle remain inside timing. The inspected Rust snapshot caches parsed VMAF models: repeated model parsing is **not** an established explanation. |
| libjxl routes | Image-bundle preparation and scorer/pool setup precede canonical timing. Bind normalization, norm selection, and lifecycle scope before accepting a ratio. |
| mDCTPSNR | The inspected canonical adapter starts a process and handles PPM input per sample; the Rust arm remains resident. Separate equivalent scoring work from process and file costs. |

These findings describe inspected adapters. Current source hashes and build
receipts cannot retroactively identify an older executable or its observations.
Any rebuilt adapter starts a new explicitly identified cohort.

The frozen prerequisite attempt also rejects a Triton plugin's unresolved
`libtriton.so` dependency under its recorded loader environment. Tagged upstream
source describing optional plugins does not prove that an installed wheel is
unreachable from a workload. Source inspection alone does not clear runtime
closure. The native-only route likewise needs its own complete closure.

## Separately verified BRISQUE test follow-up

The existing, unmerged
[`opt/brisque-simd`](https://github.com/imazen/zenmetrics/tree/opt/brisque-simd)
advanced from `ac542221f0908ed423537bbc87cbcadc3e4e4511` to
`ef9763f23b1cfd429140b6dbe9be5b432f7bb538`. The follow-up adds strided GRAY8,
neutral RGB8/GRAY8, and odd/minimum-size checks and qualifies a finite-input
comment. Independent diff review found no scoring-expression, public-API,
threshold, or expectation change in that follow-up.

At the latter commit, the CI-pinned sibling snapshot passes the locked BRISQUE
library and behavior suites: 9/9 and 10/10 tests respectively. These establish
the scoped test result, not canonical parity, ARM execution, performance, or
approval to merge the underlying optimization. The live remote branch SHA was
independently checked on 2026-10-19. Master receives no metric-path changes.

## Fixed-factor source tooling

[Oracle source patching](../iqa_compete_2026-10-16/fixed-factor-oracle.md)
landed at `9bb02ed493db667505828642d03872306538021e`. It creates separately
identified scratch copies of hash-pinned FSIM, VSI and PyIQA sources with one
factor expression changed. MATLAB FSIM/VSI remain primary references; PyIQA
FSIM/VSI are diagnostic variants, while PyIQA is primary for its SSIM names.
Source identities, factor choice and the SSIM `downsample=True` runtime
requirement remain explicit.

Independent artifact controls and an actual CLI alias regression passed before
landing. Those controls verify source preservation and patch provenance. They
execute no metric and do not qualify a fixed-factor implementation, score or
timing cell. The corresponding metric experiments remain missing.

## MAD helper source tooling

The [MAD helper ownership overlay](../iqa_compete_2026-10-16/mad-helper-oracle.md)
landed at `64548ff67d5d79e7663125faa11831d076fc50a8`. It preserves the
original source and creates a separately labeled compatibility reference.
The exact original helper build faults during cleanup on the tested Octave
6.4 environment; three ownership edits retain the numeric body. The
compatibility build completed six helper cases, with 18 output arrays
independently verified byte-for-byte against the related in-repo MATLAB shim.
That shim is corroborating evidence from a related port.

Independent artifact controls and actual CLI alias/normal-patch checks passed.
All three remote files match the reviewed source hashes. These checks cover
the source patch and its provenance; full MAD scores, timing and oracle
integration remain unqualified. The original helper artifacts are retained.
[Repeatable artifact recipes](../iqa_compete_2026-10-16/oracle.just) cover
both fixed-factor copies and this helper overlay. They execute no metric.

## Native adapter preparation

The opt-in [libjxl RGB-input scope](../iqa_compete_2026-10-16/native-rgb-scope.md)
landed at `62aaf7e94acaee2ccbe9b4f6c16f0ec3416a3559`. It moves metadata,
image bundles, RGB conversion, the used comparator and per-call object
release inside a lexical timed scope. Raw input file reads and reusable
pool lifecycle stay outside. The default scope retains its existing staging
and header placement. Independent source review and compiled-binary
argument/failed-open controls passed; the remote payload matches the review.
No scoring API ran in those controls.

Pinned libjxl source resolves the Butteraugli scalar reducer: `ComputeScore`
sets the reference on every call, then returns the maximum of the diffmap.
Historical Rust receipts explicitly select `butteraugli_max`. This resolves
the reducer/column question without qualifying the historical batches. A
current canonical executable hash and an older invocation path alone cannot
bind that executable to past observations.

The opt-in adapter starts a new cohort. Complete executable/source closure,
matched cold/warm work, numerical parity and paired ratio intervals remain
missing. The exact selected color-transform path still needs build-specific
binding before claiming equivalent preprocessing. **0/19 remains unchanged.**

## Source inventory and primary SSIMULACRA2 build

The [source census](inventory-source.md) landed at `39c30081`, with dashboard
classification at `c0009175`. It covers all 126 exact CLI registry names in
127 records: 124 included names, two excluded PieAPP aliases and one non-CLI
AIMOS exclusion. Primary-reference languages and source identities are separate
from substitute runners. Unknown gates, revisions, weights and executable
bindings remain explicit. Three hash-indexed shards stay below 30,000 bytes
each. Independent corruption, preservation and exact-coverage controls pass.
This source census does not replace a newly frozen executable inventory.

The [primary SSIMULACRA2 builder](../iqa_compete_2026-10-16/ssimulacra2-primary.md)
landed at `e0b26212`. It builds unchanged Cloudinary v2.1 source with its
required Little-CMS 2.13 dependency. Archive hash checks precede all output
mutation; fresh-directory reservation, path quoting and stage failures have
independent controls. A contained build succeeds, and its usage-only invocation
returns before image reads or scoring. Static Highway source/compiler provenance
remains unbound. Rust numerical comparisons and matched cold/warm measurements
remain missing. Neither leaf changes an existing metric path or
qualifies a replacement-panel configuration.

The opt-in primary raw-RGB adapter landed at `328c3f6a`. It compiles the
unchanged upstream scoring body and retains first-call and warm records with
per-call ingress and object release inside timing. Three synthetic geometries
(8×8, 17×19, 64×64) at ingress budgets 1 and 2 match the official file CLI's
eight-decimal score display. This verifies the two canonical ingress paths;
it does not compare Rust to C++, bind the remaining static dependency source,
or qualify timing ratios. Refusal and source-preservation controls pass.

The separately identified owned-Highway build (`763820ae`) closes that static
dependency source/compiler gap for executable `9d0fc033…`. Its 232 Highway
source files and 32 installed headers match the pinned archive. The six
canonical-ingress comparisons still agree with the official CLI's display.
The runtime CMS branch, Rust comparisons and qualified ratios remain open;
earlier executable identities retain their original limitations.

## Primary MATLAB references executed through Octave

The [verified fixed-factor adapter](../iqa_compete_2026-10-16/octave-primary.md)
landed at `e2d414c3`. The primary FSIM and VSI releases now execute at factors
1–4 on the retained 64×64 control pair. Each case selects a separately verified
source copy, and first/warm scalar strings repeat. This establishes execution
beyond the earlier metadata-only staging. Remaining Rust parity, the requested
size/content/thread grid and qualified ratios remain missing. An explicit
headless environment resolves the observed startup stall; process durations
are not used as metric speed comparisons.

The [fixed-factor source audit](fixed-factor-semantics.md) records every named
variant's current placement, including the MS-SSIM per-level-view difference
that still requires an unmerged correction. A fresh Rust comparison harness
now builds from the unchanged CI-pinned source, with source-byte checks and
retained compiler/runtime receipts; compilation alone qualifies no score.

The first direct [partial scalar comparisons](parity-fsim-vsi-64.tsv) now pass
for all eight fixed-factor FSIM/VSI names at 64×64, one synthetic pair and one
thread. Both FSIM columns are checked. Maximum observed absolute differences
are 9.231485020855246e-9 (FSIM/FSIMc, gate 5e-5) and
8.306581955475068e-5 (VSI, gate 1e-4). No numerical bound changed. This covers
only those eight input/configuration cases; it does not complete either
method's requested grid or qualify a replacement-panel timing cell.

The larger [fixed-factor numerical panels](vsi-resize-runtime.md) now complete
all factors 1–4 × four requested sizes × three retained content pairs × thread
budgets 1 and 28. FSIM/FSIMc passes 96/96 cases, checking both scalar columns,
with maximum absolute difference 1.329927450832713e-6 against the unchanged
5e-5 gate. VSI passes 96/96 with explicitly bound upstream resize M-files,
maximum difference 3.1678478384922215e-7 against the unchanged 1e-4 gate.
The older image-2.12 VSI cohort remains retained: 82/96 pass, 14 fail.
All Rust thread-budget pairs are bit-identical. Raw artifacts and process
memory/resource receipts remain bound in the two result indexes; no maps or
qualified ratios are claimed. A separate [automatic-factor panel](primary-automatic.json)
passes 48/48 cases across unsuffixed FSIM/FSIMc and VSI, including original
author factor 8 at 4K. Its maxima are the same as the fixed-factor panels.

The owner-requested MS-SSIM correction is separately pushed and verified at
[`970ea199`](https://github.com/imazen/zenmetrics/tree/970ea199538b6efc64eb109e4db81de90137919d)
on unmerged `fix/ssim-pyiqa-fixed-factor`. Its ten focused tests pass.
Across the same twelve pairs/geometries and two thread budgets, the five
unsuffixed names produce 108 bit-identical baseline/candidate score cases and
12 matching small-input refusals. Fixed-name behavior changes remain separate
before/after observations; independent primary-reference comparison and
qualified speed ratios are still missing. The integration described below brings this correction onto master; historical fixed-factor observations retain their original implementation identity.

## Primary HaarPSI scalar grid

The [primary HaarPSI scalar panel](primary-haarpsi.md) passes 48/48 cases for
both RGB and unrounded-luma names across the requested sizes, retained pairs
and thread budgets 1 and 28. Maximum absolute difference is
5.942815561876813e-8 against the unchanged 5e-5 bound. The original author
M-file is unchanged and source-bound; both process RSS values and the aggregate
run-heavy receipt are retained. Map comparisons and qualified timing ratios
remain missing.

## Primary GMSD score and map grid

The [source-bound primary C panel](primary-gmsd.md) passes all 18 even-size
cases, with bit-identical maps, zero mean difference and maximum score relative
difference 1.4294167329816467e-13 against the existing 1e-12 bound.
All six original 255×257 C cases produce an instrumented buffer overflow;
Rust retains its documented floor-sized geometry. No input cropping or timing
qualification is inferred. Each process RSS and the aggregate resource
receipt are retained with the hashed raw artifacts.

## Primary MDSI scalar grid

The [unchanged-author MDSI panel](primary-mdsi.md) passes 24/24 scalar cases,
covering the four requested sizes, three retained pairs and both thread budgets.
Maximum relative difference is 7.364215176352893e-12 against the existing 1e-9
bound. Source, runtime, input hashes, both process memory receipts and the
aggregate resource receipt are retained. The author exposes no map output;
qualified speed ratios remain missing.

## Qualification contract

Each new observation must bind source and adapters, executable hashes, inputs,
thread budgets, loader environment, and the actual paired-round schedule.
Compare equivalent cold and warm scoring work with decode/file I/O outside
timing. Retain both process-memory evidence and resource-wrapper receipts.
Keep the metric's existing numerical gate unchanged.

Flag a ratio above 1.08 against canonical or a matched earlier same-host cell.
The ratio's 95% interval must be narrower than relative ±3%; an interval
straddling 1.08 remains borderline. Additional pooled samples do not repair an
invalid pairing design. Classical methods precede learned methods. PieAPP is
excluded from this study; AIMOS is excluded as proprietary and unreleased.

Evidence paths, hashes, exact test commands, and resource receipts are in
[the metadata record](review.meta). No timing ratio is supplied because none
has qualified in this replacement panel.

## Metric implementation integration, 2026-10-19

The fixed-factor MS-SSIM correction from `970ea199` is integrated with the
luma-only FSIM API from PR 92 (`c9f98a98`). Existing combined FSIM/FSIMc CLI
outputs remain combined; callers that need only luma can use `score_luma8`
without computing chroma. Ten SSIM tests and all 23 FSIM library tests pass,
including direct-statistics, fixed-factor, strided, and exact luma agreement
checks. FSIM all-target compilation, doctests (no cases), and strict all-target
clippy pass. The debug FSIM run was interrupted during an expensive existing
FFT test; the complete release suite subsequently passed without changing
assertions.

AVX512 dependency propagation is ported from `3494d8af`, preserving existing
threading features and excluding that commit's lockfile and analysis edits.
A lean CLI build also exposed a module gate that excluded the VMAF adapter
when only `cpu-vmaf` was enabled; that gate is corrected. Runtime dispatch
retains fallback tiers. Local checks ran on an AVX2 host and do not measure
AVX512 performance or establish a Rust/canonical speed ratio.

Reproduce with `just fixed-factor-check`, `just metric-dispatch-check`, and
`just vmaf-adapter-check`, under the normal host resource wrapper. The FSIM
release run completed in 16 seconds with 0.45 GiB wrapper peak RSS; the SSIM
run completed in 21 seconds with 0.99 GiB wrapper peak RSS. These are test
process resource receipts, not per-metric performance measurements.

The separate `fix/clean-panel-diagnostic-fixture` branch correctly retains
the tied median case (`p_order=0.89`) and exercises the original rejection
threshold using continuous values (`p_order=0.01`). Both values were reproduced
from its unchanged diagnostic implementation. Its full clean-panel framework
is not imported here: its production controller is disabled, and the fixture
repair supplies neither acquisition readiness nor a qualified ratio.


## Execution update: full SSIM grid and first paired timing table

Missing remains: most classical and all learned execution matrices, qualified
speed ratios, comparable maps for methods exposing them, and complete runtime
closure. The newly published [summary.tsv](summary.tsv) is a **25/96-case
partial HaarPSI timing prefix**, with zero qualified timing cells. It is now
available to the site generator and optimization queue. The earlier missing
summary statement above describes the preceding snapshot.

All four above-1.08 cells in this prefix have exact-configuration whole-process
CPU profiles, bound in [the artifact index](haarpsi-paired-prefix.json).
The 64×64 photo RGB 28-thread warm observation is 4.025143785 ms Rust versus
0.529015065 ms canonical, paired geometric ratio 7.609710678332473,
CI95 [7.4508351580649865, 7.771973930364894]. Its resource receipt is
`run-heavy: done rc=0 67s | peak-RSS 0.08GiB | min-avail 14206MiB | peak-load 1.34`.
This is exploratory Iterate-mode evidence. The profile includes kernel
scheduling, futex wakeups and Rayon worker activity; its whole-process sample
percentages are not exclusive metric-kernel costs. `pool_map` schedules 32
bands whenever more than one worker is available, including these tiny maps.
No optimization has been applied in this review.

The original timing cohort was retired incomplete during case 26 to redesign
per-invocation sample budgets. All raw completed and partial evidence remains.
A separate full 96-case plan now declares sampling and block counts before
execution; results will not pool the old and new cohorts. Its active local
path is recorded in the [prefix metadata](haarpsi-paired-prefix.meta).

The [full PyIQA SSIM study](primary-pyiqa-ssim.md) now includes all twenty
registered names: 480 observations, 390 scored comparisons and 90 matching
size refusals. The user authorized selecting external bounds on October 19;
the [separate policy-applied view](approved-pyiqa-ssim.json) uses 1e-6 absolute
for eighteen unrounded/direct/fixed routes and 5e-5 for the two original rounded
classes. All 390 scored candidate comparisons pass. The 90 matching size
refusals remain separate; the baseline has twelve numerical misses and six
refusal mismatches. Three caller-parameter
precision controls reproduce original primary values exactly and isolate
Gaussian/luma precision effects in those cases. Original primary values and
failed timeout attempts remain available. Existing Rust golden tests are unchanged.
The [full SSIM index](primary-pyiqa-ssim.json),
[precision controls](primary-pyiqa-ssim-precision.json), and
[command/source receipts](primary-pyiqa-ssim.meta) distinguish these cohorts.

The [Wang MS-SSIM grid](primary-msssim.md) passes all 24 scalar cases under
its unchanged 1e-4 absolute bound (maximum difference 4.752062978119476e-6).
At 64 squared the caller selects the documented adaptive three-level policy;
the original default five-level refusal remains logged. Internal map comparison
and paired timing remain open. This is separate from the Metrix alternative.

## Further native-map and CMS observations, 2026-10-19

Missing: an assigned external map bound, attribution of the HaarPSI nonfinite differences, and CMS runtime tracing across the remaining input matrix. The [HaarPSI map grid](haarpsi-maps.md) preserves 24 RGB map comparisons. The 4K illustration pair has 41 Rust-NaN/primary-finite positions at each thread count, retained separately from matching NaNs. These findings do not alter scalar passes or qualify map parity.

A separate [SSIMULACRA2 CMS call-stack observation](ssim2-cms-callers.md) attributes all 2,088 `cmsDoTransform` calls on the 64×64 photo control to initialization stacks containing `JxlCmsInit`. It preserves the earlier failed zero-transform hypothesis and the exact scalar result. It is untimed and does not establish runtime closure for the full matrix.

## 2026-10-20: current worker and HDR runtime controls

MISSING: the broad warm/first-call grids, default/resident HDR score controls,
revised base HDR native/reference timings and fully qualified regression
acceptance. This update does not turn earlier partial matrices into complete
coverage. The full current [progress record](../canonical_recheck_2026-10-19/current-campaign-progress.json)
remains separate from the representative README ratios.

A [fresh classical worker](../canonical_recheck_2026-10-19/current40-worker.json)
from `ea767ba1a4029963bc56a74fb824e64455e4599e` compiles every current changed
route, including both revised base HDR names and NR routes. Its independent
source verifier matches 1,887 tracked in-tree files; external dependency source
is outside that verifier’s claim. Every changed name returned a finite score on
the common 1152² photo scheduling workload. This is not canonical parity evidence.

The [HDR reference control receipt](../canonical_recheck_2026-10-19/hdrvdp-reference-controls/controls.json)
retains all old failures: installed Octave 6.4/image 2.12 fails 20/24 official
HDR-VDP-2 sRGB goldens. With unchanged image 2.20.1 resize helpers, all 24 pass
0.005 (largest difference 2.842170943040401e-14), on 14/14 byte-identical original
raw fixtures. The nine resident-parameter preflights pass. No official HDR-VDP-3
bound is inferred from these controls, and score/timing qualification remains pending.

`just bench-iqa-canonical-compare MANIFEST NEW_RESULTS 1,8` now exposes explicit
paired thread selection and manifest scheduling. Its retained 38-route manifest
freezes 76 cases successfully; the two revised base HDR routes remain pending.
Native changed-only selection already includes all 40 current routes in measured
local ascending order. CI was not run or consulted, as directed.

## 2026-10-20: site-readable warm observation export

New retained scalar finding: [SSIMULACRA2-GPU at 64²](../canonical_recheck_2026-10-19/ssim2-gpu-small-scalar-finding.json)
returns 84.8498333349087 against primary C++ 90.86388011956463, consistently
across all six blocks at threads 1/28. The GPU and primary select four and five
pyramid levels respectively on this input. Full numerical attribution remains
open; this is a correctness observation, not an accepted parity result or a
request for additional GPU maps. Metric code and bounds remain unchanged.
The fresh ordinary CPU control matches primary exactly (90.86388011956463),
with peak-RSS 0.10 GiB, minimum available 15,411 MiB and peak load 1.33.

The later [current40 comparative command receipt](../canonical_recheck_2026-10-19/local40-current40worker-command.json)
supersedes the 38-route preparation scope above: all 40 changed routes now have
bound reference plans, producing 80 cases at threads 1/8. Preparation passed in
7 s, peak-RSS 0.06 GiB, minimum available 15,441 MiB, peak load 2.41. Revised HDR
photo controls passed at both budgets: V2 delta 0.0007070822066168603 against
0.005; V3 delta 1.7763568394002505e-15 with no assigned bound. Their four paired
timing cells are acquiring separately. The manifest chooses one valid pair per
route and does not establish full-grid completion or qualified regression results.

MISSING: the full declared size/content grids, remaining first-call comparisons,
revised HDR paired timing qualification and accepted regression decisions.

The site can read [summary.tsv](local_canonical_2026-10-20/summary.tsv): one row
for each of the 126 CLI names, with 117 measured representative timing rows and
nine explicit statuses without comparable timing. AIMOS is recorded once as an
excluded method outside that CLI inventory. All 630 acquired warm observations
are retained in 16 [indexed tables](local_canonical_2026-10-20/index.json), with
canonical labels, source revisions, plan identities and contract exclusions.
These counts include repeated cohorts and historical sources; they do not mean
630 distinct inputs or accepted passes. First-call and map evidence stays separate.

Each representative row points through its `observation` number to one actual
cell. When the README reports a median across several content pairs without an
exact median observation, the table selects a deterministic measured case from
that geometry/source/cohort. Its times and ratio belong to that case; no per-case
time is derived from an aggregate ratio. Blank bounds remain unassigned, and
flags above 1.08 do not establish an accepted regression.

Reproduce this projection under shared heavy admission with
`just bench-iqa-canonical-summary SOURCE_DIRECTORY INVENTORY_RECEIPT NEW_DIRECTORY`.
The output refuses existing paths and changed evidence. The emitted input
manifests bind every input table and the exporter source. The publication wrapper
reports rc=0, 0s, peak-RSS 0.02GiB, minimum available 15322MiB, peak load 0.90.

## Local canonical followup, 2026-10-21

MISSING: complete multi-size warm/cold grids, remaining fixed-stage confidence
intervals and runtime qualification, and remaining map comparisons. Acquired
cold summaries are 186/1968 CPU and
6/124 CUDA; these counts include failures and
refusals. [The panels](../canonical_recheck_2026-10-19/acquired-cold-panels-2026-10-21/index.json)
retain every completed block and arm-memory receipt.

[BRISQUE64](../canonical_recheck_2026-10-19/brisque-64-numerical-finding.json)
exceeds its unchanged bound at both worker settings. [HDR-v2 modern-runtime
controls](../canonical_recheck_2026-10-19/hdr-octave111-v2-controls.json) reproduce
the 1 MP discrepancy, ruling out the legacy runtime as the explanation for that
case. [The v3 controls](../canonical_recheck_2026-10-19/hdr-octave111-v3-controls.json)
retain near-machine-precision scalar agreement in two cases without assigning a
universal bound. HDR ratios remain withheld, including the incomplete 4K warm
worker that reached its frozen chunk budget.

[Completed CPU profiles](../canonical_recheck_2026-10-19/cpu-first-profiles-2026-10-20/index.json)
and three CUDA profiles preserve recording and source hashes. The whole-process
recordings do not isolate cold-clock or causal shares. Capped captures stay
partial. [Repeat commands](../../docs/IQA_CANONICAL_COMMANDS.md) include the
41-route comparative just recipe, with explicit source and output paths.

Learnings: compiler/runtime controls and clock-lifetime controls are separate
from numerical acceptance; missing bounds and failed stages supply no timing
ratio. These followups change benchmark tooling, evidence and documentation, not
production metric paths or existing numerical expectations.

### BRISQUE runtime controls completed (2026-10-21)

[Eight original-only controls](../canonical_recheck_2026-10-19/brisque-runtime-controls-2026-10-21.json) use both installed torch 2.14.1 wheels
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

The GMSD 64² photo first-call [one-thread initial20 interval](../canonical_recheck_2026-10-19/gmsd-photo64-t1-first-confirmation20.json)
is 1.207 [1.174, 1.228]; its precision target is met. The [28-thread interval](../canonical_recheck_2026-10-19/gmsd-photo64-t28-first-confirmation20.json)
is 16.737 [16.130, 17.618] and misses the precision target. A fixed additional
40-block stage is queued; every original block remains retained. Numerical
checks pass, while runtime cleanliness and regression acceptance remain
unqualified. Final acquisition wrappers both recorded 1 s / 0.02 GiB peak RSS /
peak load 1.52; minimum available was 14,435 MiB at one thread and 14,426 MiB at28.

The FLIP v1.7 255×257/t28 [first-call initial20 interval](../canonical_recheck_2026-10-19/flip17-photo-odd-t28-first-confirmation20.json)
is 1.459 [1.369, 1.556], missing the precision target. The fixed additional
40-block stage retains the full initial stage; numerical acceptance remains
unassigned. Its final acquisition wrapper recorded 2 s / 0.22 GiB peak RSS /
minimum available 14,294 MiB / peak load 1.41.

### Backend and NIQE numerical follow-up (2026-10-21)

The [oneDNN controls](../canonical_recheck_2026-10-19/brisque-backend-controls-2026-10-21.json) narrow
BRISQUE's 64² photo miss. With oneDNN's effective ISA limited to AVX2, the
unchanged PIQ source returns 75.65032958984375, versus Rust 75.64959716796875
(difference 0.000732421875). The retained default original is 75.73614501953125.
ATen-only AVX2 and the separate MKL AVX2 request do not change that default
score; the latter request does not prove MKL's effective ISA. Disabling oneDNN
gives 75.47998046875. These counterfactual diagnostics do not replace the
default canonical failure or its 0.015138428 bound. The wrapper recorded 7 s,
0.33 GiB peak RSS, minimum available 13,998 MiB and peak load 2.28.

The [native stage controls](../canonical_recheck_2026-10-19/brisque-native-original-stages.json)
reproduce both retained native scalars exactly. Actual native and original
grayscale planes are bit-identical on both photos. The default original differs
in 20 of 36 statistical features at 64² and 26 of 36 at 255×257. The oneDNN
AVX2 control changes two 64² quantized shape parameters into exact agreement
with Rust. This narrows numerical attribution to statistical feature computation;
it does not clear the default-backend failure. The native diagnostic wrapper
recorded 1 s, 0.22 GiB peak RSS, minimum available 14,262 MiB and peak load 1.87.

The [NIQE odd-photo finding](../canonical_recheck_2026-10-19/niqe-odd-numerical-finding.json) retains
three failing blocks at both 1 and 28 workers: Rust 18.807175290717275 versus
BasicSR 18.799147370663427, difference 0.008027920053848447 above the unchanged
0.000602329 bound. This is separate from the documented 64² size refusal.
Root cause and a code correction remain unverified. Earlier golden checks and
all original observations remain retained.

### Remaining cold-flag profiles (2026-10-21)

The [four additional complete CPU profiles](../canonical_recheck_2026-10-19/cpu-first-profiles-supplemental-2026-10-21/index.json)
bring completed distinct CPU cases to 19. Together with the retained partial
GMSD captures and [5-of-30 FSIM capture](../canonical_recheck_2026-10-19/fsim64-t28-capped-adopt-v1.json),
these 21 CPU flags have profiling evidence. Partial captures
remain partial; no completed scorer is repeated for publication. The five
additional CUDA profiles bring distinct GPU flag profiles to eight. These are
whole fresh-worker diagnostics; their instrumented times and sampled shares
do not qualify comparative timings or a causal fraction of a gap. Each linked
projection retains its complete resource-wrapper receipt.

The [SSIMULACRA1 64²/t1 initial20 interval](../canonical_recheck_2026-10-19/ssim1-photo64-t1-first-confirmation20.json)
is 1.250 [1.241, 1.290], missing the unchanged precision target. Its fixed
additional 40-block stage is queued and retains every original observation.
No timing exception or regression acceptance is assigned.

### NIQE feature attribution (2026-10-21)

The [stage cross-rescore](../canonical_recheck_2026-10-19/niqe-native-original-stages.json) reproduces
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

### NIQE intermediate-plane control (2026-10-21)

The [plane comparison](../canonical_recheck_2026-10-19/niqe-native-original-planes.json) retains exact native/original scalar controls and all eight observed plane comparisons. The [two-shape counterfactual](../canonical_recheck_2026-10-19/niqe-shape-counterfactual.json) substitutes only the two observed second-scale shape parameters: its diagnostic score is 18.798994557675755, versus the unchanged canonical 18.799147370663427. This adjusted feature vector is not a replacement native result; the original numerical failure and bound remain unchanged.

| Scale | Plane | Differing f32 words | Max absolute difference |
| --- | --- | ---: | ---: |
| 1 | cur | 0/36864 | 0 |
| 1 | mu | 0/36864 | 0 |
| 1 | mu_sq | 0/36864 | 0 |
| 1 | norm | 0/36864 | 0 |
| 2 | cur | 3654/9216 | 4.57763671875e-05 |
| 2 | mu | 2043/9216 | 3.0517578125e-05 |
| 2 | mu_sq | 3715/9216 | 0.0078125 |
| 2 | norm | 5234/9216 | 0.0020902156829833984 |

### Further fixed20 first-call intervals (2026-10-21)

These complete initial stages retain every original block and revalidate their numerical contracts. Runtime qualification remains unassigned.

| Case | Ratio | 95% interval | Precision target |
| --- | ---: | --- | --- |
| fsim-grid-pairA-64x64-t28 | 1.095647 | [1.048398, 1.159399] | False |
| flip-v17-grid-pairB-3840x2160-t28 | 1.618464 | [1.541807, 1.644737] | False |
| ssimulacra1-grid-pairA-64x64-t28 | 1.236499 | [1.222225, 1.264232] | True |
| ssimulacra1-grid-pairA-255x257-t1 | 1.224473 | [1.195467, 1.254944] | True |

`fsim-grid-pairA-64x64-t28` final-block resource receipt: `run-heavy: done rc=0 1s | peak-RSS 0.05GiB | min-avail 14359MiB | peak-load 2.52`.

`flip-v17-grid-pairB-3840x2160-t28` final-block resource receipt: `run-heavy: done rc=0 7s | peak-RSS 1.40GiB | min-avail 13367MiB | peak-load 3.22`.

`ssimulacra1-grid-pairA-64x64-t28` final-block resource receipt: `run-heavy: done rc=0 1s | peak-RSS 0.02GiB | min-avail 14354MiB | peak-load 2.32`.

`ssimulacra1-grid-pairA-255x257-t1` final-block resource receipt: `run-heavy: done rc=0 1s | peak-RSS 0.02GiB | min-avail 14288MiB | peak-load 1.67`.

FSIM photo64/t28 and FLIP v1.7 screen4K/t28 have fixed additional 40 stages queued. FSIM remains borderline at 1.08. These descriptive cold intervals remain separate from warm results and accepted exceptions.

### Additional completed first-call profiles (2026-10-21)

Four [CPU captures](../canonical_recheck_2026-10-19/cpu-first-profiles-additional-2026-10-21/index.json) cover SSIMULACRA1 screen64 at one and 28 threads, FSIM screen64 at 28 threads, and FLIP v1.7 AIC odd at 28 threads. Two original attempts stopped at disk preflight before recording; their completed resumptions are identified separately.

Four [neural captures](../canonical_recheck_2026-10-19/neural-first-profiles-additional-2026-10-21/index.json) cover Paq2Piq at one and eight threads, MANIQA and HyperIQA at eight threads. Their source-line attribution remains partial because split debug references are unavailable.

The [Butteraugli CUDA photo1MP capture](../canonical_recheck_2026-10-19/butteraugli1mp-t1-first-profile-v1.json) supplies CPU stacks plus CUDA API and kernel traces. These profiles cover complete native processes; their instrumented durations and sample shares do not qualify a timing exception or isolate the first-evaluation clock. Fixed interleaved confirmations remain separate.

### VMAF Studio601 photo4K current-source score control (2026-10-21)

The [current-source controls](../canonical_recheck_2026-10-19/vmaf-studio4k-current-score-controls.json) compare the EA worker against every retained canonical sample for the same photo pixels and unchanged 0.02 absolute bound. Already-acquired current native arms are adopted where available; other native calls are accuracy-only probes. These are not new comparative timing results.

| Case | Current native | Canonical first sample | Maximum absolute difference | Existing bound passes |
| --- | ---: | ---: | ---: | --- |
| vmaf-studio601-grid-pairA-3840x2160-t1 | 92.691578992858453 | 92.719035484555903 | 0.027456491697449792 | False |
| vmaf-studio601-grid-pairA-3840x2160-t28 | 92.691578992858453 | 92.719035484555903 | 0.027456491697449792 | False |
| vmaf-neg-studio601-grid-pairA-3840x2160-t1 | 90.704283702489448 | 90.731845077129009 | 0.027561374639560654 | False |
| vmaf-neg-studio601-grid-pairA-3840x2160-t28 | 90.704283702489448 | 90.731845077129009 | 0.027561374639560654 | False |

The [linked-oracle identity](../canonical_recheck_2026-10-19/vmaf-linked-oracle-version-controls.json) verifies the source archive against the exact Cargo.lock checksum. The vmaf-head-sys 0.2.0 archive, Meson source and generated linked-build header all identify libvmaf 3.2.0. Existing 3.2.1 native descriptions are not evidence of a different effective FFI oracle. Older EFA observations remain retained and independently labeled; no bound or score was changed.

A separate [existing-score-contract assessment](../canonical_recheck_2026-10-19/vmaf-existing-score-contracts-2026-10-21/index.json) applies the already-present full-pixel FFI score assertion (absolute 0.02) to retained EA VMAF 4K/v1 samples. Their original boundless plans and summaries remain unchanged; this is an additional score-only audit, with no native calls or new tolerance.

vmaf-4k: 6/26 declared cohorts are acquired; 6 pass that existing score assertion; 0 cohorts retain process/input findings without a scalar pass.

vmaf-v1: 6/26 declared cohorts are acquired; 4 pass that existing score assertion; 2 cohorts retain process/input findings without a scalar pass.


### Next immutable cold publication cut (2026-10-21)

The [additional batch](../canonical_recheck_2026-10-19/acquired-cold-additions-2026-10-21-v2/index.json) retains 67 new acquired cases, bringing the published total to 316/2092 candidates. Every older record and all immutable plan/input/source bindings are revalidated. This is a partial acquisition cut, not full coverage or regression acceptance.

The [SSIMULACRA1 screen odd/t1 capture](../canonical_recheck_2026-10-19/cpu-first-profiles-screen-odd-2026-10-21/index.json) and [Butteraugli CUDA photo1MP/t28 capture](../canonical_recheck_2026-10-19/butteraugli1mp-t28-first-profile-v1.json) supply profiling evidence for the two newly observed cold flags. Fixed 20-block confirmations retain their original clocks and bounds; these full-process profiles do not qualify a timing exception.

The [IW-SSIM CUDA-reference/t1 initial20 interval](../canonical_recheck_2026-10-19/iwssim-cuda-reference-t1-first-confirmation20.json) is 1.485914 [1.360269, 1.560884], missing the unchanged precision target. Its fixed additional40 stage is queued; every initial block remains retained. Runtime/causal qualification and accepted exceptions remain unassigned.

### Original HDR numerical grid (2026-10-21)

The [complete acquisition index](../canonical_recheck_2026-10-19/hdr-original-grid-2026-10-21/index.json) retains 48/48 warm grid cells and 96 native/reference role outcomes across both HDR names, four geometries, three content pairs and thread budgets 1/28. This is acquisition coverage; the existing incomplete gates remain incomplete. The family [summary.tsv](../canonical_recheck_2026-10-19/hdr-original-grid-2026-10-21/summary.tsv) and its metadata retain site-readable rows with intentionally blank ratios.

hdrvdp: 24/24 complete observation pairs; maximum absolute raw-score difference 0.02003825875114984. Retained fixture checks: 12 pass and 12 miss.

hdrvdp3: 21/24 complete observation pairs; maximum absolute raw-score difference 7.815970093361102e-14. Retained fixture checks: 0 pass and 0 miss.

The v2 check is the unchanged fixture-derived 0.005 TOL_Q, not a universal parity bound. V3 has no assigned original-Octave bound for this grid. The reference is the frozen Octave runner with the named image-2.20.1 resize overlay; the separately retained official Octave 11.1 controls remain separate. Incomplete warm rounds and resource-gate findings are retained. Spectral preparation differs across clocks, so comparative ratios and native-parity/timing qualification remain withheld. Every row retains its source, raw outcome, measured arm maximum RSS and run-heavy receipt.

### SSIMULACRA1 screen odd/t28 first-call capture (2026-10-21)

The [capped profile prefix](../canonical_recheck_2026-10-19/ssim1-screen-odd-t28-capped-adopt-v1.json) retains 17/30 requested worker captures for the newly observed initial-three-block ratio 1.219720689786165. Its 32 MB size stop remains explicit. Postprocessing makes no scorer calls, and all original completed observations remain retained. A separate fixed20 confirmation retains the original input, source and clock. The instrumented capture covers complete fresh worker processes, including startup and RGB loading; it does not isolate the internal first-call clock, assign an external numerical bound or qualify a runtime exception.

The [photo odd/t28 initial20 interval](../canonical_recheck_2026-10-19/ssim1-photo-odd-t28-first-confirmation20.json) is 1.141342 [1.116921, 1.159368]. Its precision target is met and the interval is above 1.08; no additional40 is selected. An external numerical bound and runtime acceptance remain unassigned. Final-block resource receipt: `run-heavy: done rc=0 1s | peak-RSS 0.02GiB | min-avail 14035MiB | peak-load 1.17`.

Further initial20 stages retain all original blocks and numerical contracts:

| Case | Ratio | 95% interval | Precision target |
| --- | ---: | --- | --- |
| ssim2-grid-pairA-255x257-t1 | 1.439884 | [1.407924, 1.471711] | True |
| iwssim-gpu-cuda-reference-pair-cid22-t28 | 1.436002 | [1.395810, 1.480209] | False |
| butteraugli-gpu-grid-pairA-64x64-t1 | 433.818280 | [416.617869, 450.459627] | False |

IW-SSIM CUDA/t28 and Butteraugli CUDA/photo64/t1 have fixed additional40 stages queued after precision misses. The Butteraugli reference is the original CPU implementation. SSIMULACRA2 photo odd/t1 meets the precision target. These cold intervals remain separate from warm headline ratios and accepted runtime exceptions.
