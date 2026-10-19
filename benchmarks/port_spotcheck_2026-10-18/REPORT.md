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
