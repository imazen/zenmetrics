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

Missing: the full inventory, the classical size/content/thread matrix, the
learned-method matrix, qualified ratio intervals, and the final site-readable
`summary.tsv`. The replacement timing panel has **0/19 qualified metric
configurations**. This denominator describes that panel, not the complete CLI
inventory. No canonical speed or parity conclusion follows from this note.

The 19 configurations are `brisque`, `butteraugli`, `butteraugli-original`,
`ssim2`, `vsi`, `vsi-k2`, `vsi-k3`, `vsi-k4`, `vsi-nodownscale`, `mdctpsnr`,
`cambi`, `cambi-aic2026`, `psnr-y-libvmaf`, `vmaf`, `vmaf-4k`, `vmaf-neg`,
`vmaf-neg-studio601`, `vmaf-studio601`, and `vmaf-v1`.

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
