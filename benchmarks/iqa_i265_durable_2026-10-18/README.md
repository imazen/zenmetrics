# i265 durable IQA launch, 2026-10-18

The [frozen launch receipt](launch.json) records the actual build, five
control runs, resource logs and first-stage progress at the observation time.
It is a launch snapshot, not a completed benchmark or perceptual qualification.
The [Linux runner contract](../../docs/IQA_DURABLE_LINUX.md) gives reproduction
and status commands.

On i265, the active plan is
`/home/lilith/benchmark-bundles/iqa-i265-20261018-coverage/sequence.json`.
Its six stages schedule 12,096 cells: all 85 compiled non-NN CPU methods,
then 23 NN CPU and 23 NN CUDA methods at 512×512; the same families follow
at ascending larger sizes. Budgets are explicitly one/eight workers on the
verified P-core set 0–7. PieAPP remains an explicit opt-in; NN maps have no
qualified adapter. Unsupported classical maps retain refusal rows.

The worker SHA-256 is
`2256ada792205730f5229e648ee2f8ae80558c59fcbebcdbdaa01d2a6002b2d2`.
The build starts from `71c18de8` and applies the receipt-bound, benchmark-only
ORT session-budget patch; the original engine and adapted engine hashes are
in the receipt. No model or metric arithmetic was changed.

PSNR/GMSD controls retain 16 terminal cells, including unsupported PSNR maps.
LPIPS-Alex has separate RTEN/CUDA cold/warm controls at both budgets. Together
they retain 24 checked placement proofs and 70 warm rounds with clean zenbench
admission gates. Cold/warm score bits match within each route. CUDA logs emit
the requested intra-op budget and no affinity errors. These controls do not
qualify every model, graph operation or input size.

Earlier CUDA controls using worker `8a8a7065` passed score identity but exposed
unconfigured ORT session pools. They remain diagnostic and are not promoted.
Corrected attempts use fresh outputs; all original bytes remain preserved.

The active user service is `iqa-durable-41cfddaddf8a.service`, started at
10:56:13 UTC. Results live under
`/home/lilith/benchmark-results/iqa-i265-20261018-coverage/` on i265.
The compact `progress.json` in each stage is authoritative for current
progress. The receipt observed the first stage incomplete and all five
successor output directories absent, confirming the current dependency wait.
Every stage uses the same `/home/lilith/tmp/i265-heavy.lock`; attempts have
1100-second watchdogs and at most two tries.

Build/preparation/control logs and the external receipt live under
`/home/lilith/tmp/iqa-i265-20261018/` on i265. Immutable bundles, inputs, model
exports and supplied CUDA/ORT libraries are hashed in each bundle's manifest
and `SHA256SUMS.json`. Full result reports and attempts remain outside git.
No remote mirror has been made by this launch.

```sh
just bench-iqa-linux /home/lilith/benchmark-bundles/iqa-i265-20261018-coverage/sequence.json status
```

## Input-contract coverage

After the launch snapshot, i265 reached `hdrvdp` and `hdrvdp3`: their 16
first scalar attempts refused the frozen sRGB8 input. These routes require
absolute luminance and explicit viewing conditions; the benchmark does not
invent them. The failed attempts remain pending bounded retry, independently
of the compact report's exhausted-cell count. These are contract refusals,
not stalled workers.

The same 512 schedule also contains `ms-ssim-pyiqa-f32-k4/k5/k6`. The
fixed-factor and five-level geometry checks in
[`ssim_pyiqa.rs`](../../crates/zenmetrics-cli/src/metrics/ssim_pyiqa.rs)
reject their final levels below the valid 11×11 window. Their i265 attempts
had not yet executed at the 11:01:35 UTC observation; no success is claimed.
The existing kernel, factors, downsampling and limits remain unchanged.
Each of these three methods retains all 96 larger-size cells, including
scalar and map eligibility observations, in the frozen larger-family bundle.
Larger cells remain scheduled; eligibility is decided at the actual size.

At 11:01:35 UTC the first stage had 335 successful and 76 unsupported cells,
16 failed first attempts, zero exhausted cells and 949 pending cells. The
pending total includes the failed attempts awaiting retry. The external
`/home/lilith/tmp/iqa-i265-20261018/input-contract-status.json` on i265
retains that observation and the exact returned HDR errors.

The subsequent i265 pass reached all three fixed-factor MS-SSIM methods and
reproduced eight failed first attempts each. Thus all 40 first-attempt
refusals match the five input-contract cohorts above. No frozen attempt,
input, limit or larger-size schedule was changed.

An audit of the first 94 successful warm protocol rows found 14 rounds in
seven rows flagged by zenbench as `HeavyProcesses(2)`. Preserve those flags:
these rows do not establish uncontended performance. All 70 warm rounds in
the separate prelaunch controls were clean. The external
`/home/lilith/tmp/iqa-i265-20261018/contention-observations.json` on i265
retains every affected attempt and original gate observation.
