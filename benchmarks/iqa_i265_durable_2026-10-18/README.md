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
