# Scalar-only priority scheduling — 2026-10-18

Owner direction replaces upcoming map scopes with scalar work only. Existing
maps, measurements and all attempts remain under their original frozen manifests.
The new queue runs the host's clean fastest-ten baseline, all six dvifmish
routes, and explicitly requested quality families before remaining scalar work,
then resumes NN CPU followed by NN GPU. This changes scheduling, not kernels,
rounds, acceptance thresholds or source input bytes.

`benchmarks/iqa_compete_2026-10-16/scalar_priority.py` acquires the same shared
host lock at a worker boundary. It refuses an owned live watchdog, disables
and stops the old services, terminates only exact output-bound idle controllers,
and proves their coordinator locks free. Retained rows are read through their
original frozen controller's validator with identity/config/intent/log hashes.
No old report is rewritten. Completed validated scalar cells are excluded;
flagged warm rows are retained as evidence but not qualified reuse. Unsupported
and two-attempt-exhausted cells retain their terminal classification. A prior
failed attempt leaves one remaining attempt, represented by a separate frozen
phase capped at one; new cells keep the original two-attempt ceiling.

Each replacement bundle has only `only_mode=score` cells and a hashed lineage
inventory binding original configs and attempt receipts. Cold repeats 0/1/2,
one/eight workers, and ten 100ms-target warm rounds stay unchanged. Input hashes
must match when using the corrected Mac worker. The service freezes and verifies
its own scheduler files and plan hash; every future phase records its preceding
new output in `priority-wait.json`. NN output identities and attempt history
are resumed under their original policy, with the new scalar phases as parents.
The disabled old predecessor configurations remain historical files.

Commands are exposed as `just iqa-scalar-priority <action> <config>` and
`just iqa-scalar-priority-tests`. Preparation/rollout uses the shared host lock
and native 12G/8-job resource wrapper. Frozen worker watchdogs, dead-reservation
recovery and timing clocks are unchanged. `status <sequence.json>` reports
actual native service state, phase outputs and current wait/progress records.

## Mac ranking qualification

The provisional ranking file SHA256 was
`56f396ae5f88f32c40f02019c8a3499b9f312c679334850740dc9af75bc3efb1`.
Its generic PSNR and SSIM warm rows were contention-flagged, so they did not
qualify selection. A new protected 512²/eight-worker scalar diagnostic used
corrected worker `c4d49bd53dbdef9d88dc664b547af49221c18e0e6b337d5a84d0bacf71a3bd19`,
the same shared lock, a private nested attempt lock, and five seconds of settle
outside clocks. Both routes passed all ten rounds cleanly: PSNR
0.46650681806274225 ms and SSIM 1.6235048592537313 ms. Original flagged rows
remain preserved. These controls supplement the ranking, not production cells.

The final clean warm scalar ranking (512², eight requested workers) is:

| rank | route | median ms |
| --- | --- | ---: |
| 1 | dvifmish-port-min-v0.2 | 0.40666233352816417 |
| 2 | psnr | 0.46650681806274225 |
| 3 | gmsd | 0.4963917322042219 |
| 4 | psnr-y-studio601 | 0.8109804302905004 |
| 5 | dvifmish-port-lap-v0.2 | 0.8410885287388051 |
| 6 | psnr-y | 0.937523458392435 |
| 7 | psnr-y601 | 0.9375892565124934 |
| 8 | psnr-cb | 1.4751313825117374 |
| 9 | psnr-cr | 1.4921237424151097 |
| 10 | ssim-pyiqa-f32-k6 | 1.592359171158655 |

Final ranking SHA256:
`7169ed839d03718caf42e888e8db12f5c15151a3152b3be2793dccbe95a95926`.
Control/ranking/boundary artifacts:
`/home/lilith/benchmark-artifacts/iqa-scalar-priority-mac-20261018/ranking-v1/`
on the coordinator and `~/tmp/iqa-scalar-priority.noindex/ranking-v1/` on mac.
PSNR enters the ten; SSIM stays outside. No flagged median is used.

## Additional requested families and quality evidence

The frozen CPU workload exposes HaarPSI (`haarpsi`, `haarpsi-y`), SSIMULACRA2
(`ssim2`, distinct from SSIM/MS-SSIM), DSSIM (`dssim`), `butteraugli`,
`butteraugli-original`, `cvvdp`, and six VMAF variants: `vmaf`, `vmaf-neg`,
`vmaf-4k`, `vmaf-v1`, `vmaf-studio601`, `vmaf-neg-studio601`. These are additional
requested routes, not deducted from the ten. All six dvifmish routes remain
independently requested. GPU variants are not inferred from CPU measurements.

Butteraugli's generic route declares max and pnorm3 columns, but this frozen
benchmark records the first headline (max). It does not provide an independent
pnorm3 timing or quality observation. The original variant remains separate.

Quality evidence comes from
[`aic2026_agreement.json`](../benchmarks/iqa_complexity_2026-10-17/analysis/aic2026_agreement.json)
and [`aic2026_columns.json`](../scripts/complexity/aic2026_columns.json), anchored
at `ee64f6226ed7f30b6a104c3b5f013982f6078b24`. This is preliminary public-split
subjective JND reconstruction: 5768 images, 42 sources, mean within-source
sign-aligned Spearman, 2000 source-bootstrap samples, seed 20261017. Published
implementation columns provide quality evidence, not the raw Mac scores.
The coordinator's measured join and bootstrap intervals are retained under
`/home/lilith/benchmark-artifacts/iqa-quality-research-20261018/mac-quality-performance.{json,csv,md}`.

Mappings exist for HaarPSI, SSIMULACRA2, DSSIM, CVVDP, VMAF and VMAF-neg.
The published Butteraugli proposal is a pnorm3-family proxy, not the max score
being timed here; its Mac max quality field is therefore null. No comparable
formal quality mapping is asserted for dvifmish, butteraugli-original or the
remaining VMAF variants. These are requested-family additions, not a formal
statistical Pareto-dominance claim. The historical i265 Pareto front is not a
Mac result and is not reused.
