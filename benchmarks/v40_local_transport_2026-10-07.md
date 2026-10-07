# V40 local transport

Zensim fit-source pin: `e4d98711dd140b155d776cfb027561cbb336f568`.
Transport pin source: `b7c2fb3f42562277f8f78c004f9861fd60dc5d55`.
Post-fit chain source: `061d9b4f2f8a40f8b8e75fa832fa5d0ce30c8fbd`.
Worker producer is current `master@origin` `e056defcc113e81e55207704dff910a7a0471d2d`,
build `c581bdb55f88`, including the reviewed claim-ownership changes.

Canonical local artifacts: `/mnt/v/output/zensim/v40-2026-10-07/`.
Program SHA: `c694bec138bc8ff76acdf627cf9711b7f1f6fd744c44a274fe3922c53fb2ff45`.
Image ID: `sha256:c90d075bc6acb03b482515ff0a9d5782f4b07c099d0f6b5649b6f91996c0158d`.
Neither source nor image nor inputs were pushed or queued by this lane.

The canonical contract verifier admits exactly control, hb4, hc4 and palette.
The uh4 notice describes an excluded arm; it is absent from transport variants.
The three reviewed manifests contain 40, 80 and 40 full-budget cells. All four
final bounded image blobs pass receipt/input/checkpoint verification and refuse
installation as local smoke. Original transport tests plus the V40 cases pass.

`harvest_driver_v40.py` is an independent driver using the pinned committed
harvester, program archive and checkpoint inspector. It checks the registered
manifest and its exact install root, binds full-budget admission metadata and
never provides a local-smoke override. `v40_postfit.sh` validates local pins,
then emits heartbeat/status, incrementally harvests verified DONE cells, freezes
the completed fresh control, or invokes the released four-source SDR scorer.
It never schedules a job. HDR/external/UPIQ reports remain separate explicit
exposure-gated commands. The existing E28/E30 scripts and live queue/caps were
not edited.

Exact future commands and approval-file names are in the artifact's
`POSTFIT_COMMANDS.md`; templates remain inert. The image's entire embedded
program inventory, baked worker SHA and Python library versions match
`IMAGE_INVENTORY.json`. The archive mirror and three randomly selected new
Parquet files match `MIRROR_CHECK.json`.
