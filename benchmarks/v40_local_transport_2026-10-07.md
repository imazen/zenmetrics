# V40 local transport

Zensim fit-source pin: `e88f9180537a9755cfa7800fa68185166213a5ae`.
Transport pin source: `3daf9b33d5a32d19e236222f675e4072373fb387`.
Post-fit chain source: `69dd8fcb084aab10171db959c380392f0eabe582`.
Worker producer is current `master@origin` `e056defcc113e81e55207704dff910a7a0471d2d`,
build `c581bdb55f88`, including the reviewed claim-ownership changes.

Canonical local artifacts: `/mnt/v/output/zensim/v40-2026-10-07/`.
Program SHA: `69e23c9a04495c27c6c0d7e2085bfc78d7a39d43eab431b8e168792bd34abc0d`.
Image ID: `sha256:a05e8f31c19d6a995fe26e389f96f7b9a883eefe3b74f57050c47f2b80a7cabe`.
Neither source nor image nor inputs were pushed or queued by this lane.

The canonical contract verifier admits exactly control, hb4, hc4, palette and
uh4. The four manifests contain 40, 80, 40 and 40 full-budget cells. E31's
owner disposition cites zensim DATA_SPLITS ddf375af and binds the immutable
TRAIN-fit manifest and legacy label hash; native feature qualification remains
false. The verifier checks its exact decision argv, pinned record, native fit
projection and input roles/weights. All five final bounded image blobs pass
receipt/input/checkpoint verification and refuse installation as local smoke.
All ten real executor smokes pass; registered-budget probes stop after epoch 000.
Original transport tests and the new E31 positive/mutation cases pass.

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
