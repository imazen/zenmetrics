# E29 local fit profile — 2026-10-07

Missing: coordinator registration of the four-source amendment/exposure
addendum and one matched control; full E29 fits/assessment; registry/R2
publication and fleet enrollment. Nothing from this preparation was pushed.

`pack_fit_program.py --profile v2e29` pins the strict D1 owners plus the E29
TRAIN adapter, fit contract, amendment and guarded HDR panel. It transports
three trainer/predictor/panel binaries and the canonical research inspector.
The prepared data root is staged locally as well as packed for the executor.
Source owner: `zensim--e29/benchmarks/E29_WORKLOG.md`.

Current program SHA256:
`44ca24737293b404e29e150ebce78632cc21d5de7b6416c3058fb53c2100ef58`.
Data SHA256:
`9c3eff1b740d2b7a77a235a16a3cd3521746af55a72bb92cddd0536674963008`.
Local image: `ghcr.io/imazen/zenfleet-worker:fit-e29-consensus-v41-w925f9783329f`.
Artifact root `/mnt/v/output/zensim/e29-2026-10-07/` and tower mirror
`/mnt/tower/output/zensim/e29-2026-10-07/` retain exact source/build/binary,
manifest, executor, resource, import-tripwire and exposure receipts.

80 arm and 40 proposed control jobs come from the actual `zenfleet-ctl
 declare-fits` owner. Registered argv stays 120 epochs × 50,000 draws and final
119; separately declared short jobs cannot install as scientific cells.
Local image tests enter `fit-cell-exec`, extract/hash the archive and bind
FIT_ROOT to the hash-named scratch root. All three short routes pass trusted
harvest; both HDR full argv routes reach epoch zero then are deliberately
stopped by the test driver. SDR stays strictly admitted; native HDR has
explicit subset provenance and false qualification, never a full-family waiver.

`declare-fits` emits its unchanged 2-GiB/four-thread packing hint. Local smokes
are actually capped at one CPU/six GiB/no swap, v3/one Rayon thread. This is
recorded separately in RESOURCE_PROFILE.json; the jobset_caps.json proposal
uses the existing jobset → memory/hosts shape and enrolls no hosts.

One import test accidentally ran the old unguarded HDR CLI and opened the
legacy 22,860-row/952-column mc944 HDR VAL default. It computed target swings,
then failed before any student prediction. No fit used those rows. The CLI is
now guarded and zero-open import/absent-control tests pass. The read remains
an explicit preparation exception, disclosed in UNINTENDED_EXPOSURE.json.
The earlier v40 image/program/smokes are preserved as superseded evidence.
The registered 3,900-row hdr_v3mix VAL, AIC and confirmation payloads remain
unopened. All 40 E30 nA3 completed checkpoints were verified and pinned through
their original trusted program, without label reads; exact E29 program reuse
parity is not asserted.
