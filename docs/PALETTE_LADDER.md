# Indexed palette ladder

`scripts/jobsys/palette_plan.py` declares metadata-only quantization requests.
It requires 320 distinct source families, 50 fit and 14 development sources
in each of five classes. Geometry uses nearest rounding with ties upward,
without upscaling. It retains every requested profile and PNG effort.

Use the canonical Resample executor for Mitchell B=C=1/3 references. Bind
its output receipt before `encode_items` emits EncodeDeclareItem rows for
`zenfleet-ctl declare-encodes`. The executor accepts `palette_ladder` knobs
only for opaque SDR zenpng inputs. `palette-joint` enables the separate
joint-present binary; the absent build must have no transitive zenquant/joint
activation. Freeze cargo feature trees, compiler, lock, source and binary
hashes with each execution program. The direct dependency pins the same
zenquant 0.1.3 used by zenpng; there is one resolved quantizer version.

The ordinary Encode output remains PNG bytes for canonical blob addressing.
A corresponding Metric job named `palette-receipt`, carrying identical knobs,
captures palette/indices, pixel and encoded hashes, parameters and stage times.
It reruns the deterministic encode; its encoded hash must equal the Encode
blob. Declare receipt jobs first for diagnostic minimum-quality profiles:
`quality-floor` is a declared outcome, while every other error fails the cell.
Only successful configurations proceed to all 33 Encode writes. No truecolor
fallback is permitted. Repeated encodes currently repeat quantization; no
compute dedup or throughput claim is made.

Score persisted Encode blobs with the existing Feature, ScoreFile and Diffmap
paths. Keep every metric variant and map and bind their hashes into the final
capture. Training captures must have an independent decode proof and full
feature-era identity. Byte/config aliases remain in provenance while training
mass is deduplicated by reference pixels, decoded pixels and feature era.
No launch or measured budget is implied by a generated manifest.

`just palette-check` exercises invented source metadata and indexed encoding
under both build variants, including the independent PNG decoder. Run it with
the shared heavy-command lock and resource wrapper.

`scripts/jobsys/budget_preflight.py` writes an exclusive metadata-only manifest:
native-size omissions, the generation union, twenty full-budget mock fit routes
per admitted host, supporting stages and explicit watchdog reservations. Host
identity is included in each task namespace. Missing caps or budget produce
UNREADY. Measurements, production quotas, placement admission and execution
receipts remain separate launch prerequisites; the generator never launches.

Registered source-family cells use the existing FitCell executor, archive,
ledger, grid coverage and decoded-checkpoint qualification owner. The
`registered` program profile accepts separately frozen public source pins.
Its image metadata carries only opaque data/contract hashes. Private cell
contracts stay in the data archive; an authorized harvester supplies the same
contract through `ZEN_REGISTERED_FIT_CONTRACT`. Missing watchdogs, changed
contracts and incomplete decoded checkpoint admission refuse installation.
