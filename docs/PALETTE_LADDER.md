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
hashes with each execution program. The direct dependency and zenpng share one resolved quantizer version; the
local dispatch patch and its review status are recorded below.

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

The registered source gate also requires the transitive native-statistics and
assessment-identity Python helpers, the Rev5 bank schema owner and C's frozen
product declaration. A self-consistent source inventory that omits one of
these required owners refuses packaging. Additional dependencies of the final
program remain part of its independently reviewed source closure.

## Reviewed dispatch integration

The executor pins the merged [zenquant PR14](https://github.com/imazen/zenquant/pull/14)
revision `8ec37ee21f0216cf9b138e407eecc470ee620493` through a root Cargo patch.
The direct requirement is 0.1.4, with one resolved quantizer shared by zenpng
and the executor. Release publishing remains a separate owner decision.

`effective_dispatch` now records the native report, including Fast/Adaptive
promotion, target-tier quality/priority/strength, format defaults, exact-palette
bypass, refinement and joint use/availability. Quality-floor outcomes retain
the partial observation. The requested-config Debug field stays separately
labelled; it is not used as an effective report.

The PNG adapter's `QuantizeOutput` is non-exhaustive and has no external
constructor. The local bridge therefore performs the reported native call and
the existing adapter call, and requires identical palette, indices and metric
bits before encoding. Successful receipts explicitly record `quantize_passes=2`,
`dispatch_output_equal=true`, both stage durations and their total. A quality
floor returns before the adapter pass and records one call. This overhead is
part of later measured preparation cost; there is no throughput claim or
implicit compute deduplication.

`just palette-dispatch-receipts <new-output-dir>` asserts and exports generated
Fast/Adaptive, target-tier/joint and quality-floor receipts in both build
variants. All ordinary palette and independent decoder tests still run through
`just palette-check`. These tests do not invoke job execution or a fleet.
