# What remains after zenpsy-api — 2026-11-09

The minimal still-image core and its erased wrappers cover typed native
assessments, named scalar views, identity, checked packed pixel inputs,
cancellation/progress, and optional real reference preparation. Heterogeneous
collections no longer require zenmetrics-abstractions. Stacking `Erased` keeps
the original native result available through `native::<T>()`.

This is not yet a production migration: **0 of the 22 direct normal-dependency
packages on zenmetrics-abstractions are migrated by these reviews.** The new
orchestrator path is opt-in, and its native adapters live in standalone
validation harnesses. The old crate remains needed by existing ports and
frontends even for capabilities the new core can express.

This source audit uses zenmetrics `785be25d` and the shared wrapper implementation
[2ee9fd21](https://github.com/imazen/zenpixels/commit/2ee9fd219fad403d09ef5f595f5504af8add3da1).
The legacy sources below are unchanged by the wrapper integration. It is an API
and caller audit, not new numerical or performance qualification. Review links:
[core](https://github.com/imazen/zenpixels/pull/111),
[codec bridge](https://github.com/imazen/zencodec/pull/196),
[native integration](https://github.com/imazen/zenmetrics/pull/113).

The subsequent [maps/NR review](https://github.com/imazen/zenpixels/pull/131)
has landed on zenpixels `d234f3c6`. Its common map views and one-image/paired
contracts address those two core API gaps; production consumers in this
inventory still use the legacy interfaces. The [follow-up report](IQA_MAPS_NR_AND_JXL_LOOP.md)
records native bridges, workbench iteration scoring and remaining zero-copy
limits. The table below preserves the consumer inventory while updating the
required migration work.

## Contracts with current consumers

| Remaining surface | Actual caller or implementation | Placement and migration gate |
| --- | --- | --- |
| Temporal sessions: push/finish, delayed frame outputs, frame indices/rate, clip pooling and failure state | [VMAF](../crates/vmaf/src/metric.rs) declares stateful history/latency and returns delayed frames; [CVVDP](../crates/cvvdp/src/pipeline.rs) pools a clip into JOD and requires a positive frame rate. | Keep a video extension or legacy compatibility layer. Compose still metrics over frames, but retain a distinct native temporal implementation for these two metrics. Preserve delayed output ordering, final drain and failure behavior. VMAF's current still `evaluate` itself drives a one-frame sequence, so removing the sequence trait would also require refactoring that adapter. |
| Native planar image input and significant-bit/range metadata | VMAF's `accepts` and `input_from` consume native YCbCr 4:2:0 U8/U16 planes. The [browser](../crates/zenmetrics-browser/src/lib.rs) builds planar input for its video metric routes. | Put shared plane views in zenpixels, then migrate ingress with exact native score/stride tests. The minimal core's packed `PixelSlice` does not replace the current `Image::Planar`; converting to RGB would change the input contract. No codec runtime is needed for an already-decoded plane view. |
| One-image/no-reference scoring and optional paired reference/distorted views | [CLI classical NR](../crates/zenmetrics-cli/src/metrics/nr.rs) wraps PIQE, NIQE and BRISQUE in `PairedNoReference`; [CLI NN](../crates/zenmetrics-cli/src/metrics/nn.rs) similarly wraps ARNIQA, MANIQA, LIQE, MUSIQ, HyperIQA, PaQ-2-PiQ and WaDIQaM NR variants. | The merged `NoReferenceMetric` and `PairedNoReference` now provide these contracts. Migrate the production NR implementations and CLI callers; the PIQE bridge currently validates the new API only. Preserve both named results where the CLI exposes reference and distorted scores. |
| Generic diagnostic map iteration, native grid geometry, support and polarity | Browser `evaluation_maps` consumes `&dyn Evaluation` and visits every published `MapView` without knowing the concrete metric result. Coarse, cropped and padded maps need their actual geometry. | The merged `Evaluation::maps` and borrowed `MapView` now provide generic native planes/placement. Migrate the browser and its legacy map adapters; keep display/resampling and richer native support metadata outside the scoring core. Diagnostics remain off by default; the four-method validation retains native diagnostics only on explicitly requested small witnesses. |
| Bound-preserving worker/model storage | CLI NN and PyIQA model caches store `Mutex<HashMap<_, Box<dyn Metric>>>` in `OnceLock`. Benchmark callbacks also use Send trait objects. | Preserve Send where a concrete caller needs it. The local core intentionally adds no global Send/Sync requirement. An erased scorer can carry `+ Send` when its native scorer does, but `ErasedEvaluation` and dynamically prepared objects do not promise Send. Prepare/consume on the executing worker and transfer owned scalar views, or introduce a separate bound-preserving wrapper when a real migration requires transport of native results. |
| Existing identity/column compatibility and conformance controls | The old [identity API](../crates/zenmetrics-abstractions/src/identity.rs) exposes canonical text, fingerprints and qualified columns; metric tests pin those outputs. Many ports call the [conformance harness](../crates/zenmetrics-abstractions/src/conformance.rs). | Keep compatibility serialization and validation during migration. Do not silently rename stored identities or drop regression coverage. Hashing, column naming and conformance machinery can live outside the minimal scoring core. The CLI also has its own `MetricKind` column mapping; this audit does not claim every writer uses `MetricId::column`. |

Temporal sessions do not inherently require zencodec. They need decoded frames,
ordering and timing. zencodec is relevant when mapping decoder output,
compositing/presentation state or an edited media timeline into that input.
Planar pixel types belong in zenpixels; metric-specific temporal pooling belongs
with the metric/session extension.

## What can become smaller

The old universal boxed `Metric`/`Evaluation`/`WarmReference` paths can migrate
to the typed core plus optional `Erased` wrappers. Native scalar outputs stay
structured; no single-number return or universal metric enum is needed. The
new orchestrator already uses those shared wrappers rather than maintaining
another private result-erasure layer.

Other legacy conveniences are not requirements on every method:

- `ColdReplay` and its owned reference copy are an explicit cold fallback, not
  genuine preparation. Keep such a fallback at the caller if needed; do not
  make it the core `PrepareReference` default.
- A declared output schema, universal score-kind taxonomy, viewing-condition
  accessor and parameter reflection can remain compatibility or optional
  inspection tools. Effective configuration still belongs in identity/native
  configuration. CVVDP and HDR-VDP implement the old viewing query, but that
  does not make one universal display model a still-image scoring requirement.
- Map folds, resampling and display helpers can remain diagnostic utilities.
  Their presence in metric implementations/tests is not evidence that every
  production scoring call needs them. HDR-VDP's custom folds are also exposed
  as named scalar outputs.
- Stateful sequence errors and stateless frame composition belong in the video
  extension. They should not enlarge the still-only error or metric hierarchy.

The old graph uses zenpixels 0.2.16 while the reviewed core uses 0.3.1. Both
versions are intentionally present in the integration lock. A blanket re-export
cannot bridge distinct pixel types or resolve planar ingress; migrate adapters
with their native input semantics intact.

## Dependency inventory and removal order

Reproduce the direct count by parsing the tracked `Cargo.toml` files and finding
`dependencies.zenmetrics-abstractions` (including renamed dependencies whose
`package` field names it). There are 17 individual metric packages:
brisque, cvvdp, fsim, gmsd, haarpsi, hdrvdp, iwssim, mad-iqa, mdctpsnr,
msssim, niqe, nlpd, piqe, psnrhvs, vif, vmaf and vsi. The other five are
zenmetrics-nn, zenmetrics-pyiqa-ports, zenmetrics-browser, zenmetrics-cli and the
cpu-profile benchmark package. These are direct packages, not a count of
individual methods inside the NN bundles.

Migrate packed full-reference scalar adapters first, comparing every named
native scalar and actual prepared path. Use the new no-reference and diagnostic
contracts when moving their current callers. Migrate shared planar views and
the two temporal implementations with their own native controls. Keep the old
facade until all 22 packages and their conformance/identity callers have moved;
then remove redundant compatibility types.

Retaining the legacy sequence/conformance surface does not qualify its known
gaps. The default `StatelessSequence` still stores but does not enforce its
`SequenceInfo`, and does not poison itself after a failed push. The scalar
conformance check rejects undeclared outputs but does not require every
declared secondary output to be returned. Those existing findings remain in
the [integration review](ZENCODEC_IQA_REVIEW.md) and
[PR review](PR_REVIEW_2026-11-04.md). New temporal/conformance migrations need
those contracts tested explicitly; wrapper tests do not fix them.

Current wrapper/native validation and its limits are recorded in the
[contract validation report](IQA_STILL_CONTRACT_VALIDATION.md).
