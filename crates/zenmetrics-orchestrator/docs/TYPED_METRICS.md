# Typed metric execution

Enable `psy` for `zenmetrics_orchestrator::psy::ContractExecutor`. It has no
GPU-umbrella dependency. Register factories with `register` for cold scoring or
`register_prepared` for native `PrepareReference` implementations. Factories
and scorers execute on the calling worker; the caller owns placement and
resource admission. Thread-local native results are supported.

`run` accepts one borrowed, colour-described `PixelSlice` reference and a slice
of candidates. All strides, formats and contexts pass to the native adapter.
For a prepared registration, actual preparation runs once and serves the batch.
The executor uses the core's `Erased` and `DynPrepareReference` wrappers. It
stores a common owned score view and retains the entire native
assessment, recoverable with `assessment.native::<ConcreteOutput>()`.

`Fallback::Refuse` is the default. `Fallback::Resource` explicitly authorizes
trying the next registered implementation only after a classified resource
failure. Completed candidates survive and are not replayed. Fallback cannot
change the effective family/variant/reference/configuration. Every result keeps
its actual implementation identity; this policy does not assert CPU/GPU
bitwise equivalence. Constructor and preparation errors retain their original
located cause in the attempt history. Cancellation and computation failures
stop the batch. An incomplete report has a completed prefix and a final error,
never invented values for pending candidates.

Routine batches refuse diagnostic requests before constructing a scorer.
Use the typed native API for explicitly bounded diagnostic witnesses and a
separate storage budget.

This additive path does not replace the existing GPU chooser, worker/session
pools, live VRAM admission, strip execution or benchmark cache. Those policies
remain orchestration concerns rather than obligations on every metric. The
current native integration tests cover prepared Butteraugli, SSIMULACRA2 and
serving Zensim, plus cold selected Margarine. They do not qualify GPU arithmetic
or measure scheduling performance.

Validation and exact source pins: [native contract report](../../../docs/IQA_STILL_CONTRACT_VALIDATION.md).

For running several different metrics, use a collection of
`Box<zenpsy_api::DynMetric<'_>>` containing `zenpsy_api::Erased::new(metric)`
and call `zenpsy_api::score` for each member. Prepared collections use
`Box<dyn zenpsy_api::DynPrepareReference>`. These need no executor. Wrapper
stacks keep the original concrete assessment as the native downcast target.

One executor's registrations are alternative implementations of the same
effective metric, in fallback order; `run` selects one, not all registrations.
Use one executor per metric when a multi-metric run also needs fallback.
The local erased result does not promise Send/Sync. Prepare and consume native
results on their executing worker, or send an owned scalar projection.

See the [remaining abstraction surfaces](../../../docs/ZENPSY_RESIDUAL_SURFACES.md)
for video, planar, no-reference and generic map consumers that remain on the
legacy contract.
