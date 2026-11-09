# Forward-compatible metric integration — 2026-11-09

This follow-up consumes the unpublished API in
[zenpixels PR 144](https://github.com/imazen/zenpixels/pull/144), pinned together
with zenpixels at `20cbfe10cfed18b90ce9c6d986d6aed4e7510b24`. The native
Butteraugli adapter uses
[`f223ca9e`](https://github.com/imazen/butteraugli/commit/f223ca9e4b606f51010b22bbb8fd8680f3cc5677),
from [its lending review](https://github.com/imazen/butteraugli/pull/22).
Zensim/SSIMULACRA2 pins and the selected Margarine vendor/teacher remain unchanged.
The root lock is regenerated through the CI-pinned snapshot, not local siblings.

## Construction and evolution

ScoreSpec, Implementation, EvalOpts and MapGrid expose accessors and builders
instead of public mutable fields. Semantic descriptors have explicit constructors;
there is no arbitrary default identity, score meaning or map geometry. EvalOpts
still has Default because score-only, uncapped evaluation is meaningful.
MapGrid constructors reject zero dimensions/cell pitch. Backend::Gpu is itself
non-exhaustive and has a constructor. Runtime output names may be owned, and
positive-infinity permission is independent of the perfect value. These
unpublished field/signature changes are queued explicitly in the core changelog.

Score and MapView remain non-exhaustive borrowed views with readable fields.
Their constructors provide external construction; they own no hidden resource
and descriptor metadata is still checked before consumption. Native assessments
remain complete structured results rather than a single universal number.

Named map requests use EvalOpts::with_map; a missing map is an explicit error.
The existing with_native_map still requests native diagnostics without promising
a particular common map. Maps remain off by default. Validation traverses erased
outputs through visitors rather than allocating boxed iterators; ordinary iterator
APIs remain available. This removes that validation allocation, not every possible
metric allocation.

## Planar and prepared work

Standard zenpixels PlanarSlice borrows planes, layout and color context. Each
plane has its own pitch and may omit unused final-row padding. The optional
PlanarMetric/PreparePlanarReference/PlanarWarmReference extension leaves packed
Metric callers usable. PreparedPlanarReference owns actual native reference work;
its input/factory mutable borrow and an operation's Pulse are not retained.
Candidates may change stride but must preserve geometry and interpretation.

Prepared lending exposes EvaluationRef only while native buffers are valid.
The checked callback validates outputs first, enforces callback cardinality and
preserves backend/consumer errors. Returned owned data cannot borrow the temporary
view. Each operation gets fresh cancellation/progress controls. Stacked Erased
wrappers and heterogeneous dynamic prepared collections forward this path.
Send is required only at a concrete worker boundary, not on all metrics/results.

## Native qualification and copies

The standalone native gate checks cold, prepared and lending scores/maps against
direct native calls at 97×73 with independent pitches, minimal backing storage
and NaN padding. All named scalar bits and map samples match. Changed color is
refused. Existing RGB8 byte-stride, erasure, PIQE, map and orchestrator controls
remain in the same gate.

| Adapter | New planar path | Boundary cost and limits |
| --- | --- | --- |
| Butteraugli | Native strided plane preparation/comparison and borrowed map callback | No input packing or second native map export on the lending path. Owned result requests still export an owned map. Native internal work/scratch remains. |
| Zensim | Owned native warm state plus reusable RGBA float packing | SDR linear BT.709 only in this new bridge; HDR remains supported by its existing explicit packed configuration. No HDR relabeling. |
| SSIMULACRA2 | Owned native warm state plus reusable RGBA float packing | Native component maps are retained. A fused map is not silently generated. No zero-copy planar claim. |
| Selected Margarine | Existing explicit packed native bridge | Frozen vendor, teacher and kernels unchanged; five scalar outputs/maps match at 64×64, 97×73 and 131×129. No new planar zero-copy claim. |

[JXL PR 143](https://github.com/imazen/jxl-encoder/pull/143) adapts owned planar preparation and lends a qualified native
Butteraugli-distance map directly into its existing block reducer. Its real
97×73 encode matches native backend bytes, scalar bits, maps and block distances.
An untraced control refuses owned-map export; requested traces deliberately keep
an owned map. Unsupported color/units/map geometry are errors. Qualification is
SDR BT.709/D65 at 80 cd/m²; other metrics need a separately qualified steering
policy. Generic diagnostic maps alone do not establish that policy.

The workbench's selected-metric iteration scoring remains a score-only observer
of saved sRGB8 reconstructions. Separate browser WASM instances still cross
memory/format boundaries; this native adapter does not make that route zero-copy.

## Reproduction and remaining surfaces

- `just iqa-native-check`: native scalar/map/stride/preparation/lending and
  heterogeneous erased controls, plus strict harness Clippy.
- `just iqa-margarine-check`: frozen selected native/frontend graph and exact
  scalar/map witnesses; strict Clippy.
- `just iqa-cross-check`: the same standalone native and Margarine i686 controls.
- JXL's `just psy-loop-check`: actual encode/trace/refusal and legacy callback
  controls, strict Clippy. Its recorded CI sibling cohort is used locally; existing
  fixed-point, feature hash, Libjxl byte-lock and divergence gates also pass.
- Core checks: no-default/std tests, Clippy/rustdoc, Rust 1.88, WASM,
  freestanding ARM and i686; codec bridge checks the corresponding core pin.

**0 of the 22 legacy normal-dependency packages is production-migrated here.**
These adapters are explicit validation tools and the new orchestrator is opt-in.
Shared planar and owned prepared/lending contracts now exist in review; temporal
push/drain/pooling, compatibility identities/columns, production model storage
and existing conformance guarantees still need their consumer migrations.
The [residual-surface inventory](ZENPSY_RESIDUAL_SURFACES.md) remains the removal
gate for zenmetrics-abstractions. No crate publication or site deployment is
performed by this work.
