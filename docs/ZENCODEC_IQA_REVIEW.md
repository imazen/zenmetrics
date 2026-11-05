# Metric abstractions and zencodec IQA integration

Source review, 2026-11-04. This is a proposed direction, not an approved API
change or a claim of end-to-end integration.

## Inspected state

- zenmetrics `master`: `d46ec235d3bb080188445f9ccf4685afdff209b9`.
- zencodec checkout: `f6ccbdc` (the unrelated local edits were confined to its
  AV1 validation tool).
- zencodecs in zenpipe: `43301b4`.
- zenpixels committed source: `5bf81d640752793454802d7a94346c3c1e97c81e`,
  package 0.3.1. Its uncommitted documentation was excluded from the build.
- Unmerged abstraction branches: `abstractions-local-planelayout` at
  `55f86753`, `feat/strided-planes-zenpixels-0.3` at `70c50f52`, and
  `feat/zenpixels-planes` at `3423aaeb`. These are distinct from master.

## Recommendation

Evolve the existing `zenmetrics-abstractions` contract into the proposed
`zencodec-iqa` contract. Preserve its metric, result, map and warm-reference
designs. First qualify its pixel boundary, session invariants and adapters;
a package rename alone does not establish interoperability.

Keep the dependency direction:

```text
zencodecs / applications
        -> metric implementations
        -> zencodec-iqa
        -> zenpixels + how-far + whereat

zencodec-iqa's optional codec adapter -> zencodec
```

Each arrow means "depends on". zencodecs may consume the IQA contract;
the contract must not depend on zencodecs. The codec core need not acquire
metric implementations.

The core should remain `no_std + alloc`, `forbid(unsafe_code)`, and independent
of codecs, CMS implementations, GPU runtimes, model loaders, filesystem I/O,
CLI configuration and fleet scheduling. An optional zencodec adapter can
borrow mapped decoded output. Registration and concrete scorer construction
belong above the contract. Keep the existing MIT/Apache contract licensing;
moving implementations requires their own provenance/license review.

## What already belongs in the contract

[`Metric`](../crates/zenmetrics-abstractions/src/metric.rs) is object-safe,
`Send`, and mutable so implementations can reuse scratch. It returns owned
[`Evaluation`](../crates/zenmetrics-abstractions/src/eval.rs) objects with
multiple named scores and maps. `NoReferenceMetric` preserves the real
single-image operation; `PairedNoReference` is an explicit convenience.

Packed inputs already use zenpixels `PixelSlice`, including row stride.
Planar inputs are validated. Warm references explicitly distinguish real
reference-work reuse from `ColdReplay`; their contract requires exact cold
parity. Temporal sessions distinguish stateless, delayed and pooled metrics.
Cancellation/progress already uses how-far, and located errors use whereat,
matching the ecosystem's control/error foundations.

[`MapGeometry`](../crates/zenmetrics-abstractions/src/map.rs) describes native
cells, origin, coverage and support. Subsampled visualization retains the
native result for folds. This is materially better than treating every map
as an input-sized image. Preserve these distinctions.

[`MetricId`](../crates/zenmetrics-abstractions/src/identity.rs) separates
family, variant, reproduced reference, implementation revision, backend and
sorted configuration. Preserve the score-revision discipline and full
canonical provenance when changing packaging.

## Changes needed before extracting or publishing the contract

### 1. Complete the shared plane migration

Master imports the removed `zenpixels::PlaneLayout`. With the inspected
zenpixels 0.3.1 source it fails at `input.rs:13` with E0432. The permissive
`>=0.2.11, <0.4.0` requirement therefore admits an incompatible source version.

`feat/zenpixels-planes` replaces the intermediate IQA-local layout with
`zenpixels::plane::{PlanarLayout, PlaneEncoding, PlaneRole, PlaneSpec}` and
shared chroma types. Its isolated abstraction tests pass with the inspected
0.3.1 source. Reuse and review that work instead of introducing another
plane vocabulary. This test does not qualify every metric consumer or the
whole branch.

Preserve native sample encoding, chroma siting, crop phase and color matrix.
Master's separate `significant_bits` field says the low bits are used; it
cannot describe P010's shifted codes. Contemporary zencodec frames already
carry `SampleEncoding`, subsampling, siting, crop and resolved CICP claims.
Do not turn native Y/Cb/Cr or auxiliary code planes into grayscale color to
make them fit an input constructor. The shared-plane branch still wraps
planes in `PixelSlice` and retains a separate significant-bit field, so its
passing tests do not finish this semantic integration.

### 2. Negotiate interpretation, not just a descriptor

`Image::format()` drops the packed slice's color context and dimensions.
`rgb8_view()` accepts descriptor equality with `RGB8_SRGB`; it does not
inspect attached ICC/CICP context. `Planes::new()` compares descriptor
transfer/range/primaries, without a complete frame interpretation.

Add explicit input requirements and preparation validation for both sides:
color interpretation, relative versus absolute luminance, precision, alpha,
geometry and minimum size. Validation must include the actual image/context;
an `accepts(InputFormat) -> bool` answer alone cannot certify it. Reuse
zenpixels-convert for approved conversions and retain a preparation receipt.
That receipt should bind the selected image/region, orientation, color and
alpha policy, any resampling, and resulting sample contract.

zencodec's `ImageOutput::map()` yields a lifetime-bound mapping guard;
`with_pixels()` can borrow packed color inside a callback. Build the IQA view
inside that lifetime. `ImageInterpretation` separately retains composition,
orientation, selected geometry, native frame signal and presentation. Resolve
these deliberately before scoring; equal array dimensions are insufficient.

The existing zencodec-render companion handles oriented canvas and explicit
color preparation. It documents full-image reconstruction/copying and leaves
planar video, resizing and HDR display mapping unqualified. Reuse its
qualified paths, but do not treat its default RGBA8 sRGB rendering as an
arbitrary HDR/native-component IQA front door.

### 3. Add operation controls and discoverable capabilities

`EvalOpts` currently contains only a map request and borrowed pulse. It has
no memory, image-size, threading or map-output budget. `to_owned_image()`,
`ColdReplay`, map resampling and result enumeration allocate infallibly.

Introduce operation-level limits with checked size arithmetic and fallible
large allocations. Reuse applicable zencodec resource/threading concepts
through the integration boundary, while keeping GPU-specific policy out of
the base traits. State whether each limit is enforced or advisory and count
retained references, scratch and map outputs. Caller-managed pools should
avoid nested frame/metric parallelism.

Declare supported input contracts, size limits, named maps, actual reference
reuse, temporal behavior and resource behavior before expensive execution.
Keep capabilities configuration/backend-specific. A typed configured scorer
can retain its current mutable execution shape; a separate reusable factory
can create independent scratch-bearing instances for applications/registries.

### 4. Integrate the existing scorer families

The common trait is implemented by native CPU ports and learned metrics.
`zenmetrics-api` remains a separate concrete enum with its own `MetricKind`,
`Backend`, score structs, sessions and pixel/HDR routes. It does not implement
the shared `Metric` trait, and the inspected GPU crates do not implement it
directly. The CLI still bridges those families separately.

Add adapters with exact scalar/map/warm parity before routing consumers
through one contract. Preserve existing scheduler memory management and
runtime reuse. GPU construction, reference residency, batching and device
sharing need explicit adapter capabilities; do not require CubeCL types in
the core. Retain the real NR operation instead of requiring a synthetic
reference for every caller.

### 5. Bind targets and results to full identity

`ScoreSpec` supplies a key and `ScoreKind`; it has no typed target, model
asset identity or preparation identity. `Evaluation` does not independently
carry the configured metric identity after it leaves the scorer. The
eight-hex-digit column suffix is a compact label, not a collision-proof
persistent cache key. Persist full identity and pair/preparation provenance.

Separate direction, scale/unit and valid identity/nonfinite behavior where
necessary. Preserve valid infinite PSNR and explicitly undefined native-map
cells. Do not normalize all scores to an invented common quality scale.
Bind a target to the exact metric profile/revision, output key and comparison
direction. Model/graph hashes and preprocessing must participate when they
can affect scores. NN adapters currently place engine names in the reference
description; represent runtime/backend and reproduced method separately.

zencodec's `LossyTarget` currently promises approximate one-pass metric
targets, while zencodecs' `transcode_to_quality` is specialized around
Profile-A zensim. An optional generic search driver should take a configured
scorer/warm reference and an encoder callback, keep limits on attempts and
resources, and return the measured result plus whether the target was met.
It must retain the selected bytes and handle non-monotonic observations.
An approximate encoder dial must never become an asserted measured target.

### 6. Strengthen session and conformance contracts

The default `StatelessSequence` stores neither the initial dimensions nor a
failed-state flag. A synthetic scorer exercising that default accepted a
changed-size second frame, then accepted another push and `finish()` after a
`DimensionMismatch`. This contradicts the documented geometry and failed-state
contract. The probe exercises default session mechanics, not metric arithmetic.

The current conformance harness checks undeclared output keys but does not
require every declared secondary output to be returned exactly once in order.
The synthetic scorer declared two outputs, returned only the headline, and
the harness reported zero violations.
Add negative controls for those omissions, changed-size sequences, error
poisoning, rejected replacement references and recovery, differing color
contexts, shifted code encodings, odd chroma crops, strided planes, resource
refusal and cancellation during work. Retain all existing numerical gates.
Put reusable test fixtures/checks behind a testkit or opt-in feature rather
than making their helpers part of the permanent base API by default.

For variable-rate media, preserve explicit presentation timestamps/durations
from zencodec instead of deriving time from `SequenceInfo::frame_rate` and
push order. Rate-only sequences remain useful for genuinely fixed-rate input.
Do not advertise row/strip streaming for algorithms that materialize frames;
introduce it only for implementations that can honor halo/finalization and
exact parity requirements.

## Qualification and implementation order

1. Review and qualify the shared-plane branch against current zenpixels and
   current zencodec mapped outputs, including interpretation refusal cases.
2. Correct sequence/conformance omissions without weakening assertions.
3. Design the limits, capabilities, target and result-envelope additions
   using actual zencodecs/encoder-search callers; review public signatures.
4. Prove one native CPU adapter, one GPU adapter and one NR adapter through
   decode/prepare/score, then migrate the other consumers.
5. Extract/rename the contract, retain compatibility re-exports where possible,
   and qualify packaging, `no_std`, browser and native targets before release.

The first useful integration is decode a selected image, resolve its
interpretation, prepare the reference once, and score multiple decoded encode
candidates under one exact score/preparation identity. Keep search,
registration and fleet execution above this contract.

## Checks performed

Source was exported into isolated scratch workspaces; the repository lockfile
and sibling sources were unchanged. Logs are under
`$HOME/tmp/iqa-review-20261104/` on the review machine.

| Source/dependency combination | Result |
| --- | --- |
| Master abstractions, registry zenpixels 0.2.16, `cargo check --no-default-features` | Pass |
| Same combination, `cargo test --no-default-features` | 34 unit tests and one doctest pass |
| Master abstractions, committed zenpixels 0.3.1 at `5bf81d64` | E0432: removed root `PlaneLayout` |
| `feat/zenpixels-planes` abstractions, same zenpixels 0.3.1 source, `cargo test --no-default-features` | 35 unit tests and one doctest pass |

These are isolated crate checks on the native host, not full-workspace,
cross-platform or end-to-end codec/IQA qualification. The normal CI-pinned
snapshot failed before compilation because its zensim pin
`96c3c9bb900b580a4d9c0266a0a0b50203757528` was unavailable from its configured
repository. No pin, lockfile, metric arithmetic or public API was changed.

The repository-wide advisory `just quality --jobs 4` also ran. Its format
check found an existing difference in `crates/zenfleet-worker/src/lib.rs`;
`just fmt-check` identified the same file. Dependency analysis failed resolving
zencodec `>=0.2.0, <0.4.0` through the local zenavif checkout. The quality
summary's zero-warning clippy count is not evidence of a successful workspace
build. Existing stale-document and unused-dependency findings remain. The
review's own whitespace and relative-link checks pass.

## Default-contract probe

This diagnostic ran as an example in the isolated `feat/zenpixels-planes`
snapshot above. Its `metric.rs` and `conformance.rs` are unchanged from master.
Observed output:

```text
changed_dimensions_accepted=true
failed_push_kind=DimensionMismatch
valid_push_after_error_accepted=true
finish_after_error_accepted=true
missing_declared_secondary_output_violations=0
```

The scorer intentionally omits one declared output to test the harness. It
is a synthetic protocol probe, not an image-quality implementation.

```rust
use std::borrow::Cow;
use zenmetrics_abstractions as za;
use za::{Metric, Evaluation, EvalOpts, Image, MetricResult};

struct Probe;
impl Metric for Probe {
    fn id(&self) -> za::MetricId {
        za::MetricId::new("probe", za::Implementation::new("probe", "0.0.0", 1))
    }
    fn outputs(&self) -> Cow<'static, [za::ScoreSpec]> {
        Cow::Owned(vec![
            za::ScoreSpec::new("score", za::ScoreKind::Difference),
            za::ScoreSpec::new("secondary", za::ScoreKind::Difference),
        ])
    }
    fn accepts(&self, format: &za::InputFormat) -> bool { za::is_rgb8_srgb(format) }
    fn evaluate(&mut self, r: Image<'_>, d: Image<'_>, opts: EvalOpts<'_>) -> MetricResult<Box<dyn Evaluation>> {
        opts.checkpoint()?;
        opts.refuse_map("probe")?;
        za::check_same_size(&r, &d)?;
        Ok(Box::new(za::ScalarEval::single("score", 0.0)))
    }
}
fn input(data: &[u8], side: u32) -> Image<'_> {
    za::zenpixels::PixelSlice::new(data, side, side, side as usize * 3, za::zenpixels::PixelDescriptor::RGB8_SRGB).unwrap().into()
}
fn main() {
    let large = [0u8; 8*8*3];
    let small = [0u8; 4*4*3];
    let mut metric = Probe;
    let mut seq = metric.sequence(za::SequenceInfo::new()).unwrap();
    seq.push(input(&large, 8), input(&large, 8), EvalOpts::new()).unwrap();
    println!("changed_dimensions_accepted={}", seq.push(input(&small, 4), input(&small, 4), EvalOpts::new()).is_ok());
    let error = seq.push(input(&large, 8), input(&small, 4), EvalOpts::new()).err().unwrap();
    println!("failed_push_kind={:?}", error.error().kind());
    println!("valid_push_after_error_accepted={}", seq.push(input(&large, 8), input(&large, 8), EvalOpts::new()).is_ok());
    println!("finish_after_error_accepted={}", seq.finish(EvalOpts::new()).is_ok());
    let violations = za::conformance::check(&mut metric, &input(&large, 8), &input(&large, 8));
    println!("missing_declared_secondary_output_violations={}", violations.len());
    for v in violations { println!("{}: {}",v.check,v.detail); }
}
```
