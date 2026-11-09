# Maps, no-reference metrics and JXL iteration — 2026-11-09

The map/NR contract in [zenpixels PR 131](https://github.com/imazen/zenpixels/pull/131)
has landed. This integration pins zenpixels and zenpsy-api together at
[`d234f3c6`](https://github.com/imazen/zenpixels/commit/d234f3c63efca06d5d7d52cb39b6c1f92961e98f).
The native byte-stride repair is incorporated from zenmetrics `4ecbfd85`;
metric source pins and the selected Margarine vendor/teacher are unchanged.

`EvalOpts::with_native_map()` requests diagnostics in the same scoring call.
`Evaluation::maps()` borrows `MapView`s: existing zenpixels `AnyPlane` storage,
`ScoreSpec` meaning and a regular `MapGrid` in input coordinates. Erasure forwards
these views while retaining the original native result. Undefined map samples
remain unchanged. Cell placement does not claim that only those pixels affected
the value; full support/component information remains on native assessments.

The current adapters expose original Zensim/Butteraugli/selected-Margarine map
storage and SSIMULACRA2's per-scale/channel component planes. Their pointer
controls establish that common views borrow those native allocations. The
SSIMULACRA2 native `diffmap()` method derives a fused map from those components;
the generic adapter does not silently allocate that derived map during iteration.
Score-only results expose no maps. Diagnostic witnesses remain explicitly
requested, in memory, and small; this change saves no maps to disk.

`NoReferenceMetric` and checked `score_image` share the existing native result,
identity and operation-control contracts. Optional `PairedNoReference` exposes
candidate outputs followed by reference outputs with `_ref` keys and retains
both native assessments. Actual preparation evaluates the reference once,
sharing it across results; only candidate diagnostic maps are requested.
The real PIQE bridge preserves direct native score bits for one-image, paired
and prepared calls at 64×64, 97×73 and 131×129 with padded rows. It is a
validation bridge over the unchanged legacy PIQE implementation, not a migration
of that package's production API.

## What the encoder needs

Source inspection uses jxl-encoder `c978839d41d770bbd76adec5a74684860e908e14`;
the only local change during inspection was its unrelated Cargo.lock. Sources:
[external callback](https://github.com/imazen/jxl-encoder/blob/c978839d41d770bbd76adec5a74684860e908e14/jxl-encoder/src/loop_metric.rs),
[backend](https://github.com/imazen/jxl-encoder/blob/c978839d41d770bbd76adec5a74684860e908e14/jxl-encoder/src/vardct/perceptual_backend.rs),
[loop](https://github.com/imazen/jxl-encoder/blob/c978839d41d770bbd76adec5a74684860e908e14/jxl-encoder/src/vardct/perceptual_loop.rs),
[XYB conversion](https://github.com/imazen/jxl-encoder/blob/c978839d41d770bbd76adec5a74684860e908e14/jxl-encoder/src/vardct/xyb.rs).

| Need | Shared metric contract | Encoder adapter policy |
| --- | --- | --- |
| One reference and repeated candidates | Actual optional reference preparation; borrowed strided pixel views | Construct once per encode and retain native reference work. |
| Global candidate/seed judgement | Named native scalar and its direction/unit | Select an output and target explicitly. A native quality score is not automatically a Butteraugli distance. |
| Local quantizer steering | Optional native map planes, meaning and grid placement | Select and qualify a spatial signal, normalize to the loop's target scale and reduce into blocks. The current reducer uses 8×8 blocks and a 16-norm times 1.2. Those are not universal IQA semantics. |
| Color interpretation | Actual pixel color metadata; explicit refusal of unsupported input | Use the same declared primaries, white/absolute scale and display interpretation for reference and reconstruction. |
| Low-copy operation | Borrow map/input views; preserve ownership and stride | Keep conversion and metric scratch reusable; reduce a borrowed map directly instead of expanding every native grid into another full-resolution array. |

The useful SDR ingress is **planar f32 linear RGB, BT.709/sRGB primaries and
D65**, before the metric's own perceptual transform. It is not encoded sRGB
and not the encoder's XYB. The encoder converts wide primaries into its XYB
working space and reconstructs linear RGB from XYB; its pointwise transform
is not Butteraugli's spatially adaptive opsin conversion. Sharing the XYB
planes would substitute a different transform.

The external callback currently receives slices/geometry without color metadata.
The loop adapter must resolve that interpretation before attaching a metric.
The native SDR witness uses Butteraugli's 80 cd/m² intensity target. HDR needs
explicit relative/absolute luminance and viewing configuration; the existing
PQ/HLG intensity dispatch does not qualify arbitrary HDR/wide-gamut external
callbacks. No new HDR encoder qualification is claimed here.

There is no single input format that makes every metric zero-copy. A metric
that accepts these linear planes can borrow them. An RGB8-only metric needs a
real conversion; reuse its destination storage rather than relabeling linear
values. Reference conversions/precomputation happen once, candidate work per
iteration. Shared planar picture ingress should use zenpixels' standard types;
this PR does not add a competing planar image hierarchy or change the core's
currently packed scoring signatures.

## Where copies still happen

The JXL-shaped native witness preserves borrowed full padded candidate buffers,
matches both Butteraugli scalar bits and map bits, reuses the caller's map
allocation, and exposes that buffer through a borrowed `MapView`. This is
interface/buffer validation, not a benchmark or a claim of no internal copies.

Inspection of pinned Butteraugli `31687022`,
[`compare_linear_planar_into`](https://github.com/imazen/butteraugli/blob/31687022761bf5b401a9f1e0ed6e6dd407afee70/butteraugli/src/precompute.rs),
finds two concrete limits: it requires `stride * height` backing storage,
including last-row padding, and copies the finished internal map into the
caller buffer before recycling its scratch. JXL already holds full padded
buffers. A minimal borrowed plane can legally omit trailing padding and cannot
generally be passed through that entry unchanged. JXL also supplies one shared
stride for its three reconstruction planes; the native entry has one stride
argument. Independently strided generic planes require compatible native
ingress or an explicit adaptation.

Removing the map-export copy needs a native lending/callback entry that lets
the consumer inspect/reduce the map before scratch is recycled. The current
owned `Metric::Output`/`WarmReference::Output` does not lend scorer scratch;
`ErasedEvaluation` also requires a static owned native result. An optional
lending extension could serve this workload, but no such API or kernel change
is implemented by the maps/NR review. It should retain checked geometry,
fresh Pulse controls and native map meaning without exposing JXL block policy.

The browser's existing external steering route converts each reconstruction
to opaque sRGB8, transfers it through JavaScript into a metric module, then
copies the returned map into the encoder. The codec and metric modules have
separate WASM instances. Borrowed Rust views alone cannot remove these transfers.
An encoder and selected metric in one Rust/WASM instance could share plane
views and use a native lending/reusable-buffer route. Independently loaded
modules or JS tensor runtimes need an explicit interop design and may require
copies. The existing preview steering path also assigns zero error to
uncovered/undefined map cells; that is an adapter policy, not a guarantee of
native map data or qualification of every map as a local refinement signal.
No end-to-end zero-copy browser claim is made.

## Workbench behavior and checks

The workbench now offers **Score iterations with selected metric** for a live
or loaded trace. It routes every stored reconstruction through the existing
score-only CPU/NR/learned dispatcher, preserves every named scalar at full
precision and reports per-frame refusals. Selecting a row selects the displayed
iteration. Stop/trace replacement prevents stale asynchronous results from
being published. Learned metrics retain their existing weight consent and
execution-provider controls. The new helper module shares the immutable asset
generation with the app, worker and WASM packages.

These observations describe the trace's **stored sRGB8 reconstruction**, not
the internal linear f32 candidate. They do not change encoded bytes, loop maps,
targets or seed selection. A scalar-only metric can observe iterations or rank
candidates; supplying local quantizer steering requires a separately qualified
spatial signal. No-reference quality values do not automatically supply one.

Local controls pass for native map storage and PIQE composition, selected
Margarine, strict harness lint, i686 Cross and the CI-pinned orchestrator graph.
`just browser-loop-score-check` checks routing for all 102 currently runnable
catalog entries with an explicit oracle; it does not execute 102 algorithms.
`just browser-loop-ui-check` exercises the real UI with deterministic RPC
responses, cancellation/refusal controls and immutable asset assembly.
The full native/WASM viewer gate additionally scores its existing two-frame
trace with PSNR; that end-to-end gate is configured here and was not run locally.

Reproduce: `just iqa-native-check`, `just iqa-margarine-check`,
`just iqa-byte-stride-check`, `just iqa-cross-check`,
`just browser-loop-score-check`, and `just browser-loop-ui-check` with
Playwright/Chromium installed and disk-backed `TMPDIR`.
