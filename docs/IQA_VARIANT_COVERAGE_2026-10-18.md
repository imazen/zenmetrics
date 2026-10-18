# IQA variant selection audit, 2026-10-18

This audit compares the original six coverage manifests on each host against
the captured 122-route worker inventory and the additional APIs inspected below.
A route's presence in a manifest is scheduling coverage, not numerical or timing
qualification. Failed or unsupported attempts are not missing scheduler cells.

## Routes absent from the original six manifests

Both hosts omit `pieapp`, deliberately opt-in under
`benchmarks/iqa_compete_2026-10-16/selection.py`. Its frozen weight directory
exists in the model mapping. It is not currently a scheduled NN model; 23 of the
24 mapped NN models were selected. `pieapp-pyiqa` is separately an uncompiled
reference alias, not a second qualified implementation.

Seven captured reference aliases are uncompiled: `topiq-fr`, `topiq-fr-pipal`,
`wadiqam-pyiqa`, `ahiq-pyiqa`, `cw-ssim-pyiqa`, `dmm-pyiqa`, and
`pieapp-pyiqa`. This does not establish that an enabled oracle environment or
future adapter cannot run them.

The six classical GPU names are also outside these six manifests:
`ssim2-gpu`, `butteraugli-gpu`, `dssim-gpu`, `iwssim-gpu`, `zensim-gpu`,
and `cvvdp-gpu`. Historical GPU experiments elsewhere are not coverage in this
current CPU-classical/NN-CPU/NN-GPU graph. Inventory `compiled=true` does not
establish runtime support: the current CLI explicitly refuses `zensim-gpu`
scoring pending validation (`metrics/mod.rs`), so it must not be called a usable
missing measurement without a qualified implementation.

## Private, unpublished and parameterized variants

All six unpublished DVIF routes are included in separate frozen supplements:
port-min-v0.2, port-2d-v0.2, port-lap-v0.2, port-lap-v0.3, and the talk-faithful
luma float/integer routes. Publication remains prohibited; benchmark execution
is authorized. The original 122-route inventory predates that adapter.

Zensim appears only as `zensim` with no per-cell profile environment override in
the inspected frozen non-NN manifest. Thus that route does not benchmark every
profile. `crates/zenmetrics-api/src/zensim_profile.rs` exposes named previews
v0.1/v0.2, A, B/B-HDR, C/C-HDR and D (feature-dependent), plus explicit custom
ZNPR bakes. The inspected zensim source at `96c3c9bb` resolves latest-preview to
B; historical measurements must retain their own pinned dependency provenance.
HDR profiles require their real luminance ingress and are not interchangeable
with this study's RGB8 SDR inputs.

The same zensim source has 21 named research profile functions in
`zensim-experimental/src/lib.rs`: A-phone, linear-bounded, preview-v0.2,
preview-v0.4, and 17 preview-v0.5 names (base, balanced, compression, ensemble,
tuner, cross-codec, tuner-v2/v3/v4, balanced-v2/v3, compression-v2/v3, calibrated
tuner-v4/balanced-v3/compression-v3, and linear). None is separately identified
in these benchmark manifests. Preview-v0.2's experimental wrapper is not
necessarily a new arithmetic variant relative to its built-in counterpart.
This is an API inventory, not a claim that every research candidate is servable.

Margarine is absent from the current worker inventory and manifests. Its
standalone unpublished package has a selected `simd-row-malta` command,
max/p1/p2/p3/p6 outputs and a separate native-UHF research control. See
`crates/margarine/README.md` and `DATA_PROVENANCE.md`. Those need their own
constructor-inclusive benchmark adapter and controls before inclusion; historical
research timings do not substitute for this protocol.

Arbitrary private/custom ZNPR bakes are not enumerated by `MetricKind::all()`.
The current graph contains no per-bake identity, and this inspection does not
claim an exhaustive inventory of private experiment archives or identify a
specific private bake as qualified. In particular, the existence of E36 training
transport is not a serving adapter or permission to relabel an E34 package.
No private weights or unpublished crates were copied into the chart dataset.

## Existing refusals retained

Mac strict Metal has five model refusals: A-FINE, ARNIQA, CKDN, DeepDC and MUSIQ.
The exact-node audit distinguishes registry misses from explicit CPU-preferred
shape placement; see the linked
[WebGPU audit](../benchmarks/mac_validation_2026-10-18/WEBGPU_NODE_AUDIT.md).
These are scheduled failures, not omitted models or stalled workers. HDRVDP
requires absolute luminance/viewing conditions and the small MS-SSIM pyramid
refusals retain their original input contract. No fabricated luminance,
downsampling changes, fallback, thresholds or attempt caps are introduced.
