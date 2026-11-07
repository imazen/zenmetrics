# Still-image IQA contract validation — 2026-11-05

The shared core is [zenpsy-api, review 111](https://github.com/imazen/zenpixels/pull/111),
a standalone package in the zenpixels repository. Consumers import
`zenpsy_api` and may alias it locally as `iqa`.
The [zencodec adapter, review 196](https://github.com/imazen/zencodec/pull/196)
only maps resolved outputs; metric implementations do not acquire a codec
runtime. Existing zenmetrics-abstractions consumers are not migrated here.
Video sessions/timing remain in the existing abstraction.

## Frozen native inputs and implementations

Both standalone harnesses pin the renamed core and its pixel types together at
[41c3cffe](https://github.com/imazen/zenpixels/commit/41c3cffe7763fb6a20e958b04e9cefb3ec4d82f6).
All four native adapters pass again on x64 with that source, including actual
SSIMULACRA2 cached maps and the original selected Margarine frontend. The
standalone manifests and lockfiles bind the exact dependency graph:

| Consumer | Source | Qualified adapter surface |
|---|---|---|
| Serving Zensim A (`latest`, single-threaded) | [5475929d](https://github.com/imazen/zensim/commit/5475929d22306846cfe533312a110dbf7d3669e9) | Native scalar/map and actual cached-reference operations |
| Butteraugli 0.9.4 | [31687022](https://github.com/imazen/butteraugli/commit/31687022761bf5b401a9f1e0ed6e6dd407afee70) | Max and pnorm3 from one operation, native map and real cached reference |
| fast-ssim2 0.9.1 | [6d27c728](https://github.com/imazen/fast-ssim2/commit/6d27c728a4094cbbcf5f04e3b61859f062951f2e) | Native scalar, fused diffmap, per-scale terms and actual cached-reference maps |
| Selected Margarine `simd-row-malta` | In-tree selected kernel from zenmetrics b50fea22; 128 rows, 512 columns, existing vendor snapshot | Max/p1/p2/p3/p6 and native map; preparation explicitly refuses |

The Zensim dependency names imazen/zensim explicitly: a local directory named
zensim can refer to the research repository. Margarine's separate manifest
retains its original Butteraugli teacher pin
[13c49cbc](https://github.com/imazen/butteraugli/commit/13c49cbcab6e2b2bb65f60e3cb28344463bacbf7)
and never replaces `vendor/butteraugli`. Its private geometry entry now returns
Send+Sync errors so the IQA envelope can retain the boxed native cause. The
legacy entry preserves its original error signature; no arithmetic changed.

Inputs are deterministic textured sRGB RGB8 pairs at 64x64, 97x73 and 131x129,
with a distributed +17 channel perturbation. Strided repeats use nine/fifteen
padding bytes and minimally sized final rows; padding has a distinct sentinel.
They are generated directly by the committed fixture code, with no decoder,
corpus, model training or external oracle involved.

## Observed controls

All four adapters pass scalar declaration, sanitized/unique column identity,
map-request scalar identity, supported reduced-map geometry and cancellation
admission controls on all three shapes. All four preserve native scalar bits
between tight and strided inputs.

Serving Zensim A, Butteraugli and fast-ssim2 preserve exact scalar bits between
cold and actual cached-reference calls at every shape. Zensim and Butteraugli
map/scalar bits match separate direct native calls at 97x73. Margarine's five
scalar outputs and full native map bits match direct selected-kernel calls at
all three shapes, including strided maps. The Margarine adapter does not
synthesize a warm cache.

SSIMULACRA2 additionally checks all native map values against direct calls on
those three shapes plus 5x3 and 130x9: cold, strided and actual cached-reference
results preserve exact score and map bits. Identity inputs produce zero maps.
The primary `diffmap` retains global support from its L4 normalization. Each
`{ssim,artifact,detail_lost}_{x,y,b}_{scale}` map retains its pyramid grid and
five-sample blur halo; sub-8 reflect-padded maps declare global support.
These error contributions are not per-pixel SSIMULACRA2 scores.

The first validation pinned fast-ssim2 before its map API and refused map
requests. That was a validation coverage gap: current fast-ssim2 already
provides both cold and cached maps, also used by the existing browser adapter.
The corrected panel updates only this standalone dependency; the root workspace
lock, metric algorithms and frozen Margarine teacher remain unchanged.

A check-budget token stops Zensim, Butteraugli and SSIMULACRA2 inside their
native calls; each retains TimedOut and an underlying native error. The
Butteraugli tiny-reference refusal retains its concrete native cause. A
Margarine configuration-error control retains the exact boxed cause allocation.
Margarine's selected kernel has no Pulse argument, so its prototype adapter
checks cancellation only before and after that call; it is not preemptible.

Both standalone harnesses pass strict Clippy. The core separately passes
native controls, strict Rustdoc/Clippy, Rust 1.88, WASM and freestanding ARM
compilation. The codec adapter separately checks resolved geometry, actual
context and original mapping-error/stop propagation.

## Reproduction and limits

From the zenmetrics root, with current Rust and git dependency access:

```sh
just margarine-vendor-check
just iqa-native-check
just iqa-margarine-check
```

Use the normal resource wrapper on shared hosts. These standalone workspaces
have their own tracked lockfiles and do not regenerate the root Cargo.lock.
CI runs both harnesses on Linux x64 and aarch64.

The [Linux x64 panel](https://github.com/imazen/zenmetrics/actions/runs/41988502159/job/127114853182)
and [Linux aarch64 panel](https://github.com/imazen/zenmetrics/actions/runs/41988502159/job/127114853279)
both pass the original native controls and strict lint at source `40590360`;
those historical panels predate the SSIMULACRA2 map correction. The corrected
source-pinned panel at `977a114a` passes on native x64 and aarch64 Linux with
strict Clippy. The existing CI matrix repeats it on both architectures.

This is interface and forwarding validation on native x86 and ARM Linux. It does not
requalify metric algorithms against published reference goldens, establish
quality/performance/memory acceptance, validate HDR or every profile/input
format, or migrate/publish any metric implementation. Margarine stays
unpublished and outside the normal workspace dependency graph.
