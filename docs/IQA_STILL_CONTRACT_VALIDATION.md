# Still-image IQA contract validation — 2026-11-05

The shared core is [zenpixels-iqa, review 111](https://github.com/imazen/zenpixels/pull/111).
The [zencodec adapter, review 196](https://github.com/imazen/zencodec/pull/196)
only maps resolved outputs; metric implementations do not acquire a codec
runtime. Existing zenmetrics-abstractions consumers are not migrated here.
Video sessions/timing remain in the existing abstraction.

## Frozen native inputs and implementations

The standalone manifests and lockfiles bind the exact dependency graph:

| Consumer | Source | Qualified adapter surface |
|---|---|---|
| Serving Zensim A (`latest`, single-threaded) | [5475929d](https://github.com/imazen/zensim/commit/5475929d22306846cfe533312a110dbf7d3669e9) | Native scalar/map and actual cached-reference operations |
| Butteraugli 0.9.4 | [31687022](https://github.com/imazen/butteraugli/commit/31687022761bf5b401a9f1e0ed6e6dd407afee70) | Max and pnorm3 from one operation, native map and real cached reference |
| fast-ssim2 0.9.0 | [aaf7705b](https://github.com/imazen/fast-ssim2/commit/aaf7705be0b923fb9c0f7fbda37e13b2aaccd3fc) | Native whole-image scalar and cached reference; map requests refuse |
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
all three shapes, including strided maps. These adapters do not synthesize a
spatial map for SSIMULACRA2 or a warm cache for Margarine.

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

This is interface and forwarding validation on native x86 Linux. It does not
requalify metric algorithms against published reference goldens, establish
quality/performance/memory acceptance, validate HDR or every profile/input
format, or migrate/publish any metric implementation. Additional native ARM
execution is tracked by CI, not inferred from the x86 results. Margarine stays
unpublished and outside the normal workspace dependency graph.
