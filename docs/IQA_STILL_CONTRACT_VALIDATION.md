# Minimal typed IQA contract validation — 2026-11-08

The current [zenpsy-api review](https://github.com/imazen/zenpixels/pull/111)
uses a typed `Metric::Output: Evaluation`, named score views and an optional
`PrepareReference` extension. The reviewed core/pixel pin is
[d9cfb73d](https://github.com/imazen/zenpixels/commit/d9cfb73d9195b26101c82b36bbd09aaf425dd073).
The [codec bridge](https://github.com/imazen/zencodec/pull/196) forwards that
concrete output without boxing or losing native fields.

## Current measured coverage

The serving Zensim, Butteraugli and fast-ssim2 source pins in the historical
source table below are unchanged. Their typed adapters preserve native scalar
bits across cold, strided and actual prepared calls at 64x64, 97x73 and 131x129.
Butteraugli's max/pnorm3 and selected Margarine's max/p1/p2/p3/p6 remain separate
named outputs. Native result objects remain directly accessible.

Zensim additionally matches independent direct `compute_hdr` calls exactly at
97x73 for all nine combinations of PQ, HLG and absolute-linear light with
BT.709, Display P3 and BT.2020 primaries. Strided repeats match exactly; a
mismatched transfer declaration refuses. The linear fixture is in cd/m²;
PQ/HLG fixtures use explicit native display parameters. This validation adapter
refuses attached overriding colour contexts on its HDR path. It does not
qualify all HDR input formats, viewing conditions or prepared HDR scoring.

Diagnostics are off by default. Explicit small witness tests retain exact
native Zensim/Butteraugli maps and every SSIMULACRA2 component/grid, including
5x3 and 130x9 edge cases. These are in-memory fixtures only; no maps are saved.
The native Butteraugli prepared API returns its internal pooling map even when
`compute_diffmap` is false; the adapter drops that scratch before returning a
score-only assessment. Selected Margarine pools its five scalars from internal
map scratch, likewise dropped unless diagnostics were requested. Neither
adapter copies or persists diagnostic arrays during routine scoring.

Margarine still uses the frozen selected kernel and teacher. Its five scalar
and opt-in native map outputs match direct calls at all three geometries,
including padded rows. Its original selected frontend compiles, and the boxed
error-allocation retention control passes. There is no synthesized preparation.

## Orchestration boundary

The opt-in `zenmetrics-orchestrator/psy` path consumes the actual core. It
registers worker-local factories, erases native results only at that boundary,
retains every named score and actual implementation identity, and prepares a
reference once for each prepared batch attempt. Real controls cover all four
methods: three native prepared implementations and cold selected Margarine.
Synthetic failures verify resource-only explicit fallback, preservation of a
completed prefix, refusal to change effective scoring configuration, original
error causes/traces, and no retry after cancellation or computation failures.
Routine batches refuse diagnostic requests before constructing a scorer.

This is an additive backend-neutral route. The legacy GPU worker pool, chooser,
benchmark cache, CLI and existing zenmetrics-abstractions consumers have not
been migrated. There is no new GPU numerical qualification, performance claim,
video contract, native planar view abstraction or no-reference interface.

## Reproduction

- `just iqa-native-check`: native scalar/stride/preparation, opt-in diagnostics,
  Zensim HDR, runtime registration/fallback and strict lint controls.
- `just iqa-margarine-check`: original selected frontend, frozen vendor check,
  exact native selected-kernel and runtime cold-registration controls.
- `just iqa-cross-check`: the same native/runtime panel and selected Margarine
  through i686 Cross.
- Core: `just iqa-check` and `just iqa-portability-check` in zenpixels.
- Codec adapter: the same recipes in zencodec.

Current local native validation passes on Linux x64 and i686 through cross. Core and codec adapter pass
Rust 1.88, no_std, WASM and freestanding ARM checks, tests/doctests and strict
Clippy/Rustdoc. Native algorithm/reference parity, every model and every input
format remain outside these integration checks. Historical observations below
retain their original source identities and scope; they are not current
minimal-contract coverage claims.

# Historical rich-contract validation — 2026-11-05

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
