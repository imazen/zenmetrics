# #34 — zenjxl sweep `encode_ms`: the parasite is in the `--plan` path, not the encoder

**Date:** 2026-09-09 · **Host:** i265 (Core Ultra 7 265K, 20 threads, 30 GiB) ·
**Data:** `jxl_encode_ms_parasite_2026-09-09.tsv` (+ `.meta` for exact commands)
**Harness:** `crates/zenmetrics-cli/examples/jxl_encode_ms_probe.rs` (committed)

Issue #34 reported that the sweep's `encode_ms` for zenjxl is an additive
~31 s/MP term that flattens a real 17–68x effort span to ~0.9x, and stated the
mechanism was not identified. This run identifies **where** it lives and rules
out four plausible causes. It does not yet name the line of code.

## Headline

**The `--knob-grid` path is clean. The `--plan` path is not.** Same binary, same
encoder, same image, same effort.

| path | 1 MP, e5 | 1 MP, e9 | effort span |
|---|--:|--:|--:|
| `--knob-grid '{"effort":[..]}'` | **131 ms** | **1499 ms** | **42x**, monotonic |
| `--plan rd_core` (`vd-e5/e9_zen_def`) | **14 247 ms** | **14 169 ms** | **1.005x**, flat |

That flat ~1.0x across e5–e9 is exactly the issue's signature, reproduced on
current `master`. On this box the parasite is ~13.5 s/MP where the fleet rollup
implied ~31 s/MP — a faster box, same shape.

Full ladder for the clean path (per the sweep-discipline size rule, so slope and
intercept are separable rather than hidden in a single "ms/MP"):

| size | MP | e1 | e5 | e9 |
|---|--:|--:|--:|--:|
| 64² | 0.0041 | 0.86 ms | 2.11 ms | 6.89 ms |
| 256² | 0.0655 | 2.30 ms | 11.47 ms | 75.38 ms |
| 512² | 0.2621 | 6.23 ms | 31.05 ms | 341.72 ms |
| 1024² | 1.0486 | 35.73 ms | 130.69 ms | 1498.51 ms |

Monotonic in both axes, 42x effort span at 1 MP. Nothing resembling 31 s/MP.

## What it is NOT — four causes ruled out by measurement

1. **Not the encoder core.** Direct `jxl_encoder::LossyConfig` at 1 MP,
   `threads=1`: 48.7 / 146.9 / 292.1 / 1678.1 ms for e1/e5/e7/e9 — a 34x span,
   monotonic.
2. **Not the zenjxl / zencodec wrapper.** Wrapper and direct agree within noise
   at every size and effort measured (e.g. 1 MP e9: 982.9 vs 987.0 ms), and
   produce byte-identical output.
3. **Not encoder construction.** `cfg.job().encoder()` measured **0.000–0.002 ms**
   at every size. `apply_threads` plus a struct move, as the source suggests.
4. **Not fleet concurrency.** 16 simultaneous encodes on this 20-thread box
   inflate per-encode wall by 2.2–4.5x — but they **multiply**; the effort span
   survives intact (256²: 60 / 206 / 2473 ms/MP at e1/e5/e9). The parasite is
   *additive* and *flattens* the span, so contention cannot be it.
5. **Not within-process accumulation.** Nine consecutive same-effort cells in one
   `--jobs 1` process: 43.7, 30.0, 30.4, 31.1, 31.6, 32.0, 33.1, 35.0, 36.3 ms.
   No index-driven degradation (the first cell is the *slowest*, i.e. warm-up).

`.with_threads(1)` — which the plan path pins per cell for content-addressing
determinism (`sweep/plan.rs`), while the knob-grid path's `apply_threads` yields
`threads=0` (ambient rayon pool) — accounts for **3.1x**, and preserves the
effort span. It is a contributor, not the parasite.

## Where it actually is: a bimodal split *within* the plan cells

Nine `rd_core` cells, all three sizes, split into two populations that hold
their membership exactly across sizes:

| population | cells | ms/MP @256² | @512² | @1024² |
|---|---|--:|--:|--:|
| **fast** | `vd-e5_libjxl_def`, `vd-e9_libjxl_def` | 1539–1780 | 1487–1631 | 1662–1767 |
| **slow** | the other 7 | 17 175–17 974 | 8289–8956 | 13 212–13 892 |

The slow population is **~12x** the fast one and covers every `mod-*` cell,
every `*_zen_*` cell, **and** `vd-e7_libjxl_def`.

That last one is the sharpest clue and the reason this writeup stops short of
naming a cause: `vd-e5_libjxl_def` and `vd-e9_libjxl_def` are fast while
`vd-e7_libjxl_def` — same mode, same strategy, same `_def` internal label, only
effort differs — sits with the slow group. **So the split is not explained by
mode, by strategy, or by effort.** Some other per-cell knob that the planner
varies is doing it. The emitted plan manifest carries only counts
(`cells`, `duplicates_merged`, `invalid_skipped`, …) and no per-cell config, so
it cannot be read off the artifact.

Also unexplained: the slow population's ms/MP is **non-monotonic in size**
(17.5k → 8.6k → 13.5k), i.e. the parasite is neither a fixed per-cell cost nor
cleanly pixel-proportional. Absolute parasite ≈ 1139 / 2235 / 14 070 ms at
0.066 / 0.262 / 1.049 MP — ×1.96 for the first ×4 in pixels, then ×6.3 for the
second. Something is superlinear between 512² and 1024².

## Consequences that hold regardless of the remaining unknown

- Every zenjxl `encode_ms` produced through a **plan-driven** sweep is
  contaminated. The canonical 2026-06-27 rollup is plan-driven.
- The **knob-grid** path's `encode_ms` is trustworthy on current `master`.
- The issue's advice stands: do not fit a picker, handicap table or cost model
  on the contaminated column.

## Next step for whoever picks this up

Dump the resolved per-cell `SweepVariant` for `vd-e5_libjxl_def` (fast) against
`vd-e7_libjxl_def` (slow) and diff them. The differing knob is the parasite.
`resolve_verified(CodecKind::Zenjxl, id, q, fp)` in `sweep/plan.rs` returns the
`PlannedConfig`; printing its `Debug` for those two ids is a few lines and needs
no fleet.

## Reproduce

```
cargo build --release -p zenmetrics-cli --no-default-features \
    --features cpu-metrics,sweep,jxl,png

# clean path
zenmetrics sweep --codec zenjxl --sources <dir> --q-grid 75 \
    --knob-grid '{"effort":[1,5,9]}' --metric zensim --jobs 1 --output k.tsv

# contaminated path
zenmetrics sweep --codec zenjxl --sources <dir> --q-grid 75 \
    --plan rd_core --plan-budget 12 --metric zensim --jobs 1 --output p.tsv

# attribution probe (encoder vs wrapper vs construction vs threads vs contention)
ZEN_PROBE_SIZES=1024 ZEN_PROBE_EFFORTS=1,5,7,9 ZEN_PROBE_THREADS=1 \
  cargo run --release -p zenmetrics-cli --features sweep,jxl,png \
    --example jxl_encode_ms_probe
```
