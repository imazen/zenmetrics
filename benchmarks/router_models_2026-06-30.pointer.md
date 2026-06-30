# Cross-codec router models — 2026-06-30

The 3 baked ZNPR router models for `zenpicker::MetaPicker::route` (lossy / lossless / auto-gate),
plus the feature sidecar they were trained from.

## SHIPPED: i8, in-crate, wired (the final state)

The f32 bakes (below, ~90 KB each) were repacked to **i8 via `zenpredict repack --dtype i8`**
(the built-in calibrated quant) → **~27 KB each (<30 KB)** with **MEASURED 0 accuracy cost** on
the real `.bin` (ground-truth via `zenpicker/examples/score_router.rs`):

| router | f32 acc | i8 acc | i8 bytes |
|---|---|---|---|
| lossy | 75.72% | **75.72%** (Δ 0) | 27101 |
| lossless | 88.37% | **88.37%** (Δ 0) | 26965 |
| gate | 98.06% | **98.07%** (Δ +0.01) | 26759 |

The i8 `.bin` are **committed into `zenpicker/benchmarks/zenpicker_router_{lossy,lossless,gate}_v0.1.bin`**
and loaded by **`zenpicker::MetaPicker::default_routers()`** (std; `include_bytes!` + 16-aligned
`AlignedModel<N>` + `OnceLock`). Verified end-to-end: `route_demo` + the `default_routers_load_and_route`
unit test. The f32 bakes below are the pre-quant originals (kept in block storage for provenance).

## Block storage

`/mnt/v/output/router-features-2026-06-30/`

| file | bytes | sha256 (16) | shape | held-out test |
|---|---|---|---|---|
| `router_lossy.bin` | 91205 | `9e251047b1e3b285` | in 102 (101 feat + target) → 6 fam | 75.7% family-acc, 3.92% mean / 0% median RD-overhead |
| `router_lossless.bin` | 90685 | `75e41ef07179c441` | in 101 → 6 fam (webp/jxl; png padded) | 88.4% family-acc |
| `router_gate.bin` | 90111 | `c640b787b482a693` | in 102 → 2 (lossy\|lossless) | 98.1% acc |
| `zenanalyze_features.parquet` | 2261870 | `b6c265e22abad6fb` | 4497 variants × 101 qualified feats + dims | sidecar (source-only) |
| `router_{lossy,lossless,gate}.bakereq.json` | — | — | BakeRequestJson inputs | re-bake with `zenpredict-bake <json> <bin>` |

R2 / Tower mirror: TODO (mirror to `zentrain` + Tower before any cleanup).

## Provenance

- **Corpus**: `/mnt/v/output/clean-picker-corpus-2026-06-26/` — 4497 renditions (the canonical
  picker variants; `o_<id>.png.scale<W>x<H>.png`).
- **Sidecar**: re-extracted with current zenanalyze via
  `examples/extract_features_for_picker --sizes native --features api` →
  **101 qualified `name@hex8` source-only features**, experimental-complete
  (`xyb444_color_loss` / `xyb_bquarter_chroma_loss` / `chroma_subsample_dct_loss` present +
  populated), **0.0000% NaN** (current tiny-cell handling, no imputer). Keyed `variant_name`.
- **Why re-extract**: the canonical 2026-06-27 parquets carried 97 named source feats **+ 372
  positional `feat_N` that are zensim (ref,distorted) PAIR features** — encode-dependent, NOT
  available at routing time. The earlier GBDT routers trained on all 469 (a leak). Clean
  retrain on 101 source-only feats: 76.2% (GBDT) vs the old leaky 75.5% — strict win. See
  [[verify-premises-before-cascading]] / [[cross-codec-meta-router-3way]].
- **Labels**: min-`encoded_bytes` family from the canonical per-codec RD curves
  (`/mnt/v/output/canonical-picker-2026-06-27/`); lossless filtered to `score_zensim>=99.999`.
- **Trainer / bake**: `scripts/picker/train_router_clean.py` (validation),
  `scripts/picker/bake_routers.py` (train MLP → BakeRequestJson, classifier→ZNPR via negate
  logits + pad to 6 CodecFamily; numpy self-verify argmin==sklearn match=1.0000), git
  `b76201fb`+. **MLP shape** `(128,64)`; f32 ⇒ ~90 KB each.
- **End-to-end verified**: `zenpicker/examples/route_demo.rs` loads the 3 `.bin`, routes a real
  variant: zq60→Webp, zq85→Jxl, zq97→Avif (lossy), Lossless→Jxl(lossless).

## Error anatomy + the zq97 dead-zone (`scripts/picker/router_error_anatomy.py`)

Lossy router on the held-out test set (shipped i8, both views of each misroute):
- All cells: mean byte-overhead **4.09%** (median 0 — most picks ARE the oracle), but a heavy
  tail (p99 64%). Misroutes: median **8.76% bytes** / **4.75 zq points** (iso-bytes quality
  deficit), tail to 807% / 154 zq (the unreachable-target cells). Acc degrades with quality:
  80% (zq45–70) → 73% (70–90) → 68% (90+).
- Dominant confusions are the RD-close modern cluster **AVIF↔JXL↔WebP** (cheap, 5–13% bytes):
  avif→jxl (n=654, 9.1%/8.7zq), jxl→avif (526, 12.8%/3.2zq), webp→avif (518, 12.5%/6.2zq).

**The zq97 "dead zone" + why AVIF beats JXL there — ROOT CAUSE is a sweep gap, not the codec.**
Swept lossy quality ceilings: avif median **96.4** (reaches ≥95 on 100%), jpeg 94 (20%), webp 90
(0%), jxl 90 (0%). **`zenjxl_lossy` was swept only to q≤90 (max zensim 94.7, ZERO rows ≥zq95)**
while jpeg/webp/avif went to q95 — so above ~zq94 jxl-lossy has NO data and the router can only
pick AVIF. jxl-lossy (VarDCT) is genuinely near-lossless-capable; this is a **coverage artifact**.
Also: only 13% of variants reach zq97 with ANY lossy; hitting zq97 costs 2.73× the zq90 bytes;
the gate correctly sends ~97% of zq97 to lossless (best-lossy/best-lossless median 1.24). And the
lossy router was trained on zq45–90, so zq97 is **extrapolation** for it. Demo `o_1016`: lossy
ceilings jpeg 92.8 / webp 84.7 / jxl 87.8 / avif 96.3 — none reaches 97; the gate mis-said LOSSY
and the lossy router (extrapolating) picked AVIF (highest ceiling) = best-effort toward an
unreachable target. **FIX: re-sweep `zenjxl_lossy` to q95** (match the others) before trusting any
AVIF-vs-JXL pick above ~zq90; route zq>~94 via the gate (lossless); don't query the lossy router
above its zq90 training range. See [[jxl-lossy-swept-only-to-q90]].
