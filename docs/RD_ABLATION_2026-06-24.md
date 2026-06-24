# RD-full search-space reduction — Pareto-ablation + knob-stability (2026-06-24)

Goal: shrink the encoder rd-full search space (fewer cells/image) and find the pixel-size threshold above
which the ideal knobs stop changing — **before** spending credits on denser jpeg/jxl/avif corpora. Done
entirely on **existing dense rd-full data (no fleet, no credits)**. Tool: `scripts/analysis/pareto_ablation.py`.

> Aside — the "~1M-cell dense corpus" is **jpeg**, not webp: `unified_v15r_zenjpeg` (1,785,696 rows). webp's
> densest old parquet is only 1,000 rows; today's CPU run gave webp 21,555. So webp isn't the dense reference.

## Data analyzed (`/mnt/v/zen/zensim-training/2026-05-07/unified/`)
| codec | parquet | imgs | cells/img | knob combos | metric | size range |
|---|---|---|---|---|---|---|
| jpeg | unified_v15r | 979 | 1824 | 96 | ssim2 | ≤1MP |
| jxl | unified_v12 | 200 | 160 | 32 | zensim | — |
| avif | unified_v12 | 200 | 20 | 4 (qm×speed only — **sparse**) | zensim | — |

(zensim polarity verified higher=better: corr(q,zensim)=0.78, corr(ssim2,zensim)=0.94.)

## (A) Combo-level ablation — knobs NEVER on the (bytes, quality) Pareto front
| codec | knobs | never-on-front (ablatable) | reduced set at ~0 RD loss |
|---|---|---|---|
| jpeg | 96 | **61** | 35 (top-48 = 0.00 loss everywhere; top-32 = max 0.96 ssim2) |
| jxl | 32 | **18** | 14 |
| avif | 4 | 2 | 2 |

**jpeg rd-full can drop ~63% of knob combos with zero RD loss → the denser run costs ~1/3.**

## (B) Per-axis defaults (bake as encoder default + ablate from the search)
- **jpeg `effort=2` is strictly dead — 0.000 RD loss ablating it → always `effort=1`** (effort=2 is pure
  wasted compute).
- **jpeg `progressive`**: dominant in the web range, BUT baseline is genuinely Pareto-optimal at HIGH q
  (median q 80; only 8% of baseline wins are at q≤35). Always-progressive costs mean 0.54 / max 10.69 ssim2
  — all in the archival range. → **web-focused picker (q5–60): always progressive; archival: keep baseline.**
- **jxl `patches=False` is dead → always `patches=True`.**
- **avif `qm=False` is dead → always `qm=True`** (quantization matrices always on).

## (C) Knob-stability vs pixel count (jpeg)
Front-knob set is **stable from ~0.15MP to 1MP** (top-16 Jaccard 1.00 across buckets); slight variation only
below 0.05MP (thumbnails). **→ the ideal knobs stop varying by ~0.15MP (~400px).** Large images do NOT need
a dense knob sweep — score them at the stable optimal knob set only. (Data caps at 1MP; no mechanism would
make the knobs re-diverge above it — but a confirming run at 2–8MP is cheap if wanted.)

## Implication for densifying jpeg/jxl/avif (the actual goal)
1. **Reduced knob set**: jpeg 96→35, jxl 32→14, avif drop qm=False — ~1/3 the cells/image.
2. **Bake the per-axis defaults** (effort=1, patches=True, qm=True; web-progressive) — ablate from the grid.
3. **"Larger" coverage is cheap**: dense knob sweep only needs ≤~0.15MP to fix the knobs; large images get
   the stable optimal knobs at a few q points (the size signal the MLP conditions on), not a full sweep.
4. **avif needs a denser rd-full first**: the analyzed sweep had only qm×speed. A small dense avif sweep
   (add tune/tiles/sharpness/etc. axes) on ~30 clustered representative images would extend the ablation
   before the big run — cheap (tokens-scale), and it's the right next step for avif specifically.
