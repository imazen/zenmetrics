# Datagen GPU-Metric Scoring — Data Inventory (2026-06-24)

Canonical record of the **(reference, encoded-variant) → perceptual-metric** data gathered for
zensim / picker / calibration training. **Goal:** for every encoded rendition in the 2026-06-23
datagen corpus, six GPU perceptual metrics **plus** the 372-dim with-iw zensim feature sidecar,
stored as joinable parquet keyed on the cell identity.

This file exists so the spend behind the data is recoverable. **Cost context:** the vast.ai fleet
that produced the avif/webp ScoreFile blobs drained the account balance to **−$0.25** (peak **92
boxes @ $17.84/hr**). The blobs + materialized parquets below are that spend's output — **do not
re-score what is already here**; resume only the incomplete tails (§6).

## 1. The six metrics + features

| column(s) | metric | backend |
|---|---|---|
| `butteraugli_max_gpu`, `butteraugli_pnorm3_gpu` | Butteraugli (max + p-norm 3) | butteraugli-gpu (cubecl) |
| `cvvdp_imazen_v0_0_1` | ColorVideoVDP (JOD) | cvvdp-gpu |
| `dssim_gpu` | DSSIM | dssim-gpu |
| `iwssim_gpu` | IW-SSIM | iwssim-gpu |
| `ssim2_gpu` | SSIMULACRA2 | ssim2-gpu |
| `zensim_score` | zensim | zensim-gpu |
| `feat_0 … feat_371` | 372-dim zensim feature vector (`with-iw` regime) | zensim-gpu |

**Cell identity** (join key across every sidecar): `(image_path, codec, q, knob_tuple_json)`.
The ScoreFile write-back additionally carries `encode_sha` = `sha256(variant bytes)`.

## 2. Status matrix

Source corpus = **1,482 renditions** (`train_renditions_2026-06-14`). There are TWO independent gaps —
**encode coverage** (how many source images were encoded at all) and **scoring coverage** (how many of the
encoded variants were metric-scored):

| codec | images / 1482 | cells scored | cells/img | 6 metrics | 372 feat | status |
|---|---|---|---|---|---|---|
| jpeg | **117 (8%)**  | 16,830 / 16,830  | 144 | ✅ | ✅ | scored-complete, but **ENCODE covers only 8% of the corpus** |
| png  | 1,482 (100%)  | 4,446 / 4,446    | 3   | ✅ | ✅ | complete (png grid ≈ 3 cells/image) |
| avif | **229 (15%)** | 29,889 / 38,472  | 168 | ✅ | ✅ | **encode 15% of images**; 78% of those scored |
| webp | 1,398 (94%)   | 128,699 / 306,162| 219 | ✅ | ✅ | encode 94%; 42% of those scored |
| hdr/zenjxl | subset | 7,980          | — | cvvdp only (1/6) | ❌ | 1/6 metrics |

**Fully-scored cells today (6 metrics + 372 feat): 179,864.** BUT because of the encode gap, jpeg & avif are
a small slice of the corpus, NOT corpus-complete:
- **Encode gap (CPU side):** jpeg encoded only **117/1482** images, avif only **229/1482**. To build a complete
  cross-codec corpus they must be RE-ENCODED on the remaining ~1,365 / ~1,253 images (CPU `score`/encode job —
  cheaper than GPU scoring). webp (1,398) / png (1,482) image coverage is near-full.
- **Scoring gap (GPU side):** of what IS encoded, webp 42% / avif 78% / jpeg+png 100% scored.

### Size coverage (CRITICAL for MLP training — log-spaced size density)

The 1,482 source corpus spans **32px → 11,648px** and DOES contain the large-size expansion (175 images
≥512px; **88 images >2048px**). But the scored data is thumbnail-skewed AND the big images are excluded
from the primary scored codec:

| max-dim bucket | full corpus | webp encoded | jpeg | avif |
|---|---|---|---|---|
| ≤128 | 1,260 (85%) | 1,260 | 98 | 197 |
| 129–512 | 47 | 47 | 2 | 2 |
| 513–2048 | 87 | 87 | 6 | 9 |
| **>2048 (big)** | **88** | **4** | 11 | 21 |

**webp is missing 84 of the 88 >2048px images — and webp's 84 un-encoded images ARE exactly those 84 big
ones** (likely excluded as too slow to encode). So the large-size expansion lives in the corpus but is
concentrated in the barely-scored jpeg/avif subsets; the most-scored codec (webp, 128k cells) is ~90%
≤128px. **The MLP large-size need is NOT met by what is currently scored** — to serve it, the >2048px
images must be encoded + scored across codecs (webp especially), not left in the jpeg/avif tails.

Two scoring methods produced this:
- **SPLIT** (jpeg/png/hdr-cvvdp): CPU encode-once + GPU `score-pairs` per metric → **per-metric**
  parquet in the codec's `sidecars/` on R2.
- **ScoreFile** (avif/webp, this session): per-chunk jobexec decodes the ref once, byte-range-fetches
  each pre-encoded variant from `variants.tar` (no re-encode), scores all 6 metrics + 372 features →
  **JSONL blobs** in `jobs/<run>/blobs/`, joined to **combined** `scores.parquet` + `features.parquet`
  by `scripts/jobsys/writeback_scores.py`.

## 3. R2 layout (bucket `codec-corpus`)

Encode corpus prefix **DGP** = `picker-sweep-2026-06-22/datagen-2026-06-23` (HDR: `…-hdr`).

```
<DGP>/ref/                         reference renditions (source PNGs)
<DGP>/<codec>/variants.tar         all encoded variants (tar; members = <stem>_<hash>_<codec>[_q..].<ext>)
<DGP>/<codec>/pairs.tsv            image_path codec q knob_tuple_json ref_path dist_path   (CANONICAL ref→variant map)
<DGP>/<codec>/omni.tsv             per-cell encode metadata (NB: webp encoded_filename is truncated — use pairs.tsv)
<DGP>/<codec>/sidecars/            SPLIT output: <metric>.parquet + zensim_features.parquet + DONE   (jpeg/png/hdr)
jobs/<run>/blobs/                  ScoreFile output: per-chunk JSONL (metric + feature rows)          (avif/webp)
jobs/<run>/variant_index.tsv       sha \t offset \t size  (byte ranges into variants.tar)
jobs/<run>/manifest.json[.gz]      per-chunk DesiredJobs
```

ScoreFile run ids:
- avif main: `jobs/datagen-zenavif-sf-20260624/` — 3,835 blobs, 29,720 unique encode_shas
- avif gap:  `jobs/datagen-zenavif-gap-sf-20260624/` — 15 blobs (paused almost immediately; the 8,583-cell
  gap the omni-built manifest had undercounted is **NOT** filled — see §6)
- webp:      `jobs/datagen-zenwebp-sf-20260624/` — 5,528 / 26,562 chunks

## 4. Schemas

**ScoreFile blob (JSONL, one object per line):**
```json
{"kind":"metric","image_path":"1552.scale128x93.png","codec":"zenavif","encode_sha":"7a71…","metric":"butteraugli-gpu","score":12.02,"scores":{"butteraugli_max_gpu":12.02,"butteraugli_pnorm3_gpu":5.04}}
{"kind":"feature","image_path":"1552.scale128x93.png","codec":"zenavif","encode_sha":"7a71…","regime":"with-iw","zensim_score":13.62,"features":[…372 floats…]}
```
A full 12-variant chunk = 72 metric rows (12×6) + 12 feature rows.

**Write-back parquet (avif/webp), `/mnt/v/zen/zensim-training/2026-06-24/unified/<codec>/`:**
- `scores.parquet`  : `image_path, q, knob_tuple_json, encode_sha,` + the 7 metric columns above
- `features.parquet`: `image_path, q, knob_tuple_json, encode_sha, zensim_score, feat_0..feat_371`

**SPLIT sidecar parquet (jpeg/png/hdr), `<DGP>/<codec>/sidecars/`:**
- `<metric>.parquet`        : `image_path, codec, q, knob_tuple_json,` + that metric's columns
- `zensim_features.parquet` : `image_path, codec, q, knob_tuple_json, zensim_score, feat_0..feat_371`

Both schemas join on `(image_path, q, knob_tuple_json)`. (SPLIT lacks `encode_sha`; the ScoreFile
write-back has it.)

## 5. How to consume / join recipe

```python
import pyarrow.parquet as pq
sc = pq.read_table("/mnt/v/zen/zensim-training/2026-06-24/unified/zenavif/scores.parquet").to_pandas()
ft = pq.read_table("/mnt/v/zen/zensim-training/2026-06-24/unified/zenavif/features.parquet").to_pandas()
df = sc.merge(ft, on=["image_path","q","knob_tuple_json","encode_sha"])   # cell -> 6 metrics + 372 feats
```
ScoreFile `encode_sha` → `(image_path, q, knob)` is recoverable from the codec's `pairs.tsv`
(`basename(dist_path)` → variant name) + `sha256(variant bytes)`; `variant_index.tsv` maps
`sha → tar offset`. `writeback_scores.py` does this join (accepts comma-sep run ids to merge runs).

## 6. What's incomplete + how to resume (needs vast credit)

| gap | size | resume |
|---|---|---|
| webp tail | 177,463 cells (58%) | top up vast → `bash scripts/jobsys/gpu_scorefile_launch.sh datagen-zenwebp-sf-20260624 zenwebp <N>` (skips done chunks) |
| avif gap | 8,583 cells (manifest built, 15 done) | `bash scripts/jobsys/gpu_scorefile_launch.sh datagen-zenavif-gap-sf-20260624 zenavif <N>` then re-write-back both runs |
| hdr 5 metrics | 7,980 cells × {butteraugli,dssim,iwssim,ssim2,zensim} + features | needs nits-domain decode (`sweep/hdr.rs decode_encoded_to_nits` + `HdrScorer`) in jobexec, OR a `sweep --hdr` score run |

All ScoreFile resume is **non-wasteful**: completed chunks have blobs in `jobs/<run>/blobs/`; the
`ZEN_SKIP_SHAS_FILE` gap mode (`build_scorefile_manifest.py`) excludes already-scored shas.

## 7. Tower mirror

Canonical training parquets mirrored to `/mnt/tower/output/zensim-training/2026-06-24/unified/<codec>/`
(see §8 for sha verification). Tower is the source-of-truth backup; `/mnt/v` is the working cache.

## 8. Provenance

- **Encode corpus:** `picker-sweep-2026-06-22/datagen-2026-06-23` — renditions of
  `/mnt/v/output/imazen-26-features/train_renditions_2026-06-14/*.png` (plan-mode cells; identity
  `(image_path, codec, q, knob_tuple_json)`, knob = `{"cell","fp","plan"}`).
- **Codec commits:** see `~/work/zen/DATA_PROVENANCE.md` for the per-codec HEAD SHAs used for the
  2026-06-23 backfill (RD curves shift between codec revs — do not mix-and-match across backfills).
- **Scoring code:** zenmetrics `jobexec` ScoreFile path + `scripts/jobsys/writeback_scores.py`
  (commit recorded in `git log` for this doc).
- **cvvdp column** `cvvdp_imazen_v0_0_1` is the imazen cvvdp-gpu output (NOT pycvvdp v0.5.4 — that
  variant uses column `cvvdp_pycvvdp_v054`; none present in this corpus yet).

## COMPLETE CPU corpus (2026-06-24-cpu) — full coverage, no GPU

A second, **CPU-only** scoring pass that closes the coverage gaps the GPU set had (jpeg 8% / avif 15% image
coverage, no large sizes). Run entirely on **Hetzner CPU** — no GPU, no vast credit (the vast account was at
−$0.25 the whole time), held **<$1/hr**. The unlock: `ssim2` + `zensim` are the two metrics with CPU
implementations, so the encode + score + features all run on cheap CPU boxes.

**Metrics:** `score_ssim2` + `score_zensim` + the 372-dim `with-iw` zensim feature vector (`feat_0..feat_371`).
The 4 GPU-only metrics (cvvdp / iwssim / dssim / butteraugli) are NOT in this pass — deferred to a GPU run
when vast credit returns.

**Coverage:** full 1,482-image corpus + a 4.2–16MP big-image tier (57 imgs/codec). **477,288 cells.**

| codec | cells | plan | coverage |
|---|---|---|---|
| jpeg | 206,928 | rd_core (~136/img) | full 1482 + 57 big |
| avif | 244,440 | rd_core (~168/img) | full + 57 big |
| webp | 21,555 | rd_core (~14/img) | full + big — **SPARSE**; original was modes_full (~219/img), denser re-run is a pending user decision |
| png | 4,365 | rd_core (3/img) | lossless, full + big |
| **total** | **477,288** | | |

**Layout:** `/mnt/v/zen/zensim-training/2026-06-24-cpu/unified/<codec>/{omni.tsv,features.parquet}` + Tower
mirror `/mnt/tower/output/zensim-training/2026-06-24-cpu/<codec>/`. Encoded variants persisted on R2 under
`picker-sweep-2026-06-22/runs/dgcpu-*/variants/`. `_MANIFEST.json` in the unified dir.

**Deferred:** ~27 renditions >16MP (monster sizes up to 102MP — outside the MLP range, too slow here) + the
4 GPU-only metrics.

**Lessons:** the consolidator had a per-run `box-N` filename collision that silently dropped ~80k jpeg cells —
caught by verification, fixed (per-run subdirs). webp ran rd_core (sparse) vs its original modes_full.
avif-big oversubscribed ccx53 cores (nested rayon, loadavg ~32); png-big's metric on 11–16MP images is
single-core-slow. Tooling: `scripts/sweep/hetzner_cpu_sweep.sh` (+`--feature-output`, MINPX/MAXPX window),
`scripts/jobsys/consolidate_cpu_sweep.py`.
