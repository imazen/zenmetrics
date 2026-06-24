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

| codec | cells scored | 6 metrics | 372 feat | primary location | complete? |
|---|---|---|---|---|---|
| jpeg | 16,830 | ✅ | ✅ | R2 `…/zenjpeg/sidecars/` | ✅ **DONE** |
| png  | 4,446  | ✅ | ✅ | R2 `…/zenpng/sidecars/`  | ✅ **DONE** |
| avif | 29,889 / 38,472 | ✅ | ✅ | `/mnt/v` unified + R2 blobs | ⚠️ **78%** (gap paused) |
| webp | 128,699 / 306,162 | ✅ | ✅ | `/mnt/v` unified + R2 blobs | ⚠️ **42%** (paused) |
| hdr/zenjxl | 7,980 | **cvvdp only (1/6)** | ❌ | R2 `…-hdr/zenjxl/sidecars/` | ⚠️ **1/6 metrics** |

**Fully-scored cells (all 6 metrics + 372 features): 179,864** = jpeg 16,830 + png 4,446 + avif 29,889 +
webp 128,699. Plus hdr/zenjxl 7,980 cells with cvvdp only. This is the materialized output of today's spend.

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
