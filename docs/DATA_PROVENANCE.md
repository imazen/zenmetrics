# Backfill data provenance

Records where metric backfill sidecars live in R2, what codec
versions produced them, and how to read them.

Future sessions (zensim retraining, zenpredict baking, picker
training) should consult this doc to know which dataset is which
and avoid mixing data from different codec generations — since
codecs like `jxl-encoder` are actively developed and RD curves
shift across commits, conflating runs gives garbage models.

---

## Success path summary (2026-05-19)

Two production runs are now complete end-to-end:

- `cvvdp-v15rc-2026-05-18` — **2568 omni sidecars + 2568 zensim
  300-feature parquets** (513,570 zenjpeg cells).
- `omni-multi-codec-2026-05-19` — **365 omni sidecars + 365 zensim
  300-feature parquets** (mixed zenwebp + zenavif + zenjxl + v13_zenjpeg).

The pipeline that delivered them lives in
`zenmetrics/crates/vastai-fleet` (Rust) and
`zenmetrics/scripts/sweep/` (launchers + onstarts). The
end-to-end recipe is in `scripts/sweep/README.md` under
**"The proven end-to-end pipeline (2026-05-19)"**.

Key facts a downstream consumer (zensim retrain, zenpredict bake)
needs to know:

1. **All cells have 6 GPU metric scores** — `cvvdp_imazen_v0_0_1`,
   `zensim_score_gpu`, `ssim2_score_gpu`, `butteraugli_max_gpu`,
   `butteraugli_pnorm3_gpu`, `dssim_score_gpu`, `iwssim_score_gpu`.
2. **All cells have a 300-feature zensim vector** in the sibling
   `zensim_features/<chunk>.parquet`. Join the two parquets on
   `(image_path, codec, q, knob_tuple_json)`.
3. **Encoded variants are preserved on R2** at
   `s3://zentrain/<run>/encoded/<chunk>/<encoded_filename>` so
   future metric backfills can re-use them without re-encoding.
   The `encoded_r2_uri` column on the omni parquet has the full
   URI per row.
4. **The 372-feature historical CSVs** at
   `/mnt/v/zen/zensim-training/2026-05-15-full-features/` come
   from a different zensim build (~2026-05-15) and have
   different `f_<i>` semantics from the 300-feature parquets
   produced here. Do NOT join them by feature index — they're
   feature-index-incompatible across zensim versions.

---

## Zensim 372-feature corpora (2026-05-15)

Local-only (not on R2). Used for zensim V_20a + V_20b training.

**Path:** `/mnt/v/zen/zensim-training/2026-05-15-full-features/`

```
aic3_features_372col_2026-05-15.{csv,parquet}    — 1.6 MB parquet
cid22_features_372col_2026-05-15.{csv,parquet}   — 10.7 MB parquet  (4292 pairs)
kadid_features_372col_2026-05-15.{csv,parquet}   — 24.6 MB parquet
konjnd_features_372col_2026-05-15.{csv,parquet}  — 2.6 MB parquet   (1008 pairs aligned)
konjnd_full_features_372col_2026-05-15.csv        — 76 104 pairs metric-anchored superset
tid_features_372col_2026-05-15.{csv,parquet}     — 7.5 MB parquet
_MANIFEST.md                                      — full schema + corpus policy
```

**Schema (374 columns):** `ref_basename`, `human_score`, `f0`..`f371`.

The 372 zensim features break down as (from the manifest):

- `basic` features: 13/ch × 3ch × 4 scales = 156
- `peaks` (max + p95): 6/ch × 3ch × 4 scales = 72
- `masked` (gated `extended_features=true`): 6/ch × 3ch × 4 scales = 72
- `IW pool` (gated `compute_iw_features=true`): 6/ch × 3ch × 4 scales = 72
- **Total = 372 features per pair**

**Corpus training policy** (load-bearing per zensim/CLAUDE.md):

| Corpus      | `human_score` units                       | Training use |
|---|---|---|
| KADID-10k   | DMOS 1–5 (lower better)                   | OK to train (human MOS) |
| TID2013     | MOS 0–9 (higher better)                   | OK to train (human MOS) |
| CID22       | MCOS / 100 (0–1)                          | **VALIDATION ONLY** — sacred |
| KonJND-1k   | per-source mean PJND threshold            | OK as auxiliary |
| KonJND-full | gpu_ssimulacra2 / 100                     | OK (metric-anchored, not human) |
| AIC-3 CTC   | score.jnd (signed JND units)              | **VALIDATION ONLY** |

**Important:** the 372-feature columns came from a SPECIFIC zensim
build at extraction time (2026-05-15). zensim is actively developed
and the feature definitions may shift between commits. If retraining
against these CSVs, **do NOT mix with features re-extracted from a
newer zensim** — feature column indices are not stable across
zensim versions.

The zen-metrics omni backfills (described below) ran with
`zensim-gpu` (which emits only `score_zensim_gpu`, no extended
features). To get a fresh 372-feature vector for the cvvdp-v15rc
or omni-multi-codec corpora, a separate extraction pass is needed
that runs CPU `zensim` with `--feature-output`. The Rust worker
supports this via the `feature_output: Option<PathBuf>` field on
`SweepConfig`, but the deployed v22/v23 binaries don't currently
plumb it through `inline.rs`. That's a one-line change + a fresh
run if the existing 372-feature corpora aren't sufficient.

---

## Active backfills

### `cvvdp-v15rc-2026-05-18` — v15rc_zenjpeg full omni sweep

| Field | Value |
|---|---|
| Status | **Complete** — 2568 / 2568 omni sidecars + 2568 / 2568 zensim feature parquets (2026-05-19) |
| R2 omni prefix | `s3://zentrain/cvvdp-v15rc-2026-05-18/omni/` |
| R2 features prefix | `s3://zentrain/cvvdp-v15rc-2026-05-18/zensim_features/` |
| R2 encoded prefix | `s3://zentrain/cvvdp-v15rc-2026-05-18/encoded/<chunk_id>/` |
| Sidecar count | 2568 parquets, one per chunk, 200 rows each |
| Total rows | 513,570 (matches input parquet row count) |
| Input parquet | `s3://zentrain/unified-2026-05-07/unified_v15rc_zenjpeg.parquet` |
| Sources prefix | `s3://zentrain/sweep-v15rc-2026-05-07/sources/` |
| Codec | `zenjpeg` only |
| Metrics | `zensim-gpu`, `ssim2-gpu`, `butteraugli-max-gpu`, `butteraugli-pnorm3-gpu`, `cvvdp-imazen-v0.0.1`, `dssim-gpu`, `iwssim-gpu` + zensim 300-feat vector |
| Worker image | `ghcr.io/imazen/zen-metrics-sweep:v22` initial → `v23` for reencode → `v24` for feature backfill |
| Approx burn | ~$3-4 across all three passes (initial omni + 346-chunk reencode + feature-backfill) |

**Note:** the initial v22 omni run left 346 of the 2568 chunks
with `encoded_filename: null` in their omni sidecars (no encoded
variants saved). Those 346 were re-encoded 2026-05-19 against v23
image and their feature parquets backfilled afterward. The
omni-sidecar count stays 2568 throughout — the reencode passes
overwrote existing omni sidecars in place rather than creating
duplicates.

### `omni-multi-codec-2026-05-19` — v12 webp+avif+jxl + v13 jpeg omni sweep

| Field | Value |
|---|---|
| Status | **Complete** — 365 / 365 omni sidecars + 365 / 365 zensim feature parquets (2026-05-19) |
| R2 omni prefix | `s3://zentrain/omni-multi-codec-2026-05-19/omni/` |
| R2 features prefix | `s3://zentrain/omni-multi-codec-2026-05-19/zensim_features/` |
| R2 encoded prefix | `s3://zentrain/omni-multi-codec-2026-05-19/encoded/<chunk_id>/` |
| Codecs | `v12_zenwebp` (5), `v12_zenavif` (20), `v12_zenjxl` (160), `v13_zenjpeg` (180) |
| Input parquets | `s3://zentrain/unified-2026-05-07/unified_v12_zen{webp,avif,jxl}.parquet` + `unified_v13_zenjpeg.parquet` |
| Sources prefix | `s3://zentrain/sweep-v15-2026-05-06/sources/` (different from v15rc!) |
| Metrics | Same 6 GPU metrics as the v15rc run |
| Worker image | `ghcr.io/imazen/zen-metrics-sweep:v22` (most) + `v23` (final stranded chunks) |
| Approx burn | ~$1.50 (8-12 box fleet @ $0.06/hr × ~2 hr) |

---

## Codec commit hashes pinned during each backfill

The unified Rust worker at `zenmetrics/crates/vastai-fleet` links
codec crates as **path deps to local sibling worktrees**. These
checkouts are sometimes on experimental branches; the table below
records the SHA each backfill actually saw at link time.

### `cvvdp-v15rc-2026-05-18` (v22 image, built 2026-05-19)

| Crate | Path dep | HEAD commit | Notes |
|---|---|---|---|
| `zenpng` | `~/work/zen/zenpng` | `4ec04ca` | `fix/security-audit-2026-05-06` |
| `zenjpeg` | `~/work/zen/zenjpeg/zenjpeg` | `bdc7f4c` | `fix/security-audit-2026-05-06` |
| `zenwebp` | `~/work/zen/zenwebp` | `60fd977` | `fix/security-audit-2026-05-06` |
| `zenavif` | **crates.io v0.1.7** | — | Local `docs/speed6-tx-rdo-opt-in` lacked `__expert`. Stayed on crates.io 0.1.7. |
| `zenjxl` | git rev | `9ac0cd5` | imazen/zenjxl rev pin in workspace [patch.crates-io] |
| `jxl-encoder` | git rev | `6b8eefc1` | imazen/jxl-encoder rev pin in workspace [patch.crates-io] |

### `omni-multi-codec-2026-05-19` (v22 image initially, v23 for stranded)

Same codec set as v15rc above for the v22 batch. Most of the multi-
codec sidecars came from v22.

The handful of late v23 chunks (the 1 stranded zenavif plus any
re-runs) link a slightly different codec stack — all codecs are
now path deps to local sibling worktrees:

| Crate | Path dep | HEAD commit | Notes |
|---|---|---|---|
| `zenpng` | `~/work/zen/zenpng` | `4ec04ca` | unchanged |
| `zenjpeg` | `~/work/zen/zenjpeg/zenjpeg` | `bdc7f4c` | unchanged |
| `zenwebp` | `~/work/zen/zenwebp` | `60fd977` | unchanged |
| `zenavif` | `~/work/zen/zenavif--main` | `37a529e` | New worktree of zenavif `main` (HEAD as of 2026-05-19). Restored `pub mod expert` + `__expert`. |
| `zenjxl` | `~/work/zen/zenjxl--main` | `9ac0cd5` | New worktree of zenjxl `main`. Same commit as the v22 git rev pin. |
| `jxl-encoder` | `~/work/zen/jxl-encoder/jxl-encoder` | `7de1db87` | Local checkout HEAD. **NEWER than the v22 binary's git rev pin (`6b8eefc1`)** — includes W44-68 perf series (DCT32 suppression on screenshot content + earlier W44-66/67 ledger work). |

**Important caveat for zensim retraining:** the v22 multi-codec
sidecars and the v23 multi-codec sidecars were produced by
DIFFERENT `jxl-encoder` commits. v22's jxl-encoder is `6b8eefc1`
(pre-W44-66 ledger); v23's is `7de1db87` (post-W44-68). RD curves
for `v12_zenjxl-*` chunks differ between v22-produced rows and
v23-produced rows. If you're training a metric that's sensitive
to RD shape, **stratify by chunk_id ranges**:

- v22 produced approximately the first 160 jxl chunks before the
  rebuild; v23 finished a stranded handful afterward. To stratify
  precisely, look at the `mtime` of each sidecar in R2:
  `s5cmd ls s3://zentrain/omni-multi-codec-2026-05-19/omni/v12_zenjxl-*.parquet`
  — anything uploaded after 2026-05-19T10:24Z came from a v23 box.

---

## Sidecar schema (all backfills)

Every sidecar in `omni/` has this Arrow schema (zstd-compressed
parquet, ~20-40 KB each):

```text
image_path                  : Utf8       — full local path the worker saw at encode time
codec                       : Utf8       — `zenjpeg` / `zenwebp` / `zenavif` / `zenjxl`
q                           : Int64      — quality knob value, codec-specific scale
knob_tuple_json             : Utf8       — JSON object of the full knob point
encoded_bytes               : Int64
encode_ms                   : Float64
encoded_filename            : Utf8       — basename of the saved encoded variant
                                          (under encoded/<chunk>/)
decode_ms                   : Float64
score_zensim_gpu            : Float64
score_ssim2_gpu             : Float64
score_butteraugli_max_gpu   : Float64
score_butteraugli_pnorm3_gpu: Float64
score_cvvdp_imazen_v0_0_1   : Float64    — JOD scale 0..10, 10 = imperceptible
score_dssim_gpu             : Float64    — distance, 0 = identical
score_iwssim_gpu            : Float64    — [0, 1], 1 = identical
chunk_id                    : Utf8
run_id                      : Utf8
encoded_r2_uri              : Utf8       — full s3:// path to the encoded variant
```

**Encoded variants** (the raw codec-output bytes) land at
`s3://zentrain/<run>/encoded/<chunk>/<filename>` and can be re-decoded
for future metric runs without re-encoding.

---

## Reading the data (Python)

```python
import pyarrow.parquet as pq
import s3fs

fs = s3fs.S3FileSystem(
    endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    key=R2_ACCESS_KEY_ID,
    secret=R2_SECRET_ACCESS_KEY,
)

# Single chunk
t = pq.read_table(
    "zentrain/cvvdp-v15rc-2026-05-18/omni/v15rc_zenjpeg-0000.parquet",
    filesystem=fs,
)

# Whole run (slower — 2568 files for v15rc, 365 for multi-codec)
import pyarrow.dataset as ds
dataset = ds.dataset(
    "zentrain/cvvdp-v15rc-2026-05-18/omni/",
    format="parquet",
    filesystem=fs,
)
t = dataset.to_table(columns=["image_path", "codec", "q", "knob_tuple_json",
                              "score_cvvdp_imazen_v0_0_1", "score_zensim_gpu"])
```

---

## Where the unified Rust worker source lives

- Worker binary: `zenmetrics/crates/vastai-fleet/` (cargo crate)
- CLI: `vastai-fleet worker --run-id <id> --chunks-r2 <uri>`
- Docker image: `ghcr.io/imazen/zen-metrics-sweep:v23` (post 2026-05-19)
- Dockerfile: `zenmetrics/Dockerfile.sweep.v23`
- Launchers: `zenmetrics/scripts/sweep/launch_{single_instance,backfill}.sh`
- Onstart: `zenmetrics/scripts/sweep/onstart_unified.sh`
- Util monitor: `zenmetrics/scripts/sweep/fleet_util_snapshot.sh`

The worker links `zen-metrics-cli` as a library so `run_sweep`
runs in-process (one cubecl init per worker process, not per
group). This is what gave the 2.7x throughput vs the bash
predecessor — see commit `24313a0` and `4e760b6` for the Phase A
+ Phase B writeups.

---

## Future runs — keeping this doc accurate

When you start a new backfill, append a new section here with:

1. Run ID + R2 prefixes (inputs, outputs, sources)
2. Codec set + their HEAD commits at link time (use the snippet below)
3. Metrics enabled
4. Worker image tag
5. Approx burn

Codec HEAD snippet (run from any zen-metrics binary's build host):

```bash
for c in zenpng zenwebp zenjpeg zenavif--main zenjxl--main jxl-encoder; do
    cd ~/work/zen/$c 2>/dev/null && \
        echo "$c $(git rev-parse HEAD) $(git branch --show-current)"
done
```
