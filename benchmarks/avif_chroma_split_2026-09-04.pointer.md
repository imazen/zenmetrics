# Pointer — `avif-chroma-2026-09-04` (chroma-split arm harvest + re-cut outputs)

Record: [`avif_chroma_split_2026-09-04.md`](avif_chroma_split_2026-09-04.md) §7 ·
Registration: same file, §1-6 · Companion doc updated with the unconfounded
re-cut: [`avif_backend_selection_2026-09-03.md`](avif_backend_selection_2026-09-03.md)

**Not in git** (>30 KB rule, and these are generated tables). Canonical copies:

| location | path |
|---|---|
| local | `/mnt/v/output/avif-chroma-2026-09-04/` — includes a `scoreblobs/` working cache (263 MB, 1,544 files; NOT mirrored below — it is a local sync of the score run's own primary blob store, see Inputs) |
| LAN store | `s3://zentrain/analysis/avif-chroma-2026-09-04/` — **VERIFIED 9 objects / 645,663 B** |
| Tower | `/mnt/user/coefficient/output/avif-chroma-2026-09-04/` — **VERIFIED 9 files, sha256 spot-matched** (written over SSH — the `/mnt/tower` NFS mount has a history of `StaleNetworkFileHandle` on the dev box) |

**Era pin:** `2026-09-04` — zenavif `6dfdf6f` / zenrav1e `7ad86844`; fleet image
`ghcr.io/imazen/zenfleet-worker:exec-avifchroma-f15bb3a5` (digest
`sha256:dad86ae95b75a36438b4e3a968c65c7fef513ec381c5a7f05930a56ffeedb3dd`).
Producer commit `f15bb3a5`. **Never join across eras** — the doc's §3 era
control (2,880/2,880 byte-identical) is what licenses reading `br444` against
`brsdr`'s Stage-B-era scored cells; nothing else here crosses eras.

**Score coverage: 5,760 / 5,760 cell rows = 100.00 %** (`br420` 2,880/2,880,
`br444` 2,880/2,880 — both `zenfleet-ctl gap` = `0 of 2880`). The 5,760 cell
rows join to **5,179 distinct `encode_sha`** (581 exact-duplicate bitstreams,
expected AVIF byte-convergence, not missing data); every one of the 5,179
distinct bitstreams carries both `ssim2` and `zensim` features (`rows UNSCORED
(no ssim2): 0`, `rows missing bytes: 0`). `avifdoe_chroma_harvest.sh` is
idempotent; this is a clean full-coverage read, re-run once at this fill level
with an identical result (no partial-fill artifact).

**Verdict: CHROMA. k₁₁ = 11 of 11, k = 16 of 16** — the maximum-strength
reading against the pre-registered rule (k₁₁ ≥ 9 → CHROMA). Independently
verified by hand against `tables/reach_per_image.tsv`, not just the harvest
script's own printed summary line. Full writeup: doc §7.2.

**Producers (both canonical owners, extended not forked):**

- `scripts/jobsys/avifdoe_chroma_harvest.sh` (new; one-command pairs→sizes→
  harvest→re-cut driver)
- `scripts/jobsys/avifdoe_chroma_analyze.py` (new; era/scorer controls, the
  three BD-rate axes, the reach-90 verdict — imports its stats, computes no
  new ones)
- `scripts/jobsys/avifdoe_harvest.py` (canonical harvester, reused unmodified)

**Inputs:**

- `br420_ledger` — `s3://zentrain/jobs/avifdoe-rav-br420-20260904/ledger/`
- `br444_ledger` — `s3://zentrain/jobs/avifdoe-rav-br444-20260904/ledger/`
- `score_blobs` — `s3://zentrain/jobs/avifdoe-cs-sf-cpu-20260904/blobs/` (1,544
  NDJSON blobs, 35,908 lines; this run's OWN primary store — the local
  `scoreblobs/` cache above is a sync of this, not a second canonical copy)
- `refs` — `s3://codec-corpus/avif-doe-1024-2026-09-01/`
- `stagea_scored` — `/mnt/v/output/zensim-avifdoe/doe_scored_2026-09-02.parquet`
- `crop_manifest` — `/mnt/v/output/avif-doe-1024-2026-09-01/crop_manifest_2026-09-01.tsv`
- `brsdr_pairs` — `~/tmp/avifdoe_pairs/avifdoe-rav-brsdr-20260903_pairs.tsv`
  (the Stage-B-era 4:4:4 comparator the era control licenses reading against)

## Files (9, 645,663 B)

| file | bytes | sha256 |
|---|--:|---|
| `chroma_scored_2026-09-04.parquet` | 261,404 | `e5bc9468a842988f3ef899326b884aae8c62112607f6dd0c33f59ecf885d7e49` |
| `sizes.tsv` | 367,988 | `7bfadd6cecd86c52eda97816c70f7c3f89c425b1f2d142a65aa40a0607008adf` |
| `tables/axis_backend_true.tsv` | 3,356 | `5bf9fc9561cf02c48de528483da640b781f35b456769fc23d19d7184a992946a` |
| `tables/axis_chroma_true.tsv` | 3,374 | `d333b726ed284d8a93b5798a1a129e1b17dc76801dc54bd364fb9fe6e59ec7a1` |
| `tables/axis_published_confounded.tsv` | 3,368 | `ebf01523d8bec1c595cdd607ee47ad74e6373493b6230b234151fa401d0adb6c` |
| `tables/era_control.tsv` | 109 | `bdc52c8d485161de38b7136e3dae8a265dd4b6c1ed474c9f13802b2e4081be7d` |
| `tables/notes.json` | 3,877 | `e01cd780cb824f0b83fd678a1f2fbf1c579285ac0032708630899444e6336230` |
| `tables/reach_per_image.tsv` | 2,107 | `dbe3cc3d415406a3df54b8dd4c171868ce184be3a8035007c53c1ed118d6cc2e` |
| `tables/scorer_control.tsv` | 80 | `3c8348643cd0f8210c6fceea67587d21a24ef4341409ff63c128be2b728707a2` |

### What each table is

- **`chroma_scored_2026-09-04.parquet`** — the harvested tidy table, 5,760 rows
  × 18 cols, one row per cell (both arms), `metrics_present` carried (5,760/5,760
  `ssim2|zensim_features`, 0 null)
- **`sizes.tsv`** — `encode_sha -> bytes`, from both arms' blob listings
- **`tables/era_control.tsv`** — `br444` vs `brsdr`'s 9-q subset, byte identity
  fraction (2,880/2,880 = 1.000000, ERA INERT)
- **`tables/scorer_control.tsv`** — max\|Δ ssim2\| on shared bitstreams scored
  by two different score runs (0.0, SCORER INERT)
- **`tables/axis_chroma_true.tsv`** — per-image BD-rate, `br420` vs `br444`
  (backend held, chroma varies) — THE clean chroma-bytes axis
- **`tables/axis_backend_true.tsv`** — per-image BD-rate, `br420` vs `svt420`
  (chroma held at 4:2:0, backend varies) — THE clean backend-bytes axis
- **`tables/axis_published_confounded.tsv`** — per-image BD-rate, `br444` vs
  `svt420` — the ORIGINAL joint (backend×chroma) comparison, kept for
  before/after continuity, not a new finding
- **`tables/reach_per_image.tsv`** — THE decision table: per-image max
  achieved ssim2 and reach-90 boolean for all three arms (`br420`/`br444`/
  `svt420`), all 32 images
- **`tables/notes.json`** — machine-readable verdict + both axis stat blocks +
  the in-wave one-ladder cross-check, everything in the doc's §7 read from
  this file (nothing hand-recomputed)
