# aom-rs still-diverging cells after the 2026-08-30 requeue — planes + messages (pointer)

Block storage: `/mnt/v/output/avifaom-2026-08-30/poison-planes-2026-08-30` (323 MB, 158 files — **not in git**).
Companion outcome table: `/mnt/v/output/avifaom-2026-08-30/requeue_round_2026-08-30_outcome.tsv`
(312 rows: job_id, status, error_class, image_path, w, h, pixels, q, knobs, size_class).

## What this is

The 41 cells of `avifaom-enc-20260830` that still refuse on the fixed executor
(`ghcr.io/imazen/zenfleet-worker:exec-zensim944hdr-d7982e9b`, zenmetrics master `d7982e9b`,
zenav1-aom `cb76cda9f9fe3df6b551fedc86bff568b89b649d` = KB-41 roots #22-#23 + KB-42 roots A-C).
Per cell: `<w>x<h>_cq<N>_s<S>.{y,u,v,json}` — the EXACT u8 planes the port and the C oracle
were both handed (`ZEN_AOMRS_DUMP_PLANES`), plus `{w,h,cw,ch,cq_level,speed,matrix}`.
`messages_full.tsv` carries the executor's own verdict per cell.

## What they are (measured, all 41)

- **41 / 41 are `aom-rs: PORT DIVERGED from the C oracle`** — never `DEFERRED`, never a crash.
  The arm refused to emit unverified bytes, exactly as designed; the ledger's `encoder_panic`
  class is the worker's generic non-zero-exit label, not a panic.
- Split: **31 screen-detected** (`screen_tools=true`, the KB-41 family) and
  **10 photo** (`screen_tools=false`) — the photo class is a *different* residual from the
  screen-tools arm the recent roots closed.
- Divergences are TINY: frame-OBU payload |delta| median **26 B**, max **1,620 B**, and
  **3 cells are byte-length EQUAL but content-different** (7046.scale256x256 cq25 among them —
  a 2,615 B payload, the smallest reproducer in the set).
- Sizes run 256x256 … 4000x3000; speeds 4 / 6 / 8; cq 9 … 62.

## Not to be retried blindly

These are deterministic port divergences. Re-queueing them re-runs and re-poisons them (done
twice now, both times with identical outcomes). They belong to the zenav1-aom port lane.

## Round 3 outcome (2026-08-30 21:28-21:48Z) — 28 of the 41 CLOSED

Re-queued on `ghcr.io/imazen/zenfleet-worker:exec-zensim944hdr-8662064f` (zenmetrics
`8662064f`, zenav1-aom `58204b295dd0` = KB-41 roots #24-#26, tree clean before AND
after the build; digest `sha256:72a0b143…`). `zenfleet-ctl requeue --classes
encoder_panic` pardoned all **59** still-failing cells (the 41 here + the 18 tiny
C3 ones); the three LAN boxes drained them in ONE pass each (r7900x done=18,
r3500 done=9, r5900xt done=5).

| | before | after |
|---|---|---|
| distinct done | 125,941 | **125,973** (+32) |
| still failing | 59 | **27** |

**The 27 that remain are exactly the two registered open classes**, cell for cell
(`benchmarks/avifaom_round3_2026-08-30_open.tsv`):

- **14 tiny** (< 0.1 MP: 12x `8468.scale59x128` + `5052.scale78x128` +
  `8020.scale115x128`) — PARITY C3's unported `av1_determine_sc_tools_with_encoding`.
  Four of the original 18 closed as a side effect of roots #24/#25.
- **13 large** — PARITY's **root #27**, the CNN convolve's own RTCD dispatch
  (`av1_cnn_convolve_no_maxpool_padding_valid`, port pinned to `_c`, encoder runs
  `_avx2`): the 3 screen `--cpu-used 6` cells (2320x3408, 2550x3300, 2590x3209) and
  all 10 photo cells (1728x3072 / 1730x3072 / 2415x3528 / 2765x4096 x2 /
  2896x4096 / 3150x4000 x2 / 4000x3000 x2).

Scored + harvested in the same round: `avifaom-sf-{gpu,cpu}-gap4-20260830`
(16/16 jobs each, 0 failed), write-back over all 8 score runs against
`pairs_aom_full4.parquet` -> `harvest-2026-08-30` (125,973 scores x 10 cols,
features 125,972 x 944, `miss_sha=0 miss_score=0`), views ->
`views-2026-08-30` (train 102,405 / eval8 23,567). The pre-round harvest is
preserved as `harvest-2026-08-30-pre-round3.bak`.
