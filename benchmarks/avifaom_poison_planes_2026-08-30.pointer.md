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
