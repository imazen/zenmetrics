# jxlav1-trial-20261028 — imazen-26-JXL AV1 backend trial sweep

Status: encode wave running (declared 2026-10-28 ~02:55Z). Pointer doc — artifacts
live in the LAN object store, not in this repo.

## Goal

Refresh codec-choice data for the coefficient pareto viewer: measure the two
pure-Rust AV1-still backends on a representative imazen-26 JXL subset, score
with ssim2 + butteraugli (max-norm and 3-norm), extract zensim rev7 features,
feed zenanalyze heuristics. Encodes are preserved (content-addressed blobs).

## Corpus (train bucket only)

- Source: `/mnt/tower/output/imazen-26-render-v4-2026-10-15/jxl-srgb/`
  (render-v4 JXL corpus; manifest `manifests/jxl-srgb.tsv`, 2,112 images).
- Selector: `scripts/jobsys/jxlav1_subset.py` — deterministic, folder-stratified,
  pixel-count-spread picks; `bucket == "train"`, `<= 16 MP` cap.
- Result: 96 sources (152 MB), sha256-verified against the manifest at copy time.
- Staged: `s3://codec-corpus/imazen-26-jxl-srgb-train-v4-20261028/` (flat
  basenames; workers resolve via `ZEN_CORPUS_PREFIX`) + sibling
  `…/imazen-26-jxl-srgb-train-v4-20261028.subset.tsv` (name/folder/dims/mp/form/
  colour/sha256/manifest_path).
- Selection stats: `train` = 1,067 candidates ≤16 MP of 1,067 train rows;
  all 21 folders represented (2..24 picks proportional to folder share).

## Cells / jobset

- Emitter: `scripts/jobsys/jxlav1_cells.py` (avifsvt_cells.py shape).
- Grid: `q ∈ {1, 5..70 step5, 72..100 step2}` (30 pts) × `speed ∈ {4,6,8}`
  × `backend ∈ {svt-rs, aom-rs}` = 17,280 cells.
- Knob tuple: `{"backend":"<b>","speed":S}`; codec `zenavif`. q is a
  per-backend label, not a cross-backend quality scale (zenavif convention;
  aom-rs q→cq mediator is in encode.rs §aom-rs).
- Jobset: `s3://zentrain/jobs/jxlav1-enc-20261028/manifest.json` (17,280
  declared jobs). Manifest built by `zenfleet-ctl declare-encodes` from
  `cells-main-{svt,aom}.jsonl` in the run dir.

## Executor image

- `ghcr.io/imazen/zenfleet-worker:exec-zenav1-jxl-0539c685`
- Built in the lock snapshot (`~/tmp/zenmetrics-lock-snap`), musl static-pie:
  `zenmetrics-cli` `--features sweep,png,jpeg,webp,avif,jxl,cpu-metrics,
  hdr-gainmap,avif-svt,avif-aom`; `zenfleet-worker` musl.
- zenmetrics tree = commit 0539c685 (working-copy empty at build time);
  zensim pin 96c3c9bb (stamped ZENSIM_BUILD_COMMIT); zenav1-aom pin 5d779f86;
  zenav1-svt pin 2255a342 (lock-resolved git rev).
- aom-sys-ref: `x86_64-unknown-linux-musl` has no libaom cmake toolchain
  mapping at this pin — the build panics in `cmake_toolchain_for`. Worked
  around by seeding `upstream/build-x86_64-unknown-linux-musl/` with the
  host `upstream/build/` gnu objects + matching `.aom-oracle-sha` stamp
  (`03087864… -ffp-contract=off clean`); the sanctioned path (glibc oracle
  objects + musl fopen64 shim, fca352dd).
- ghcr push denied (token scope) → image shipped via `docker save | ssh docker
  load` to all four workers directly.

## Fleet (permitted hosts only: i134, r5600g, r3x00)

| host | cores used | notes |
|---|---|---|
| i134-zen | 16 | direct ssh |
| r5600g-zen | 12 | via dev relay (r5900xt keys not authorized) |
| r3500-zen | 6 | via dev relay, --memory 10g |
| r3800x-zen | 8 (cpuset 0-7) | via dev relay; box already runs another lane's zen-score containers |

All launched `ZEN_LONG_LIVED=1 ZEN_CORE_OVERSUBSCRIBE=1 ZEN_REQUIRE_SNAPSHOT=0
ZEN_CORPUS_BUCKET=codec-corpus ZEN_CORPUS_PREFIX=… ZEN_FLEET_BUCKET=zentrain`
via `lan_score_launch.sh` (r* via dev's copy).

## Smoke first (done)

`jobs/jxlav1-smoke-20261028`: 240 cells (4 sources × 1 speed × 30q × 2
backends) — `done=240 failed=0` on i134 in ~2 min. Verified `blobs/` +
`ledger/*.parquet` rows (`status=done`, `output_sha`) in the store.

Local `sweep` smoke (same encode path, this box, q60 only): aom-rs byte-verify
passed on all 4 sources incl. 1024×1024 (the Aug-era 720:1080 divergence band
appears closed at pin 5d779f86 — the fleet wave confirms over the full grid;
divergent cells would fail loudly and stay recoverable via requeue).

q60 spot-check (matched-rate, bytes±0.1%): svt-rs ssim2 73.1 vs aom-rs 32.6 on
the 6.9 MP photo; line-plot flips (aom 69.7 vs svt 62.1). Signal is real and
worth the sweep.

## Open follow-ups

- [ ] score wave: ScoreFile {ssim2, butteraugli} over `blobs/` (preserve
  encodes; score after decode-once-per-source batching).
- [ ] feature wave: zensim rev7 — needs zensim pin bump (96c3c9bb predates
  Rev7) + `feature-rev` executor + `ZENSIM_FORMULA_REV=7` launch env.
- [ ] aom-rs coverage over the 720:1080 min-dim band (42/96 sources) — ledger
  will show whether any cells refuse.
- [ ] harvest ledgers → parquet rollup → coefficient viewer input.
- [ ] worker drain: `docker rm -f zen-score-enc` on each box when done.
