# zensim-720 feature backfill — Hetzner CPU fleet (runbook)

Add the **v2 720-feature vector** (`feat_0..feat_719` = v1-372 `with-iw` ++ v2-348)
to the canonical picker training corpora, on cheap Hetzner CPU boxes. The GPU zensim
kernel is disabled (v2 is CPU-only); the executor emits 720 features, **no score**.

**Status 2026-07-20:** machinery + idle autoshutdown + the canonical declare path are
BUILT and VALIDATED end-to-end on a real Hetzner pair (see "Validation" below). The
full multi-corpus run is a deliberate scale step (hundreds of GB of declare
streaming + a multi-hour fleet) — fire it per the recipe below.

## What emits 720

- Image `ghcr.io/imazen/zenfleet-worker:exec` (CPU, `ZEN_FLEET_IMAGE_CPU` in
  `scripts/jobsys/fleet.env`). Built from `--features sweep,png,jpeg,webp,avif,jxl,cpu-metrics`.
- `metric=zensim` or `zensim-gpu` → a `feature` row `{regime:"v2-ab", features:[720]}`
  (no score row). Gated on `cpu-metrics`, so a CPU-only box emits 720 (commit 1ed8204d).
- Rebuild: `PUSH=1 bash scripts/jobsys/build_executor_image.sh` after
  `cargo build --release -p zenmetrics-cli --no-default-features --features sweep,png,jpeg,webp,avif,jxl,cpu-metrics`.

## Corpus map (the corpora to backfill)

Canonical picker datasets — **7 codecs × {train,validate,test}**:
`s3://zentrain/canonical/2026-07-01-zensimA/<codec>/{train,validate,test}.parquet`
(codecs: `zenjpeg_lossy zenwebp_lossy zenwebp_lossless zenavif_lossy zenjxl_lossy
zenjxl_lossless zenpng_lossless`). They have `score_zensim` + the old 372 features but
NOT `feat_372..719` — that's what this backfill adds.

Per split the lean declare input is the sibling **`pairs.<split>.parquet`** under
`s3://zentrain/canonical/2026-06-27/<codec>/` — columns `ref_path, dist_path,
image_path, codec, q, knob_tuple_json, dist_member, dist_tar`. Each split spans ~8
`box-N.tar` variant tars (3–8 GB each) under the source run, e.g. zenjpeg_lossy:
`s3://zentrain/jxl-lossy/runs/mandfix2-zenjpeg-1782584881/variants/box-{0..7}.tar`
(the `dist_tar` column names them; get the source run from each dir's `_MANIFEST.json`
→ `source_run_prefix`). **Refs:** `s3://codec-corpus/clean-picker-corpus-2026-06-26/`
(the `ref_path` basename; CORPUS_PREFIX = `clean-picker-corpus-2026-06-26`).

`build_scorefile_from_pairs.py` auto-detects both the `dist_tar/dist_member` schema
(the `pairs.*` files) and the canonical `variant_tar_r2_url/variant_r2_url` schema
(the feature-table `train.parquet`) — commit cc8a2449.

## Recipe — per (codec, split, tar)

Declare is per-tar (the tool filters `pairs` to one tar and streams it once to build
the sha→offset index). Set the zensim-only metric list so CPU boxes emit just 720:

```bash
cd ~/work/zen/zenmetrics
set -a; . ~/.config/cloudflare/r2-credentials; set +a
RUN=zensim720-zenjpeg_lossy-train-box0     # one run-id per corpus (share across its tars)
# declare (ZEN_SCOREFILE_METRICS is honored if you patch it; else edit METRICS in the tool to ["zensim-gpu"])
python3 scripts/jobsys/build_scorefile_from_pairs.py \
  ~/tmp/pairs.train.parquet \
  s3://zentrain/jxl-lossy/runs/mandfix2-zenjpeg-1782584881/variants/box-0.tar \
  $RUN
# launch a PAIR (self-destruct ON by default; small backstop while validating)
ZEN_TAR_OVERRIDE=s3://zentrain/jxl-lossy/runs/mandfix2-zenjpeg-1782584881/variants/box-0.tar \
ZEN_CORPUS_PREFIX_OVERRIDE=clean-picker-corpus-2026-06-26 \
ZEN_MAX_RUNTIME_MIN=240 TYPES="cpx52 cpx42 cx43" LOCATIONS="fsn1 nbg1 hel1" RESUME=1 \
  bash scripts/jobsys/hetzner_scorefile_launch.sh $RUN 2
bash scripts/jobsys/fleet status $RUN          # or: fleet watch $RUN
# teardown is automatic on drain; manual: bash scripts/jobsys/teardown_fleet.sh $RUN
```

Outputs land at `s3://codec-corpus/jobs/$RUN/blobs/` as JSONL `feature` rows
(`encode_sha`, `regime:"v2-ab"`, `features:[720]`); rejoin to the corpus by
`encode_sha` (= sha256 of the variant bytes, in `variant_index.tsv`).

## Idle AUTOSHUTDOWN (built here — Hetzner had none)

`hetzner_scorefile_launch.sh` boxes SELF-DESTRUCT via the hcloud API (commit
18b7fc10): the worker runs foreground `--restart no`, and when the gap drains
(`ZEN_IDLE_PASSES` consecutive no-work passes) the entrypoint exits clean and the
cloud-init calls `destroy_self` (metadata instance-id → `DELETE`). A hard backstop
(`ZEN_MAX_RUNTIME_MIN`, default 720) kills a hung box regardless. Token is HOST-only.
`ZEN_SELF_DESTRUCT=0` opts out (then only `teardown_fleet.sh` stops billing).
**Note:** `zenfleet-core::idle` / `fleet watch --destroy` do NOT stop Hetzner boxes
(vast-only / dashboard-alarm-only) — the cloud-init self-destruct is the mechanism.

## "Optimal" notes

- **Decode/extract reuse (verified 2026-07-20):** the ScoreFile path decodes the ref
  ONCE per source and each variant ONCE (byte-range from R2, no re-encode). The v1
  ref XYB pyramid is precomputed ONCE per chunk and reused across the chunk's variants
  (`precompute_ref_ctx` + `extract_features_regime_with_ctx`, bit-identical; **~15%
  off** a 12-variant chunk). **Remaining win:** the v2-348 block still rebuilds the ref
  pyramid per variant (`zensim::compute_v2_features` has no precomputed-ref API), and
  v1+v2 don't share a pyramid — that ~2/3 of extraction needs a **combined-720-with-ref
  shared-pyramid API in the zensim crate** (proposed to the feature-v2 session; ~40%+).
- zensim parallelizes internally and plateaus ~8 concurrent — a box with ≥8 vCPU
  (`cpx52`=16, `cx43`=8) saturates it; more, smaller boxes beat fewer huge ones on
  cost only up to the per-box plateau. A *pair* is the minimum; scale box count to
  the deadline.
- **Declare bandwidth is the bottleneck:** each box-N.tar is 3–8 GB and there's no
  prebuilt index, so `build_scorefile_from_pairs.py` streams the whole tar. Run the
  DECLARE on a well-connected box (a Hetzner box in the same region), not a home dev
  box — streaming ~50 GB/codec over a home link is the slow part, not the scoring.
- Per-box rate: MEASURE on the first real tar (do not extrapolate the 96×96 smoke's
  ~120 cells/s — real renditions are larger/slower). ~1.9× the v1-372 cell cost.

## Validation (2026-07-20, run `zensim720-hzsmoke-20260720-013548`)

Tiny self-contained corpus (2 refs + 6 jpg variants). A **pair** of `cpx22/fsn1`
boxes launched with the self-destruct cloud-init: both pulled `:exec`, one scored the
2 chunks (6 variants), **6/6 `feature` rows `regime=v2-ab len=720`** landed in
`blobs/`, and **both boxes SELF-DESTRUCTED on drain** (0 remaining, no manual
teardown; the 3 persistent dev boxes untouched). Cost ≈ €0.001. This exercises the
identical ScoreFile path (R2 ref fetch + tar byte-range + decode + 720 extract +
lease/ledger + autoshutdown) the canonical run uses.

## UPDATE 2026-07-20 — variants are in TARS, not encodes/ (byte-range path)

**Discovery:** the canonical `pairs.*.parquet` `dist_path` points at `encodes/<file>`, but those
individual objects **only exist for `zenjpeg_lossy`** (pre-extracted). The other **6 codecs' `encodes/`
prefix is EMPTY (404)** — their bytes live in the sweep run's per-box tars
`s3://zentrain/jxl-lossy/runs/<sweep>/variants/box-N.tar` (`dist_tar` column). ~188GB, ~53 tars, 4.26M
variants. Sweeps: zenavif `mandfix4-zenavif-1782593621`(8), zenjxl_lossy `jxl-lossy-vardct-1782609551`(24),
zenwebp lossy+lossless SHARE `mandfix2-zenwebp-1782584881`(9), zenjxl_lossless `jxl-modular-1782596759`(10),
zenpng `mandfix2-zenpng-1782584881`(2).

**Path chosen (byte-range / "seekable", per user):** do NOT re-upload 4.26M objects. Instead:

1. **Index on a Hetzner box** — `scripts/jobsys/index_tars_driver.sh` (loops all codecs, ~3 concurrent)
   calls `scripts/jobsys/index_tar_byterange.py <tar> <codec> <run> zentrain`: streams the tar ONCE
   (`s5cmd cat | tarfile r|`), and from `m.offset_data`+`m.size` writes a 4-col index
   `dist_member\toffset\tsize\tdist_member` + a ScoreFile manifest (ref derived from the filename:
   `(.+?)_[0-9a-f]{16}_` → `+.png`) to `jobs/bf-<tag>-tN/`. One run per tar.
2. **Score via the byte-range mode** — the existing launcher with `ZEN_TAR_OVERRIDE=<the tar>` and
   **NO `ZEN_ENCODES_PREFIX`** (that would force direct-object). Refs shared across all codecs at
   `zentrain/refs/clean-picker-corpus-2026-06-26/`. Oversubscribe `ZEN_CORE_OVERSUBSCRIBE=3`.
3. **jobexec fix (required):** `variant_index()` now caches the index on disk keyed by URI (download
   once/box, atomic rename) — without it byte-range re-downloads the index per cell (the 30MB×N
   bottleneck direct-object dodged). Keep per-tar indexes ~5MB so the per-process parse stays cheap.

**New knob:** `ZEN_CORE_OVERSUBSCRIBE` (float ≥1, default 1) multiplies the `can_admit` core budget for
I/O-bound feature backfills; RAM stays bounded by `can_admit`. Measured ~29 var/s/box at 8-wide →
~22-wide oversubscribed.

**Overnight autonomy:** `scripts/jobsys/backfill_overnight_manager.py` (run on the dev box, nohup) —
discovers declared `bf-z*-t*` runs, keeps ≤CAP hzsf-bf boxes (~$2/hr; CAP counts the draining zenjpeg
fleet too), launches ONE oversubscribed byte-range box per undone run without a live box, self-terminates
when all runs complete (or after HOURS), then tears down the index box. Boxes self-destruct on drain, so
the fleet is self-bounding. Run: `CAP=56 HOURS=7 nohup python3 scripts/jobsys/backfill_overnight_manager.py &`.

**Validated 2026-07-20:** zenpng + zenavif byte-range blobs = `kind=feature`, `{720}` features,
`regime=v2-ab`, no score. zenjpeg_lossy runs via the original direct-object path (`bf-zjl2`).

## COST (2026-07-20) — use cx43, price-cap, NOT cpx42

Hetzner hourly (gross, verified via `hcloud server-type describe`): **cpx42 EUR0.1314/hr** vs
**cx43 EUR0.0296/hr for the SAME 8 vCPU / 16GB** (4.4× cheaper) — cx33 EUR0.016 (4 vCPU), cx23 EUR0.0104
(2 vCPU). **Always launch cx43** (fall back cx33/cx23); never cpx42 unless cx-series is capacity-out.
`backfill_overnight_manager.py` caps on projected fleet PRICE (`MAX_EUR`, default 1.6 EUR/hr scoring +
~0.13 index box < $2/hr), NOT a box count — box prices vary 8×, so a count cap silently blew the budget
(23 cpx42 = ~$3.3/hr when the user asked for $2). Run: `MAX_EUR=1.6 TYPES="cx43 cx33 cx23" HOURS=8
nohup python3 scripts/jobsys/backfill_overnight_manager.py &`.
