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
