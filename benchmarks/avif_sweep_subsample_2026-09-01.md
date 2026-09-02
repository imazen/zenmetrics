# AVIF svt-rs/aom-rs sweep over a curated imazen26 subsample (2026-09-01)

**User directive (2026-09-01):** get zenavif encodes running on the new pure-Rust
`zenav1-svt` / `zenav1-aom` backends over a **carefully representative subsample**
of imazen26 training files, to build predictive models for **tuning, size (bytes),
and speed (encode time)**. Explicitly supersedes the 2026-07-13 "avif datagen
halted" note for *this* directed SDR sweep (that halt was scoped to avif-**HDR**
datagen per `feedback_zenavif_in_flux_no_datagen.md`; this is SDR). Compute: home
LAN fleet + local only — **no paid cloud**.

## 0. What already existed (read first — avoids duplicating the 2026-08-30 wave)

A prior wave (`benchmarks/balance_campaign_2026-08-28.md` "FRESH AVIF WAVE — SVT
BACKEND" / "AOM ARM", zensim repo) already ran both backends over **ALL 1,455
train-side renditions** of `train_renditions_2026-06-14` at **3 sampled speeds**
`{4,6,8}` × the same 30-point q grid, for zenfleet job-system exercise + zensim
944-feature training data. It is fully drained (svt: 130,590 cells harvested;
aom: 125,688/126,360 done, 312 poison = understood port divergences owned by a
separate KB-41 lane in `zenav1-aom`). It did **not**: (a) curate a representative
subsample (it used the exhaustive full set), (b) sweep every speed/cpu-used
preset (only 3 samples), (c) attach host/thread/contention metadata for a speed
model, or (d) target size/encode_ms as first-class modeling outputs. This sweep
is a **complement**, not a duplicate: same executor/backends, disjoint purpose
(dense speed axis on a curated subsample, for a size+speed+tuning model), new
run names (`avifsub-{svt,aom}-enc-20260901`, vs. the prior `avifsvt-enc-20260830`
/ `avifaom-enc-20260830`).

Reused as-is: the executor's `backend`/`speed` knob dispatch
(`zenmetrics-cli/src/sweep/encode.rs::encode_avif`, features `avif-svt`/
`avif-aom`), the cell emitter `scripts/jobsys/avifsvt_cells.py` (unmodified —
its 30-point q-grid and per-backend `--speeds` list already fit this task), the
zenfleet job system (`zenfleet-ctl declare-encodes` / `zenfleet-worker`), and the
imazen26 k-means representative-selection methodology
(`scripts/imazen26_recluster_even.py`, zenanalyze
`benchmarks/imazen26_cluster_ablation_2026-06-14.py`). One extension was made to
the clustering script (below); nothing else was rewritten.

## 1. Subsample: K-means, train-only origins, whole native images

**Population + split.** `zenmetrics/scripts/imazen26_recluster_even.py`, run
against `/mnt/v/output/imazen-26-features/imazen26_features_2026-06-13.parquet`
(the original `feat_`-column population used by the canonical ablation study;
sha256 `aab7c7d2f06687a069283f0e43b9fc31087f216a3d6120b12231d0ef619af774`).
`size_class == "native"` only. Train/holdout split is the canonical imazen-26
least-significant-digit rule (`zensim/docs/DATA_SPLITS.md` §2a,
`origin_split.py::split_of`): last digit of the numeric origin id ∈
`{0,2,4,6,8}` → train. The script filters to **even ids only** (`--parity 0`) —
odd ids (validate `{1,3,5}` + test `{7,9}`) never enter the clustering
population, so contamination is structural, not a post-hoc filter. Verified:
`ODD-id reps (MUST be 0 for parity=0): 0`.

**Extension made (small, additive):** the script clustered over EVERY
`crop_label` (whole image `full` + 10 sub-crops `c50/c25 × {center,tl,tr,bl,br}`)
— correct for its original picker-source-selection purpose, but a crop is a
*virtual* region (no standalone file exists for it), so a pick would need a
crop step before it's encode-ready. Added `--crop-label` (optional, defaults to
the old unfiltered behavior — zero change for existing callers) to restrict the
population to `crop_label == "full"` (whole native images only), so every pick's
`image_path` is directly an existing, ready-to-encode PNG. Also threaded
`width`/`height` through to the output TSV (needed for the corpus naming
convention below). Diff is additive; no existing row-selection logic changed.

**Method** (unchanged from the ablation study): 84 zenanalyze content features
(13 geometry features — `pixel_count`, `*_dim`, `aspect*`, `block_misalignment*`,
`log_padded*`, `bitmap_bytes`, `channel_count` — excluded; size is a
densification axis applied *to* picks, not a selection axis), z-scored over the
selected population, `sklearn.cluster.KMeans(n_clusters=K, n_init=10,
random_state=0)`, one representative per cluster = the **centroid-nearest
member** (Euclidean, standardized space).

**Population size:** 1,082 train-origin whole native images (of 1,978 total
even+odd whole-images; 82 content features after z-score).

**K = 32** (order 24–48 per the sweep-lane brief; sized down from the ablation
study's K=500 picker-training precedent because *this* population is whole
images only — no crop-level redundancy to absorb — and the downstream cost per
representative is far higher here: dense speed × dense q × 2 backends per
source, not one feature-extraction pass).

**Command:**
```
uv run --with scikit-learn --with pyarrow --with numpy python3 scripts/imazen26_recluster_even.py \
  --parquet /mnt/v/output/imazen-26-features/imazen26_features_2026-06-13.parquet \
  --select-k 32 --parity 0 --crop-label full --seed 0 \
  --out-manifest /mnt/v/output/avifsvt-subsample-2026-09-01/reps_K32_full_even_2026-09-01.tsv
```
Output: `selected 32 reps (32 distinct images) → …`; `ODD-id reps: 0`;
`crop_label distribution: {'full': 32}`; **4 singleton clusters kept** (genuine
outliers, not merged away — per the k-means-for-representativeness discipline).

**Cluster table** (cluster_id, cluster_size = population count that cluster
represents, content_class, origin dims): see
`/mnt/v/output/avifsvt-subsample-2026-09-01/avif_subsample_picks_2026-09-01.tsv`
(committed pointer below; not the raw TSV — see §6). Content-class distribution
of the 32 picks (k-means rebalances by feature diversity, not raw corpus share —
same effect the ablation study documented at K=500):

| content_class | picks |
|---|--:|
| 7000-lilith-plots | 6 |
| 8100-lilith-web-screenshots | 5 |
| 9226-lilith-ai-products | 4 |
| 9094-lilith-ai-illustrations | 3 |
| 6000-lilith-scans-public-patents | 3 |
| 1400-lilith-nature | 3 |
| 9000-lilith-ai-clipart | 2 |
| 6600-ia-scans-manuscript-illustrations | 2 |
| 3000-art-institute-of-chicago-photos | 1 |
| 1600-lilith-food | 1 |
| 1200-lilith-interiors | 1 |
| 1000-lilith-photos-general | 1 |

12 distinct content classes covered by 32 picks (of ~18 total classes in the
corpus) — plots/screenshots (feature-diverse, low raw share) are up-weighted;
product photography (homogeneous white-background shots, high raw share) is
down-weighted, matching the K=500 ablation's documented rebalancing.

**Excluded origins:** every odd-id origin (validate `{1,3,5}` + test `{7,9}` —
1,247 of 2,157 whole native images) is excluded **by construction** (the
`--parity 0` filter), not by a post-hoc drop — this is the contamination guard:
imazen26 is a first-class eval axis (`G-IM26`) reading the canonical bigcodec
TEST views, so training-adjacent artifacts (this sweep's encodes) must never be
built from val/test-origin content. Sub-crop units (`c50`/`c25` × 5 positions,
~91% of the native population) are excluded by the new `--crop-label full`
filter — they're virtual regions, not separate files; a future wave could add
them with a real crop step.

**Size handling:** 2 of 32 picks (`6602`, `6604` — Haeckel manuscript-plate
scans, 36.1 MP / 35.1 MP) exceed the executor's established practical cap (16
MP — a *throughput*, not parity, decision inherited from the 2026-08-30 wave's
KB-41 finding). Rather than drop them (losing the only 2 picks of a
feature-diverse content class), they're downscaled to fit ≤16 MP via **Lanczos**
(Pillow, even-dimension rounding to avoid odd-dim chroma-subsampling edge
cases): 4865×7207 → 3286×4868, 4961×7277 → 3302×4844. The other 30 picks are
native-resolution symlinks, byte-identical to source. Recorded per-pick in the
`treatment` column (`native_symlink` | `downscaled_lanczos`).

**Corpus naming:** each pick is named `<origin_id>.scale<W>x<H>.png` (symlink or
downscaled copy) to satisfy `avifsvt_cells.py`'s existing `\.scale(\d+)x(\d+)\.`
filename regex **with zero changes to that script** — the alternative (teaching
the emitter to read PNG dimensions itself) would touch code that 130k+ cells of
the prior wave already depend on being unchanged.

## 2. Grid (registered before launch)

| axis | values | source |
|---|---|---|
| sources | 32 (K-means picks, §1) | this doc |
| codec | `zenavif` | fixed |
| backend | `svt-rs`, `aom-rs` | both, per user directive |
| **svt-rs speed** | **1..=10 (all 10)** | zenavif's own dial; SVT preset 0..=13 linear internally |
| **aom-rs speed** | **0..=9 (all 10)** | raw `--cpu-used`; **NOT the same range as svt-rs** — caught by the local smoke test (see §3), which is exactly why the smoke-before-launch gate exists |
| q | 30 points: `1, 5,10,…,70 (step 5), 72,74,…,100 (step 2)` | `avifsvt_cells.py::q_grid()`, unmodified — already dense at both ends per the sweep-axes discipline |
| size | **native only (no extra downscale axis) — SCOPED OUT** | see below |
| threads/cell | 1 (single-threaded encode; concurrency comes from N parallel cells, not intra-cell threading) | executor design |

**Total cells: 32 × 10 × 30 × 2 = 19,200** (9,600 svt-rs + 9,600 aom-rs).

**Size axis explicitly scoped out.** The mandate allows a "native + 2–4
downscales" axis "if the budget allows." The local smoke test (§3) found
aom-rs's per-cell cost is dominated by (q, speed) — NOT by pixel count — to a
degree that a second size axis would multiply an already large aom-rs budget
(~167 CPU-h at ONE size, §3) rather than add proportionally. Given LAN-only
compute, adding sizes was cut in favor of full speed-preset density (the task's
stated *primary* axis of interest for the speed model) and the full 30-point q
grid. This is a deliberate, documented cut, not a silent one.

## 3. Smoke test (local, before any fleet launch) — and what it caught

Per the two-stage-launch mandate, before declaring anything: local single-cell
(then multi-cell) timing on a fresh `zenmetrics` build (`--features
sweep,png,jpeg,webp,avif,jxl,cpu-metrics,avif-svt,avif-aom`, zenmetrics HEAD
`fc971815`).

**Bug caught by the smoke test, before any fleet minute was spent:** the
original knob-grid (`{"backend":["svt-rs","aom-rs"],"speed":[1,4,7,10]}`, one
shared 1–10 range for both backends) failed immediately —
`aom-rs speed=10 out of the byte-verified --cpu-used range 0..=9`. The two
backends' speed dials are **not the same range** (svt-rs 1–10 via zenavif's own
abstraction; aom-rs 0–9, the raw libaom `--cpu-used`). Fixed by giving each
backend its own `--knob-grid` invocation with its native range (§2 table)
instead of a single shared grid — this is also why the grid is registered as
two separate per-backend axes rather than one shared "speed" column.

**Timing (real measurements, `--jobs 1`, single-threaded, this dev box)**:

*svt-rs*, 2 tiny sources (0.30 MP) × 5 q × 10 speeds (100 cells, 0 failures) +
1 large source (1442, 12.0 MP) × 5 q × 5 speeds (25 cells, 0 failures):

| speed | median @0.30MP | median @12.0MP |
|--:|--:|--:|
| 1 | 4,227 ms | 21,624 ms |
| 2 | 1,887 ms | — |
| 3 | 522 ms | 6,002 ms |
| 4 | 306 ms | — |
| 5 | 92 ms | 2,763 ms |
| 6 | 34 ms | — |
| 7 | 12 ms | 121 ms |
| 8–10 | 12 ms | 120 ms |

*aom-rs*, 2 tiny sources (0.30 MP) × 4 q × 10 speeds (40 cells before the smoke
was stopped — see below) + 1 large source (12.0 MP) × 1 q × 1 speed (spot check):

| speed | median @0.30MP | @12.0MP |
|--:|--:|--:|
| 0 | **98,981 ms** | not measured (too expensive to spot-check locally) |
| 1 | 38,970 ms | not measured |
| 2 | 12,773 ms | not measured |
| 3 | 2,990 ms | not measured |
| 4 | 2,111 ms | not measured |
| 5 | 1,266 ms | not measured |
| 6 | 400 ms | **2,904 ms** (measured) |
| 7 | 224 ms | not measured |
| 8 | 93 ms | not measured |
| 9 | 23 ms | not measured |

The aom-rs local smoke was **deliberately stopped at 40/100 cells** (all 10
speeds fully sampled at q∈{1,30,60,90}, tight variance within each speed —
max/median ratio ≤1.9 in every group) rather than run to completion or extended
to more large-image speeds: a full local aom-rs low-speed sweep on a 12–16 MP
image would itself cost minutes-to-tens-of-minutes **per cell** serially, which
is exactly the workload the fleet (not a local smoke test) exists to absorb in
parallel. This is the honest boundary of local smoke-testing per the two-stage
mandate — real calibration is the fleet's first chunk (§5), not more local
serial waiting.

**Budget estimate (labeled, NOT a validated benchmark — a planning number per
the "smoke calibrates sec/cell" mandate).** Per-speed cost was fit to a
power-law in megapixels (`cost ∝ MP^k`) using the two real size points above
(k=0.44–0.92 per svt-rs speed, individually fit; k=0.538 single-point estimate
for aom-rs, applied uniformly across its speeds since only one large-image
point exists), then summed over the actual 32-source MP distribution (min 0.25,
median 1.57, p75 12.0, max 16.0 MP) × 30 q values:

| backend | estimated CPU-hours | dominant cost |
|---|--:|---|
| svt-rs | ~7.5 | speed=1 (46%), speed=2 (28%) |
| aom-rs | ~167 | speed=0 (63%), speed=1 (25%) — **88% of aom-rs cost is speeds 0–1 alone** |
| **total** | **~175 CPU-hours** | |

This is an extrapolation from 2–3 measured size points per backend and should
be read as a **planning estimate with real uncertainty**, not a benchmark
result — flagged explicitly per the "never extrapolate performance numbers"
discipline. The fleet's first chunk (§5) is the actual calibration; this number
only sets expectations for fleet sizing and `ZEN_PASS_TIMEOUT`.

**Consequence for fleet config:** aom-rs speed 0–1 cells can individually run
past a minute even at modest sizes. Per the campaign's own documented gotcha
("the worker's per-pass timeout defaults to `ZEN_PASS_TIMEOUT=1800`; a chunk of
big cells cannot finish → rc=124 every pass, zero progress while looking
alive"), every worker in this wave launches with **`ZEN_PASS_TIMEOUT=14400`**
(4 h), matching the fix that doc prescribes for exactly this shape of workload.

**Correction found after writing the estimate above (checked before declaring,
not after): the extreme aom-rs low-speed cost is a SCREEN-CONTENT-DETECTION
artifact, not a general size/quantizer effect — and both local smoke-test
images happened to be screenshots.** `imazen/zenav1-aom#14` (the port's
byte-divergence band, originally filed against 1024×745/1280×800) is
**CLOSED**: its own resolution states "the '720 ≤ min(w,h) < 1080 band' of the
title was a coincidence of which train renditions are screen-detected" — the
real cause was under-costed palette/IntraBC search on screen-detected frames
(fixed across KB-41 roots #3-#21, census 30/30 → 102/102 → 104/104 byte-exact,
including the exact repro dimensions). **No `--exclude-min-dim` band exclusion
is needed for this wave** — the port is validated byte-exact for both photo
and screen content as of 2026-08-30. What IS still true: screen-detected
content pays a real throughput tax (IntraBC DV search, "~80 s per 1080p cell at
cpu6" per the issue's own numbers), not a correctness one. My two tiny smoke
images (`8288`, `8434`) are **both** `8100-lilith-web-screenshots` content —
i.e. exactly the class that pays this tax — so the §3 aom-rs low-speed
estimate (~167 CPU-h, calibrated on those two images) is very likely a
substantial **overestimate** for the 27/32 picks that are photos, illustrations,
scans, or product shots (not screen-detected, no DV-search tax). Only 5/32
picks are `8100-lilith-web-screenshots`; the `7000-lilith-plots` class (6/32 —
flat-color line art/charts) may or may not trigger the detector and is an open
question this wave's own data will answer. Not re-measured locally (would cost
the same serial local time the two-stage launch mandate exists to avoid) — the
fleet's real per-source, per-speed timing is what lands in the harvest, and
this note exists so a reader doesn't take the §3 number as tighter than it is.

## 4. Fleet topology (home LAN + local only — no cloud)

| host | role | capacity used | notes |
|---|---|---|---|
| **r7900x** (`192.168.50.27`, ex-"lianli" — see below) | primary CPU worker | 24 threads, uncapped | dedicated always-Ubuntu worker, `zen-worker` unit present but drained/stopped from the prior wave; `enroll_running_node.sh --start` reactivates it; `exec-zensim944hdr-03bdf64b` image already cached locally (verified: has `avif-svt`/`avif-aom` baked in, built after the 2026-08-30 campaign) |
| **tower** (`192.168.50.170`) | secondary CPU worker, CAPPED | `--cpuset-cpus` leaving ≥8 cores, `--cpu-shares=256`, `--memory=40g` | live household media server (Plex/sabnzbd/*arr stack all running, load 1.2–1.5 on a 32-thread box at observation time) — media has priority, never uncapped |
| **local (wsl/dev box)** | tertiary CPU worker, MODEST cap | via `run-heavy`, explicitly capped well below full 28 cores | a sibling Claude session is concurrently doing unrelated zensim work on this same physical box (`.workongoing` claim: "Profile D no-tax refactor") — kept light to avoid contending with it |
| jason / ian (node-2/node-3, kids' PCs) | **not used** | — | both observed reachable + idle (Ubuntu-booted, load ≤0.25) at check time, so not disqualified by the yank-rule protocol — but skipped anyway as a conservative call: r7900x + tower + local already give ~55–70 combined cores against a 19,200-cell/~175-CPU-h grid, and the task brief says skip on any doubt. Documented here rather than silently omitted. |

**Naming note:** the fleet doc `zensim/CLAUDE.md`'s "lianli" reference (hostname
`lilith-lianli`, IP `.27`) is superseded by the current neutral-ID convention in
the private homefleet roster — the box is now `r7900x` everywhere in current
docs/NODES.md (renamed for its silicon, not its case). Same box, same IP, same
role (dedicated always-Ubuntu worker) — just the current name.

**Corpus:** the 32 source PNGs are uploaded to
`s3://codec-corpus/avif-subsample-2026-09-01/` (LAN store; verified 32/32 objects
present via `aws s3 ls`, not s5cmd, per the SeaweedFS listing-undercount
caveat) so every worker (none of which mount `/mnt/v`) can fetch them via
`ZEN_CORPUS_BUCKET=codec-corpus ZEN_CORPUS_PREFIX=avif-subsample-2026-09-01`.

## 5. Runs

- `avifsub-svt-enc-20260901` — svt-rs encode, `s3://zentrain/jobs/avifsub-svt-enc-20260901/`
- `avifsub-aom-enc-20260901` — aom-rs encode, `s3://zentrain/jobs/avifsub-aom-enc-20260901/`
- Score runs declared after the encode gate passes (ssim2-gpu + zensim CPU +
  butteraugli, matching the prior wave's metric set), named
  `avifsub-{svt,aom}-sf-{cpu,gpu}-20260901`.

**Speed-model validity columns.** The job-system `pairs` output already carries
a `worker` column (verified against the prior wave's `pairs_svt.parquet`:
`['ref_path','dist_path','image_path','codec','q','knob_tuple_json',
'encode_sha','metric','worker','provider']`) and each `EncodedCell` carries
`encode_ms` (`Instant::now()`-measured inside the encode call,
`zenmetrics-cli/src/sweep/encode.rs`). At harvest time this sweep joins a small
static `host_meta` table (host → thread_count, shared: bool) onto `worker`,
giving every encode_ms row: host id, thread count (1 — single-threaded per
cell, concurrency is cross-cell), and contended (`true` for tower + local,
`false` for r7900x). This is a metadata join at harvest, not a core-schema
change — no zenfleet plumbing was touched for it.

## 6. Artifacts

- `benchmarks/avif_sweep_subsample_2026-09-01.md` (this file, zenmetrics repo —
  "the repo where the sweep harness lives")
- `/mnt/v/output/avifsvt-subsample-2026-09-01/`:
  - `reps_K32_full_even_2026-09-01.tsv` — raw k-means output (image_path, crop_label,
    content_class, cluster_id, cluster_size, width, height)
  - `sources_manifest_2026-09-01.tsv` — origin_id → final treatment (symlink vs.
    downscaled) mapping
  - `avif_subsample_picks_2026-09-01.tsv` — **the merged provenance table** (cluster
    table + corpus key + treatment + source path), sorted by cluster_id
  - `sources/` — the 32 corpus-ready PNGs (`<id>.scale<W>x<H>.png`)
- `s3://codec-corpus/avif-subsample-2026-09-01/` — same 32 PNGs, LAN-store mirror
  (what the fleet actually reads)
- Extended: `zenmetrics/scripts/imazen26_recluster_even.py` (`--crop-label` flag,
  width/height passthrough — additive, backward-compatible)
