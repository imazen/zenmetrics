# AVIF autotune v1 — one canonical training view, two bakes, and an honest scorecard

**2026-09-04.** Curation + training lane. No new encodes: this is the union of
every AVIF DOE wave that has already been scored, joined into one trainable
view, and the first knob-picker trained on it.

Consumer contract (what the wiring lane loads, and the one breaking difference
from today's runtime): **`/mnt/v/zen/avif-autotune-2026-09-04/AUTOTUNE_CONTRACT.md`**.
Per-file sha256 + every exclusion: that dir's `_MANIFEST.json`.

---

## 0. TL;DR

- **The view:** 111,400 harvested rows from 6 scored parquets → **79,368 rows**
  after measured exclusions, over **143 configs × 64 (image, corpus) rows × 117
  named zenanalyze features**, era-labelled, split 26 train / 6 `eval8` origins.
- **The bytes head works, unevenly.** Held-out mean regret **13.4 %** (core,
  48 cells) / **14.5 %** (full, 143 cells) against the per-row oracle. On
  screenshots the core bake is **0.7 % mean, exactly optimal on 92 % of
  decisions**; on photo and scan it is 17–19 %.
- **The backend head FAILS.** **54.0 %** agreement with the measured per-image
  winner against a **67.7 % always-`zenav1-svt` baseline** — worse than the
  constant. It gets svt-winners right and essentially never recovers a
  zenrav1e win (0/25 and 3/27 on the two eval images where zenrav1e wins).
  Reported as a fail; not shipped as a decision.
- **The time head is not usable as a budget gate** — the trainer's own
  `TIME_HEAD_R2` gate fires (median per-cell R² 0.316 core, **−1.981** full,
  bar 0.60). The emitted `encode_ms` LUT is what a runtime should use, and even
  that is modelled, not measured.
- **Three era-scoped facts were re-derived here from the bytes, not inherited
  from a doc** (§2): the cross-era byte identity, the dead-knob census, and the
  speed-dial aliasing.
- **There is no canonical holdout, and there cannot be one from this corpus.**
  All 32 references are even-origin by construction, so the `{1,3,5}` /
  `{7,9}` buckets are structurally empty (§3).

---

## 1. What went in

| source table | era | rows read | rows in the view |
|---|---|--:|--:|
| `zensim-avifdoe/doe_scored_2026-09-02.parquet` (a0r/a1/a2/ag) | 2026-09-01 | 49,120 | 37,696 |
| `zensim-avifdoe-b6/b6_scored_2026-09-02.parquet` | 2026-09-01 | 25,056 | 25,056 |
| `zensim-avifdoe-b6/naive_native_control_scored_2026-09-02.parquet` | 2026-09-01 | 7,189 | 6,549 |
| `zensim-avifdoe-eradelta/eradelta_scored_2026-09-03.parquet` (a1/b1/c1) | 2026-09-03 | 15,648 | 11,328 |
| `zensim-avifdoe-eradelta/c1_ag_scored_2026-09-03.parquet` | 2026-09-03 | 1,920 | 1,632 |
| `avif-backend-2026-09-03/br_scored_2026-09-03.parquet` (brnat/brsdr) | 2026-09-03 | 12,467 | 11,603 |
| **union after cell-merge + exclusions** | | **111,400** | **79,368** |

(The per-table "rows in the view" columns sum above 79,368 because 12,000 cells
appear in two eras and are merged — see §2.1.)

Carried but NOT trained on: `cells_hdr.parquet` (3,680 rows, Track T2, 10-bit PQ,
16 refs) — a separate regime with no knob axis. §6.

Per-image inputs: the G-CROP re-extracted budget features
(`avif-doe-1024-2026-09-01/_gcrop/budget_features.tsv`) and a **native feature
table extracted this pass** (`avifsvt-subsample-2026-09-01/_natfeat/native_features.tsv`,
32 rows × 117 features, same extractor + argv as the G-CROP run). Both were
needed because **the two corpora share all 32 filenames and 13 of them are
different pixels** — the crop corpus's `1442.scale4000x3000.png` is a 1024²
window, the native corpus's is the whole 4000×3000 image.

**Gate (new, and it passed):** the 19 references that are byte-identical
between the corpora have **byte-identical feature rows in both tables —
2,223 / 2,223 cells**. That is simultaneously a corpus-identity check and an
extractor-determinism check across the 2026-09-01 → 2026-09-04 gap.

---

## 2. Three things re-derived from the bytes

The view builder recomputes these instead of trusting the records, because two
of the records disagree with each other about one of them.

### 2.1 Cross-era byte identity — 12,000 / 12,000

12,000 cell identities `(corpus, image, arm, q)` appear in both the 2026-09-01
and 2026-09-03 eras. **All 12,000 carry the same `encode_sha`; 0 conflicts.**
So the era-delta wave's rows and Stage-A's rows are one population for exactly
those cells, and the builder merges them into one row (keeping whichever era
carried butteraugli). Any cell whose eras disagreed would have been dropped and
listed; none did.

### 2.2 The dead-knob census — and it settles a documented contradiction

For each `(corpus, speed, bit_depth, knobs)` arm, the fraction of its cells whose
`encode_sha` equals the same-`(corpus, speed, bit_depth)` control's at the same
`(image, q)`:

| arm | corpus | speed | cells | identical | verdict |
|---|---|--:|--:|--:|---|
| `scm3` | budget | 4 | 288 | 288 | **inert** |
| `tn0` | budget | 4 | 288 | 288 | **inert** |
| `scm3` | budget | 6 | 288 | 288 | **inert** |
| `scm3-tn0` | budget | 6 | 288 | 288 | **inert** |
| `tn0` | budget | 6 | 288 | 288 | **inert** |
| `scm3` | native | 4 | 96 | 96 | **inert** |
| `tn0` | native | 4 | 96 | 96 | **inert** |
| `tn0` | native | 6 | 288 | 288 | **inert** |

`avif_doe_stageA_2026-09-02.md` §3 says `scm3` is byte-identical to the control
at both presets; `avif_eradelta_analysis_2026-09-03.md` reads (in its byte-level
paragraph) as saying the opposite. **The measurement above agrees with Stage-A
at speeds 4 and 6 — and `scm3` at speed 7 is NOT flagged**, which is consistent
with the era-delta record's separate "screen-exclusive at speed 7" finding.
Speed-7 `scm3` cells are kept.

Plus **29 pair-arms** whose bytes equal a single-knob arm's on every shared cell
(e.g. `acb1-scm3` ≡ `acb1`, `mtx32-tn0` ≡ `mtx32`), and **svt presets 8/9/10 on
the native corpus, byte-identical to preset 7** on 251/118/247 shared cells.
Total dropped: **10,614 rows** (9,984 inert/alias + 630 speed-alias).

### 2.3 What else was excluded

| rule | rows | why |
|---|--:|---|
| `excluded_run` | 96 | `t1d` — 24 of its 96 cells encoded 4.5–7 h before either zenav1-svt#18 fix landed; superseded by era-delta `c1`, same plan/corpus/ladder |
| `unscored_cell` | 10 | no `m_ssim2` (never labelled, not a failure) |
| `measured_inert_or_aliased_arm` | 9,984 | §2.2 |
| `measured_speed_alias` | 630 | §2.2 |

`encode_sha` is used as a many-to-one **attribute**, never as the row key: the
row identity is the CELL `(corpus, image, arm, q)`. Keying on the sha would
drop one of every byte-identical pair and destroy exactly the inertness signal
§2.2 measures.

---

## 3. The split, and why it is not the canonical one

Every one of the 32 references has an **even** origin id — the corpus was
k-means-selected under `--parity 0` precisely so that no val/test-origin
content could reach a training artifact. The consequence is that
`origin_split.split_of` returns `train` for all of them and the canonical
`{1,3,5}` validate / `{7,9}` test buckets are **structurally empty**.
`train_hybrid.py` correctly refuses that corpus:

> `0 validation rows (no origins ending in 1/3/5). Train on the full imazen-26
> corpus, not a train-biased even-only set.`

Rather than weaken that guard, the trainer gained an explicit, declared hook —
`SPLIT_RULE = "even_only_eval8"` — implementing the **registered** even-only
sub-split (`zensim/docs/DATA_SPLITS.md` line 158, the `avifgen-2026-08-06`
precedent, owner `avifgen_training_views.py`): `{0,2,4,6}` train, `{8}` the
leg-side eval holdout. It **hard-errors if any origin is not canonical-train**,
so it cannot be used to launder odd-origin content into training, and it
hard-errors if it ever produces test rows. Default behaviour is unchanged.

Result: 26 train origins / **6 `eval8` origins**, zero overlap, zero rows with a
canonical val or test origin. The 6 happen to span all five coarse content
classes — `1008` photo, `6018` + `6038` scan, `7058` plot, `8288` screenshot,
`9118` ai-gen.

**`eval8` is a leg-side holdout, not the canonical one.** A genuine
generalization estimate for this picker needs an odd-origin AVIF encode wave,
which does not exist.

---

## 4. The scorecard

Regret = picked bytes ÷ cheapest cell in the view that reaches the target − 1,
over the `eval8` leg, at 23 reachable `ssim2` targets.

| bake | cells | n | mean | p50 | p90 | max | exactly optimal |
|---|--:|--:|--:|--:|--:|--:|--:|
| `core` (cross-size-verified, svt-only) | 48 | 266 | **13.4 %** | 8.8 % | 31.1 % | 78.7 % | 21.1 % |
| `full` (both backends, all knobs) | 143 | 293 | **14.5 %** | 8.3 % | 41.7 % | 78.6 % | 21.8 % |

Per coarse content class, mean regret (full / core):

| class | n (full) | full | core |
|---|--:|--:|--:|
| screenshot | 44 | 5.9 % | **0.7 %** (92 % exactly optimal) |
| plot | 38 | **4.0 %** | 4.5 % |
| scan | 110 | **11.0 %** | 18.6 % |
| photo | 50 | 25.5 % | **17.0 %** |
| ai-gen | 51 | 26.6 % | **11.5 %** |

By size class: `medium` 5.9 % / 0.7 %, `large` 16.0 % / 15.4 %. By target band
(core): q<30 8.9 %, 30–70 14.1 %, 70+ **16.0 %** — worst exactly where web
traffic lives, and the corpus is thinnest at the top (`DATA_STARVED_SIZE` fires
on 60 `(size_class, zq)` cells).

### Backend pick — FAILS

Reference: `avif-backend-2026-09-03/parquet/backend_per_image.parquet::winner_full`,
the per-image BD-rate over the pooled best-over-speeds frontier.

**The reference is a BUDGET-corpus verdict** — its `pixels` column reads
1,048,576 for every cropped reference, because the `zenrav1e` arm (`brsdr`) only
ever ran on the 1024 squared corpus. So the comparison is scoped to budget rows:
**161 comparable decisions, with 132 native rows counted NOT-COMPARABLE rather
than as misses.** (An earlier pass of this analysis scored all 293 and read
58.0 %; that applied a crop verdict to native pixels, which is exactly the
corpus collision the DOE's own gap-fill header warns about. The scoped numbers
below supersede it.)

| | agreement |
|---|--:|
| model (`full` bake) | **54.0 %** |
| always `zenav1-svt` | **67.7 %** |
| always `zenrav1e` | 32.3 % |

| image (budget leg) | measured winner | model agrees |
|---|---|--:|
| `1008.scale3000x4000` | zenrav1e | **0 / 25** |
| `6018.scale2320x3408` | svt | 29 / 30 |
| `6038.scale2479x3230` | svt | 26 / 26 |
| `7058.scale1024x1024` | svt | 11 / 27 |
| `8288.scale375x667` | svt | 18 / 26 |
| `9118.scale1536x1024` | zenrav1e | **3 / 27** |

The head has learned "pick svt". The most likely mechanism is in the data, not
the model: **the entire zenrav1e arm ran on the 1024 squared budget corpus
only**, so the picker sees zenrav1e evidence at exactly one size and none at
native. n = 6 held-out images for a per-image binary decision is also too few to
establish one either way.

Two further caveats on the reference itself: it is a **(backend × chroma)**
verdict (svt is 4:2:0 and zenrav1e 4:4:4 in every cell, 1,114 `av1C` boxes, 0
exceptions), and the backend-selection record's own §4 says the decisive
per-image signal is **reach**, not bytes — svt-4:2:0 cannot reach ssim2 90 on
16/32 references. A bytes-argmin picker cannot express "this backend cannot get
there at all".

### Time head — not usable

Trainer safety gate `TIME_HEAD_R2` fires on both bakes: median per-cell R²
**0.316** (core) and **−1.981** (full) against a 0.60 bar. Held-out relative
error p50 71 % / 83 %, p90 104 % / 135 %; rank agreement with the label is
SROCC 0.83 / 0.71. And the label is itself modelled — the view's `encode_ms`
comes from the speed instrument's `α + β·MP` fits, whose pooled form is flagged
`linear_model_failed` on 20/20 arms with β spreading up to **24.3×** across
sources. A time budget should use the LUT, and should treat it as an estimate
with a ±20 % floor even then.

### Both bakes fail the trainer's safety gate

`safety_report.passed = false` on both, baked with `--allow-unsafe`. Violations:
`OVERFIT` (train→val gap +5.4 pp core, +6.4 pp full, bar 2.0), `HIGH_OVERHEAD`
(bar 5 %), `PER_ZQ_TAIL` on 9 (core) / 20 (full) targets, `PER_SIZE_TAIL` on
`large`, `DATA_STARVED_SIZE` on 60 cells, `TIME_HEAD_R2`, and on `full` one
`WORST_ROW` at 100.9 %. Those thresholds were calibrated for corpora two orders
of magnitude larger than 32 references; the violations are a fair description of
a small-corpus v1, and they are recorded rather than tuned away.

---

## 5. What the numbers say to buy next

Ranked by what would move the scorecard, and all of it is encode work the view
cannot substitute for:

1. **An odd-origin AVIF wave.** Without it there is no canonical holdout and
   every number above is a leg-side estimate on 6 images.
2. **A zenrav1e native-size arm** (~2,880 cells / ~19 CPU-h, the backend
   record's rank-3 gap). This is the direct fix for the failed backend head.
3. **A zenrav1e 4:2:0 arm** (~24 CPU-h + one change to
   `avif_config_from_knobs`, rank 1). Until it exists, "backend" and "chroma"
   are one column.
4. **`encode_ms` persisted by the fleet path.** Everything about time here is
   a model of a model. One column in the ledger schema removes that.
5. **More plot/screen references** (n = 6 and 5). The two classes where the
   picker is *best* are the two whose estimates rest on the fewest images.

---

## 6. HDR is carried, not modelled

`cells_hdr.parquet` (3,680 rows, `t2a` + `t2b`, 16 refs, 10-bit PQ) is in the
view for completeness and joins nothing to the SDR table. Track T2 swept only
the speed/preset dial with **zero DOE knobs**, on a corpus where backend,
chroma and matrix all differ at once, scored in absolute nits through PU21 —
a different scalar from the SDR rows' u8-sRGB `ssim2` despite the shared column
name. Its own record calls it a baseline and states it establishes exactly two
pre-registered questions. **No HDR bake was trained and none should be from
this data.**

---

## 7. Owners touched (all extended, none forked)

| change | owner | why |
|---|---|---|
| `SPLIT_RULE` config hook + `even_only_eval8` | `zenanalyze/zentrain/tools/train_hybrid.py` | §3; default path unchanged, guard intact, hard-errors on any non-train origin |
| `zentrain.input_layout` metadata key | `zenanalyze/tools/bake_picker.py` | makes the runtime-scoped size one-hot self-describing instead of a hardcoded 4 |
| `avif_autotune_view.py` (new) | `zenmetrics/scripts/jobsys/` | the union step; imports `frontier`/`pooled_front`/`COARSE` from the Stage-A + brem analyzers, computes no statistic they own |
| `avif_autotune_validate.py` (new) | `zenmetrics/scripts/jobsys/` | forward pass from `zentrain/tools/_predict_lib.forward`, SROCC from `zenstats` via `zen_stats.py`, backend winner READ from `backend_per_image.parquet` |
| `native_features.tsv` (new data) | `zenanalyze/.../extract_features_imazen26_crops` | same binary, same argv shape as the G-CROP run |
