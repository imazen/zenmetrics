# AVIF knob-tuning design of experiments — `zenav1-svt` + `zenav1-aom` (2026-09-01)

**Goal (user directive, 2026-09-01):** produce the data a predictive model needs to
choose *optimal encoder-knob combinations* from `(target quality, zenanalyze features
of the reference image)`, jointly optimising **quality / size / speed**, for the
`zenav1-svt` and `zenav1-aom` backends reached through `zenavif`. Model training is a
follow-on lane; this document designs, budgets, registers and launches the
**experiment**.

**Registered BEFORE any declaration or launch**, per the sweep/calibration discipline
(`~/work/zen/CLAUDE.md`). Companions, all of which this design is built on:

| doc | what this design takes from it |
|---|---|
| [`avif_knob_dossier_2026-09-01.md`](avif_knob_dossier_2026-09-01.md) | the per-backend shortlists (§8), the 24 hazards, the version pins, and every dead-cell exclusion below |
| [`avif_sweep_subsample_2026-09-01.md`](avif_sweep_subsample_2026-09-01.md) | the K=32 image subsample (reused unchanged), the fleet topology, the launch/stop procedure |
| [`avif_sweep_permutation_retrofit_2026-09-01.md`](avif_sweep_permutation_retrofit_2026-09-01.md) | the canonical machinery (§1), the measured per-speed cost table (§9.1), the flattened denominators, §6's clean-stop options |
| `~/work/zen/zenjpeg/docs/VARIANT_GENERATION.md` | pattern 4 (identity = resolved state) and the rule *"every exclusion must be proven by encode"* |
| `zenmetrics/docs/JXL_LOSSY_KNOBSPACE_ABLATION_PROGRAM.md` | the declare → gap → reconcile precedent |

---

## 0. TL;DR

1. **The running sweep is the DOE's control arm, and it is half wasted.** The svt-rs
   arm (default envelope × 7 presets × 29 q) is exactly the `deviations = 0` stratum a
   factorial design measures against — **keep it, unchanged**. The aom-rs arm is
   spending **80 % of its remaining CPU on `cpu-used` 0–1 at 29-point q density**, for
   a *defaults-only* control. **Shrunk to 5,184 cells (from 9,280): saves 128.8 CPU-h**
   (§4), justified by the measured fact that aom encode time is flat in `q`
   (retrofit §9.2: every q column is 14.27–14.62 % of 7 sampled, uniform = 14.29 %) —
   so a speed model needs **no** q density, and an RD frontier at `cpu-used` 0/1 needs
   5 points, not 29.
2. **Design shape: quality-laddered fractional factorial, not a Cartesian grid.**
   `tune` is the outermost axis on both backends (hazard **H-4**); main effects are the
   `deviations = 1` stratum and pairwise interactions the `deviations = 2` stratum of
   the *existing* `SweepBuilder` machinery — so the design **is** the planner's own
   deviation ladder, not a new mechanism.
3. **Every knob arm carries its own 9-point q ladder**, a strict subset of the control's
   29-point grid. Arms are compared at **matched quality by RD interpolation**, not at a
   pre-computed anchor — which removes this design's dependency on baseline *scores*
   that do not exist yet (§2.3), while staying denser at low q than high q.
4. **`encode_ms` is not persisted anywhere by the running wave** (measured, §6). The
   `encode` job kind writes only the encoded bytes; `score_file` scores persisted blobs
   without re-encoding. **The speed model currently has zero input.** Fixed by a
   dedicated, uncontended timing block (§3.5) rather than by trusting fleet-contended
   times.
5. **The pixel budget is a free variable set by the permutation count, not a
   constraint on it** (user directive, §2.4). Knob screening runs on
   **1024×1024 content-aware CROPS at native resolution** — crops, not
   downscales, because the knobs under test are high-frequency machinery and a
   resampler removes exactly what they act on. That is what let the knob space
   *grow*: A1 from 3 presets to all 7, A2 from 16 images back to 32, A3 from one
   dense speed to three. ≈ **3,300 permutations per image**.
6. **Total: 117,435 cells, ≈ 70.5 CPU-h encode.** Roughly 1.5 days of LAN-fleet
   encode at measured throughput, plus scoring. It fits *because* of the aom
   shrink and the pixel budget together.

---

## 1. What the model needs, and what that implies for the design

The deliverable is a function `(target_quality, features(ref)) → knobs`, optimising a
three-way objective. Three consequences drive every choice below.

**(a) Per-(image, knobset) RD *curves*, not points.** "Which knobs at target quality T"
is only answerable if each arm's rate/quality curve can be evaluated at T. So every
knob arm gets a q ladder, and arms are compared by interpolating each curve to a common
quality — the BD-rate construction. A single anchored q per image would answer the
question at one T only, and would inherit any error in the anchor.

**(b) Main effects must be *de-aliased*, or the model learns the wrong attribution.**
`tune=iq` on both backends forces 9–10 other knobs (**H-4**). A factorial that crosses
`tune` with `{qm, sharpness, variance-boost, cdef, chroma-deltaq, deltaq, max_tx_size,
screen-detection}` does not measure those knobs — it measures `tune` nine times. The
design makes `tune` outermost and, more importantly, hashes the **resolved** post-tune
configuration (§5.1) so an aliased spelling *cannot* be declared as a separate cell.

**(c) Speed is a different measurement than size and quality.** Bytes and quality are
deterministic functions of (image, knobs); encode time is a function of (image, knobs,
host, contention, threads). They must not be harvested by the same instrument. §3.5
separates them.

---

## 2. Fixed inputs (reused, not re-derived)

### 2.1 Corpus — the K=32 subsample, unchanged

`/mnt/v/output/avifsvt-subsample-2026-09-01/sources/`, mirrored at
`s3://codec-corpus/avif-subsample-2026-09-01/`. 32 k-means representatives over 1,082
train-origin whole native imazen26 images, 84 content features, centroid-nearest member
per cluster; odd-id (val/test) origins excluded **by construction**. Provenance:
`avif_subsample_picks_2026-09-01.tsv`. Distribution as measured here:

- **MP:** min 0.25, median 1.57, mean 5.05, max 16.00; 32 values spanning
  0.25 / 0.37 / 1.02–1.57 (×9) / 3.69 / 6.71 / 7.91–8.01 (×3) / 12.00 (×6) / 15.99 / 16.00.
  Four size decades are present without adding a resample axis, which is why §3 does
  **not** re-introduce one for the RD blocks (it does for timing — §3.5).
- **Content:** 12 classes / 32 picks (plots 6, screenshots 5, ai-products 4,
  ai-illustrations 3, patent scans 3, nature 3, clipart 2, manuscript scans 2, and one
  each of art-photos, food, interiors, general photos).

### 2.2 Backends and their true axis ranges

| | svt-rs (`zenav1-svt`, SVT-AV1 v4.2.0) | aom-rs (`zenav1-aom`, libaom v3.14.1) |
|---|---|---|
| speed dial | zenavif `speed` 1..=10 → **7 distinct presets** {0,1,3,4,6,7,9}; 7/8/9/10 all → preset 9 (**H-2**, measured) | `--cpu-used` 0..=9, **injective** (gated by `aom_rs_speed_dial_is_injective`) |
| quality dial | `quality_to_qp_gated`: q 98 ≡ q 100 (QP 1) | `aom_rs_cq_level`: q 98 ≡ q 100 (cq 1) |
| distinct q | **29** of the 30 grid points | **29** of the 30 |
| encoding | the port encodes alone | the port **and** the C oracle run, and byte-identity is required (**H-5**, §5.3) |

### 2.3 The quality ladder (registered)

**Control (`deviations = 0`) arm:** the existing 29-point grid
`1, 5,10,…,70, 72,74,…,98` (q 100 merged into q 98).

**Knob arms — the 9-point ladder:** `q ∈ {5, 15, 25, 35, 45, 60, 76, 90, 96}`.

- Every point is a member of the control grid, so a knob cell at q=X is directly
  comparable to the default cell at q=X on the same image, with **no interpolation and
  no anchor error**.
- **Low-q density ≥ high-q density**, per the discipline: 5 points at q ≤ 45 (span 40)
  against 4 points at q > 45 (span 51). The weighting is deliberately stronger than
  "equal", because achieved quality is compressive at the top of the dial — measured on
  this very corpus (retrofit §3.2): image `8288` moves ssim2 70.96 → 77.19 over
  q 50 → 90 and only 77.19 → 77.52 over q 90 → 98, i.e. **most of the quality range
  lives below q 60**.
- q 96 retains a near-lossless point (the stated weak zone for every learned metric)
  without paying for the merged q 98/100 pair.
- 3-point probe ladder, where a block only needs to detect an interaction's sign:
  `q ∈ {15, 45, 90}`.

**Why a ladder and not a pre-computed quality anchor.** The brief's preferred anchoring
uses the already-scored baseline to place per-image q at target ssim2 bands. **Those
scores do not exist**: measured at 2026-09-02T01:55Z the score runs hold **29 (svt) and
7 (aom) completed score jobs** and have not advanced in 20 minutes — no score worker is
running anywhere on the fleet (§6). Blocking the DOE on that would idle the fleet.
Since every ladder point is a control grid point, the anchor is recovered *at analysis
time* for free, exactly and per image, as soon as the baseline scores land — the ladder
is a superset of what anchoring would have selected, at 9 points instead of 3.

### 2.4 The pixel budget is a FREE VARIABLE set by the permutation count

**User directive, 2026-09-01 mid-flight:** *"we may need to downscale or crop
images to make encode time achievable for the permutation set we want to test
per image… the permutation set you actually want should drive the design — don't
shrink the knob space to fit native-size costs."*

This inverts the sizing: the design picks the permutation set first, then buys
the pixels that fit it. Concretely it is what let §3 go from a
budget-constrained shape (3 presets, 16 images for interactions) to the **full**
one (all 7 effective presets for main effects, all 32 images for interactions,
all 23 aom arms).

**Decision: crop to 1024×1024; never upscale; native stays native below budget.**

> **AMENDED 2026-09-02 (§12.2).** The rule as *implemented* also passes through
> an image with **either axis already ≤ 1024**, which the counts below did not
> anticipate. The as-built corpus is **13 cropped / 19 native**, 37.02 MP against
> the 33.55 MP this budget implies (**+10.3 %**, 10 of 32 references). The
> amendment stands and the corpus was **not** rebuilt; the arithmetic, the one
> property actually lost, and the reason are in §12.2. The table below is the
> ORIGINAL registration, kept for the record — read the as-built column beside it.

| source | transform | n (registered) | n (as built) |
|---|---|--:|--:|
| MP ≤ 1.05 | **native, untouched** (never upscale) | 9 | 9 |
| either axis ≤ 1024, MP > 1.05 | native (unanticipated clause) | — | 10 |
| MP > 1.05 | **1024×1024 crop at native resolution**, content-aware window | 23 | 13 |

**Why crop and not downscale — the load-bearing choice.** A Lanczos or Mitchell
downscale is a low-pass filter, and **the knobs under test are largely
high-frequency machinery**: `ac_bias` is literally "RD bias toward
high-frequency error… texture and grain retention", variance boost keys on
per-superblock *variance*, `sharpness` biases the deblocking filter, CDEF is a
directional de-ringing filter, `max_tx_size` decides whether 64×64 transforms
are available, and SVT's screen-content detector keys on the crisp edges and
flat palette-able regions that resampling is precisely designed to soften.
Screening those knobs on resampled inputs measures them on content the
resampler has already partly removed — an attenuation that is both large and
knob-specific, i.e. exactly the kind of bias that survives into a trained model.
A native-resolution crop keeps the grain, the sharpness and the noise floor
intact; what it changes is *which* content, which the selection rule below
controls and the transfer gate (§3.8) tests.

**Why 1024 px and not 512.** The floor has to leave the encoder's machinery room
to behave representatively:

- **Tiles are an axis in this design** (`tile_cols_log2 = 1`, and `(1,1)`).
  AV1's minimum tile width is 256 px and SVT resolves an over-large request down
  to what the geometry supports rather than erroring, so a 2-column tiling needs
  ≥ 512 px and a 2×2 tiling needs ≥ 512 in *both* axes to be a real
  measurement rather than a silent degradation. 1024 gives 2× headroom on both.
- **Superblocks**: SVT uses 128×128 at presets ≤ M6 and 64×64 at ≥ M7. 1024 is
  8×128 and 16×64 — a full grid at either size, and an exact multiple of both,
  so the screening arm carries **no partial-superblock cells**. That removes a
  real confound from knob screening (partial-SB coding is its own code path);
  the cost is that partial-SB behaviour is only exercised by the native arms,
  which is where §3.8's gate looks.
- **The intercept lesson** (workspace sweep discipline §1): at 0.25 MP an svt
  speed-7 encode is 13 ms and is dominated by fixed overhead; at 1.05 MP it is
  ~44 ms and the per-pixel term leads. 1 MP is in the slope-dominated regime at
  every preset, so a knob effect measured there is not an artifact of α.
- CDEF works on 64×64 units and loop-restoration units are 64–256 px; both have
  many instances at 1024².

**Two picks sit below the floor and stay there.** `8288.scale375x667` (0.25 MP)
and `8434.scale414x896` (0.37 MP) are real phone screenshots and upscaling is
forbidden, so they enter the screening arm at native size. **The tile axis and
any partition-depth conclusion is NOT read on those two images** — recorded here
rather than dropping them, because they are the only members of the tiny bucket
and the byte-side intercept needs them.

**Crop selection — cluster-preserving, not center and not max-activity.** A
center crop lands on flat background often enough to matter; a
maximum-activity crop biases the whole wave toward busy content and would
inflate every knob's measured effect. The rule instead **preserves the parent's
place in the k-means feature space that selected it**:

1. Enumerate candidate 1024×1024 windows on a 5×5 stride grid over the source,
   plus the exact center (≤ 26 candidates).
2. Extract the same **84 zenanalyze content features** the subsample clustering
   used, z-scored with the *population's* mean/σ (not the candidates').
3. Choose the window minimising Euclidean distance to the **parent image's own**
   feature vector.
4. **Verify** the chosen crop's nearest k-means centroid is still the parent's
   cluster. Record the distance, the assigned cluster, and — where it moves —
   the class shift, per pick. **EXECUTED 2026-09-02 — §12.1: 11 of 13 crops
   preserved their cluster, 2 moved, and all 19 native references reproduced
   theirs exactly (the control that makes the other 13 readable).**

**The transformed image IS the reference.** Each crop is persisted
content-addressed alongside the native corpus
(`s3://codec-corpus/avif-doe-1024-2026-09-01/`), named
`<origin_id>.crop1024.png` so `avifsvt_cells.py`'s filename convention still
parses, with a manifest row carrying `source_id`, the crop rect `(x, y, w, h)`,
`transform = crop-native` (or `native`), the source sha256 and the crop sha256.
**zenanalyze features are re-extracted from the crop**, never inherited from the
parent: the tuning model conditions on features of *what was actually encoded*.
The score jobs emit a feature vector per scored variant already, so the
downstream table gets this for free; the selection step above is the only place
that needs its own extraction.

**Cost effect, measured-model.** Capping at 1.05 MP takes the corpus from
161.6 to 32.1 megapixels (5.0×), but encode cost is **sublinear** in pixels
(fitted `k` = 0.37–0.94 for svt, 0.54 for aom), so the honest saving is
**2.1× (aom, and svt's fast presets) to 3.7× (svt speed 4)** — not 5×. Stating
it as 5× would be the extrapolation error the discipline forbids. Per-(speed,
one q, all 32 images) at the budget, from the same fits as §4.1:

| svt speed → preset | native | at budget | aom cpu-used | native | at budget |
|---|--:|--:|---|--:|--:|
| 1 → 0 | 330.4 s | 205.9 s | 0 | 12,551 s | 6,013 s |
| 2 → 1 | 186.5 s | 100.9 s | 1 | 4,942 s | 2,367 s |
| 3 → 3 | 107.4 s | 38.2 s | 2 | 1,620 s | 776 s |
| 4 → 4 | 95.7 s | 26.2 s | 4 | 267.7 s | 128 s |
| 5 → 6 | 44.5 s | 10.1 s | 6 | 50.7 s | 24.3 s |
| 6 → 7 | 7.5 s | 3.0 s | 8 | 11.8 s | 5.7 s |
| 7 → 9 | 3.7 s | 1.35 s | 9 | 2.9 s | 1.4 s |

**The second win is the score side, which was the binding constraint** (§6):
scoring cost is per-variant and scales with *pixels*, so a 1.05 MP variant costs
roughly a fifth of the 5.05 MP corpus mean. That is what makes a 119k-cell wave
affordable at all.

---

## 3. The design

### 3.0 Structure

```
                       tune  (OUTERMOST — H-4: rewrites 9 other knobs)
                        │
      ┌─────────────────┼──────────────────┬───────────────────┐
   A0/A0R control    A1 main effects    A2 pairwise         AG transfer gate
   deviations = 0    deviations = 1     deviations = 2      native vs budget
   29-q × 7 presets  9-q × 7 presets    9-q × 1 preset      3-q × 2 presets
   native + budget   × 32 imgs @budget  × 32 imgs @budget   × 32 imgs @native
```

Every knob block runs at the **1024² pixel budget** (§2.4); the control runs at
**both** sizes, which is what makes the two comparable and gives the bytes
intercept for free (§3.9).

`deviations` is **not a new concept** — it is `SweepCell::deviations`, computed by
`zenavif::sweep::cross()` as the number of axes whose value index is non-zero, with
index 0 pinned to the production default on every axis. `SweepBuilder::with_max_deviations`
already bounds it. So "main effects, then pairwise" is the planner's own ladder, and
`run.plan.json` reports every stratum the design drops.

### 3.1 A0 — control (already in flight; svt keep, aom shrink)

The `deviations = 0` stratum at each backend's default envelope. This is the reference
every knob arm is differenced against, **and** the per-image RD backbone.

| arm | grid | size | cells |
|---|---|---|--:|
| **A0-svt** | 7 presets × 29 q × 32 img — **unchanged, in flight** | native | 6,496 |
| **A0-aom** | **SHRUNK** (§4): `cpu-used {4,6,8,9}` × 29 q + `{2,3,5,7}` × 9 q + `{0,1}` × 5 q, minus 5 poison cells | native | 5,179 |
| **A0R-svt** | 7 presets × 29 q × 32 img — the **same-size default** every svt knob arm is differenced against | budget | 6,496 |
| **A0R-aom** | `cpu-used {2..9}` × 9 q + `cpu-used 1` × 3 q | budget | 2,400 |

A0-aom's 5-point ladder at `cpu-used` 0/1 is `q ∈ {5, 25, 45, 76, 96}`.

**A0R is not optional.** A knob cell at 1024² must be differenced against a
*default* cell at 1024², on the same image, at the same q. The A1 plan already
emits its `deviations = 0` stratum at the 9 ladder points, so the minimum was
free; A0R buys the other 20 q points so the reduced-size RD curves are as dense
as the native ones, for 3.1 CPU-h. `cpu-used 0` is deliberately absent from
A0R-aom (6,013 s per q point even at budget); the native A0 arm carries it.

### 3.2 A1 — main effects, svt-rs

17 single-deviation knob arms × **all 7 effective presets** (speeds
{4, 6, 7, 2, 5, 3, 1} → presets {4, 7, 9, 1, 6, 3, 0}) × the 9-point ladder ×
**32 images at budget** = **34,272 cells**.

This is the §2.4 authorization spent: at native size the same block over 3
presets cost 4.55 CPU-h and could not have afforded the slow end at all; at
budget the **whole** ladder costs 12.1 CPU-h, of which speed 1 (preset 0) alone
is 8.75. The speeds axis is ordered `[4, 6, 7, 2, 5, 3, 1]` — default first,
speed 1 last — precisely so the budget gate (§7.3) sheds preset 0 first and
nothing else.

| # | axis | levels (index 0 = current default) | dossier | why these levels |
|--:|---|---|---|---|
| 1 | `tune` | **1 = PSNR**, 0 = VQ, 3 = IQ | §8.1 #3, H-4 | zenavif drives `EncodePipeline` at SVT **mainline** defaults today — tune 1, no QM, no variance boost, sharpness 0. Tune 3 is the only mode upstream marks *still-image only*; tune 0 is upstream's "greater sharpness" recommendation. Tune 2/4 excluded: 4 (MS-SSIM) forces the same block as 3 minus the IQ-only bits and would mostly merge; 5 is `TUNE_FILM_GRAIN` in this port, **not** VMAF (**C-1**). |
| 2 | variance boost | **off**, on(str 2, oct 5), on(str 3, oct 5), on(str 3, oct 7) | §8.1 #4 | upstream: *"strength 3 is best for still images"*, octile 4–7. Strength 4 **excluded — it saturates to strength 3's plan** (`avif.rs:806`), so it is 3 distinct levels, not 4. Harness clamps 1..=4 / 1..=8 per **H-10** (release-mode OOB panic on `pub` fields guarded only by `debug_assert`). |
| 3 | QM levels | **off**, (4,10), (2,10), (8,15) | §8.1 #5 | SVT ships QM **off**; libaom's all-intra default window is 4/10 and its image tune forces 2/10; 8/15 is SVT's own documented default window when enabled. Three upstream opinions = three levels. |
| 4 | `sharpness` | **0**, 3, 7 | §8.1 #6 | both backends' image tunes force **7**. Treated as **CATEGORICAL, never ordinal** (H-7 in §8.2 / the aom analysis): on libaom `sharpness == 3` is a discrete switch at 8 unrelated sites and the quantizer term has integer-division plateaus. Three levels, fitted as factors. |
| 5 | screen-content | **None (preset-derived)**, `Some(3)` | §8.1 #7, H-6 | forces the anti-alias-aware detector at any preset; decisive on the 11/32 screenshot+plot picks. |
| 6 | `ac_bias` | **0.0**, 1.0, 3.0 | §8.1 #8 | live in mainline (`pipeline.rs:2041` ungated), default 0.0 ⇒ entirely unexplored. 1.0 is the fork's own default; 3.0 probes the upper half of 0.0–8.0. |
| 7 | `max_tx_size` | **64**, 32 | §8.1 #9 | tune-IQ picks it *by qp* (32 at qp ≤ 45) — upstream saying the optimum is quality-dependent, which the 9-point ladder resolves directly. |
| 8 | tiles (log2 cols,rows) | **(0,0)**, (1,0), (1,1) | §8.1 #10 | the parallelism/efficiency lever that is not preset. Bytes-moving, so it is a *modelled* axis, unlike `thread_count` (byte-inert). |
| 9 | `bit_depth` | **Auto (8)**, 10 | §8.1 #11 | 10-bit reduces banding in smooth gradients at a bitrate cost — a real still axis. |

Non-default level count: 2+3+3+2+1+2+1+2+1 = **17**.

**Excluded from A1 by construction, each citing its hazard:**

| excluded | reason |
|---|---|
| `sb_size_override` | **H-11** — SB128 is signalled but the 128 root is forced-SPLIT; a size delta there is signalling/CDF, not partition quality. Would be a mislabelled sample. |
| `rc_mode` Cqp vs Crf | **H-3** — the port refuses `aq_mode != 0` and independently verifies `Cqp ≡ Crf` byte-for-byte on a still. One factor, not two. |
| `rc_mode` Vbr/Cbr | **H-8** — silently ignores both `qp` and `target_bitrate` on frame 0 and uses QP 30 regardless. Does not error. |
| `aq_mode != 0` | refused by the port (`pipeline.rs:927`). |
| `tune` 4, 5 | 4 mostly merges into 3; 5 is FILM_GRAIN in this port, not VMAF (**C-1**). |
| `tf_strength`, `kf_tf_strength`, `qp_scale_compress_strength`, `noise_adaptive_filtering` | **H-7** — zero consumers crate-wide. |
| `thread_count` | byte-inert with one tile; a throughput-only axis (**U-10**), held at 1. |
| 4:4:4 / 4:2:2 / 12-bit, superres, everything multi-frame | refused by the port (§4.2 of the dossier). |
| variance-boost strength 4 | saturates to strength 3 (`avif.rs:806`). |
| speeds 8, 9, 10 | **H-2** — all preset 9, identical bytes *and* identical times (measured to the millisecond, retrofit §9.1). |
| q 100 | merges with q 98 on both backends (measured, retrofit §3). |

### 3.3 A1b — FOLDED INTO A1 (superseded by §2.4)

A1b was a 3-q probe at preset 1, existing only because native-size costs made the
slow presets unaffordable in the main block. At the pixel budget the **whole**
preset ladder fits in A1 at full 9-q density, so the probe is redundant and is
not declared. Its scientific purpose — detecting a knob effect that *inverts* at
the slow end, where the preset table turns SG restoration, Wiener, filter-intra
and wedge prediction on — is now served better by presets 0 and 1 at 9 q instead
of 3, and remains a registered Stage-B trigger (**B-5**).

### 3.4 A2 — pairwise interactions, svt-rs

All two-deviation combinations of the 17 A1 levels — `((Σd)² − Σd²)/2 = (289 − 37)/2 =`
**126 composed strata**, 137 with the singles — × **speed 6** × the 9-point ladder ×
**all 32 images at budget** = **39,456 cells** before merges.

- **Speed 6 (preset 7)** is the single point: it is the neighbourhood of libavif's own
  default (`--speed 6`), and it costs 0.85 % of the svt arm's encode CPU.
- **32 images, not 16.** The earlier draft cut the image axis in half purely to fit
  native-size cost, which cost 3 content classes' outliers and 40 % of the population
  coverage. At budget the full corpus costs **1.02 CPU-h** for this block, so the cut
  is withdrawn: interactions are now measured on exactly the same content the main
  effects are. (The 16-image class-stratified subset survives in §9 as the
  pre-registered *fallback* if the budget gate fires.)
- **Aliased pairs cost nothing.** A pair like `(tune=3, qm=(4,10))` resolves to the same
  configuration as `tune=3` alone, because `apply_tune_overrides` forces exactly that QM
  window. The planner merges it by fingerprint and reports it in
  `run.plan.json::duplicates_merged`. That is H-4 being *neutralised by construction*
  rather than by a hand-maintained exclusion list.

### 3.5 A3 — main effects, aom-rs

23 single-deviation arms, all `aom_bench::ToggleKnobs` fields whose C counterpart is
emitted by `ToggleKnobs::c_ctrls()` and therefore drivable from `zenmetrics` **without
touching the `zenav1-aom` repo**:

| group | arms | note |
|---|--:|---|
| quantization matrices | 2: (4,10), (2,10) | driven C-side by `EncodeCell::c_encode_qm`, a separate entry point — its own sub-arm. **H-23**: the still-image QM formulas are *decreasing* in qindex and effectively use only levels 4–10, so the selecting branch is held fixed. |
| trellis | 2: `disable_trellis_quant` 0, 1 | default is **3**, not 0 — `aomcx.h:716` states the wrong default (**H-22.2**); trust `av1_cx_iface.c:289` + `arg_defs.c`. |
| partition envelope | 5: `max_partition_size_px` 64, 32; `enable_{rect,ab,1to4}_partitions=0` | sizes restricted to {4,8,16,32,64,128} in the harness — **H-21**: C-release limps on illegal values while the port panics. |
| intra mode set | 8: `smooth`, `paeth`, `cfl`, `directional`, `diagonal`, `angle_delta`, `filter_intra`, `intra_edge_filter` each `=0` | for a still, intra **is** the encoder — the richest cheap factor block on either backend. |
| transform search | 5: `enable_tx64=0`, `enable_rect_tx=0`, `enable_flip_idtx=0`, `reduced_tx_type_set=1`, `use_intra_dct_only=1` | `enable_tx_size_search=0` **excluded**: C asserts against combining it with `enable_tx64=0` (`encodeframe.c:2461`), and it is independently inert at `cpu-used ≥ 8` (**U-9**). |
| screen tools | 1: `tune_content_screen=1` | the only way to force palette + IntraBC **deterministically** (bypasses the detector), which makes them designable factors instead of a content lottery (**H-6**). |

Grid: 23 arms × `cpu-used {4, 6, 8}` × 9 q × **32 images at budget** =
**19,872 cells**, 9.1 CPU-h. (At native size the same coverage was 24 CPU-h and
had to be cut to one dense speed plus two 3-q probes; §2.4 buys the full
three-speed × 9-q block instead.)

- `cpu-used 6` is libavif's default speed.
- `cpu-used 4` is the quality end of the practical range.
- `cpu-used 8` is the **non-RD-mode** probe: libaom silently disables deltaq modes 1–5
  and `enable_tx_size_search` above `cpu-used 7` (**H-17**, **U-9**). Which of these 23
  arms goes inert there is a fact the model must know, and it is exactly the kind of
  "measured null that is really a structural zero" the dossier warns about. Measuring it
  at 3 q costs 0.25 CPU-h.
- `cpu-used 4` on 16 images is the slow-end interaction probe (the aom analogue of A1b).

**Excluded from A3:** `--tune=iq` / `ssimulacra2` (**PORT-MISSING at the harness layer**
— ported and byte-gated in `aom-encode`, but `ToggleKnobs` has no tune field, so wiring
it means editing the `zenav1-aom` repo; **out of scope for this wave by the brief, and
recorded in §8 as the single highest-value follow-up**); `deltaq_mode2`/`deltaq_mode3`
(port-side only, not emitted by `c_ctrls` ⇒ every cell would fail byte-identity);
`coeff_cost_upd_freq`/`mode_cost_upd_freq` (**present but port-side inert** — a sweep
there produces *false nulls*, worse than no knob); `cdf_update_mode=2` (≡ 1 on a lone
KEY frame); `disable_tx_stats_prune` (anti-vacuity witness, not a product knob);
`--arnr-*` (**H-7**, no alt-ref on a still); `--denoise-noise-level` values (**H-19** —
the value is discarded in all-intra, only the `>0` gate survives); `min-q`/`max-q`
(**H-20** — `0,0` is a back door into lossless that skips the lossless checks);
CICP/colour (harness-pinned, **H-15**); `--row-mt`/threads (**U-10** — C libaom is
almost certainly not bit-exact across thread counts; held fixed).

### 3.6 A4 — the timing block (separate instrument, uncontended)

**Finding that forces this block:** `encode_ms` is **not persisted by the running
wave**. `jobexec` emits it only on the `metric` job kind (`jobexec.rs:1826`, `:1848`),
which re-encodes; the `encode` kind writes the encoded bytes to stdout (they *are* the
content-addressed output) and the `score_file` kind scores persisted blobs without
re-encoding. The ledger schema has no timing column. **So the sweep as running produces
no speed-model input at all.**

Rather than change the ledger schema or the content-addressing contract, timing is
harvested by the canonical single-host tool — `zenmetrics sweep`, whose output parquet
already carries `encode_ms` (`assemble/mod.rs:176`) — run **serially, on r7900x, after
the encode fleet drains**, with:

- **an explicit size axis** — the one dimension the launch doc scoped out. Timing is the
  measurement that *needs* it: the discipline requires fitting `total = α + β·pixels`
  and reporting both terms, and "ms/MP" alone is meaningless without the intercept.
  Ladder: 64², 256², 512², 1024², 2048², 4096² Lanczos derivatives of 4 corpus sources
  (2 photo, 1 screenshot, 1 plot) — 24 inputs spanning three decades including the tiny
  bucket where fixed cost dominates.
- **all 7 svt presets and all 10 `cpu-used` values** at 3 q points (measured: per-q
  cost is flat to ±1.2 % on aom and ±3.3 % on svt — retrofit §9.2 — so q density buys
  a speed model nothing);
- **the A1/A3 knob arms** at 2 sizes and 1 q, so the model can price a knob's *time*,
  not only its bytes;
- `threads = 1`, `--jobs 1`, ASLR left on, ≥ 3 repeats, **minimum** reported.

≈ 1,800 timed encodes. Recorded with host id, thread count, and a `contended=false`
flag; every fleet-harvested time stays labelled **ranking-only** (retrofit §7).

### 3.8 AG — the cross-size transfer gate (pre-registered, blocking)

Reduced-size screening is only worth anything if a knob's effect at 1024² points
the same way as at native size. That is an assumption, so it gets a gate rather
than a paragraph of reassurance — and at speed 6/4 it is nearly free.

**Grid:** all **17 main-effect arms** × speeds **{4, 6}** × 3 q `{15, 45, 90}` ×
**all 32 images at NATIVE size** = **3,264 cells, 1.47 CPU-h**. The matching
reduced-size half is already inside A1, so the gate costs only its native leg.

**Criterion, fixed before the data exists.** For each (image, arm, speed),
compute the BD-rate against that image's default arm at the same size, over the
3 shared q points. Then, per arm:

| test | bar |
|---|---|
| **T1 direction** | sign agreement between native and budget BD-rate on ≥ **80 %** of the 32 images, counting only images where `\|BD-rate\|` at native ≥ 0.5 % |
| **T2 rank** | Spearman of the 32 per-image BD-rates, native vs budget, ≥ **0.7** |
| **T3 magnitude** | median `\|BD-rate_budget − BD-rate_native\|` ≤ **1.0 %**, and no systematic sign in that residual (binomial p ≥ 0.05) |

An arm passing all three is screened at budget and its Stage-B follow-up may
stay at budget. An arm failing **T1** is **not** screened at reduced size: it is
promoted to a native-size arm in Stage B and its A1/A2 numbers are annotated as
size-conditional. An arm failing only T2/T3 keeps its *direction* (which is what
Stage-B triggers key on) with the magnitude flagged.

Expected failures, named in advance so a hit is not a surprise: `ac_bias`,
`sharpness` and forced screen-content are the arms whose mechanism is most
plausibly resolution-coupled — cropping preserves native HF content, which is
exactly why crop was chosen over downscale (§2.4), but a crop still changes the
*ratio* of frame area to feature scale, which is what the detector and the
deblocking filter see.

### 3.9 The α + β·pixels decomposition, free from the size pair

Running the control at **both** sizes on the same images at the same (speed, q)
gives every `(image, speed, q)` cell two pixel counts, which is a two-point fit
of `total = α + β · pixels` per the discipline's requirement to report the
intercept *and* the slope, never a bare "ms/MP" or "bpp":

- **Bytes: free, from the fleet.** `encoded_bytes` is recorded for every cell, so
  A0 (native) × A0R (budget) yields the `header_bytes + content_bpp · pixels`
  split per (image, speed, q) with **no extra encodes**. This is the term the
  discipline warns about hardest — a ~1 KB AVIF container is +0.4 bpp on a
  thumbnail and ~0 at 4K, so a bitrate model without it ships wrong defaults at
  the small end. The 2 sub-0.4 MP picks are what make the intercept identifiable.
- **Time: NOT free, and not obtainable from these runs at all** — `encode_ms` is
  not persisted by the fleet path (§6). The A4 block is the instrument, and it
  carries a **6-point** size ladder (64², 256², 512², 1024², 2048², 4096²)
  rather than two points, because a 2-point fit of a term the sweeps show to be
  sublinear in pixels would be a straight line through a curve.

Stating which half is free and which is not is the point: the size pair is a
genuine opportunity for the *bytes* model and buys the *speed* model nothing.

### 3.7 Totals

| block | size | cells | encode CPU-h |
|---|---|--:|--:|
| A0-svt control (in flight, unchanged) | native | 6,496 | 6.25 |
| A0-aom control (**shrunk** from 9,280) | native | 5,179 | 32.4 |
| A0R-svt reduced control | budget | 6,496 | 3.11 |
| A0R-aom reduced control | budget | 2,400 | 5.00 |
| A1 svt main effects (17 arms × 7 presets × 9 q × 32) | budget | 34,272 | 12.1 |
| A2 svt pairwise (137 sets × 1 preset × 9 q × 32) | budget | 39,456 | 1.02 |
| A3 aom main effects (23 arms × 3 speeds × 9 q × 32) | budget | 19,872 | 9.10 |
| AG cross-size transfer gate | native | 3,264 | 1.47 |
| **fleet total** | | **117,435** | **≈ 70.5** |
| A4 timing (single-host, not a fleet run) | ladder | ~1,800 | ~2 |

Of the fleet total, **4,415 cells are already done** (3,562 svt + 853 aom at
2026-09-02T01:55Z).

**The math the directive asked for, laid out.** Permutations per image × sec per
cell at the chosen pixel budget × K images, against fleet capacity:

| block | permutations / image | sec/cell @budget (mean over the 32) | × K=32 | encode |
|---|--:|--:|--:|--:|
| A1 | 17 arms × 7 presets × 9 q = **1,071** | 0.353 s | 34,272 cells | 12.1 h |
| A2 | 137 sets × 1 preset × 9 q = **1,233** | 0.093 s | 39,456 cells | 1.02 h |
| A3 | 23 arms × 3 speeds × 9 q = **621** | 1.65 s | 19,872 cells | 9.10 h |
| A0R | 7 × 29 + aom 75 = **278** | 1.26 s | 8,896 cells | 8.11 h |
| AG | 17 × 2 × 3 = **102** (native) | 1.62 s | 3,264 cells | 1.47 h |

**≈ 3,300 permutations per image**, against a native-size design that could
afford roughly a third of that. Capacity: the LAN fleet is r7900x (24 threads,
uncapped) + tower (capped, ≥ 8 cores left for the household) + local (2 workers,
cpuset-capped at 6 cores each) ≈ **46 effective cores**. 70.5 CPU-h of encode is
**≈ 1.5 days** wall at full occupancy, plus scoring.

**Scoring is the co-equal cost and is measured, not assumed.** 117k variants ×
3 CPU metrics + a zenanalyze feature vector each. Per-variant score cost scales
with pixels, so the budget cells are ~5× cheaper than the native mean — but this
lane has **not measured the per-variant rate** (nothing was scoring, §6).
Registered gate **G-SCORERATE**: after the first 200 scored variants land,
compute the realised rate and re-check this budget; if the total lands above
50 CPU-h of scoring, the de-scope rule (§7.3) fires on A2 first.

---

## 4. Fleet arbitration — what was cut from the running wave, with the math

### 4.1 The cost model, and its validation

Per-speed cost was fitted `ms = a · MP^k` in log-log from the **measured** 5-image
tables (svt: retrofit §9.1, 5 images 0.25–12.0 MP × 10 speeds; aom: launch doc §3,
2 images at 0.30 MP × 10 speeds + one 12.0 MP point at speed 6, k = 0.537 applied
uniformly), then summed over the true 32-image MP distribution.

**Validation of the method, not of the inputs:** reproducing the two prior lanes'
totals gives **6.25 CPU-h** for the svt control (their labelled estimate: ~7.5) and
**161.2 CPU-h** for the aom control as declared (theirs: ~167). Within 17 % and 4 %
respectively. The *inputs* keep their original uncertainty: the aom numbers are
calibrated on two images that are **both screen-detected screenshots** — the class that
pays the IntraBC DV-search tax — so **every aom CPU-h below is an upper bound**, as the
launch doc itself flagged. The svt numbers are direct measurements on 5 corpus images.

### 4.2 svt-rs — KEEP, unchanged

| | |
|---|--:|
| declared (flattened) | 6,496 |
| done at 01:55Z | 3,562 rows, of which 630 are the now-undeclared speed-8/9/10 aliases |
| measured throughput 01:35→01:55Z | **3,036 cells/h** |
| remaining | ≈ 3,666 cells ≈ **1.2 h wall** |

It is the DOE's control arm, it costs 6.25 CPU-h in total, and it is the only source of
a **7-preset × 29-q** response surface at defaults. Cutting it would save ~2 CPU-h and
destroy the baseline. **No change.**

### 4.3 aom-rs — SHRINK, 9,280 → 5,184 cells

| `cpu-used` | q points | cells | CPU-h kept | CPU-h if left at 29 q | saved |
|--:|--:|--:|--:|--:|--:|
| 0 | 5 | 155¹ | 17.4 | 101.1 | **83.7** |
| 1 | 5 | 160 | 6.9 | 39.8 | **32.9** |
| 2 | 9 | 288 | 4.0 | 13.0 | 9.0 |
| 3 | 9 | 288 | 0.9 | 3.1 | 2.1 |
| 4 | 29 | 928 | 2.2 | 2.2 | 0 |
| 5 | 9 | 288 | 0.4 | 1.3 | 0.9 |
| 6 | 29 | 928 | 0.4 | 0.4 | 0 |
| 7 | 9 | 288 | 0.07 | 0.23 | 0.16 |
| 8 | 29 | 928 | 0.10 | 0.10 | 0 |
| 9 | 29 | 928 | 0.02 | 0.02 | 0 |
| **total** | | **5,184** | **32.4** | **161.2** | **128.8 (80 %)** |

¹ `1432.scale3000x4000.png × cpu-used 0` is **excluded by construction** — 5 cells,
`failed / encoder_panic` in the live ledger, a `zenav1-aom` port defect owned by the
KB-41 lane. Annotated, not silently dropped; a fixed port re-declares them additively.

**Why this specific shrink is not information loss:**

1. **Encode time is flat in q — measured.** aom: every q column is 14.27–14.62 % of
   7 sampled, uniform = 14.29 %, `q100/mean = 0.999×`. svt: `q100/mean = 1.033×`
   (retrofit §9.2). A **speed** model therefore extracts nothing from q density; the
   full `cpu-used` ladder — which the shrink **keeps at every one of its 10 values** —
   is the entire signal.
2. **An RD frontier needs shape, not resolution.** `cpu-used` 0 and 1 exist in this
   design to bound what any knob can achieve. A 5-point ladder spanning
   q 5 → 96 with a smooth monotone curve fixes that bound; points 6..29 refine a
   frontier nobody will operate on (aomenc `cpu-used 0` on a 12 MP still is minutes per
   image).
3. **The production band keeps full density.** `cpu-used {4, 6, 8, 9}` — which brackets
   libavif's default of 6 — retains all 29 q points, so the cross-backend
   quality-matched comparison and the per-image anchor recovery are unaffected.
4. **The freed capacity is what pays for the DOE.** 128.8 CPU-h against ≈ 12.4 CPU-h
   for all four new knob blocks. The wave stops buying a defaults-only frontier at
   30-point resolution and starts buying knob effects.

**Measured aom throughput** 01:35→01:55Z: **540 cells/h**, on a cell mix dominated by
`cpu-used ≥ 2`. Remaining after the shrink ≈ 4,400 cells, ≈ 31 CPU-h, of which 24 CPU-h
is the 315 kept `cpu-used` 0/1 cells.

### 4.4 How the shrink is applied

By **option (b)** of retrofit §6 — *shrink the declaration*: re-declare the smaller cell
set through the canonical builder and swap `manifest.json`; workers re-read the manifest
each pass and stop claiming what is no longer declared. Reversible (the current manifest
is preserved beside it), **loses no ledger row, kills no worker, orphans no completed
cell**. Option (a) (`control.json` pause) is inert for these workers — they were launched
without `ZEN_CONTROL_KEY` (verified from their container env). **Every DOE run launched
by this lane sets `ZEN_CONTROL_KEY=jobs/<run>/control.json`** so it is pausable.

---

## 5. Machinery — how this is declared

### 5.1 zenavif: one new axis, resolved-state fingerprints

`zenavif::sweep::SweepAxes` gains **one** axis, `svt_knobs: Vec<SvtKnobSet>`, where
`SvtKnobSet` carries the nine §3.2 knobs and knows its own deviation count. Every
pre-existing preset pins `vec![SvtKnobSet::default()]`, so cell ids, deviation counts
and fingerprints are unchanged — pinned by the existing
`backend_axis_does_not_move_archived_zenravif_fingerprints` golden test, which replays
12 `(cell, q, fp)` triples minted before the backend axis existed.

Two properties are load-bearing:

- **`fingerprint()` hashes the RESOLVED configuration.** It builds the
  `svtav1::HdrForkConfig` the encoder will actually run and calls the port's own
  `apply_tune_overrides(qp)` (public, idempotent, and a documented no-op for tunes
  0/1/2/5) before hashing. Consequence: `(tune=3, qm=off)` and `(tune=3, qm=(4,10))`
  have **one** fingerprint and are declared once, with the alias recorded. H-4's
  aliasing is removed by the same mechanism that removed the speed-dial aliasing —
  resolved-state identity, per VARIANT_GENERATION pattern 4 — rather than by a
  hand-written exclusion table that can drift.
- **The harness clamps what the port only `debug_assert`s.** `variance_boost_strength`
  to 1..=4 and `variance_octile` to 1..=8 (**H-10**: `var_boost.rs` indexes a
  `[f64; 5]` and computes `octile · SUBBLOCKS_IN_OCTILE − 1` behind assertions that
  vanish in release; a fleet worker is a release build).

Knob values reach the encoder through `expert::SvtParams` — the existing, explicitly
**unstable** expert surface whose module doc already names "sweeping parameter
combinations to feed a picker / regression / calibration training pipeline" as its
purpose. `EncoderConfig`'s fields are `pub(crate)`, so this is additive.

New plans: `svt_doe_main` (A1/A1b) and `svt_doe_pairwise` (A2).

### 5.2 Declaration path and cell identity

svt blocks declare through the **canonical plan path**, which emits the canonical
`{"cell","fp","plan"}` `knob_tuple_json` — so these runs are picker-dataset compatible,
`assemble/flat_picker.rs` yields non-NULL `cell`/`fp`/`knob_plan`, and
`check_mandatory_coverage.py` works on them:

```sh
zenmetrics sweep --codec zenavif --plan svt_doe_main \
  --sources <32 picks> --q-grid 5,15,25,35,45,60,76,90,96 \
  --dry-run --emit-cells cells.jsonl --emit-cells-image-path basename \
  --output <run>            # writes <run>.plan.json
zenfleet-ctl declare-encodes --cells cells.jsonl --out manifest.json
```

The running A0 wave **stays on the raw knob-map schema** — migrating it would change
`knob_tuple_json`, which is half the content-addressed `CellId`, and re-run all 4,415
completed cells (retrofit §5.2). The DOE runs are new run names, so they start clean on
the canonical schema. **Two schemas will coexist in this campaign; that is deliberate
and recorded here.**

The aom blocks have no `EncoderConfig` representation (the port is driven directly from
`zenmetrics-cli`), so they declare via `--knob-grid` **with a registered resolver in
`sweep/dedup.rs`** — per the retrofit's rule: *"do not add a knob to `AVIF_KNOBS` and
sweep it with `--knob-grid` unless you also register a resolver, otherwise the new axis
is un-deduplicated by construction."*

### 5.3 The aom byte-identity constraint, and the gate that protects it

Every aom-rs cell requires the port's frame-OBU payload to be byte-identical to the C
oracle's (`encode.rs:1286-1295`). The A0 arm bootstraps from `c_encode_defaults()` —
"a plain `aomenc --allintra` encode with NO coding-tool flags". The knob arm must drive
the C side too, via `EncodeCell::c_encode_ctrls(ctrls)` — whose base is
`c_encode`, i.e. **`--enable-restoration=0`, a NON-default config**.

**Pre-registered gate G-AOM-BASE (must pass before A3 is declared):** on 2 corpus
images × 2 q, `c_encode_ctrls([(AV1E_SET_ENABLE_RESTORATION, 1)])` must produce a stream
whose sha256 equals `c_encode_defaults()`. If it does, the knob arm sits on the same
base as the control and the two are comparable; if it does not, **A3 is not declared**,
the finding is recorded, and the aom lane reverts to control-only. This is
"every exclusion must be proven by encode" applied to an *inclusion*.

**Pre-registered gate G-AOM-ARM:** the 23 arms are smoke-run on 2 images × 2 q × 1 speed
before the fleet declaration. Any arm producing a byte divergence is **dropped from the
declaration with its error class recorded** and handed to the KB-41 lane as a port lead
— never silently retried, never left to fail 400 cells deep.

---

## 6. Blocking finding: nothing is scoring

Measured 2026-09-02T01:35Z → 01:55Z, from the gap-fill loop's own log: encode blobs
2,241 → 3,269 (svt) and 718 → 969 (aom), while **score blobs stayed at 29 and 7**. No
score worker process exists on the dev box, and r7900x runs only `avifsub-svt-r7900x`
and `avifsub-aom-r7900x` (both encode). The score runs are declared and their manifests
are being refreshed every 5 minutes by the gap-fill loop, but **nothing is consuming
them**.

Consequences and actions:

1. Quality is the DOE's dependent variable. Encoded bytes without scores are half a
   measurement. **Score workers are launched as part of this wave's rollout**, on
   capacity freed by the aom shrink, and the score fill is reported alongside the encode
   fill from here on.
2. This is why §2.3 does not anchor on baseline scores.
3. Score cost is **per variant and independent of the encode speed knob**, so it scales
   with *cells*, not CPU-hours: 56,080 cells × 3 CPU metrics + a zensim feature vector.
   At an assumed ~3 s/variant that is ≈ 47 CPU-h — comparable to the whole encode
   budget, and the reason A2 was cut to 16 images rather than 32.

---

## 7. Analysis plan and Stage-B triggers (registered now; Stage B runs after A)

### 7.1 Analysis

1. **Per (image, arm, speed): an RD curve** over the 9-point ladder — `(bytes, ssim2)`,
   with zensim and butteraugli as secondary responses. `ssim2` is the primary quality
   response (the standing north star for non-photo content, and 11/32 picks are
   plots/screenshots).
2. **Effect size = BD-rate of the arm against the `deviations = 0` control at the same
   (image, speed) AND THE SAME SIZE**, integrated over the ladder's quality span.
   The same-size requirement is why A0R exists: differencing a budget-size knob
   cell against a native-size default would fold the crop into the knob effect.
3. **Main effects** = per-arm BD-rate distribution over the 32 images, reported as
   median and IQR, per speed and per content class.
4. **Interactions** = observed pairwise BD-rate minus the additive prediction from the
   two main effects, per image.
5. **Speed** is modelled separately from A4: `encode_ms = α + β·pixels` per
   `(backend, speed, arm)`, both terms reported, never a bare "ms/MP".
6. **Categoricals stay categorical**: `sharpness` and `tune` are factors, never fitted
   as ordinal trends (**H-14**, and libaom's `sharpness` quantizer plateaus).
7. **Duplicate rows are removed before fitting** — any row whose `output_sha` already
   appears for the same `image_path` (the 715+ pre-flatten alias burns; retrofit §7).

### 7.2 Stage-B triggers (mechanical; the training lane may execute them)

| id | trigger | follow-up |
|---|---|---|
| **B-1** | knob `k` has `median_image \|BD-rate\| ≥ 1.5 %` **or** `IQR ≥ 3 %` at any speed | dense grid on `k`: 5 levels × the full 29-q ladder × 32 images × speeds {4,6,7} |
| **B-2** | pair `(k1,k2)` deviates from the additive prediction by `≥ 1.0 %` BD-rate on `≥ 25 %` of images | full `k1 × k2` grid, 3 levels each, 9-q, 32 images |
| **B-3** | `k`'s BD-rate median has **opposite signs** in ≥ 2 content classes, each `\|median\| ≥ 1 %` | content-stratified dense follow-up + an explicit interaction term in the model |
| **B-4** | the A4 speed fit's held-out MAPE `> 25 %` for any `(backend, speed)` | extend the timing block's size ladder / knob coverage until ≤ 25 % |
| **B-5** | any A1 main effect **inverts sign** across the preset ladder (median per-image BD-rate of opposite sign at two presets, each `\|median\| ≥ 1 %`) | that knob is fitted with an explicit preset interaction, and its Stage-B grid is run at both the inverting presets |
| **B-6** | an arm fails **T1** of the cross-size gate (§3.8) | that knob's Stage-B grid runs at **native** size; its A1/A2 numbers stay annotated as size-conditional |

**Stage-B budget envelope:** ≤ 60,000 cells, ≤ 60 CPU-h encode, ≤ 50 CPU-h score,
declared as `avifdoe-{svt,aom}-b-<date>` through the same canonical builder.

### 7.3 Pre-registered de-scope rule (budget gate)

Shed in this fixed order, each step a declaration shrink per §4.4 (reversible, no
ledger row lost), re-checking after each:

1. **A2 → the 16-image class-stratified subset** (§9). Halves the largest block
   for the smallest scientific loss; the subset is pre-registered so the choice
   is not made after seeing results.
2. **A1 → drop speed 1 (preset 0).** It is 8.75 of A1's 12.1 CPU-h on its own,
   and the axis is already ordered so `collapse_one_axis` takes it first.
3. **A1 → drop speed 3 (preset 3).** Next most expensive; presets 1, 4, 7, 9
   still span the ladder.
4. **A3 → drop `cpu-used 4`.** 7.1 of A3's 9.1 CPU-h.

Triggers: measured throughput 6 h after launch implying **> 4 days** to drain,
or **G-SCORERATE** landing scoring above 50 CPU-h. **q density is never cut**
and the knob set is never cut — those are the two things §2.4's authorization
exists to protect.

---

## 8. Known gaps this wave does NOT close (recorded, not hidden)

| gap | why not here | value |
|---|---|---|
| **aom `--tune=iq` / `ssimulacra2`** | fully ported and byte-gated in `aom-encode`, but absent from `aom_bench::ToggleKnobs`; wiring it means editing the **`zenav1-aom` repo**, which this lane does not touch | **the single highest-value knob on either backend** — libavif's *default* for colour on libaom ≥ 3.13.0, so today's aom-rs cells measure a configuration the ecosystem does not ship |
| aom `deltaq-mode 6` (Variance Boost) | reachable only through `tune=iq` or a `ToggleKnobs` field that does not exist | the direct analogue of SVT's headline still-image lever |
| svt `--enable-cdef` / `--enable-restoration` / `--enable-palette` | PORT-MISSING in `zenav1-svt` (SH bit hardcoded / preset-derived); the dossier rates all three "shallow" to wire, but it is another repo | libaom's all-intra default **disables CDEF for images** — a directly testable cross-backend hypothesis SVT cannot currently answer |
| aom `coeff/mode_cost_upd_freq` | present but **port-side inert**; a sweep would produce false nulls | needs the `pack.rs` split its own doc comment describes |
| **U-2** (is SVT `aq_mode=2` really a no-op on a still?) | needs a C SVT-AV1 build; the port asserts it, nobody measured it | decides whether the port's `aq_mode` refusal is a capability gap or a no-op |
| a resample/size axis on the RD blocks | the 32 picks already span 0.25–16 MP; a second size axis multiplies the aom budget rather than adding to it | A4 carries the size axis where it is mandatory (timing) |

---

## 9. Frozen inputs

**A2 16-image FALLBACK subset** — A2 now runs on all 32 (§3.4); this list is the
pre-registered first step of the de-scope ladder (§7.3), fixed now so it cannot
be chosen after seeing results. Rule: largest cluster of each of the 12 content
classes, then the 2 smallest-MP and 2 largest-MP remaining picks; 12/12 classes,
MP 0.25–16.00, 60.2 % population coverage:

```
1008.scale3000x4000.png  1220.scale3000x4000.png  1420.scale3000x4000.png
1432.scale3000x4000.png  1634.scale3000x4000.png  3006.scale3000x2235.png
6038.scale2479x3230.png  6602.scale3302x4844.png  6604.scale3286x4868.png
7076.scale1024x1024.png  8288.scale375x667.png    8434.scale414x896.png
8446.scale2560x1440.png  9032.scale1024x1536.png  9100.scale1024x1536.png
9954.scale1024x1536.png
```

**Run names**

| run | contents | corpus |
|---|---|---|
| `avifsub-svt-enc-20260901` / `avifsub-aom-enc-20260901` | A0 native control (existing; aom shrunk in place) | `codec-corpus/avif-subsample-2026-09-01` |
| `avifdoe-svt-a0r-20260901` / `avifdoe-aom-a0r-20260901` | A0R same-size control | `codec-corpus/avif-doe-1024-2026-09-01` |
| `avifdoe-svt-a1-20260901` | A1 main effects | budget corpus |
| `avifdoe-svt-a2-20260901` | A2 pairwise | budget corpus |
| `avifdoe-aom-a3-20260901` | A3 aom main effects | budget corpus |
| `avifdoe-svt-ag-20260901` | AG cross-size transfer gate | native corpus |
| `avifdoe-*-sf-cpu-20260901` | score runs (`ssim2,zensim,butteraugli` + a zenanalyze feature vector per variant) | — |

**Budget corpus** (`s3://codec-corpus/avif-doe-1024-2026-09-01/`, mirrored at
`/mnt/v/output/avif-doe-1024-2026-09-01/`): 32 references — 9 native symlinks
(MP ≤ 1.05) and 23 `<origin_id>.crop1024.png` content-aware crops — plus
`crop_manifest_2026-09-01.tsv` carrying, per pick: `origin_id`, `transform`
(`native` | `crop-native`), `crop_rect` (x, y, w, h), `source_sha256`,
`crop_sha256`, the parent's cluster id, the crop's **re-extracted** 84-feature
vector, its distance to the parent's vector, its own nearest centroid, and a
`class_shift` flag where those differ.

## 10. Execution status

**Done and verified.**

| step | evidence |
|---|---|
| Design registered | this document, `6bc743b8` on `zenmetrics master@origin` |
| A0-aom shrunk 9,280 → 5,179 | `declare-encodes` on the flattened cells; verified a **strict subset with byte-identical retained jobs** (`new-is-subset-of-old: True`, `NEW-not-in-old: 0`, `removed: 4,101`) ⇒ no `JobId` moved, every completed cell reused. 483 of 793 done cells fall inside the new grid; the other 310 keep their ledger rows and blobs, they are simply no longer declared. |
| Prior manifests preserved | `s3://zentrain/jobs/avifsub-aom-enc-20260901/manifest.pre-doeshrink-2026-09-01.json` (and the earlier `manifest.pre-flatten-2026-09-01.json`), plus a local `.bak` |
| Workers picked it up | aom workers restarted on r7900x + tower + local; the local pair relaunched **cpuset-capped (6 cores each), `nice -n 19 ionice -c 3`**, with `--control-r2-key` so this lane's half of the wave IS pausable |
| `control.json` seeded | `{"paused":false,"drain":false}` for both encode runs |
| Score-side stall found | §6 — score blobs flat at 29 (svt) / 7 (aom) across 20 minutes while encode blobs grew by 1,279; no score worker running anywhere |

**Design revised mid-flight** (2026-09-01, user directive) to make the per-image
pixel budget a free variable set by the permutation count — §2.4, §3.8, §3.9.
Net effect: the knob space **grew** rather than shrank. A1 went from 3 presets to
all 7; A2 went from 16 images back to 32; A3 went from one dense speed plus two
3-q probes to three speeds at full 9-q; A1b was folded into A1 and dropped; a
same-size reduced control (A0R) and a cross-size transfer gate (AG) were added.
≈ 3,300 permutations per image, ≈ 70.5 CPU-h of encode, 117,435 cells.

**Implemented, NOT yet compiled or declared** (see the blocker below):

- `zenavif`: `expert::SvtParams` (the nine §3.2 knobs, with `clamped()` for
  hazard H-10 and `resolved(qp)` transcribing the port's own
  `apply_tune_overrides`); `EncoderConfig::{with_svt_params, svt_params}`;
  `encoder_svt_rs::apply_svt_params` on the colour pipeline; the
  `SweepAxes::svt_knobs` axis with per-set deviation counting, the
  `-tn/-vbst/-qml/-shp/-scm/-acb/-mtx/-tl` id grammar and its parser, the
  resolved-state fingerprint extension, and the `svt_doe_main` /
  `svt_doe_pairwise` plans. Eight new tests, including the two invariants that
  make the axis safe to add (a default knob set moves **no** id and **no**
  fingerprint) and the H-4 merge (`tune 3` absorbs the knobs it forces).
- `zenmetrics`: `scripts/jobsys/avifdoe_declare.sh` (the declare path + G-DEDUP
  readout), `scripts/jobsys/avifdoe_build_budget_corpus.py` (the §2.4 budget
  corpus: cluster-preserving 1024² crops, native passthrough below budget, the
  full provenance manifest, and a **loud non-zero exit** when the G-CROP feature
  re-extraction has not run), and the zenavif plan-name error string.
  **No other zenmetrics change is needed**: `build_zenavif_plan` dispatches plan
  names generically through `zenavif::sweep::SweepAxes::by_name`, so the two new
  plans are reachable from `--plan` without touching the executor.

**Known gap in the budget-corpus builder, stated rather than papered over:** it
selects the crop *position* from the existing imazen26 feature parquet's
per-position crop rows, which is principled and needs no new tooling — but the
**crop's own 84-feature re-extraction** (G-CROP) needs the imazen26 extractor
pointed at the new PNGs, which this lane could not run. The script refuses to
report success without it. The per-variant feature vectors the score jobs emit
are from the *encoded pair*, so the tuning model's own features already describe
the crop and never the parent; what is outstanding is the cluster-assignment
sanity check, not the model's inputs.

**BLOCKER (environment, not the design): the dev box's `/tmp` filesystem hit its
quota and every shell invocation now fails.** Confirmed by a direct
`EDQUOT: write` on a `/tmp/claude-*/scratchpad` file; the harness appends
`&& pwd -P >| /tmp/claude-*-cwd` to every command, so an unwritable `/tmp` turns
into a bare `exit 1` with no output on *every* command — including `true`.
Consequences and state:

- `cargo build` / `cargo test` / `jj commit` cannot run, so the zenavif change
  above is **unverified and uncommitted** (it is on disk under
  `/home/lilith/work/zen/zenavif`, which is a different, healthy filesystem).
- The A0R / A1 / A2 / AG declarations have **not** been made, and the budget
  corpus has **not** been built. The A0 shrink landed before the outage and is
  the wave's live state: `avifsub-svt-enc-20260901` draining the native control
  (3,562 of 6,496 done, ~3,036 cells/h ⇒ ~1.2 h left) and
  `avifsub-aom-enc-20260901` draining the shrunk 5,179-cell grid on r7900x +
  tower + the two cpuset-capped local workers.
- **A3's aom knob threading is not written.** aom-rs has no `EncoderConfig` —
  it is driven straight from `zenmetrics-cli` — so each arm must be applied to
  **both** the port's `ToggleKnobs` and the C oracle via
  `EncodeCell::c_encode_ctrls`, with a resolver registered in `sweep/dedup.rs`,
  and gates G-AOM-BASE + G-AOM-ARM run first. Registered here; not started.
  The svt arm carries the DOE if A3 slips.
- The r7900x and tower workers are unaffected — they are separate boxes and
  keep draining the shrunk A0 grid.
- **The two local workers are spinning uselessly and must be killed first.**
  MEASURED from `~/tmp/avifdoe/local_aom_worker.log`: every chunk fails with
  `ledger write … failed: parquet External: Disk quota exceeded (os error 122)`,
  dozens in a row. They claim a chunk, spend CPU encoding it, then cannot write
  the ledger row — so the work is discarded, the claim expires, and the cell
  returns to the gap for a remote worker. **No data is lost** (the ledger is
  append-only, blobs are content-addressed, claims are leases), but the local
  box is burning CPU for nothing and is a prime suspect for exhausting the quota
  in the first place. Kill by recorded PID
  (`~/tmp/avifdoe/local_{svt,aom}_worker.pid`) — never `pkill -f`, which
  self-matches the invoking shell — and do not relaunch without pointing
  `TMPDIR` at a filesystem with room. r7900x and tower are unaffected and have
  been draining throughout.
- Recovery, in order (full command sequence:
  `~/tmp/avifdoe_RESUME_AFTER_SHELL.md`):
  1. kill the local workers, then free space on the `/tmp` filesystem;
  2. `cargo test -p zenavif --features sweep,encode-svt-rs,__expert`;
  3. `scripts/jobsys/avifdoe_build_budget_corpus.py` + the G-CROP extraction,
     then upload the budget corpus to
     `s3://codec-corpus/avif-doe-1024-2026-09-01/`;
  4. `scripts/jobsys/avifdoe_declare.sh --smoke`, then the full declare;
  5. **start score workers** — nothing has been scoring (§6), and the DOE's
     dependent variable is quality.
- One verification the outage prevented: whether any `SweepAxes { .. }` struct
  literal exists outside `src/sweep.rs` (examples/tests). Four in-crate literals
  were found and updated by hand; the compiler will name any others.

**Gates that must pass before each fleet scale-up** (two-stage launch):

| gate | what it checks |
|---|---|
| G-DEDUP | `--dry-run` flatten count vs the naive cross product; every merge explained by a fingerprint mediator or a dossier hazard, recorded in `<run>.plan.json` |
| G-FIRSTCELL | after the 1-worker smoke, the first cell's **encoded bytes AND its score row** are listed in the store with the `aws` CLI (never s5cmd — it undercounts on the LAN store) |
| G-AOM-BASE | §5.3 — the ctrls path reproduces the defaults base byte-for-byte |
| G-AOM-ARM | §5.3 — all 23 aom arms byte-match on the smoke set, or are dropped with their error class recorded |
| G-CLAMP | `variance_boost_strength ∈ 1..=4`, `variance_octile ∈ 1..=8` enforced in the harness (**H-10**) |
| G-CONTROL | every DOE worker launched with `ZEN_CONTROL_KEY` set, so the wave is pausable (unlike A0's) |
| G-CROP | every budget-corpus reference has a manifest row with its crop rect + both sha256s, its features **re-extracted from the crop**, and its cluster assignment checked against the parent's — with any class shift recorded, not silently accepted (§2.4) |
| G-XSIZE | §3.8's T1/T2/T3, run on the AG cells before any A1/A2 conclusion is published. An arm failing T1 is **not** screened at reduced size |
| G-SCORERATE | the realised per-variant score cost, measured on the first 200 scored variants, re-checked against the §3.7 budget (§7.3 fires on a miss) |

---

## 11. Execution record — 2026-09-02, the post-outage recovery lane

Everything below is measured, in the order it happened. Two gates fired and one
of them stopped the wave; both are recorded with what they caught.

### 11.1 zenavif compiled for the first time

The knob axis had never been compiled. It needed 10 fixes, all mechanical, none
touching the design: 4× `E0063` (test-local `Stratum` literals missing the new
`svt` field — all `Av1Backend::Zenravif` cells, where `SvtParams::default()` is
inert, so their assertions are unchanged), 1× `E0382` (the safety-invariant test
borrowed a config after `with_svt_params` consumed it), and 5×
`clippy::field_reassign_with_default` carrying the same `#[allow]` and rationale
the sibling test already had. Result: **clippy `-D warnings` clean, 438 tests
pass, 0 fail, rustfmt clean** (zenavif `dd61d459`).

Both registered safety invariants hold, and the design constants this plan
asserts are checked by the tests and hold: **17** single-deviation knob sets,
**17+120 = 137** pairwise sets, **24** strata at one q for `svt_doe_main`.

⚠ **The resume file's build command is wrong and will mislead the next reader**:
it names a `sweep` feature, which is a *zenmetrics* feature — zenavif has none,
and gates `pub mod sweep` on `__expert`. The working invocation is
`cargo test -p zenavif --features encode-svt-rs,__expert`.

### 11.2 Budget corpus — built, but it is NOT the corpus §2.4 describes

`avifdoe_build_budget_corpus.py` produced **32 references, 13 cropped + 19
native**. §2.4 and the resume file both expect **23 cropped + 9 native**. The
cause is the passthrough condition:

```python
if mp <= BUDGET_MP or w <= args.side or h <= args.side:
```

An image passes through **natively if EITHER dimension is ≤ 1024**, regardless of
its pixel count — which is a different rule from the comment above it ("never
crop something already at or under budget"). So every `1024×1536` / `1536×1024`
pick (1.57 MP, 50 % over budget) and `1440×900` (1.30 MP) is encoded at native
size. **Measured: 37.02 MP total against the 33.55 MP the budget implies —
+10.3 %, across 10 of 32 references.** That is a modest overshoot, not a
wave-invalidating one, but §3.9's `α + β·pixels` fit and every CPU-h figure in
§3.7 are stated against 1.048 MP references and should be read with it.

Two references (`8288.scale375x667`, `8434.scale414x896`) are recorded
`below_tile_floor=true`: below AV1's 512 px-per-axis floor for a 2×2 tiling, so
**the tile axis is not measurable on them**. That is correctly recorded in the
manifest rather than silently degraded.

The corpus is uploaded and verified with the `aws` lister: **32 objects** at
`s3://codec-corpus/avif-doe-1024-2026-09-01/`.

### 11.3 G-CROP: NOT SATISFIED — and the flag that is supposed to satisfy it is a hole

The builder exits non-zero with `*** G-CROP NOT SATISFIED ***`, which is correct:
the crops' own features have not been re-extracted. **No A1/A2/A3 conclusion may
be published until they are.**

But the escape hatch is worse than the gate. `--features-cmd` is documented as
the way to satisfy G-CROP, and the manifest records `feature_recheck="todo"`
when it is passed and `"PENDING"` when it is not — while the gate fails only on
`"PENDING"`. **`avifdoe_build_budget_corpus.py` never executes the command: the
script imports no `subprocess` and calls nothing.** So passing `--features-cmd`
flips the manifest to a passing value and satisfies the gate *without extracting
a single feature* — on the one gate this plan calls "the one place the wave can
silently change what an image is". Not used here; the corpus carries an honest
`PENDING`. The flag must either run the command and perform the cluster check,
or be removed.

### 11.4 Declaration — 49,120 cells, and G-DEDUP explained

| run | plan | q | cells | merged |
|---|---|--:|--:|--:|
| `avifdoe-svt-a0r-20260901` | `svt_speed_dense` | 29 | 6,496 | 87 |
| `avifdoe-svt-a1-20260901` | `svt_doe_main` | 9 | 6,912 | 0 |
| `avifdoe-svt-a2-20260901` | `svt_doe_pairwise` | 9 | 33,984 | 171 |
| `avifdoe-svt-ag-20260901` | `svt_doe_transfer` (native corpus) | 3 | 1,728 | 0 |

`invalid_skipped: 0` on all four — the confirmation that `avif-svt` is live in
the declaring build, since without it every svt stratum would land there.

**G-DEDUP PASSES, but the script's own warning needs a correction.** It says a
knob plan reporting `duplicates_merged: 0` is a red flag. That is true only for
plans that can *express* a collision. `svt_doe_main` and `svt_doe_transfer` run
at `--max-deviations 1`, where no cell spells out two knobs at once, so hazard
H-4 has nothing to merge and **0 is structurally correct there, not a warning
sign**. Where the collision is expressible it fires exactly as designed: in the
pairwise plan `s6-svt-420-tn3` (tune=IQ) absorbed **9** cells that spell out the
knobs tune 3 forces — `vbst1.2.5`, `vbst1.3.5`, `qml1.4.10`, `qml1.2.10`,
`qml1.8.15`, `shp3`, `shp7`, `scm3`, `mtx32` — visible in the plan's own
`aliases` record.

### 11.5 G-FIRSTCELL FAILED, and what it caught

First stage-1 worker on r7900x, one worker, A1 only: **215 ledger rows, 0 blobs,
every row `status: failed, error_class: encoder_panic`.** The gate did exactly
its job — this was caught on one worker, not on thirty.

Root cause, established with a control arm rather than assumed: the fleet image
`ghcr.io/imazen/zenfleet-worker:exec-avifsub-svtaom-4a6876b9` **predates the knob
axis**. Given a real source file it answers
`unknown zenavif plan "svt_doe_main"; expected rd_core, modes_full,
modes_full_alpha, or scalar_dense`. (A first attempt to prove this was
inconclusive — with an empty sources dir *both* the real and a deliberately
bogus plan name return `no source files found`, because the source check runs
first. The control is what showed the test was measuring nothing.) The
counter-evidence: the same cells encode **24/24, encode-fail=0** through a
freshly built binary. So the code was fine and the image was stale — the
opposite of the obvious reading.

**Fix: a new image**, built from a musl-static zenmetrics + zenfleet-worker
carrying zenavif `dd61d459` and zenmetrics `b0bb8340`, published as a new **tag
on the existing package** (per the one-package-many-tags rule):

```
ghcr.io/imazen/zenfleet-worker:exec-avifdoe-svtknobs-b0bb8340
sha256:a948e4dfabfbdbacd6f2d0b1ff4d5e4520c6b0753173587555ae2b151b1c4be9
```

Verified before relaunch: it resolves all four new plans, its control arm
correctly rejects a bogus name *and now lists all eight plans*, and both
binaries report `statically linked` (the documented glibc trap avoided).

**G-FIRSTCELL then PASSES**: 39 blobs in the first 45 s, and the first blob is a
well-formed AVIF — `ftypavif` + `mif1miaf` magic, `file(1)` "ISO Media, AVIF
Image", 241,518 bytes.

**G-CONTROL is verified working, not merely configured**: `control.json` was
flipped to `paused` mid-incident to stop the failing wave, and the workers
honoured it. This is the capability the A0 wave lacks.

Two image-build gotchas worth baking into the runbook: r7900x's docker has **no
buildx**, so `Dockerfile.executor`'s `COPY --chmod` fails on the legacy builder
(fixed by installing the buildx plugin at user level — the script header's own
first-choice remedy); and `build_executor_image.sh` treats the
`zenfleet-worker` overlay as optional ("image keeps base's") while
`Dockerfile.executor` **COPYs it unconditionally**, so omitting it fails the
build rather than falling back.

### 11.6 Scoring — the §6 blocking finding is cleared

A capped score worker now runs on tower (`avifsub-score-svt-tower`,
`--cpuset-cpus=22-23 --cpu-shares=256 --memory=16g`, leaving 8 cores free for
the household per the tower-safety rule). `avifsub-svt-sf-cpu-20260901` score
blobs moved **29 → 36 and rising** within minutes, after being flat at 29 for
the whole preceding wave. Nothing about the score runs needed changing — they
were correctly declared and refreshed, and simply had no consumer.

### 11.7 The stale image poisoned 93 % of A1, and the pardon that recovered it

The 215 ledger *files* the stale worker wrote were not 215 cells. Measured after
the fix, with `zenfleet-ctl report`:

```
avifdoe-svt-a1-20260901: declared=6912 ever_done=480 live_done=480 failed-only=6432
```

**6,432 of 6,912 cells — 93 % of the main-effects arm — were poisoned by an
image defect**, and a fresh worker on the *correct* image drained straight out
(`done=0 failed=0 ... already_poison=6432`, then `drained (idle 12 consecutive
no-work passes) — exiting clean`). A run can therefore be fully unblocked at the
code level and still be dead at the ledger level; **fixing the image is not
finishing the recovery.**

Recovered with `zenfleet-ctl requeue`, which exists for exactly this ("the
faulty-worker era WAS the fault"), scoped rather than blanket:

```
zenfleet-ctl requeue --run avifdoe-svt-a1-20260901 \
    --classes encoder_panic --before <2026-09-02T03:50:00Z>
→ appended 6432 pardon rows (failed/worker_lost/attempts=0)
```

The `--before` scope was **verified, not assumed**: cutoffs at 03:50Z, 03:55Z
and 04:00Z all return exactly 6,432, identical to the unscoped count, which
proves every failure predates the fix and the corrected era contributed none —
so the pardon cannot mask a genuine current-era failure. `attempts=0` with a
TRANSIENT class is the reconcile path's own retry vocabulary; a PENDING pardon
row would deadlock (the documented `stale_claim_after=None` class), and
`reassert` is the wrong tool here — it only reverses buried *done* rows and
reports 0 for this run. Genuine failures re-poison with current-era evidence
after exactly one retry, so nothing is permanently laundered.

**Verified working, not merely applied**: A1 blobs resumed **458 → 545** on the
next worker.

### 11.8 Fleet snapshot and what is still open

```
SNAPSHOT 2026-09-02T04:49Z
run                                   blobs   ledger   declared
avifsub-svt-enc-20260901               6293      295       9600
avifsub-aom-enc-20260901               3030      100       5179
avifdoe-svt-a0r-20260901                  0        0       6496
avifdoe-svt-a1-20260901                 604      290       6912
avifdoe-svt-a2-20260901                5101       73      33984
avifdoe-svt-ag-20260901                   0        0       1728
avifsub-svt-sf-cpu-20260901             103       15          -
avifsub-aom-sf-cpu-20260901             208        5          -
```

- **G-CROP remains PENDING** (§11.3). Encoding is not gated on it; **conclusions
  are.**
- **A0R and AG have manifests + control keys uploaded but no workers yet.** A1
  runs on r7900x, A2 on a local worker.
- **Local workers are back, with the outage's fix applied**:
  `TMPDIR=/home/lilith/tmp/zfw-scratch`, so `jobexec`'s never-evicted source
  cache stages on `/home` (822 GB free) instead of the quota'd `/tmp`. Verified:
  1.2 GB staged in the new scratch within a minute, **`/tmp` flat at 1 %, and
  zero `Disk quota exceeded` lines** where the pre-outage run logged dozens.
- **A3 (the aom knob arms)** is untouched, as §5.3 registers.

---

## 12. Gap closure — 2026-09-02, the three gaps §11 left open

§11 closed the recovery honestly and named what it could not finish. This
section closes those three. Everything below is measured; where a decision was
made instead of a measurement, the deciding fact is stated first.

### 12.1 G-CROP is CLEARED, and the escape hatch is gone

**The hole, restated.** `--features-cmd` was documented as satisfying G-CROP and
flipped the manifest from `PENDING` to `todo` when passed — while the script
imported no `subprocess` and executed nothing. The gate failed only on
`PENDING`, so the flag cleared it without extracting a feature.

**Fix: the flag is REMOVED, not repaired.** A per-image "print 84 numbers"
command was the wrong shape anyway — the canonical extractor
(`zenanalyze/examples/extract_features_imazen26_crops`, the binary that produced
the population parquet) is manifest-driven and batch. So
`avifdoe_build_budget_corpus.py` now takes:

- `--extractor <path>` — the canonical binary. The builder writes a manifest of
  its own 32 outputs, **runs it once** (`--sizes "" --crop-fractions ""` ⇒ one
  `full`/`native` row per reference), and treats a non-zero exit, a missing
  output, or a short row count as fatal.
- `--cluster-model <npz>` — the fitted clustering geometry, emitted by a new
  additive `--out-model` on `scripts/imazen26_recluster_even.py` (the clustering
  owner): content-feature names, the NaN-fill medians, the surviving-column
  mask, `mu`/`sd`, and the 32 k-means centroids. Assignment happens **in that
  space**; re-standardising locally would have answered a different question,
  because the builder's own z-scoring spans a different population (all
  parities, all crop labels) than the clustering's (train-even, `full` only).
- `--native-drift-tol` (default 0.5).

The two must be given together, and neither alone clears the gate.

**The gate now validates itself, which is what makes it un-fakeable.** 19 of the
32 references are symlinks — *bit-identical* to the parents that were clustered.
So re-extracting them must reproduce the parents' own vectors and clusters.
Those 19 rows are a control the extractor cannot satisfy by accident: a missing,
no-op, wrong-schema or garbage extractor fails them, and the builder exits
non-zero. A crop that moves is **recorded, not fatal** (§2.4 asks for the shift,
not for its absence); a *native* that moves is fatal.

`scripts/jobsys/test_avifdoe_budget_gate.py` pins it shut — 7 hermetic cases,
all passing, no corpus or network: a missing binary, `exit 3`, **`exit 0` that
extracts nothing (the original hole, verbatim)**, a short row count, honest
features passing, a native landing in the wrong cluster, and a native inside the
right cluster but beyond the drift tolerance (so both control arms are shown
load-bearing, not just one).

**The extractor-drift control, run first because the rest is meaningless
without it.** The population parquet is from 2026-06-13; today's zenanalyze is
`18971ef`. Re-extracting the 32 **parent** images and diffing against their
stored rows over the 97 shared content columns:

| | |
|---|---|
| cells differing by > 1e-6 relative | **33 of 3,104** |
| worst column | `edge_slope_stdev`, 2.68e-2 rel (19.341875 today vs 18.836258 stored) |
| **displacement in the clustering's z-space** | **max 0.0253, mean 0.0012** |

Cluster distances are of order 2–7 and decision margins of order 0.3–8, so
extractor drift is 1–2 orders of magnitude too small to move an assignment. The
drift is real but not load-bearing, and that is a measurement, not an assumption.

Schema note: the parquet carries **110** `feat_*` columns and today's extractor
emits **117**, with the 110 an exact name-and-order prefix of the 117 — textbook
append-only growth (`chroma_subsample_dct_loss` + six `highlight_*` from the HDR
tier). The comparison is defined over the shared 110 only; the 7 new ones have
no 2026-06-13 counterpart.

**Result.**

```
G-CROP: features 133b93d80c15581f… over 32 references; native control PASSED (19/19)
G-CROP: 2 of 13 crops changed cluster (recorded, not fatal):
  1442.scale4000x3000.png  cluster 0 -> 25  (parent_z_dist 24.4071, margin 0.3021)
  1634.scale3000x4000.png  cluster 25 -> 10 (parent_z_dist 4.5834,  margin 1.6077)
```

**The two shifts, and what they mean.** Neither is a defect; both are the crop
doing exactly what §2.4 said it would.

- **`1442.scale4000x3000`** (4000×3000 night aurora, `1400-lilith-nature`) — its
  parent cluster **0 was a SINGLETON**, i.e. the parent was an outlier no other
  image resembled. The 1024² crop is **24.41 z-units** from the parent and lands
  in cluster 25 (n=78) with a margin of only **0.302** — a boundary assignment.
  Cropping removed precisely the whole-frame structure that made it an outlier.
  This is the significant one: it is the pick whose *representativeness* the crop
  most changed, and any A1/A2 conclusion that leans on it should say so.
- **`1634.scale3000x4000`** (`1600-lilith-food`) — 4.58 z-units, cluster 25 → 10
  (n=42), margin 1.608. An ordinary interior reassignment.

**"Preserved" is not the same as "unchanged", and one row proves it.**
`6604.scale3286x4868` is recorded PRESERVED, but its parent's cluster 26 is also
a singleton, so the centroid *is* the parent and the distance to it is the
displacement: **67.72** — the largest content change in the corpus, in a row the
verdict column calls preserved. Read `parent_z_dist` beside `feature_recheck`,
never `feature_recheck` alone. (The identity `cluster_dist == parent_z_dist` on
the singleton rows is also the internal check that the replayed geometry is the
clustering's own.)

**Provenance, recorded in three places.** The manifest gains six append-only
columns — `source_path`, `assigned_cluster`, `cluster_dist`, `cluster_margin`,
`parent_z_dist`, `features_sha256` — with the original 14 keeping their
positions so any positional reader still parses. Alongside it,
`gcrop_provenance.json` carries the extractor path and full argv, the features
sha256, the cluster-model sha256, the population parquet, the tolerance, and the
shift list. Both, plus the model and the feature table, are uploaded:

| artifact | sha256 |
|---|---|
| `_gcrop/budget_features_2026-09-02.tsv` (32 × 117 feat) | `133b93d80c15581f6eeb6a6e29995f3589178d2f22ca2710210fe0e891ddf754` |
| `kmeans_K32_full_even_2026-09-02.npz` | `a1211b950e7c7d36fe8e059d2918c5fd6167cc3ae6b6400b6d8b486edbf4ee7e` |

R2: `s3://codec-corpus/avif-doe-1024-2026-09-01/` now holds the 32 sources plus
`crop_manifest_2026-09-01.tsv`, `gcrop_provenance.json`,
`kmeans_K32_full_even_2026-09-02.npz`, `_gcrop/budget_features_2026-09-02.tsv`.

**Two independent reproductions, both byte-exact.** The `--out-model` extension
to the clustering owner is **inert**: replaying the registered command reproduces
`reps_K32_full_even_2026-09-01.tsv` byte-identically (sha
`1ce1dbcd90ffc690b4f7330cfac9ca595be8a00b6036dcc4c5e2e3d212fb1a29`), with the
same `units=1082 content_feats=82`, 0 odd-id reps and 4 singleton clusters. And
the fixed builder, run into a scratch directory first, reproduced the live
corpus exactly — **32/32 on-disk sha256 identical, 0 differences across the 9
identity columns** — before anything in the live directory was touched; the
extraction itself is deterministic across two runs with different working
directories (**0 differing cells over 32 × 117**). The pre-gate manifest is kept
as `crop_manifest_2026-09-01.tsv.pre-gcrop.bak`.

**G-CROP is CLEARED. A1/A2/A3 conclusions are no longer blocked by it** — with
the two shifts and the `6604` displacement carried into any write-up.

### 12.2 The corpus deviation is AMENDED, not rebuilt — and the deciding fact is 13,532

**The deciding fact, measured before the decision.** Blobs already encoded
against the *current* budget-corpus references, 2026-09-02T04:53Z:

| run | blobs |
|---|--:|
| `avifdoe-svt-a1-20260901` | 902 |
| `avifdoe-svt-a2-20260901` | 12,630 |
| **total** | **13,532** |

and rising while the fleet ran. `avifdoe-svt-a0r-20260901` and
`avifdoe-svt-ag-20260901` had 0. Twenty minutes later, at 05:13Z, the same two
runs read **3,481 + 21,902 = 25,383** — the number a rebuild would have had to
discard grew by 11,851 during this lane alone, which is the direction of the
decision, not a coincidence.

**Cell identity is content-addressed on the reference bytes — verified, not
assumed.** For the same `image_path` `1008.scale3000x4000.png`, the declared
`inputs` sha is `06fc50b2…` in A1/A2/A0R and `7c36c631…` in AG; those are exactly
that row's `crop_sha256` and `source_sha256`. So a rebuilt reference produces
*new* cells: the 13,532 existing blobs would be **orphaned, not silently wrong**.
That removes the correctness argument for rebuilding and leaves only the cost
one — and the cost is 13,532 encodes plus a redeclaration of three runs, on a
wave that had just recovered from a 93 %-poison incident (§11.7).

**What the deviation actually costs.** The 10 affected references:

| reference | geometry | MP | superblock-exact? |
|---|---|--:|---|
| `9444` `9954` `9074` `9032` `9136` `9100` (1024×1536) | 1024×1536 | 1.573 | **yes** (128 and 64) |
| `9118` `9830` `9654` (1536×1024) | 1536×1024 | 1.573 | **yes** (128 and 64) |
| `8134.scale1440x900` | 1440×900 | 1.296 | **no** |

Those 10 contribute 15.452 MP where the budget implies 10.486 — an excess of
**+4.966 MP**, i.e. the corpus's **37.02 vs 33.55 MP (+10.3 %)**. So:

- **Cost is a 10.3 % overshoot on a budget that was itself a free variable**
  (§2.4's whole premise), not a breach of a constraint.
- **The tile axis is unaffected**: every one of the 10 has both axes ≥ 900, well
  clear of AV1's 512 px-per-axis floor for a 2×2 tiling.
- **Superblock purity loses exactly ONE reference.** Nine of the ten are exact
  multiples of both 128 and 64, so §2.4's "no partial-superblock cells" property
  holds for them unchanged. Only **`8134.scale1440x900`** breaks it. (The corpus
  has four partial-superblock natives in total; the other three —
  `8288.scale375x667`, `8434.scale414x896`, `8414.scale1280x800` — are under
  budget and would have been native under either rule, so they are not caused by
  this deviation.)

**Decision: AMEND.** Rebuilding would discard 13,532 encodes and force a
three-run redeclaration to remove a 10.3 % pixel overshoot and one reference's
superblock purity. §2.4 is amended in place above with the as-built column and a
dated note; the module docstring of `avifdoe_build_budget_corpus.py` now states
the rule truthfully (the old comment claimed "never crop something already at or
under budget", which is not what the condition does), and the builder prints the
MP overshoot on every run. **Spec and code now agree**, which was the actual
requirement — the split was a documentation defect, not a data defect.

Read every `α + β·pixels` fit (§3.9) and CPU-h figure (§3.7) against 37.02 MP,
not 33.55.

### 12.3 The jobexec source cache is content-addressed and swept

**The outage's cause** (§11): `resolve_source_raw` cached every fetched source
at `${TMPDIR}/jobexec_src_<pid>_<basename>`, kept it forever, and the fleet runs
a fresh process per cell — 10,413 dead pids held 60 distinct images as ~46
copies each = **22.93 GB**. §11's mitigation moved `TMPDIR` off the quota'd
volume, which bought room without fixing the growth.

**Fix, in `crates/zenmetrics-cli/src/jobexec.rs`:**

1. **Content-addressed key.** `src_cache_path(uri)` keys on `sha256(resolved
   URI)` (32 hex chars) plus the basename as a readable suffix — the basename is
   kept because it carries the extension, which the HDR decode path dispatches
   on. Copies collapse from (processes × images) to (images): the **46× fix**.
   The repo already had this pattern one function away — the variant index is
   URI-keyed and atomically published — so this is adopting the local
   precedent, not inventing a scheme.
2. **Basename-only was NOT an option, and the reason is in this very plan.**
   §2.4 keeps the corpus key *unchanged* across the crop, so
   `avifsvt-subsample-2026-09-01/1442.scale4000x3000.png` and
   `avif-doe-1024-2026-09-01/1442.scale4000x3000.png` are different pixels under
   one filename. Only the full URI separates them. (The old PID key avoided this
   by accident and only across processes — two such fetches in ONE `--serve`
   process collided and the second silently got the first's bytes. That latent
   bug is closed by the same change.)
3. **Per-writer `.part`.** With `dst` now shared, a `dst.part` sibling would let
   two concurrent workers interleave into one file and rename the mixture into
   place. `.part` now carries pid + a sequence counter; publication is still an
   atomic rename of a fully-written file, so a cache hit is always whole.
4. **Eviction, lazily once per process.** `sweep_src_cache_once()` runs on the
   first source resolution — no entry-point wiring, and a long-lived `--serve`
   worker pays it exactly once — removing entries under the `jobexec_src_`
   prefix untouched for `ZEN_JOBEXEC_SRC_CACHE_MAX_AGE_HOURS` (default 24; `0`
   disables). It collects the **legacy PID-scoped files** the outage left behind
   and orphaned `.part` files, and it bounds the remaining axis (a box that
   walks a large corpus would otherwise keep every image it ever touched).
5. **Concurrency safety is structural, not hoped for.** A cache hit `touch`es
   the entry, so "age" is time-since-last-**use**, not since download; the
   sweeper only ever matches its own prefix; and a wrongly-collected entry costs
   one re-download, never a wrong or partial read.

**Tests** (4 new, `cargo test -p zenmetrics-cli --no-default-features --features
jobexec,hdr --bin zenmetrics` → **24 passed, 0 failed**): the same basename from
two corpora must not share an entry; the key is deterministic and carries no
pid; the sweeper collects stale + legacy + orphaned-`.part` while keeping a
fresh entry **and never touching a foreign file however old**; and a touch
resets the age clock. `cargo fmt` clean.

**`build_executor_image.sh` now agrees with `Dockerfile.executor`.** The script
treated the `zenfleet-worker` overlay as optional ("image keeps base's") while
the Dockerfile `COPY`s it unconditionally — so the "optional" branch was an
unreachable outcome that turned a clear precondition into an opaque `COPY
failed` minutes into a build. Made **required**, with the build command in the
error. That is also the safer half of the disagreement: a silently-inherited
stale worker is exactly what cost this wave 93 % of A1 (§11.5/§11.7).

**These land in the repo now and change nothing running.** The fleet uses pinned
image tags (`exec-avifdoe-svtknobs-b0bb8340`); it picks the cache fix up at the
next image roll. No image was rolled here, because §12.2 chose not to rebuild
and therefore forced no redeclaration.

Pre-existing and NOT introduced here, recorded so the next reader does not chase
it: `cargo clippy -p zenmetrics-cli --no-default-features --features jobexec,hdr`
fails on `cvvdp-gpu` (`build_cvvdp_inner` is never used) — a dependency crate
that clippies clean on its own, i.e. a feature-combination dead-code issue. Zero
clippy diagnostics mention `jobexec.rs`. Likewise `--features jobexec` *without*
`hdr` does not compile the bin at all (`zenflate` unresolved at jobexec.rs:1583,
`metric_runtime` missing at :1001 — both inside `hdr`-gated regions with
un-gated callers).

### 12.4 AG's arithmetic, rechecked against the as-built corpus — it is degenerate

**§3.8's cell count is right; its statistics are not, on this corpus.** The grid
arithmetic checks out (17 arms × 2 speeds × 3 q × 32 images = 3,264), and AG is
confirmed to run on the **native** corpus by data rather than by the doc — its
declared `inputs` sha is the reference's `source_sha256`, while A1/A2/A0R carry
the `crop_sha256` (§12.2).

**But 19 of the 32 references are native passthroughs, so for those the "budget"
and "native" legs are the SAME ENCODE of the SAME PIXELS.** T1/T2/T3 are all
defined over "the 32 images", and 19 of those comparisons are identities:

- **T3 cannot fail.** Its statistic is the *median* of 32 residuals, of which 19
  are exactly 0 — so the median is 0 by construction, for every arm, whatever
  the 13 crops do. A bar that cannot be failed is not a gate.
- **T1 is weakened by more than half.** "Sign agreement on ≥ 80 % of 32" = ≥ 26
  images, of which 19 agree automatically; the effective bar on the informative
  subset is 7 of 13 (**54 %**), not 80 %. An arm could flip sign on six of the
  thirteen real crops and still pass.
- **T2 is inflated.** 19 points sit exactly on the identity line, which drags the
  Spearman toward 1 mechanically.
- The residual's binomial sign test likewise has 13 non-zero terms, not 32.

**Registered correction (analysis-side, blocking on AG's readout).** T1/T2/T3
must be computed over the **13 cropped references only** — the subset where
"budget" and "native" denote different things — with the bars restated against
n=13, and the 19 identity pairs reported separately as the null check they
actually are (any non-zero residual there is a bug in the pipeline, not a
transfer effect). n=13 is thin for a Spearman; state the CI rather than a bare
coefficient.

**Also registered, not executed:** AG's native leg for the 19 passthroughs
duplicates encodes A1 already declared at the same size on the same bytes —
19/32 of its cells. AG has **0 blobs and 0 ledger rows**, so dropping them is
free *if* AG is redeclared before it runs. Not done here: redeclaring another
lane's run mid-wave is exactly the move §11.7 warns about, and the wave owner
should make that call. Flagged, with the arithmetic, so the choice is informed.

---

## 13. Scoring scale-up — 2026-09-02, from a trickle to parity

§11.6 declared the §6 blocking finding "cleared" because one capped tower worker
moved `avifsub-svt-sf-cpu` off 29. That was true and insufficient: the DOE's own
four runs had **no score declaration at all**, and the one worker was 2 cores.
This section measures the trickle, names its three causes, and fixes them.

### 13.1 The diagnosis — three causes, one of them dominant

Measured 2026-09-02T05:18Z, before any change:

| run | encode blobs | score run | score blobs |
|---|--:|---|--:|
| `avifsub-svt-enc-20260901` | 6,496 | `avifsub-svt-sf-cpu-20260901` | 148 |
| `avifsub-aom-enc-20260901` | 3,414 | `avifsub-aom-sf-cpu-20260901` | 256 |
| `avifdoe-svt-a1-20260901` | 4,124 | — | — |
| `avifdoe-svt-a2-20260901` | 21,902 | — | — |
| `avifdoe-svt-a0r-20260901` | 0 | — | — |
| `avifdoe-svt-ag-20260901` | 0 | — | — |

**Cause 1 (dominant): the DOE backlog was UNDECLARED.** `aws s3 ls
s3://zentrain/jobs/` returns four `avifdoe-*` encode runs and **zero**
`avifdoe-*-sf-*` score runs. 26,026 encode blobs — every byte the DOE itself had
produced — had nothing to score them. The pre-existing 5-minute gap-fill loop
(`~/tmp/avifsub_gapfill_loop.sh`) was **alive** (round 51 at 05:13Z, not dead as
assumed) but its `for backend in svt aom` loop is hard-coded to the two
`avifsub-*-enc` runs of the *naive* wave. It never had a DOE branch to lose.

**Cause 2: the fleet held exactly ONE score worker.**
`avifsub-score-svt-tower` — `--cpuset-cpus=22-23`, i.e. **2 cores** — bound to
`ZEN_RUN=jobs/avifsub-svt-sf-cpu-20260901`. No worker served the aom score run
(its blobs sat flat at 256 across rounds 48-51 while its declared job count rose
263 → 287), and the **32-thread dev box was running zero workers of any kind**
(load 0.68).

**Cause 3 is NOT per-pair cost or store I/O.** Both were measured and cleared —
see 13.2, which also records the wrong answer this lane published to itself
first.

### 13.2 A measurement that was wrong, and the check that caught it

The first timing run reported **6.72 s for a 12-pair chunk (0.56 s/pair)** and
was **wrong** — it measured failure, not work. The local `zenmetrics` binary
returned, for every row, `chooser: no measurements for metric 'ssim2'` with
**zero numeric scores**, in 6.72 s.

Acting on it would have been worse than slow. A first local worker launched on
that binary wrote **322 `done` ledger rows over 333 error blobs in under three
minutes** — a 93 %-poison incident of exactly the §11.7 shape, self-inflicted,
and it looked like a *triumph* at the time: 257 blobs in 134 s, ~13× the rate the
CPU budget allows. **The tell was the implausibility, not an error message** —
nothing failed, exit codes were 0, blobs were well-formed JSONL of the right size
and row count. What caught it was opening a blob and counting numeric scores.

Root cause: the local binary was built without `--no-default-features`
(`--features sweep,png,jpeg,webp,avif,jxl,cpu-metrics,avif-svt,avif-aom`), which
pulls the orchestrator in; its persistent capability profile
(`~/.cache/zenmetrics/capability_<hash>.toml`) carries an **empty `[metrics]`
table**, so `Orchestrator::choose_backend` returns `ChooserError::UnknownMetric`
for every metric. The box's CPU changed (the profile reads
`AMD Ryzen 9 9950X3D`, 32 logical cores, against the 7950X/28-core this repo's
docs assume), so its `machine_hash` is new and the older populated profiles do
not apply. `ZENMETRICS_USE_LEGACY_SCHEDULER=1` does **not** bypass it — jobexec's
score path routes through the chooser regardless. `zenmetrics score --metric
ssim2` on the *same* ref/blob pair returns `ssim2=19.796691`, which is what makes
this hard to see: the binary is fine everywhere except the path the fleet uses.

The canonical fleet build (`scripts/jobsys/build_executor_image.sh:15`) uses
`--no-default-features --features sweep,png,jpeg,webp,avif,jxl,cpu-metrics` and
has no orchestrator, which is why the tower worker — with **no capability cache
at all** — has been producing correct scores the whole time.

**Resolution: use the proven image, not a speculative rebuild.** All local
workers run `ghcr.io/imazen/zenfleet-worker:exec-avifsub-svtaom-4a6876b9`, the
tag already producing correct scores on tower.

**The 322 poisoned rows were removed, not pardoned.** `requeue` acts on
failed/poison rows by `error_class` and `reassert` reverses buried *done* rows —
neither targets a row that is genuinely `done` over a garbage blob. Every row in
the snapshot was verified to be the single smoke worker's
(`worker: {'wsl-score-smoke': 323}`, 322 done + 1 failed) with no other worker's
work in the run, so the run's `ledger/`, `blobs/`, `claims/` and `_probe/` were
cleared and re-compacted to `0 done + 0 newest-failed`. The cells simply re-enter
the gap. This is only safe because the run was eight minutes old and entirely
self-inflicted; on a shared run the answer is a scoped pardon.

**Two `--run` prefix traps, hit and recorded.** `zenfleet-ctl declare-scorefiles
--run jobs/<run>` and `compact --run jobs/<run>` both write to
`s3://<bucket>/jobs/jobs/<run>/` — the tools prepend `jobs/` themselves and take
the **bare run name**, while the worker's `ZEN_RUN` takes `jobs/<run>`. The
orphaned `jobs/jobs/` declaration was deleted. (The pre-existing gap-fill loop
sidesteps this by using `--manifest-out` plus an explicit `aws s3 cp`.)

**A fresh run needs a snapshot before a strict worker will start.** The first
correct launch crash-looped 7× on `FATAL: ZEN_REQUIRE_SNAPSHOT=1 but no
ledger_snapshot.parquet`; `zenfleet-ctl compact --run <bare> --upload` (which
also needs `~/tmp/zen-snaps/` to exist) is the fix, and is preferable to
`ZEN_REQUIRE_SNAPSHOT=0` because it keeps the strict invariant for every later
worker.

### 13.3 What was declared

```sh
zenfleet-ctl pairs --ledger s3://zentrain/jobs/avifdoe-svt-<run>-20260901/ledger/ \
  --refs-prefix s3://codec-corpus/avif-doe-1024-2026-09-01/ \
  --blobs-prefix s3://zentrain/jobs/avifdoe-svt-<run>-20260901/blobs/ \
  --out ~/tmp/avifdoe_pairs/<run>_pairs --endpoint $EP
zenfleet-ctl declare-scorefiles --pairs .../a1_pairs.parquet --pairs .../a2_pairs.parquet \
  --run avifdoe-svt-sf-cpu-20260902 --bucket zentrain --endpoint $EP \
  --metrics ssim2,zensim,butteraugli --full-uri
```

One new score run, **`avifdoe-svt-sf-cpu-20260902`**, covering A1 + A2:
**3,241 jobs / 38,796 pairs** at declaration (05:21Z), **3,373 / 40,356** one
round later as A1 encodes landed. Chunk = 12 pairs; one blob = one chunk × 3
metrics = 36 JSONL rows (12 `ssim2` + 12 `butteraugli` numeric, 12 `zensim`
carrying a 720-wide `features` vector and no scalar — the same shape the naive
wave's blobs already have).

**AG is deliberately NOT declared here.** `avifdoe-svt-ag-20260901` encodes the
**native** corpus (its declared `inputs` sha is `source_sha256`, not
`crop_sha256` — §12.4), so it needs a different `--refs-prefix`; it has 0 blobs,
and §12.4 registered that the wave owner may drop 19/32 of its cells before it
runs. Declaring it against the crop prefix would have silently scored the wrong
pixels — the §12.3 collision, which is the exact hazard that makes the two
corpora share filenames. The recurring loop prints a loud notice if AG ever
produces a blob.

### 13.4 Recurring declaration, and how it fails loud

`~/tmp/scorescale_gapfill.sh` (detached, PID in
`~/tmp/scorescale_gapfill.pid`, log `~/tmp/scorescale_gapfill.log`) re-runs
`pairs` → `declare-scorefiles` → `compact` for A1/A2/A0R every 5 minutes.
`declare-scorefiles` is idempotent per cell, so it coexists with the older
naive-wave loop rather than fighting it.

Its predecessor's failure mode was dying silently. This one writes
`~/tmp/scorescale_gapfill.heartbeat` every round with
`round= errors_total= errors_this_round= score_blobs=`, prints a `❌` line for
any failed canonical command, and skips a run with 0 encode blobs by name rather
than erroring. A dead loop is a stale heartbeat mtime; a sick loop is a rising
`errors_total`.

### 13.5 Metric set reduced to `ssim2,zensim` — USER DIRECTIVE, mid-lane

**User directive, 2026-09-02, verbatim: "you can skip butteraugli even."** All new
score declarations for the DOE waves drop butteraugli and carry **`ssim2,zensim`**
(the zensim row is the 720-wide `features` vector, not a scalar).

**The already-declared 3-metric jobs could NOT be re-declared in place, and the
reason is structural:** `DesiredJob::job_id()` is `JobId::of(&self.kind,
&self.inputs)` (`zenfleet-core/src/ledger.rs:118`) and `kind` **carries the metric
list**. Changing `["ssim2","zensim","butteraugli"]` → `["ssim2","zensim"]`
therefore changes **every** job_id in the run: the re-declaration is not an
edit, it is a second, disjoint job set. That is precisely the "churn the ledger
or double-declare" case, so the sanctioned fallback was taken — **the recurring
declaration was switched going forward**, at a recorded boundary:

| | |
|---|--:|
| last 3-metric blob count (05:42:35Z) | **907** |
| first 2-metric declaration | 3,424 jobs / 40,896 pairs (05:42:11Z) |

**Consequence the analysis lane must key on: rows are HETEROGENEOUS.** The 907
blobs written before the boundary carry 36 JSONL rows each (12 `ssim2` + 12
`butteraugli` numeric + 12 `zensim` feature rows); every blob after carries 24
(12 `ssim2` + 12 `zensim`). Verified on both sides of the cut, 0 errors. **Key on
the metrics present per row, never on a fixed row count or a fixed metric set.**
The pre-boundary butteraugli values are valid and were kept — extra data, not a
defect — but they cover only the ~11 k pairs scored before 05:42Z, so butteraugli
is **not** a corpus-wide column and must not be treated as one.

Those 907 blobs' cells are re-scored under the new 2-metric job_ids (the old
job_ids remain `done` in the ledger and are simply not in the current manifest).
That re-work was accepted deliberately: it is ~11 k pairs against the ~40 k
pairs that get the saving.

### 13.6 Worker topology

Every score worker runs the **proven** image
`ghcr.io/imazen/zenfleet-worker:exec-avifsub-svtaom-4a6876b9`, launched through
the canonical `scripts/jobsys/lan_score_launch.sh` (against `localhost` for the
dev box — it resolves the store via `~/.config/zen/s3env.sh` exactly as a remote
box does, so there is no second launch path). `ZEN_CONTROL_KEY` is set on every
worker, so the whole fleet is pausable with the run's `control.json`.

| host | worker | run | cpuset | mem | note |
|---|---|---|---|--:|---|
| dev | `dev-doe1` | `avifdoe-svt-sf-cpu-20260902` | 0-11 | 24 g | DOE, the bulk |
| dev | `dev-doe2` | `avifdoe-svt-sf-cpu-20260902` | 12-19 | 18 g | DOE |
| dev | `dev-aomsf` | `avifsub-aom-sf-cpu-20260901` | 20-23 | 10 g | naive aom (had NO consumer) |
| dev | `dev-svtsf` | `avifsub-svt-sf-cpu-20260901` | 20-23 | 10 g | naive svt (tower alone = 12 h) |
| tower | `tower-score-svt-1` | `avifsub-svt-sf-cpu-20260901` | 22-23 | 16 g | pre-existing, untouched |
| r7900x | — | — | — | — | **encode only; nothing added** |

**Dev-box cap honoured:** all containers sit inside cpuset `0-23`, leaving
**cores 24-31 (8 of 32) free**, per the machine-safety rule. Measured under
load: 38-40 GiB RAM available, per-container RSS 1.0-1.6 GiB against caps of
10-24 GiB, no swap growth. Load average reads 60+ because
`lan_score_launch.sh` sets `ZEN_CORE_OVERSUBSCRIBE=2` and the runnable queue is
deep, but the cpuset confines it and `vmstat` still showed 27-31 % idle — the
oversubscription is why a 4th worker fit on shared cores rather than a reason to
back off.

**r7900x was deliberately left alone.** It was carrying the A1 encode lane; it
was observed before and after (load 11.00, 11 live `zenmetrics` encode
processes) and **no score worker was added there**, so encode throughput could
not regress from this lane. Tower's pre-existing 2-core score worker was left
exactly as it was.
### 13.7 Measured rates — before, after, and after the metric cut

All figures are `aws` lister deltas over timed windows against the LAN store, on
the runs named. **Before** is the state this lane inherited.

| window | run | Δblobs | seconds | blobs/h | pairs/h |
|---|---|--:|--:|--:|--:|
| **before** (§11.8 → 05:18Z) | `avifsub-svt-sf-cpu` | — | — | **~20** | ~240 |
| **before** | `avifsub-aom-sf-cpu` | flat at 256 (rounds 48-51) | — | **~0** | 0 |
| **before** | `avifdoe-*` (all four) | — | — | **0 (undeclared)** | 0 |
| after, 3-metric (05:32:52→05:36:07Z) | `avifdoe-svt-sf-cpu-20260902` | +239 | 196 | **4,389** | 52,670 |
| after, 3-metric (05:36:58→05:40:44Z) | same | +273 | 218 | **4,508** | 54,100 |
| after, 2-metric (05:42:35→05:44:55Z) | same | +336 | 140 | 8,640 | 103,700 |
| **after, 2-metric (05:42:35→05:50:49Z)** | same | **+1,152** | **487** | **8,514** | **102,168** |
| after (05:40:44→05:44:55Z) | `avifsub-svt-sf-cpu` | +30 | 251 | **430** | 5,160 |
| after (05:40:44→05:44:55Z) | `avifsub-aom-sf-cpu` | +9 | 251 | **129** | 1,550 |

**Headline: the DOE score lane went from 0 (nothing declared) to ~8,600
blobs/h ≈ 103 k pairs/h, and `avifsub-svt-sf-cpu` from ~20/h to ~430/h.**

**Dropping butteraugli bought ~1.9×, more than the ~0.27 s of ~0.6-0.9 s
predicted.** 4,508 → 8,514 blobs/h (487 s window; three shorter windows read 8,124-8,928) across the boundary, same workers, same
cores, same image — so at 1024² butteraugli was closer to *half* the 3-metric
per-pair cost than a third. This is a measured before/after on one machine with
everything else held constant, not an estimate.

**A per-pair cost figure is deliberately NOT quoted from the first timing run.**
That run measured failure (§13.2). The honest per-pair number is the one implied
by the fleet windows above: ~102 k pairs/h across the 20 DOE-serving cores ≈ **0.70 CPU-s/pair**
for `ssim2,zensim` at 1024², and ≈ 1.33 CPU-s/pair for the 3-metric set.
### 13.8 Where the constraint now sits — scoring is no longer it

**Encode state at 05:44Z** (`zenfleet-ctl report`, live-gap semantics):

| run | declared | live_done | gap | workers |
|---|--:|--:|--:|---|
| `avifdoe-svt-a2-20260901` | 33,984 | 33,984 | **0 — COMPLETE** | — |
| `avifdoe-svt-a1-20260901` | 6,912 | 6,432 | 480 | r7900x |
| `avifdoe-svt-a0r-20260901` | 6,496 | 0 | 6,496 | **none** |
| `avifdoe-svt-ag-20260901` | 1,728 | 0 | 1,728 | **none** |
| `avifsub-svt-enc-20260901` | 6,496 | 6,496 | 0 | — |
| `avifsub-aom-enc-20260901` | 9,280 | ~4,100 | ~5,200 | tower |

**Projected score completion vs encode completion:**

- **DOE (A1+A2), the wave's bulk: ~2,270 jobs left at 8,514 blobs/h ⇒ complete
  ≈ 06:07Z**, i.e. **within ~16 minutes**, against an A2 encode leg that is
  already finished and an A1 leg with 480 cells to go.
- `avifsub-svt-sf-cpu`: ~400 jobs at 116-430 blobs/h ⇒ ≈ 06:45-08:20Z, and it accelerates
  the moment the DOE workers drain and are repointed.
- `avifsub-aom-sf-cpu` is **encode-bound, not score-bound**: its scorer idles
  waiting for aom encode (~5,060 cells left at ~415/h ⇒ ~12 h).

**So the brief's target — "finish scoring within ~a day of encodes finishing" —
is met with a very large margin, and the binding constraint has moved.** Scoring
now drains faster than encode produces. The open items are encode-side and
belong to the wave owner:

1. **A0R (6,496 cells) and AG (1,728 cells) have no encode workers at all** —
   unchanged since §11.8. Until they run, 8,224 DOE cells cannot be scored
   because they do not exist.
2. **A1's 480-cell gap** is `failed-only`; whether those are genuine failures or
   just un-retried is not diagnosed here (this lane did not touch encode
   declarations).
3. **AG additionally needs a score declaration with the NATIVE refs-prefix** when
   it runs — see §13.3. The recurring loop declares A0R automatically and prints
   a loud notice for AG rather than guessing its prefix.

---

## 14. Wave completion + Stage A — 2026-09-02

**Full record: [`avif_doe_stageA_2026-09-02.md`](avif_doe_stageA_2026-09-02.md).**
This section is the pointer and the corrections this lane owes §3, §7 and §13.

### 14.1 The wave is complete

All four svt-rs encode runs reached live-gap 0 with **zero failed cells**:
A1 6,912 · A2 33,984 · A0R 6,496 · AG 1,728 = **49,120 cells, all scored**.
A0R and AG had no encode workers when the lane opened; both were launched here.

**§13.8's "A1 480-cell `failed-only` gap" never existed.** A1's ledger shows the
run finished at **04:47:21Z**, ~57 min before that snapshot, 6,912/6,912
live-done, with **zero post-fix `encoder_panic`** — the pardon of §11.7 worked
and nothing was laundered. Nothing was requeued and no poison was annotated.
The reading was stale, which is worth knowing: `report` can under-read a
*finished* run.

### 14.2 Corrections this lane owes the plan

1. **§3.2's cell arithmetic is impossible under its own isolation rule.** "17
   arms × all 7 effective presets = 34,272" cannot be declared at
   `--max-deviations 1`, because a non-default preset *is* the one permitted
   deviation. zenavif's own test fixes the real design at **24 strata**
   (1 default + 16 knob arms + 6 speeds + 1 bit-depth), i.e. the declared
   **6,912**. Consequence: **knob main effects exist at two presets, s4 and s6,
   not seven**; **B-5** is evaluable across one preset pair; and §3.3's claim
   that folding A1b away was safe because "presets 0 and 1 at 9 q" would cover
   it is **not true** — no knob arm exists at preset 0 or 1.
2. **§3.9's bytes decomposition is not identifiable from the crop/native pair.**
   A crop is a different image, not a smaller one, so the two-point intercept
   absorbs the content difference: **SROCC(α, q) median 0.943** across 91
   (image, speed) groups, α climbing **731 → 59,176 bytes** across the ladder
   (81×) and going negative on 781 of 2,639 fits. The "free from the
   fleet" claim does not hold; a same-content size ladder would be needed.
3. **§3.8's transfer gate is thinner than §12.4 restated.** On top of the n=13
   correction, AG's declared grid carries knob arms at **speed 4 only** (the s6
   cells spend their deviation on the preset) and omits `bd10`; n = 11 after the
   two degenerate crops. **2 PASS (`mtx32`, `qml1.8.15`) · 7 PARTIAL · 2 genuine
   FAIL-T1 (`acb3`, `shp3`, which fire B-6) · 3 NOT-MEASURED · 2 INERT.** Its
   most useful output is a result §6 alone would have got wrong: **tiling's
   bitrate cost is largely a reduced-size artefact** — `tl1.0` +0.65 % at budget
   vs **−0.12 % at native**, `tl1.1` 8.6× smaller, carrying the only two
   significant T3 sign tests in the set (p = 0.012, 0.001).
4. **§7.2's Stage-B envelope is exceeded by the triggers it defines.** Honouring
   every trigger costs **447,636 cells against a registered 60,000 — 7.5×**
   (B-1 17 · B-2 23 · B-3 13 · B-6 2). Prioritisation is a decision and it is
   the coordinator's; **no Stage-B wave was declared here.**
5. **A recurring score declaration re-does its own work.** `zenfleet-ctl pairs`
   emits the same rows in a **different order** on consecutive runs against a
   frozen ledger, and `declare-scorefiles` chunks by row order into the
   `job_id`, so every gap-fill round re-declares the run as fresh jobs:
   `declared=4,128` vs `ever_done=16,476`, a **4.0× multiplier** with zero
   failures. A waste bug, not a correctness bug; the fix is a deterministic sort
   in `pairs`. Not applied mid-wave.

### 14.3 The finding for the port program

**`tune=0` and `screen_content_mode=Some(3)` produce byte-identical bitstreams
to the default on 288/288 cells at both presets**, while the harness's own
resolved-state fingerprint separates them. `tune=3` *does* change the bitstream,
so `tune` is partly wired. Minimal repro, cells affected (8,972) and the
fingerprint table are in the Stage-A record §3. **Not fixed here** — zenavif is
another lane's subject.

---

## 15. Stage B — trigger B-6 declared and running (2026-09-02)

**USER DECISION, 2026-09-02: "go with B-6 first".** This lane declares and runs
**only** B-6. The other 53 triggers (§10) stay undeclared; the remaining
Stage-B envelope is unspent and uncommitted.

### 15.1 The two arms, and the cell arithmetic as VERIFIED

B-6 fires when an arm fails **T1** of the cross-size transfer gate — its
direction at the 1024² screening budget does not carry to native — and its
follow-up is "*that knob's Stage-B grid runs at native size*" (§7.2). Stage A
§8.1 certified only `mtx32` and `qml1.8.15` for reduced-size screening and
found exactly two arms that **fail**:

| arm | knob | A1 level | Stage-A T1 | median BD @1024² | median BD @native |
|---|---|---|---|--:|--:|
| `acb3` | `SvtParams::ac_bias` = 3.0 (`0.0..=8.0`, default 0.0) | 3.0 | **FAIL-T1** (dir 0.62 vs bar) | +0.07 % | −0.43 % |
| `shp3` | `SvtParams::sharpness` = 3 (`0..=7`, default 0) | 3 | **FAIL-T1** (dir 0.62, 8/11 refs) | +0.50 % | +0.41 % |

The Stage-B grid is B-1's registered follow-up shape — **5 levels × the full
29-q ladder × 32 images × speeds {4, 6, 7}** — at native size.

**Registered: 27,840.** Recomputed here and it reproduces exactly:
`2 knobs × (5 × 29 × 32 × 3) = 2 × 13,920 = 27,840`.

**Declared: 25,056.** The difference is **2,784** and it is not an error in
either number — it is the two knobs' **shared default level**. `ac_bias` 0.0
and `sharpness` 0 are the *same configuration* (the default stratum), so a
plan that carries both knobs on one axis spells 9 levels, not 10. The
trigger-list figure is the per-knob sum, which counts that control block once
per knob; the declaration counts it once, full stop. `2,784 = 1 level × 29 q ×
32 images × 3 speeds`.

Declared shape, from the planner's own audit manifest:

```
plan svt_doe_b6: 783 cells/image × 32 sources = 25,056
  = 27 strata (9 knob levels × 3 speeds) × 29 q
duplicates_merged 0 · invalid_skipped 0 · compute_tier_skipped 0
q_coarsenings 0 · dropped_axes 0 · over_budget false
```

The 9 levels: the shared default, `acb` {1.0, 3.0, 5.0, 8.0}, `shp` {1, 3, 5,
7}. Each knob **retains both levels A1 already measured** (`acb` 1.0/3.0, `shp`
3/7) so every native point has a budget-size partner to difference against —
that is the whole purpose of the trigger — and each adds the range A1 never
reached: `ac_bias` above 3.0 (the dossier calls 3.0 "the upper half"; it is
37.5 % of the range, and 4.0–8.0 was untouched) and the odd `sharpness` steps.
`sharpness` stays **categorical** (H-14): five levels is a level set for
factor fitting, never an ordinal trend.

**The two byte-inert knobs are absent by construction.** `tune = 0` and
`screen_content_mode = Some(3)` (§3, imazen/zenav1-svt#17) consumed 8,972
Stage-A cells producing bitstreams identical to the default. Neither appears in
any B-6 level, and `svt_doe_b6_is_two_dense_grids_sharing_one_control` asserts
it rather than trusting it.

### 15.2 `--max-deviations 2`, and why 1 would have silently halved the block

`speeds` is itself an axis, so a knob level at speed 6 or 7 costs **two**
deviations — the same arithmetic §4.1 found in A1, where it collapsed the
"7 presets" claim to two. At `--max-deviations 1` this block emits **11**
strata instead of 27: only the speed-4 leg plus two bare controls. Measured,
and pinned by a test.

Two is safe here for a structural reason, not a lucky one: `svt_knobs` is
**ONE axis**, so no cell can ever spell two knobs at once. The block is
main-effects-only despite the looser bound. That is also why
**`duplicates_merged: 0` is correct here** rather than the red flag the
declare script warns about — no collision is *expressible*, so there is
nothing to merge (the same argument §11.4 makes for `svt_doe_main`).

### 15.3 Gates — five, none assumed

1. **Corpus identity (native, not crop).** All **32/32** files at
   `s3://codec-corpus/avif-subsample-2026-09-01/` sha256-match the local
   native sources, **0 mismatches**; against the budget twin, **13 differ and
   19 are byte-identical passthroughs** — exactly the as-built 13/19 split
   §12.2 records. Stronger still, the collision is impossible at *cell* level,
   not merely avoided: the declared `source_sha` for cropped ref `6604` is
   `769b0df4…` from the native prefix and `4ac38273…` from the crop prefix, so
   the two corpora produce **disjoint CellIds**.
2. **Declaration determinism.** Two independent declare rounds into separate
   directories produced **sha-equal** `_cells.jsonl` (`951a323d…`) and
   `_manifest.json` (`41f4acb2…`).
3. **Pairs determinism.** `zenfleet-ctl pairs` run twice against the same
   ledger state gave a **sha-equal** parquet (`82aea675…`) — the deterministic
   sort from `08215e84` holds on this run, so the 4.0× re-declaration
   multiplier §14.2(5) recorded does not apply here.
4. **G-FIRSTCELL, encode.** One worker, 4 cores: 60 blobs in the first ~7 s.
   First blob is a well-formed AVIF — `ftypavif` + `mif1miaf` magic, `file(1)`
   "ISO Media, AVIF Image", 27,675 bytes — and it **decodes and scores**
   (ssim2 59.42), which magic bytes alone would not have shown.
5. **G-FIRSTCELL, score.** First score blob carries the documented §13.3
   shape: 12 `ssim2` rows with real scalars (67.9998) and 12 `zensim` rows
   each carrying a 720-wide `features` vector and no scalar.

### 15.4 Cost — MEASURED through the real executor, not extrapolated

Every number below comes from running `zenmetrics jobexec` — the worker's own
entry point — on the real native corpus, locally, before the fleet was scaled.

**Per-speed totals over all 32 images at q45, default knobs** (not a per-pixel
rate applied to a corpus — the actual sum):

| speed | preset | 32-image total @ q45 | 29-q ladder factor | CPU-h per knob level |
|---|---|--:|--:|--:|
| s4 | 4 | **165.30 s** | 26.1–30.1 | 1.257 |
| s6 | 7 | **13.16 s** | 30.1 | 0.110 |
| s7 | 9 | **6.63 s** | 29.2–30.0 | 0.055 |

⇒ **13.2 CPU-h encode for the whole wave** (9 levels, knob-cost factor 1.03
from the measured `acb8`/`shp7` deltas) — **0.22× the registered 60 CPU-h
envelope.** Blobs total **14.3 GB**, mean 0.57 MB/cell, against 20 TB free on
the LAN store. **The B-6 budget is not at risk and no de-scope is needed.**

Three things the probe established that the plan did not know:

- **`ac_bias = 8.0` is safe AND live.** `SvtParams::clamped()` does **not**
  clamp `ac_bias` (it clamps the variance-boost pair, the QM levels,
  `max_tx_size` and `sharpness`), so the top of the documented range was a
  genuine release-mode out-of-range risk — the **H-10** hazard class. It
  encodes cleanly and *moves bytes* (29,049 vs the default's 28,516 at
  s4/1.57 MP), so it is a real level, not another inert knob.
- **Cost is not linear in pixels.** Preset 4 runs **1.842 MP/s at 1.57 MP but
  0.591 MP/s at 16 MP** — 3.1× worse per pixel at the large end. An `α + β·pixels`
  fit or a single "ms/MP" would misprice this block at whichever end it was not
  fitted on.
- **Preset dominates everything.** s4 is **17–27×** s6/s7 and carries **88 %**
  of the wave's CPU. Consequently **cell-count progress overstates real
  progress**: at 2,010 cells done the wave was 8.02 % of cells but only
  **6.27 % of the work** (1.28×), because the cheap strata drain first. Report
  B-6 fill work-weighted, not by cell count.
- **Bytes and time have different q shapes.** The 29-q ladder factor is ~27–30
  for *time* but **53–56 for bytes**. High-q cells are cheap to compute and
  expensive to store; the storage plan and the CPU plan cannot share a factor.

### 15.5 Machinery — four changes, all in the canonical owners

| change | where | why |
|---|---|---|
| `svt_doe_b6` plan + test | zenavif `43423054` | the grid itself; the test pins levels, deviations, strata at max-dev 1 *and* 2, id distinctness and id **round-trip** (`svt_doe_cell_ids_roundtrip` only walks the pairwise plan, so `acb5`/`acb8`/`shp1`/`shp5` had no coverage anywhere) |
| plan names get ONE owner | zenavif `386b82f8`, zenmetrics `6b3c41fe` | see §15.6 |
| `avifdoe_declare.sh --stage-b6` | zenmetrics `8d5d3d93` | declares through the canonical builder; hard-codes the native corpus and `--max-deviations 2` so the two traps above cannot be re-hit by hand |
| `avifdoe_score_gapfill.sh` run list parameterised | zenmetrics (this commit) | `ZEN_DOE_RUNS="<run>=<refs-prefix> …"`; B-6 runs as a **second instance of the same loop**, not a fork. The run→refs pairing stays explicit because mismatching it is the silent-garbage hazard that script exists to prevent |

Fleet image **`ghcr.io/imazen/zenfleet-worker:exec-avifdoe-b6-6b3c41fe`**
(digest `sha256:2495afda…`), a new **tag on the existing package** per the
one-package-many-tags rule. Verified before launch: both binaries
`statically linked`; the real plan resolves 27 cells/image; the **control arm**
rejects a bogus plan *and now lists `svt_doe_b6`*; `capabilities` advertises
`avif` + `avif-svt` so the claim-time gate admits these cells.
(`exec-avifdoe-b6-8d5d3d93` was built and pushed first, then superseded by the
§15.6 fix **before any worker ran on it** — it is unused, not stale-in-service.)

### 15.6 A diagnostic that would have lied — the plan list had drifted

zenmetrics' "unknown zenavif plan" error hand-typed the plan names, and its own
comment called itself "the human-readable mirror" of `SweepAxes::by_name`. It
drifted the instant `svt_doe_b6` landed: the message still named eight plans
and omitted the ninth.

That is not cosmetic. **That message IS the control arm §11.5 uses to prove a
fleet image is not stale** — the check that cost this wave 93 % of its
main-effects arm when it was skipped. A future reader running it against a
perfectly good image would have been told the plan they had just declared does
not exist. It was caught here only because the control arm was actually run
(and the first run of it was itself useless — an empty sources dir makes the
real and bogus arms return the *same* "no source files found", exactly the trap
§11.5 documents; a real source had to be mounted before the test measured
anything).

Fixed at the owner: `PLANS` is now one static table that `by_name` looks up in
and `names()` returns, so a second copy is no longer expressible, and
zenmetrics renders the message from `names()`. A test asserts the two agree and
that all nine wire-contract names survive.

### 15.7 Topology, and a correction to the fleet's assumed state

**The fleet was not idle when this lane opened.** `avifsub-aom-r7900x` and
`avifsub-aom-tower` had been up **11 hours**; their run
`avifsub-aom-enc-20260901` was **COMPLETE (live-gap 0, 5,179 declared /
5,484 ever-done)** and both workers were re-touching poison for zero output —
zenfleet's own `idle` waste, on the two boxes this wave needed. Both were
stopped before launch; nothing was destroyed (the run is complete).

| box | role | cap | note |
|---|---|---|---|
| r7900x (24c) | encode `r7900x-b6enc` | `--cpuset-cpus 0-19 --memory 32g` | dedicated worker box, 4 cores left |
| tower (32c) | encode `Tower-b6enc` | `--cpuset-cpus 0-19 --cpu-shares 256 --memory 24g` | **live household media server** — the tower rule, never uncapped |
| dev (32c) | score `dev-b6sf` | `--cpuset-cpus 20-29 --memory 24g` | shares the box with the Stage-A scoring lane's 4 workers; local encode was deliberately NOT added (load was already climbing and the Stage-A lane is another lane's live work) |

Score run **`avifdoe-svt-b6-sf-cpu-20260902`**, metrics **`ssim2,zensim`**
(butteraugli stays dropped per the standing user directive), fed by the
parameterised gap-fill loop every 180 s with heartbeat + `errors_total`
(`~/tmp/b6_score_gapfill.heartbeat`).

**Two launch frictions worth baking into the runbook.** A fresh run has no
`ledger_snapshot.parquet`, so `ZEN_REQUIRE_SNAPSHOT=1` fails the worker loud on
every pass; the fix is `zenfleet-ctl compact --run <run> --upload` (which is
preferable to `ZEN_REQUIRE_SNAPSHOT=0` because it keeps the strict invariant for
every later worker), and the declare script's "next steps" block does not
mention it. And `control.json`'s schema is `{"paused":false,"drain":false}` —
a plausible-looking `{"state":"running"}` is not it.

### 15.8 Corrections this lane owes the record

- **There is no 16320×7612 image in this corpus.** The 32 native refs run
  **0.25–16.00 MP** (largest `6604.scale3286x4868.png`), 161.59 MP total,
  median 1.57. Any B-6 costing that assumed a ~124 MP pano is wrong by ~8× on
  its worst cell.
- **Stage-A §10's B-6 cost line is right, and its per-knob framing is what
  makes it differ from the declaration.** Both numbers are correct for what
  they count; see §15.1. Nothing needs re-registering.

---

## 16. A3 REDESIGNED — the C oracle leaves the tuning-data path (2026-09-02)

**Registered, not declared.** A3 still awaits the user's Stage-B decisions; this
section replaces its *design* only. Nothing here declares a run, changes a live
declaration, or touches the draining waves.

### 16.1 What invalidates the §3.5 / §5.3 design

USER RULE **"IMAZEN-ONLY IMAGING/CODEC SOFTWARE"** (`~/work/zen/CLAUDE.md`,
2026-09-02):

> NEVER reach for imaging or codec software not written by imazen — for encoding,
> decoding, probing, fixture generation, format/metadata inspection, oracles, or
> gates — **especially anywhere in a pipeline that develops predictive models
> designed to tune imazen software.** […] C references (libaom, SVT-AV1, libjxl,
> …) are for **differential port validation inside the port repos only** — never
> as tuning-data admission gates, never as the encoder behind sweep cells, never
> as the truth a product model is trained against. **A port's own shipped
> behavior is the ground truth for tuning it.**

§3.5 selected A3's 23 arms by the criterion *"all `aom_bench::ToggleKnobs` fields
whose C counterpart is emitted by `ToggleKnobs::c_ctrls()`"*, and §5.3 required
each arm to *"drive the C side too, via `EncodeCell::c_encode_ctrls(ctrls)`"*
behind gates **G-AOM-BASE** and **G-AOM-ARM**, both of which are sha-equality
tests against C libaom v3.14.1. Under the rule above, that design is invalid as a
source of tuning data on three separate counts, and the third is the one that
bites hardest:

1. **C as admission gate.** G-AOM-ARM drops an arm from the declaration when the
   port's payload diverges from C. The surviving arm set is then a function of C
   libaom's decisions, so the model's factor coverage is C-selected.
2. **C as factor-space definition.** Restricting the arms to fields `c_ctrls()`
   can express means the *design matrix itself* is bounded by the C CLI surface.
   §3.5 already records the cost: `--tune=iq`/`ssimulacra2` and
   `deltaq_mode2`/`deltaq_mode3` are excluded **for no reason except that C
   cannot be driven to match** — port-side knobs, deliberately unmeasured.
3. **C inside the encode itself.** MEASURED at
   `crates/zenmetrics-cli/src/sweep/encode.rs:1307-1380` — this is not a gate,
   it is the encoder. Every aom-rs cell today:
   - calls `aom_sys_ref::ref_init()` and runs a **full C libaom encode**
     (`cell.c_encode_defaults()`), which is the *sequence/frame header source*;
   - reads the **screen-content decision out of the C stream**
     (`aom_bench::stream_allows_screen_content_tools(&bootstrap)`) and feeds it
     into the port's `ToggleKnobs` — so C's detector, not the port's, chooses
     the port's coding tools;
   - refuses to emit unless `port_payload == oracle_payload`;
   - **splices the port's payload into the oracle's OBU frame**
     (`splice_frame_obu(&bootstrap, &port_payload)`) — the emitted AVIF's
     sequence header is C's bytes.

   So the C oracle is simultaneously encoder (partially), probe, and admission
   gate on the *existing A0 aom arm*, not merely on the proposed A3 one.

### 16.2 The root cause is an API-shape mismatch, not a policy slip

zenmetrics drives zenav1-aom through **`aom-bench`**, which is the port's
**differential-validation harness**, not a product encoder API. This is
structural, not incidental — every port encode takes a C stream by signature:

```rust
// zenav1-aom/crates/aom-bench/src/lib.rs:1150, :1176
pub fn port_encode(&self, bootstrap: &[u8]) -> Vec<u8>
pub fn port_encode_with(&self, bootstrap: &[u8], knobs: &ToggleKnobs) -> Vec<u8>
```

Its own doc calls the argument a *"bootstrap header-field parse"*. There is no
`port_encode` overload that derives its own sequence header. `zenmetrics-cli`'s
`Cargo.toml:60-66` already states the fact plainly — *"the port still bootstraps
its sequence/frame HEADER FIELDS from the oracle stream […] so the oracle is a
build+run dependency of this arm until the port derives headers standalone"* —
so nothing here is newly discovered; what is new is that the rule now makes that
dependency **disqualifying for tuning data**, where before it was a caveat.

**Consequence, stated plainly:** deleting the byte-identity compare would not
purge C. The header would still be C's and the screen decision would still be
C's. The aom arm cannot be de-oracled at the zenmetrics layer at all.

### 16.3 The corrected A3 design

**A3's aom knob arms drive zenav1-aom's own encoder directly, and the port's own
output is the ground truth.** Concretely:

- **Factor space = the PORT's knob surface**, `aom_bench::ToggleKnobs` (or
  whatever the standalone entry point exposes) — **not** the subset `c_ctrls()`
  can mirror. §3.5's exclusions 2 and 3 (`--tune=iq`/`ssimulacra2`,
  `deltaq_mode2`/`deltaq_mode3`) are re-classified from EXCLUDED to
  **eligible**, because the only reason they were cut was C-drivability. They
  return to the design on their own merits, subject to the usual cost gate.
  The other §3.5 exclusions stand: they rest on *port-side* or *AV1-semantic*
  reasoning (H-7 no alt-ref on a still, H-19 the value is discarded in
  all-intra, H-20 the lossless back door, H-15 harness-pinned CICP, U-9
  structural inertness above `cpu-used 7`), not on C parity.
- **Ground truth = the port's emitted bitstream.** Bytes, `encode_ms`, and every
  metric score come from the port's own stream. Nothing is compared to C to
  decide whether a cell counts.
- **Validity gate = the port's OWN correctness, self-contained**: the emitted
  stream must decode through the port's own decoder,
  `aom_decode::frame::decode_frame_obus` (`zenav1-aom/crates/aom-decode`), and
  the AVIF must read back its declared depth/config through `zenavif-parse` —
  the R1 route the landed gate G3 and
  `sweep::encode::aom_rs_depth_tests::emitted_bitstream_carries_the_requested_depth`
  already use. This is the imazen-only replacement for a C-parity gate, and it
  is a *stronger* product statement: byte-equality with C never proved the
  stream decodes; a decode does.
- **G-AOM-BASE and G-AOM-ARM are WITHDRAWN as tuning-data gates.** Both are
  sha-equality tests against C. They are re-registered, unchanged in content, as
  **port-repo differential checks** — zenav1-aom's business, in zenav1-aom's
  test tree, on zenav1-aom's schedule. An arm that diverges from C is **no
  longer dropped from the declaration**; the divergence is recorded as a
  *metadata column* on the cell (`c_parity: matched | diverged | not_measured`)
  and the cell's data stands on its own decode-verified merit.
- **The control arm changes accordingly.** §5.3's control was
  `c_encode_defaults()` — a C encode. The corrected control is the port at
  `ToggleKnobs::default()` with the port's own header derivation, i.e. a
  single-deviation design whose base is the port's own default configuration.
  That is what a model tuning zenav1-aom needs to price a knob against.
- **The port's screen-content detector replaces C's.** §5.3's mirror-the-oracle
  step (`stream_allows_screen_content_tools`) exists only because the header
  came from C. With the port deriving its own header, `allow_screen_content_
  tools` is the port's decision, as it is in production.

### 16.4 The prerequisite, and who owns it

The corrected design has exactly one blocker, and it is **not in this repo**:

> **PREREQ-AOM-STANDALONE** — zenav1-aom must expose an encode entry point that
> derives its own sequence + frame header (no `bootstrap: &[u8]`), emitting a
> complete AV1 temporal unit. Owner: **zenav1-aom**, in its own repo.

Until it lands, the honest status of the aom lane is:

| | status |
|---|---|
| **A3 as designed in §3.5/§5.3** | **INVALID for tuning data** (§16.1). Not declared. Not to be declared. |
| **A3 as redesigned in §16.3** | **BLOCKED on PREREQ-AOM-STANDALONE.** Registered, not declared. |
| **The A0 aom arm now draining** | Produces C-bootstrapped, C-gated, C-header-spliced cells. **Not admissible as tuning-data for a model that tunes zenav1-aom.** Its cells remain valid as *port-parity evidence* — which is what the harness was built to produce. |
| **The svt arm** | Unaffected. `zenav1-svt` is driven through zenavif's own `Av1Backend::SvtRs`, muxed by `zenavif-serialize`, with no C reference anywhere in the path. §9.x's *"the svt arm carries the DOE if A3 slips"* is now the load-bearing plan, not a fallback. |

**No cells are re-run and nothing is deleted on this finding.** The A0 aom cells
are content-addressed and already paid for; they are re-*labelled*, not
discarded. What changes is what may be trained on them.

### 16.5 What was NOT changed, and why

- **The differential C validation itself is untouched and remains correct
  practice.** The rule scopes it to the port repos, which is exactly where
  `s4cov_*`, `encoder_gate_*`, and `config_permutations.rs` live. Nothing in
  zenav1-aom is being asked to stop comparing against libaom.
- **`sweep/encode.rs`'s byte-identity compare is left in place.** It is the only
  thing standing between this arm and emitting unverified bytes *given* that the
  header is spliced from C. Removing it without PREREQ-AOM-STANDALONE would make
  the arm worse, not more compliant. The HBD band gate above it was re-keyed the
  same day to cite the port's own record and this harness constraint rather than
  C parity (`sweep/encode.rs`, `aom_rs_depth_tests::hbd_outside_the_byte_
  verified_speed_band_is_refused_by_name`).

---

## 17. B-6 COMPLETE and analysed — the screening failure was miscalibration, not a wrong direction (2026-09-02)

**Record:** [`avif_doe_stageB6_analysis_2026-09-02.md`](avif_doe_stageB6_analysis_2026-09-02.md)
(+ its `.pointer.md`). This section is the completion note; the record carries
the tables, gates and limitations. **Analysis lane only — nothing was declared,
launched or stopped here.**

### 17.1 Completion

| run | declared | live_done | failed-only | verdict |
|---|--:|--:|--:|---|
| `avifdoe-svt-b6-20260902` (encode, native) | 25,056 | 25,056 | 0 | COMPLETE |
| `avifdoe-svt-b6-sf-cpu-20260902` (score, `ssim2,zensim`) | 2,112 | 5,907 | 0 | COMPLETE |

**Score coverage is 100% counted correctly.** The score run's `done > declared`
is the pre-sort-fix rework echo — score jobs are chunk-keyed, so a
re-declaration after the chunk sort changed mints new job identities over the
same cells. Deduped by cell: **23,489 distinct bitstreams, 23,489 scored, 0
cells missing `ssim2`, 0 missing bytes.** §15.1's declared shape reproduces
exactly (27 strata × 928, every stratum full).

### 17.2 The registered question, answered

B-6's registered purpose was to find out whether reduced-size screening was
*miscalibrated* or *directionally wrong* for the two arms that failed T1.
**Answer: miscalibrated.** On every (speed, knob) cell where **both** legs carry
an effect above ±0.5%, budget and native agree on **sign 1.000 of the time**.

- **`acb3` is NOT-MEASURED at native, not FAIL.** Only 2 of 11 cropped
  references clear ±0.5% at speed 4 and **0 of 11** at speed 6. Stage A's
  T1 = 0.25 was four references that cleared the floor on a **3-point** ladder.
- **`shp3`'s failure is real and speed-4-specific**: FAIL-T1 at s4 (T1 **0.75**
  vs the 0.80 bar, up from 0.62) and **PASS at s6** (T1 **1.000**). §4.2's
  collapsed speed axis is why Stage A could not see this.
- **T1 has a construction defect.** Its denominator counts a reference by its
  *native* effect only and asks nothing of the budget leg, so it grades sign
  agreement against **budget-side noise**. §8.2 named the vanishing-effect half
  of this; the surviving-effect half is worse and produced one of B-6's two
  arms. **Registered as a defect, NOT fixed here** — amending a pre-registered
  bar after seeing results is the coordinator's call.

### 17.3 The knob verdicts

- **`ac_bias`: no native effect at any level.** 12 (speed, level) medians span
  **−0.03% to +0.39%** BD-rate; at speed 7 every level is within ±0.02%.
  `acb8` — the unclamped **H-10** level — is genuinely live (byte-identical to
  the control on only 6.7% of cells, median **+0.559%** bytes, p95 +5.67%),
  confirming §15.4's probe at corpus scale, and it is **the only `ac_bias` level
  with a defensible effect and it is a LOSS**. All 2,784 `acb8` cells are flagged
  as exercising an argument `SvtParams::clamped()` does not clamp.
- **`sharpness`: a pure bit cost, rising toward the fast presets.** `shp7`
  **+7.15% / +7.99% / +9.46%** at s4 / s6 / s7; `shp5` +5.57 / +5.67 / +7.52;
  `shp3` +1.20 / +1.49 / +2.75; **`shp1` free** (+0.00 to +0.23%, moves the
  bitstream on 90% of cells and its size by +0.000%). Class spread is
  **10.97 pp** (`shp7` @s6: plot +0.17%, ai-gen +11.13%). **B-3 does not fire**
  — no B-6 knob has opposite-signed class medians past ±1%.
- **Neither earns a place in the per-image knob set.** A perfect oracle over all
  8 levels buys a corpus median **−0.23 / −0.09 / −0.01 pp**; a realistic
  speed-4-trained rule buys **−0.05 to −0.24 pp**. `ac_bias` is not learnable
  (per-image sign stable across the three presets on 6–15 of 31 images);
  `sharpness` is (26–29 of 31) but what is learnable is *"leave it at 0"*.
  **One lead, n = 1–2**: `7004` and `7058` (both `plot`, 1.05 MP) are where
  sharpness pays, −7.41% and −5.02% for `shp7` at s7.

### 17.4 QM × sharpness at native — NOT MEASURED, and it stays that way

B-6 carries **no QM axis** and no cross-wave join can manufacture one (Stage A's
pair cells are 1024² crops whose native twins are provably different pixels:
0/2,535 byte-identical). What B-6 re-measures is the **sharpness half of the
additive baseline**, which shifts **−0.39 pp** at corpus level (−1.86 pp on
cropped references alone, but 19 of 32 references are passthroughs contributing
exactly zero). A −0.4 pp shift does not overturn §7.1's −5.2 to −5.5% residual,
so **size is not a plausible explanation for the synergy** — but that is an
argument about one input, not a measurement. **Confirming it needs a
`(qml × shp)` grid at native; no declared wave contains one.**

### 17.5 Three corrections this wave owes the record

1. **§15.4's preset column is wrong.** Measured by byte identity against the
   naive preset × q sweep, the mapping is literal: `s4`↔4, `s6`↔**6**, `s7`↔**7**
   — **928/928 byte-identical** each, **0/928** against every other naive speed.
   The CPU-cost measurements in §15.4 are unaffected; only the labels are.
2. **Stage A §5.3's `6006` / `6018` exclusion is CROP-specific.** At native both
   are ordinary images — `6006` yields BD-rates at all three speeds and is the
   *largest* effect in the wave (`shp7` @s7: **+18.72%**); only `6018` at speed 4
   still degenerates. **A native wave must re-test degeneracy at native rather
   than inheriting the budget exclusion list.**
3. **The naive sweep's speeds 7, 8, 9 and 10 are byte-identical** (1–6 are
   mutually distinct) — the preset saturates at 7, so its speed-8/9/10 strata
   (630 cells) measure preset 7 three more times. Same *shape* as §3's inert-knob
   finding but on the **speed** axis. Flagged for the port program; **no issue
   opened by this lane.**

### 17.6 Budget spent, and what it says about the remaining triggers

B-6 was costed at **13.2 CPU-h encode** (§15.4) against the 60 CPU-h Stage-B
envelope — **0.22×** — and it retired the two arms §10 called *"the cheapest
high-value cells"*. It bought: two knobs removed from the tuning-model candidate
set, a construction defect found in the gate that generates B-6 triggers, and a
native-size sharpness table that the QM × sharpness follow-up (B-2's strongest
cluster, ~15,552 cells) would otherwise have had to buy for itself.

**Still undeclared: the other 53 triggers.** Nothing here changes §10.1's 7.5×
overrun or prioritises what comes next — that remains the coordinator's decision.
Two inputs this wave adds:

- **Anything keyed on `acb3` or `shp3` at reduced size should be re-scoped or
  dropped**, and the same question should be asked of the 3 NOT-MEASURED arms
  (`acb1`, `tl1.0`, `tl1.1`): their verdict may likewise be "no effect to
  transfer" rather than "untested".
- **B-2's QM × sharpness cluster is now the only route to the synergy question**,
  and B-6 has removed the concern that its residual is a size artefact.

---

## 18. Stage-B remainder + the timing instrument — declared, launched, registered (2026-09-03)

**Records:** [`avif_stageB_remainder_2026-09-03.md`](avif_stageB_remainder_2026-09-03.md)
and [`avif_speed_instrument_2026-09-03.md`](avif_speed_instrument_2026-09-03.md).
This section is the pointer; those carry the grids, gates and limitations.

### 18.1 §3.6's A4 timing block is BUILT — B-4 is no longer structurally unevaluable

§3.6 identified that `encode_ms` is persisted by no fleet path and specified a
single-host instrument. It is now running on **r7900x, exclusively and uncontended**.
Three corrections to §3.6's design, all measured:

1. **It is `--no-score`.** §3.6 did not say to disable scoring. MEASURED on the first
   launch: **23 min of wall clock against 15.0 s of `encode_ms`** — scoring was ~99 %
   of the run, *and* running a multi-threaded metric on every core between two
   single-threaded timed encodes is a systematic perturbation of the measured
   quantity. `zenmetrics sweep --no-score` was added for this (`3f7281d1`); the first
   run was discarded and relaunched.
2. **The ladder is CROPS, not §3.6's "Lanczos derivatives".** §2.4 already argues
   this for RD — a resampling kernel removes exactly the high-frequency signal — and
   it binds harder for *time*, which is driven by that residual: a Lanczos-reduced
   64² tile is not a small image, it is a smooth one, and fitting α on it charges the
   intercept for a content change.
3. **A backend axis was added** (coordinator requirement, 2026-09-03): svt-rs **and**
   zenrav1e, full 10-speed dial each, because backend selection is now a model output.

**Already measured, before the fits:** svt-rs is **52× faster** than zenrav1e at
1024²/s6/q45 (143.9 vs 7,539.8 ms) *and* 40 % smaller *and* higher zensim; the dials
**alias differently** (svt saturates at preset 9, zenrav1e still moves at speed 10
and aliases at 7/8 on another image); zenrav1e's per-pixel cost has a **3.4× content
spread**. **The knob-time half of §3.6 is BLOCKED with the blocker named** — the
sweep's AVIF knob vocabulary does not carry the DOE deviations, and the plan path
that does runs through the jobexec kind that persists no `encode_ms`.

### 18.2 Two Stage-B runs, 16,768 cells of the ~34,944 remaining

`avifdoe-svt-brnat-20260903` (7,488, NATIVE) is §17.4's missing **native QM ×
sharpness grid**, as a complete 4×3 factorial, plus mtx32 and tune×tile; A2 ran the
same plan at budget with the same ladder, so it is a clean **size A/B on the
interaction**. `avifdoe-rav-brsdr-20260903` (9,280, BUDGET) is the **zenrav1e SDR RD
arm** — the corpus had zero zenrav1e SDR coverage. Neither needed a zenavif change
or a new image. `vbst`, `scm3` and `acb` are excluded with citations (§10.1(4);
era-delta's 0/288 at speed 6; §17.3). Scoring was declared **at launch**.

### 18.2b ⛔ §3.6's q-flatness premise is FALSIFIED

§3.6 justified pricing the timing block at 3 q points with: *"measured: per-q cost
is flat to ±1.2 % on aom and ±3.3 % on svt — retrofit §9.2 — so q density buys a
speed model nothing."* The instrument tested it rather than inheriting it.
**MEASURED (S1b, 180 cells): svt-rs median relative spread over q {15, 45, 90} is
0.752 (75.2 %), max 3.528; zenrav1e median 0.433, max 2.671.** The svt median is
**~23×** the registered tolerance.

**Cost RISES with quality** — 46 of 60 cells monotone up, **0** monotone down. So
**the speed model needs a q axis**; (backend, speed, pixels) alone is not
sufficient. Every α + β coefficient the instrument reports is **q45-specific**.

This also touches the Stage-B de-scope ladder, whose step 1 ("29-q → 9-q") was
costed as ~69 % of cost for ~69 % of cells on the flatness premise — with cost
rising in q, *which* points are cut changes the saving. Record:
[`avif_speed_instrument_2026-09-03.md`](avif_speed_instrument_2026-09-03.md) §6.4b.

An anomaly is flagged there and left open: all 14 non-monotone cells are zenrav1e
speeds 2-4, where q90 runs up to **3.0× faster** than q45 — counter-mechanistic and
unexplained by this instrument.

### 18.3 Three corrections this lane owes the record

1. **`avifdoe_build_budget_corpus.py`'s pixel budget did not follow `--side`.**
   `BUDGET_MP` was hardcoded to the default side's 1.048576, so at any other side the
   passthrough test used the wrong threshold — a sub-1.048 MP source passed through
   whole where a crop was asked for (MEASURED: `8288` at `--side 64`). **Fixed**
   (`6b9101b7`); at side 1024 the two are the same number and the registered corpus
   rebuilds **5/5 byte-identical**, so nothing already encoded is affected.
2. **The drained-worker restart loop is systemic, not incidental.** `unless-stopped`
   plus the worker's self-exit-on-drain means that the moment a run completes, docker
   restarts the worker forever. Found on **five** containers today across three
   hosts — B-6's on r7900x and tower at **2,466 restarts each**, and another lane's
   three era-delta workers at 458 / 561 / 178 — every one reporting `done=0` and
   re-fetching a 14 MB manifest each cycle. §15.7 found the same shape once; it is
   now clearly a pattern worth a fix at the launcher, not repeated cleanup.
3. **`rsync -a` on the native corpus stages BROKEN LINKS.** 30 of its 32 entries are
   symlinks into `/mnt/v/output/imazen-26-png/`, so a plain `rsync -a` to a fleet box
   produced 30 dangling links and 2 real files — a run would have quietly encoded 2
   images and merely looked small. Use `--copy-links` and verify
   `find -type f | wc -l` before arming anything.
