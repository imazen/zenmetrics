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
5. **Total: 56,080 cells, ≈ 46 CPU-h encode + ≈ 47 CPU-h score.** Two days of LAN-fleet
   wall time at measured throughput. It fits *because* of the aom shrink.

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

---

## 3. The design

### 3.0 Structure

```
                       tune  (OUTERMOST — H-4: rewrites 9 other knobs)
                        │
      ┌─────────────────┼──────────────────┐
   A0 control        A1 main effects    A2 pairwise
   deviations = 0    deviations = 1     deviations = 2
   29-q × 7 presets  9-q × 3 presets    9-q × 1 preset × 16 imgs
                     (+A1b 3-q × preset 1)
```

`deviations` is **not a new concept** — it is `SweepCell::deviations`, computed by
`zenavif::sweep::cross()` as the number of axes whose value index is non-zero, with
index 0 pinned to the production default on every axis. `SweepBuilder::with_max_deviations`
already bounds it. So "main effects, then pairwise" is the planner's own ladder, and
`run.plan.json` reports every stratum the design drops.

### 3.1 A0 — control (already in flight; svt keep, aom shrink)

The `deviations = 0` stratum at each backend's default envelope. This is the reference
every knob arm is differenced against, **and** the per-image RD backbone.

| arm | grid | cells |
|---|---|--:|
| **A0-svt** | 7 presets × 29 q × 32 img — **unchanged** | 6,496 |
| **A0-aom** | **SHRUNK** (§4): `cpu-used {4,6,8,9}` × 29 q + `{2,3,5,7}` × 9 q + `{0,1}` × 5 q, × 32 img, minus 5 poison cells | 5,184 |

A0-aom's 5-point ladder at `cpu-used` 0/1 is `q ∈ {5, 25, 45, 76, 96}`.

### 3.2 A1 — main effects, svt-rs

17 single-deviation knob arms × **speeds {4, 6, 7}** (= SVT presets 4, 7, 9) × the
9-point ladder × 32 images = **14,688 cells**.

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

### 3.3 A1b — the slow-preset interaction probe, svt-rs

17 arms × **speed 2** (= SVT preset 1) × 3-point ladder × 32 img = **1,632 cells**.

Rationale: A1 covers presets 4/7/9. A knob's effect can invert at the slow end (the
preset table turns SG restoration, Wiener, filter-intra, wedge/diff-weighted prediction
on below M4/M7/M9, so the search a knob perturbs is a different search). This block is
the cheapest instrument that can detect such an inversion: it asks only for the *sign*
of each main effect at preset 1, which 3 q points answer. Speeds 1 and 3 are not probed
— speed 1 is 43.5 % of the arm's whole encode cost for a preset that no production
still encode uses.

### 3.4 A2 — pairwise interactions, svt-rs

All two-deviation combinations of the 17 A1 levels — `((Σd)² − Σd²)/2 = (289 − 37)/2 =`
**126 strata** — × **speed 6** × the 9-point ladder × **16 images** = **18,144 cells**.

- **Speed 6 (preset 7)** is the single point: it is the neighbourhood of libavif's own
  default (`--speed 6`), and it costs 0.85 % of the svt arm's encode CPU.
- **16 images, not 32.** Per the brief's stated priority ("cut representative-image
  count before cutting q density"), the image axis is what gives. Selection rule,
  deterministic: **the largest cluster of each of the 12 content classes, then the 2
  smallest-MP and 2 largest-MP remaining picks.** Result: 12/12 content classes,
  MP 0.25–16.00 (all four size decades retained — an interaction that only appears on
  tiny or on 16 MP inputs is still visible), 60.2 % of the 1,082-image population.
  Frozen list in §9.
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

Grid: 23 arms × `cpu-used 6` × 9 q × 32 img (6,624) + 23 × `cpu-used 8` × 3 q × 32 img
(2,208) + 23 × `cpu-used 4` × 3 q × 16 img (1,104) = **9,936 cells**.

- `cpu-used 6` is libavif's default speed and carries the dense ladder.
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

### 3.7 Totals

| block | cells | encode CPU-h |
|---|--:|--:|
| A0-svt control (in flight, unchanged) | 6,496 | 6.25 |
| A0-aom control (**shrunk** from 9,280) | 5,184 | 32.4 |
| A1 svt main effects | 14,688 | 4.55 |
| A1b svt slow-preset probe | 1,632 | 2.64 |
| A2 svt pairwise | 18,144 | 1.19 |
| A3 aom main effects | 9,936 | ~4.0 |
| **fleet total** | **56,080** | **≈ 51** |
| A4 timing (single-host, not a fleet run) | ~1,800 | ~2 |

Of the fleet total, **4,415 cells are already done** (3,562 svt + 853 aom at
2026-09-02T01:55Z).

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
   (image, speed)**, integrated over the ladder's quality span.
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
| **B-5** | any A1 main effect **inverts sign** between A1b (preset 1) and A1 (presets 4/7/9) | that knob gets the full preset ladder, 9-q, 32 images |

**Stage-B budget envelope:** ≤ 60,000 cells, ≤ 60 CPU-h encode, ≤ 50 CPU-h score,
declared as `avifdoe-{svt,aom}-b-<date>` through the same canonical builder.

### 7.3 Pre-registered de-scope rule (budget gate)

If measured throughput 6 h after the DOE launch implies **> 4 days** to drain, **A2 is
withdrawn** (declaration shrunk, per §4.4) — it is the largest block, the most
dispensable (interactions refine an attribution that main effects already give), and
the last declared. A1/A1b/A3 are not withdrawn; if they alone exceed the envelope, the
image axis is cut per §3.4's rule before any q density is.

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

**A2 16-image subset** (rule: largest cluster of each of the 12 content classes, then
the 2 smallest-MP and 2 largest-MP remaining picks; 12/12 classes, MP 0.25–16.00,
60.2 % population coverage):

```
1008.scale3000x4000.png  1220.scale3000x4000.png  1420.scale3000x4000.png
1432.scale3000x4000.png  1634.scale3000x4000.png  3006.scale3000x2235.png
6038.scale2479x3230.png  6602.scale3302x4844.png  6604.scale3286x4868.png
7076.scale1024x1024.png  8288.scale375x667.png    8434.scale414x896.png
8446.scale2560x1440.png  9032.scale1024x1536.png  9100.scale1024x1536.png
9954.scale1024x1536.png
```

**Run names**

| run | contents |
|---|---|
| `avifsub-svt-enc-20260901` / `avifsub-aom-enc-20260901` | A0 control (existing; aom shrunk in place) |
| `avifdoe-svt-a1-20260901` | A1 + A1b |
| `avifdoe-svt-a2-20260901` | A2 |
| `avifdoe-aom-a3-20260901` | A3 |
| `avifdoe-*-sf-cpu-20260901` | score runs (`ssim2,zensim,butteraugli` + a zenanalyze feature vector per variant) |

**Gates that must pass before each fleet scale-up** (two-stage launch):

| gate | what it checks |
|---|---|
| G-DEDUP | `--dry-run` flatten count vs the naive cross product; every merge explained by a fingerprint mediator or a dossier hazard, recorded in `<run>.plan.json` |
| G-FIRSTCELL | after the 1-worker smoke, the first cell's **encoded bytes AND its score row** are listed in the store with the `aws` CLI (never s5cmd — it undercounts on the LAN store) |
| G-AOM-BASE | §5.3 — the ctrls path reproduces the defaults base byte-for-byte |
| G-AOM-ARM | §5.3 — all 23 aom arms byte-match on the smoke set, or are dropped with their error class recorded |
| G-CLAMP | `variance_boost_strength ∈ 1..=4`, `variance_octile ∈ 1..=8` enforced in the harness (**H-10**) |
| G-CONTROL | every DOE worker launched with `ZEN_CONTROL_KEY` set, so the wave is pausable (unlike A0's) |
