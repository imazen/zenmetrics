# AVIF speed instrument — the tuning model's third axis

**Status: PROTOCOL REGISTERED, RUN IN FLIGHT 2026-09-03.** §1–§5 are fixed before
the data exists. §6 carries what has landed; §7 is honest about what has not.
Companion: [`avif_stageB_remainder_2026-09-03.md`](avif_stageB_remainder_2026-09-03.md)
(the RD arms, on the fleet).

---

## 0. Why this exists

**`encode_ms` is persisted nowhere.** Plan doc
[`avif_doe_plan_2026-09-01.md`](avif_doe_plan_2026-09-01.md) §3.6 found it and the
era-delta wave re-confirmed it: `jobexec` emits `encode_ms` only on the `metric` job
kind (which re-encodes), while the DOE runs `encode` (whose output *is* the
content-addressed bytes) and `score_file` (which scores persisted blobs without
re-encoding). The ledger schema has no timing column. So **the entire 130k-cell DOE
produces no speed-model input at all**, and trigger **B-4** — "the A4 speed fit's
held-out MAPE > 25 %" — has been NOT EVALUABLE since Stage A.

The tuning model needs three axes: bytes, quality, and **time**. Two are covered.

**And as of 2026-09-03 it needs time PER BACKEND.** Coordinator design requirement,
mid-flight: *"we want the model to be able to pick the best backend for an image."*
Backend selection is a model **output**, so a speed model that only knows svt cannot
express the decision it is being asked to make.

---

## 1. The owner, and why nothing new was written

`zenmetrics sweep` **already persists `encode_ms`**
(`crates/zenmetrics-cli/src/sweep/run.rs:1219`). It is therefore the owner, and this
instrument is a **protocol driver around it** — `scripts/jobsys/avif_speed_instrument.sh`
— not new timing code. Two things were added to the owner and nothing was forked:

- **`--no-score`** (this session, `3f7281d1`): encode and time every cell, score
  nothing. Implemented as the *absence* of an implementation — the flag leaves
  `metrics` empty, which every scoring loop in `sweep::run` already treats as
  nothing-to-score and which the header builder already renders as no `score_*`
  columns. It **conflicts with** `--metric` rather than silently winning: silently
  dropped scores are indistinguishable from a scorer that failed on every cell.
- The 7-rung size ladder is built by the **existing** budget-corpus builder
  (`scripts/jobsys/avifdoe_build_budget_corpus.py`) run at seven `--side` values.
  Zero new cropping code.

### 1.1 Why `--no-score` is a correctness fix, not a tidiness one

MEASURED, and it is the reason the first launch was discarded: on r7900x the
instrument's first run carried **23 minutes of wall clock against 15.0 s of
`encode_ms`** — scoring and its per-cell overhead were **~99 %** of the run. Beyond
the cost, it is a *bias*: a multi-threaded perceptual metric run on every core
between two single-threaded encodes leaves the package in a different boost and
thermal state than the encode it precedes. That is a systematic perturbation of
exactly the quantity being measured.

Byte-neutrality verified: the same cell encodes to **71,552 bytes** with and without
the flag; only the score column disappears. Rate after the change: **66 rows in 40 s**
against 57 rows in 20 minutes — ~20× on the cheap cells.

---

## 2. Protocol

Per the perf discipline (zensim `CLAUDE.md`, "PERF MEASUREMENT: the noise floor at
2304² is 10 %, and it is ASLR"):

| rule | how it is met here |
|---|---|
| **ONE binary, runtime arms only** | backend, speed and q are all `--knob-grid` / `--q-grid` values of the same process, so a build-layout shift cannot masquerade as an arm |
| **Arms INTERLEAVED inside a pass** | both backends live in **one** `--knob-grid`, so the sweep's own Cartesian walk alternates them per image — not two separate runs compared after the fact |
| **min-of-N over N separate PROCESS STARTS, ASLR on** | 3 full passes, each its own process; layout is an input, not noise to be averaged |
| **uncontended** | r7900x, `--jobs 1`, nothing else on the box (verified: load 0.05, zero containers, at launch) |
| **not niced** | `nice`/`ionice` bite only under contention; on an idle dedicated box they add scheduling artifacts to the thing being measured. r7900x is a dedicated LAN worker, not the shared dev box or the household tower |
| **drift control** | the 3 passes re-measure every cell, so the per-cell spread across passes **is** the drift control; it is reported, not assumed away |

**Host, recorded because it is part of the measurement:** r7900x, AMD Ryzen 9 7900X,
12C/24T, governor `powersave`, glibc 2.43. Binary sha256
`cdbc45a418b5d3b9bbf63aff8aabd82a85cf9e14e2af412f6e9c61035906eb7e`.

**Governor note, stated rather than fixed:** `powersave` (amd-pstate) is not the
most stable clock source. It was left alone rather than changed under a wave — the
min-of-N estimator is the discipline's own answer to clock excursions, and the
pass-to-pass spread in §6 is what will show whether that was sufficient.

---

## 3. The grid

### 3.1 S1a — the α + β·pixels ladder

**32 ladder inputs × 10 speeds × 2 backends × q45 × 3 passes.**

The ladder is **crops, not downscales**, and that is load-bearing. Plan §2.4 argues
it for RD — "a resampling kernel is a low-pass filter, i.e. it removes exactly the
signal those knobs act on" — and it applies at least as hard to *time*: encode cost
is driven by high-frequency residual, so a Lanczos-reduced 64² tile is not a small
image, it is a **smooth** image, and fitting α on it would charge the intercept for
a content change. A native-resolution crop varies pixel count and holds content
statistics fixed, which is precisely what an `α + β·pixels` fit needs to identify.

Every rung of a source is centred on the **same** cluster-preserving crop position,
so the rungs are concentric windows on one content sample.

| source | class | native | rungs (px, square) |
|---|---|---|---|
| `1008` | photo, general | 3000×4000 | 64 128 256 512 1024 2048 2816 |
| `1442` | photo, nature | 4000×3000 | 64 128 256 512 1024 2048 2816 |
| `6602` | manuscript scan | 3302×4844 | 64 128 256 512 1024 2048 2816 |
| `6006` | patent scan, 1-bit | 2320×3408 | 64 128 256 512 1024 2048 |
| `8446` | web screenshot | 2560×1440 | 64 128 256 512 1024 |

**32 inputs, 4,096 → 7,929,856 px — 3.3 decades**, tiny bucket included, per the
discipline's size mandate. **No rung is ever an upscale**: a rung is emitted only
where the builder produced a true `side × side` crop, which is why the last two
sources are short. That asymmetry is a stated limitation (§7), not a silent one.

### 3.2 S1b — the q-flatness probe

**3 sizes (`1442` at 256/1024/2048) × 10 speeds × 2 backends × q {15, 45, 90} × 3 passes.**

The plan asserts per-q encode cost is flat (±1.2 % aom, ±3.3 % svt, retrofit §9.2).
The brief says **verify, don't inherit** — and no such number exists for zenrav1e at
all. Splitting this out is what makes S1a affordable at the full 7-rung ladder: q is
a separate question from the size fit, so it is asked on 3 sizes rather than 32.

### 3.2b S1c — the content-class block

**32 corpus images × 2 sizes (native + budget) × q45 × 3 passes**, svt at all 10
speeds and zenrav1e at **{4, 7, 10}**.

S1a's ladder buys 3.3 decades of pixels from **5** sources, i.e. 5 of the corpus's
**12** content classes. S1c buys **all 12**, at two sizes. Neither substitutes for
the other and both are needed: content changes β by a MEASURED **3.4×** on
zenrav1e, while pixel count is what α + β is a fit *in*. Run second so the α + β
deliverable is not held behind it.

zenrav1e is restricted to 3 speeds here because at native the corpus is **161.59 MP**
and its slow end costs 47–161 s/MP summed over the dial — the full ladder there
would cost more than the rest of the instrument combined. svt keeps all 10 (it is
~52× cheaper). **That restriction is a budget decision and is stated, not hidden:
zenrav1e's speeds 1–3 have no content-class coverage.**

### 3.3 Both backends, full 10-speed dial

svt-rs and zenrav1e (`backend: "svt-rs"` / `"zenravif"`), speeds 1–10 each. The
dials alias **differently** and measuring the alias is free:

| MEASURED, r7900x, `1442` 1024², q45 | s1 | s4 | s7 | s10 |
|---|--:|--:|--:|--:|
| svt-rs `encode_ms` | 3959.1 | 489.4 | 17.36 | **17.29** |
| svt-rs bytes | 3572 | 3726 | 4163 | **4163** |
| zenrav1e `encode_ms` | 24110.5 | 2545.0 | 1009.3 | 300.5 |
| zenrav1e bytes | 4559 | 5640 | 5896 | **6628** |

svt saturates at preset 9 — speeds 7–10 byte- and time-identical (H-2, and B-6
§17.5(3) found the same on the naive sweep). **zenrav1e's dial is still moving at
speed 10**, and on a different image (`8288`) it aliases at **7/8** instead
(15583 B, 747.8/742.2 ms). Assuming one backend's alias structure transfers to the
other would have silently thrown away a third of the zenrav1e dial.

### 3.4 aom-rs — PLANNED-BLOCKED

Not in this instrument. It needs `--features avif-aom` (a build this box does not
have: verified, every local binary answers *"this build lacks the `avif-aom`
feature"*) **and** era pins post-#15. Timing a port still landing byte-identity
fixes measures a moving target. It re-enters by re-running this same script with an
`avif-aom` binary — no design change.

---

## 4. What this instrument does NOT cover, and why

**The knob-time axis is registered and BLOCKED, with the blocker named.** Plan §3.6
wanted "the A1/A3 knob arms at 2 sizes and 1 q, so the model can price a knob's
*time*". It cannot be done through this path today:

- `zenmetrics sweep`'s AVIF knob vocabulary is `speed, lossless, backend,
  partition_range, lrf, fast_deblock` (`sweep/encode.rs:191`). The DOE's knobs —
  `tune`, the QM window, `scm`, `bd10`, `sharpness`, `ac_bias` — **are not in it**;
  they are fleet-side deviations reachable only through the zenavif plan path.
- The plan path runs through `jobexec`, which is exactly what does not persist
  `encode_ms`.

So the two options are (a) extend the sweep's AVIF vocabulary to the DOE deviations,
or (b) add `encode_ms` persistence to the `encode` job kind. Both are real work with
a schema or contract consequence; neither is a thing to do mid-wave. **Registered,
scoped, not silently dropped.** What ships instead: the backend × speed × size ×
q surface, which is what backend picking actually needs.

**VERIFIED at the API level, 2026-09-03 — the blocker is real, and here is its exact
shape.** Option (a) is not a zenmetrics-only change. The `--knob-grid` path builds a
zenavif **`EncoderConfig`**, and that type exposes only `bit_depth()` and
`with_qm(bool)` of the knobs this DOE sweeps — there is **no** setter for `tune`,
`sharpness`, `ac_bias`, screen-content forcing, or the QM *window* (`with_qm` is a
boolean, not the `qml<min>.<max>` range the arms use). Those live on `SvtParams`,
reached only through the `SweepAxes` plan path. **So (a) requires a zenavif change**,
and zenavif is currently claimed by another lane — which is why this stays registered
rather than attempted.

**What IS reachable today without touching zenavif**, and is the natural next
tranche: `bd10` via `bit_depth()`, QM **on/off** via `with_qm()`, plus the sweep's
existing AVIF knobs (`partition_range`, `lrf`, `fast_deblock`). That is a partial
knob-time block — enough to price bit-depth and the coarse QM switch, not the QM
window or the tune/sharpness/ac_bias family. Recorded so the next lane knows the
split rather than re-deriving it.

---

### 4.1 S2 — the reachable knob-time partial, PRE-REGISTERED and ready to run

Registered now so the next lane runs it rather than re-deriving it. It needs **no
zenavif change** — but it is **not zero-change**: `bit_depth` and `qm` are **not in
the sweep's `AVIF_KNOBS`** today (which is `speed, lossless, backend,
partition_range, lrf, fast_deblock`), so S2 requires first adding those two names and
wiring them to `EncoderConfig::bit_depth()` / `with_qm()`. That is a **zenmetrics-only**
edit of a few lines against setters that already exist, which is exactly what makes
this the reachable tranche and the `qml`/`tune`/`shp`/`acb` family the blocked one.

**Grid:** the 5 ladder sources at 2 rungs (**512², 1024²**) × **q45** × 3 passes,
svt-rs at speeds **{4, 6, 7}** (the DOE's own presets), across:

| arm | knob-grid | prices |
|---|---|---|
| control | `{}` | baseline |
| `bd10` | `{"bit_depth":["ten"]}` | 10-bit's time cost — the one B-3 arm with a measured RD win |
| `qm-on` | `{"qm":[true]}` | the coarse QM switch (NOT the `qml` window) |
| `pr` | `{"partition_range":[...]}` | |
| `lrf` / `fastdb` | `{"lrf":[...]}`, `{"fast_deblock":[...]}` | |

**≈ 10 inputs × 3 speeds × 6 arms × 3 passes ≈ 540 timed encodes**, well under an
hour at svt speeds — svt's measured β at s6 is 65.0 ms/MP, so the whole block is
minutes of encode, not hours. Run it with the **same protocol as S1a** (`--no-score`,
`--jobs 1`, min over process starts, uncontended host) and analyse with
`avif_speed_analyze.py`.

⚠ **Two things this block CANNOT price, and they are the expensive ones:** the QM
*window* (`qml1.2.10` etc.) and the whole `tune` / `sharpness` / `ac_bias` /
screen-content family. Those need `SvtParams`, i.e. a zenavif change (§4). Do not
let a clean S2 result read as "the knob-time axis is covered".

⚠ **`bit_depth` and `qm` must be added to `AVIF_KNOBS` first** — `reject_unknown_knobs`
hard-errors on an unrecognised knob (correct behaviour, but it will abort the block on
launch). `partition_range`, `lrf` and `fast_deblock` are already in the vocabulary.

## 5. Analysis plan (fixed before the data)

1. **Fit `encode_ms = α + β·pixels` per (backend, speed)**, over S1a's 32 inputs.
   **Report BOTH terms, always.** A bare "ms/MP" is meaningless without the
   intercept — at 64² the intercept dominates and at 2816² the slope does.
2. **Do not assume the fit is linear.** The zensim perf work found per-pixel cost
   *rising* with size, which makes a linear fit return a negative intercept — the
   model failing, not a saving. If α comes out negative, report it as a model
   failure and fit `α + β·pixels^γ` instead, stating the change.
3. **Content is a factor, not noise.** β is fitted per source as well as pooled;
   the measured 3.4× content spread on zenrav1e (§6) means a single pooled β would
   be wrong for both ends.
4. **q-flatness is a verdict, not an assumption**: per (backend, speed, size),
   report max relative spread of `encode_ms` across q {15, 45, 90}. The plan's
   ±3.3 % svt claim is the thing under test. **ANSWERED in §6.4b — falsified.**
5. **Speeds are categorical.** svt speeds 7–10 are one encoder; zenrav1e's aliases
   are image-dependent. No ordinal trend is fitted across a saturation seam.
6. **min over the 3 passes** is the reported statistic; the pass-to-pass spread is
   reported beside it as the drift control.

---

## 6. Results — FIRST FITS (pass 1, partial: 16 of 32 ladder inputs)

**Scope, stated first.** S1a **passes 1 AND 2 are COMPLETE — 640 cells each**, so
drift control is measured (§6.4c). The α+β table below is the min-of-2 estimate; it
moves the coefficients by well under 1 % from pass 1 alone, as the 0.44 % median
drift implies. Originally recorded as: S1a **pass 1 COMPLETE — all 640 cells** (32 ladder
inputs × 10 speeds × 2 backends), across all **5 sources**. Still **one pass**, so
**no drift control yet** (the tool prints `NOT MEASURABLE`), and **q-flatness NOT
MEASURED** (S1b runs after S1a). Passes 2-3 and S1c are in flight.

### 6.1 The headline: the α + β·pixels model is RIGHT, and pooling is what breaks

| | pooled R² | per-source R² (median) | β spread across sources |
|---|---|---|---|
| all 20 arms (full pass, 5 sources) | **0.627 – 0.906** | **0.9969 – 0.9998** | **1.95× – 24.30×** |

A pooled fit that scores 0.60 looks like a failed model. It is not. **The same arm
fit per source lands at R² 0.999**, and the pooled residual is entirely β varying
with **content**. The analyzer reports this as
`POOLING_NOT_MODEL (per-source fit is clean)` on **20 of 20 arms**.

**This is the load-bearing result for the tuning model: a single (backend, speed) β
is wrong by up to 24.3×.** And it is already wrong by **5.4× between two
PHOTOGRAPHS** — `1008` (general photo) vs `1442` (nature photo), both 12 MP, same
content class, fitting β = 15,961.6 vs 2,953.5 ms/MP at svt speed 1. Adding the
scan, patent and screenshot sources takes the spread to 24.3×. So the speed model
must be **feature-conditioned per image**, not a per-(backend, speed) constant —
which is exactly what backend picking needs, since the question is always "how long
will *this* image take".

**Six arms now fit a NEGATIVE intercept** (svt 1/3/4/5, zenrav1e 9) — the classic
linear-model-failure tell. It is still pooling, not the model: those same arms fit
per source at R² 0.9969-0.9998. A negative α is what a pooled line does when it is
averaging incompatible slopes.

The nonlinearity check comes back clean on the same evidence: the power fits give
**γ = 0.93–1.09**, i.e. near-linear in pixels, and no arm has a negative intercept.
Cost is linear in pixels *within* a source.

### 6.2 α and β, both terms, per (backend, speed)

Pooled over the available sources — quote **with** the caveat above, never the β
alone.

| speed | svt α (ms) | svt β (ms/MP) | zenrav1e α (ms) | zenrav1e β (ms/MP) | β ratio |
|--:|--:|--:|--:|--:|--:|
| 1 | 874.53 | 9,514.7 | 345.16 | 35,276.0 | 3.7× |
| 2 | 424.06 | 4,117.0 | 817.43 | 23,322.6 | 5.7× |
| 3 | 105.34 | 1,323.0 | 255.80 | 13,968.5 | 10.6× |
| 4 | 34.75 | 769.0 | 261.84 | 6,480.9 | 8.4× |
| 5 | 0.29 | 287.3 | 69.48 | 1,511.4 | 5.3× |
| 6 | 2.33 | 55.0 | 53.85 | 2,101.5 | 38.2× |
| 7 | 0.50 | **28.41** | 33.16 | **1,963.4** | **69.1×** |
| 8 | 0.51 | **28.33** | 34.03 | **1,962.3** | 69.3× |
| 9 | 0.54 | **28.32** | −0.59 | 708.5 | 25.0× |
| 10 | 0.52 | **28.33** | 0.36 | 393.7 | 13.9× |

*(rows 1-6 are the partial-pass values and are superseded by the full-pass fits in
`speed_alpha_beta.tsv`; svt 1 now fits α = −254.4, β = 13,244.4 and zenrav1e 1
α = 2,008.8, β = 49,233.5.)*

**The intercept is not decorative.** At svt speed 1, α = 874.5 ms — at the 64² rung
(0.004 MP) the slope contributes 39 ms and the intercept 875. A "ms/MP" number alone
would misprice that cell by ~20×. This is the discipline's α-matters case, measured.

### 6.3 Both aliases confirmed across the whole ladder, not one cell

- **svt speeds 7, 8, 9, 10** fit β = 28.409 / 28.334 / 28.322 / 28.330 ms/MP and
  α = 0.500 / 0.514 / 0.543 / 0.518. Four independent **32-point** regressions
  landing within **0.3 %** — the preset-9 saturation (H-2) confirmed as a property
  of the whole size ladder and all 5 content classes, not the single cell §3.3
  showed.
- **zenrav1e speeds 7 and 8** fit β = 1963.36 / 1962.33 — **0.05 % apart**. The 7/8
  alias first seen on `8288` reproduces across the full ladder.
  zenrav1e's 9 and 10 remain distinct (603.2, 328.1), so its dial genuinely runs
  longer than svt's.

### 6.4 Backend picking — what the speed axis already says

At matched *dial position* svt-rs dominates zenrav1e on time by **3.7× to 64.8×**,
and the gap is widest exactly in the fast-preset region a web pipeline would use.
Combined with the single-cell RD read (§3.3: at 1024²/s6/q45 svt is also 40 %
smaller **and** higher zensim), the SDR case for zenrav1e is looking thin — which is
precisely why the Stage-B `brsdr` arm runs at budget size first, and why a native
zenrav1e leg is registered as *gated on that result* rather than pre-bought.

⚠ **Dial position is not matched quality.** These ratios are per-speed-index, and
the two backends map speed to work differently. The quality-matched comparison is
the RD wave's job, in quality space, not q space.

### 6.4b Q-FLATNESS: the registered assumption is FALSIFIED

The plan inherited a claim from the retrofit (§9.2) that per-q encode cost is flat
— **"±1.2 % on aom and ±3.3 % on svt — so q density buys a speed model nothing"** —
and §3.2 registered *verify, don't inherit*. S1b pass 1 (180 cells: 3 sizes × 10
speeds × 2 backends × q {15, 45, 90}) verifies it, and it does not hold:

| backend | median rel. spread over q | max | registered claim |
|---|--:|--:|--:|
| **svt-rs** | **0.752 (75.2 %)** | **3.528 (352.8 %)** | ±3.3 % |
| **zenrav1e** | **0.433 (43.3 %)** | **2.671 (267.1 %)** | not stated |

**The svt median is ~23× the registered tolerance and the worst cell is ~107×.**

**Direction is unambiguous: cost RISES with quality.** Over all 60 (image, backend,
speed) cells: **46 rise monotonically across q {15, 45, 90}, 0 fall monotonically**,
14 are non-monotone. That is the expected mechanism — a lower quantizer keeps more
coefficients and does more RD work — but its *size* is what was mis-registered.
Example, `1442` at 1024², svt speed 1: **2,358.6 → 3,903.9 → 6,101.2 ms**.

**CONFIRMED ON TWO PASSES — it is not measurement noise.** S1b pass 2 (180 more
cells) landed 22:46:01Z; recomputing the verdict as **min-of-2** barely moves it:

| backend | pass 1 median / max | min-of-2 median / max |
|---|--:|--:|
| svt-rs | 0.7524 / 3.5276 | **0.7508 / 3.4988** |
| zenrav1e | 0.4325 / 2.6712 | **0.4325 / 2.6712** |

Noise is what `min()` removes. A spread that survives min-of-2 unchanged — moving
0.2 % relative on svt and **0.0 %** on zenrav1e — is a property of the encoder, not
of the measurement. Compare the drift control on the same instrument: **0.44 %
median** (§6.4c). The q effect is ~170× that.

**Three consequences, and the first is the important one:**

1. **The speed model needs a q axis.** A model over (backend, speed, pixels) alone
   is not sufficient; at fixed size and speed, q moves cost by a median of 75 % on
   svt. The claim that q density buys a speed model nothing is the opposite of what
   the data says.
2. **Every α + β fit in §6.1-§6.3 is q45-SPECIFIC** and must be labelled that way.
   S1a is a single-q block by design (§3.1) — which was the right call for
   affordability, but it means those coefficients describe q45 and no other point.
3. **A registered de-scope step rested on this.** §4.1 of the companion doc costed
   "29-q → 9-q" as cutting ~69 % of cost with ~69 % of cells, which assumes rough
   flatness. With cost rising in q, *which* q points are cut changes the saving —
   another reason that ladder should be rebuilt from measured per-axis cost.

**ANOMALY, flagged and NOT explained away.** All 14 non-monotone cells are
**zenrav1e at speeds 2-4**, where q90 comes out *faster* than q45 — on `1442` at
1024², speed 2 reads **12,811.7 / 14,933.7 / 5,005.7 ms** at q15/45/90, i.e. q90 is
**3.0× faster** than q45. Time falling as quality rises is counter-mechanistic. It
is reproducible across the three sizes in this block but has **no explanation
here**; it wants the per-q bitstream sizes and rav1e's speed/quantizer coupling,
which is the analysis lane's work, not this instrument's.

### 6.4c DRIFT CONTROL — measured, and the instrument is reproducible

S1a pass 2 completed 22:29:52Z (640 cells), so the drift control the protocol
promised is now measurable rather than `NOT MEASURABLE`.

| statistic over (max−min)/min across the 2 passes, n=640 | value |
|---|--:|
| median | **0.44 %** |
| p90 | **1.82 %** |
| max | 20.0 % |

**The worst spreads are the noise floor, not drift.** All eight worst cells are
`1008.crop0064` — the 64² tile, where a pass reads **0.175 ms** and absolute clock
resolution dominates. Split by cost: cells with min < 10 ms have median spread
**0.79 %**, cells with min ≥ 1000 ms have **0.42 %**. The instrument is *more*
reproducible exactly where the numbers matter.

**N = 3 is comfortably sufficient here**, and the concern §7 registered from the
zensim perf work (≥ 15 process starts needed to tame 10 % ASLR bimodality at 2304²)
**does not materialise**: these encodes span ms to minutes, so per-process layout is
a far smaller share than it is of a ~350 ms whole-image walk.

**Bonus determinism check: 640/640 cells are BYTE-IDENTICAL across the two passes.**
Both encoders are deterministic on this corpus, and the contended window changed
timings only, never output.

### 6.6b The 21:39Z contention had NO measurable effect — §6.6's caution is superseded

§6.6 recorded the contended window and warned that drift after 21:39Z would conflate
machine drift with another lane's build. **Measured, it did not.** The ladder walk is
alphabetical, so the contended window falls on the LATE sources — and those have the
*lowest* spread:

| source (walk order →) | 1008 | 1442 | 6006 | 6602 | 8446 |
|---|--:|--:|--:|--:|--:|
| median spread | **1.11 %** | 0.50 % | 0.21 % | **0.26 %** | **0.24 %** |

A contention signature would put the elevated spread on `6602`/`8446`, which ran
during the build. It is on `1008` instead — the first source walked, and the one
carrying the sub-millisecond 64² cells. **Whole-pass durations agree to 0.04 %**:
pass 1 S1a ran 114.48 min, the partly-contended pass 2 ran 114.53 min.

Mechanism: the sweep is `--jobs 1` single-threaded on a 24-thread box, so an 8-job
build had spare cores to use. **§6.6's window record stands as provenance; its
warning is retracted on evidence.** The estimator design (min over process starts)
was sound but did not have to do any work here.

### 6.5 Runtime, and a cost estimate that was wrong

The registered S1a pass estimate was ~2,591 s of encode. **MEASURED: 2,839 s
bought only 27.13 of the ladder's 47.55 MP** — the photo sources cost ~104.6 s/MP
summed over the 20 (backend, speed) cells, and the scan/screenshot sources that
remain are dearer. A full S1a pass is therefore **~1.7 h**, not 43 min, and with
S1b a pass is ~2.3-2.6 h; three passes plus S1c run overnight. Nothing is
blocked on that: a chained harvester pushes the outputs to the LAN store on the
`COMPLETE` marker, and a chained launcher starts S1c only after that marker AND
after the last `zenmetrics` leaves the box, so the two never overlap and r7900x
exclusivity holds unattended.

The estimate was low for the same reason §6.1 is the headline — it was built from
one image's speed curve, and β is a function of content.

**Artifacts:** `~/speedinstr/out/run2/` on r7900x (`s1a_pass{1,2,3}.tsv`,
`s1b_pass{1,2,3}.tsv`, `COMPLETE`), auto-pushed by a chained harvester to
`s3://zentrain/instruments/avif-speed-2026-09-03/run2/`; S1c to
`~/speedinstr/out/run3_s1c/` via a chained launcher that waits on run2's marker so
the two never overlap on the host. Fits reproduce with
`scripts/jobsys/avif_speed_analyze.py --s1a <pass tsvs> --s1b <probe tsvs>
--out-dir DIR`.

### 6.6 CONTENDED WINDOW on r7900x from 2026-09-03T21:39Z — and why the estimator survives it

The instrument reserves r7900x and §2 records "uncontended" as a precondition, with
the box verified at load 0.05 and zero containers at launch. **At 21:39Z another
lane started `cargo-nextest nextest run --workspace -j 8` on it** — multiple `rustc`
and `rust-lld` processes at ~100 % each. Box load went 1.00 → 3.66.

**Scope of the contamination, stated precisely rather than hand-waved:**

| block | window | status |
|---|---|---|
| S1a pass 1 (640 cells) | 18:25:11 – 20:19:40Z | **CLEAN** — verified load 1.00, single-core, zero containers |
| S1b pass 1 (180 cells) | 20:19:40 – 20:35:20Z | **CLEAN** |
| S1a pass 2 | 20:35:20Z – | clean until **21:39Z**, contended after |
| pass 3, S1c | later | unknown, depends on that lane |

**Every result published in §6.1-§6.4b is from pass 1 and is therefore clean.** The
α + β fits, the 24.3× content spread, both alias confirmations and the q-flatness
falsification all predate 21:39Z.

**And the estimator is built for exactly this.** §2 reports **min over N process
starts**, not a mean — contention can only inflate a timing, never deflate it, so a
contended pass loses to a clean one under `min()`. Since pass 1 is clean for every
cell, every cell retains at least one uncontended sample. What contention *does*
damage is the **drift control**: the pass-to-pass spread will now conflate
machine-state drift with another lane's build, so a large spread on cells measured
after 21:39Z is not evidence about this instrument. **Report drift from clean passes
only, and say which those were.**

**Not fixed by killing it.** The build is another lane's live work; the reservation
was a plan, not a lock. The durable fix is a claim mechanism for the box — the
`.workongoing` marker covers repos, not hosts — which is worth having before the
next single-host timing wave.

## 7. Limitations — stated before any result

1. **S1a's ladder covers 5 of 12 content classes**; S1c covers all 12 but only at
   two sizes and, for zenrav1e, only 3 speeds. A per-(backend, speed) β for a class
   outside S1a's five is therefore a two-point estimate, not a ladder fit.
2. **Screen content is absent above 1024²** in the ladder. No corpus source is both
   screen-class and ≥ 2048² — the largest screenshot is 3.69 MP and the largest plot
   1.05 MP. β for screen content at the large end is therefore **not measured**, and
   given the 3.4× content spread already seen, it must not be interpolated from the
   photo sources.
2. **Two sources cannot reach the top rungs**, so the ladder is ragged (7/7/7/6/5).
   Pooled fits are unbalanced by construction; per-source fits are not.
3. **One host, one governor, one glibc.** Every absolute millisecond here is an
   r7900x number. Backend *ratios* are within-binary and within-host, so they
   travel; absolute times do not.
4. **N = 3 passes** is the floor the brief set. The zensim perf work needed ≥ 15
   process starts to tame ASLR bimodality at 2304², but that was a ~350 ms
   whole-image walk; these encodes span 1 ms to ~180 s, so layout is a much smaller
   share. The measured pass-to-pass spread in §6 is what decides whether 3 was
   enough — if it exceeds a few percent on the cheap cells, N must rise.
5. **The knob-time axis is missing** (§4), so a knob's *time* price is still
   unknown; only the speed dial's is measured.
6. **Three ladder cells came back as passthroughs and were dropped**
   (`8446` at sides 2048 and 2816, `6006` at 2816). That is the builder's
   `w <= side or h <= side` clause behaving **correctly** — you cannot crop a
   2560x1440 image to a 2816 square — and it is why the ladder is ragged. It is
   *not* a defect.
7. **A separate, genuine defect in the same rule was found and FIXED**
   (`avifdoe_build_budget_corpus.py`): `BUDGET_MP` was a hardcoded 1.048576 rather
   than derived from `--side`, so at any side other than 1024 the passthrough test
   ran against the wrong threshold. It did **not** affect this ladder (all five
   sources are >= 3.69 MP, far above the constant), but it would silently break any
   ladder built from small sources: MEASURED, `8288` (0.25 MP) at `--side 64`
   passed through whole under the old rule and now crops to 64x64. The pixel budget
   now follows `--side`; at side 1024 the two are byte-for-byte the same number, and
   rebuilding the registered corpus after the change reproduced it **5/5
   byte-identical**.
