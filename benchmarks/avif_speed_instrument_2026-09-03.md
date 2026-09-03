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

---

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
   ±3.3 % svt claim is the thing under test.
5. **Speeds are categorical.** svt speeds 7–10 are one encoder; zenrav1e's aliases
   are image-dependent. No ordinal trend is fitted across a saturation seam.
6. **min over the 3 passes** is the reported statistic; the pass-to-pass spread is
   reported beside it as the drift control.

---

## 6. Results

**PENDING — the run is in flight.** This section is deliberately empty rather than
filled with the discarded first run's numbers. What is already established and
carried above: the `--no-score` cost/bias measurement (§1.1), the two backends'
speed/byte curves at 1024² and the alias structures (§3.3), and the zenrav1e content
spread (47.2 s/MP photo vs 160.9 s/MP screenshot, summed over the 10-speed dial).

Outputs will land at `~/speedinstr/out/run2/` on r7900x as
`s1a_pass{1,2,3}.tsv` + `s1b_pass{1,2,3}.tsv`, with `COMPLETE` as the terminal
marker, and are copied to `/mnt/v/output/avif-speed-instrument-2026-09-03/` with a
`_MANIFEST.json` carrying `build_commit`, the binary sha256 and the host block.

---

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
