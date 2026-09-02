# AVIF knob DOE — Stage A: the wave, and what it measured

**Date:** 2026-09-02 · **Design of record:**
[`avif_doe_plan_2026-09-01.md`](avif_doe_plan_2026-09-01.md) (§7.1 analysis,
§7.2 Stage-B triggers, §3.8 transfer gate, §3.9 bytes model, §12.4 degeneracy
correction) · **Lane:** wave completion + Stage-A analysis.

This document is the completion record for the svt-rs half of the AVIF knob DOE
and the pre-registered Stage-A read of it. It ends with a mechanical Stage-B
trigger list and a budget; **it does not declare Stage B** — that decision is
the coordinator's.

---

## 0. TL;DR

1. **The wave is COMPLETE.** All four svt-rs encode runs reached live-gap 0 with
   **zero failed cells**: A1 6,912, A2 33,984, A0R 6,496, AG 1,728 — 49,120
   cells, and every one of them scored.
2. **The A1 "480 failed-only" gap in the plan's §13.8 never existed.** A1's
   ledger shows the run finished at **04:47:21Z**, ~57 minutes before that
   snapshot was taken, with 6,912/6,912 live-done and **zero post-fix
   `encoder_panic`**. There is no poison to annotate and no port-bug lead here.
3. **Two of the seventeen knob arms are dead.** `tune=0` and
   `screen_content_mode=Some(3)` produce **byte-identical bitstreams to the
   default on 288 of 288 cells, at both presets** — while the harness's own
   resolved-state fingerprint says they are different configurations. That is a
   knob plumbed to the fingerprint but not to the bitstream. It consumed
   **8,972 cells** of this wave and must not consume any of Stage B's.
4. **The plan's §3.2 cell arithmetic is impossible under its own isolation
   rule.** §3.2 registers "17 arms × all 7 effective presets" = 34,272 cells,
   but `--max-deviations 1` means a non-default preset *is* the one permitted
   deviation, so a cell cannot carry both. zenavif's own test asserts the real
   design: **24 strata**. The declared 6,912 is correct; the doc is wrong. The
   consequence is load-bearing: **knob main effects exist at two presets (s4 and
   s6), not seven**, so §7.1(3)'s "per speed" and trigger **B-5** are evaluable
   across exactly one preset pair.
5. **The AG transfer gate is real but thin, and it blocks.** It is computable on
   **13 references** (§12.4: 19 of 32 are native passthroughs where "budget" and
   "native" are the same encode), at **speed 4 only** (the declared AG carries
   knob arms at s4, not the {4,6} §3.8 registered), and over 3 q points.
6. **Four integrity gates passed**, none of them assumed: the AG corpus
   identity (3 independent proofs), A0-native ≡ A0R on all shared passthrough
   cells, A0R ≡ the in-run controls, and BD-rate bit-parity against zenavif's
   own implementation.

---

## 1. Wave completion

### 1.1 Final state

All counts from `zenfleet-ctl report` (live-gap semantics) against the LAN store.

| run | grid | declared | live_done | failed-only | verdict |
|---|---|--:|--:|--:|---|
| `avifdoe-svt-a1-20260901` | main effects, budget | 6,912 | 6,912 | 0 | COMPLETE |
| `avifdoe-svt-a2-20260901` | pairwise, budget | 33,984 | 33,984 | 0 | COMPLETE |
| `avifdoe-svt-a0r-20260901` | same-size control, budget | 6,496 | 6,496 | 0 | COMPLETE |
| `avifdoe-svt-ag-20260901` | transfer gate, **native** | 1,728 | 1,728 | 0 | COMPLETE |
| **total** | | **49,120** | **49,120** | **0** | |

**A0R and AG had no encode workers at all** when this lane opened — and A0R is
the control arm every knob effect is differenced against, so it was the critical
path. Both were launched here; neither produced a single failed cell.

### 1.2 The A1 gap that had already closed

The plan's §13.8 recorded `A1 ... live_done 6,432, gap 480, failed-only`. That
was a **stale read**, not a real gap. Reduced from A1's own 501 ledger files
(26,208 rows):

| era | status | error_class | rows |
|---|---|---|--:|
| pre-fix | `done` | — | **480** |
| pre-fix | `failed` | `encoder_panic` | 6,432 |
| pre-fix | `poison` | `encoder_panic` | 6,432 |
| post-fix | `failed` | `worker_lost` | 6,432 ← the §11.7 `requeue-pardon` rows |
| post-fix | `done` | — | **6,432** |

- **Zero post-fix `encoder_panic`.** Every panic predates the image fix, exactly
  as §11.7's `--before` scoping predicted, and the pardon did not launder a
  live failure.
- **480 + 6,432 = 6,912**, a clean partition: the 480 are the cells that
  succeeded before the stale image was replaced, the 6,432 are the pardoned
  remainder.
- Last ledger write **04:47:21Z**; the §13.8 snapshot that reported a 480 gap
  was taken at 05:44Z. The likely cause is a listing undercount on the object
  store — the same caveat the subsample doc already records for `s5cmd` — and it
  is worth knowing that `report` can under-read a *finished* run.

**So: no cells were requeued, no poison was annotated, and A1 contributes no
port-bug lead.** The only structural encoder finding in this wave is §3 below,
and it is not a failure — it is a silence.

### 1.3 What was launched, and the launcher gap it exposed

`scripts/jobsys/lan_score_launch.sh` — the canonical LAN launcher — forwarded
`ZEN_CORPUS_PREFIX` but **not** `ZEN_CORPUS_BUCKET`, so `jobexec`
(`jobexec.rs:400`) fell back to the *run* bucket when resolving
`cell.image_path`. Every corpus living in its own read-only bucket — the normal
case, `codec-corpus` — was therefore unreachable through it, which is why the
earlier encode arms were launched by hand. Fixed additively (`e9a64c60`); the
two launches below are its live verification.

| host | worker | run | cap | note |
|---|---|---|---|---|
| r7900x | `doe-a0r`, `doe-a0r2`, `doe-a0r3` | A0R | 16g/8g/8g | one worker was CPU-starved by the co-resident uncapped aom container (100% vs 1001%) with ~13 cores idle; **three workers took A0R from 1,803 to 6,132 cells/h (3.4×)** |
| tower | `doe-ag` | AG | cpuset 0-7, shares 256, 12g | leaves 10 of 32 cores free — the household floor is 8 |
| dev | `doe-ag-local` | AG | cpuset 24-30, 16g | added only after the score run reached gap 0 and its workers went idle; peak RSS 1.3 GiB against the cap |

The scale-up is the reusable lesson: **on a box shared with an uncapped
container, one zenfleet worker does not expand to fill the idle cores — add
workers, not cores.** The lease-based claim makes that safe by construction.

---

## 2. The AG corpus identity — proven three ways

A1/A2/A0R encode the **1024-budget crop** corpus; AG encodes the **native** one.
The two corpora share **all 32 filenames**, so pointing AG's scoring at the crop
prefix would silently compare native encodes against crop references. This is
the §12.3 collision, and it is the reason the previous lane refused to declare AG
scoring at all. It is now settled by measurement, not by reading the plan:

| # | layer | result |
|--:|---|---|
| 1 | **corpus** — sha256 of all 32 files in each prefix | **19 byte-identical** passthroughs, **13 genuinely different** crops |
| 2 | **manifest** — declared `inputs[0]` vs the file sha | AG == **native** sha **32/32**; A0R == **crop** sha **32/32** |
| 3 | **runtime** — an AG blob for a *cropped* reference (`6006`) | `ispe` = **2320×3408** (native), not 1024×1024 (crop) |

Layer 3 is the one that matters: layers 1-2 prove what was *declared*, layer 3
proves what the worker *fetched*. Cross-check: AG's declared sha equals the crop
sha on exactly **19/32** — precisely the passthroughs, where the two are the same
bytes.

Scoring was then declared with `--refs-prefix
s3://codec-corpus/avif-subsample-2026-09-01/` and verified live: **AG's pairs
rows resolve 100% under the native prefix, A0R's 100% under the crop prefix.**
The wiring is committed in `scripts/jobsys/avifdoe_score_gapfill.sh` with the
proof in its header, so the next reader does not have to re-derive it.

---

## 3. THE FINDING: two knob arms never reach the bitstream

This is the wave's most actionable result and it does not depend on any modelling
choice — it is byte equality.

### 3.1 What was measured

For every single-deviation arm, the fraction of its `(image, q)` cells whose
encoded bitstream is **byte-identical to the default arm** at the same
`(image, speed, q)`:

| knob | meaning | identical @ s4 | identical @ s6 |
|---|---|--:|--:|
| **`scm3`** | `screen_content_mode = Some(3)` | **288/288 = 100%** | **288/288 = 100%** |
| **`tn0`** | `tune = 0` (VQ) | **288/288 = 100%** | **288/288 = 100%** |
| `mtx32` | `max_tx_size = 32` | 72/288 = 25.0% | 79/288 = 27.4% |
| `qml1.8.15` | QM window (8,15) | 32/288 = 11.1% | 32/288 = 11.1% |
| `acb1` | `ac_bias = 1.0` | 17/288 = 5.9% | 37/288 = 12.9% |
| `acb3` | `ac_bias = 3.0` | 16/288 = 5.6% | 24/288 = 8.3% |
| `shp3` / `shp7` | `sharpness` 3 / 7 | 5.6% / 5.2% | 0% / 0% |
| `bd10`, `qml1.2.10`, `qml1.4.10`, `tl1.0`, `tl1.1`, `tn3`, `vbst×3` | — | 0% | 0% |

`scm3` and `tn0` are not weak effects. They are **no effect at all**, on every
image, at every quality point, at both presets.

### 3.2 The knob reaches the fingerprint but not the encoder

The harness's own resolved-state fingerprint — the hash that is supposed to make
two configurations distinguishable — **does** separate them:

| arm | fp @ q5 | fp @ q45 | fp @ q96 |
|---|---|---|---|
| `s4-svt-420` (default) | `39121e954d30bfe5` | `525f02196fc98147` | `27ef653437e59dc7` |
| `s4-svt-420-tn0` | `ee1082fec17e38e5` | `98b717369c47c847` | `c19e9bac140044c7` |
| `s4-svt-420-scm3` | `edf9fd1ddcd7f88b` | `66969d4a0b2866c1` | `bc0f079c597a1b41` |
| `s4-svt-420-tn3` | `cd9ed696474abe9a` | `a3567e1386ad8a08` | `d3624754e847ea88` |

So the DOE believed it was varying something. **Minimal repro** — image
`7004.scale1024x1024.png`, q 45, speed 4, plan `svt_doe_main`, corpus
`avif-doe-1024-2026-09-01`:

| arm | `encode_sha` | bytes |
|---|---|--:|
| `s4-svt-420` | `e07ecbc4f28e4339…` | 52,306 |
| `s4-svt-420-tn0` | `e07ecbc4f28e4339…` | 52,306 ← **same bitstream** |
| `s4-svt-420-scm3` | `e07ecbc4f28e4339…` | 52,306 ← **same bitstream** |
| `s4-svt-420-tn3` | `24cfb87a676e5b69…` | 53,709 ← tune *does* work at 3 |

**`tune` is partly wired: `tune=3` changes the bitstream, `tune=0` does not.**
That narrows the lead considerably — it is not "tune is ignored", it is "the
tune-0 path resolves to the same encoder state as the default tune-1 path". For
`screen_content_mode`, `Some(3)` is a candidate out-of-range value (the plan's
own hazard H-10 documents clamping of out-of-band `pub` fields).

**Not diagnosed further and deliberately not fixed here** — zenavif is another
repo and another lane's subject. The repro above is the deliverable for the AV1
port program.

### 3.3 The blast radius, and why 27 A2 "interactions" are not interactions

Because the two arms are exact no-ops, **every pair containing one of them is a
byte-identical alias of the other knob's single arm.** Verified, not inferred —
all 27 such A2 pairs are identical on **288/288** cells each, and `tn0-scm3` is
an alias of the bare control:

```
s6-svt-420-tn0-mtx32      ==  s6-svt-420-mtx32       288/288
s6-svt-420-scm3-qml1.2.10 ==  s6-svt-420-qml1.2.10   288/288
s6-svt-420-tn0-scm3       ==  s6-svt-420             288/288
   … 27 of 27 pairs, all exact
```

| block | cells on an inert arm | share of block |
|---|--:|--:|
| A1 | 576 | 8.3% |
| A2 | 8,352 | 24.6% |
| AG | 44 | 2.5% |
| **total** | **8,972** | **21.8% of A1+A2+AG** |

Of A2's 118 declared strata, **29 (24.6%) carry no information**: 2 singles that
equal the control and 27 pairs that equal a single. The interaction analysis
below therefore reports on **74 informative pairs**, and the Stage-B trigger
evaluation excludes inert knobs by construction rather than letting them
silently score a residual of exactly zero.

This is not wasted measurement — "these two knobs do nothing" is a genuine and
useful result. It is wasted *repetition* if Stage B re-buys it.

---

## 4. Registered vs observed

| block | registered (§) | declared | what the data actually is | status |
|---|---|--:|---|---|
| **A0R** | §3.1 — 7 presets × 29 q × 32 img, budget | 6,496 | exactly that; labels `s1..s7-svt-420` | **as registered** |
| **A1** | §3.2 — "17 arms × all 7 effective presets × 9 q × 32" = **34,272** | **6,912** | 24 strata × 9 q × 32 = 1 default + 16 knob arms + 6 non-default speeds + 1 bit-depth, **all knob arms at speed 4** | **doc arithmetic wrong; data correct** |
| **A2** | §3.4 — 126 pairs + singles × speed 6 × 9 q × 32 | 33,984 | 118 strata × 9 q × 32 = 1 control + 16 singles + **101** pairs (`bd10` absent) | as registered, post-merge |
| **AG** | §3.8 — 17 arms × speeds **{4,6}** × 3 q × 32 = 3,264 | **1,728** | 18 strata × 3 q × 32 = 16 knob arms **at s4 only** (no `bd10`) + s4 control + a bare s6 | **speed axis collapsed** |
| **A3** (aom main effects) | §3.5 — 23 arms | — | **never declared** | ABSENT |
| **A4** (timing block) | §3.6 — 6-point size ladder | — | **never declared** | ABSENT |

### 4.1 The A1 discrepancy is the plan's error, not the wave's

§3.2 asks for 17 knob arms crossed with all 7 presets. That is **not expressible
under `--max-deviations 1`**: speed 4 is index 0, so a non-default preset *is*
the one permitted deviation and cannot be combined with a knob deviation.
zenavif's own test states the real design in a comment and asserts it:

```rust
// svt_doe_main_at_one_deviation_is_isolated_main_effects  (zenavif src/sweep.rs)
// speed 4 and bit-depth Auto are index 0, so: 1 default + 16
// knob arms + 6 speeds + 1 bit-depth = 24 strata at one q.
assert_eq!(plan.cells.len(), 24, …);
```

24 × 9 × 32 = **6,912** — the declared count, to the cell. So the declaration is
right and §3.2's "34,272" (and the 12.1 CPU-h / "speed 1 alone is 8.75" cost
model built on it) describes a grid that could never have been declared.

**Consequences, all load-bearing:**

- §7.1(3)'s main effects "per speed" exist at **two** presets — s4 from A1 and
  s6 from A2's singles — not seven.
- **B-5** (main effect inverts sign across the preset ladder) is evaluable
  across exactly **one preset pair**, s4 vs s6, for the 16 knobs present in
  both. `bd10` appears at s4 only, so B-5 is **NOT COMPUTABLE** for it.
- §3.3's stated reason for folding A1b away — "presets 0 and 1 at 9 q instead of
  3" — did not happen: no knob arm exists at preset 0 or 1. A1b's scientific
  purpose (does a knob invert at the slow end?) is **still open**, and is now a
  Stage-B question rather than something A1 answered.
- §7.3's de-scope ladder ("A1 → drop speed 1", "→ drop speed 3") would have shed
  *bare preset controls*, not knob×preset cells.

### 4.2 AG's speed axis collapsed the same way

`svt_doe_transfer` sets `speeds = vec![4, 6]`, but under `--max-deviations 1` the
s6 cells spend their deviation on the preset, so the only s6 stratum that
survives is the **bare control**. The gate therefore runs at **speed 4 only**,
and `bd10` — which AG also drops — has **no transfer evidence at all**.

### 4.3 Both absent blocks matter, and one cannot be recovered from this data

- **A3 (aom-rs main effects) was never declared.** Every knob result in this
  document is **svt-rs only**. There is no aom axis to report and none can be
  derived; the "top knobs per backend" ask has exactly one backend.
- **A4 (the timing block) was never declared, and speed is not recoverable
  here.** `encode_ms` is not persisted by the fleet path — confirmed against the
  ledger schema (`job_id, image_path, codec, q, knob_tuple_json, output_sha,
  status, error_class, attempts, ts, worker, provider, kind_json`) and the score
  blobs (`kind, image_path, codec, encode_sha, metric, score, scores, runtime` /
  `regime, features`). Neither carries a duration, and `ts` is a ledger-write
  time on concurrently-chunked work. **No speed model is reported and none is
  estimated** — §3.9 already registered that the size pair buys the bytes model
  everything and the speed model nothing.

---

## 5. Method, and the gates that make it trustworthy

### 5.1 Instruments

| step | owner | note |
|---|---|---|
| score-blob → tidy table | `scripts/jobsys/avifdoe_harvest.py` | joins on `encode_sha`; computes no statistics |
| RD / BD-rate / main effects / interactions | `scripts/jobsys/avifdoe_stagea_analyze.py` | §7.1 |
| transfer gate / bytes model / triggers | `scripts/jobsys/avifdoe_stagea_gates.py` | §3.8, §3.9, §7.2 |
| rank statistics | **`zenstats`** via `zensim/scripts/lib/zen_stats.py` | nothing correlation-shaped is hand-rolled |
| BD-rate | mirrors `zenavif/scripts/rd_gap/bd_arm.py` | **parity-gated, see 5.2** |

**Rows are heterogeneous and the code keys on that.** The score run changed
metric sets mid-flight (§13.5), so every row carries `metrics_present`:

| `metrics_present` | rows |
|---|--:|
| `ssim2 \| zensim_features` (2-metric, post-boundary) | majority |
| `butteraugli \| butteraugli_max \| butteraugli_pnorm3 \| ssim2 \| zensim_features` | the pre-boundary remainder |

**`ssim2` is the only corpus-wide scalar quality response**, exactly as §7.1
designates it. `zensim` is emitted as a 720-wide **feature vector**
(`kind:"feature"`, `regime:"v2-ab"`) and carries **no scalar**, so it is not a
second response without running a model over it. `butteraugli` covers only the
pre-boundary pairs and is **not** a corpus-wide column — it is not used for any
ranking here.

**Cells, not bitstreams, are the row set.** Encoding is content-addressed, so
the 42k cells collapse to ~28k distinct blobs; keying rows on `encode_sha` would
drop one of every byte-identical pair and destroy precisely the inertness signal
of §3. The harvester iterates cells and attaches scores to them.

### 5.2 Four gates, all passed, none assumed

| gate | what it rules out | result |
|---|---|---|
| **BD-rate parity** | a private BD-rate that quietly disagrees with the house one | identical to `zenavif/scripts/rd_gap/bd_arm.py` on 200 random ladders, **max \|Δ\| = 0.0 exactly** |
| **A0-native ≡ A0R on passthroughs** | the two size legs differing in *configuration* rather than in pixels — which would confound every BD-rate and the whole bytes model | **386 / 386 shared `(image, speed, q)` cells byte-identical, 0 differ** |
| **A0R ≡ in-run controls** | cross-run differencing being invalid | **306 / 306 byte-identical, 0 differ** |
| **AG passthrough null** | the transfer gate measuring a pipeline artefact instead of a size effect | on the 19 passthroughs the native and budget legs are the same encode; **0 residual violations** |

The second and third are what license the whole differencing scheme: A0R can
serve as the control for A1 and A2 arms because it is provably the *same
encoder configuration*, and the bytes model can attribute native-vs-budget
differences to pixel count because nothing else differs.

### 5.3 Exclusions, stated rather than silent

**Two references are excluded from BD-rate with cause: `6006` and `6018`.** Both
are crops of scanned patent pages that landed on near-blank regions. Their rate
does not respond to quality at all and their `ssim2` saturates:

| reference | bytes q5 → q96 | ratio | ssim2 q5 → q96 |
|---|---|--:|---|
| `6006.scale2320x3408` | 6,486 → 7,045 | **1.1×** | 71.9 → **100.0** |
| `6018.scale2320x3408` | 29,215 → 29,387 | **1.0×** | 74.4 → **100.0** |

A Pareto frontier over such a ladder collapses below the 4-point minimum, so no
BD-rate is defined. This is a property of the content, not a pipeline fault, and
it is the same crop-degeneracy family §12.2 already flagged.

**No other reference is dropped.** An earlier pass appeared to lose `9654` and
`1634` as well; that was a stale score-blob snapshot, and both have complete,
well-behaved ladders once the sync caught up. Recording it because it is the
failure mode most likely to recur: **a partial sync looks exactly like
degenerate content.**

### 5.4 Content classes are small, and the class verdicts need reading with care

The 32 references carry 12 fine content classes, sized 6, 5, 4, 3, 3, 3, 2, 2,
1, 1, 1, 1. A median over n = 1 is one observation. Results are therefore
reported over **five coarse families** — `photo` (7), `plot` (6),
`screenshot` (5), `ai-gen` (9), `scan` (5) — with the fine class kept in the
per-image table. The plan's own "11/32 screenshot+plot" split (§2.1) is exactly
`plot` + `screenshot` here.

**The G-CROP `feature_recheck` verdict must be read beside `parent_z_dist`, and
three crops are flagged wherever a class-keyed result uses them** (features
sha `133b93d8…`):

| reference | verdict | `parent_z_dist` | why it is flagged |
|---|---|--:|---|
| `1442.scale4000x3000` | **SHIFTED** 0→25 | **24.41** | genuinely changed cluster |
| `1634.scale3000x4000` | **SHIFTED** 25→10 | 4.58 | genuinely changed cluster |
| `6604.scale3286x4868` | "preserved" | **67.72** | *trivially* preserved — it stayed only because nothing was nearer; the largest z in the corpus |
| `6602.scale3302x4844` | "preserved" | 16.15 | same shape, less extreme |

Four of the twelve fine classes have a single member, so "preserved" is
trivially true for them too. **Class medians in §8 are directional evidence, not
estimates**, and any B-3 trigger resting on a class with n < 3 is marked
PROVISIONAL in the trigger table.

