# AVIF era-delta wave — analysis — 2026-09-03

Analysis of the three arm-sets registered in
[`avif_newera_sweep_2026-09-03.md`](avif_newera_sweep_2026-09-03.md) against the
audit in [`avif_newera_delta_2026-09-03.md`](avif_newera_delta_2026-09-03.md).
Companion doc for the HDR arm:
[`avif_hdr_rd_baseline_2026-09-03.md`](avif_hdr_rd_baseline_2026-09-03.md) —
kept **separate** because the two waves answer unrelated registered questions
against different corpora, instruments and pins; nothing joins across them.

**Scope discipline.** Every number below is either (a) an exact byte fact from a
content-addressed identity join, or (b) a BD-rate / paired read computed inside
ONE era against that era's own in-run control. **No row from this wave is ever
pooled with a Stage-A row.** Where the two eras are compared it is
*effect-vs-effect* — each era's own `main_effects.tsv`, produced by the same
tool with the same `--control inrun` instrument — and it is labelled as such.

---

## 0. TL;DR

1. **STABILITY — held, and held EXACTLY.** Arm-set A re-ran `svt_doe_main`
   unchanged at the new pin and reproduced Stage-A's bitstreams **byte-for-byte
   on every shared cell**. Arm-set B — a *different* plan, written this era —
   independently reproduced the same bytes on 3,456 speed-4/control cells
   (vs Stage-A `a1`) and 2,880 speed-6 knob cells (vs Stage-A `a2`). Because the
   bytes are identical **and the two waves' scorers agree exactly** (§2.5),
   every covered Stage-A effect is not merely "within CI" at the new pin — it is
   the *same number*. No statistical test was needed, and none is reported as if
   it were the evidence. Scope is enumerated in §2.2b: 7 of the 16 speed-6 knobs
   were **not** re-measured this wave and keep their Stage-A numbers unverified.
2. **`scm3` at speed 7 (preset 9) is REAL, and it is screen-content-exclusive.**
   First measurement through the sweep interface, statistics-free: at speed 7,
   `scm3` changes the bitstream on **90 of 288 cells (31.25 %)**; the divergence
   is **0/144 on photo + AI-generated content** and **90/144 on
   plot + screenshot + scan**. At speeds 4 and 6 it is **0/288 and 0/288** —
   reproducing the dossier's "dead at preset ≤ 7" on a completely different
   instrument than the port-level probe that found the preset-9 edge.
3. **`c1` supersedes `t1d` everywhere, and the answer INVERTS.** 72 of 96 cells
   reproduce byte-identically; **24 differ**, on exactly the 8 images the
   multi-tile predicate selects — zero false positives, zero false negatives.
   T1-d's registered n for the cross-size question is **13 images, not 32**, and
   **8 of those 13 (61.5 %) are the corrupt ones**. On the clean read, at the
   registered n = 13: q45 is **−0.69 % bytes for +0.244 ssim2, and bd10
   DOMINATES its 8-bit twin on 10 of 13 images**; q90 is +18.0 % bytes for
   +2.00 ssim2. The superseded block reported median Δssim2 **−104.80** at q90
   with 8 of 13 DOMINATED. **A BD-rate is NOT MEASURED for this block in either
   era** — a 3-point ladder cannot satisfy the BD-rate owner's ≥ 4-point guard,
   and the guard was not loosened to produce one.
4. **What did NOT move ≥ 1 pp:** nothing on the replicated axis — the ≥ 1 pp
   movement question is answered `0.0000 pp` there, by construction, not by a
   test. The things that DID move are the two axes that had no prior value:
   speed 7 (new) and the bd10-native block (previously wrong).

---

## 1. Coverage — what was analysed, and what was live

Every run below was verified with `zenfleet-ctl report` before any number was
computed; a run with a live gap is not analysed.

<!--TABLE:COVERAGE-->

**A scoring gap was found and closed by this lane, not worked around.** The
wave was declared with no score-side gapfill loop and no scoring worker, so at
pickup **0 of 15,648 cells had been scored** and none would have been. The
canonical loop (`scripts/jobsys/avifdoe_score_gapfill.sh`) was started against
the three runs with the run→refs-prefix map set explicitly — **a1/b1 →
`avif-doe-1024-2026-09-01` (crop), c1 → `avif-subsample-2026-09-01` (native)** —
which is the documented same-filenames-different-pixels hazard, and a worker was
launched. Nothing about the declaration was changed.

---

## 2. The stability question, answered at the byte level

### 2.1 Arm-set A reproduces Stage-A exactly

`avifdoe_era_compare.py` joins the two runs on the cell identity
`(image_path, q, knob_tuple.cell)` and compares `encode_sha` — the content
address of the encoded bitstream. This is an exact test: two runs agree or they
do not.

| comparison | shared cells | byte-identical | differing |
|---|--:|--:|--:|
| `avifdoe-svt-a1-20260901` → `avifdoe-svt-eradelta-a1-20260903` | 6,912 | **6,912** | **0** |

**100.00 % identical, on all 24 strata.** The registration predicted this
("*the expectation is that most of these 6,912 cells reproduce old bytes
exactly — that expectation is itself the thing being tested*"). It is not
"most"; it is all.

### 2.2 Arm-set B reproduces it too, from a different plan

Arm-set B is a **new** `SweepAxes` (`svt_doe_era_delta_r1`), written this era,
with its own knob list and speed list. Its speed-4 legs and its bare speed-6 /
speed-7 controls overlap Stage-A's A1 cell space, and the registration called
that overlap a free internal-consistency check. It passes:

| stratum | shared cells | byte-identical |
|---|--:|--:|
| `s4-svt-420` (control) | 288 | 288 |
| `s4-svt-420-{qml1.2.10, qml1.4.10, scm3, shp3, shp7, tn3, vbst1.2.5, vbst1.3.5, vbst1.3.7}` | 288 each | 288 each |
| `s6-svt-420` (control) | 288 | 288 |
| `s7-svt-420` (control) | 288 | 288 |
| **total vs `a1`** | **3,456** | **3,456** |

and its **speed-6 knob legs** against Stage-A's `a2` (the run that supplied
every speed-6 knob effect in the Stage-A tables):

| stratum | shared cells | byte-identical |
|---|--:|--:|
| `s6-svt-420` (control) | 288 | 288 |
| `s6-svt-420-{qml1.2.10, qml1.4.10, scm3, shp3, shp7, tn3, vbst1.2.5, vbst1.3.5, vbst1.3.7}` | 288 each | 288 each |
| **total vs `a2`** | **2,880** | **2,880** |

So the reproduction is not an artifact of re-running the same plan object: two
independently-constructed declarations, one of them new code, land on the same
bytes at both presets.

### 2.2b Exactly which cells this verification does and does not cover

Stating this plainly, because "the era changed nothing" is a claim whose scope
is the thing that matters:

| axis | covered by | status |
|---|---|---|
| **speed 4, all 17 knob arms** (incl. `bd10`) + the 6 non-default-speed controls | arm-set A ↔ Stage-A `a1`, 6,912 cells | **verified byte-identical** |
| **speed 6, 9 knobs** (`tn3`, `shp3`, `shp7`, `vbst1.2.5/1.3.5/1.3.7`, `qml1.2.10`, `qml1.4.10`, `scm3`) | arm-set B ↔ Stage-A `a2`, 2,880 cells | **verified byte-identical** |
| **speed 6, the other 7 knobs** (`acb1`, `acb3`, `mtx32`, `qml1.8.15`, `tl1.0`, `tl1.1`, `tn0`) | — | **NOT re-measured this wave.** Arm-set B's registered knob list is the delta audit's risk list, which does not include them. They keep their Stage-A numbers, unverified at the new pin. |
| **speed 7, every knob** | arm-set B | **new axis** — Stage-A never crossed a knob with speed 7, so these are first measurements, never a stability check |
| **native corpus, `bd10`** | arm-set C | §4 |
| anything outside these strata (12-bit, superres, tiles, fork mode) | — | out of scope, and the byte join says nothing about it |

### 2.3 Why this is a stronger result than the audit could promise

The audit's §1 records that **Stage-A's own encodes were built against a
floating `path` dependency on `zenav1-svt`, not a git-rev pin** — before
`85af725` there was no fixed sha, so "what Stage-A saw" is only bounded, not
known. The audit therefore had to reason forward from a *commit range*
(§2.1: only 2 of 33 commits are AVIF-reachable).

The byte join sidesteps that entirely. Whatever binary produced the Stage-A
blobs, and whatever binary the `2ca060f4`-pinned image contains, **they emit
identical bytes on every one of 13,248 verified cell-pairs** — 6,912 (A↔`a1`)
+ 3,456 (B↔`a1`) + 2,880 (B↔`a2`) — covering **9,792 distinct cell identities**
across 34 strata and 7 presets. That is a measurement of the delta, not an
inference about it.

### 2.4 Stability verdict table

Because the contributing bitstreams are byte-identical, each effect's BD-rate
is the same number in both eras — the response is a deterministic function of
(bytes, decoded pixels), and both are unchanged. The verdict column below is
produced mechanically by `avifdoe_era_compare.py --effects-old/--effects-new`
at a registered **1.0 pp** movement threshold, differencing the two eras' own
`main_effects.tsv` (both `--control inrun`, so the instrument is held fixed and
only the era varies).

**Arm-set A vs Stage-A** (speed-4 knobs + `bd10`; Stage-A's speed-6 rows have
no arm-set A counterpart and read NOT-MEASURED-new, which is correct — arm-set A
replicates `svt_doe_main`, and `svt_doe_main` carries knobs at speed 4 only):

<!--TABLE:STABILITY-->

**Arm-set B vs Stage-A** (its speed-4 legs joined against `a1`, its speed-6 legs
against `a2`; its speed-7 legs have no Stage-A counterpart and read
NOT-MEASURED-old, which is the correct label for a new axis):

<!--TABLE:STABILITYB-->

### 2.5 Scorer drift is separately accounted for

Identical bytes make one more thing measurable that is normally confounded:
because the bitstream is provably the same object, **any difference in the
score is the scorer's, not the encoder's**. The two waves were scored by
different worker images, so this is not hypothetical.

<!--TABLE:SCOREDRIFT-->

---

## 3. `scm3` at speed 7 (preset 9) — the first sweep-side measurement

The delta audit (§4.1) found, with a port-level `knob_byte_identity` probe on
its own fixtures, that `screen_content_mode = Some(3)` stops being inert at
raw preset 8/9, and that **speed 7 is the only product-reachable point** (the
`speed → preset` map sends 7/8/9/10 all to preset 9). Arm-set B measures that
through the sweep, on the DOE corpus, for the first time.

**The byte-level result needs no statistics** — `scm3` either changed the
bitstream or it did not:

| speed (preset) | class | cells | byte-identical to control | differing |
|---|---|--:|--:|--:|
| 4 (p4) | all | 288 | 288 | **0** |
| 6 (p7) | all | 288 | 288 | **0** |
| **7 (p9)** | photo | 63 | 63 | **0** |
| **7 (p9)** | ai-gen | 81 | 81 | **0** |
| **7 (p9)** | scan | 45 | 27 | **18** (2 of 5 images) |
| **7 (p9)** | screenshot | 45 | 18 | **27** (3 of 5 images) |
| **7 (p9)** | plot | 54 | 9 | **45** (5 of 6 images) |
| **7 (p9)** | **total** | **288** | **198** | **90 (31.25 %)** |

Three things worth stating precisely:

- **The speed-4 and speed-6 zeros are a reproduction, not a null.** Stage-A's
  own `main_effects.tsv` reports `scm3` at exactly `0.0000` with a
  `[0.0000, 0.0000]` CI at both speeds — because the arm *was* the control.
  This wave reproduces that on 576 cells at the new pin, and the reproduction is
  at the *bitstream* level: all 576 are byte-identical to their control, so the
  zero is a fact about the encoder, not a metric that happened to land on zero.
- **At speed 7, the median is still `0.0000` — and reporting only the median
  would hide the whole effect.** 22 of 32 images are exactly zero (inert), so
  the median sits on them; the mean is **−16.73 %** and the minimum is
  **−88.86 %**. The right statistic here is the effect **conditional on the knob
  firing**, which §3.1 gives, plus the count of images it fires on. A knob whose
  effect is confined to a content class will always look dead in a
  corpus-median.
- **The divergence is content-exclusive, not content-graded.** Photo and
  AI-generated content are at 0/144. Every differing cell is
  plot / screenshot / scan. That is the behaviour a screen-content detector
  should have, and it is the first time we have seen it through the product's
  own knob surface.
- **AI-generated content behaves like photo here**, not like screen content —
  9 images, 81 cells, zero divergence. Worth knowing before anyone assumes
  "synthetic ⇒ screen".

### 3.1 Effect size

| content class | images where `scm3` fires | median BD-rate % | min | max |
|---|--:|--:|--:|--:|
| plot | 5 | -66.93 | -85.44 | -18.57 |
| scan | 2 | -87.53 | -88.86 | -86.20 |
| screenshot | 3 | -25.99 | -33.23 | -24.46 |
| **all firing** | **10** | **-50.08** | **-88.86** | **-18.57** |

| image | class | `scm3` BD-rate % | `tn3` BD-rate % |
|---|--:|--:|--:|
| `6018.scale2320x3408.png` | scan | -88.86 | — |
| `6006.scale2320x3408.png` | scan | -86.20 | -85.57 |
| `7050.scale1024x1024.png` | plot | -85.44 | -85.83 |
| `7052.scale1024x1024.png` | plot | -82.54 | -84.14 |
| `7004.scale1024x1024.png` | plot | -66.93 | -68.75 |
| `8414.scale1280x800.png` | screenshot | -33.23 | -38.44 |
| `8434.scale414x896.png` | screenshot | -25.99 | -35.09 |
| `8288.scale375x667.png` | screenshot | -24.46 | -30.53 |
| `7042.scale1024x1024.png` | plot | -23.19 | -13.73 |
| `7058.scale1024x1024.png` | plot | -18.57 | -26.80 |

On the 22 images where `scm3` is inert, `tn3` at speed 7 still gives a median **-10.88 %** (n = 22) from its other eight aliased fields.

### 3.1b Where the BD-rate comes from — the paired matched-q bytes

A −50 % median BD-rate is large enough that it deserves the underlying bytes on
the page rather than only the integral. `scm3` vs its own speed-7 control, on
three of the firing images:

| image (class) | q | bytes with `scm3` | bytes control | Δ bytes % | ssim2 `scm3` | ssim2 control | Δ ssim2 | verdict |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `6018` (scan) | 5 | 28,925 | 41,984 | -31.10 | 73.419 | 12.857 | +60.562 | DOMINATES |
| `6018` (scan) | 15 | 29,054 | 87,219 | -66.69 | 92.776 | 38.098 | +54.678 | DOMINATES |
| `6018` (scan) | 25 | 29,029 | 128,913 | -77.48 | 96.300 | 56.036 | +40.263 | DOMINATES |
| `6018` (scan) | 35 | 29,124 | 174,528 | -83.31 | 96.369 | 69.908 | +26.460 | DOMINATES |
| `6018` (scan) | 45 | 29,183 | 217,741 | -86.60 | 96.706 | 79.964 | +16.742 | DOMINATES |
| `6018` (scan) | 60 | 29,310 | 285,571 | -89.74 | 97.680 | 88.347 | +9.333 | DOMINATES |
| `6018` (scan) | 76 | 29,388 | 340,061 | -91.36 | 98.264 | 92.503 | +5.760 | DOMINATES |
| `6018` (scan) | 90 | 29,536 | 434,757 | -93.21 | 100.000 | 94.636 | +5.364 | DOMINATES |
| `6018` (scan) | 96 | 29,625 | 511,253 | -94.21 | 100.000 | 95.305 | +4.695 | DOMINATES |
| `7050` (plot) | 5 | 35,562 | 28,427 | +25.10 | 50.043 | -7.684 | +57.727 | TRADE |
| `7050` (plot) | 15 | 35,770 | 51,002 | -29.87 | 58.737 | 21.760 | +36.977 | DOMINATES |
| `7050` (plot) | 25 | 36,079 | 101,720 | -64.53 | 65.186 | 37.898 | +27.288 | DOMINATES |
| `7050` (plot) | 35 | 36,625 | 176,390 | -79.24 | 73.968 | 55.488 | +18.480 | DOMINATES |
| `7050` (plot) | 45 | 37,388 | 234,286 | -84.04 | 78.790 | 66.309 | +12.481 | DOMINATES |
| `7050` (plot) | 60 | 38,716 | 335,430 | -88.46 | 82.425 | 77.423 | +5.002 | DOMINATES |
| `7050` (plot) | 76 | 40,205 | 423,574 | -90.51 | 84.036 | 82.035 | +2.001 | DOMINATES |
| `7050` (plot) | 90 | 44,359 | 542,873 | -91.83 | 83.522 | 83.691 | -0.170 | TRADE |
| `7050` (plot) | 96 | 53,311 | 650,943 | -91.81 | 84.568 | 84.261 | +0.307 | DOMINATES |
| `8414` (screenshot) | 5 | 26,606 | 19,617 | +35.63 | -7.580 | -19.951 | +12.371 | TRADE |
| `8414` (screenshot) | 15 | 38,729 | 40,287 | -3.87 | 24.208 | 10.380 | +13.827 | DOMINATES |
| `8414` (screenshot) | 25 | 43,734 | 63,592 | -31.23 | 36.451 | 30.110 | +6.341 | DOMINATES |
| `8414` (screenshot) | 35 | 58,857 | 96,178 | -38.80 | 50.459 | 48.145 | +2.314 | DOMINATES |
| `8414` (screenshot) | 45 | 81,307 | 129,816 | -37.37 | 62.260 | 59.249 | +3.012 | DOMINATES |
| `8414` (screenshot) | 60 | 106,760 | 189,911 | -43.78 | 71.814 | 70.505 | +1.309 | DOMINATES |
| `8414` (screenshot) | 76 | 149,937 | 249,802 | -39.98 | 77.913 | 75.085 | +2.828 | DOMINATES |
| `8414` (screenshot) | 90 | 224,278 | 358,302 | -37.41 | 79.031 | 75.740 | +3.291 | DOMINATES |
| `8414` (screenshot) | 96 | 279,514 | 430,396 | -35.06 | 79.470 | 76.101 | +3.369 | DOMINATES |

Two things this makes visible that the BD-rate alone does not. **The win is not
uniform in q** — at the very lowest q, palette/IntraBC coding costs *more* bytes
(`7050` +25.1 %, `8414` +35.6 % at q5) while buying a great deal of quality
(+57.7 and +12.4 ssim2), which is a trade the BD-rate integration prices
correctly and a per-q byte comparison would misread. And **the top of the
ladder is where it pays**: on `6018` (a scanned document) q90 goes 434,757 B →
29,536 B (**14.7×**) while ssim2 goes 94.636 → **100.000** — the screen-content
tools are coding the page essentially exactly at a fifteenth of the rate, and
`scm3` DOMINATES its control on all nine of that image's ladder points.

### 3.2 The single-cell prior, and what it is not

The wave's launch report carried a single-cell observation of a screen tile
going **2,898 B → 930 B** with `scm3` at preset 9 — a ~68 % byte drop.
**That number is not reproduced here and is not evidence for anything in this
document.** This lane did not observe it and did not re-run it; the
corpus-level effect above is the measurement, and it is what any claim should
rest on.

It is worth recording only that the corpus measurement is **consistent in
magnitude** with that prior rather than in tension with it: on the firing
images the paired per-q read shows byte reductions from −24 % to −94 % (e.g.
`6018.scale2320x3408.png` goes −66.7 % at q15 and −93.2 % at q90), so a ~68 %
drop on one hand-picked screen tile sits squarely inside the observed range.
Consistency is not confirmation — the prior remains one cell with no
provenance in this lane's data.

---

## 3b. Speed 7 is a NEW axis — the nine knobs' first measurement there, and B-5

Stage-A never crossed a knob with speed 7 (preset 9); arm-set B does. These are
**first measurements, not stability checks**, and nothing in this section is
compared against a Stage-A number.

| knob | n | median BD-rate % | 95% CI | IQR | mean | images improved | min | max |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `qml1.2.10` | 32 | -4.8889 | [-6.0343, -2.9078] | 4.5370 | -4.3309 | 29/32 | -14.1627 | 12.8881 |
| `qml1.4.10` | 32 | -4.0875 | [-5.1068, -2.5737] | 3.5100 | -3.7512 | 29/32 | -13.9825 | 9.7750 |
| `scm3` | 32 | 0.0000 | [0.0000, 0.0000] | 23.5062 | -16.7317 | 10/32 | -88.8574 | 0.0000 |
| `shp3` | 32 | 2.7838 | [2.1217, 3.1061] | 1.6826 | 2.5975 | 3/32 | -4.3885 | 7.2652 |
| `shp7` | 32 | 9.1038 | [7.0338, 9.5592] | 4.5769 | 8.1177 | 2/32 | -8.4281 | 16.6559 |
| `tn3` | 31 | -13.7313 | [-18.5620, -10.1941] | 18.9363 | -21.4314 | 28/31 | -85.8321 | 12.3397 |
| `vbst1.2.5` | 32 | 0.0026 | [-2.2622, 1.9353] | 8.5979 | 1.8549 | 12/32 | -12.3601 | 44.4831 |
| `vbst1.3.5` | 32 | 0.0758 | [-1.4988, 3.8141] | 10.8566 | 3.4660 | 11/32 | -12.6760 | 51.7889 |
| `vbst1.3.7` | 32 | -1.4601 | [-2.2943, -0.0430] | 3.1140 | -1.4887 | 22/32 | -9.9487 | 12.0725 |

**Why n is 30–32 and not always 32.** Two scanned documents — `6006` and
`6018` — have an RD ladder whose Pareto frontier collapses to **2 points** at
speeds 4 and 6 (quality saturates at ssim2 100.0 across the top of the ladder
and is non-monotone in bytes below it), so the BD-rate owner's ≥ 4-point guard
declines to produce a number for them. That is the guard working, not a data
gap, and it is **not new**: Stage-A's own tables carry the same n for the same
images — the `shp7` speed-4 image set is *identical* between Stage-A's `a1` and
this wave's `b1` (30 images, same 30). At speed 7 the saturation disappears
(the fast preset never reaches ssim2 100) and only `tn3` loses one image.

**Stage-B trigger B-5 (registered evaluation, now on three presets instead of
two).** B-5 fires when a knob's |median BD-rate| ≥ 1 % at two presets *with
opposite signs*. Adding speed 7 as a third point:

| knob | s4 (p4) | s6 (p7) | s7 (p9) | presets with \|median\| ≥ 1 % | B-5 fires? |
|---|--:|--:|--:|--:|--:|
| `qml1.2.10` | -0.29 | -2.59 | -4.89 | 2 | no |
| `qml1.4.10` | -0.42 | -2.03 | -4.09 | 2 | no |
| `scm3` | +0.00 | +0.00 | +0.00 | 0 | no |
| `shp3` | +0.91 | +1.65 | +2.78 | 2 | no |
| `shp7` | +5.47 | +7.31 | +9.10 | 3 | no |
| `tn3` | -7.03 | -4.48 | -13.73 | 3 | no |
| `vbst1.2.5` | +0.38 | +0.01 | +0.00 | 0 | no |
| `vbst1.3.5` | +0.78 | +0.32 | +0.08 | 0 | no |
| `vbst1.3.7` | -0.97 | -1.39 | -1.46 | 2 | no |

**B-5 still does not fire** — and the reason is now more interesting than
"no flip found". Every knob whose effect is large enough to test is **monotone
in preset**: `qml1.2.10` and `qml1.4.10` strengthen from ≈0 at speed 4 to
several percent at speed 7, `shp3`/`shp7`'s cost rises toward the fast presets,
and the `vbst*` arms decay toward zero. Stage-A read `qml1.2.10` as
"sign-flipping between presets" from two points; with three, it is a knob whose
benefit grows with preset and merely passes through zero near speed 4.

`shp7`'s rise (`+5.47 → +7.31 → +9.10 %` at s4/s6/s7) independently reproduces
the Stage-B6 native-corpus finding of `+7.15 / +7.99 / +9.46 %` at the same
three speeds, on a *different corpus* — the budget crops here vs B-6's native
images. The two agree on direction, on ordering, and to within ~1.7 pp on
magnitude.

**A reproducibility note on the CIs, not the medians.** Arm-set B's speed-4 and
speed-6 medians reproduce Stage-A's `stagea_inrun` values *exactly* (`shp7`
5.4676 / 7.3077, `tn3` −7.0342 / −4.4807, `qml1.2.10` −0.2895 / −2.5853,
`scm3` 0.0000 / 0.0000 — every digit), which is what byte identity plus an
identical scorer predicts. The bootstrap **CI** bounds can differ in the third
decimal (`shp3` s4: `[0.0111, 2.0097]` in Stage-A vs `[0.0130, 2.0097]` here)
because `median_ci` resamples the per-image list in *insertion order* and the
two runs enter their images in a different order. The resampling distribution
is the same; the particular draw is not. Registered as an observation, not
changed — making it order-invariant would move every published Stage-A CI, and
that is a deliberate decision for the owner, not a drive-by.

---

## 4. Arm-set C — the clean bd10-native read, and the supersession of `t1d`

### 4.1 The corruption is confirmed, and its extent is exactly predicted

| comparison | shared cells | byte-identical | differing |
|---|--:|--:|--:|
| `avifdoe-svt-t1d-20260902` → `avifdoe-svt-eradelta-c1-20260903` | 96 | **72** | **24** |

The 24 differing cells are **all 3 q-points of exactly 8 images**:
`1008`, `1220`, `1420`, `1432`, `1442`, `1634` (the 12 MP 3000×4000-class
portraits/landscapes) and `6602`, `6604` (≈16 MP).

**Predicate check.** Applying zenav1-svt #18's own tile-forcing predicate —
`width > 4096` **OR** sb-aligned area `> 4096 × 2304 = 9,437,184 px` — to the
32 native corpus images selects **8 images, and they are exactly these 8**.
Zero false positives, zero false negatives. The blast radius is
mechanism-confirmed, not correlational.

The other 72 cells reproducing byte-identically is simultaneously a live check
on the audit's §2.1 claim that the era delta reaches nothing else on the AVIF
path — and it passes.

### 4.2 ⛔ The corruption covers 8 of the 13 images the question actually has

`avif_hdr_arm_plan_2026-09-02.md` §10.4a registers a restriction that is easy
to lose: **19 of the 32 DOE images are byte-identical passthroughs** between
the native and 1024² builds, so a *cross-size* question has **n = 13**
(`transform == crop-native`), not 32. All 8 corrupt images are ≥ 12 MP and
therefore all 8 are in that 13.

> **So the pre-fix `t1d` answer to "does bd10 survive native size" rested on 5
> clean images and 8 with structurally wrong pixels — 61.5 % of its own
> effective n was invalid.** This is why `c1` supersedes it *everywhere*
> rather than partially.

### 4.3 The answer — and the honest shape of it

**A BD-rate is NOT MEASURED for this block, in either era.** The BD-rate owner
(`avifdoe_stagea_analyze.py::bd_rate`, parity-gated against zenavif
`scripts/rd_gap/bd_arm.py`) requires ≥ 4 ladder points on both sides; the
`svt_doe_t1_bd10_transfer` plan is a **3-point probe** (`q ∈ {15,45,90}`). It
was never possible to compute one here, before the fix or after, and loosening
the guard to manufacture one would be a gate relaxation. The registered
question "does −1.02 % survive native resolution" therefore cannot be answered
in the same currency as the −1.02 % it is asked about — that is a property of
the block's design, not of this analysis.

The artifact says so too: `c1_paired/main_effects.tsv` is **a header and no
rows**. The analyzer ran, found no computable BD-rate, and wrote an empty table
rather than a zero — which is the behaviour NOT-MEASURED should have.

What the block *can* answer is the paired matched-q read: at each `q`, bd10's
bytes and quality against its 8-bit twin.

**THE REGISTERED READ — n = 13 `crop-native` images (the cross-size question)**

| q | n images | median Δbytes % | 95% CI | median Δssim2 | 95% CI | DOMINATES | DOMINATED | TRADE |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 15 | 13 | -8.27 | [-8.41, -6.76] | -3.780 | [-4.01, -2.05] | 0 | 2 | 11 |
| 45 | 13 | -0.69 | [-1.20, -0.31] | +0.244 | [+0.06, +0.62] | 10 | 0 | 3 |
| 90 | 13 | +18.01 | [+14.86, +21.68] | +1.996 | [+0.80, +2.94] | 0 | 0 | 13 |

**The 19 `native` passthroughs — no size transfer to measure here, reported as the internal control**

| q | n images | median Δbytes % | 95% CI | median Δssim2 | 95% CI | DOMINATES | DOMINATED | TRADE |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 15 | 19 | -7.86 | [-11.89, -5.06] | -4.912 | [-5.34, -3.46] | 0 | 0 | 19 |
| 45 | 19 | -0.38 | [-1.17, +0.01] | +0.082 | [-0.19, +0.44] | 8 | 3 | 8 |
| 90 | 19 | +18.20 | [+13.63, +20.83] | +0.559 | [+0.33, +1.27] | 0 | 2 | 17 |

**⛔ SUPERSEDED — the same read on the pre-#18-fix block, for scale only**

| q | n images | median Δbytes % | 95% CI | median Δssim2 | 95% CI | DOMINATES | DOMINATED | TRADE |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 15 | 13 | -8.20 | [-8.27, -6.50] | -15.405 | [-42.12, -2.93] | 0 | 2 | 11 |
| 45 | 13 | -0.69 | [-1.05, -0.31] | -67.209 | [-83.88, +0.06] | 2 | 0 | 11 |
| 90 | 13 | +18.01 | [+14.81, +21.71] | -104.800 | [-114.52, +0.00] | 0 | 8 | 5 |

**⛔ SUPERSEDED — its passthrough half, which is BYTE-IDENTICAL to c1's and so reproduces c1's numbers exactly**

| q | n images | median Δbytes % | 95% CI | median Δssim2 | 95% CI | DOMINATES | DOMINATED | TRADE |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 15 | 19 | -7.86 | [-11.89, -5.06] | -4.912 | [-5.34, -3.46] | 0 | 0 | 19 |
| 45 | 19 | -0.38 | [-1.17, +0.01] | +0.082 | [-0.19, +0.44] | 8 | 3 | 8 |
| 90 | 19 | +18.20 | [+13.63, +20.83] | +0.559 | [+0.33, +1.27] | 0 | 2 | 17 |

### 4.4 How much the answer actually changed on the corrupt images

The 8 multi-tile images, pre-fix (`t1d`) against post-fix (`c1`), same control,
same q, same paired read. This is the size of the error that was in the record:

| image | q | Δbytes % pre-fix | Δbytes % post-fix | Δssim2 pre-fix | Δssim2 post-fix | verdict pre-fix | verdict post-fix |
|---|--:|--:|--:|--:|--:|--:|--:|
| `1008` | 15 | -8.25 | -8.41 | -67.731 | -3.843 | TRADE | TRADE |
| `1008` | 45 | -0.72 | -0.74 | -112.501 | +0.616 | TRADE | DOMINATES |
| `1008` | 90 | +18.18 | +18.16 | -143.622 | +4.182 | DOMINATED | TRADE |
| `1220` | 15 | -8.15 | -8.10 | -40.128 | -4.843 | TRADE | TRADE |
| `1220` | 45 | -1.03 | -1.02 | -99.072 | +0.244 | TRADE | DOMINATES |
| `1220` | 90 | +14.03 | +14.02 | -141.837 | +2.195 | DOMINATED | TRADE |
| `1420` | 15 | -7.59 | -7.54 | -42.125 | -4.391 | TRADE | TRADE |
| `1420` | 45 | -1.05 | -1.20 | -78.067 | +0.014 | TRADE | DOMINATES |
| `1420` | 90 | +21.71 | +21.68 | -114.524 | +2.942 | DOMINATED | TRADE |
| `1432` | 15 | -6.50 | -6.76 | -49.434 | -3.192 | TRADE | TRADE |
| `1432` | 45 | -2.07 | -1.93 | -94.408 | +0.239 | TRADE | DOMINATES |
| `1432` | 90 | +38.10 | +38.03 | -106.081 | +3.945 | DOMINATED | TRADE |
| `1442` | 15 | -8.26 | -8.30 | -2.927 | -2.049 | TRADE | TRADE |
| `1442` | 45 | -2.72 | -2.66 | -46.951 | +2.140 | TRADE | DOMINATES |
| `1442` | 90 | +66.29 | +66.30 | -106.862 | +2.355 | DOMINATED | TRADE |
| `1634` | 15 | -8.20 | -8.33 | -45.945 | -3.780 | TRADE | TRADE |
| `1634` | 45 | -1.57 | -1.57 | -83.875 | +0.371 | TRADE | DOMINATES |
| `1634` | 90 | +27.14 | +27.11 | -126.487 | +3.541 | DOMINATED | TRADE |
| `6602` | 15 | -9.35 | -9.44 | -34.916 | -3.224 | TRADE | TRADE |
| `6602` | 45 | -0.54 | -0.53 | -67.209 | +0.029 | TRADE | DOMINATES |
| `6602` | 90 | +17.81 | +17.80 | -88.325 | +0.459 | DOMINATED | TRADE |
| `6604` | 15 | -9.53 | -9.51 | -15.406 | -3.830 | TRADE | TRADE |
| `6604` | 45 | -0.31 | -0.31 | -76.184 | +0.092 | TRADE | DOMINATES |
| `6604` | 90 | +14.81 | +14.86 | -104.800 | +0.795 | DOMINATED | TRADE |

The pre-fix rows are shown **only** to size the correction; they are superseded
and must not be cited as measurements of anything.

Three things this table settles:

- **The error was in the pixels, not the rate.** Δbytes is essentially unchanged
  pre-to-post on every one of the 24 cells (e.g. `1442` q90: +66.29 % → +66.30 %).
  The broken encoder produced a normally-sized bitstream containing wrong pixels
  — which is exactly why a byte-count sanity check would never have caught it,
  and why the quality column is where the damage shows.
- **The verdict flips on every affected cell.** All 8 images go
  `DOMINATED → TRADE` at q90 and `TRADE → DOMINATES` at q45. A reader of the
  pre-fix block would have concluded that 10-bit encoding is *catastrophically*
  worse at native size (median Δssim2 **−104.80** at q90 over the 13 images);
  the clean read says **+2.00**.
- **It independently reproduces the fix's own verification.** `1008` at q90
  reads **+4.182** here — the same number `avif_hdr_arm_plan_2026-09-02.md`
  §10.4f recorded from the local recon gate ("*bd10 now beats its 8-bit twin by
  +4.18*"), arrived at through a completely different path: fleet encode, fleet
  scoring, this analysis. Two independent instruments, same value to three
  decimals.

**Instrument note for this table.** The pre-fix (`t1d`) scores come from the
`avifdoe-svt-t1-sf-cpu-20260902` score run and the post-fix (`c1`) scores from
`avifdoe-svt-eradelta-sf-cpu-20260903` — different scoring images. That is safe
for `ssim2` specifically: this lane measured `ssim2` to be **bit-identical
across four different builds** on identical bitstreams (§2.5 here, and §1.2b of
the HDR companion doc). It would **not** be safe for `zensim`, which is why no
`zensim` number appears in this document.

**The registered answer to Q4 ("does the effect survive native resolution?"):
YES, and if anything it is slightly stronger on the large images.** The two
halves of this table are a size contrast by construction — the 19 passthroughs
are sources that were already at or below the budget size, the 13 `crop-native`
are the genuinely large ones — and at q45, the q where `bd10` is closest to a
free win:

| population | median Δbytes % | median Δssim2 | images where bd10 DOMINATES |
|---|--:|--:|---|
| budget corpus, 1024², 32 images (Stage-A instrument, for reference) | −0.39 | +0.280 | 18/32 |
| **native, 19 passthrough (small) images** | **−0.38** | **+0.082** | **8/19** |
| **native, 13 `crop-native` (large) images — the registered n** | **−0.69** | **+0.244** | **10/13** |

There is no size cliff here: the sign, the magnitude and the dominance rate all
carry over, and the large-image half is marginally the better of the two. Note
this is the *paired* read, not a BD-rate, so it is not directly comparable to
Stage-A's −1.02 %/−1.22 % BD-rate figure — see the guard note above.

**Control provenance, declared rather than buried.** The 8-bit control is
`avifdoe-svt-ag-20260901`'s `s4-svt-420` stratum — the exact shape match
(native corpus, speed 4, `q ∈ {15,45,90}`, 32 images) the HDR plan §10.4a
identifies. It was encoded at the OLD pin. That is a **cross-era control** and
is recorded as such in every row of `paired_per_q.tsv` (`ctl_run` column). It
is sound here for a reason that is measured, not assumed: the control is
**8-bit**, and zenav1-svt #18 lived in `dr_predict_hbd`, the *high-bit-depth*
directional intra path. The 8-bit path's era-invariance is not taken on faith —
§2.1's 6,912-cell join is almost entirely 8-bit cells and is 100 % identical.

---

## 5. What this wave does NOT establish

- **Nothing about the other 31 commits.** The identity join proves the delta
  changes no bytes *on the configuration space this wave covers*. It says
  nothing about configurations outside it (12-bit, superres, tiles, fork mode,
  the `hdr_fork()` constructor path of §4.2).
- **No BD-rate for bd10 at native size** (§4.3), in either era.
- **No answer to Stage-B's B-2 (QM × sharpness).** Unchanged as the highest-value
  remaining item; the registration already reconciled it and this wave did not
  absorb it.
- **No speed-7 comparison against Stage-A** for any knob but the controls —
  Stage-A never crossed a knob with speed 7, so the speed-7 knob legs are a
  **new axis**, reported as first measurements, never as a stability check.
- **`encode_ms` is still not persisted**, so B-4 remains NOT EVALUABLE. This
  wave changes nothing there.
- **`svt_doe_t1_bd10_knobs` (`avifdoe-svt-t1b-20260902`, 4,320 cells, 0 done)**
  is still an open pre-existing gap. Flagged, not adopted.
- **7 of the 16 speed-6 knobs are not re-verified** (§2.2b). Their Stage-A
  numbers stand, but this wave did not test them at the new pin.
- **Nothing about `scm3`'s mechanism.** *Why* `screen_content_mode` becomes live
  at preset 9 is untraced — `sc_detect.rs` scoping, a preset-gated path, or
  something else. That is root-cause work in the port repo. This wave measures
  the effect and its content boundary, nothing about its cause.
- **No `scm3` speed sweep above 7.** Speeds 8, 9 and 10 all alias to preset 9,
  so they are the same cell; nothing was learned about presets the product dial
  cannot reach (preset 8 in particular, where the port-level probe first saw the
  divergence, is unreachable and stays unmeasured *as a product configuration*).
- **No `zensim` number anywhere.** The DOE emits `zensim` as a 720-wide feature
  vector, and the scalar is instrument-era-stamped (measured in the HDR
  companion doc, §1.2b: 11.14 points between two scoring images on identical
  bytes). `ssim2` is the corpus-wide scalar response, exactly as the DOE plan's
  §7.1 designates.

---

## 6. Reproduction

Data pointer with paths, shas and the exact command chain:
[`avif_eradelta_analysis_2026-09-03.pointer.md`](avif_eradelta_analysis_2026-09-03.pointer.md).

Tools (all in `scripts/jobsys/`, all committed):
`avifdoe_score_gapfill.sh` → `avifdoe_harvest.py` →
`avifdoe_stagea_analyze.py --control inrun` (BD-rate + the paired read) →
`avifdoe_era_compare.py` (cross-era identity + effect stability).
BD-rate parity against `zenavif/scripts/rd_gap/bd_arm.py` is asserted on every
run (`--parity-check`), and was **exact (max |Δ| 0.0)** for this wave.

**No rank statistic appears in this document**, so `zenstats` is not called and
is not being bypassed: BD-rate is an integration (the DOE analyzer owns it,
parity-gated), the medians and their CIs come from that same analyzer's
percentile bootstrap, and the identity results are exact counts. A SROCC/PLCC
number would have to come from `zenstats` via `panel`; none is reported.
