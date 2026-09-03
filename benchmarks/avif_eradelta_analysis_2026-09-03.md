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
   `scm3` changes the bitstream on **90 of 288 cells (31.3 %)**; the divergence
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
| **7 (p9)** | **total** | **288** | **198** | **90 (31.3 %)** |

Three things worth stating precisely:

- **The speed-4 and speed-6 zeros are a reproduction, not a null.** Stage-A's
  own `main_effects.tsv` reports `scm3` at exactly `0.0000` with a
  `[0.0000, 0.0000]` CI at both speeds — because the arm *was* the control.
  This wave reproduces that on 576 cells at the new pin.
- **The divergence is content-exclusive, not content-graded.** Photo and
  AI-generated content are at 0/144. Every differing cell is
  plot / screenshot / scan. That is the behaviour a screen-content detector
  should have, and it is the first time we have seen it through the product's
  own knob surface.
- **AI-generated content behaves like photo here**, not like screen content —
  9 images, 81 cells, zero divergence. Worth knowing before anyone assumes
  "synthetic ⇒ screen".

### 3.1 Effect size

<!--TABLE:SCM3EFFECT-->

### 3.1b Where the BD-rate comes from — the paired matched-q bytes

A −50 % median BD-rate is large enough that it deserves the underlying bytes on
the page rather than only the integral. `scm3` vs its own speed-7 control, on
three of the firing images:

<!--TABLE:SCM3BYTES-->

Two things this makes visible that the BD-rate alone does not. **The win is not
uniform in q** — at the very lowest q, palette/IntraBC coding costs *more* bytes
(`7050` +25.1 %, `8414` +35.6 % at q5) while buying a great deal of quality
(+57.7 and +12.4 ssim2), which is a trade the BD-rate integration prices
correctly and a per-q byte comparison would misread. And **the top of the
ladder is where it pays**: on `6018` (a scanned document) q90 goes 435 KB →
29.5 KB at ssim2 100.000 vs 94.636 — the screen-content tools are coding the
page essentially exactly at a fourteenth of the rate.

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

<!--TABLE:S7EFFECTS-->

**Stage-B trigger B-5 (registered evaluation, now on three presets instead of
two).** B-5 fires when a knob's |median BD-rate| ≥ 1 % at two presets *with
opposite signs*. Adding speed 7 as a third point:

<!--TABLE:B5-->

**B-5 still does not fire** — and the reason is now more interesting than
"no flip found". Every knob whose effect is large enough to test is **monotone
in preset**: `qml1.2.10` and `qml1.4.10` strengthen from ≈0 at speed 4 to
several percent at speed 7, `shp3`/`shp7`'s cost rises toward the fast presets,
and the `vbst*` arms decay toward zero. Stage-A read `qml1.2.10` as
"sign-flipping between presets" from two points; with three, it is a knob whose
benefit grows with preset and merely passes through zero near speed 4.

`shp7`'s rise (`+5.47 → +7.53 → +9.33 %` at s4/s6/s7) independently reproduces
the Stage-B6 native-corpus finding of `+7.15 / +7.99 / +9.46 %` at the same
three speeds, on a *different corpus* — the budget crops here vs B-6's native
images.

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

What the block *can* answer is the paired matched-q read: at each `q`, bd10's
bytes and quality against its 8-bit twin.

<!--TABLE:C1PAIRED-->

### 4.4 How much the answer actually changed on the corrupt images

The 8 multi-tile images, pre-fix (`t1d`) against post-fix (`c1`), same control,
same q, same paired read. This is the size of the error that was in the record:

<!--TABLE:C1VST1D-->

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
