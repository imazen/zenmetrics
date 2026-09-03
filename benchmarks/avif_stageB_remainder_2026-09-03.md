# AVIF DOE — Stage-B remainder: the native interaction block + the zenrav1e SDR arm

**Status: PRE-REGISTERED AND LAUNCHED 2026-09-03. No results here.** Grid, budget
arithmetic, ranking, bars and de-scope rules are all fixed *before* the data exists;
analysis is a later lane. Companion: [`avif_speed_instrument_2026-09-03.md`](avif_speed_instrument_2026-09-03.md)
(the timing axis, single-host on r7900x).

Predecessors, all load-bearing here: [`avif_doe_plan_2026-09-01.md`](avif_doe_plan_2026-09-01.md)
(§7.2 the trigger list, §7.3 the de-scope ladder, §15 B-6),
[`avif_doe_stageA_2026-09-02.md`](avif_doe_stageA_2026-09-02.md) (§10 the 53 triggers and
their ranking), [`avif_doe_stageB6_analysis_2026-09-02.md`](avif_doe_stageB6_analysis_2026-09-02.md)
(§17.4 the QM×sharpness gap), [`avif_eradelta_analysis_2026-09-03.md`](avif_eradelta_analysis_2026-09-03.md)
(the `scm3` mooting and the pin-stability licence).

---

## 0. TL;DR

1. **Two runs, 16,768 cells, ~19 CPU-h estimated encode** — against a remaining
   envelope of **~35,000 cells / ~46.8 CPU-h** (60,000 / 60 registered, minus B-6's
   25,056 / 13.2). The envelope is a **ceiling, not a target**: the balance is
   deliberately unspent and held for the gated follow-ons in §5, because the arms
   that would fill it are the ones Stage A itself told us to deprioritise.
2. **Rank 1 is QM × sharpness at native**, and the block that buys it is a
   **complete 4×3 factorial** in (qml level, shp level) — B-2's registered
   "3 levels each" shape, *exceeded* on the qml axis, not a subset of it.
3. **A zenrav1e SDR RD arm is included and is coordinator-required** (design
   requirement, 2026-09-03: *"we want the model to be able to pick the best backend
   for an image"*). The corpus has **zero** zenrav1e SDR coverage today.
4. **Neither run needed a zenavif change or a new fleet image.** The interaction
   block is the existing `svt_doe_pairwise` plan pointed at the native corpus and
   stratum-filtered; the zenrav1e arm goes through the builder's other canonical
   entry point, `--knob-grid`, because `backend` is already a first-class zenavif
   sweep knob and `zenravif` is its default.
5. **Scoring was declared at launch**, in the same session, before this document was
   written. Three prior waves launched with nothing scoring; that is the failure this
   bullet exists to not repeat.

---

## 1. Budget — the arithmetic, shown

| | cells | encode CPU-h |
|---|--:|--:|
| Stage-B envelope (plan §7.2) | 60,000 | 60.0 |
| less B-6, COMPLETE (plan §17.1, §17.6) | −25,056 | −13.2 |
| **remaining** | **34,944** | **46.8** |
| this wave — `brnat` | 7,488 | ~4 (est.) |
| this wave — `brsdr` | 9,280 | ~24 (est.) |
| **this wave total** | **16,768** (48.0 % of remaining) | **~28 (est.)** |
| **held, unspent** | **18,176** | **~19** |

**The CPU-h figures are ESTIMATES and are labelled as such.** Their basis:

- `brnat`: B-6 encoded 25,056 native cells in 13.2 CPU-h = **1.90 s/cell** blended
  over speeds {4, 6, 7}. This block is s6-only, i.e. mid-ladder, so 7,488 × 1.9 s
  ≈ 3.95 CPU-h. Confidence: high — same corpus, same encoder, same speed range.
- `brsdr`: MEASURED on r7900x this session, and the measurement is why the estimate
  moved. Summed over the 10-speed dial at q45, zenrav1e costs **47.2 s/MP** on a
  photo (`1442`, 1024² crop) but **160.9 s/MP** on a screenshot (`8288`, 0.25 MP) —
  a **3.4× content spread**. At a ~80 s/MP blended mean over the budget corpus's
  37.02 MP that is ~0.82 CPU-h per q point, ×29 q ≈ **23.9 CPU-h**. Confidence:
  low-to-moderate — two content points, and it assumes per-q flatness (§4).

**The estimate is gated, not trusted** — see G-RATE in §4.

---

## 2. `brnat` — the native s6 interaction block (rank 1)

**Run:** `avifdoe-svt-brnat-20260903` · **corpus:** NATIVE
(`s3://codec-corpus/avif-subsample-2026-09-01/`) · **plan:** `svt_doe_pairwise`,
stratum-filtered to 26 of 118 · **ladder:** the registered 9-point knob ladder
`5,15,25,35,45,60,76,90,96` · **26 × 9 × 32 = 7,488 cells** (dry-run VERIFIED
exactly, filtered from 33,984).

### 2.1 Why this is rank 1

Three independent readings put it there and none of them is this lane's opinion:

- Stage A §10.1(3): the QM × sharpness cluster is **"the strongest measured
  structure in the wave."**
- The era-delta wave re-flagged it **unchanged** as "the single highest-value
  remaining Stage-B item," and confirmed none of the QM or sharpness knobs sit in
  the AVIF-relevant commit delta.
- B-6 §17.4 established both halves of the gap: **no declared wave contains a
  native (qml × shp) grid**, and B-6 *removed* the competing explanation — the
  size-driven shift in the additive baseline is **−0.39 pp** at corpus level
  against a **−5.2 to −5.5 %** synergy residual, so size is not a plausible cause.

### 2.2 The strata, and what they are a factorial in

| group | strata | n |
|---|---|--:|
| control (the additive baseline) | `s6-svt-420` | 1 |
| main-effect legs | `qml1.2.10` `qml1.4.10` `qml1.8.15` `shp3` `shp7` `mtx32` `tn0` `tn3` `tl1.0` `tl1.1` | 10 |
| **rank 1** — QM × sharpness | `qml{1.2.10,1.4.10,1.8.15}` × `shp{3,7}` | 6 |
| **rank 3** — mtx32 × {QM, sharpness} | `qml{…}-mtx32` (3) + `shp{3,7}-mtx32` (2) | 5 |
| **rank 4** — tune × tile | `tn{0,3}` × `tl{1.0,1.1}` | 4 |
| | **total** | **26** |

The control + 3 qml singles + 2 shp singles + 6 pairs are a **complete 4 × 3
factorial** in (qml ∈ {default, 1.2.10, 1.4.10, 1.8.15}, shp ∈ {default, 3, 7}).
B-2's registered follow-up shape is "full k1×k2 grid, 3 levels each"; this is 4
levels × 3 levels, so the interaction is **over**-covered on the qml axis, not
approximated.

### 2.3 What the block gets for free

A2 already ran **this same plan, at the same 9-q ladder, on the same 32 images, at
BUDGET size**. So `brnat` is a clean **size A/B on the interaction itself**, cell
for cell, with no new svt encodes on the budget side. That is why the block is
worth 7,488 cells rather than the 15,552 §17.6 costed it at: §17.6 priced B-2's
generic per-pair grid; this prices the plan's own strata, which the fleet can
execute today.

### 2.4 What is deliberately ABSENT — a ranking, not an oversight

- **Every `vbst*` stratum.** Stage A §10.1(4): the variance-boost arms are "the most
  expensive and the least certain," they fire on **IQR rather than median**, and it
  says to deprioritise them **first**. Six of the 17 B-1 triggers are `vbst`.
- **Every `scm3` pair — MOOTED, and this is a measurement, not a judgement call.**
  The era-delta wave measured `scm3` through the sweep interface, statistics-free:
  at **speed 6 it changes the bitstream on 0 of 288 cells** (0/288 at speed 4 too).
  It is real only at speed 7, where it is screen-content-exclusive (90/144 on
  plot+screenshot+scan, **0/144** on photo+AI). An interaction with a knob that is
  byte-identical to the control at this preset **cannot exist**. Cite:
  `avif_eradelta_analysis_2026-09-03.md` §0.2 and §3.
- **Every `acb*` stratum.** B-6 retired `ac_bias` outright: 12 (speed, level)
  medians span **−0.03 % to +0.39 %** at native, and `acb8` — the only level with a
  defensible effect — is a **loss** (§17.3).

---

## 3. `brsdr` — the zenrav1e SDR RD arm (rank 2, coordinator-required)

**Run:** `avifdoe-rav-brsdr-20260903` · **corpus:** BUDGET
(`s3://codec-corpus/avif-doe-1024-2026-09-01/`) · **knob-grid**
`{"backend":["zenravif"],"speed":[1..10]}` · **ladder:** the full 29-point control
grid · **10 × 29 × 32 = 9,280 cells** (dry-run VERIFIED exactly).

### 3.1 Registration authority and the gap it fills

Coordinator design requirement, 2026-09-03, mid-flight: *"we want the model to be
able to pick the best backend for an image."* Backend selection becomes a model
**output**, so the data must support per-image cross-backend comparison at matched
quality. Today it cannot: **svt coverage is deep and zenrav1e SDR coverage is
zero** — zenrav1e appears only in the 16-reference HDR arm, where svt wins −43 %
on 16 of 16. Defaults-first; zenrav1e's own knob surface is a later tranche.

### 3.2 Why BUDGET corpus, and why the 29-point ladder

Both choices exist to make the comparison exact rather than approximate:

- **A0R-svt already holds the 29-q control ladder on these exact references.** So
  this arm is directly comparable to svt cell-for-cell with **no new svt encodes**.
- **q values do NOT align across backends.** Matching therefore happens at
  *analysis* time in **quality space** (achieved ssim2 / zensim), never on q — the
  same convention T2 used. A dense 29-point ladder is what makes that matching
  accurate; it is not q-density for its own sake.

### 3.3 The full 10-speed dial is measured even where it aliases

The two backends alias **differently**, and assuming otherwise is exactly the kind
of transfer this wave exists to test. MEASURED on r7900x this session:

| | speed 1 | 4 | 7 | 10 |
|---|--:|--:|--:|--:|
| svt-rs bytes (`1442`, 1024²) | 3572 | 3726 | 4163 | **4163** |
| zenrav1e bytes (same) | 4559 | 5640 | 5896 | **6628** |

svt saturates at preset 9 — speeds 7–10 byte- **and** time-identical (H-2, and
B-6 §17.5(3) found the same on the naive sweep). **zenrav1e's dial is still
moving at speed 10.** A separate probe on `8288` found zenrav1e aliasing at
**7/8** instead (15583 bytes and 747.8/742.2 ms). Aliased cells are free
confirmation; an assumed alias structure is a silent hole.

### 3.4 aom-rs — PLANNED-BLOCKED, not forgotten

The third backend joins as a third arm when its era pins land post-#15. Timing or
RD-profiling a port that is still landing byte-identity fixes would measure a
moving target. Registered here so the omission is explicit.

---

## 4. Gates and de-scope, all fixed before the data exists

| id | gate | action on failure |
|---|---|---|
| **G-SMOKE-1** | dry-run cell counts equal the registered arithmetic (7,488 / 9,280) | do not declare |
| **G-STRATA** | all 26 requested strata resolve in the plan | **hard error in the filter script** — a stratum the plan stopped spelling would silently shrink the block |
| **G-LIVE** | the filtered strata are not silently collapsing to the control | inspect before declaring |
| **G-IMAGE** | the fleet image's bogus-plan control arm lists `svt_doe_pairwise`, **run against a real source** | rebuild image; an empty sources dir makes both arms return the same error (plan §11.5) |
| **G-FIRSTCELL** | one real cell of each arm encodes inside the image | stop, do not scale |
| **G-RATE** | after the first ~500 `brsdr` cells, the realised s/cell must keep the run inside the remaining envelope | fire the §4.1 de-scope |
| **G-SCORE** | score jobs declared **at launch**, not after | — |

**Status: ALL SEVEN GATES PASS.** Evidence in §6.

**G-RATE, MEASURED — and the first measurement was wrong, which is recorded rather
than quietly replaced.**

| window | cells | CPU-s/cell | `brsdr` projection | wave total | verdict |
|---|--:|--:|--:|--:|---|
| 89 s paired sample | 84 | 7.17 | 18.5 CPU-h | 22.5 | PASS (52 % margin) |
| **cumulative from launch** | **323** | **15.42** | **39.7 CPU-h** | **43.7** | **PASS (7 % margin)** |

**The 89-second window was unrepresentative and the cumulative figure supersedes
it.** The cause is already documented in §3 and §1: zenrav1e's per-pixel cost has a
**3.4× content spread**, and the worker walks the corpus in image order, so any
short window samples one content class rather than the mix. A rate sampled over
~1.5 minutes of a multi-hour run is not a rate. Both numbers are kept here because
the error is instructive: the short-window figure would have been reported as a
comfortable 52 % margin when the real margin is 7 %.

**The gate as pre-registered PASSES (43.7 ≤ 46.8), so the §4.1 de-scope does NOT
fire** — a gate that passes is not re-argued after the fact. But the margin is now
inside the uncertainty of the estimate itself, so G-RATE is treated as a
**standing** gate rather than a one-shot: the cumulative rate is re-checked as the
run drains, and de-scope step 1 fires the moment the projection crosses 46.8.

Note the **cell** budget — the constraint the brief set — is not close: 16,768 of
~34,944 is **48 %**. It is the plan's CPU-h envelope that is tight, and only because
zenrav1e is genuinely expensive (its speed-1 cell costs 14.99 s on a 0.25 MP
screenshot).

**G-SCORERATE — the co-equal constraint, measured on the current mix.** The
registered score ceiling is 50 CPU-h. Measured on the live `dev-brsf` container:
**2,437 CPU-s over 225 scored cells = 10.8 CPU-s/cell**.

That figure is **native-dominated** — `brnat` (5.05 MP mean) was 46 % filled while
`brsdr` (1.157 MP mean) was 5 %, so almost every scored cell so far is a native one,
and score cost scales with pixels. A whole-run projection therefore depends on a
mix that has not been observed yet, and multiplying by a pixel ratio to guess it is
exactly the extrapolation this repo forbids. **What is established: the native leg
alone projects to ~22.5 CPU-h of the 50.** The blended figure is re-measured once
`brsdr` cells start entering the scorer; if the total crosses 50 CPU-h, the plan's
own de-scope (§7.3) fires on the score side.

Wall ETAs, work-weighted from live counters rather than from cell counts:
`brnat` ~48 cells/min → **~1 h**; `brsdr` at 8.47 of 10 cores busy → **~4 h**.

### 4.2 Completion — `brnat` is DONE, and a correction to how fill was being read

**`avifdoe-svt-brnat-20260903`: COMPLETE — 7,488 / 7,488 DONE cells, 0 failed**
(2026-09-03T19:38Z, ~66 min wall on tower under household caps). The worker then
drained and began the restart loop described in §6; it was stopped and the box
repointed at `brsdr`.

**⛔ A blob count is NOT a cell count, and this lane read it as one for the first
hour.** Blobs are **content-addressed**, so any two cells whose encoder output is
byte-identical collapse onto **one** blob. `brnat` finished at **7,488 cells but
only 5,817 distinct blobs** — **1,671 cells (22.3 %) produced bytes another cell
had already produced.** Progress reported off the blob prefix therefore reads
~78 % at completion and would never reach 100 %, which is exactly the shape of a
stall. B-6 saw the same thing and said so (§17.1: "23,489 distinct bitstreams" for
25,056 declared cells); this lane rediscovered it the slow way.

**Read fill from the ledger**, i.e. the `pairs: N DONE cells` line the gap-fill loop
emits every round, never from `aws s3 ls .../blobs/ | wc -l`.

**The dedup rate is also a result, not just an accounting note.** 22.3 % of this
block's cells are byte-identical to some other cell in it, and the block is 26
strata over 32 images at 9 q. A stratum whose bytes equal the control's on an image
has **zero** effect there — so that 22.3 % is an upper bound on how much of the
declared grid can carry any interaction signal at all, and the per-stratum
breakdown is a cheap first cut for the analysis lane (it needs no scores). Same
caveat as always: byte-identity is measured per cell, so it must be counted per
(stratum, image), never pooled.

### 4.1 Pre-registered de-scope (fires on G-RATE only)

In this fixed order, re-checking after each step:

1. **`brsdr` 29-q → the 9-point knob ladder** (9,280 → 2,880 cells). This is the
   first cut because 9 points still clears the BD-rate owner's ≥ 4-point guard, so
   quality-space matching survives it; only the *precision* of the match degrades.
2. **`brsdr` drops speeds 1–2** — the two most expensive cells on the dial
   (speed 1 alone is 14.99 s on a 0.25 MP screenshot) and the two least likely to
   be product choices.
3. **`brnat` drops the rank-4 tune × tile group** (4 pairs + their 4 singles).

**Never cut:** the QM × sharpness factorial, or the control. Those are the two
things this wave's authorisation exists to protect.

---

## 5. What is held unspent, ranked, so a future tranche is mechanical

18,176 cells and ~19 CPU-h remain after this wave. In priority order:

| rank | item | cells | why not now |
|--:|---|--:|---|
| 5 | **zenrav1e at NATIVE**, 9-q, 10 speeds | 2,880 (~19 CPU-h) | **Gated on `brsdr` deliberately** — cheapest-discriminating-first. A single r7900x cell already has svt at 52× the speed, 40 % fewer bytes and a higher zensim than zenrav1e (§6). If `brsdr` confirms svt dominates on SDR RD *and* speed, a native zenrav1e leg is money spent on a settled question. |
| 6 | `vbst*` B-1 dense grids (6 triggers) | ~83,520 if honoured | Stage A ranks these last: IQR-triggered, most expensive, least certain. Needs its own budget decision, not a leftover. |
| 7 | B-3 content-stratified follow-ups (13 triggers) | ~123,540 if honoured | Several rest on classes with n < 3 and are already marked PROVISIONAL. |
| 8 | `tl`/`tn` **multi-speed** B-1 ladders | 13,920/knob | `brnat` covers them at s6 only; the speed axis needs `svt_doe_main`, i.e. a different plan and a bigger block. |
| 9 | aom-rs third backend arm | — | PLANNED-BLOCKED on era pins post-#15 (§3.4). |

**Honouring every remaining trigger costs 447,636 cells against a 60,000 envelope
— 7.5×** (Stage A §10). That ratio is why this document ranks and cuts rather than
enumerating.

---

## 6. Execution record — 2026-09-03

**Machinery, all in the canonical owners, no forks:**

| change | where |
|---|---|
| `--stage-b-remainder` mode | `scripts/jobsys/avifdoe_declare.sh` |
| `declare_block_filtered` + `declare_block_knobgrid` | same |
| the stratum filter, hard-erroring on a missing name | `scripts/jobsys/avifdoe_filter_cells.py` (new) |
| score gap-fill | **reused unforked** via `ZEN_DOE_RUNS`, as designed |

**Gate evidence.**

- G-SMOKE-1: `brnat` "strata requested: 26 all present cells kept: 7488 of 33984";
  `brsdr` "emitted 9280 declare items".
- G-LIVE: on `8288` at native, control **17,899 B** vs `qml1.2.10-shp3` **17,384 B**
  vs `qml1.8.15-shp7` **19,407 B** vs `tn3-tl1.1` **19,012 B** — distinct, so the
  strata are live.
- G-IMAGE / G-FIRSTCELL, inside
  `ghcr.io/imazen/zenfleet-worker:exec-avifhbd-eradelta-e015344f` with a real
  source mounted: the bogus-plan arm lists all 13 plans including
  `svt_doe_pairwise`; `brnat` encodes **118/118** cells and `brsdr` **1/1**, zero
  failures.
- G-SCORE: score run `avifdoe-br-sf-cpu-20260903`, metrics **`ssim2,zensim`** (the
  standing directive drops butteraugli), declared in round 1 — 60 encoded cells →
  6 score jobs — **before this document existed**.

**Topology (observe-before-load; r7900x is reserved).**

| box | role | cap |
|---|---|---|
| tower (32c) | encode `Tower-brnat` | `--cpuset-cpus 0-19 --cpu-shares 256 --memory 24g` — live household media server, never uncapped |
| dev (32c) | encode `dev-brsdr` | `--cpuset-cpus 20-29 --memory 24g` |
| dev (32c) | score `dev-brsf` | `--cpuset-cpus 8-15 --memory 24g` |
| **r7900x (24c)** | **NONE — reserved** | the uncontended timing instrument runs there until it writes `COMPLETE`; a fleet worker on that box would invalidate every `encode_ms` it is measuring |

Before launch, two **drained** B-6 containers were stopped — one on r7900x and one
on tower, each in a `unless-stopped` restart loop with **2,466 restarts**, reporting
`done=0 … drained` every pass and burning ~27 % of a core for zero output. The B-6
run is COMPLETE (25,056/25,056), so nothing was destroyed; this is zenfleet's own
`idle` waste, on the two boxes this wave needed, exactly as plan §15.7 found it.

**Era labelling.** Both runs are pinned to the era-delta image
`exec-avifhbd-eradelta-e015344f`. The era-delta wave established that this pin
reproduces Stage A **byte-for-byte on 6,912 of 6,912 shared cells**, which is what
licenses comparing `brnat` against A2's Stage-A-era budget cells. Rows stay labelled
by pin regardless; never cross-join an unlabelled row.

---

## 7. Limitations — stated before any result

1. **s6 only.** `brnat` inherits `svt_doe_pairwise`'s single preset. B-6 showed
   sharpness's transfer behaviour is **speed-specific** (FAIL-T1 at s4, PASS at s6),
   so nothing here may be stated about the interaction at s4 or s7.
2. **`brnat` is a plan-expressible factorial, not an arbitrary one.** The qml axis
   carries the 3 levels the plan spells; a finer QM sweep is a different block.
3. **The `brsdr` CPU estimate rests on two content points** and an unverified
   per-q-flatness assumption. The speed instrument's S1b block measures that
   assumption directly; until it lands, treat §1's 23.9 CPU-h as provisional.
4. **T1's construction defect is inherited, not fixed.** B-6 §17.2 found T1 grades
   sign agreement against budget-side noise. Amending a pre-registered bar after
   seeing results is the coordinator's call; this wave does not re-use T1, and any
   new gate here is registered fresh (§4).
5. **No timing.** `encode_ms` is still not persisted by the fleet path. That axis is
   the companion document's, on a single uncontended host, by construction.
