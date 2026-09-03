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

1. **Two runs, 16,768 cells, MEASURED 28.6 CPU-h encode** — against a remaining
   envelope of **~34,944 cells / 46.8 CPU-h** (60,000 / 60 registered, minus B-6's
   25,056 / 13.2). The envelope is a **ceiling, not a target**: the balance is
   deliberately unspent and held for the gated follow-ons in §5, because the arms
   that would fill it are the ones Stage A itself told us to deprioritise.
   **`brnat` is COMPLETE** (7,488/7,488, 0 failed); `brsdr` is draining.
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
6. **Two of this lane's own measurements were wrong and are corrected in place, not
   quietly replaced** (§4.2, §4.3, §4.4): a blob count is not a cell count
   (content-addressing collapses byte-identical cells, so `brnat` reads 78 % at
   completion), and the G-RATE figure that fired the de-scope was **recomputation
   from 1800 s pass timeouts**, not intrinsic cost. The clean rate is **9.52
   CPU-s/cell**, the de-scope was **reverted**, and the full 29-q ladder stands.

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

### 4.3 ⛔ The G-RATE measurement was CONTAMINATED, and the de-scope was fired on it

**Found 2026-09-03T19:50Z, after firing.** The `brsdr` workers were running
zenfleet's **resource-aware concurrent mode** — *"LPT + can_admit, chunk target
300s"* — while `fleet-entrypoint.sh` kills a pass at **`ZEN_PASS_TIMEOUT`, default
1800 s**. Two consecutive passes died that way:

```
[FLEET-ERROR 19:03:17Z] pass 26 TIMED OUT after 1800s — worker hung.
  worker| spot preemption — released chunk claim chunk-9b4beae… for fast requeue
         (its cells re-enter the gap; any already-flushed cells in it stay Done)
[FLEET-ERROR 19:33:17Z] pass 27 TIMED OUT after 1800s — worker hung.
```

The chunk sizer admits cells against a cost model calibrated on svt-class work.
**zenrav1e is 3.7×–64.8× slower per cell** (companion instrument, §3.3), so a
"300 s" chunk of zenrav1e cells runs far past 1800 s, the pass is killed, the claim
is released, and **every cell in it that had not yet flushed is recomputed**.

**Consequence: the CPU-per-cell figures in §4.2 measure waste, not intrinsic cost.**
The 15.42 → 28.1 CPU-s/cell "rise" is not content ordering as the earlier note
assumed — it is recomputation accumulating. Both numbers are upper bounds of unknown
tightness, and **G-RATE was failed against them**, which is why the de-scope fired.

**What was done, in order, and why the order matters.** The de-scope is *reversible
by construction* (§4.4's shrink preserves the previous manifest), so it was left in
place rather than hastily reverted, and the operational defect was fixed first:
both workers were relaunched with **`ZEN_CHUNK_WALL_SEC=0`** — the serial per-cell
path the worker's own log advertises — under which **every completed cell flushes
immediately** and a pass timeout can therefore lose at most one cell. The clean
rate is measured on that path, and only then is the de-scope revisited. Reverting
first would have re-measured the same contaminated number on a bigger grid.

**The launcher already supported the fix and this lane did not use it.**
`lan_score_launch.sh` passes `ZEN_CHUNK_WALL_SEC` and `ZEN_PASS_TIMEOUT` through,
and its own comment says *"big-cell workloads (HDR diffmaps) exceed the 1800s
default and get [an override]"*. A backend measured at up to 24 s for a single
1 MP cell is a big-cell workload; it should have been launched that way. **Any
future zenrav1e or aom-rs fleet arm must set `ZEN_CHUNK_WALL_SEC=0` (or a raised
`ZEN_PASS_TIMEOUT`) at launch** — the concurrent chunk sizer is calibrated for svt
and silently mis-sizes for anything much slower.

**Registered correction to the de-scope LADDER, not applied here.** Measured from
the speed instrument's dial curve, **speeds 1–2 carry ~73 % of zenrav1e's total
cost across the 10-speed ladder** while being only 20 % of the cells. Step 2 of
§4.1 ("drop speeds 1–2") is therefore a far better CPU-per-science trade than
step 1 ("29-q → 9-q", 69 % of cells for ~69 % of cost). **The registered order was
NOT changed mid-flight** — reordering a de-scope ladder after seeing which arm is
expensive is exactly what pre-registration exists to prevent — but the ladder's
order should be rebuilt from measured per-axis cost for the next wave.

### 4.4 RESOLUTION — clean rate measured, de-scope REVERTED, full 29-q grid restored

With both workers on the serial per-cell path (no timeouts on either host), a paired
two-host cgroup-CPU / ledger sample over **429 s and 90 cells**:

| | CPU-s/cell | full 9,280-cell grid | wave total vs 46.8 |
|---|--:|--:|---|
| contaminated (concurrent + 1800 s kill) | 15.42 → 28.1 | 57–72 CPU-h | **FAIL** |
| **clean (serial, no waste)** | **9.52** | **24.6 CPU-h** | **28.6 → PASS, 39 % margin** |

**So the de-scope was never warranted; it was fired on recomputation.** The shrink
was **REVERTED** — `manifest_PRE_DESCOPE_29q.json` restored to `manifest.json`,
verified at 4,425,398 B — and the arm keeps its full 29-point ladder, which is what
makes quality-space matching against A0R-svt precise. Nothing was lost either way:
no ledger row, no completed cell, no worker.

**But serial is not the shipping configuration, and the same sample shows why.**
Local and tower each burned **exactly 1.0 core** for 429 s — `ZEN_CHUNK_WALL_SEC=0`
removes the waste by removing the concurrency. Projected wall time on that path:
**~10.2 h**. The correct configuration keeps concurrency and simply gives chunks
time to finish:

**`ZEN_PASS_TIMEOUT=7200` with the default concurrent chunker** — MEASURED after
relaunch: local **902 %** and tower **905 %** CPU (~18 cores), zero timeouts, full
29-q manifest. At 9.52 CPU-s/cell that is ~113 cells/min, i.e. **~1.1 h** for the
remaining grid instead of 10.2.

**The generalisable rule, since this cost three relaunches to find:** zenfleet's
concurrent chunker sizes work against an svt-calibrated cost model and the entrypoint
kills a pass at 1800 s by default. For any backend materially slower than svt
(zenrav1e here; aom-rs next), **raise `ZEN_PASS_TIMEOUT` — do not reach for
`ZEN_CHUNK_WALL_SEC=0`.** The serial flag is the right *diagnostic* (it isolates
intrinsic cost from recomputation by making every cell flush) and the wrong
*production* setting (it throws away 9× the throughput).

### 4.5 Worker topology, MEASURED three ways — oversubscribed concurrency is the worst of them

The `brsdr` arm was run under three worker configurations and each was measured
with the **authoritative** counter (`zenfleet-ctl compact`'s `distinct_done`) on
both ends of the window, after two earlier attempts were misled by lagging proxies:

| configuration | cells/min | CPU-s/cell | 7,290 remaining cells |
|---|--:|--:|---|
| concurrent, `OVERSUBSCRIBE=2`, 1800 s kill | — | 15.4 → 28.1 (waste) | timed out, recomputing |
| concurrent, `OVERSUBSCRIBE=2`, 7200 s | **26.2** | **34.8** | 4.6 h wall, **70.5 CPU-h** |
| **serial, ×1 worker/host** | 12.6 | **9.52** | 9.6 h wall, 19.3 CPU-h |
| **serial, ×4 workers/host (shipped)** | ~50 (projected) | 9.52 | ~2.4 h wall, 19.3 CPU-h |

**Oversubscribed concurrency costs 3.7× the CPU for 2.1× the throughput** — and the
CPU figure is what the Stage-B envelope is denominated in, so it is the axis that
decides. At 34.8 CPU-s/cell the remaining grid alone is **70.5 CPU-h against 46.8
remaining**; at 9.52 it is 19.3 and fits.

**Cause:** `lan_score_launch.sh` hardcodes `ZEN_CORE_OVERSUBSCRIBE=2`, so a 9-core
cpuset runs ~18 simultaneous AV1 encodes. AV1 encoding is memory-bandwidth- and
cache-heavy with a large per-encode working set, so oversubscription does not
overlap I/O stalls — it thrashes. Serial-per-worker at 1 core each avoids it while
**scaling out instead of up**: eight 1-core workers give ~4× the throughput of two
9-core oversubscribed ones at ~27 % of the CPU.

**Shipped:** 8 serial workers (4 local on cores 20-23, 4 tower on 0-3 with
`--cpu-shares 256`), each `ZEN_CHUNK_WALL_SEC=0 ZEN_LONG_LIVED=1
--restart on-failure:5`.

### 4.5b The serial workers ALSO timed out — because I dropped the setting that fixed it

**MEASURED, and the cause was mine.** The 8 serial workers were relaunched with
`ZEN_CHUNK_WALL_SEC=0 ZEN_LONG_LIVED=1` but **without** `ZEN_PASS_TIMEOUT=7200` —
the override established one section earlier as necessary for this backend. Their
process cmdline therefore read `timeout 1800`, and:

```
[FLEET-ERROR 20:54:53Z] pass 1 TIMED OUT after 1800s — worker hung.
  worker| spot preemption — released claim 9f590d85… for fast requeue
```

A serial worker claims a batch and walks it at ~9.5 s/cell, so 1800 s covers only
~190 cells; a larger claim never finishes, the pass is killed, and the claim is
released. **`distinct_done` sat at 2020 for 38 minutes while eight cores ran flat
out — ~4 CPU-h burned for zero net progress.** The tells were all visible and I read
past them for half an hour: `docker stats` showing 100 % CPU on **42.9 MiB** of
memory (an encode of a 1 MP image is hundreds of MB), `zenfleet-worker` with
**0.08 s** of accumulated CPU, and a worker process only 9 minutes old inside a
38-minute-old container.

**The serial path is not immune to the timeout — nothing is.** `ZEN_PASS_TIMEOUT`
governs the *pass*, and both chunkers put many cells in a pass. The correct
configuration for a backend materially slower than svt is **both** settings
together:

```
ZEN_CHUNK_WALL_SEC=0   # CPU-efficient: no oversubscription thrash (§4.5)
ZEN_PASS_TIMEOUT=7200  # so a pass of that work can actually finish
ZEN_LONG_LIVED=1       # for a worker trailing a live run
```

Relaunched with all three, verified in the live cmdline (`timeout 7200`).

### 4.5c ⛔ CORRECTION to §4.5: `ZEN_CHUNK_WALL_SEC=0` removes the CLAIM BOUND, not just concurrency

§4.5 recommended serial-per-worker as the CPU-efficient configuration. **That
recommendation was wrong for a large gap, and this is the correction.**

MEASURED: with 8 serial workers relaunched at 21:05Z and **zero** pass timeouts,
`distinct_done` sat at 2020 for **1 h 45 m**. The workers were demonstrably
encoding the whole time — `brsdr0`'s current cell showed **153.34 s of CPU over
2:33 elapsed**, ~100 % of a core — yet **zero passes had completed**.

**The mechanism:** `ZEN_CHUNK_WALL_SEC` bounds how much work a worker CLAIMS, and
setting it to 0 does not merely disable concurrency — it removes the bound, so the
worker claims the **entire remaining gap** and writes its ledger only at pass end.
With ~7,260 cells left at a measured ~13.8 s/cell mean (from the companion
instrument's zenrav1e β), one such pass would take **~28 hours**. It would never
complete, the 7200 s timeout would fire, and every unflushed cell — an estimated
**~3,600 cells / ~14 CPU-h** of completed work — would be discarded.

**So serial mode is unusable for a large run at any timeout.** The earlier §4.5
measurement that made it look good (12.6 cells/min at 9.52 CPU-s/cell) was taken
when the gap happened to be small enough that passes still closed; it does not
generalise, and neither did the recommendation drawn from it.

**The configuration that satisfies all three constraints** — bounded flushes, no
oversubscription thrash, and passes that finish:

```
ZEN_CHUNK_WALL_SEC=300   # BOUNDED claim -> regular flushes, bounded loss window
ZEN_PASS_TIMEOUT=7200    # a bounded chunk of slow cells still finishes
ZEN_LONG_LIVED=1         # worker trailing a live run
--cpuset-cpus <ONE core> # OVERSUBSCRIBE=2 then means 2 concurrent encodes, not 18
```

The last line is the trick that keeps §4.5's real finding (oversubscription thrashes
memory-bandwidth-bound AV1) while discarding its wrong remedy: the launcher hardcodes
`ZEN_CORE_OVERSUBSCRIBE=2`, so **shrinking the cpuset is how you bound concurrency**,
not disabling chunking. Eight 1-core workers give 8 cores of encoding at 2 concurrent
encodes each.

**Acceptance test, run rather than assumed — PASSED in 3 minutes.** The bar was a
flush within 20 minutes of relaunch. MEASURED: `distinct_done` 2080 → **2110 at
22:54:29Z, 3 minutes after launch**, against **1 h 45 m and zero** under the serial
configuration. That is the property §4.5's configuration silently lacked, and it is
now a gate rather than an assumption.

**Loss accounting, honestly.** The serial workers did flush once right at the end —
`distinct_done` read 2020 at 22:50 and **2080** at relaunch, so 60 cells landed. The
rest of their 1 h 45 m across 8 cores was discarded: at the measured ~13.8 s/cell
that is on the order of **3,500 cells / ~13 CPU-h**, which is the price of the wrong
recommendation in §4.5 and is counted against this lane, not the tooling.

### 4.5d ⛔ CORRECTION to §4.5's ATTRIBUTION: it is CPU PINNING, not oversubscription

§4.5 concluded that `ZEN_CORE_OVERSUBSCRIBE=2` "thrashes memory-bandwidth-bound
AV1". A third measured point falsifies that attribution. All three, same corpus,
same backend, `distinct_done` on both ends of the window:

| config | concurrent encodes | cpuset shape | cells/min | CPU-s/cell |
|---|---|---|--:|--:|
| 2 workers × **9-core** cpuset, oversub 2 | 18 on 9 cores | **wide** | 26.2 | 34.8 |
| 8 workers × **1-core** cpuset, oversub 2 | 16 on 8 cores | **narrow** | **21.0** | **22.9** |
| 8 workers × **1-core** cpuset, oversub 1 | 8 on 8 cores | narrow | 12.0 | 40.0 |

**Rows 1 and 2 have essentially the same oversubscribe ratio (2:1) and differ 1.5×
in cost.** What differs is cpuset *width*: with a 1-core cpuset each encode is
pinned and keeps its cache; with a 9-core cpuset the scheduler migrates 18 encodes
across 9 cores and they lose it. **Rows 2 and 3 have identical pinning and differ in
concurrency** — and 2-per-core *beats* 1-per-core by 1.75×, because the second cell
fills the first's serial and I/O phases.

**So oversubscription HELPS here; wide cpusets hurt.** My §4.5 remedy (serial /
`oversub=1`) was the worst of the three configurations on both axes, and it was
adopted on a measurement that confounded the two effects. The corrected rule:

> **Pin encodes with narrow per-worker cpusets, and keep oversubscription at 2.**
> Scale by adding 1-core workers, not by widening a worker's cpuset.

`ZEN_CORE_OVERSUBSCRIBE` is now forwardable by the launcher, but the measured
default (2) is the right one for this backend — the knob's value here was in being
able to *test* it, not in changing it.

### 4.8 FINAL G-RATE DECISION — de-scope step 1 FIRES and STAYS

The decisive measurement, on the configuration §4.5d established as best, after the
config churn had stopped: **1200 s window, 390 cells, `distinct_done` on both ends,
zero timeouts.**

| | |
|---|--:|
| rate | **19.5 cells/min** |
| cost | **24.6 CPU-s/cell** |
| full 9,280-cell grid | **63.5 CPU-h** |
| remaining envelope | 46.8 CPU-h |
| **verdict** | **OVER by 36 % — G-RATE FAILS** |

**Why this supersedes the §4.4 revert.** That revert rested on 9.52 CPU-s/cell from
a **429 s** window taken on the serial configuration — which §4.5c later proved was
silently claiming the entire gap and flushing nothing. The number was measured on a
broken config over a window too short to span a pass. This one is **2.8× longer, on
a verified-healthy config, with an acceptance-tested flush cadence**. It is the best
measurement this lane has and the decision follows it.

**De-scope step 1 fired** (29-q → 9-q, 2,880 cells), the shrink applied through
§4.4's sanctioned path with `manifest_PRE_DESCOPE_29q.json` preserved beside it.
Arithmetic at the measured rate:

| | cells | CPU-h |
|---|--:|--:|
| `brsdr` spent | 2,950 | 20.2 |
| `brsdr` remaining in the 9-q subset | ~1,965 | 13.4 |
| `brsdr` total | | **33.6** |
| + `brnat` | 7,488 | ~4.0 |
| **wave** | | **37.6 of 46.8 — FITS** |

**⚠ The shrink did NOT take effect until the workers were RESTARTED.** §4.4's
sanctioned path swaps `manifest.json`, but `ZEN_LONG_LIVED=1` pins the manifest
(the fetch is outside the pass loop — the same trap that idled the scorer for 3 h).
MEASURED: after the swap, both hosts still reported `manifest ready (4425398
bytes)`, the pre-shrink size. Restarting all 8 workers brought them to
`1373158 bytes`. **Any manifest change — de-scope, re-declaration, gap-fill —
requires restarting long-lived workers.** Bounded chunks (§4.5c) are what made that
restart cheap: ≤ 300 s of work per worker rather than the 1 h 45 m the unbounded
configuration would have thrown away.

**The ~2,035 already-encoded cells outside the 9-q subset are not waste.** They are
valid denser-q coverage on the images reached first, the gap-fill scores them from
the ledger regardless of the manifest, and they stay in the run. The wave therefore
delivers the registered 9-q grid **plus** partial 29-q coverage — which is strictly
more than the de-scope promised.

**What the de-scope costs, stated:** quality-space matching against A0R-svt is now
9-point rather than 29-point on the cells that remain. That still clears the BD-rate
owner's ≥ 4-point guard, which is why this was step 1 of the ladder. §6.4b's finding
that cost **rises** with q also means the cut points were the expensive ones, so the
saving is at least proportional.

### 4.6 Every short-window fleet rate here is LUMPY, and the envelope is a BRACKET

**Ledger parquets are written per PASS** (`pass-<worker>-<n>.parquet`), and a pass
claims a large batch of the gap **in serial mode too**. MEASURED: with all 8 serial
workers pinned at ~100 % CPU each, `distinct_done` moved **0 cells in 480 s** — not
a stall, just no pass boundary inside the window. Conversely the earlier 429 s
window caught 90 cells. **Short-window rates here are pass-lumpy, so any of them —
including this lane's 9.52 CPU-s/cell — is an estimate with a wide error bar, not a
measurement.** The four proxies that misled this lane in turn were blob counts,
ledger-file counts, the gap-fill DONE line, and short `distinct_done` windows.

**The independent anchor comes from the companion instrument**, which measures
`encode_ms` directly with no claim/chunk/ledger machinery in the path. Pricing the
`brsdr` grid from the complete S1a pass-1 fits (zenrav1e, 10-speed β sum
**118,967 ms/MP**, α sum 4,797 ms) against the budget corpus's 37.02 MP × 29 q × 32
images:

| source | `brsdr` grid | wave total vs 46.8 |
|---|--:|---|
| speed instrument, direct `encode_ms` | **36.7 CPU-h** (35.5 slope + 1.2 intercept) | 40.7 — fits, 13 % margin |
| fleet short-window, 9.52 CPU-s/cell | 24.5 CPU-h | 28.5 — fits, 39 % margin |

**Both fit; treat 24.5–36.7 CPU-h as a bracket rather than picking one.** The
instrument-derived figure is the better-founded of the two (direct measurement, no
fleet machinery) but it applies the LADDER corpus's pooled β to the BUDGET corpus,
and this same instrument measured β varying **24.3× with content** — so that step
carries exactly the error the headline result warns about. Neither number is
quotable to three figures.

**What the bracket decides, and it is decided:** serial-per-worker is the
CPU-efficient configuration at either end of it, so it is the right choice under the
uncertainty. The de-scope stays reverted because the full grid fits at both ends.

### 4.7 Monitor the LIVENESS signal, not the lumpy counter

The first progress monitor armed for `brsdr` alerted on `distinct_done` not moving
for 30 minutes. It fired — **as a false positive**. Under `ZEN_PASS_TIMEOUT=7200` a
legitimate pass can exceed 30 minutes before it flushes, so the stall bar was
shorter than the interval it was policing. A monitor that cries wolf on healthy
behaviour is its own failure mode: the next real stall arrives looking identical to
the last three false ones.

**What settled it was a direct liveness test**, and it is the one to monitor:
`zenmetrics jobexec` accumulating CPU proportional to elapsed time. MEASURED at the
moment of the false alarm: `brsdr1`'s encoder had **138.07 s of CPU over 2:18
elapsed** (~100 % of a core on a single cell) and `brsdr0` was 2 s into a fresh one,
with **zero** `TIMED OUT` lines since the relaunch. Compare the genuinely broken
state 40 minutes earlier: `zenfleet-worker` at **0.08 s** of CPU after 9 minutes.

The replacement monitor alerts on (a) any `TIMED OUT` in the last 30 min and (b) no
encoder burning CPU for 20 min, and reports progress when the counter does move.
Both are direct; neither depends on flush granularity.

**Also worth carrying: cell cost varies enormously within this arm.** A single cell
held one core for over 2 minutes while the arm's mean is ~9.5 s — consistent with
the companion instrument's measured 24.3× content spread and its 3.7×-69× speed-dial
spread. Any per-cell timeout or chunk sizing calibrated on a mean will be wrong for
this backend.

**The generalisable rule:** for a memory-bandwidth-bound codec, **scale workers
out, not concurrency up** — and never measure a fleet rate from blob counts, ledger
file counts, or the gap-fill's DONE line, all three of which lag or dedupe. Use
`zenfleet-ctl compact`'s `distinct_done` on both ends of a fixed window.

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
