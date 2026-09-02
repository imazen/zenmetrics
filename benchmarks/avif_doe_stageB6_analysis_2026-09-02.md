# AVIF knob DOE — Stage B, trigger B-6: what `ac_bias` and `sharpness` do at NATIVE size

**Date:** 2026-09-02 · **Design of record:**
[`avif_doe_plan_2026-09-01.md`](avif_doe_plan_2026-09-01.md) §15 (the B-6
declaration), §7.1 (analysis), §7.2 (triggers), §3.8 (the transfer gate), §12.4
(the degeneracy correction) · **Stage-A record:**
[`avif_doe_stageA_2026-09-02.md`](avif_doe_stageA_2026-09-02.md) §8 (the gate
that fired B-6), §7.1 (the QM × sharpness synergy) · **Lane:** analysis only —
this lane declared nothing, launched nothing, and touched no worker.

B-6 is Stage B's first user-approved wave. It re-runs, at **native size**, the
two arms Stage A proved were **not screenable at the 1024² budget**. This is
the completion record and the pre-registered read of it.

---

## 0. TL;DR

1. **The wave is COMPLETE and fully scored.** 25,056/25,056 encode cells,
   live-gap 0; **23,489/23,489 distinct bitstreams scored, 0 cells missing
   `ssim2`, 0 missing bytes.** The score run's ledger reads `declared 2112 /
   done 5907` — that is the known rework echo, not 3,795 extra cells; counted by
   distinct scored cell it is exactly 100%.
2. **The screening was MISCALIBRATED, not directionally wrong.** On every
   (speed, knob) cell where **both** legs carry an effect above ±0.5%, budget
   and native agree on **sign 1.000 of the time** — 6/6, 10/10, 11/11, 11/11,
   1/1. Stage A's two FAIL-T1 verdicts were largely a **construction defect in
   T1**, which counts refs by their *native* effect only and therefore scores
   sign agreement against **budget-side noise** (§4.2).
3. **`ac_bias` has essentially no native effect at all.** Across 12 (speed,
   level) cells the corpus median BD-rate spans **−0.03% to +0.39%**, and only
   **2 of 11** cropped references clear ±0.5% at speed 4 and **0 of 11** at
   speed 6. So `acb3` is **NOT-MEASURED at native, not FAIL** — there is no
   direction to transfer. Stage A's T1 = 0.25 was four coin flips.
4. **`shp3`'s failure is real but speed-4-specific.** Re-run on the definitive
   instrument it stays **FAIL-T1 at speed 4** (T1 **0.75** vs the 0.80 bar; Stage
   A read 0.62) and **PASSES at speed 6** (T1 **1.00**). Stage A could not see
   this: its gate ran at speed 4 only.
5. **Sharpness is expensive at native, monotone in level, and it gets WORSE as
   the preset gets faster** — `shp7` costs **+7.15% / +7.99% / +9.46%** of bits
   at s4 / s6 / s7. `shp1` is the exception: it moves the bitstream on ~90% of
   cells but costs **+0.00% to +0.23%** — free.
6. **Content spread is enormous and it is not a sign flip.** `shp7` at speed 6
   spans **+0.17% (plot) to +11.13% (ai-gen)** — **10.97 pp** across five
   families. **B-3 does not fire** for any B-6 knob: every class median above
   ±1% is a *cost*.
7. **Neither knob earns a place in the per-image knob set.** A perfect
   per-image oracle over all 8 levels buys a corpus median of **−0.23 / −0.09 /
   −0.01 pp** at s4 / s6 / s7; a *realistic* rule trained at speed 4 and applied
   at s6/s7 buys **−0.05 to −0.24 pp**. `ac_bias` is not even learnable (its
   per-image sign is stable across the three speeds on **6–15 of 31** images).
8. **Four integrity gates passed, none assumed**, including a new cross-run one:
   B-6's native encodes are **byte-identical to Stage A's AG native leg on
   576/576 shared cells** and to A1/A2/A0R on **3,705/3,705 passthrough** cells,
   while differing on **0/2,535 cropped** cells — exactly as the corpus design
   requires.
9. **Two corrections the record owes itself** (§7): plan §15.4's preset column
   is wrong — `s6` is preset **6** and `s7` is preset **7**, proven by
   928/928 byte identity — and the naive sweep's speeds **7, 8, 9 and 10 are
   byte-identical**, so 630 of its cells measure preset 7 three extra times.

---

## 1. Wave completion, and the score count that has to be read carefully

| run | grid | declared | live_done | failed-only | verdict |
|---|---|--:|--:|--:|---|
| `avifdoe-svt-b6-20260902` | 9 levels × 29 q × 32 images × speeds {4,6,7}, **native** | 25,056 | 25,056 | 0 | COMPLETE |
| `avifdoe-svt-b6-sf-cpu-20260902` | score, `ssim2,zensim` | 2,112 | 5,907 | 0 | COMPLETE |

**The score run's `done > declared` is not extra work and not a gap.** It is the
pre-sort-fix rework echo: score *jobs* are chunk-keyed, so a re-declaration
after the chunk sort changed produces new job identities over the same cells.
The number that matters is **distinct scored cells**, and it is complete:

```
cell rows from pairs tables: 25056
  distinct encode_sha:       23489      <- content-addressed dedup, 1,567 aliases
encode sizes:                23489
scored bitstreams:           23489      <- every one
rows missing bytes:          0
rows UNSCORED (no ssim2):    0
metrics_present:             25056 x "ssim2|zensim_features"   (uniform; 2-metric era)
```

The declared shape reproduces exactly: **27 strata × 928 = 25,056**, and every
one of the 27 strata carries exactly 928 cells (= 29 q × 32 images). Nothing is
ragged.

**One reference is short at one speed.** `6018.scale2320x3408` yields no BD-rate
at speed 4 — its rate does not respond to quality there, so the Pareto frontier
falls below the 4-point minimum. It is fine at speeds 6 and 7. Every other
reference is usable at every speed, which is itself a finding: see §7.

---

## 2. Method, and the four gates

Instruments, all canonical owners — **nothing statistical is re-implemented**:

| step | owner |
|---|---|
| score blobs + ledger → tidy table | `scripts/jobsys/avifdoe_harvest.py` (extended here, §9) |
| BD-rate, frontier, median bootstrap CI, content-class map | `scripts/jobsys/avifdoe_stagea_analyze.py` (imported) |
| exact two-sided binomial | `scripts/jobsys/avifdoe_stagea_gates.py` (imported) |
| SROCC | **`zenstats`** via `zensim/scripts/lib/zen_stats.py` |
| this wave's driver | `scripts/jobsys/avifdoe_stageb6_analyze.py` (new; imports the three above) |

**Sign convention, inherited unchanged: NEGATIVE BD-rate = the arm needs FEWER
bits at matched quality = the arm WINS.**

**The control is the native `deviations = 0` arm at the same (image, speed)**,
per §7.1(2) — B-6's own in-run control, on the same 29-q ladder. Quality
response is `ssim2`, the only corpus-wide scalar (§8, item 1).

**Categorical discipline (§7.1(6), H-14): `sharpness` is a FACTOR.** Levels are
reported as levels. No ordinal trend is fitted and no slope is quoted anywhere
in this document. Where the level ordering is described in prose it is a
description of four measured numbers, not a model.

### 2.1 Gate 1 — BD-rate parity

Identical to `zenavif/scripts/rd_gap/bd_arm.py` on 200 random ladders,
**max |Δ| = 0.0 exactly**. Same gate, same result, as Stage A §5.2.

### 2.2 Gate 2 — cross-run byte identity (new, and it is what licenses §4)

19 of the 32 references are **native passthroughs** (`crop_sha == source_sha`),
so a *budget*-run encode of one is literally the **same bitstream** as B-6's
native encode. AG encoded the native corpus, so it must match on **both**
classes. Anything else would mean the legs differ in *configuration* rather than
in *pixels*, and every comparison in §4 would be confounded.

| Stage-A run | reference class | shared cells | byte-identical | expected | verdict |
|---|---|--:|--:|--:|---|
| `a0r` | passthrough | 1,653 | 1,653 | 1.0 | PASS |
| `a0r` | cropped | 1,131 | 0 | 0.0 | PASS |
| `a1` | passthrough | 1,197 | 1,197 | 1.0 | PASS |
| `a1` | cropped | 819 | 0 | 0.0 | PASS |
| `a2` | passthrough | 855 | 855 | 1.0 | PASS |
| `a2` | cropped | 585 | 0 | 0.0 | PASS |
| `ag` | passthrough | 342 | 342 | 1.0 | PASS |
| `ag` | **cropped** | **234** | **234** | 1.0 | PASS |

The last row is the strong one: **AG's native leg and B-6 are the same encoder,
the same configuration and the same pixels on every shared cell**, so B-6 is a
strict 29-q / 3-speed upgrade of the 3-q / 1-speed instrument the transfer gate
originally ran on.

### 2.3 Gate 3 — the control is the same instrument two ways

The brief's named native-defaults control is the drained naive `avifsub-svt-enc`
preset × q sweep. It is a **different run, declared a day earlier, by a different
plan**, so it is an independent check on B-6's own control arm. Differencing
every B-6 arm against it instead:

| | |
|---|---|
| (speed, knob) cells compared | **24** |
| cells whose median BD-rate is **identical to 4 dp** | **21** |
| max \|Δ\| over all 24 | **0.0015 pp** (three speed-7 cells) |

Stage A's equivalent check (A0R vs the in-run controls) gave median \|Δ\| 0.386 pp
and max 1.574 pp. Here it is **0.000**, because the two controls are not merely
equivalent — they are the **same bitstreams**: `sN-svt-420` is byte-identical to
naive speed *N* on **928/928** shared cells at each of N = 4, 6, 7, and
byte-**distinct** from every other naive speed. The three residual cells come
from the naive sweep's own small coverage gaps (937–942 of 960 per speed), not
from a difference in configuration.

### 2.4 Gate 4 — the passthrough null, restated for this wave

For the budget-vs-native comparison in §4 the 19 passthroughs are an identity
pair, so their residual must be exactly zero. **Q-matched leg: 0 violations of
19 on all 8 cells.** (The full-29q leg shows 19/19 by construction — the two
legs use different ladders over the same bitstreams. That is a ladder-density
difference, not a pipeline fault, and it is why the **q-matched leg is the
primary read** and the 29-q leg a secondary one.)

---

## 3. Native full-ladder main effects

Median per-image BD-rate over the corpus, vs the native default at the same
(image, speed). CI is a 10,000-resample percentile bootstrap over images
(seeded). `wins` counts images needing fewer bits.

**`sharpness` — a factor, four levels**

| level | s4 median | 95% CI | IQR | wins/n | s6 median | 95% CI | wins/n | s7 median | 95% CI | wins/n |
|---|--:|---|--:|--:|--:|---|--:|--:|---|--:|
| `shp1` | +0.00% | [+0.00, +0.04] | 0.09 | 8/31 | **+0.14%** | [+0.04, +0.21] | 6/32 | **+0.23%** | [+0.15, +0.47] | 3/32 |
| `shp3` | **+1.20%** | [+0.88, +1.72] | 2.35 | 7/31 | **+1.49%** | [+1.10, +2.01] | 2/32 | **+2.75%** | [+2.12, +3.10] | 2/32 |
| `shp5` | **+5.57%** | [+3.95, +8.00] | 5.80 | 2/31 | **+5.67%** | [+4.28, +7.91] | 3/32 | **+7.52%** | [+7.01, +8.68] | 2/32 |
| `shp7` | **+7.15%** | [+5.60, +10.02] | 6.66 | 1/31 | **+7.99%** | [+6.33, +10.23] | 4/32 | **+9.46%** | [+8.60, +11.38] | 2/32 |

**`ac_bias` — the range A1 never reached**

| level | s4 median | 95% CI | IQR | wins/n | s6 median | 95% CI | wins/n | s7 median | 95% CI | wins/n |
|---|--:|---|--:|--:|--:|---|--:|--:|---|--:|
| `acb1` | +0.04% | [−0.08, +0.10] | 0.33 | 15/31 | **−0.03%** | [−0.05, −0.01] | 22/32 | −0.00% | [−0.01, +0.01] | 18/32 |
| `acb3` | +0.05% | [−0.16, +0.30] | 0.48 | 14/31 | −0.03% | [−0.06, +0.01] | 20/32 | +0.01% | [−0.01, +0.02] | 11/32 |
| `acb5` | +0.11% | [−0.04, +0.44] | 0.88 | 12/31 | −0.00% | [−0.04, +0.04] | 17/32 | +0.01% | [−0.01, +0.03] | 12/32 |
| **`acb8`** | **+0.39%** | [+0.17, +0.90] | 1.20 | 9/31 | **+0.11%** | [+0.01, +0.22] | 11/32 | +0.02% | [−0.01, +0.04] | 13/32 |

**Bold = the bootstrap CI of the median excludes zero.**

### 3.1 What the tables say

- **Sharpness is a pure bit cost at native, at every level ≥ 3 and every
  preset, and the cost RISES as the preset gets faster** (s4 → s7: +7.15% →
  +9.46% at level 7). It wins on 1–4 images of 32.
- **`shp1` is free.** Its BD-rate median is +0.00% at speed 4 and its whole
  effect is under a quarter of a percent everywhere. It is not inert — it
  changes the bitstream on **89.6%** of cells — it just changes the *size* by a
  median of **+0.000%** (p95 +0.01%). A knob that moves bytes without moving
  bytes: the level exists, it does nothing measurable to rate.
- **`ac_bias` is, at native, effectively a null knob.** Its entire 12-cell
  median range is **−0.03% to +0.39%**, and the three cells whose CI excludes
  zero do so at magnitudes (−0.03%, +0.11%, +0.39%) far below any decision bar.
  At speed 7 it is dead outright: every level median is within ±0.02%.
- **`acb8` — the unclamped H-10-class level — is genuinely live**, confirming
  plan plan §15.4's single-cell probe at corpus scale. It is byte-identical to the control
  on only **6.7%** of cells and moves size by a median **+0.559%** (p95
  **+5.67%**). It is also **the only `ac_bias` level with a defensible effect
  and it is a LOSS** (+0.39% bits at s4, +0.99% on `ai-gen`). *All 2,784 `acb8`
  cells are flagged as exercising an argument `SvtParams::clamped()` does not
  clamp.*

**Per-level byte movement, all 2,784 cells per level** (the liveness check —
none of these are the §3 inert pair):

| level | byte-identical to control | median Δbytes | p5 … p95 |
|---|--:|--:|---|
| `acb1` | 11.46% | +0.037% | −0.17 … +1.00% |
| `acb3` | 8.73% | +0.182% | −0.18 … +2.53% |
| `acb5` | 7.79% | +0.330% | −0.21 … +3.93% |
| `acb8` | 6.68% | +0.559% | −0.24 … +5.67% |
| `shp1` | 10.42% | +0.000% | +0.00 … +0.01% |
| `shp3` | 1.58% | +4.972% | +0.00 … +17.87% |
| `shp5` | 1.58% | +9.992% | +0.00 … +39.12% |
| `shp7` | 1.58% | +11.256% | +0.00 … +46.02% |

### 3.2 Per content class — huge spread, no sign flip

Median BD-rate by coarse family (`photo` 7, `plot` 6, `screenshot` 5, `ai-gen`
9, `scan` 5). **Class n is 5–9; these are directional evidence, not estimates**
(Stage A §5.4). `*` = bootstrap CI excludes zero.

**`sharpness`**

| speed | level | photo | plot | screenshot | ai-gen | scan |
|--:|---|--:|--:|--:|--:|--:|
| 4 | `shp1` | +0.02 | +0.00 | +0.00 | +0.09 | −0.01 |
| 4 | `shp3` | +1.15\* | +2.06 | **−0.35** | +1.11\* | +1.74\* |
| 4 | `shp5` | +4.64\* | +8.64 | +0.91 | +8.00\* | +4.92\* |
| 4 | `shp7` | +6.20\* | +10.44\* | +2.71\* | +10.36\* | +6.00\* |
| 6 | `shp3` | +1.46\* | +0.14 | +0.91\* | +2.48\* | +0.99\* |
| 6 | `shp5` | +5.73\* | +0.15 | +3.86\* | +8.41\* | +3.45 |
| 6 | `shp7` | +7.77\* | **+0.17** | +6.94\* | **+11.13\*** | +4.49 |
| 7 | `shp3` | +2.01\* | +0.73 | +2.91\* | +2.94\* | +7.16\* |
| 7 | `shp5` | +7.41\* | +2.63 | +7.43\* | +9.85\* | +12.36\* |
| 7 | `shp7` | +9.71\* | +2.91 | +8.36\* | +13.05\* | **+13.74\*** |

**`ac_bias`** — the whole table lies in **[−0.21, +0.99]**; the largest cell is
`acb8` on `ai-gen` at s4 (+0.99%, CI includes zero). It is reproduced in
`b6_main_effects_by_class.tsv` and not tabulated here, because no cell in it
reaches a magnitude worth a row.

**The spread is the finding, and it is not a sign flip.** `shp7` at s6 runs
**+0.17% on plots and +11.13% on ai-gen — 10.97 pp**. At s7 it is +2.91% on
plots and +13.74% on scans — 10.83 pp. **B-3 does not fire** for any B-6 knob:
checked mechanically, no knob has opposite-signed class medians with both
|median| ≥ 1%. Every class median past ±1% is a cost.

**Class medians hide a bimodal `plot` family.** All six plots are 1.05 MP, and
`shp7` splits them cleanly at every speed:

| speed | 7004 | 7058 | 7050 | 7052 | 7076 | 7042 |
|--:|--:|--:|--:|--:|--:|--:|
| 4 | **−3.04** | +3.17 | +15.04 | +10.14 | +10.74 | +12.01 |
| 6 | **−1.46** | −0.37 | +0.09 | +0.24 | +7.27 | +13.44 |
| 7 | **−7.41** | −5.02 | +0.54 | +5.28 | +7.05 | +8.72 |

The s6 `plot` median of +0.17% is the midpoint of a family spanning −1.46% to
+13.44%. **This is a per-IMAGE property, not a class property** — which is
exactly the case a per-image picker exists to exploit, and §6 prices it.

---

## 4. The screening-failure story, quantified — the wave's registered purpose

### 4.1 The test, and why it is a better one than Stage A could run

B-6 fired because `acb3` and `shp3` failed **T1** of the cross-size gate. Stage
A's gate compared **AG native at 3 q points** against **A1 budget at the same 3
points**, on 11 cropped references, at **speed 4 only** (Stage A §8). Two things change here:

- the native leg becomes B-6's **29-q ladder at speeds 4 AND 6**, and
- the budget leg becomes the **full 9-q screening ladder** the arms actually
  published — which is the honest test of *"did the screening's own numbers
  transfer"*.

All **9** budget q points {5,15,25,35,45,60,76,90,96} are inside B-6's 29-q
ladder, so the native leg is **restricted to exactly those 9** for the primary
read. Both legs then integrate the same quality span and the residual is a
**size** effect, not a ladder-density effect. Per plan §12.4 the test runs on the **13
cropped references** only (11 usable). Bars are unchanged: T1 ≥ 0.80, T2 ≥ 0.70,
T3 ≤ 1.0% with no systematic sign.

### 4.2 The gate re-run

| speed | knob | n crop | n eff | T1 | T2 SROCC | T3 med \|resid\| | T3 p | passthrough violations | **B-6 verdict** | *Stage-A verdict* |
|--:|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 4 | `acb1` | 11 | **0** | — | 0.818 | 0.140 | 0.549 | 0/19 | **NOT-MEASURED** (0 of 11 refs move ≥0.5%) | *NOT-MEASURED* |
| 4 | `acb3` | 11 | **2** | — | 0.518 | 0.155 | 0.549 | 0/19 | **NOT-MEASURED** (2 of 11, need 3) | ***FAIL-T1*** |
| 4 | `shp3` | 11 | 8 | **0.750** | 0.682 | 0.466 | 0.549 | 0/19 | **FAIL-T1** | ***FAIL-T1*** |
| 4 | `shp7` | 11 | 11 | **1.000** | 0.818 | 0.927 | 0.549 | 0/19 | **PASS** | *PARTIAL* |
| 6 | `acb1` | 11 | **0** | — | 0.164 | 0.050 | 0.065 | 0/19 | **NOT-MEASURED** | *not run (s4 only)* |
| 6 | `acb3` | 11 | **0** | — | 0.455 | 0.086 | 0.549 | 0/19 | **NOT-MEASURED** | *not run* |
| 6 | `shp3` | 11 | 10 | **1.000** | 0.709 | 0.771 | 0.065 | 0/19 | **PASS** | *not run* |
| 6 | `shp7` | 11 | 11 | **1.000** | 0.846 | 1.972 | 0.227 | 0/19 | **PARTIAL** | *not run* |

### 4.3 The answer, plainly

**Reduced-size screening was miscalibrated for these knobs. It was not
directionally wrong.** Three measurements say so:

**(a) When both legs carry an effect, the direction always agrees.** T1's own
denominator counts a reference if its *native* effect clears ±0.5% — it asks
nothing of the budget leg. Require **both**:

| speed | knob | n | sign agree, all 11 | sign agree, \|native\|≥0.5 (**T1 as written**) | sign agree, **BOTH ≥0.5** | n both | BOTH ≥0.25 |
|--:|---|--:|--:|--:|--:|--:|--:|
| 4 | `acb1` | 11 | 0.909 | — (n=0) | — | 0 | 1.000 (n=1) |
| 4 | `acb3` | 11 | 0.818 | 0.500 | **1.000** | 1 | 1.000 (n=2) |
| 4 | `shp3` | 11 | 0.636 | **0.750** | **1.000** | 6 | 0.857 (n=7) |
| 4 | `shp7` | 11 | 1.000 | 1.000 | **1.000** | 11 | 1.000 |
| 6 | `acb1` | 11 | 0.273 | — (n=0) | — | 0 | — (n=0) |
| 6 | `acb3` | 11 | 0.818 | — (n=0) | — | 0 | — (n=0) |
| 6 | `shp3` | 11 | 1.000 | 1.000 | **1.000** | 10 | 1.000 |
| 6 | `shp7` | 11 | 1.000 | 1.000 | **1.000** | 11 | 1.000 |

**Every evaluable cell is 1.000.** The FAIL-T1 verdicts are produced entirely by
references where the *budget* leg sits inside the noise band, so its sign is a
coin flip that T1 nonetheless scores. This is the Stage A §8.2 construction problem
Stage A named — *"T1's denominator is the effect itself"* — measured on the
definitive instrument, and it is worse than Stage A realised: the flaw is not
only that a vanishing effect reads as NOT-MEASURED, it is that a **surviving**
effect gets graded against noise on the other leg.

**(b) `acb3` has no direction to transfer.** At speed 4 the 11 cropped
references span **−0.56% to +0.99%** at native and **−0.82% to +0.93%** at
budget — **82% of native and 64% of budget values sit inside ±0.5%**. At speed 6
it is **100% and 100%**. Stage A's T1 = 0.25 was **four** references that
cleared the floor on a **3-point** ladder. On a 29-point ladder over the same
references, **two** clear it at s4 and **none** at s6. `acb3` is NOT-MEASURED at
native; calling it FAIL-T1 was a category error the 3-q instrument invited.

**(c) `shp3`'s failure is real, and it is speed-4-specific.** T1 rises 0.62 →
**0.75** on the better instrument but stays under the 0.80 bar, and the four
disagreeing references are all cases where one leg reads ≈0 (e.g. `6602`:
budget **−0.07%**, native **+0.67%**; `8446` screenshot: budget **+0.07%**,
native **−0.40%**). At speed 6 the same knob reads **T1 = 1.000, T2 = 0.709,
T3 = 0.771% — PASS**. Stage A could not have found this: Stage A §4.2's collapsed speed
axis left its gate at speed 4 only.

### 4.4 Magnitude — the honest version

The budget leg does **not** demonstrably overstate on a per-image basis:

| speed | knob | \|budget\|>\|native\| | binomial p | median per-image ratio |
|--:|---|--:|--:|--:|
| 4 | `acb3` | 3/5 | 1.000 | 2.000 |
| 4 | `shp3` | 5/8 | 0.727 | 1.016 |
| 4 | `shp7` | 4/11 | 0.549 | 0.982 |
| 6 | `shp3` | **9/11** | **0.065** | **1.625** |
| 6 | `shp7` | 8/11 | 0.227 | 1.266 |
| | **pooled** | **29/46** | **0.104** | |

So: **at speed 6 the budget leg overstates sharpness by ~1.3–1.6× on 8–9 of 11
references (suggestive, p = 0.065 and 0.227, n = 11); at speed 4 it does not
(median ratios 1.02 and 0.98); pooled it is not established (p = 0.104).**

*A ratio of medians would have told a tidier and less true story* — median
|budget| / median |native| is 1.02–1.63 across these cells, which is **not** the
median of the per-image ratios. The per-image test is the one reported.

**The error does not scale with the size jump.** Over the 41 effect-bearing
(shp3, shp7) references spanning 3.7–16.0 MP native against a fixed 1024² crop:
SROCC(native MP, |budget|/|native|) = **0.068**; SROCC(native MP, residual) =
**0.157**; median ratio by MP tercile **0.94 / 1.33 / 1.06**. Whatever the
miscalibration is, "a bigger crop-to-native jump makes it worse" is **not** it.

### 4.5 What this licenses

- **Screening `sharpness` at reduced size is safe for DIRECTION at both
  presets, and for MAGNITUDE at speed 4 only.** Its speed-6 magnitude should not
  be quoted from a 1024² read.
- **Screening `ac_bias` at reduced size is a waste of cells**, not because the
  budget lies but because **there is nothing to screen** — the knob has no
  native effect at any level it can reach.
- **T1 should require BOTH legs to clear the effect floor.** As written it
  grades sign agreement against noise, and that alone produced one of B-6's two
  arms. This is a defect in the gate, not in the data, and it is registered here
  rather than fixed silently — changing a pre-registered bar after seeing
  results is the coordinator's call, not this lane's.

---

## 5. QM × sharpness at native — what CANNOT be concluded, stated first

**B-6 carries no QM axis.** Its 9 levels are the shared default, `acb`
{1,3,5,8} and `shp` {1,3,5,7}. There is **no `qml*` cell in this wave**, no
`(qml, shp)` pair, and no cross-wave join that would manufacture one: Stage A's
pair cells are 1024² crops of 13 references whose native twins are different
pixels (§2.2 proves it, 0/2,535 byte-identical). **So the synergy is
NOT-MEASURED at native — neither confirmed nor refuted.** No number in this
section is a native reading of the interaction.

**What B-6 does re-measure is one half of the additive baseline.** Stage A's
residual is `observed_pair − (main_qml + main_shp)`, with **both** main effects
at 1024². B-6 gives `main_shp` at native on matched references:

| speed | level | population | median budget | median native | additive-baseline shift |
|--:|---|---|--:|--:|--:|
| 4 | `shp3` | 11 cropped | +1.231% | +0.812% | **−0.419 pp** |
| 4 | `shp3` | 30 matched refs | +0.914% | +0.779% | −0.135 pp |
| 4 | `shp7` | 11 cropped | +6.132% | +5.205% | **−0.927 pp** |
| 4 | `shp7` | 30 matched refs | +5.468% | +5.269% | −0.198 pp |
| 6 | `shp3` | 11 cropped | +2.014% | +1.234% | **−0.780 pp** |
| 6 | `shp3` | 30 matched refs | +1.732% | +1.428% | −0.304 pp |
| 6 | `shp7` | 11 cropped | +8.942% | +7.085% | **−1.857 pp** |
| 6 | `shp7` | 30 matched refs | +7.418% | +7.024% | −0.393 pp |

**Read the "30 matched refs" rows for anything corpus-level**: 19 of Stage A's
32 references are passthroughs whose budget and native legs are the *same
encode*, so they contribute a shift of exactly zero and the cropped-only column
overstates the corpus effect by ~4×.

**The one conclusion available.** Stage A's `(qml1.2.10, shp7)` and
`(qml1.4.10, shp7)` residuals sit at **−5.2 to −5.5%** on 87–90% of images. The
sharpness half of their additive baseline moves by **−0.39 pp** at corpus level
going to native. **A −0.4 pp shift does not overturn a −5.4 pp residual**, so
size is not a plausible explanation for the synergy — but that is an argument
about one input, not a measurement of the output. **Confirming it needs a
`(qml × shp)` grid at native, which no declared wave contains.**

---

## 6. Tuning-model verdict — neither knob earns a place

The ask is whether `ac_bias` and `sharpness` belong in a per-image knob set at
native, at what levels, for which content, at what stakes. The stakes are priced
three ways, from most generous to most realistic.

**(a) The oracle** — perfect per-image level selection, an unachievable upper
bound (a level "wins" only if BD-rate < 0):

| speed | family | images with any win | median oracle gain | mean oracle gain | best single image |
|--:|---|--:|--:|--:|---|
| 4 | `ac_bias` | 21/31 | −0.160 pp | −0.496 pp | `7052` −2.96% |
| 4 | `sharpness` | 14/31 | 0.000 pp | −0.262 pp | `7004` −3.04% |
| 4 | **both** | 23/31 | **−0.234 pp** | −0.656 pp | `7004` −3.04% |
| 6 | **both** | 29/32 | **−0.089 pp** | −0.178 pp | `7004` −1.46% |
| 7 | **both** | 23/32 | **−0.014 pp** | −0.437 pp | `7004` −7.41% |

**(b) The win rates are noise.** Pooled over all 95 (speed, image) cells:

| level | wins | win rate | median gain *when it wins* | wins > 0.5% | wins > 1.0% |
|---|--:|--:|--:|--:|--:|
| `acb1` | 55/95 | 0.579 | **−0.032%** | 4/95 | 2/95 |
| `acb3` | 45/95 | 0.474 | −0.074% | 4/95 | 2/95 |
| `acb5` | 41/95 | 0.432 | −0.080% | 5/95 | 1/95 |
| `acb8` | 33/95 | 0.347 | −0.101% | 7/95 | 2/95 |
| `shp1` | 17/95 | 0.179 | −0.125% | 0/95 | 0/95 |
| `shp3` | 11/95 | 0.116 | −0.779% | 8/95 | 5/95 |
| `shp5` | 7/95 | 0.074 | −1.325% | 4/95 | 4/95 |
| `shp7` | 7/95 | 0.074 | −1.458% | 4/95 | 4/95 |

`acb1` "wins" on 58% of cells — by a median of **0.032%**. That is a coin flip
with a rounding error attached, not a lever.

**(c) Learnability, and the realistic rule.** A picker must recognise the win
from something stable. Is a knob's per-image effect stable across the preset
axis?

| level | SROCC s4~s6 | s6~s7 | s4~s7 | sign stable across all 3 |
|---|--:|--:|--:|--:|
| `acb1` | 0.196 | 0.253 | 0.128 | **6/31** |
| `acb3` | 0.172 | 0.205 | 0.180 | 9/31 |
| `acb5` | 0.296 | 0.042 | 0.058 | 11/31 |
| `acb8` | 0.596 | 0.270 | 0.003 | 15/31 |
| `shp1` | 0.476 | 0.782 | 0.375 | 16/31 |
| `shp3` | 0.503 | 0.545 | 0.267 | **26/31** |
| `shp5` | 0.491 | 0.675 | 0.147 | **29/31** |
| `shp7` | 0.595 | 0.700 | 0.242 | **29/31** |

**`ac_bias` is not learnable** — its per-image sign survives all three presets
on 6–15 of 31 images, i.e. at or barely above chance. **`sharpness` is
learnable** (26–29 of 31), but what is learnable about it is *"this costs
bits"*. Only **one image of 32** (`7004.scale1024x1024`, a plot) has `shp7`
winning at all three speeds; one more wins at two, one at one, and 28 never win.

A concrete achievable rule — *enable the level only where it won at speed 4,
apply at s6/s7*:

| level | target | fires on | median BD when fired | corpus median | mean corpus gain |
|---|--:|--:|--:|--:|--:|
| `shp3` | s6 | 7/31 | +0.270% | +1.513% | **−0.049 pp** |
| `shp3` | s7 | 7/31 | +1.756% | +2.656% | **−0.142 pp** |
| `shp5` | s6 | 2/31 | +1.672% | +5.726% | −0.043 pp |
| `shp5` | s7 | 2/31 | +0.617% | +7.487% | **−0.204 pp** |
| `shp7` | s6 | 1/31 | −1.458% | +8.218% | −0.047 pp |
| `shp7` | s7 | 1/31 | **−7.406%** | +9.250% | **−0.239 pp** |

### 6.1 Verdict

| knob | levels | verdict | stake |
|---|---|---|---|
| **`ac_bias`** | 1.0, 3.0, 5.0, 8.0 | **NO — drop from the per-image knob set at native.** Effect is inside ±0.4% at every level and speed, is *not* learnable across presets, and its only level with a defensible effect (`8.0`) is a **loss**. | oracle −0.16 pp @s4, −0.01 pp @s7 |
| **`sharpness`** | 1 | **NO — free but pointless.** Moves the bitstream on 90% of cells, moves its size by +0.000%. | 0 |
| **`sharpness`** | 3, 5, 7 | **NO as a tuning lever; YES as a default-off flag.** Costs +1.2% to +9.5% of bits corpus-wide and up to **+18.7%** on a single image, wins on 7–11 of 95 cells. | realistic rule buys −0.05 to −0.24 pp against a +9.5% downside if enabled blind |

**The one lead worth naming, and it is n = 1–2.** `7004.scale1024x1024` and
`7058.scale1024x1024` — both `plot`, both 1.05 MP — are where sharpness pays,
and it pays *large*: −7.41% and −5.02% for `shp7` at speed 7. Whatever
distinguishes them from the other four plots (which lose +0.54% to +8.72% on the
same knob at the same speed and size) is the only thing in this wave that could
justify a per-image sharpness decision. **Two images is a lead, not a rule**, and
this wave cannot say what the distinguishing feature is.

**The wins do not follow size, and the appearance that they do is an artefact.**
32 of the 36 wins ≥0.5% land on passthrough references, and every win ≥1.0% is
on an image ≤1.30 MP — but the corpus's passthroughs *are* its small images
(9 of 32 refs are ≤1.1 MP), and the rank correlation between native size and
BD-rate is weak throughout (SROCC 0.00–0.43, median ≈0.15). Sharpness is
directionally cheaper on small images at the fast presets (s7 `shp7`: +7.05% at
≤1.1 MP vs +10.56% above) — worth a note, not a rule.

---

## 7. Corrections to the record

### 7.1 Stage A's `6006` / `6018` exclusion is CROP-specific and does not carry

Stage A §5.3 excluded both references from BD-rate: their 1024² crops landed on
near-blank regions of scanned patent pages, so rate did not respond to quality
and `ssim2` saturated at 100.0. **At native size both are ordinary images.**
`6006.scale2320x3408` yields BD-rates at **all three speeds** — and is in fact
the *largest* effect in the wave (`shp7` at s7: **+18.72%**). `6018` yields
BD-rates at speeds 6 and 7 and only degenerates at speed 4.

**So B-6 runs on 31–32 references, not 30.** Any future native wave should
re-test degeneracy at native rather than inheriting the budget exclusion list.

### 7.2 Plan §15.4's preset column is wrong

plan §15.4's cost table maps `s4`→preset 4, **`s6`→preset 7, `s7`→preset 9**.
Measured by byte identity against the naive preset × q sweep, the mapping is
**literal**: `s4`↔4, `s6`↔**6**, `s7`↔**7**, each **928/928 byte-identical** and
**0/928** against every other naive speed. The CPU-cost measurements in plan §15.4
are unaffected (they were taken on the cells, whatever they are named); only the
preset labels are wrong.

### 7.3 The naive sweep's speeds 7, 8, 9 and 10 are the same encoder

The naive `avifsub-svt-enc-20260901` sweep declares speeds 1–10. Speeds **7, 8,
9 and 10 are byte-identical to one another on every shared cell** (and 1–6 are
mutually distinct). The preset saturates at 7. Its speed-8/9/10 strata — 630
cells — measure preset 7 three more times. This is the same *shape* as Stage A
Stage A §3's inert-knob finding (a configuration axis plumbed to the fingerprint but not
to the bitstream) but on the **speed axis**, and it is worth a look from the port
program: whether presets above 7 are meant to exist in `zenavif`'s speed mapping
at all. **Not filed as an issue here** — that is the port lane's call, and this
lane opens no issues.

---

## 8. Limitations — what this wave cannot tell you

1. **One quality response.** `ssim2` is the only corpus-wide scalar. `zensim` is
   emitted as a 720-wide **feature vector** with **no scalar** (`kind:"feature"`,
   `regime:"v2-ab"`), so it is not a second opinion without running a model over
   it, and butteraugli was dropped from this run by standing directive. **This
   matters more here than in Stage A**: `sharpness` is a *perceptual sharpening*
   control, so "it costs bits at matched ssim2" and "it is not worth enabling"
   are different claims, and only the first is measured. Whether sharpness buys
   anything a sharpness-blind metric cannot see is **NOT MEASURED** — not zero,
   not disproven, **unmeasured**.
2. **One backend.** svt-rs only. A3 was never declared; no aom-rs statement is
   derivable from this wave.
3. **No speed axis.** `encode_ms` is still not persisted by the fleet path, so
   no `ms/MP` and no `α + β·pixels` fit appears here, with or without an
   intercept. B-4 remains NOT EVALUABLE.
4. **No QM axis, so no native synergy.** §5.
5. **Three presets, and the slow end is untested.** Speeds 4, 6, 7 (presets 4,
   6, 7). Presets 0–3, where SG restoration / Wiener / filter-intra turn on, are
   absent — and the sharpness cost is *rising* toward the fast end, so
   extrapolating it downward is unwarranted.
6. **Class n is 5–9, and one class is bimodal.** Every per-class median is
   directional evidence. `plot` in particular is two populations (§3.2).
7. **The budget-vs-native comparison rests on 11 references.** plan §12.4's
   degeneracy is structural: 19 of 32 references have no size contrast at all.
   Every T1/T2/T3 number in §4.2 has n = 11, and n = 11 is where a binomial
   p-value of 0.065 lives.
8. **`6018` contributes no speed-4 row**, so speed-4 medians are n = 31 while
   speeds 6 and 7 are n = 32.
9. **No Tower mirror this session.** `/mnt/tower` returned a stale NFS handle;
   outputs are on `/mnt/v` and the LAN store (§9). Recorded as unavailable, not
   as done.

---

## 9. Outputs, and the two owners extended

**Analysis outputs** — local `/mnt/v/output/zensim-avifdoe-b6/`, LAN store
`s3://zentrain/analysis/avif-doe-stageB6-2026-09-02/` (20 objects, 1.77 MB).
Nothing is in git (>30 KB rule); the file list with sizes and sha256 is
[`avif_doe_stageB6_analysis_2026-09-02.pointer.md`](avif_doe_stageB6_analysis_2026-09-02.pointer.md).

**Raw inputs** stay in their canonical home: encode bitstreams
`s3://zentrain/jobs/avifdoe-svt-b6-20260902/blobs/` (23,489 objects), score
blobs `s3://zentrain/jobs/avifdoe-svt-b6-sf-cpu-20260902/blobs/` (5,907
NDJSON), native corpus `s3://codec-corpus/avif-subsample-2026-09-01/`.

**Regeneration:**

```sh
zenfleet-ctl pairs --ledger s3://zentrain/jobs/avifdoe-svt-b6-20260902/ledger/ \
    --refs-prefix s3://codec-corpus/avif-subsample-2026-09-01/ \
    --blobs-prefix s3://zentrain/jobs/avifdoe-svt-b6-20260902/blobs/ --out b6_pairs
avifdoe_harvest.py --score-dir <b6 score blobs> --sizes <b6 blob listing> \
    --pairs b6=b6_pairs.tsv --out b6_scored.parquet
avifdoe_stageb6_analyze.py --b6-scored b6_scored.parquet \
    --stagea-scored /mnt/v/output/zensim-avifdoe/doe_scored_2026-09-02.parquet \
    --naive-scored naive_scored.parquet \
    --crop-manifest /mnt/v/output/avif-doe-1024-2026-09-01/crop_manifest_2026-09-01.tsv \
    --native-dims native_dims.tsv --outdir stageb6 \
    --parity-check ~/work/zen/zenavif/scripts/rd_gap/bd_arm.py
```

**Two canonical owners were EXTENDED rather than forked** (per the
no-duplicate-implementations rule):

- **`avifdoe_harvest.py`** now accepts the naive sweep's knob-tuple shape
  (`{"backend":"svt-rs","speed":N}`, no `cell`, no chroma) by synthesizing a
  DOE-vocabulary control label. The synthesis **asserts chroma**, so it is
  backed by the 928/928 byte-identity measurement in §2.3 and the comment says
  so — a future backend whose naive default is not 4:2:0 will be caught by
  re-running that check, which is named in the code.
- **`avifdoe_stagea_analyze.py`**'s control-source label was hardcoded
  `"in-run-9q"`. B-6's in-run control is **29-q**, so the label would have
  misreported the instrument. It now carries the actual point count. **Verified
  non-regressive: re-running the Stage-A `--control a0r` analysis after the
  change reproduces all five of its published tables BYTE-IDENTICALLY**
  (`main_effects.tsv`, `bd_per_image.tsv`, `interactions.tsv`,
  `main_effects_by_class.tsv`, `arm_byte_identity.tsv`).

`avifdoe_stageb6_analyze.py` is new and **imports** `bd_rate`, `frontier`,
`median_ci`, `q1q3` and the content-class map from the Stage-A analyzer,
`binom_two_sided` from the Stage-A gates, and SROCC from `zenstats`. It
re-implements no statistic.
