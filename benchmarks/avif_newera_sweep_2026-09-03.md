# AVIF new-era sweep registration — 2026-09-03

> **✅ RUN AND ANALYSED — results:
> [`avif_eradelta_analysis_2026-09-03.md`](avif_eradelta_analysis_2026-09-03.md)**
> (2026-09-03). Headline: arm-set A reproduced Stage-A's bitstreams
> **byte-identically on every shared cell**, and arm-set B did the same
> independently on 3,456 cells (vs `a1`) + 2,880 (vs `a2`) — so every covered
> effect is the same number, not merely within CI. `scm3` at speed 7 fires on
> **90/288 cells, exclusively on screen-adjacent content**, worth a **median
> −50.08 % BD-rate** conditional on firing. Arm-set C's clean bd10-native read
> **inverts** the superseded `t1d` answer.

Companion to [`avif_newera_delta_2026-09-03.md`](avif_newera_delta_2026-09-03.md)
(read that first — this doc assumes its findings). Registers what gets
declared this wave, what's reconciled/deprioritized from the outstanding
Stage-B envelope, and the budget.

---

## 1. Grid — three arm-sets, all reusing the registered 9-point ladder

**Quality ladder** (unchanged from DOE plan §2.3, reused deliberately —
already low-q-dense per the workspace discipline: 5 points ≤ q45 vs 4 above,
and every point is a control-grid member so no interpolation/anchor error):
`q ∈ {5, 15, 25, 35, 45, 60, 76, 90, 96}`.

**Corpus:** the same budget corpus (`avif-doe-1024-2026-09-01`, 32 images,
1024² content-aware crops) for arm-sets A and B, so results are directly
comparable to Stage-A's own tables cell-for-cell. Arm-set C uses the **native**
corpus (`avif-subsample-2026-09-01`) by construction — that's the whole point.

### Arm-set A — stability re-run (exact A1 replication at the new pin)

Re-declares `SweepAxes::svt_doe_main()` **unchanged** — same 24 strata (1
default + 16 knob arms + 6 non-default-speed controls + 1 bit-depth arm) × 9q
× 32 images. **6,912 cells, zero new code.** This is a direct, paired
(same image, same config, only the encoder pin differs) comparison against
Stage-A's own A1 table (§6 of the Stage-A doc) for every knob at speed 4.
Content-addressed dedup means any cell whose bytes come out byte-identical to
the old-pin run costs nothing to re-verify at score time (§2 of the delta
audit already found only 2 of 33 upstream commits are even reachable from
this path, so the *expectation* is that most of these 6,912 cells reproduce
old bytes exactly — that expectation is itself the thing being tested).

### Arm-set B — new/targeted: the priority list from the delta audit's risk
list, at the two product-reachable presets never crossed with a knob before

New `SweepAxes` plan, `svt_doe_era_delta_r1` (implemented in zenavif, §2
below), `--max-deviations 2` — following `svt_doe_b6`'s own established
shape exactly (a restricted `svt_knobs` list × an explicit `speeds` list,
full cross, every cell ≤2 deviations), rather than inventing asymmetric
per-knob speed restriction the builder doesn't otherwise support:

| knob | why it's on the priority list |
|---|---|
| `tn3` (tune=3) | §5.1 top risk — forces scm3, which is live at speed 7 |
| `shp7` | largest/most consistent cost (§6.1 Stage-A) |
| `shp3` | same family, smaller magnitude |
| `vbst1.2.5` | large-tail arm (§5.3) |
| `vbst1.3.5` | large-tail arm |
| `vbst1.3.7` | large-tail arm |
| `qml1.2.10` | sign-flips between presets (§5.4) |
| `qml1.4.10` | sign-flips between presets |
| `scm3` | dead at speed ≤6/preset ≤7 but live at speed 7 (delta audit §4.1) |

× **speeds {4, 6, 7}** (matching `svt_doe_b6`'s own speed set, so speed 4 is
a free internal-consistency check against arm-set A, and speed 6/7 are the
two points that matter). **10 knob levels (1 default + 9 above) × 3 speeds =
30 combinations × 9q × 32 images = 8,640 cells.** The known-zero legs
(`scm3`×speed4, `scm3`×speed6, every knob's speed-4 leg already in arm-set A)
are accepted rather than hand-carved out — content-addressed dedup makes
them free, and a uniform full cross is far less likely to have a declaration
bug than an asymmetric one.

### Arm-set C — REVISED: re-run an existing, already-corrupted native bd10
block, not a new one (delta audit §6.0)

**This was going to be new code until the pre-declaration duplication check
(the mandatory step before writing anything — "NO DUPLICATE
IMPLEMENTATIONS") found `SweepAxes::svt_doe_t1_bd10_transfer()` already
exists** (`zenavif` `bcd7978`, 2026-09-02 08:30) and is exactly this
question: bd10 on the native corpus, speed 4, the 3-point probe ladder
(`q∈{15,45,90}`). It was already run, as `avifdoe-svt-t1d-20260902` —
complete, 96/96 — but **encoded 09:57–10:16 on 2026-09-02, hours before
either #18 fix.** 24 of its 96 cells (the 8 native images that force
multi-tiling) ran the exact broken configuration; the delta audit's §6.0
walks the evidence. **No new Rust code — this arm-set is a fresh declaration
of the existing plan under a new run name** (content-addressed cells mean
re-declaring into the old run name would treat its 96 wrong-pixel cells as
already-done and skip them — same shape as `avifhbd-t2a-fix-20260902`'s own
restart).

96 cells (32 images × 3 q-points × 1 speed), same size as the block it
replaces. The 72 single-tile cells are expected to reproduce byte-identical
to the pre-fix run (a live check on this audit's §2.1 claim that only 2 of
33 commits are AVIF-reachable); the 24 multi-tile cells are expected to move
from strongly negative to strongly positive, matching §10.4f's own
verification table.

**Total new declaration: 6,912 (A) + 8,640 (B) + 96 (C, re-run) = 15,648
cells.** Compare Stage-A's whole wave (117,435 cells, ≈70.5 CPU-h) — this is
**13.3%** of that. Per §15.4's measured SVT cost profile (preset dominates:
s4 is 17-27× s6/s7; budget-corpus cells are cheap; C's 96 native cells are a
rounding error against A/B), the estimate is **≤3.5 CPU-h encode**,
comfortably inside the fleet capacity freed by t2a-fix/t2b's completion (§8
of the delta audit).

**Also found in the same duplication check, explicitly NOT adopted into this
wave**: `svt_doe_t1_bd10_knobs` (bd10 × the 15 live knobs, budget corpus,
speed 6) is declared as `avifdoe-svt-t1b-20260902` — **4,320 cells, 0 done,
never actually run.** It predates and is unrelated to the #18 fix (budget
corpus, single-tile by construction, not corrupted) — a genuine pre-existing
gap, flagged for the coordinator, not silently absorbed into an era-pin
verification wave that isn't what declared it.

### No tune-vmaf arm, no aom arm

Both explicitly excluded — falsified/blocked respectively, per the delta
audit §3 and §7. Registering either would either measure nothing (tune-vmaf —
there is no config surface to vary) or build on another lane's actively
moving in-flight commits in a shared repo (aom).

---

## 2. Implementation (zenavif, additive, tested)

`SweepAxes::svt_doe_era_delta_r1()` (arm-set B only — the only genuinely new
plan) added next to `svt_doe_b6()` in `src/sweep.rs`, following the same
shape: a `Vec<SvtParams>` restricted knob list, an explicit small speed
list, `PLANS` table entry (so the "unknown plan" diagnostic can't silently
drift the way §15.6 of the DOE plan already found once), and a cell-count
test pinning 8,640 (10 knob levels × 3 speeds, `--max-deviations 2`, the
same arithmetic shape as `svt_doe_b6`'s own 9×3=27). Arm-set A reuses
`svt_doe_main()` verbatim; arm-set C
reuses `svt_doe_t1_bd10_transfer()` verbatim (§1) — neither needs any zenavif
code change, only a fresh declaration under a new run name.

---

## 3. Stage-B B-1/B-2/B-3 reconciliation

Full trigger table: Stage-A doc §10 (55 triggers, 447,636 cells if all
honoured, 7.5× the 60,000-cell envelope). Only B-6 (2 triggers, `acb3`/`shp3`)
has been declared and run to date (§17 of the DOE plan — COMPLETE, neither
knob earned a place in the tuning set). Reconciling the remaining 53 against
this era's findings:

| trigger | status this wave | reason |
|---|---|---|
| **B-1 `tn3`@{s4,s6}** | **partially absorbed** | arm-set B adds the speed-7 leg (the one B-1's own follow-up spec asks for: "speeds {4,6,7}") crossed with the mechanistically-linked scm3 finding. The full 5-level dense grid (13,920 cells) is **not** run this wave — this is a targeted single-level (tn3 is already level "tune=3", there is no "tune=3.5") speed extension, not the registered B-1 follow-up shape. Remaining: re-run the *other 4 tune levels* B-1 would want, at native size, if `tn3` proves stable — coordinator's call, not executed here. |
| **B-1, the other 15 triggers** (qml×2, shp3/shp7, tl1.0/tl1.1, vbst×3 @ both speeds) | **not absorbed — arm-set B re-measures at s6/s7 but does not run the 5-level dense grid any of them registered for** | Arm-set B is a stability/coverage check, not the Stage-B follow-up. `shp3`/`shp7`/`vbst*`/`qml*` are covered by arm-set B at the new pin, which tells us whether the *existing* Stage-A numbers are still trustworthy — it does not buy the dense grid B-1 asks for. Genuinely undeclared; §10.1 of the Stage-A doc already ranks these (vbst arms "most expensive and least certain" — deprioritize; QM×sharpness is B-2's job, not B-1's). |
| **B-2 QM×sharpness cluster** (6 pairs, ~15,552 cells) | **not run — flagged as the strongest remaining candidate, unchanged from Stage-A's own read** | This is real, i.e. the single highest-value remaining Stage-B item by the existing analysis (§10.1 point 3: "the strongest measured structure in the wave"), and nothing in this era's findings changes that read (none of the QM or sharpness knobs are in the AVIF-relevant 2-commit delta). Registering it is squarely a budget/priority decision for the coordinator, not something this era-delta lane should absorb into a pin-verification wave. |
| **B-2, the other 17 pairs** | **not run** | Lower-value per §10.1 point 4 (mostly `vbst*`-involving, tail-driven, IQR-triggered — "characterising a tail, which a 5-level dense grid on 32 images is not obviously the right instrument for"). Unchanged assessment. |
| **B-3 `bd10`@s4** | **has native evidence already, but it's a re-run of corrupted data, not new coverage** | `svt_doe_t1_bd10_transfer` (a T1-d block, pre-existing) already asks the native-bd10 question at speed 4 — but its stored run encoded before either #18 fix, with 24/96 cells structurally wrong (delta audit §6.0). Arm-set C is that block re-declared fresh, not a B-3 answer in its own right (single speed, 3-point probe ladder, not B-3's full content-stratified dense follow-up). Content-stratification (the literal B-3 ask) is not separately run — the native measurement will have per-image results a follow-up could bucket by content family. |
| **B-3, the other 12 triggers** (mtx32×2, qml×4, tn3@s4, vbst×5) | **not run** | Same read as their B-1 entries — arm-set B re-measures some of these at new presets but does not run the content-stratified dense follow-up. |
| **B-4** | still **NOT EVALUABLE** | A4 (the timing block) was never declared in Stage-A and isn't declared here either — `encode_ms` still isn't persisted by the fleet path. Out of this lane's scope (it's an instrumentation gap, not an era-pin question). |
| **B-5** | still **does not fire** | Unchanged — no knob has |median| ≥1% at both presets with opposite signs, on the one preset pair (s4 vs s6) Stage-A could evaluate it on. Arm-set B's speed-7 data is a bonus third point for this trigger's evaluation once scored, at no extra declaration cost. |

**Net effect on the 7.5× overrun**: this wave spends 15,648 of the 60,000-cell
Stage-B envelope (21%) on stability verification + two newly-open axes,
touches 2 of 6 open trigger classes (partial), and explicitly does **not**
attempt to close the overrun — that remains a budget/priority call for
whoever owns the next Stage-B wave, informed by: **the era bump itself
doesn't change B-2's or most of B-1's priority ranking** (their knobs are
untouched by the 2-commit AVIF-relevant delta), so the existing §10.1
guidance (QM×sharpness first, vbst arms last) still stands without needing
re-derivation.

---

## 4. Budget vs fleet capacity

- **Freed capacity**: `avifhbd-t2a-fix-20260902` (3,248 cells) and
  `avifhbd-t2b-20260902` (432 cells) are both COMPLETE (delta audit §8) — the
  workers that were draining them (tower's `zen-score-t2afix`, plus whatever
  local capacity fed t2b) are idle and available.
- **Concurrent, not-to-disturb**: `zen-score-b6enc` on tower (late score
  compaction for an already-COMPLETE B-6 encode run, ~1400 min budget, will
  self-exit like t2afix did), and the AOM lane's `avifsub-{aom,svt}-sf-cpu-
  20260901` scoring loop locally. Neither competes for the same corpus/plan
  namespace as this wave, so no interference is expected.
- **This wave's estimate**: ≤3 CPU-h encode (§1), well inside a single-box
  budget. Scoring: 2 metrics (`ssim2,zensim`) per the standing user directive
  that dropped butteraugli from the default set (DOE plan §13.5) — 12,672
  cells × 2 metrics = 25,344 score jobs, comparable in scale to a single Stage-
  A sub-block.
- **Dense-low-q**: satisfied by construction — the reused 9-point ladder is
  already low-q-weighted (§1), and arm-set C's native bd10 cells use the same
  ladder rather than a sparser one, honoring the workspace's q5-q60 density
  mandate.
- **Dead-cell exclusions**: `tn0` and `scm3`-below-speed-7 are excluded from
  every arm-set by construction (not merely by post-hoc filtering) — the new
  `svt_doe_era_delta_r1` plan simply never emits those combinations, so there
  is no "declare then discard" waste.
