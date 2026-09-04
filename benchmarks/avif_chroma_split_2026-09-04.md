# AVIF chroma split — the zenrav1e 4:2:0 arm, and what it is allowed to conclude

**REGISTRATION. Written 2026-09-04 while the arm was encoding and BEFORE any
cell of it was scored.** Every cut, bar, exclusion and decision rule below is
fixed here. The results section is appended later and adds no rule.

Closes the rank-1 data gap of
[`avif_backend_selection_2026-09-03.md`](avif_backend_selection_2026-09-03.md)
§6 — the document whose own §0.6 calls the confound *"the single most important
caveat in this document."*

Era pin: **NEW**, and deliberately so — see §3. zenavif `6dfdf6f` / zenrav1e
`7ad86844`; fleet image `ghcr.io/imazen/zenfleet-worker:exec-avifchroma-f15bb3a5`
(digest `sha256:dad86ae95b75a36438b4e3a968c65c7fef513ec381c5a7f05930a56ffeedb3dd`).
Producer commit `f15bb3a5`. **Never join across eras** — §3 is how this arm earns
the right to be read against the old one, or fails to.

---

## 1. The confound, and why no existing data can split it

`avif_config_from_knobs` pinned `Yuv420` for `backend=svt-rs` and left zenravif
on zenavif's `Yuv444` default. **There was no chroma knob for AVIF at all.** So
`backend` and `chroma` were the same column:

| arm | backend | chroma | bitstreams read |
|---|---|---|--:|
| `brsdr` | zenrav1e | **4:4:4**, seq_profile 1 | 640 / 640 |
| `brnat` + `a0r` | svt | **4:2:0**, seq_profile 0 | 448 / 448 + 26 / 26 |

Zero exceptions, measured out of the `av1C` box. Every cross-backend number ever
published on this corpus is therefore a **(backend × chroma)** number, including
the one that drives the picker's decision surface:

> svt-4:2:0 **cannot reach ssim2 90 on 16 of 32 references at any q, at any
> speed** — on plots it tops out at 40.5 / 54.4 / 66.7 / 67.1 while zenrav1e
> reaches 94.2 / 94.4 / 90.9 / 93.2 on the same images. The best of all 118 svt
> arms in A2 does not fix it.

That is either **the most valuable signal in the DOE** (a backend that cannot
serve synthetic content) or **an artefact of the chroma column riding along with
it** (4:2:0 cannot serve synthetic content, and svt merely never got to try
4:4:4). The two readings imply opposite pickers. No existing cell distinguishes
them, because no zenrav1e cell was ever 4:2:0.

**The knob now exists** (`f15bb3a5`), gated by a read-back test that fails
without it. §2 is the arm that uses it.

---

## 2. The arms

Both on the **BUDGET** corpus `/mnt/v/output/avif-doe-1024-2026-09-01/sources`
(32 refs; worker prefix `s3://codec-corpus/avif-doe-1024-2026-09-01/`), the
**9-q ladder** `5,15,25,35,45,60,76,90,96`, zenravif speeds **1–10**.
32 × 9 × 10 = **2,880 cells each**, dry-run verified exactly, `0 merged by
resolved state`.

| run | knob grid | what it is |
|---|---|---|
| `avifdoe-rav-br420-20260904` | `{"backend":["zenravif"],"chroma":["420"],"speed":[1..10]}` | **the arm** — the cell that has never existed |
| `avifdoe-rav-br444-20260904` | `{"backend":["zenravif"],"speed":[1..10]}` | **the era control** (§3) |

Score run `avifdoe-cs-sf-cpu-20260904`, metrics **`ssim2,zensim`**, no `--hdr`,
**declared at launch** (round 1 declared 26 jobs off the first 270 encoded
cells), both arms in ONE score run so the scorer cannot differ between them.

**Why the 9-q ladder and not `brsdr`'s 29.** `brsdr`'s ledger holds 4,979 cells,
but only **2,880 of them are a balanced fully-crossed grid**; the other 2,099 are
29-q extras on *whichever images the workers reached first* — an uncontrolled
subset, not a designed stratum. The 9-q grid is the only cell-for-cell match, so
it is the whole comparison and **the 2,099 extras are excluded from every number
in this document**.

---

## 3. The era control — pre-registered, and it can fail

The chroma knob is a zenmetrics change, so this arm ships a **new fleet image**.
zenavif has moved **28 commits** past `brsdr`'s pin `56179fcb`, including a
breaking `0.1.8 → 0.2.0` and +290 lines in `src/encoder.rs`. **This is a new era
and it is not assumed inert.**

`br444`'s knob tuple carries **no chroma token**, so its CellIds are identical to
`brsdr`'s 9-q subset *by construction* — verified before launch: the two 2,880-line
cells files are **byte-identical after sorting**, and the two manifests are the
same 1,373,158 bytes. So `output_sha` is directly comparable, cell for cell.

**Decision rule, fixed now:**

* **`br444` reproduces `brsdr` byte-for-byte on 2,880 / 2,880** → the era is
  proven inert for zenravif SDR. The arm may additionally be read against
  `brsdr`'s scored cells, and that is stated with the fraction.
* **Anything less than 2,880 / 2,880** → the era moved. **`br444` becomes the
  only 4:4:4 comparator**, every number is within-era, and no `brsdr` figure is
  joined to this wave. The reproduction fraction is reported either way.

This is the same instrument the era-delta wave used (6,912 / 6,912
byte-identical) and the same discipline: **byte identity is the evidence, not a
statistical test that the difference is small.**

### 3.1 …and a SCORER control, which the era control does not cover

The era control proves the **encoder** is inert. The backend axis compares
`br420` — scored by *this* wave's score run — against svt cells scored by
Stage-A's. **If the scorer moved, that axis is inadmissible and §3 would not
have caught it.**

The control costs nothing: `br444`'s blobs are byte-identical to `brsdr`'s and
the two were scored by *different* score runs, so joining on `encode_sha` and
differencing `m_ssim2` isolates the scorer exactly.

`scorer_control.tsv`, and **`NOT RUN` is a distinct named outcome from a zero
delta**, so a missing input can never read as a pass. Both controls are
recomputed on the full arm at completion.

---

## 4. The question, and the decision rule that answers it

**Q: is the reach failure on plots and screens a CHROMA property or a BACKEND
property?**

The instrument is per-image and deterministic: `max achieved ssim2` over that
image's 90 cells (9 q × 10 speeds), per arm. No fitting, no interpolation, no
statistic. The pre-registered comparison set is the **16 references where
svt-4:2:0's max achieved ssim2 is below 90**, listed here so it cannot be chosen
after the fact:

| class | reference | svt-420 max | zenrav1e-444 max |
|---|---|--:|--:|
| plot | `7004.scale1024x1024` | 40.48 | 94.20 |
| plot | `7042.scale1024x1024` | 67.14 | 93.21 |
| plot | `7050.scale1024x1024` | 84.80 | 94.50 |
| plot | `7052.scale1024x1024` | 89.31 | 95.19 |
| plot | `7058.scale1024x1024` | 54.36 | 94.38 |
| plot | `7076.scale1024x1024` | 66.74 | 90.92 |
| screenshot | `8134.scale1440x900` | 87.17 | 93.62 |
| screenshot | `8288.scale375x667` | 77.62 | 93.70 |
| screenshot | `8414.scale1280x800` | 79.69 | 90.78 |
| screenshot | `8434.scale414x896` | 86.39 | 93.49 |
| screenshot | `8446.scale2560x1440` | 87.33 | 91.85 |
| scan | `6602.scale3302x4844` | 76.44 | 91.72 |
| scan | `6604.scale3286x4868` | 80.07 | 94.38 |
| ai-gen | `9032.scale1024x1536` | 89.81 | 90.68 |
| ai-gen | `9118.scale1536x1024` | 88.75 | 94.53 |
| ai-gen | `9444.scale1024x1536` | 88.41 | **89.98** |

(`9444` is the honest edge case: zenrav1e-**444** misses 90 too, by 0.02. It is
in the set because svt misses it, and it is flagged wherever it is counted.)

**The rule, fixed now.** Let `k` = the number of those 16 on which
**zenrav1e-4:2:0** also fails to reach ssim2 90, and `k₁₁` the same over the 11
plot + screenshot references (the sharpest split in the wave: svt fails 11/11
there and 0/7 on photos).

| outcome | reading | consequence for the picker |
|---|---|---|
| **k₁₁ ≥ 9** | **CHROMA.** 4:2:0 itself cannot reach the band on synthetic content; svt was never the cause. | Decision 1 (chroma) is the whole lever. Backend identity does **not** predict reach. The published headline must be re-worded from "svt cannot" to "4:2:0 cannot". |
| **k₁₁ ≤ 2** | **BACKEND.** zenrav1e serves these at 4:2:0 where svt cannot. | The published headline survives as a genuine backend property, and the chroma confound, while real, did not manufacture it. |
| **3 ≤ k₁₁ ≤ 8** | **CONDITIONAL.** Reported per image and per class, with no headline. | Both features are needed; the picker needs the interaction, not either main effect. |

The thresholds are 9 and 2 out of 11 because the two clean readings are "nearly
all" and "nearly none"; anything between them is a real interaction and gets
reported as one rather than rounded to a story. **No outcome here is a null
result** — all three change the decision surface, which is why the arm is worth
running whichever way it lands.

### 4.0.1 ADDENDUM (2026-09-04, before any result) — the ladder is itself a reach constraint

The 16-reference set above was derived from svt's **29-q** grid, which reaches
q98. Both chroma arms run the **9-q** ladder, which stops at q96. Measured cost
of that restriction, on svt's own scored cells:

* achieved max ssim2 drops by up to **2.106** points (worst: `1432`, 92.74 → 90.63);
* it flips **exactly one** image's reach-90 verdict — `9954.scale1024x1536`,
  90.49 → 89.64 — which is **ai-gen, not plot/screenshot**, so **the k₁₁
  denominator of 11 is untouched and the decision rule in §4 is unaffected**.

(The same computation reproduces the published `svt_qmax` exactly on all 32
images from the 29-q cells, which is the check that the instrument is the same
one.)

Two consequences, both adopted:

1. **Every arm is read on ONE ladder.** The analyzer restricts svt to the same 9
   q points, so no comparison is confounded by ladder density. A `br420` miss is
   then never attributable to a ceiling the other arms did not also have.
2. **An in-wave reading is reported beside the registered one**: which images
   svt-420 misses *here*, and whether `br420` and `br444` miss them too. That
   version of the question is immune to ladder density by construction, because
   all three arms share the ladder. The registered 16-image set stays PRIMARY —
   it is what was registered — and both are printed rather than one being
   quietly substituted for the other.

---

## 4.1 The two clean axes

With `br420` in hand the published `+7.3 %` (zenrav1e-444 vs svt-420 — backend
and chroma **jointly**) decomposes for the first time:

* **TRUE BACKEND axis** — `br420` vs `brnat`/`a0r` svt-420. Same chroma, both
  backends.
* **TRUE CHROMA axis** — `br420` vs `br444`. Same backend, both chromas, and
  since these are the same 2,880 CellIds on the same images, it is fully paired.

Statistics for both are the ones the backend doc already used, unchanged and not
re-derived here: **BD-rate over the 9-q ladder, median over 32 images, 95 % CI,
sign test**, plus the band-restricted read over ssim2 ∈ [30, 95]. Both are
reported per image and condensed by content class.

---

## 5. Gates, and exclusions

**Pre-launch (all PASSED before any worker scaled):**

| gate | requirement | result |
|---|---|---|
| G-CELLS | 2,880 cells per arm, dry-run | 2,880 / 2,880, `0 merged` |
| G-TUPLE | `br420` carries `chroma` on every cell; `br444` on none | 2,880 / 2,880 and 0 / 2,880 |
| G-CELLID | `br444` cells byte-identical to `brsdr`'s 9-q set | **identical** (sorted diff empty) |
| G-CHROMA | `av1C` **and** AV1 sequence header of real fleet blobs read the requested subsampling | **6 / 6 PASS**, `chroma 420`, `seq_profile 0` |

G-CHROMA is the one that matters and it is the first-cell gate: six blobs pulled
from `br420`'s own `blobs/` prefix, parsed by `zenavif-parse` through
`examples/avif_depth_verify` (the same R1 + R2 routes gate G3 uses for bit
depth). Under the pre-`f15bb3a5` code every one of them would have read `444` /
`seq_profile 1`.

**Widened to a census once both arms were producing** — 40 blobs per arm, drawn
from each run's own `blobs/` prefix, same instrument:

| arm | `av1C` chroma | `seq_profile` | depth | n |
|---|---|---|--:|--:|
| `br420` | **420** | **0** (Main) | 8 | 40 / 40 |
| `br444` | **444** | **1** (High) | 8 | 40 / 40 |

Zero exceptions either way. This is what splits the confound *at the bitstream
level* rather than by declaration: `br420` emits the same `seq_profile 0` /
4:2:0 the svt arms emit on 448 / 448 of theirs, from the zenrav1e backend, and
`br444` reproduces `brsdr`'s measured `seq_profile 1` / 4:4:4 on 640 / 640.

The census is also **how `br444`'s chroma is established at all.** Its knob tuple
carries no chroma token — that is what makes its CellIds `brsdr`'s — so the
harvest deliberately reports its `chroma` column as `None` rather than
synthesizing 4:4:4 from zenavif's default (`avifdoe_harvest.py`: "synthesizing
`-420` here would be convention dressed as evidence"). The 4:4:4 label above is
a **measurement**, on the same footing as the `NAIVE_BACKEND_CHROMA` assertion
that rule protects.

**Exclusions, fixed now:**

1. The 2,099 uncontrolled 29-q `brsdr` extras — excluded from everything (§2).
2. BD-rate requires the owner's **≥ 4 overlapping quality points**. An image
   that does not overlap is **NOT MEASURED**, never imputed and never dropped
   silently: it is reported with its `n`.
3. Cells that fail to encode are reported as a fraction, never dropped.
4. `zensim` is emitted as a **720-wide feature vector, not a scalar**, so
   **`ssim2` is the only corpus-wide scalar quality response** and is the
   matching axis everywhere — a limitation of the score run, not a choice here.
5. Coverage is counted at the **cell** level by joining through `encode_sha`. A
   blob-prefix count is not a coverage number (blobs are content-addressed and
   dedup; score blobs are additionally chunked).

**Topology** (per §4.5d of the Stage-B doc, which *corrects* §4.5: the win is
narrow cpusets, not low oversubscribe): 16 workers × **1-core cpuset**,
`ZEN_CORE_OVERSUBSCRIBE=2`, `ZEN_PASS_TIMEOUT=7200` — 8 on dev (cores 20–27),
8 on tower (cores 0–7, `--cpu-shares 256 --memory 24g`, household caps).
r7900x is **not used**: the timing instrument holds it.

---

## 6. What this arm does NOT answer

* **svt at 4:4:4** — blocked in the encoder (`zenavif/src/encoder_svt_rs.rs:391`),
  so the (backend × chroma) square has three corners, not four. The sweep now
  refuses that cell **by name** rather than silently pinning 420, but refusing is
  not measuring: no conclusion here covers it.
* **Native size** — budget-1024² only, exactly as §3 of the backend doc was.
* **4:2:2** — `EncodeChromaSubsampling` names two variants; 4:2:2 has no config
  to resolve to and is refused by name.

---

## 7. Results (harvested 2026-09-04, post-reboot)

**Coverage: 5,760 / 5,760 cell rows scored, 0 unscored, 0 missing bytes.**
`zenfleet-ctl gap` reads `0 of 2880` on both `br420` and `br444`. The 5,760 cell
rows join to **5,179 distinct `encode_sha`** (581 exact-duplicate bitstreams
across cells — expected AVIF byte-convergence at high q/slow-speed, not a
defect); every one of the 5,179 distinct bitstreams carries both `ssim2` and
`zensim` features. `avifdoe_chroma_harvest.sh` is idempotent and was re-run
clean at this fill level; nothing here is a partial read.

### 7.1 Controls — both read exactly as the registration required to license anything else

| control | reading |
|---|---|
| era (encoder) | **2,880 / 2,880 byte-identical** — `br444` reproduces `brsdr`'s 9-q subset exactly. **ERA INERT.** |
| scorer | **max \|Δ ssim2\| = 0.0** on 2,880 shared bitstreams. **SCORER INERT.** |

Both controls are unconditional passes, not "close enough" reads. Per §3's rule,
this means `br444` may additionally be read against `brsdr`'s scored cells (era
inert), and the backend axis (§7.3) is not an artefact of a scorer that moved
between waves.

### 7.2 The verdict — CHROMA, and it is not close

**k₁₁ = 11 of 11, k = 16 of 16.** Every one of the 16 pre-registered
svt-4:2:0-fails-90 references — all 11 plot+screenshot images included —
is ALSO a reach-90 failure for **zenrav1e at 4:2:0** (`br420`). Per the fixed
rule (k₁₁ ≥ 9 → CHROMA), this is the maximum-strength reading the design can
produce: not one of the 11 plot/screenshot images is spared. Independently
verified by hand against `tables/reach_per_image.tsv` (not just the script's
own printed line) — `br420_reaches90` is `NO` on all 16 registered images.

**The published headline is re-worded per the pre-registered consequence**:
"svt cannot reach ssim2 90 on plots/screenshots" becomes **"4:2:0 cannot reach
ssim2 90 on plots/screenshots"** — the backend behind the 4:2:0 encode does not
matter; zenrav1e (a materially different encoder/RD-control implementation from
svt) hits the identical ceiling, on the identical image set.

**In-wave, one-ladder reading (§4.0.1's immune-to-ladder-density check, computed
independently on the same 9-q ladder for all three arms) corroborates it with a
second, non-overlapping method**: on this ladder svt-420 fails 90 on **17**
images (16 registered + `9954`, the pre-flagged ladder-edge flip — ai-gen, not
plot/screenshot, so it changes nothing about k₁₁). **`br420` also fails on all
17 of those 17** (full intersection). **`br444` fails on only 2 of the 17**
(`9444` and `9954` — both pre-flagged edge cases: `9444` was registered in §4 as
"the honest edge case" where even 4:4:4 misses 90 by a hair, and `9954` is the
ladder-restriction flip from §4.0.1). Two independent countings of "does the
zenrav1e arm reach where svt fails" — the pre-registered 16/11-image set and the
in-wave 17-image set — land on the same answer.

Per-image detail: `tables/reach_per_image.tsv` (32 rows, `br420_max_ssim2` /
`br444_max_ssim2` / `svt420_max_ssim2` / the three `*_reaches90` booleans).

### 7.3 The two clean axes — bytes tell a DIFFERENT story than reach, and that difference IS the finding

| axis | comparison | n | median BD-rate | 95% CI | wins (a / b) | sign p |
|---|---|--:|--:|---|---|--:|
| **chroma_true** | `br420` (a) vs `br444` (b) — backend HELD, chroma varies | 32 | **+0.08%** | [−0.97, +0.93] | 15 / 17 | 0.860 |
| **backend_true** | `br420` (a) vs `svt420` (b) — chroma HELD at 4:2:0, backend varies | 31 (1 NOT MEASURED, BD-rate's ≥4-point overlap guard) | +10.19% | [+0.25, +19.15] | 10 / 21 | 0.071 |
| published_confounded (original, for reference) | `br444` (a) vs `svt420` (b) — backend AND chroma jointly, as originally published | 31 (1 NOT MEASURED) | +5.60% | [−1.83, +14.19] | 12 / 19 | 0.281 |

**Read this table together with §7.2, not instead of it.** Holding backend
constant and varying only chroma (`chroma_true`) moves BD-rate by a median of
**+0.08%** — statistically indistinguishable from zero (CI straddles 0, sign
test a coin flip at p=0.860). Holding chroma constant and varying backend
(`backend_true`) moves it by +10.19%, svt-favouring, CI excludes zero but only
just (lower bound +0.25) and the sign test does not clear 0.05. **So chroma is
not a bytes-efficiency lever at matched achievable quality — it is a quality
CEILING.** The two axes measure different things and neither substitutes for
the other: `chroma_true` says "when both arms can reach a quality, they cost
about the same bytes to get there"; §7.2 says "one of the two arms frequently
cannot get there no matter what it spends." A reader who only saw
`chroma_true`'s near-zero median would wrongly conclude chroma barely matters;
a reader who only saw §7.2 would not know that byte cost, where both arms
succeed, is roughly chroma-neutral. Both are true and both are needed.

`published_confounded` (the original backend×chroma-confounded comparison this
whole arm exists to split) reproduces the qualitative shape of the originally
published figure (svt-leaning, wide CI, not significant on this 9-q/31-image
read) — consistent with, not contradicting, the original finding; this arm
explains *why* it read that way rather than overturning it.

Full per-image BD-rate: `tables/axis_backend_true.tsv`,
`tables/axis_chroma_true.tsv`, `tables/axis_published_confounded.tsv`.

### 7.4 What changes in the decision surface

**Decision 1 (backend doc §4) is CONFIRMED as chroma, not backend, and is now
measured rather than inferred from a confound.** A picker that reaches for
"switch backend" to fix a plot/screenshot quality ceiling will not fix it —
zenrav1e hits the same wall at 4:2:0. The only lever that opens the ceiling is
the chroma knob itself (4:4:4), which costs essentially nothing in bytes where
both chromas can already succeed (§7.3) and is the sole lever that makes the
otherwise-unreachable images reachable at all.

**Decision 2 (time/backend budget) is untouched by this arm** — it was never a
chroma question — but should now be read as "which backend, at 4:4:4 chroma
where reach demands it, within the time budget," not "which backend, full
stop." This arm does not re-measure the time axis (§6, native-size and
knob-time gaps are unchanged).

### 7.5 Ranked gaps — revised

Backend-doc §6 rank 1 ("a zenrav1e 4:2:0 arm... the ONLY way to split the
backend effect from the chroma effect") is **CLOSED by this arm**. Remaining,
renumbered:

| rank | gap | status |
|--:|---|---|
| 1 (was 2) | An svt 4:4:4 arm, or a recorded decision that svt is 4:2:0-only | still blocked upstream in zenavif/zenav1-svt |
| 2 (was 3) | Native-size cross-backend AND cross-chroma coverage | this arm is budget-1024² only (§6) |
| 3 (was 4) | The QM×sharpness interaction at s4/s7 | unchanged |
| 4 (was 5) | S1c content-class speed splits | **now available** — see `avif_speed_instrument_2026-09-03.md`, S1c section (companion lane, same session) |
| 5 (was 6) | More plot/screen references (n=6/5) | unchanged — the CHROMA verdict rests on the same small n as the original finding it explains |
| 6 (was 7) | A second scalar quality response | unchanged |
| 7 (was 8) | aom-rs as a third arm | PLANNED-BLOCKED, unchanged |

### 7.6 Outputs

`/mnt/v/output/avif-chroma-2026-09-04/` — `chroma_scored_2026-09-04.parquet`
(5,760 rows × 18 cols) + `tables/{era_control,scorer_control,axis_backend_true,
axis_chroma_true,axis_published_confounded,reach_per_image,notes}.{tsv,json}`.
Triple-mirror + pointer: `avif_chroma_split_2026-09-04.pointer.md`.
