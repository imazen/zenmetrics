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
