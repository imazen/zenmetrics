# REV2 WAVE — the recalculation, the GPU copy, and the refit

**Pre-registered 2026-09-06, before the wave ran.** Gates below were written
before their results and are never edited to match one. A gate that fails is
reported failed, not re-scoped. Amendments are appended with a date and the
thing that changed the decision, never folded into the original text.

**User directive (verbatim, 2026-09-05):** *"we should fix arithmetic defects
aggressively before shipping, and perhaps change feature definitions and
formulas to make monotone linear models better. we have a fleet to
recalculate"*.

**This lane is the execution of** [`PLAN_REV2_RECALC_2026-09-06.md`](PLAN_REV2_RECALC_2026-09-06.md)
now that its one blocker — the arm token — has landed. It owns four things and
nothing else:

1. the **fourth hand-copy** of the moved arithmetic, in `zensim-gpu`;
2. the **image** that carries a rev2-capable extractor;
3. the **wave** that recomputes the tables;
4. the **refit** of the shipped SDR chain (and its corruption head) at rev2.

It does **not** own the arithmetic (zensim's rev2 lane), the arm choice (R6 /
R6b), or the decision to make revision 2 the shipped revision. `SHIPPED_REVISION`
stays `Rev1` unless the refit passes every gate in §5 and the user approves the
flip.

---

## 0. What revision 2 IS, restated so the wave's scope is checkable

Three registered eras, flipped together behind `ZENSIM_FORMULA_REV=2`:

| era token | defect | arm, decided by measurement | slots moved at 372 | at pools-live 944 | at zeroed ext944/ext924 |
|---|---|---|--:|--:|--:|
| `v1ssimcap` | F4 — unbounded SSIM luminance `1 − D²` | `Clamp` = `max(0, 1 − D²)` | 132 | 132 | 36 |
| `freecomp` | F5 — free-40 raw-moment route parity | reassociation (both halves) | 0 | 11 | 0 |
| `v1hfgain` | F17 — unbounded `contrast_inc` | `SaturatingExcess` = `g/(g+1)` | 12 | 12 | 12 |
| **combined, v1 side** | | | **144** | **144** | **48** |

Two facts about this table are load-bearing for every gate below and are quoted
from the lanes that measured them, not re-derived here:

* **`Clamp` is bit-identical to revision 1 on every pixel this box has.** F4's
  pathology fires on 0 of the 217,756 rows R6 fitted and scored; the 5,814,302
  on record belongs to a bigcodec sweep with no local pixels. So a rev1-vs-rev2
  delta measured on the eval corpora is **F17 alone** (plus F5 at 944 widths),
  and any gate that reports "F4 moved N cells" on those corpora is reporting
  N = 0 by construction, which is an identity and not evidence.
* **F17 fires everywhere.** It moves 12 basic slots at every width and every
  pool state, on five distortion corpora and the training leg.

## 1. THE GPU COPY — pre-registered decision rule

`zenmetrics/crates/zensim-gpu` is the **fourth hand-copy** of this arithmetic
(zensim's own three are now behind one owner each — `ssim_form`, `hf_gain_form`).
Per NO DUPLICATE IMPLEMENTATIONS it must either move with the revision or refuse
to serve it. The rule, fixed before looking at how hard it is:

* **Port it** if the change is expressible without inventing new plumbing this
  crate has never used, AND a parity test can actually be RUN — on a real GPU,
  or on CubeCL's own CPU runtime, which is a real execution of the same kernels.
* **Refuse it** — a loud, named error at construction, plus pinning the wave's
  oracle to the CPU walk — if either half fails. A GPU path that silently serves
  rev1 numbers to a rev2 caller is the exact defect class this whole revision
  exists to remove.

`zensim-gpu` is a **sanctioned gated mirror**, not a stray duplicate: it can
never call zensim's CPU kernels, and `cpu_parity` / `extended_parity` /
`cpu_gpu_feature_sweep` hold it exact against them. The mirror therefore also
mirrors the **revision selection**, with the same `ZENSIM_FORMULA_REV` spelling
and the same fallback, rather than reaching for a new public API in zensim
(the public surface is closed without approval).

**G-GPU.1** — rev1 through the GPU path is **byte-identical** to today's GPU
path, on every fixture the existing parity suite covers. The revision switch is
not allowed to cost revision 1 a single bit.
**G-GPU.2** — the ported slots agree with the CPU rev2 walk to the suite's
existing tolerance, on a run that actually executed, on inputs where the ported
arithmetic **differs** from rev1 (a fixture where `contrast_inc > 0`). A test
whose inputs never enter the changed regime proves nothing and is reported as
NOT MEASURED.
**G-GPU.3** — anything the GPU cannot serve at rev2 **refuses loudly** and names
itself; it never silently falls back.

## 2. THE IMAGE

Built from zensim `main@origin` (the rev2b tip) + zenmetrics `master@origin`
(the BMP arm), as a new **tag on the canonical `ghcr.io/imazen/zenfleet-worker`
package**. The plan it inherits (`PLAN_REV2_RECALC` §3 G-IMAGE) applies
unchanged. Two gates are added here because the wave is a *revision* wave:

**G-IMG2.1 (rev1 reproduction, re-run on the NEW build).** The new image's
extractor at `ZENSIM_FORMULA_REV=1` reproduces the stored postC 372 root
**bit-exactly** — `to_bits()`, zero cells differ. A new binary is not trusted on
the strength of the old binary's gate.
**G-IMG2.2 (rev2 reproduction against the local R6b tables).** The new image's
extractor at `ZENSIM_FORMULA_REV=2` is **bit-exact** against the rev2 tables the
R6b lane extracted locally, on the same pairs. This is the gate that proves the
fleet and the local lane are the same era; without it the wave's rows are not
comparable to the numbers the arm was chosen on.

**Both must read 0 cells differ before a second box is launched.** A wave whose
first chunk fails either is stopped, not rounded.

**A named hazard, recorded before it can bite:** the previously-built
`exec-featjobs-*` tag advertises `feature-jobs` and therefore **can claim Feature
cells**, but predates the BMP decode arm — so it would fail every LIVE cell, and
`encoder_panic` is deterministic, so the reconciler poisons on the first failure.
Only the new tag may be enrolled for this wave; the roster is checked before
launch, not after.

## 3. THE WAVE — order, and what each leg is worth

Priority is the user's product order intersected with "what has pixels a worker
can reach". Every leg is declared with `revision=2` and the **declared**
`feature_set_id` echoed into every row (never re-derived worker-side).

| # | leg | rows | pixels | why here |
|---|---|--:|---|---|
| a | 372 eval corpora (8) + instruments + probes + anchor | 43,870 + inst. | LAN store (staged) | every gate and every verdict reads these |
| b | fast-class training legs (safesyn) | 196,086 | bitstreams, local | the refit's training input |
| c | KADIS negrich, then full | 167,034 / 700 k | distorted on object store | corruption head + negative tail |
| d | bigcodec picker TEST views first | 5.74 M | object store | capacity-bound; last |

**Pre-registered efficiency rule for (b):** the R6b lane already extracted a
safesyn leg at rev2 **locally**. It is **verified, not recomputed** — a fleet
sample is compared `to_bits()` against it, and only if the sample diverges is the
whole leg re-run. Registering the local table as canonical on a passing sample is
cheaper by ~196 k rows and is exactly as defensible, because the sample is the
same gate the whole leg would have faced.

**G-WAVE2.1** — every emitted table's `_MANIFEST.json` carries `build_commit`,
`feature_set_id`, `formula_revision`, **decoder era per format**, per-file
sha256, and row counts. Decoder era is not optional: it was measured at 73 % of
an extractor era, so a table that does not record it cannot be compared with one
that does.
**G-WAVE2.2** — outputs are mirrored to the LAN store **and** Tower with a sha
manifest before any cleanup.
**G-WAVE2.3** — the 6 byte-COPY corpora (aic4, nonphoto, imazen26, sdr25,
hfnlproxy, hf_nearlossless) are excluded from every "N corpora recalculated"
count. A zero delta there is an identity, not evidence.

## 4. THE REFIT — what is fitted, on what, graded how

Two arms, both fitted on **rev2** legs and graded on **rev2** instruments:

* **(i) the shipped D chain** — the ADD156 lineage that `ZensimProfile::D`
  resolves to today, reproduced from its own embedded `zentrain.repro` at both
  registered slices (`0..155`, `0..227`), with its post-training chain
  (spline / winsor / pack) unchanged except for the inputs' revision.
* **(ii) the fast-class candidate `S372_S228_H128_p`**, k=3 split seeds.

Graded against **the rev1 incumbents on the rev1 instruments** — i.e. each side
read on its own era. **A rev1 number and a rev2 number are never differenced
directly**; the era shift is model-specific (measured: exactly 0.00000 for a
basic-block-only bake vs |Δ| 0.489 for another on the same corpus), so a
cross-era subtraction is not a delta, it is two rulers.

## 5. SHIP GATES for installing a rev2 bake into `ZensimProfile::D`

All of these, or the result is a proposal with a table and no install:

* **G-SHIP.1 rank** — CID22 ≥ today's D (0.8633) with a bootstrap CI stated,
  and no corpus outside its CI-stated noise.
* **G-SHIP.2 dial contract** — G-ADDR **6 of 6 contract rows**, graded against
  the `peer_ssim2` pins. The ssim2 TRUTH is a property of ssim2 and does not
  move with a zensim feature revision; the **dial** values do, so the dial half
  is re-read at rev2 and the reference half is not re-derived.
* **G-SHIP.3 floors** — resolvable floors on all 5 codecs.
* **G-SHIP.4 inversions** — two-reference inversion check.
* **G-SHIP.5 perf** — W4 unchanged. The revision adds no kernel cost (one
  `max` and one divide on already-loaded values, both outside the pixel loop for
  F17); this is **stated as the expectation and then measured**, not assumed.
* **G-SHIP.6 corruption head** — a rev2 head refit on rev2 features, reported at
  T=0.9 and T=0.95 against the rev1 head, with the gate pass rate measured **with
  the head in front of the rev2 D**, and honest-cell FP per codec.
* **G-SHIP.7 hygiene** — weights ≤ 30 KB, manifest, `profile.rs`, tests +
  clippy + fmt + public-API zero delta, CHANGELOG + docs + annotations + ledger,
  board cells under a **distinct rev2 era suffix**, board regenerated, gates run.

`SHIPPED_REVISION` remains a per-bake declaration: installing a rev2 D does not
move C / CHdr, which stay rev1 pending the user's own toggle decision.

## 6. What would make THIS lane wrong

* Declaring the wave without re-running the rev1 gate on the NEW binary — the
  old gate proves the old binary.
* Comparing a rev1 number with a rev2 number as if the difference were a delta.
* Counting byte-copy corpora in a recalculation claim.
* Shipping a GPU rev2 path whose changed arithmetic was never executed.
* Letting a stale `feature-jobs`-capable image claim cells in this wave.
* Installing a bake because most gates passed. The install rule is all of §5.

---

# 7. RESULTS

## 7.1 ⛔ G-IMG2.1 FAILED — and it found a defect nobody was looking for: **the extractor's output depends on which libc the binary was linked against**

The gate §2 pre-registered (rev1 reproduction, re-run on the NEW binary) **failed on
the first try**, and chasing it produced the most consequential finding of this lane.

**What failed.** The fleet binary — `x86_64-unknown-linux-musl`, built from zensim
`88477e38` + zenmetrics `master@origin` — against the stored postC 372 root:

| corpus | rows | cells | differ | worst \|Δ\| |
|---|--:|--:|--:|--:|
| csiq | 866 | 322,152 | **77** (0.0239 %) | 1.11e-16 |
| tid | 3,000 | 1,116,000 | **328** (0.0294 %) | 1.11e-16 |

Every delta is exactly **one ULP at f64**. That is not a formula difference.

**What it is NOT — measured, not reasoned.** The obvious suspect was source drift
between the root's `build_commit` (`4fbd8ff8`) and `main@origin` (`88477e38`) — 163
commits including the whole revision-2 refactor, the feature-system phases and two
`blur.rs` perf lanes. Rebuilding the executor with zensim pinned at **`4fbd8ff8`
itself** reproduced **the same 77 cells**. So the drift is not in the source.

**What it IS.** The same tree built for the **native glibc** target reads
**0 of 322,152 cells differ**. Two conclusions follow, and the first is a
substantial positive result this lane did not set out to produce:

* **zensim `4fbd8ff8` → `88477e38` moves ZERO rev1-372 feature bits.** The
  revision-2 selector refactor, `feature_defs`/`Layout`/`Plan`, the servability
  work and both kernel-lane perf commits are bit-exact for revision 1 at 372,
  verified on 866 real pairs × 372 slots against a root built before any of them.
* **The musl build and the glibc build of the SAME source disagree.**

**Localisation — the slot pattern names the function.** Over csiq, 66 distinct
slots differ, spread across all four blocks (basic 19 / peaks 14 / masked 16 /
iw 17). Inside the basic block the differing positions are **exactly
`≡ 1, 4, 7 (mod 13)`** — which are precisely the three slots computed as
`(Σx⁴/n).powf(0.25)` (`feature_v2.rs:5400/5403/5406`, `streaming.rs:580/583/585`).
The peaks block's analogues are `.powf(0.125)` (`feature_v2.rs:5316-5318`), and
masked/IW carry the same L4 family. **Every other slot in the v1 path uses `sqrt`,
which is IEEE-correctly-rounded in hardware and therefore libc-independent.**

**Mechanism, confirmed by a probe with nothing to do with zensim.** 400,000 random
doubles spanning the magnitude range of a pooled 4th raw moment, `x ** 0.25`
evaluated through each libc's `pow`:

| pair | mismatches / 400,000 |
|---|--:|
| glibc 2.43 (Ubuntu 26.04, this box) vs glibc 2.36 (Debian bookworm) | **0** |
| glibc vs **musl** (alpine) | **276 (0.069 %)** |

First musl divergence: `57076.535008512925 ** 0.25` → glibc `15.456615376437254`,
musl `15.456615376437256`. Predicted feature-cell rate from that: 0.069 % × (144
L4/L8 slots ÷ 372) ≈ **0.027 %**, against the observed **0.0239 %**. The mechanism
is closed.

## 7.2 The defect, stated as a defect

**`powf` is not correctly rounded and is not specified to agree between libc
implementations. The v1 extractor calls it on 144 of its 372 slots. Therefore the
same pixels produce different features depending on how the binary was linked** —
and the fleet links musl (static, deliberately, so a worker is immune to the base
image's glibc version) while every local table on this box was produced with glibc.
A wave that mixed them would silently carry two eras, which is the exact defect
class revision 2 exists to remove.

**Blast radius, structural: 144 slots at width 372** — the three L4 slots per
(scale, channel) in each of basic / masked / IW (36 each) plus the three L8 peaks
slots per (scale, channel) (36). It is a different 144 from revision 2's combined
144; the two sets overlap only where a bounded SSIM term feeds an L4 pool.

**The fix is derived, not chosen.** `x^(1/4) = sqrt(sqrt(x))` and
`x^(1/8) = sqrt(sqrt(sqrt(x)))`. `sqrt` is correctly rounded and hardware-implemented
on every target this crate builds for, so the composition is bit-identical
everywhere — and it is also *more* accurate than a `pow` call, as well as cheaper.
There is no arm to select: for these two exponents the replacement is unique.

**It is NOT landed here, and that is deliberate.** Changing it moves revision-1
bytes, so it is an era break and needs the same treatment F4, F5 and F17 got — a
registered era token, a runtime-selectable owner, a measured blast radius and a
decision recorded before the numbers exist. Registering it as a fourth era of
revision 2 would also invalidate the R6b lane's already-extracted rev2 tables.
**Registered here as a defect with its mechanism, its exact blast radius and its
derived fix; the era registration belongs to zensim's revision lane.**

## 7.3 What this lane did instead, so the wave is not blocked

The wave needs a fleet binary whose output is bit-identical to the local glibc
tables it will be joined against. Two facts make that reachable without touching
any arithmetic:

* **glibc 2.36 and glibc 2.43 agree bit-for-bit** on `pow(x, 0.25)` (§7.1's probe,
  0 of 400,000). So a binary built against the executor base image's own glibc
  (Debian bookworm, 2.36) reproduces this box's glibc output exactly.
* The gate is the proof, not the argument: the bookworm-glibc binary is put through
  **the same G-IMG2.1** before it is allowed to claim a cell.

**The musl binary is not thrown away and not silently retired** — it remains the
correct choice for any wave whose tables are never compared with a glibc-produced
one, and it is the reason the finding surfaced at all.
