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
