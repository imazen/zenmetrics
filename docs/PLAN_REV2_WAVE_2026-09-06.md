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

## 7.4 THE WAVE RAN — 43,870 rows at revision 2, on one box, in ~2.5 minutes

Job set `s3://zentrain/jobs/rev2feat372-20260906`: **3,907 `JobKind::Feature` jobs**,
`regime 372`, `revision 2`, `feature_set_id
basic+peaks+masked+iw@w372/v1postc_rev2#d16a1091` declared once and echoed into
every row. Executed by ONE worker (`r7900x-feat`, 24 cores,
`ZEN_CORE_OVERSUBSCRIBE=1`) on the glibc image, claimed 05:37:02Z and drained
clean (exit 0) with **3,907 of 3,907 blobs** written.

| corpus | rows | vs postC at **rev1** | vs r6b `satexcess` at **rev2** |
|---|--:|---|---|
| csiq | 866 | **0 differ** / 322,152 | **0 differ** |
| tid | 3,000 | **0 differ** / 1,116,000 | **0 differ** |
| konjnd | 1,008 | **0 differ** / 374,976 | **0 differ** |
| live | 779 | **0 differ** / 289,788 | **0 differ** |
| aic3 | 600 | **0 differ** / 223,200 | **0 differ** |
| kadid | 10,125 | **0 differ** / 3,766,500 | **0 differ** |
| cid22 | 4,292 | **FAIL** 191,378 / 1,596,624 — see §7.5 | **536 rows differ** — §7.5 |
| pipal | 23,200 | NOT COMPARABLE — the staged set is the 23,200-pair superset, the stored root holds 21,800 (the 7-per-ref cut the staging lane could not reconstruct) | not in the r6b root |
| **total** | **43,870** | **6,092,616 cells, 0 differ** (the 6 comparable corpora) | **6,092,616 cells, 0 differ** (6 corpora) |

Both columns cover the same six corpora and therefore the same 6,092,616 cells.
cid22 fails identically in both columns and for the same reason (§7.5); pipal has
no comparator at either revision because the stored root's row cut has never been
reconstructed — that is a pre-existing open item, not something this wave broke.

**G-IMG2.1 and G-IMG2.2 both PASS.** The second one is the load-bearing result:
the fleet executor at `ZENSIM_FORMULA_REV=2`, running a different code path from
the local `extract_features_372col`, reproduces the R6b lane's `satexcess` tables
**bit-for-bit**. That independently confirms the composition argument the R6b lane
could only derive — that at width 372, revision 2 IS the `satexcess` arm, because
`Clamp` moves zero cells on these pixels and `freecomp` is unreachable without an
append block.

Root: `/mnt/v/zen/zensim-training/2026-09-06-full-features-372-rev2/`
(`_MANIFEST.json` carries `build_commit`, `feature_set_id`, `formula_revision`,
per-format decoder era, per-file sha256, row counts, and the pairs corrections
below). Mirrored to `s3://zentrain/eval-roots/…` and Tower.

**Two pairs-file defects the gate found and this root corrects**, both invisible
until something demanded bit-exactness:

* **konjnd — 4 of 1,008 rows named the WRONG distorted file.** The LAN-staged TSV
  derived the JND level with round-half-to-EVEN (Python's `round`); the stored
  root used round-half-UP. The four rows whose PJND is exactly `X.5`
  (`SRC0437` 58.5, `SRC0823`/`SRC0904`/`SRC0993` 30.5) therefore pointed at
  `_030`/`_058` instead of `_031`/`_059`. Feature deltas up to **0.068** — a real
  difference, not rounding. Re-extracting the four with the half-up file is
  bit-exact to the stored root, which proves the diagnosis rather than asserting it.
* **live — the staged TSV was in directory order**, not the stored table's order.
  A **stable sort by `basename(ref_path)`** reproduces the stored order exactly on
  all 779 rows (grouping AND within-group), as §7.5 of the recalc plan found.

## 7.5 ⛔ A SECOND DEFECT THE WAVE FOUND: 12.5 % of CID22 is decoded by a NON-IMAZEN decoder

cid22 is the only corpus where the fleet disagrees with the local roots, and the
disagreement is **exactly the JPEG rows**: cid22's distorted side is 3,756 `.png`
+ **536 `.jpg`**, and **100 % of the `.jpg` rows differ while 0 % of the `.png`
rows do** — ~360 of 372 slots each, max |Δ| **9.5e-3**.

Cause, traced to source rather than inferred: the postC root's cid22 / kadid /
tid / pipal legs are produced by **`zensim-validate --extract-only`**, which
decodes with **`image::open()`** — the third-party `image` crate
(`zensim-validate/src/main.rs:830, 854, 1720, 1744`). kadid (PNG) and tid (BMP)
are bit-exact against the fleet because both decoders produce identical pixels
for those formats; **JPEG is where they part.** `extract_features_372col` was
migrated to the imazen decoders on 2026-09-04 (`shared/zen_decode.rs`);
`zensim-validate` never was — the "extraction is not migration" failure this
project has already paid for once.

**Why it matters beyond this wave:** CID22 is the gold human-MOS validation
holdout that every model in the project is graded on, and 12.5 % of its rows are
therefore graded on pixels no imazen decoder produces. The r6b rev2 tables
inherit it (they were built by the same two producers), so an in-era A/B on them
is still single-confound and correct — but every published CID22 number is read
partly through a foreign decoder, and the `_MANIFEST.json` `decoder_era` string
names only `zen_decode.rs`, so the fact is invisible in the provenance.

**This root's cid22 is the imazen-decoder one.** The price of the defect — what a
bake's CID22 SROCC actually does across the two decode eras — is NOT measured
here; it needs one bake scored on both tables, and that is registered, not run.

## 7.6 The 944 wave — the first rev2 extraction at a width where F5 is REACHABLE

Job set `s3://zentrain/jobs/rev2feat944-20260906`: the same 3,907 cells at
`regime 944`, `revision 2`, id
`basic+peaks+masked+iw+v2+append+append2@w944/era2r4_rev2#b782e349` (the slots
hash is unchanged from its non-rev2 sibling — the era names the build class, the
hash names the slot set). Four boxes (r7900x 24c, r5900xt 32c, i265 20c, r3500
6c), all observed idle with no `zen*` container before launch,
`ZEN_CORE_OVERSUBSCRIBE=1`, **~3 minutes wall**, all 3,907 cells done.

This is the leg the 372 wave could not be: at 944 the append block exists, so
**all three revision-2 eras are live** (F4 at its 36 zeroed-pool slots, F17 at 12,
F5/`freecomp` at 11) instead of only F17. Nothing existed at rev2 × 944 before.

Root: `/mnt/v/zen/zensim-training/2026-09-06-full-features-944-rev2/`, mirrored to
the LAN store and Tower. Two gates, both MEASURED rather than inferred from the
regime string:

* **Zeroed pools.** `f156..371` nonzero count = **0** across csiq/konjnd/live/tid/
  aic3. The `folded720append2` shape is confirmed, not assumed.
* **Cross-regime basic block.** `f0..155` against the 372 rev2 root on the SAME
  pairs: 404,794 of 975,468 cells differ, **worst |Δ| 1.96e-8** (worst relative
  1.7e-3, on a value of 4.5e-6). That is buffered-walk vs streaming-fold
  accumulation order, not a formula difference — but it is NOT zero, so **the two
  roots must not be column-mixed.** Each is internally consistent; that is what
  the era token is for.

## 7.7 What this lane did NOT do, named rather than omitted

* **safesyn, KADIS and bigcodec are not in either wave.** safesyn's 47 GB of
  bitstreams are absent from the LAN store and KADIS's distorted PNGs are R2-only,
  so both need a staging pass before a worker can reach them; bigcodec is
  capacity-bound behind those. The fleet is not the bottleneck — one box did
  43,870 rows in 2.5 minutes and four did the 944 leg in three — **pixel staging
  is**, and that is the next lane's first task, not a compute problem.
* **The six byte-COPY corpora** (aic4, nonphoto, imazen26, sdr25, hfnlproxy,
  hf_nearlossless) are in neither root and are excluded from every count here.
  Their pixels are gone; a zero delta there would be an identity, not evidence.
* **The `powf` era is registered, not landed** (§7.2), and no arithmetic was
  changed by this lane.
* **The CID22 decoder defect is priced at zero so far** — nobody has scored a bake
  on both cid22 tables, so its cost in SROCC is unknown, not small.

## 7.8 The CONTROL wave, and what a rev2 flip costs a bake that is NOT refitted

A third wave — `s3://zentrain/jobs/rev1feat372-20260906`, the same 3,907 cells at
**revision 1** — was run so that every comparison has one variable. Root:
`/mnt/v/zen/zensim-training/2026-09-06-full-features-372-rev1-fleet/`. With it:

* **rev1-fleet vs postC** isolates the DECODER (same revision, same walk).
* **rev1-fleet vs rev2-fleet** isolates the REVISION (same producer, decoders,
  image and libc).

### The CID22 foreign-decoder defect, PRICED

§7.5 said the cost was "NOT measured; it needs one bake scored on both tables."
It is measured now, on all three shipped SDR bakes, CID22 SROCC:

| bake | postC (`image`-crate cid22) | rev1-fleet (all-imazen) | Δ |
|---|--:|--:|--:|
| Profile **D** (the SDR default) | +0.8633 | +0.8632 | **−0.0001** |
| Profile **B** | +0.8821 | +0.8821 | **0.0000** |
| Profile **A** (v47-QAT) | +0.8655 | +0.8654 | **−0.0001** |

**≤ 1e-4 on every shipped SDR bake.** So the defect is real, it is a genuine
provenance and reproducibility failure — 536 rows of the gold holdout are not
reproducible by any imazen-decoder extractor and the manifests misdescribe the
decoder — and it is **NOT a ranking defect**. It stays `annotated`, not
`invalidated`, in `benchmarks/eval_annotations.json`. The refit lane measured the
same quantity independently on its own rev2 D bake and got **2e-5**, ~1/200th of
the effect it was trying to see.

### The revision flip, unrefitted — SROCC delta rev1-fleet → rev2-fleet

| bake | CID22 | KonJND | AIC-3 | CSIQ | LIVE | TID |
|---|--:|--:|--:|--:|--:|--:|
| **D** | **0.00000** | **0.00000** | **0.00000** | **0.00000** | **0.00000** | **0.00000** |
| **B** | 0.00000 | −0.00120 | 0.00000 | −0.00010 | −0.00140 | +0.00010 |
| **A** (v47-QAT) | **+0.00030** | 0.00000 | 0.00000 | **+0.00090** | 0.00000 | **+0.00080** |

**The SDR default is exactly unaffected** — six corpora, zero to five decimals —
which is what you would expect of a 28-input bake whose only F17 exposure is
`f116` and `f155`. A moves slightly UP on three corpora; B moves ≤0.0014 in mixed
directions. This corroborates R6b's own serve-skew estimate (|6e-5| for D) with a
different instrument and extends it to B and A.

**It does NOT say the flip is free for a REFITTED bake.** The refit lane measured
the opposite sign in the D *chain* — revision 2 costs CID22 −0.00456
[−0.00549, −0.00364] at s156, CI-excluding, against R6b's +0.00272 on a denser
model class. Serving an existing bake on rev2 features and refitting a bake on
rev2 features are different questions with different answers.

### A defect in this lane's own output, recorded rather than quietly fixed

**PIPAL in all three new roots is FEATURES ONLY and is NOT SCOREABLE.** The
LAN-staged pipal pairs TSV carries `ref_path` + `dist_path` and no target column
at all — the only one of the eight that does — so the harvested parquet has no
`human_score` and `bake_verdict` refuses it. Deliberately not repaired by deriving
an elo here: a PIPAL normalisation that did not match `zensim-validate`'s own
`load_pipal` would be worse than the gap. Recorded in all three manifests; the fix
is to re-stage pipal's pairs with the target column.

### A tooling note worth one line

A root that declares a `feature_set_id` whose ERA is not yet in
`benchmarks/feature_sets_registry.json` makes `bake_verdict` refuse every bake with
`SlotsNotPopulated: the bake READS N slot(s) the table does not POPULATE`. The
slots ARE populated; the era is what could not be resolved. Rebuilding
`bake_verdict` after registering the era fixes it with no flag. The refusal is the
registry doing its job; only the message is misleading.
