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

## 7.9 ⚠ TWO rev2 tags exist on the canonical package. Only ONE may be enrolled.

| tag | libc | rev1 gate vs the postC root | use |
|---|---|---|---|
| `exec-featrev2glibc-88477e38` | **glibc** (bookworm 2.36) | **PASS — 0 of 6,092,616 cells** | **the wave's image.** Every table in §7.4–§7.8 came from it. |
| `exec-featrev2-88477e38` | musl (static) | **FAIL — 77/322,152 csiq, 328/1,116,000 tid** | **do NOT enroll for any wave whose tables are compared with a glibc-produced one.** |

The musl tag was built first, is pushed, and is deliberately **not deleted** — it
is the artifact that produced the §7.1 measurement, and deleting the evidence for
a finding to tidy a registry is the wrong trade. It is annotated here instead.
Both were built from the same source; the only difference is the target triple.

**A musl worker is not wrong in general** — it is the right choice for a wave
whose output is only ever compared with other musl output, and static linking is
why the fleet is immune to the base image's glibc version. It is wrong *here*,
because every stored root this wave must join was produced with glibc.

The older `exec-featjobs-*` tag remains the separate hazard §2 named: it advertises
`feature-jobs`, so it CAN claim Feature cells, and it predates the BMP arm, so it
would fail every LIVE cell deterministically and poison them. Neither of the two
tags above has that gap.

## 7.10 ⛔ THE TRAINING LEG DOES NOT TRANSFER: the fleet's AVIF decoder is a different era

safesyn — the 196,086-row training leg, 16.6 GB of bitstreams — was staged to the
LAN store (`s3://codec-corpus/safesyn-rev2-2026-09-06/`, 196,086 images + 3,218
sources, complete after a retry pass for **one source whose filename contains
spaces**, which s5cmd's whitespace-split `run` format had dropped: 57 objects) and
extracted at revision 2 by four boxes (`rev2safesyn372-20260906`, 7,407 jobs).

Against the R6b lane's LOCAL `satexcess` safesyn table — same pairs, same
revision, row alignment verified **0 of 196,086 mismatched on BOTH `ref_basename`
and `human_score`** — the fleet table differs on **37,379,073 of 72,943,992 cells
(51.2 %)**, worst |Δ| **0.136**.

That is not the extractor. Broken down by the distorted side's container (52
sampled basic slots per row):

| container | rows | rows differing | worst \|Δ\| |
|---|--:|--:|--:|
| `.jpg` | 111,068 | 63.8 % | **1.11e-07** |
| `.jxl` | 26,362 | 64.5 % | **1.04e-07** |
| `.webp` | 24,655 | 63.9 % | **8.57e-08** |
| **`.avif`** | **34,001** | **99.3 %** | **0.0217** |

**JPEG, JXL and WebP agree to ~1e-7 — rounding. AVIF disagrees by 0.0217 on 99.3 %
of its rows, five orders of magnitude larger.** The eval corpora never saw this
because their distorted sides ship as decoded PNG/BMP; safesyn is the first leg in
this wave that decodes real bitstreams in-process, and it is where the two decode
paths part.

**So the fleet cannot currently produce a training leg that is interchangeable
with the local one.** `zensim`'s `shared/zen_decode.rs` (at `e34f937d`, the era the
R6b tables record) and `zenmetrics-cli/src/decode.rs` (at zenmetrics `master`) do
not resolve to the same AVIF decoder — and zenavif is under an explicit hold for a
backend rewrite, so this is exactly the moving part you would expect. Which one is
"right" is NOT decided here and must not be guessed: both are imazen decoders, and
the answer is whichever era the consuming tables were built in.

**Consequences, stated so nobody has to rediscover them:**

* The fleet safesyn table is published as its own era —
  `/mnt/v/zen/zensim-training/2026-09-06-safesyn-rev2/` — and **must not be mixed
  with the R6b gram or any table built from it.** The refit lane's gram stays the
  local one; nothing about its A/B changes.
* A fleet-extracted training leg needs the AVIF decoder **pinned** to the era of
  the tables it will join, and that pin has to be a declared, checked property of
  the image, not an accident of which commit it was built from. Today it is an
  accident.
* This is the same class as §7.5's CID22 finding — a decoder difference invisible
  in the provenance — but two orders of magnitude larger, and it lands on the
  TRAINING side rather than the eval side, where nothing downstream would have
  flagged it.
* **The eval roots in §7.4/§7.6/§7.8 are unaffected**: their distorted sides are
  PNG/BMP, and six of seven comparable corpora are bit-exact.

## 7.11 The GPU copy — MEASURED on a real Vulkan device, and a correction to its own count

**Decision taken: PORT, not refuse** (§1's rule), and G-GPU.2 is **MEASURED**, not
waived — by an instrument better than either option the brief offered. This box has
Mesa **lavapipe** (Vulkan 1.4), so the `wgpu` backend runs the actual CubeCL
codegen path — WGSL → naga → SPIR-V → Vulkan dispatch — and only the final
SPIR-V→machine-code step is LLVM-on-CPU. That is categorically different from
`cubecl/cpu`, which bypasses GPU codegen entirely and would have proved nothing.

| gate | result | evidence |
|---|---|---|
| **G-GPU.1** rev1 byte-identity | **PASS** | 29,700 `to_bits()` values, PRE-PORT binary vs post-port at rev1, `cmp` clean — 3 sizes × 5 fixture pairs × 3 regimes × {cold, warm-ref} + strip + PU-HDR. Negative control: the pre-port binary is revision-blind. |
| **G-GPU.2** rev2 agreement, executed | **PASS** | Full 114-test suite on lavapipe. rev1 104/6; rev2 (`ZENSIM_FORMULA_REV=2`, so CPU zensim and the GPU read the SAME variable) 104/6 with an **identical failure set**, using the suite's own tolerances with nothing invented. **Negative control:** the pre-port tree at rev2 fails **17 MORE** — `cpu_gpu_feature_sweep` ×12, `cpu_parity` ×2, `extended_parity`, `odd_dim_320x241`, `pu_xyb_parity` — exactly the divergence the port removes. |
| **G-GPU.3** loud refusal | **PASS** | `ensure_diffmap_state` returns `Result` and refuses a revision it cannot serve; tested both directions plus the served case. |

**Not covered, and named rather than glossed:** hardware-vendor f32 behaviour (fma
contraction, denormals). The changed arithmetic is one `max` and one divide on
already-loaded values, so vendor sensitivity is not expected — but that is
reasoning, not measurement. `scripts/zensim_gpu_rev2_gate.sh` closes it in one
command on r7900x's GTX 1060; it was not run because that box was executing this
wave's safesyn leg, and a contended GPU run is worth less than an honest gap.

**F5 is structurally absent**, read from source and pinned by a test:
`ZensimFeatureRegime` is Basic 228 / Extended 300 / WithIw 372, with no raw-moment
accumulator, no `GLOBAL_*` slots and no append kernel.

### Three things §1 of this plan got wrong

1. **EIGHT F4 sites, not four.** `kernels/diffmap.rs` holds **four** more — the
   three channels of `per_scale_weighted_ssim_kernel` plus its host-scalar
   reference — and zensim's CPU diffmap routes through the same `ssim_form` owner
   as the feature walk, so leaving them would have served rev1 pixels to a rev2
   caller: the exact G-GPU.3 defect. The sum, spelled out so the next reader can
   check rather than re-derive it: **`fused` 2 + `masked_iw` 1 +
   `masked_iw_strip` 1 + `diffmap` 4 = 8.** (`diffmap.rs` contains a FIFTH
   `mu_diff`, at `:797`, which is the `#[cfg(test)]` rev1 control listed below and
   is deliberately not ported — counting it here is what produced the retired
   NINE.) *(The landed commit message says NINE; that count is
   wrong and is corrected here. The CHANGELOG states no count and is accurate.)*
2. **`per_scale_weighted_ssim_kernel` DOES have a launch site** — the plan said the
   only unlaunched kernel was `masked_iw_kernel`, which is true, but it implied the
   diffmap kernels were out of scope. They are not.
3. **The anti-vacuity trap is structural, not probabilistic.** §1 worried a fixture
   "might not enter the changed regime". Worse: **F4 cannot be reached from 8-bit
   sRGB at all** — 0 of 29,700 values move. The **PU-XYB (HDR)** route is what makes
   it live: 192 SSIM-derived values move, `ssim_max` **5.4275 → 1.0**. Every SDR
   fixture is vacuous for F4 by construction, which is also *why* R6 found no
   pathology in 217,756 rows — every R6 corpus is SDR.

### Two remaining hand-copies, both accounted for

* `diffmap.rs:797`, inside `#[cfg(test)]` — a hand-mirror that pins the **rev1**
  form against the rev1-pinned wrapper. Deliberately NOT ported: porting it would
  delete the control. Same shape as zensim's own `legacy_scalar`.
* `examples/b_channel_diagnostic.rs:182` — `#[allow(dead_code)]`, on no shipped or
  scored path, and missed by the original survey because that survey grepped only
  `src/`. Its own comment says the next investigator will take it **verbatim**,
  which is precisely how a revision-1 copy walks into a revision-2 investigation.
  Now carries a comment naming the revision it computes and how to port it.

**Method note, transferable:** distrusting the plan's site count was right; a single
`grep` over `src/` was not enough either, and a count re-read off one's own list is
not a count. The check that worked was run only at verification time — enumerate
every occurrence on the LANDED tree and classify each as ported /
deliberately-not / missed.

## 7.12 Final wave inventory

Five job sets, **all drained clean**, on four LAN boxes with no paid cloud:

| job set | regime | rev | jobs | rows | boxes | wall |
|---|---|--:|--:|--:|--:|---|
| `rev2feat372-20260906` | 372 | 2 | 3,907 | 43,870 | 1 | ~2.5 min |
| `rev2feat944-20260906` | 944 | 2 | 3,907 | 43,870 | 4 | ~3 min |
| `rev1feat372-20260906` (CONTROL) | 372 | 1 | 3,907 | 43,870 | 2 | ~5 min |
| `rev2safesyn372-20260906` | 372 | 2 | 7,407 | 196,086 | 4 | ~6 min |
| `rev2safesyn944-20260906` | 944 | 2 | 7,407 | 196,086 | 4 | ~11 min |
| **total** | | | **26,535** | **523,782** | | **~28 min** |

Five roots on disk, all mirrored to `s3://zentrain/eval-roots/<name>/` and
`tower:/mnt/user/coefficient/output/zensim-archive-2026-09-06/`, each with a
`_MANIFEST.json` (`build_commit`, `feature_set_id`, `formula_revision`, decoder era
per format, per-file sha256, row counts):

* `2026-09-06-full-features-372-rev2` — 8 corpora, 43,870 rows
* `2026-09-06-full-features-944-rev2` — 8 corpora, 43,870 rows
* `2026-09-06-full-features-372-rev1-fleet` — the control, 43,870 rows
* `2026-09-06-safesyn-rev2` — 196,086 × 375
* `2026-09-06-safesyn-944-rev2` — 196,086 × 947, **1.05 GB**

**Pixels staged and reusable on the LAN store**: `eval372-rev2-2026-09-06/`
(124,742 objects, 25 GB) and `safesyn-rev2-2026-09-06/` (199,304 objects, 16.6 GB).

**Not attempted, and why**: KADIS distorted (R2-only; needs scoped temp creds and a
staging pass) and bigcodec (5.7 M rows, unstaged). Neither is compute-bound — one
box did 43,870 rows in 2.5 minutes. **The harvester is the next limit, not the
fleet**: `--feature-corpus` materialises every column as a Python list, so peak RSS
is ~`rows × features × 32 B` — MEASURED at ~12 GB for 196,086 × 944, which is fine
here and is NOT fine for bigcodec. A chunked `pq.ParquetWriter` is the fix and is
named in the tool's own header rather than left to be discovered.

## 7.13 The 944 root is USABLE by a real 944 bake — verified, with one cosmetic gap

Scoring shipped Profile C (`c_sdr_purity944_2026-08-29.bin`, 667 read ids, highest
`f941`) against `2026-09-06-full-features-944-rev2`: `bake_verdict` **derives** the
width as 944, reads the declared
`basic+peaks+masked+iw+v2+append+append2@w944/era2r4_rev2#b782e349` out of the
manifest, and proceeds. No refusal. The only note is
`EraUnknown: era not established (bake unknown, table era2r4_rev2)` — a property of
that *bake*, which declares no era, not of the root.

**One cosmetic gap, named rather than fixed:** none of the five new roots is listed
in `zensim_validate::eval_roots`, so every run prints `era UNKNOWN (not a
registered root)` before going on to read the era correctly from the
`_MANIFEST.json`. Harmless for a root passed explicitly — the feature-set line
immediately below it resolves — but registering the paths there would make a
flagless run self-describing, which is the whole point of that owner. A small
additive change in zensim, left for the lane that owns that file.

## 7.14 The fast-class arm at rev2 — INERT, and a method finding that outranks it

> **Evidence for §7.14, §7.15 and §7.16 lives in the refit lane's own record**, not
> here: zensim **`benchmarks/rev2_refit_2026-09-06.md`** (`main@origin`
> `d244c245`) — the per-codec tables, the pre-registration and its amendment, the
> gate-by-gate verdict, the bake shas and the confounds — with artefacts at
> **`/mnt/v/output/zensim/rev2-refit-2026-09-06/`** (277 files, per-file sha256,
> `build_commit`, `formula_revision`, decoder era per format, the BLAS pin, the
> libc). The three sections below summarise; that document measures.


**Result: revision 2 is inert on the fast-class model class.** k=3 (seeds
4004/4005/4006), rev1-in-era → rev2-in-era, identical recipe and pack chain:

| corpus | rev1 | rev2 | Δ | **seed spread** |
|---|--:|--:|--:|--:|
| CID22 | 0.88885 | 0.88854 | −0.00031 | **0.00600** |
| KonJND | 0.48878 | 0.49097 | +0.00219 | **0.02712** |
| AIC-3 | 0.79854 | 0.79625 | −0.00229 | **0.00745** |
| CSIQ | 0.96051 | 0.95866 | −0.00185 | **0.00492** |
| LIVE | 0.96188 | 0.96316 | +0.00128 | **0.00532** |
| TID | 0.94881 | 0.95071 | +0.00190 | **0.00253** |
| KADID *(train==val)* | 0.94212 | 0.94365 | +0.00153 | **0.00322** |

**Every delta is smaller than the seed spread of the arm it is measured in.** It
remains NOT SHIPPABLE at both revisions (contract 4/6 — C5 fails with all 400
identity rows outside the band, identity dial 94.1999; A7r fails 5 of 5), and the
one axis that moves gets **worse**: C6 goes 441 → 946 grid cells out-scoring a
perfect copy. It stays an arm, not a candidate — where §4 already had it.

### ⛔ The method finding: a single-seed paired bootstrap is confidently wrong about its own sign

Seed-**matched** paired bootstraps (B = 2,000, seed 20260905, s4004 vs s4004 …) of
the *same* rev1-vs-rev2 comparison:

| seed | CID22 Δ | 95 % CI |
|---|--:|---|
| 4004 | **−0.00113** | [−0.00217, −0.00008] |
| 4005 | **+0.00240** | [+0.00129, +0.00349] |
| 4006 | **−0.00215** | [−0.00311, −0.00133] |

**The sign flips across seeds and all three CIs exclude zero.** One experiment
that, read at a single seed, would have reported a CI-excluding LOSS on two seeds
and a CI-excluding WIN on the third. The mechanism: a paired bootstrap resamples
**pairs**, so it cannot see **initialisation** variance — and on this architecture
that is the larger term.

**The converse is what makes this actionable rather than merely cautionary.** A
**deterministic** fit with no seed — the convex lasso chain the ADD156 D lineage
uses — has no initialisation term, so its paired-bootstrap CI *is* credible. That
is exactly why the D chain's rev2 **CID22 −0.00456 [−0.00549, −0.00364] stands as a
G-SHIP.1 failure** while this one does not. Registered in zensim
`benchmarks/eval_annotations.json` as
`single-seed-paired-bootstrap-cannot-see-init-variance-2026-09-06` (`f9c3643a`),
with the rule: report a seeded class's rank delta as a k-seed mean beside its seed
spread, and reserve single-arm bootstrap CIs for deterministic fits. **Not enforced
in tooling.**

**Free control noted in passing:** `bake_dial_refit pack` is deterministic — all
three rev1 id100 bakes reproduced byte-identically across two independent
invocations.

## 7.15 G-SHIP.6 — the corruption head at rev2, and the one result that outlives this wave

**Stated first, because it is what a reader needs:** the outcome that could have
changed the install decision — a rev2 head materially better than the rev1 head,
making the revision worth its rank cost — **did not happen.**

Corpus gated before use: 117,276 rows, `ref_id` / `family` / `is_corruption` /
`severity` / `region` / `kind` / `content_class` **positionally identical** to the
rev1 twin, exactly the twelve F17 slots moved, the closed form `g/(g+1)`
reproducing the extraction to **4.3e-8**, `d228` parity passing on the same folds.
BLAS pinned at 4 threads and recorded (the head's bake is thread-dependent until
that defect is fixed at its owner).

| arm | T | detection rev1 → rev2 | ladder FP | near-lossless FP |
|---|---|---|---|---|
| logistic | 0.9 | 86.06 → **90.98** | 11.27 → **14.75** | 49.40 → **55.95** |
| logistic | 0.95 | 77.44 → 76.67 | 6.81 → 7.27 | 35.71 → 37.20 |
| **hgb** | 0.9 | 98.90 → **99.04** | **1.23 → 1.23** | **2.38 → 2.38** |
| **hgb** | 0.95 | 97.91 → 97.66 | 0.36 → 0.31 | 2.08 → 1.79 |
| hgb_drop8 | 0.9 | 97.77 → 97.32 | 0.46 → 0.31 | 2.08 → 1.79 |
| hgb_drop8 | 0.95 | 96.52 → 97.18 | 0.31 → 0.31 | 1.79 → 1.79 |

DEPLOY **`pass_q20`** (the headline column): logistic 91.07 → 95.98, **hgb 99.85 →
99.85**, hgb_drop8 99.55 → 99.11; D dial alone 23.07 → 22.32. **`pass_q10` reads
12.50 → 12.35 for every arm at every threshold on both eras** — the dial-alone
value, exactly as §7.14's registered caveat predicts, because the q10 anchor's dial
is −4.964 at rev1 and −6.128 at rev2, both below the floor. **Predicted before the
run, then confirmed.**

### ★ The tree is INVARIANT to the revision, to nine decimals

At T = 0.9 the HGB head's ladder-FP delta is **exactly 0.000000000**, its
near-lossless-FP delta is **exactly 0.000000000**, and all five per-codec cells are
identical to 9 dp. **Operationally: the corruption head does not need refitting when
the revision flips.** That removes it as a blocker from any future rev2 attempt —
worth more than any single number in the table above.

### A prediction that was WEAKENED before the fact, then half falsified anyway

*(Corrected after the first draft of this section credited the prediction with more
than it claimed. The ordering is the point — not who caught it.)*

The lane's **first** draft predicted that the logistic would gain and **the tree
would be invariant**, because the logistic is linear in a value the ±8 standardised
clip destroys. It **narrowed that to the RELATIVE claim alone — "the logistic gains
more than the tree" — BEFORE seeing any number**, on the reasoning that the ±8 clip
applies to the tree as well, so tree-invariance was not claimable a priori. The
measurement then confirmed the narrowed version and falsified the other half.

**A prediction weakened before the fact is a different object from one that was
right**, and a record that blurs them turns a narrowed claim into a vindicated one.
What survives here is the narrowed claim.

**The asymmetry is the whole argument for pre-registration, and it is worth stating
as a rule rather than as an anecdote.** Narrowing the claim cost nothing, because no
number existed yet — that is *only* true because it was written down before the run.
Leaving the flattering version standing after publication would have been expensive
and hard to undo. Cheap before, expensive after: that gap is what pre-registration
buys, and it is available to any lane willing to write the prediction down first.

**Relative claim (the narrowed one) HOLDS:** pAUC₅ moves **+0.54** for the logistic
against **−0.05** for the tree, 11×. **Directional claim FAILS:** the logistic does not improve in the sense that matters — at T = 0.9 its
+4.92 pt of detection is bought with **+3.48 pt of ladder FP and +6.55 pt of
near-lossless FP**, worse on every codec (webp near-lossless 12.77 % → 31.91 %, jxl
47.62 % → 56.19 %). Bounding the feature moved where the linear threshold *lands*;
it did not make the head **separate** corruption from honest content any better, and
+0.54 on a 0–100 pAUC scale says so plainly.

Also measured: the corruption corpus carries revision-1 `contrast_inc` up to
**1.30e7** — higher than negrich's 1.22e7 — all of it pinned to that ±8 clip. F17's
unboundedness is larger on this population than anywhere previously recorded.

## 7.16 ⚠ A stale binary left on purpose, and the trap it sets

Two build dirs are deliberately **not** reclaimed —
`/home/lilith/work/zen/zensim/target-refit-rev2` (1.6 GB) and
`target-refit-rev2-ccorpus` (92 MB) — because the W4 runner may want a binary
matching the refit lane's numbers.

Those numbers, and what the binaries were built to produce, are in zensim
`benchmarks/rev2_refit_2026-09-06.md` — named here because a warning about a stale
artifact is useless without a path to the thing it belongs to.

**They were built at zensim `88477e38`, which is already several commits behind
`main@origin`.** So they are the **right** binaries for reproducing §7.15's numbers
and the **wrong** ones for measuring anything about current main. A stale binary is
exactly the artifact that gets reused without checking — it runs, it produces
plausible numbers, and nothing fails. Named here so that reuse is a decision rather
than an accident. Delete them freely once W4 is done; nothing references them.
