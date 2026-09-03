# AVIF HDR-10 RD baseline — Track T2 analysis — 2026-09-03

Analysis of `avifhbd-t2a-fix-20260902` (svt) and `avifhbd-t2b-20260902`
(zenrav1e), the two blocks registered in
[`avif_hdr_arm_plan_2026-09-02.md`](avif_hdr_arm_plan_2026-09-02.md) §4.3 and
launched in its §10.4c / §10.4f. Companion doc:
[`avif_eradelta_analysis_2026-09-03.md`](avif_eradelta_analysis_2026-09-03.md) —
kept **separate**; the two waves share no corpus, no instrument and no pin, and
nothing joins across them.

**This answers exactly two pre-registered questions** (plan §5.1) and nothing
else:

- **Q5** — what is the rate/quality curve of 10-bit AVIF on HDR stills?
  *A baseline. No PASS/FAIL bar, by registration.*
- **Q6** — how do the two wired HDR AVIF arms differ?
  *A **contrast**, not a controlled comparison.*

---

## 0. TL;DR

<!--TABLE:TLDR-->

---

## 0b. INSTRUMENT AND CORPUS LIMITATION (reproduced verbatim, plan §3.4)

> **INSTRUMENT AND CORPUS LIMITATION (registered 2026-09-02).**
> On the SDR-paired track (T1) reference and both decodes are 8-bit, so the
> ssim2/zensim instrument is fully valid and the BD-rate numbers stand unqualified.
> On the HDR track (T2), **provided it is scored through `sweep --hdr`**, the path is
> f32 end-to-end and no 8-bit quantization occurs; if it is scored through
> `score-pairs --hdr` / jobexec, ssim2 and zensim are u8-shelled and the numbers order
> encodes only coarsely.
> **No result from this arm may be stated as a banding claim**, for two independent
> reasons, either sufficient alone:
> **(i) instrument** — on any 8-bit-quantized route, requantization does not merely hide
> the effect, it **manufactures the opposite one** (§3.5); and
> **(ii) corpus** — the encode-wired HDR references are **gain-map-reconstructed from
> 8-bit bases**, so their smooth gradients were 8-bit-quantized before reconstruction and
> the banding 10-bit prevents is **present in the reference**.
> A banding claim requires **both** a never-8-bit-quantized source (pool A EXR or
> si-hdr, i.e. TODO-2) **and** the PQ `--hdr` route (which does preserve 10 bits and does
> unlock BANDVIS on its 10-bit-calibrated PU constants). Until both hold, banding is
> **NOT MEASURED** — never a null, never a zero.
> **Do not cite HDR-VDP-2.2**: in-tree but unvalidated and unwired — its own header says
> "no number out of this crate should be published as an HDR-VDP-2 score"
> (`zenmetrics/crates/hdrvdp/src/lib.rs:18-49`, `publish = false`).

**Which branch of that limitation applies here: the FAITHFUL one.** The fleet
scoring path carries `7051921a` (faithful f32 fleet HDR scoring), and the
proof was re-run by this lane on one of this wave's own cells — §1.2 below.
So the "u8-shelled / coarse ordering" clause does **not** apply to the numbers
in this document, and the banding prohibition **does** — it is corpus-driven
here, not instrument-driven, and reason (ii) alone is sufficient.

---

## 1. Coverage and route

### 1.1 Runs

<!--TABLE:COVERAGE-->

### 1.1b Why the svt arm is `t2a-fix` and not `t2a` — this corpus is the worst
case for zenav1-svt #18

Worth stating because it is not incidental. #18 lived at the intersection of
**10-bit × multi-tile × low preset**, and T2-a sits in all three at once: every
cell is 10-bit PQ by construction; the 16 HDR references run **4000×2252
(9.0 MP) to 4284×5712 (24.5 MP)**, and applying #18's own tile-forcing
predicate (`width > 4096` OR sb-aligned area `> 4096×2304`) to them selects
**15 of 16** — the single exception, `1231_interiors` at 4000×2252, misses the
area threshold by 4 %; and four of its seven presets (0, 1, 3, 4) are
inside the 0–5 band where the directional HBD intra path `dr_predict_hbd` is
reachable. The original `avifhbd-t2a-20260902` was therefore *maximally*
exposed, which is why it was abandoned at 120/3,248 and re-declared as a fresh
run on the fixed binary rather than requeued.

The companion doc's arm-set C measures the same defect's blast radius on the
SDR side and finds the tile-forcing predicate selects the affected images with
zero false positives and zero false negatives, which is the evidence that the
restart was necessary rather than precautionary.

### 1.2 G5 — the route proof, re-run on this wave's own artifact

The gate is not satisfiable by absence of failure; it needs positive evidence
in the run log that the PQ `--hdr` route was taken. The registered
zero-instrumentation discriminator is that `--hdr-transfer` is **inert on the
faithful f32 route** while the u8 shell's own test asserts `|pu − pq| > 1e-9`.

Re-run inside the exact scoring image the fleet used
(`ghcr.io/imazen/zenfleet-worker:exec-avifhbd-t2fix-64252836`), on cell
`p0-zenav1svt-hdr10 / 1070_general_stone-temple-ruins… / q=1`
(`encode_sha 2eb3912d…`):

| invocation | ssim2 |
|---|---|
| `score-pairs --hdr --hdr-transfer pq` | `-113.52625369804514` |
| `score-pairs --hdr --hdr-transfer pu-rescale` | `-113.52625369804514` |
| **\|Δ\|** | **0.0** |

⇒ inert ⇒ **faithful f32 route.** And the value is **bit-equal to the score the
fleet stored for that cell**, so the stored corpus is on the same route, not
merely a hand-run of it.

**TODO-0's tripwire, also verified live on this wave's own reference:** the
same pair scored *without* `--hdr` is refused —

> `PNG signals an HDR transfer via cICP (transfer=16, PQ): refusing to crush it
> through the 8-bit SDR decode path — score it with --hdr instead`

— and `score-pairs` then exits non-zero rather than reporting an empty success.
A mis-routed T2 cell fails; it does not return a plausible number.

### 1.2b The quality column is `ssim2` — and a correction this lane owes the record

The T2 corpus was touched by **two different scoring images**: `avifhbd-t2b`'s
432 cells were scored 2026-09-02 by `exec-avifhbd-t2-89d0fb64`, and everything
scored 2026-09-03 (all of `avifhbd-t2a-fix`, **plus a full re-score of t2b**)
by `exec-avifhbd-t2fix-64252836`. The re-score gives 351 `(encode_sha, metric)`
pairs scored by *both* images, on identical bitstreams:

| metric | pairs compared | differing | max \|Δ\| |
|---|--:|--:|--:|
| `ssim2` | 351 | **0** | **0.0000000000** |
| `zensim` | 351 | 225 | **1.156e-7** |

**So both images agree**: `ssim2` bit-exactly, `zensim` to within ordinary
float non-determinism. There is **no material instrument era split inside this
corpus**, and the analysis does not depend on there being one.

> **⚠ CORRECTION — an earlier draft of this section claimed the two images
> disagreed on `zensim` by 11.14 points, from a single hand-run.** That was
> wrong, and the way it was wrong is worth keeping. The 11.14 gap
> (`88.96769457440139` stored vs `77.82319440419373` re-run) is real, but it is
> between the **fleet ScoreFile executor path** and a hand-run
> **`score-pairs --metric zensim`** — *not* between the two images. A third
> build reproduces the same shape in the other direction (local
> `zenmetrics-cli 0.6.0` returns `−44.47` where the fleet stored `−136.72`).
> The fleet's `zensim` comes from its 372-feature `with-iw` record
> (`kind:"feature"`, `regime:"with-iw"`, whose `zensim_score` equals the metric
> record's `score` exactly), while `score-pairs` scores with the CLI's shipped
> default profile unless `--zensim-profile` says otherwise. **`score-pairs
> --metric zensim` and the ScoreFile executor's `zensim` are different
> quantities and must not be compared** — which is a finding in its own right,
> and one that generalises past this arm.
>
> The n = 1 measurement was not wrong about its own cells; it was wrong about
> *what varied between them*. The 351-pair check is what distinguishes the two
> explanations, and it should have come first.

Two decisions follow, and only the first is load-bearing:

- **Every number in this document is `ssim2`.** That was already the DOE's
  designated corpus-wide scalar (plan §7.1) and it is now measured
  build-invariant across three builds. `zensim` is not used for Q5 or Q6.
- **The 48 pre-era score blobs are still EXCLUDED from the harvest** — now as
  hygiene rather than necessity. The 2026-09-03 wave re-scores every t2b cell,
  and the harvest is last-write-wins over a filename-sorted glob, so leaving
  both generations in would make each cell's provenance depend on a sha
  ordering. Excluding them makes the table one instrument end to end by
  construction; the measurement above says it would not have changed a number.
  The excluded key list is `t2_preera_blobs.txt` in the output dir.

### 1.3 G0.5 — the primaries × content cross-tab, published with every content claim

<!--TABLE:CROSSTAB-->

> **Registered analysis restriction (plan §4.3).** Primaries are nearly
> determined by content in this corpus. No content-class conclusion may be drawn
> without this cross-tab, and no primaries/gamut conclusion without the content
> one; where the two cannot be separated the finding is
> **"content-or-gamut, not separable in this corpus"**, never attributed to one.

---

## 2. Q5 — the RD baseline

<!--TABLE:Q5-->

---

## 3. Q6 — the backend contrast

**Three differences at once.** T2-a is `zenav1-svt`, **4:2:0**, BT.2020nc
matrix; T2-b is `zenavif`/zenrav1e, **4:4:4 GBR identity-PQ**. Any number below
is the sum of backend + chroma + matrix, and cannot be attributed to one of
them. The plan registered it as a contrast for exactly this reason.

**No preset ↔ speed mapping is asserted or used.** `speed_to_svt_preset` maps
zenavif's product speed dial onto svt presets *within the svt backend*; it says
nothing about rav1e's own speed scale. The analysis therefore reports a full
arm × arm matrix and a per-backend **Pareto envelope** (pool every dial setting
a backend offers on that image, take the frontier), which is the product-level
comparison that needs no cross-backend dial equivalence.

**The two ladders are not equally dense, and the envelope is not neutral about
that.** T2-a offers 7 presets × 29 quality points = 203 candidates per image;
T2-b offers 3 speeds × 9 = 27. A denser grid produces a frontier that hugs the
true RD curve more closely, so the envelope credits T2-a for *having more
operating points*, which is a real product difference but not the same claim as
"the encoder is better". Read the envelope together with the arm × arm matrix
below it: if a single svt preset beats a single zenavif speed by about the same
margin as the envelopes do, the advantage is the encoder; if the envelope gap
is much larger, part of it is ladder density. The registered framing —
a **contrast**, not a controlled comparison — covers this as well as the
backend/chroma/matrix confound.

<!--TABLE:Q6-->

---

## 4. Bit depth — what is and is not paired here

**Both T2 arms are 10-bit PQ. There is no 8-bit HDR arm, and none was
declared.** The plan's §4.3 table lists T2-a and T2-b only, both 10-bit; the
HDR encode path emits 10-bit by construction (T2-a via
`sweep/hdr.rs to_yuv420_bd10`, T2-b via `zenavif::encode_rgb16`, which at the
pinned commit `bcd7978` ignores `config.bit_depth` and always codes 10). So an
8-vs-10 **paired** read on the HDR corpus is **NOT MEASURED — never a null**,
and it is not a gap this wave left open: it was never in scope.

The 8-vs-10 paired reads that *do* exist live on the SDR track: T1 on the
budget corpus (`bd10`, plan §3.3 — "T1 carries no instrument limitation"), and
its native leg, whose clean re-run is
[`avif_eradelta_analysis_2026-09-03.md`](avif_eradelta_analysis_2026-09-03.md)
§4. Read the depth question there, not here.

**G3 (10-bit decode-verify) is inherited, not re-run.** Plan §10.4c records
T2-b 5/5 and T2-a 4/4 PASS on real fleet blobs — `av1C`, sequence header and
decoder `ImageInfo` all agreeing at depth 10, `decoder_transfer = 16` (PQ),
with negative controls at `--expect-depth 8` exiting 1. This lane did not
re-run G3 and does not claim to have.

**The `av1C` mis-signalling on T2-a's blobs (§10.4c) does not touch these
numbers.** The container advertised `seq_profile = 1` / 4:4:4 while the AV1
sequence header said profile 0 / 4:2:0; decoders (ours included) read the
sequence header, all three depth reads agree at 10, and the defect was fixed
upstream in `zenavif ae9a354f`. The stored blobs are annotated, not
re-encoded — so **T2-a's rate and quality are valid, and a consumer reading
chroma from the container of a pre-fix blob would be misled.**

---

## 5. What this document does NOT establish

- **Nothing about banding.** §0b, reason (ii) alone.
- **No 10-bit-display claim, no aom comparison, no claim about SVT-AV1 itself**
  — explicitly not questions of this arm (plan §5.1).
- **No content-class attribution that is separable from gamut** on this corpus
  (§1.3).
- **No controlled backend comparison** — §3 is a three-way-confounded contrast.
- **No HDR knob effects.** The HDR knob surface is two knobs wide and the knob
  block was never declared (plan §4.3, gated on TODO-3).
- **No cross-era join.** `avifhbd-t2a-20260902` (the pre-#18-fix run, paused at
  120/3,248 with structurally invalid cells) is **not** read here, and its rows
  are never pooled with `avifhbd-t2a-fix-20260902`'s.
- **Nothing about `zensim` on this corpus** (§1.2b). The stored T2 parquet
  carries a `m_zensim_score` column and it is single-instrument as published,
  but no claim in this document rests on it. Note the correction in §1.2b: the
  two scoring images agree on `zensim` to 1.16e-7, and the 11-point gap this
  lane first reported was between the ScoreFile executor and a hand-run
  `score-pairs --metric zensim` — two different quantities, not two eras.

---

## 6. Reproduction

Data pointer with paths, shas and the exact command chain:
[`avif_hdr_rd_baseline_2026-09-03.pointer.md`](avif_hdr_rd_baseline_2026-09-03.pointer.md).

Tools (`scripts/jobsys/`): `avifdoe_score_gapfill.sh` (with
`ZEN_DOE_SCORE_HDR=1` — an HDR wave declared without `--hdr` poisons every job
with `encoder_panic`, which is how 46 scorefiles were lost on the first
attempt) → `avifdoe_harvest.py` → `avifhbd_t2_analyze.py`. BD-rate parity
against `zenavif/scripts/rd_gap/bd_arm.py` is asserted on every run
(`--parity-check`) and was **exact (max |Δ| 0.0)**.

**No rank statistic appears in this document**, so `zenstats` is not called and
is not being bypassed: BD-rate is an integration (the DOE analyzer owns it,
parity-gated), the medians and their CIs come from that same analyzer's
percentile bootstrap, and the identity results are exact counts. A SROCC/PLCC
number would have to come from `zenstats` via `panel`; none is reported.
