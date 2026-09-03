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

1. **Q5 — the baseline exists now.** The 16-reference PQ corpus is encoded at 7 svt presets x 29 quality points and 3 zenrav1e speeds x 9, all 10-bit PQ, all scored f32-faithful. Median achieved ssim2 spans **-85.8 to 86.1** (zenav1-svt) and **-59.6 to 77.0** (zenrav1e); the full per-(arm, q) curve is `t2_rd_baseline.tsv`. There is no PASS/FAIL bar here by registration — this is the control every future HDR arm differences against.
2. **Q6 — the contrast.** Per-backend Pareto envelope: **zenav1-svt -43.01 % BD-rate vs zenrav1e** (95% CI [-48.63, -37.97], 16/16 images), and arm-for-arm the spread runs **-45.38 %** (`p0-zenav1svt-hdr10` vs `s7-zenavif-hdr10`) to **-36.70 %** (`p9-zenav1svt-hdr10` vs `s4-zenavif-hdr10`). Three things differ at once (backend, chroma, matrix) and the ladders are not equally dense, so this is a **contrast**, not a controlled comparison, and not a claim about either encoder in isolation.
3. **The instrument, measured rather than assumed.** On 351 cells scored by BOTH images that touched this corpus, `ssim2` is bit-identical (0 differing) and `zensim` agrees to 1.16e-7 — no material era split. Every number here is `ssim2` anyway, and §1.2b carries a correction this lane owes the record plus a trap worth knowing: `score-pairs --metric zensim` and the ScoreFile executor's `zensim` are different quantities.
4. **Not measured, and not a null:** banding (corpus, §0b), any 8-vs-10 depth pairing on HDR (no 8-bit HDR arm was ever declared — §4), HDR knob effects (never declared), and any content-vs-gamut attribution this corpus cannot separate (§1.3).

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

| run | declared | done | failed-only | verdict |
|---|--:|--:|--:|---|
| `avifhbd-t2a-fix-20260902` (T2-a, svt, encode) | 3248 | 3248 | 0 | COMPLETE |
| `avifhbd-t2b-20260902` (T2-b, zenrav1e, encode) | 432 | 432 | 0 | COMPLETE |

| harvest | cells | scored (`ssim2`) | unscored |
|---|--:|--:|--:|
| `t2a` | 3248 | 3248 | 0 |
| `t2b` | 432 | 432 | 0 |
| **both arms** | **3680** | **3680** | **0** |

### 1.1b Why the svt arm is `t2a-fix`: this corpus is the worst case for #18

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

| content category | primaries | n refs |
|---|--:|--:|
| 1000-lilith-photos-general | Display-P3 | 1 |
| 1200-lilith-interiors | BT.709 | 3 |
| 1400-lilith-nature | BT.709 | 2 |
| 1400-lilith-nature | Display-P3 | 9 |
| 1600-lilith-food | BT.709 | 1 |
| **total** | **BT.709 / Display-P3** | **6 / 10** |

> **Registered analysis restriction (plan §4.3).** Primaries are nearly
> determined by content in this corpus. No content-class conclusion may be drawn
> without this cross-tab, and no primaries/gamut conclusion without the content
> one; where the two cannot be separated the finding is
> **"content-or-gamut, not separable in this corpus"**, never attributed to one.

---

## 2. Q5 — the RD baseline

**Read the shared-q table first — it is where the Q6 result comes from.** The
two arms are not offset by a constant; their curves have different shapes:

- **The two dials are not comparable, and reading the table across a row would
  mislead.** At q45 svt sits at **0.083 bpp / ssim2 2.6** and zenrav1e at
  **0.235 bpp / 29.0** — svt is *both* cheaper *and* lower-quality at the same
  dial position. The same q means different operating points on the two
  backends, which is precisely why Q6 below is a BD-rate over *achieved
  quality* and never a per-q comparison.
- **svt's whole ladder is shifted toward lower rate, and it reaches further.**
  Its median quality ceiling is **ssim2 86.1** against zenrav1e's **77.0**, so
  the BD-rate integration is bounded above by zenrav1e's reach and svt's extra
  headroom sits *outside* the compared span rather than inflating the result.
- **The rate floors differ by an order of magnitude** — median minimum encode
  **9–10 KB** (svt) vs **91–93 KB** (zenrav1e). Part of that is 4:2:0 against
  4:4:4, one of the three confounded axes, not an encoder verdict.

| arm (median bpp / median ssim2) | q5 | q25 | q45 | q60 | q76 | q90 | q96 |
|---|--:|--:|--:|--:|--:|--:|--:|
| `p0-zenav1svt-hdr10` | 0.009 / -67.0 | 0.034 / -41.4 | 0.082 / 4.5 | 0.161 / 34.8 | 0.370 / 59.5 | 1.080 / 77.7 | 1.885 / 84.1 |
| `p1-zenav1svt-hdr10` | 0.009 / -67.1 | 0.034 / -41.7 | 0.082 / 4.0 | 0.163 / 34.4 | 0.370 / 58.7 | 1.102 / 77.5 | 1.956 / 84.3 |
| `p3-zenav1svt-hdr10` | 0.009 / -67.3 | 0.034 / -42.7 | 0.083 / 3.3 | 0.165 / 33.7 | 0.370 / 58.5 | 1.111 / 77.2 | 1.974 / 84.1 |
| `p4-zenav1svt-hdr10` | 0.009 / -67.6 | 0.034 / -43.5 | 0.083 / 2.6 | 0.165 / 33.3 | 0.369 / 58.1 | 1.114 / 77.1 | 1.987 / 84.1 |
| `p6-zenav1svt-hdr10` | 0.009 / -68.2 | 0.035 / -44.4 | 0.090 / 2.4 | 0.178 / 33.4 | 0.374 / 57.8 | 1.137 / 76.8 | 2.019 / 83.3 |
| `p7-zenav1svt-hdr10` | 0.009 / -68.9 | 0.036 / -45.5 | 0.090 / 1.4 | 0.181 / 32.8 | 0.369 / 57.5 | 1.147 / 77.0 | 2.035 / 83.3 |
| `p9-zenav1svt-hdr10` | 0.009 / -69.3 | 0.034 / -47.0 | 0.090 / 0.4 | 0.185 / 32.2 | 0.362 / 56.7 | 1.144 / 76.9 | 2.029 / 82.6 |
| `s4-zenavif-hdr10` | 0.059 / -59.2 | 0.146 / 4.8 | 0.235 / 29.0 | 0.314 / 38.9 | 0.491 / 53.8 | 0.837 / 69.1 | 1.204 / 77.0 |
| `s6-zenavif-hdr10` | 0.060 / -59.6 | 0.149 / 4.9 | 0.242 / 29.1 | 0.323 / 39.1 | 0.499 / 53.8 | 0.855 / 69.0 | 1.201 / 76.9 |
| `s7-zenavif-hdr10` | 0.060 / -59.6 | 0.151 / 4.6 | 0.244 / 29.2 | 0.325 / 38.9 | 0.501 / 53.6 | 0.863 / 68.8 | 1.210 / 76.8 |

*(the full curve — every arm x every ladder point, with the second quality column — is `t2_rd_baseline.tsv`; only the 7 q values present in BOTH ladders are shown here)*

| arm | backend | n refs | ladder pts | median min ssim2 | median max ssim2 | median min bytes | median max bytes |
|---|--:|--:|--:|--:|--:|--:|--:|
| `p0-zenav1svt-hdr10` | zenav1-svt | 16 | 29 | -84.156 | 86.078 | 9 KB | 6.16 MB |
| `p1-zenav1svt-hdr10` | zenav1-svt | 16 | 29 | -85.022 | 86.145 | 9 KB | 6.25 MB |
| `p3-zenav1svt-hdr10` | zenav1-svt | 16 | 29 | -84.095 | 86.095 | 9 KB | 6.27 MB |
| `p4-zenav1svt-hdr10` | zenav1-svt | 16 | 29 | -84.438 | 86.144 | 9 KB | 6.22 MB |
| `p6-zenav1svt-hdr10` | zenav1-svt | 16 | 29 | -85.827 | 86.059 | 10 KB | 6.28 MB |
| `p7-zenav1svt-hdr10` | zenav1-svt | 16 | 29 | -85.774 | 86.104 | 10 KB | 6.28 MB |
| `p9-zenav1svt-hdr10` | zenav1-svt | 16 | 29 | -85.835 | 86.041 | 10 KB | 6.26 MB |
| `s4-zenavif-hdr10` | zenavif | 16 | 9 | -59.216 | 76.957 | 91 KB | 2.11 MB |
| `s6-zenavif-hdr10` | zenavif | 16 | 9 | -59.577 | 76.925 | 93 KB | 2.11 MB |
| `s7-zenavif-hdr10` | zenavif | 16 | 9 | -59.585 | 76.838 | 93 KB | 2.13 MB |

**Within-backend dial ordering** (each backend against itself; negative = the test arm needs fewer bits)

*zenav1-svt (preset dial)*

| test arm | reference arm | n | median BD-rate % | 95% CI | images where test wins |
|---|--:|--:|--:|--:|--:|
| `p0-zenav1svt-hdr10` | `p1-zenav1svt-hdr10` | 16 | -1.3766 | [-1.5495, -1.2181] | 16/16 |
| `p0-zenav1svt-hdr10` | `p3-zenav1svt-hdr10` | 16 | -3.8533 | [-4.3331, -3.3008] | 16/16 |
| `p0-zenav1svt-hdr10` | `p4-zenav1svt-hdr10` | 16 | -4.6831 | [-5.2075, -3.9892] | 16/16 |
| `p0-zenav1svt-hdr10` | `p6-zenav1svt-hdr10` | 16 | -9.9963 | [-11.6992, -8.3774] | 16/16 |
| `p0-zenav1svt-hdr10` | `p7-zenav1svt-hdr10` | 16 | -13.0360 | [-14.8985, -10.4501] | 16/16 |
| `p0-zenav1svt-hdr10` | `p9-zenav1svt-hdr10` | 16 | -13.2040 | [-15.6491, -10.0118] | 16/16 |
| `p1-zenav1svt-hdr10` | `p0-zenav1svt-hdr10` | 16 | 1.3959 | [1.2339, 1.5739] | 0/16 |
| `p1-zenav1svt-hdr10` | `p3-zenav1svt-hdr10` | 16 | -2.4064 | [-2.6163, -1.9363] | 16/16 |
| `p1-zenav1svt-hdr10` | `p4-zenav1svt-hdr10` | 16 | -3.2597 | [-3.8547, -2.5231] | 16/16 |
| `p1-zenav1svt-hdr10` | `p6-zenav1svt-hdr10` | 16 | -8.4187 | [-9.4797, -7.1640] | 16/16 |
| `p1-zenav1svt-hdr10` | `p7-zenav1svt-hdr10` | 16 | -11.4741 | [-12.4238, -9.1042] | 16/16 |
| `p1-zenav1svt-hdr10` | `p9-zenav1svt-hdr10` | 16 | -11.4770 | [-14.3914, -8.8701] | 16/16 |
| `p3-zenav1svt-hdr10` | `p0-zenav1svt-hdr10` | 16 | 4.0080 | [3.4135, 4.5305] | 0/16 |
| `p3-zenav1svt-hdr10` | `p1-zenav1svt-hdr10` | 16 | 2.4657 | [1.9746, 2.6866] | 0/16 |
| `p3-zenav1svt-hdr10` | `p4-zenav1svt-hdr10` | 16 | -0.8440 | [-1.0547, -0.6715] | 16/16 |
| `p3-zenav1svt-hdr10` | `p6-zenav1svt-hdr10` | 16 | -5.8609 | [-7.2259, -5.3752] | 16/16 |
| `p3-zenav1svt-hdr10` | `p7-zenav1svt-hdr10` | 16 | -8.9780 | [-10.1482, -7.2217] | 16/16 |
| `p3-zenav1svt-hdr10` | `p9-zenav1svt-hdr10` | 16 | -9.2310 | [-11.0602, -6.9410] | 16/16 |
| `p4-zenav1svt-hdr10` | `p0-zenav1svt-hdr10` | 16 | 4.9132 | [4.1550, 5.4947] | 0/16 |
| `p4-zenav1svt-hdr10` | `p1-zenav1svt-hdr10` | 16 | 3.3695 | [2.5885, 4.0093] | 0/16 |
| `p4-zenav1svt-hdr10` | `p3-zenav1svt-hdr10` | 16 | 0.8512 | [0.6760, 1.0660] | 0/16 |
| `p4-zenav1svt-hdr10` | `p6-zenav1svt-hdr10` | 16 | -5.3054 | [-6.0693, -4.3683] | 16/16 |
| `p4-zenav1svt-hdr10` | `p7-zenav1svt-hdr10` | 16 | -8.2445 | [-9.5419, -6.5245] | 16/16 |
| `p4-zenav1svt-hdr10` | `p9-zenav1svt-hdr10` | 16 | -8.4771 | [-10.1131, -6.2735] | 16/16 |
| `p6-zenav1svt-hdr10` | `p0-zenav1svt-hdr10` | 16 | 11.1065 | [9.1433, 13.2492] | 0/16 |
| `p6-zenav1svt-hdr10` | `p1-zenav1svt-hdr10` | 16 | 9.1927 | [7.7169, 10.4724] | 0/16 |
| `p6-zenav1svt-hdr10` | `p3-zenav1svt-hdr10` | 16 | 6.2261 | [5.6805, 7.7886] | 0/16 |
| `p6-zenav1svt-hdr10` | `p4-zenav1svt-hdr10` | 16 | 5.6027 | [4.5679, 6.4614] | 0/16 |
| `p6-zenav1svt-hdr10` | `p7-zenav1svt-hdr10` | 16 | -3.2286 | [-3.9504, -2.2631] | 15/16 |
| `p6-zenav1svt-hdr10` | `p9-zenav1svt-hdr10` | 16 | -2.9046 | [-5.0931, -1.7831] | 16/16 |
| `p7-zenav1svt-hdr10` | `p0-zenav1svt-hdr10` | 16 | 14.9907 | [11.6695, 17.5068] | 0/16 |
| `p7-zenav1svt-hdr10` | `p1-zenav1svt-hdr10` | 16 | 12.9613 | [10.0161, 14.2150] | 0/16 |
| `p7-zenav1svt-hdr10` | `p3-zenav1svt-hdr10` | 16 | 9.8639 | [7.8117, 11.2944] | 0/16 |
| `p7-zenav1svt-hdr10` | `p4-zenav1svt-hdr10` | 16 | 8.9856 | [7.0102, 10.5485] | 0/16 |
| `p7-zenav1svt-hdr10` | `p6-zenav1svt-hdr10` | 16 | 3.3364 | [2.3155, 4.1128] | 1/16 |
| `p7-zenav1svt-hdr10` | `p9-zenav1svt-hdr10` | 16 | -0.3534 | [-1.0598, 0.1172] | 10/16 |
| `p9-zenav1svt-hdr10` | `p0-zenav1svt-hdr10` | 16 | 15.2138 | [11.1257, 18.5524] | 0/16 |
| `p9-zenav1svt-hdr10` | `p1-zenav1svt-hdr10` | 16 | 12.9650 | [9.7335, 16.8106] | 0/16 |
| `p9-zenav1svt-hdr10` | `p3-zenav1svt-hdr10` | 16 | 10.1697 | [7.4587, 12.4356] | 0/16 |
| `p9-zenav1svt-hdr10` | `p4-zenav1svt-hdr10` | 16 | 9.2626 | [6.6934, 11.2510] | 0/16 |
| `p9-zenav1svt-hdr10` | `p6-zenav1svt-hdr10` | 16 | 2.9942 | [1.8154, 5.3665] | 0/16 |
| `p9-zenav1svt-hdr10` | `p7-zenav1svt-hdr10` | 16 | 0.3552 | [-0.1167, 1.0736] | 6/16 |

*zenavif / zenrav1e (speed dial)*

| test arm | reference arm | n | median BD-rate % | 95% CI | images where test wins |
|---|--:|--:|--:|--:|--:|
| `s4-zenavif-hdr10` | `s6-zenavif-hdr10` | 16 | -1.7253 | [-2.1557, -0.9684] | 15/16 |
| `s4-zenavif-hdr10` | `s7-zenavif-hdr10` | 16 | -2.7298 | [-3.2302, -1.9686] | 16/16 |
| `s6-zenavif-hdr10` | `s4-zenavif-hdr10` | 16 | 1.7562 | [0.9779, 2.2034] | 1/16 |
| `s6-zenavif-hdr10` | `s7-zenavif-hdr10` | 16 | -1.0570 | [-1.4468, -0.6613] | 16/16 |
| `s7-zenavif-hdr10` | `s4-zenavif-hdr10` | 16 | 2.8067 | [2.0082, 3.3392] | 0/16 |
| `s7-zenavif-hdr10` | `s6-zenavif-hdr10` | 16 | 1.0684 | [0.6657, 1.4681] | 0/16 |

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

> **The ladder-density worry is RESOLVED by the matrix, and the answer is that
> density is not what is driving the margin.** All 21 cross-backend arm pairs
> land between **−36.70 %** and **−45.38 %**, and the envelope's −43.01 % sits
> inside that range rather than beyond it. The most conservative single-arm
> comparison available — svt's *slowest* preset against zenrav1e's *fastest*
> speed (`p9` vs `s4`) — is still **−36.70 %, winning on 16 of 16 images**. A
> denser ladder buys a few points at most here; the bulk of the gap is not it.
> What the gap *is* remains three-way confounded (backend, chroma, matrix), and
> nothing below attributes it to any one of the three.

**Per-backend Pareto envelope** — the dial-free product comparison

| test backend | reference backend | n refs | median BD-rate % | 95% CI | IQR | images where test wins | min | max |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `zenav1-svt` | `zenavif` | 16 | -43.0108 | [-48.6316, -37.9680] | [-49.4381, -37.9802] | 16/16 | -57.0034 | -18.2158 |
| `zenavif` | `zenav1-svt` | 16 | 75.7271 | [61.2513, 95.5535] | [61.2536, 97.9640] | 0/16 | 22.2730 | 132.5765 |

**By content class, with the primaries cross-tab carried** (§1.3's restriction: where content and gamut cannot be separated on this corpus, the finding is *content-or-gamut*)

| test | reference | content category | primaries | n | median BD-rate % | min | max |
|---|--:|--:|--:|--:|--:|--:|--:|
| `zenav1-svt` | `zenavif` | 1000-lilith-photos-general | Display-P3 | 1 | -18.2158 | -18.2158 | -18.2158 |
| `zenav1-svt` | `zenavif` | 1200-lilith-interiors | BT.709 | 3 | -38.3264 | -38.9945 | -26.1183 |
| `zenav1-svt` | `zenavif` | 1400-lilith-nature | BT.709 | 2 | -46.3453 | -52.0799 | -40.6107 |
| `zenav1-svt` | `zenavif` | 1400-lilith-nature | Display-P3 | 9 | -47.0634 | -57.0034 | -19.6261 |
| `zenav1-svt` | `zenavif` | 1600-lilith-food | BT.709 | 1 | -48.5575 | -48.5575 | -48.5575 |

**Full arm × arm matrix, cross-backend cells only** — no preset↔speed equivalence is implied by any row here

| test arm | reference arm | n | median BD-rate % | 95% CI | test wins |
|---|--:|--:|--:|--:|--:|
| `p0-zenav1svt-hdr10` | `s4-zenavif-hdr10` | 16 | -43.5860 | [-49.1129, -38.0703] | 16/16 |
| `p0-zenav1svt-hdr10` | `s6-zenavif-hdr10` | 16 | -44.6325 | [-50.2997, -39.0059] | 16/16 |
| `p0-zenav1svt-hdr10` | `s7-zenavif-hdr10` | 16 | -45.3783 | [-51.4451, -40.1207] | 16/16 |
| `p1-zenav1svt-hdr10` | `s4-zenavif-hdr10` | 16 | -42.1904 | [-48.2921, -37.2176] | 16/16 |
| `p1-zenav1svt-hdr10` | `s6-zenavif-hdr10` | 16 | -43.7288 | [-49.4514, -38.1473] | 16/16 |
| `p1-zenav1svt-hdr10` | `s7-zenavif-hdr10` | 16 | -44.3964 | [-49.7019, -39.2965] | 16/16 |
| `p3-zenav1svt-hdr10` | `s4-zenavif-hdr10` | 16 | -40.8717 | [-46.5439, -35.1004] | 16/16 |
| `p3-zenav1svt-hdr10` | `s6-zenavif-hdr10` | 16 | -42.2403 | [-47.7436, -35.9621] | 16/16 |
| `p3-zenav1svt-hdr10` | `s7-zenavif-hdr10` | 16 | -42.9354 | [-48.1916, -37.3154] | 16/16 |
| `p4-zenav1svt-hdr10` | `s4-zenavif-hdr10` | 16 | -40.0774 | [-46.0384, -34.1855] | 16/16 |
| `p4-zenav1svt-hdr10` | `s6-zenavif-hdr10` | 16 | -41.6729 | [-47.2503, -35.1107] | 16/16 |
| `p4-zenav1svt-hdr10` | `s7-zenavif-hdr10` | 16 | -42.5412 | [-47.5123, -36.4322] | 16/16 |
| `p6-zenav1svt-hdr10` | `s4-zenavif-hdr10` | 16 | -38.0846 | [-43.1377, -28.5494] | 16/16 |
| `p6-zenav1svt-hdr10` | `s6-zenavif-hdr10` | 16 | -39.3457 | [-44.0011, -29.5757] | 16/16 |
| `p6-zenav1svt-hdr10` | `s7-zenavif-hdr10` | 16 | -39.9218 | [-44.6342, -30.7643] | 16/16 |
| `p7-zenav1svt-hdr10` | `s4-zenavif-hdr10` | 16 | -37.0167 | [-41.3395, -24.3551] | 16/16 |
| `p7-zenav1svt-hdr10` | `s6-zenavif-hdr10` | 16 | -38.1331 | [-42.2283, -25.4383] | 16/16 |
| `p7-zenav1svt-hdr10` | `s7-zenavif-hdr10` | 16 | -38.7210 | [-42.8768, -26.6940] | 16/16 |
| `p9-zenav1svt-hdr10` | `s4-zenavif-hdr10` | 16 | -36.7044 | [-40.5038, -21.9022] | 16/16 |
| `p9-zenav1svt-hdr10` | `s6-zenavif-hdr10` | 16 | -37.8309 | [-41.4078, -23.0253] | 16/16 |
| `p9-zenav1svt-hdr10` | `s7-zenavif-hdr10` | 16 | -38.4234 | [-42.0668, -24.3241] | 16/16 |
| `s4-zenavif-hdr10` | `p0-zenav1svt-hdr10` | 16 | 77.4405 | [61.5305, 96.6060] | 0/16 |
| `s4-zenavif-hdr10` | `p1-zenav1svt-hdr10` | 16 | 73.0617 | [59.3311, 93.7585] | 0/16 |
| `s4-zenavif-hdr10` | `p3-zenav1svt-hdr10` | 16 | 69.1989 | [54.0991, 87.0694] | 0/16 |
| `s4-zenavif-hdr10` | `p4-zenav1svt-hdr10` | 16 | 66.9198 | [51.9542, 85.3170] | 0/16 |
| `s4-zenavif-hdr10` | `p6-zenav1svt-hdr10` | 16 | 61.5228 | [39.9568, 76.8222] | 0/16 |
| `s4-zenavif-hdr10` | `p7-zenav1svt-hdr10` | 16 | 58.7723 | [32.1966, 71.3984] | 0/16 |
| `s4-zenavif-hdr10` | `p9-zenav1svt-hdr10` | 16 | 57.9890 | [28.0446, 68.7460] | 0/16 |
| `s6-zenavif-hdr10` | `p0-zenav1svt-hdr10` | 16 | 80.8274 | [63.9724, 101.2060] | 0/16 |
| `s6-zenavif-hdr10` | `p1-zenav1svt-hdr10` | 16 | 77.9203 | [61.6916, 97.8294] | 0/16 |
| `s6-zenavif-hdr10` | `p3-zenav1svt-hdr10` | 16 | 73.2908 | [56.1575, 91.3640] | 0/16 |
| `s6-zenavif-hdr10` | `p4-zenav1svt-hdr10` | 16 | 71.5812 | [54.1087, 89.5744] | 0/16 |
| `s6-zenavif-hdr10` | `p6-zenav1svt-hdr10` | 16 | 64.9059 | [41.9965, 79.3778] | 0/16 |
| `s6-zenavif-hdr10` | `p7-zenav1svt-hdr10` | 16 | 61.6396 | [34.1171, 73.8718] | 0/16 |
| `s6-zenavif-hdr10` | `p9-zenav1svt-hdr10` | 16 | 60.8549 | [29.9128, 71.2129] | 0/16 |
| `s7-zenavif-hdr10` | `p0-zenav1svt-hdr10` | 16 | 83.2806 | [67.0388, 105.9522] | 0/16 |
| `s7-zenavif-hdr10` | `p1-zenav1svt-hdr10` | 16 | 80.0604 | [64.7661, 98.8146] | 0/16 |
| `s7-zenavif-hdr10` | `p3-zenav1svt-hdr10` | 16 | 75.4032 | [59.5360, 93.0187] | 0/16 |
| `s7-zenavif-hdr10` | `p4-zenav1svt-hdr10` | 16 | 74.1948 | [57.3175, 90.5208] | 0/16 |
| `s7-zenavif-hdr10` | `p6-zenav1svt-hdr10` | 16 | 66.5015 | [44.4342, 81.4064] | 0/16 |
| `s7-zenavif-hdr10` | `p7-zenav1svt-hdr10` | 16 | 63.1948 | [36.4145, 75.8230] | 0/16 |
| `s7-zenavif-hdr10` | `p9-zenav1svt-hdr10` | 16 | 62.4077 | [32.1425, 73.1411] | 0/16 |

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
