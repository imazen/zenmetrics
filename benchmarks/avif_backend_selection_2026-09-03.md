# AVIF DOE — Stage-B remainder: the native QM×sharpness surface, and the backend table

**Analysis of the two runs registered in
[`avif_stageB_remainder_2026-09-03.md`](avif_stageB_remainder_2026-09-03.md).**
Everything below was pre-registered there; nothing in the cut, the bars or the
exclusions moved after the data existed. Era pins throughout: **2026-09-03**
(zenavif `56179fcb` / zenav1-svt `2ca060f4` / zenrav1e `e4883037`, fleet image
`exec-avifhbd-eradelta-e015344f`). No number here is joined across eras.

Companions: [`avif_speed_instrument_2026-09-03.md`](avif_speed_instrument_2026-09-03.md)
(the wall-time axis), [`avif_doe_stageB6_analysis_2026-09-02.md`](avif_doe_stageB6_analysis_2026-09-02.md)
(§17.4, the gap this closes), [`avif_eradelta_analysis_2026-09-03.md`](avif_eradelta_analysis_2026-09-03.md)
(the pin-stability licence).

---

## 0. TL;DR

1. **Score coverage is COMPLETE: 12,467 / 12,467 cells (100.00 %).** It was
   verified *before* any analysis and stated then as a fraction — `brnat`
   7,488/7,488, `brsdr` 4,971/4,979 (99.84 %), 8 cells still in the scorer. Those
   8 landed while this was being written, and **the entire analysis was re-run at
   full coverage rather than arguing the residue was negligible**: every Q1 table
   is byte-identical and the only change anywhere in the wave is one scan image's
   BD-rate moving **0.009 pp** (−2.5001 → −2.5091). No median, CI, win count or
   p-value moved. Coverage is measured at the CELL level by joining through
   `encode_sha`; the blob prefix is content-addressed and would have read
   78 % / 80 % at completion (registration §4.2).
2. **Q1 — the QM × sharpness synergy HOLDS AT NATIVE, and it is a near-total
   cancellation, not a partial one.** At s6, `shp7` alone costs **+7.02 %**
   BD-rate; `qml1.2.10` alone saves **−2.86 %**; the additive model predicts
   **+4.17 %**; the joint arm actually reads **−0.03 %**. The interaction
   residual is **−4.70 pp** (95 % CI [−5.80, −3.99], 26/30 images negative,
   sign test p = 5.9 × 10⁻⁵). `qml1.4.10 × shp7` repeats it at −4.68 pp.
3. **The synergy is a two-level effect, not a QM-axis trend.** `qml1.8.15` — the
   third level — barely interacts (−0.48 pp with `shp7`, −0.20 pp with `shp3`)
   and is byte-identical to the control on 11 % of its cells. The 4×3 factorial
   is what shows this; a 3-level design that happened to pick 1.8.15 would have
   concluded the interaction was small.
4. **Size barely moves it.** On the 13 genuinely-cropped references the
   native-minus-budget shift in the residual is **+0.22 pp** ([+0.05, +0.30],
   n = 66, p = 0.0043) — detectable, and an order of magnitude below the effect.
   The 19 passthrough references are a **null control that reads exactly
   0.0000 on 114 of 114 cells**, which is also the integrity gate on the whole
   crop-vs-native comparison.
5. **Content splits it cleanly, and `plot` is the exception**: ai-gen −6.33 pp,
   photo −5.24, scan −4.87, screenshot −4.05, **plot ≈ 0.00 (p = 0.41)**.
6. **⛔ Q2 — THE BACKEND TABLE IS NOT A BACKEND COMPARISON. It is a
   (backend × chroma) comparison, and the chroma difference is total.**
   MEASURED by reading the `av1C` box out of **1,114 bitstreams**: every
   `brsdr` / zenrav1e cell is **seq_profile 1, 4:4:4 (640/640)**; every
   `a0r` / `brnat` / svt cell is **seq_profile 0, 4:2:0 (448/448 and 26/26)**.
   Zero exceptions. This is structural in the sweep tool, not a mis-specified
   grid (§4.1), and it is the single most important caveat in this document.
   **UPDATE 2026-09-04 — SPLIT. See §8.** The confound is resolved: it is
   **CHROMA, not backend** (k₁₁ = 11 of 11 on the pre-registered rule). Item 8
   below is re-worded per that finding's own pre-registered consequence.
7. **Read through that confound, the headline numbers are:** zenrav1e-4:4:4
   needs **+7.3 %** more bits than svt-4:2:0 at matched ssim2 (median over 32
   images, 95 % CI [−1.99, +19.81], **11 of 32 images go the other way**, sign
   test p = 0.11 — **not** a significant global win for either side). The
   band-restricted reading agrees: +8.83 % over ssim2 ∈ [30, 95], same 11/32.
8. **The decisive axis is not bytes, it is REACH and TIME.** ~~svt at 4:2:0~~
   **UPDATE 2026-09-04 (§8): re-worded per the chroma-split arm's own
   pre-registered consequence — "svt" → "4:2:0."** **4:2:0
   cannot reach ssim2 90 on 16 of 32 references at any q, at any speed** —
   on plots it tops out at **40.5 / 54.4 / 66.7 / 67.1** while 4:4:4 reaches
   94.2 / 94.4 / 90.9 / 93.2 on the same images — **and zenrav1e at 4:2:0 hits
   the identical ceiling on the identical images** (k₁₁ = 11/11, §8): this is a
   chroma property, not a backend one. **The best of all 118 svt arms
   in A2 does not fix it** (39.99 / 53.83 / 66.18 / 67.05). In the other
   direction, at a 100 ms budget zenrav1e is over-budget on **31 of 32**
   references and svt wins by default; per dial position svt is **3.9× to 69.3×
   faster** at 1 MP — the time/backend axis is untouched by the chroma split
   (§8.4).
9. **So backend selection is genuinely image-conditioned and the model needs it
   as an output — but the feature that decides it is chroma-vs-content, not
   backend identity.** The ranked gap list is §6; **gap 1 is a zenrav1e 4:2:0
   arm**, which is cheap, which the encoder already supports, and which no
   existing data can substitute for.

---

## 1. Score coverage — verified before anything was analysed

Encoding was declared COMPLETE by the launch lane (registration §4.9). Scoring
was still draining, so coverage was verified first and is reported as a
fraction. **The scorer works on BLOBS and blobs are content-addressed**, so a
blob-prefix count is not a coverage number: `brnat`'s 7,488 cells collapse onto
5,817 distinct bitstreams (22.3 % dedup, registration §4.2). Coverage is
therefore computed by joining every ledger cell to the scored `encode_sha` set.

| run | DONE cells | distinct blobs | cells with `ssim2` | blobs scored | cell coverage |
|---|--:|--:|--:|--:|--:|
| run | at first harvest | **final** |
|---|--:|--:|
| `avifdoe-svt-brnat-20260903` | 7,488 / 7,488 (100.00 %) | **7,488 / 7,488 (100.00 %)** |
| `avifdoe-rav-brsdr-20260903` | 4,971 / 4,979 (99.84 %) | **4,979 / 4,979 (100.00 %)** |
| total | 12,459 / 12,467 (99.94 %) | **12,467 / 12,467 (100.00 %)** |

(Distinct blobs: `brnat` 5,817, `brsdr` 4,607 — the numbers a blob-prefix count
would have reported as "progress".)

**8 `brsdr` cells (0.16 % of that run) were unscored at first harvest** and were
carried as `metrics_present = None`, never dropped. The scorer stayed live
throughout (five containers; blobs 2,706 → 2,721), the 8 landed, and **the whole
analysis was re-run against 100 % coverage.** The re-run is the measurement that
replaces the argument: **every table in §2 is byte-identical**, three backend
tables changed, and the sole numeric delta in the wave is `6602`'s BD-rate
(−2.5001 → −2.5091) and its banded twin (−4.2913 → −4.3082). The 11/21 win
split, every median, every CI and every p-value are unchanged. **All numbers
below are the 100 %-coverage run.**

Every scored cell carries the same metric set — `ssim2 | zensim_features`, on
**10,416 of 10,416** scored bitstreams — so this wave has none of Stage A's
mid-flight metric heterogeneity. `zensim` is emitted as a **720-wide feature
vector**, not a scalar, so **`ssim2` is the only corpus-wide scalar quality
response** and is the matching axis everywhere below. That is a limitation of
the score run, not a choice made here.

`brsdr`'s ledger holds 4,979 cells against its de-scoped 2,880-cell manifest —
the 2,099 retained 29-q extras (registration §4.8), independently reproduced
here. Its `blobs/` prefix holds 5,776 objects, i.e. **1,169 orphans** left by
cells that were encoded under the original 9,280-cell manifest and dropped from
the ledger when it was shrunk. Orphan blobs are ignored: the ledger is the cell
set.

---

## 2. Q1 — QM × sharpness at native

### 2.1 What was measured

`brnat` is the complete **4 × 3 factorial** the registration promised:
qml ∈ {default, 1.2.10, 1.4.10, 1.8.15} × shp ∈ {default, 3, 7}, all 26 strata,
32 images, the 9-point knob ladder, speed 6. Per image, BD-rate of each arm is
taken against **its own run's `s6-svt-420` control** over the overlapping
achieved-ssim2 span, and the interaction is the Stage-A owner's construction
verbatim:

```
resid(k1,k2 | image)  =  BD(k1+k2)  −  [ BD(k1) + BD(k2) ]
```

Negative = the joint arm needs fewer bits than the additive model predicts =
synergy. CIs are 10,000-resample percentile bootstraps of the median over
**images**; the sign test is the Stage-A gates' `binom_two_sided`.

**Two gates ran before any of it.** BD-rate is **identical to zenavif's
`scripts/rd_gap/bd_arm.py` on 200 random ladders (max |Δ| = 0.0)**, and the
`brnat` control is **byte-identical to A2's control on 171 of 171 shared
(passthrough-image, q) cells** while differing on 117 of 117 cropped ones —
exactly the pattern that proves the two legs differ in pixels and in nothing
else.

### 2.2 The interaction surface — the cancellation, shown term by term

Native, speed 6, medians over 32 images, BD-rate %, negative = fewer bits.

| qml | shp | qml alone | shp alone | additive prediction | **observed joint** | **residual** | 95 % CI | n_neg/30 | p |
|---|---|--:|--:|--:|--:|--:|---|--:|--:|
| 1.2.10 | 3 | −2.86 | +1.43 | −1.43 | −2.16 | **−1.19** | [−1.53, −0.81] | 25 | 3.2e−4 |
| 1.4.10 | 3 | −2.55 | +1.43 | −1.12 | −1.76 | **−1.06** | [−1.36, −0.83] | 25 | 3.2e−4 |
| 1.8.15 | 3 | −0.66 | +1.43 | +0.77 | +0.64 | **−0.20** | [−0.33, −0.03] | 22 | 0.016 |
| 1.2.10 | 7 | −2.86 | **+7.02** | **+4.17** | **−0.03** | **−4.70** | [−5.80, −3.99] | 26 | 5.9e−5 |
| 1.4.10 | 7 | −2.55 | **+7.02** | **+4.47** | **+0.32** | **−4.68** | [−5.18, −3.95] | 27 | 8.4e−6 |
| 1.8.15 | 7 | −0.66 | +7.02 | +6.37 | +6.06 | **−0.48** | [−0.74, −0.05] | 22 | 0.016 |

**The verdict: Stage A's budget-size synergy holds at native, and at the two
aggressive QM levels it is a near-total cancellation.** Sharpness 7 alone costs
+7.02 % bits. Combined with `qml1.2.10` it costs **−0.03 %** — the QM window
absorbs essentially the whole sharpness penalty, and the residual (−4.70 pp) is
of the same magnitude and sign as the −5.2 to −5.5 pp Stage A measured at budget
size. `|resid| ≥ 1 pt` on **93 %** of images for both `shp7` pairs.

Pooled over all six factorial cells the native median residual is **−1.07 pp**
([−1.46, −0.80], 147/180 negative, p = 2.2 × 10⁻¹⁸). Do not quote that pooled
figure as "the synergy": it averages the strong `shp7` cells with the nearly-inert
`qml1.8.15` ones.

### 2.3 The finding the 4×3 shape bought — the QM axis is not a trend

`qml1.8.15` is a different animal from the other two levels on **every** axis:
its own main effect is −0.66 % against −2.86 / −2.55; its residual with `shp7`
is −0.48 pp against −4.70 / −4.68; and it is **byte-identical to the control on
11.1 % of its cells** (32 of 288). Registration §2.2 called this block a
factorial that "over-covers the qml axis" relative to B-2's 3-level shape. That
over-coverage is what makes the result interpretable: with three levels chosen
differently, this reads as a weak interaction rather than a strong one confined
to two levels.

**Treat (qml, shp) as a categorical cell, never as an ordinal trend.** That is
the plan's own §7.1(6) rule for sharpness, and the QM axis now needs it too.

### 2.4 The crop-vs-native A/B, with its null control

A2 ran the same plan, same 9 q, same 32 images at 1024² budget size, so the
size A/B is paired cell-for-cell. **It is only meaningful on the 13 genuinely
cropped references**: for the other 19 the budget corpus *is* the native pixels,
the encodes are byte-identical (gate: 171/171), and the delta is identically
zero by construction. Pooling all 32 would dilute a real effect with 19
self-comparisons.

| scope | n | median Δ(native − budget) | 95 % CI | n_neg | p |
|---|--:|--:|---|--:|--:|
| **cropped 13 refs** | 66 | **+0.2172 pp** | [+0.0548, +0.2965] | 21 | 0.0043 |
| passthrough 19 refs (**NULL CONTROL**) | 114 | **0.0000** | [0.0000, 0.0000] | 0 | — |
| all 32 (**diluted — do not cite**) | 180 | 0.0000 | [0.0000, 0.0000] | 21 | — |

**The synergy is very slightly weaker at native**, by +0.22 pp against a −4.7 pp
effect — 4.6 % of the effect size. B-6 §17.4 had already removed size as a
plausible *explanation* of the synergy (−0.39 pp shift in the additive baseline
against a −5.2 to −5.5 % residual); this measures the interaction itself and
reaches the same place from the other direction. The null control reading
exactly 0.0000 on 114 of 114 cells is the evidence that the pairing is sound.

### 2.5 Content classes — and the one class with no synergy

Median residual by coarse class, native, `qml × shp7`:

| class | n imgs | qml1.2.10 | qml1.4.10 | qml1.8.15 |
|---|--:|--:|--:|--:|
| ai-gen | 9 | **−6.33** | −5.07 | −0.83 |
| photo | 7 | **−5.24** | −4.76 | −0.98 |
| scan | 3 | −4.87 | −4.64 | −0.41 |
| screenshot | 5 | −4.05 | −4.19 | −0.03 |
| **plot** | 6 | **+0.01** | +0.00 | +0.03 |

Pooled over the six factorial cells the class picture is: ai-gen −2.02
(50/54 negative, p = 3.8e−11), photo −1.18 (**42/42**, p = 4.5e−13), scan −1.14
(16/18), screenshot −0.83 (24/30), **plot +0.007 (15/36 negative, p = 0.41 — no
effect).**

**Plot content shows no QM × sharpness synergy at all.** §5 explains why that is
almost certainly the same phenomenon as the svt quality ceiling in Q2: at 4:2:0
the plot encodes are chroma-limited well below the quality range where the QM
window and the sharpness filter interact, so there is nothing for them to trade
against.

### 2.6 A free result: `tune=0` is a no-op, and the inert census

The registration (§4.2) flagged the per-(stratum, image) byte-identity census as
a cheap first cut that needs no scores. Run:

| stratum | cells | identical to control | frac inert | images fully inert |
|---|--:|--:|--:|--:|
| **`tn0`** | 288 | **288** | **1.0000** | **32 / 32** |
| `mtx32` | 288 | 72 | 0.2500 | 2 |
| `qml1.8.15` | 288 | 32 | 0.1111 | 0 |
| `qml1.8.15-mtx32` | 288 | 2 | 0.0069 | 0 |
| all 21 others | 288 each | 0 | 0.0000 | 0 |

**`tune=0` is the encoder's default**: 288 of 288 cells on all 32 images are
byte-identical to the control, so the `tn0`, `tn0-tl1.0` and `tn0-tl1.1` strata
carry no information beyond `tl1.0` / `tl1.1` alone — visible in §2.7 as
`tn0-tl1.x` reading exactly the same BD-rate as `tl1.x`, and as a 0.0000
residual on 30/30 images. Three of 26 strata were spent confirming a default.
**Any future grid should byte-check its "level 0" arms against the control
before spending cells on them.**

### 2.7 The other main effects at native (s6), for the record

| arm | median BD % | 95 % CI | IQR | n_neg/30 |
|---|--:|---|--:|--:|
| `tn3` | **−7.69** | [−12.01, −3.38] | 12.76 | 22 |
| `qml1.2.10` | −2.86 | [−3.92, −2.00] | 2.62 | 24 |
| `qml1.4.10` | −2.55 | [−3.27, −1.92] | 1.76 | 24 |
| `qml1.8.15` | −0.66 | [−0.80, −0.33] | 0.89 | 22 |
| `tn0` | 0.0000 | [0, 0] | 0.00 | 0 |
| `mtx32` | +0.15 | [−0.04, +1.10] | 1.54 | 11 |
| `tl1.0` | +0.52 | [+0.33, +1.04] | 0.92 | 6 |
| `tl1.1` | +1.19 | [+0.71, +1.76] | 1.95 | 1 |
| `shp3` | +1.43 | [+0.87, +2.13] | 1.98 | 5 |
| `shp7` | +7.02 | [+5.40, +9.36] | 6.09 | 3 |

`tune=3` is the largest single win at native (−7.69 %, consistent with the −4 to
−7 % the coordinator's summary carries) **but it is also by far the most
variable knob in the block** — IQR 12.76 pp, range −23.2 to +19.8, 8 of 30
images regressed. It is a per-image decision, not a default.

**Stated limitation, inherited from the plan:** `brnat` is **speed 6 only**
(`svt_doe_pairwise` carries a single preset). B-6 established that sharpness's
transfer behaviour is speed-specific, so **nothing here may be restated at s4 or
s7.**

---

## 3. Q2 — the backend table

### 3.1 ⛔ Read this before any number in §3: the comparison is confounded with chroma

The registration asked for the first per-image SDR cross-backend comparison, on
the budget corpus, with `brsdr` (zenrav1e, defaults) against A0R-svt (svt-rs,
defaults). Both arms ran their backend's defaults, as designed. **The defaults
differ in chroma subsampling, and that was not visible in the knob tuples.**

MEASURED, by reading the `av1C` `AV1CodecConfigurationRecord` out of the
bitstreams themselves — a stratified census covering every (image × speed)
combination in both runs, plus every one of `brnat`'s 26 arms:

| run | backend | bitstreams checked | images | seq_profile | chroma | exceptions |
|---|---|--:|--:|--:|---|--:|
| `brsdr` | zenrav1e | **640** | 32 | 1 | **4:4:4** | **0** |
| `a0r` | svt-rs | **448** | 32 | 0 | **4:2:0** | **0** |
| `brnat` | svt-rs | 26 (all arms) | 1 | 0 | 4:2:0 | 0 |

The mechanism is in the sweep tool, and it is deliberate on the svt side:
`crates/zenmetrics-cli/src/sweep/encode.rs::avif_config_from_knobs` pins
`Yuv420` **only** when `backend=svt-rs` — because zenavif's svt backend
*rejects* anything else (`encoder_svt_rs.rs::reject_unsupported_config`:
*"Av1Backend::Zenav1Svt encodes 4:2:0 only … the 4:4:4 default is zenravif-only
for now"*) — and leaves the zenravif arm on zenavif's `EncoderConfig` default,
which is **`EncodeChromaSubsampling::Yuv444`** (`encoder.rs:684`). **No
`chroma` / `subsampling` knob is wired for AVIF at all**; the AVIF path never
reads the `subsampling` knob (that one is zenjpeg's).

So every byte, quality and reach difference in §3 is **jointly attributable to
backend and to chroma**, and cannot be split with the data that exists. The
harvest extension this lane wrote refuses to synthesize a chroma token into the
`brsdr` arm labels for exactly this reason (§7); the census above is what turns
"unknown" into "measured".

**Everything below is true of the two things as configured — which is a real
product question — and is NOT a statement about the encoders.**

### 3.2 Bytes at matched quality

Per image, both backends' points are pooled across their whole speed dial into
one Pareto frontier, and BD-rate is integrated over the overlapping achieved-ssim2
span. Negative = zenrav1e needs fewer bits.

| scope | n | median BD % | 95 % CI | zenrav1e wins | p |
|---|--:|--:|---|--:|--:|
| **all, full overlap** | 32 | **+7.31** | [−1.99, +19.81] | **11 / 32** | 0.110 |
| **all, banded ssim2 ∈ [30, 95]** | 32 | **+8.83** | [−0.96, +19.48] | **11 / 32** | 0.110 |
| plot | 6 | +73.31 (banded +42.52) | [−9.24, +112.78] | 2 / 6 | 0.69 |
| photo | 7 | +7.62 | [−7.05, +26.37] | 3 / 7 | 1.00 |
| ai-gen | 9 | +7.00 | [−2.10, +19.62] | 3 / 9 | 0.51 |
| scan | 5 | +7.00 | [−2.50, +69.49] | 1 / 5 | 0.38 |
| screenshot | 5 | +2.89 | [−9.68, +155.99] | 2 / 5 | 1.00 |

The banded reading exists because a full-overlap BD-rate on these ladders
integrates from ssim2 ≈ −60 upward (the q1/q5 cells), so the deep-negative tail
carries integration weight nobody ships in. **It was computed as a check on the
default and it confirms it** — same median direction, same 11/32 split, no
ordering change. Both are reported; neither replaces the other.

**No global winner.** The median favours svt-4:2:0 by 7–9 %, the CI crosses
zero, and the sign test does not reject a coin flip. Per band, the byte ratio
(zenrav1e ÷ svt, < 1 means zenrav1e is smaller) is:

| band (ssim2) | n | median ratio | zenrav1e wins | p |
|---|--:|--:|--:|--:|
| low 30–50 | 29 | 1.065 | 8 / 29 | **0.024** |
| mid 50–70 | 29 | 1.043 | 10 / 29 | 0.136 |
| **high 70–85** | 27 | **1.012** | **13 / 27** | **1.00 — a tie** |
| near-lossless 85–95 | 16 | 1.097 | 3 / 16 | **0.021** |

**In the high band — where most web delivery decisions live — the two are
indistinguishable on bytes**, 1.2 % apart with the wins split 13/14.

### 3.3 The result that actually decides the table: svt-4:2:0 has a quality ceiling

Maximum achievable ssim2 per image, across the entire ladder (all q, all
speeds), for each arm — plus the best of all **118** svt arms in A2 as a
control on whether any knob rescues it:

| image | class | svt a0r max | best of 118 A2 svt arms | zenrav1e max | gap |
|---|---|--:|--:|--:|--:|
| `7004` | plot | **40.48** | 39.99 | **94.20** | +53.7 |
| `7058` | plot | **54.36** | 53.82 | **94.38** | +40.0 |
| `7076` | plot | 66.74 | 66.18 | 90.92 | +24.2 |
| `7042` | plot | 67.14 | 67.04 | 93.21 | +26.1 |
| `6602` | scan | 76.44 | 76.22 | 91.72 | +15.3 |
| `8288` | screenshot | 77.62 | 77.52 | 93.70 | +16.1 |
| `8414` | screenshot | 79.69 | 79.67 | 90.78 | +11.1 |
| `6604` | scan | 80.07 | 79.67 | 94.38 | +14.3 |
| `7050` | plot | 84.80 | 84.46 | 94.50 | +9.7 |
| … 14 photo/ai-gen refs | | 88–93 | 88–92 | 90–94 | ≈ 0 |
| `6006`, `6018` | scan | **100.00** | 100.00 | 94.2 / 96.0 | −4 to −6 |

Census over the corpus:

| threshold | svt a0r | best-of-118 A2 svt | zenrav1e |
|---|--:|--:|--:|
| max achievable ssim2 **< 90** | **16 / 32** | 19 / 32 | **1 / 32** |
| **< 85** | 9 / 32 | 9 / 32 | **0 / 32** |
| **< 80** | 7 / 32 | 8 / 32 | **0 / 32** |

**On plot and screen content, svt at 4:2:0 cannot be driven to high quality at
all** — `7004` saturates at ssim2 40.5 at q98 — **and no knob in the DOE fixes
it.** The A2 column is the load-bearing part: 118 arms, including every QM,
sharpness, tune, tile, matrix-coefficient, ac-bias, screen-content-mode and
variance-boost level the plan spells, and the best of them lands within 1 point
of the default. This is a chroma-resolution limit, not a rate-control one, which
is also the most parsimonious explanation for **why plot content shows no
QM × sharpness synergy in §2.5**: the plot encodes never reach the quality
region where those two knobs trade against each other.

The two `6006`/`6018` scans invert it — svt reaches ssim2 100.00 where zenrav1e
stops at 94–96 — so the ceiling is a content-conditioned property, not a
uniform one.

### 3.4 The third axis — wall time

Joined from the speed instrument (`build_commit d1928710`,
`speed_alpha_beta{,_per_source}.tsv`). **Its stated limitations are carried into
every table cell, not just into this paragraph:** every α/β is **q45-specific**;
β is **single-threaded WALL time** (conservative by ~27 % for large threaded svt
frames); the **pooled** fit is unreliable (R² 0.62–0.91, β varies up to **24.33×**
with content) while the **per-source** fit is clean (R² 0.9929–0.9997) but exists
for **5 sources only**; and **content-class speed splits (S1c) are NOT MEASURED
and are never interpolated here.**

Predicted encode time at 1 MP, per dial position:

| speed | svt-rs ms | zenrav1e ms | ratio |
|--:|--:|--:|--:|
| 1 | 12,911 | 50,942 | 3.9× |
| 4 | 935 | 9,539 | 10.2× |
| 6 | 66 | 2,971 | 44.9× |
| **7** | **28.7** | **1,987** | **69.1×** |
| 10 | 28.7 | 392 | 13.7× |

svt saturates at 7–10 (β 28.2 ms/MP, within 0.4 %); zenrav1e is still moving at
10. **At their respective fast ends svt is ~14× faster; at matched dial position
in the mid-range it is 45–69× faster.**

Because the bytes-optimal speed is 1–2 for *both* backends — an operating point
nobody ships — the decision-relevant view is iso-time: at budget T, each backend
may use its cheapest-bytes speed that both reaches the quality band and fits T.
**"Cannot reach the band at any speed" and "reaches it but misses the budget"
are counted separately**; merging them would read a capability gap as a speed
problem.

| band | budget | zenrav1e wins | svt wins | zenrav1e over-budget | **svt cannot reach band** |
|---|--:|--:|--:|--:|--:|
| low | 100 ms | 0 | 1 | **31** | 0 |
| low | 500 ms | 6 | 15 | 11 | 0 |
| low | 2,000 ms | 7 | 24 | 1 | 0 |
| mid | 100 ms | 0 | 1 | 29 | 0 |
| mid | 2,000 ms | 8 | 22 | 0 | 2 |
| high | 100 ms | 0 | 1 | 26 | 0 |
| high | 500 ms | 6 | 11 | 10 | 4 |
| high | 2,000 ms | 7 | 20 | 0 | **5** |
| near-lossless | 500 ms | 3 | 7 | 6 | **11** |
| near-lossless | 2,000 ms | 1 | 15 | 0 | **15** |

**Under ~500 ms, svt wins by default because zenrav1e cannot finish.** Above
~2 s both fit, and svt wins bytes on roughly three quarters of the corpus — but
on the near-lossless band it is *unable to serve* 15 of 32 references at all.
Those are the two ends of the same story: **svt is one to two orders of
magnitude cheaper and modestly smaller where it can operate, and there is a
substantial slice of the corpus where it cannot operate at the target quality.**

### 3.5 The per-image win table, condensed

Full table: `tables/backend_per_image.tsv` (32 rows, both BD variants, both
frontiers' achieved spans, and per-band byte readings). Condensed by class,
bytes only, best-over-speeds:

| class | n | zenrav1e wins bytes | svt wins bytes | zenrav1e wins high band | svt cannot reach ssim2 90 |
|---|--:|--:|--:|--:|--:|
| photo | 7 | 3 | 4 | 3 / 7 | **0 / 7** |
| ai-gen | 9 | 3 | 6 | 5 / 9 | 3 / 9 |
| screenshot | 5 | 2 | 3 | 3 / 5 | **5 / 5** |
| scan | 5 | 1 | 4 | 2 / 4 | 2 / 5 |
| **plot** | 6 | 2 | 4 | 0 / 2 | **6 / 6** |
| **all** | **32** | **11** | **21** | **13 / 27** | **16 / 32** |

The last column is the sharpest split in the wave: **svt-4:2:0 fails to reach
ssim2 90 on 11 of the 11 plot + screenshot references, and on 0 of the 7
photos.** Bytes barely separate the two configurations; reach separates them
along the content axis almost perfectly.

---

## 4. Q3 — what the decision surface looks like now

Combining this wave with the standing results — `scm3` (screen, −50 %, speed-7
only), `tune=3` (−4 to −7 %), `bd10-native`, and the two answers above — the
feature-conditioned picker's surface has **four decisions, of very unequal
maturity**:

**Decision 1 — chroma, and it now outranks backend.** The only measured lever
that changes what quality is *reachable* on synthetic content is 4:4:4 vs 4:2:0,
worth up to **+54 ssim2 points of headroom** on plots. No rate knob substitutes
for it (118 arms tried). Any picker that routes plot/screen content through a
4:2:0 path has capped its own output before it starts. **This is the highest-value
signal in the DOE so far and it was discovered as a confound, which means it has
never been swept.** **UPDATE 2026-09-04 — no longer a confound, MEASURED (§8):
k₁₁ = 11 of 11 on the pre-registered CHROMA/BACKEND rule. A picker that switches
*backend* to fix a plot/screenshot reach failure will not fix it — zenrav1e hits
the identical wall at 4:2:0 on the identical 11 images. Chroma is the only lever
that opens the ceiling, and (§8.3) it costs ~0 bytes where both chromas can
already reach a quality — the cost of 4:4:4 is not a bytes tradeoff, it is the
thing that makes otherwise-unreachable content reachable at all.**

**Decision 2 — backend, conditioned on a time budget first and content second.**
Below ~500 ms/frame it is not a choice: zenrav1e cannot finish on 31 of 32
references. Above ~2 s it is a real per-image choice with an 11/21 split and a
*tied* high band. The picker should treat the time budget as a hard gate applied
before any RD reasoning, then use content class + the reach question, and only
then bytes.

**Decision 3 — (qml, shp) as a joint categorical cell, s6, non-plot content.**
`qml1.2.10 × shp7` buys sharpness-7's behaviour at ≈ **0 %** bitrate cost against
+7.02 % for sharpness alone, on 26/30 images, with the same sign at both sizes.
The cell is the unit: the axes are not ordinal (level 1.8.15 breaks the trend)
and the interaction does not decompose. On plot content the cell is worth
nothing — but see Decision 1 for why.

**Decision 4 — `tune=3` per image, never as a default.** Median −7.69 % but IQR
12.76 pp with 8/30 images regressing, so it needs a per-image predictor, which
is precisely the kind of output the model exists to produce.

**And one thing the surface should NOT carry:** `tune=0` is the default (288/288
byte-identical), so it is not a decision at all.

---

## 5. Limitations — the honest list

1. **The backend arm is confounded with chroma, totally and uniformly** (§3.1).
   Nothing in §3 is a statement about encoders.
2. **`ssim2` is the only scalar response.** `zensim` was emitted as a feature
   vector, so no zensim-space matching was possible. On plot/screen content in
   particular, an ssim2 ceiling of 40 is a strong claim resting on one metric —
   though the fact that a *different* configuration reaches 94 on the same image
   with the same metric means it is not a metric saturation artifact.
3. **`brnat` is speed 6 only.** Nothing about the interaction may be restated at
   s4 or s7 (B-6 measured that sharpness transfer is speed-specific).
4. **The qml axis is the three levels the plan spells.** A finer QM sweep is a
   different block, and §2.3 shows the axis is not smooth enough to interpolate.
5. **Speed coefficients are q45-anchored, single-threaded wall time, and pooled
   where no per-source fit exists** (5 of 32 sources have one). The instrument
   itself falsified q-flatness — cost RISES with q — so the iso-time table's
   time column is a q45 reading used as a band, not a per-cell prediction.
6. **S1c content-class speed splits are NOT MEASURED.** No content-conditioned
   time number appears anywhere here, interpolated or otherwise.
7. **`brsdr` is 53.7 % of its original grid** (the 9-q ladder complete plus
   2,099 retained 29-q extras), so the quality-space match on that side is
   9-point-dense for every image and denser only on some. That degrades match
   *precision*, not validity — 9 points clears the BD-rate owner's ≥ 4-point
   guard on every cell used.
8. ~~8 `brsdr` cells unscored~~ — **RESOLVED**: they landed during the analysis and everything was re-run at 100.00 % coverage (§1). Kept in this list so the sequence is visible rather than tidied away.
9. **A0R covers svt speeds 1–7**; 8–10 are byte- and time-identical to 7
   (measured), so no coverage is lost, but the table's svt speed column never
   exceeds 7.
10. **n is small per content class** (3–9 images). Class-level medians are
    directional; the corpus-level ones are the load-bearing figures.

---

## 6. Ranked data gaps before the training lane

| rank | gap | why it blocks the model | cost |
|--:|---|---|---|
| **1** | ~~A zenrav1e 4:2:0 arm~~ | **✅ CLOSED 2026-09-04.** The chroma-split arm (`avif_chroma_split_2026-09-04.md` §7) split it: **CHROMA, k₁₁ = 11/11**. Kept in this list so the sequence is visible rather than tidied away. | done |
| **2** | **An svt 4:4:4 arm**, or a recorded decision that svt is 4:2:0-only | Decides whether "use 4:4:4 on synthetic content" is a *chroma* output the picker can set on either backend, or a *backend* output. Today `zenavif` hard-rejects it, so this is a zenavif/zenav1-svt feature question before it is a data question. | blocked upstream |
| **3** | **Native-size cross-backend AND cross-chroma coverage** (registration §5 rank 5) | Every §3/§8 number is budget-1024² only. §2.4 shows size effects are small for the QM×sharpness interaction, but nothing establishes that for the backend/chroma reach ceiling, which is exactly the kind of thing that could be size-dependent. | 2,880+ cells / ~19+ CPU-h |
| **4** | **The interaction at s4 and s7** | s6-only by construction, and B-6 measured that sharpness transfer is speed-specific — so the single most valuable knob pair in the DOE is unmeasured at two of three production presets. Needs `svt_doe_main`, a different plan. | ~15k cells |
| **5** | ~~S1c content-class speed splits~~ | **✅ LANDED 2026-09-04.** `avif_speed_instrument_2026-09-03.md` §8: between-coarse-class median spread 3.92× (1.5–2.2× at production speeds); within-class spread up to 52.6× — the 5-coarse-class label leaves real signal on the table vs the corpus's existing 12-fine-class one. **⚠ PRE-BACKEND-CHANGE ERA**: the AV1 backend is under active re-development in a concurrent session; these coefficients will need re-measurement once it lands. | done |
| **6** | **More plot/screen references** (n = 6 and 5) | Plot is the class that breaks both headline results — no QM×shp synergy, and the whole quality ceiling. Every claim about it rests on 6 images. | corpus work |
| **7** | **A second scalar quality response** (zensim scalar, or butteraugli) | Single-metric conclusions on synthetic content, per §5.2. | score-side |
| **8** | **aom-rs as a third arm** | PLANNED-BLOCKED on era pins post-#15 (registration §3.4); unchanged. | — |

---

## 7. Outputs, owners, and reproduction

**Analysis outputs** — local `/mnt/v/output/avif-backend-2026-09-03/`
(41 files, 2.08 MB: 18 TSVs, 18 parquets, the 12,467-row harvested table, both
pairs tables, the chroma census, `_MANIFEST.json` with per-file sha256).
Nothing is in git (>30 KB rule); file list and hashes:
[`avif_backend_selection_2026-09-03.pointer.md`](avif_backend_selection_2026-09-03.pointer.md).

**Two owners touched, both EXTENDED rather than forked**, per the
no-duplicate-implementations rule:

- **`scripts/jobsys/avifdoe_harvest.py`** now understands the SDR backend-arm
  knob tuple (`{"backend":"zenravif","speed":N}`) that the builder's
  `--knob-grid` entry point emits, labelling it `s<N>-zenravif` and carrying the
  speed in the explicit `dial` / `dial_kind` columns. **The label deliberately
  carries no chroma token** — the existing `-420` assertions are backed by a
  928/928 byte-identity measurement and no equivalent existed for zenravif;
  synthesizing one would have been convention dressed as evidence, and §3.1 is
  what evidence looks like. The new branch is **strictly additive by
  construction** (backends already in `NAIVE_BACKEND_CHROMA` are excluded, so it
  is unreachable for every tuple shape the previous code handled) and
  **verified non-regressive on real data: re-harvesting with the change
  reproduces all 7,488 `brnat` rows BYTE-IDENTICALLY across all 18 columns.**
- **`scripts/jobsys/avifdoe_brem_analyze.py`** is new and re-implements no
  statistic: `frontier`, `bd_rate`, `median_ci`, `q1q3` and the content-class
  map are imported from `avifdoe_stagea_analyze.py`, `binom_two_sided` from
  `avifdoe_stagea_gates.py`, and `curves` / `load_scored` from
  `avifdoe_stageb6_analyze.py`. The interaction residual is the Stage-A owner's
  construction verbatim.

**Gates that ran, and passed:**

| gate | result |
|---|---|
| BD-rate parity vs `zenavif/scripts/rd_gap/bd_arm.py`, 200 random ladders | **max \|Δ\| = 0.0** |
| `brnat` control ≡ A2 control on passthrough refs | **171 / 171 byte-identical** |
| …and differs on cropped refs | **0 / 117 identical** (as required) |
| size-A/B null control on passthrough refs | **0.0000 on 114 / 114** |
| harvest non-regression on `brnat` | **7,488 rows byte-identical, 18/18 columns** |
| chroma census exceptions | **0 of 1,114 bitstreams** |

**Regeneration:**

```sh
zenfleet-ctl pairs --ledger s3://zentrain/jobs/avifdoe-svt-brnat-20260903/ledger/ \
    --refs-prefix s3://codec-corpus/avif-subsample-2026-09-01/ \
    --blobs-prefix s3://zentrain/jobs/avifdoe-svt-brnat-20260903/blobs/ --out brnat_pairs
zenfleet-ctl pairs --ledger s3://zentrain/jobs/avifdoe-rav-brsdr-20260903/ledger/ \
    --refs-prefix s3://codec-corpus/avif-doe-1024-2026-09-01/ \
    --blobs-prefix s3://zentrain/jobs/avifdoe-rav-brsdr-20260903/blobs/ --out brsdr_pairs
avifdoe_harvest.py --score-dir <br score blobs> --sizes <both blob listings> \
    --pairs brnat=brnat_pairs.tsv --pairs brsdr=brsdr_pairs.tsv --out br_scored.parquet
avifdoe_brem_analyze.py --br-scored br_scored.parquet \
    --stagea-scored /mnt/v/output/zensim-avifdoe/doe_scored_2026-09-02.parquet \
    --crop-manifest /mnt/v/output/avif-doe-1024-2026-09-01/crop_manifest_2026-09-01.tsv \
    --native-dims /mnt/v/output/zensim-avifdoe-b6/native_dims.tsv \
    --speed-coef      /mnt/v/output/avif-speed-instrument-2026-09-03/speed_alpha_beta.tsv \
    --speed-coef-per-source \
        /mnt/v/output/avif-speed-instrument-2026-09-03/speed_alpha_beta_per_source.tsv \
    --outdir tables --parity-check ~/work/zen/zenavif/scripts/rd_gap/bd_arm.py
```

**⚠ Commit provenance — this document and its two scripts landed inside commit
`c041466e`, whose message is another lane's** ("measure: threading explanation
CONFIRMED by controlled test + Track T COMPLETE"). Cause: `jj` snapshots the
whole working copy on *every* command, and this analysis ran in the primary
checkout alongside a live speed-instrument lane, so that lane's next `jj`
invocation swept these in-progress files into its own change and pushed them.
Nothing was lost or altered — the files are verified byte-correct on
`master@origin` — but `git log` will not lead a reader here from the commit
subject, hence this note. **The lesson is general: an additive path-scoped
`.workongoing` protects against concurrent EDITS to the same files, and does
nothing about a shared `jj` working copy.** Two lanes writing to one checkout
need a sibling `jj workspace`, not a marker convention.

**Raw inputs** stay in their canonical homes: encode bitstreams
`s3://zentrain/jobs/avifdoe-{svt-brnat,rav-brsdr}-20260903/blobs/`, score blobs
`s3://zentrain/jobs/avifdoe-br-sf-cpu-20260903/blobs/`, corpora
`s3://codec-corpus/avif-{subsample,doe-1024}-2026-09-01/`.

---

## 8. The chroma confound is SPLIT — unconfounded re-cut (2026-09-04)

**⚠ PRE-BACKEND-CHANGE ERA (2026-09-04): the AV1 backend is being actively
re-developed in a concurrent session; the byte/reach numbers below are a
zenavif `6dfdf6f` / zenrav1e `7ad86844` read and will need re-measurement
after the backend redevelopment lands (the user's own estimate: ~2x faster).
No new encodes were run for this section — it harvests an arm that had
already finished encoding before the local box rebooted.**

Full registration + methodology: `avif_chroma_split_2026-09-04.md` §1-6.
Full results: same file §7. This section is the pointer + the decision-surface
update; numbers are not re-derived here.

**The gap this closes**: §0.6/§3.1 above established that every cross-backend
number in §3 is a joint (backend × chroma) attribution — `brsdr`/zenrav1e was
always 4:4:4, `brnat`/`a0r`/svt was always 4:2:0, zero exceptions on 1,114
bitstreams. `avif_config_from_knobs` had no chroma knob for AVIF at all. The
chroma-split arm added the knob (`f15bb3a5`) and ran zenrav1e at 4:2:0
(`br420`, the cell that never existed) plus a same-era zenrav1e-4:4:4 control
(`br444`), both on the 9-q ladder, both scored by one score run.

**Controls, both unconditional passes**: era (encoder) 2,880/2,880
byte-identical — `br444` reproduces `brsdr`'s 9-q subset exactly, so `br444`
may be read against `brsdr`'s Stage-B-era scored cells. Scorer: max |Δ ssim2|
= 0.0 on 2,880 shared bitstreams.

**Verdict: CHROMA, not backend.** k₁₁ = 11 of 11 (every pre-registered
plot+screenshot svt-4:2:0-fail image is ALSO a zenrav1e-4:2:0 fail); k = 16 of
16 overall. This is the maximum-strength reading against the pre-registered
rule (k₁₁ ≥ 9 → CHROMA). Corroborated by an independent ladder-density-immune
reading: of svt-420's 17 in-wave-ladder misses, `br420` also misses all 17/17;
`br444` misses only 2/17 (both pre-flagged edge cases).

**The bytes axes say something DIFFERENT from the reach axis, and that
difference is itself the finding.** Holding backend constant and varying only
chroma moves BD-rate by a median of **+0.08%** (CI [−0.97, +0.93], sign test
p = 0.860 — indistinguishable from zero). Holding chroma constant and varying
backend moves it +10.19% svt-favouring (CI [+0.25, +19.15], sign p = 0.071).
**So 4:2:0 is not a bytes-efficiency choice where both chromas can already
reach a target quality — it is a hard reach CEILING that no amount of bitrate
opens.** A picker reasoning about "chroma cost" in bytes alone would
underweight it by an order of magnitude relative to its actual effect on
reachable quality.

**Decision-surface consequences** (full text: §4 Decision 1, updated inline
above): a picker that reaches for "switch backend" to fix a plot/screenshot
quality ceiling will not fix it. The only lever that opens the ceiling is the
chroma knob (4:4:4), essentially free in bytes where both chromas already
succeed. The time/backend axis (Decision 2) is untouched — it was never a
chroma question.

**Ranked gaps**: §6 rank 1 (the zenrav1e 4:2:0 arm) and rank 5 (S1c
content-class splits, landed the same session — see
`avif_speed_instrument_2026-09-03.md` §8, same PRE-BACKEND-CHANGE caveat) are
both closed; the table above is updated in place with strikethrough + pointer
per this doc's own convention for resolved items (§5 point 8).

Outputs, triple-mirrored, sha256-verified: `avif_chroma_split_2026-09-04.pointer.md`.
