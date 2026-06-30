# How the codec picker decides (in plain terms)

zenpicker answers one question: **given an image, a quality target, and which formats are
allowed, which codec family should encode it?** This is the human-readable version of what we
found — the decision, what each choice costs, and how the data earned it.

The decision has a fixed shape:

```
1. lossless?  — caller asked, OR target ≥ ~96 (near-perfect → store exact)
2. viable     — allowed ∩ can-REPRESENT-the-image ∩ FITS-the-time-budget
3. ORDER the viable codecs best-first, take the top
```

Steps 1–2 are **facts** (format specs + a cost model) — read them and you know they're right.
**Step 3, the order, is the only place judgment lives**, and it has two forms:

- **Image-aware (default).** A *linear projection* of zenanalyze features predicts each codec's
  bytes-at-target; order = cheapest first. Fit on confound-corrected data, it routes **one-shot at
  3.85% over the perfect oracle**. It's one linear layer — the weights are readable, not a net.
- **Fixed prior (fallback).** `JXL > AVIF > WebP > JPEG > GIF` lossy, `JXL > WebP > PNG > GIF`
  lossless — used when no features are available, or as the pure-audit path. **The corrected data
  confirms this order** (it isn't just asserted).

Both run behind the same capability + budget filter, so the pick is sane for **any** subset of
allowed formats — one, several, or none.

---

## 1. Capability — facts from the format specs

| the image needs… | can | cannot |
|---|---|---|
| alpha / transparency | png, webp, jxl, avif, gif | **jpeg** |
| HDR / > 8-bit | jxl, avif, png(16) | jpeg, webp, gif |
| a lossy encode | jpeg, webp, jxl, avif, gif | **png** |
| a lossless encode | png, webp, jxl, gif | jpeg, avif |

No fit, no thresholds. (`zenpicker::content_capability`, read from the zenanalyze Offer.)

---

## 2. Budget — the four modes

Codecs differ wildly in encode time: JXL/AVIF are slow (AVIF especially at the slow speeds that
give it good RD), WebP/JPEG/PNG are fast. The mode sets a latency ceiling; the rule drops any codec
whose own per-image estimate exceeds it (`AllowedFamilies::viable`):

- **RealtimeFastest** — tight → JXL/AVIF too slow → fall through to WebP/JPEG.
- **RealtimeBalanced** — looser → JXL/AVIF may fit at mid effort.
- **QueuedBalanced / QueuedAggressive** — no latency gate → best-RD survivor (usually JXL/AVIF).

The codec supplies its own time estimate; the picker never guesses. And a *fast* encode hurts RD,
so slow codecs dropping out under a tight budget is the correct call, not a regret.

---

## 3. Lossy vs lossless

One threshold on the quality target (`LOSSLESS_QUALITY ≈ 96`): below → lossy, above → lossless.
Near-perfect quality is cheaper stored exactly than squeezed from a lossy encoder. **JPEG never
wins lossy bytes-at-quality and PNG never wins lossless** — they're in the lists for compatibility,
last among the viable.

---

## 4. The order — image-aware linear projection (what carries the picker)

Image-blindness is *expensive*. Measured held-out, one-shot, as extra bytes vs the perfect oracle,
on the confound-corrected data:

| order | extra bytes (avg) | worst 10% |
|---|---|---|
| image-blind always-JXL | **30%** | 82% |
| image-blind always-AVIF | 22% | 58% |
| **image-aware linear projection** | **3.85%** | **14%** |
| a full neural net (MLP) | ~4% | ~11% |
| *(2–3 trial encodes — ruled out, multi-shot)* | *0%* | *0%* |

The best codec genuinely depends on the image. Here's where each one wins (oracle, by size ×
target quality):

```
              low quality (zq50)      high quality (zq88)
  large >1M   AVIF 85%                AVIF 73%      ← AVIF everywhere
  medium      AVIF 62%                JXL 52%
  small       AVIF 55%                JXL 54%
  tiny <64k   WebP 59%                JXL 53%       ← WebP low-q → JXL high-q
```

Read it as: **big images → AVIF; tiny images → WebP when quality is low, JXL when it's high; the
middle is AVIF/JXL.** A fixed order can't express that, which is why it leaves 22–30% on the table.
The linear projection captures it for ~4%, with weights you can read — every codec's bytes scale
with size + target quality; the *differentiators* are `info_weight` (texture/entropy),
`uniformity`, and `max_dim`. A 27 KB MLP buys nothing over the linear projection. One-shot
throughout (trial-encoding is off the table — "one shot is mandatory").

---

## 5. How the data earned it — statistical correction, no re-sweep

The order only became trustworthy after fixing confounds that were quietly poisoning it. **All
corrected with the data already in hand — no new encodes:**

1. **AVIF speed artifact (the one that mis-ranked it below WebP).** We have AVIF at speeds
   **s2/s4/s6/s8**; the measured per-2-step byte factor is only ~0.97, so AVIF's best swept speed
   is already near RD-optimal — it was barely understated. Re-measured with a clean RD metric
   (`bytes_to_reach`: cheapest encode reaching the target), AVIF beats WebP at zq ≥ 75 even raw.
   (`scripts/picker/avif_speed_correct.py`)
2. **Coverage (missing-not-at-random).** Codecs reach different quality ranges, so a naïve oracle
   credits whoever happens to have data there. Fixed with **paired pairwise** comparison — compare
   two codecs only on images where both reach the target. (`corrected_ranking.py`, `picker_data.py`,
   gate `check_quality_coverage.py`)
3. **Corpus skew.** The corpus is small-image-heavy, which flattered WebP. Fixed by **stratifying
   by size × quality and reweighting uniformly** over strata.

Result — the confound-corrected pairwise ranking: **AVIF 2.07 ≈ JXL 2.05 ≫ WebP 1.22 > JPEG 0.65.**
"WebP > AVIF" was the artifact, not reality. (Earlier fixes in the same trail: the `experimental`
feature gap — now 101 qualified features, 0 NaN; and a feature leak — zensim *pair*-features that
aren't available at pick time, removed without accuracy loss.)

Everything in §4 is measured on this corrected data.

---

## 6. What ships, what's next

- **Capability + budget + lossy/lossless** (§1–3) — pure rules, in `zenpicker` today.
- **Fixed-prior order** (`zenpicker::family_rule`) — the no-features fallback / audit path;
  data-confirmed.
- **Image-aware linear projection** (§4) — validated at 3.85%; **being baked as the lossy router**
  (replacing the MLP with the interpretable linear projection on corrected data, via the existing
  `zenpredict` bake path — `route()` already consumes a per-family-score router through
  `RouteDecision::resolve`).

Optional later (not required for the above): **quality-targeted sampling** — encode each codec to a
common achieved-quality grid instead of a shared `q`, to extend clean comparison past ~zq88 (today
near-lossless targets route to the lossless side, which is correct anyway). Not a blocker — the
statistical correction above already gives a trustworthy order without it.

---

*Analysis: `scripts/picker/{avif_speed_correct,corrected_ranking,linear_projection_order,export_projection}.py`,
`picker_data.py`, `check_quality_coverage.py`. Projection weights + provenance:
[`benchmarks/picker_projection_2026-06-30.pointer.md`](../benchmarks/picker_projection_2026-06-30.pointer.md).*
