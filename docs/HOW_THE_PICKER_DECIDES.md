# How the codec picker decides (in plain terms)

zenpicker answers one question: **given an image, a quality target, and which formats are
allowed, which codec family should encode it?** This doc is the human-readable version of what
we found — the actual decision tree, what each choice costs, and how we got here.

The headline: **family choice is a short, auditable RULE, not a model** — and it does the sane
thing for *any* subset of formats (one allowed, several, or none). Shipped as
`zenpicker::family_rule`.

---

## 1. The rule (what ships)

```
family_rule(image, target, allowed, budget):
  1. lossless?  — caller asked, OR target quality ≥ ~96 (near-perfect → store exact)
  2. viable     — allowed
                  ∩ can REPRESENT the image   (capability — format-spec facts)
                  ∩ FITS the time budget       (drop codecs too slow for the latency)
  3. return the highest-priority viable codec:
        lossy    : JXL > AVIF > WebP > JPEG > GIF
        lossless : JXL > WebP > PNG > GIF
     → None only if nothing allowed can encode it (e.g. lossy asked, only PNG allowed)
```

Verify it by reading the two preference lists + the capability table — no fitted thresholds (bar
the one quality crossover), no black box.

**Capability — obviously correct, from the format specs:**

| the image needs… | can | cannot |
|---|---|---|
| alpha / transparency | png, webp, jxl, avif, gif | **jpeg** |
| HDR / > 8-bit | jxl, avif, png(16) | jpeg, webp, gif |
| a lossy encode | jpeg, webp, jxl, avif, gif | png |
| a lossless encode | png, webp, jxl, gif | jpeg, avif |

**The priority is a codec-reality PRIOR, deliberately not fit from our sweep.** JXL and AVIF are
the modern high-efficiency codecs (JXL first; **AVIF second** — AV1 intra); WebP is the older but
ubiquitous fallback; JPEG/PNG are compatibility floors; GIF is niche. We anchor on the prior
because the sweep currently mis-ranks WebP *above* AVIF — an artifact, not reality: AVIF was
encoded only to speed 4 (never the RD-optimal 0–2), the comparison is trapped in the low-quality
zone where AVIF's lead is thinnest, and the corpus skews small. A reader knows AVIF > WebP for
lossy, so the rule reflects that — until the data earns the right to refine it (§4).

---

## 2. Budget — the four modes

Codecs differ wildly in encode time: JXL/AVIF are slow (AVIF especially at the slow speeds that
give it good RD), WebP/JPEG/PNG are fast. The mode sets a latency ceiling and the rule drops any
codec whose own per-image encode estimate exceeds it (`AllowedFamilies::viable`):

- **RealtimeFastest** — tight budget → JXL/AVIF too slow → falls through to WebP/JPEG.
- **RealtimeBalanced** — looser → JXL/AVIF may fit at mid effort.
- **QueuedBalanced** — no latency gate → best-RD survivor (usually JXL).
- **QueuedAggressive** — no gate, max effort → best-RD, with headroom for an offline verify.

Same priority + the budget gate = "the best codec by priority that's fast enough." The codec
supplies its own time estimate; the picker never guesses. And because a *fast* encode hurts RD
(the same speed-4 effect that hobbled AVIF), slow codecs dropping out under a tight budget is the
correct call, not a regret.

---

## 3. Lossy vs lossless

One threshold on the quality target (`LOSSLESS_QUALITY ≈ 96`): below ~zq94 → lossy, above ~zq97 →
lossless, a thin contested 95–96 band. Near-perfect quality is cheaper stored exactly than
squeezed out of a lossy encoder. **JPEG never wins bytes-at-quality and PNG never wins lossless** —
they're in the lists only for compatibility, last among the viable.

---

## 4. What content-adaptation would buy (deferred — §1 is what ships)

The §1 rule is content-*blind* (always JXL first among the viable). A picker exists to tame the
tail — the images where WebP/AVIF beat JXL one-shot. Measured as **extra bytes vs the perfect
oracle**, one-shot, on the corrected data:

| approach | extra bytes (avg) | worst 10% | size |
|---|---|---|---|
| always JXL (no picker) | **18%** | 48% | a constant |
| a 6-leaf size/detail tree | 8.8% | 24% | a few `if`s |
| an 8–16-leaf tree | 6–8% | 18–20% | a dozen `if`s |
| 2 learned "codec-fit" scores + a 16-leaf tree | **4.6%** | 14% | ~1 KB |
| the full neural net (MLP) | **4.0%** | 11% | 27 KB |
| *(2–3 trial encodes, keep smallest)* | *0.04% = oracle* | *0%* | *— multi-shot, ruled out* |

Read it as: **always-JXL is fine on the typical image (median miss ~2%) but has an ugly tail.** A
few size/detail/chroma questions halve it; a tiny learned model nearly closes it; the 27 KB net
buys only ~0.6% over a ~1 KB one. **All of these are deferred**, for two reasons: (1) they're fit
on the same data that mis-ranks AVIF below WebP, so they'd bake that artifact in (an early tree
literally routed small clean images to WebP — likely the AVIF-understated effect); (2) one-shot is
mandatory, so the trial-encode oracle is off the table. Once the data is fixed (§6 + AVIF speed
0–2), a content composite can *nudge* the §1 prior — never invert it.

---

## 5. How we got to a trustworthy rule (the findings trail)

The §1 rule only became trustworthy after fixing data problems that were quietly poisoning the
picker — recorded so they don't come back:

1. **Missing features (the "experimental" gap).** The chroma-loss + IDCT-roundtrip features were
   compiled out by default, so training never saw them. Fixed: `experimental` on by default;
   re-extracted **101 qualified source features, 0 NaN**.
2. **A feature leak.** The old training table mixed source-content features with **zensim
   pair-features** (computed from a specific distorted encode) — not available at pick time.
   Removing them didn't hurt accuracy and made the picker honest.
3. **Cross-codec sampling bias (the big one) — two faces.** The sweep dials a generic `q` that
   each codec maps to a *different* achieved quality, so coverage is ragged. (a) Above ~zq90 AVIF
   was *over*-credited (only it had data there) — fixed by the support-aware data layer
   (`scripts/picker/picker_data.py`, compare only where each codec has measured support) + a gate
   (`check_quality_coverage.py`); the corrected win-rate is **JXL 45% / WebP 31% / AVIF 23%**.
   (b) AVIF is also *under-encoded* — swept only to **speed 4**, never the RD-optimal 0–2 — which,
   with a small-skewed corpus, makes WebP look better than AVIF (it isn't). Both are why §1's order
   is a codec-reality **prior**, not a data fit. Write-up:
   [`CROSS_CODEC_QUALITY_SAMPLING.md`](CROSS_CODEC_QUALITY_SAMPLING.md).

Everything measured here is on the *corrected* data.

---

## 6. What ships, and what's left

**Ships now: the §1 `zenpicker::family_rule`** — capability + budget gate + codec-reality
priority. One-shot, no ML runtime, robust to any format subset, verifiable by reading. (The 3 MLP
routers exist + are de-biased at `zenpicker::MetaPicker::default_routers()`, kept behind the `api`
surface for the ~0.6% of tail they shave when you trust them — but the rule is the default.)

**Still open** — to let content-adaptation safely *refine* the prior (§4), the data must first be
trustworthy:

1. **Quality-targeted sampling** — encode each codec to hit a common achieved-quality grid instead
   of a shared `q` (a new sweep mode), so all codecs are comparable past ~zq88 (today the support
   runs out there; near-lossless targets route to the lossless side, which is correct anyway).
2. **Re-sweep AVIF at speed 0–2** (not 4), so its curves reflect the codec, not the encoder knob.

With both, a content composite can be fit to *nudge* (never invert) the §1 prior, with the gate
(`check_quality_coverage.py`) in front of training to keep it honest.

---

*Sources: `scripts/picker/{reduce_picker,tree_reduce,tree_composite,tree_export,router_error_anatomy}.py`,
`picker_data.py`, `check_quality_coverage.py`. Models + provenance:
[`benchmarks/router_models_2026-06-30.pointer.md`](../benchmarks/router_models_2026-06-30.pointer.md).*
