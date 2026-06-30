# How the codec picker decides (in plain terms)

zenpicker answers one question: **given an image, a quality target, and which formats are
allowed, which codec family should encode it?** This doc is the human-readable version of what
we found — the actual decision tree, what each choice costs, and how we got here.

The headline: **choosing the family is mostly easy.** Two codecs never win and get dropped, one
codec is the default, a quality threshold splits lossy from lossless, and a small size/detail
rule handles the rest. No neural network required.

---

## 1. The lossy decision tree (the intuitive form)

Picking among the lossy codecs reduces to a tree whose **first question is image size**, then
detail and chroma — readable straight off:

```
                         ┌─ shorter side ≤ ~130 px ?  (is it a SMALL image?)
                         │
          ┌──────────────┴───────────────┐
       YES (small)                     NO (medium / large)
          │                                │
   chroma noise?                     how much detail/texture? (laplacian variance)
   ├─ clean      → WebP              ├─ flat / smooth
   └─ noisy      → JXL              │   ├─ very flat        → AVIF
                                    │   └─ a little texture → JXL
                                    └─ detailed/busy        → JXL
```

In words:

- **Small images (shorter side ≲ 130 px)** — thumbnails, icons, tiny crops — go to **WebP** when
  the color is clean, **JXL** when there's chroma noise. WebP earns its keep here; it was built
  for exactly this size class.
- **Larger images** split on **detail**:
  - **Flat / smooth** content (gradients, screenshots, low-texture photos) → **AVIF**, whose
    intra prediction shines on smooth regions — *unless* there's a bit of texture, then **JXL**.
  - **Detailed / busy** content (most photographs) → **JXL**, the generalist that wins whenever
    there's real high-frequency content.
- **JPEG is never the answer** for bytes-at-quality, and **PNG never wins** lossless. They stay
  in the toolbox only for compatibility (legacy decoders, no-alpha targets) — not because they
  ever beat the others.

That's the whole content-dependent part. Everything else is a constant or a threshold.

---

## 2. The other two decisions (even simpler)

- **Lossy or lossless?** A threshold on the quality target. Below ~zq94 → lossy; above ~zq97 →
  lossless; the 95–96 band is the only genuinely contested sliver. (Near-perfect quality is
  cheaper to reach by storing it exactly than by pushing a lossy encoder to its limit.)
- **Which lossless codec?** **Always JXL.** It's the smallest on 88% of images; WebP edges it on
  the other 12% (costing ~2% if you don't bother checking); PNG never wins.

---

## 3. What each rung costs (so you can pick your complexity)

Measured as **extra bytes vs. the perfect oracle**, one-shot (no trial encodes), on the
unbiased, fairly-sampled data:

| approach | extra bytes (avg) | worst 10% | size |
|---|---|---|---|
| just always use JXL | **18%** | 48% | a constant |
| the 6-leaf tree above | 8.8% | 24% | a few `if`s |
| an 8–16-leaf tree | 6–8% | 18–20% | a dozen `if`s |
| 2 learned "codec-fit" scores + a 16-leaf tree | **4.6%** | 14% | ~1 KB |
| the full neural net (MLP) | **4.0%** | 11% | 27 KB |
| *(if you could afford 2–3 trial encodes)* | *0.04% = oracle* | *0%* | *— but multi-shot* |

Read it as: **always-JXL is fine on the typical image (its median miss is ~2%) but has an ugly
tail** — sometimes WebP/AVIF beat it badly. The whole point of a picker is taming that tail. A
six-question tree halves it; a dozen questions or a tiny learned model nearly closes it. The
27 KB neural net buys only ~0.6% more than a ~1 KB model — real, but the difference between
"good" and "tight," not "works" vs "broken."

If trial encodes were allowed, encoding the top 2–3 and keeping the smallest **is** the oracle —
but one-shot is a hard requirement, so the tree/model is what stands in for that.

---

## 4. How we got to a clean answer (the findings trail)

The clean tree above only emerged after fixing three data problems that were quietly poisoning
the picker — worth recording so they don't come back:

1. **Missing features (the "experimental" gap).** The chroma-loss + IDCT-roundtrip features were
   compiled out by default, so training never saw them. Fixed: `experimental` is on by default in
   zenanalyze; re-extracted **101 qualified source features, 0 NaN**. (One of those features —
   `xyb_bquarter_chroma_loss` — shows up as a split in the bigger tree, so it earns its place.)
2. **A feature leak.** The old training table mixed source-content features with **zensim
   pair-features** (computed from a specific distorted encode) — information not available at
   pick time. Removing them didn't hurt accuracy and made the picker honest.
3. **A cross-codec sampling bias (the big one).** The sweep dials a generic quality `q` that each
   codec maps to a *different* achieved quality, so above ~zq90 some codecs simply had no data —
   and the oracle silently credited whoever did (AVIF). Corrected, the ranking **flips**: JXL is
   the most-often-best lossy codec (45%), not AVIF (23%). The fix lives in the data layer
   (`scripts/picker/picker_data.py`: compare codecs only where each has measured support) plus a
   gate (`check_quality_coverage.py`) that refuses biased data. Full write-up:
   [`CROSS_CODEC_QUALITY_SAMPLING.md`](CROSS_CODEC_QUALITY_SAMPLING.md).

Everything in §1–§3 is measured on the *corrected* data.

---

## 5. What we'd ship, and what's left

**Ship the tree (or the ~1 KB 2-score model) as the family rule** — rules + a quality threshold
+ JXL-default + the small size/detail/chroma tree. It's auditable, one-shot, needs no ML runtime,
and is within a couple percent of the neural net. Keep the MLP behind a flag for the rare case
that last ~0.6% matters. (The 3 MLP routers are built, shipped, and de-biased today at
`zenpicker::MetaPicker::default_routers()`; this is the proposal to demote them to optional.)

**What's still open:** the tree is only trustworthy up to ~zq88, because that's as high as the
sweep measured *all* codecs (see §4.3). To pick reliably at near-lossless quality we need
**quality-targeted sampling** — encode each codec to hit a common achieved-quality grid rather
than a shared `q` — which is a new sweep mode (the one remaining build). Until then, route
near-lossless targets through the lossless side, which is correct anyway.

---

*Sources: `scripts/picker/{reduce_picker,tree_reduce,tree_composite,tree_export,router_error_anatomy}.py`,
`picker_data.py`, `check_quality_coverage.py`. Models + provenance:
[`benchmarks/router_models_2026-06-30.pointer.md`](../benchmarks/router_models_2026-06-30.pointer.md).*
