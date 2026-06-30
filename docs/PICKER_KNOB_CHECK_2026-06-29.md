# Picker recovery + cheap knob-check disambiguation (2026-06-29)

Two intertwined threads from this session: (1) the **experimental-feature exclusion
disaster** and its recovery, and (2) the **cheap knob-check** design for the
multi-cell codecs whose K=1 pickers are feature-ceiling-limited.

## 1. The experimental-feature disaster (root cause + fix — SHIPPED)

zenanalyze's `experimental` Cargo feature gated the XYB chroma-loss pair
(`xyb444_color_loss` 138 / `xyb_bquarter_chroma_loss` 139) OFF by default.
`FeatureSet::SUPPORTED` shrinks when the flag is off, so **every picker feature
extraction silently dropped those columns** — and `keep_features()` (keep-all)
couldn't add what wasn't extracted. Net effect: pickers trained on an
impoverished feature set for an unknown period.

**Fixes shipped:**
- zenanalyze `default = ["experimental"]` (commit `cc514652`) — useful features
  can't be silently extracted-out again. Build + lib tests green.
- `keep_features()` hard guard (zenmetrics `68d9cc1f`): training **fails loudly**
  (SystemExit 3) if the chroma-loss columns are absent; `ZEN_ALLOW_MISSING_EXPERIMENTAL=1`
  override for corpora that legitimately lack them (HDR). Verified: enriched passes,
  non-enriched exits 3, override returns.
- `zenfleet-vastai` zenanalyze dep pins `features = ["experimental"]` (`ccc192c6`)
  so future sweeps' source-features always include them.

**No re-sweep was needed** — features are source-content descriptors, independent
of encode params, so every encode + GPU-metric score stayed valid. Recovery =
re-extract features (local, ~7 s for 4497 renditions) + re-merge + retrain.

**Recovery results (experimental-enriched, even/odd clean split, held-out TEST):**
the chroma-loss features *improved* the codecs that benefit:
- jpeg_zq: 58.3% → **64.9%** argmin_acc (+6.6 pp), TEST 2.21% ✓
- jpeg_ssim2: **64.2%** ✓ ; jxl_ssim2 **100%** ✓ ; jxl_zq **93.8%** ✓

## 2. webp/avif K=1 is feature-ceiling-limited — and it's MULTI-KNOB

webp/avif K=1 pickers fail the safety gate (~2.8% / ~6.6% TEST overhead, teacher ≈
student). Two features were tried and **measured not to clear it**:
- experimental spatial `xyb_bquarter_chroma_loss`: teacher 36.9% → 37.1% (no move)
- a new **IDCT-roundtrip chroma-subsampling feature** `chroma_subsample_dct_loss`
  (id 140): per chroma 8×8 block, FDCT → quant(jpegli-D2) → IDCT, then the energy a
  2× subsample+upsample removes. Math-validated (5/5 unit tests), strong signal
  (0–21.5, 90% nonzero). Teacher **still 37.0%**, and as a *global* picker input it
  was **net-negative** (regressed jpeg_ssim2 64.2% → 60.4%). Kept in zenanalyze
  (experimental) but NOT in any global picker's input.

**Why no single feature helps:** webp VP8 lossy is **always 4:2:0** (no 420-vs-444
choice — the IDCT feature's target decision doesn't exist for webp). webp's 39 cells
are `method` (m2/m4/m6 + the vp8l lossless family) × `filter`
(def/mpass/parity/plim50/smooth) × `sharp_yuv` (on/off). The K=1 limit is a
**multi-knob composition**, not one discriminator. RD-optimal-cell distribution at a
web-relevant operating point (target zensim 65, n=4497):
- method: `vp8-m6` 80%, but **~16% want `vp8l` (lossless)** — high-impact graphics/
  screen split
- filter: fully content-split (plim50 24 / parity 20 / def 15 / smooth 15 / mpass 11) — no default
- sharp_yuv: 54/46 split — **chroma-rule predictable** (Spearman vs RD benefit:
  chroma_complexity +0.372, **chroma_subsample_dct_loss +0.368**, colourfulness +0.352)

## 3. Cheap knob-check (the design — real-time, no encode)

Metric-K-verify (encode each top-K candidate → decode → score → pick best) is an
**offline** cost; only jpeg's fast encode makes it borderline real-time. For
real-time we want a **cheap disambiguator**: resolve the specific knob-differences
among the picker's top-K using content+knob-semantic rules, **no encode**.

The IDCT feature *failed as a global input but works as a targeted knob-rule input*
(+0.368 for sharp_yuv) — exactly the reframe. Design:
- Decompose the cell choice into per-knob decisions, each individually content-
  predictable (proven for sharp_yuv; lossy/lossless is a strong graphics-feature
  signal; filter is the open question).
- Each knob with a content-split best value gets a rule; near-100%-default knobs get
  a code default. Apply the rules to the picker's top-K to pick the cell.
- Integrate into `zenpredict::picker_safety::resolve_pre_argmin` as a post-argmin /
  top-K refinement stage (no encode, no metric).

## 4. Next steps (active arc)
1. Validate the per-knob decomposition beats the monolithic K=1 picker (achieved RD
   vs oracle) for webp — build the rules (lossy/lossless graphics-rule, sharp_yuv
   chroma-rule, filter rule-or-default) and measure.
2. Same analysis for avif (it DOES have 420/422/444 — the IDCT feature applies there
   as a subsampling knob-rule).
3. Wire the knob-check into the runtime safety pipeline; keep metric-K-verify as an
   explicit offline mode.
4. Ship the clean recovery pickers (jxl intact; jpeg re-trained clean, IDCT dropped).
