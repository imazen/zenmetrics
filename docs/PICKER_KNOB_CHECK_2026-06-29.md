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

## 4. Knob-check PROVEN (webp, held-out odd-origin TEST, target zensim 65)

- **Separability ceiling:** decomposed oracle (pick each knob independently) = 0.86%
  mean overhead vs joint oracle; 76% within 1%, **93% within 3%**. webp's lossy knobs
  (method × filter × sharp_yuv) ARE separable.
- **End-to-end (per-knob HistGB classifiers on content features, held-out):**
  - method 93.7% acc (m6-dominant + lossy/lossless split — predictable)
  - sharp_yuv 62.2% (the chroma rule; chroma_subsample_dct_loss is a top input)
  - **filter 31.6%** (5-way, ~20% random floor — content-underdetermined)
  - achieved **2.29% mean overhead — beats the monolithic K=1 picker (2.8%)**; median
    0.97%, p90 5.65%.
- **The filter is the residual bottleneck:** ~1.4% of the gap to the 0.86% ceiling is
  filter mis-prediction, and it drives the p90 tail (so the cheap knob-check alone may
  still graze the safety gate's tail thresholds). The 5 filters (def/mpass/parity/
  plim50/smooth) are near-RD-equivalent + weakly content-correlated.

**Verdict:** the cheap, no-encode knob-check WORKS (separable + beats monolithic K=1),
and it localizes webp's hardness to one dimension (filter). method + sharp_yuv ship as
cheap content rules.

## 5. Next steps (decision + build)
1. **Filter handling — RESOLVED: size/budget-adaptive (user, 2026-06-29).** webp encode
   is expensive at large sizes, so the codec picks the mode from a resource/time budget
   + a `PickerStrategy` enum:
   - **one-shot** (large images / tight budget): the cheap no-encode knob-check — accept
     the ~2.29% (better than the 2.8% monolithic; fully real-time).
   - **multi-shot** (small images / budget allows): targeted *filter-only* verify —
     encode the 2-3 candidate filters at the fixed predicted method+sharp_yuv, keep the
     best. Only the one ambiguous dimension is verified (not the full 39-cell K-verify).
   - **`Auto`**: gate on `pixel_count` vs the budget's affordable encode passes
     (estimate webp encode cost ∝ pixels). Metric-K-verify is the explicit fully-offline
     mode. Foundation built in `zenpredict` (`encode_strategy`): the budget→passes
     decision; the picker emits ranked candidates; the codec runs the encode loop.
2. Wire the knob-check into `zenpredict::picker_safety::resolve_pre_argmin` as a
   post-argmin per-knob refinement; keep metric-K-verify as an explicit OFFLINE mode.
3. Repeat for avif (it DOES have 420/422/444 — the IDCT feature applies there as a real
   subsampling knob-rule, unlike webp).

## DONE this cycle
- Experimental disaster fixed + shipped (cc514652 / 68d9cc1f / ccc192c6).
- Clean recovery pickers shipped to codec crates: **zenjpeg v0.2** (ssim2 64.2% / zq
  64.9%, K=3) + **zenjxl v0.2** (ssim2 100% / zq 93.8%, K=1) on `origin/main`,
  superseding pre-experimental v0.1. IDCT feature kept in zenanalyze (knob-rule), out
  of the global pickers (net-negative there).
- `encode_strategy` foundation on zenanalyze/zenpredict `main` (18a99393): `PickerStrategy
  {OneShot, MultiShot, Auto}` + `EncodeBudget` + `passes()`, 4 tests.
- IDCT-roundtrip feature on zenanalyze `main` (6499cf26): id 140, experimental-gated,
  golden re-blessed, discriminant boundary bumped (141 now first-unused). Worktree
  merged + cleaned up.

## Multi-shot loop design (2026-06-29)

Outer controls (built, `zenpredict::encode_strategy`): `PickerStrategy {OneShot,
MultiShot, Auto}` (mode) + `EncodeBudget` (multi-axis ceiling: max_passes ∧
max_trial_pixels ∧ max_ms). `resolve(strategy, n_candidates, image_pixels,
est_ms_per_encode)` → trial count (codec supplies its own est_ms); `time_exhausted(
elapsed_ms)` → runtime safety stop.

Inner loop (to build):
- **Picker emits candidates WITH predicted (zensim, bytes)** — not just a rank — so the
  search can order/navigate them. They differ in the ambiguous knob (webp `filter`)
  ± a quality `q` step.
- **Directed search**, not a blind sweep: caller gives a target (target zq or byte
  ceiling) + a preference (quality-priority vs bytes-priority). After each trial measure
  achieved (zensim, bytes); overshoot the target → step toward a leaner candidate,
  undershoot → step up; select the ambiguous knob among trials that landed near target.
  Converges in fewer trials, bounded by `EncodeBudget`.
- **Pairwise (streaming) evaluation for peak RAM**: a tournament that holds only the
  reference (decoded) + the current trial (decoded) + a transient diffmap; the running
  best is `(score, encoded_bytes)`, NOT a decoded image. Per trial: encode → decode →
  zensim(ref, decoded) → score; beat best ⇒ keep encoded bytes + score, drop the decoded
  trial + diffmap. Peak RAM = ref + 1 trial + 1 diffmap, **independent of K** (2 images
  is the reference-metric floor).

## REMAINING (the integration arc)
1. **zenpredict**: extend `resolve_pre_argmin` to emit the ranked candidate list with
   predicted (zensim, bytes) + a small directed-search-policy helper (next-candidate
   from achieved-vs-target + preference). Pure/testable, no codec-path risk.
2. **zenwebp multi-shot loop**: directed search + streaming pairwise zensim eval,
   bounded by `EncodeBudget::resolve` / `time_exhausted`. First codec integration.
3. **One-shot rules**: the picker's top-1 is the one-shot pick (method/sharp_yuv are
   already content-predictable); no separate baking needed.
4. **avif**: has real 420/422/444 → the committed IDCT feature applies there as a
   subsampling knob-rule (unlike webp). Repeat the separability + per-knob analysis.

## zenpicker codec router (design, 2026-06-29)

Route an encode to the best CODEC FAMILY (jpeg/webp/jxl/avif/png) given four inputs:
- **format allowlist** — `AllowedFamilies` (exists in zenpicker).
- **target quality** — zq or ssim2, engineered into the features (`zq_norm`) so the
  meta-model is quality-conditioned (families win at different q: avif/jxl aggressive,
  jpeg/webp high-q).
- **resource budget** — `EncodeBudget` + a per-family encode-cost estimate.
- **mode** — `EncodeMode {RealtimeFastest, RealtimeBalanced, QueuedBalanced,
  QueuedAggressive}` (BUILT, zenpredict): a latency × effort profile. `is_realtime()`
  gates codec viability (real-time prefers fast codecs); `strategy()` → per-codec
  `PickerStrategy` (realtime→OneShot, QueuedBalanced→Auto, QueuedAggressive→MultiShot);
  the Fastest/Balanced/Aggressive tier also drives the codec's effort knob.

Logic: `argmin(meta_model, features+zq)` over `allowlist ∩ viability(mode, budget,
per_family_est_ms)` — the viability mask drops families too slow for the budget/mode
(realtime masks avif/jxl when `max_ms` is tight). The chosen family then runs its
per-codec picker + the multi-shot loop (`mode.strategy()`).

Data EXISTS: the canonical per-codec parquets (`s3://zentrain/canonical/2026-06-27/`)
carry per-family RD (zensim/ssim2 + bytes) + `encode_ms` + 372 features. Join across
families → label = best family per (image, target-quality); `encode_ms` → the per-family
cost model. So the meta-router is trainable (zq-conditioned, like the per-codec pickers).

Build: (a) `viable_families(allowlist, mode, budget, per_family_est_ms)` mask + enhanced
`MetaPicker::pick` (structural, pure/testable, no data); (b) join the canonical parquets
→ meta-router training data; (c) train the zq-conditioned meta-model + bake (supersedes
the current allowlist+features-only `MetaPicker`).
