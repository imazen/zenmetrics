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

**Quality is the PRIMARY routing axis (user 2026-06-29) — VALIDATED.** Best-family share
over 2307 common lossy variants (min `encoded_bytes` at the target zensim, 7-q RD curves
interpolated) shifts cleanly with quality: avif 42% @zq50 (aggressive) → webp/jxl/avif
~balanced @zq60–75 → jxl 53% @zq85 → avif 56% @zq90. **jpeg is ~0% RD-optimal** — its
value is compatibility/speed (entered via the allowlist or a realtime profile, not RD).
So the meta-model must be quality-conditioned and trained quality-dense. The canonical
grid is 7 q-levels (5/15/30/50/70/85/95) — spanning, and the per-family RD curves
interpolate to dense target qualities; a denser re-sweep would sharpen the family
crossings. **(a) shipped** (`viable` + `EncodeMode`).

**First lossy router (GBDT, the obvious shape) — works.** `scripts/picker/train_lossy_router_gbdt.py`,
held-out (train.parquet → test.parquet, dense target_zq 45–90 step 3, inputs = 469
zenanalyze features + w/h + target_zq): family-acc **75.6%** vs 36.1% always-jxl
baseline (2×). Per-zq 80.9% / 74.3% / 72.3% (low→high). `target_zq` adds **+7.8pp**
(75.6 vs 67.8 without it) — a real axis; content carries the rest (67.8% alone). Design
(user 2026-06-29): **three routers** — lossy (quality-conditioned, this one), lossless
(no quality, min bytes), auto-gate (lossy-vs-lossless given target); model inputs
zenanalyze-all + dims + PixelDescriptor + target zq; constraints = budget(`viable`) +
allowlist + descriptor-capability (alpha→no jpeg, HDR→jxl/avif — rules, not learned).

RD overhead vs oracle (the metric that matters — held-out): **median 0%** (nails the
best family for the majority), **mean 3.91%**, p90 10.81%, can't-reach 1.3%. So
misroutes are mostly between RD-close families (free); the cost concentrates in a p90
tail — clearable by a **meta multi-shot family-verify** (queued mode → encode the top-2
families, keep the best), the family-level analogue of the within-codec knob-check. The
same `EncodeMode` / `EncodeBudget` / `directed_search` apply at BOTH levels (family and
knob). Next: the meta multi-shot + GBDT→MLP bake.

**Lossless router** (`scripts/picker/train_lossless_router_gbdt.py`, GBDT, features+dims,
NO target_zq — lossless has no quality dial; objective = fewest bytes). GOTCHA caught: the
canonical `zenpng_lossless` dataset is only **53.8% truly lossless** — its `modes_full`
sweep mixes in LOSSY palette-quantized png encodes (score_zensim 37–100). A naive
`min(encoded_bytes)` per family compares png's lossy small files against jxl/webp's
true-lossless ones and wrongly concludes "png wins 99%". FIX (mandatory for any lossless
picker): filter `score_zensim>=99.999` before min-bytes. After the filter the ranking is
the expected one — true-lossless oracle winner **jxl 87.7% / webp 12.3% / png 0%** (png
never optimal; jxl/png median 0.711, webp/png 0.758). Router: **91.2% family-acc, 0.69%
mean / 0% median RD overhead** (jxl-vs-webp misroutes are RD-close).

**Auto-gate** (`scripts/picker/train_auto_gate_gbdt.py`, GBDT, features+dims+target_zq →
lossy|lossless; label = best-lossy-bytes(zq) < best-true-lossless-bytes, or lossy can't
reach the target). Sharp, intuitive crossover at **zq~96**: lossless-better is 0.7%→3%
across zq 45–90, 13.7% at zq95, **36.4% at zq96** (the knee), 96.8% at zq97, 100% at zq98.
Router-acc ≥99% everywhere except the 95–96 knee (94%/78% — the genuinely content-dependent
band). Baked rule of thumb: target <~94 zensim → lossy; >~97 → lossless; 95–96 is the
content-aware contested band where the gate earns its keep. ALL THREE routers shipped:
`train_{lossy_router,lossless_router,auto_gate}_gbdt.py`.

**Model shapes** (lossy router, `train_lossy_router_shapes.py`): GBDT **75.5% / 3.90% mean
RD overhead** beats every MLP shape — MLP(256,128,64) 73.9% / 4.65%, MLP(128,64) 73.1% /
4.90%, GBDT→MLP distilled 73.1% / 4.55% (distillation does NOT recover the gap — it's MLP
capacity, not labels; matches the within-codec K=1 finding). So GBDT is the better model;
the bakeable ZNPR MLP gets within ~1.5pp acc / ~0.75pp RD overhead → viable to ship at
~27KB. GBDT eats NaN tiny-cell features natively; the MLP path needs a median imputer (as
the bake already does). Next: bake the MLP routers (ZNPR) + wire into zenpicker's MetaPicker
(viable() mask + per-family est_ms cost model) — needs a MetaPicker public-API shape (3
routers + descriptor-capability rules), so propose + get approval before adding the API.

**API LANDED (2026-06-30, zenanalyze main).** User approved the single-`route()` shape +
MLP/ZNPR, and directed "use zenanalyze-api types exclusively" — so the routing surface is
built on the `Offer` contract, not raw `&[f32]`/zenpixels. Shipped in `zenpicker`:
`QualityTarget {Zq|Ssim2|Lossless}`, `RouteDecision {family, lossless, ranked}` +
`resolve()` (masked argmin, ranked for the queued meta multi-shot), `content_capability(&Offer)`
(alpha→no jpeg, hdr/bit-depth→jxl/avif/png — rules), `AllowedFamilies::{intersect, LOSSY,
LOSSLESS}`, and **`MetaPicker::route(offer, target, allowed, mode, latency, est_ms)`** — narrow
(caller ∩ capability ∩ viable) → auto-gate (explicit Lossless bypasses; else the gate model) →
branch family router → resolve. `with_router(gate, lossless)` builder; 26 tests incl. 6
end-to-end with tiny baked gate/lossy/lossless models; clippy+fmt clean; no_std+alloc builds.
Commits 1f9e6913 (core) + e025a0a3 (route). REMAINING — the real-model bake: re-extract the
router corpus features with current zenanalyze (NaN-free tiny cells, no imputer — the canonical
2026-06-27 feat_* predate the tiny-handling fix so they carry NaN), train the 3 MLPs on clean
features, bake to ZNPR (cells=families via zentrain `train_hybrid`; gate is 2-class), ship the
.bin set + wire a consumer. Memory: [[cross-codec-meta-router-3way]].

**Clean re-extraction + contract fix DONE (2026-06-30).** Re-extracted all 4497 router-corpus
renditions with current zenanalyze via `examples/extract_features_for_picker --sizes native
--features api` → **101 qualified `name@hex8` source features, experimental-complete
(xyb444_color_loss / xyb_bquarter_chroma_loss / chroma_subsample_dct_loss all present +
populated), 0.0000% NaN** (current tiny-cell handling needs no imputer). Sidecar:
`/mnt/v/output/router-features-2026-06-30/zenanalyze_features.parquet` (4497 × 101 + dims;
keyed `variant_name`). **Caught a LEAK in the old routers:** the canonical 2026-06-27 parquets
carry 97 named source feats + **372 positional `feat_N` that are zensim (ref,distorted) PAIR
features** (verified: constant-per-variant for named, 330-distinct-per-330-rows for `feat_N`) —
not available at routing time. The earlier GBDT routers trained on all 469, so they ingested
encode-dependent inputs. Retrained the lossy router on the clean 101 source-only features:
**76.2% family-acc (BEATS the old leaky 75.5%)**, 3.92% mean / 0% median RD overhead — so the
zensim features were dead weight, and the fix is a strict win (valid, 4.6× smaller feature set).
Dims (`width`/`height`) add only +0.4pp (76.2→75.8 without) → DROP them: router = 101 qualified
+ target, so `route()`'s `reuse_for(101 qual) + append(target)` works unchanged, no API/Offer
change. Trainer: `scripts/picker/train_router_clean.py`. REMAINING: bake the 3 MLPs (101-feat
classifier → ZNPR, negate logits + pad to 6 CodecFamily outputs; gate 2-out) via `zenpredict-bake`,
verify, ship `.bin`.
