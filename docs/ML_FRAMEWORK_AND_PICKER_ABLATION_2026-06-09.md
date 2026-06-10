# ML Framework Eval + Picker Feature/Knob Ablation (2026-06-09)

Durable summary of the 2026-06-09 candle/burn/linfa-vs-zen-ML analysis, plus the
**picker feature/knob ablation design** (which is captured nowhere else). The eval
sections are terse and point at the runnable artifacts; §4 (ablation) is the meat.

**Sources / cross-refs:**
- `crates/burn-ranknet-spike/README.md` — burn training spike + GBDT teacher/student spike (runnable, measured).
- `zenmetrics/CLAUDE.md` "## burn:" section — kernels-abandoned ≠ training-viable; don't add burn to the zenforks graph.
- Commits: `21044c63`, `704b19dd`, `95dec9e3` (zenmetrics) — burn ranknet + de-trap + model-size measurement.
- `zenanalyze/zenpicker-train/src/bin/cluster_features.rs` (commit `96ccf86`) — linfa k-means stratified-sampling tool (the ablation corpus selector).
- Live, related: `zenanalyze/zenpicker-train/src/bin/picker_tree_ab.rs` (another session, real-sweep tree A/B) — the natural instrument for §4.

---

## 1. Framework verdict — candle / burn / linfa vs zen ML (3 layers)

| layer | verdict |
|---|---|
| **Inference** (`zenpredict`, ~20KB, ZNPR) | **KEEP.** Wins on wasm binary size + small-tensor dispatch + the baked decision pipeline. NOT a no_std argument — the workspace is std-using; the real constraint is wasm binary size (every KB ships to the browser). burn-inference fine server-side. candle disqualified (≈300K-SLoC deps, eager dispatch ~8.5× off PyTorch on small models, rayon/memmap2 awkward on wasm). |
| **Training** (`zentrain` Python + `zensim-train-core` hand-rolled Rust) | **burn is the opening.** Unifies the split stack, gives autodiff (unblocks FiLM/MoE), and rides the SAME cubecl 0.10 zenmetrics already pins. Proven: `burn-ranknet-spike` trains a RankNet MLP w/ custom pairwise+monotonic loss → 0.998 pair-acc. |
| **GPU metrics** (hand `#[cube]` kernels) | **KEEP hand kernels.** `burn-conv-spike` measured burn GEMM conv2d 4.32× slower on 1-ch separable. burn only relevant here for differentiable metrics (someday). |

**Classical ML:** linfa 0.8.1 (pure-Rust) for k-means stratified sampling (shipped: `cluster_features`) and a credible picker RF/GBDT A/B. No GBDT in linfa → **forust-ml 0.5.0** (pure-Rust XGBoost-algo) for trees.

## 2. burn training: viable, SEPARATE binary, NOT in the zenforks graph

Architecture decision: training runs as a **standalone binary** (burn + own cubecl)
exchanging **parquet/ZNPR data** with the zenforks-cubecl metric crates — they never
share a cargo graph, so the "two-cubecls" problem is **sidestepped by construction**.
The fork can't extend into burn (`zenforks-cubecl` is a *renamed* package; `[patch]`
re-sources but can't rename; the rename exists so GPU crates can be *published*).
Only **differentiable metrics** would force one graph. Full mechanics + 3 reconciliation
options in `crates/burn-ranknet-spike/README.md` + the CLAUDE.md burn section.

## 3. GBDT teacher / GD-MLP student (measured, synthetic methodology demo)

Framing: GBDT (incl. LambdaMART for ranking) is the better **learner** on tabular
features; the GD-MLP is the better **deployable** (smooth/monotone, tiny ZNPR bake).
So **distill** the teacher into the shippable student rather than choose.

Measured on interaction-heavy synthetic data (`gbdt-teacher-compare`, held-out pair-rank):
- direct MLP **~0.838–0.846**, GBDT teacher **0.861**, distilled MLP **~0.847–0.863**.
- Teacher edge **+0.015–0.023** (trees catch axis-aligned + multiplicative interactions the small MLP misses). Recovered-% noisy (student unseeded; edge small vs init variance → use 5-seed averaging).
- **Model size (measured):** GBDT = **975 KB raw JSON / 109 KB gzipped** (forust JSON-only) vs **~27 KB** zensim ZNPR MLP bake → ~36×/~4× bigger. **This is the shippability case for distilling teacher→MLP**, not shipping 100 trees into a codec/wasm hot path.
- I/O caveat: the demo is scalar-out / abstract-features-in — **no `zq` column**. Real picker conditions on target quality (add `zq` as an input col; trivial for GBDT).

---

## 4. Picker feature/knob ABLATION design

**Inputs = zenanalyze features (~102 active, IDs 0–121; +90 dense-percentile on a branch). Outputs = encoder knobs (per codec: chroma_scale, lambda/trellis, q/distance, effort, XYB, AQ, deringing, progressive, dequant_bias, …).** Goal: scope the picker — which features to extract, which knobs to predict — *before* heavy training, to cut model size, feature-extraction cost, and overfitting/OOD risk.

### 4.1 It's a conditional matrix, not two rankings
Feature relevance is **conditional on which knob and which quality regime.** Features that predict `chroma_scale` (chroma complexity, Cb/Cr sharpness, skin-tone) are nearly disjoint from those for `lambda` (DCT energy, noise floor, texture) or `effort` (size/time budget). The real object is a **features × knobs × zq-band (× mode)** importance tensor. A global feature ranking will delete features that were the *only* signal for some knob. **`zq` (target quality) must be an input** — at high zq almost nothing about content moves the knob; at low q (q5–q40, where structural decisions live) content dominates.

### 4.2 Input side — ablate by redundancy *cluster*, not by feature
- zenanalyze features have tight redundancy clusters (ρ≥0.95; dendrogram already at `zenanalyze/benchmarks/feature_groups_*`). Drop one of two correlated features and importance migrates to its twin → false "both unimportant." **Ablate at the cluster / leave-one-group-out level.**
- Prefer **permutation / LOGO importance over gain** (gain is biased toward high-cardinality continuous features; underrates binary-ish flags like `is_grayscale`).
- **Per surviving knob.** Payoff is concrete: Tier 1/2/3 cost ~1/2/3 ms per 4MP — if Tier-3 DCT features (compressibility, AQ map, noise floor, quant survival) move no surviving knob, drop a whole tier *and* shrink the model.

### 4.3 Output side — the underexploited, higher-value half
Few systematically ask **"which knobs are worth predicting?"** A knob earns a head only if **both**:
1. **Real RD spread** — sweep it, measure best-vs-worst metric delta. If optimal vs a fixed default differs <~0.5 zq / sub-JND butteraugli across the corpus → **pin the default, delete the head.**
2. **Content-dependent optimum** — if the best setting is the same image-to-image, a constant beats a picker. Only predict knobs whose argmax *moves with content*.

Pinning dead knobs collapses output dimensionality, shrinks the model, removes failure modes, and reduces the future sweep grid. **Watch knob coupling** (chroma_scale × lambda, XYB × trellis interact) — measure spread/optimum-variance on a **joint** sweep, not one-knob-at-a-time.

### 4.4 Methodology guardrails (this is a source-informing measurement)
Survivors become `const`s in source → CLAUDE.md sweep discipline binds:
- **Dense q + 4 size buckets + content-stratified.** A feature irrelevant at q90 can be decisive at q20. Use `cluster_features` (linfa k-means) so the ablation corpus spans content classes — don't compute importances on a photo-only/Kodak-ish corpus.
- **Per-mode / per-codec.** Chroma features matter at 4:2:0 not 4:4:4; importances differ XYB on/off. Ablate per mode or fold the mode flag in as an input.
- **Judge by RD, not training loss.** Gate = "does dropping this feature/knob cost real bytes/quality on held-out content?" (BD-rate / zq-at-fixed-bytes via actual encode→metric round-trip), never the GBDT's training loss.

### 4.5 GBDT is the right INSTRUMENT (even if you ship an MLP)
Train **one GBDT per knob**, read importances → populate the features×knob matrix without the MLP seed-CI loop. The GBDT is a feature-selection oracle here, not just a teacher. `picker_tree_ab.rs` (live in zenanalyze) is positioned to emit exactly this: per-knob importance ranking + per-knob RD spread = the ablation.

### 4.6 Where it misleads — caveats
- **Interactions:** features that only matter jointly (sign-XOR-like — exactly what beat the MLP in §3) are underrated by single-feature permutation importance. Group/LOGO + retrain is gold standard but expensive.
- **Gain-bias:** don't ship the final cut off tree gain alone; permutation or SHAP for the decision.
- **Knob coupling:** can't ablate coupled knobs independently — joint sweep.

### 4.7 Sequencing (do output ablation FIRST — bigger lever)
1. **Sweep** (substrate; persist encoded bytes + all metric variants per the persist rule).
2. **Output ablation** — per-knob RD-spread + optimum-variance → pin the dead knobs.
3. **Input ablation** — per-surviving-knob GBDT permutation importance, by redundancy cluster; build the features×knob matrix; drop features below threshold for *all* survivors (weight by tier cost).
4. **Validate** — retrain picker (MLP or distilled) on pruned I/O; confirm held-out RD within tolerance vs the full model.

---

## 5. Status / what's live (2026-06-09)
- `cluster_features` (linfa k-means stratified sampling) **shipped** zenanalyze main `96ccf86` — the §4.4 corpus selector.
- `burn-ranknet-spike` + `gbdt-teacher-compare` **shipped** zenmetrics master (`95dec9e3`) — §2/§3 instruments.
- `picker_tree_ab.rs` **live** (another session, zenanalyze) — the §4.5 instrument; its natural output is the per-knob importance + RD-spread tables of §4.2/§4.3.
- Machine-safety: all ML builds must run under `nice` (global CLAUDE.md, 2026-06-09 lockup incident).
