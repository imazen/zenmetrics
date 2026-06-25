# JXL Lossy Knob-Space Ablation — Long-Term Program

**Status:** active. Started 2026-06-25. Pilot = jxl lossy; method generalizes to webp/avif/jpeg
+ the lossless/modular partition + the cross picker.

## The goal

Find the **minimal multidimensional knob shape** a JXL lossy picker should explore — which knobs,
and which *crosses* (interactions), carry **content-dependent RD value worth picking** — and push
everything else into **code** (a fixed default, or a feature-derived rule baked into the encoder).
Iterate fast on the Hetzner fleet until the grid stabilizes and a trained picker's achieved RD ≈ the
oracle. The early rounds *discover the questions* (which axes even matter); the later rounds *answer*
them (the picker, and a leaner codec whose defaults absorbed the settled knobs).

We already know the knob *surface* (≈50 `LossyConfig` fields + 14 expert `LossyInternalParams` — see
the 2026-06-25 history mine). What we don't know, and this program resolves:
- which knobs move bytes/quality enough to be worth picking at all (vs. a fixed default);
- which are **content-dependent** (→ a picker axis or a codec rule) vs **universal** (→ a code constant);
- which **interact** (→ keep the cross) vs **separable** (→ code the main effects, drop the cross);
- and therefore the smallest (axes × crosses) shape that captures ~all achievable RD.

## Decision rules — code vs picker vs rule

For every knob and every cross, the sweep data forces one verdict:

| finding in the data | verdict |
|---|---|
| never on the Pareto front, or byte-inert | **CODE** — drop the axis, keep the default |
| best value is the same regardless of content | **CODE** — set that value as the default |
| best value is a deterministic function of cheap features | **CODE RULE** — bake the rule into the encoder, not the picker |
| best value is content-dependent **and** materially moves RD | **PICKER axis** — the picker chooses it |
| joint optimum ≠ (main-effect optima combined) | **keep the CROSS** — a genuine interaction |
| joint optimum = main effects combined | **separable** — code the main effects, drop the cross |

The picker should end up choosing among the *smallest* set that captures the gain; everything
separable, universal, or inert becomes code. A knob graduating from "swept" to "coded" is a
**success**, not a loss — it shrinks both the codec's surface and the picker's input space.

## The iteration loop (time-efficient)

1. **Design the grid** from current knowledge: *cross* the candidate-interacting axes; *single-
   deviation-probe* the unmeasured ones (the 9 ASSERTED-only experts); all efforts **e1–e9**; dense
   distance (denser at low-q AND at d≥3 where the W44 RD wedges live); content strata.
2. **Sweep fast on Hetzner** via the job system (zenfleet: per-cell, resumable declare→gap→reconcile).
   Lossy is **0.2 GB/cell**, so a cheap CPU fleet rips through it. **Persist everything** — variants +
   every metric variant + diffmaps + features — to `zentrain`, so re-analysis never re-encodes.
3. **Analyze.** Per axis/cross: Pareto-front win-rate, RD-gain magnitude, content-dependence (does the
   optimal value vary by content class / feature?), and interaction (joint vs main-effects). GBDT
   feature-importance + RD-spread (the picker-ablation methodology in
   `docs/ML_FRAMEWORK_AND_PICKER_ABLATION_2026-06-09.md`).
4. **Prune + pivot.** Apply the decision rules: drop the losers, promote the interactors, refine the grid.
5. **Edit the jxl crates.** Code the settled defaults/rules (jxl-encoder / zenjxl sweep axes); shrink the
   sweep to the surviving pickable shape. Re-pin codec-commit provenance (don't mix pre/post-edit rows).
6. **Repeat** until: no new axis earns its place for two rounds running, **and** the trained picker's
   achieved RD ≈ oracle. The residual oracle gap is the signal — it points to the next axis to add, or
   tells us to stop.

## Yes — it re-sweeps as the grid ablates/expands, and incrementally

This program **does more sweeps in response to discoveries** — that's the whole loop, not a one-shot.
Each cycle that changes the grid re-runs the fleet, but the job system's **declare→gap→reconcile**
makes it a cheap *delta*, never a from-scratch re-run:

- **EXPAND** (a discovery says "cross axis X with Y", or "add effort eN", or "the oracle gap wants knob
  Z"): declare the *superset* manifest; the **gap = only the not-yet-done cells**, so the fleet encodes
  ONLY the new combinations — every existing cell is already content-addressed in the ledger and is
  skipped. Adding a dimension costs only the new dimension's cells.
- **ABLATE** (a knob/cross is proven inert/universal → coded): drop it from the next declare. Its already
  encoded variants + scores **stay persisted** for reuse/audit (persist-everything) — never re-encoded,
  never deleted. The next picker just trains on the smaller surviving surface.
- **Re-analysis is free** (no encode): a new metric, a new feature, a new Pareto cut re-reads the
  persisted sidecars/variants. Only an *encode-changing* grid edit triggers fleet work.

So the loop genuinely grows and shrinks the explored shape over time, and the **goal condition is
unsatisfiable without it** — "grid stable (no new axis earns its place two rounds)" can only be
*verified* by actively probing candidate expansions and watching them fail to win, and the oracle gap
can only close by adding the axes that close it. Caveats: a *large* expansion (a whole new cross
dimension) is a flagged fork — it surfaces for a go-ahead before spending the fleet, rather than
auto-expanding; routine prune-and-code of a clearly-dead axis is autonomous. More cells = more fleet
spend, bounded by the standing guardrails (kill idle boxes, `--plan-budget`, persist-no-re-encode).
This incrementality is *the* reason the vehicle is the job system, not a monolithic sweep.

## Phases (concrete first cycles)

- **P0 — main effects (cheap, first).** Single-deviation probes of every candidate lossy knob × e1–e9 ×
  dense distance × content strata. Output: which knobs *ever* win, and whether their best value is
  content-dependent. Immediately codes the inert/universal ones. This is the bulk of the
  question-discovery.
- **P1 — pairwise crosses** among P0 survivors: detect the interactions. Output: which crosses to keep
  (everything else → code the main effects).
- **P2 — code the settled.** Push universal / rule-derivable knobs into jxl-encoder defaults or rules;
  shrink the sweep to the pickable shape. This is where the codec gets leaner.
- **P3 — picker train + oracle gap.** Train lossy / lossless / cross pickers on the reduced shape; measure
  achieved-vs-oracle RD. The residual gap → the next axis → loop back to P0/P1 on it.
- **e1–e9 all sweep in the first pass** — each effort adds a real VarDCT gate (e9 = lz77 +
  enhanced_clustering at the kTortoise ceiling), so excluding any of them risks wrongly burying a knob
  that only pays off at that effort. **e10–e12** only ever enter under `--features butteraugli-loop` —
  without it they are byte-identical to e9 on the lossy path (butteraugli_iters / seeds only, both
  feature-gated).

## Discipline + efficiency constraints

- **Fleet:** Hetzner CPU via the job system (per-cell ≤1.5 GB even for modular, resumable). codec-corpus
  is **read-only** (corpus); runs write to **zentrain**; never `zen-tuning-ephemeral`.
- **Persist-everything** (ML-pipeline discipline): variants content-addressed + all metric variants +
  diffmaps + features, so a new metric or a re-analysis never re-encodes.
- **Schema:** reuse `jxl-encoder/benchmarks/lossy_pareto_2026-04-30.tsv`'s columns (+ CVVDP) — but
  *re-run on current master*: that file and the off-repo `/mnt/v/zen/picker-training/zenjxl-2026-05-04/`
  pareto predate the W44 cost-model retuning, so their absolute bytes are stale.
- **Honor the known skip-list** (byte-inert): `fine_grained_step{4,8}`, `faster_decoding` lossy tier 2,
  modular predictor 5/15 (#67), `nb_rcts 0`, rct19, gather_dedup, `lossy_search_seeds` (without buttloop).
  `try_dct32` is gate-shadowed under Zenjxl defaults — sweep it only under a Libjxl-strategy stratum.
- **Content-gate gotcha:** `adapt_to_image_lossy_with_smoothness` + try_dct32 self-suppression flip knobs
  by image+distance. The sweep must **pin via raw `LossyInternalParams` overrides** (they bypass the
  gates) *or* stratify by content class so each cell's gate fires consistently — otherwise the data is
  contaminated by the encoder's own dispatch.
- **Metric coverage:** every measured RD verdict to date is butteraugli/SSIM2-gated; the CVVDP cost-model
  was never re-fit. Carry CVVDP through every sweep so the picker can be metric-agnostic.
- **Provenance:** codec-commit SHA on every sweep (`build.rs` → plan manifest); never mix codec revs.

## Known starting state (2026-06-25 history mine)

- **Measured-valuable** (cross / keep): `ans_histogram_strategy` Approximate at e≤7 (−1.36%); `strategy`
  {zen,libjxl,lean} gates; the `entropy_mul_table` content-gated "variant-Z" family (high-d-photo /
  screenshot suppressed).
- **Measured-negative** (do NOT re-test without a new mechanism): `k_ac_quant`=0.65 global/gated, cfl
  two-pass widen to e5/6, cfl Newton port (+7.82%), try_dct64 widen to e5/6, global entropy_mul lift.
- **Unmeasured — the P0 prizes** (wired for libjxl parity, never RD-ablated): `try_dct16`, `try_dct4x8_afv`,
  `k_info_loss_mul_base`, the `entropy_mul_table` rebalance itself, `cfl_two_pass` (the feature),
  `chromacity_adjustment`, `patch_ref_tree_learning`, `non_aligned_eval`, `enhanced_clustering_vardct`.
- **`k_ac_quant`** is the sanctioned **learned-dispatch** axis (issue #25): its scalar ladder cells are the
  per-image dispatch training data — keep it as a picker axis, not a code constant.
- **Blocked:** `chroma_subsampling=420` — the single highest-value missing web axis — returns
  `InvalidConfig` today (jxl-encoder #47 chunk 4 unwired). Unblocking it is a jxl-encoder task and a
  prerequisite for the web picker to ever choose 420.

## Current state / next action

- Partition mechanism shipped: `scripts/sweep/partition_cells_by_mode.py` (lossy/lossless split).
- **Next:** draft a `lossy_dense` `SweepAxes` in zenjxl — P0 single-deviation-probe sizing first
  (cross distance × e1–e9 × strategy × gaborish × ans; probe the 9 unmeasured experts + the k_ac_quant
  ladder + entropy_mul presets one-deviation-at-a-time → ~3–5k cells/image), widen to crosses only where
  P0 shows promise.
- **Gated on:** the concurrent "one Rust worker" refactor settling before the first fleet run (don't fire
  a multi-hour sweep into a moving worker). Bucket design + per-cell memory both verified.
