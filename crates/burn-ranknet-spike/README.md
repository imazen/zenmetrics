# burn-ranknet-spike

**Question:** Can [`burn`](https://burn.dev) (0.21) train a RankNet-style
perceptual-metric / picker MLP — the kind `zensim-train-core` currently trains
with **hand-rolled backprop** — using autodiff and a *custom* multi-term loss?

**Verdict: PASS.** burn autodiff trained a 8→16→1 MLP on a pairwise-ranking +
monotonicity-hinge loss to **0.998 pair-ranking accuracy** (from 0.46 random
baseline) in 400 epochs on CPU. The loss that `zensim-train-core` hand-derives
(`mlp.rs` backward + `adam.rs` + the per-sample-α / pool-head chain rules) is
~15 lines of tensor ops here, with gradients produced by autodiff rather than
manual derivation.

```
epoch    0  loss 0.7323 (rank 0.7022, mono 0.0602)  pair-acc 0.4635
epoch   50  loss 0.3325 (rank 0.3270, mono 0.0111)  pair-acc 0.9370
epoch  100  loss 0.1492 (rank 0.1487, mono 0.0010)  pair-acc 0.9870
epoch  200  loss 0.0746 (rank 0.0745, mono 0.0002)  pair-acc 0.9980
epoch  400  loss 0.0373 (rank 0.0372, mono 0.0001)  pair-acc 0.9980
```

Run it: `cargo run --release` (CPU, ~seconds, no GPU driver needed).

## What this is NOT

This is **not** a re-litigation of `burn-conv-spike`. That crate answered a
*different* question — burn/cubek **conv kernels for GPU metric compute** —
and the verdict there was **ABANDONED** (4.32× slower than the hand-written
separable Gaussian stencil at 4000×3000). Training small MLPs is orthogonal to
GPU-kernel throughput: this spike tests **autodiff + custom-loss + optimizer
ergonomics**, and on that axis burn is a clean win over the hand-rolled path.

## Why it matters

zen's metric/picker **training** is split across Python (`zentrain`, sklearn +
torch) and hand-rolled Rust (`zensim-train-core`), with **no autodiff** — which
is exactly why FiLM / MoE heads sit in "deferred" (hand-deriving their gradients
is too error-prone). This spike confirms the training half of the
"train in burn → bake to ZNPR → infer in zenpredict" path:

- Custom RankNet (pairwise-logistic) + monotonicity loss = plain tensor ops.
- Adam + the train loop = `optim.step(lr, model, grads)`; the whole `adam.rs`
  and the manual chain rules go away.
- burn is generic over `Backend`, so the same `Mlp<B>` swaps from the
  `ndarray` CPU backend used here to the **cubecl** CUDA/wgpu backend (the same
  cubecl 0.10 line zenmetrics already pins) by changing one type alias —
  giving GPU training that `zensim-train-gpu` declares but doesn't implement.

## Architecture decision (2026-06-09): separate binaries, NO shared cargo graph

Training runs as a **standalone binary** (burn + its own cubecl) that talks to
the metric/codec side **only through data** — parquet feature/label sidecars in,
ZNPR bakes out. zenmetrics' GPU metric crates keep the published `zenforks-cubecl`
fork; the trainer never imports them. So burn and `zenforks-cubecl` **never share
one cargo graph**, and the "two-cubecls" problem is **avoided by construction**,
not a pending risk.

Why the fork can't simply extend into burn (so nobody re-litigates this):
`zenforks-cubecl` is a *renamed* package (`cubecl = { package = "zenforks-cubecl" }`
in our manifests). burn's own backend crates depend on the real `cubecl-core`,
and you can neither rewrite burn's source nor `[patch]`-rename it (`[patch]`
re-sources, it cannot rename). The rename itself exists so our GPU crates can be
**published** (`[patch]` is build-local and doesn't survive `cargo publish`).
Publishing-with-fork and composing-with-burn want opposite mechanisms — so we
keep them in **separate binaries** and hand off data, not tensors.

The **only** scenario that forces them into one graph is **differentiable metrics**
(burn autodiff backprop *through* an existing zenforks-cubecl kernel, for
end-to-end encoder training). Not on the table today. If it ever is, the options
are: (a) upstream the fork deltas to vanilla cubecl — check first whether
burn-*training* even needs them (they're metric-kernel correctness, not training);
(b) `[patch]` that one crate onto the git fork (original `cubecl-*` names), kept
free of any `zenforks-cubecl` dep; (c) nuclear — fork burn's backend crates too.
**Do NOT add `burn` to the zenmetrics workspace's (zenforks) graph expecting it
to "just work" — it won't, and that's intentional, not a bug to fix.**

## Binaries

- `burn-ranknet-spike` (`src/main.rs`) — the ergonomics proof above.
- `gbdt-teacher-compare` (`src/bin/teacher_compare.rs`) — GBDT (forust) teacher
  vs the GD-MLP student, plus distillation, on interaction-heavy synthetic data.
  Answers "is a tree-ensemble teacher worth distilling into the shippable MLP?"

  **Result (synthetic, 2026-06-09):** held-out pair-ranking accuracy — direct MLP
  **0.842**, GBDT teacher **0.861** (+0.019: trees capture the axis-aligned +
  multiplicative interactions the small MLP misses), distilled MLP **0.847**
  (recovers only ~24% of the gap). Honest read: the GBDT edge is real, but
  distilling into the SAME 10->16->1 student only partly transfers it — student
  *capacity*, not training signal, is the limiter here. To capture the teacher's
  edge you'd widen the student, ship the GBDT directly, or accept the gap.
  Real-picker numbers need real labeled sweep data; this is a methodology demo.

*NOT a published crate. Spike only.*
