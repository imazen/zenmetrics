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

## Open question this does NOT settle

This crate uses the **`ndarray` backend**, which pulls **no cubecl at all**, and
lives in its own `[workspace]` to dodge the parent graph. So it proves the
*ergonomics* decisively but does **not** yet prove that burn's cubecl backend
can **coexist in one cargo graph with the `zenforks-cubecl` fork** (a *renamed*
package — `zenforks-cubecl-core` ≠ `cubecl-core` to cargo). That fork
reconciliation is the remaining integration risk before burn-on-GPU lands in the
real zenmetrics graph; see the ML-framework eval notes. Resolving it is the
natural follow-up (options: upstream the fork deltas, `[patch]` burn's backends
onto the fork, or bridge at the byte layer).

*NOT a published crate. Spike only.*
