# burn-conv-spike — one-shot perf spike informing the cvvdp-gpu Burn port decision

This is **not a maintained crate.** It exists in-tree as a
paper trail: a reproducible perf comparison that decided the
`cvvdp-gpu` Burn-port question for us. The conclusion is
captured in
[`crates/cvvdp-gpu/docs/BURN_PORT_PLAN.md`](../cvvdp-gpu/docs/BURN_PORT_PLAN.md)'s
"Status: ABANDONED" banner.

## The question

Should we replace `cvvdp-gpu`'s hand-written 5-tap separable
Gaussian downscale (`kernels::pyramid::downscale_kernel`) with
two `cubek_convolution::conv2d` calls (vertical 5×1 then
horizontal 1×5), routing through Burn's auto-tuned conv
implementations?

The original `BURN_PORT_PLAN.md` (tick 190) claimed this would
"recover cuDNN-class performance" and cover ~85 % of pycvvdp's
pipeline natively. Before committing 3–5 days to the rewrite,
the spike measures the actual cost.

## The numbers (RTX 5070 sm_120, 4000×3000 → 2000×1500 f32, 100-iter median)

| Path | ns/op | vs hand-written |
|---|---|---|
| `cvvdp-gpu downscale_kernel` (hand-written direct stencil) | **338 692** | 1.00× |
| cubek separable `SimpleSyncCyclic + Mma` (best) | 1 462 306 | **4.32× slower** |
| cubek separable `SimpleSyncCyclic + Cmma` | 1 704 000 | 5.03× slower |
| cubek separable `SimpleSyncStrided + Cmma` | 1 690 000 | 4.99× slower |
| cubek separable `SimpleSyncTilewise + Cmma` | 1 686 000 | 4.98× slower |
| cubek separable `SimpleAsyncCyclic + Cmma` | (FAILED — 16-byte stride alignment requirement) | — |

Parity sanity check: middle-pixel `rel_diff = 0.000156` between
hand-written and cubek separable (well under the 1e-3 tolerance
the spike uses). cubek's zero-padding deviates from cvvdp's
reflection at edges but interior math matches — confirms the
4.32× ratio is real, not a wrong-output artifact.

## Verdict

**ABANDON the Burn port.** cubek-convolution's matmul-routed
conv is the wrong tool for 1-channel separable kernels:
- CMMA 16×16×16 tensor-core tiles waste 15/16 of the work when
  `in_channels = out_channels = 1`.
- im2col → GEMM doubles memory traffic vs. a direct stencil.
- Even the MMA path (4.32× slower) doesn't recover.

Recommended actionable lever instead: shared-memory tiling /
register tiling of the existing `downscale_kernel` direct
stencil — that's where the real perf headroom is for our
1-channel separable use case.

## Re-running

This crate has its own `[workspace]` root (cubek-convolution
0.2 requires `cubecl ^0.10.0` and excludes pre-releases; the
parent workspace pins `cubecl 0.10.0-pre.4`). The parent
workspace excludes it via `Cargo.toml`'s `exclude = [...]`
list.

```bash
cd crates/burn-conv-spike
cargo run --release
```

Output mirrors to stdout. Requires a CUDA-capable GPU; the
cubecl-cuda runtime loads CUDA via `dlopen` so `nvcc` doesn't
need to be on PATH.

## Don't extend this crate

If a future tick re-investigates Burn (different version,
different cubek release, different pipeline stage), spin up a
new spike crate at e.g. `crates/burn-conv-spike-v2/`. Keeping
this one frozen at the configuration that produced the
"abandoned" verdict preserves the paper trail. The verdict
itself lives in `BURN_PORT_PLAN.md`.
