# zenmetrics-nn

Learned full-reference image-quality metrics, run from ONNX graphs exported from
each model's upstream reference implementation. Plan and licensing:
[`docs/PYIQA_PORT_PLAN.md`](../../docs/PYIQA_PORT_PLAN.md).

| model | columns | engines | parity target | worst \|Δ\| (18 pairs, 64–2048 px) |
|---|---|---|---|---|
| LPIPS v0.1 (`Lpips`, `LpipsNet`) | `lpips_alex`, `lpips_vgg`, `lpips_squeeze` | rten, ort-cuda | `lpips` 0.1.4, CPU fp32 | rten ≤ 1.1e-7, ort-cuda ≤ 9.0e-8 |
| DISTS (`Dists`) | `dists` | rten, ort-cuda | `DISTS_pytorch` 0.1, CPU fp32, no resize | rten 1.8e-7, ort-cuda 1.8e-7 |
| ST-LPIPS (`Lpips`, `LpipsNet::{AlexShiftTolerant,VggShiftTolerant}`) | `stlpips_alex`, `stlpips_vgg` | rten, ort-cuda | `stlpips_pytorch` 0.0.2 | rten ≤ 4.2e-8, ort-cuda ≤ 6.0e-8 |
| DeepDC (`DeepDc`) | `deepdc` | rten, ort-cuda | `DeepDC_PyTorch` 0.2, ImageNet input norm, no resize | rten 1.2e-7, ort-cuda 1.2e-7 |

Engines (`Engine`): `rten` (default feature, pure Rust, CPU); `ort` feature:
ONNX Runtime 1.31 with the CUDA provider, TF32 off, deterministic compute,
loaded at run time from `ORT_DYLIB_PATH`. Timings:
[`benchmarks/nn_lpips_phase0_2026-10-15.md`](../../benchmarks/nn_lpips_phase0_2026-10-15.md).

Weights are not bundled. Export a model directory, then load it:

```sh
just nn-export-lpips <python-with-lpips> <ladder-dir> <out-dir> vgg
just nn-export-dists <python-with-DISTS_pytorch> <ladder-dir> <out-dir>
```

```rust
use zenmetrics_abstractions::{EvalOpts, Metric};
use zenmetrics_nn::{Engine, Lpips, LpipsNet};
let mut lpips = Lpips::from_dir("out-dir", LpipsNet::Vgg, Engine::Rten)?;
let eval = lpips.evaluate(reference, distorted, EvalOpts::new())?;
```

On rten, `warm_reference` caches the reference's normalised features, so
scoring N candidates against one reference runs the trunk N + 1 times instead
of 2N. On ort-cuda it does not yet (the whole graph runs per call).

Tests: `cargo test -p zenmetrics-nn` runs the head and preprocessing unit tests.
`just nn-parity <model-dir> <ladder-dir> [rten|ort-cuda] [features] [nets]`
checks scores against the exported upstream goldens.
