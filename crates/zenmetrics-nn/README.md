# zenmetrics-nn

Learned full-reference image-quality metrics, run from ONNX graphs exported from
each model's upstream reference implementation. Plan and licensing:
[`docs/PYIQA_PORT_PLAN.md`](../../docs/PYIQA_PORT_PLAN.md).

| model | engines | parity target | worst \|Δ\| (18 pairs, 64–2048 px) |
|---|---|---|---|
| LPIPS v0.1, VGG16 (`lpips_vgg`) | rten (CPU), ort-cuda | `lpips` 0.1.4, CPU fp32 | rten 4.5e-8, ort-cuda 6.0e-8 |

Engines: `rten` (default feature, pure Rust, CPU); `ort` feature: ONNX Runtime
1.31 with the CUDA provider, TF32 off, deterministic compute, loaded at run time
from `ORT_DYLIB_PATH`. Timings: [`benchmarks/nn_lpips_phase0_2026-10-15.md`](../../benchmarks/nn_lpips_phase0_2026-10-15.md).

Weights are not bundled. Export a model directory, then load it:

```sh
just nn-export-lpips-vgg <python-with-lpips> <ladder-dir> <out-dir>
```

```rust
use zenmetrics_abstractions::{EvalOpts, Metric};
let mut lpips = zenmetrics_nn::LpipsVgg::from_dir("out-dir")?;
let eval = lpips.evaluate(reference, distorted, EvalOpts::new())?;
```

On rten, `warm_reference` caches the reference's normalised features, so
scoring N candidates against one reference runs the trunk N + 1 times instead
of 2N. On ort-cuda it does not yet (the whole graph runs per call).

Tests: `cargo test -p zenmetrics-nn` runs the head and preprocessing unit tests.
`just nn-parity <model-dir> <ladder-dir> [rten|ort-cuda] [features]` checks
scores against the exported upstream goldens.
