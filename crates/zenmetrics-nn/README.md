# zenmetrics-nn

Learned full-reference image-quality metrics, run from ONNX graphs exported from
each model's upstream reference implementation. Plan and licensing:
[`docs/PYIQA_PORT_PLAN.md`](../../docs/PYIQA_PORT_PLAN.md).

| model | status | engine | parity target |
|---|---|---|---|
| LPIPS v0.1, VGG16 (`lpips_vgg`) | CPU slice | rten | `lpips` 0.1.4, CPU fp32 |

Weights are not bundled. Export a model directory, then load it:

```sh
just nn-export-lpips-vgg <python-with-lpips> <ladder-dir> <out-dir>
```

```rust
use zenmetrics_abstractions::{EvalOpts, Metric};
let mut lpips = zenmetrics_nn::LpipsVgg::from_dir("out-dir")?;
let eval = lpips.evaluate(reference, distorted, EvalOpts::new())?;
```

`warm_reference` caches the reference's normalised features, so scoring N
candidates against one reference runs the trunk N + 1 times instead of 2N.

Tests: `cargo test -p zenmetrics-nn` runs the head and preprocessing unit tests.
`just nn-parity <model-dir> <ladder-dir>` checks scores against the exported
upstream goldens.
