# zenmetrics-nn

Learned full-reference image-quality metrics, run from ONNX graphs exported from
each model's upstream reference implementation. Plan and licensing:
[`docs/PYIQA_PORT_PLAN.md`](../../docs/PYIQA_PORT_PLAN.md).

| model | columns | engines | parity target | 1 JND | worst \|Δ\| in JND (15–21 pairs, 64–4096 px) |
|---|---|---|---|---|---|
| LPIPS v0.1 (`Lpips`, `LpipsNet`) | `lpips_alex`, `lpips_vgg`, `lpips_squeeze` | rten, ort-cuda | `lpips` 0.1.4, CPU fp32 | 0.065 / 0.194 / 0.080 | rten ≤ 1.4e-6, ort-cuda ≤ 1.2e-6 |
| DISTS (`Dists`) | `dists` | rten, ort-cuda | `DISTS_pytorch` 0.1, CPU fp32, no resize | 0.094 | rten 1.9e-6, ort-cuda 1.9e-6 |
| ST-LPIPS (`Lpips`, `LpipsNet::{AlexShiftTolerant,VggShiftTolerant}`) | `stlpips_alex`, `stlpips_vgg` | rten, ort-cuda | `stlpips_pytorch` 0.0.2 | 0.023 / 0.066 | rten ≤ 1.8e-6, ort-cuda ≤ 2.6e-6 |
| DeepDC (`DeepDc`) | `deepdc` | rten, ort-cuda | `DeepDC_PyTorch` 0.2, ImageNet input norm, no resize | 0.064 | rten 1.9e-6, ort-cuda 1.9e-6 |
| CKDN (`Ckdn`) | `ckdn` | rten, ort-cuda | `researchmm/CKDN` epoch-14 checkpoint, `cat(dist, ref, dist, ref)` → `dr`, eval transform: short side →329 then cc288 (dist bicubic-aa, ref nearest) | 0.0053¹ | rten 5.7e-5, ort-cuda 5.1e-5 |
| A-FINE (`Afine`) | `afine` | rten, ort-cuda | `ChrisDud0257/AFINE` afine.pth + OpenAI ViT-B-32, raw `afine_all` (lower is better) | 0.507¹ | rten 1.4e-5, ort-cuda 1.9e-5 |
| DreamSim (`Dreamsim`) | `dreamsim` | rten, ort-cuda | `dreamsim` 0.21.2 ensemble, LoRA folded, `1 − cosine` on centred unit concat, PIL-bicubic 224×224 | 0.0176 | rten 1.2e-5, ort-cuda 2.1e-5 |
| ARNIQA (`Arniqa`, no-reference) | `arniqa`, `arniqa_ref` | rten, ort-cuda | `miccunifi/ARNIQA` `single_image_inference.py`, kadid10k regressor, 5-crop full + PIL-bilinear half | 0.0538 | rten 6.65e-6 (= 3.58e-7 score), ort-cuda same |
| MANIQA (`Maniqa`, no-reference) | `maniqa`, `maniqa_ref` | rten, ort-cuda | `IIGROUP/MANIQA` koniq10k eval path, cv2-INTER_CUBIC 224 single forward, score [0,1] | 0.0214 | rten 1.70e-4 (= 3.6e-6 score), ort-cuda (see card) |
| LIQE (`Liqe`, no-reference, consent-gated) | `liqe`, `liqe_ref` | rten, ort-cuda | `zwx8981/LIQE` `LIQE.forward`, 495-prompt text bank, top-left-224-patch×15 quirk | 0.3938 | rten 1.25e-5, ort-cuda 4.96e-5 |
| MUSIQ (`Musiq`, no-reference) | `musiq`, `musiq_ref` | rten, ort-cuda | `google-research/musiq` `run_predict_image.py`, koniq_ckpt.npz, 3-scale TF-gaussian 32² patches → transformer, MOS-scale | 1.969 | rten 1.20e-4, ort-cuda 6.98e-5 |

Parity tolerance is 1e-3 JND per column (`tests/parity.rs`). One JND is the median score
of each column on KonJND-1k pairs at the observers' just-noticeable threshold
(`models/jnd_scale.tsv`; method:
[`benchmarks/nn_konjnd_scale_2026-10-15.md`](../../benchmarks/nn_konjnd_scale_2026-10-15.md)).
¹ CKDN and A-FINE have image-dependent identical-pair scores, so their JND is the pooled
median over score(pair) − score(reference vs itself): |delta| for CKDN, signed for
A-FINE.

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
