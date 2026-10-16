# zenmetrics-nn

Learned full-reference image-quality metrics, run from ONNX graphs exported from
each model's upstream reference implementation. Plan and licensing:
[`docs/PYIQA_PORT_PLAN.md`](../../docs/PYIQA_PORT_PLAN.md).

| model | columns | engines | parity target | 1 JND | worst \|Δ\| in JND (model-specific golden sets) |
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
| HyperIQA (`Hyperiqa`, no-reference) | `hyperiqa`, `hyperiqa_ref` | rten, ort-cuda | `SSL92/hyperIQA` `demo.py` model on a documented deterministic 5x5 crop grid (upstream crops at random), koniq_pretrained.pkl, ResNet-50 hypernetwork, trained on 0–100 MOS with an unbounded linear output | 4.462 | rten 3.42e-6, ort-cuda 3.42e-6 |
| PaQ-2-PiQ (`Paq2piq`, no-reference) | `paq2piq`, `paq2piq_ref` | rten, ort-cuda | `baidut/paq2piq` `InferenceModel.predict` (MIT code; CC-BY-NC weights, consent-gated), RoIPoolModel-fit.10.bs.120.pth, ResNet-18 + 1+20x20 RoIPool grid, ~0-100 MOS | 1.229 | rten 2.51e-5, ort-cuda 1.28e-5 |
| WD (`WassersteinDistortion`, full-reference) | `wd_s1` … `wd_s5` | rten, ort-cuda | Cool-Chic raw RGB VGG16 pre-pool, five sigma maps; [validation](../../benchmarks/wd_port_2026-10-16/README.md) | 0.3822 / 0.2477 / 0.1602 / 0.1038 / 0.06745 | rten ≤1.93e-6, ort-cuda ≤3.45e-6 (25 pairs, including non-square and 2048²) |
| WaDIQaM (`WadiqamFr`, `WadiqamNr` no-reference) | `wadiqam_fr_live`, `wadiqam_fr_tid`, `wadiqam_nr_live(_ref)`, `wadiqam_nr_tid(_ref)` | rten, ort-cuda | `dmaniry/deepIQA` `evaluate.py` CPU fp32, `top="weighted"`, non-overlapping 32² stride-32 patches at native res, raw u8→f32; MIT code + weights | 29.93¹ / 5.21¹ / 23.55 / 2.76 | rten ≤ 7.8e-6, ort-cuda ≤ 5.5e-6 (29 pairs incl. non-square + 4096²) |

Parity tolerance is 1e-3 JND per column (`tests/parity.rs`). One JND is the median score
of each column on KonJND-1k pairs at the observers' just-noticeable threshold
(`models/jnd_scale.tsv`; method:
[`benchmarks/nn_konjnd_scale_2026-10-15.md`](../../benchmarks/nn_konjnd_scale_2026-10-15.md)).
¹ CKDN, A-FINE and WaDIQaM-FR have image-dependent identical-pair scores, so their JND
is the pooled median over score(pair) − score(reference vs itself): |delta| for CKDN
and WaDIQaM-FR, signed for A-FINE.

Engines (`Engine`): `rten` (default feature, pure Rust, CPU); `ort` feature:
ONNX Runtime 1.31 with the CUDA provider, TF32 off, deterministic compute,
loaded at run time from `ORT_DYLIB_PATH`. Timings:
[`benchmarks/nn_lpips_phase0_2026-10-15.md`](../../benchmarks/nn_lpips_phase0_2026-10-15.md).

PaQ-2-PiQ accepts native-resolution inputs up to 4096 pixels per axis. Its
[review receipt](../../benchmarks/paq2piq_review_2026-10-16.md) records export
commands, small-image goldens and default-scheduler validation.

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

### MUSIQ and HyperIQA export qualification

Both export recipes require a goldens TSV and its complete image directory.
They fail for missing image pairs, nonfinite scores or differences above
`1e-3 * JND` in score units, with scales read from `models/jnd_scale.tsv`.
`just nn-export-checks` exercises rejection controls without loading models.
HyperIQA exports have a fixed batch of 25 and score the documented deterministic
crop grid; its linear output is unbounded. MUSIQ errors distinguish the 8192
patch-row limit (`Error::TooManyPatches`) from a degenerate aspect ratio
(`Error::DegenerateAspectRatio`). These are additive public error variants.

[Review qualification receipt](../../benchmarks/nr_learned_review_2026-10-15.meta)
records the complete exports, rejection controls and both engine parity runs.

### PieAPP

`PieApp` reconstructs the CVPR 2018 network from the paper and checkpoint
metadata, with official `compute_score` used only as an opaque oracle.
The weights require explicit noncommercial consent (`PieApp::WEIGHTS`);
CLI callers set `ZENMETRICS_NN_ACCEPT_WEIGHTS=pieapp`. No weights are bundled.
It scores valid 64×64 patches at stride six, in batches of at most 32, on
`rten` or `ort-cuda`. Both dimensions must be at least 64. Warm references
reuse planar conversion only. See the [model card](models/pieapp/card.toml)
and [sampling divergence](../../DIVERGENCES.md#pieapp-sampling-and-official-entry-point-2026-10-16).
Both engines pass 24 pairs at 1e-3 JND using the complete 1008-pair KonJND
calibration; see the [qualification record](../../benchmarks/pieapp_2026-10-16.md).
