# Porting pyiqa deep metrics to Rust — plan (2026-10-14)

Goal: make it cheap to add learned IQA models from
[IQA-PyTorch (pyiqa)](https://github.com/chaofengc/IQA-PyTorch) to zenmetrics, at equal or
better speed than PyTorch, with WebGPU as a stretch goal. Today
[`METRIC_PROVENANCE.md`](METRIC_PROVENANCE.md) lists DISTS, LPIPS ×4, PieAPP, WaDIQaM,
DeepDC, DreamSim, TOPIQ ×2, AHIQ and ST-LPIPS ×2 as "out of scope (no torch)".

Evidence:
- [`PYIQA_MODEL_INVENTORY_2026-10-14.md`](PYIQA_MODEL_INVENTORY_2026-10-14.md): every
  learned model, its backbone, licence, preprocessing traps, ONNX exportability, and
  pyiqa's published speed table.
- [`../benchmarks/nn_runtime_shootout_2026-10-14.md`](../benchmarks/nn_runtime_shootout_2026-10-14.md):
  rten / tract / burn / ORT / PyTorch on VGG16 and ResNet50, CPU and RTX 4080, on i265.

Owner decision (2026-10-14): models whose code or weights are noncommercial go in a
separately licensed crate, **`zenmetrics-pyiqa-ports`**. Reusing zentract / zensr /
zensally is optional; improving them with what we learn here is welcome.

## 1. What the evidence says

1. **Licensing splits the model list in two.** pyiqa's code is PolyForm Noncommercial
   1.0.0 and its weight mirror is CC BY-NC-SA 4.0. Permissive ports come from each
   model's upstream repo and weights (LPIPS BSD-2, DISTS MIT, DeepDC MIT, ST-LPIPS BSD-2,
   DreamSim MIT, MANIQA/ARNIQA Apache, LIQE/AHIQ/CKDN MIT). TOPIQ and WaDIQaM weights
   exist only as pyiqa trainings; PieAPP's official weights are NC; CLIP-IQA, QualiCLIP,
   TReS are NC upstream. Those go in `zenmetrics-pyiqa-ports`.
2. **Four backbones cover most of the list.** VGG16/19 + AlexNet → LPIPS ×4, DISTS,
   DeepDC, ST-LPIPS ×2 (8 of the AIC deep-metric columns). ResNet50 → TOPIQ ×10, HyperIQA, AHIQ,
   ARNIQA ×8, TReS, CKDN. ViT-B/8,16,32 and CLIP RN50 → MANIQA, AHIQ, DreamSim, LIQE,
   A-FINE, CLIP-IQA, QualiCLIP.
3. **No single pure-Rust runtime matches PyTorch on both CPU and GPU** (measured on
   i265, two sizes per model):
   - CPU: **rten 0.27** is at parity single-threaded (0.97–1.00×) and on multithreaded
     VGG (0.91–1.04×); ResNet50 multithreaded trails 1.28–1.45×. tract 1.14–2.2×,
     burn-flex 1.4–3.6×. ORT is faster than PyTorch but is C++ (oracle only).
   - GPU: **burn-cuda** trails PyTorch fp32 by 1.2–2.1× at batch 1; PyTorch fp16 is a
     further ~1.7× faster. **burn-vulkan is 100–120× slower on VGG** (5–11× on ResNet50);
     not root-caused, and it is the closest native proxy we have for WebGPU.
4. **The PyTorch-kernel bar and the pyiqa-as-used bar are different bars.** pyiqa's own
   table times decode + host→device copy + Python per call: DISTS 0.125 s and LPIPS-vgg
   0.143 s per ~1 MP image on a V100, where the network itself is single-digit ms on a
   4080 (not measured at 1 MP on V100; the gap is an inference, to be measured in phase
   0). Beating pyiqa end-to-end is mostly I/O, batching and reference reuse; matching raw
   PyTorch kernels is the harder target.
5. **burn cannot join the zenmetrics cargo graph** (it needs upstream `cubecl` 0.11
   under its real name; zenmetrics pins the renamed `zenforks-cubecl` 0.10.2 by git rev,
   see `CLAUDE.md` "burn"). Anything built on zenforks-cubecl also cannot be published
   to crates.io while the fork is a git pin, so the separately licensed crate could not
   depend on it.
6. **The `burn-conv-spike` "4.32× slower" verdict does not apply.** It timed a
   1-channel separable blur; these networks have 64–2048 channels, which is what
   implicit-GEMM conv on tensor cores is for.

## 2. Architecture

### Repos and crates

| repo | crate(s) | licence | depends on |
|---|---|---|---|
| `imazen/zenmetrics-nn` (new, name open) | `zenmetrics-nn` (engines, preprocessing, weight store, backbones, permissive metric heads as feature-gated modules), `zenmetrics-nn-cli`, `zenmetrics-nn-plugin` (cdylib) | AGPL-3 + commercial, like the rest | rten, burn, burn-onnx, burn-store (crates.io) |
| `imazen/zenmetrics-pyiqa-ports` (new) | `zenmetrics-pyiqa-ports` | noncommercial (proposed: PolyForm Noncommercial 1.0.0, matching pyiqa) | `zenmetrics-nn` from crates.io |
| `imazen/zenmetrics` (this repo) | `zenmetrics-cli` gains NN metric kinds via plugin or subprocess | unchanged | nothing new in the graph |

Why a separate repo rather than an excluded crate here: it keeps upstream cubecl out of
this repo's lockfile and CI sibling pins, and both downstream users (this repo and the
NC crate) consume it the same way, as a published crate or binary. An excluded
workspace inside this repo (the `crates/margarine` pattern) also works if one repo is
preferred.

Metric heads for other people's metrics stay modules inside `zenmetrics-nn`, not
bare-name crates: the bare-name rule in `CLAUDE.md` is for metrics we publish first.

### Engines

| engine | role | why |
|---|---|---|
| rten (`.onnx` loaded at runtime) | default CPU, wasm-SIMD | only pure-Rust engine at PyTorch CPU parity |
| burn via `burn-onnx` codegen (same `.onnx`) | CUDA, Vulkan/Metal, WebGPU, wasm | one generated source for every GPU backend |
| ORT, PyTorch | oracles in `tools/`, never shipped | C++ |

Both shipped engines consume the **same exported `.onnx`**, so one golden set checks
both. Ops that ONNX handles badly (AHIQ's DeformConv2d, MUSIQ's hash position embedding
and multiscale patching, crop schedules, LPIPS/DISTS tap reductions) live in Rust
around the graph, not in it.

### Adding a model

A model is a directory `models/<name>/`:

- `card.toml`: FR/NR, upstream repo + commit, paper, code and weight licences, weight
  URLs + sha256, input normalisation, resize/crop schedule, score range and direction,
  oracle (upstream package or pyiqa), parity tolerance.
- `export.py`: loads the **upstream** reference implementation, exports backbone + head
  to ONNX with dynamic H/W (patching and cropping outside the graph), and dumps CPU-fp32
  goldens: per-tap tensor hashes and stats, plus final scores on the golden corpus.
- Rust glue: preprocessing, crop schedule, and any head ops outside the graph.

Once its backbone exists, a model that reuses it should be mostly card + export
script + head glue. That is the "easy" target; it is an estimate until phase 1 has
shipped two or three models.

### Weights

- Never inside the crate package (crates.io 10 MB limit; VGG16 features alone are
  ~58 MB). Fetched on first use or by `zenmetrics-nn fetch <model>` into a cache dir,
  sha256-verified against the card.
- Permissive weights: upstream URLs, with an imazen R2 mirror for reliability where the
  licence allows redistribution (BSD/MIT/Apache).
- NC weights: fetched from their original location and converted on the user's machine,
  so we never redistribute an adapted CC BY-NC-SA file. Loading NC models needs an
  explicit opt-in. Whether burn-store can read the `.pth` pickles directly is unverified.

### Integration with this repo

- **Fleet:** `zenmetrics-nn-cli score-pairs` writes the same parquet sidecar schema as
  `zenmetrics score-pairs`. zenfleet jobexec runs it as a separate binary, which is the
  data-not-tensors architecture already chosen on 2026-06-09. The worker image bakes the
  binary; weights come from R2 at boot like other per-sweep data.
- **CLI and Python:** a plugin loader in `zenmetrics-cli` (zentract's pattern: a small
  versioned C ABI taking strided f32 planes plus a model id and returning a score). The
  cost is one copy per image and no GPU buffer sharing with zenmetrics' own kernels.
  `zenmetrics-py` links zenmetrics-api, so it needs the plugin too. Subprocess is the
  first step; the plugin follows.
- `CliMetricKind` gains `lpips`, `lpips-vgg`, `dists`, …; `METRIC_PROVENANCE.md` and
  `AIC2026_METRICS_AND_FITTING.md` rows move from "out of scope" as each lands.

## 3. Performance plan

Two bars, reported separately, both swept over 64²–4096² per the sweep rules:

- **Bar A, pyiqa as used:** pairs per second scoring a directory, including decode, on
  the same host. Levers we have that pyiqa doesn't: Rust decode (zencodecs), pinned
  uploads, batching pairs on GPU, and a **warm reference** for FR metrics (run the
  backbone on the reference once per N candidates, the same pattern as
  `ButteraugliReference`; for N ≥ 2 it removes up to half the backbone work).
- **Bar B, raw PyTorch fp32 kernels:** CPU is already at parity on VGG with rten. To
  close the rest:
  1. Fuse tap reductions (LPIPS unit-normalise → diff → 1×1 → spatial mean; DISTS
     moments) so taps are never written out.
  2. Multithreaded ResNet50 on rten (1.28–1.45× gap): profile first; upstream fix or a
     zensr-style hand conv for the hot shapes.
  3. GPU f16, gated on score parity, not assumed.
  4. Only if fleet GPU throughput demands it: port cubek's implicit-GEMM conv/matmul
     into zenforks-cubecl. That needs the fork resynced to cubecl 0.11 and published
     first. It is also the GPU conv zensr's spike said it lacked.

Report peak RSS and VRAM with every number. Measure on dev (Zen 5, AVX-512) as well as
i265 before drawing CPU conclusions.

## 4. WebGPU

- **Route 1, burn's wgpu/WebGPU backend** (the same generated model). Blocked on
  root-causing burn-vulkan's 100× VGG result. Then measure in Chrome for lpips-alex,
  lpips-vgg and dists at 256², 512² and 1 MP. WGSL has no cooperative-matrix path in
  burn or in zenforks-cubecl, so expect plain tiled-GEMM speed.
- **Route 2, wasm CPU** with rten's wasm-SIMD kernels. LPIPS-alex has 2.47 M parameters
  and pyiqa times it at 0.17–0.21 s per ~1 MP on desktop CPUs, so it is plausible in a
  browser without a GPU. This is the safe first browser target.

## 5. Correctness

- **Oracle per model is the upstream reference package**: `lpips`, `DISTS_pytorch`,
  `dreamsim`, the upstream repos. pyiqa is the oracle only where pyiqa's variant is the
  target (the NC crate, AIC columns produced with pyiqa). Its variants differ from
  upstream in places: MUSIQ uses bicubic where Google's code uses TF gaussian resize, up
  to 1.09 MOS apart.
- **Three gates:** export (ONNX under ORT vs PyTorch), engine (rten and burn vs ORT, per
  tap), end-to-end (score vs oracle on the golden corpus at CPU fp32 with TF32 off).
  Tolerances are proposed per metric in `PARITY_TOLERANCES.md` style and need sign-off;
  never widened silently.
- **Golden corpus:** CID22 validation subset + KADID/TID pairs + a size ladder
  (32²–4096²). Pooling depth sets a minimum input size per backbone (VGG pools four or
  five times); record it on each card.
- **Preprocessing is where parity breaks.** The inventory's §4 lists every
  normalisation, resize and crop rule. Each card spells it out, and each model gets a
  preprocessing parity test before any network parity test.
- **Licence caveat for oracles:** running pyiqa inside a commercial company's
  development may not count as a noncommercial purpose under PolyForm NC, even if
  nothing ships. Permissive models avoid pyiqa entirely; get counsel's view for the NC
  crate's oracle runs.

## 6. Model order

| wave | crate | models | backbone work |
|---|---|---|---|
| 1 | zenmetrics-nn | LPIPS alex/vgg/squeeze, LPIPS+ (reimplemented from the TOPIQ paper), DISTS, DeepDC, ST-LPIPS alex/vgg | VGG16, VGG19, AlexNet, SqueezeNet, BlurPool (clean-room) |
| 2 | zenmetrics-nn | ARNIQA, CKDN, PaQ2PiQ, LIQE, A-FINE, MANIQA, DreamSim | ResNet50/18, CLIP ViT-B/32, ViT-B/8 + Swin, ViT-B/16 |
| 2 | zenmetrics-pyiqa-ports | TOPIQ FR ×2 + NR, WaDIQaM, PieAPP, HyperIQA, DBCNN, NIMA, CLIP-IQA(+), QualiCLIP, TReS | reuses wave-2 backbones |
| 3 | zenmetrics-nn | MUSIQ (choose oracle first), AHIQ (DeformConv in Rust; pin the timm weights it silently depends on) | MUSIQ transformer, DeformConv |
| hold | — | Q-Align / Q-ReAlign / Compare2Score (LLMs, 2–19 GB), FID, DMM, TOPIQ face variants (need a face detector) | — |

Retraining TOPIQ and WaDIQaM on permissively licensed data would move them to the
permissive crate. That is a separate project and is not planned here.

## 7. Phase 0 (first concrete steps)

1. **Bar A baseline:** on i265 (CPU + RTX 4080) and dev (CPU), time upstream `lpips`,
   `DISTS_pytorch` and pyiqa end-to-end on a 64²–4096² ladder with pyiqa's own
   protocol, and raw PyTorch fp32/fp16 compute on the same ladder. Commit under
   `benchmarks/`.
2. **Root-cause burn-vulkan's VGG slowness.** It decides whether WebGPU route 1 is
   viable.
3. **Vertical slice:** LPIPS-vgg end-to-end in Rust (card, export script, rten CPU,
   burn-cuda, warm reference, parquet sidecar), gated on parity with upstream `lpips`.
   This sets the template every later model copies.

## 8. Spillover to zentract, zensally, zensr

- **zentract** (tract-onnx 0.22.3 cdylib; CPU, f32, tract multithreading not enabled):
  an rten engine behind the same ABI would be 1.14–1.26× faster single-threaded and
  1.3–2.2× multithreaded on these CNNs, and drops tract's assembly build, which needed a
  git pin for Windows-ARM64. Enabling tract's `multithread-mm` is a free interim step.
- **zensally** (UltraFace, MicroSalNet via tract): tiny nets, so native GPU gains little
  at batch 1. rten would remove the Windows-ARM64 pin and is probably faster
  (extrapolated from these CNNs, not measured on its models). The real gain is
  in-browser face and saliency detection through rten wasm or burn WebGPU.
- **zensr** (hand archmage SIMD conv, measured faster than tract): keep its CPU engine.
  Its GPU path needs the tiled implicit-GEMM conv that Bar B step 4 would add to
  zenforks-cubecl, so that kernel would be shared.

## 9. Decisions needed

1. Licence text for `zenmetrics-pyiqa-ports` (proposed: PolyForm Noncommercial 1.0.0).
2. A new `imazen/zenmetrics-nn` repo, or an excluded workspace inside this repo; and
   the core crate's name (avoid `zeniqa`-like names).
3. Counsel: running pyiqa as an internal oracle; ImageNet-trained backbone weights.
