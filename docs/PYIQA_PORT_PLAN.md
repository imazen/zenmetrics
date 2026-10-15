# Porting pyiqa deep metrics to Rust — plan (2026-10-15, rev 3)

Goal: make it cheap to add learned IQA models from
[IQA-PyTorch (pyiqa)](https://github.com/chaofengc/IQA-PyTorch) to zenmetrics, at equal or
better speed than PyTorch, with WebGPU as a stretch goal. Today
[`METRIC_PROVENANCE.md`](METRIC_PROVENANCE.md) lists DISTS, LPIPS ×4, PieAPP, WaDIQaM,
DeepDC, DreamSim, TOPIQ ×2, AHIQ and ST-LPIPS ×2 as "out of scope (no torch)".

Evidence:
- [`PYIQA_MODEL_INVENTORY_2026-10-14.md`](PYIQA_MODEL_INVENTORY_2026-10-14.md): every
  learned model, its backbone, licence, preprocessing traps, ONNX exportability, and
  pyiqa's published speed table.
- [`PYIQA_UPSTREAM_LICENSES_2026-10-15.md`](PYIQA_UPSTREAM_LICENSES_2026-10-15.md): the
  original repository behind every pyiqa model and metric, its code and weight terms read
  as text, and the verdict for a zenmetrics port. Supersedes the inventory's licence
  column.
- [`../benchmarks/nn_runtime_shootout_2026-10-14.md`](../benchmarks/nn_runtime_shootout_2026-10-14.md):
  rten / tract / burn / ORT / PyTorch on VGG16 and ResNet50, CPU and RTX 4080, on i265,
  plus the burn + zenforks-cubecl coexistence probe.

Owner decisions (2026-10-14, licensing refined 2026-10-15):
- **Minimise noncommercial licensing.** Don't relicense our code wholesale. Trace each
  model to its original repository and port from there when its terms are better than
  pyiqa's. A separately licensed crate exists only for a specific model that requires
  it (§2).
- **All the new crates live in this repo and workspace.**
- An **optional C++ backend (ONNX Runtime) is fine** as long as pure-Rust engines exist
  for the same models.
- Reusing zentract / zensr / zensally is optional; improving them with what we learn is
  welcome.
- **i265's GPU flips a bit every few minutes**: a mismatching GPU run there is rerun
  before it counts (§5).

## 1. What the evidence says

1. **Licensing is decided by weights, not code.** pyiqa's code is PolyForm Noncommercial
   1.0.0 and its weight mirror is CC BY-NC-SA 4.0, but 24 of its 40 neural entries have
   permissive upstream code, and the upstream weights are permissive for LPIPS, DISTS,
   DeepDC, DMM, ST-LPIPS (BlurPool rewritten), WaDIQaM (official deepIQA, MIT), ARNIQA,
   MANIQA, MUSIQ, A-FINE, FGResQ and DreamSim. AHIQ, CKDN, HyperIQA, LIQE, NIMA and UNIQUE
   publish weights with no stated terms. TOPIQ, DBCNN and PaQ-2-PiQ weights are NC;
   PieAPP, CLIP-IQA, QualiCLIP and URanker are NC upstream; TReS's agreement forbids
   distribution. Full table: [`PYIQA_UPSTREAM_LICENSES_2026-10-15.md`](PYIQA_UPSTREAM_LICENSES_2026-10-15.md).
2. **Four backbones cover most of the list.** VGG16/19 + AlexNet → LPIPS ×4, DISTS,
   DeepDC, ST-LPIPS ×2 (8 of the AIC deep-metric columns). ResNet50 → TOPIQ ×10,
   HyperIQA, AHIQ, ARNIQA ×8, TReS, CKDN. ViT-B/8,16,32 and CLIP RN50 → MANIQA, AHIQ,
   DreamSim, LIQE, A-FINE, CLIP-IQA, QualiCLIP.
3. **No single pure-Rust runtime matches PyTorch on both CPU and GPU** (measured on
   i265, two sizes per model):
   - CPU: **rten 0.27** is at parity single-threaded (0.97–1.00×) and on multithreaded
     VGG (0.91–1.04×); ResNet50 multithreaded trails 1.28–1.45×. tract 1.14–2.2×,
     burn-flex 1.4–3.6×. **ONNX Runtime beats PyTorch** (0.78–0.85× single-threaded).
   - GPU: **burn-cuda** trails PyTorch fp32 by 1.2–2.1× at batch 1; PyTorch fp16 is a
     further ~1.7× faster. ORT's CUDA EP ran ResNet50 in 1.24 / 1.78 ms vs PyTorch fp32
     2.26 / 2.61 (ORT with its default TF32, so not like for like). **burn-vulkan is
     100–120× slower on VGG** (5–11× on ResNet50); not root-caused, and it is the
     closest native proxy we have for WebGPU.
4. **The PyTorch-kernel bar and the pyiqa-as-used bar are different bars.** pyiqa's own
   table times decode + host→device copy + Python per call: DISTS 0.125 s and LPIPS-vgg
   0.143 s per ~1 MP image on a V100, where the network itself is single-digit ms on a
   4080 (the network-only V100 time at 1 MP was not measured; phase 0 measures it).
   Beating pyiqa end-to-end is mostly I/O, batching and reference reuse; matching raw
   PyTorch kernels is the harder target.
5. **burn and zenforks-cubecl coexist in one binary** (measured, `coexist-probe/`). burn
   pulls upstream cubecl 0.11 and wgpu 30; our GPU crates pull zenforks-cubecl 0.10.2 and
   wgpu 29. Cargo keeps both generations because they differ in version and source, and
   the `[lib] name = "cubecl_*"` shims don't collide because rlib file names carry a
   hash. The probe interleaved burn matmuls with `ssim2-gpu` on the same RTX 4080 on
   CUDA and on wgpu; ssim2 scores were bit-identical across all rounds and both runs.
   The rename only stops burn from *using* the fork: no tensor, handle or client crosses
   between them, only host data. Costs and couplings:
   - two GPU runtimes, memory pools and kernel caches per process (VRAM budgeting must
     count both);
   - both sides share **one `cudarc` 0.19.10** with the same feature set with or
     without burn; recheck `cargo tree -i cudarc -f '{p} [{f}]'` when either side bumps;
   - burn's tree is all crates.io, so burn-based crates stay publishable; anything that
     depends on the git-pinned fork does not.
6. **The `burn-conv-spike` "4.32× slower" verdict does not apply.** It timed a
   1-channel separable blur; these networks have 64–2048 channels, which is what
   implicit-GEMM conv on tensor cores is for.

## 2. Architecture

### Crates (all in this workspace)

| crate | licence | contents | GPU deps |
|---|---|---|---|
| `crates/zenmetrics-nn` (name open) | AGPL-3 + commercial | engine trait + engines, preprocessing, weight store, model cards, backbones, permissive metric heads (feature-gated modules) | burn (crates.io) only; **never zenforks-cubecl**, so it stays publishable |
| `zenmetrics-cli`, `zenmetrics-api`, `zenmetrics-py` | unchanged | `nn` feature links `zenmetrics-nn` directly | both cubecls in one binary (§1.5) |

Linking directly replaces the plugin/subprocess boundary the first draft proposed:
no copy across an ABI, one CLI, and `zenmetrics-py` gets the metrics for free.

**Licensing (rev 3).** Every model's code is ours, under the crate's normal licence:
ported from a permissive upstream (keeping its notice in a `LICENSE-<origin>` file, the
gmsd convention), or written from the paper when the upstream is NC, has no licence, or
is pyiqa's own (TOPIQ). We never port from pyiqa's files and never read TReS's code.
Noncommercial status belongs to **weights**: each card records `weights_licence`
(`permissive` / `unknown` / `nc`), we redistribute only `permissive` weights, and
loading `unknown` or `nc` weights fetches them from their origin and requires an
explicit opt-in that names the terms. So no crate, binary or default build is
noncommercial; the user who opts in to NC weights takes on their terms. A separately
licensed crate is made for one model only if its code cannot be written independently;
none has needed it so far. This replaces rev 2's single `zenmetrics-pyiqa-ports` crate
(owner, 2026-10-15: "only do specific crates when required").

Workspace hygiene that comes with it: `Cargo.lock` gains burn's tree (the probe's lock
is ~8,900 lines), so regenerate with `scripts/ci/lock.sh --regen`, never a plain cargo
run in a dev tree. CI builds the `nn` crates in their own jobs so the existing metric
jobs don't pay burn's compile time.

Metric heads for other people's metrics stay modules inside `zenmetrics-nn`, not
bare-name crates: the bare-name rule in `CLAUDE.md` is for metrics we publish first.

### Engines

| engine | build | role |
|---|---|---|
| rten (`.onnx` loaded at runtime) | pure Rust, default | CPU and wasm-SIMD; PyTorch parity on CPU |
| burn via `burn-onnx` codegen (same `.onnx`) | pure Rust, default for GPU | CUDA, Vulkan/Metal, WebGPU, wasm |
| ONNX Runtime via `ort` | **optional C++ feature `ort`**, off by default | fastest CPU (0.78–0.85× PyTorch), CUDA EP for fleet throughput; also the export oracle |
| PyTorch | `tools/` only | oracle |

All engines consume the **same exported `.onnx`**, so one golden set checks every
engine, and any model available on `ort` is also available on rten and burn (the
owner's condition for a C++ backend). Each engine reports its name in the score's
impl tag so a sweep can stay on one backend. Ops that ONNX handles badly (AHIQ's
DeformConv2d, MUSIQ's hash position embedding and multiscale patching, crop schedules,
LPIPS/DISTS tap reductions) live in Rust around the graph, not in it.

ORT's own WebGPU execution provider belongs to onnxruntime-web (JavaScript), not to the
Rust crate on wasm, so it is not a browser route for us.

### Adding a model

A model is a directory `models/<name>/`:

- `card.toml`: FR/NR, upstream repo + commit, paper, code source (`upstream-port` or
  `paper`) and its licence, `weights_licence` class with the terms URL, weight
  URLs + sha256, input normalisation, resize/crop schedule, score range and direction,
  oracle (upstream package or pyiqa), parity tolerance.
- `export.py`: loads the **upstream** reference implementation, exports backbone + head
  to ONNX with dynamic H/W (patching and cropping outside the graph), and dumps CPU-fp32
  goldens: per-tap tensor hashes and stats, plus final scores on the golden corpus.
- Rust glue: preprocessing, crop schedule, and any head ops outside the graph.

Once its backbone exists, a model that reuses it should be mostly card + export script +
head glue. That is the "easy" target; it is an estimate until phase 1 has shipped two
or three models.

### Weights

- Never inside the crate package (crates.io 10 MB limit; VGG16 features alone are
  ~58 MB). Fetched on first use or by an explicit fetch command into a cache dir,
  sha256-verified against the card.
- Permissive weights: upstream URLs, with an imazen R2 mirror for reliability where the
  licence allows redistribution (BSD/MIT/Apache).
- `unknown` and `nc` weights: fetched from their original location (not pyiqa's mirror
  when an original exists) and converted on the user's machine, so we never
  redistribute them. Loading them needs an explicit opt-in. Whether burn-store can read the `.pth` pickles directly is unverified.

### Integration

- `CliMetricKind` gains `lpips`, `lpips-vgg`, `dists`, …; `score-pairs` writes them into
  the normal parquet sidecars, so zenfleet jobexec needs nothing new beyond baking
  weights into or fetching them alongside the worker image.
- `METRIC_PROVENANCE.md` and `AIC2026_METRICS_AND_FITTING.md` rows move from "out of
  scope" as each lands.

## 3. Performance plan

Two bars, reported separately, both swept over 64²–4096² per the sweep rules:

- **Bar A, pyiqa as used:** pairs per second scoring a directory, including decode, on
  the same host. Levers we have that pyiqa doesn't: Rust decode (zencodecs), pinned
  uploads, batching pairs on GPU, and a **warm reference** for FR metrics (run the
  backbone on the reference once per N candidates, the same pattern as
  `ButteraugliReference`; for N ≥ 2 it removes up to half the backbone work).
- **Bar B, raw PyTorch fp32 kernels:** CPU is already at parity on VGG with rten and
  ahead with `ort`. To close the rest on the pure-Rust engines:
  1. Fuse tap reductions (LPIPS unit-normalise → diff → 1×1 → spatial mean; DISTS
     moments) so taps are never written out.
  2. Multithreaded ResNet50 on rten (1.28–1.45× gap): profile first; upstream fix or a
     zensr-style hand conv for the hot shapes.
  3. GPU f16, gated on score parity, not assumed.
  4. Only if fleet GPU throughput demands it and `ort` CUDA is not acceptable there:
     port cubek's implicit-GEMM conv/matmul into zenforks-cubecl (needs the fork resynced
     to cubecl 0.11). It is also the GPU conv zensr's spike said it lacked.

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
  target (pyiqa-retrained weights, AIC columns produced with pyiqa). Its variants differ from
  upstream in places: MUSIQ uses bicubic where Google's code uses TF gaussian resize, up
  to 1.09 MOS apart.
- **Three gates:** export (ONNX under ORT vs PyTorch), engine (rten, burn and `ort` vs
  the reference, per tap), end-to-end (score vs oracle on the golden corpus at CPU fp32
  with TF32 off). Tolerances are proposed per metric in `PARITY_TOLERANCES.md` style and
  need sign-off; never widened silently.
- **Goldens are CPU-generated.** i265's RTX 4080 flips a bit every few minutes (owner
  report; `CLAUDE.md` Known Bugs), so the parity harness reruns every mismatching GPU
  case automatically, records both runs, and calls a mismatch real only if it
  reproduces. No single GPU run on that box becomes a golden.
- **Golden corpus:** CID22 validation subset + KADID/TID pairs + a size ladder
  (32²–4096²). Pooling depth sets a minimum input size per backbone (VGG pools four or
  five times); record it on each card.
- **Preprocessing is where parity breaks.** The inventory's §4 lists every
  normalisation, resize and crop rule. Each card spells it out, and each model gets a
  preprocessing parity test before any network parity test.
- **Licence caveat for oracles:** running pyiqa inside a commercial company's
  development may not count as a noncommercial purpose under PolyForm NC, even if
  nothing ships. Permissive models avoid pyiqa entirely; get counsel's view before any
  oracle run that needs pyiqa itself.

## 6. Model order

| wave | crate | models | backbone work |
|---|---|---|---|
All waves live in `zenmetrics-nn`; the weights column is the card's `weights_licence`.

| wave | models | weights | backbone work |
|---|---|---|---|
| 1 | LPIPS alex/vgg/squeeze, LPIPS+ (reimplemented from the TOPIQ paper), DISTS, DeepDC, ST-LPIPS alex/vgg | permissive | VGG16, VGG19, AlexNet, SqueezeNet, BlurPool (from the paper) |
| 2 | ARNIQA, MANIQA, A-FINE, FGResQ, DreamSim, WaDIQaM (deepIQA weights) | permissive | ResNet50/18, CLIP ViT-B/32, ViT-B/8 + Swin, ViT-B/16 |
| 2 | CKDN, LIQE, HyperIQA, NIMA, UNIQUE | unknown (opt-in) | reuses wave-2 backbones |
| 2 | TOPIQ FR ×2 + NR, DBCNN, PaQ-2-PiQ, PieAPP, CLIP-IQA(+), QualiCLIP, WaDIQaM (pyiqa retrain) | nc (opt-in); code from papers | reuses wave-2 backbones |
| 3 | MUSIQ (Apache weights; choose oracle first), AHIQ (from the paper; DeformConv in Rust; pin the timm weights it silently depends on) | permissive / unknown | MUSIQ transformer, DeformConv |
| hold | Q-Align / Q-ReAlign / Compare2Score (LLMs, 2–19 GB), FID, DMM, TOPIQ face variants (need a face detector) | — | — |
| no | TReS (agreement forbids distribution and assigns derivatives), CNNIQA, IQT, MA-CLIP, MetaIQA, MS-SWD (no licence; paper-only if ever wanted) | — | — |

Retraining TOPIQ or DBCNN on permissively licensed data would make their weights
permissive. That is a separate project and is not planned here.

## 7. Phase 0 (first concrete steps)

1. **Bar A baseline:** on i265 (CPU + RTX 4080) and dev (CPU), time upstream `lpips`,
   `DISTS_pytorch` and pyiqa end-to-end on a 64²–4096² ladder with pyiqa's own
   protocol, and raw PyTorch fp32/fp16 compute on the same ladder. Commit under
   `benchmarks/`.
2. **Root-cause burn-vulkan's VGG slowness.** It decides whether WebGPU route 1 is
   viable.
3. **Vertical slice:** `crates/zenmetrics-nn` with LPIPS-vgg end to end (card, export
   script, rten CPU, burn-cuda, `ort` behind its feature, warm reference, `score-pairs`
   column), gated on parity with upstream `lpips`. This sets the template every later
   model copies.

**Status 2026-10-15** ([`nn_lpips_phase0_2026-10-15.md`](../benchmarks/nn_lpips_phase0_2026-10-15.md)):
step 1 done on i265 (upstream `lpips`/`DISTS_pytorch`, CPU and 4080, fp32/TF32-off/fp16,
64²–4096²; dev not yet measured). Step 3's CPU part is in `crates/zenmetrics-nn`:
LPIPS-vgg on rten, warm reference, parity 4.5e-8 against upstream on 18 pairs,
1.13–1.32× faster than PyTorch CPU from 512² up and 9–16 % slower at 64–256².
Later on 2026-10-15: step 2 done (burn's f32 3×3 conv on wgpu falls back to
`conv_direct`, [`burn_wgpu_conv_2026-10-15.md`](../benchmarks/burn_wgpu_conv_2026-10-15.md));
burn-cuda measured and rejected for now (deterministic deviation up to 1.3e-4 from
upstream); the `ort` feature (ONNX Runtime CUDA, TF32 off, deterministic) matches
upstream to 6e-8 and is the GPU engine. Wave 1 so far: LPIPS alex/vgg/squeeze and DISTS
on rten and ort-cuda, parity ≤ 1.8e-7; CLI `--metric lpips-alex|lpips-vgg|lpips-squeeze|dists`
behind the `nn` / `nn-ort` features. Remaining wave 1: DeepDC, ST-LPIPS (BlurPool from the
paper), LPIPS+. CPU optimisation is running as an overnight SWE-2 lane (branch `nn-opt`).
Checklist for each new model: `crates/zenmetrics-nn/ADDING_MODELS.md`.

## 8. Spillover to zentract, zensally, zensr

- **zentract** (tract-onnx 0.22.3 cdylib; CPU, f32, tract multithreading not enabled):
  an rten engine behind the same ABI would be 1.14–1.26× faster single-threaded and
  1.3–2.2× multithreaded on these CNNs, and drops tract's assembly build, which needed a
  git pin for Windows-ARM64. Enabling tract's `multithread-mm` is a free interim step.
  If `zenmetrics-nn`'s engine layer turns out general, it could replace zentract's
  engine outright.
- **zensally** (UltraFace, MicroSalNet via tract): tiny nets, so native GPU gains little
  at batch 1. rten would remove the Windows-ARM64 pin and is probably faster
  (extrapolated from these CNNs, not measured on its models). The real gain is
  in-browser face and saliency detection through rten wasm or burn WebGPU.
- **zensr** (hand archmage SIMD conv, measured faster than tract): keep its CPU engine.
  Its GPU path needs the tiled implicit-GEMM conv that Bar B step 4 would add to
  zenforks-cubecl, so that kernel would be shared.

## 9. Decisions needed

1. The core crate's name (avoid `zeniqa`-like names).
2. Counsel: running pyiqa as an internal oracle; ImageNet-trained backbone weights;
   OpenAI CLIP's model-card scope ("any deployed use case … out of scope"); A-FINE weights
   trained on the authors' NC DiffIQA; LLaMA-2 terms behind Compare2Score / Q-Align.
3. `fsim` and `vsi` port research-only MATLAB releases whose headers forbid use "as the
   basis of a commercial software or hardware product without first obtaining
   permission" ([licence report](PYIQA_UPSTREAM_LICENSES_2026-10-15.md) §3): ask the
   authors for permission or get counsel's view.
