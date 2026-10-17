# ort-webgpu engine receipt (2026-10-17, i265, RTX 4080 via Vulkan)

`Engine::OrtWebgpu` (feature `ort-webgpu`, implies `ort`): the same exported
ONNX graphs on ONNX Runtime's **WebGPU plugin execution provider**. Sessions
set deterministic compute and forbid CPU fallback
(`with_disable_cpu_fallback`), so any graph that leaves a node on
the CPU EP fails to commit. The WebGPU EP keeps memory patterns ON (the CUDA
BFCArena retention bug does not apply; peak device memory is unchanged
either way — measured, see below).

## Provider provenance (live-checked 2026-10-16)

- ORT core: `onnxruntime` 1.31.x shared library, loaded via `ORT_DYLIB_PATH`
  (`ort = "=2.0.0-rc.14"` + `load-dynamic`). The core `.so` contains WebGPU
  EP internals but does **not** register the provider by itself;
  `get_available_providers()` lists only CUDA/TRT/CPU and requesting
  `WebGpuExecutionProvider` in the providers list silently falls back to
  CPU (verified).
- WebGPU plugin EP: official `onnxruntime-ep-webgpu` wheel **0.4.0**
  (latest on PyPI 2026-10-16), linux x86_64:
  <https://pypi.org/project/onnxruntime-ep-webgpu/>. Contains
  `libonnxruntime_providers_webgpu.so`; registered at run time via
  `Environment::register_ep_library` from `ZENMETRICS_NN_WEBGPU_EP_LIB`.
  After registration `get_ep_devices()` reports two GPU adapters:
  Intel iGPU (vendor id `0x8086`) and RTX 4080 (vendor id `0x10de`,
  metadata `Discrete:1`); real inference verified on both.
- ORT plugin-EP/device API reference (ort crate 2.0.0-rc.14):
  `Environment::devices()` + `SessionBuilder::with_devices` +
  `with_disable_cpu_fallback` — the crate's `ep::WebGPU` type exists but
  targets a statically-linked provider; the plugin path is used instead.

## Parity vs exported upstream goldens (gate 1e-3 JND + bit-identical repeat)

All runs `just nn-parity <model-dir> <ladder> ort-webgpu model-tests,ort-webgpu <net>`.

| model | result | worst JND |
|---|---|---|
| lpips_alex | PASS | 2.28e-6 |
| lpips_squeeze | PASS | 1.50e-6 |
| lpips_vgg | **PARTIAL** (≤1024² pass; OOM 2048²) | 3.1e-7 (scored pairs) |
| stlpips_alex | PASS | 2.57e-6 |
| stlpips_vgg | **PARTIAL** (≤1024² pass; OOM 2048²) | 4.5e-7 (scored pairs) |
| dists | **PARTIAL** (≤1024² pass; OOM 2048²) | 1.20e-5 (scored pairs) |
| dreamsim | PASS | 1.35e-5 |
| maniqa | PASS | 8.35e-6 |
| liqe | PASS | 4.42e-5 |
| hyperiqa | PASS | 5.98e-6 |
| paq2piq | PASS | 3.13e-5 |
| pieapp | PASS, repeats bit-identical | 9.19e-7 |
| wd_s1..s5 | PASS | ≤ 2.5e-6 |
| wadiqam_fr_live / fr_tid / nr_live / nr_tid | PASS | 2.37e-6 / 5.09e-6 / 2.02e-7 / 6.81e-6 |
| afine, arniqa, ckdn, deepdc, musiq | LOAD FAIL | — |

**Load failures (capability, not a bug):** those five graphs carry
CPU-assigned host ops — `Cast`, `ConstantOfShape`, `EyeLike`, `Gather`,
`SequenceAt` dynamic-shape housekeeping that ORT will not place on the
WebGPU EP. With fallback forbidden the session refuses to commit;
`from_dir` reports `Error::BadModel` (environment/registration problems
report `Error::Engine` — e.g. `ZENMETRICS_NN_WEBGPU_EP_LIB` unset or not a
file). On ort-cuda these nodes land on the CPU EP silently.

**2048² OOM (capacity):** `vkAllocateMemory` → `VK_ERROR_OUT_OF_DEVICE_MEMORY`
inside the fused pair graphs' `Sub` bind group on the 16 GB adapter
(device-VRAM peak 15,690 MiB sampled by nvidia-smi during the attempt).
Every scored pair at ≤1024² matches the goldens. The same graphs complete
the full 64–2048² ladder on ort-cuda.

**OOM behavior — measured in-process** (`webgpu_probe`, DISTS on device 1,
memory pattern ON): 256² ok → 2048² returns `Err(MetricError{kind:Backend,
Engine("WebGPU validation failed … CreateBuffer … OOM")})` — no abort, no
panic → 256² ok again with a **bit-identical** score (session not
poisoned) → alternating 256²↔512² ×3 in one session, all bit-identical
repeats. No size guard added: the failure is a returned error and the
session stays usable, matching the brief's "guard only if it aborts or
poisons".

**Memory pattern ON vs OFF — measured** (same probe, DISTS, device 1,
nvidia-smi `memory.used` @ 200–250 ms): identical. Peak during the 2048²
attempt 15,690 MiB both ways; isolated 1024² run (40 evals, one session)
peaks at 5,298 MiB and holds it as a plateau; ≤512² steady-state ≤ 400
MiB both ways; identical Err-not-poisoned-stable-alternation. The CUDA
BFCArena retention bug it is disabled for on the CUDA path does not
exist in this EP, so ON stays.

## Adapter selection

`ZENMETRICS_NN_WEBGPU_DEVICE` = index into the EP's GPU list, else the
best vendor-name rank (NVIDIA > AMD/ATI > other); more than one
equally-ranked GPU without the pin is an `Error::Engine` naming every
candidate. The safe ort API exposes no vendor-id/discreteness metadata
and the crate forbids `unsafe`, so the rank and the env pin are the
resolution.

**Measured on i265:** `vendor()` returns `""` for both adapters (the
`Discrete` flag and vendor id live in metadata the safe API does not
wrap), so both rank "other" and the unset case errors —
`ambiguous WebGPU adapter: 2 equally-ranked GPUs; set
ZENMETRICS_NN_WEBGPU_DEVICE to one of: 0 (vendor "", id 32103, type GPU),
1 (vendor "", id 9988, type GPU)`. The parity/bench data above ran under
the earlier last-enumerated auto-pick on the same adapter (index 1); the
OOM/memory-pattern probes below ran with `ZENMETRICS_NN_WEBGPU_DEVICE=1`.

## nn_bench: lpips_vgg @ 256/1024/2048, budget 3 s/size·mode

`ZENMETRICS_NN_ENGINE=<eng> just snap-cargo @ run --release -p zenmetrics-nn \
 --features model-tests,ort-webgpu --example nn_bench -- \
 ~/tmp/nn-opt2/models-all ~/tmp/nn-phase0/ladder <out.tsv> 1415 256,1024,2048 3`

median ms (`n` ≥ 3, budget 3 s/mode·size; `—` = `VK_ERROR_OUT_OF_DEVICE_MEMORY`):

| engine | mode | 256 | 1024 | 2048 |
|---|---|---|---|---|
| rten | compute | 77.96 | 1134.85 | 4441.11 |
| rten | warm | 44.74 | 633.96 | 2430.28 |
| rten | e2e | 78.12 | 1166.08 | 4407.71 |
| ort-cuda | compute | 3.82 | 65.50 | 270.58 |
| ort-cuda | warm | 3.52 | 67.20 | 280.37 |
| ort-cuda | e2e | 4.15 | 77.68 | 315.94 |
| ort-webgpu | compute | 8.47 | 136.32 | — |
| ort-webgpu | warm | 9.67 | 140.75 | — |
| ort-webgpu | e2e | 11.03 | 146.10 | — |

WebGPU lands ~2.2× slower than CUDA at 256² and ~2.1× at 1024², ~9× faster
than rten at 256² and ~8× at 1024², and cannot reach 2048² on the 16 GB
adapter for this fused-pair graph. `warm` shows no benefit on either ort
engine (whole graph per call, as documented).

Bench TSVs committed beside this receipt, plus `parity_summary.tsv`
(per model × size rollup of the parity run). The full 568-row per-pair
receipt is over the generated-data commit cap — `parity.pointer.md`
records the host/path/sha256 on i265. Campaign logs:
`~/tmp/webgpu-campaign/{parity-*,loadfail-*,bench-*}.log` +
`~/tmp/webgpu-campaign/status` on i265.
