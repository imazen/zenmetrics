# Integrating zensim / SSIMULACRA2 / butteraugli into an encoder search loop

Task #163, 2026-08-30. Host: Apple M4 Pro (12c, 24 GB), macOS. Build base
`f240bb93`. Provenance, statistics and caveats: `metric_loop_2026-08-30.meta`.

**The question.** A quality-targeting search loop scores one fixed reference
against many distorted candidates. What does one *more* candidate cost once the
reference is warm, and where does a GPU start winning?

**The answer, in one line.** Warm-reference always — it pays back by the second
candidate at every size, on CPU and GPU alike, and every parity delta is exactly
zero. Whether to use a GPU at all **depends on the GPU's memory architecture,
not on the image size or the candidate count**: on a unified-memory GPU (Metal,
measured here) the crossover is 2–4 candidates and the encoder's 3–5 already
clears it; on discrete CUDA it is 14–299 and the encoder never gets there
unless the metric instance is reused across images.

---

## 0. Four premises in the brief that did not survive measurement

| premise | what is actually true |
|---|---|
| "CPU zensim and fast-ssim2 have no reference-reuse API; determine whether one exists" | **Both have one, and both are already in production use.** `zensim::Zensim::precompute_reference{,_linear_planar}` → `compute_with_ref*` (`zensim/src/metric.rs:1977`, `:2303`), and `fast_ssim2::Ssimulacra2Reference::new` → `compare` / `compare_with` (`fast-ssim2/src/precompute.rs:230`, `:329`, `:360`). jxl-encoder already calls both (`ssim2_loop.rs:172`, `zensim_loop.rs:1017`). There is no missing CPU warm-reference type for any of the three. |
| "the metrics take sRGB u8 or a PixelSlice" — implying u8 ingress is the thing to measure | **The encoder loops are linear f32 planar end to end.** `perceptual_loop.rs:2275` / `zensim_loop.rs:1147` produce `recon_r/g/b` as planar linear f32 at `padded_width` stride and score the full frame. No u8 appears in any of the three loops. Timing the u8 path answers a different question, so this work re-measured the planar f32 path. |
| butteraugli-gpu's cached-ref "is identical to the one-shot score — verify that claim" | **The claim is true and vacuous.** In strip mode `set_reference_srgb_u8` only does `self.cached_ref_strip = Some(ref_rgb.to_vec())` (`butteraugli-gpu/src/opaque.rs:703`) and `compute_with_reference_srgb_u8` calls `self.inner.compute_srgb_u8(&held, dis_rgb, …)` (`:726`) — the full one-shot, plus a host `Vec` clone per call. It scores identically because it does identical work. See §5; this is the most consequential GPU finding here. |
| implied: strided input might force a packed copy | **butteraugli and zensim both take an arbitrary `stride` on both sides and are bit-identical to the tight path** (measured, §4). fast-ssim2 accepts `ImgRef` but `ToLinearRgb` always materialises a packed `LinearRgbImage`, so it alone forces a repack. |

A fifth, found on the way: **`fast-ssim2`'s `rayon` feature is off in every
consumer, and that is correct** — enabling it measured 1.00–1.04× at 256²–2048²
(inside noise) and **0.31× at 64², a 3.2× regression**. See §3.

---

## 1. The recommendation, per metric

CPU, planar linear f32, multi-core (crate defaults). `N` = candidates per reference.

| metric | shape to use | per candidate @1 MP | vs cold one-shot | pays back after | ingress |
|---|---|---|---|---|---|
| **butteraugli** | `ButteraugliReference::new_linear_planar` once → `compare_linear_planar_into` per candidate | **22.4 ms** | 0.55× | **N ≥ 2** | none — pass `padded_width` straight through |
| **zensim** | `precompute_reference_linear_planar` once → `compute_with_ref_and_diffmap_linear_planar` per candidate, **at the padded stride** | **11.4 ms** | 0.78× | **N ≥ 2** | 1.7–3.9 % today, and it is avoidable — see §4 |
| **SSIMULACRA2** | `Ssimulacra2Reference::new` once → `compare_with(&mut CompareContext)` per candidate | **28.1 ms** | 0.58× | **N ≥ 2** | 5–8 %, **not** avoidable through today's API |

GPU: depends on the GPU. On **unified memory (Metal / Apple integrated)** the
GPU beats this table from N ≥ 3 at ≥ 1 MP — setup is only 45–60 ms. On
**discrete CUDA** it does not, until N ≥ 14 (or always, if the instance is
reused across images). Full numbers and the decision function: §5.

### Cost model, validated

    T(N) = precompute + N x per_candidate

Measured against a real 5-candidate loop at all 7 sizes and all 3 metrics, the
model errs by **−2.9 % to +1.8 %** (`loop_n5_model_err_pct` in the TSVs). It is
safe to plan with.

### Fitted terms — report both, always

`alpha` (per-call fixed cost) fitted on 64²–512², `beta` (per-pixel slope) on
1024²–4096². Fitting one line over the whole ladder returns a *negative*
intercept for every cell — an artifact of the 16 MP point dominating unweighted
OLS, not a fixed cost. Multi-core:

| metric | alpha (ms) | alpha as % of a 64² call | beta (ms/MP) | single-thread beta | parallel speedup |
|---|---|---|---|---|---|
| butteraugli | 0.200 | 71 % | 23.4 | 66.3 | **2.83×** |
| zensim | 0.332 | 141 % | 9.90 | 44.4 | **4.49×** |
| SSIMULACRA2 | ~0.00 | ~0 % | 46.6 | 47.6 | **1.02× (serial)** |

Two things fall out of that table that a "ms/MP" figure alone would hide:

- **zensim's multi-core alpha exceeds its entire 64² call.** Threading it at
  thumbnail sizes is a *regression*: 0.315 ms MT vs 0.249 ms ST for one 64²
  one-shot. Below roughly 128², run zensim single-threaded.
- **SSIMULACRA2 does not parallelise.** Its slope is the same on 1 core and 12.
  At 1 MP it is the cheapest of the three single-threaded (26.7 ms vs
  butteraugli's 58.4) and the *most expensive* multi-core (28.1 vs 22.4). If
  cores are available, that ranking inverts — pick accordingly.

Per-megapixel cost is U-shaped, so quote it with the size attached:

| ms/MP (MT, warm) | 64² | 256² | 1024² | 4096² |
|---|---|---|---|---|
| butteraugli | 68.7 | 35.0 | 21.4 | 23.3 |
| zensim | 57.6 | 20.2 | 10.9 | 10.0 |
| SSIMULACRA2 | 23.7 | 23.4 | 26.8 | **44.3** |

SSIMULACRA2's per-pixel cost nearly doubles at 16 MP (working set overflow);
zensim is flat and is the cheapest large-image metric of the three by 2×.

---

## 2. What the warm reference actually saves, and why the three differ

The saving is not a property of "having a warm reference" — it is set by how
much of the pipeline is reference-only.

| metric | hoisted out of the per-candidate path | measured warm/cold |
|---|---|---|
| **SSIMULACRA2** | ref XYB + pyramid + `mu1 = blur(ref)` + `sigma1_sq = blur(ref²)` — 2 of 5 per-scale blurs and 1 of 2 XYB transforms (`fast-ssim2/src/precompute.rs:264-297`) | **0.582–0.589**, flat |
| **butteraugli** | ref XYB + the 10-plane `separate_frequencies` psycho decomposition + `precompute_reference_mask`, at both resolutions (`butteraugli/src/precompute.rs:295-326`) | **0.605–0.640**, flat |
| **zensim** | ref XYB + the 4-level downscale pyramid, and *nothing else* (`zensim/src/streaming.rs:2776-2857`) | **0.883–0.938**, flat |

zensim saves least because its blur is deliberately fused across both images —
`fused_blur_h_ssim` computes `blur(src)`, `blur(dst)`, `blur(src²+dst²)` and
`blur(src·dst)` in one pass (`zensim/src/blur.rs:2289-2292`), so `mu1` cannot be
hoisted without unfusing it. A 3-output `fused_blur_h_ssim3` "for the
cached-reference-moments path" exists but is wired only into the non-default
`feature-regime-v2` regime. **Do not chase zensim's warm-reference saving; ~10 %
(ST) to ~22 % (MT) is the structural ceiling of the shipped v1 path.**

The ratios are flat across three orders of magnitude of pixel count, on both
thread configurations. That is the strongest single result here: **one number
per metric predicts the warm saving at any size.**

---

## 3. Threading

`fast-ssim2` gates parallelism behind a non-default `rayon` feature. No consumer
enables it — jxl-encoder (`Cargo.toml:65`), zenmetrics-api (`:165`),
zenmetrics-orchestrator (`:119`) all take `features = ["imgref"]`. That looked
like free performance left on the floor. It is not:

| fast-ssim2 `rayon` off → on, multi-core, min-of-2 | 64² | 256² | 1024² | 2048² | 4096² |
|---|---|---|---|---|---|
| speedup | **0.31×** | 1.00× | 1.04× | 1.03× | 0.98× |

No gain above the noise floor anywhere, and rayon's dispatch overhead makes a
0.1 ms kernel **3.2× slower**. Leave it off. (Recorded as a comment on the
harness's dependency so a future session does not "fix" it back.)

butteraugli (`rayon` on by default) and zensim (`threads` on by default) do
parallelise — 2.83× and 4.49× respectively on 12 cores.

---

## 4. Pixel-format cost, and one avoidable copy in jxl-encoder

The encoder holds planar linear f32 at `padded_width`. What each metric charges
to accept it:

- **butteraugli — nothing.** `new_linear_planar(r, g, b, w, h, stride, params)`
  and `compare_linear_planar{,_into}(r, g, b, stride)` take the stride
  natively. `perceptual_backend.rs:664` already passes `padded_width`.
- **zensim — nothing, if you use the stride argument.**
  `compute_with_ref_and_diffmap_linear_planar` takes `stride`, and
  `zensim_loop.rs:1397` correctly passes `padded_width`.
  **But `zensim_backend.rs:363-366` does not**: it copies all three distorted
  planes strided→tight into scratch and then calls with `stride == width`. The
  callee accepted a stride all along. Worse, its `padded_width == width` fast
  path still runs a full `copy_from_slice` of all three planes. Measured cost of
  that copy: **1.7 % of the per-candidate wall at 1 MP, 3.9 % at 16 MP** (MT).
  Deleting it is free — the scores are bit-identical (below).
- **SSIMULACRA2 — 5–8 %, and unavoidable today.** `ToLinearRgb` always
  materialises a packed `LinearRgbImage` (`fast-ssim2/src/input.rs:25-58`), so
  `ssim2_loop.rs:319-325`'s planar→`Vec<[f32; 3]>` repack cannot be elided from
  the caller side. **fast-ssim2 is the one metric of the three that violates the
  workspace's "multi-row pixel APIs handle stride natively at no cost on the
  packed path" rule.** Closing it needs a planar entry point on fast-ssim2
  (`Ssimulacra2Reference::new_linear_planar` + a planar `compare_with`), which
  does not exist. That is the only genuinely missing API found in this work.

### Numeric parity — measured, not assumed

Every delta is **exactly 0.0** at all 7 sizes (`PARITY:*` rows in the TSVs):

| check | result |
|---|---|
| butteraugli: padded stride vs tight stride | **0** |
| zensim: padded stride vs tight stride | **0** |
| zensim: repeated warm compare vs first | **0** |
| SSIMULACRA2: warm reference vs cold `compute_ssimulacra2` | **0** |
| SSIMULACRA2: `compare_with(ctx)` vs allocating `compare` | **0** |

So none of the recommendations here changes any metric's output. Note the
zero-allocation entry points (`compare_linear_planar_into`, `compare_with`) are
free of wall-time benefit at these sizes — within noise of the allocating
variants — but they are bit-identical and they remove per-candidate allocation,
which is why they are still the ones to use in a loop.

---

## 5. GPU

Two datasets, and they disagree — which is the finding.

**(a) Metal, measured here, same box as the CPU numbers.** Driver:
`crates/zenmetrics-api/examples/loop_gpu_probe.rs`, run through the umbrella
`Metric` API (what production uses), `Backend::Wgpu`, 15 reps × 2 runs, min of
medians. Data: `metric_loop_gpu_metal_2026-08-30.tsv`.

**(b) CUDA / RTX 5070, quoted from `gpu_coldstart_2026-05-29.tsv`.** Not
re-measured; this host has no CUDA. Read (b) as "a different class of machine",
not a same-box A/B — and note trap 2 below before quoting its butteraugli row.

### (a) Metal — the crossover is 2–4 candidates

Setup = `metric_new + first_compute` (constructor + first-dispatch shader
compile). `N*` = candidates on one image before GPU total beats this host's
12-core CPU total.

| metric | mode | size | setup (ms) | GPU/cand (ms) | CPU/cand (ms) | **N\*** | warm/cold |
|---|---|---|---|---|---|---|---|
| butteraugli | Auto | 512² | 17.0 | 4.04 | 8.40 | 3 | **1.010** |
| butteraugli | **Full** | 512² | 13.4 | 6.68 | 8.40 | 4 | 1.171 |
| butteraugli | Auto | 1024² | 55.7 | 10.47 | 22.41 | 4 | **1.011** |
| butteraugli | **Full** | 1024² | 44.7 | 8.83 | 22.41 | **3** | 0.791 |
| butteraugli | Auto | 2048² | 201.5 | 35.47 | 96.98 | 3 | **1.003** |
| butteraugli | **Full** | 2048² | 166.1 | 22.33 | 96.98 | **2** | 0.656 |
| SSIMULACRA2 | Auto | 512² | 32.2 | 4.20 | 6.37 | 14 | 0.642 |
| SSIMULACRA2 | Auto | 1024² | 59.7 | 9.33 | 28.08 | **3** | 0.674 |
| SSIMULACRA2 | Auto | 2048² | 214.2 | 21.68 | 117.91 | **2** | 0.654 |
| zensim | Auto | 512² | 7.4 | 6.71 | 3.57 | never | 0.965 |
| zensim | Auto | 1024² | 24.3 | 9.94 | 11.45 | 15 | 0.894 |
| zensim | Auto | 2048² | 33.1 | 22.35 | 43.21 | **2** | 0.890 |

At 256² the GPU never wins for any metric — CPU is 1.3–2.3 ms/candidate there
and no GPU dispatch competes with that.

Why Metal is so different from CUDA: **there is no NVRTC and no discrete-memory
upload.** First-dispatch shader compile is 4–50 ms (CUDA: 130–915 ms) and
`metric_new` is 0.08–171 ms (CUDA: 40–5734 ms; zensim-gpu is ~0.1 ms because it
allocates lazily). Total setup at 1024² is **45–60 ms on Metal vs 574–653 ms on
CUDA** — an order of magnitude, and it is the whole crossover.

**The `warm/cold` column is the empirical proof of the butteraugli-gpu no-op.**
Under `Auto` it is 1.002 / 1.010 / 1.011 / 1.003 at 256²/512²/1024²/2048² —
dead flat at 1.0, the cached reference saves nothing. Under `Full` it falls to
0.79 at 1 MP and 0.66 at 4 MP. Same scores in every cell, both modes:
butteraugli 20.7495 / 20.9070 / 22.3951 / 25.3907, cold == warm exactly.
`Full` is also the *cheaper* constructor (166 vs 201 ms at 2048²).
**Always build butteraugli-gpu with `MemoryMode::Full` for a search loop.**
zensim's 0.89–0.98 and SSIMULACRA2's 0.64–0.68 on Metal match their CPU ratios
(0.78 / 0.58) — the same structural split as §2, on a different processor.

### (b) CUDA — the crossover is 14–299 candidates

`N*` = candidates on **one** image before GPU total beats CPU total, where
GPU total = `cold_total + N x gpu_warm` and CPU total = `precompute + N x cpu_warm`.

| metric | size | CPU/cand (ms) | GPU cold total (ms) | GPU/cand (ms) | **N\* fresh process** | **N\* instance reused** |
|---|---|---|---|---|---|---|
| butteraugli | 512² | 8.40 | 498.7 | 1.54 | 72 | 1 |
| butteraugli | 1024² | 22.41 | 653.4 | 3.60 | 34 | 1 |
| butteraugli | 4096² | 391.20 | 4923.9 | 50.20 | 14 | 1 |
| zensim | 512² | 3.57 | 570.3 | 1.66 | 299 | 1 |
| zensim | 1024² | 11.45 | 574.1 | 3.27 | 70 | 1 |
| zensim | 4096² | 167.31 | 914.2 | 37.80 | 7 | 1 |
| SSIMULACRA2 | 512² | 6.37 | 396.2 | 3.96 | 163 | 1 |
| SSIMULACRA2 | 1024² | 28.08 | 610.0 | 6.50 | 28 | 1 |
| SSIMULACRA2 | 4096² | 742.40 | 6740.5 | 47.70 | 9 | 1 |

jxl-encoder's actual candidate counts, from the effort ladder
(`effort.rs:1428-1447` × `lossy_search_seeds_for`, `effort.rs:2083-2092`):

| effort | e≤7 | e8 | e9/e10 | e11 | e12 | e13+ |
|---|---|---|---|---|---|---|
| compares | 0 | **3** | **5** | 18 | 68 | 132 |

### The decision function

```
use_gpu(metric, pixels, N, gpu_kind, instance_reused_across_images):
    if instance_reused_across_images:  return true        # N* == 1 everywhere
    if pixels < 0.25 MP:               return false       # CPU is 1-3 ms/cand; nothing beats it
    if gpu_kind == unified_memory:                        # Metal / integrated
        return N >= 3    at >= 1 MP                       # measured N* = 2-4
        return N >= 14   at 0.25 MP                       # ssim2 14, butter 3-4, zensim never
    else:                                                 # discrete, NVRTC/PCIe
        return N >= N_star_cuda(metric, pixels)           # 14-299, table above
```

For the encoder specifically:

- **Unified-memory GPU (Metal, Apple integrated) at ≥1 MP: GPU wins from
  e8 (N = 3).** Setup is 45–60 ms and per-candidate is 2.5× cheaper than CPU.
  Below 0.25 MP, CPU — every metric.
- **Discrete CUDA, e8 / e9 / e10 (N = 3–5): CPU, at every size.** Not close —
  the smallest CUDA N\* is 7, and at ≤1 MP it is 28–299.
- **Discrete CUDA, e11 (N = 18): CPU below ~4 MP.** At 16 MP, zensim (7) and
  SSIMULACRA2 (9) cross; butteraugli (14) crosses marginally.
- **Discrete CUDA, e12+ (N = 68–132): GPU wins at ≥1 MP** for all three.
- **Fleet/batch scoring — one instance over many images: GPU always,
  N\* = 1, both architectures.** Setup is paid once for the whole run. cubecl's
  device registry is a process-global `static`
  (`cubecl-common/src/device/handle/mutex.rs:34`), so client init is once per
  process and shared across *all* metrics; what does not amortise is
  `metric_new` (up to 3.8 s at 16 MP for butteraugli on CUDA, 5.7 s for ssim2,
  ~0 for zensim) and the first-dispatch shader/NVRTC compile. Hold the instance.

**The two deciding variables are instance reuse and GPU memory architecture —
not image size and not candidate count.** A per-image search loop cannot
amortise *discrete* GPU setup, because NVRTC compile and device upload dominate;
on unified memory there is barely any setup to amortise, so the same loop wins.
That is why a single "does GPU pay off" answer does not exist and the brief's
framing (one crossover in N and pixels) is one dimension short.

### GPU traps that would invalidate a naive benchmark

1. **`ButteraugliOpaque::new(...)`'s warm reference is a no-op above 224 px
   tall.** `resolve_auto` tries Strip *first* — "even if Full fits, butter is
   strip-preferred" (`butteraugli-gpu/src/memory_mode.rs:190-194`) — whenever
   `height > MIN_STRIP_BODY + 2*HALO_ROWS` = 64 + 160 = **224**. In strip mode
   `set_reference_srgb_u8` stores a host `Vec` (`opaque.rs:703`) and
   `compute_with_reference_srgb_u8` replays `compute_srgb_u8(&held, dis)`
   (`opaque.rs:726`) — the full one-shot, plus a `Vec` clone per call. **You must
   construct with `MemoryMode::Full` to get a real device-cached reference.**
   The umbrella maps `MemoryMode::Auto → butteraugli_gpu::MemoryMode::Auto`
   (`zenmetrics-api/src/memory_mode.rs:107`), so umbrella-built butteraugli gets
   the replay path. **Measured on Metal (§5a): `warm/cold` = 1.002 / 1.010 /
   1.011 / 1.003 at 256²/512²/1024²/2048² under `Auto`, versus 0.95 / 1.17 /
   0.79 / 0.66 under `Full`.** This is no longer a source reading.
2. **The committed butteraugli-gpu `warm_per_call_ms` predates this behaviour.**
   `gpu_coldstart_2026-05-29.tsv` is dated 2026-05-29; the strip held-reference
   landed 2026-05-31 (`302ff4fc`, "#160"). Before it, strip-mode `set_reference`
   *errored*; after it, it silently replays. The 1.5–50 ms warm figures are
   whole-image numbers and are not what `ButteraugliOpaque::new(...)` delivers
   today. Re-measure with an explicit `MemoryMode::Full` before quoting them for
   an Auto-constructed instance.
3. **zensim-gpu's warm-ref `Score` path needs a profile.** With
   `ZensimParams::new()` or `with_weights(...)`, `compute_with_reference_srgb_u8`
   returns `NoCachedReference` (`zensim-gpu/src/opaque.rs:876`). Use
   `ZensimParams::default_weights()`, or bench `compute_features_with_reference_*`.
4. **Shader/NVRTC compilation is lazy** — it happens on first dispatch, not in
   the constructor. Warm up one full call before timing.
5. **Metal**: do not enable `fast-reduction` for ssim2/butteraugli (Metal
   silently no-ops `Atomic<f32>::fetch_add`). zensim-gpu's
   `cached_ref_slot_rebuild::gpu_score_correct_after_foreign_size_rebuild` and
   `diffmap_invariants::invariant_1_*` are known-red on Metal (repo CLAUDE.md) —
   the first one is specifically a *cached-reference* failure after a size
   rebuild, so a warm-reference GPU path on Metal needs that resolved before it
   can be recommended.

### Batching — measured on Metal

`ssim2-gpu`'s `Ssim2Batch<R>` holds one cached reference and uploads N distorted
images in a single pinned buffer, then dispatches batched kernels. Measured with
the crate's existing `examples/bench_batch.rs` (its `WgpuRuntime` arm needed no
change), 256², median of 5 post-warmup — `metric_loop_batch_metal_2026-08-30.tsv`:

| batch N | sequential /img | batched /img | **Metal speedup** | CUDA speedup (2026-05-02, same driver + size) |
|---|---|---|---|---|
| 1 | 4.10 ms | 4.30 ms | 0.95× | 1.02× |
| 2 | 4.08 ms | 2.09 ms | **1.95×** | 2.00× |
| 4 | 4.03 ms | 1.41 ms | **2.86×** | 2.84× |
| 8 | 4.04 ms | 0.72 ms | **5.65×** | 3.30× |
| 16 | 4.06 ms | 0.56 ms | **7.24×** | 3.58× |

Batching beats sequential warm-reference calls from N = 2, and on Metal it keeps
scaling where CUDA plateaus — 7.24× vs 3.58× at N = 16, per-image dropping from
4.06 ms to 0.56 ms. At N = 1 it is a 5 % loss (the pack/finalise overhead with
nothing to amortise it over), so gate it on N ≥ 2.

Availability differs by metric: `ssim2-gpu` has `Ssim2Batch`; `butteraugli-gpu`
has `ButteraugliBatch` (one flat `N×W×H×3` buffer, fixed N, panics rather than
returning `Result`); **`zensim-gpu` has no batch API at all.** Neither batch type
is reachable from the opaque layer — both are typed-API only (`cubecl-types`).

**This does not apply to jxl-encoder's loops as written.** They are sequential by
construction: candidate `i+1`'s quantisation field is derived from candidate
`i`'s score, so there are never N independent candidates in flight. Batching pays
for a search that evaluates several candidates per round — multi-seed sweeps,
trellis alternatives, or the outer seed loop (`perceptual_loop.rs:1816` runs 2–4
seeds, but sequentially and with different quant state, so even that would need
restructuring). Treat the 2–7× as available to a *redesigned* search, not as a
speedup the current loop can pick up.

---

## 6. Concrete follow-ups

1. **`zensim_backend.rs:363-366` (jxl-encoder): delete the strided→tight copy**
   and pass `padded_width` as `stride`. Bit-identical (measured), worth 1.7 % at
   1 MP / 3.9 % at 16 MP per candidate. Same for `cvvdp_backend.rs`, which
   mirrors it.
2. **`rate_control.rs` (jxl-encoder) is the only cold-reference loop left** —
   3 iterations of `butteraugli_linear` with *both* images rebuilt as
   `Vec<RGB<f32>>` per call (`tile_distmap.rs:211-219`). At 3 candidates a warm
   reference is already past break-even (N\* = 2): ~35 % off its metric time.
3. **fast-ssim2 needs a planar entry point.** `Ssimulacra2Reference::new_linear_planar`
   + a planar `compare_with` would remove a 5–8 % per-candidate repack and fix
   the only stride-rule violation of the three. This is the one API that does
   not exist.
4. **Re-measure butteraugli-gpu warm-reference with `MemoryMode::Full`** and
   correct `gpu_coldstart_2026-05-29.tsv`'s butteraugli row, or annotate it.
5. **zensim below ~128²: force single-threaded.** Its MT dispatch alpha
   (0.332 ms) exceeds a whole 64² compare.
