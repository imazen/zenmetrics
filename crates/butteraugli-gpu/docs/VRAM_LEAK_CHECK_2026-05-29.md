# butteraugli-gpu VRAM leak check — 2026-05-29

Task #147 (`claude-butter-leak`). Confirms whether butteraugli-gpu's GPU
buffers leak across repeated use, or whether the steady-state VRAM is a
healthy CubeCL memory-pool plateau (the reuse path #144 flagged).

Host: RTX 5070 (12 GiB, WSL2), driver 596.21, CUDA runtime 13.2,
CudaRuntime backend, release build. Driver:
`crates/butteraugli-gpu/examples/vram_leak_check.rs`. In-repo data:
`crates/butteraugli-gpu/benchmarks/vram_leak_check_2026-05-29.tsv`
(per-cell summary, 22 rows). Full per-cycle series (1680 rows) is in
block storage — see
`benchmarks/vram_leak_check_2026-05-29_per_cycle.pointer.md`.

## TL;DR — NO LEAK

Across all three usage patterns, all four memory modes, and both 1 MP and
16 MP, butteraugli-gpu's VRAM **plateaus** — it does not grow per cycle.
#144's observation is confirmed: the first `set_reference` allocates the
working set; every subsequent reference / score / instance **reuses** it
(in-place for `set_reference`/`compute`, via CubeCL's pool for
construct→drop cycles). The strongest evidence is at 16 MP, where the
working set is ~3.8 GiB: 100 distinct `set_reference` calls hold at
**3840 MiB and return to exactly 3840 MiB**, and 30 full instance
construct→score→drop cycles (each allocating ~3.2 GiB) hold at **~3830 MiB
± noise** instead of OOMing the card. A genuine leak at this size would
exhaust 12 GiB in a handful of cycles.

This is a **healthy CubeCL pool-reuse plateau, not a leak.**

## Method

Three checks, one TSV row per cycle:

1. **`many_scores`** — one instance, `set_reference` once (warm modes) or
   cold per-call (`full`/`strip`), then score N=100 **distinct** distorted
   images. Leak ⇒ monotonic per-cycle growth; reuse ⇒ flat plateau.
2. **`many_new_refs`** — one instance, `set_reference` with **distinct
   pixel content** N=100 times, score after each. This is #144's exact
   reuse path. Leak ⇒ grows per new ref; reuse ⇒ flat.
3. **`create_drop`** — construct → score → **drop**, repeat N=30. CubeCL
   pools buffers *across* `Drop`, so the absolute VRAM does **not** return
   to the pre-allocation baseline even when healthy. PASS = the value
   plateaus (not climbing every cycle); FAIL = strictly increasing.

Modes: `full` (`new`+`compute`), `warm_ref` (`new`+`set_reference`+
`compute_with_reference`), `strip` (`new_strip`+`compute_strip`),
`warm_ref_strip` (`new_strip`+Mode-E cached ref). Sizes: 1 MP
(1024×1024), 16 MP (4096×4096).

### VRAM probe

`nvidia-smi --query-gpu=memory.used` (global card used MiB). Under WSL2,
`--query-compute-apps=used_memory` returns **empty** (the GPU paravirt
layer hides per-PID accounting), so the global figure is the only signal.
Each sample is the **min of 4–5 reads** after `block_on(client.sync())` +
a 60 ms settle, so CubeCL's deferred frees land first and a *transient*
allocation by a concurrent GPU process rides above the minimum and is
filtered. The delta is reported relative to a process-start baseline; the
absolute used-MiB is logged on every row.

### Shared-card caveat — why two passes

This host ran a concurrent `zensim` GPU eval during the measurement
window. Because the probe is **global**, that process contaminated the
*fast* 1 MP cells of the first pass (one cell even went to delta = **−1200
MiB** — impossible for a leak, proving contamination, not a butteraugli
defect). The canonical numbers below come from a clean pass that gates
each cell on a quiet GPU window (util < 5–8 %, free > 6.5–7.5 GiB), uses
min-of-4/5 sampling, and auto-retries a 1 MP cell if its delta goes
negative or its range exceeds 80 MiB. The analyzer additionally fits a
slope on the **lower envelope** (rolling-min) of the delta series, which
is robust to additive transients: a genuine leak pushes the floor up
monotonically; another process's churn cannot.

The 16 MP cells were unaffected (longer duration + larger working set
averaged out the transients) and matched the independently-measured peak
VRAM sweep (`gpu_vram_sweep_2026-05-28.tsv`) to within ~5 % — e.g. peak
`16mp full` 3.91 GiB vs this run's 3.74 GiB resident plateau.

## Source-level finding (why no leak is structurally expected)

`Butteraugli<R>` (`crates/butteraugli-gpu/src/pipeline.rs`) holds a
**fixed set** of `cubecl::server::Handle` fields — sRGB staging, planar
linear/XYB planes, blur planes, frequency bands, block-diff accumulators,
mask, diffmap, temps, blur LUTs. There is **no growing `Vec<Handle>`**.
The only nested instances are:
- `half_res: Option<Box<Butteraugli<R>>>` — one sibling, allocated at
  construction for `new_multires*`.
- `ref_cache_full: Option<Box<Butteraugli<R>>>` — the strip-mode Mode-E
  whole-image cache, lazily allocated **once** on first `set_reference`
  (`set_reference_strip_mode`, ~L932) and reused thereafter.

`set_reference_with_options` (whole-image path) allocates **no new plane
handles** — it overwrites the existing planes in place
(`populate_linear_from_srgb` / `apply_opsin` / `separate_frequencies` /
`compute_mask_pipeline_reference_only`). This is exactly #144's
"subsequent new refs REUSE" claim, confirmed in source.

Per-call **transient** handles do exist (dist/ref sRGB upload staging in
`compute_with_reference_inner` / `populate_linear_from_srgb`; the
reduction kernel's small `max_bits`/`sums`/`partials` handles in
`kernels/reduction.rs`) and are dropped at end of call. Whether they
accumulate (leak) or are recycled by CubeCL's pool (plateau) is what the
runtime measurement decides — and it decides **plateau**.

## Results (clean pass)

Per (check, mode, size): `floor` = median of the post-warmup
lower-envelope (the resident working set, MiB); `range` = max−min delta
across cycles (MiB, mostly nvidia-smi 1-MiB quantization noise);
`slope_env` = MiB/cycle on the lower envelope over the post-warmup window
(≈ 0 ⇒ no growth). Full series in the TSV.

| check         | mode            |   MP |  N | floor MiB | range MiB | slope_env MiB/cyc | verdict |
|---------------|-----------------|-----:|---:|----------:|----------:|------------------:|---------|
| many_scores   | full            |  1.0 |100 |       260 |        34 |            -0.342 | PASS    |
| many_scores   | full            | 16.8 |100 |      3831 |        34 |            +0.005 | PASS    |
| many_scores   | warm_ref        |  1.0 |100 |       223 |        33 |            -0.430 | PASS    |
| many_scores   | warm_ref        | 16.8 |100 |      3841 |        34 |            -0.292 | PASS    |
| many_scores   | strip           |  1.0 |100 |       143 |        33 |            +0.128 | PASS    |
| many_scores   | strip           | 16.8 |100 |       319 |        34 |            +0.025 | PASS    |
| many_scores   | warm_ref_strip  |  1.0 |100 |       304 |         9 |            +0.107 | PASS    |
| many_scores   | warm_ref_strip  | 16.8 |100 |      4128 |        33 |            -0.358 | PASS    |
| many_new_refs | full            |  1.0 |100 |       265 |        33 |            +0.207 | PASS    |
| many_new_refs | full            | 16.8 |100 |      3840 |        42 |            -0.416 | PASS    |
| many_new_refs | warm_ref        |  1.0 |100 |       223 |        33 |            -0.261 | PASS    |
| many_new_refs | warm_ref        | 16.8 |100 |      3840 |        33 |            -0.195 | PASS    |
| many_new_refs | warm_ref_strip  |  1.0 |100 |       320 |         0 |            +0.000 | PASS    |
| many_new_refs | warm_ref_strip  | 16.8 |100 |      4191 |        33 |            -0.069 | PASS    |
| create_drop   | full            |  1.0 | 30 |       256 |         0 |            +0.000 | PASS    |
| create_drop   | full            | 16.8 | 30 |      3830 |        41 |            +0.430 | PASS    |
| create_drop   | warm_ref        |  1.0 | 30 |       256 |         1 |            +0.034 | PASS    |
| create_drop   | warm_ref        | 16.8 | 30 |      3829 |        33 |            +0.000 | PASS    |
| create_drop   | strip           |  1.0 | 30 |       128 |         0 |            +0.000 | PASS    |
| create_drop   | strip           | 16.8 | 30 |       320 |         0 |            +0.000 | PASS    |
| create_drop   | warm_ref_strip  |  1.0 | 30 |       352 |         8 |            -0.322 | PASS    |
| create_drop   | warm_ref_strip  | 16.8 | 30 |      4128 |        33 |            -0.036 | PASS    |

All 22 cells PASS. The two `many_scores` 1 MP cells the first pass had
contaminated (`warm_ref`, `strip`) are the clean-re-run values above —
e.g. `warm_ref` 1 MP went from a contaminated 1346 MiB range / +16.6
slope to a clean 33 MiB range / −0.43 slope; `strip` 1 MP from a −1200
MiB (impossible) delta to a clean 143 MiB floor.

**Verdict criterion:** a leak is *monotonic* floor growth. Because the
global probe is MiB-quantized, a 1 MP working set wobbles in a ~33 MiB
band (one allocator block) as pure noise — so a cell is flagged only if
the lower-envelope end exceeds its start by > 48 MiB **and** the envelope
is mostly non-decreasing (≥ 70 % of steps ≥ 0). Every cell's envelope
end−start is between −33 and +18 MiB (below the 48 MiB band; several
negative), so **0 cells flag**. A single quantization step that returns
down (e.g. `many_new_refs warm_ref` 1 MP: 256 for 60 cycles → 289 →
back to 272) is correctly classified PASS.

### What the numbers mean per check

- **many_scores** (one ref, many scores): floor flat. 16 MP `full` slope
  +0.005 MiB/cyc over 100 scores — i.e. the dist sRGB upload staging +
  reduction scratch are recycled, not leaked.
- **many_new_refs** (#144's reuse path): the headline. 100 **distinct**
  references on one instance, floor flat at all sizes; 16 MP `full`
  3840→3840 (returns exactly), `warm_ref_strip` 1 MP range = **0 MiB**.
  In-place plane reuse confirmed.
- **create_drop**: bounded pool. 1 MP `full` range = **0 MiB** over 30
  reconstructions; 16 MP `full` holds ~3830 MiB instead of 30 × 3.2 GiB
  → the pool recycles dropped buffers; `Drop` does not leak.

## Verdict

**No VRAM leak in butteraugli-gpu.** All 22 cells plateau. The reuse #144
saw (first ref allocates ~3.99 GiB at 16 MP / 3990 ms; subsequent refs at
0.76 ms) is genuine in-place / pool reuse with a stable resident working
set, **not** unbounded growth. The steady-state plateau equals the
one-shot peak working set for each (mode, size), consistent with the
independent peak-VRAM sweep.

## Regression guard

`crates/butteraugli-gpu/tests/vram_no_leak.rs` (cuda-gated, same contract
as the other GPU integration tests — built by CI's
`--features cuda --all-targets` job, run only on a real GPU): three tests
(`many_scores_one_instance_no_leak`,
`many_new_references_one_instance_no_leak`,
`create_drop_instances_plateau_no_leak`) assert the post-warmup VRAM floor
grows by ≤ 96 MiB over the run (both end−start and max−min). Healthy reuse
is ~0; a real leak blows past the gate in the first dozen cycles.

Verified on this host (RTX 5070, 87.2 s, `--test-threads=1`):
- `create_drop`: 40 cycles, growth = 0, span = 0 MiB — PASS.
- `many_new_refs` (the #144 reuse path): 120 distinct refs, growth = 33,
  span = 34 MiB (quantization) — PASS.
- `many_scores`: 120 scores, growth = **−33** MiB (floor dropped),
  span = 33 MiB — PASS.
