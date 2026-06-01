# zenmetrics-api ideal-API redesign — spec (task #159)

Nothing is published on crates.io, so this is a free redesign. Goal: **obvious to
use, nothing surprising, Auto never changes the score.** This doc is the coherent
implementation target for the 13-crate change — implement against it; don't diverge.

## Principles

- The 90% case is one call. Intent (one-off vs warm, speed vs memory) is expressed by
  *which entry point you call* + one optional knob — never a flag you can fumble.
- **`Auto` never changes the score.** It only ever selects modes/backends proven
  score-safe; score-shifting options (CappedPyramid; butter Strip before #158;
  cross-backend beyond the documented tolerance) are never auto-selected.
- Inputs are **typed**: decoded pixels carry their own format + dims (zenpixels), so
  there is no `(w, h, &[u8])` mismatch footgun. `&[u8]` means **encoded file bytes**.

## Backend model (the big change)

```rust
pub enum Backend {
    Auto,          // pick the best AVAILABLE backend: a working GPU device else Cpu
    Cuda, Wgpu, Hip,
    Cpu,           // the OPTIMIZED native CPU crates (fast-ssim2, butteraugli 0.9.4,
                   // dssim-core, zensim, in-tree cvvdp/iwssim) — the fast path
    CpuReference,  // the cubecl-cpu path (runs GPU kernels on CPU; slow; parity/debug
                   // only — this is what `Backend::Cpu` WRONGLY meant before this redesign)
}
```

- **`Backend::Cpu` flips meaning** to the optimized native crates (what users want).
  The current cubecl-cpu reference becomes `Backend::CpuReference`. The umbrella gains an
  optimized-CPU dispatch mirroring `zenmetrics-orchestrator`'s `cpu_adapter` (it already
  wires fast-ssim2/dssim-core/butteraugli/zensim/iwssim/cvvdp behind `cpu-*` features).
- **`Backend::Auto`** needs capability detection. `detect_gpu()`/`detect_cpu()` already
  exist in `zenmetrics-orchestrator/src/{gpu,cpu}.rs` — hoist the minimal availability
  probe into a shared spot (a small `zenmetrics-capability` module/crate, or the umbrella
  behind a feature) so both the api and the orchestrator use one detector. Auto resolves
  to a working CUDA/Wgpu device if present, else `Cpu`. Resolution is observable
  (`Backend::resolve_auto() -> Backend`) so it is never a black box.

## Inputs (zenpixels, not u8)

- Decoded pixels: a **zenpixels** pixel type (the format+dims-carrying `PixelSlice` /
  `PixelBuffer` the workspace already uses; `Metric::compute_pixels` takes `PixelSlice`
  today — make it the primary input). No `width`/`height` args at the call site.
- Encoded bytes (a PNG/JPEG file): `&[u8]`, via the `*_encoded` entry points, decoded
  internally (zenpixels-convert / the relevant decoder).

## Entry points

```rust
// one-off pair — Auto-safe optimal mode, reuse-context = one-off:
pub fn score(metric: Metric, backend: Backend, reference: &PixelSlice, distorted: &PixelSlice) -> Result<Score>;

// warm: one reference, many distorted (reuse implied by the entry point):
pub fn warm_reference(metric: Metric, backend: Backend, reference: &PixelSlice) -> Result<Warm>;
impl Warm { pub fn score(&mut self, distorted: &PixelSlice) -> Result<Score>; }

// encoded file bytes, decoded internally:
pub fn score_encoded(metric: Metric, backend: Backend, reference: &[u8], distorted: &[u8]) -> Result<Score>;

// optional priority (default Speed); ties broken among modes that FIT the VRAM cap:
//   score(...).priority(Priority::Memory)
pub enum Priority { Speed, Memory }

// expert escape — force any explicit MemoryMode (opt-in, documented may-not-be-score-safe):
//   Metric::Cvvdp.build(backend).memory_mode(MemoryMode::StripPair{..}).finish()
```

`MetricSession` / `OwnedSessionMetric` stay as the lower-level reusable handles; the
above are the obvious front door layered over them.

## Mode selection (data-grounded — task #157)

Auto maps `(metric, size, reuse-context, priority)` → the **fastest SCORE-SAFE mode that
fits**, seeded from `benchmarks/mode_wall_2026-05-31.csv`:
- one-off + large → strip / (cvvdp) StripPair; one-off + small → Full.
- warm → Full (strip only for VRAM feasibility).
- Never CappedPyramid (JOD-shifting). butter Strip is now safe (#158) so it is eligible.

## Tests — brute-force GPU-vs-CPU

A broad cross-backend matrix (cuda-gated where GPU is required, but the CPU arm always
runs): **every metric × {Cpu (optimized), Cuda, (Wgpu where built)} × sizes 256/512/1024**,
asserting:
1. Every backend produces a finite, in-range score for every entry point
   (`score` / `warm_reference` / `score_encoded`).
2. **CPU vs GPU agree within each metric's documented cross-backend tolerance**
   (per-metric; some are tight, some looser — record the measured band, no graceful skips).
3. `Backend::Auto` resolves to the expected backend per availability (GPU present → GPU;
   forced no-GPU → Cpu) and never to `CpuReference`.
4. `Backend::Cpu` runs the OPTIMIZED crate, not cubecl-cpu (assert via a perf/identity
   marker or the routed crate's signature).

## Implementation phases (each lands + verifies on master)

1. Backend model: add `Auto` + `CpuReference`, flip `Cpu` → optimized; hoist a shared
   availability detector; `resolve_auto()`.
2. Optimized-CPU dispatch in the umbrella (the 6 CPU crates, mirroring `cpu_adapter`).
3. zenpixels inputs across the scoring surface + the `*_encoded` decode path.
4. Entry points (`score` / `warm_reference` / `score_encoded`) + `Priority` + the
   data-grounded resolver (#157 Phase B); reshape `score_pair` into `score`.
5. Brute-force cross-backend tests.

Open: confirm the `Cpu`/`CpuReference` naming flip with the user before landing phase 1
(it changes the meaning of the existing `Backend::Cpu`).
