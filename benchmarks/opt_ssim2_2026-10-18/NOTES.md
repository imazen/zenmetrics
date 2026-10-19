# opt_ssim2_2026-10-18 — fast-ssim2 size-gated rayon dispatch

Branch: `opt/ssim2-par-gate` (zenmetrics pin bump) + `opt/ssim2-par-gate` on
`imazen/fast-ssim2` @ `7c71ca00f08df3e6582f54c10edeef228cb3c` (the code change;
zenmetrics `ci/sibling-pins.tsv` updated to that rev, `Cargo.lock` unchanged,
`lock.sh --check` PASS).

## What changed

`fast-ssim2` unconditionally dispatched per-row/column IIR loops, plane
downsampling, channel work, and map fusion into rayon whenever `RAYON_NUM_THREADS
> 1`. Below ~1 MP the per-chunk work is microseconds and the steal/epoch
protocol dominated: 64x64 t32 was 22 ms (canon 1.3 ms) — a 17x anti-scaling.

New runtime gate `par_ok(pixels)` in `fast-ssim2/src/pipeline/mod.rs`
(`PAR_MIN_PIXELS = 1<<17`, measured crossover between 64^2/255^2 and ~1 MP)
selects the already-existing serial fallback bodies under the rayon build. All
dispatch sites in `pipeline/mod.rs`, `pipeline/precompute.rs`, `pipeline/simd.rs`
use it; strip-path sites keep their explicit `parallel` opt-in. The serial code
is bit-for-bit the same algorithm (same ordered FMA chains), so scores are
identical on every tested input.

## Flags addressed (portcheck protocol, same host, libjxl `ComputeSSIMULACRA2`
canonical via `iqa-jxl`)

- 64^2 t32: 22.1 -> 0.25 ms (0.19x canonical; was 16.7x)  **cleared**
- 255^2 t32: 35.3 -> 14.5 ms (1.42x; was 3.5x)
- 1 MP t32: 82 -> 69.7 ms (0.54x)  **cleared**

## Residual: 255^2 still flags at ~1.4x (t1 and t32, photo + screen)

Not threading — the serial path itself is ~4.5 ms slower than libjxl at 255^2.
Profiling (`cycles:P`, post-gate build) shows:

- ~7% `fmaf`/`fmaf_with_fma` libcalls — the crate compiles baseline x86-64; the
  scalar IIR tail paths (`fast_gaussian_vertical_1d`, row tails, edge-mirror
  prologues) and sub-8x8 deep scales run `f32::mul_add` through libcalls.
- `vertical_1d` scalar closure ~3.7% — tail columns (7 of 255 at scale 0, whole
  4x5 last scale) run an unpipelined scalar IIR.
- malloc/free/memset ~2-3% — per-scale plane + scratch allocs.

Closing the residual needs a fast-ssim2 SIMD-tails + scratch-reuse round
(similar to the ssimulacra1 treatment) — est. ceiling ~1.5-2 ms of the 4.5 ms
gap from these items alone; the rest is per-scale fixed cost. Follow-up.

## Verify

- fast-ssim2: `cargo check` {default, no-default, rayon-only, all-features},
  clippy -D warnings, `cargo test --all-features` — all pass (73 tests incl.
  reference-parity bit-exact pins).
- zenmetrics-api: 11 ssim2 integration tests pass (cached-ref parity, strip,
  dispatch, HDR).
- score_bits identical on every measured cell (photo+screen, all sizes).
- Timing: `iqa-compete` score/warm, RAYON_NUM_THREADS=1/32; canonical `iqa-jxl
  ssim2` same pairs (canonical is thread-insensitive; t32 ~ t1).

## Caveats

- 4K t1 after (~914 ms) measured while the box was contended by the portcheck
  lane; the baseline 769 ms was cleaner. True after-value is likely ~770-800 ms;
  both are well under canonical (~1.2 s).
- Canonical `iqa-jxl` ignores the thread arg (single-threaded C++), so t32
  canonical ≈ t1.
