# hdrvdp CPU optimisation pass — 2026-08-29

**Host:** Apple M4 Pro (12-core aarch64 laptop), macOS Darwin 25.5.0, rustc 1.98.0,
workspace `release` profile (opt-level 3, `lto = "thin"`), every run under
`nice -n 19`, **no** `-C target-cpu=native`. Raw rows:
[`hdrvdp_perf_2026-08-29.tsv`](hdrvdp_perf_2026-08-29.tsv).

**Method.** One scored pair (`ColorEncoding::Luminance`, synthetic textured HDR
field + 3 % grating distortion) through `hdrvdp::hdrvdp` via
`crates/hdrvdp/examples/perf_probe`; best of 2–3 reps per size (reps agree to
&lt; 2 %). Interleaved paired statistics for the small sizes and hot kernels:
`cargo bench -p hdrvdp --bench pipeline` (zenbench).

**Bit-identity gate.** Every stage below reproduces the *exact* pre-optimisation
output bits: `tests/bit_lock.rs` (dual-path lock against frozen verbatim copies
of every pipeline stage + FNV bit-hash of the shared constant tables,
mutation-verified on three deliberate perturbations), plus the `perf_probe`
end-to-end FNV digest per size, unchanged from baseline through every commit:

| side | digest |
|---|---|
| 64 | `0xfccf99da9363dad1` |
| 97 | `0x6b2ea6ab77db7a03` |
| 256 | `0xb7bf4ac18a3dce32` |
| 1024 | `0x0c3c45ea79735ecc` |
| 4096 | `0xbe959ded56ea5a44` |

## End-to-end wall time (seconds, best of reps)

| stage (commit) | 64² | 97² | 256² | 1024² | 4096² |
|---|---|---|---|---|---|
| baseline (e58e985f) | 0.016 | 0.078 | 0.285 | 4.801 | 81.890 |
| + up_conv rewrite (1223a583) | 0.013 | 0.061 | 0.157 | 2.709 | 46.904 |
| + corr_dn rewrite (2447406a) | 0.010 | 0.052 | 0.096 | 1.688 | 30.038 |
| + FFT plans (5c156c2e) | 0.008 | 0.017 | 0.066 | 1.112 | 18.468 |
| + invariant hoists (624615e8) | 0.008 | 0.016 | 0.061 | 1.047 | 17.472 |
| + box3x3/imresize (5cc2c63d) | 0.008 | 0.016 | 0.057 | 0.973 | 16.295 |
| + conv pad-row dedup (see git log) | 0.008 | 0.015 | 0.055 | 0.946 | 15.737 |
| **total speedup** | **2.00×** | **5.20×** | **5.18×** | **5.07×** | **5.20×** |

97² is the odd-size (Bluestein FFT) probe: the FFT-plan commit alone was 3.06×
there because the direct code rebuilt the chirp *and* re-transformed the chirp
filter on every 1-D transform.

## `total = α + β·pixels` fit

A single line does not fit 4 decades well — the slope grows mildly with size
(cache) — so both the global least-squares fit and the segment slopes are
given. Fits use the four power-of-two sizes (97² sits on a different code path).

| | α (ms) | β (µs/px) | segment slopes (µs/px): 64→256 / 256→1024 / 1024→4096 |
|---|---|---|---|
| baseline | ≈ 0 (−117 by global LSQ — the 4096² point dominates; the small-size extrapolation gives −2 ms, i.e. no measurable intercept) | 4.887 | 4.38 / 4.59 / 4.90 |
| optimised | ≈ 5 (small-size extrapolation; global LSQ −16) | 0.972 | 0.80 / 0.93 / 0.97 |

The optimised build's ~5 ms fixed cost is now visible at 64² (the per-pixel
work no longer buries it); it is per-call setup (Photoreceptor JND tables,
spectral matrix, FFT plans). A caller scoring many pairs at one geometry could
amortise it, but that needs an API for reusable state — owner call, see below.

## What changed (each landed as its own commit, each bit-identical)

1. **up_conv** (was 47 % of the metric, `sample` closure = 2×`reflect1` + 2
   divisibility tests *per tap*): materialise the zero-upsampled reflect1
   extension once per call, correlate against a precomputed non-zero tap list,
   8 outputs per block (LLVM vectorises across outputs; each output keeps its
   serial accumulation chain in the original ky-major order). 1.75–1.82× end
   to end at 256²–4096².
2. **corr_dn** (26 %): same shape — reflect-pad once (interior rows are
   `copy_from_slice`), dense blocked correlation, contiguous fast path for
   step 1, strided for the step-2 `lofilt`. All taps participate (the original
   had no zero-skip) so `+0.0·x` lands identically. 1.56–1.64× more.
3. **FFT plans** (butterfly `sin_cos` was recomputed per block; Bluestein
   rebuilt chirp + chirp-filter FFT per call): per-length plan with per-stage
   twiddles (same `expi(ang·k)` expressions), shared across `fft2` rows/cols
   and `conv_fft_real`'s forward+inverse; 8-column blocks in the column pass.
   1.45–3.06× more.
4. **Invariant hoists**: `Photoreceptor::{lum_min,lum_max}` memoised (were
   `10^±5` powf per *lookup*, 6 exp10 per pixel); optical MTF filter built
   once per pair instead of per image (`visual_pathway_with_filter`);
   `decompose` moves pyramid planes instead of cloning. ~1.06× more.
5. **box3x3 interior fast path + imresize restructure**: branch-free interior
   for the mutual-masking blur (identical nine additions in identical order);
   imresize per-output weight rows sliced (kills bounds checks) and the
   vertical pass blocked 8 wide. ~1.07× more.
6. **conv_fft_real pad-row dedup**: every padded row below the image is the
   same all-`pad_value` row, and a DFT is a deterministic function of its
   input — transform one, copy into the rest (forward row pass only).
   ~1.03× more.

## Profile evidence

macOS `sample` (10 s) on `perf_probe 512`, top of stack, before → after:

| symbol | baseline | after all 5 |
|---|---|---|
| `up_conv` | 47 % | ~10 % |
| `corr_dn` | 26 % | ~18 % |
| `fft_radix2(_planned)` + `fft2` + sincos | ~10 % | ~18 % |
| libm `pow` (masking transducer) | ~4 % | ~17 % |
| `imresize` | 1.2 % | ~3 % |
| `mutual_masking` | 1.1 % | ~1 % |

(Percentages are of a run ~5× faster on the right column; the survivors are
the bit-identically-irreducible parts, see below.)

## Tried / considered and NOT landed (with reasons — don't re-try blindly)

- **`w == 1` twiddle-skip in the FFT butterflies** (skip `mul` by `1+0i`):
  NOT bit-safe — `re·1 − im·0` differs from `re` when `re` is `-0.0` (gives
  `+0.0`), and exact-zero intermediates with either sign are common in FFTs of
  structured images. Rejected by construction, not measured.
- **FMA (`mul_add`) in any kernel**: changes rounding (the bit lock's mutation
  test M3 proves it goes red). Not eligible without an owner decision to
  re-baseline the metric.
- **Real-input FFT (rfft) / split-radix / radix-4**: all reorder or refactor
  the float DAG → different bits. Same owner gate.
- **corr_dn micro-variants** (indexed `for j` instead of `iter_mut().zip`,
  BLK 8 → 16): both measured *neutral* on the 256² kernel bench (57.9–61.3
  Mops/s across variants, within run-to-run noise) — LLVM already
  SLP-vectorises the products and the ordered per-output `fadd` chain is the
  floor; reverted. The ordered-reduction serialisation is the bit-exactness
  tax; only reassociation (owner-gated) lifts it.
- **Uniform-grid direct indexing instead of `interp1_linear`'s binary search**
  in the masking CSF lookup (~2 %): the grid is only uniform up to libm
  rounding (`log10(10^x)`), so direct index computation can pick a different
  interval at cell boundaries → different bits. Rejected.

## Remaining, ranked (all need either owner input or are small)

1. **libm `pow` in the masking transducer, now ~17 % of what's left.**
   Bit-identically irreducible (any vectorised/approximate pow changes bits —
   cvvdp's `vpow` is ≈128 ULP by design). If the owner ever re-baselines the
   metric (e.g. for the planned `hdrvdp-gpu`, which cannot be bit-equal to f64
   CPU anyway), a vectorised pow is the single biggest CPU win left.
2. **FFT butterflies ~18 %**: an SoA (split re/im) layout would let LLVM
   vectorise pairs of butterflies with the identical per-butterfly DAG
   (bit-safe in principle) — a substantial rewrite; not attempted this pass.
3. **Threading**: every stage is per-pixel/per-band parallel, and the crate is
   deliberately dependency-free (no rayon) — owner call on the policy. The
   fleet already parallelises across pairs, so per-pair threads may be moot.
4. **Reusable per-geometry state** (Photoreceptor tables + FFT plans + MTF
   filter, ~5 ms/call): needs a public API for a session object — owner call
   (public API changes need approval).
5. **Fusing the four `bfilts` convolutions** (one padded source, 4
   accumulators): saves 3 padding passes + 4× tap-window loads; corr_dn is
   near memory/issue bound so expect ≤ ~5 % end to end.
