# Pilot reconciliation: ssim, gmsd, ssim2 (2026-10-17)

Two lanes counted the same three metrics without seeing each other's files
(`agy`: Antigravity CLI, Gemini 3.8 Flash; `swe`: Devin SWE-2). Source commit
13aa694a (identical metric source to master 55ca0b69); fast-ssim2 99b98cb9.
Totals at 1024×1024 under the v1 evaluator, before rulings:

| metric | agy flops/px | swe flops/px | agy halo | swe halo |
|---|---:|---:|---:|---:|
| ssim | 708.0 | 756.3 | 5 | 10 |
| gmsd | 31.5 | 31.9 | 2 | 3 |
| ssim2 | 831.8 | 1363.7 | 160 | 447 |

Rulings (settled against source, not averaged):

- **ssim2 cube root** (`fast-ssim2/src/pipeline/xyb.rs`): computed inline,
  not a library call, so its arithmetic counts (convention 1). swe correct;
  agy recorded it as `pow` and listed the expansion as missing.
- **ssim2 blur** (`fast-ssim2/src/pipeline/gauss.rs:204-240`, `:271-313`):
  horizontal pass per output 3 mul + 10.5 fma + 3 add (4-way blocked
  recurrence), vertical pass 6 fma + 3 add + 3 sign flips + 2 add. swe
  correct; agy undercounted.
- **ssim2 halo**: neither. Per-level blurs do not chain; the downsample chain
  does; horizontal and vertical passes act on different axes (convention 4).
- **ssim strip overlap** (`classical.rs`, row bands recomputing 10 halo rows
  per band): real work of this implementation, not of SSIM; class
  `implementation` (convention 2). swe counted it, agy did not.
- **ssim halo**: 5 per axis (11-tap separable Gaussian). swe summed the two
  axes (10); agy correct.
- **gmsd downsample reach**: a 2×2 box reaches 1 source sample, so the
  cumulative halo is 1 + 1·2 = 3. swe correct.
- **bytes**: swe recorded per-tap reads as bytes; bytes are compulsory traffic
  (convention 3).

Both lanes revise their pilot tallies under conventions v2; the revised
totals are recorded in the coverage table of `../README.md`.
