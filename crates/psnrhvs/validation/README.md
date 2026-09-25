# psnrhvs validation

Golden-score provenance for `src/tests.rs`. The constants pinned in
`tests::golden_scores` and `tests::rgb8_luma8_goldens` were produced by
running the reference implementation under GNU Octave.

## Regenerating

1. Obtain `psnrhvsm.m` (Nikolay Ponomarenko, <http://ponomarenko.info>).
   It is licensed for educational/research use and is therefore **not
   redistributed** in this repo. Place it in this directory.
2. `octave-cli --no-gui --quiet gen_goldens.m` — the single-plane rows
   (`tests::golden_scores`).
3. `octave-cli --no-gui --quiet gen_goldens_rgb.m` — the per-channel +
   BT.601-luma rows (`tests::rgb8_luma8_goldens`). The reference is
   single-channel only, so the multi-plane `rgb8` convention (mean of
   the three channel scores) is verified as the mean of three
   single-plane reference runs.

## Files

- `gen.m` — deterministic integer-pattern image generators (identical
  integer arithmetic to the Rust `gen*` helpers in `src/tests.rs`).
  NOTE: kinds that route through `bitxor` return `uint64`; Octave
  applies *integer* arithmetic to `double * uint64`, so luma mixes must
  cast `double(plane)` first (see `gen_goldens_rgb.m`).
- `dct2.m` — orthonormal DCT-II matching MATLAB `dct2` semantics
  (Octave's `image` package provides `dct2`; this local copy keeps the
  generator self-contained).
- `gen_goldens.m`, `gen_goldens_rgb.m` — golden emitters.
- `rung.m` — score-and-print helper (`name rows cols hvs_m hvs`).

Tolerance: the Rust port computes per-block energies in f32 with an f64
global accumulation, so goldens assert `|score − reference| ≤ 1e-3`
(the observed deltas are ≤ ~4e-4 dB).
