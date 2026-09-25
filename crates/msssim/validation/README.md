# MS-SSIM validation assets

`gen_goldens.m` produces the golden scores asserted in `src/tests.rs`
under GNU Octave. It drives the authors' reference implementation,
`msssim.m` + `ssim_index_new.m` (Z. Wang, E. P. Simoncelli, A. C.
Bovik, "Multi-scale structural similarity for image quality
assessment", IEEE Asilomar 2003), as archived in
`LabForComputationalVision/MAD_Competition` (the `msssim/` folder is
Wang's own distribution from `ece.uwaterloo.ca/~z70wang`).

The reference files are **not committed** — copy `msssim.m` and
`ssim_index_new.m` next to this README. The reference needs the
Octave `image` package (`fspecial`, `imfilter`): Debian
`octave-image`, then `pkg load image`.

Run from this directory:

    octave --no-gui --quiet gen_goldens.m
    octave --no-gui --quiet rgb_golden.m

Deterministic `gen.m` patterns are shared verbatim with the Rust test
generators, so inputs are bit-identical on both sides.

Golden set (15 rows): plane pairs at even/odd dims covering level
counts 1, 2, 3 and 5 (`level` is passed explicitly with
`weight(1:level)` — the same semantics as the port's auto-level), the
176-min-dim 5-level boundary, identical and constant inputs (all
`1.0`), a constant-vs-textured case, a negative-mean case exercising
the complex-power real-part semantics (`g1_g2_16`: `mssim_array(1) <
0`, score `= |m|^w·cos(πw)`), and an RGB pair scored on the unrounded
`0.2989/0.5870/0.1140` luma (Octave `rgb2gray` rounds to uint8, so
`rgb_golden.m` computes the coefficients explicitly).
