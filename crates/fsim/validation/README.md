# FSIM validation assets

`gen_goldens.m` produces the golden scores asserted in `src/tests.rs`
under GNU Octave. It drives the authors' reference implementation,
`FR_FSIMc.m` (L. Zhang, D. Zhang, X. Mou, D. Zhang, "FSIM: a feature
similarity index for image quality assessment", IEEE TIP 20(8), 2011).

The reference is distributed under a research-only license, so it is
**not committed** — obtain `FR_FSIMc.m` (and the `phasecong2`/`scharr`
helpers it calls, all in the same file) from the authors' FSIM
package and drop it next to this README. The local `fspecial.m` shim
covers the only image-package call the reference makes
(`fspecial('average', F)`).

Run from this directory:

    octave --no-gui --quiet gen_goldens.m

Deterministic `gen.m` patterns are shared verbatim with the Rust test
generators, so inputs are bit-identical on both sides.
