# VIFp validation assets

`gen_goldens.m` produces the golden scores asserted in `src/tests.rs`
under GNU Octave. It drives the authors' pixel-domain reference
implementation, `vifp_mscale.m` — the multiscale scalar-GSM release
from the Laboratory for Image and Video Engineering (LIVE, UT
Austin), archived at
`metrix_mux/metrix/vif/vifp_release/vifp_mscale.m` in
`sattarab/image-quality-tools`. The algorithm is the computationally
simpler pixel-domain derivative of

> H. R. Sheikh and A. C. Bovik, "Image Information and Visual
> Quality", IEEE Transactions on Image Processing 15(2), 2006.

The reference file is **not committed** — its University of Texas
copyright header permits use/copying but requires the notice to
appear in all copies; obtain `vifp_mscale.m` separately and drop it
next to this README to regenerate. The generator needs the Octave
`image` package (`fspecial`): Debian `octave-image`, then the script
runs `pkg load image` itself.

Run from this directory:

    octave --no-gui --quiet gen_goldens.m
    octave --no-gui --quiet rgb_golden.m

Deterministic `gen.m` patterns are shared verbatim with the Rust test
generators, so inputs are bit-identical on both sides.

Golden set (17 rows): plane pairs at even/odd dims (64×64 through
512×512, incl. 37×41 where the deep scales' `'valid'` maps go empty
and 24×24 where only scale 1 survives), identical inputs
(`0.999999999992` — the reference does *not* score exactly 1: the GSM
gain `g = s/(s+1e-10)` stays just below one), and the degenerate
`NaN` cases the reference produces whenever the denominator
accumulator is zero — `min(w,h) < 17` (no `'valid'` map), constant
reference (`sigma1_sq ≡ 0`), all-zero input — plus constant-distorted
(`num = 0` → `0.0`) and the unrounded-luma RGB case.

## f64 note

The reference's `1e-10` masks sit ~4 orders above f64's noise floor
on the 0–255 scale; an f32 statistics plane leaves ~`1e-3` residual
variance on flat input and defeats every mask (constant inputs would
score finite instead of `NaN`). The port therefore runs `f64`
end-to-end — the goldens then land within `5e-13`, i.e. residual
`libm::log10`/`fspecial` rounding only.
