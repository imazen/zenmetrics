# VSI validation assets

`gen_goldens.m` produces the golden scores asserted in `src/tests.rs`
under GNU Octave. It drives the authors' reference implementation,
`VSI.m` (L. Zhang, Y. Shen, H. Li, "VSI: a visual saliency-induced
index for perceptual image quality assessment", IEEE TIP 23(10),
2014), distributed from the author's site
(`cslinzhang.github.io/home/VSI/`).

The reference is distributed under a research-only license, so it is
**not committed** — obtain `VSI.m` (which contains `SDSP`, `logGabor`,
`RGB2Lab` and helpers inline) from the authors' package and drop it
next to this README. The reference calls two image-package functions;
install Debian `octave-image` (or `pkg load image` with the forge
package) — `imresize` is load-bearing, it is **not** shimmed because
the antialiased-bilinear resample is part of the algorithm's
semantics.

Run from this directory:

    octave --no-gui --quiet gen_goldens.m

Deterministic `gen.m` patterns are shared verbatim with the Rust test
generators, so inputs are bit-identical on both sides. Channel
assignment for RGB cases: `R = gen(k)`, `G = gen(mod(k,5)+1)`,
`B = gen(mod(k+1,5)+1)`.

`imresize_check.m` is the harness that verified the crate's
`conv_interp_vec` port (antialiased triangle kernel + symmetric
whole-point padding + full-kernel-sum normalisation) bit-identical to
`imresize(A,[m n],'bilinear')` at 17 scale combinations
(`MAX 0.000000e+00`). Re-run to re-verify after touching the resize
code.

Golden set (14 rows): RGB pairs at even/odd sizes, `F=1` and `F=2`
decimation regimes (384² and 520×400), identical pairs (score `1.0`),
constant/zero pairs (score `NaN` — the reference's saliency
normalisation degenerates), and a replicated-gray-channel case.
