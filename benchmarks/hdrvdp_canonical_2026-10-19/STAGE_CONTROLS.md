# Decoded pixels and spectral interpolation controls

Neither AIC2026 HDR invocation is identified. The original all-61,
six-decimal raw-Q equality requirement remains unchanged. This continuation
adds exact decoded-input checks and stage controls, without setting grids,
score fitting or shipped metric changes. Published-column agreement and
native-versus-upstream agreement are measured separately.

## Exact decoded samples

All **122/122** reference/distorted PNGs produce identical packed RGB8
samples under four independent decoding paths:

- Octave 11.1.0 `imread`, the upstream scoring driver's actual ingress.
- FFmpeg 8.0.1 PNG decode to `rgb24`, used as the independent byte oracle.
- Pillow 12.3.0, with mode and geometry checked before reading bytes.
- Rust image 0.25.10 `image::open(...).to_rgb8()`, exactly the retained
  native scoring probe's ingress, rebuilt with its unchanged locked graph.

Each decoder matches **290,223,360 channel bytes**: zero unequal bytes,
maximum absolute code difference zero. Native and Octave dimensions also
match on every image. The Octave dump explicitly permutes dimensions to
packed row-major RGB; no colour transform is applied. These results exclude
decoder differences among these tested paths on this panel. They do not
establish the unavailable historical MATLAB decoder's behavior.

Commands start with `just hdrvdp-canonical-pixels PAIRS NEW_OUTPUT`.
The native extension uses `hdrvdp-canonical-build-decode`,
`hdrvdp-canonical-decode`, then a fresh `hdrvdp-canonical-pixels` output
with the native dump directory as its third argument. Every output path
must be new. Inputs, dimensions, per-image hashes, code differences and
all raw bytes remain under `~/tmp/hdrvdpcx/stages/` on r5900xt.

## Release and runtime distinction

Official 2.2.1 requests `'cubic'` in its spectral loader; 2.2.2 requests
`'pchip'`. [Octave](https://docs.octave.org/latest/One_002ddimensional-Interpolation.html)
still treats those names identically. [MATLAB's documented R2020b change](https://www.mathworks.com/help/matlab/ref/double.interp1.html)
uses cubic convolution for `'cubic'`, with a spline fallback for irregular
samples. Therefore an Octave 2.2.1 replay does not test the modern-MATLAB
interpretation of that call.

The diagnostic modern model uses Keys convolution with parameter −1/2,
quadratic endpoint extension on uniform samples, and not-a-knot splines
otherwise, with zero outside the source domain. Its boundary handling is
explicit and unqualified against actual MATLAB. Small controls cover
constants, lines, quadratics, exact knots, out-of-domain fill and a known
negative lobe that distinguishes the convolution from PCHIP. On the actual
spectral queries, a separately evaluated polynomial and SciPy CubicSpline
provide independent model checks. This is a tested model, not a claim of
MATLAB runtime identity.

## Stage coverage

Frozen source-inclusion probes compile the unchanged native interpolation,
HDR2 spectral and HDR3 spectral modules and their data. Type-only argument
shims let the standalone diagnostic binary call private spectral helpers;
no shipped library visibility or numerical source changes. Source SHA256
manifests bind every copied module/data file. The final probe compiles
without warnings. Earlier preliminary probes and their logs remain retained.

The upstream driver calls each release's actual `load_spectral_resp`.
It records the 420-point wavelength grid, resampled cone-log/rod/display
columns, processed receptor sensitivities, display emission and mixing
integrals. HDR3 additionally records XYZ matching functions, XYZ-to-native
conversion, age24 optical-density filter, age-adjusted emission and the
final normalized mixing matrix. Its age adjustment executes the original
release's optical-density block directly from the caller-supplied source.

The independent PCHIP check uses [SciPy's PchipInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html).
Both ordinary native grids and exactly matched upstream query points are
compared, separating grid-rounding differences from interpolation arithmetic.
The sampled native/upstream source tables are numerically identical; the
cone/rod file formatting differs. Native HDR3 extrapolates raw sub-380-nm
receptor tails, then discards the first 20 samples before use. Those unused
raw differences must be distinguished from the processed sensitivities.

The pre-age stage run shows actual 2.2.1/2.2.2 loaders are bit-identical in
Octave across every sampled entry. Native HDR2's mixing matrix differs from
2.2.2 by at most 1.942890293094024e-16. The modern-cubic model changes it
by as much as 3.901188103089259e-05. These are measured stage differences,
not inferred end-to-end score bounds.

Two targeted full-panel controls follow: the modeled modern interpolation,
and an Octave HDR2 run using the native port's twelve measured mixing
integrals. HDR2 consumes the spectral tables through this matrix; the
second intervention therefore measures the Q effect of that stage directly.
Neither control fits anything to published scores.
