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

## Complete results

The [complete stage comparison table](spectral_comparisons.tsv) preserves
cardinality, exact-entry counts, median/maximum differences and the location
and values of each maximum. Native-versus-upstream results are separate
from published-column residuals.

| HDR3 native versus upstream 3.0.7 stage | samples | maximum absolute difference |
|---|---:|---:|
| Processed receptors | 1680 | 1.7763568394002505e-15 |
| Age24 optical-density filter | 420 | 2.220446049250313e-16 |
| Age-adjusted OLED spectra | 1260 | 5.429684479807406e-16 |
| Final normalized mixing matrix | 12 | 1.1102230246251565e-16 |
| XYZ-to-native matrix | 9 | 7.105427357601002e-15 |

The age24 filter is not identity: its native/upstream agreement is measured
after executing the actual optical-density stage. The discarded raw
receptor-tail discrepancy disappears before the scored receptor stage.

| full 61-pair HDR2 oracle versus published column | median Q error | maximum Q error | median site-JND error | maximum site-JND error | exact printed matches |
|---|---:|---:|---:|---:|---:|
| Modeled modern cubic/spline | 0.026448804274338045 | 0.21998219107626227 | 0.0039474697874885 | 0.04922876833055878 | 0/61 |
| Native spectral matrix substitution | 0.02650919246411121 | 0.2199231820780838 | 0.003953365176821766 | 0.049215451593439985 | 0/61 |

The modern model changes the original upstream Q by median
5.713591006184515e-05, maximum 0.00020115879955540095.
It does not explain the published-column gap and does not qualify actual
MATLAB endpoint/runtime behavior.

Substituting the native matrix changes upstream Q by median
7.105427357601002e-15, maximum 7.105427357601002e-14.
The resulting oracle versus the unchanged native Q still differs by median
0.0086901909133843, maximum 0.014212375528195764.
Thus the measured native spectral-stage differences do not explain that
separate port residual. No native score was corrected.

HDR3 full scoring was not rerun: its prior 61-pair upstream-versus-CSV
median remains 0.0013692098323083712 Q and maximum 0.012360806855934214 Q.
The new HDR3 observations qualify spectral stages, not new published scores.

## Reproduction and retained evidence

All commands use the shared heavy flock and run-heavy with a 24G cap.
Native spectral modules/data are source-hash verified unchanged. The
qualified decoder-comparison script is frozen as
`pixels-native-shape-r3/pixel_control.qualified.py`; a later unused-import
cleanup in the maintained script changes no measurements. All raw
per-image data and per-stage arrays stay outside git.

```sh
just hdrvdp-canonical-build-stages NEW_NATIVE_DIRECTORY
just hdrvdp-canonical-stages V221 V222 V307 NATIVE_PROBE NEW_OUTPUT
just hdrvdp-canonical-check-cubic
just hdrvdp-canonical-prepare-cubic COMPILED_V222 NEW_CUBIC_TREE
just hdrvdp-canonical-prepare-matrix COMPILED_V222 NATIVE_STAGE_TSV NEW_MATRIX_TREE
# Run each new tree through the existing complete-panel oracle recipe:
just hdrvdp-canonical-oracle NEW_TREE PAIRS NEW_OUTPUT 2 200 0 37.84 rgb-bt.709 precise
```

On r5900xt the raw root is `~/tmp/hdrvdpcx/stages/`. Pixel qualification is
`pixels-native-shape-r3/`; stage qualification is `spectra-aged-r2/`.
The two score attempts are `hdr2-modern-cubic-all61` and
`hdr2-native-mixing-all61`, each with frozen driver, source-tree hashes,
exact argv, input hash and complete log in its metadata.

| artifact | SHA256 |
|---|---|
| Pixel summary | `16942ca2fac7c37a811d9ac6f4e0c454ac6ee40ea8ffca4cf45dbd192a4ccb01` |
| Per-image pixel hash table (43,180 bytes; retained outside git) | `36143f09c21e343b390497289f0f3d73977a7822593fa215801a3db8fd8747b0` |
| Complete spectral receipt | `60b2d5d080f5ec101d35a4f0eaa47803f0696feb4e246c5fbe760aae2340834e` |
| Native spectral source manifest | `97072a2f4f8a0f57730bd9d2abd8203c05bd4a39d096ea87d3d6f939cd7882dc` |
| Modern modeled Q output | `7baa63acc3cc3f2c3e1467a4c6c0978a04ab80a088146713e5fad473e30b1b20` |
| Native-matrix Q output | `f6035764a88ac13a230542091e5b89808ccf12a4e2a41838ef3795587e5c9891` |

Resource receipts (commands/logs retained alongside each attempt):

```text
Pixel shape+sample qualification: run-heavy: done rc=0 13s | peak-RSS 0.09GiB | min-avail 50993MiB | peak-load 4.86
Expanded spectral stage control: run-heavy: done rc=0 1s | peak-RSS 0.09GiB | min-avail 51159MiB | peak-load 1.38
Modeled modern full61: run-heavy: done rc=0 287s | peak-RSS 0.49GiB | min-avail 50166MiB | peak-load 3.94
Native-matrix full61: run-heavy: done rc=0 285s | peak-RSS 0.48GiB | min-avail 50270MiB | peak-load 2.86
```

## Remaining limits

The tested decoding paths and spectral/PCHIP stages now have explicit
comparisons and end-to-end HDR2 controls. They do not identify the exact
historical invocation or establish an actual MATLAB runtime comparison.
MATLAB was absent from the checked local/dev locations. Historical MATLAB
release/toolbox details and executable HDR dataset calls remain missing
from the inspected resources. Model boundary conventions remain unqualified
without that runtime. A wholly single-precision pipeline is still a
lower-confidence untested hypothesis; the earlier test covered only the
transfer conversion, and no inspected written evidence specifies that route.
These limits do not justify a fitted score map or relaxing any gate.
No metric implementation, native score, numerical assertion or public API
changed. NIQE/PIQE remains outside this lane.
