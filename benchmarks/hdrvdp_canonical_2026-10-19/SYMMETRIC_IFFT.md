# Symmetric inverse FFT control

The canonical AIC2026 HDR2 and HDR3 calls remain unidentified. This control
retains the original requirement: all 61 raw scores equal the published
six-decimal values. It changes only the benchmark oracle, with no fitted
correction or shipped metric change.

[MathWorks documents](https://www.mathworks.com/help/matlab/ref/ifft2.html)
that `ifft2(F,'symmetric')` ignores the negative-frequency half. The retained
Octave compatibility adapter uses `real(ifft2(F))`. These operations differ
on deliberately asymmetric spectra, so this was a concrete untested
compatibility control. The documentation does not identify which 2-D storage
axis is retained. The helper measures both Hermitian completions; it does
not claim verified MATLAB runtime equivalence.

`scripts/symmetric_ifft_control.m` retains nonnegative rows, reconstructs
negative rows with conjugate/reversed-column partners, and handles DC and
Nyquist rows separately. Its column convention transposes this operation.
For every actual invocation it also computes the original real inverse,
records maximum spectral conjugacy defect and output differences, and
returns the selected convention. `HDRVDP_FFT_AUDIT=0` computes only the
selected completion/inverse for the second full-panel run; synthetic controls
require its output to be exactly identical to audit mode. No transform size, padding, input, filter,
display parameter or pooling operation changes. HDR2's default
`do_si_gauss=false` leaves the separate Gaussian FFT branch inactive.

The independent small controls pass for odd/even real spectra, explicitly
conflicting halves with a known inverse, an interior conflict distinguishing
axes, and MathWorks' documented 2x2 example. These assertions qualify the
helper's stated operations; they do not supply the missing MATLAB oracle.

```sh
just hdrvdp-canonical-check-symmetric
just hdrvdp-canonical-prepare-symmetric "$COMPILED_OCTAVE_TREE" "$NEW_TREE"
HDRVDP_FFT_AXIS=rows just hdrvdp-canonical-oracle \
  "$NEW_TREE" "$PAIRS" "$NEW_OUTPUT" 2 200 0 37.84 rgb-bt.709 precise
```

The wrapper freezes the helper, all upstream file hashes, driver, input
manifest, chosen axis, argv and runtime log. Each output path must be new.
Raw artifacts stay under `~/tmp/hdrvdpcx/symmetric/` on r5900xt.

## Remaining independent controls and blockers

- An actual MATLAB/Octave comparison with identical decoded input and
  stage outputs would test FFT backend, interpolation and toolbox numerics.
  No MATLAB executable was found in the checked local/dev locations. The
  Hermitian-completion experiment does not remove this runtime blocker.
- Spectral interpolation is a concrete stage for that comparison: release
  2.2.1 uses `interp1(...,'cubic',0)` in `load_spectral_resp.m`; 2.2.2 uses
  `'pchip'`. A useful control would compare the actual sampled cone/rod and
  display spectra before scoring. An unmeasured version substitution is
  not evidence of an explanation.
- Exact decoded uint8 pixel arrays from MATLAB, Octave and an independent
  PNG decoder can be compared before transfer conversion. Original PNG
  bytes are already verified; that alone does not establish decoder identity.
  No ICC/gAMA/cHRM metadata exists in this panel, and sRGB-chunk strata both
  retain residuals, so a generic colour-profile theory is insufficient.
- Earlier work tests single-precision transfer conversion, not every
  intermediate stage of a wholly single-precision pipeline. Testing such a
  pipeline needs an explicit runnable call and stage controls; it is not
  supported as the dataset route by the written conditions inspected.

The executable dataset invocation, exact historical runtime/toolbox versions
and any private preprocessing remain unavailable in the inspected public
resources. The available site/CTC settings do not specify them. There is no
evidence-backed finite grid that can establish absence of all undocumented
calls. No further broad grid or arbitrary fitted correction is justified.
The previous MEX/fallback and six-pair 3.0.6/3.0.7 controls retain their
measured exclusions. Neither failed exact reproduction permits modifying
production metric code to follow an unidentified target.
