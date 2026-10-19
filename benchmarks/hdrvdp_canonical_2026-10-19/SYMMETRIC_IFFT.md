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

## Complete row-retention result

All 61 pairs completed. Compared with the original compiled Octave oracle,
median Q change is 7.105427357601002e-15 and maximum is
9.237055564881302e-14. Versus published HDR2: median Q residual
0.026509192464104103, maximum 0.21992318207815487; site-JND median
0.003953365176822543, maximum 0.04921545159345575. Spearman is
0.9998413537810682 and **0/61** match printed precision.

The audit covers 488 actual FFT calls. Maximum spectral Hermitian defect
is 1.5417889814026006e-08 in absolute frequency-domain units; the maximum
per-call relative defect is 7.904616340102982e-16. Maximum spatial output
change is 1.7053025658242404e-13 for either completion versus the real
inverse; maximum row/column output difference is 1.4210854715202004e-13.
These maxima come from individual calls, not an extrapolation between
stages. They do not explain the measured 0.0265 Q median CSV residual.

Output SHA256: `5767b63c9394238a33b6da325294ab83028b4b65fcfdf6bc458d854b305fe07f`.
Tree-manifest SHA256: `e9e9037c8aa12d7d88797254989353efc17aedd4ccdb489329ab31b0152a58c2`.
Driver SHA256: `1338151d3c677da507c294f6d4e4a9285bd51215d25e89f0725625ece86d8a34`.
Input-pairs SHA256: `9a017bcaa1b9ffc127a3daea1cc854a303708bd2091f4ba45ef78c9f2bbf82fa`.
Source base: `8a2738f6d7537ebbae77b61018f63f27797bd694`; helper commit `257c403a`.
Exact command: the recipe above with tree `~/tmp/hdrvdpcx/upstream/v2-symmetric-control`,
pairs `~/tmp/aicfix-data/pairs.tsv`, output
`~/tmp/hdrvdpcx/symmetric/hdr2-rows-all61.tsv` and the extracted statistics
1.8.2 path recorded by its `.meta.json`. Its `.log` retains:

```text
run-heavy: done rc=0 495s | peak-RSS 0.66GiB | min-avail 49972MiB | peak-load 3.36
```

## Complete column-retention result and qualification

All 61 pairs completed. Median Q change versus the original compiled oracle
is 7.105427357601002e-15, maximum 9.947598300641403e-14.
Versus published HDR2: median raw Q residual 0.026509192464104103,
maximum 0.21992318207811934; site-JND median 0.003953365176821766,
maximum 0.04921545159344776; Spearman 0.9998413537810682; **0/61** printed matches.
Row versus column end-to-end Q difference: median 7.105427357601002e-15,
maximum 5.684341886080802e-14 on all 61 pairs. Neither completion explains the column gap.
These controls do not identify the dataset invocation or qualify actual MATLAB
runtime identity. Both canonical calls remain missing; HDR3 was not rerun.

The column run uses the same preparation recipe into a new tree
`~/tmp/hdrvdpcx/upstream/v2-symmetric-columns`, with
`HDRVDP_FFT_AXIS=columns HDRVDP_FFT_AUDIT=0`; all other oracle arguments
match the row run. Synthetic controls require audit/non-audit output identity.
Output: `~/tmp/hdrvdpcx/symmetric/hdr2-columns-all61.tsv`.

- Output SHA256: `33f7ba8b3f211f3eff63dfa4250bd695db58ef528549375bc098a0b9701f22dc`.
- Tree-manifest SHA256: `7ccb7a61057590ee19b0c26cacd9d41315aee805a88d31d3e24ec326af97b444`.
- Driver SHA256: `3bacf1e03377a18d3f0a229ccf4614afb2d301f2afbd71f3e5bd6a58e28b73d4`.
- Source base: `d252ddf107534d3611a12b6666e9edf5a70fb7d9`.

```text
run-heavy: done rc=0 325s | peak-RSS 0.51GiB | min-avail 50396MiB | peak-load 1.72
```

Both synthetic FFT controls and the existing strict harness regression pass.
The regression retains incomplete/duplicate refusal, independent 6/61-pair
collection and immutable failed attempts. The original 28 attempts remain
unchanged; these are two additional complete 61-pair experiments, both with
zero exact printed matches. No metric code, public API, fitted correction,
threshold, suppression or author contact changed.

## CI evidence

All six earlier research pushes through `82dd076e` were again verified as
ancestors of `origin/master`. Their main CI runs were cancelled by later
pushes; the last push's dashboard, native benchmark fleet and two guards
passed. This is not a green full-CI claim.

Inherited master `8a2738f6` has [failed lint job 117892362497](https://github.com/imazen/zenmetrics/actions/runs/39200417903/job/117892362497).
Unchanged `-D warnings` rejects `assign_op_pattern` in PIQE kernels
(lines 86/87/185/186), NIQE kernels (91/95/224/246/247), and
`needless_borrow` in NIQE lib.rs (702/703/705). That commit added the kernel
assignments and changed AGGD scratch bindings into references. The borrowed
call text is older, but those binding changes make its extra borrows redundant.
These metric files are outside this research's scope. No metric edit on master
or lint suppression was made; full-tree CI remains an external qualification
blocker for this lane. The actual failed-job log and diff/blame attribution
are retained under the raw evidence directory.
