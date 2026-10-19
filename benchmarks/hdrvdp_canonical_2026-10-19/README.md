# HDR-VDP canonical-configuration research

**Neither AIC2026 HDR column has an exact canonical reproduction yet.**
The gate is equality to each published six-decimal score on all 61 crop
pairs. A small residual, rank correlation of one, or a six-pair screen does
not pass. Canonical-versus-CSV and Rust-versus-canonical are separate checks.

- [Data provenance](PROVENANCE.md): original columns and crop bytes.
- [Official versions and written configuration evidence](UPSTREAM.md).
- `scripts/oracle.m`: upstream Octave driver with explicit input conversion.
- `scripts/run_oracle.py`: serialized resource wrapper, full logs, immutable
  driver copy, complete upstream-tree hash manifest and exact argv.
- `scripts/collect.py`: every attempt, including failures and diagnostic
  panels; no fitted score correction. `compare.py` refuses incomplete or
  duplicate full panels.

Commands run from the repository root. Sources and images are caller-supplied;
no upstream implementation or per-image published scores are bundled here.
For the retained paths below, the host is r5900xt. `HDRVDP_STATISTICS_PATH`
points to an extracted Octave statistics package when it is not installed.

```sh
just hdrvdp-canonical-oracle "$ORACLE_TREE" "$PAIRS" "$NEW_OUTPUT" \
  3 200 10 37.842505866 XYZ precise
python3 benchmarks/hdrvdp_canonical_2026-10-19/scripts/compare.py \
  "$PUBLISHED" "$CONFIG" HDR_VDP_3 "$NEW_OUTPUT" q
just hdrvdp-canonical-grid "$GRID_JSON"
just hdrvdp-canonical-harness-check "$SCRATCH" "$PUBLISHED" "$CONFIG"
```

`oracle.m` positional arguments after the three paths are version (2/3),
peak, code-rounding bits (0 disables), ppd, encoding, matrix/conversion,
then optional upstream `key=value` arguments. Fixed task is quality; v3
uses CPU and age24. Defaults are taken from the hashed release. `precise`
uses the seven-decimal sRGB D65 matrix; `iec` uses four decimals; `octave`
uses the installed `rgb2xyz`; `single` evaluates the transfer in single
precision before returning to double; `display` passes normalized codes to
the release's own display encoding; `gog` uses a 1000:1, gamma2.2,
zero-ambient display model. The display encoding owns its luminance, so
its peak argument must describe that fixed helper rather than rescale it.
No fitted parameter or residual correction is used.

`prepare_v2_octave.py` copies a pristine release into a new tree, removes
only the unsupported `ifft2(...,'symmetric')` arguments, adds a `matrix.h`
compatibility include, and builds upstream corrDn/upConv/pointOp with the
matching extracted `mkoctfile`. `prepare_v306.py` removes the same FFT
arguments, enables the return-value reconstruction fallback, and relocates
the unchanged abstract class to its Octave-required @-folder. These trees
are explicitly adapted Octave oracles. They do not establish MATLAB
floating-point identity. Failed setup attempts remain in the raw evidence.

Every scratch output path must be new. The grid file is a JSON list of
argument arrays for the same oracle command. A failed attempt stops that
grid rather than silently skipping a configuration. Metric code changes,
if needed after identification, belong on an unmerged fix branch.

## Evidence

The original 61-pair controls and diagnostic screens are retained with raw
logs and source manifests on r5900xt under `~/tmp/hdrvdpcx/`. No fitted
correction was applied. The full-pair controls establish the residuals in
upstream; sampled configuration exclusions do not identify the dataset
invocation. The port remains unchanged.

[HDR2 trials](hdrvdp2_trials.tsv) and [HDR3 trials](hdrvdp3_trials.tsv)
record all 28 upstream attempts: 25 scoring executions completed (three
61-pair controls and 22 six-pair screens); three setup executions failed
and remain visible. Every completed trial has **zero** printed matches on
its panel. The 203-nit CTC lead, zero/mean surround, old peak sensitivity,
MTF off, CRT/LED spectra, luminance ingress, fixed sRGB display helper,
GOG display helper, code rounding, ppd precision, alternate colour matrices,
single-precision transfer and the 3.0.6 version screen do not reproduce the
column under the recorded calls. These are exclusions of those calls,
not an exhaustive proof about every possible combination.

| complete upstream control versus CSV | n | median raw Q error | max raw Q error | median site-JND error | max site-JND error |
|---|---:|---:|---:|---:|---:|
| HDR2 2.2.2, compiled pyramid, Octave FFT adaptation | 61 | 0.026509192464118314 | 0.2199231820780625 | 0.003953365176822543 | 0.04921545159343488 |
| HDR3 3.0.7, closest retained configuration | 61 | 0.0013692098323083712 | 0.012360806855934214 | 0.0013356867286653529 | 0.012807267024098623 |

The [separate consistency table](controls.tsv) shows the HDR2 MEX/fallback
difference is at most 1.4921397450962104e-13 Q across all 61 pairs; it does
not explain the HDR2 column residual. Native HDR2 versus that upstream
Octave control still differs by median 0.0086901909133843 Q and
maximum 0.014212375528188659 Q. Native HDR3 versus upstream differs by at
most 4.973799150320701e-14 Q, and its fresh replay matches the retained
native Q values exactly. The 3.0.6 version screen differs from 3.0.7 by at
most 1.2168044349891716e-13 Q on six pairs.

[Residual strata](strata.tsv) retain category, sRGB-chunk and codec groups
with their actual sample sizes. [Luminance correlations](strata-correlations.tsv)
are exploratory and do not identify a cause. All images are 840×944, so
this panel cannot identify a size dependence. Per-pair diagnostics stay in
`~/tmp/hdrvdpcx/residual-private.json` on r5900xt.

[Qualification metadata](qualification.meta) records input, source, driver,
runtime and raw-evidence identities plus resource receipts. Reproduce the
control table with `scripts/summarize_controls.py SCRATCH OUTPUT_TSV`.
Runtime logs contain setup warnings and full output; no failed attempt was
converted into a score. MATLAB itself and the authors' executable HDR-VDP
invocations remain missing. No metric change was made and no fix branch was opened; the dataset
configuration remains unidentified.

To recreate the isolated native replay workspace, use
`just hdrvdp-canonical-prepare-rust NEW_DIRECTORY`, then
`just hdrvdp-canonical-build-rust NEW_DIRECTORY/Cargo.toml`. The preparation
checks every copied runtime source hash. It leaves the shipped Cargo.lock
and manifests untouched and omits unused test dependencies only in the
scratch manifest. The saved replay binary is
`~/tmp/hdrvdpcx/bin/vdpprobe` on r5900xt; its hash is in the metadata.
The oracle wrapper requires this lane's claimed `.workongoing` file.
