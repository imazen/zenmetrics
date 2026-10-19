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
