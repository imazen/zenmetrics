# Future HDR raw-table-resident control design

Source/design only. No new scorer, fixture execution, build or production edit.
The retained numerical controls remain valid. Under coordinator/root provisional
judgment, with no human-approved new exception, existing base-HDR timings remain
asymmetric descriptive observations, outside strict acceptance; no frozen plan
or result is changed by this design. Base sRGB-display and 200-nit XYZ AIC
conventions remain separate cohorts.

The exact official `load_spectral_resp.m` bound to the base controls has SHA256
`f51b19ae6a874bf5474a0bee367d9564fac84b091b9d343bc7c7a6e72e6ca560`.
Its line 10 obtains a raw numeric table with `dlmread(file_name)`; lines 11–18
construct the wavelength grid, allocate responses and perform column-wise PCHIP
interpolation. The current fixture caches the outputs of this whole function
before clocks (`first_boundary_hdrvdp_resident.py:105–116`), so the current shim
returns already interpolated arrays (`:46–60`).

The exact EA native warm calls still prepare spectra. V2 `metric.rs:241,282`
calls `spectral::emission_spectra` and `lmsr_matrix`; `spectral.rs:68–101,108–145`
parses embedded CSV and interpolates emission/cone/rod curves. V3
`v3/metric.rs:214` calls `emission_columns`; `v3/pathway.rs:227` calls
`receptor_sensitivities`; `v3/spectral.rs:84–97,148–152` parses and PCHIP-resamples.
MTF/JND caches do not cache these loader results. Scalar equality cannot prove
that this parameter preparation occupies the same timing boundary.

## Minimal successor

1. Freeze a new source/config/plan identity. Retain byte-identical original
   metric, observer, emission and `load_spectral_resp.m` bodies and original
   display/task/output settings. Do not reuse the current interpolated-array
   cache or qualify the canceled native probe.
2. Before clocks, call the captured original builtin `dlmread` on only the
   exact-bound spectral files. Store its raw numeric matrices, including the
   original wavelength column, with exact class/shape/IEEE-754 bits. Bind file
   bytes, exact accepted call spellings, helper/runtime identities and matrices.
3. Install a separate fail-closed `dlmread` fixture wrapper for those exact
   one-argument keys. Return the raw numeric matrix using Octave value semantics;
   refuse unknown filenames, alternate argument forms, cache mutation and
   unexpected source resolution. Other original helper bodies stay unchanged.
4. Inside every canonical clock, perform the original RGB8 conversion and
   original metric call. The unchanged `load_spectral_resp.m` now receives the
   raw matrix from the wrapper, then executes its original grid construction,
   allocation and PCHIP interpolation inside the clock. Include temporary release.
   Keep input reads, raw-table loading, fixture setup and audits outside clocks.
5. Root admits new default-versus-raw-table-resident ABBA at both budgets,
   exact raw-table and original-loader output controls, unknown-key/argument
   refusal, post-call immutability, source-resolution checks, complete runtime
   snapshots and actual cell-protocol controls. Preserve existing bounds and
   V3 null bound. New controls cannot inherit current cache receipts.

This removes filesystem loading and retains **original spectral interpolation**
in canonical timing. It also hoists CSV text parsing: native embedded-CSV parsing
still remains timed. Label that residual difference explicitly. This design
alone does not establish a strict match for all parameter preparation. A parser-
resident byte-stream design would require separate original-parser/runtime
controls; it is not implemented or assumed here. Do not replace original parsing
or interpolation with independently written arithmetic.
