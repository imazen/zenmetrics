# Official HDR resident-parameter benchmark fixture

This implements the separate fixture proposed in
[the parameter I/O source review](first-boundary-hdr-parameter-io.md).
Preparation runs no Octave, inference, tests, compiler or benchmark. It copies
the frozen original source trees and verifies each file against its retained
SHA256. Existing helpers, plans, binaries and golden results remain unchanged.

`first_boundary_hdrvdp_resident.py PARENT_CURRENT40_FOLDER RESIZE_SOURCE_FOLDER
NEW_OUTPUT_FOLDER` emits two distinct runtime plans: installed image 2.12, and
installed image 2.12 with the unchanged image 2.20 `imresize.m` and private
helper. Both use Octave 6.4. The installed package version and selected resize
path are asserted separately. Neither preparation nor original/resident
identity controls qualify the runtime against external goldens.

## Adapter boundary and unchanged arithmetic

The original APIs consume spectral filenames, with unconditional observer
loads. The new adapter calls the **original** `load_spectral_resp` with each
exact-bound physical file before any evaluator invocation. It saves both
original outputs, including the original wavelength grid, interpolation,
shape and class. It then installs a separate read-only path shim and seals
the cache. V2 has three exact bare request keys. V3 has three exact absolute
keys and the bare `ciexyz31.csv` request from `hdrvdp_iturgb2native`.

The shim has no filesystem access, miss delegation, lazy fill or path
normalization. Unknown keys fail. The cache owner permits one installation
and reads only; replacement and reset fail. Returned arrays use Octave value
semantics, which the independent controls must verify. There are no counters
or profiler instrumentation in the metric or its timed body.

All original metric/observer/helper M-files remain byte-identical. The
preparer also checks the literal evaluator body between `tic` and `toc`
against the retained adapter, including its occurrence in the warmup. RGB8
conversion, original top-level evaluation and result release retain their
boundary. Cone exponentiation, rod insertion, spectral integration and the
v3 display-primary matrix solve remain inside the original call. Parameter
I/O and interpolation move outside first/warm clocks through original-helper
preloading; import, raw input read and source/runtime hashing also precede
clocks. A first process has no earlier metric evaluation. A warm process has
the retained one untimed warmup and five timed evaluations.

V2 keeps its CCFL default and Q output. V3 keeps LED-LCD-sRGB, quality,
30 ppd, CPU, age 24, surround none and Q_JOD. The AIC 200-nit XYZ convention
is separate and unchanged. Cache lookups remain part of evaluator timing;
this fixture does not claim identical reference/native setup costs.

## Controls and execution protocol

Each runtime plan contains nine preflight processes:

- Two version-specific array controls compare fresh original-helper outputs
  against cache results by class, shape and every `num2hex` bit. They modify
  returned arrays and require unchanged subsequent results, and require
  replacement, reset and unknown-key refusals.
- Seven independent missing-key controls omit each of the three v2/four v3
  entries in turn. The actual original metric call must fail with that exact
  cache-miss identifier and request. A path-resolution bypass cannot pass by
  completing the score. These diagnostic calls are not timing observations.

Each plan retains all104 score cases: the 512 photo and full original
12-pair grid, both metrics, first/warm, threads 1/28. The separate
`first_boundary_hdrvdp_resident_run.py PLAN NEW_RESULTS` runs preflight first,
then default/resident/resident/default in four fresh processes for every
selected case. Every requested warm sample must be finite, repeat-identical,
and bit-identical to the other arm. Scalar `%.17g` values must also agree
with independently emitted `num2hex` strings. Post-evaluation cache-array
checks run after the measurement clock.

Use `--preflight-only` for the nine prerequisite controls. `--panel photo512`
selects 8 of104 score cases, `--panel original12` selects96, and the default
`--panel all` selects104. The result reports the selected coverage; partial
controls never imply full-panel acceptance. A failed preflight refuses score
controls. A scorer exit or numerical mismatch retains the attempt and allows
the remaining arms/cases to execute. Source/closure drift refuses further
execution.

The coordinator wraps the complete runner in existing shared heavy admission,
with disk TMPDIR, complete driver stdout/stderr and a new exclusive output
directory. Full 1/28-thread panels need a wrapper budget that permits28;
preflight-only uses one. The controller does not acquire admission itself.
Per-process timeouts remain1100 seconds. The copied reviewed golden-driver
utilities retain complete process stdout/stderr, time-v memory receipts and
recursive `ldd` logs. Installed Octave/image M/OCT, parameter/helper source and
library-closure snapshots are compared before/after every process; normalized
source/library closure must also stay identical across the run. No native
metric is executed by this controller.

All emitted plans and control results remain `unmatched_scope`,
`matched_scope=false`, with zero qualified timing cells. Passing exact
default/resident controls establishes this adapter substitution only. Root
still reviews runtime golden evidence, exhaustive native source provenance,
native numerical controls and remaining scope before using timing ratios.
The existing v2 external `5e-3` bound is unchanged; this official v3 photo/grid
cohort still has no assigned external bound.
