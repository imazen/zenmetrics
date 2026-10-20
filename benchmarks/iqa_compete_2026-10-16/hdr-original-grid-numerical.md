# Original HDR warm numerical grid

`hdr_original_grid_numerical.py` observes the frozen official default HDR-VDP2
and HDR-VDP3 sRGB-display calls against the unchanged current40 worker. It selects
48 original-grid warm cases: two metrics ×12 RGB8 pairs ×1/28 threads. Each case
runs one default and one native child, alternating order by case index:96 children.
The fixed photo512/1–8 resident callable and all existing bundles remain unchanged.

This source was prepared without executing tests or metric controls. Root must
review the frozen source and admit its tests, setup audit and one small two-child
control before the campaign. The synthetic suite mocks every executor/runtime
capture; its96 simulated children are not metric observations. The retained-row
shape control requires `HDR_NUMERICAL_RETAINED_ROWS` pointing to the frozen four
production stdout/config copies; missing fixtures fail rather than skip. Reading
those old rows does not execute a scorer or add numerical grid coverage.

## Commands and admission

`just bench-iqa-canonical-hdr-numerical-check RETAINED_ROWS` runs the protocol
controls normally and with Python assertions disabled. The explicit argument
names the frozen four-row fixture directory. Invoke it under shared heavy
admission with disk-backed `TMPDIR`; it does not execute a metric.
`just bench-iqa-canonical-hdr-numerical ...` exposes the controller's commands
with the admission rules below.

A frozen private request binds the exact selected48 inventory, headless parent,
passed nine-preflight receipt, current40 build/source receipts, helper and
controller hashes, and admission tools. All hashing, retained receipt audits,
runtime/library resolution and scoring belong inside both existing shared locks
and `run-heavy --mem 12G --jobs 1` or `--jobs 28`. `prepare`, `audit`, `cell`,
`check` and `finish` require that caller admission. Root's frozen launcher supplies
it. `queue` supplies it separately per case, exposes each pending/running result
through stdout plus dispatch command/PID/log/exit files, and must not have an outer
shared lock around the whole panel. Its caller remains the visible supervisor.

Sequence: admit synthetic tests normally and with `-O`; `prepare` a new control
root; `audit` it; `queue --case-id <first-case>`; admitted read-only `check`.
Only after independent review, prepare a separate campaign root, audit it, queue
all48 cases, then admit `finish`. Exact root-local paths/hashes/argv are supplied
in the private frozen bundle, not guessed from an old suffix. Every output path
must be new. There is no automatic retry, overwrite, resume or terminal reuse.

The audit reuses the corrected pinned worker's context/seal validation of every
retained preflight log and snapshot, without rerunning old goldens. Each child
gets Octave and separate native-executable recursive ELF dependency snapshots
before/after; hashing and closure checks remain admitted. Missing evidence,
source/preflight/runtime drift or an incomplete controller stops dispatch.
Ordinary child exit/timeout/protocol/numerical failures retain both role attempts
and allow independent cases to continue. Full-grid finalization refuses missing
cell receipts rather than reporting them as successful skips.

## What the observations mean

Original: five finite `SAMPLE`/`SCORE_HEX` pairs, exact scalar bits within the
child. Raw RGB file loading and setup occur before its clock; RGB8 conversion,
unchanged original call and temporary release occur inside. The default spectral
loader's filesystem I/O, parsing and interpolation remain in that original call.
The official100nit peak/1nit black/30ppd setting uses v2 Q and v3 Q_JOD;
v3 uses quality/CPU/age24/surround-none. The separate200nit XYZ AIC adapter is
preserved. Runtime is Octave6.4/image2.12 with the frozen unchanged image2.20.1
two-M-file resize overlay, not a full image-package upgrade.

Native: existing warm configs declare five samples, but this exact worker
requires ten warm zenbench rounds. The bound grid config must explicitly set
`probe_only=false`; the row may omit that optional field or emit false. Every
other config key must be emitted unchanged. Require mode/only-mode=score,
timing-phase=warm and the unchanged production model adapter disabled; reject
probe records or a true probe flag. Require a complete status-ok10-round row,
retain all raw resource flags, and compare the finite initialization/warmup
scalar recorded in that row. CPU timed return values are not individually
observed; the row's self-match does not prove timed-call stability. This run
supplies no native first-call qualification.

Every raw score, f64 bits, observed difference and process outcome is retained.
v2 keeps absolute0.005/relative0; v3 keeps bound/pass null and native parity
unqualified. Resource flags do not disappear when numerical observations are
complete: none of these outputs qualifies timing. All ratios stay null,
`matched_scope=false`, `qualified_timing_cells=0` because parameter preparation
scopes differ. Separate native warm1/28 bit comparisons preserve disagreements;
they do not discharge the retained exact first/warm thread gate. Canonical/direct,
default/resident ABBA, first-call and timed-return gates remain incomplete or
unobserved with their original text retained. No cold claims, maps, calibration,
production arithmetic changes or human-approved scope exception are introduced.
