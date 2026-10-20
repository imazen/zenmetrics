# Current40 official HDR successor plans

Missing: root source-verification receipt, code review, admitted static tests and
real original-call/native controls. No scorer, compiler or comparison was run by
this lane. All prepared plans have `execution_ready=false` and zero qualified
timing cells. Earlier helpers, plans, observations and 200-nit AIC routes remain
unchanged. The optional native probe is preserved, uncompiled; use the full
current worker for this successor.

## Bound worker and reference bodies

- Source: `ea767ba1a4029963bc56a74fb824e64455e4599e`.
- Binary SHA-256: `19898e546917c9b62badcf1733ec74aacb3e02b4e14d88af1e31923a88a54005`, independently checked against the supplied receipt.
- Supplied receipt SHA-256: `778f3d2b81ca5e903bceeafff3cc98d4c310e5ac955bacec223edd4c2742e277`.
- Build contract: `compete,metrics-parallel,zenmetrics-cli/nr`, no overlays, thin LTO, native-CPU flags unset. The preparer refuses a different contract.
- Production `hdrvdp_aic.rs` SHA-256: `6b790071435a1bcc943f7d3da17b2353f6de434d4a526abb3dfeee4560c31b94`, identical to the previously reviewed 4f89155 body. This limited check does not substitute for root's exhaustive build-source verification.

Official v2 calls `hdrvdp(distorted,reference,'sRGB-display',30,{})` and reports
`Q`. Official v3 calls `hdrvdp3('quality',distorted,reference,'sRGB-display',30,
{'use_gpu',false,'quiet',true,'age',24,'surround','none'})` and reports `Q_JOD`.
Both official v3 encoding switches lowercase their input; the separate Python
oracle's documented case-sensitive behavior does not require changing these
original M-files. Source bodies remain unchanged.

| Prepared panel | Raw pairs | Budgets | Metrics | Phases | Cases |
| --- | ---: | --- | --- | --- | ---: |
| Retained 512×512 S67 photo | 1 | 1, 28 | hdrvdp, hdrvdp3 | first, warm | 8 |
| Original grid: three contents × 64², 255×257, 1024², 3840×2160 | 12 | 1, 28 | hdrvdp, hdrvdp3 | first, warm | 96 |

The photo uses the exact retained `pairA/cid22` RGB hashes from the original
512 plan. The full grid uses all original twelve pair identities and both
original budgets; duplicates, omissions and replacement of 28 with 8 refuse
preparation. Eight-thread cases are not needed for these requested panels. If
the changed-recipe default needs them, prepare a separate successor.

Four plans separate panel and phase. Every case binds its raw inputs, original
reference tree, native config and emitted candidate/direct-control M-files.
Native commands point to the exact current40 worker, not the optional probe.
The supplied receipt and production glue are copied without editing. All new
source/config/script/plan hashes are retained privately in the preparation index.

## Bounds and acceptance

V2 keeps `TOL_Q=5e-3` from `crates/hdrvdp/tests/golden_srgb_display.rs`. Do not
raise it on a discrepancy. V3 has no assigned original-Octave bound for this
photo/grid cohort: its plan uses null and cannot declare native parity. Its
existing `TOL_Q_JOD=1e-9` against the frozen NumPy reference corpus is recorded
separately and remains unchanged; it is not substituted for an absent official
cohort bound. Original/direct adapter controls require exact scalar f64 bits for
both metrics, independently of either native gate.

First entries have one metric invocation and no earlier scoring. Warm entries
have one explicit untimed prime, then five timed calls. Inputs and runtime/package
setup precede clocks; RGB8 /255 conversion, original option/spectral-file I/O,
evaluation and result release remain timed. No manual 200-nit EOTF, ten-bit code
rounding or XYZ matrix is used. Parameter storage and runtime setup remain
explicit scope differences, so ratios stay withheld with `matched_scope=false`.

Declared runtime is the retained Octave 6.4/image 2.12 environment. Entries assert
those versions and record selected helper paths. V2's committed validation
records a discrepancy there against the qualified 11.1/image 2.20.1 golden
runtime. Original/direct wiring controls can compare the same declared runtime;
new native parity remains diagnostic until the runtime is qualified. A newer
runtime needs a new frozen bundle and plans, including its image/shared-library
closure; do not edit the existing scripts or version assertions.

## Code and control order

[first_boundary_hdrvdp_current40.py](first_boundary_hdrvdp_current40.py) only
prepares source/configuration. It launches no native or Octave process. The
entry emitter remains
[first_boundary_hdrvdp_octave.py](first_boundary_hdrvdp_octave.py).

Root executes controls only after code review, exhaustive source verification
and shared heavy admission:

1. Run the existing six HDR entry/refusal static controls plus the six new current40 identity/grid/bound controls. These use invented fixtures, not reference-scoring evidence.
2. On the 512 photo, run each original candidate and independent direct-call entry twice in separate fresh processes, alternating order. Require exact scalar bits across every result. The direct entry uses an independent interleaved-matrix transpose/reshape.
3. Run current40 first and warm configs with the same inputs/budgets. Preserve errors, score bits, original-call outputs and deltas. Require native thread/phase identity; any mismatch is a finding. V2 retains its existing bound; v3 remains unqualified without a bound.
4. Apply the same controls to every original-grid case, including native/reference refusals. Do not skip cases or infer larger-size qualification from the photo control.

The static-check command is:

```sh
python3 -B -m unittest discover -s benchmarks/iqa_compete_2026-10-16 -p 'first_boundary_hdrvdp*checks.py'
```

The caller holds both shared heavy locks, runs through `run-heavy --mem 12G`,
uses disk-backed `TMPDIR`, and retains complete output. No CI or native build is
part of these plans. Root must add the source-verification receipt and complete
runtime closure to a new execution plan before running any command.
