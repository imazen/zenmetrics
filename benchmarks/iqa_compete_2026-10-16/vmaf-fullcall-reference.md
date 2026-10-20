# VMAF complete-model-lifetime reference preparation

The frozen source fixture passed independent review and its admitted C++ link
and [twelve exact-score controls](../canonical_recheck_2026-10-19/vmaf-fullcall-controls.json).
All 72 old/new scalar observations match in binary64. This qualifies the
reference fixture; native comparisons and timing acceptance remain separate.
Existing reference/native sources, binaries, plans, results and bounds are retained.

[libvmaf_rgb_fullcall.cpp](libvmaf_rgb_fullcall.cpp) is an additive successor of
`libvmaf_rgb_compare.cpp`. Its only scoring-body change moves the original
embedded `vmaf_model_load` and `vmaf_model_destroy` into every COLD/SAMPLE clock
for model routes. RGB conversion, model names/configuration, context/picture
creation, single-frame scoring, close and temporary vector release retain the
old body and order. Raw file reads, argument/environment setup and stdout remain
outside. No model scoring precedes COLD. The old nine-entry argv form is accepted
(program plus metric, width, height, reference, distorted, threads, samples and
convention). The explicit SCOPE header identifies the successor.

The model-lifetime boundary now includes embedded model parsing/allocation and
release per call. Native and author internal model representations/parsers are
different; this does not assert identical model-preparation work, full timing
qualification, native parity or an externally assigned bound for unbounded
routes. Non-model routes remain accepted by the inherited adapter but are not
part of this control fixture.

## Exact control contract

The new [stdlib-only driver](vmaf_fullcall_oracle.py) links only the new C++
wrapper to the retained unchanged libvmaf archive. It invokes no Cargo, installer
or model download. Compile flags match the retained older wrapper build:
`-O3 -Wall -Wextra -ffp-contract=off -std=c++17`, with frozen source/build header
copies, archive, `-pthread -lm`. Header and compiler backend/runtime identities
are captured alongside compile/version/query commands and complete logs. The
archive's own historical compiler settings are not changed or reconstructed.

Twelve comparisons use the original `vmaf-plan.json` photo512 case rows:
`vmaf`, `vmaf-neg`, `vmaf-4k`, `vmaf-v1`, `vmaf-studio601` and
`vmaf-neg-studio601`, each at threads1/8. The original row/argv/geometry/RGB
convention, raw input paths/hashes and numerical-bound metadata are retained in
the frozen cases. The driver validates those cases against the unchanged plan;
it does not apply or assign native/canonical numerical bounds.

Each old/new child is fresh and runs one COLD plus two SAMPLE observations:
24 children,72 scalar observations. Require successful process exit, exact
VERSION/SCOPE and record-count/order protocol, positive integer durations, finite
binary64 scores and identical f64 bits for all six observations within each
old/new comparison. No cross-budget identity requirement is added. A failure
retains its raw logs and fails the case; score/protocol failure in one arm still
allows the other arm and remaining independent cases to execute. Identity drift
also fails. The driver computes no speed ratios or timing-acceptance decision.

Both original and successor receive explicit environment budgets before exec;
the unchanged libvmaf context still receives its original `cfg.n_threads` from
argv. This is configured thread control, not measured utilization. Per-child
timeouts terminate only their own newly created process group. Existing and
failed build/control directories are never overwritten or resumed.

## Freeze and admission

The immutable private bundle includes the new C++/driver/docs, exact old C++,
source/build public headers, cases and a SHA256 manifest. Its externally supplied
freeze digest binds original archive/old executable/plan/build metadata/input
identities and compiler/inspection tools. No compiler, ldd or library/scorer runs
while preparing that bundle. All runtime inspection belongs to the later
admitted driver execution.

The driver requires `--admitted`: this is the caller's explicit attestation that
both existing shared locks and run-heavy12G admission are held. It is not a
kernel proof of lock ownership or memory containment. Root's frozen launcher
and retained resource-wrapper log supply the admission evidence. Compilation,
tool queries, recursive ldd closure capture, asset hashing and24 scalar children
are sequential under that admission. Inputs/source/archive/tools are checked
before/after. Build and controls each retain recursive runtime snapshots and
refuse drift. The controls verify that their runtime closure equals the build's
retained post-link closure. Source/receipt/binary digests link both phases.

The interface is:

```text
python3 -B vmaf_fullcall_oracle.py --admitted
  --freeze /absolute/new-bundle/freeze.json --freeze-sha EXACT_SHA256
  --build-output /absolute/new-build --controls-output /absolute/new-controls
```

The exact root-admitted command, frozen source manifest and missing/verified
status live in `~/tmp/handoff/vmaf-fullcall-oracle-20261020/report.md`. Root owns
review, actual execution, publication and workspace cleanup; this lane does not
push. No active helper/plan/result is edited. A passing scalar control would
qualify only this fixture's old/new exact scores; native speed, full model-scope
acceptance and runtime cleanliness require their own existing contracts.
