# Prerequisite status after bounded harness acceptance

No acquisition, landing, public push or metric/shared-producer change. Coverage
remains 0/19; the old pilot stays rejected; no ratio or CI is qualified. The
containment and archive-prefix repairs accepted on 1e904dac remain unchanged.

## Real identity-only attempt: rejected

The accepted harness was copied into a new directory and invoked under both
remote admission locks in established order, then run-heavy (12 GiB, one job).
No scorer or model import was invoked. The sealer captured the frozen baseline,
input/model copies, build receipts and a new staged-source inventory. All 22,924
source entries match the retained inventory, with zero additions/removals/changes;
all five staged overlay-file hashes match their preload/map receipts. This is
current identity consistency, not newly proven source-to-binary reproducibility.
The retained executable SHA256 remains
6692a32d5d3d226574b5002068ca062881b1e16f7bc0c06040d1254808c682b0.

Full native dependency closure refuses Triton's libTritonPluginsTestLib.so:
its NEEDED libtriton.so is unresolved by ldd in the bound loader environment.
The installed _C/libtriton.so exists, but its presence does not establish the
plugin's actual load context. No search-path override, skipped root or canonical
environment mutation was used. The attempt remains rejected; canonical.json and
manifest.json were not produced, and full pre/post verification was not reached.
Resolving the loading-context contract needs isolated controls and independent
review before another real seal. Old attempts and this partial attempt remain.
Resource receipt: rc1, 24s, peak RSS 0.05 GiB, minimum available 13812 MiB,
peak load 1.62, 12 GiB hard cap, one job. These are sealing resources.

A separate real import-identity subcheck then captured 12 discovered roots and
8,751 executable caches under both locks/run-heavy. Both immediate pre/post
inventory checks passed without scorer execution. Binding SHA256:
779c1ea9b481add31cde115cbfb94f1dead19e82d9966f1d8b544674cb098a88.
This validates only import inventory stability, not full schema2 identity or
native loading. Resource receipt: rc0, 8s, peak RSS 0.03 GiB, minimum available
13881 MiB, peak load 2.13, 12 GiB cap/jobs1.

## Second paired-NR scalar: synthetic validation only

PairedNoReference::scores already exposes distorted `brisque` and reference
`brisque_ref`; the existing driver discards the second value. IQA_ROW.ref_score
continues to mean its first distorted warmup result and is never relabeled.

`patch_nr_result.py SOURCE NEW_OUTPUT` emits a separate benchmark driver and
`nr_observer.rs`, with an overlay hash receipt. It does not modify its source or
any shared producer. No new metric evaluation is added: the observer extracts
both scalars from the existing returned vector, preserves the headline, rejects
missing/swapped/nonfinite columns and any across-call bit change, and emits
`nr_result_schema=paired-nr-v1`, `nr_scores`, count and stability. The existing
input conversion, result-vector allocation, metric call and destruction remain
inside timing. The extra observer bookkeeping is also timed; it is not claimed
free or proven scope-equivalent. Review its timing impact before acquisition.

A future separately built/sealed cohort must declare native protocol
`iqa-row-nr-v1` and bind its observer/overlay receipt. The accepted historical
`iqa-row` remains parseable for identity/history but lacks the reference result
and now fails block qualification. Both arms must supply finite reference and
distorted scores, each satisfying the same original parity bound. No production
binary was rebuilt or executed here; actual second-result numeric parity remains
pending. A rebuild is a new cohort, never retroactive historical evidence.

Synthetic Rust tests compile the observer alone and the actual adapted make_call
function against an invented CLI returning two distinct scalars. They establish
extraction/routing, not full metric-graph compilation or live metric parity.
Python tests cover new schema validation, reference-only parity failure and refusal
of missing reference output. Existing acceptance assertions remain; valid block
fixtures now explicitly contain both scalar outputs.

## Diagnostic and production-policy review

[Gate trace plan](GATE_TRACE_PLAN.md) records the selected ac675d3 trace limits,
proposed exact-snapshot attribution schema and rejected-only diagnostic scope.
Parser/retention controls pass, but live snapshot instrumentation/correlation is
not implemented or qualified. No diagnostic scorer should run yet.

[Fixed-look policy](FIXED_LOOK_POLICY.md) proposes 120 accepted paired blocks,
60 per order, at most 144 attempted slots, one final look and an exact
order-statistic median interval. It retains the <=3% relative interval and 1.08
crossing rules, with explicit rejection caps, stop rules and order/dependence
assumptions. It needs independent statistical review and implementation before
production; no new observations informed its choices.

The canonical supervisor's provisional correction is incorporated: staged VMAF
caches immutable models and keeps clones timed; live unpatched source is not the
cohort. Current source/feature observations cannot bind historical executables.
CAMBI-AIC2026 luma disagreement remains pending validation; no speculative plane
fix or full audit acceptance is implied. Remaining 18 adapters belong to their
assigned source audits; this work does not duplicate them.

## Retained controls and next gates

43 Python acceptance tests pass, including compiled synthetic Rust routing;
three standalone Rust observer tests pass. Final suite run-heavy: rc0, 21s,
peak RSS 0.16 GiB, minimum available 51353 MiB, peak load 1.13, 2 GiB cap/jobs1.
Standalone Rust: rc0, 0s, peak RSS 0.16 GiB, minimum available 51422 MiB,
peak load 1.36, 2 GiB cap/jobs1. No metric performance claim follows.

Private evidence: /mnt/v/aic2026-evidence-private/clean-panel-prerequisites-2026-10-19/.
Checksums bind logs, rejected real seal diagnostics, source consistency, selected
trace source, staged overlay proposal and immutable diff. Independent review must
assess these sources and policy before routing a further bounded step. Gates:
loading-context resolution + complete real seal/pre/post; new observer binary and
live both-scalar parity; exact-snapshot trace readiness then separately authorized
diagnostic; accepted adapter audits; reviewed/implemented fixed-look inference.
