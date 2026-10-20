# HDR ABBA controls with admission per arm

These new helpers leave the active resident v3 source, 104-case plan and
preflight outputs unchanged. Preparation creates a separate cohort and visible
queue; it executes no test, inference, preflight or admission command.

`first_boundary_hdrvdp_arm_queue.py prepare PARENT_PLAN PREFLIGHT_RESULT
NEW_FOLDER --heavy-wrapper WRAPPER --lock LOCK1 --lock LOCK2` prepares all104
original cases (416 ABBA arms). The fixed parent contract remains both base
metrics, photo512 plus the original12 grid, first/warm and threads1/28.

Adding `--photo512-threads8` prepares a **separate** four-case/sixteen-arm
supplement: both base metrics, photo512, first/warm, explicit threads8. It
derives the retained one-thread photo identities, byte-copies both reference
scripts, and writes new native configs differing only in `threads=8`.
Request scripts, configurations, case specifications and predecessor identity
are hash-bound. It neither substitutes eight for28 nor changes the original
104 cases. The supplemental controls must pass before the eventual comparative
command can combine established thread-one cases with explicit thread-eight
cases for its1/8 defaults.

## Passed-preflight seal

Run `seal NEW_FOLDER --cohort-sha COHORT_SHA --preflight-sha RESULT_SHA`
only after reviewing the passed nine-preflight receipt. An admitted worker
requires the explicitly supplied result hash, exact original plan/controller
identity, nine explicit passed outcomes, complete array/key markers, matching
nested receipts/log hashes and equal before/after runtime snapshots across
all nine controls. It then hashes every retained preflight artifact into a
new read-only `preflight-seal.json`, with the common normalized runtime/source
closure. Original preflight files are read only. No preflight is rerun.

The receipt's summary alone does not hash its nested runtime snapshots. The
seal binds the retained files present at this verification step; it does not
independently establish their historical authorship. Root's review of the
passed original receipt remains part of admission.

The original104 and thread-eight supplement have distinct cohort/seal hashes,
while reusing the same passed original preflight artifacts. Neither accepts a
null, inferred, failed, partial or different-source preflight receipt.

## Admission and exact ABBA controls

The outer command is:

```
queue NEW_FOLDER --cohort-sha COHORT_SHA --seal-sha SEAL_SHA --case-id EXACT_ID
```

It runs one case's four arms in default/resident/resident/default order. Omit
`--case-id` for the complete selected cohort. Add `--ordinal 0`,1,2 or3 with
an exact case ID to dispatch just that arm. The admitted worker requires and
revalidates all preceding terminal arms before scoring; later-arm selectors
cannot bypass the ABBA order.

**Run the outer coordinator without the shared heavy locks.** It takes only
its own metadata lock. Each dispatch independently acquires the two configured
shared locks and invokes the bound resource wrapper with12G and that case's
explicit1/28/8 worker budget. Source/input/preflight hashes, runtime snapshots,
one original scalar process and post-attempt validation all occur inside that
admission. Locks are released before the next arm is dispatched. Wrapping the
outer queue in those same shared locks would wait on its own inner admission.

Each arm retains the original1100-second evaluator limit and complete scalar
stdout/stderr, commands, memory receipt, recursive `ldd` logs and before/after
runtime snapshots. Source/library closure and non-thread runtime environment
must match the sealed preflight across the complete queue; only explicitly
declared thread budgets vary. Every resident arm still performs its original
untimed bootstrap and post-clock array-bit checks. First/warm arithmetic and
finite, repeat and hexadecimal scalar gates come from the copied hash-bound
v3 helpers, unchanged.

A numeric or original-process failure retains a failed observation and allows
remaining ABBA arms/cases to run. Source, runtime, seal, order or admission
failure stops scheduling. Four observations and identical finite scalar bits
are required for a passed case. Retained arm receipts are never overwritten;
the next admitted arm revalidates its predecessors. There are no automatic
retries or inferred recovery of incomplete attempts. A forced interruption or
resource kill can leave an incomplete attempt, which remains blocked with its
owned dispatch PID/logs preserved. Operator-directed recovery needs a separate
explicit attempt/cohort; it cannot erase the original evidence.

## Visible queue and limits

`queue-catalog.json` is the immutable case/arm inventory. `queue-status.json`
is the latest view, with each arm's state, pass field, receipt path and latest
queue event. Every earlier view is retained under `views/`; events are separate
append-only files under `events/`. Each dispatch records its command, owned PID,
combined wrapper log and exit status under `admission/`. Scalar artifacts live
under `arms/`, and the fourth admitted arm writes the exact ABBA case receipt
under `cases/`. `status NEW_FOLDER` reads the visible view without admission.

All cohorts/results retain `unmatched_scope`, `matched_scope=false` and zero
qualified timing cells. No native metric is executed by this queue. The
explicit thread-eight native configs are preparation artifacts for later
comparative integration. V2's existing external5e-3 bound is unchanged, and no
official v3 photo/grid bound is assigned here. Source preparation is verified;
sealing, queue behavior and actual controls still require admitted execution
and independent review.
