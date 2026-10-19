# Changed-piece monitor repair: owned nonmetric scope only

The prior snapshot/count/cached-origin result remains bounded-accepted. This
repair does not qualify general attribution, a scorer, or acquisition. Generic
acceptance remains 59/60 RED; diagnostic proposal 9d69f271 is still UNAPPLIED.
The existing master BRISQUE change b52fadb6 is an ancestor of baseline73c04;
oldopt is a different implementation. No metric behavior is remeasured here.

## Controller lifetime and actual record joins

The validator joins controller PID/start against the registered owner entries
in actual monitor snapshots and against held, launch and outer records. Child
and rival parent PIDs must both equal that controller PID. The separate held and
launch JSON files are hash-bound and must equal their copies in the outer
receipt. Held/launch/post-cleanup lock identities and token must agree.

Rehashed actual-file negatives separately change controller start, child start,
rival parent, held owner and chronology. Both historical actual runs also pass
the stronger identity join using derived receipts which name their original
held/launch records. Those derived receipts do not overwrite the original bytes.
Additional controls consistently change all claimed controller starts, leaving
the independent snapshot contradictory; the snapshot join rejects them.

## One uncertain clock offset

`monitor_join.common_offset` intersects the launch/READY origin bound with the
GO/exit admissible interval for every retained sample/counter-window start,
publication, check, gate entry/hook and read timestamp. It includes refreshes
not used by a fresh check. Held/launch receipt ordering, parent chronology,
snapshot/check ordering and nonoverlapping serial gate events are checked.
No nonempty intersection is presented as exact clock equality.

The review counterexample has launch/READY [100,200], GO/exit [300,400], and
check times100/300. Both old pairwise predicates pass; the common intersection
is empty and refuses. A feasible boundary case passes the same schema and actual
byte/event joins. Both historical actual receipt intervals agree with the
reviewer's independently derived intervals.

## Publication age, decisions, missing and stale paths

The isolated full-pin ac675d3a3f5334f4be6749a61b37995c9f3d34fb overlay now also
instruments monitor publication and cached reads. Publication records the exact
Instant stored with the SystemState. The read records the exact observed Instant
and duration used by the unchanged one-second stale predicate. ResourceGate
emits its stored observation_age_ns, actual clean/reason/state and selected
snapshot, separately from hook emission and refresh sample time. The old
check.observation_age_ns is explicitly labeled hook-minus-refresh age; it is not
used as evidence of the gate's publication age.

The actual nonmetric lib-test performs40 fresh check-and-record calls, stops and
joins its owned sampler, waits1100ms for normal stale handling, then explicitly
clears the stopped cache through a test-only helper to exercise the real missing
branch. It does not rewrite timestamps, force refreshes for reporting, alter
thresholds, or reclassify the decision to create evidence. Missing/stale records
have unknown decision/status, no fabricated clean result, and no fresh check.
The final publication is retained; it is unused by fresh checks but used by the
stale read. This distinction explains the generic validator's unconsumed list.

Changed/rehashed gate age, read age, publication, clean/reason, heavy count,
missing/stale status and gate ordering are rejected. No assertion or threshold
in the existing generic suite changed, and that suite was not rerun.

## Bounds and remaining gates

The sampler and gate arithmetic remain pinned, but logging changes execution
load/timing. Proc counters support identity and bracketing only: their recorded
interval is neither sysinfo's internal denominator nor an exact duration for
each per-process counter read. Other gate APIs, disabled-gate behavior and
poisoned-cache locking are not qualified by these controls.

Two actual panel admission paths are absent on this host. The explicit-lock
control refuses missing files before launching a child, retains the failure,
and never creates/replaces/truncates/removes admission locks. Fixture-lock
controls run under the shared heavy lock and run-heavy but do not establish
production admission identity. Actual production-lock proof remains a gate.

Compiled source, lockfile, binary, build log, config, raw events, controller and
helper source/bytecode hashes, interpreter executable, outer and ownership
records are retained. These are a diagnostic receipt, not complete Python loader
closure or compiler reproducibility qualification. `attribution_ready` stays
false pending independent changed-piece review and remaining cohort gates.
Canonical PIQ, native scope/norm/config/closure, observer cohort and statistical
selection model remain separate. No historical refusal explanation follows.
The separately owned libjxl adapter is untouched; no matched-timing claim.

Private receipts (r5900xt):
`/mnt/v/aic2026-evidence-private/clean-panel-monitor-repair-2026-10-19/`.
The SHA256SUMS manifest binds every retained attempt, final source/export/build,
binary and focused join controls. Old failed/generic receipts remain intact.
