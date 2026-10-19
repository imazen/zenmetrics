# Isolated monitor attribution control

Historical initial control; the changed-piece repair and current limits are in
[MONITOR_REPAIR.md](MONITOR_REPAIR.md).

## Existing BRISQUE implementation distinction

Commit b52fadb6e996e89b3d28fa1bcb45c2070afcf9cd changes existing BRISQUE code,
including reductions. Local ancestry inspection confirms it is an ancestor of
73c04e41, the master-derived baseline. Oldopt ef9763f2 is not present in this
local object store; its distinction is owner-supplied and must not be erased.
This control does not build or measure BRISQUE, establish authorization for that
historical landing, or infer identical scores/timings across those cohorts.

## Scope and implementation

`stage_monitor_overlay.py` exports pinned zenbench ac675d3 into a NEW directory,
checks unique source anchors, and hashes the original and patched files. It adds
`monitor_overlay.rs` to that copy only. The actual SystemMonitor refresh retains
an attribution value inside its returned SystemState; actual cached
ResourceGate::check_and_record_status calls emit checks referring to that value.
The original CPU/load/count/temperature/RAM calculation and gates are unchanged.
The source copy is compiled as a lib-test executable with no metric dependencies.
The target invokes only the owned nonmetric monitor test, not benchmark scoring.

The diagnostic hook records the exact counted PID set and registered owner set
from the classification's sysinfo refresh, with Linux start/group/session/cgroup
identity and cumulative counters read around that refresh. Those proc counters
bracket the refresh; they are NOT the internal sysinfo CPU-percentage denominator.
Missing/reused identities are explicit errors and fail the control. Other public
gate methods and missing/stale background observations are not covered by this
instrumentation. This is a bounded check-and-record path control, not complete
production instrumentation.

The clock is the child's monotonic Instant origin. Parent CLOCK_MONOTONIC launch
and READY observations bound its offset. Cross-process correlation is an interval,
not an asserted exact timestamp equivalence. The interval includes process
startup/handshake latency; no timing or overhead claim follows.

## Owned process and artifact joins

`monitor_control.py` holds two owned fixture flocks, records kernel lock proof,
and binds the direct monitor child through PID/start identity and exact argv.
The enclosing invocation uses run-heavy and the shared heavy lock; the two
fixture locks are explicitly NOT proof of production admission-lock ownership.
A controller-owned busy sibling is intentionally absent from the monitor's
registered owner list, so it exercises actual foreign-heavy classification.
Cleanup signals only direct Popen-owned unreaped children, then reaps them before
releasing fixture locks. No shared process is signalled.

The controller checks source files against the manifest before and after running,
compares the actual /proc/child/exe bytes to the compiled binary receipt, and
binds config, controller, source, raw event stream and outer receipt hashes.
It requires cached snapshot reuse, counted rival identity and excluded monitor
identity. Negative controls reject wrong age, counted identity, executable hash
and outer start identity using the same validators as the positive path.

The retained runtime-r3 receipt is the final build-bound control; earlier
runtime-r1/r2 evidence remains historical and is not relabeled. All controls
remain diagnostic; `attribution_ready` stays false. A final unused refresh is
retained rather than dropped.

## Remaining gates

Independent overlay/runtime review, complete missing/stale-check instrumentation,
reviewed cross-process clock semantics and any future real cohort integration
remain required. No old heavy-process refusal is reconstructed or attributed.
The canonical PIQ oracle, native ingress/norm/closure, statistical selection model
and production acquisition remain unqualified. The full generic suite is still
red at the newly reached tied-median diagnostic expectation; its proposed repair
is separate and unapplied pending explicit owner approval. Only the earlier
schedule expectation was owner-approved and changed.

Private evidence (observed on r5900xt):
`/mnt/v/aic2026-evidence-private/clean-panel-monitor-2026-10-19/`.
`SHA256SUMS` binds source export, overlay files, build logs/receipt, executable,
all three runtime attempts and the approved-fixture failure/observer receipts.
The original failed builds and earlier generic failing receipts are preserved.
