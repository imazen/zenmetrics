# HeavyProcesses(4) diagnostic readiness

The old attempt has no trace identifying four processes. Clean outer accounting
cannot establish a false positive. No diagnostic scorer is authorized here.

Inspected selected git zenbench ac675d3 source: platform.rs prints foreign
pid/name/cpu_pct when ZENBENCH_GATE_TRACE exists; gate.rs prints clean/reason/state.
The gate stores observation_age_ns internally, but these stderr lines do not bind
snapshot IDs, snapshot time, start ticks, group/session/cgroup or CPU deltas.
Therefore the selected trace alone is NOT sufficient for exact attribution.
Raw source excerpts and hashes belong in the private prerequisite evidence.

Proposed minimal future diagnostic: one native BRISQUE warm attempt only, always
retained as rejected diagnostic (no ratio, clean samples or promotion). Both locks
and existing resource/inner gates stay active. Enable ZENBENCH_GATE_TRACE=1 and
retain complete stdout/stderr, configuration, binary/source/overlay hashes, outer
contention and containment receipts. No canonical timing arm is needed to identify
a native gate refusal. No retries without a new routed decision.

Before that attempt, a separately reviewed benchmark-only zenbench snapshot must
emit each exact monitor snapshot ID and monotonic time plus each gate check's
snapshot ID/time/age. For the processes counted by that exact refresh, retain PID,
start ticks, group/session/cgroup, ownership basis, counted flag, CPU percentage,
cumulative CPU ticks before/after and interval. Join to contemporaneous outer
receipts by monotonic clock and stable identity; missing or reused identities
remain unknown. External polling alone cannot reconstruct an earlier snapshot.
Do not modify shared zenbench or relax the gate. Instrumentation creates a new
binary/cohort and must be hash-bound; it is not suitable as production timing.

The parser retains raw lines and refuses to infer readiness from current traces.
The proposed structured attribution validator checks identity, time/age arithmetic,
counts and artifact hash shapes. Synthetic controls qualify parsing only, not
live instrumentation, actual hash contents or causality. A future consumer must
verify those bound artifact hashes, link every snapshot to the emitted raw trace
and reject missing observations. Instrumentation and that runtime join remain
readiness gates; no claim that the plan is executable as-is follows.
