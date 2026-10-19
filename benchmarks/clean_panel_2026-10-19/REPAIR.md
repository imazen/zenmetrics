# Independent-review repair

The original candidate `bb796cb3` must not land. Its pilot remains rejected and
coverage remains **0/19**. This repair phase runs synthetic and owned-child
controls only; it authorizes no scorer acquisition, production statistics or
public push. Shared producer helpers and metric code are unchanged.

| Review blocker | Repair and acceptance control |
|---|---|
| Committed sealer layout | Resolve `build_baseline.sh` beside the sealer. Seal a tiny fake build/environment from the committed location, without a scorer or Rust build. The baseline identity now hashes the exact saved JSON, without the earlier mutable-list self-reference. |
| Lost exceptional receipts | Start a block receipt before verification; persist each returned arm immediately and raw execution status before parsing. Parse/launch errors, missing files and ordinary interruption finalize a rejected block with exception type, phase, remaining-arm states and artifact hashes. STOP is checked between blocks. |
| Asserted rather than observed workload | Admit only the implemented BRISQUE adapter. Validate frozen config, executable/helper/adapter bindings, emitted metric/shape/mode/phase/thread/input fields and output-column identity. Canonical records VERSION, workload, weights/input hashes and exact sample IDs. Emitted PID/group/executable must match the contained child tree. |
| Surviving timeout descendants | A harness-local Linux subreaper owns the serialized child tree, including reparented or session-escaped descendants. PID/start-time checks and pidfds protect signal identity. TERM then KILL escalation is followed by reaping and an empty-tree check before returning. Tests cover a direct child exiting while its descendant ignores TERM, session escape, interruption and both admission locks remaining held until cleanup. |
| Missing native dependency | All installed environment ELF files, including torch extensions/libraries, become dependency roots. Bind resolved transitive files, symlink targets, RPATH/RUNPATH/NEEDED, explicit loader environment and loader configuration. Re-resolve before/after attempts. Synthetic shadow-path/loader mutations and a real three-library ELF dependency chain reject changed transitive bytes. |

`just clean-panel-check` is the acceptance suite. Its fixtures never invoke a
metric. The owned-child controls use only processes they create. The ELF control
compiles three tiny C shared libraries and never invokes them as scorers.
Run the suite under the shared admission lock and `run-heavy`.

`just clean-panel-before-check HISTORICAL_DIRECTORY` is a separate negative
control. Export `panel.py`, `seal_pilot.py` and `build_baseline.sh` from `bb796cb3`
into their repository layout first. It runs the same acceptance assertions
against the historical modules. Its successful diagnostic result means those
assertions fail on the old implementation, not that the old implementation is
acceptable. The reviewer separately reproduced the inherited timeout survivor;
the new owned-child tests require its termination and reaping.

## Limits retained for re-review

- No new real pilot or trace-enabled gate diagnosis has run. The historical
  `HeavyProcesses(4)` refusal is neither overridden nor called a false positive.
- The canonical adapter now retains both NR results and checks repeated sample
  consistency. Native `IQA_ROW.ref_score` is the first distorted timing result,
  **not** the NR reference result. The native driver does not expose that second
  result here; cross-arm reference-result parity remains unvalidated. The existing
  distorted-score bound is unchanged.
- Containment is Linux-only and requires `/proc`, subreaper support and pidfds.
  It refuses pre-existing controller children. After bounded TERM/KILL escalation,
  an unkillable owned task records `containment-blocked` and retains admission;
  it never returns a terminal success while that task remains. SIGKILL of the
  controller or storage failure cannot guarantee a terminal on-disk receipt.
- The native closure covers installed ELF roots and their resolved dependencies.
  An arbitrary dynamically constructed `dlopen` target outside those roots is
  not proven by static closure alone. The real canonical environment has not been
  resealed in this repair phase; no claim of a newly qualified production runtime
  is made. New real receipts must use schema 2; old receipts are not upgraded.
- Every timing decision remains unqualified. The pilot command permits at most
  four blocks; its fixed-look bootstrap code is descriptive only. Production
  stopping/retry policy, order balance and temporal dependence still require
  independent review. The remaining eighteen adapters remain unqualified.
