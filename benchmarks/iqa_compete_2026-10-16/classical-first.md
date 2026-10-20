# Classical evaluator first calls

Run the comparative protocol with
`just bench-iqa-canonical-classical-first PLAN NEW_OUTPUT --blocks 3`.
For resident-model NN first calls, the separate
`just bench-iqa-canonical-first-confirm PLAN CASE NEW_OUTPUT --blocks 20`
command acquires a fixed alternating stage and bootstraps the median ratio.
It retains every block and refuses an interval on an execution, score or
numerical-gate failure. Its interval remains descriptive; it does not certify
runtime isolation or replace the warm measurements.

NN first-call confirmations require an explicit finite nonnegative absolute
bound and a passing gate in every block. The strict controller records its
numerical dependencies and bootstrap method. The earlier controller remains
available for already-running bounded stages; its bytes are preserved.

`compare_classical_first.py PLAN NEW_OUTPUT --blocks 3` acquires one first
evaluation per fresh process, alternating canonical/native order. It uses
the frozen native worker's `timing_phase=cold`, `only_mode=score` path.
Raw RGB reads and process launch are outside both evaluator clocks.
Native scorer construction is inside; scorer destruction after elapsed
capture is outside. This is not process startup, OS-cache-cold performance,
or the resident-model NN protocol.

The output must be a new directory under the caller's `~/tmp`. Each block
holds both existing shared heavy locks and runs through the 12-GiB wrapper.
Lock waiting has no fixed timeout and is outside the benchmark timer:
shared queues may legitimately exceed five minutes. Admitted blocks and
individual arms retain deadlines. Input,
source-plan, review, helper, interpreter and worker bindings are checked
inside admission before and after execution. Every arm retains argv, exit
status, stdout, stderr and GNU time memory observations. Native worker
metadata is retained without interpreting resource flags as qualification.
Wrapper logs retain the resource-wrapper observations.

Plans name explicit source-plan cases, preserve every original field and
bound, and add binary/source identities plus `first_call_scope` declarations
for construction, parameter I/O, runtime setup and teardown. The original
plan is copied byte-for-byte into output. Source-plan bindings must be
present unchanged. Scope admission additionally uses the hash-bound
84-route review. The plan phase must equal `scorer_construct_first_eval`;
missing or different phases refuse before execution. NN and GPU names are
not accepted.

Only reviewed `first_evaluation_candidate` routes with an explicit matched
scope can execute. NIQE's existing COLD follows two untimed pair calls and
is refused. DSSIM preconstructs its canonical context outside the timer;
native cold includes context construction. The six model VMAF routes and
other hoisted constructors similarly need separate boundary work.
The two AIC2026 HDR-VDP aliases are included SDR routes with unresolved cold
parameter-I/O scope; they are distinct from the base typed-input names.
No frozen adapter or numerical expectation is changed by this tool.

The canonical command receives `samples=1`, satisfying references that
require a positive sample count. It emits one COLD observation followed by
a SAMPLE call. That later call cannot contaminate the earlier first-call
observation: all stdout is retained, but only COLD enters this comparison.
Exactly one finite `COLD ns score` is required, with positive elapsed time.
Native output must contain exactly one successful score
`IQA_ROW` matching name, geometry and thread count, with cold phase, `n=1`,
positive finite `cold_elapsed_ns` and finite score. Missing, malformed,
duplicate and refusal observations cannot become timing ratios.

The summary is the median native/canonical ratio over all requested
alternating blocks. Missing blocks, errors, unmatched scope or any failed
numerical gate withhold it. An absent absolute bound records a delta and
`passed=null`; it neither invents a parity gate nor reports a failure.
All ratios are descriptive, with no confidence interval or regression
acceptance. Historical warm windows are never paired with these calls.

The initial private control plan selects FLIP and RGB PSNR, 512×512,
one thread, using their explicit revised source plans. FLIP constructs
its error-map evaluation inside the clock and returns the declared f64
mean. PSNR has no scorer object or external parameters on either side;
RGB ingress and the scalar callable are timed. Their existing null bounds
remain null. Python imports/runtime configuration are outside the clock;
native first-use runtime work inside the evaluator remains included.
Thus these are matched declared evaluator calls, not matched interpreter
or thread-runtime startup costs. Author DSSIM is withheld for the concrete
constructor mismatch above; changing its source lineage does not move its
clock.

Run synthetic controls with the project's Python test runner under shared
heavy admission:

```sh
python3 -B -m unittest discover -s benchmarks/iqa_compete_2026-10-16 \
  -p test_classical_first.py -v
```

Set `TMPDIR` to a new disk-backed directory under `~/tmp`, hold both shared
heavy locks, and invoke this command through `run-heavy --mem 12G --jobs 1`.
Tests execute only tiny invented protocol workers. Real FLIP/PSNR controls
and any performance conclusions require separate coordinator execution.
