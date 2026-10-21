# Classical first-call verification metadata

`compare_classical_first_verified.py PLAN NEW_OUTPUT --blocks 3` is an
additive successor to `compare_classical_first.py`. The original remains
unchanged for retained plans and active acquisitions. Metric calls, clocks,
protocols, numerical bounds, admission, deadlines and summary gates are copied
without changes. No performance magnitude is established by this source change.

Each `verify()` invocation parses each distinct source-plan path once and
indexes its cases. It checks each source's binding map once against the plan's
binding map. Every invocation starts with empty metadata caches; there is no
mtime cache or cached hash. Every original bound file is still content-hashed
on every invocation, in the original order. Required-file checks remain per
case. A successful two-arm block still verifies initially, before and after
each arm, and finally: six complete invocations.

The successor requires its own file hash in addition to all old helper
bindings. Duplicate source-case IDs and malformed unused source cases are
rejected during indexing, where the old linear lookup could accept a matching
entry before the malformed tail. Other parsing behavior is retained. This
adds explicit refusal; it does not accept a formerly refused contract.

Run `test_classical_first_verified.py` through the project's Python test runner
under shared resource admission and explicit disk-backed `TMPDIR`. Controls
compare old/new validator outcomes and hash traces on inert file fixtures,
check one parse/map comparison per source per invocation, mutate same-mtime
source contents between invocations, and reject post-arm tampering before
the next arm. Block tests mock arm execution; no scorer subprocess is used.
Run normal and optimized Python, then separately admit read-only verification
of a real large-binding plan. Source preparation alone is not a test result.

A successor request must retain every old case field, bound/null, input,
source plan, review, native binary/source identity and binding. Add this
controller's hash, retain the old controller hash, and use the new controller
for the outer command and admitted block children. Freeze a NEW request/output
with the prior request/plan hashes bound; do not edit existing requests or
replace a selected large case with a smaller fixture. Metadata caching does
not authorize narrowing bindings or reducing any verification frequency.

Use `just bench-iqa-canonical-classical-first-verified-check` under shared heavy
admission to run the inert controls normally and with assertions disabled.
`just bench-iqa-canonical-classical-first-verified PLAN NEW_OUTPUT` runs a
prepared successor plan; each block admits itself, so give the coordinator no
outer heavy lock. The plan must bind the successor as well as every original
helper and source artifact.

To surface completed cold-call flags while a campaign continues, run
`just bench-iqa-canonical-classical-first-flags NEW_SNAPSHOT REGRESSION_LOG
--root RESULTS_ROOT` (repeat `--root` for separately frozen campaigns). This
reads constructor/first-evaluation summaries, records every matched ratio
above 1.08 and retains the case geometry, worker budget and original bounds.
It acquires no measurements and assigns no missing numerical bound. Every
capture uses a new output file; partially written summaries remain pending.
The flags require fixed-stage confirmation and profiling before interpretation;
the capture does not qualify runtime cleanliness or confidence intervals.

After a complete fixed confirmation stage, use
`just bench-iqa-canonical-classical-first-interval RESULTS NEW_OUTPUT` under
shared heavy admission. An additional fixed stage uses `--additional RESULTS`;
it must link to the initial confirmation plan and retain its exact cases and
bindings. The analysis retains every block, verifies order and numerical gate
records, and needs at least twenty blocks. A missing bound produces an explicit
unqualified numerical result. These intervals describe acquired first-call
variability and never accept a regression or establish runtime cleanliness.

`just bench-iqa-canonical-classical-first-profile PLAN CASE NEW_OUTPUT` captures
CPU stacks from the unchanged worker in fresh processes. Supply shared heavy
admission, disk-backed `TMPDIR`, and explicit `--repeats`/`--frequency` when
profiling short calls. The recording includes process startup, input loading
and teardown; sampled percentages do not isolate the internal cold clock.
Instrumented wall times must stay outside comparison results.

If capture succeeded but postprocessing failed, `--adopt CAPTURE_DIRECTORY`
validates the retained request, configuration and complete cold observations
before processing the existing data. It runs no additional scorer. Original
files remain intact. Per-task recordings may lack a CPU field; the helper does
not infer CPU placement from them. Recordings are capped at 32 MiB, with a
1500 MiB free-disk guard before each invocation.

`just bench-iqa-canonical-classical-first-profiles-publish INDEX NEW_DIRECTORY`
exports completed fresh-worker recordings, including adopted captures. It checks
plan/artifact hashes and revalidates every retained worker scalar before writing
bounded JSON projections. The index gives each capture/adoption resource log its
own role; an earlier failed postprocessor remains visible after successful
adoption. This exporter keeps whole-process cycle samples separate from warm
profiles and comparative timing evidence. It does not isolate the internal cold
clock, identify worker-thread spans or qualify a regression.

`just bench-iqa-canonical-classical-first-profile-adopt-capped PLAN CASE CAPTURE NEW_DIRECTORY`
retains a recording stopped by perf's explicit size cap. It revalidates every
completed worker, records a possible interrupted final worker separately and
keeps requested/completed counts distinct. It executes no scorer and never
promotes an incomplete capture to benchmark evidence. This partial diagnostic
has its own receipt schema and remains separate from complete-profile exports.

Completed campaign aggregates are checked against the declared case set and
each retained case summary. Capture counts each case once and rejects altered
aggregate records. Run `just bench-iqa-canonical-classical-first-flags-check`
under shared heavy admission for the inert metadata controls in normal and
optimized Python. These controls execute no scorers or threshold changes.
