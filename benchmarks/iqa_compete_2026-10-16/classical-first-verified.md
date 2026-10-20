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
