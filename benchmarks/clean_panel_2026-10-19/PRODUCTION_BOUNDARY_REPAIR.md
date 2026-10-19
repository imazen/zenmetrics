# Launch-to-supervision guard placement

The enclosing cleanup guard is now active before Popen. Retry state is
initialized before launch; the returned handle and retained observations are
passed to the unchanged supervision operations. A finally block re-enters
supervision with that same handle/state until cleanup is proven. It does not
launch again and does not unlock merely because an exception occurred.
Release happens only after the positive cleanup proof, including when the
original interruption is subsequently propagated.

Three owned sys.settrace controls interrupt the first line after successful
Popen return and the adjacent retry-loop/header entry. They wait for the exact
finite child's identity before injection, then require both lock acquisitions
to refuse and retain exact kernel fdinfo while the child remains alive. All
three wrappers retain admission, reap after natural completion, persist
cleanup and reverse release, then propagate KeyboardInterrupt (rc -2).
The outer regression harness passes; rc -2 is not mislabeled wrapper success.
Final control resources: rc0,7s,peak-RSS0.03GiB,min-avail52915MiB,peak-load0.50.

The production verifier and lock helper are unchanged from the prior repair.
Prior bounded P2 acceptance, six guarded fault successes, stronger independent
identity-through-child-exit/release-precheck controls and fcc monitor acceptance
remain intact. Old failing receipts are preserved. Late release-event failure
can still produce rc1/incomplete evidence after verified child cleanup; this
is not a false-success guarantee. Unknown Popen-handle quarantine remains
source-only with no recovery claim. SIGKILL/fatal interpreter or host failure,
and unseen escaped descendants, remain outside these bounded controls.

No production invocation, native build, scoring, generic audit, expectation
change, acquisition, push or landing. Independent boundary-only re-review is
required. Generic59/60red, diagnostic proposal unapplied,0/19 and
attribution_ready=false are unchanged.
