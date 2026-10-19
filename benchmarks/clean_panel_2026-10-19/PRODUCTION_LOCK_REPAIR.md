# Separate exception-retention and production-join repair

The prior production wrapper candidate is independently rejected. Its successful
archived run remains historical evidence; it does not establish exception safety.
Prior bounded monitor and containment acceptance is unchanged.

After Popen returns, all identity, inventory, receipt and cleanup-proof operations
run inside a lock-retaining retry boundary. A persistent fault quarantines the
wrapper with both descriptors live, including after direct-child exit. Logging
failure cannot escape quarantine. Interrupts do not authorize release. A Popen
exception with no returned handle is conservatively unknown launch state and
quarantines indefinitely; no absence-of-child claim is inferred. SIGKILL or host
failure is outside this in-process protection.

Cleanup requires direct-child reap, retained PID/start identities gone, and a
fresh empty owned-session inventory. The required control/controller identities
must also be gone. Path/kernel/receipt checks and cleanup-confirmed persistence
precede either unlock. Session sampling still cannot prove absence of unseen
escaped descendants. Unknown cleanup never becomes success through a timeout.

Owned controls exercise six faults: immediate post-Popen identity, inventory,
interrupt, final receipt write, cleanup event write and cleanup/quarantine log
write. Each retains a kernel fdinfo witness and actual nonblocking refusal on
both fixture locks while cleanup is pending. Persistent faults are removed only
by the test driver after observing retention; then natural child cleanup and
reverse release complete. The indefinite unknown-Popen-handle branch is a
conservative source path, not a tested launch-recovery mechanism.

The verifier joins exact kernel owner/device/inode records, nonempty ancestry
with controller/command/wrapper PID-start identity, before/after lifetime,
release descriptor/path identity, positive cleanup/reap status and required exit
inventories. All five independently produced fully rehashed mutants reject
AFTER the unchanged artifact/event/monitor checks pass. The retained positive
archive still passes. Reviewer evidence is read only.

Final focused run: rc0,9s,peak-RSS0.06GiB,min-avail52931MiB,peak-load0.11.
These are owned control resources, not metric performance. Independent review
is pending; no production execution, generic audit, scoring, threshold change,
push or landing. Generic59/60 RED, diagnostic proposal unapplied,0/19 and
attribution_ready=false remain. Native build preparation is a separate commit.
