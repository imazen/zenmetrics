# Exceptional containment and executable bytecode repair

Review candidate only. No acquisition or landing is authorized. Historical
candidates and the rejected pilot remain unchanged; coverage is 0/19, with no
qualified ratio or confidence interval.

## Containment

Required pidfd facilities are probed before launch. Once a child can exist,
process discovery, identity-safe signaling, reaping and receipt failures retry
inside containment while both caller admission locks remain held. Failed scans
mean unknown containment, never empty. SIGINT/SIGTERM record cancellation without
unwinding quarantine. Release requires an empty scan, durable terminal receipt
and another empty scan. An already-open diagnostic file retains initial failures
when receipt replacement fails. PID/start-time verification after pidfd opening
is retained; there is no numeric-PID signaling fallback.

The actual panel.acquire controls inject pidfd EMFILE, process-scan EIO,
signaling EPERM and receipt ENOSPC. Each has transient and persistent variants;
the latter requests cancellation while failure remains active. An external
supervisor checks both locks, explicitly enables recovery, requires empty/reaped
children before release, and owns a bounded PID-safe cleanup backstop. Facility
preflight and pre-existing-child refusal are also covered.

## Executable caches

The sealer now hashes environment bytecode and records cache membership across
resolved interpreter search roots, standard library, environment libraries and
script directories. Verification rediscovers these roots and compares file
membership, hashes and symlink targets. Zip/egg search-root files are hashed as
whole files. Python environment overrides are removed consistently for discovery
and execution. The no-write setting is retained but is not treated as a read ban.
Fresh-interpreter controls execute timestamp-valid substituted bytecode while
source bytes stay unchanged, then require rejection for added and mutated caches.
A sourceless pyc control is included. Installed canonical packages were untouched.

## Evidence and limits

The independent review of 458a68b4 recorded both locks released while its owned
child remained alive after injected EMFILE, and unchanged VALUE=1 source executing
VALUE=2 bytecode. Those reviewer logs are preserved with attribution in the private
repair2 evidence archive. This repair passes `just clean-panel-check`: 34 tests,
including eight end-to-end quarantine fault cases. The final run reports rc=0,
15 seconds, peak RSS 0.03 GiB, minimum available 51394 MiB, peak load 0.28,
2 GiB hard cap and one job. These are control resources, not metric measurements.

Evidence archive: `/mnt/v/aic2026-evidence-private/clean-panel-repair2-2026-10-19/`.
Its SHA256SUMS binds both local runs and the attributed prior reviewer evidence;
final-review separately binds the immutable candidate diff and identity.

Linux procfs/subreaper/pidfd and serialized controller-child ownership remain
requirements. SIGKILL, interpreter death and kernel failure cannot be quarantined
by Python; no uninterruptible-kernel-task control is claimed. Pre/post provenance
checks do not make concurrent mutate-and-restore attacks impossible. Real schema2
reseal/runtime validation, native second NR-result parity, HeavyProcesses(4), the
remaining 18 adapters and production stopping/order/dependence qualification are
still separate unresolved gates. No scorer, benchmark or trace was run.
