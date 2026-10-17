# Fresh benchmark runs after reboot

Build every host from one committed master revision and its pinned siblings.
Record that revision and each host's executable SHA256; executable hashes differ
across platforms. Keep earlier observations under their original source and
binary identities. A fresh run uses a new output directory and does not import
completed cells from older executables.

Set `IQA_RUN_HEAVY` to the executable resource wrapper on each host, or install
`run-heavy` on PATH. The launcher refuses a missing wrapper. Linux cells retain
their 16 GiB limit and macOS cells their 12 GiB monitored guard. macOS uses
`gtimeout --foreground` so the wrapper's process-group termination reaches the
timed child. The macOS guard is not a cgroup hard cap.

Before reboot, let active cells finish, stop only the owned queues, and copy raw
JSONL, input/model manifests, source patches, immutable executables and all
referenced logs/receipts. Record active PIDs and queue commands. Other sessions'
jobs require coordination with their owners; the benchmark does not reboot a
shared host or stop unrelated processes.

After reboot, resolve CPU, RAM, disk, GPU/runtime and competing jobs again.
Verify the existing raw-pixel and model hashes. Reboot alone does not establish
a quiet window: media services and other sessions can restart. Use the recorded
before/after process views and sampled CPU contention labels. Linux affinity
must exclude occupied CPUs and their SMT siblings; record the effective set.
macOS has no certified OS affinity placement in this study.

For each host, export the pinned source and build with the existing
`build_cpu.sh` recipe under the configured wrapper and host lock. Set
`IQA_GPU_RUNTIME=wgpu` for Metal, `cuda` only where a real NVIDIA runtime is
available, and leave it unset for CPU-only builds. Use a dedicated
`LOCK_SNAP_DIR`; do not reuse an earlier executable's path.

Run `run.py --coverage-first` against the existing `pairs.json`, using
`--sizes cid22,aic-small,aic-medium,aic-large,2k,4k,8k` and `--threads 1,all`.
Pass the existing model root and per-metric model map. Use separate output
directories for CPU, GPU and competitor routes; collect them with `collect.py`
before auditing with `coverage.py`. The defaults require five samples at CID22,
AIC-small and AIC-medium; three at AIC-large/2K/4K; one at 8K. Each cell has one
existing bounded scope. Unsupported maps and size/resource refusals remain
explicit rows. Compare GPU/competitor scores once at CID22; do not rerun parity
at every size. Report per-size present, timed, unsupported and refused counts.

The native libvmaf/libjxl launcher currently targets Linux. Windows supervision
and the remaining macOS acceptance work are described in [PORTABILITY.md](PORTABILITY.md).
The older remote-host receipts use frozen source and external orchestration;
their observations do not qualify a newly built master executable automatically.
