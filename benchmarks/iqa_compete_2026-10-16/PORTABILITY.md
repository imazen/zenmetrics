# Remaining host qualification

The i265 matrix is incomplete. No remote-host or borrowed-PC measurement is
claimed here. The borrowed machine remains pending coordinator availability;
this study does not probe or reboot it.

The implemented collection, reference and chunk identities include host and OS.
Charts separate OS testbeds. Windows and Linux on the same hardware must retain
the same physical-host alias but different OS records and output directories.
Raw RGB bytes and model identities must match the original manifests; paths can
differ. Build the frozen source with runtime dispatch and record each executable,
compiler, dynamic runtime and driver identity. The current Python launchers
explicitly refuse Windows until its native supervisor is qualified.

## Borrowed Windows/Linux host

Use zenbench's existing native process supervisor and OS peak counters as the
starting point, with a small result-file bridge around the IQA driver. Its
[pinned fleet contract](https://github.com/imazen/zenbench/blob/ac675d3/docs/fleet.md)
separates supervisor wall time from in-process timing and describes Windows
process-handle peak resident and private-commit counters. Preserve that distinction;
only the IQA driver's own samples become compute latency. Reuse the existing
[zenmetrics execution adapter](../../scripts/jobsys/benchmark_exec.py) for receipt
and parent-lifetime conventions.

A hard Windows memory ceiling still needs implementation and real-host controls.
The pinned [nice supervisor](https://github.com/imazen/zenbench/blob/ac675d3/src/bin/support/nice.rs)
sets Windows priority but explicitly refuses requested memory ceilings outside
Linux; it cannot be substituted for that guard. A Windows supervisor must assign
the child to a Job before measured work, enforce a configured committed-memory
ceiling, terminate/reap the entire Job on timeout and retain failure receipts.
Microsoft documents [Job memory limits](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information)
and the separate [peak working-set counter](https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters).
Committed bytes are not resident bytes.

Before timing, qualify native execution, one/all-core budgets, child cleanup,
transient peak capture, memory refusal, timeout, busy-host refusal and raw-input
identity. Windows lacks POSIX load average: record its absence explicitly and
qualify a measured CPU-utilization/competing-process admission gate, retaining
its exact settings and controls. Do not substitute zero for an unavailable load
counter. Capture the actual GPU adapter and refuse software rendering. ORT uses
an explicit dynamic library, session worker budgets and background-I/O opt-out.
Then run CID22/2K, the size ladder and competitors on Windows, followed by the
same inputs and frozen source on Linux. The coordinator controls boot changes.

## Other Unix hosts

Use isolated owned checkouts on r7900x and mac; leave primary checkouts intact.
Resolve hardware, free RAM, disk and runtime availability on arrival. The existing
launcher selects Linux resource scopes or the separately qualified macOS wrapper.
Native-map and scalar calls retain their original timing boundaries. Record the
actual Metal/Vulkan adapter on hosts without NVIDIA inventory, and reject logged
device-service failures even if a call returned a score. The Linux reservation
recovery helper does not establish macOS recovery support. These are remaining
runtime acceptance tasks, not completed platform support.
