# Durable Linux IQA timing

The Linux sequence uses the same frozen inputs, cold/warm worker protocol,
per-attempt watchdog and dead-ticket recovery as the
[Mac runner](IQA_DURABLE_MAC.md). Each family runs at 512×512 before any
larger input: non-NN CPU, NN CPU, NN CUDA, then those families again with
sizes ordered by pixel count. NN maps have no qualified adapter; NN cells
are scalar-only. Routine NN selection excludes PieAPP unless explicitly
selected. This is a single-pair S67 scaling study, not quality qualification.

## Placement and resource boundaries

Prepare the base CPU bundle with `--zenbench /absolute/zenbench --cpu-set
0,1,2,3,4,5,6,7` after resolving the machine's topology with `zenbench
cpu-info`. These IDs describe the measured i265 topology; resolve them again
on another host. Explicit thread budgets remain 1 and 8, rather than “all.”

`run-heavy` creates a fresh memory-limited scope for each attempt. Placement
must therefore be enforced inside that scope: a controller's inherited
thread affinity or an earlier scope does not establish the worker boundary.
Before exec, the worker constrains its own scope's `cpuset.cpus`, pins its
calling thread and asks the frozen zenbench CLI to verify the performance
class and exact CPU set. A refusal produces no measurement. The attempt
retains `cpu-placement.json` with zenbench topology, enforcement, affinity,
memory limit and swap limit. Warm measurement retains zenbench admission
observations and all ten rounds. Cold attempts use the shared heavy lock;
they do not execute zenbench's warm-round admission gate.

The host must expose the cpuset controller through the user manager's
ancestors. Some managers delegate memory but leave cpuset files owned by
root: `AllowedCPUs` can appear in `systemctl show` while the kernel boundary
remains unrestricted. The worker uses passwordless `sudo -n tee` only on
the cpuset path resolved from its own kernel cgroup membership. It requires
an app scope under its current UID's manager, and then verifies the actual
boundary with zenbench. Missing delegation or unavailable sudo remains a
failed attempt. This setup does not persist controller delegation across a
reboot. Resource and placement receipts must be inspected on a new host.

## Build and prepare

Build under the shared machine heavy lock and resource wrapper:

```sh
LOCK_SNAP_DIR="$HOME/tmp/zenmetrics-lock-snap" just iqa-compete-build durable
```

This mode uses the pinned Cargo snapshot and existing ORT thread-budget
patch. It retains native scorer construction for cold timing. The patch
sets intra-op workers to the cell budget and inter-op workers to one;
environment variables alone do not configure ORT sessions. Retain the
snapshot's `ort-threads.meta.json` with the build receipt. The NN timing
control checks each CUDA attempt's emitted `IQA_ORT_THREADS` value.

Prepare CPU and NN bundles using the existing durable preparation recipes.
NN CUDA preparation accepts `--engine ort-cuda --ort-library LIBRARY` and
repeated `--runtime-library FILE` arguments. Core, provider and explicitly
supplied dependency libraries are copied and hashed. Their frozen directory
becomes the child's library path. CUDA provider initialization is required;
unsupported graph nodes can still execute on that session's CPU provider.
This must not be described as a graph with exclusively GPU operations.

```sh
just iqa-linux-coverage-prepare CPU NN_CPU NN_CUDA NEW_DIRECTORY BINARY BUILD_JSON
just bench-iqa-linux NEW_DIRECTORY/sequence.json install
just bench-iqa-linux NEW_DIRECTORY/sequence.json status
```

The plan binds all six manifest hashes. Its systemd user service survives
terminal disconnects and restarts a lost controller. Each successor requires
its predecessor's exact terminal inventory and released coordinator lock.
Exhausted failures stay failed and permit the next family; integrity failures
stop the service. `service.log` and `service-command.json` sit beside the plan.
Each output has its own `progress.json`, report and preserved attempts.
The transient service is not installed for boot-time restart.

Run `just iqa-durable-check` and `just iqa-durable-nn-check PYTHON_WITH_ONNX`
for supervisor, dependency and export controls. Run the real NN control with
`just iqa-nn-timing-check BUNDLE NEW_OUTPUT --lock MACHINE_HEAVY_LOCK
--threads 1` (and 8 separately). Earlier attempts without explicit ORT session
budgets remain diagnostic, even if their numerical protocol passed.

`just iqa-cpu-timing-check BUNDLE NEW_OUTPUT --lock MACHINE_HEAVY_LOCK`
requires PSNR scalar cold/warm identity, explicit unsupported PSNR maps and
GMSD cold/warm/scalar/map bit identity at the frozen 512 input. It retains
all attempts and the original timing-round requirements.
