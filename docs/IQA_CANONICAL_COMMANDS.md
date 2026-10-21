# Canonical IQA benchmark commands

`just bench-iqa` and `just bench-iqa-changed` freeze new scalar CPU campaigns.
The changed-only selector includes 41 current routes, including the revised base
HDR names and Daala RGB preprocessing. Its [local scheduling table](../benchmarks/rebenchmark_since_f7668c7/order_current_local41_2026-10-20.tsv)
retains measured warm medians from one common 1152×1152 photo crop; it is not a
speed prediction for other images or thread budgets. Historical 27-, 38- and 40-route
tables remain available for their frozen plans.
For the current local canonical comparison, the [41-route receipt](../benchmarks/canonical_recheck_2026-10-19/local41-scoped-fullcall-command.json) freezes the original references, numerical bounds and setup caveats:

```sh
just bench-iqa-canonical-compare "$HOME/tmp/local-canonical-20261019T2330/changed-canonical-sources-current41-scoped-v1.json" /absolute/new-results 1,8
```

This complete manifest uses one/eight workers because the frozen base HDR cell
accepts those budgets. Four HDR routes keep timing-scope caveats. Its82declared
cases are not82acquired or passing observations; full size/content grids have
separate plans. New output paths are required.

They require explicit worker, build receipt, source revision, verified RGB pair
manifest, new bundle/output paths, resource wrapper and shared host lock:

```sh
just bench-iqa-changed --binary /absolute/iqa-compete \
  --pairs /absolute/pairs.json --build-receipt /absolute/build.json \
  --source-commit REVISION --bundle /absolute/new-bundle \
  --output /absolute/new-results --lock "$HOME/tmp/HOST-heavy.lock" \
  --run-heavy "$HOME/work/claudehints/scripts/run-heavy"
```

Build and qualify the native worker beforehand under the same shared lock.
On hosts requiring checked performance-core placement, also supply the qualified
`--zenbench` executable and explicit `--cpu-set`. Preparation verifies committed
RGB hashes and build receipt identity while holding admission. Each measurement
then acquires that same shared lock; preparation releases it before the durable
runner starts, preventing nested-lock deadlock.

The protocol is three fresh-process cold repeats, including construction and
first evaluation after RGB loading, plus ten clean warm zenbench rounds with a
100 ms target. Warm scoring evaluates the full pair repeatedly after initialization.
The six geometries are 512×512, 840×944, 1680×1888, 3355×2516, 2048×1080 and
3840×2160, with scalar scores at threads 1 and 8. The wrapper uses 12G/8 jobs;
its actual platform resource receipt, including peak memory, remains in worker
logs. Native score and resource gates are unchanged. A controller `ok` alone
does not establish that every warm round passed its gate.

Existing bundle/output paths are refused. Resume retained campaigns through
`just iqa-durable-run MANIFEST OUTPUT WRAPPER LOCK`; preserve their original
manifest, worker, attempt caps and lineage. `iqa-durable-prepare` now explicitly
selects scalar scores, six non-8K sizes, threads 1/8 and three cold repeats.
The low-level Python freezer retains explicit map/size options for historical
work; those options do not change already frozen campaigns.

`bench-iqa-supplemental` retains the historical portable 5/3/1 panel.
`bench-iqa-advanced-supplemental` retains historical Python/GPU/NN experiments.
Their samples must remain separate from canonical observations. Kernel benchmarks,
profilers and parity controls answer different questions and do not replace this
comparison protocol. NN campaigns require separately qualified adapters and run
last in the current three-host program.

Check command construction and admission with `just iqa-canonical-command-check`.

Historical `iqa-compete-*` panel executors also carry a `-supplemental` suffix.
Recipes named smoke, probe, control, tests or check are diagnostics; their
intentionally short budgets are not comparison measurements. Historical recipes
that explicitly request 8K remain supplemental and are not part of the active
non-8K campaign. Build recipes do not themselves qualify timing observations.

New durable runners also acquire shared admission for initial full asset hashing.
This prevents coordinator preflight CPU work from overlapping another admitted
measurement. Hashing remains outside cold and warm clocks; frozen older runners
retain their original identities.

After a campaign is terminal, `just iqa-canonical-gate-retry BUNDLE RESULTS
NEW_DIRECTORY LOCK` validates its retained attempt seals and freezes only warm
cells with resource flags or missing rounds and an original attempt remaining.
`retry-plan.json` records the exact maximum attempts for each successor and
lists exhausted unqualified cells. It does not install a service or grant new
attempts. Replacements retain the parent report hash; final selection counts
each logical cell once. Clean cells and input-contract refusals are not repeated.

A create-once reservation beside the parent report prevents preparing another
successor against the same remaining caps. Preserve that reservation if
preparation fails; inspect its recorded successor before taking further action.
Existing manually prepared successor lineages must be audited before using this
helper on their original parent.

Exports retain the executed worker engine when the worker reports it, including
weightless private CW-SSIM’s native Rust route. The NN engine environment remains
a fallback for older workers; absent engine evidence remains null.

The [captured input-contract audit](../benchmarks/iqa_input_contracts_2026-10-20/README.md)
distinguishes HDR luminance/viewing refusals and fixed-factor MS-SSIM window
refusals from unavailable implementations or worker stalls. Preserve these raw
attempts and limits; larger valid geometries remain in the frozen queue.

### Changed-only successors after a broad campaign

`prepare_changed_successor.py BUNDLE RESULTS NEW_DIRECTORY METRICS_JSON --lock LOCK`
requires the old coordinator to have stopped and acquires its coordinator lock.
It rebuilds the report from sealed attempts, retains completed changed cells,
excludes unchanged cells and partitions unfinished cells by remaining original
attempt budget. A create-once parent reservation prevents duplicate successors.
Run each frozen successor with exactly the recorded `max_attempts`; never rerun
completed cells to replace a campaign identifier. The approved changed list has
27 routes. Original results and source/version identities remain separate.

On 2026-10-20 the Mac/i265 operational scope was restricted to the approved
27 changed/added routes. Broad parent campaigns remain retained evidence, not
instructions to finish unchanged methods again. The i265 boundary snapshot
retained 896 completed changed cells and froze 400 unfinished cells: 48 with
one remaining attempt and 352 with two. Successful parent cells are never
selected into these successors. r7900x retains its full non-NN scope because
it has no earlier timing panel. Progress denominators must follow the active
scope (1296 changed cells versus 4608 full-inventory cells), not silently mark
excluded unchanged cells complete.

### Completion and master remeasurement (owner direction, 2026-10-20)

The owner asked to bring the dataset to 100 % on i265, mac and r7900x, re-measuring only methods whose code
changed and never repeating finished cells. Every public classical series on the three hosts was built from
`519a6ecb`. Between that and master `d1950fa2` exactly 12 routes changed: the eight PSNR routes
(`psnr_kernels.rs`/`classical.rs` SIMD kernels), `flip`, `flip-v17` (SIMD kernels and the `zenmetrics-math`
powf lanes), `ws-hv` (SIMD kernels) and `dssim` (dependency pin 331b56cf → 22e9ab03). Sibling pins did not
change. The worker crate's `compete` feature newly enables AVX-512, but the CLI crate's `cpu-metrics` already
did at `519a6ecb`; the worker's directly compiled CLI modules serve maps only, which these scalar campaigns
do not measure. Find this list again with
`jj diff --from 519a6ecb --to <rev> --name-only crates Cargo.toml Cargo.lock ci/sibling-pins.tsv benchmarks/heaptrack/drivers/cpu_profile`.

- Unchanged routes keep their original worker: `prepare_changed_successor.py --purpose remaining` froze each
  parked parent's unfinished cells for the 51 routes not owned by its earlier changed successor and not among the
  12 (i265: 796 cells, mac: 1,536), with the parent's remaining attempt caps. mac's parked dvifmish larger-size
  campaign resumed with its own frozen worker (522 cells pending at resume).
- The 12 routes are re-measured on all three hosts with a master worker (`build_canonical_worker.sh … d1950fa2`;
  i265 binary `041081cf`, mac `f1a41d4f` with the private counter correction, r7900x `f26d5a01`), 576 cells
  each, the full six-size, two-thread, cold-and-warm protocol, attempts 2, timeout 1,100 s, with i265 placed on
  CPUs 0–7 as before.
- `brief_data.py` never pools builds: a cell measured by both uses the newer source once its series has as
  many rows, and counts the older one as superseded.
- `hdrvdp`/`hdrvdp3` refused 8-bit sRGB until `4f89155c` routed it through upstream's `sRGB-display`
  encoding (100 cd/m² peak, 1 cd/m² black, 30 ppd; checked against official HDR-VDP-2.2.2 goldens). Their
  cells are re-measured on all three hosts with workers built at `4f89155c` (i265 `06615a9e`), 96 cells each
  (`*-hdrvdp-srgb`). The earlier refusals stay recorded; they were never timings.
- r7900x needs no other completion run: its other failed cells are input-contract refusals. mac's 65 exhausted
  `private-nn-production` cells hit the 12 GiB monitored memory guard; they stay refusals unless the guard
  is changed by the owner.

Resolve the live state rather than trusting this list: `systemctl --user list-units 'iqa-*20261020*'` on Linux,
`launchctl list | grep iqa-rerun` on mac, and each campaign's `progress.json`. mac's controllers are
Background LaunchAgents (`LimitLoadToSessionType=Background`), which load without a console login.

### Saved-result publication during an active campaign

`bench-iqa-capture-saved` reads atomic reports and finalized sealed attempts,
without adopting unfinished workers or stopping measurements. Capture windows
and file hashes identify what existed when each host was read. Raw reports
stay outside public git. `bench-iqa-publish-saved` emits only allowlisted timing,
resource and provenance fields, with explicit preferred-row identity and
without per-image scores. Different binaries remain separate series; portable
panels cannot substitute for canonical cells. The
[2026-10-20 saved snapshot](../benchmarks/iqa_saved_2026-10-20T0056Z/README.md)
retains partial observations, refusals and gates separately.

### Rust versus original implementations

`just bench-iqa-canonical PLAN NEW_RESULTS` runs the explicit original/reference
commands in a frozen comparison plan. Select a subset with `--metrics name,name`;
`--blocks 3 --samples 5` controls the alternating acquisition blocks. Every arm
retains its command, stdout, stderr, process memory receipt and numerical result.
The runner serializes measurements under the local host resource locks.

Create the plan with `just bench-iqa-canonical-plan` and explicit `--binary`,
`--pairs`, `--source`, `--python`, `--extras`, `--brisque-weights`, `--ssim1`,
`--ssim2`, `--output`, `--metrics`, `--sizes` and `--threads`. These paths name
installed reference assets; the command does not download or silently substitute
an implementation. Additional original implementations use `--registry FILE`,
which supplies their commands, numerical bounds, scope and artifact hashes.
Both RGB inputs and executable/reference assets are verified before and after
each block. Existing plan and result paths are refused.

Ratios mean **Rust runtime / canonical runtime**: below 1 is faster. They are
descriptive medians of alternating block ratios. A missing bound remains an
observed numerical difference, rather than an asserted parity pass. A mismatched
input or timed boundary withholds its ratio. This comparative protocol is
separate from the frozen cold3/warm10 CPU campaign above.

`just bench-iqa-canonical-publish NEW_DIRECTORY RESULTS...` validates retained
blocks and emits compact source-bound tables for the README generator.
`just bench-iqa-canonical-check` checks parsing and result projection. Builder
recipes for original DSSIM, SSIMULACRA1, Butteraugli and libvmaf accept explicit
source paths; run builds under the same resource wrapper and host locks.

For an explicitly stopped broad parent, `prepare_gate_retry.py --stopped-parent
--metrics EXACT_NAMES_JSON` can select only named, unqualified warm cells with
remaining original attempts. The exclusive coordinator lock still proves the
old controller is absent; every prior attempt/config/log seal is verified.
The default still requires a terminal campaign. Completed clean cells are never
selected, and create-once reservations prevent reusing the retry cap.

The current comparative entry point checks every canonical warm score against
the plan's existing bound. Small nondeterministic reductions remain recorded;
they are not rejected solely for differing between warm calls. Retained older
plans keep their original runner. Optional plan fields `arm_timeout_seconds`
and `rust_chunk_budget_ms` extend acquisition budgets for slow neural models;
the native worker's 850000 ms cap and ten warm rounds remain unchanged.

`just bench-iqa-canonical-resume RESULTS --blocks N --samples N` finishes
missing blocks after the original coordinator has stopped. It preserves both
completed and interrupted attempts. `just bench-iqa-canonical-interval RESULTS
NEW_JSON` bootstraps complete alternating-block ratios, without pooling inner
samples, and records every native resource flag. An interval is descriptive;
flagged observations never become an accepted regression through resampling.

Original neural comparisons can use an already frozen graph bundle.
`bench-iqa-canonical-model-verify` checks explicitly selected copies against its
existing hash manifest. `bench-iqa-canonical-original-nn-plan` freezes original
LPIPS, ST-LPIPS, DISTS and DeepDC callables, graph identities, source, reference
weights and the existing JND/1000 bounds. Build and prepare under shared
admission; run neural measurements after the classical comparisons.

For a comparative run over an explicitly declared changed-route manifest, use:

```sh
just bench-iqa-canonical-changed /absolute/reference-manifest.json /absolute/new-results
```

The manifest supplies `binary`, `rust_source` and `metric_plans` (one explicit
reference-plan path for each changed name). The command verifies those bindings,
freezes `new-results.plan.json`, then compares threads 1 and 8 in ascending
retained historical duration. Each case keeps its valid geometry, input hashes,
reference convention and existing numerical bound. This order comes from the
committed changed-route timing list; it is not a prediction for a different host.
Both output paths must be new. Use `--metrics NAME,...` to restrict execution of
the frozen plan, or pass `--blocks N --samples N` for acquisition counts.

`bench-iqa-canonical-author-plan MANIFEST NEW_DIRECTORY` freezes separately
specified original author-network callers. The manifest names the reference
Python, helper, configuration, source/weight assets and already verified native
graph directory for each metric. The existing JND/1000 column bound is read from
`jnd_scale.tsv`. It retains the requested inference engine and explicit weight
consent. Constructor controls do not establish scoring parity.

`bench-iqa-canonical-queue MANIFEST` executes explicit `{plan, output, options}`
entries sequentially. Optional `{results, blocks}` prerequisites require their
complete frozen case list; `--wait` keeps the visible controller waiting for
those cases. Each acquired block still obtains both host locks and its memory
cap. Process failures remain in their cohort and do not discard later queued
checks. Existing output directories are refused.

`bench-iqa-canonical-status MANIFEST --details` reads completed-case and block
counts, pending cases and retained process/numerical findings. It starts no
workers and does not qualify results; historical setup failures remain counted
even when a separate replacement cohort succeeds.

`bench-iqa-canonical-round-audit PUBLIC_JSON NEW_RAW_MANIFEST COHORT...` checks
that every retained successful Rust arm records exactly ten warm observations
in its count, timing list and round means. Short or inconsistent arms fail the
audit. Missing arms remain explicitly counted. The small publication binds a
fresh raw hash manifest; neither a zero-finding prefix nor its arm count proves
complete metric coverage or performance qualification.

Use `canonical_wadiqam_inference_cell.py` for original Chainer WaDIQaM
references. Its CPU aliases live in the author modules, preserving Chainer's
own backend dispatch, and gradient recording is disabled. The unchanged
original model classes and oracle pooling functions supply the scalar.
`bench-iqa-canonical-wadiqam-inference PYTHON QUEUE NEW_RECEIPT`, under shared
heavy admission, checks all four models against gradient-enabled calls before
replacing unexecuted reference plans. It retains the old plan files and refuses
acquired outputs or plans already backed up. Existing numerical bounds, native
settings and budgets remain intact.

`bench-iqa-canonical-interval RESULTS NEW_JSON --additional SUCCESSOR_RESULTS`
combines a completed fixed follow-up only when its recorded parent plan, source,
worker and case contract agree. All blocks remain in the primary estimate. A
separate diagnostic counts whole blocks with at least ten clean native resource
observations; this subset does not turn the result into an accepted regression.

Both PieAPP routes are excluded from the owner-directed local canonical recheck.
AIMOS is excluded: proprietary, with no released implementation identified.

`bench-iqa-canonical-warm-profile WORKER CONFIG NEW_DIRECTORY` uses Linux perf
control FIFOs to enable sampling after the worker's existing completed-warmup
record. Model construction and initialization are excluded; benchmark calibration
and measured rounds remain included. It adds no metric timers or counters.
Run under the same host locks, thread budget and resource wrapper as the compared
case. The command retains perf's report, raw recording hashes, full logs and
configuration identity. Its default sampling frequency is 199 Hz.

Original-network plan preparation validates the 1100-second arm budget against
the existing supervisor; a larger plan value is refused before scoring.

`bench-iqa-canonical-flags QUEUE NEW_AUDIT_DIRECTORY LOCAL_REGRESSION_LOG`
reads completed comparisons, logs each newly observed ratio above 1.08 and
prepares a fixed 20-block follow-up plus a warm-profile configuration. It retains
incomplete cases as pending. Repeating the reader adds only newly completed
flags; it never replaces an acquired comparison or changes a numerical bound.

The original learned-metric dispatcher exposes RTEN and optional ORT CUDA/WebGPU,
with no ORT CPU engine. A frozen worker control explicitly refuses `ort-cpu`
before scoring. The separate PyIQA port routes do expose ORT CPU. Neural ratios
carry R/O for the executed RTEN/ORT CPU engine; an unsupported original-engine
request never supplies a comparison ratio. Source and controls are retained in
`benchmarks/canonical_recheck_2026-10-19/original-engine-refusal.json`.
The original ST-LPIPS 0.0.2 package also imports `torchsummary` (the local runtime
uses 1.5.1), even though that utility is not called during inference.

### Explicit comparative thread selection

`bench-iqa-canonical-compare` freezes the exact reference manifest scope and runs
Rust against the declared originals in the manifest's measured scheduling order:

```sh
just bench-iqa-canonical-compare /absolute/reference-manifest.json /absolute/new-results 1,8
```

Use `1,28` for this local host's full admitted CPU budget. Additional arguments
such as `--blocks 3 --samples 5` follow the thread selection. Both plan and result
paths must be new. Every selected route needs an explicit reference plan; the
command refuses missing routes or changed source/input bindings. The manifest
selects the native worker and reference implementations; it performs no build or
download. The current native changed-only inventory has 40 routes. A comparison
manifest covers only its declared routes; the retained 38-route manifest does
not include the two revised base HDR routes pending reference qualification.

### Per-name and per-cell spot-check tables

`just bench-iqa-canonical-summary SOURCE_DIRECTORY INVENTORY_RECEIPT NEW_DIRECTORY`
exports one status/representative row for every inventory name and separate
source-bound tables of all acquired warm cells. Run it under shared heavy
admission. Aggregate README ratios never supply invented individual timings:
the representative table always uses a retained case and links its observation
number to the full tables. `bench-iqa-canonical-summary-check` checks this
selection and refusal of changed evidence. See the
[local export](../benchmarks/port_spotcheck_2026-10-18/local_canonical_2026-10-20/index.json).

### Current forty-route local comparison, 2026-10-20

The comparative command admits preparation under both local heavy locks and
the 12 GiB wrapper, then releases them before the comparison runner acquires
admission for each block. Invoke this command directly; an outer lock would
prevent its inner preparation from starting. Low-level preparation recipes
still require caller-held admission.

The latest [scope-qualified local command receipt](../benchmarks/canonical_recheck_2026-10-19/local40-scoped-current40worker-command.json)
prepares 80 numerical cases for all 40 routes at threads 1/8, with comparable
timing ratios withheld on 22 cases across 11 routes. On this box:

```sh
just bench-iqa-canonical-compare "$HOME/tmp/local-canonical-20261019T2330/changed-canonical-sources-current40-scoped-v3.json" /absolute/new-results 1,8
```

Plan preparation passed with every original numerical/input/callable contract
unchanged. This is preparation evidence, not 80 completed measurements or a
human-approved exception. The older manifest and receipts below remain historical.

The later [timing-boundary review](../benchmarks/iqa_compete_2026-10-16/current40-timing-audit/matrix.md)
identifies setup differences on four HDR routes, six VMAF model routes and
DSSIM. A manifest may now supply `scope_dispositions`, a mapping from route
name to its source-backed explanation. `bench-iqa-canonical-compare` records
those caveats in a new plan, retains all numerical checks and withholds a
comparable timing ratio for those cases. It does not rewrite older plans or
observations, loosen a numerical bound or approve an exception. Unknown route
names and empty explanations are refused.

`just bench-iqa-canonical-scoped-check` verifies ratio withholding, preservation
of numerical gates and refusal to overwrite retained plans. The local base HDR
reference fixture supports only threads 1/8; the generic `1,28` example above
does not qualify it for a 28-thread request.

The [current40 command receipt](../benchmarks/canonical_recheck_2026-10-19/local40-current40worker-command.json)
records successful preparation of all 40 changed routes at threads 1/8 (80 cases),
including the revised base HDR names. On this box, run:

```sh
just bench-iqa-canonical-compare "$HOME/tmp/local-canonical-20261019T2330/changed-canonical-sources-current40.json" /absolute/new-results 1,8
```

The manifest uses the verified current40 native worker. Each reference retains
one valid pair and geometry; this command does not expand the full four-size,
three-content grid. The HDR entries cover one 512² photo pair with resident
spectral parameters, after original/resident exact-bit controls and actual cell
protocol controls passed. HDR-VDP-2 keeps its 0.005 bound; HDR-VDP-3 has no assigned
external bound. Timed comparisons and their resource gates remain separate from
successful preparation. The older 38-route manifests remain frozen predecessors.

### Source-reviewed grid preparation

`bench-iqa-canonical-grid-scope-plan SOURCE PAIRS REVIEWED NEW_PLAN --metrics NAMES
--threads COUNTS` expands the original reference contracts and carries forward
the independently reviewed timing caveats. Run this preparation recipe under
shared heavy admission, then run the completed plan separately. Numerical,
input and callable contracts must match between the original and reviewed
source plans; inconsistent flags or reasons are refused.

The advertised output appears only after qualification is complete. Staging
files remain beside it, including failed or interrupted preparation evidence;
existing outputs are preserved. `bench-iqa-canonical-grid-scope-check` covers
those guarantees and preservation of numerical bounds. The earlier
`frozen_grid_scope_20261020.py` preserves the exact source used to prepare the
active seven-route VMAF/CAMBI campaign. Its known future-use limits do not change
that inspected campaign, whose 168 cases retained all six model-route caveats.

### Recovering original canonical comparisons

`just bench-iqa-canonical-block-resume PLAN EXISTING_OUTPUT` resumes original
three-block, five-warm-sample campaigns. Add `--mode first` only for the original
three-block first-evaluation campaign. Fixed 20/40-block confirmation campaigns
keep their separate controllers. `bench-iqa-canonical-queue-resume MANIFEST`
handles a stopped original warm queue; `--after-queue MANIFEST` retains its
classical-before-neural prerequisite.

Recovery validates completed numerical classifications, retains failed outcomes,
and acquires only missing blocks using the original helper and budgets. An
interrupted attempt stays in place; the new attempt gets a distinct directory.
Original plans, completed block data and case summaries are never replaced.
Recovery journals identify each reused/new block by path and SHA256. Root TSV
rows are appended after validating existing rows, with the old TSV retained in
the journal. These descriptive measurements do not establish runtime acceptance.

If the original plan did not bind the block helper, supply both
`--original-helper /absolute/compare_canonical_budgeted.py` and
`--original-helper-sha256 SHA256`, after checking the old controller's command
and the retained source. Recovery records this explicit identity separately;
it does not add a binding retroactively to the frozen plan. Each original block
still performs its own asset checks and uses shared resource admission.
Use `--flags DIRECTORY --regression-log FILE` on queue recovery to audit every
newly completed case. Check recovery with `just bench-iqa-canonical-recovery-check`.
