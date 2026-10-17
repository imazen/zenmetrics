# Timing and qualification boundaries

This document describes the adapters. Completed coverage is recorded separately;
source availability and a successful build do not establish benchmark completion.

| Adapter | Resident input | Excluded setup | Timed operation |
|---|---|---|---|
| Rust CLI CPU | RGB8 bytes | file read, initial model/scorer warmup | CLI metric call, internal color conversion, output release |
| Rust native maps | RGB8 or native plane images | file read, scorer construction | full pair evaluate with native maps and output release |
| Rust CUDA/WGPU | RGB8 bytes | device instance/buffer allocation, initial JIT warmup | full pair compute, internal upload/conversion, reduction/readback; map readback when offered |
| libvmaf | u8 YUV/gray planes | file read, RGB component conversion, context/model/feature registration, picture allocation/fill | submit pictures, extract features, flush, predict score |
| libjxl | normalized encoded-sRGB ImageBundle | raw file read/unpack, metadata and worker pool | ComputeScore or ComputeSSIMULACRA2, internal color conversion and output release |
| numpy/PyTorch | ndarray or normalized tensors on selected device | import, raw read, normalization/upload, model construction | public metric call, internal conversions, synchronization, scalar extraction/output release |

The RGB-to-native-buffer preparation differs between APIs and is stated in each
row. The benchmark follows each implementation's native input type rather than
adding an artificial shared input conversion inside its call. GPU instances
retain buffers, without caching reference features. CUDA timing includes the
synchronous readback needed to return a scalar, not only kernel submission.

Rust uses interleaved zenbench arms, warmup and at least ten rounds. Percentiles
are over round batch means; iteration counts and raw rounds are retained.
Native C/C++ adapters use one warmup and ten individual timed calls. Python
uses ten calls or at least three calls totaling three seconds. Every sample
is retained. All cells have an outer process timeout below twenty minutes and a
host memory limit. Timeout or memory refusal is an explicit row, not a timing.
Admission is separate from measurement: a high-load refusal is retained and
retried after releasing the heavy-job lock.

The VMAF snapshot adapter caches immutable parsed model parameters. Its owned
clones remain timed. The original parser and every scoring expression remain
unchanged; the committed bitwise receipt covers scalar and native-map outputs.
PIQ BRISQUE also reads SVM parameters through its loader on every call. Its
benchmark loader caches the deserialized values and returns fresh tensor clones,
preserving the public function's in-place support-vector transpose. Each child
compares original-loader and cached-loader score bits before timing. Neither
adapter modifies a shipped crate or an installed oracle environment.

Every competitor compares its headline with the same-pair Rust score. Existing
parity gates determine qualification; exact comparison is used when no applicable
cross-version bound is documented. Bound values and their source are recorded.
For Butteraugli C++, the pinned crate's reference_parity.rs uses a 0.001 relative
gate. CUDA mismatches repeat a complete cell and retain both attempts. A residual
that reproduces remains a variant mismatch and is graphed separately. A variant
comparison is not evidence of matching algorithm semantics; the libvmaf VIF and
PSNR-HVS chroma conventions are explicitly labeled.

The Python package APIs and conventions are documented by their projects:
[PIQ functions](https://piq.readthedocs.io/en/latest/functions.html),
[scikit-image metrics](https://scikit-image.org/docs/stable/api/skimage.metrics.html),
[TorchMetrics SSIM](https://lightning.ai/docs/torchmetrics/latest/image/structural_similarity),
[pytorch-msssim](https://github.com/VainF/pytorch-msssim),
[LPIPS](https://github.com/richzhang/PerceptualSimilarity),
[ST-LPIPS](https://github.com/abhijay9/ShiftTolerant-LPIPS),
[DeepDC](https://github.com/h4nwei/DeepDC),
[FLIP](https://github.com/NVlabs/flip), and
[ColorVideoVDP](https://github.com/gfxdisp/ColorVideoVDP).

Third-party code is confined to this benchmark harness and external tools.
PyIQA is not installed, imported or invoked by these adapters. The legacy
CLI metric names containing a package name are identifiers of the existing Rust
implementations; they do not cause a Python dependency to load.

Slow Rust cells can span independently bounded chunks. Every zenbench group
still requests ten rounds. If its chunk budget ends first, it emits a partial
record with the actual rounds; that record cannot qualify alone. The controller
releases the scope/lock, starts another warmed chunk, and combines actual round
means only after at least ten have completed. Input, executable, thread budget
and sampling configuration must match. Score-bit disagreement is preserved.
Each chunk's load, log hash and resource receipt remain in the raw result;
aggregate resource seconds sum separate lock holds. No duration or sample is
estimated from an incomplete run.

The extended GPU driver checks every returned score using zenbench's deferred
output Drop, after the measured interval. Native map release remains inside the
metric adapter's call. Rows retain the last actual result and count any score
bit variation across zenbench warmup/calibration/timed calls. A variation or
out-of-bound CPU comparison triggers a second complete cell and links both
attempts. The initial GPU qualification summary predates this additional
per-call check; the full matrix uses the qualified extended driver.

A metric call may return multiple scalar columns. The timing covers that public
call's outputs; the primary comparison uses its first declared column (for
example WD scale 1 or PSNR-HVS Y). Paired no-reference routes score both images,
returning distorted and reference columns; the BRISQUE Python adapter does the
same two-image work. Each row records the declared output columns. Extra native
outputs and any differences in public API workload are stated rather than
presented as an isolated single-plane or single-image cost.

CVVDP CPU and GPU native-map entry points write into a caller-owned vector.
The adapter retains that vector across warmup and timed calls; warmup establishes
its capacity. Map computation and writing/readback remain timed, while final
caller-buffer destruction occurs after measurement. Butteraugli GPU copies its
native map into a returned host vector and drops that vector within the call.
These distinct public output-buffer contracts are retained in both adapters.

Static charts use Matplotlib from the owned competitor environment. Pass its
Python executable as the optional third argument to `just iqa-compete-charts`.
The script writes standalone SVGs and an HTML index. Markers represent observed
sizes; connecting lines join measured points without fitting or extrapolation.

Linux scope failures can leave zenbench admission records. The driver emits its
PID, process group and timestamp before warmup. Recovery is limited to that
reported process: the owner and every group member must have exited, the record
must match the owner/time window, and its guard must be acquired without waiting.
The original record is archived verbatim and retained under a backup name; live
or other-owner reservations stay in place. Seven OS-backed controls and a real
PieAPP timeout recovery exercise this path. Rust-header build validation is
recorded separately from the Python recovery controls.

Resource-wrapper receipts cover the entire admitted command, including untimed
input loading and warmup. Their peak RSS is a process-scope measurement, not
an isolated timed-kernel heap measurement.

The CLI learned-metric route caches loaded models in a process-wide `OnceLock`
([NN adapter](../../crates/zenmetrics-cli/src/metrics/nn.rs)). The first callable
warmup loads the selected model; later timed calls use that model directly.
The `IQA_WARMUP` duration measures that first score callable after persistent
scorer construction. Earlier input loading and eager constructor work are also
untimed but are outside that diagnostic duration.

Learned and no-reference CLI routes without native maps record `unsupported`
for that mode independently of score startup, consent, geometry or scope
failures. Reclassification retains the original status in raw metadata.
Competitor map capabilities remain specific to each public entry point.

Host load or low available-memory launch refusals are admission attempts, not
metric outcomes. They wait outside the shared lock and do not replace valid
partial rounds. Resume and collection inspect retained failure logs, including
legacy log pointers, to exclude those attempts from completed cells. A real
control resumed four saved rounds across an admission refusal and combined
4 + 5 + 5 actual rounds; the ten-round minimum remains unchanged.

Reference selection keys include the combined raw-pixel SHA-256, metric,
size label and thread budget. Each newly completed competitor cell reloads
its supplied CPU ledgers, so references appended by a concurrent CPU controller
can qualify subsequent cells. A different pixel hash never qualifies even if
the size label matches. Existing saved rows without a reference remain explicitly
uncompared until a separate comparison is recorded; this does not retime them.

`just iqa-compete-requalify` records a separate saved-row comparison receipt.
It shares the live controller's parity gates and asserts that sample count,
statistics, observed score, load and resource fields do not change. Inputs and
reference ledgers are hashed; a ledger changing during reference selection
refuses the receipt. Any newly discovered GPU mismatch without a linked earlier
repeat is listed as requiring reproduction. Updating a comparison never supplies
new timing samples or substitutes for that repeat.

ORT sessions require explicit thread configuration in addition to the environment.
The benchmark snapshot adapter sets intra-op threads to the cell's
`RTEN_NUM_THREADS` budget and inter-op threads to one in each image, pair and
trunk builder. It retains deterministic compute and all provider/graph settings.
Each constructed session reports its configured budget outside timing. The
collector keeps earlier ORT observations without that receipt as diagnostics,
not qualified 1T/all-core timings. A separate native-worker affinity probe checks
the installed runtime. See [ORT thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).

Compact summaries retain comparison availability, parity bound/source, reference
identity and configured ORT budgets. Charts put measurements lacking CPU
comparisons, and mismatches still requiring reproduction, in separate panels.
A valid timing alone does not establish same-score parity.

`just iqa-compete-coverage` audits the collected Rust rows against the frozen
inventory, requested sizes and thread budgets. GPU-name routes and learned
ORT-CUDA routes are requested explicitly for the host. Its completeness gate
refuses missing cells, insufficient sampling, diagnostic adapters, unexpected
process failures, missing GPU/CPU comparisons and pending reproductions.
Explicit supported input/build/resource refusals count as attempts, never as
timings. The receipt lists all missing/pending cells and reports extra routes
separately; testing this audit does not establish matrix completion.

GPU logs containing a device-service-thread panic or WGPU validation/binding
rejection cannot qualify a returned score. The controller preserves the returned
samples as diagnostic data, marks the score unqualified and requests a separate
repeat. Failed dispatch rows cannot be combined with successful partial rounds.
This handles the documented asynchronous WGPU failure mode before other-host
runtime qualification; pattern controls are not a substitute for running there.

ORT-CUDA benchmark children set `ORT_DISABLE_TELEMETRY=1` before loading the
runtime. ORT 1.31's [full opt-out](https://github.com/microsoft/onnxruntime/blob/v1.31.0/onnxruntime/core/platform/telemetry_environment.h)
avoids initializing the background uploader, rather than merely suppressing
individual events through the runtime API. The collector requires this opt-out
and the explicit session-budget receipt. Its setting is part of chunk identity.
The worker control retains exact one/twenty-thread assertions; default-telemetry
and opt-out diagnostic runs remain separate. Attribution of any extra thread
requires the measured per-thread cases, not this configuration alone.

Host and complete OS identity now participate in result collection, CPU-reference
matching, resume keys and chunk identity. Windows/Linux observations on the same
machine cannot replace one another. Coverage audits handle one host/OS at a time;
charts and compact summary names retain the distinction. Remaining native-host
supervisor and qualification work is described in [portability](PORTABILITY.md).

Optional bounded batches hold the host lock across several separately supervised
cells. They retain the same native child timers and per-cell run-heavy receipts;
no process setup or waiting enters the samples. Each child receives the remaining
hold deadline, with at most 1100 seconds plus bounded termination cleanup.
If the remaining batch budget interrupts a later cell, that attempt is retained
as scheduling evidence and retried in a fresh hold. Load/memory admission refusals
release the hold before waiting. The lock descriptor is inherited by supervised
children, so losing the controller cannot release the lock around a live child.
Explicit one/twenty-thread budgets and the ten-round minimum are unchanged.

The PIQ panel includes its public VGG16 LPIPS and DISTS classes as well as
its classical functions. Their model construction is untimed and inputs retain
PIQ's documented RGB [0,1] convention with internal ImageNet normalization.
See [PIQ class APIs](https://piq.readthedocs.io/en/latest/modules.html).
PIQ's FSIM scalar call and the Rust CLI call have different output workloads:
the latter also computes FSIMc. Rows retain that distinction explicitly instead
of substituting an implementation with a different public interface.
