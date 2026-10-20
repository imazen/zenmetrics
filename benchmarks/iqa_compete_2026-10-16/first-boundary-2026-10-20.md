# Classical first-call boundaries and sRGB-display successor

Missing: no new adapter has acquired original-call bit controls, native numerical
observations or qualified timing in this lane. C/C++ emitted adapters need
separate builds. The new base HDR-VDP cohort needs the full current worker and a
bound official runtime. Preparation and source inspection do not qualify scores.

## Scope and source identity

The 84-route static review has SHA-256
`9a608cf99e5b90d7c0f7a4041978d2903424c2805e277f33cbb668afdabba1ca`.
Its 19 non-candidate classical routes are enumerated below. Every cited review
source used in this lane matches the review's stored hash. Original observations,
helpers, plans and acceptance gates remain unchanged. No production metric or
public API changes are proposed here.

The frozen native `iqa_compete_timing/mod.rs:5–15` includes `make_call` and the
first evaluation, captures elapsed time, then destroys the returned closure.
`iqa_compete.rs:98–149` constructs CVVDP's scorer in that closure's factory;
most other classical routes construct and destroy their local scorer inside
`run_metric_display`. Raw RGB reads precede both clocks. Lazy runtime creation
must be declared explicitly; imports are not an OS-cache-cold runtime.

## All 19 route dispositions

Paths below are relative to this benchmark directory unless prefixed `crates/`.
Line references identify the hash-bound review source, rather than later edits.

| Route | Verified scope problem or distinction | Additive disposition |
| --- | --- | --- |
| brisque | `canonical_cpu_cell.py:40–52` loads immutable parameters before timing and clones owned support vectors inside; `crates/brisque/src/lib.rs:678–685` has a zero-sized constructor. | Preserve resident-parameter first evaluation; file deserialization is not missing native construction. No constructor rewrite. |
| butteraugli | `libjxl_compute.cpp:54–64,98–112` hoists the thread pool and suppresses the first elapsed value. | Emit COLD; construct the runner on the first timed call and retain it beyond elapsed capture. Preserve conversion/comparator/metric bodies. |
| cvvdp | `canonical_cvvdp_cell.py:18,20–28` hoists scorer construction. Native `iqa_compete_maps.rs:20–46` constructs a retained scorer within cold. | Move construction into the first call and retain scorer beyond elapsed capture. Constructor JSON reads remain an explicit reference/native storage distinction. |
| dssim | `dssim_reference.c:27–41` hoists/finally frees its context; native `cpu_dispatch.rs:227–236` constructs per scalar call. | Construct and destroy context inside the timed call. Bind the author-library lineage independently; old fork and author are separate baselines. |
| hdrvdp-aic2026 | `canonical_hdrvdp_cell.py:34–49` applies the 200-nit linear RGB convention; official v2 `hdrvdp.m:247–251` and `load_spectral_resp.m:10` read spectral files within evaluation. | Preserve existing AIC convention. Withhold a strict matched startup ratio until original disk reads versus native embedded spectral parsing are declared. |
| hdrvdp3-aic2026 | Same adapter applies ten-bit code rounding, 200-nit EOTF and XYZ, reports Q. Official v3 `hdrvdp_visual_pathway.m:58–71` reads sensitivities inside evaluation. | Preserve AIC convention and Q output. Do not use base-route sRGB-display/Q_JOD as a replacement. |
| iwssim | `canonical_iwssim_cell.py:33–41` stages model construction; `crates/zenmetrics-api/src/cpu_dispatch.rs:191–203` constructs native state inside the scalar call. | Construct the unchanged original IW_SSIM object inside each score call; preserve legacy eig compatibility. |
| iwssim-piq | `canonical_iwssim_yiq_cell.py:14–26` substitutes PIQ's unrounded Y ingress then delegates to the staged original adapter. Native `metrics/mod.rs:3282–3308` constructs per call. | Same constructor timing correction with the exact original PIQ rgb2yiq ingress; no rounded-gray substitution. |
| niqe | `canonical_niqe_resident_cell.py:68–70` evaluates original and resident paths before its timer at 78. | Move the original/resident exact-bit gate to independent fresh-process controls; the emitted replacement has no scoring precheck. Retain owned parameter-array copies. Original COLD remains historical, pre-evaluated. |
| nlpd | `canonical_classical_cell.py:31–43` stages LaplacianPyramid and immutable tensor construction; original `NLPD_Pytorch/nlp.py:25–35` creates filters in its constructor. | Construct the unchanged original model inside each call; preserve RGB conversion and compare body. |
| nlpd-iqa | Same staged branch; original `IQA_pytorch/NLPD.py:19–31` creates filters, pads and upsampler. | Construct within each call; retain the rounded BT.709 Y ingress and as_loss=False. |
| proposal-butteraugli | Same libjxl suppressed first call and staged pool as butteraugli; p-norm map and pooling are already timed. | Same additive runner/COLD change; preserve map and p-norm body. |
| ssim2 | `ssimulacra2_primary_compute.cpp:79–113` emits a first score but creates its ingress runner before the clock. | Construct runner inside first timing; keep runner alive beyond elapsed capture. Keep the Cloudinary v2.1 body and caller-only canonical scoring. |
| vmaf | `libvmaf_rgb_compare.cpp:97–103` loads the model before timing. Frozen native `crates/vmaf/src/v0.rs:62–78` parses and caches embedded model on first call. | Load/destroy original model inside each canonical call. Preserve YUV709 conversion/context/features. |
| vmaf-4k | Same staging, model vmaf_4k_v0.6.1. | Same constructor-inclusive adapter; retain its exact model choice. |
| vmaf-neg | Same staging, model vmaf_v0.6.1neg. | Same adapter; retain NEG features/model. |
| vmaf-neg-studio601 | Same staging plus explicit studio601 input conversion. | Same adapter, preserve the 601 argument; never substitute 709. |
| vmaf-studio601 | Same staging plus explicit studio601 input conversion. | Same adapter, preserve the 601 argument. |
| vmaf-v1 | Same staging, model vmaf_v1.0.16_3d0h. Frozen native `crates/vmaf/src/lib.rs:173–189` parses/caches on first use. | Same constructor-inclusive adapter; preserve v1 chroma and model selection. |

The preparer is [first_boundary_prepare.py](first_boundary_prepare.py). It
requires exact reviewed source bytes and writes new artifacts and receipts
without overwriting originals. The control driver is
[first_boundary_control.py](first_boundary_control.py). Each original/candidate
arm runs twice in separate fresh processes, alternating order. Every emitted
headline and reference-image NR score must have identical f64 bits; failures and
raw logs are retained. Neither emission nor synthetic controls establish original
source parity. Run real controls before promotion or acquisition.

Use only COLD from the constructor-inclusive candidates for first-call
comparisons. Their subsequent SAMPLE values support bit controls; they do not
establish a matched warm boundary. In particular CVVDP recreates its model in
the candidate, while the native warm closure retains its scorer; the frozen
native VMAF model cache also changes warm setup work. Historical warm adapters
and their observations remain separate.

For libjxl/Cloudinary, a runner retained after elapsed capture matches native
process-lifetime worker teardown more closely than timing its destruction.
Runtime implementation, worker count and thread-budget behavior still require
observation; constructor placement alone does not prove equivalent runtime work.

## New base HDR-VDP successor at 4f89155

Production `crates/zenmetrics-cli/src/metrics/hdrvdp_aic.rs` at
`4f89155cd8ef766c285fcfbfed9d40def047a836` has SHA-256
`6b790071435a1bcc943f7d3da17b2353f6de434d4a526abb3dfeee4560c31b94`.
All 44 HDR-VDP source/data/manifest files checked against the retained build
source are byte-identical. The API `lib.rs` re-export is identical as well.
The cached compiler identity matches installed Rust 1.99.0 (b940084d7).
This supports proposing cached artifact reuse; no new linking or execution is
claimed. The optional [native probe](first_boundary_hdrvdp.rs) includes the full
unmodified frozen production module. It is preserved, uncompiled. The full
current worker will carry both production routes instead.

| Route | Original official call | Headline |
| --- | --- | --- |
| hdrvdp | `hdrvdp(distorted,reference,'sRGB-display',30,{})` | `result.Q` |
| hdrvdp3 | `hdrvdp3('quality',distorted,reference,'sRGB-display',30,{'use_gpu',false,'quiet',true,'age',24,'surround','none'})` | `result.Q_JOD` |

Both start from RGB8 /255; the original metric performs its own sRGB display
mapping, `99*linear+1` cd/m². V3 selects the encoding's original LED-LCD-sRGB
emission. There is no ten-bit quantization, manual 200-nit EOTF or XYZ matrix.
Those operations belong to the separately retained AIC2026 adapter.

[first_boundary_hdrvdp_octave.py](first_boundary_hdrvdp_octave.py) emits a new
entry script without launching Octave. It checks explicit runtime/image-package
versions and original entry resolution, reads raw RGB before timing, and records
runtime function paths. `first` executes once without any earlier score. `warm`
performs one explicit untimed warmup then emits five samples by default. Timed
work includes conversion, original option/spectral setup, evaluation and output
release. Original parameter-file I/O is included; native uses embedded data.
These are explicitly labeled evaluator measurements, with matched-scope ratios
withheld until that storage boundary is resolved.

The `--direct-control` entry reads interleaved pixels as a 3×N matrix and uses an
independent transpose/reshape to form H×W×3. It calls the exact same original
function in separate fresh processes. Compare all printed scalar bits for both
first/warm phases, twice in alternating process order. Use the same raw pairs,
source trees, runtime and thread limits. This qualifies the adapter wiring, not
native numerical equivalence. Missing data, version mismatch and changed bound
files must refuse execution; no missing-file skips or new tolerance is allowed.

[first_boundary_hdrvdp_plan.py](first_boundary_hdrvdp_plan.py) takes the retained
AIC pair inventory only to preserve raw input identities and thread budgets. It
creates new case IDs, configs, reference/direct scripts and a preparation ledger.
Bind the full current worker's source, binary, build receipt and complete official
runtime/package closure in a new execution plan before running. Never replace or
edit the predecessor plan. Native `hdrvdp3` declares the JOD column, not AIC Q.

The v2 successor's committed validation records Octave 11.1.0/image 2.20.1.
It explicitly records a blur discrepancy on Octave 6.4/image 2.12. Local installed
directories show the older runtime/package; no Octave process was launched in
this lane. The emitter accepts an explicitly declared runtime but records whether
it matches the documented golden runtime. Legacy-runtime results remain labeled
diagnostics. Do not treat a version label as observed runtime qualification; the
entry must execute and retain its runtime receipt.

Original-call controls require bit identity. New native comparisons have no
inherited per-pair bound from the old AIC plans (their bounds are null). Report
observed deltas until a real applicable bound is supplied. The v2 5e-3 golden
assertion applies to its existing generated fixture panel; it is unchanged.
No accepted old score or typed-input refusal is relabeled as this successor.
