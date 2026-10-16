# Cool-Chic Wasserstein Distortion — 2026-10-16

Implemented on `wd-port`, based on `nn-lpips-vgg` at `552bb3be`.
Core `18d1875d`, area bound `339a52f8`, CLI `0e89c64a`, KonJND
`65bd274b`, both-engine gates `419c040c`.

This is the pinned [Cool-Chic convention](https://github.com/Orange-OpenSource/Cool-Chic/blob/main/coolchic/training/metrics/wasserstein.py),
not an assertion that the AIC-4 preprocessing has been identified. Reference
source, checkpoint and graph hashes are recorded in [export.json](export.json).
The BSD source and license notices are retained in
[LICENSE-wd](../../crates/zenmetrics-nn/LICENSE-wd).

The metric uses raw sRGB `[0,1]`, VGG16 pre-pool ReLU taps 3/8/15/22,
five mean/variance levels, zero-padded 3×3 binomial filtering, variance floor
5e-7 and the filtered sigma-map weights. All five `log2_sigma=1..5` outputs
share the feature extraction and moment passes. The CUDA engine accelerates
the trunk; the statistics remain in Rust. Warm references reuse VGG features.
The API supports native strided input, refuses maps, and returns errors for
size mismatch, either axis below 8/above 4096, or area above 9,000,000 pixels
(1,048,576 on 32-bit targets). The area bound covers all supplied AIC sizes. All 61 native controls complete,
including the largest 3355×2516 pair.

## Verified numerical panel

All 25 pairs pass on both engines: 250 comparisons across five columns,
including 32², 64²–2048² and portrait/landscape odd sizes. Aligned padded rows,
warm/cold identity and the zero identical-pair score are checked. The oracle
calls Cool-Chic's unmodified `wasserstein_distortion` per tap and rebuilds
`multi_wasserstein_distortion`'s constant sigma map and f32 per-tap sum around it; large planes use channel chunks to
bound oracle memory, combining each mean by its channel count. All 1,008
KonJND pairs were scored with repeated identical results; no mismatches were
reported. The pooled median is one JND because the identical-pair WD is zero.
JPEG and BPG each contribute 504 pairs. No calibration coefficients were fit.

| Column | KonJND median | RTen worst JND | CUDA worst JND |
|---|---:|---:|---:|
| wd_s1 | 0.3821742833 | 1.871543347e-6 | 2.495391130e-6 |
| wd_s2 | 0.2477349937 | 1.924787254e-6 | 1.924787254e-6 |
| wd_s3 | 0.1602388099 | 1.859869180e-6 | 2.975790687e-6 |
| wd_s4 | 0.1038497202 | 1.147901885e-6 | 3.443705654e-6 |
| wd_s5 | 0.0674495548 | 1.767384379e-6 | 1.767384379e-6 |

The gate is 1e-3 JND per column. [Per-engine errors](ladder_parity.tsv),
[codec medians](konjnd_scales.tsv), and [measurement digest](konjnd_scales.json)
retain the results. Source goldens are
[wd_goldens.tsv](../../crates/zenmetrics-nn/models/wd/wd_goldens.tsv).

## Landing checks and reproduction

`just nn-wd-gates <models> <square-ladder>` runs debug/release NN tests,
workspace fmt, exact CI workspace clippy, NN all-target clippy with both
engines, both requested CLI feature checks, inventory/routing regressions,
RTen/CUDA parity and a real default-scheduler CLI oracle assertion for all
five columns. All pass. The first attempt exposed invalid 13-byte test padding;
15-byte padding is pixel-aligned and preserves every strided/numeric assertion.
The strict inventory count is 104 metric families after adding WD.

Other recipes: `nn-wd-reference`, `nn-wd-score`, `nn-wd-kon-jnd`,
`nn-wd-parity`, `nn-wd-cli-smoke`, `nn-wd-aic-probe`. Heavy commands use
`flock ~/tmp/i265-heavy.lock` plus `run-heavy --mem 16G --jobs 16`, disk-backed
`TMPDIR`, and `LOCK_SNAP_DIR=~/tmp/zenmetrics-lock-snap-wd`.

Observed on i265: models `~/tmp/wd-port/models`; square/non-square ladders
`~/tmp/nn-phase0/{ladder,ladder-nonsquare}`; full calibration rows
`~/tmp/wd-port/konjnd-wd.tsv` (SHA-256 in the digest); full logs
`~/tmp/aic-variants-scratch/round12-*.log`. Weights and images are not committed.
ORT resolves at `~/tmp/nnrt-research/venv-cu/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.31.0`;
its CUDA provider dependencies were checked with `ldd` before use.

Measured wrapper receipts (no extrapolated time or memory claims):

```
baseline fmt/clippy: rc=0 76s | peak-RSS 0.83GiB | min-avail 19566MiB | peak-load 4.30
CPU upstream oracle, 25 pairs: rc=0 400s | peak-RSS 7.91GiB | min-avail 14459MiB | peak-load 18.30
unit tests plus six-pair CUDA pilot: rc=0 14s | peak-RSS 1.24GiB | min-avail 21293MiB | peak-load 9.31
KonJND, all 1008 pairs: rc=0 811s | peak-RSS 1.28GiB | min-avail 21235MiB | peak-load 14.69
initial complete gates: rc=0 134s | peak-RSS 8.23GiB | min-avail 14226MiB | peak-load 9.88
post-memory-fix complete gates: rc=0 182s | peak-RSS 10.69GiB | min-avail 13075MiB | peak-load 10.58
```

The separate AIC probe runs under a single hard one-hour clock. Its residuals
and coverage are recorded after completion, separately from upstream parity.

## Full-resolution memory qualification

The initial CUDA native-control sequence completed 57 pairs before hitting the
16-GiB cgroup cap at S67 (3355×2516). The interrupted wrapper's zero RSS is
invalid because its measurement process was terminated. No cap was raised.
`ae582a3b` disables the WD host arena and consumes each ORT output after copying
it, bounding retained output memory across changing shapes. All four remaining
pairs complete: `/usr/bin/time -v` reports maximum RSS 11,210,496 KiB, elapsed
55.58 s, exit 0 (`~/tmp/wd-port/aic-round12/retry4.time`). The six-pair pilot's
five-column scores are bit-identical before/after. Both-engine numerical errors
remain unchanged on the complete ladder; every landing gate passes again.
These are recovery-panel measurements, not a measured before/after RSS ratio.

## AIC convention search — unmatched

The single clock began 04:49:05 UTC and ended 05:08:33 UTC on 2026-10-16,
including native controls, the resource correction, selection and full winner
verification. The 60-minute deadline was 05:49:05 UTC. Previously measured
24-grid conventions were read and not repeated. Native Rust controls cover all
61 pairs. Historical ImageNet controls were reused at their archived six-decimal
precision on the three selection pairs.

The new search evaluates pre-ReLU/post-pool taps, raw/normalized/range/linear/
BGR/luma inputs, 256/512 short-side resize, feature-rescaled sigma, and seven
pooling forms. It completes 113 combinations: two previous/native controls plus
111 new combinations (including six new aggregations of those controls).
[Selection ranking](aic_selection.tsv) uses worst then median site JND over
S01–S03; it fits no coefficients. The selected configuration is raw RGB [0,1],
post-pool taps 4/9/16/23, sigma `max(k−mean(log2(input_size/feature_size)),0)`,
and sum across taps. Every winner pair was evaluated twice; all 61 reproduce
exactly, with no repeat mismatches. This full panel includes the selection
pairs and is not an independent held-out test. Other candidates were not
measured on all 61 pairs, so “best” means selected on S01–S03.

| Column | Winner median absolute error | Winner max absolute error | Winner median site JND | Winner max site JND |
|---|---:|---:|---:|---:|
| WD_s1 | 0.2346362472 | 1.8276839256 | 0.2405550604 | 1.3984686018 |
| WD_s2 | 0.2219514847 | 1.6570122242 | 0.2310363687 | 1.4947104439 |
| WD_s3 | 0.2223619223 | 2.0376310349 | 0.2902315124 | 2.3648735969 |
| WD_s4 | 0.1776740551 | 1.4441973567 | 0.3724274401 | 2.4017891693 |
| WD_s5 | 0.1123020649 | 0.8913221955 | 0.3459800033 | 2.1742165445 |

These residuals do **not** establish an AIC match. The shipped `wd` remains the
Cool-Chic convention. Its native-control median site errors across columns are
0.9974450227, 1.1794214028, 1.2246033057, 1.0541206793 and 0.8888083345 JND.
[AIC residuals](aic_residuals.tsv) and [all predictions](aic_predictions.tsv)
retain full precision. Site JND is `abs(a*predicted^c − a*published^c)` using the
supplied mapping (b=0, sign=+1); it is distinct from the KonJND port-parity scale.

[Receipt](aic_receipt.json) records input/artifact digests, versions, coverage
and timing. Full probe artifacts remain at `~/tmp/wd-port/aic-round12/search`
on i265; native/recovery logs are retained beside it. New search wrapper:
`rc=0 150s | peak-RSS 1.82GiB | min-avail 19468MiB | peak-load 8.42`.
A first four-second attempt failed on integer sigma tensor dtype; the corrected
float32 attempt retains the original clock. `nn-wd-aic-report` compacts existing
measurements without repeating inference.

The largest 3355×2516 pair also completes on RTen under the same 16-GiB cap:
`rc=0 34s | peak-RSS 11.08GiB | min-avail 11620MiB | peak-load 8.22`.
`/usr/bin/time -v` reports 11,615,712 KiB maximum RSS, elapsed 34.12 s,
exit 0 (`~/tmp/wd-port/aic-round12/largest-rten.time`). All five scores agree
with retained CUDA controls within 1.1046152371067653e-7 KonJND;
[full-precision comparison](largest_engine_parity.tsv). This is a two-engine
capacity/parity control, separate from the timed convention search.

At the 9,000,000-pixel area cap itself (3000×3000, coordinator rerun after
review), maximum RSS is 12,346,372 KiB on RTen (50.43 s) and 11,880,128 KiB on
ONNX Runtime CUDA (36.43 s), both under the 16-GiB cap; the two engines agree to
1.5e-8 in every column ([cap_memory.tsv](cap_memory.tsv)). Most of it is the two
images' four VGG16 taps, held at once (3000×3000 taps are 4.0 GiB per image from
their shapes: 64·9 M + 128·2.25 M + 256·0.56 M + 512·0.14 M floats). Running
the statistics head per tap would bound it by the largest tap pair instead; not
done here.
