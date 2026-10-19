# Twenty-minute favorite-metric probe — 2026-10-19

## Owner-approved tolerance update (after the probe)

The owner requested changing the two comparison tolerances. The live probe
checker now uses **0.002 absolute for `msssim` vs `octave:msssim`** and
**2e-8 absolute for `gmsd` vs `pyiqa:gmsd`**. Both retained comparisons pass
these new bounds; controls just above each bound reject. The PyIQA GMSD
competition path uses the same shared policy. Same-implementation Rust goldens
and original-C GMSD bounds are unchanged. Historical packets and the original
exclusions below retain the policy that applied when measured.

Use the pair command check with arguments
`probe_pair_arm.py --check METRIC REFERENCE ${baseline} ${candidate}`.
Reference names are explicit; an unsupported combination refuses. Input hashes
must agree and empty/nonfinite scores refuse. This accepts scalar agreement for
those variants, not resource cleanliness or full benchmark qualification.

### Additional native comparison: libvpx

Source checked 2026-10-19:
[libvpx fastssim.c](https://chromium.googlesource.com/webm/libvpx/+/refs/heads/main/vpx_dsp/fastssim.c),
blob `4d32a02a55549613a09b8302773586f97fb18e42`.
`vpx_calc_fastssim` implements a four-scale gradient-based approximation. It
starts with downsampling, uses renormalized scale weights, and adds luminance
at the coarsest scale. It exposes raw per-plane scores through output pointers;
the return value converts the 0.8Y+0.1U+0.1V combination to decibels.

A comparison adapter should report the raw Y output with an explicit luma and
bit-depth contract; directly comparing its dB return to the Rust [0,1] scalar
would be wrong. Keep it named as the libvpx fast variant, separately from the
five-scale Wang route and existing `msssim-libvmaf` implementation. No libvpx
metric was built or timed in this change.

## Original timed-probe record

Owner budget: 12:56:11–13:16:11 UTC, including admission and reporting.
Used the existing `zenbench pair` Iterate framework, resident-input warm timings,
randomized whole-invocation A/B with A/A controls, and prebuilt scorers. Ten
metrics attempted; IW-SSIM could not produce canonical timings. No qualification,
CI interval, or current-master performance claim is made.

**Actionable leads: VMAF first; SSIMULACRA2 second on the 7950X.** VMAF still
has an ingress/model-lifecycle scope mismatch. SSIMULACRA2 changes direction
between the two CPUs. Profile these paths before selecting an optimization.
FSIM, VSI, HaarPSI and NLPD do not look like the first throughput targets here.
MS-SSIM and the GMSD alternative need oracle agreement before speed rankings.

## Observations

Ratio is Rust/canonical latency: below 1 means Rust took less time. These are
exploratory observations, including resource-contaminated samples, not accepted
performance ratios. `1mp` is 1024×1024 except VMAF's historical-input `1mp` case
label, which actually used **2048×1080**. `4k` is 3840×2160. All request one worker.
One-block cases have no between-block precision estimate.

| CPU | Case | Canonical ms | Rust ms | Ratio | Paired blocks | Maximum score delta |
|---|---|---:|---:|---:|---:|---:|
| 7950X | haarpsi-pairA-1mp-t1-warm | 125.469 | 25.001 | 0.199 | 3 | 6.52861e-09 |
| 7950X | vsi-1mp | 480.499 | 69.228 | 0.144 | 2 | 3.33223e-06 |
| 7950X | butteraugli-1mp | 282.771 | 197.362 | 0.698 | 1 | 7.15256e-07 |
| 7950X | ssim2-1mp | 146.875 | 159.986 | 1.089 | 1 | 1.13687e-13 |
| 7950X | fsim-1mp | 359.742 | 25.983 | 0.072 | 1 | 1.83197e-10 |
| 7950X | nlpd-1mp | 388.786 | 22.401 | 0.058 | 1 | 5.84443e-06 |
| 7950X | msssim-1mp | 284.897 | 21.447 | 0.075 | 1 | 0.00123805 |
| 7950X | gmsd-1mp | 52.386 | 1.732 | 0.033 | 1 | 1.02932e-08 |
| 7950X | vmaf-1mp | 49.514 | 108.606 | 2.193 | 1 | 0.0151753 |
| 7950X | vmaf-4k | 196.035 | 387.567 | 1.977 | 2 | 0.00147484 |
| 5900XT | ssim2 | 173.683 | 124.984 | 0.720 | 2 | 1.13687e-13 |
| 5900XT | butteraugli | 361.500 | 294.848 | 0.816 | 2 | 7.15256e-07 |
| 5900XT | ssim2-4k | 1497.245 | 1344.348 | 0.898 | 1 | 5.95435e-12 |
| 5900XT | butteraugli-4k | 2975.123 | 2148.433 | 0.722 | 1 | 7.39098e-06 |

## Interpretation and exclusions

- Butteraugli: libjxl 0.12 RGB-input scope, maximum output; Rust additionally
  computes pnorm3. Observed delta is below the existing 0.001 relative bound.
- SSIMULACRA2: libjxl 0.12 alternative, not independently qualified standalone
  lineage. Scores differ in low floating-point bits; **not bit-identical**.
- VMAF: native uses preconverted BT.709 YUV420; Rust includes RGB ingress.
  Both measured sizes meet the existing 0.02 scalar bound. This does not isolate
  kernel cost or bind model-loading lifetime equivalence.
- FSIM: original MATLAB FeatureSIM automatic-factor source via Octave. 1MP
  scalar difference is below 5e-5.
- VSI: original MATLAB automatic-factor source via Octave, installed resize
  implementation not matched. 1MP difference is below 1e-4; 4K reference
  attempt timed out and contributes no ratio.
- HaarPSI: original MATLAB via Octave; numeric check passed at 5e-5.
- NLPD: authors' Python/PyTorch RGB reference; observed difference below 1e-5.
- MS-SSIM: selected metrix_mux Octave variant differs by 0.00123805, exceeding
  the existing 1e-4 bound. Timing is **not an equivalent-output Rust win**.
- GMSD: PyIQA alternative differs by 1.02932e-8, exceeding the selected 1e-14
  bound. Its A/A varied by 13.3%. No canonical performance conclusion. The
  original C helper exists but its current driver has no internal timing output;
  no wall-time substitution was made.
- IW-SSIM: original failure lacked pyrtools. Private probe-only pyrtools 1.0.11
  supplied that dependency; retry then failed on the reference's removed
  `torch.eig` API. No reference arithmetic or environment was altered to bypass it.

## Binding and limits

Rust source: `e2d414c33a6d3c7da50a818fa36e56654d3ebeef`, binary SHA256
`b9279b4af7032a35a11b7bd6da2ddb9d258ec044475ea97eb7d99e69aaf8ea6d`.
This is the available frozen build, **not today's newly integrated master**.
Native libjxl binary SHA256:
`de04cd3180d7ac40e2eac84464fe191ac5ee713201a0a96b393888fe691fb4e2`.
Runtime dispatch and libraries differ between CPUs; do not pool hosts.

The broad diagnostic plan inherited an incorrect HaarPSI baseline revision
label. Actual per-case commands/specifications and raw outputs are retained;
that label must not be used as source provenance. The later large-image plan
corrects the label. Full native/Python runtime closure is unqualified.

Generic transport deliberately retained unchecked outcomes; scalar deltas above
are post-run comparisons with the existing bounds, not retroactive qualification.
The original HaarPSI plan used its existing check command. CPU placement and
resource observations are retained, including dirty observations; taskset plus
requested worker budgets does not prove all inner runtime placement.

Both existing remote heavy locks serialized work. Local native probes used the
local heavy lock. A dead prior local admission reservation was backed up only
after its guard, absent owner PID, and absent owner process group/session were
checked. No foreign processes were signalled. Old failures remain intact.

Resource receipts (8 GiB run-heavy cap, jobs=1): remote broad panel rc2, 54s,
peak RSS 0.90 GiB, min available 13757 MiB, peak load 3.21; local 1MP panel rc0,
31s, peak RSS 0.25 GiB; local Butteraugli 4K retry rc0, 55s, peak RSS 1.91 GiB.
The initial local 4K attempt timed out on Rust Butteraugli; the fresh retry used
45s child timeout and unchanged scoring, rather than dropping the failure.

Private evidence: `/mnt/v/aic2026-evidence-private/favorites-probe-20261019T125611/`.
It retains plans, packet.json.gz, experiment.json, block receipts, selected.csv,
raw scorer logs, score vectors, timing vectors, and failed attempts. The small
transport is `benchmarks/iqa_compete_2026-10-16/probe_pair_arm.py`; its AST was
verified identical to the executed transport apart from the docstring/formatting.
No metric arithmetic, numerical bound, or qualification threshold changed.
