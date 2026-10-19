# Fixed-factor source audit

Observed 2026-10-19. Source baseline:
[`e2d414c33a6d3c7da50a818fa36e56654d3ebeef`](https://github.com/imazen/zenmetrics/tree/e2d414c33a6d3c7da50a818fa36e56654d3ebeef).
FSIM comparison branch:
[`c9f98a987cd2f0fae8f7f2033770a41bb057f3ea`](https://github.com/imazen/zenmetrics/tree/c9f98a987cd2f0fae8f7f2033770a41bb057f3ea).
PyIQA reference: 0.1.16, source commit
[`18dd7a19694e94aac21019170e3f5e63d6b4e19e`](https://github.com/chaofengc/IQA-PyTorch/blob/18dd7a19694e94aac21019170e3f5e63d6b4e19e/pyiqa/archs/ssim_arch.py).
The [source patcher](../iqa_compete_2026-10-16/fixed-factor-oracle.md)
binds the exact reference bodies. This is a source audit, not numerical
qualification. Existing metric paths on master remain unchanged.

| CLI names | Current placement of fixed factor | Source disposition |
|---|---|---|
| `fsim-nodownscale`, `fsim-k2`, `fsim-k3`, `fsim-k4` | `Fsim::with_downscale(Factor(k))` selects the same RGB/opponent decimation path as Auto. | Factor selection matches the requested substitution structurally; full primary-reference numeric grid remains missing. |
| `vsi-nodownscale`, `vsi-k2`, `vsi-k3`, `vsi-k4` | `Vsi::with_downscale(Factor(k))`; SDSP remains on original inputs, then the same decimator processes opponent and saliency planes. | Factor selection matches the requested substitution structurally; full primary-reference numeric grid remains missing. |
| `ssim-pyiqa-f32-k2`, `ssim-pyiqa-f32-k3`, `ssim-pyiqa-f32-k4`, `ssim-pyiqa-f32-k5`, `ssim-pyiqa-f32-k6` | One complete-block average pool before the single-scale statistics call. | Pool placement matches patched PyIQA `ssim(..., downsample=True)` structurally; preprocessing and numerical parity remain separate checks. |
| `ssimc-pyiqa-f32-k2`, `ssimc-pyiqa-f32-k3`, `ssimc-pyiqa-f32-k4`, `ssimc-pyiqa-f32-k5`, `ssimc-pyiqa-f32-k6` | The same single-scale pool applied independently to the three channels. | Same source disposition as luma SSIM; no numerical qualification inferred. |
| `ms-ssim-pyiqa-f32-k2`, `ms-ssim-pyiqa-f32-k3`, `ms-ssim-pyiqa-f32-k4`, `ms-ssim-pyiqa-f32-k5`, `ms-ssim-pyiqa-f32-k6` | One initial fixed pool, followed by the five-level pyramid on that pooled image. | **Different placement from patched PyIQA:** the reference pools the scoring view at each level while evolving the unpooled level image separately. An unmerged correction and regression checks remain required. |

The five unsuffixed SSIM-family names (`ssim-pyiqa`, `ssim-pyiqa-f32`,
`ssimc-pyiqa-f32`, `ms-ssim-pyiqa`, `ms-ssim-pyiqa-f32`) select factor 1.
PyIQA defaults to `downsample=False`; these paths must retain their existing
output and code paths when fixed variants are corrected.

## MS-SSIM correction boundary

In `crates/zenmetrics-cli/src/metrics/ssim_pyiqa.rs`, `score` currently calls
`boxdown` before its scale loop. The primary PyIQA `ms_ssim` calls `ssim` at
every level; the patched factor is local to that call. Its outer `pool2d(2)`
evolves the original level image, not the factor-pooled scoring view.
The factor therefore does **not** compound as k to the number of levels.

The initial-pool and per-level-view arrangements differ at odd boundaries;
their size checks also follow different floor/ceiling sequences. A correction
must validate `floor(level_dimension / k)` at each scored level, pool only
that level's statistics input, and advance the original level through the
unchanged twofold pyramid pool. Retain all other preprocessing, constants,
weights, gates and unsuffixed behavior. This is internal implementation work,
without a new public API, on an unmerged `fix/ssim-pyiqa-fixed-factor` branch.

The reviewed source patcher accepts k1–k4. k5/k6 are listed because they are
real CLI names; this audit does not claim an independent patched oracle for
those factors. External parity gates for these SSIM names are still unbound
in the source census. Neither observed error nor a configured harness default
may substitute for an existing documented gate.

FSIM's existing unmerged branch preserves `Factor(k)` in both RGB and luma-only
paths. It has not been rewritten or merged by this audit. No new VSI/SSIM
correction branch, before/after numeric result or base-name bit-parity result
is claimed here. The successful 64×64 canonical factor executions are recorded
separately in [the Octave adapter record](../iqa_compete_2026-10-16/octave-primary.md).
