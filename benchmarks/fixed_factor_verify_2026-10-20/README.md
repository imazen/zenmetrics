# Fixed-factor verification: retained numerical evidence

This is a re-aggregation of already committed measurements, not a new metric run. The owner's fixed-factor definition was numerically exercised against single-expression patched primary sources. No i265 commands, new build, score run, numerical bound change or timing claim is involved.

The original panels use three content pairs, four geometries (64×64, 255×257, 1024×1024, 3840×2160), and 1/28 threads on Ryzen 9 7950X / WSL2. These are not the newly staged 61 AIC2026 840×944 pairs. The aggregate correlations mix these retained geometries and thread observations; they are not a 61-image AIC rank correlation.

## Results

SSIM: 390 scored comparisons pass and 90 size refusals match across twenty names. FSIM/FSIMc: 96 fixed-factor cases, 192 checked scalar values, all pass. VSI with the explicit upstream image-2.20.1 two-M-file resize overlay: 96/96 fixed-factor cases pass. Automatic-factor controls add 48 cases / 72 scalar values, all pass. The table below preserves these scalar values and separately counts refusals.

| Variant | Scored | Matching refusals | Median abs Δ | Max abs Δ | Spearman |
|---|---:|---:|---:|---:|---:|
| `fsim` | 24 | 0 | 3.4636025e-09 | 1.3299275e-06 | 0.99956322 |
| `fsim-k2` | 24 | 0 | 3.0188326e-09 | 4.6791292e-07 | 1 |
| `fsim-k3` | 24 | 0 | 2.7716475e-09 | 1.8109097e-07 | 0.99934505 |
| `fsim-k4` | 24 | 0 | 2.4113337e-09 | 1.384976e-08 | 0.99978154 |
| `fsim-nodownscale` | 24 | 0 | 9.2341608e-09 | 1.3299275e-06 | 0.99956322 |
| `fsimc` | 24 | 0 | 3.335813e-09 | 1.3294678e-06 | 0.99934505 |
| `fsimc-k2` | 24 | 0 | 3.6273775e-09 | 4.6788052e-07 | 0.99978154 |
| `fsimc-k3` | 24 | 0 | 2.6802952e-09 | 1.8116696e-07 | 0.99934505 |
| `fsimc-k4` | 24 | 0 | 2.3901449e-09 | 1.3761492e-08 | 0.99978154 |
| `fsimc-nodownscale` | 24 | 0 | 9.3600589e-09 | 1.3294678e-06 | 0.99934505 |
| `ms-ssim-pyiqa` | 18 | 6 | 9.3465038e-08 | 6.0410393e-07 | 0.99895996 |
| `ms-ssim-pyiqa-f32` | 18 | 6 | 1.0922804e-08 | 3.8032964e-08 | 0.99947957 |
| `ms-ssim-pyiqa-f32-k2` | 12 | 12 | 2.2752564e-09 | 1.0223588e-08 | 0.99821905 |
| `ms-ssim-pyiqa-f32-k3` | 12 | 12 | 8.9438634e-10 | 5.1847843e-09 | 1 |
| `ms-ssim-pyiqa-f32-k4` | 12 | 12 | 3.3579345e-10 | 2.9739777e-09 | 1 |
| `ms-ssim-pyiqa-f32-k5` | 12 | 12 | 2.2317359e-10 | 2.0381685e-09 | 1 |
| `ms-ssim-pyiqa-f32-k6` | 6 | 18 | 1.1709872e-10 | 1.4596903e-09 | 0.9701425 |
| `ssim-pyiqa` | 24 | 0 | 7.3994055e-07 | 1.5453514e-05 | 0.99869138 |
| `ssim-pyiqa-f32` | 24 | 0 | 7.4836021e-08 | 1.4846956e-07 | 0.99890913 |
| `ssim-pyiqa-f32-k2` | 24 | 0 | 2.2647866e-08 | 7.6222859e-08 | 1 |
| `ssim-pyiqa-f32-k3` | 24 | 0 | 9.4108312e-09 | 5.0499528e-08 | 0.99978154 |
| `ssim-pyiqa-f32-k4` | 24 | 0 | 4.984217e-09 | 4.3720012e-08 | 0.99978154 |
| `ssim-pyiqa-f32-k5` | 24 | 0 | 3.3608995e-09 | 3.3145366e-08 | 0.99956322 |
| `ssim-pyiqa-f32-k6` | 18 | 6 | 1.6352756e-09 | 1.8000097e-08 | 0.99947957 |
| `ssimc-pyiqa-f32` | 24 | 0 | 7.9662282e-08 | 1.7274589e-07 | 0.99956322 |
| `ssimc-pyiqa-f32-k2` | 24 | 0 | 2.9812983e-08 | 8.751443e-08 | 0.99934505 |
| `ssimc-pyiqa-f32-k3` | 24 | 0 | 1.5872177e-08 | 5.9953023e-08 | 0.99956322 |
| `ssimc-pyiqa-f32-k4` | 24 | 0 | 1.0208215e-08 | 5.1366634e-08 | 0.99978154 |
| `ssimc-pyiqa-f32-k5` | 24 | 0 | 7.2579231e-09 | 3.9246671e-08 | 0.99890913 |
| `ssimc-pyiqa-f32-k6` | 18 | 6 | 4.0115846e-09 | 2.2675629e-08 | 0.99844115 |
| `vsi` | 24 | 0 | 3.7474697e-09 | 3.1678478e-07 | 1 |
| `vsi-k2` | 24 | 0 | 4.08818e-09 | 3.1571421e-07 | 1 |
| `vsi-k3` | 24 | 0 | 3.8961427e-09 | 2.7397799e-07 | 1 |
| `vsi-k4` | 24 | 0 | 3.5283673e-09 | 2.6648203e-07 | 1 |
| `vsi-nodownscale` | 24 | 0 | 3.8659857e-09 | 3.1678478e-07 | 1 |

## Source and oracle contracts

- [SSIM numerical grid](https://github.com/imazen/zenmetrics/blob/4377e1ad83feb7b1586b11fc5ebe140b187f27ec/benchmarks/port_spotcheck_2026-10-18/primary-pyiqa-ssim.md), [source/data index](https://github.com/imazen/zenmetrics/blob/4377e1ad83feb7b1586b11fc5ebe140b187f27ec/benchmarks/port_spotcheck_2026-10-18/primary-pyiqa-ssim.json) and [authorized policy](https://github.com/imazen/zenmetrics/blob/4377e1ad83feb7b1586b11fc5ebe140b187f27ec/benchmarks/port_spotcheck_2026-10-18/approved-pyiqa-ssim.json). Source fix `970ea199538b6efc64eb109e4db81de90137919d` is an ancestor of the inspected origin/master `4377e1ad83feb7b1586b11fc5ebe140b187f27ec`. The complete `ssim_pyiqa.rs` is byte-identical between those revisions. Corrected MS-SSIM pools each scoring view while evolving the unpooled twofold pyramid.
- [FSIM/VSI numerical grid and resize control](https://github.com/imazen/zenmetrics/blob/4377e1ad83feb7b1586b11fc5ebe140b187f27ec/benchmarks/port_spotcheck_2026-10-18/vsi-resize-runtime.md), [original FSIM index](https://github.com/imazen/zenmetrics/blob/4377e1ad83feb7b1586b11fc5ebe140b187f27ec/benchmarks/port_spotcheck_2026-10-18/runner-image212.json), [VSI overlay index](https://github.com/imazen/zenmetrics/blob/4377e1ad83feb7b1586b11fc5ebe140b187f27ec/benchmarks/port_spotcheck_2026-10-18/runner-image2201.json), [automatic controls](https://github.com/imazen/zenmetrics/blob/4377e1ad83feb7b1586b11fc5ebe140b187f27ec/benchmarks/port_spotcheck_2026-10-18/primary-automatic.json). VSI and the measured FSIM RGB scalar kernels retain their measured arithmetic; later FSIM changes add a configurable luma path. This report does not execute that added route.
- [Oracle single-expression patch contracts](../iqa_compete_2026-10-16/fixed-factor-oracle.md) and [PyIQA executable adapter](../iqa_compete_2026-10-16/probe_pyiqa_ssim.py). PyIQA is 0.1.16; only `f` is patched, with `downsample=True`. Author FeatureSIM.m/VSI.m replace only automatic `F` with k. FSIM is primary MATLAB code under Octave; PyIQA FSIM/VSI diagnostic substitutes are not used.

Measured Rust source: FSIM/VSI `e2d414c33a6d3c7da50a818fa36e56654d3ebeef`, binary SHA256 `b9279b4af7032a35a11b7bd6da2ddb9d258ec044475ea97eb7d99e69aaf8ea6d`; corrected SSIM candidate `970ea199538b6efc64eb109e4db81de90137919d`, binary SHA256 `a438615020db8b0fb7ef86f61c74c992a354942ee10c3f79f72a1efb2eb1a36b`. Input manifest SHA256 `83d61ef3ce1877a20e07a2c8f8ac0df0a761a0a77903a5784b011a74b17f0712`.

Author source SHA256: FeatureSIM.m `4b23dc2881c71dfa996e40fd84b9b946f3934b36ecfb3ee4dfbafa5505bcbba9`; VSI.m `5939b9053966f58377e08052b5e2d0e0e3f954040fe732625591363285433954`; PyIQA ssim_arch.py `a0eb8e0bedfae99d9d014371a77b7d9afb110c71a22b6cda06a4992da0d951e8`.

## Bounds and limitations

Existing absolute inclusive bounds remain: unrounded SSIM-family 1e-6; rounded SSIM/MS-SSIM 5e-5; FSIM/FSIMc 5e-5; VSI 1e-4. This does not establish 1e-9 relative agreement. The retained PyIQA Gaussian is formed in float32 and promoted; selected precision controls explain differences without replacing the primary outputs. The historical VSI image-2.12 runner has 14/96 failures, maximum 0.010004985003426614, and remains retained. Selecting the unchanged image-2.20.1 resize files resolves those observations without changing metric code or gates.

The new 61-pair panel and a newly built current-master binary are not verified here. At 840×944, fixed-factor MS-SSIM k5/k6 cannot supply an 11×11 final scoring view under the current unchanged contract; an executable rerun must count matching refusals separately. No map or performance qualification is claimed.

## Reproduction and artifact integrity

The original commands, exact execution/source hashes and resource receipts remain in [SSIM provenance](https://github.com/imazen/zenmetrics/blob/4377e1ad83feb7b1586b11fc5ebe140b187f27ec/benchmarks/port_spotcheck_2026-10-18/primary-pyiqa-ssim.meta) and [FSIM/VSI provenance](https://github.com/imazen/zenmetrics/blob/4377e1ad83feb7b1586b11fc5ebe140b187f27ec/benchmarks/port_spotcheck_2026-10-18/primary-fixed-panel.meta). Existing recipes live in `benchmarks/iqa_compete_2026-10-16/oracle.just`: `pyiqa-ssim-panel`, `octave-panel`, `octave-image-source-panel`, and `octave-automatic-panel` (consult their arguments there). This aggregation verifies each indexed shard SHA256, verifies equality to the inspected origin/master objects, recomputes per-variant deltas and inclusive gates, and computes Spearman from average ranks with ties. Source bindings are in [provenance.json](provenance.json).
