# Port speed versus canonical implementations

Rebenchmark target: DSSIM PR 199, exact source
`22e9ab033d5e5ac41ee961de00e7ae9bd959b634` (pin change `28742d06`).
The retained measurements below use the earlier frozen build containing
`331b56cf`; their results are not relabeled as measurements of the new pin.

Snapshot: **2026-10-20 00:35 UTC**. CPU: **Ryzen 9 7950X / WSL2**.

**Rust/C** is Rust time divided by canonical time; lower is faster. Ratios are approximate descriptive warm comparisons. Sizes and thread counts are shown per row. The observations do not establish complete size/content/thread coverage or qualified regression intervals.

| Coverage | Count |
| --- | ---: |
| CLI routes in the catalog, including named variants | 126 |
| Routes with fresh retained comparison observations | 46 |
| Routes with measured comparable timing ratios | 45 |
| DSSIM native Rust baseline | 1.0× |
| Routes without a fresh comparison in this snapshot | 80 |

**DSSIM is natively Rust. Its canonical native-Rust baseline is 1.0×.** This baseline is a reference convention. The separately timed C-API adapter retains a parser error in the evidence ledger.

[Comparison ledger](benchmarks/canonical_recheck_2026-10-19/README.md) · [Exact-name observations](benchmarks/canonical_recheck_2026-10-19/results.json) · [Cohort/input/source bindings](benchmarks/canonical_recheck_2026-10-19/cohorts.json) · [Full references, papers and model terms](docs/MODEL_CARDS.md)

Frozen metric source: `efae6b8fed9e57346157d92dffc3cdb57c44560e`. Observation-file SHA256 for this snapshot: `93b1688231dfef350a713dced476bb48d76e14485b0780782b607047944f5661`.

`pass` uses an assigned existing bound; a displayed Δ has no assigned acceptance bound. Fractions retain partial coverage. `prior` denotes older recorded evidence. A dash means no fresh comparable ratio in this snapshot.

## Current ratio table

| Metric | Rust/C | Pixels | Threads | Check |
| --- | ---: | --- | ---: | --- |
| `brisque` | 0.119× | 512×512 | 1 | 170/170 pass |
| `cambi` | 1× | 512×512 | 1 | Δ 0 |
| `cambi-aic2026` | 0.676× | 512×512 | 1 | Δ 0 |
| `dssim` | **1.0×** | baseline | — | native Rust canonical |
| `flip` | 0.334× | 512×512 | 1 | Δ 3.6e-14 |
| `flip-v17` | 0.336× | 512×512 | 1 | Δ 0 |
| `fsim` | 0.219× | 255×257 | 1 | 6/6 pass |
| `fsim-k2` | 0.18× | 255×257 | 1 | 6/6 pass |
| `fsim-k3` | 0.158× | 255×257 | 1 | 6/6 pass |
| `fsim-k4` | 0.141× | 255×257 | 1 | 6/6 pass |
| `fsim-nodownscale` | 0.211× | 255×257 | 1 | 6/6 pass |
| `haarpsi` | 0.121× | 255×257 | 1 | 6/6 pass |
| `haarpsi-y` | 0.133× | 255×257 | 1 | 6/6 pass |
| `mdsi` | 0.209× | 255×257 | 1 | 6/6 pass |
| `ms-ssim-pyiqa-f32` | 0.0139× | 1152×1152 | 1 | 1/3 pass |
| `msssim` | 0.0278× | 255×257 | 1 | 6/6 pass |
| `msssim-libvmaf` | 0.354× | 512×512 | 1 | Δ 1.5e-07 |
| `niqe` | 0.182× | 512×512 | 1 | 170/170 pass |
| `piqe` | 0.103× | 512×512 | 1 | 170/170 pass |
| `psnr` | 0.207× | 512×512 | 1 | Δ 0 |
| `psnr-y` | 0.142× | 512×512 | 1 | Δ 2e-12 |
| `psnr-y-libvmaf` | 0.762× | 512×512 | 1 | Δ 0 |
| `psnr-y601` | 0.145× | 512×512 | 1 | Δ 2.4e-12 |
| `ssim-libvmaf` | 0.362× | 512×512 | 1 | Δ 6e-08 |
| `ssim-pyiqa-f32` | 0.0096× | 3840×2160 | 1 | 2/6 pass |
| `ssim-pyiqa-f32-k2` | 0.0252× | 1152×1152 | 1 | 1/3 pass |
| `ssim-pyiqa-f32-k3` | 0.0413× | 1152×1152 | 1 | 1/3 pass |
| `ssim-pyiqa-f32-k4` | 0.0674× | 1152×1152 | 1 | 1/3 pass |
| `ssim-pyiqa-f32-k5` | 0.16× | 1152×1152 | 1 | 1/3 pass |
| `ssim-pyiqa-f32-k6` | 0.28× | 1152×1152 | 1 | 1/3 pass |
| `ssim2` | 0.585× | 512×512 | 1 | Δ 0 |
| `ssimc-pyiqa-f32` | 0.0133× | 1152×1152 | 1 | 1/3 pass |
| `ssimulacra1` | 1.12× | 512×512 | 1 | Δ 3.8e-09 |
| `vif` | 0.0762× | 255×257 | 1 | 6/6 pass |
| `vifvec` | 0.0956× | 255×257 | 1 | 6/6 pass |
| `vmaf` | 0.885× | 512×512 | 1 | 2/2 pass |
| `vmaf-4k` | 0.895× | 512×512 | 1 | Δ 0 |
| `vmaf-neg` | 0.873× | 512×512 | 1 | 2/2 pass |
| `vmaf-neg-studio601` | 0.945× | 512×512 | 1 | 2/2 pass |
| `vmaf-studio601` | 0.911× | 512×512 | 1 | 2/2 pass |
| `vmaf-v1` | 0.743× | 512×512 | 1 | Δ 0.00023 |
| `vsi` | 0.0776× | 255×257 | 1 | 6/6 pass |
| `vsi-k2` | 0.0676× | 255×257 | 1 | 6/6 pass |
| `vsi-k3` | 0.0629× | 255×257 | 1 | 6/6 pass |
| `vsi-k4` | 0.0622× | 255×257 | 1 | 5/6 pass |
| `vsi-nodownscale` | 0.078× | 255×257 | 1 | 6/6 pass |

## Speed findings

| Metric | Rust/C | Scope | Status |
| --- | ---: | --- | --- |
| `ssimulacra1` | 1.12× | 512×512, 1 thread | descriptive flag above 1.08; interval qualification pending |

## Remaining fresh comparisons

| Metric | Rust/C | Existing evidence |
| --- | ---: | --- |
| `afine` | — | prior |
| `ahiq` | — | prior |
| `ahiq-pyiqa` | — | prior |
| `arniqa` | — | prior |
| `butteraugli` | — | unchecked |
| `butteraugli-gpu` | — | unchecked |
| `butteraugli-original` | — | unchecked |
| `ciede2000` | — | unchecked |
| `ckdn` | — | prior |
| `cvvdp` | — | prior |
| `cvvdp-gpu` | — | unchecked |
| `cw-ssim` | — | unchecked |
| `cw-ssim-pyiqa` | — | prior |
| `deepdc` | — | prior |
| `dists` | — | prior |
| `dmm` | — | partial |
| `dmm-pyiqa` | — | prior |
| `dreamsim` | — | prior |
| `dssim-gpu` | — | unchecked |
| `fsim-y` | — | unchecked |
| `gmsd` | — | prior |
| `hdrvdp` | — | prior |
| `hdrvdp-aic2026` | — | unchecked |
| `hdrvdp3` | — | prior |
| `hdrvdp3-aic2026` | — | unchecked |
| `hyperiqa` | — | prior |
| `iwssim` | — | unchecked |
| `iwssim-gpu` | — | unchecked |
| `iwssim-piq` | — | prior |
| `liqe` | — | prior |
| `lpips-alex` | — | prior |
| `lpips-squeeze` | — | prior |
| `lpips-vgg` | — | prior |
| `mad` | — | unchecked |
| `maniqa` | — | prior |
| `mdctpsnr` | — | prior |
| `ms-gmsd` | — | unchecked |
| `ms-gmsdc` | — | unchecked |
| `ms-ssim-pyiqa` | — | unchecked |
| `ms-ssim-pyiqa-f32-k2` | — | unchecked |
| `ms-ssim-pyiqa-f32-k3` | — | unchecked |
| `ms-ssim-pyiqa-f32-k4` | — | unchecked |
| `ms-ssim-pyiqa-f32-k5` | — | unchecked |
| `ms-ssim-pyiqa-f32-k6` | — | unchecked |
| `musiq` | — | prior |
| `nlpd` | — | prior |
| `nlpd-iqa` | — | prior |
| `paq2piq` | — | prior |
| `pieapp` | — | excluded by owner |
| `pieapp-pyiqa` | — | excluded by owner |
| `proposal-butteraugli` | — | unchecked |
| `psnr-cb` | — | unchecked |
| `psnr-cr` | — | unchecked |
| `psnr-y-studio601` | — | unchecked |
| `psnr-ycbcr611` | — | unchecked |
| `psnrhvs` | — | prior |
| `psnrhvs-daala` | — | prior |
| `psnrhvs-y` | — | unchecked |
| `ssim` | — | unchecked |
| `ssim-pyiqa` | — | unchecked |
| `ssim2-gpu` | — | unchecked |
| `ssimc` | — | unchecked |
| `ssimc-pyiqa-f32-k2` | — | unchecked |
| `ssimc-pyiqa-f32-k3` | — | unchecked |
| `ssimc-pyiqa-f32-k4` | — | unchecked |
| `ssimc-pyiqa-f32-k5` | — | unchecked |
| `ssimc-pyiqa-f32-k6` | — | unchecked |
| `stlpips-alex` | — | prior |
| `stlpips-vgg` | — | prior |
| `topiq-fr` | — | gate |
| `topiq-fr-pipal` | — | gate |
| `wadiqam-fr-live` | — | prior |
| `wadiqam-fr-tid` | — | prior |
| `wadiqam-nr-live` | — | prior |
| `wadiqam-nr-tid` | — | prior |
| `wadiqam-pyiqa` | — | prior |
| `wd` | — | prior |
| `ws-hv` | — | unchecked |
| `zensim` | — | in-house metric; no independent canonical |
| `zensim-gpu` | — | unchecked |

## Coverage first: 15-minute work order

Complete the first comparison for each uncovered name before adding more content pairs, thread budgets or optimization experiments. Preserve running acquisition blocks and their frozen bindings; make scheduling changes between blocks.

| Budget | Work |
| --- | --- |
| 0–2 min | Harvest completed observations and select uncovered names from the prepared plans. |
| 2–12 min | One shared pair, one thread, three alternating blocks and three samples per new name; retain 1152² where k6 requires it. Serialize heavy execution. |
| 12–15 min | Publish completed rows and explicit blockers; use remaining time for small adapter/parser fixes that recover a row. |

| Prepared queue | Additional names |
| --- | ---: |
| PyIQA SSIM-family variants | 12 |
| Classical/native references | 15 |
| Original MATLAB references | 5 |
| Unique additional names | 32 |

These prepared queues offer a conditional target of **78/126** fresh comparison rows if all execute successfully. This is a scheduling target; execution time and successful completion remain unmeasured. Profile and optimize after the breadth pass, except for a small blocker fix that directly recovers coverage.

## Repeat the comparative benchmarks

Prepare a plan with explicit installed references and input hashes, then run:

```sh
just bench-iqa-canonical PLAN.json NEW_RESULTS_DIRECTORY --samples 3
```

Use `--metrics NAME,...` to select a named subset of a prepared plan. The [comparison ledger](benchmarks/canonical_recheck_2026-10-19/README.md#reproduce) documents plan preparation and evidence publication.
