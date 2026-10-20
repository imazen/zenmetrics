# Native resumable AIC scoring qualification (2026-10-20)

The frozen native-qualified rescorer runs the ten selected columns on all 61 staged RGB8 840×944 AIC crop pairs: **610 finite scores**, with all pre/post-run input/model/executable identity checks passing. This is a 61-pair qualification panel, not a completed 9,618-pair campaign. The [full-precision CSV](native-aic61.csv) contains only our calculated cells and copied metadata; other input metric columns are empty. No published score is substituted for a missing calculation.

## Resume proof

1. A bounded one-pair invocation checkpoints ten values and exits 2 with 600 cells pending.
2. The identical command without `--limit-pairs` executes only the remaining 60 pairs and exits 0 with 610 scored cells.
3. A further identical invocation exits 0, adds no attempts, and leaves all scalar checkpoint values unchanged. There are 61 pair checkpoints and exactly 122 adapter attempts: one compare call plus one feature-helper call per pair.

Every CSV value parses back to the exact binary64 scalar stored in its checkpoint. Completed columns are not rerun when another column fails. The OS-backed controls exercise real interruption/owned-child cleanup, concurrent-writer and live-orphan refusal, asset/binary/input drift, corrupt receipts, relocated identical files, view selection, explicit JND fits, and nonfinite/ambiguous or unhealthy-child rejection. The current 23 Python controls pass. The native precision helper additionally establishes exact JSON versus score/compare/batch text parity for PSNR, FLIP-v17 and WS-HV. The Rust compare-output unit control covers ordinary, tiny and large binary64 values.

## Published-table observations

`HDR_VDP_2` uses native `hdrvdp-aic2026` Q, and `HDR_VDP_3` uses `hdrvdp3-aic2026` Q. The source comparison table below measures raw absolute differences against the supplied drop cropped cells, which agree with July on these shared columns. It assigns **no new numerical bound** and does not replace existing author-reference qualification. WS-HV remains a native implementation with an unresolved exact published convention. ADM2 and VIF scales use the full-precision VMAF feature helper in a separate per-pair decode; all regular metrics share the compare call's decoded bitmaps.

| Column | n | Median abs Δ | Max abs Δ |
|---|---:|---:|---:|
| `ADM2` | 61 | 1.7254051446258245e-05 | 0.0011669352572181158 |
| `HDR_VDP_2` | 61 | 0.02777679661814858 | 0.22998204858633642 |
| `HDR_VDP_3` | 61 | 0.0013692098323154767 | 0.012360806855933326 |
| `PSNR` | 61 | 7.9061490616538777e-05 | 0.0012816242561370927 |
| `VIF-vmaf-scale0` | 61 | 0.0026614194755554266 | 0.012330098644256604 |
| `VIF-vmaf-scale1` | 61 | 0.00039695586395260829 | 0.0041535836944580629 |
| `VIF-vmaf-scale2` | 61 | 0.00019354322052000938 | 0.0041164167861938994 |
| `VIF-vmaf-scale3` | 61 | 0.00014791741943354086 | 0.003888785919189508 |
| `VIF_vmaf` | 61 | 4.2286022186477368e-05 | 0.009415630849838319 |
| `WS-HV` | 61 | 0.00078221174740078503 | 0.0040969276943787492 |

## Identity and reproduction

Host: Ryzen 9 7950X / WSL2. Requested thread environment: eight per process. Source build snapshot `48ef93edd591` is master `0578ad29` plus the full-precision export patch; that patch landed as `f8ae4743`. No metric arithmetic changed. This source label records the actual local snapshot rather than claiming the build happened after its later landing.

Use [the runner contract](../../docs/AIC_RESCORE.md) and `just aic-rescore-check`. Exact local commands, stdout/stderr, source-bound plan and complete checkpoint attempts remain under `$HOME/tmp/ffverify/rescore/`. The full panel used the frozen runner SHA256 below, preserved locally as `runner-native-qualified.py`. Subsequently added guards refuse unowned existing output files and preserve foreign staging-file collisions. Those guards pass all 23 controls and a current-runner native PSNR smoke invocation (one scored pair, 60 pending, hashes verified); the full 610-cell panel was not repeated after these IO-only changes. CSV projection refresh is independent of the immediately durable per-pair scalar checkpoints.

```sh
just aic-rescore run --plan hdr-union-plan.json \
  --binary BUILD/zenmetrics --vmaf-features BUILD/examples/aic_vmaf_features \
  --source-commit 48ef93edd591 --output hdr-final-scores --threads 8 \
  --limit-pairs 1
# Remove --limit-pairs for completion; repeat again to verify no-op resume.
```

SHA256 bindings:

| Artifact | SHA256 |
|---|---|
| Frozen native-qualified runner | `d0579f0a904a7362bf89de2940ec500f44bad17eb2a7521f3bf22e4847208831` |
| Current runner (ownership guards) | `27a45e4bd4f0386429f6bf49d9f798c169b1fffc8e47e80033ed401c973c6ed0` |
| Plan | `1640eded9efab3a1053cc3daf0bf9eab079ba65a0ac0f8c6a02108188ea3fb24` |
| Selected published CSV | `3fae4552b1b56397a83fcd4cd27f94cdb05efc335b9f53b3cd33287748b6fb44` |
| Final run identity | `5a61e19e94fc301c7a71ed23c67e0249b6e7c163ed969010bfb6247e2c9107b7` |
| Native output CSV | `5254fe7d1e2e85768a197ff5d5f2e926a9c70b5d24aa474ab0b9abbb9ed64d32` |
| Comparison TSV | `f6697f5b1fd8b505586b2ba624f38131f48db7e9b467c679f79a9ddfefea263d` |
| CLI binary | `bc7bee27d02fcb1ddbe4479222f3cc8fe196872f6634fa98d516516b152dab6f` |
| Feature helper | `9f5291b35b58b399fa340a3e4a4cf9a381058a45d2a086bbe2661e934e39cc44` |

Whole-step resource receipts (not per-metric performance claims):

- `hdr-final-first.log`: `run-heavy: done rc=2 2s | peak-RSS 0.89GiB | min-avail 11449MiB | peak-load 2.10`
- `hdr-final-resume.log`: `run-heavy: done rc=0 68s | peak-RSS 0.93GiB | min-avail 9839MiB | peak-load 5.57`
- `hdr-final-noop.log`: `run-heavy: done rc=0 1s | peak-RSS 0.02GiB | min-avail 11432MiB | peak-load 3.09`
- `precision-control.log`: `run-heavy: done rc=0 1s | peak-RSS 0.12GiB | min-avail 8018MiB | peak-load 2.19`

Strict CLI Clippy was attempted on the checked source. It is blocked by two pre-existing `clippy::assign_op_pattern` errors in `zenmetrics-math/src/glibc.rs` at lines 158 and 177. This change does not alter that arithmetic or relax the strict check. The precision unit/native controls and rescorer controls pass independently. Full diagnostic log: `$HOME/tmp/ffverify/rescore/clippy.log`.
