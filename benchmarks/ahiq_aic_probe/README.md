# AHIQ AIC-4 convention probes (2026-10-15)

Upstream-torch probes that locate the convention behind the published AIC-4
`AHIQ` column (`~/tmp/aic-sub-crop/img` crop pairs + `metrics_subset.csv`).

## Resolution (AHIQ fix round)

**The published column is pyiqa's AHIQ pipeline under timm 1.x**, not any
upstream-native convention:

- pyiqa `archs/ahiq_arch.py`: ImageNet mean/std normalisation of `[0,1]`
  tensors, `uniform_crop(crop_size=224, crop_num=20)` —
  `arch_util.uniform_crop` lays a 5x5 grid `(i*(h-224)//4, j*(w-224)//4)`
  and keeps the first 20 positions (the bottom row is dropped),
  deterministic, mean over crops.
- pyiqa builds the ViT by the bare name; under timm 1.x that resolves to
  `vit_base_patch8_224.augreg2_in21k_ft_in1k` — *not* the augreg
  checkpoint upstream's pinned timm 0.5.4 uses. Under augreg2 + imnet +
  grid20 the column reproduces to ~1e-4 on every measured pair; under the
  correct augreg backbone the same grid is +0.06..+0.11 off.
- `ahiq_aic_probe.py` is the canonical probe (upstream `test.py`-path
  scoring: SaveOutput hooks + deform_fusion + Pixel_Prediction, explicit
  backbone tags, `AHIQ_VIT_TAG` env override). It scores every row of a
  supplied CSV under each named convention — `s9`, `stream20`, `s20`,
  `s9_imnet`, `stream20_imnet`, `grid20`, `grid20_imnet` — writes
  `<conv>`/`<conv>_delta` columns and a seed-jitter std block (stream
  conventions only; grid20 is deterministic). `aic_ahiq.tsv` is the
  committed all-61-pair result under the winning convention.
- Verified artifact: `aic_ahiq.tsv`; log
  `~/tmp/nn-port-scratch/ahiq_probe_grid20imnet.log` on i265.

## Pre-fix-round exploration scripts (historical, augreG2 backbone)

The `ahiq_aic_probe_{norm,imnet,imnet2,stream,jitter,grid,g2,p16,fullres,
tile}.py` scripts are the earlier convention hunt. They loaded the bare
timm name and therefore probed the augreg2 checkpoint — the correct
backbone had not been identified yet — and ran upstream's `Full`-style
reimplementation rather than the hook path. Their residual figures
(including the "imnet + stream20 ~0.009" near-match on five pairs) were
coincidental compensation of the wrong backbone and are superseded by the
canonical probe. Kept for the exploration record only; do not run them.

Logs: `~/tmp/nn-port-scratch/ahiq_aic_probe_*.log`, `ahiq_probe_*.log`
on i265.
