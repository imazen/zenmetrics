# AHIQ AIC-4 convention probes (2026-10-15)

Upstream-torch probes that locate the convention behind the published AIC-4
`AHIQ` column (`~/tmp/aic-sub-crop/img` crop pairs + `metrics_subset.csv`).
Each exec's the shared header of `ahiq_aic_probe.py` (timm vit_base_patch8_224
+ resnet50 pretrained backbones + the IIGROUP/AHIQ epoch-32 checkpoint's
deform_fusion/Pixel_Prediction head) and scores one crop convention.

- `ahiq_aic_probe.py` — baseline variants on the native [-1,1] input
  (9-crop, 20 random crops @seed 1919 per-image, resize-to-224, cv2-ish
  bilinear, 5-crop). All ~0.03-0.13 above the column.
- `ahiq_aic_probe_p16.py` — the OTHER checkpoint (AHIQ_vit_p16_epoch28):
  1.3-1.6 raw, wrong model entirely.
- `ahiq_aic_probe_fullres.py` — same baseline set on the fullres images
  (`~/tmp/aic-sub/img`); still ~0.03-0.11 above the fullres column.
- `ahiq_aic_probe_grid.py` / `ahiq_aic_probe_g2.py` — dense uniform grids
  2x2..7x7, flips, (d,r) swap under [-1,1]: score surface flat ~0.60-0.66,
  no crop convention reaches the column.
- `ahiq_aic_probe_norm.py` — input-normalisation sweep on S02: [-1,1] 0.614,
  raw u8 -1.58, **ImageNet 0.487 vs published 0.4836**, BGR 0.619. The
  column was computed on ImageNet-mean/std inputs, not upstream's [-1,1].
- `ahiq_aic_probe_imnet.py` / `ahiq_aic_probe_imnet2.py` — under ImageNet
  norm: per-image reseeded s9/s20, uniform grids, 5/10/18-flip ensembles.
  Published sits between s9 and s20.
- `ahiq_aic_probe_stream.py` — ImageNet norm + upstream default
  `n_ensemble=20` on ONE continuing np.seed(1919) stream in CSV order:
  S02 -0.0003, S03 -0.0024, S01 -0.0115, S04 -0.0092, S05 +0.0093
  (median |d| ~0.009 ~ 0.2 JND). Per-image 20 draws confirmed
  (batch-shared draws are excluded by S02). Best verified convention;
  residual is consistent with unpublished draw-order detail.
- `ahiq_aic_probe_jitter.py` — n_ensemble {10,12,16,20} + seed jitter +
  median-of-20 (started, results in ahiq_aic_probe_jitter.log).
- `ahiq_aic_probe_tile.py` — centred 224-tiling variants (unused result).

Logs: `~/tmp/nn-port-scratch/ahiq_aic_probe_*.log` on dev.
