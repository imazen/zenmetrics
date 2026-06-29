# Fleet-LOO feature ablation — zenjxl_lossy / ssim2

- config: `zenjxl_lossy_dense`  picker_target: `ssim2_a`  seed: 12345  features: 97  pareto_cells: 1416555
- baseline overhead: val 3.910%  test 4.030%
- ranked on **val** delta; keep-threshold 0.05pp
- single-feature variants: 97  (must-keep 74 / droppable-looking 23)

## Top features that MATTER (highest LOO importance)

| rank | feature | val Δpp | test Δpp |
|---|---|---|---|
| 1 | feat_aq_map_mean | +1.220 | +1.570 |
| 2 | feat_luma_histogram_entropy | +1.020 | +0.960 |
| 3 | feat_quant_survival_uv_p25 | +0.920 | +1.200 |
| 4 | feat_quant_survival_y_p10 | +0.870 | +0.930 |
| 5 | feat_cb_vert_sharpness | +0.820 | +0.880 |
| 6 | feat_alpha_present | +0.810 | +0.010 |
| 7 | feat_quant_survival_y_p50 | +0.810 | +0.610 |
| 8 | feat_gradient_fraction_smooth | +0.780 | +0.360 |
| 9 | feat_aq_map_p90 | +0.780 | +0.690 |
| 10 | feat_log_padded_pixels_16 | +0.780 | +0.510 |
| 11 | feat_distinct_color_bins | +0.730 | +0.890 |
| 12 | feat_max_dim | +0.730 | +0.260 |
| 13 | feat_noise_floor_uv_p75 | +0.720 | +0.570 |
| 14 | feat_log_aspect_abs | +0.700 | +0.120 |
| 15 | feat_laplacian_variance_p1 | +0.670 | +0.230 |
| 16 | feat_high_freq_energy_ratio | +0.660 | +0.650 |
| 17 | feat_laplacian_variance_p99 | +0.610 | +0.400 |
| 18 | feat_noise_floor_y_p10 | +0.610 | +0.510 |
| 19 | feat_noise_floor_y_p25 | +0.590 | +0.890 |
| 20 | feat_info_weight_p90 | +0.550 | +0.230 |

## Droppable-looking tail (single-LOO ~0 — MUST be joint-verified, not dropped on single-LOO alone)

```
feat_uniformity feat_laplacian_variance_p50 feat_flat_color_block_ratio feat_edge_slope_stdev feat_quant_survival_y_p75 feat_cb_horiz_sharpness feat_noise_floor_y_p75 feat_laplacian_variance_p75 feat_chroma_luma_covariance_cb feat_noise_floor_y_p1 feat_aq_map_p50 feat_grayscale_score feat_luma_kurtosis feat_dct_compressibility_y feat_quant_survival_y_p25 feat_info_weight_mean feat_bitmap_bytes feat_chroma_luma_covariance_cr feat_gradient_fraction feat_patch_fraction_fast feat_chroma_complexity feat_cr_vert_sharpness feat_palette_fits_in_256
```
