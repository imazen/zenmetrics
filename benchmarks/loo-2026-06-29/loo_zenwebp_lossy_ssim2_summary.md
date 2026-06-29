# Fleet-LOO feature ablation — zenwebp_lossy / ssim2

- config: `zenwebp_picker`  picker_target: `ssim2_a`  seed: 12345  features: 97  pareto_cells: 2526165
- baseline overhead: val 2.510%  test 2.530%
- ranked on **val** delta; keep-threshold 0.05pp
- single-feature variants: 94  (must-keep 26 / droppable-looking 68)

## Top features that MATTER (highest LOO importance)

| rank | feature | val Δpp | test Δpp |
|---|---|---|---|
| 1 | feat_colourfulness | +0.250 | -0.130 |
| 2 | feat_alpha_used_fraction | +0.180 | +0.040 |
| 3 | feat_alpha_bimodal_score | +0.180 | +0.040 |
| 4 | feat_noise_floor_uv_p75 | +0.150 | +0.010 |
| 5 | feat_aq_map_p90 | +0.140 | +0.030 |
| 6 | feat_noise_floor_y_p5 | +0.130 | +0.220 |
| 7 | feat_aspect_min_over_max | +0.130 | +0.120 |
| 8 | feat_chroma_luma_covariance_cr | +0.110 | +0.040 |
| 9 | feat_info_weight_mean | +0.100 | -0.030 |
| 10 | feat_orientation_energy_ratio | +0.100 | -0.050 |
| 11 | feat_quant_survival_uv_p75 | +0.090 | -0.020 |
| 12 | feat_channel_count | +0.090 | +0.070 |
| 13 | feat_luma_kurtosis | +0.090 | +0.150 |
| 14 | feat_is_grayscale | +0.080 | +0.040 |
| 15 | feat_palette_log2_size | +0.080 | +0.170 |
| 16 | feat_block_misalignment_8 | +0.080 | -0.040 |
| 17 | feat_cr_horiz_sharpness | +0.070 | +0.200 |
| 18 | feat_uniformity | +0.060 | +0.110 |
| 19 | feat_pixel_count | +0.060 | +0.040 |
| 20 | feat_laplacian_variance_p1 | +0.060 | +0.190 |

## Droppable-looking tail (single-LOO ~0 — MUST be joint-verified, not dropped on single-LOO alone)

```
feat_laplacian_variance_p50 feat_min_dim feat_block_misalignment_32 feat_luma_histogram_entropy feat_aq_map_std feat_bitmap_bytes feat_quant_survival_uv feat_noise_floor_uv_p50 feat_quant_survival_uv_p25 feat_variance_spread feat_grayscale_score feat_log_pixels feat_aq_map_p5 feat_laplacian_variance_p90 feat_gradient_fraction feat_variance feat_flat_color_block_ratio feat_cr_peak_sharpness feat_edge_density feat_aq_map_p1 feat_cb_vert_sharpness feat_aq_map_mean feat_aq_map_p75 feat_laplacian_variance feat_dct_compressibility_y feat_noise_floor_y_p50 feat_alpha_present feat_aq_map_p10 feat_info_weight_p90 feat_log_padded_pixels_16 feat_aq_map_p99 feat_patch_fraction_fast feat_max_dim feat_quant_survival_y feat_chroma_complexity feat_cr_sharpness feat_noise_floor_y feat_noise_floor_uv feat_log_aspect_abs feat_chroma_luma_covariance_cb feat_noise_floor_y_p75 feat_laplacian_variance_p75 feat_noise_floor_y_p1 feat_quant_survival_uv_p50 feat_noise_floor_y_p25 feat_dct_compressibility_uv feat_laplacian_variance_peak feat_noise_floor_uv_p90 feat_quant_survival_uv_p10 feat_distinct_color_bins feat_noise_floor_y_p90 feat_laplacian_variance_p99 feat_patch_fraction feat_aq_map_p95 feat_noise_floor_y_p10 feat_high_freq_energy_ratio feat_quant_survival_y_p1 feat_noise_floor_uv_p25 feat_aq_map_p50 feat_skin_tone_fraction feat_quant_survival_y_p5 feat_log_padded_pixels_32 feat_laplacian_variance_p5 feat_cb_peak_sharpness feat_palette_fits_in_256 feat_quant_survival_y_p50 feat_laplacian_variance_p10 feat_quant_survival_y_p10
```
