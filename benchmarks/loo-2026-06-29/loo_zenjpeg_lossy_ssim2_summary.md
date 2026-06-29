# Fleet-LOO feature ablation — zenjpeg_lossy / ssim2

- config: `zenjpeg_picker`  picker_target: `ssim2_a`  seed: 12345  features: 97  pareto_cells: 3806550
- baseline overhead: val 8.230%  test 7.880%
- ranked on **val** delta; keep-threshold 0.05pp
- single-feature variants: 83  (must-keep 5 / droppable-looking 78)

## Top features that MATTER (highest LOO importance)

| rank | feature | val Δpp | test Δpp |
|---|---|---|---|
| 1 | feat_laplacian_variance_p5 | +0.220 | +0.050 |
| 2 | feat_min_dim | +0.210 | +0.590 |
| 3 | feat_gradient_fraction_smooth | +0.160 | +0.260 |
| 4 | feat_luma_kurtosis | +0.110 | +0.250 |
| 5 | feat_cb_peak_sharpness | +0.060 | +0.240 |
| 6 | feat_quant_survival_y | -0.060 | -0.080 |
| 7 | feat_cr_sharpness | -0.070 | +0.080 |
| 8 | feat_distinct_color_bins | -0.080 | +0.000 |
| 9 | feat_dct_compressibility_uv | -0.100 | -0.170 |
| 10 | feat_channel_count | -0.100 | -0.220 |
| 11 | feat_variance | -0.110 | -0.460 |
| 12 | feat_noise_floor_y_p50 | -0.120 | -0.070 |
| 13 | feat_cb_vert_sharpness | -0.120 | -0.080 |
| 14 | feat_pixel_count | -0.120 | -0.090 |
| 15 | feat_noise_floor_y_p90 | -0.140 | -0.040 |
| 16 | feat_bitmap_bytes | -0.140 | -0.110 |
| 17 | feat_laplacian_variance_p10 | -0.160 | -0.210 |
| 18 | feat_cb_horiz_sharpness | -0.170 | -0.030 |
| 19 | feat_aq_map_p1 | -0.190 | -0.070 |
| 20 | feat_log_padded_pixels_8 | -0.200 | -0.130 |

## Droppable-looking tail (single-LOO ~0 — MUST be joint-verified, not dropped on single-LOO alone)

```
feat_quant_survival_y feat_cr_sharpness feat_distinct_color_bins feat_dct_compressibility_uv feat_channel_count feat_variance feat_noise_floor_y_p50 feat_cb_vert_sharpness feat_pixel_count feat_noise_floor_y_p90 feat_bitmap_bytes feat_laplacian_variance_p10 feat_cb_horiz_sharpness feat_aq_map_p1 feat_log_padded_pixels_8 feat_aq_map_mean feat_noise_floor_y feat_noise_floor_y_p1 feat_laplacian_variance_p1 feat_colourfulness feat_aq_map_p50 feat_palette_log2_size feat_chroma_luma_covariance_cr feat_cr_horiz_sharpness feat_block_misalignment_8 feat_aq_map_std feat_info_weight_mean feat_flat_color_block_ratio feat_luma_histogram_entropy feat_is_grayscale feat_quant_survival_y_p10 feat_grayscale_score feat_laplacian_variance_p50 feat_orientation_energy_ratio feat_edge_slope_stdev feat_chroma_luma_covariance_cb feat_noise_floor_y_p10 feat_dct_compressibility_y feat_info_weight_p90 feat_block_misalignment_32 feat_noise_floor_uv feat_edge_density feat_high_freq_energy_ratio feat_log_padded_pixels_16 feat_alpha_present feat_noise_floor_uv_p25 feat_aq_map_p75 feat_max_dim feat_cr_vert_sharpness feat_noise_floor_uv_p75 feat_aq_map_p90 feat_log_aspect_abs feat_quant_survival_uv_p25 feat_aq_map_p95 feat_chroma_complexity feat_log_padded_pixels_32 feat_uniformity feat_laplacian_variance_p90 feat_alpha_used_fraction feat_patch_fraction_fast feat_alpha_bimodal_score feat_laplacian_variance_peak feat_aq_map_p99 feat_log_pixels feat_cr_peak_sharpness feat_palette_fits_in_256 feat_quant_survival_uv feat_laplacian_variance_p99 feat_aq_map_p5 feat_aspect_min_over_max feat_noise_floor_uv_p50 feat_cb_sharpness feat_laplacian_variance feat_aq_map_p10 feat_laplacian_variance_p75 feat_variance_spread feat_gradient_fraction feat_noise_floor_uv_p90
```
