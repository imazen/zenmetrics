"""Shared zentrain picker-config base. Outputs on /home (stable; /mnt/v volatile).
Two targets via PICKER_TARGET env: ssim2 (score_ssim2_gpu) | zensim_a (score_zensim_gpu).
Plan cell-id decomposed per codec (categorical cells + scalar) so the trainer's
DATA_STARVED_CELL gate holds and the scalar knob is learned."""
import csv, os
from pathlib import Path

PP = Path("/home/lilith/picker-pp")
ZQ_TARGETS = list(range(30, 70, 5)) + list(range(70, 95, 2))  # web-relevant reachable band

def paths(codec):
    t = os.environ.get("PICKER_TARGET", "ssim2")
    return (PP / "train" / f"{codec}.{t}.pareto.parquet",
            PP / "train" / f"{codec}.features.tsv",
            PP / "models" / f"{codec}_predict_{t}_v0.1.json",
            PP / "models" / f"{codec}_predict_{t}_v0.1.log")

_WANTED = [
    "feat_variance","feat_edge_density","feat_uniformity","feat_chroma_complexity",
    "feat_cb_sharpness","feat_cr_sharpness","feat_flat_color_block_ratio","feat_colourfulness",
    "feat_laplacian_variance","feat_variance_spread","feat_grayscale_score","feat_cb_horiz_sharpness",
    "feat_cb_vert_sharpness","feat_cb_peak_sharpness","feat_cr_horiz_sharpness","feat_cr_vert_sharpness",
    "feat_cr_peak_sharpness","feat_high_freq_energy_ratio","feat_luma_histogram_entropy",
    "feat_dct_compressibility_y","feat_dct_compressibility_uv","feat_patch_fraction_fast",
    "feat_quant_survival_y","feat_quant_survival_uv","feat_aq_map_mean","feat_aq_map_std",
    "feat_noise_floor_y","feat_noise_floor_uv","feat_edge_slope_stdev","feat_gradient_fraction",
    "feat_palette_density","feat_alpha_used_fraction","feat_alpha_bimodal_score","feat_pixel_count",
    "feat_log_pixels","feat_aspect_min_over_max","feat_channel_count","feat_aq_map_p75",
    "feat_aq_map_p90","feat_aq_map_p95","feat_aq_map_p99","feat_noise_floor_y_p50","feat_noise_floor_y_p90",
    "feat_laplacian_variance_p50","feat_laplacian_variance_p75","feat_laplacian_variance_p90",
    "feat_laplacian_variance_p99","feat_laplacian_variance_peak","feat_quant_survival_y_p10",
    "feat_luma_kurtosis","feat_gradient_fraction_smooth",
]
def keep_features(features_path):
    with open(features_path) as f: have = set(next(csv.reader(f, delimiter="\t")))
    return [c for c in _WANTED if c in have]
