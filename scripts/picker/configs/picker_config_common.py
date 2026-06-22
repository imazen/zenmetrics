"""Shared zentrain picker-config base for the unified-sweep pipeline.

All four codecs (zenjpeg/zenwebp/zenjxl/zenavif) train from a zenmetrics
`--plan` sweep adapted by `omni_to_pareto.py`. The plan cell-id is treated as an
opaque categorical config (the picker picks among the swept variants per
(content-features, target-ssim2)); `q` is the trainer's quality dial. The target
metric is SSIMULACRA2 (written into the `zensim` column by the adapter), per the
"ssim2-approximating, not zensim Profile-A" goal.

A per-codec module sets PARETO/FEATURES/OUT_* paths and does
`from picker_config_common import *` then `KEEP_FEATURES = keep_features(FEATURES)`.
"""
import csv
from pathlib import Path

PP = Path("/mnt/v/output/picker-pipeline-2026-06-22")

# Achieved-quality (ssim2) target grid the picker is trained for. Floor at 30
# (tiny images have a ~54 ssim2 floor — they can't reach <30 at any q, so a
# global grid below 30 structurally starves the tiny size class); ceiling at 94
# (>95 is rarely reachable). 30..65 step 5 (the aggressive-web band) + 70..94
# step 2 (perceptibility band). Web-relevant; avoids unreachable-target gate
# noise. Per-size target grids would let medium/large reach lower — future work.
ZQ_TARGETS = list(range(30, 70, 5)) + list(range(70, 95, 2))

# NOTE: each codec config defines its OWN parse_config_name + CATEGORICAL_AXES +
# SCALAR_AXES, decomposing its plan cell-id grammar so each categorical cell
# holds >=3 member configs (the trainer's DATA_STARVED_CELL gate) and the scalar
# knob (e.g. trellis lambda) is learned continuously. `q` is the quality dial.

# Proven ablation-validated content-feature subset (from the production zenjpeg
# picker config); filtered at import to those actually present in the codec's
# features TSV so a per-codec extraction difference degrades gracefully.
_WANTED = [
    "feat_variance", "feat_edge_density", "feat_uniformity", "feat_chroma_complexity",
    "feat_cb_sharpness", "feat_cr_sharpness", "feat_flat_color_block_ratio",
    "feat_colourfulness", "feat_laplacian_variance", "feat_variance_spread",
    "feat_grayscale_score", "feat_cb_horiz_sharpness", "feat_cb_vert_sharpness",
    "feat_cb_peak_sharpness", "feat_cr_horiz_sharpness", "feat_cr_vert_sharpness",
    "feat_cr_peak_sharpness", "feat_high_freq_energy_ratio", "feat_luma_histogram_entropy",
    "feat_dct_compressibility_y", "feat_dct_compressibility_uv", "feat_patch_fraction_fast",
    "feat_quant_survival_y", "feat_quant_survival_uv", "feat_aq_map_mean", "feat_aq_map_std",
    "feat_noise_floor_y", "feat_noise_floor_uv", "feat_edge_slope_stdev", "feat_gradient_fraction",
    "feat_palette_density", "feat_alpha_used_fraction", "feat_alpha_bimodal_score",
    "feat_pixel_count", "feat_log_pixels", "feat_aspect_min_over_max", "feat_channel_count",
    "feat_aq_map_p75", "feat_aq_map_p90", "feat_aq_map_p95", "feat_aq_map_p99",
    "feat_noise_floor_y_p50", "feat_noise_floor_y_p90", "feat_laplacian_variance_p50",
    "feat_laplacian_variance_p75", "feat_laplacian_variance_p90", "feat_laplacian_variance_p99",
    "feat_laplacian_variance_peak", "feat_quant_survival_y_p10", "feat_luma_kurtosis",
    "feat_gradient_fraction_smooth",
]


def keep_features(features_path) -> list[str]:
    with open(features_path) as f:
        have = set(next(csv.reader(f, delimiter="\t")))
    kept = [c for c in _WANTED if c in have]
    missing = [c for c in _WANTED if c not in have]
    if missing:
        import sys
        sys.stderr.write(f"[picker-config] {len(missing)} wanted features absent from "
                         f"{Path(features_path).name}: {missing[:6]}{'...' if len(missing) > 6 else ''}\n")
    return kept
