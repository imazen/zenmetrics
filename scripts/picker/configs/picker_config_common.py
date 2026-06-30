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
    # Use the curated _WANTED first (stable order), then EVERY remaining feat_
    # column available. The picker is feature/cell-limited (teacher≈student), so
    # widening the input is the lever to hug the pareto tighter; the GBDT teacher
    # is robust to redundancy and the val→test gap flags any MLP overfit.
    with open(features_path) as f: header = next(csv.reader(f, delimiter="\t"))
    have = set(header)
    import sys
    # GUARD (2026-06-29): the XYB chroma-loss features are the DIRECT
    # chroma-subsampling-cost signal. Extracting without them silently cripples
    # multi-cell (webp/avif) picker discrimination — the off-frontier 420-vs-444
    # disaster. zenanalyze now ships `experimental` ON by default; if these are
    # still absent, the features TSV was built with an old/opted-out analyzer.
    # DEMAND them — fail loudly unless an explicit, visible override is set.
    _REQUIRED_EXPERIMENTAL = ["feat_xyb_bquarter_chroma_loss", "feat_xyb444_color_loss"]
    _missing = [c for c in _REQUIRED_EXPERIMENTAL if c not in have]
    if _missing:
        print("\n" + "=" * 72
            + f"\nFATAL: training features missing experimental chroma-loss columns: {_missing}"
            + f"\n  features file: {features_path}"
            + "\nThese are the direct chroma-subsampling-cost signal; training without them"
            + "\nsilently cripples multi-cell (webp/avif) 420-vs-444 discrimination — the"
            + "\noff-frontier picker disaster (2026-06-29). Re-extract with a zenanalyze built"
            + "\nwith the `experimental` feature (now the default)."
            + "\nOverride ONLY if this corpus legitimately lacks them: ZEN_ALLOW_MISSING_EXPERIMENTAL=1"
            + "\n" + "=" * 72, file=sys.stderr)
        if os.environ.get("ZEN_ALLOW_MISSING_EXPERIMENTAL") != "1":
            raise SystemExit(3)
    # A curated _WANTED feature silently absent is the same failure mode, smaller:
    # a config asking for a feature it never gets. Surface it (don't fail).
    _wanted_missing = [c for c in _WANTED if c not in have]
    if _wanted_missing:
        print(f"WARNING: _WANTED features absent from {features_path}: {_wanted_missing}", file=sys.stderr)
    wanted = [c for c in _WANTED if c in have]
    extra = [c for c in header if c.startswith("feat_") and c not in set(_WANTED)]
    return wanted + extra

# Heavy-tailed, strictly-positive features that span orders of magnitude
# (measured on imazen26_train_features_2026-06-22: pixel_count tail 9352x,
# laplacian_variance 365x, luma_kurtosis 367x, the chroma horiz/vert/peak
# sharpness family 80-180x, etc.). Fed raw to the trainer's StandardScaler
# their outliers dominate and the bulk collapses — log1p (0 params, applied
# BEFORE StandardScaler and baked into the model JSON so inference matches)
# compresses the tail to a smooth, near-Gaussian input. Bounded-[0,1] and
# low/left-skew features (skew < ~1.5) are left raw — StandardScaler handles
# them and a log would only distort. winsor/clip_then_log1p are avoided here
# because they need corpus-specific [p1,p99] params (FEATURE_TRANSFORM_PARAMS);
# log1p is parameter-free and corpus-stable.
_LOG1P_FEATURES = [
    "feat_pixel_count", "feat_variance", "feat_laplacian_variance",
    "feat_laplacian_variance_p50", "feat_laplacian_variance_p75",
    "feat_high_freq_energy_ratio", "feat_dct_compressibility_y",
    "feat_dct_compressibility_uv", "feat_cb_horiz_sharpness",
    "feat_cb_vert_sharpness", "feat_cb_peak_sharpness", "feat_cr_horiz_sharpness",
    "feat_cr_vert_sharpness", "feat_cr_peak_sharpness", "feat_luma_kurtosis",
]
def feature_transforms(features_path):
    """log1p map restricted to KEEP_FEATURES actually present in this TSV.
    Set PICKER_NO_TRANSFORMS=1 to disable (for A/B ablation)."""
    if os.environ.get("PICKER_NO_TRANSFORMS"):
        return {}
    keep = set(keep_features(features_path))
    return {f: "log1p" for f in _LOG1P_FEATURES if f in keep}
