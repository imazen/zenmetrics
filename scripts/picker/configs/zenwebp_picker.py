"""zenwebp picker config — predict-zensim-a target, FLEET sweep data.

Points PARETO/FEATURES at the fleet-swept webp data adapted to trainer
format at /mnt/v/output/zenmetrics/picker-fleet/train/ (webp_zensim.* —
the `zensim` column carries the zensim-a score for this target).

Same modes_full cell-id decomposition + log1p FEATURE_TRANSFORMS as the
predict-ssim2 sibling (zenwebp_ssim2.py); the only difference is the
fleet pareto read. modes_full cell-id: vp8|vp8l -m<method>_<tuning>[-syuv].
Categorical format x tuning x syuv; scalar = method (0/2/4/6).
"""
import re
from picker_config_common import keep_features, feature_transforms, paths, ZQ_TARGETS  # noqa: F401

CODEC = "zenwebp"
# PARETO/FEATURES/OUT resolve via paths(CODEC) keyed on PICKER_TARGET (like
# zenjpeg/zenavif) so KEEP_FEATURES is computed from the SAME features file the
# merge writes — not a stale _FLEET path that predates the all-origin corpus.
# A caller may still override the data via train_hybrid --pareto/--features
# (the merge does, belt-and-suspenders).
PARETO, FEATURES, OUT_JSON, OUT_LOG = paths(CODEC)
KEEP_FEATURES = keep_features(FEATURES)
FEATURE_TRANSFORMS = feature_transforms(FEATURES)
CATEGORICAL_AXES = ["format", "tuning", "syuv"]
SCALAR_AXES = ["method"]
SCALAR_SENTINELS = {}
SCALAR_DISPLAY_RANGES = {"method": (0.0, 6.0)}


def parse_config_name(name: str) -> dict:
    fmt = name.split("-")[0]
    m = re.search(r"-m(\d+)", name)
    method = float(m.group(1)) if m else 0.0
    syuv = "syuv" if name.endswith("-syuv") else "no"
    tuning = name.split("_", 1)[1].replace("-syuv", "") if "_" in name else "def"
    return {"format": fmt, "tuning": tuning, "syuv": syuv, "method": method}
