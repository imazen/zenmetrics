"""zenwebp picker config — predict-ssim2 target, FLEET sweep data.

Points PARETO/FEATURES at the fleet-swept webp data adapted to trainer
format at /mnt/v/output/zenmetrics/picker-fleet/train/ (webp_ssim2.* —
the `zensim` column carries the ssim2-gpu score for this target).

Mirrors zenwebp_picker.py's recipe (same modes_full cell-id decomposition
+ log1p FEATURE_TRANSFORMS) so the predict-ssim2 and predict-zensim-a
pickers differ ONLY in which fleet pareto they read — making their
held-out bytes-overhead directly comparable (and comparable to the jpeg
fleet _predict_ baselines, which used the _picker recipe with transforms).

modes_full cell-id: vp8|vp8l -m<method>_<tuning>[-syuv].
Categorical format x tuning x syuv; scalar = method (0/2/4/6).
"""
import re
from pathlib import Path
from picker_config_common import keep_features, feature_transforms, ZQ_TARGETS  # noqa: F401

CODEC = "zenwebp"
_FLEET = Path("/mnt/v/output/zenmetrics/picker-fleet")
PARETO = _FLEET / "train" / "webp_ssim2.pareto.parquet"
FEATURES = _FLEET / "train" / "webp_ssim2.features.tsv"
OUT_JSON = _FLEET / "models" / f"{CODEC}_predict_ssim2_v0.1.json"
OUT_LOG = _FLEET / "models" / f"{CODEC}_predict_ssim2_v0.1.log"
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
