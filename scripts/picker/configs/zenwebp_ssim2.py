"""zenwebp picker config (ssim2). modes_full cell-id: vp8|vp8l -m<method>_<tuning>[-syuv].
Categorical format x tuning x syuv; scalar = method (0/2/4/6)."""
import re
from picker_config_common import keep_features, PP, ZQ_TARGETS  # noqa: F401
CODEC = "zenwebp"
PARETO = PP / "train" / f"{CODEC}.pareto.parquet"
FEATURES = PP / "train" / f"{CODEC}.features.tsv"
OUT_JSON = PP / "models" / f"{CODEC}_ssim2_v0.1.json"
OUT_LOG = PP / "models" / f"{CODEC}_ssim2_v0.1.log"
KEEP_FEATURES = keep_features(FEATURES)
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
