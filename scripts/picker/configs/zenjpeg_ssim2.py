"""zenjpeg picker config (ssim2 target), unified-sweep pipeline.

rd_core cell-id grammar: `<strategy>_<trellis>_small_<sub>`
  strategy in {gls,jp3,moz,pw4} (categorical); trellis in {t0,tr14.5,tr14.75+dc}
  (t0=off; tr<lambda>[+dc]); sub in {420,444} (categorical).
Decomposed so each categorical cell (strategy x sub = 8) holds the 3 trellis
variants (DATA_STARVED_CELL threshold 3); trellis lambda is a learned scalar.
"""
from picker_config_common import keep_features, PP, ZQ_TARGETS  # noqa: F401

CODEC = "zenjpeg"
PARETO = PP / "train" / f"{CODEC}.pareto.parquet"
FEATURES = PP / "train" / f"{CODEC}.features.tsv"
OUT_JSON = PP / "models" / f"{CODEC}_ssim2_v0.1.json"
OUT_LOG = PP / "models" / f"{CODEC}_ssim2_v0.1.log"
KEEP_FEATURES = keep_features(FEATURES)

CATEGORICAL_AXES = ["strategy", "sub"]
SCALAR_AXES = ["trellis_lambda"]
SCALAR_SENTINELS = {"trellis_lambda": 0.0}
SCALAR_DISPLAY_RANGES = {"trellis_lambda": (0.0, 25.0)}


def parse_config_name(name: str) -> dict:
    parts = name.split("_")
    strategy = parts[0]
    trellis = parts[1]
    sub = parts[-1]
    lam = 0.0 if trellis == "t0" else float(trellis[2:].split("+")[0])
    return {"strategy": strategy, "sub": sub, "trellis_lambda": lam}
