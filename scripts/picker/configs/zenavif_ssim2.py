"""zenavif picker config (ssim2). rd_core cell-id: s<speed>[-noqm][-420][-bd10].
Categorical sub(420/444) x bd(8/10) x qm(on/off) = 8 cells; scalar = speed."""
from picker_config_common import keep_features, PP, ZQ_TARGETS  # noqa: F401
CODEC = "zenavif"
PARETO = PP / "train" / f"{CODEC}.pareto.parquet"
FEATURES = PP / "train" / f"{CODEC}.features.tsv"
OUT_JSON = PP / "models" / f"{CODEC}_ssim2_v0.1.json"
OUT_LOG = PP / "models" / f"{CODEC}_ssim2_v0.1.log"
KEEP_FEATURES = keep_features(FEATURES)
CATEGORICAL_AXES = ["sub", "bd", "qm"]
SCALAR_AXES = ["speed"]
SCALAR_SENTINELS = {}
SCALAR_DISPLAY_RANGES = {"speed": (0.0, 10.0)}
def parse_config_name(name: str) -> dict:
    speed = float(name.split("-")[0][1:])  # s2 -> 2
    return {"sub": "420" if "-420" in name else "444",
            "bd": "10" if "-bd10" in name else "8",
            "qm": "off" if "-noqm" in name else "on", "speed": speed}
