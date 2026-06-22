"""zenjxl picker config (ssim2). rd_core cell-id: mod-e<eff>_def | vd-e<eff>_<variant>_def.
Categorical mode(mod/vd) x variant(none/libjxl/zen) = 3 cells; scalar = effort."""
from picker_config_common import keep_features, PP, ZQ_TARGETS  # noqa: F401
CODEC = "zenjxl"
PARETO = PP / "train" / f"{CODEC}.pareto.parquet"
FEATURES = PP / "train" / f"{CODEC}.features.tsv"
OUT_JSON = PP / "models" / f"{CODEC}_ssim2_v0.1.json"
OUT_LOG = PP / "models" / f"{CODEC}_ssim2_v0.1.log"
KEEP_FEATURES = keep_features(FEATURES)
CATEGORICAL_AXES = ["mode", "variant"]
SCALAR_AXES = ["effort"]
SCALAR_SENTINELS = {}
SCALAR_DISPLAY_RANGES = {"effort": (1.0, 9.0)}
def parse_config_name(name: str) -> dict:
    mode = name.split("-")[0]            # mod / vd
    toks = name[len(mode) + 1:].split("_")  # e5/def  or  e7/zen/def
    return {"mode": mode, "variant": toks[1] if len(toks) >= 3 else "none",
            "effort": float(toks[0][1:])}
