import re
from picker_config_common import keep_features, feature_transforms, paths, ZQ_TARGETS  # noqa
CODEC="zenwebp"; PARETO,FEATURES,OUT_JSON,OUT_LOG=paths(CODEC); KEEP_FEATURES=keep_features(FEATURES); FEATURE_TRANSFORMS=feature_transforms(FEATURES)
CATEGORICAL_AXES=["format","tuning","syuv"]; SCALAR_AXES=["method"]; SCALAR_SENTINELS={}; SCALAR_DISPLAY_RANGES={"method":(0.0,6.0)}
def parse_config_name(n):
    mm=re.search(r"-m(\d+)",n); return {"format":n.split("-")[0],"tuning":n.split("_",1)[1].replace("-syuv","") if "_" in n else "def","syuv":"syuv" if n.endswith("-syuv") else "no","method":float(mm.group(1)) if mm else 0.0}
