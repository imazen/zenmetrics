from picker_config_common import keep_features, feature_transforms, paths, ZQ_TARGETS  # noqa
CODEC="zenjpeg"; PARETO,FEATURES,OUT_JSON,OUT_LOG=paths(CODEC); KEEP_FEATURES=keep_features(FEATURES); FEATURE_TRANSFORMS=feature_transforms(FEATURES)
CATEGORICAL_AXES=["strategy","sub"]; SCALAR_AXES=["trellis_lambda"]; SCALAR_SENTINELS={"trellis_lambda":0.0}
SCALAR_DISPLAY_RANGES={"trellis_lambda":(0.0,15.0)}  # data max ~14.8; tight range => scalar head uses full [0,1]
import re as _re
_TR = _re.compile(r"tr(\d+(?:\.\d+)?)")  # leading trellis-lambda float after the `tr` prefix
def parse_config_name(n):
    # scalar_dense grammar (2026-06): strategy[+bracket] _ t0|tr<lambda>[cpl±Ncl1] _ sub _ chroma[-blur<x>]
    # Robust to the cpl/blur/bracket sub-knobs the old parser crashed on (e.g. `jp3_tr14.75cpl+1cl1_small_420`).
    p = n.split("_")
    t = p[1] if len(p) > 1 else ""
    if t == "t0" or not t.startswith("tr"):
        tl = 0.0
    else:
        m = _TR.match(t)
        tl = float(m.group(1)) if m else 0.0
    return {"strategy": p[0], "sub": p[-1], "trellis_lambda": tl}
