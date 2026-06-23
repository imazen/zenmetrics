from picker_config_common import keep_features, paths, ZQ_TARGETS  # noqa
CODEC="zenjxl"; PARETO,FEATURES,OUT_JSON,OUT_LOG=paths(CODEC); KEEP_FEATURES=keep_features(FEATURES)
CATEGORICAL_AXES=["mode","variant"]; SCALAR_AXES=["effort"]; SCALAR_SENTINELS={}; SCALAR_DISPLAY_RANGES={"effort":(1.0,9.0)}
def parse_config_name(n):
    m=n.split("-")[0]; tk=n[len(m)+1:].split("_"); return {"mode":m,"variant":tk[1] if len(tk)>=3 else "none","effort":float(tk[0][1:])}
