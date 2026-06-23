from picker_config_common import keep_features, paths, ZQ_TARGETS  # noqa
CODEC="zenavif"; PARETO,FEATURES,OUT_JSON,OUT_LOG=paths(CODEC); KEEP_FEATURES=keep_features(FEATURES)
CATEGORICAL_AXES=["sub","bd","qm"]; SCALAR_AXES=["speed"]; SCALAR_SENTINELS={}; SCALAR_DISPLAY_RANGES={"speed":(0.0,10.0)}
def parse_config_name(n):
    return {"sub":"420" if "-420" in n else "444","bd":"10" if "-bd10" in n else "8","qm":"off" if "-noqm" in n else "on","speed":float(n.split("-")[0][1:])}
