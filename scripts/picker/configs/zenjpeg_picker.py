from picker_config_common import keep_features, feature_transforms, paths, ZQ_TARGETS  # noqa
CODEC="zenjpeg"; PARETO,FEATURES,OUT_JSON,OUT_LOG=paths(CODEC); KEEP_FEATURES=keep_features(FEATURES); FEATURE_TRANSFORMS=feature_transforms(FEATURES)
CATEGORICAL_AXES=["strategy","sub"]; SCALAR_AXES=["trellis_lambda"]; SCALAR_SENTINELS={"trellis_lambda":0.0}
SCALAR_DISPLAY_RANGES={"trellis_lambda":(0.0,15.0)}  # data max ~14.8; tight range => scalar head uses full [0,1]
def parse_config_name(n):
    p=n.split("_"); t=p[1]; return {"strategy":p[0],"sub":p[-1],"trellis_lambda":0.0 if t=="t0" else float(t[2:].split("+")[0])}
