import csv
from zenjxl_lossy_dense import (CODEC, PARETO, FEATURES, OUT_JSON, OUT_LOG,
    CATEGORICAL_AXES, SCALAR_AXES, SCALAR_SENTINELS, SCALAR_DISPLAY_RANGES,
    parse_config_name, ZQ_TARGETS)  # noqa
with open(FEATURES) as f: _hdr=next(csv.reader(f,delimiter='\t'))
KEEP_FEATURES=[c for c in _hdr if c.startswith('feat_')]
from picker_config_common import feature_transforms  # noqa
FEATURE_TRANSFORMS=feature_transforms(FEATURES)
