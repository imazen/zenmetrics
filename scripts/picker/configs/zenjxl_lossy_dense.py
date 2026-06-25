# Picker config for the JXL lossy knob-space ablation `lossy_dense` plan.
#
# Unlike `zenjxl_picker.py` (rd_core: mode/strategy/effort only), this decomposes the
# FULL lossy_dense cell-id grammar so the picker can choose among the knob set:
#   vd-e{effort}_{strategy}_{knob-label}[-flag...]
# e.g. vd-e7_zen_def, vd-e7_libjxl_def, vd-e7_zen_kaq0.65, vd-e7_zen_def-gab0, vd-e5_zen_def
# Under lossy_dense (--max-deviations 1) exactly one of {effort!=7, strategy!=zen,
# knob!=def, a flag} deviates; the picker learns which (effort, strategy, knob, flag)
# minimizes bytes at the target quality given the image's content features.
#
# PARETO/FEATURES are overridden to the run's files at launch (cf. the zenwebp fleet
# configs); the parse function is validated by scripts/picker/test_lossy_dense_parse.py.
import re
from picker_config_common import keep_features, feature_transforms, paths, ZQ_TARGETS  # noqa

CODEC = "zenjxl_lossy_dense"
PARETO, FEATURES, OUT_JSON, OUT_LOG = paths(CODEC)
KEEP_FEATURES = keep_features(FEATURES)
FEATURE_TRANSFORMS = feature_transforms(FEATURES)

CATEGORICAL_AXES = ["strategy", "knob", "flag"]
SCALAR_AXES = ["effort"]
SCALAR_SENTINELS = {}
SCALAR_DISPLAY_RANGES = {"effort": (1.0, 9.0)}  # lossy_dense sweeps e1..e9

_RE = re.compile(r"^vd-e(\d+)_([a-z]+)_([^-]+)(?:-(.*))?$")


def parse_config_name(n):
    m = _RE.match(n)
    if not m:
        # Defensive default = the production-default cell (all index-0).
        return {"strategy": "zen", "knob": "def", "flag": "none", "effort": 7.0}
    eff, strat, label, flags = m.group(1), m.group(2), m.group(3), m.group(4)
    return {
        "strategy": strat,
        "knob": label,
        "flag": flags if flags else "none",
        "effort": float(eff),
    }
