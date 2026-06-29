"""zenavif HDR picker config — target = **ssim2** (no-GPU HDR-picker track; ssim2
scores HDR via the PU21-integrated feeding). TEMPLATE — see the two prerequisites
below before using.

FEATURE SET = SDR keep_features + the 6 depth-tier HDR features, auto-included by
keep_features() from the HDR features TSV (see zenjxl_hdr.py for the full note).
Stage the HDR features TSV at PP/train/zenavif_hdr.features.tsv before training.

AXES mirror the SDR zenavif picker (speed scalar; sub / bd / qm categorical). For an
HDR knob-grid sweep, omni_to_pareto.py renders knob_tuple_json's sorted knob MAP to a
canonical config_name ('def' | 'k1=v1,k2=v2...', "," separated); the parse decodes it.

PREREQUISITES (both unmet as of 2026-06-29 — this config cannot be exercised yet):
  1. **No zenavif HDR encode path.** crates/zenmetrics-cli/src/sweep/hdr.rs
     ::validate_hdr_sweep accepts ONLY zenjxl; zenavif 10-bit PQ + CICP encode must be
     wired (~2 fns per the hdr-picker-blocked-encode-infra memory) before any zenavif
     HDR sweep can run.
  2. **Knob-grid axis names UNVERIFIED.** The avif knob names below
     (speed/sub/bd/qm) are mirrored from the SDR rd_core grammar; once the HDR encode
     path lands, confirm them against zenavif's codec_knobs / encode_avif knob reads
     and adjust CATEGORICAL/SCALAR + parse_config_name to match the actual --knob-grid.
"""
from picker_config_common import keep_features, feature_transforms, paths, ZQ_TARGETS  # noqa

CODEC = "zenavif_hdr"
PARETO, FEATURES, OUT_JSON, OUT_LOG = paths(CODEC)  # PICKER_TARGET=ssim2
KEEP_FEATURES = keep_features(FEATURES)
FEATURE_TRANSFORMS = feature_transforms(FEATURES)

CATEGORICAL_AXES = ["sub", "bd", "qm"]
SCALAR_AXES = ["speed"]
SCALAR_SENTINELS = {}
SCALAR_DISPLAY_RANGES = {"speed": (2.0, 6.0)}


def _kv(name):
    """Decode the knob-grid config_name 'def' | 'k1=v1,k2=v2...' -> {k: v_str}."""
    out = {}
    if name and name != "def":
        for tok in name.split(","):
            if "=" in tok:
                k, v = tok.split("=", 1)
                out[k] = v
    return out


def parse_config_name(name):
    kv = _kv(name)
    return {
        "speed": float(kv.get("speed", 4)),
        "sub": kv.get("sub", "444"),
        "bd": kv.get("bd", "10"),  # HDR defaults to 10-bit
        "qm": kv.get("qm", "on"),
    }
