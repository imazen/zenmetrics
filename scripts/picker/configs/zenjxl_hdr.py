"""zenjxl HDR picker config — target = **ssim2** (the trustworthy HDR metric that
scores without a GPU: ssim2 reads HDR through the PU21-integrated feeding, so an
HDR ssim2 sweep is CPU-runnable). cvvdp is the gold HDR target but is GPU-only;
this config deliberately targets ssim2 per the no-GPU HDR-picker track.

FEATURE SET = SDR keep_features + the 6 depth-tier HDR features. No special wiring
needed: picker_config_common.keep_features() keeps EVERY ``feat_*`` column in the
features TSV, and the HDR features TSV (built by zenanalyze
``examples/extract_hdr_features``, ``--features experimental,hdr``) carries the SDR
content features PLUS the depth tier:
  feat_peak_luminance_nits  feat_p99_luminance_nits  feat_hdr_headroom_stops
  feat_hdr_pixel_fraction   feat_wide_gamut_peak     feat_wide_gamut_fraction
(measured 2026-06-29 on imazen-26-hdr: 5/6 vary strongly; wide_gamut_fraction is
constant 0.0 on this corpus — all DisplayP3/Bt709, no Bt2020 primaries — so it
carries no signal HERE; the GBDT teacher ignores a constant column.) Stage the HDR
features TSV at PP/train/zenjxl_hdr.features.tsv before training.

config_name grammar — KNOB-GRID, **not** the plan cell-id. HDR sweeps run
``zenmetrics sweep --hdr --knob-grid ...``; ``--plan`` is rejected for HDR
(crates/zenmetrics-cli/src/sweep/hdr.rs::validate_hdr_sweep). So knob_tuple_json is
a sorted-key knob MAP, not ``{"cell",..}``. omni_to_pareto.py renders that map to a
canonical config_name: ``def`` for ``{}`` or ``k1=v1,k2=v2...`` (sorted keys, ","
separated because knob names contain "_"); the parse below decodes that form.

AXES = zenjxl lossy knobs reachable via the knob-grid EXPERT path
(encode_jxl_expert in crates/zenmetrics-cli/src/sweep/encode.rs): effort (scalar) +
gaborish / patches / error_diffusion (categorical booleans). The exact CATEGORICAL/
SCALAR set MUST match the ``--knob-grid`` used by the sweep.

LIMITATION (read before sweeping): the SDR-ablation top jxl-lossy axes —
epf / k_ac_quant / try_dct* / entropy_mul — are NOT exposed by the knob-grid expert
path (they are only reachable via ``--plan``, which HDR does not accept). Until
encode_jxl_expert gains those LossyConfig builders (or ``--plan`` is wired to the HDR
encode path), the HDR jxl picker is limited to the knobs listed above. Extend the
axes here in lockstep with whatever the eventual HDR knob-grid sweeps.
"""
from picker_config_common import keep_features, feature_transforms, paths, ZQ_TARGETS  # noqa

CODEC = "zenjxl_hdr"
# PICKER_TARGET=ssim2 -> PP/train/zenjxl_hdr.ssim2.pareto.parquet,
# PP/train/zenjxl_hdr.features.tsv, PP/models/zenjxl_hdr_predict_ssim2_v0.1.{json,log}
PARETO, FEATURES, OUT_JSON, OUT_LOG = paths(CODEC)
KEEP_FEATURES = keep_features(FEATURES)
FEATURE_TRANSFORMS = feature_transforms(FEATURES)

CATEGORICAL_AXES = ["gaborish", "patches", "error_diffusion"]
SCALAR_AXES = ["effort"]
SCALAR_SENTINELS = {}
SCALAR_DISPLAY_RANGES = {"effort": (1.0, 9.0)}


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
    # Defaults = zenjxl/libjxl encoder defaults (the 'def' / {} cell).
    return {
        "effort": float(kv.get("effort", 7)),
        "gaborish": kv.get("gaborish", "true"),
        "patches": kv.get("patches", "true"),
        "error_diffusion": kv.get("error_diffusion", "false"),
    }
