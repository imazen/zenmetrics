"""ColorVideoVDP reference implementation wrapper.

Uses Rafal Mantiuk's `pycvvdp` (https://github.com/gfxdisp/ColorVideoVDP),
which ships the canonical reference implementation. cvvdp expects
sRGB-encoded display values; we pass raw uint8 frames in HWC layout and
let the metric's internal display model handle the rest.

The score is reported as JOD (Just-Objectionable Differences). Higher is
better; 10.0 is "indistinguishable from reference". Typical viewable
quality lands in the 7.5-9.5 range.

Display profile is selected via the env var ZEN_METRICS_CVVDP_DISPLAY
(default 'standard_4k', matching the default in pycvvdp's own CLI). See
`pycvvdp.vvdp_display_photo_eotf.list_displays()` for available names.
"""
import os

import numpy as np
import torch
import pycvvdp


class Scorer:
    name = "cvvdp"

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        display_name = os.environ.get(
            "ZEN_METRICS_CVVDP_DISPLAY", "standard_4k"
        )
        # `pycvvdp.cvvdp` is the canonical entry point. heuristics='auto'
        # mirrors the package's CLI default. We pin display_geometry to
        # 'standard_fhd' if the caller requests it, but otherwise let the
        # display name carry the full geometry+EOTF spec.
        self.metric = pycvvdp.cvvdp(
            display_name=display_name,
            device=self.device,
            heuristics="auto",
        )

    def score(self, ref_bytes: bytes, dist_bytes: bytes, w: int, h: int) -> float:
        ref = np.frombuffer(ref_bytes, dtype=np.uint8).reshape(h, w, 3)
        dist = np.frombuffer(dist_bytes, dtype=np.uint8).reshape(h, w, 3)
        # `predict` accepts HWC uint8 for single-frame still-image runs.
        # It returns (JOD-score: float, diff-map: tensor); the diff map
        # would be useful for debugging but we discard it here.
        jod, _ = self.metric.predict(ref, dist, dim_order="HWC")
        # `jod` is a 0-dim torch tensor when the metric runs on GPU and a
        # python float in some configurations — normalise via float().
        return float(jod)
