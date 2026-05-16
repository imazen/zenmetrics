"""IW-SSIM (Information-Weighted SSIM, Wang & Li 2011) wrapper.

Uses the `piq` library (https://github.com/photosynthesis-team/piq),
which ships a PyTorch port of the original Matlab reference. Output is
in [0, 1]; higher is better, 1.0 is identical.

GPU is used when available — the metric is small and runs in
milliseconds on a modern CUDA card. CPU fallback is acceptable for
small-batch sweeps.

The piq function accepts (B, C, H, W) tensors in [0, 1]. We move data
to the chosen device once per call and copy back the scalar — keeping
the surface API symmetric with the cvvdp wrapper.
"""
import os

import numpy as np
import torch
import piq


class Scorer:
    name = "iwssim"

    def __init__(self) -> None:
        prefer_gpu = os.environ.get("ZEN_METRICS_IWSSIM_DEVICE", "auto")
        if prefer_gpu == "cpu":
            self.device = "cpu"
        elif prefer_gpu == "cuda":
            self.device = "cuda"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def score(self, ref_bytes: bytes, dist_bytes: bytes, w: int, h: int) -> float:
        ref = (
            np.frombuffer(ref_bytes, dtype=np.uint8)
            .reshape(h, w, 3)
            .astype(np.float32)
            / 255.0
        )
        dist = (
            np.frombuffer(dist_bytes, dtype=np.uint8)
            .reshape(h, w, 3)
            .astype(np.float32)
            / 255.0
        )
        ref_t = (
            torch.from_numpy(ref.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        )
        dist_t = (
            torch.from_numpy(dist.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            value = piq.information_weighted_ssim(
                ref_t, dist_t, data_range=1.0, reduction="mean"
            )
        return float(value.item())
