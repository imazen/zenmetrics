"""Compare pycvvdp's band_freqs vs cvvdp-gpu's at 4000x3000."""

import torch
import pycvvdp
from pycvvdp.lpyr_dec import weber_contrast_pyr

W, H = 4000, 3000
m = pycvvdp.cvvdp(display_name="standard_4k", heatmap="none")
ppd = m.pix_per_deg
print(f"ppd = {ppd}")

lpyr = weber_contrast_pyr(W, H, ppd, device=torch.device("cuda"), contrast="weber_g1")
print(f"pycvvdp band_freqs ({len(lpyr.band_freqs)} entries): {list(lpyr.band_freqs)}")
print(f"pycvvdp get_band_count(): {lpyr.get_band_count()}")
print(f"pycvvdp height: {lpyr.height}")
print(f"pycvvdp pyr_shape ({len(lpyr.pyr_shape)} entries): {lpyr.pyr_shape}")
