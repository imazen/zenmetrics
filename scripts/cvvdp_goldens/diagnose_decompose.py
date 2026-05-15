"""Inspect actual shape of pycvvdp's decompose output."""

import torch
import pycvvdp
from pycvvdp.lpyr_dec import weber_contrast_pyr

W, H = 4000, 3000
m = pycvvdp.cvvdp(display_name="standard_4k", heatmap="none")
ppd = m.pix_per_deg
lpyr = weber_contrast_pyr(W, H, ppd, device=torch.device("cuda"), contrast="weber_g1")

# Real input — cvvdp pyramid takes a 4D Y_lum (B, C, H, W).
img = torch.rand(1, 1, H, W, device="cuda") * 100.0 + 1.0  # luminance
bands_struct = lpyr.decompose(img)
print(f"decompose returned type: {type(bands_struct)}")
print(f"  len: {len(bands_struct)}")

# weber_contrast_pyr.decompose returns (bands, log_l_bkg) typically.
for i, b in enumerate(bands_struct):
    print(f"\nbands_struct[{i}] (type {type(b).__name__}):")
    if isinstance(b, list):
        for j, sub in enumerate(b):
            if isinstance(sub, torch.Tensor):
                print(f"  [{j}] shape: {tuple(sub.shape)}, dtype: {sub.dtype}")
            else:
                print(f"  [{j}] {type(sub).__name__}: {sub}")
    elif isinstance(b, torch.Tensor):
        print(f"  shape: {tuple(b.shape)}, dtype: {b.dtype}")
