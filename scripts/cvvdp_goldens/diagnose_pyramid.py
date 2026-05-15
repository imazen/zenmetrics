"""Print pycvvdp's pyramid depth + band frequencies at 4000x3000."""

import numpy as np
import torch
import pycvvdp
from pycvvdp.lpyr_dec import weber_contrast_pyr

W, H = 4000, 3000

m = pycvvdp.cvvdp(display_name="standard_4k", heatmap="none")
ppd = m.pix_per_deg
print(f"ppd = {ppd}")

# weber_contrast_pyr takes (W, H, ppd, device).
lpyr = weber_contrast_pyr(W, H, ppd, device=torch.device("cuda"), contrast="weber_g1")
print(f"lpyr type: {type(lpyr).__name__}")

# Common attributes on pycvvdp pyramid:
for attr in ("height", "width", "ppd", "band_freqs", "num_levels", "n_levels",
             "pyr_shape", "min_freq", "max_freq"):
    if hasattr(lpyr, attr):
        v = getattr(lpyr, attr)
        if callable(v):
            try:
                v = v()
            except Exception as e:
                v = f"(call failed: {e})"
        print(f"  {attr}: {v}")

# Run decomp on a synthetic input and report the band count + shapes.
img = torch.zeros(1, 1, H, W, device="cuda")
try:
    bands = lpyr.decompose(img)
    print(f"  decompose returned {len(bands)} bands")
    for i, b in enumerate(bands):
        if isinstance(b, torch.Tensor):
            print(f"    band[{i}]: shape={tuple(b.shape)}")
        else:
            print(f"    band[{i}]: {type(b).__name__}")
except Exception as e:
    print(f"  decompose failed: {e}")

# Many cvvdp versions store the per-band frequency list as `band_freqs`
# or `freqs`. Walk module-level fns too.
for attr in dir(lpyr):
    if attr.startswith("_"):
        continue
    v = getattr(lpyr, attr, None)
    if isinstance(v, torch.Tensor):
        print(f"  tensor attr {attr}: shape={tuple(v.shape)}, val={v.detach().cpu().numpy().tolist() if v.numel() < 30 else '(large)'}")
    elif isinstance(v, (int, float, list, tuple)):
        print(f"  attr {attr}: {v}")

# Print num pyramid levels via get_band_count() if available.
for fn in ("get_band_count", "get_band_freqs", "get_band_size"):
    if hasattr(lpyr, fn):
        f = getattr(lpyr, fn)
        try:
            print(f"  {fn}(): {f()}")
        except TypeError:
            # may require args
            for k in range(20):
                try:
                    print(f"  {fn}({k}): {f(k)}")
                except Exception as e:
                    print(f"  {fn}({k}) stopped at {k}: {e}")
                    break
