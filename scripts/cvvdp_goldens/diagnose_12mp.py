"""Diagnose pycvvdp's pyramid + band setup at 4000x3000.

cvvdp-gpu currently caps the pyramid at MAX_LEVELS=8. If pycvvdp
uses more levels at 12 MP, that alone explains the 0.586 JOD drift
between cvvdp-gpu and pycvvdp on the synth pair.
"""

import numpy as np
import torch
import pycvvdp

W, H = 4000, 3000

# Same synth pair shape as bench_12mp_cuda.py
yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
r = ((xx * 17 + yy * 5) % 251).astype(np.uint8) + 40
g = ((xx * 11 + yy * 13) % 247).astype(np.uint8) + 40
b = ((xx * 7 + yy * 19) % 241).astype(np.uint8) + 40
ref = np.stack([r, g, b], axis=-1)
dist = np.stack(
    [
        np.maximum(r.astype(np.int16) - 8, 0).astype(np.uint8),
        np.maximum(g.astype(np.int16) - 4, 0).astype(np.uint8),
        np.minimum(b.astype(np.int16) + 12, 255).astype(np.uint8),
    ],
    axis=-1,
)

m = pycvvdp.cvvdp(display_name="standard_4k", heatmap="none")

# pycvvdp creates an internal CSF object with the band frequencies +
# pyramid depth at metric-instantiation time. Inspect what it picked.
print("=== pycvvdp instance ===")
print(f"display_name: standard_4k")
print(f"display.pixels_per_degree: {m.display_photometry.pix_per_deg(None) if hasattr(m.display_photometry, 'pix_per_deg') else 'n/a'}")
print(f"display object: {type(m.display_photometry).__name__}")

# Run predict to populate the internal pyramid / band structures.
jod, stats = m.predict(dist, ref, dim_order="HWC")
print(f"\n=== predict result ===")
print(f"jod: {float(jod):.6f}")
print(f"stats keys: {list(stats.keys()) if hasattr(stats, 'keys') else 'n/a'}")

# Try to extract band setup from the metric's internals.
# pycvvdp uses LaplacianPyramid + ContrastSensitivity_pix internally.
print(f"\n=== metric internals ===")
for attr in dir(m):
    if attr.startswith("_"):
        continue
    if attr in ("predict",):
        continue
    val = getattr(m, attr, None)
    if callable(val):
        continue
    repr_str = repr(val)
    if len(repr_str) > 200:
        repr_str = repr_str[:200] + "..."
    print(f"  {attr}: {repr_str}")

# Inspect the underlying pyramid + CSF setup.
print(f"\n=== display_photometry ===")
for attr in dir(m.display_photometry):
    if attr.startswith("_"):
        continue
    val = getattr(m.display_photometry, attr, None)
    if callable(val):
        continue
    print(f"  {attr}: {val}")

# pycvvdp's predict_video_source goes through `lpyr_dec.LaplacianPyramid`.
# We want to know: at H=3000 W=4000, how many pyramid levels does it
# use? What are the per-level frequencies?
try:
    # ImLpyr_pix is the still-image Laplacian pyramid class.
    from pycvvdp import lpyr_dec
    print(f"\n=== lpyr_dec module ===")
    print(f"  dir: {[n for n in dir(lpyr_dec) if not n.startswith('_')]}")

    # Try to manually construct + measure the band count.
    # ImLpyrBase classes vary across cvvdp versions; try a few names.
    for cls_name in ["ImLpyrBase", "weber_contrast_pyr", "LaplacianPyramid",
                     "lpyr_dec", "lpyr_dec_imp"]:
        cls = getattr(lpyr_dec, cls_name, None)
        if cls is not None:
            print(f"  found: {cls_name} = {cls}")
except Exception as e:
    print(f"\nintrospection failed: {e}")
