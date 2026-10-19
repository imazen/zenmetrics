"""Owner-approved cross-implementation probe bounds (2026-10-19).

These are scalar comparison policies, not same-implementation golden tolerances.
"""

ABSOLUTE_BOUNDS = {("msssim", "octave:msssim"): 0.002, ("gmsd", "pyiqa:gmsd"): 2e-8}

# User delegated selection on 2026-10-19 after reviewing the complete retained
# grid and precision controls. These policies cover PyIQA 0.1.16 only.
PYIQA_SSIM_BOUNDS = {
    **{name: 1e-6 for name in ("ssim-pyiqa-f32", "ssimc-pyiqa-f32", "ms-ssim-pyiqa-f32")},
    **{f"{base}-k{k}": 1e-6
       for base in ("ssim-pyiqa-f32", "ssimc-pyiqa-f32", "ms-ssim-pyiqa-f32")
       for k in range(2, 7)},
    "ssim-pyiqa": 5e-5,
    "ms-ssim-pyiqa": 5e-5,
}
