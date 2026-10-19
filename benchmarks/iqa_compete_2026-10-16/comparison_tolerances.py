"""Owner-approved cross-implementation probe bounds (2026-10-19).

These are scalar comparison policies, not same-implementation golden tolerances.
"""

ABSOLUTE_BOUNDS = {("msssim", "octave:msssim"): 0.002, ("gmsd", "pyiqa:gmsd"): 2e-8}
