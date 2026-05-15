"""Fair-comparison benchmark for pycvvdp's CUDA still-image path
PLUS golden-emission for cvvdp-gpu's parity tests.

Times `pycvvdp.cvvdp.predict()` on a 4000x3000 synthetic RGB pair
(same shape + same byte-level construction as `examples/time_12mp.rs`
on the cvvdp-gpu side) so the two timings are directly comparable.
Also produces a JSON manifest of pycvvdp golden JOD values for
small fixtures so the Rust parity tests can gate against
canonical reference values, not host-scalar self-checks.

Run from the repo root:
    cd scripts/cvvdp_goldens
    uv venv .venv --python python3.10
    uv pip install --python .venv/bin/python \\
        torch --index-url https://download.pytorch.org/whl/cu124
    uv pip install --python .venv/bin/python \\
        'cvvdp==0.5.4' 'pillow>=10' 'numpy>=1.26'
    .venv/bin/python bench_12mp_cuda.py

Writes `pycvvdp_synth_goldens.json` in the script directory.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

import pycvvdp

W, H = 4000, 3000
ITERS = 5

# Synth-pair constructors. Each matches a cvvdp-gpu test fixture
# byte-for-byte so the GPU runs the same input and we can compare
# the JOD output directly.
def synth_pair_12mp(w=W, h=H):
    """Matches examples/time_12mp.rs synth_pair() and the 12 MP
    parity test (compute_dkl_jod_matches_pycvvdp_at_12mp_synth).
    """
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
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
    return ref, dist


def synth_pair_odd_dim(w=73, h=91):
    """Matches the 73×91 odd-dim parity test's construction —
    distinct R/G/B patterns from x/y position, with a small DIST
    perturbation. Tests ceil-div pyramid invariants at odd
    dimensions.
    """
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    r = ((xx * 8) % 256).astype(np.uint8)
    g = ((yy * 8) % 256).astype(np.uint8)
    b = (((xx + yy) * 4) % 256).astype(np.uint8)
    ref = np.stack([r, g, b], axis=-1)
    dist = np.stack(
        [
            np.maximum(r.astype(np.int16) - 8, 0).astype(np.uint8),
            np.maximum(g.astype(np.int16) - 4, 0).astype(np.uint8),
            np.minimum(b.astype(np.int16) + 12, 255).astype(np.uint8),
        ],
        axis=-1,
    )
    return ref, dist


def main():
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}")
        print(f"cuda capability: {torch.cuda.get_device_capability(0)}")

    ref_np, dist_np = synth_pair_12mp()
    metric = pycvvdp.cvvdp(display_name="standard_4k", heatmap="none")

    # Produce the 73×91 odd-dim golden first (small, fast, used by
    # the cvvdp-gpu odd-dim parity test for a canonical-reference
    # gate instead of a host-scalar self-check). One call — value
    # is deterministic.
    odd_ref, odd_dist = synth_pair_odd_dim()
    print("odd-dim 73x91 golden:")
    odd_jod, _ = metric.predict(odd_dist, odd_ref, dim_order="HWC")
    print(f"  jod = {float(odd_jod):.4f}\n")
    odd_jod_val = float(odd_jod)

    # Warm up: first 12 MP .predict() triggers Torch graph
    # compilation, kernel JIT, allocator warmup. Don't time it.
    print("warming up (uncached)...")
    t = time.perf_counter()
    jod_warm, _ = metric.predict(dist_np, ref_np, dim_order="HWC")
    dt_warm = time.perf_counter() - t
    print(f"  warm-up call: {dt_warm * 1000:.1f} ms, jod={float(jod_warm):.4f}")

    times = []
    print(f"running {ITERS} timed iters...")
    for i in range(ITERS):
        # torch.cuda.synchronize ensures we measure end-to-end GPU work,
        # not just the CPU dispatch.
        torch.cuda.synchronize()
        t = time.perf_counter()
        jod, _ = metric.predict(dist_np, ref_np, dim_order="HWC")
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        times.append(dt)
        print(f"  iter {i}: {dt * 1000:.1f} ms, jod={float(jod):.4f}")

    times.sort()
    median = times[ITERS // 2]
    total_px = W * H
    print()
    print(f"pycvvdp 12 MP CUDA timing ({ITERS} iters, median):")
    print(f"  predict():         {median * 1000:.1f} ms  -> {median * 1e9 / total_px:.1f} ns/px")
    print()
    print(f"cvvdp-gpu tick 171 reference (same host):")
    print(f"  compute_dkl_jod (cold):           36.1 ns/px")
    print(f"  compute_dkl_jod_with_warm_ref:    20.6 ns/px")

    # Emit goldens manifest. Floats stored at 4-decimal precision to
    # match the granularity Rust tests use (any drift > 1e-4 surfaces
    # in the tolerance comparison).
    manifest = {
        "schema_version": 1,
        "pycvvdp_version": "0.5.4",
        "display_name": "standard_4k",
        "pix_per_deg": float(metric.pix_per_deg) if hasattr(metric, "pix_per_deg") else None,
        "fixtures": {
            "synth_4000x3000": {
                "shape_hw": [H, W],
                "construction": "synth_pair_12mp",
                "jod": float(jod_warm),  # use median-equivalent first reading
            },
            "synth_73x91_odd": {
                "shape_hw": [91, 73],
                "construction": "synth_pair_odd_dim",
                "jod": odd_jod_val,
            },
        },
    }
    out_path = Path(__file__).parent / "pycvvdp_synth_goldens.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote goldens manifest: {out_path}")


if __name__ == "__main__":
    main()
