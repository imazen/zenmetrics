"""Fair-comparison benchmark for pycvvdp's CUDA still-image path.

Times `pycvvdp.cvvdp.predict()` on a 4000x3000 synthetic RGB pair
(same shape + same byte-level construction as `examples/time_12mp.rs`
on the cvvdp-gpu side) so the two timings are directly comparable.

Run from the repo root:
    cd scripts/cvvdp_goldens
    uv venv .venv --python python3.10
    uv pip install --python .venv/bin/python \\
        torch --index-url https://download.pytorch.org/whl/cu124
    uv pip install --python .venv/bin/python \\
        'cvvdp==0.5.4' 'pillow>=10' 'numpy>=1.26'
    .venv/bin/python bench_12mp_cuda.py

Output:
    pycvvdp 12 MP CUDA timing (5 iters, median):
      predict():         <ms>  -> <ns/px>
"""

import time

import numpy as np
import torch

import pycvvdp

W, H = 4000, 3000
ITERS = 5

# Synth pair matches the cvvdp-gpu time_12mp construction
# (same per-pixel arithmetic, so the GPU sees identical inputs).
def synth_pair():
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
    return ref, dist


def main():
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}")
        print(f"cuda capability: {torch.cuda.get_device_capability(0)}")

    ref_np, dist_np = synth_pair()
    metric = pycvvdp.cvvdp(display_name="standard_4k", heatmap="none")

    # Warm up: first .predict() triggers Torch graph compilation,
    # kernel JIT, allocator warmup. Don't time it.
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


if __name__ == "__main__":
    main()
