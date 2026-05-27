"""Head-to-head wall-time bench at 4096² between:
  - pycvvdp (the reference Python implementation, CUDA backend)
  - cvvdp-gpu (our Rust implementation, both Full and Mode B / StripPair)

Same synthetic ref/dist pair on both sides; CUDA warm-up first; median
of N timed iterations. Run from anywhere; uses absolute paths.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import pycvvdp

REPO = Path("/home/lilith/work/zen/zenmetrics").resolve()
SIZE = 4096
N_ITERS = 5

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


def synth_pair_4096(size: int):
    rng = np.random.default_rng(0xC0FFEE)
    ref = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    dist = ref.copy()
    dx, dy = 3, 2
    dist[dy:, dx:, :] = ref[:-dy, :-dx, :]
    return ref, dist


def bench_pycvvdp(ref_np, dist_np, n=N_ITERS):
    ref_t = torch.from_numpy(ref_np).to(device).float() / 255.0
    dist_t = torch.from_numpy(dist_np).to(device).float() / 255.0
    metric = pycvvdp.cvvdp(display_name="standard_4k", device=device)
    metric.predict(ref_t, dist_t, dim_order="HWC")
    torch.cuda.synchronize()
    ts = []
    last_jod = None
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        jod, _ = metric.predict(ref_t, dist_t, dim_order="HWC")
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
        last_jod = float(jod)
    return ts, last_jod


def bench_cvvdp_gpu(mode: str, n=N_ITERS):
    """Drive cvvdp-gpu via a small inline Rust program built once.

    `mode` is one of "full" or "strip_pair". Returns (per-iter seconds, jod).
    """
    bin_dir = REPO / "target/release/examples"
    example = bin_dir / "bench_4096_one_mode"
    if not example.exists():
        raise RuntimeError(
            f"Build first: cargo build --release -p cvvdp-gpu --features cuda --example bench_4096_one_mode"
        )
    times = []
    last_jod = None
    for _ in range(n):
        r = subprocess.run(
            [str(example), mode, str(SIZE), str(N_ITERS)],
            capture_output=True,
            text=True,
            check=True,
        )
        info = json.loads(r.stdout.splitlines()[-1])
        times = info["per_iter_seconds"]
        last_jod = info["jod"]
        break  # one driver call already does N internally
    return times, last_jod


def fmt_summary(label, ts, jod):
    arr = np.array(ts)
    median = float(np.median(arr))
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    n_pix = SIZE * SIZE
    return (
        f"{label:25s}  p25={p25*1000:7.1f} ms  p50={median*1000:7.1f} ms  "
        f"p75={p75*1000:7.1f} ms  ({median * 1e9 / n_pix:.2f} ns/px)  jod={jod:.4f}"
    )


def main():
    print(f"\n== {SIZE}² head-to-head: pycvvdp vs cvvdp-gpu ==\n")
    ref, dist = synth_pair_4096(SIZE)

    print("warming up pycvvdp...")
    t_py, jod_py = bench_pycvvdp(ref, dist)
    print(fmt_summary("pycvvdp (CUDA)", t_py, jod_py))

    for mode in ("full", "strip_pair"):
        print(f"running cvvdp-gpu {mode}...")
        t_us, jod_us = bench_cvvdp_gpu(mode)
        label = "cvvdp-gpu " + ("Full" if mode == "full" else "Mode B")
        print(fmt_summary(label, t_us, jod_us))


if __name__ == "__main__":
    main()
