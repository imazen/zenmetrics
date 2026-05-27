"""Five-way bench: pycvvdp CPU + CUDA, cvvdp-gpu Full + Mode B, cvvdp-cpu."""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import pycvvdp

REPO = Path("/home/lilith/work/zen/zenmetrics").resolve()
N_ITERS = 5
SIZES = (int(s) for s in sys.argv[1:]) if len(sys.argv) > 1 else (1024, 2048, 4096)

torch.set_grad_enabled(False)


def synth_pair(size: int):
    rng = np.random.default_rng(0xC0FFEE)
    ref = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    dist = ref.copy()
    dx, dy = 3, 2
    dist[dy:, dx:, :] = ref[:-dy, :-dx, :]
    return ref, dist


def bench_pycvvdp(ref_np, dist_np, device_str, n=N_ITERS):
    dev = torch.device(device_str)
    ref_t = torch.from_numpy(ref_np).to(dev).float() / 255.0
    dist_t = torch.from_numpy(dist_np).to(dev).float() / 255.0
    metric = pycvvdp.cvvdp(display_name="standard_4k", device=dev)
    metric.predict(ref_t, dist_t, dim_order="HWC")
    if device_str == "cuda":
        torch.cuda.synchronize()
    ts, last_jod = [], None
    for _ in range(n):
        if device_str == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        jod, _ = metric.predict(ref_t, dist_t, dim_order="HWC")
        if device_str == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
        last_jod = float(jod)
    return ts, last_jod


def bench_cvvdp_gpu(mode, size, n=N_ITERS):
    example = REPO / "target/release/examples/bench_4096_one_mode"
    r = subprocess.run(
        [str(example), mode, str(size), str(n)],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(r.stdout.splitlines()[-1])
    return info["per_iter_seconds"], info["jod"]


def bench_cvvdp_cpu(size, n=N_ITERS):
    example = REPO / "target/release/examples/bench_one_mode"
    r = subprocess.run(
        [str(example), str(size), str(n)],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(r.stdout.splitlines()[-1])
    return info["per_iter_seconds"], info["jod"]


def fmt(label, ts, jod, size):
    arr = np.array(ts)
    p50 = float(np.median(arr))
    n_pix = size * size
    return (
        f"{label:30s}  p50={p50 * 1000:9.1f} ms  ({p50 * 1e9 / n_pix:6.2f} ns/px)  jod={jod:.4f}"
    )


def main():
    for size in SIZES:
        print(f"\n== {size}² ==")
        ref, dist = synth_pair(size)

        # CUDA paths first (fastest, smallest sample size needed)
        t, jod = bench_pycvvdp(ref, dist, "cuda")
        print(fmt("pycvvdp CUDA", t, jod, size))

        t, jod = bench_cvvdp_gpu("full", size)
        print(fmt("cvvdp-gpu Full (cubecl-CUDA)", t, jod, size))

        t, jod = bench_cvvdp_gpu("strip_pair", size)
        print(fmt("cvvdp-gpu Mode B (cubecl-CUDA)", t, jod, size))

        # CPU paths
        t, jod = bench_cvvdp_cpu(size, n=max(2, N_ITERS // 2) if size >= 4096 else N_ITERS)
        print(fmt("cvvdp-cpu (Rust SIMD)", t, jod, size))

        t, jod = bench_pycvvdp(ref, dist, "cpu", n=max(2, N_ITERS // 2) if size >= 4096 else N_ITERS)
        print(fmt("pycvvdp CPU", t, jod, size))


if __name__ == "__main__":
    main()
