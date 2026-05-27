"""Four-way bench at 1024²: pycvvdp CPU + CUDA, cvvdp-gpu Full + Mode B."""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import pycvvdp

REPO = Path("/home/lilith/work/zen/zenmetrics").resolve()
SIZE = 1024
N_ITERS = 5

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
    # warm-up
    metric.predict(ref_t, dist_t, dim_order="HWC")
    if device_str == "cuda":
        torch.cuda.synchronize()
    ts = []
    last_jod = None
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


def bench_cvvdp_gpu(mode, size=SIZE, n=N_ITERS):
    example = REPO / "target/release/examples/bench_4096_one_mode"
    r = subprocess.run(
        [str(example), mode, str(size), str(n)],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(r.stdout.splitlines()[-1])
    return info["per_iter_seconds"], info["jod"]


def fmt(label, ts, jod, size):
    arr = np.array(ts)
    p50 = float(np.median(arr))
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    n_pix = size * size
    return (
        f"{label:30s}  p25={p25*1000:8.1f} ms  p50={p50*1000:8.1f} ms  "
        f"p75={p75*1000:8.1f} ms  ({p50 * 1e9 / n_pix:6.2f} ns/px)  jod={jod:.4f}"
    )


def main():
    print(f"\n== {SIZE}² four-way bench ==\n")
    ref, dist = synth_pair(SIZE)

    print("warming up pycvvdp-CUDA...")
    t_py_cuda, jod_py_cuda = bench_pycvvdp(ref, dist, "cuda")
    print(fmt("pycvvdp CUDA", t_py_cuda, jod_py_cuda, SIZE))

    print("warming up pycvvdp-CPU (this may take a while)...")
    t_py_cpu, jod_py_cpu = bench_pycvvdp(ref, dist, "cpu")
    print(fmt("pycvvdp CPU", t_py_cpu, jod_py_cpu, SIZE))

    print("running cvvdp-gpu Full...")
    t_us_full, jod_us_full = bench_cvvdp_gpu("full")
    print(fmt("cvvdp-gpu Full (cubecl-CUDA)", t_us_full, jod_us_full, SIZE))

    print("running cvvdp-gpu Mode B...")
    t_us_b, jod_us_b = bench_cvvdp_gpu("strip_pair")
    print(fmt("cvvdp-gpu Mode B (cubecl-CUDA)", t_us_b, jod_us_b, SIZE))


if __name__ == "__main__":
    main()
