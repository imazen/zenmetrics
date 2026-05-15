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


def synth_pair_256_chroma_shift(w=256, h=256):
    """256×256 ref with a chrominance-only distortion: G channel
    gains a uniform +16 offset (clamped), R and B unchanged. Tests
    the DKL RG/VY response in isolation — the other 256² fixtures
    (blur3x1, blur1x3, noise) perturb all three channels roughly
    equally, so they don't isolate chromatic vs achromatic sensitivity.

    dist[y,x,R] = ref[y,x,R]
    dist[y,x,G] = clamp(ref[y,x,G] + 16, 0, 255)
    dist[y,x,B] = ref[y,x,B]

    Pure integer ops, bit-stable across NumPy + Rust.
    """
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    r = ((xx * 17 + yy * 5) % 251).astype(np.uint8) + 40
    g = ((xx * 11 + yy * 13) % 247).astype(np.uint8) + 40
    b = ((xx * 7 + yy * 19) % 241).astype(np.uint8) + 40
    ref = np.stack([r, g, b], axis=-1)
    dist = ref.copy()
    dist[..., 1] = np.clip(ref[..., 1].astype(np.int16) + 16, 0, 255).astype(np.uint8)
    return ref, dist


def synth_pair_256_blur1x3(w=256, h=256):
    """256×256 ref with a 3-pixel VERTICAL average distortion.
    Complement to `synth_pair_256_blur3x1` (horizontal) — together
    they exercise both axes of the separable Gaussian pyramid.

    dist[y,x,c] = (ref[y,x,c] + ref[(y+1)%h,x,c] + ref[(y+2)%h,x,c]) // 3

    Pure integer ops, bit-stable across NumPy + Rust.
    """
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    r = ((xx * 17 + yy * 5) % 251).astype(np.uint8) + 40
    g = ((xx * 11 + yy * 13) % 247).astype(np.uint8) + 40
    b = ((xx * 7 + yy * 19) % 241).astype(np.uint8) + 40
    ref = np.stack([r, g, b], axis=-1)
    ref16 = ref.astype(np.uint16)
    ref16_y1 = np.roll(ref16, shift=-1, axis=0)
    ref16_y2 = np.roll(ref16, shift=-2, axis=0)
    dist = ((ref16 + ref16_y1 + ref16_y2) // 3).astype(np.uint8)
    return ref, dist


def synth_pair_256_noise(w=256, h=256):
    """256×256 ref with per-pixel-per-channel deterministic
    additive noise. Tests a noise-type distortion (no spatial
    correlation), complementary to the spatial-blur fixture.

    noise[y,x,c] = ((x * 73 + y * 137 + c * 211) % 64) - 32
    dist[y,x,c] = clamp(ref[y,x,c] + noise[y,x,c], 0, 255)

    Pure integer arithmetic — bit-stable across NumPy + Rust.
    """
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    r = ((xx * 17 + yy * 5) % 251).astype(np.uint8) + 40
    g = ((xx * 11 + yy * 13) % 247).astype(np.uint8) + 40
    b = ((xx * 7 + yy * 19) % 241).astype(np.uint8) + 40
    ref = np.stack([r, g, b], axis=-1)

    cc = np.arange(3, dtype=np.int64).reshape(1, 1, 3)
    yy3 = yy.astype(np.int64)[..., None]
    xx3 = xx.astype(np.int64)[..., None]
    noise = ((xx3 * 73 + yy3 * 137 + cc * 211) % 64) - 32
    dist = np.clip(ref.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
    return ref, dist


def synth_pair_256_blur3x1(w=256, h=256):
    """256×256 reference with a deterministic 3-pixel horizontal
    average as the distortion. Tests parity at the common 256² size
    used by the v1 manifest corpus, but using pure-integer DIST
    construction so the Rust test can reproduce the exact bytes
    without depending on the zenmetrics-corpus PNG/JPEG files.

    dist[y,x,c] = (ref[y,x,c] + ref[y,(x+1)%w,c] + ref[y,(x+2)%w,c]) // 3

    Pure integer ops (u8 → u16 → u8 floor-div) — bit-stable across
    NumPy and Rust.
    """
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    r = ((xx * 17 + yy * 5) % 251).astype(np.uint8) + 40
    g = ((xx * 11 + yy * 13) % 247).astype(np.uint8) + 40
    b = ((xx * 7 + yy * 19) % 241).astype(np.uint8) + 40
    ref = np.stack([r, g, b], axis=-1)
    # Horizontal 3-pixel average with wrap. Cast to u16 before the
    # sum to avoid overflow at u8 + u8 + u8 (could hit 765 > 255).
    ref16 = ref.astype(np.uint16)
    ref16_x1 = np.roll(ref16, shift=-1, axis=1)
    ref16_x2 = np.roll(ref16, shift=-2, axis=1)
    dist = ((ref16 + ref16_x1 + ref16_x2) // 3).astype(np.uint8)
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
    print(f"  jod = {float(odd_jod):.4f}")
    odd_jod_val = float(odd_jod)

    # 256x256 fixture with deterministic blur-3x1 distortion. Tests
    # the common 256² size without depending on the zenmetrics-corpus
    # PNG/JPEG files (so Rust tests can run in fully offline contexts).
    blur256_ref, blur256_dist = synth_pair_256_blur3x1()
    print("256x256 blur3x1 golden:")
    blur256_jod, _ = metric.predict(blur256_dist, blur256_ref, dim_order="HWC")
    print(f"  jod = {float(blur256_jod):.4f}")
    blur256_jod_val = float(blur256_jod)

    # 256x256 fixture with per-pixel additive noise (uncorrelated
    # across pixels and channels). Complementary to the spatial-blur
    # fixture — both tap different parts of the CSF/masking response.
    noise256_ref, noise256_dist = synth_pair_256_noise()
    print("256x256 noise golden:")
    noise256_jod, _ = metric.predict(noise256_dist, noise256_ref, dim_order="HWC")
    print(f"  jod = {float(noise256_jod):.4f}")
    noise256_jod_val = float(noise256_jod)

    # 256x256 fixture with VERTICAL 3-pixel blur — complement to
    # blur3x1 so the test sweep covers both axes of the separable
    # pyramid passes.
    vblur256_ref, vblur256_dist = synth_pair_256_blur1x3()
    print("256x256 blur1x3 golden:")
    vblur256_jod, _ = metric.predict(vblur256_dist, vblur256_ref, dim_order="HWC")
    print(f"  jod = {float(vblur256_jod):.4f}")
    vblur256_jod_val = float(vblur256_jod)

    # 256x256 fixture with chrominance-only distortion (G+16). Isolates
    # the DKL RG/VY channel response — other 256² fixtures perturb all
    # three R/G/B channels roughly equally.
    chroma256_ref, chroma256_dist = synth_pair_256_chroma_shift()
    print("256x256 chroma_shift golden:")
    chroma256_jod, _ = metric.predict(chroma256_dist, chroma256_ref, dim_order="HWC")
    print(f"  jod = {float(chroma256_jod):.4f}\n")
    chroma256_jod_val = float(chroma256_jod)

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
            "synth_256x256_blur3x1": {
                "shape_hw": [256, 256],
                "construction": "synth_pair_256_blur3x1",
                "jod": blur256_jod_val,
            },
            "synth_256x256_noise": {
                "shape_hw": [256, 256],
                "construction": "synth_pair_256_noise",
                "jod": noise256_jod_val,
            },
            "synth_256x256_blur1x3": {
                "shape_hw": [256, 256],
                "construction": "synth_pair_256_blur1x3",
                "jod": vblur256_jod_val,
            },
            "synth_256x256_chroma_shift": {
                "shape_hw": [256, 256],
                "construction": "synth_pair_256_chroma_shift",
                "jod": chroma256_jod_val,
            },
        },
    }
    out_path = Path(__file__).parent / "pycvvdp_synth_goldens.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote goldens manifest: {out_path}")


if __name__ == "__main__":
    main()
