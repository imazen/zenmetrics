"""Dump pycvvdp's Weber-contrast pyramid bands at chroma_shift sentinels.

Tick 197 next-stage parity probe (after tick 196's DKL fix
verified DKL is now bit-identical). If our Weber bands match
pycvvdp's at sentinel pixels, drift is downstream (CSF / masking
/ pool). If they don't, weber is the source.
"""

import json
from pathlib import Path

import numpy as np
import torch

import pycvvdp
from pycvvdp.lpyr_dec import weber_contrast_pyr
from pycvvdp import display_model as dm

W, H = 256, 256


def synth_pair_256_chroma_shift():
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    r = ((xx * 17 + yy * 5) % 251).astype(np.uint8) + 40
    g = ((xx * 11 + yy * 13) % 247).astype(np.uint8) + 40
    b = ((xx * 7 + yy * 19) % 241).astype(np.uint8) + 40
    ref = np.stack([r, g, b], axis=-1)
    dist = ref.copy()
    dist[..., 1] = np.clip(ref[..., 1].astype(np.int16) + 16, 0, 255).astype(np.uint8)
    return ref, dist


def main():
    ref, dist = synth_pair_256_chroma_shift()
    metric = pycvvdp.cvvdp(display_name="standard_4k", heatmap="none")
    display_photo = metric.display_photometry
    ppd = float(metric.pix_per_deg)

    # Convert HWC uint8 to pycvvdp's expected shape (1, C, F, H, W) f32.
    def to_dkl(rgb_bytes):
        V_hwc = torch.as_tensor(rgb_bytes, dtype=torch.float32) / 255.0
        V = V_hwc.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # (1, 3, 1, H, W)
        L = display_photo.forward(V)
        dkl_torch = display_photo.linear_2_target_colorspace(L, "DKLd65")
        return dkl_torch  # (1, 3, 1, H, W)

    dkl_ref = to_dkl(ref)
    dkl_dist = to_dkl(dist)

    # pycvvdp interleaves test/ref/test/ref/... across the channel axis
    # for predict's lpyr.decompose. Match that: (1, 8, 1, H, W) with
    # ordering [test_A, ref_A, test_RG, ref_RG, test_VY, ref_VY,
    #          test_trans, ref_trans] — for still-image we have 0
    # transient channels effectively, so 6 channels. weber_contrast_pyr
    # accepts the 6-channel still-image case.
    # Stack interleaved: ABC means [t_A, r_A, t_RG, r_RG, t_VY, r_VY]
    interleaved = torch.stack(
        [
            dkl_dist[0, 0],  # test A
            dkl_ref[0, 0],   # ref  A
            dkl_dist[0, 1],  # test RG
            dkl_ref[0, 1],   # ref  RG
            dkl_dist[0, 2],  # test VY
            dkl_ref[0, 2],   # ref  VY
        ],
        dim=0,
    ).unsqueeze(0)  # (1, 6, 1, H, W)

    lpyr = weber_contrast_pyr(W, H, ppd, device=torch.device("cpu"), contrast="weber_g1")
    bands, log_l_bkg = lpyr.decompose(interleaved)
    print(f"pycvvdp pyramid: {len(bands)} bands, height={lpyr.height}")
    for i, b in enumerate(bands):
        print(f"  band[{i}] shape: {tuple(b.shape)}")

    # Sentinel pixels (level-0 coordinates).
    l0_sentinels = [
        (0, 0), (0, 255), (255, 0), (255, 255),
        (128, 128), (64, 192), (192, 64), (100, 100),
        (37, 91), (191, 13),
    ]

    out = {
        "schema_version": 1,
        "fixture": "synth_256x256_chroma_shift",
        "ppd": ppd,
        "n_bands": len(bands),
        "bands": [],
    }

    # For each band, sample at the level-0-mapped coordinate scaled
    # down by 2^k (with ceil-div per cvvdp).
    for k, b in enumerate(bands):
        bh = b.shape[-2]
        bw = b.shape[-1]
        # Scale-down factor from level 0 to level k: divisor = 2^k
        # using ceil-div (matches pycvvdp's pyramid construction).
        # We just sample at (y // 2^k, x // 2^k) clamped to the band's
        # actual shape.
        band_rec = {"k": k, "shape_hw": [bh, bw], "samples": []}
        # band shape: (1, 6, 1, bh, bw). channels:
        # [test_A, ref_A, test_RG, ref_RG, test_VY, ref_VY]
        for y0, x0 in l0_sentinels:
            yk = min(y0 // (1 << k), bh - 1)
            xk = min(x0 // (1 << k), bw - 1)
            sample = {
                "y0": y0, "x0": x0, "yk": yk, "xk": xk,
                "test_A": float(b[0, 0, 0, yk, xk]),
                "ref_A":  float(b[0, 1, 0, yk, xk]),
                "test_RG": float(b[0, 2, 0, yk, xk]),
                "ref_RG":  float(b[0, 3, 0, yk, xk]),
                "test_VY": float(b[0, 4, 0, yk, xk]),
                "ref_VY":  float(b[0, 5, 0, yk, xk]),
            }
            band_rec["samples"].append(sample)
        out["bands"].append(band_rec)

    path = Path(__file__).parent / "pycvvdp_weber_chroma_shift.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote: {path}")


if __name__ == "__main__":
    main()
