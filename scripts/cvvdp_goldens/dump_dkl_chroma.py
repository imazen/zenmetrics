"""Dump pycvvdp's DKL planes at 10 sentinel pixels of the
chroma_shift fixture. Used to localize the 0.117 JOD drift
surfaced in tick 191.

If our compute_dkl_planes matches pycvvdp at these pixels, the
drift is downstream of color transform (in the pyramid, CSF,
masking, or pool). If it doesn't, the color transform itself is
the source (despite the SRGB_LINEAR_TO_DKL matrix matching at
f64 precision — tick 192).
"""

import json
from pathlib import Path

import numpy as np
import torch

import pycvvdp
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

    # Convert HWC uint8 to pycvvdp's expected shape (1, C, F, H, W) f32 in [0, 1].
    # pycvvdp's display model + DKL transform happens inside the metric
    # pipeline; reproducing it directly requires walking the same code path.
    #
    # video_source_array drops in for our needs.
    from pycvvdp import video_source

    # standard_4k is the default — its EOTF is sRGB.
    display_photo = metric.display_photometry  # already configured
    print(f"Display: y_peak={display_photo.Y_peak}, ", end="")
    y_black, y_refl = display_photo.get_black_level()
    print(f"y_black={float(y_black):.6f}, y_refl={float(y_refl):.6f}")

    # Run the display model forward + DKL transform manually so we
    # get the intermediate L_DKL planes pycvvdp would feed to the
    # pyramid. The conversion stages:
    #   sRGB byte / 255  ->  V in [0,1]
    #   forward(V)       ->  L (cd/m^2, linear RGB)
    #   L @ rgb2abc.T    ->  DKL (after matmul with combined matrix)
    def to_dkl(rgb_bytes):
        # Run through pycvvdp's actual code path: source_2_target_colorspace
        # which composes EOTF forward + matmul exactly as predict() does
        # (line 268-269 of display_model.py). Shape conversion:
        # (H, W, 3) uint8 → (1, 3, 1, H, W) f32 in [0, 1].
        V_hwc = torch.as_tensor(rgb_bytes, dtype=torch.float32) / 255.0
        V = V_hwc.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # (1, 3, 1, H, W)
        # Apply EOTF forward → linear cd/m^2 RGB
        L = display_photo.forward(V)
        # linear → DKLd65 via pycvvdp's internal helper
        dkl_torch = display_photo.linear_2_target_colorspace(L, "DKLd65")
        # (1, 3, 1, H, W) → (H, W, 3)
        dkl = dkl_torch.squeeze(0).squeeze(1).permute(1, 2, 0).numpy()
        return dkl

    dkl_ref = to_dkl(ref)
    dkl_dist = to_dkl(dist)

    # Sentinel pixels: corners + center + a few off-grid points.
    sentinels = [
        (0, 0), (0, 255), (255, 0), (255, 255),
        (128, 128), (64, 192), (192, 64), (100, 100),
        (37, 91), (191, 13),
    ]
    out = {
        "schema_version": 1,
        "fixture": "synth_256x256_chroma_shift",
        "shape_hw": [H, W],
        "channels": ["A", "RG", "VY"],
        "sentinels": [],
    }
    for y, x in sentinels:
        rec = {
            "y": int(y),
            "x": int(x),
            "ref_srgb_u8": [int(ref[y, x, 0]), int(ref[y, x, 1]), int(ref[y, x, 2])],
            "dist_srgb_u8": [int(dist[y, x, 0]), int(dist[y, x, 1]), int(dist[y, x, 2])],
            "ref_dkl_f32": [float(dkl_ref[y, x, 0]), float(dkl_ref[y, x, 1]), float(dkl_ref[y, x, 2])],
            "dist_dkl_f32": [float(dkl_dist[y, x, 0]), float(dkl_dist[y, x, 1]), float(dkl_dist[y, x, 2])],
        }
        out["sentinels"].append(rec)
        print(
            f"  ({y:>3},{x:>3})  ref_rgb={tuple(rec['ref_srgb_u8'])}  "
            f"ref_dkl=({rec['ref_dkl_f32'][0]:.4f}, {rec['ref_dkl_f32'][1]:.4f}, {rec['ref_dkl_f32'][2]:.4f})"
        )

    path = Path(__file__).parent / "pycvvdp_dkl_chroma_shift.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote: {path}")


if __name__ == "__main__":
    main()
