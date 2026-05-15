"""Dump pycvvdp's T_p (post-CSF) at chroma_shift sentinels.

Tick 199 stage-3 parity probe (after tick 198 confirmed Weber is
bit-identical). T_p is what `apply_masking_model` computes:
  T_p = T * S * ch_gain
where T is the weber-contrast band, S is the per-pixel CSF
sensitivity (with sensitivity_correction baked in), ch_gain is
the per-channel weighting ([1, 1.45, 1] for non-baseband,
[1, 1, 1] for baseband-bypass).

Note: pycvvdp's lpyr.get_band(B_bands, bb) multiplies non-edge
bands (k in [1, n-1)) by 2.0. We mirror that here so the T_p
values reflect exactly what apply_masking_model would see.
"""

import json
from pathlib import Path

import numpy as np
import torch

import pycvvdp
from pycvvdp.lpyr_dec import weber_contrast_pyr

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
    sens_corr = float(metric.sensitivity_correction)

    def to_dkl(rgb_bytes):
        V_hwc = torch.as_tensor(rgb_bytes, dtype=torch.float32) / 255.0
        V = V_hwc.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        L = display_photo.forward(V)
        return display_photo.linear_2_target_colorspace(L, "DKLd65")

    dkl_ref = to_dkl(ref)
    dkl_dist = to_dkl(dist)

    # Interleaved channel stack for weber_contrast_pyr: [t_A, r_A, t_RG, r_RG, t_VY, r_VY].
    interleaved = torch.stack(
        [
            dkl_dist[0, 0], dkl_ref[0, 0],
            dkl_dist[0, 1], dkl_ref[0, 1],
            dkl_dist[0, 2], dkl_ref[0, 2],
        ],
        dim=0,
    ).unsqueeze(0)  # (1, 6, 1, H, W)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    interleaved = interleaved.to(device)
    lpyr = weber_contrast_pyr(W, H, ppd, device=device, contrast="weber_g1")
    bands, log_l_bkg_pyr = lpyr.decompose(interleaved)
    rho_band = lpyr.band_freqs  # one rho per band

    n_bands = len(bands)
    print(f"pycvvdp: {n_bands} bands, rho_band={list(rho_band)}")
    print(f"ppd={ppd}, sens_corr={sens_corr}")

    # ch_gain = [1, 1.45, 1, 1.] for 4ch, sliced to [1, 1.45, 1] for 3ch still-image.
    ch_gain_nonbb = np.array([1.0, 1.45, 1.0], dtype=np.float32)
    csf = metric.csf

    l0_sentinels = [
        (0, 0), (0, 255), (255, 0), (255, 255),
        (128, 128), (64, 192), (192, 64), (100, 100),
        (37, 91), (191, 13),
    ]

    out = {
        "schema_version": 1,
        "fixture": "synth_256x256_chroma_shift",
        "ppd": ppd,
        "sensitivity_correction": sens_corr,
        "ch_gain_nonbb": ch_gain_nonbb.tolist(),
        "n_bands": n_bands,
        "bands": [],
    }

    for k, b in enumerate(bands):
        is_baseband = (k == n_bands - 1)
        # band_mul: pycvvdp's lpyr.get_band returns bands[k] * band_mul
        # where band_mul = 1.0 for k=0 OR k=n-1, 2.0 otherwise.
        band_mul = 1.0 if (k == 0 or is_baseband) else 2.0
        rho_k = float(rho_band[k])
        bh = b.shape[-2]
        bw = b.shape[-1]
        logL_bkg = log_l_bkg_pyr[k]  # (1, 2 or other, 1, bh, bw) — has 2 channels at baseband too?

        # CSF expects logL_bkg of shape (..., 1, 1, ch_height, ch_width)
        # and returns S of shape (..., 1, 1, ch_height, ch_width).
        # For each of 3 channels, compute S separately, apply sens_corr.
        sens_corr_factor = 10.0 ** (sens_corr / 20.0)

        # Compute S per channel. cc=0 (A), 1 (RG), 2 (VY).
        # logL_bkg has shape [1, 2 or 6, 1, bh, bw] — pycvvdp's apply_masking_model
        # uses logL_bkg[:, 1:2, ...] (the REF side, channel 1).
        if logL_bkg.shape[-4] >= 2:
            logL_ref = logL_bkg[:, 1:2, :, :, :]
        else:
            logL_ref = logL_bkg[:, 0:1, :, :, :]

        # Compute S per channel and broadcast to (3, bh, bw).
        # At baseband, logL_ref has spatial dims (1, 1) — S is a
        # scalar per channel; broadcast it.
        S_list = []
        for cc in range(3):
            S_cc = csf.sensitivity(rho_k, 0, logL_ref, cc, 0)
            S_cc = S_cc * sens_corr_factor
            # Squeeze leading dims, broadcast to (bh, bw) if scalar.
            S_cc_flat = S_cc.view(-1)
            if S_cc_flat.numel() == 1:
                S_cc_2d = S_cc_flat.expand(bh, bw)
            else:
                S_cc_2d = S_cc.view(bh, bw)
            S_list.append(S_cc_2d)
        S = torch.stack(S_list, dim=0)  # (3, bh, bw)

        # Weber bands have shape (1, 6, 1, bh, bw). Test = 0::2, Ref = 1::2.
        T_test = b[0, 0::2, 0] * band_mul  # (3, bh, bw)
        T_ref = b[0, 1::2, 0] * band_mul

        # T_p = T * S * ch_gain (for non-baseband) or T * S (for baseband).
        if is_baseband:
            ch_gain = torch.ones(3, dtype=torch.float32, device=device).view(3, 1, 1)
        else:
            ch_gain = torch.as_tensor(ch_gain_nonbb, device=device).view(3, 1, 1)
        T_p_test = T_test.to(device) * S * ch_gain
        T_p_ref = T_ref.to(device) * S * ch_gain

        band_rec = {"k": k, "rho": rho_k, "band_mul": band_mul, "is_baseband": is_baseband, "shape_hw": [bh, bw], "samples": []}
        for y0, x0 in l0_sentinels:
            yk = min(y0 // (1 << k), bh - 1)
            xk = min(x0 // (1 << k), bw - 1)
            sample = {
                "y0": y0, "x0": x0, "yk": yk, "xk": xk,
                "t_p_test_a":  float(T_p_test[0, yk, xk]),
                "t_p_ref_a":   float(T_p_ref[0, yk, xk]),
                "t_p_test_rg": float(T_p_test[1, yk, xk]),
                "t_p_ref_rg":  float(T_p_ref[1, yk, xk]),
                "t_p_test_vy": float(T_p_test[2, yk, xk]),
                "t_p_ref_vy":  float(T_p_ref[2, yk, xk]),
            }
            band_rec["samples"].append(sample)
        out["bands"].append(band_rec)
        print(f"band {k} rho={rho_k:.3f} band_mul={band_mul} sample[0]: t_p_test_a={band_rec['samples'][0]['t_p_test_a']:.4f}")

    path = Path(__file__).parent / "pycvvdp_tp_chroma_shift.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote: {path}")


if __name__ == "__main__":
    main()
