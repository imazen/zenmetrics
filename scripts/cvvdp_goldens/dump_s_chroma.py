"""Dump pycvvdp's S (raw CSF sensitivity, pre-sens_corr) at
chroma_shift sentinels.

Tick 202 stage-5 parity probe (after tick 201 localized 7% rel
divergence in D at band 4, attributed to masking amplification of
the 0.9% rel T_p drift from tick 199).

Since T (weber) is bit-identical (tick 198) and ch_gain is constant
([1, 1.45, 1]), the only T_p input that can carry the 0.9% drift is
S. Dumping raw S (`csf.sensitivity(rho, omega=0, logL_ref, cc, sigma=0)`)
at chroma_shift sentinels isolates whether the CSF lookup is the
divergence source.

If raw S matches between pycvvdp and our `sensitivity_scalar` at
0.005% rel (= f32 noise floor), the divergence is in
`sens_corr_factor` application order. If raw S diverges, the CSF
lookup table or interp method is the source.

Schema: per-band per-channel S values at the same sentinel pixels
as the T_p / D dumps, indexed by REF's log_l_bkg per pixel.
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
    sens_corr_factor = 10.0 ** (sens_corr / 20.0)
    csf = metric.csf

    def to_dkl(rgb_bytes):
        V_hwc = torch.as_tensor(rgb_bytes, dtype=torch.float32) / 255.0
        V = V_hwc.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        L = display_photo.forward(V)
        return display_photo.linear_2_target_colorspace(L, "DKLd65")

    dkl_ref = to_dkl(ref)
    dkl_dist = to_dkl(dist)

    interleaved = torch.stack(
        [
            dkl_dist[0, 0], dkl_ref[0, 0],
            dkl_dist[0, 1], dkl_ref[0, 1],
            dkl_dist[0, 2], dkl_ref[0, 2],
        ],
        dim=0,
    ).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    interleaved = interleaved.to(device)
    lpyr = weber_contrast_pyr(W, H, ppd, device=device, contrast="weber_g1")
    _, log_l_bkg_pyr = lpyr.decompose(interleaved)
    rho_band = lpyr.band_freqs

    n_bands = len(log_l_bkg_pyr)
    print(f"pycvvdp: {n_bands} bands, ppd={ppd}, sens_corr={sens_corr}")

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
        "sens_corr_factor": sens_corr_factor,
        "n_bands": n_bands,
        "bands": [],
    }

    for k in range(n_bands):
        rho_k = float(rho_band[k])
        logL_bkg = log_l_bkg_pyr[k]
        bh = logL_bkg.shape[-2]
        bw = logL_bkg.shape[-1]

        # REF side: pycvvdp uses logL_bkg[..., 1:2, :, :, :] for CSF.
        if logL_bkg.shape[-4] >= 2:
            logL_ref = logL_bkg[:, 1:2, :, :, :]
        else:
            logL_ref = logL_bkg[:, 0:1, :, :, :]

        # Compute S per channel (raw, NO sens_corr applied).
        S_list = []
        for cc in range(3):
            S_cc = csf.sensitivity(rho_k, 0, logL_ref, cc, 0)
            S_cc_flat = S_cc.view(-1)
            if S_cc_flat.numel() == 1:
                S_cc_2d = S_cc_flat.expand(bh, bw)
            else:
                S_cc_2d = S_cc.view(bh, bw)
            S_list.append(S_cc_2d.to(device))
        S = torch.stack(S_list, dim=0)  # (3, bh, bw)

        # Also expose logL_ref at the sentinel pixels so we can
        # pass it into our host_scalar's sensitivity_scalar in the
        # parity test (same input → same expected output).
        logL_ref_2d = logL_ref.view(bh, bw).to(device)

        band_rec = {
            "k": k,
            "rho": rho_k,
            "shape_hw": [bh, bw],
            "samples": [],
        }
        for y0, x0 in l0_sentinels:
            yk = min(y0 // (1 << k), bh - 1)
            xk = min(x0 // (1 << k), bw - 1)
            sample = {
                "y0": y0, "x0": x0, "yk": yk, "xk": xk,
                "log_l_bkg_ref": float(logL_ref_2d[yk, xk]),
                "s_raw_a":  float(S[0, yk, xk]),
                "s_raw_rg": float(S[1, yk, xk]),
                "s_raw_vy": float(S[2, yk, xk]),
            }
            band_rec["samples"].append(sample)
        out["bands"].append(band_rec)
        print(
            f"band {k} rho={rho_k:.3f} shape=({bh},{bw}) "
            f"S_raw sample[0]: a={band_rec['samples'][0]['s_raw_a']:.4e} "
            f"rg={band_rec['samples'][0]['s_raw_rg']:.4e} "
            f"logL={band_rec['samples'][0]['log_l_bkg_ref']:.4f}"
        )

    path = Path(__file__).parent / "pycvvdp_s_chroma_shift.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote: {path}")


if __name__ == "__main__":
    main()
