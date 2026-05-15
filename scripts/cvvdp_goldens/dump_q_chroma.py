"""Dump pycvvdp's Q_per_ch (post spatial-pool, pre band-pool) at
chroma_shift for the still-image 3-channel path.

Stage-7 probe: after tick 203 confirmed cvvdp-gpu's GPU JOD matches
host_scalar exactly (both 9.5476 vs pycvvdp 9.6649), and Weber +
CSF S + D bands are all bit-identical to pycvvdp by tick 198/202/201
respectively, the remaining 0.117 JOD drift must sit in pool stages
OR in something subtler that the per-band probes missed.

This dump captures pycvvdp's Q_per_ch[bs, ch, frame, band] — the
output of `lp_norm(D, beta=2, dim=(-2,-1), normalize=True)`. A
parity test against our `lp_norm_mean` over `compute_dkl_d_bands`
output will tell us if the spatial pool itself is the source.

If Q_per_ch matches: the divergence is in band/channel pooling
(Q_sc / Q_tc / met2jod). If Q_per_ch diverges: the spatial lp_norm
itself or its safe_pow application diverges.
"""

import json
from pathlib import Path

import numpy as np
import torch

import pycvvdp
from pycvvdp.lpyr_dec import weber_contrast_pyr
from pycvvdp.cvvdp_metric import safe_pow

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
    sens_corr_factor = 10.0 ** (float(metric.sensitivity_correction) / 20.0)

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
    bands, log_l_bkg_pyr = lpyr.decompose(interleaved)
    rho_band = lpyr.band_freqs

    n_bands = len(bands)
    print(f"pycvvdp: {n_bands} bands, ppd={ppd}")

    csf = metric.csf
    ch_gain_nonbb = torch.as_tensor([1.0, 1.45, 1.0], device=device).view(3, 1, 1)

    Q_per_ch = []
    for k in range(n_bands):
        is_baseband = (k == n_bands - 1)
        band_mul = 1.0 if (k == 0 or is_baseband) else 2.0
        # pycvvdp's process_block_of_frames overrides baseband rho to
        # 0.1 cy/deg (cvvdp_metric.py:628). Replicate so the dump
        # matches what the actual predict() pipeline computes.
        rho_k = 0.1 if is_baseband else float(rho_band[k])
        logL_bkg = log_l_bkg_pyr[k]
        bh = bands[k].shape[-2]
        bw = bands[k].shape[-1]

        if logL_bkg.shape[-4] >= 2:
            logL_ref = logL_bkg[:, 1:2, :, :, :]
        else:
            logL_ref = logL_bkg[:, 0:1, :, :, :]

        S_list = []
        for cc in range(3):
            S_cc = csf.sensitivity(rho_k, 0, logL_ref, cc, 0)
            S_cc = S_cc * sens_corr_factor
            S_flat = S_cc.view(-1)
            if S_flat.numel() == 1:
                S_2d = S_flat.expand(bh, bw)
            else:
                S_2d = S_cc.view(bh, bw)
            S_list.append(S_2d.to(device))
        S = torch.stack(S_list, dim=0)

        T_band = (bands[k][0, 0::2, 0] * band_mul).to(device)
        R_band = (bands[k][0, 1::2, 0] * band_mul).to(device)

        if is_baseband:
            D = (T_band - R_band).abs() * S
        else:
            T_p = T_band * S * ch_gain_nonbb
            R_p = R_band * S * ch_gain_nonbb
            M_mm = metric.phase_uncertainty(torch.min(T_p.abs(), R_p.abs()))
            mask_p = float(metric.mask_p)
            mask_q = metric.mask_q[:3].to(device).view(3, 1, 1)
            term = safe_pow(M_mm.abs(), mask_q)
            xcm_weights = torch.reshape(2.0 ** metric.xcm_weights, (4, 4))[:3, :].to(device)
            M = torch.zeros_like(term)
            for cc in range(3):
                M[cc] = (term * xcm_weights[:, cc].view(3, 1, 1)).sum(dim=0)
            D_u = safe_pow((T_p - R_p).abs(), mask_p) / (1.0 + M)
            D = metric.clamp_diffs(D_u)

        # Spatial pool exactly as cvvdp_metric does it (Tensor-p path
        # of lp_norm: safe_pow over both inner and outer ops).
        beta = metric.beta  # Tensor scalar = 2
        D_5d = D.unsqueeze(0).unsqueeze(2)  # (1, 3, 1, bh, bw)
        N = bh * bw
        # lp_norm(D, beta, dim=(-2,-1), normalize=True):
        #   inner = sum(safe_pow(D, beta), dim=(-2,-1))     # (1,3,1)
        #   q = safe_pow(inner / N, 1/beta)
        inner = safe_pow(D_5d.abs(), beta).sum(dim=(-2, -1))
        q_band = safe_pow(inner / float(N), 1.0 / beta)  # (1, 3, 1)
        Q_per_ch.append({
            "k": k,
            "rho": rho_k,
            "shape_hw": [bh, bw],
            "q_a":  float(q_band[0, 0, 0]),
            "q_rg": float(q_band[0, 1, 0]),
            "q_vy": float(q_band[0, 2, 0]),
        })
        print(
            f"band {k} rho={rho_k:.3f} Q_per_ch: a={Q_per_ch[-1]['q_a']:.6e} "
            f"rg={Q_per_ch[-1]['q_rg']:.6e} vy={Q_per_ch[-1]['q_vy']:.6e}"
        )

    out = {
        "schema_version": 1,
        "fixture": "synth_256x256_chroma_shift",
        "ppd": ppd,
        "beta_spatial": float(metric.beta),
        "beta_sch": float(metric.beta_sch),
        "beta_tch": float(metric.beta_tch),
        "image_int": float(metric.image_int),
        "jod_a": float(metric.jod_a),
        "jod_exp": float(metric.jod_exp),
        "n_bands": n_bands,
        "bands": Q_per_ch,
    }
    path = Path(__file__).parent / "pycvvdp_q_chroma_shift.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote: {path}")


if __name__ == "__main__":
    main()
