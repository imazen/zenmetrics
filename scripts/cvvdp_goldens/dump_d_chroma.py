"""Dump pycvvdp's D (post-masking, post-PU-blur, pre-pool) at
chroma_shift sentinels.

Tick 201 stage-4 parity probe (after tick 198 confirmed Weber is
bit-identical and tick 199 found T_p REF-side diverges 0.89% rel).

D is the **per-pixel masked difference** that gets fed into the
spatial lp_norm pool. For non-baseband bands with the mult-mutual
masking model + xchannel-masking on, the chain is:

    T_p = T * S * ch_gain                   # T from weber band
    R_p = R * S * ch_gain
    M_mm = phase_uncertainty(min(|T_p|, |R_p|))
    term[c] = safe_pow(|M_mm[c]|, mask_q[c])
    M[cc] = sum_{in_c} (xcm_weights[in_c, cc] * term[in_c])
    D[c] = clamp_diffs( safe_pow(|T_p - R_p|, mask_p) / (1 + M[c]) )

For baseband (last band): `D = |T - R| * S` (no masking model).

If D bands match pycvvdp's D at the same sentinel pixels, the
0.117 JOD drift sits in pool / accumulation order. If D diverges,
the masking model is the source.

Goldens land at `pycvvdp_d_chroma_shift.json`. Schema mirrors the
T_p dump: per-band records with sentinel-pixel samples.
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

    def to_dkl(rgb_bytes):
        V_hwc = torch.as_tensor(rgb_bytes, dtype=torch.float32) / 255.0
        V = V_hwc.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        L = display_photo.forward(V)
        return display_photo.linear_2_target_colorspace(L, "DKLd65")

    dkl_ref = to_dkl(ref)
    dkl_dist = to_dkl(dist)

    # Interleaved channel stack for weber_contrast_pyr:
    #   [t_A, r_A, t_RG, r_RG, t_VY, r_VY].
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
    rho_band = lpyr.band_freqs

    n_bands = len(bands)
    print(f"pycvvdp: {n_bands} bands, rho_band={list(rho_band)}")
    print(f"ppd={ppd}, sens_corr={sens_corr}, sens_corr_factor={sens_corr_factor:.6f}")

    csf = metric.csf
    ch_gain_nonbb = np.array([1.0, 1.45, 1.0], dtype=np.float32)

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
        "mask_p": float(metric.mask_p),
        "mask_q": metric.mask_q[:3].tolist(),
        "n_bands": n_bands,
        "bands": [],
    }

    for k, b in enumerate(bands):
        is_baseband = (k == n_bands - 1)
        band_mul = 1.0 if (k == 0 or is_baseband) else 2.0
        rho_k = float(rho_band[k])
        bh = b.shape[-2]
        bw = b.shape[-1]
        logL_bkg = log_l_bkg_pyr[k]

        # pycvvdp's apply_masking_model uses REF's log_L_bkg (the
        # `[..., 1:2, ...]` slice) for the CSF lookup on BOTH the
        # test and ref sides.
        if logL_bkg.shape[-4] >= 2:
            logL_ref = logL_bkg[:, 1:2, :, :, :]
        else:
            logL_ref = logL_bkg[:, 0:1, :, :, :]

        # Compute S per channel and broadcast to (3, bh, bw).
        S_list = []
        for cc in range(3):
            S_cc = csf.sensitivity(rho_k, 0, logL_ref, cc, 0)
            S_cc = S_cc * sens_corr_factor
            S_cc_flat = S_cc.view(-1)
            if S_cc_flat.numel() == 1:
                S_cc_2d = S_cc_flat.expand(bh, bw)
            else:
                S_cc_2d = S_cc.view(bh, bw)
            S_list.append(S_cc_2d)
        S = torch.stack(S_list, dim=0).to(device)  # (3, bh, bw)

        # Weber bands have shape (1, 6, 1, bh, bw). Test = 0::2, Ref = 1::2.
        T_band = (b[0, 0::2, 0] * band_mul).to(device)
        R_band = (b[0, 1::2, 0] * band_mul).to(device)

        if is_baseband:
            # Baseband bypass: D = |T - R| * S (no masking model).
            D = (T_band - R_band).abs() * S
        else:
            ch_gain = torch.as_tensor(ch_gain_nonbb, device=device).view(3, 1, 1)
            T_p = T_band * S * ch_gain  # (3, bh, bw)
            R_p = R_band * S * ch_gain

            # apply_masking_model — pycvvdp's mult-mutual + xchannel.
            # Reshape to (batch, ch, frame, h, w) so we can reuse
            # metric.apply_masking_model directly.
            T_p_5d = T_p.unsqueeze(0).unsqueeze(2)  # (1, 3, 1, bh, bw)
            R_p_5d = R_p.unsqueeze(0).unsqueeze(2)
            S_5d = torch.ones_like(T_p_5d)  # apply_masking_model expects S
            # but for mult-mutual the formula `T_p = T * S * ch_gain` is
            # internal to apply_masking_model — we already applied it
            # above. So pass S=1 and T=T_p, R=R_p, and ch_gain has been
            # baked. But apply_masking_model multiplies again by S and
            # ch_gain internally. We have to inline the apply step
            # ourselves instead of reusing the metric method.

            # Inline mult-mutual + xchannel masking, mirroring
            # cvvdp_metric.apply_masking_model's "mult-mutual" branch.
            M_mm = metric.phase_uncertainty(torch.min(T_p.abs(), R_p.abs()))
            mask_p = float(metric.mask_p)
            mask_q = metric.mask_q[:3].to(device).view(3, 1, 1)
            from pycvvdp.cvvdp_metric import safe_pow

            term = safe_pow(M_mm.abs(), mask_q)
            # mask_pool: M[cc] = sum_in xcm_weights[in, cc] * term[in]
            # xcm_weights shape after reshape: (4, 4); take first 3 rows.
            xcm_weights = torch.reshape(2.0 ** metric.xcm_weights, (4, 4))[:3, :].to(device)
            M = torch.zeros_like(term)
            for cc in range(3):
                M[cc] = (term * xcm_weights[:, cc].view(3, 1, 1)).sum(dim=0)

            D_u = safe_pow((T_p - R_p).abs(), mask_p) / (1.0 + M)
            # clamp_diffs: default dclamp_type is "soft" with max_v = 10**d_max.
            # D_clamped = max_v * D_u / (max_v + D_u).
            D = metric.clamp_diffs(D_u)

        band_rec = {
            "k": k,
            "rho": rho_k,
            "band_mul": band_mul,
            "is_baseband": is_baseband,
            "shape_hw": [bh, bw],
            "samples": [],
        }
        for y0, x0 in l0_sentinels:
            yk = min(y0 // (1 << k), bh - 1)
            xk = min(x0 // (1 << k), bw - 1)
            sample = {
                "y0": y0, "x0": x0, "yk": yk, "xk": xk,
                "d_a":  float(D[0, yk, xk]),
                "d_rg": float(D[1, yk, xk]),
                "d_vy": float(D[2, yk, xk]),
            }
            band_rec["samples"].append(sample)
        out["bands"].append(band_rec)
        print(
            f"band {k} rho={rho_k:.3f} band_mul={band_mul} bb={is_baseband} "
            f"sample[0]: d_a={band_rec['samples'][0]['d_a']:.4e} "
            f"d_rg={band_rec['samples'][0]['d_rg']:.4e}"
        )

    path = Path(__file__).parent / "pycvvdp_d_chroma_shift.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote: {path}")


if __name__ == "__main__":
    main()
