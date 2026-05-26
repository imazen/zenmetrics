#!/usr/bin/env python3
"""Finding A bisection: dump pycvvdp per-stage intermediates for the
synth_jpeg_q60 cell on iphone_14_pro (1025 nit) vs standard_phone
(500 nit), to localize where the high-peak-luminance divergence enters.

Run:
    python3 scripts/cvvdp_goldens/diagnose_hipeak.py /tmp/cvvdp-hipeak [situation]

Dumps, per display:
  - the achromatic luminance range after display-encode (DKL Y)
  - per-band: rho, logL_bkg min/mean/max, sensitivity S[A/RG/VY] min/mean/max
  - per-band Q_per_ch contribution
  - final Q_per_ch and JOD
"""
import sys
import numpy as np
import torch
from PIL import Image
import pycvvdp

sit_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/cvvdp-hipeak"
sit = sys.argv[2] if len(sys.argv) > 2 else "synth_jpeg_q60"

ref = np.asarray(Image.open(f"{sit_dir}/images/{sit}.ref.png").convert("RGB"))
dist = np.asarray(Image.open(f"{sit_dir}/images/{sit}.dist.png").convert("RGB"))
print(f"situation={sit}  ref={ref.shape} dist={dist.shape}", file=sys.stderr)


def dump_one(display_name):
    m = pycvvdp.cvvdp(display_name=display_name, heatmap=None)
    print(f"\n===== {display_name}  Y_peak={float(m.display_photometry.get_peak_luminance()):.2f} "
          f"sens_corr={float(m.sensitivity_correction):.4f} csf_sigma={float(m.csf_sigma):.4f} =====")

    # Hook the CSF sensitivity to dump per-band stats.
    orig_sens = m.csf.sensitivity
    band_log = []

    def hooked_sens(rho, omega, logL_bkg, cc, sigma):
        S = orig_sens(rho, omega, logL_bkg, cc, sigma)
        ll = logL_bkg.flatten()
        Sf = S.flatten()
        band_log.append({
            "rho": float(rho), "omega": float(omega), "cc": int(cc),
            "logL_min": float(ll.min()), "logL_mean": float(ll.mean()), "logL_max": float(ll.max()),
            "S_min": float(Sf.min()), "S_mean": float(Sf.mean()), "S_max": float(Sf.max()),
        })
        return S

    m.csf.sensitivity = hooked_sens

    # Hook do_pooling_and_jods to capture the per-channel-per-band Q.
    orig_pool = m.do_pooling_and_jods
    captured = {}

    def hooked_pool(Q_per_ch):
        captured["Q_per_ch"] = Q_per_ch.detach().clone()
        return orig_pool(Q_per_ch)

    m.do_pooling_and_jods = hooked_pool

    jod, _ = m.predict(dist, ref, dim_order="HWC")
    m.csf.sensitivity = orig_sens
    m.do_pooling_and_jods = orig_pool

    Q = captured["Q_per_ch"]  # [B, all_ch, frames, bands]
    Qf = Q.squeeze().cpu().numpy()  # [all_ch, bands]
    print("  per-band Q (A / RG / VY):")
    for bb in range(Qf.shape[-1]):
        print(f"    band {bb}: A={Qf[0,bb]:.6f} RG={Qf[1,bb]:.6f} VY={Qf[2,bb]:.6f}")

    # band_log has all_ch entries per band, in band order. all_ch=3 (Y,RG,VY) for images.
    all_ch = 3
    n_bands = len(band_log) // all_ch
    print(f"JOD = {float(jod):.6f}   n_bands={n_bands}")
    for bb in range(n_bands):
        eY = band_log[bb * all_ch + 0]
        eR = band_log[bb * all_ch + 1]
        eV = band_log[bb * all_ch + 2]
        print(f"  band {bb}: rho={eY['rho']:.4f}  "
              f"logL[{eY['logL_min']:.3f},{eY['logL_mean']:.3f},{eY['logL_max']:.3f}]  "
              f"S_A[{eY['S_min']:.2f},{eY['S_mean']:.2f},{eY['S_max']:.2f}]  "
              f"S_RG[{eR['S_min']:.2f},{eR['S_mean']:.2f},{eR['S_max']:.2f}]  "
              f"S_VY[{eV['S_min']:.2f},{eV['S_mean']:.2f},{eV['S_max']:.2f}]")
    return float(jod)


for d in ("standard_phone", "iphone_14_pro"):
    dump_one(d)
