"""AIC2026 published-column table shared by assemble.py, july_diff.py and repro_check.py.

Origin: the aic-full-repro lane's assemble.py (dev, 2026-10-20), which produced
benchmarks/aic2026_full_repro_2026-10-20/; moved here unchanged so the committed results can be rebuilt.
COLS: published column -> (our score file / CLI route, score column or 'auto', site-JND map key).
JNDS: JND_<col> -> (our base column, site-JND map key). site_jnd_mappings.json is the AIC site's per-column
mapping jnd(x) = a * |sign * (x - b)| ** c.
"""
import json, math, statistics
from pathlib import Path

MAP = json.loads((Path(__file__).with_name('site_jnd_mappings.json')).read_text())


def sitejnd(mapkey, x):
    p = MAP.get(mapkey)
    if p is None or 'sign' not in p:
        return None
    return p['a'] * (abs(p['sign'] * (x - p['b'])) ** p['c'])


def mdct_jnd(db):
    return 0.000475997802 * (max(0.0, 80.0 - db) ** 2.3182)


# published column -> (score file, score column name or 'auto', site-map key)
COLS = {
 'CVVDP': ('cvvdp', 'auto', 'CVVDP'),
 'SSIMULACRA2': ('ssim2', 'ssim2', 'SSIMULACRA2'),
 'PSNR-Y': ('psnr-y-studio601', 'psnr_y_studio601', 'PSNR-Y'),
 'SSIM': ('ssim-libvmaf', 'ssim_libvmaf', 'SSIM'),
 'MS-SSIM': ('msssim-libvmaf', 'msssim_libvmaf', 'MS-SSIM'),
 'PSNR-HVS': ('psnrhvs-daala', 'psnrhvs_daala', 'PSNR-HVS-vmaf'),
 'VMAF-neg': ('vmaf-neg-studio601', 'vmaf_neg_studio601', 'VMAF-neg'),
 'proposal-Butteraugli': ('butteraugli', 'butteraugli_pnorm3', 'proposal-Butteraugli'),
 'proposal-mDCTPSNR': ('@mdct_jnd', None, 'proposal-mDCTPSNR'),
 'IW-SSIM': ('iwssim-piq', 'auto', 'IW-SSIM'),
 'VIF': ('vifvec', 'auto', 'VIF'),
 'GMSD': ('gmsd', 'auto', 'GMSD'),
 'NLPD': ('nlpd-iqa', 'nlpd_iqa', 'NLPD'),
 'HaarPSI': ('haarpsi', 'auto', 'HaarPSI'),
 'FSIM': ('fsim-y', 'auto', 'FSIM'),
 'FSIMc': ('fsim', 'fsimc', 'FSIMc'),
 'DSSIM': ('dssim', 'dssim', 'DSSIM'),
 'VMAF': ('vmaf-studio601', 'vmaf_studio601', 'VMAF'),
 'PSNR-HVS-Y': ('psnrhvs-daala', 'psnrhvs_daala_y', 'PSNR-HVS-Y-vmaf'),
 'CAMBI': ('cambi', 'cambi', 'CAMBI'),
 'PSNR-Cb': ('psnr-cb', 'psnr_cb', 'PSNR-Cb'),
 'PSNR-Cr': ('psnr-cr', 'psnr_cr', 'PSNR-Cr'),
 'PSNR-HVS-Cb': ('psnrhvs-daala', 'psnrhvs_daala_cb', 'PSNR-HVS-Cb-vmaf'),
 'PSNR-HVS-Cr': ('psnrhvs-daala', 'psnrhvs_daala_cr', 'PSNR-HVS-Cr-vmaf'),
 'ADM2': ('aic_vmaf_features', 'adm2', 'ADM2'),
 'VIF_vmaf': ('aic_vmaf_features', 'vif_vmaf', 'VIF_vmaf'),
 'CIEDE2000': ('ciede2000', 'ciede2000', 'CIEDE2000'),
 'HDR_VDP_2': ('hdrvdp-aic2026', 'hdrvdp_aic2026_q', 'HDR_VDP_2'),
 'HDR_VDP_3': ('hdrvdp3-aic2026', 'hdrvdp3_aic2026_q', 'HDR_VDP_3'),
 'PSNR': ('psnr', 'psnr', 'PSNR'),
 'SSIM-pyiqa': ('ssim-pyiqa', 'ssim_pyiqa', 'SSIM-pyiqa'),
 'MS-SSIM-pyiqa': ('ms-ssim-pyiqa', 'ms_ssim_pyiqa', 'MS-SSIM-pyiqa'),
 'CW-SSIM': ('cw-ssim', 'cw_ssim', 'CW-SSIM'),
 'VSI': ('vsi', 'auto', 'VSI'),
 'SSIMc': ('ssimc', 'ssimc', 'SSIMc'),
 'FLIP': ('flip-v17', 'flip_v17', 'FLIP'),
 'PSNR-YCbCr611': ('psnr-ycbcr611', 'psnr_ycbcr611', 'PSNR-YCbCr611'),
 'mDCT-PSNR': ('mdctpsnr', 'auto', 'mDCT-PSNR'),
}
# JND_<x> column -> (our base column, site-map key)
JNDS = {
 'JND_CVVDP': ('CVVDP', 'CVVDP'),
 'JND_SSIMULACRA2': ('SSIMULACRA2', 'SSIMULACRA2'),
 'JND_PSNR-Y': ('PSNR-Y', 'PSNR-Y'),
 'JND_SSIM': ('SSIM', 'SSIM'),
 'JND_MS-SSIM': ('MS-SSIM', 'MS-SSIM'),
 'JND_PSNR-HVS': ('PSNR-HVS', 'PSNR-HVS-vmaf'),
 'JND_VMAF-neg': ('VMAF-neg', 'VMAF-neg'),
}
META = ['distorted', 'source', 'codec', 'codec_acronym', 'distortion_level', 'bpp',
        'codec_id', 'enc_setting', 'enc_value']


def ranks(x):
    o = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    for k, i in enumerate(o):
        r[i] = k
    return r


def pear(a, b):
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    sa = sum((x - ma) ** 2 for x in a)
    sb = sum((y - mb) ** 2 for y in b)
    if sa == 0 or sb == 0:
        return float('nan')
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(sa * sb)


def write_report(pub, header, ours, report, worst_path):
    """Per-column comparison of our values (ours[col][distorted] -> float) with the published rows."""
    rep = open(report, 'w')
    rep.write('column\tcli\tn\texact_eq\tmedian_d\tmax_d\tmedian_dJND\tmax_dJND\tpearson\tspearman\tmatch\n')
    worst = open(worst_path, 'w')
    jnd_report_cols = set(JNDS)
    for c in header:
        if c in META or c not in COLS:
            continue
        tag, _, mk = COLS[c]
        pairs = []
        dec = 0
        for row in pub:
            pv = row[c]
            if pv in ('', 'nan', 'NaN'):
                continue
            ov = ours[c].get(row['distorted'])
            if ov is None:
                continue
            # published decimal precision
            dec = max(dec, len(pv.partition('.')[2].rstrip('0')) if '.' in pv else 0)
            pairs.append((row['distorted'], ov, float(pv), pv))
        if not pairs:
            rep.write(f'{c}\t{tag}\t0\t0\t\t\t\t\t\t\tnodata\n')
            continue
        n = len(pairs)
        eq = sum(1 for _, o, _, ps in pairs if f'{o:.{dec}f}' == ps or repr(o) == ps)
        dd = [abs(o - p) for _, o, p, _ in pairs]
        dj = []
        for _, o, p, _ in pairs:
            jo, jp = sitejnd(mk, o), sitejnd(mk, p)
            dj.append(abs(jo - jp) if jo is not None else float('nan'))
        djok = [x for x in dj if x == x]
        pj = pear([p[1] for p in pairs], [p[2] for p in pairs])
        sp = pear(ranks([p[1] for p in pairs]), ranks([p[2] for p in pairs]))
        med_j = statistics.median(djok) if djok else float('nan')
        match = 'MATCH' if djok and med_j <= 1e-3 else ('n/a-jnd' if not djok else 'MISS')
        rep.write(f'{c}\t{tag}\t{n}\t{eq}\t{statistics.median(dd):.4g}\t{max(dd):.4g}\t'
                  f'{med_j:.4g}\t{(max(djok) if djok else float("nan")):.4g}\t{pj:.5f}\t{sp:.5f}\t{match}\n')
        if match == 'MISS':
            order = sorted(range(n), key=lambda i: dj[i] if dj[i] == dj[i] else -1, reverse=True)
            worst.write(f'## {c} ({tag}) median_dJND={med_j:.4g}\n')
            for i in order[:5]:
                name, o, p, _ = pairs[i]
                worst.write(f'{name}\tours={o!r}\tpublished={p!r}\t|d|={abs(o-p):.4g}\t|dJND|={dj[i]:.4g}\n')
            worst.write('\n')
    # JND_ columns are recomputed-by-construction (site map); report their raw deltas vs published
    for c in sorted(jnd_report_cols):
        base, mk = JNDS[c]
        pairs = []
        for row in pub:
            ov = ours[base].get(row['distorted'])
            if ov is None or row[c] in ('', 'nan'):
                continue
            pairs.append((abs(sitejnd(mk, ov) - float(row[c])), row['distorted']))
        if pairs:
            dd = [p[0] for p in pairs]
            rep.write(f'{c}\tsitejnd({mk})\t{len(dd)}\t-\t{statistics.median(dd):.4g}\t{max(dd):.4g}\t\t\t\t\tinfo-only\n')
    rep.close()
    worst.close()
