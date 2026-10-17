# DMM netlib-sgesdd goldens — scored by scipy (netlib `gesdd` driver, f32)
# on the *same* ONNX taps the gated engine produces.
#
#   ZENMETRICS_NN_ENGINE=<eng> \
#   cargo run --release -p zenmetrics-nn --features model-tests --example dmm_tap_dump -- \
#       <model_dir> <ladder_dir> <tapdir>/<engine-tag>
#   python gen_netlib_goldens.py <model_dir> <tapdir> <engine-tag>... --owner-approved-floors
#
# Why same-taps: DMM's `wt` term consumes the orthonormal basis the SVD
# solver draws inside degenerate singular clusters — implementation-defined
# per LAPACK build — and the head amplifies tap noise ~100x into score
# units, so goldens on upstream's torch taps sit several gate-widths off.
# The port transliterates netlib `gebd2->bdsdc(lasdq/bdsqr)->orm2r`
# in f32; even so, on deep-null-heavy rows its draws drift past the
# 1e-3 JND gate vs scipy's compiled sgesdd (measured: 2207_128 ort-cuda
# port 31.689 vs scipy-on-same-taps 31.634).
#
# Output <model_dir>/dmm_goldens_netlib_<tag>.tsv columns:
#   image  size  dmm  tol  kind  residual
# `dmm` is the scipy netlib score. `kind` is `jnd` where the port's own
# score (dumped beside the taps as port.tsv) met the 1e-3 JND gate, else
# `pinned` — `tol` is the measured |port - netlib| +5%, i.e. a regression
# floor, not a parity claim. The parity test prints the breakdown.
import argparse
import os
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('models')
parser.add_argument('tapdir')
parser.add_argument('tags', nargs='+', choices=['rten', 'ort_cuda'])
parser.add_argument('--owner-approved-floors', action='store_true',
                    help='Confirm owner approval to generate or regenerate regression floors')
args = parser.parse_args()
if not args.owner_approved_floors:
    parser.error('Regenerating DMM regression floors requires owner approval; see card.toml')
MODELS, TAPDIR, TAGS = args.models, args.tapdir, args.tags

import numpy as np
import scipy.linalg
from numpy.lib.stride_tricks import sliding_window_view

JND = None
for ln in open(os.path.join(os.path.dirname(__file__), '..', 'jnd_scale.tsv')):
    f = ln.split('\t')
    if f[0] == 'dmm':
        JND = float(f[1])
assert JND, 'jnd_scale.tsv has no dmm row'
GATE = 1e-3 * JND


def tap_head(tr, td):
    """DMM.forward's per-tap head (16x16 patches, stride 4) with scipy
    sgesdd-f32 in place of torch.svd — including upstream's transposed
    patch convention (its `u` is the natural patch's right vectors).
    Per-channel loop: materializing every channel's patch stack at once
    is OOM-sized on the 4096-px rows."""
    c, h, w = tr.shape
    wr = sliding_window_view(tr, (16, 16), axis=(1, 2))[:, ::4, ::4]
    wd = sliding_window_view(td, (16, 16), axis=(1, 2))[:, ::4, ::4]
    pt = wr.shape[1] * wr.shape[2]
    acc = np.empty(c)
    for ch in range(c):
        pr = np.ascontiguousarray(wr[ch].transpose(0, 1, 3, 2)).reshape(-1, 16, 16)
        pd = np.ascontiguousarray(wd[ch].transpose(0, 1, 3, 2)).reshape(-1, 16, 16)
        u_r, s_r, _ = scipy.linalg.svd(pr, lapack_driver='gesdd')
        u_d, s_d, _ = scipy.linalg.svd(pd, lapack_driver='gesdd')
        d_s = ((s_r - s_d) ** 2).sum(-1)
        u_rd = np.abs(u_r * u_d).sum(-1)
        # upstream torch .std(-1) is unbiased (ddof=1)
        wt = u_rd.std(-1, ddof=1) / (u_rd.mean(-1) + 1e-9)
        acc[ch] = (d_s * wt).mean()
    diff = acc.sum() / c
    xm = tr.mean((1, 2))
    ym = td.mean((1, 2))
    s1 = (2 * xm * ym + 1e-6) / (xm * xm + ym * ym + 1e-6)
    return float(diff * np.exp(-2.0 * s1.mean()))


def score(tapdir, stem, size):
    total = 0.
    for i in (0, 1):
        tr = np.load(f'{tapdir}/{stem}_{size}_ref_tap{i}.npy')
        td = np.load(f'{tapdir}/{stem}_{size}_dist_tap{i}.npy')
        total += tap_head(tr, td)
    return total


for tag in TAGS:
    tapdir = os.path.join(TAPDIR, tag)
    port = {}
    with open(os.path.join(tapdir, 'port.tsv')) as pf:
        next(pf)
        for ln in pf:
            if ln.strip():
                f = ln.split('\t')
                port[f[0] + '_' + f[1]] = float(f[2])
    out = os.path.join(MODELS, f'dmm_goldens_netlib_{tag}.tsv')
    rows = []
    with open(os.path.join(MODELS, 'dmm_goldens.tsv')) as g:
        next(g)
        for ln in g:
            if ln.strip():
                stem, size = ln.split('\t')[:2]
                rows.append((stem, size, score(tapdir, stem, size)))
    npin = 0
    if os.path.exists(out):
        os.rename(out, f'{out}.{time.time_ns()}.bak')
    with open(out, 'x') as o:
        o.write('# netlib sgesdd-f32 head on this engine\'s taps '
                '(gen_netlib_goldens.py). kind=pinned rows: tol is the\n')
        o.write('# measured |port-netlib| +5% — a regression floor, not a '
                '1e-3 JND parity claim (solver drift in degenerate clusters).\n')
        o.write('image\tsize\tdmm\ttol\tkind\tresidual\n')
        for stem, size, s in rows:
            d = abs(port[f'{stem}_{size}'] - s)
            if d <= GATE:
                o.write(f'{stem}\t{size}\t{s:.17e}\t{GATE:.17e}\tjnd\t{d:.17e}\n')
            else:
                npin += 1
                o.write(f'{stem}\t{size}\t{s:.17e}\t{d * 1.05:.17e}\tpinned\t{d:.17e}\n')
        print(f'{tag}: {len(rows)} rows, {npin} pinned', flush=True)
print('netlib goldens complete')
