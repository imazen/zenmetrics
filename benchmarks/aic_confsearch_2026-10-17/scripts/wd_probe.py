"""WD convention probe. Implements the Wasserstein Distortion statistics as in google/codex
codex/loss/wasserstein.py (and Cool-Chic's port of it), with switchable conventions:
  dist  sq   : (ma-mb)^2 + (sa-sb)^2 per location (codex, Cool-Chic)
        sqrt : sqrt of that (Ballé et al. CVPR'25 / Qiu et al. paper formula)
  floor 5e-7 (Cool-Chic) | 0 (codex lower_limit)
  sigma const (Cool-Chic) | rescale: relu(log2s - mean log2(img/feature size)) (codex)
Features: torchvision VGG16 IMAGENET1K_V1 taps 3,8,15,22,29 (pre-pool ReLU), plus the network
input as a 'pix' layer, at image scales 1, 1/2, 1/4 (2x2 average pooling).
usage: wd_probe.py <pairs.tsv> <out.tsv> [input=raw,imnet]
Writes one row per (pair, input, dist, floor, scale, feature, sigma_mode, log2sigma) = per-feature WD."""
import csv, sys, time
import numpy as np, torch, torch.nn.functional as F, torchvision
from PIL import Image
torch.set_grad_enabled(False)
pairs, out = sys.argv[1], sys.argv[2]
inputs = (sys.argv[3].split('=', 1)[1] if len(sys.argv) > 3 else 'raw,imnet').split(',')
import os
DISTS = os.environ.get('DISTS', 'sq,sqrt').split(',')
FLOORS = [float(x) for x in os.environ.get('FLOORS', '5e-7,0').split(',')]
NLEV = int(os.environ.get('NLEV', '5'))
DOWN = os.environ.get('DOWN', 'avg')
vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features[:30].eval()
TAPS = (3, 8, 15, 22, 29)
K = torch.outer(torch.tensor([.25, .5, .25]), torch.tensor([.25, .5, .25])).view(1, 1, 3, 3)
def lowpass(x, stride): return F.conv2d(x, K, stride=stride, padding=1)
def feats(x):
    res = {'pix': x[0].unsqueeze(1)}
    for i, lay in enumerate(vgg):
        x = lay(x)
        if i in TAPS: res[f'tap{i}'] = x[0].unsqueeze(1)  # (C,1,H,W) as Cool-Chic's rearrange
    return res
def stats(f, n=None):
    n = n or NLEV
    sq = f * f; ms, vs = [], []
    for _ in range(n):
        m = lowpass(f, 1); p = lowpass(sq, 1); ms.append(m); vs.append(p - m * m)
        f = m[..., ::2, ::2]; sq = p[..., ::2, ::2]
    return ms, vs
def level_maps(fa, fb, dist, floor, st):
    """channel-mean distortion map per level (levels 0..5); st = (stats(fa), stats(fb))"""
    (ma, va), (mb, vb) = st
    L = [(fa - fb) ** 2]
    for a, b, x, y in zip(ma, mb, va, vb):
        L.append((a - b) ** 2 + (torch.sqrt(torch.clamp_min(x, floor)) - torch.sqrt(torch.clamp_min(y, floor))) ** 2)
    if dist == 'sqrt': L = [torch.sqrt(l) for l in L]
    return [l.mean(0, keepdim=True) for l in L]  # (1,1,h,w)
def wd_from_maps(maps, ls0):
    ls = torch.full_like(maps[0], float(ls0)); d = torch.zeros(1)
    for i, m in enumerate(maps):
        w = F.relu(1 - (ls - i).abs()); d = d + (w * m).mean()
        if i > 0: ls = lowpass(ls, 2)
    return float(d)
IMN = (torch.tensor((.485, .456, .406)).view(1, 3, 1, 1), torch.tensor((.229, .224, .225)).view(1, 3, 1, 1))
def load(p): return torchvision.transforms.functional.to_tensor(Image.open(p).convert('RGB')).unsqueeze(0)
rows = list(csv.DictReader(open(pairs), delimiter='\t'))
f = open(out, 'w'); f.write('distorted\tinput\tdist\tfloor\tscale\tfeature\tsigma\tlog2s\twd\n')
for r in rows:
    t0 = time.time(); A, B = load(r['dist_path']), load(r['ref_path']); H, W = A.shape[2:]
    for inp in inputs:
        for sc in (1, 2, 4):
            ds = (lambda X: F.avg_pool2d(X, sc)) if DOWN == 'avg' else (lambda X: F.interpolate(X, size=(H // sc, W // sc), mode='bilinear', align_corners=False, antialias=True))
            a = ds(A) if sc > 1 else A; b = ds(B) if sc > 1 else B
            if inp == 'imnet': a, b = (a - IMN[0]) / IMN[1], (b - IMN[0]) / IMN[1]
            fa, fb = feats(a), feats(b)
            for name in fa:
                h, w = fa[name].shape[-2:]
                mlr = (np.log2(H / h) + np.log2(W / w)) / 2
                st = (stats(fa[name]), stats(fb[name]))
                for dist in DISTS:
                    for floor in FLOORS:
                        maps = level_maps(fa[name], fb[name], dist, floor, st)
                        for s in range(0, NLEV + 2):
                            for mode, ls0 in (('const', s), ('rescale', max(s - mlr, 0.0))):
                                f.write(f"{r['distorted']}\t{inp}\t{dist}\t{floor}{os.environ.get('TAG','')}\t{sc}\t{name}\t{mode}\t{s}\t{wd_from_maps(maps, ls0)!r}\n")
            del fa, fb
    f.flush(); print(r['distorted'], f'{time.time()-t0:.1f}s', flush=True)
