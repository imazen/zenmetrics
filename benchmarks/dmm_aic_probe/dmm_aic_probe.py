#!/usr/bin/env python3
# DMM AIC-4 column probe — reproduces the published `DMM` column of the AIC-4
# 2026 metrics_subset.csv bit-exactly. Findings (2026-10-16):
#   * the column was computed with pyiqa's dmm_arch.py convention:
#     prepare_image_adt -> square anamorphic F.interpolate(size=tar_size,
#     mode='bilinear', align_corners=False) — NOT upstream DMM_PyTorch's
#     aspect-preserving F.resize on a PIL image;
#   * on CUDA (torch.svd = cusolver draws). On CPU the identical pipeline is
#     ~9e-3 median off — DMM's wt term is solver-draw sensitive.
# Run: zenmetrics batch --metric dmm produces `aic_dmm.tsv` (native convention);
# this script produces `aic_dmm_pyiqa{,_cuda}.tsv`.
import csv, sys, types, importlib.util
import torch
import torchvision.transforms.functional as TF
from PIL import Image

PYIQA = '/home/lilith/tmp/pyiqa-research/IQA-PyTorch'
PAIRS = '/home/lilith/tmp/nn-phase0/aic-crop/pairs.tsv'
PUB = '/home/lilith/tmp/aic-sub-crop/metrics_subset.csv'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

# import pyiqa's dmm_arch without its package __init__ (avoids yaml/tqdm deps)
pkg = types.ModuleType('pyiqa')
utils = types.ModuleType('pyiqa.utils')
reg = types.ModuleType('pyiqa.utils.registry')
class Registry(dict):
    def register(self, name=None):
        def deco(cls): return cls
        return deco
reg.ARCH_REGISTRY = Registry()
sys.modules['pyiqa'] = pkg
sys.modules['pyiqa.utils'] = utils
sys.modules['pyiqa.utils.registry'] = reg
spec = importlib.util.spec_from_file_location('dmm_arch', f'{PYIQA}/pyiqa/archs/dmm_arch.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

net = mod.DMM().to(DEV)
pub = {}
with open(PUB) as f:
    for r in csv.DictReader(f):
        pub[r['distorted']] = float(r['DMM'])
rows = []
with open(PAIRS) as f:
    for r in csv.DictReader(f, delimiter='\t'):
        im_r = TF.to_tensor(Image.open(r['ref_path']).convert('RGB'))[None].to(DEV)
        im_d = TF.to_tensor(Image.open(r['dist_path']).convert('RGB'))[None].to(DEV)
        s = net(im_d, im_r).item()
        rows.append((r['distorted'], s, pub.get(r['distorted'])))
out = f'/home/lilith/tmp/nn-port-scratch/aic_dmm_pyiqa_{DEV}.tsv'
with open(out, 'w') as f:
    f.write('distorted\tdmm_pyiqa\tdmm_published\tabs_diff\n')
    for n, s, p in rows:
        f.write(f'{n}\t{s:.10f}\t{p:.10f}\t{abs(s - p):.3e}\n')
ds = sorted(abs(s - p) for _, s, p in rows)
print(f'{DEV}: n={len(rows)} median|d|={ds[len(ds)//2]:.3e} max|d|={ds[-1]:.3e} -> {out}')
