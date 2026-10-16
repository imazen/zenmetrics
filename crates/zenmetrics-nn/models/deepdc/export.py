"""Export DeepDC (upstream `DeepDC_PyTorch` 0.2, MIT; Zhu, Chen, Wang, Ding,
"Deep Distance Correlation") for zenmetrics-nn.

usage: export.py LADDER_DIR OUT_DIR [GOLDEN_SIZES]

The trunk graph ends at each tap's double-centred distance matrix (the
upstream `_DCDM`, a C x C matrix per VGG19 tap), so what leaves the engine is
five small matrices instead of five feature maps, and a reference's matrices
can be cached. Output shape per tap is [1, C, C, 1] (C x C as a 4-D map).

Writes, into OUT_DIR:
  deepdc_trunk.onnx    x[1,3,H,W] in [0,1] -> 5 DCDM matrices (conv1_2,
                       conv2_2, conv3_4, conv4_4, conv5_4 of VGG19), with
                       upstream's ImageNet input normalisation switched on
                       (see with_input_norm).
  deepdc_full.onnx     r, d [1,3,H,W] in [0,1] -> scalar: upstream forward
                       DeepDC()(r, d).
  deepdc.meta          `channels 64,128,256,512,512` and `min_size n`.
  deepdc_goldens.tsv   upstream scores (CPU, fp32), no resize (upstream's
                       demo `prepare_image` resizes the short side to 224; the
                       model itself takes any size).
  deepdc_export.meta   versions and sha256 of every output.
"""
import re
import hashlib, os, sys, platform, types
import numpy as np, torch
# DeepDC_PyTorch 0.2 starts with `import imp` (never used); `imp` was removed in
# Python 3.12, so register an empty stand-in before importing it.
sys.modules.setdefault('imp', types.ModuleType('imp'))
from DeepDC_PyTorch import DeepDC
from PIL import Image

LADDER, OUT = sys.argv[1], sys.argv[2]
GOLDEN_SIZES = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else '64,128,256,512,1024').split(',')]
os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)

def with_input_norm(model):
    """Turn on upstream's own `use_input_norm` (ImageNet mean/std), the way
    its `MultiVGGFeaturesExtractor.__init__` sets it up. The package default
    is off, but its source comments "Correlation: use_norm = True", pyiqa's
    author-contributed `deepdc` normalises, and the JPEG AIC-4 `DeepDC`
    column matches only with it on (8 cropped pairs, |d| <= 3e-6)."""
    fe = model.features_extractor
    fe.use_input_norm = True
    fe.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
    fe.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    return model

ref_model = with_input_norm(DeepDC()).eval()   # upstream model: computes the goldens
m = with_input_norm(DeepDC()).eval()           # same, patched for export below

def dcdm_dynamic(self, x):
    """Upstream `DeepDC._DCDM`, operation for operation, except that the
    reshape takes `-1` for the pixel count: upstream reads it from
    `x.data.shape` as a Python int, which ONNX tracing freezes to the trace
    image's size."""
    batch, dim = x.shape[0], x.shape[1]
    x = x.reshape(batch, dim, -1)
    t = torch.log((1. / (torch.tensor(dim) * torch.tensor(dim))))
    I = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batch, 1, 1).type(x.dtype)
    I_M = torch.ones(batch, dim, dim, device=x.device).type(x.dtype)
    x_pow2 = x.bmm(x.transpose(1, 2))
    dcov = I_M.bmm(x_pow2 * I) + (x_pow2 * I).bmm(I_M) - 2 * x_pow2
    dcov = torch.clamp(dcov, min=0.0)
    dcov = torch.exp(t) * dcov
    dcov = torch.sqrt(dcov + 1e-5)
    return dcov - 1. / dim * dcov.bmm(I_M) - 1. / dim * I_M.bmm(dcov) + 1. / (dim * dim) * I_M.bmm(dcov).bmm(I_M)

m._DCDM = types.MethodType(dcdm_dynamic, m)

class Trunk(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        feats = self.m.features_extractor(x)
        outs = []
        for f in feats.values():
            d = self.m._DCDM(f)
            outs.append(d.reshape(1, d.shape[1], d.shape[2], 1))
        return tuple(outs)

class Full(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, r, d):
        return self.m(r, d).reshape(())

trunk = Trunk(m).eval()
x = torch.rand(1, 3, 64, 64)
chns = [t.shape[1] for t in trunk(x)]
torch.onnx.export(trunk, (x,), os.path.join(OUT, 'deepdc_trunk.onnx'), opset_version=17, dynamo=False,
                  do_constant_folding=True, input_names=['x'], output_names=[f'tap{i}' for i in range(len(chns))],
                  dynamic_axes={'x': {2: 'h', 3: 'w'}})
torch.onnx.export(Full(m).eval(), (x, x.clone()), os.path.join(OUT, 'deepdc_full.onnx'), opset_version=17,
                  dynamo=False, do_constant_folding=True, input_names=['r', 'd'], output_names=['deepdc'],
                  dynamic_axes={'r': {2: 'h', 3: 'w'}, 'd': {2: 'h', 3: 'w'}})

with torch.no_grad():
    _a, _b = torch.rand(1, 3, 96, 80), torch.rand(1, 3, 96, 80)
    assert float((m(_a, _b) - ref_model(_a, _b)).abs()) == 0.0, 'patched DCDM differs from upstream'

def ok(n):
    try:
        trunk(torch.zeros(1, 3, n, n))
        return True
    except RuntimeError:
        return False
min_size = next(n for n in range(1, 257) if ok(n) and all(ok(k) for k in range(n, n + 8)))
with open(os.path.join(OUT, 'deepdc.meta'), 'w') as f:
    f.write('channels ' + ','.join(map(str, chns)) + f'\nmin_size {min_size}\n')

def load(p):
    a = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)

with open(os.path.join(OUT, 'deepdc_goldens.tsv'), 'w') as g:
    g.write('image\tsize\tdeepdc\n')
    for name in sorted(os.listdir(LADDER)):
        # Square ladder pairs only (`<src>_<size>_ref.png`); other models keep
        # non-square goldens (`<src>_<w>x<h>_*`) in the same directory.
        if not re.fullmatch(r'\d+_\d+_ref\.png', name):
            continue
        stem, size = name.split('_')[0], int(name.split('_')[1])
        if size not in GOLDEN_SIZES:
            continue
        r, d = load(os.path.join(LADDER, name)), load(os.path.join(LADDER, name.replace('_ref', '_dist')))
        score = float(ref_model(r, d).item())
        g.write(f'{stem}\t{size}\t{score:.9g}\n')
        print('deepdc', stem, size, score, flush=True)

def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()
with open(os.path.join(OUT, 'deepdc_export.meta'), 'w') as f:
    f.write(f'torch {torch.__version__}\nDeepDC_PyTorch 0.2 (use_input_norm=True)\npython {platform.python_version()}\nhost {platform.node()}\n')
    for fn in ('deepdc_trunk.onnx', 'deepdc_full.onnx', 'deepdc.meta', 'deepdc_goldens.tsv'):
        f.write(f'sha256 {sha(os.path.join(OUT, fn))} {fn}\n')
print('done deepdc min_size', min_size, 'channels', chns)
