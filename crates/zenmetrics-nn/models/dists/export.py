"""Export DISTS (upstream `DISTS_pytorch` 0.1, MIT) for zenmetrics-nn.

usage: export.py LADDER_DIR OUT_DIR [GOLDEN_SIZES]

Writes, into OUT_DIR:
  dists_trunk.onnx   x[1,3,H,W] in [0,1] -> the five VGG16 stage outputs
                     (ImageNet mean/std normalisation and the L2-pooling
                     stages inside the graph). DISTS's sixth "tap" is the raw
                     input itself, which the Rust head takes from the pixels.
  dists_full.onnx    r, d [1,3,H,W] in [0,1] -> scalar: upstream forward.
  dists_alpha.f32,   alpha / (sum alpha + sum beta) and beta / (same), as the
  dists_beta.f32     upstream forward computes them (f32), little-endian, in
                     tap order [input 3, 64, 128, 256, 512, 512].
  dists.meta         `channels 3,64,128,256,512,512` and `min_size n`.
  dists_goldens.tsv  upstream scores (CPU, fp32), no resize (the package's
                     prepare_image default).
  dists_export.meta  versions and sha256 of every output.
"""
import re
import hashlib, os, sys, platform
import numpy as np, torch
from DISTS_pytorch import DISTS
from PIL import Image

LADDER, OUT = sys.argv[1], sys.argv[2]
GOLDEN_SIZES = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else '64,128,256,512,1024').split(',')]
os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)
m = DISTS().eval()

class Trunk(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        return tuple(self.m.forward_once(x)[1:])

class Full(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, r, d):
        return self.m(r, d).reshape(())

trunk = Trunk(m).eval()
x = torch.rand(1, 3, 64, 64)
torch.onnx.export(trunk, (x,), os.path.join(OUT, 'dists_trunk.onnx'), opset_version=17, dynamo=False,
                  do_constant_folding=True, input_names=['x'], output_names=[f'tap{i}' for i in range(5)],
                  dynamic_axes={'x': {2: 'h', 3: 'w'}, **{f'tap{i}': {2: f'h{i}', 3: f'w{i}'} for i in range(5)}})
torch.onnx.export(Full(m).eval(), (x, x.clone()), os.path.join(OUT, 'dists_full.onnx'), opset_version=17,
                  dynamo=False, do_constant_folding=True, input_names=['r', 'd'], output_names=['dists'],
                  dynamic_axes={'r': {2: 'h', 3: 'w'}, 'd': {2: 'h', 3: 'w'}})
w_sum = m.alpha.sum() + m.beta.sum()
(m.alpha / w_sum).reshape(-1).numpy().astype('<f4').tofile(os.path.join(OUT, 'dists_alpha.f32'))
(m.beta / w_sum).reshape(-1).numpy().astype('<f4').tofile(os.path.join(OUT, 'dists_beta.f32'))
assert m.alpha.numel() == sum(m.chns) == 1475

def ok(n):
    try:
        return all(t.shape[2] > 0 and t.shape[3] > 0 for t in trunk(torch.zeros(1, 3, n, n)))
    except RuntimeError:
        return False
min_size = next(n for n in range(1, 257) if ok(n) and all(ok(k) for k in range(n, n + 8)))
with open(os.path.join(OUT, 'dists.meta'), 'w') as f:
    f.write('channels ' + ','.join(map(str, m.chns)) + f'\nmin_size {min_size}\n')

def load(p):
    a = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)

with open(os.path.join(OUT, 'dists_goldens.tsv'), 'w') as g:
    g.write('image\tsize\tdists\n')
    for name in sorted(os.listdir(LADDER)):
        # Square ladder pairs only (`<src>_<size>_ref.png`); other models keep
        # non-square goldens (`<src>_<w>x<h>_*`) in the same directory.
        if not re.fullmatch(r'\d+_\d+_ref\.png', name):
            continue
        stem, size = name.split('_')[0], int(name.split('_')[1])
        if size not in GOLDEN_SIZES:
            continue
        r, d = load(os.path.join(LADDER, name)), load(os.path.join(LADDER, name.replace('_ref', '_dist')))
        score = float(m(r, d).item())
        g.write(f'{stem}\t{size}\t{score:.9g}\n')
        print('dists', stem, size, score, flush=True)

def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()
with open(os.path.join(OUT, 'dists_export.meta'), 'w') as f:
    f.write(f'torch {torch.__version__}\nDISTS_pytorch 0.1\npython {platform.python_version()}\nhost {platform.node()}\n')
    for fn in ('dists_trunk.onnx', 'dists_full.onnx', 'dists_alpha.f32', 'dists_beta.f32', 'dists.meta', 'dists_goldens.tsv'):
        f.write(f'sha256 {sha(os.path.join(OUT, fn))} {fn}\n')
print('done dists min_size', min_size)
