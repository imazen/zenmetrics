"""Export one ST-LPIPS network (upstream `stlpips_pytorch` 0.0.2, BSD-2;
Ghildyal & Liu, ECCV 2022) for zenmetrics-nn. Same head as LPIPS; the trunk
uses blur-pooled downsampling and the shift-tolerant weights.

usage: export.py LADDER_DIR OUT_DIR NET [GOLDEN_SIZES]     NET = alex | vgg

Writes, into OUT_DIR (prefix P = stlpips_<net>):
  P_trunk.onnx   x[1,3,H,W] in [-1,1] -> the network's taps (LPIPS ScalingLayer
                 + every slice of the upstream `pretrained_networks` module).
  P_full.onnx    r, d [1,3,H,W] in [-1,1] -> scalar: the whole upstream forward,
                 for engines that keep everything on the device.
  P_lin.f32      the lin layers' 1x1-conv weights, little-endian f32, in tap order.
  P.meta         `channels c0,c1,...` and `min_size n` (smallest square input
                 whose every tap is non-empty), one per line.
  P_goldens.tsv  upstream scores (CPU, fp32) on the ladder, plus per-tap terms.
  P_export.meta  versions and sha256 of every output.

The tap reductions (unit-normalise over channels, squared difference, 1x1 lin,
spatial mean, sum over taps) stay outside the trunk graph so a reference's
normalised taps can be cached and reused across candidates.
"""
import re
import hashlib, os, sys, platform
import types
import numpy as np, torch
# stlpips_pytorch 0.0.2's invariant_networks does `from IPython import embed`
# (a debugging leftover, never called); stub it rather than install IPython.
_ipy = types.ModuleType('IPython'); _ipy.embed = lambda *a, **k: None
sys.modules.setdefault('IPython', _ipy)
from stlpips_pytorch import stlpips, utils as st_utils
from PIL import Image

LADDER, OUT, NET = sys.argv[1], sys.argv[2], sys.argv[3]
GOLDEN_SIZES = [int(s) for s in (sys.argv[4] if len(sys.argv) > 4 else '64,128,256,512,1024').split(',')]
P = f'stlpips_{NET}'
os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)
m = stlpips.LPIPS(net=NET, variant='shift_tolerant', verbose=False).eval()
N = m.L

class Trunk(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.scale = m.scaling_layer
        self.net = m.net
    def forward(self, x):
        return tuple(self.net.forward(self.scale(x)))[:N]

class Full(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, r, d):
        return self.m(r, d).reshape(())

trunk = Trunk(m).eval()
x = torch.rand(1, 3, 64, 64) * 2 - 1
torch.onnx.export(trunk, (x,), os.path.join(OUT, f'{P}_trunk.onnx'), opset_version=17, dynamo=False,
                  do_constant_folding=True, input_names=['x'], output_names=[f'tap{i}' for i in range(N)],
                  dynamic_axes={'x': {2: 'h', 3: 'w'}, **{f'tap{i}': {2: f'h{i}', 3: f'w{i}'} for i in range(N)}})
torch.onnx.export(Full(m).eval(), (x, x.clone()), os.path.join(OUT, f'{P}_full.onnx'), opset_version=17,
                  dynamo=False, do_constant_folding=True, input_names=['r', 'd'], output_names=['lpips'],
                  dynamic_axes={'r': {2: 'h', 3: 'w'}, 'd': {2: 'h', 3: 'w'}})
chns = list(m.chns)
lin = np.concatenate([l.model[-1].weight.detach().numpy().reshape(-1) for l in m.lins]).astype('<f4')
assert lin.size == sum(chns), (lin.size, chns)
lin.tofile(os.path.join(OUT, f'{P}_lin.f32'))

def ok(n):
    try:
        return all(t.shape[2] > 0 and t.shape[3] > 0 for t in trunk(torch.zeros(1, 3, n, n)))
    except RuntimeError:
        return False
min_size = next(n for n in range(1, 257) if ok(n) and all(ok(k) for k in range(n, n + 8)))
with open(os.path.join(OUT, f'{P}.meta'), 'w') as f:
    f.write('channels ' + ','.join(map(str, chns)) + f'\nmin_size {min_size}\n')

def load(p):
    a = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0) * 2 - 1

def contribs(r, d):
    fr, fd = trunk(r), trunk(d)
    out = []
    for k in range(N):
        a, b = st_utils.normalize_tensor(fr[k]), st_utils.normalize_tensor(fd[k])
        out.append(float(m.lins[k]((a - b) ** 2).mean()))
    return out

with open(os.path.join(OUT, f'{P}_goldens.tsv'), 'w') as g:
    g.write('image\tsize\tlpips\t' + '\t'.join(f'tap{i}' for i in range(N)) + '\n')
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
        c = contribs(r, d)
        assert abs(sum(c) - score) <= 1e-5 * max(1.0, abs(score)), (score, sum(c))
        g.write(f'{stem}\t{size}\t{score:.9g}\t' + '\t'.join(f'{v:.9g}' for v in c) + '\n')
        print('stlpips', NET, stem, size, score, flush=True)

def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()
with open(os.path.join(OUT, f'{P}_export.meta'), 'w') as f:
    f.write(f'torch {torch.__version__}\nstlpips_pytorch 0.0.2\npython {platform.python_version()}\nhost {platform.node()}\n')
    for fn in (f'{P}_trunk.onnx', f'{P}_full.onnx', f'{P}_lin.f32', f'{P}.meta', f'{P}_goldens.tsv'):
        f.write(f'sha256 {sha(os.path.join(OUT, fn))} {fn}\n')
print('done', NET, 'min_size', min_size)
