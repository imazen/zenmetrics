"""Export LPIPS-vgg (upstream `lpips` 0.1.4, BSD-2) for zenmetrics-nn.

Writes, into OUT_DIR:
  lpips_vgg_trunk.onnx  input x[1,3,H,W] in [-1,1]; ScalingLayer + VGG16 features
                        through relu5_3; outputs the five LPIPS taps
                        (relu1_2, relu2_2, relu3_3, relu4_3, relu5_3). Dynamic H, W.
  lpips_vgg_full.onnx   inputs r, d [1,3,H,W] in [-1,1] -> scalar: the whole
                        upstream forward in one graph, for engines that keep
                        everything on the device (ort CUDA).
  lpips_vgg_lin.f32     the five NetLinLayer 1x1-conv weight vectors, little-endian
                        f32, concatenated (64+128+256+512+512 = 1472 values).
  goldens.tsv           upstream LPIPS scores (CPU, fp32, TF32 n/a) on the ladder,
                        plus per-tap contributions.
  export.meta           versions and sha256 of every output.

The tap reductions (unit-normalise over channels, squared difference, 1x1 lin,
spatial mean, sum over taps) stay outside the graph so a reference's
normalised taps can be cached and reused across candidates.
"""
import hashlib, os, sys, platform
import numpy as np, torch, lpips
from PIL import Image

LADDER, OUT = sys.argv[1], sys.argv[2]
GOLDEN_SIZES = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else '64,128,256,512,1024').split(',')]
os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)
m = lpips.LPIPS(net='vgg', version='0.1', verbose=False).eval()
TAPS = (3, 8, 15, 22, 29)

class Trunk(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.scale = m.scaling_layer
        self.features = torch.nn.Sequential(*[getattr(m.net, f'slice{i}') for i in range(1, 6)])
    def forward(self, x):
        x = self.scale(x)
        outs = []
        for s in self.features:
            x = s(x)
            outs.append(x)
        return tuple(outs)

trunk = Trunk(m).eval()
x = torch.rand(1, 3, 64, 64) * 2 - 1
onnx_path = os.path.join(OUT, 'lpips_vgg_trunk.onnx')
torch.onnx.export(trunk, (x,), onnx_path, opset_version=17, dynamo=False, do_constant_folding=True,
                  input_names=['x'], output_names=[f'tap{i}' for i in range(5)],
                  dynamic_axes={'x': {2: 'h', 3: 'w'}, **{f'tap{i}': {2: f'h{i}', 3: f'w{i}'} for i in range(5)}})
class Full(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, r, d):
        return self.m(r, d).reshape(())

torch.onnx.export(Full(m).eval(), (x, x.clone()), os.path.join(OUT, 'lpips_vgg_full.onnx'), opset_version=17,
                  dynamo=False, do_constant_folding=True, input_names=['r', 'd'], output_names=['lpips'],
                  dynamic_axes={'r': {2: 'h', 3: 'w'}, 'd': {2: 'h', 3: 'w'}})
lin = np.concatenate([l.model[-1].weight.detach().numpy().reshape(-1) for l in m.lins]).astype('<f4')
assert lin.size == 1472, lin.size
lin.tofile(os.path.join(OUT, 'lpips_vgg_lin.f32'))

def load(p):
    a = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0) * 2 - 1

def contribs(r, d):
    fr, fd = trunk(r), trunk(d)
    out = []
    for k in range(5):
        a, b = lpips.normalize_tensor(fr[k]), lpips.normalize_tensor(fd[k])
        out.append(float(m.lins[k]((a - b) ** 2).mean()))
    return out

with open(os.path.join(OUT, 'goldens.tsv'), 'w') as g:
    g.write('image\tsize\tlpips\ttap0\ttap1\ttap2\ttap3\ttap4\n')
    for name in sorted(os.listdir(LADDER)):
        if not name.endswith('_ref.png'):
            continue
        stem, size = name.split('_')[0], int(name.split('_')[1])
        if size not in GOLDEN_SIZES:
            continue
        r, d = load(os.path.join(LADDER, name)), load(os.path.join(LADDER, name.replace('_ref', '_dist')))
        score = float(m(r, d).item())
        c = contribs(r, d)
        assert abs(sum(c) - score) <= 1e-5 * max(1.0, abs(score)), (score, sum(c))
        g.write(f'{stem}\t{size}\t{score:.9g}\t' + '\t'.join(f'{v:.9g}' for v in c) + '\n')
        print(stem, size, score, flush=True)

def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()
with open(os.path.join(OUT, 'export.meta'), 'w') as f:
    f.write(f'torch {torch.__version__}\nlpips {lpips.__version__ if hasattr(lpips, "__version__") else "0.1.4"}\n')
    f.write(f'python {platform.python_version()}\nhost {platform.node()}\n')
    for fn in ('lpips_vgg_trunk.onnx', 'lpips_vgg_full.onnx', 'lpips_vgg_lin.f32', 'goldens.tsv'):
        f.write(f'sha256 {sha(os.path.join(OUT, fn))} {fn}\n')
print('done')
