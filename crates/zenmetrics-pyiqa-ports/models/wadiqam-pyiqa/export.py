"""Export pyiqa's `wadiqam_fr` (WaDIQaM-FR, KADID checkpoint WaDIQaM-kadid-f7541ea5.pth) for
zenmetrics-pyiqa-ports (src/wadiqam.rs).

usage: export.py LADDER_PAIRS_TSV OUT_DIR

Run with the oracle venv (torch 2.14.1+cpu, pyiqa 0.1.16), TORCH_HOME at its weight cache.
pyiqa's eval protocol (archs/wadiqam_arch.py): input ToTensor in [0, 1]; 32x32 patches at
stride 32 from `range(0, h - 32, 32)` x `range(0, w - 32, 32)` (so a dimension that is a
multiple of 32 drops its last full patch, and dimensions <= 32 have none); shared VGG-ish trunk
-> 512 per patch; `feat = cat(f_ref, f_dist, f_dist - f_ref)`; q = fc_q(feat),
w = fc_w(feat) + 1e-8; score = sum(q * w) / sum(w).

Writes into OUT_DIR:
  wadiqam_pyiqa_trunk.onnx  x[P,3,32,32] in [0,1] -> h[P,512]
  wadiqam_pyiqa_head.onnx   f[P,1536] -> q[P,1], w[P,1] (w includes the 1e-8)
  wadiqam_pyiqa.meta        channels 512 / min_size 33
  wadiqam_pyiqa_goldens_ladder.tsv  distorted, ref_w, ref_h, wadiqam_pyiqa (pyiqa itself)
  wadiqam_pyiqa_export.meta versions, host, sha256 of every output
and replays every ladder pair through onnxruntime CPU, printing the worst |onnx - pyiqa|.
"""
import csv, hashlib, os, platform, sys
import numpy as np, torch, torch.nn as nn
import pyiqa, onnxruntime as ort
from PIL import Image

PAIRS, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)
COL = 'wadiqam_pyiqa'
metric = pyiqa.create_metric('wadiqam_fr', device='cpu', as_loss=False)
net = metric.net.eval()


class Trunk(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.features = n.features

    def forward(self, x):
        return self.features(x).reshape(-1, 512)


class Head(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.fc_q, self.fc_w, self.eps = n.fc_q, n.fc_w, n.eps

    def forward(self, f):
        return self.fc_q(f), self.fc_w(f) + self.eps


trunk, head = Trunk(net).eval(), Head(net).eval()
p_trunk = os.path.join(OUT, f'{COL}_trunk.onnx')
p_head = os.path.join(OUT, f'{COL}_head.onnx')
torch.onnx.export(trunk, (torch.rand(4, 3, 32, 32),), p_trunk, input_names=['x'], output_names=['h'],
                  dynamic_axes={'x': {0: 'p'}, 'h': {0: 'p'}}, opset_version=17, dynamo=False)
torch.onnx.export(head, (torch.rand(4, 1536),), p_head, input_names=['f'], output_names=['q', 'w'],
                  dynamic_axes={'f': {0: 'p'}, 'q': {0: 'p'}, 'w': {0: 'p'}}, opset_version=17, dynamo=False)
with open(os.path.join(OUT, f'{COL}.meta'), 'w') as f:
    f.write('channels 512\nmin_size 33\n')

pairs = list(csv.DictReader(open(PAIRS), delimiter='\t'))
st = ort.InferenceSession(p_trunk, providers=['CPUExecutionProvider'])
sh = ort.InferenceSession(p_head, providers=['CPUExecutionProvider'])


def patches(path):
    a = np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
    x = (a.astype(np.float32) / np.float32(255)).transpose(2, 0, 1)
    h, w = a.shape[:2]
    ps = [x[:, i:i + 32, j:j + 32] for i in range(0, h - 32, 32) for j in range(0, w - 32, 32)]
    return np.stack(ps), w, h


worst = 0.0
rows = []
for r in pairs:
    s = float(metric(r['dist_path'], r['ref_path']).flatten()[0])
    pr, w, h = patches(r['ref_path'])
    pd, _, _ = patches(r['dist_path'])
    hr = st.run(None, {'x': pr})[0]
    hd = st.run(None, {'x': pd})[0]
    q, wt = sh.run(None, {'f': np.concatenate([hr, hd, hd - hr], 1)})
    o = float((q.astype(np.float64) * wt).sum() / wt.astype(np.float64).sum())
    worst = max(worst, abs(o - s))
    rows.append((r['distorted'], w, h, s))
    print(r['distorted'], s, o, flush=True)
with open(os.path.join(OUT, f'{COL}_goldens_ladder.tsv'), 'w') as f:
    f.write(f'distorted\tref_w\tref_h\t{COL}\n')
    for d, w, h, s in rows:
        f.write(f'{d}\t{w}\t{h}\t{s!r}\n')
print('worst |onnx - pyiqa| over the ladder:', worst)
with open(os.path.join(OUT, f'{COL}_export.meta'), 'w') as f:
    f.write(f'pyiqa {pyiqa.__version__}\ntorch {torch.__version__}\nonnxruntime {ort.__version__}\n'
            f'host {platform.node()}\nworst_onnx_vs_pyiqa_ladder {worst!r}\n')
    for n in sorted(os.listdir(OUT)):
        if n.startswith(COL) and not n.endswith('_export.meta'):
            f.write(f'sha256 {n} {hashlib.sha256(open(os.path.join(OUT, n), "rb").read()).hexdigest()}\n')
