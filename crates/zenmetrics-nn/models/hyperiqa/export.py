"""Export HyperIQA (upstream SSL92/hyperIQA, MIT; S. Su et al., CVPR 2020)
for zenmetrics-nn.

usage: export.py UPSTREAM_DIR CKPT_PKL OUT_DIR [GOLDENS_TSV]

UPSTREAM_DIR is a checkout of https://github.com/SSL92/hyperIQA (pin:
c42e7279, 2023-12-14); `models.py` is imported verbatim. CKPT_PKL is
`pretrained/koniq_pretrained.pkl` from the project's Google Drive link
(no stated licence — the metric is consent-gated). GOLDENS_TSV, when
given, is `oracle.py`'s output and the exported ONNX graph is checked
against it on the same deterministic crop grid.

The export is ONE graph: upstream's HyperNet (ResNet-50 + LDA head +
hyper convs producing the target network's weights) followed by the
TargetNet expressed as batched matmuls — upstream realises it as
grouped 1x1 convs with `groups = batch`, which cannot export with a
dynamic batch; `bmm` is the same math. The exported HyperNet requires fixed batch 25. Input `x [25,3,224,224]` (each row
one normalised crop), output `hyperiqa [1]` = the batch mean, as upstream
averages the per-crop scores.

Files written (all prefixed `hyperiqa`):
  hyperiqa.onnx        x[25,3,224,224] -> score `hyperiqa`[1]; fixed batch
                       25 (the legacy dynamic input label does not make
                       HyperNet's traced reshape operations dynamic).
  hyperiqa.meta        `channels 1` and `min_size 1` (every input resizes
                       to 512x384 first, so any nonzero input is legal).
  hyperiqa_export.meta versions and sha256 of every output.

Preprocessing is NOT in the graph: PIL-bilinear u8 resize to 384x512
(WxH; torchvision `Resize((512,384))`), /255, ImageNet normalisation, and
the deterministic 5x5 crop grid — all replicated in `src/hyperiqa.rs`.
"""
import hashlib
import os
import platform
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

UPSTREAM, CKPT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
GOLDENS = sys.argv[4] if len(sys.argv) > 4 else None
IMG_DIR = sys.argv[5] if len(sys.argv) > 5 else None
os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)
sys.path.insert(0, UPSTREAM)
import models  # noqa: E402

TARGET_IN, F1, F2, F3, F4 = 224, 112, 56, 28, 14


def build():
    models.weights_init_xavier = lambda m: None
    orig = models.resnet50_backbone
    models.resnet50_backbone = lambda *a, **k: orig(a[0], a[1], pretrained=False)
    m = models.HyperNet(16, 112, TARGET_IN, F1, F2, F3, F4, 7)
    models.resnet50_backbone = orig
    missing, unexpected = m.load_state_dict(
        torch.load(CKPT, map_location='cpu', weights_only=True), strict=False)
    assert all('num_batches_tracked' in k for k in missing), missing
    assert not unexpected, unexpected
    return m.eval()


class HyperIQA(nn.Module):
    """HyperNet + TargetNet as batched matmuls + batch-mean (upstream's
    per-crop score average)."""

    def __init__(self, hyper):
        super().__init__()
        self.hyper = hyper

    @staticmethod
    def fc(q, w, b, sigmoid):
        # w [b,fo,fi,1,1] -> [b,fo,fi]; q [b,fi] -> [b,fi,1]
        q = torch.bmm(w.reshape(w.shape[0], w.shape[1], w.shape[2]), q.unsqueeze(-1))
        q = q.squeeze(-1) + b
        return torch.sigmoid(q) if sigmoid else q

    def forward(self, x):
        p = self.hyper(x)
        b_ = x.shape[0]
        q = p['target_in_vec'].reshape(b_, TARGET_IN)
        q = self.fc(q, p['target_fc1w'], p['target_fc1b'], True)
        q = self.fc(q, p['target_fc2w'], p['target_fc2b'], True)
        q = self.fc(q, p['target_fc3w'], p['target_fc3b'], True)
        q = self.fc(q, p['target_fc4w'], p['target_fc4b'], True)
        q = self.fc(q, p['target_fc5w'], p['target_fc5b'], False)
        return q.reshape(1, b_).mean(1, keepdim=True).reshape(1)


T = torchvision.transforms
XS = [round(i * (384 - 224) / 4) for i in range(5)]
YS = [round(j * (512 - 224) / 4) for j in range(5)]


def prep(path):
    img = Image.open(path).convert('RGB')
    img = T.functional.resize(img, (512, 384), T.InterpolationMode.BILINEAR)
    img = T.functional.to_tensor(img)
    img = T.functional.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    return torch.stack([img[:, y:y + 224, x:x + 224] for y in YS for x in XS])


model = HyperIQA(build()).eval()
if IMG_DIR:
    probe = prep(os.path.join(IMG_DIR, sorted(os.listdir(IMG_DIR))[0]))
else:
    torch.manual_seed(0)
    probe = torch.rand(25, 3, 224, 224)
onnx_path = os.path.join(OUT, 'hyperiqa.onnx')
torch.onnx.export(
    model, probe, onnx_path,
    input_names=['x'], output_names=['hyperiqa'],
    dynamic_axes={'x': {0: 'b'}},
    opset_version=17, dynamo=False,
)

with open(os.path.join(OUT, 'hyperiqa.meta'), 'w') as f:
    f.write('channels 1\nmin_size 1\n')

meta = {
    'python': platform.python_version(),
    'torch': torch.__version__,
    'torchvision': torchvision.__version__,
    'numpy': np.__version__,
    'pillow': Image.__version__,
    'upstream': 'SSL92/hyperIQA@c42e7279',
    'ckpt_sha256': hashlib.sha256(open(CKPT, 'rb').read()).hexdigest(),
}
with open(os.path.join(OUT, 'hyperiqa_export.meta'), 'w') as f:
    for k, v in meta.items():
        f.write(f'{k} {v}\n')
    for fn in ('hyperiqa.onnx', 'hyperiqa.meta'):
        h = hashlib.sha256(open(os.path.join(OUT, fn), 'rb').read()).hexdigest()
        f.write(f'sha256 {fn} {h}\n')

# self-check: torch model vs upstream TargetNet path on the probe
with torch.no_grad():
    paras = model.hyper(probe)
    t = models.TargetNet(paras)
    want = float(t(paras['target_in_vec']).float().mean())
    got = float(model(probe))
print(f'self-check upstream-vs-export |d| = {abs(want - got):.3e}')

if GOLDENS:
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    worst = 0.0
    assert IMG_DIR, 'GOLDENS_TSV requires IMG_DIR'
    for line in open(GOLDENS).read().splitlines()[1:]:
        f = line.split('\t')
        stem, size = f[0], f[1]
        for kind, want_s in (('dist', f[2]), ('ref', f[3])):
            path = os.path.join(IMG_DIR, f'{stem}_{size}_{kind}.png')
            if not os.path.exists(path):
                print(f'skip {stem}_{size}_{kind}: not in IMG_DIR')
                continue
            got_s = sess.run(['hyperiqa'], {'x': prep(path).numpy()})[0][0]
            d = abs(got_s - float(want_s))
            worst = max(worst, d)
            print(f'{stem}_{size}_{kind} golden {float(want_s):.6f} onnx {got_s:.6f} |d| {d:.2e}')
    print(f'worst |d| vs goldens: {worst:.3e}')
