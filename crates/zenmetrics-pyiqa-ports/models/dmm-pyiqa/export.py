"""Export pyiqa's `dmm` trunk (VGG16 IMAGENET1K_V1 with L2 pooling, taps relu3_3 / relu4_3) for
zenmetrics-pyiqa-ports (src/dmm.rs).

usage: export.py OUT_DIR

Run with the oracle venv (torch 2.14.1+cpu, pyiqa 0.1.16), TORCH_HOME at its weight cache.
pyiqa's DMM (archs/dmm_arch.py) differs from upstream DMM_PyTorch's packaged preprocessing only
in `prepare_image_adt`: when min(h, w) > 128 the [0,1] tensor is resized with
F.interpolate(size=tar, mode='bilinear', align_corners=False) to a SQUARE tar x tar,
tar = max(int(min/48)*32, 128) (the public zenmetrics port keeps the aspect ratio with a PIL
short-side resize). The head (unfold 16/4 per channel, SVD, wt, glb) is host-side Rust.

Writes dmm_pyiqa_trunk.onnx (x[1,3,H,W] in [0,1] -> t3[1,256,H/4,W/4], t4[1,512,H/8,W/8]),
dmm_pyiqa.meta (`channels 256,512`, `min_size 128`), dmm_pyiqa_export.meta (versions, sha256),
and checks the trunk against pyiqa's FeaturesExtractor on a random input.
"""
import hashlib, os, platform, sys
import torch, torch.nn as nn
import pyiqa, onnxruntime as ort

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)
COL = 'dmm_pyiqa'
net = pyiqa.create_metric('dmm', device='cpu', as_loss=False).net.eval()
fe = net.features_extractor


class Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.fe = fe

    def forward(self, x):
        y = self.fe(x)
        return y['relu3_3'], y['relu4_3']


t = Trunk().eval()
x = torch.rand(1, 3, 160, 192)
t3, t4 = t(x)
y = fe(x)
assert torch.equal(t3, y['relu3_3']) and torch.equal(t4, y['relu4_3'])
p = os.path.join(OUT, f'{COL}_trunk.onnx')
torch.onnx.export(t, (x,), p, input_names=['x'], output_names=['t3', 't4'],
                  dynamic_axes={'x': {2: 'h', 3: 'w'}, 't3': {2: 'h4', 3: 'w4'}, 't4': {2: 'h8', 3: 'w8'}},
                  opset_version=17, dynamo=False)
s = ort.InferenceSession(p, providers=['CPUExecutionProvider'])
o3, o4 = s.run(None, {'x': x.numpy()})
print('onnx vs torch taps:', float(abs(o3 - t3.numpy()).max()), float(abs(o4 - t4.numpy()).max()))
with open(os.path.join(OUT, f'{COL}.meta'), 'w') as f:
    f.write('channels 256,512\nmin_size 128\n')
with open(os.path.join(OUT, f'{COL}_export.meta'), 'w') as f:
    f.write(f'pyiqa {pyiqa.__version__}\ntorch {torch.__version__}\nonnxruntime {ort.__version__}\n'
            f'host {platform.node()}\n')
    for n in sorted(os.listdir(OUT)):
        if n.startswith(COL) and not n.endswith('_export.meta'):
            f.write(f'sha256 {n} {hashlib.sha256(open(os.path.join(OUT, n), "rb").read()).hexdigest()}\n')
print('done')
