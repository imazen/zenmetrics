"""Export ARNIQA (upstream miccunifi/ARNIQA, Apache-2.0; Agnolon, Galteri,
Bertini, Del Bimbo, "ARNIQA: Learning Distortion Manifold for No-Reference
Image Quality Assessment", WACV 2024) for zenmetrics-nn.

usage: export.py LADDER_DIR OUT_DIR UPSTREAM_DIR WEIGHTS_DIR [GOLDEN_SIZES]

UPSTREAM_DIR is a clone of github.com/miccunifi/ARNIQA (no pip package; the
licence-audited pin is 66d16eb0ff1e1655872d32c0c233614a3922aaad). WEIGHTS_DIR
holds the GitHub-release weights `ARNIQA.pth` (encoder) and
`regressor_<dataset>.pth` (a TorchScript TorchLinearRegression); the dataset
is `kadid10k`, the default of upstream's single_image_inference.py.

Inference convention (single_image_inference.py, CPU fp32 — its
torch.cuda.amp.autocast block is a no-op off CUDA):
  img    = the sRGB8 image, u8
  img_ds = torchvision Resize((h//2, w//2)) == PIL BILINEAR u8 resample
  each   -> center_corners_crop: five 224x224 crops (centre, TL, BL, TR, BR),
            zero-padded where the crop exceeds the image (TF.crop)
  each   -> ToTensor() (u8 -> f32 / 255) -> Normalize(ImageNet) -> [5,3,224,224]
  score  = mean over the 5 crops of regressor(cat(enc(img), enc(img_ds))),
           then _scale_score to [0,1]: kadid10k maps MOS 1..5 as (s-1)/4.

Files written (all prefixed `arniqa`):
  arniqa_trunk.onnx   x[1,3,h,w] in 0..1 -> emb[1,2048,1,1]: ImageNet
                      normalisation inside the graph, ResNet50 (ARNIQA.pth,
                      no fc), L2-normalised `f` — the regressor's per-scale
                      embedding (the SimCLR projector is not used).
  arniqa_full.onnx    x[5,3,224,224], x_ds[5,3,224,224] in 0..1 -> scalar
                      `arniqa`: upstream's forward end to end.
  arniqa_head.f32     regressor weights[4096] then bias[1], LE f32.
  arniqa.meta         `channels 2048` and `min_size n` (probed: the
                      half-scale view needs w//2,h//2 >= 1).
  arniqa_goldens.tsv  `image, size, arniqa, arniqa_ref`: upstream CPU fp32
                      scores of the ladder's dist and ref images.
  arniqa_export.meta  versions and sha256 of every output.
"""
import hashlib, os, sys, platform
import numpy as np, torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

LADDER, OUT, UPSTREAM, WEIGHTS = sys.argv[1:5]
GOLDEN_SIZES = [int(s) for s in (sys.argv[5] if len(sys.argv) > 5 else '64,128,256,512,1024').split(',')]
sys.path.insert(0, UPSTREAM)
import models.resnet as resnet_mod  # noqa: E402  upstream ResNet (torchvision resnet50 + projector)

os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)

MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
CROP = 224

encoder = resnet_mod.ResNet(embedding_dim=128, pretrained=False, use_norm=True)
encoder.load_state_dict(
    torch.load(os.path.join(WEIGHTS, 'ARNIQA.pth'), map_location='cpu', weights_only=True))
encoder.requires_grad_(False).eval()
backbone = encoder.model          # Sequential: resnet50 without its fc
regressor = torch.load(os.path.join(WEIGHTS, 'regressor_kadid10k.pth'),
                       map_location='cpu', weights_only=False).eval()
REG_W = regressor.weights.detach().to(torch.float32)      # [1, 4096]
REG_B = regressor.biases.detach().reshape(1).to(torch.float32)


class Enc(torch.nn.Module):
    """One normalised crop -> the 2048-d L2-normalised backbone embedding `f`
    (the regressor input; ARNIQA's SimCLR projector output is unused)."""

    def __init__(self):
        super().__init__()
        self.register_buffer('mean', MEAN)
        self.register_buffer('std', STD)

    def forward(self, x):
        x = (x - self.mean) / self.std
        f = backbone(x)                    # [n, 2048, 1, 1]
        f = F.normalize(f.view(-1, 2048), dim=1)
        return f.view(-1, 2048, 1, 1)


class Full(torch.nn.Module):
    """The upstream forward verbatim on the two five-crop batches."""

    def __init__(self):
        super().__init__()
        self.register_buffer('mean', MEAN)
        self.register_buffer('std', STD)
        self.register_buffer('w', REG_W, persistent=False)
        self.register_buffer('b', REG_B, persistent=False)

    def forward(self, x, x_ds):
        f = F.normalize(backbone((x - self.mean) / self.std).view(-1, 2048), dim=1)
        f_ds = F.normalize(backbone((x_ds - self.mean) / self.std).view(-1, 2048), dim=1)
        # The scripted TorchLinearRegression's predict is `x @ w.T + b`
        # verbatim (it cannot itself be re-traced inside this trace).
        s = torch.hstack((f, f_ds)).matmul(self.w.t()).add(self.b).squeeze(1)
        s = (s - 1.0) / 4.0                    # _scale_score, kadid10k
        return s.mean(0).reshape(())


enc = Enc().eval()
full = Full().eval()
x = torch.rand(1, 3, CROP, CROP)
x5 = torch.rand(5, 3, CROP, CROP)

for name, mod, args, inputs, outs, dyn in (
    ('arniqa_trunk', enc, (x,), ['x'], ['emb'], {'x': {2: 'h', 3: 'w'}}),
    ('arniqa_full', full, (x5, x5.clone()), ['x', 'x_ds'], ['arniqa'], {}),
):
    torch.onnx.export(mod, args, os.path.join(OUT, name + '.onnx'),
                      opset_version=17, dynamo=False, do_constant_folding=True,
                      input_names=inputs, output_names=outs, dynamic_axes=dyn)

with open(os.path.join(OUT, 'arniqa_head.f32'), 'wb') as f:
    f.write(REG_W.numpy().astype('<f4').tobytes())
    f.write(REG_B.numpy().astype('<f4').tobytes())

# The composed trunk path must equal the literal upstream forward.
with torch.no_grad():
    a = torch.rand(5, 3, CROP, CROP)
    emb = torch.stack([enc(a[i:i + 1]).flatten() for i in range(5)])
    comp = ((regressor(torch.hstack((emb, emb))) - 1.0) / 4.0).mean()
    ref = full(a, a.clone())
    d = float((comp - ref).abs())
    print('composed vs upstream |d| =', d)
    assert d <= 1e-6, d


def crops01(pil):
    """center_corners_crop + ToTensor -> [5,3,224,224] f32 0..1."""
    w, h = pil.size
    cx, cy = w // 2, h // 2
    boxes = [(cy - CROP // 2, cx - CROP // 2), (0, 0), (h - CROP, 0),
             (0, w - CROP), (h - CROP, w - CROP)]
    out = np.zeros((5, 3, CROP, CROP), dtype=np.float32)
    for i, (top, left) in enumerate(boxes):
        c = np.asarray(TF.crop(pil, top, left, CROP, CROP), dtype=np.float32) / 255.0
        out[i] = c.transpose(2, 0, 1)
    return torch.from_numpy(out)


def prep(pil):
    """The host-side transform: full view + PIL-BILINEAR half view, 5 crops."""
    w, h = pil.size
    ds = pil.resize((max(w // 2, 1), max(h // 2, 1)), Image.BILINEAR)
    return crops01(pil), crops01(ds)


def oracle(path):
    x, x_ds = prep(Image.open(path).convert('RGB'))
    return float(full(x, x_ds).item())


def ok(n):
    try:
        oracle_size = crops01(Image.new('RGB', (n, n)))
        enc(oracle_size[0:1])
        return n // 2 >= 1
    except Exception:
        return False


min_size = next(n for n in range(1, 513) if ok(n) and all(ok(k) for k in range(n, n + 8)))
with open(os.path.join(OUT, 'arniqa.meta'), 'w') as f:
    f.write('channels 2048\nmin_size %d\n' % min_size)

with open(os.path.join(OUT, 'arniqa_goldens.tsv'), 'w') as g:
    g.write('image\tsize\tarniqa\tarniqa_ref\n')
    for name in sorted(os.listdir(LADDER)):
        if not name.endswith('_ref.png'):
            continue
        stem, size = name.split('_')[0], int(name.split('_')[1])
        if size not in GOLDEN_SIZES:
            continue
        sr = oracle(os.path.join(LADDER, name))
        sd = oracle(os.path.join(LADDER, name.replace('_ref', '_dist')))
        g.write(f'{stem}\t{size}\t{sd:.9g}\t{sr:.9g}\n')
        print('arniqa', stem, size, sd, sr, flush=True)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


with open(os.path.join(OUT, 'arniqa_export.meta'), 'w') as f:
    f.write(f'torch {torch.__version__}\nmiccunifi/ARNIQA @66d16eb0\npython {platform.python_version()}\nhost {platform.node()}\n')
    for fn in ('arniqa_trunk.onnx', 'arniqa_full.onnx', 'arniqa_head.f32',
               'arniqa.meta', 'arniqa_goldens.tsv'):
        f.write(f'sha256 {sha(os.path.join(OUT, fn))} {fn}\n')
print('done arniqa min_size', min_size)
