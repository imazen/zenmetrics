"""Export CKDN (upstream researchmm/CKDN, MIT; Zheng, Fu, Zeng, Zha, Luo,
"Learning Conditional Knowledge Distillation for Degraded-Reference Image
Quality Assessment", ICCV 2021) for zenmetrics-nn.

usage: export.py LADDER_DIR OUT_DIR UPSTREAM_DIR CKPT [GOLDEN_SIZES]

UPSTREAM_DIR is a clone of github.com/researchmm/CKDN (no pip package);
CKPT is upstream's trained checkpoint (Google Drive link in its README).

CKDN's model takes a 12-channel input: torch.chunk(x, 4) = (restored,
degraded-reference, second-restored, pristine-reference). Its forward
returns [dr, fr, nr, diff]; predict_score.py uses output[:,0] = dr, which
depends only on channel groups 0 and 1 (dr = sigmoid(fc(avgpool(
csp(qse(img0) - dte(img1)))))). As a plain full-reference metric the natural
degenerate feed is cat(dist, ref, dist, ref): then fr == dr, nr is constant
and diff == 0, so the whole metric is dr.

The export therefore splits into:
  ckdn_qse.onnx   x[1,3,H,W] in 0..255 -> qse branch encoding
                  [1,256,H/4,W/4] (ImageNet mean/std in the 0..255 domain,
                  inside the graph: upstream's PrefetchLoader feeds uint8
                  floats and subtracts mean*255 / divides by std*255).
  ckdn_dte.onnx   same input -> dte branch encoding [1,256,H/4,W/4].
  ckdn_tail.onnx  diff[1,256,h,w] -> dr: csp + avgpool + fc_ + sigmoid,
                  output [1,1,1,1].
  ckdn_full.onnx  r, d [1,3,H,W] in 0..255 -> scalar dr: the whole upstream
                  forward on cat(d, r, d, r), output [:,0].
  ckdn.meta       `channels 256` (encoder tap width) and `min_size n`.
  ckdn_goldens.tsv  upstream dr on the ladder (CPU, fp32) under the eval
                  transform each side applies first: resize so the short
                  side is int(288/0.875)=329 (long side floored, keeping
                  aspect) then centre-crop 288 — antialiased bicubic for the
                  distorted image, NEAREST for the reference. That asymmetry
                  is upstream's own convention:
                  timm/data/paired_transforms_tv04.py resizes img with
                  `interpolation` (bicubic per resolve_data_config) and the
                  reference target with `interpolation_tg=Image.NEAREST`,
                  under predict_score.py's img_size=288, crop_pct=0.875.
                  It is also what the AIC-4 published column was scored with
                  (median |d| 4e-5 over 61 pairs).
  ckdn_export.meta  versions and sha256 of every output.
"""
import re
import hashlib, os, sys, platform
import numpy as np, torch
import torchvision.transforms.functional as TF
from PIL import Image

LADDER, OUT, UPSTREAM, CKPT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
GOLDEN_SIZES = [int(s) for s in (sys.argv[5] if len(sys.argv) > 5 else '64,128,256,512,1024').split(',')]
sys.path.insert(0, UPSTREAM)
import ckdn  # noqa: E402  (upstream ckdn.py; timm/ only needed by its scripts)

os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)

# predict_score.py strips the DataParallel `module.` prefix this way.
sd = torch.load(CKPT, map_location='cpu', weights_only=False)['state_dict']
sd = {k.split('module.', 1)[1]: v for k, v in sd.items()}
m = ckdn.ResNet(ckdn.Bottleneck, [3, 4, 6, 3]).eval()
res = m.load_state_dict(sd, strict=False)
assert not res.missing_keys and not res.unexpected_keys, (res.missing_keys, res.unexpected_keys)
m.requires_grad_(False)

# Upstream's PrefetchLoader: uint8 -> float -> (x - mean*255) / (std*255),
# per 3-channel group of the 12-channel input.
MEAN = torch.tensor([x * 255 for x in (0.485, 0.456, 0.406)], dtype=torch.float32).view(1, 3, 1, 1)
STD = torch.tensor([x * 255 for x in (0.229, 0.224, 0.225)], dtype=torch.float32).view(1, 3, 1, 1)


class Enc(torch.nn.Module):
    """qse_1/dte_1 + maxpool + qse_2/dte_2 on a normalised 0..255 input."""

    def __init__(self, conv, layer):
        super().__init__()
        self.conv, self.pool, self.layer = conv, m.maxpool, layer
        self.register_buffer('mean', MEAN)
        self.register_buffer('std', STD)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.layer(self.pool(self.conv(x)))


class Tail(torch.nn.Module):
    """diff -> dr: csp + avgpool + flatten + fc_ + sigmoid."""

    def __init__(self):
        super().__init__()
        self.csp, self.pool, self.fc = m.csp, m.avgpool, m.fc_

    def forward(self, diff):
        x = torch.flatten(self.pool(self.csp(diff)), 1)
        return torch.sigmoid(self.fc(x)).reshape(1, 1, 1, 1)


class Full(torch.nn.Module):
    """The whole upstream forward on cat(d, r, d, r), output [:,0]."""

    def __init__(self):
        super().__init__()
        self.m = m
        self.register_buffer('mean', MEAN.repeat(1, 4, 1, 1))
        self.register_buffer('std', STD.repeat(1, 4, 1, 1))

    def forward(self, r, d):
        x = torch.cat([d, r, d, r], dim=1)
        x = (x - self.mean) / self.std
        return self.m(x)[:, 0].reshape(())


def score(r, d):
    """The composed trunk+tail path, for goldens and self-checks."""
    qse, dte, tl = Enc(m.qse_1, m.qse_2).eval(), Enc(m.dte_1, m.dte_2).eval(), Tail().eval()
    return tl(qse(d) - dte(r)).reshape(())


x = torch.rand(1, 3, 64, 64) * 255.0
qse, dte, tl, full = Enc(m.qse_1, m.qse_2).eval(), Enc(m.dte_1, m.dte_2).eval(), Tail().eval(), Full().eval()
q = qse(x)
diff = torch.rand(1, int(q.shape[1]), 16, 16) * 2 - 1

for name, mod, args, inputs, outs, dyn in (
    ('ckdn_qse', qse, (x,), ['x'], ['enc'], {'x': {2: 'h', 3: 'w'}}),
    ('ckdn_dte', dte, (x,), ['x'], ['enc'], {'x': {2: 'h', 3: 'w'}}),
    ('ckdn_tail', tl, (diff,), ['diff'], ['dr'], {'diff': {2: 'h', 3: 'w'}}),
    ('ckdn_full', full, (x, x.clone()), ['r', 'd'], ['ckdn'], {'r': {2: 'h', 3: 'w'}, 'd': {2: 'h', 3: 'w'}}),
):
    torch.onnx.export(mod, args, os.path.join(OUT, name + '.onnx'),
                      opset_version=17, dynamo=False, do_constant_folding=True,
                      input_names=inputs, output_names=outs, dynamic_axes=dyn)

# The composed path must equal the literal upstream forward.
with torch.no_grad():
    a, b = torch.rand(1, 3, 96, 80) * 255.0, torch.rand(1, 3, 96, 80) * 255.0
    d = float((tl(qse(b) - dte(a)) - full(a, b)).abs())
    print('composed vs upstream |d| =', d)
    assert d <= 1e-6, d


def ok(n):
    try:
        full(torch.rand(1, 3, n, n) * 255.0, torch.rand(1, 3, n, n) * 255.0)
        return True
    except RuntimeError:
        return False


min_size = next(n for n in range(1, 513) if ok(n) and all(ok(k) for k in range(n, n + 8)))
with open(os.path.join(OUT, 'ckdn.meta'), 'w') as f:
    f.write(f'channels {int(q.shape[1])}\nmin_size {min_size}\n')


def load(p):
    a = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32)  # u8 as f32, 0..255
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def prep(t, nearest):
    """The eval transform: resize short side to int(288/0.875)=329 (long side
    floored, torchvision's Resize(int) semantics) then centre-crop 288.
    `nearest` selects the reference side's kernel (the published convention:
    bicubic+antialias for the distorted image, nearest for the reference)."""
    h, w = int(t.shape[2]), int(t.shape[3])
    nh, nw = (329 * h // w, 329) if w <= h else (329, 329 * w // h)
    interp = TF.InterpolationMode.NEAREST if nearest else TF.InterpolationMode.BICUBIC
    return TF.center_crop(TF.resize(t, [nh, nw], interpolation=interp,
                                    antialias=not nearest), 288)


with open(os.path.join(OUT, 'ckdn_goldens.tsv'), 'w') as g:
    g.write('image\tsize\tckdn\n')
    for name in sorted(os.listdir(LADDER)):
        # Square ladder pairs only (`<src>_<size>_ref.png`); other models keep
        # non-square goldens (`<src>_<w>x<h>_*`) in the same directory.
        if not re.fullmatch(r'\d+_\d+_ref\.png', name):
            continue
        stem, size = name.split('_')[0], int(name.split('_')[1])
        if size not in GOLDEN_SIZES:
            continue
        r, d = load(os.path.join(LADDER, name)), load(os.path.join(LADDER, name.replace('_ref', '_dist')))
        s = float(full(prep(r, True), prep(d, False)).item())
        g.write(f'{stem}\t{size}\t{s:.9g}\n')
        print('ckdn', stem, size, s, flush=True)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


with open(os.path.join(OUT, 'ckdn_export.meta'), 'w') as f:
    f.write(f'torch {torch.__version__}\nresearchmm/CKDN (git, upstream ckdn.py)\npython {platform.python_version()}\nhost {platform.node()}\n')
    for fn in ('ckdn_qse.onnx', 'ckdn_dte.onnx', 'ckdn_tail.onnx', 'ckdn_full.onnx',
               'ckdn.meta', 'ckdn_goldens.tsv'):
        f.write(f'sha256 {sha(os.path.join(OUT, fn))} {fn}\n')
print('done ckdn min_size', min_size, 'channels', int(q.shape[1]))
