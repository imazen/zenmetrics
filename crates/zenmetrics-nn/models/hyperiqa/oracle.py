"""HyperIQA reference scorer (upstream SSL92/hyperIQA, MIT;
S. Su et al., "Blindly Assess Image Quality in the Wild Guided by a
Self-Adaptive Hyper Network", CVPR 2020).

usage: oracle.py UPSTREAM_DIR CKPT_PKL OUT_TSV IMG...

UPSTREAM_DIR is a checkout of https://github.com/SSL92/hyperIQA (pin:
c42e7279, 2023-12-14); `models.py` is imported verbatim; `demo.py` is not imported.
CKPT_PKL is `pretrained/koniq_pretrained.pkl` (Google Drive, README
"pre-trained model on the koniq-10k dataset"; no stated licence).

OUT_TSV has columns `image<TAB>size<TAB>hyperiqa<TAB>hyperiqa_ref`.
Each row groups the matching STEM_SIZE_dist.png and STEM_SIZE_ref.png inputs;
these are the goldens `export.py` checks and `tests/parity.rs` replays.

DEVIATION FROM UPSTREAM, intentional: upstream's eval transform is
`Resize((512,384))` + `RandomCrop(224)` scored `test_patch_num=25` (demo:
10) times and averaged — i.e. nondeterministic. This oracle uses the same
upstream network on a deterministic 5x5 crop grid instead: x-origins
round(i*160/4) and y-origins round(j*288/4) for i,j in 0..5 over the
512x384 (HxW) frame — 25 crops, the same count as `test_patch_num`. The
model and its weights are upstream's, byte for byte; only the crop
placement is ours.

`HyperNet`'s constructor asks torch hub for the ImageNet ResNet-50 init,
which `load_state_dict` then overwrites anyway — we patch in
`pretrained=False` (and stub `weights_init_xavier`, which upstream never
reaches: its `uniform_(1.0, 0.02)` would raise) so no unrelated weights
are fetched. The checkpoint covers every parameter; only
`num_batches_tracked` buffers are absent (accepted via strict=False).
"""
import sys

import torch
import torchvision
from PIL import Image

UPSTREAM, CKPT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
IMGS = sys.argv[4:]
sys.path.insert(0, UPSTREAM)
import models  # noqa: E402


def build():
    models.weights_init_xavier = lambda m: None
    orig = models.resnet50_backbone
    models.resnet50_backbone = lambda *a, **k: orig(a[0], a[1], pretrained=False)
    m = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7)
    models.resnet50_backbone = orig
    missing, unexpected = m.load_state_dict(
        torch.load(CKPT, map_location='cpu', weights_only=True), strict=False)
    assert all('num_batches_tracked' in k for k in missing), missing
    assert not unexpected, unexpected
    return m.eval()


T = torchvision.transforms
# deterministic 5x5 grid of 224 crops on the resized (512, 384) frame
XS = [round(i * (384 - 224) / 4) for i in range(5)]
YS = [round(j * (512 - 224) / 4) for j in range(5)]


def prep(path):
    img = Image.open(path).convert('RGB')
    img = T.functional.resize(img, (512, 384), T.InterpolationMode.BILINEAR)
    img = T.functional.to_tensor(img)
    img = T.functional.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    return torch.stack([img[:, y:y + 224, x:x + 224] for y in YS for x in XS])


def score(m, x):
    with torch.no_grad():
        paras = m(x)
        t = models.TargetNet(paras)
        for p in t.parameters():
            p.requires_grad = False
        return float(t(paras['target_in_vec']).float().mean())


model = build()
with open(OUT, 'w') as f:
    f.write('image\tsize\thyperiqa\thyperiqa_ref\n')
    for p in IMGS:
        name = p.rsplit('/', 1)[-1]
        if not name.endswith('_ref.png'):
            continue
        stem, size = name.split('_')[0], name.split('_')[1]
        sr = score(model, prep(p))
        sd = score(model, prep(p.replace('_ref.png', '_dist.png')))
        f.write(f'{stem}\t{size}\t{sd:.10f}\t{sr:.10f}\n')
        f.flush()
        print(stem, size, f'{sd:.6f}', f'{sr:.6f}', flush=True)
