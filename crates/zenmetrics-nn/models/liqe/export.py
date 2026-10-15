"""Export LIQE (upstream zwx8981/LIQE, MIT code © 2024 Weixia Zhang —
weights via Google Drive/Baidu, terms unstated: Unknown, consent-gated
host-side; Zhang, Ding, Ma, Zhai et al., "LIQE: Blind Image Quality
Assessment via Vision-Language Correspondence", CVPR 2023) for
zenmetrics-nn.

usage: export.py LADDER_DIR OUT_DIR UPSTREAM_DIR CKPT_PATH [GOLDEN_SIZES]

UPSTREAM_DIR is a clone of github.com/zwx8981/LIQE pinned
a64655f7b84b9543b513394472851884cde5b257. CKPT_PATH is LIQE.pt (the
README's pre-trained-weight link, Google Drive
1GoKwUKNR-rvX11QbKRN8MuBZw2hXKHGh). The model is a fine-tuned OpenAI
CLIP ViT-B/32 (both towers): clip.load("ViT-B/32") under
zwx8981's state dict.

Inference convention (LIQE.forward, the module demo2.py drives):
  x = ToTensor()(img) — u8 -> f32 0..1, NO resize
  x = CLIP-normalise inside the module, then unfold 224x224 patches,
      stride 32 (LIQE.forward's self.step; partial borders dropped)
  sel[i] = (batch // 15) * i — for the single-image call batch=1, so
      sel = [0]*15: the top-left 224x224 patch scored 15 times
  quality = sum_k (k+1) * softmax(mean_patches(scale * img_feat @
      text_feat.T)) marginals over the 495 (5 quality x 9 scene x 11
      distortion) joint prompts.

Files written (all prefixed `liqe`):
  liqe_trunk.onnx   x[1,3,224,224] in 0..1 -> emb[1,512,1,1]: CLIP
                    normalisation inside, ViT-B/32 image tower,
                    L2-normalised.
  liqe_full.onnx    x[15,3,224,224] in 0..1 -> quality[1,1,1,1].
  liqe_head.f32     logit_scale.exp()[1] then text_features [495,512]
                    L2-normalised, LE f32.
  liqe.meta         `channels 512` and `min_size 224`.
  liqe_goldens.tsv  `image, size, liqe, liqe_ref`: upstream CPU fp32
                    scores of the ladder's dist and ref images.
  liqe_export.meta  versions and sha256 of every output.
"""
import hashlib, os, sys, platform
import numpy as np, torch
import torch.nn.functional as F
from itertools import product

LADDER, OUT, UPSTREAM, CKPT = sys.argv[1:5]
GOLDEN_SIZES = [int(s) for s in (sys.argv[5] if len(sys.argv) > 5 else '64,128,256,512,1024').split(',')]
sys.path.insert(0, UPSTREAM)
os.chdir(UPSTREAM)
import clip as clip_mod  # noqa: E402  openai-clip
from LIQE import LIQE as LIQE_mod  # noqa: E402

OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

net = LIQE_mod(CKPT, 'cpu').eval()          # LIQE.pt over CLIP ViT-B/32
net.requires_grad_(False)
TEXT = net.text_features.detach()           # [495,512], L2-normalised
LOGIT_SCALE = net.model.logit_scale.detach().exp().reshape(1)
assert TEXT.shape == (495, 512)


class Enc(torch.nn.Module):
    """One 224x224 crop in 0..1 -> CLIP-normalise -> image tower -> L2
    -normalised 512-d embedding."""

    def __init__(self):
        super().__init__()
        self.register_buffer('mean', CLIP_MEAN)
        self.register_buffer('std', CLIP_STD)

    def forward(self, x):
        f = net.model.encode_image((x - self.mean) / self.std)
        return F.normalize(f, dim=1).view(-1, 512, 1, 1)


class Full(torch.nn.Module):
    """The upstream forward on the 15 selected patches -> scalar."""

    def __init__(self):
        super().__init__()
        self.register_buffer('mean', CLIP_MEAN)
        self.register_buffer('std', CLIP_STD)
        self.register_buffer('text', TEXT, persistent=False)
        self.register_buffer('scale', LOGIT_SCALE, persistent=False)

    def forward(self, x):                    # x [15,3,224,224], 0..1
        f = net.model.encode_image((x - self.mean) / self.std)
        f = F.normalize(f, dim=1)
        logits = self.scale * f @ self.text.t()     # [15,495]
        p = F.softmax(logits.mean(0, keepdim=True), dim=1)   # [1,495]
        q = p.view(1, 5, 9, 11).sum(3).sum(2)       # quality marginals
        w = torch.arange(1.0, 6.0)
        return (q * w).sum(1).reshape(-1, 1, 1, 1)


enc = Enc().eval()
full = Full().eval()
x1 = torch.rand(1, 3, 224, 224)
x15 = torch.rand(15, 3, 224, 224)

for name, mod, args, inputs, outs in (
    ('liqe_trunk', enc, (x1,), ['x'], ['emb']),
    ('liqe_full', full, (x15,), ['x'], ['liqe']),
):
    torch.onnx.export(mod, args, os.path.join(OUT, name + '.onnx'),
                      opset_version=17, dynamo=False, do_constant_folding=True,
                      input_names=inputs, output_names=outs)

with open(os.path.join(OUT, 'liqe_head.f32'), 'wb') as f:
    f.write(LOGIT_SCALE.numpy().astype('<f4').tobytes())
    f.write(TEXT.numpy().astype('<f4').tobytes())


def upstream_forward(path):
    """The literal module path: unfold 224/32, sel = [0]*15 for B=1."""
    from PIL import Image
    from torchvision.transforms import ToTensor
    x = ToTensor()(Image.open(path).convert('RGB')).unsqueeze(0)
    q, _scene, _dist = net(x)
    return float(q.item())


def oracle(path):
    """Our split: host-side unfold + 15x patch-0 selection, trunk enc."""
    from PIL import Image
    import numpy as np
    a = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    h, w = a.shape[:2]
    patch = a[:224, :224].transpose(2, 0, 1)          # top-left 224 crop
    x = torch.from_numpy(patch).unsqueeze(0).expand(15, -1, -1, -1)
    return float(full(x.contiguous()).item())


# Self-check: the export wrapper equals the literal LIQE.forward.
with torch.no_grad():
    ref = upstream_forward(os.path.join(LADDER, '1415_256_dist.png'))
    got = oracle(os.path.join(LADDER, '1415_256_dist.png'))
    print(f'wrapped vs upstream: {got} vs {ref} |d| = {abs(got - ref)}')
    assert abs(got - ref) <= 1e-6

with open(os.path.join(OUT, 'liqe.meta'), 'w') as f:
    f.write('channels 512\nmin_size 224\n')

with open(os.path.join(OUT, 'liqe_goldens.tsv'), 'w') as g:
    g.write('image\tsize\tliqe\tliqe_ref\n')
    for name in sorted(os.listdir(LADDER)):
        if not name.endswith('_ref.png'):
            continue
        stem, size = name.split('_')[0], int(name.split('_')[1])
        if size not in GOLDEN_SIZES or size < 224:
            continue  # upstream unfold yields 0 patches under 224
        sr = upstream_forward(os.path.join(LADDER, name))
        sd = upstream_forward(os.path.join(LADDER, name.replace('_ref', '_dist')))
        g.write(f'{stem}\t{size}\t{sd:.9g}\t{sr:.9g}\n')
        print('liqe', stem, size, sd, sr, flush=True)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


with open(os.path.join(OUT, 'liqe_export.meta'), 'w') as f:
    f.write(f'torch {torch.__version__}\nzwx8981/LIQE @a64655f7\npython {platform.python_version()}\nhost {platform.node()}\n')
    for fn in ('liqe_trunk.onnx', 'liqe_full.onnx', 'liqe_head.f32',
               'liqe.meta', 'liqe_goldens.tsv'):
        f.write(f'sha256 {sha(os.path.join(OUT, fn))} {fn}\n')
print('done liqe')
