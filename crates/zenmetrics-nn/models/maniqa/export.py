"""Export MANIQA (upstream IIGROUP/MANIQA, Apache-2.0; S. Yang, T. Wu, S. Shi,
S. Lao, Y. Gong, M. Cao, J. Wang, Y. Yang, Tsinghua University IIGROUP +
vivo, "MANIQA: Multi-dimension Attention Network for No-Reference Image
Quality Assessment", CVPRW 2022) for zenmetrics-nn.

usage: export.py LADDER_DIR OUT_DIR UPSTREAM_DIR WEIGHTS_DIR [GOLDEN_SIZES]

UPSTREAM_DIR is a clone of github.com/IIGROUP/MANIQA, pinned
f573d862401243c45641951a0303b8059c1c8577. WEIGHTS_DIR holds the GitHub-release
checkpoint `ckpt_koniq10k.pt` (Koniq10k tag) — the dataset
predict_one_image.py defaults to and the in-the-wild corpus.

Inference convention — the deterministic upstream eval path
(train_maniqa.py validation for dataset_name=koniq10k, num_avg_val=1,
crop_size=224; the README's predict_one_image.py is the same model but
samples np.random crops, so it is not reproducible host-side):
  img = cv2.imread BGR u8 -> cv2.resize((224,224), INTER_CUBIC) -> BGR2RGB
  x   = u8/255 CHW, Normalized (x-0.5)/0.5 -> [1,3,224,224]
  score = net(x)[0]   (five_point_crop(0) on a 224 image is the image itself)

  The cv2.resize call below runs with IPP disabled (set before the cv2
  import): the wheels' IPP path intercepts u8 resize and differs from
  OpenCV's own scalar path on ~4% of pixels, while the scalar path is
  bitwise-identical between cv2 4.12 and 5.0 — that is the convention the
  Rust port reproduces (see resize_bicubic_cv2_u8).

The model: vit_base_patch8_224 trunk (fine-tuned weights inside the
checkpoint), blocks 6..9 token features concat [784,3072] -> 2x TABlock(784)
-> conv 3072->768 -> SwinTransformer stage -> 2x TABlock -> conv 768->384 ->
SwinTransformer stage -> per-token fc_score*fc_weight weighted mean.

Files written (all prefixed `maniqa`):
  maniqa_full.onnx    x[1,3,224,224] normalized (x-0.5)/0.5 -> score[1,1,1].
  maniqa.meta         `channels 1` and `min_size` (any size resizes to 224).
  maniqa_goldens.tsv  `image, size, maniqa, maniqa_ref`: upstream CPU fp32
                      scores of the ladder's dist and ref images.
  maniqa_export.meta  versions and sha256 of every output.
"""
import hashlib, os, sys, platform

os.environ['OPENCV_IPP'] = 'disabled'  # pin OpenCV's own scalar u8 path

import numpy as np, torch
import cv2
import PIL

LADDER, OUT, UPSTREAM, WEIGHTS = sys.argv[1:5]
GOLDEN_SIZES = [int(s) for s in (sys.argv[5] if len(sys.argv) > 5 else '64,128,256,512,1024').split(',')]
sys.path.insert(0, UPSTREAM)
sys.path.insert(0, os.path.join(UPSTREAM))  # vendored timm/models in-repo
os.chdir(UPSTREAM)  # upstream imports are repo-relative (models.*, timm.*)
import models.maniqa as maniqa_mod  # noqa: E402

OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)

net = maniqa_mod.MANIQA(
    embed_dim=768, num_outputs=1, dim_mlp=768, patch_size=8, img_size=224,
    window_size=4, depths=[2, 2], num_heads=[4, 4], num_tab=2, scale=0.8)
_missing, _unexpected = net.load_state_dict(
    torch.load(os.path.join(WEIGHTS, 'ckpt_koniq10k.pt'),
               map_location='cpu', weights_only=True),
    strict=False)
# Upstream loads with strict=False (the checkpoint carries a few extra
# keys) — every *model* parameter must still be filled; never accept a
# silently-random layer.
assert not _missing, f'checkpoint missing model keys: {_missing}'
print(f'checkpoint: {len(_unexpected)} extra keys ignored, 0 missing')
net.requires_grad_(False).eval()


class Full(torch.nn.Module):
    """The upstream forward without the CUDA/loop scaffolding: hooks capture
    ViT blocks 6..9 during the trace; the per-image weighted mean is
    vectorized. Output [1,1,1,1] so rten's trunk Tap carries the scalar."""

    def __init__(self):
        super().__init__()
        self.n = net

    def forward(self, x):
        n = self.n
        _ = n.vit(x)
        f = n.extract_feature(n.save_output)          # [b,784,3072]
        n.save_output.outputs.clear()
        x = f.transpose(1, 2)                          # b c (h w)
        for tab in n.tablock1:
            x = tab(x)
        x = x.view(-1, 3072, 28, 28)
        x = n.swintransformer1(n.conv1(x))             # [b,768,28,28]
        x = x.flatten(2)                               # b c (h w)
        for tab in n.tablock2:
            x = tab(x)
        x = x.view(-1, 768, 28, 28)
        x = n.swintransformer2(n.conv2(x))             # [b,384,28,28]
        x = x.flatten(2).transpose(1, 2)               # [b,784,384]
        fs = n.fc_score(x)
        w = n.fc_weight(x)
        s = (fs * w).flatten(1).sum(1) / w.flatten(1).sum(1)
        return s.reshape(-1, 1, 1, 1)  # [b,1,1,1]: one-tap trunk output


def upstream_forward(net, x):
    """Reference: the literal upstream forward, CUDA calls removed (CPU)."""
    _ = net.vit(x)
    x = net.extract_feature(net.save_output)
    net.save_output.outputs.clear()
    x = x.transpose(1, 2)
    for tab in net.tablock1:
        x = tab(x)
    x = x.view(-1, 3072, 28, 28)
    x = net.swintransformer1(net.conv1(x))
    x = x.flatten(2)
    for tab in net.tablock2:
        x = tab(x)
    x = x.view(-1, 768, 28, 28)
    x = net.swintransformer2(net.conv2(x))
    x = x.flatten(2).transpose(1, 2)
    score = torch.tensor([])
    for i in range(x.shape[0]):
        f = net.fc_score(x[i])
        w = net.fc_weight(x[i])
        _s = torch.sum(f * w) / torch.sum(w)
        score = torch.cat((score, _s.unsqueeze(0)), 0)
    return score


full = Full().eval()
x = torch.rand(1, 3, 224, 224)

torch.onnx.export(full, (x,), os.path.join(OUT, 'maniqa_full.onnx'),
                  opset_version=17, dynamo=False, do_constant_folding=True,
                  input_names=['x'], output_names=['maniqa'])

# Self-check: the vectorized wrapper must equal the literal loop.
with torch.no_grad():
    a = torch.rand(1, 3, 224, 224)
    d = float((full(a) - upstream_forward(net, a)).abs().max())
    print('wrapped vs upstream |d| =', d)
    assert d <= 1e-6, d


def prep(path):
    """cv2.imread -> resize 224 INTER_CUBIC (u8) -> RGB, /255, (x-0.5)/0.5."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(path)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


def oracle(path):
    return float(full(prep(path)).item())


with open(os.path.join(OUT, 'maniqa.meta'), 'w') as f:
    f.write('channels 1\nmin_size 1\n')  # any input resizes to 224

with open(os.path.join(OUT, 'maniqa_goldens.tsv'), 'w') as g:
    g.write('image\tsize\tmaniqa\tmaniqa_ref\n')
    for name in sorted(os.listdir(LADDER)):
        if not name.endswith('_ref.png'):
            continue
        stem, size = name.split('_')[0], name.split('_')[1]
        if not size.isdigit() or int(size) not in GOLDEN_SIZES:
            continue
        size = int(size)
        sr = oracle(os.path.join(LADDER, name))
        sd = oracle(os.path.join(LADDER, name.replace('_ref', '_dist')))
        g.write(f'{stem}\t{size}\t{sd:.9g}\t{sr:.9g}\n')
        print('maniqa', stem, size, sd, sr, flush=True)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


with open(os.path.join(OUT, 'maniqa_export.meta'), 'w') as f:
    f.write(f'torch {torch.__version__}\nIIGROUP/MANIQA @f573d862\npython {platform.python_version()}\nhost {platform.node()}\n'
            f'cv2 {cv2.__version__} (OPENCV_IPP=disabled)\nnumpy {np.__version__}\nPillow {PIL.__version__}\n')
    for fn in ('maniqa_full.onnx', 'maniqa.meta', 'maniqa_goldens.tsv'):
        f.write(f'sha256 {sha(os.path.join(OUT, fn))} {fn}\n')
print('done maniqa')
