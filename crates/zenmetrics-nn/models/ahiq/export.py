# AHIQ export (upstream IIGROUP/AHIQ, CVPRW 2022, MIT).
#
# `python export.py <ladder_dir> <out_dir> <ckpt.pth> <patch_size>`
#
# AHIQ scores one 224x224 crop: ResNet-50 layer-1 Bottleneck taps (3x256ch
# @56x56) are deformed by offsets learned from the reference ViT features
# (vit_base_patch8_224 blocks 0-4, CLS dropped, concat 3840 @28x28), then a
# pixel-score regressor pools f . w / sum(w). The eval protocol crops the
# image nine ways (five_point_crop idx 0-8: corners, edge centres, centre)
# and averages — crops are deterministic, no resizing anywhere.
#
# torchvision's DeformConv2d has no ONNX op: it is decomposed in-graph as
# nine bilinear grid_sample taps + a 1x1 conv (weight[c',c,ky,kx] folded to
# [c', k*C+c]), verified <= 4e-7 against torchvision on random inputs.
#
# Written files (in <out_dir>):
#  ahiq_trunk.onnx   x [-1,1] [1,3,224,224] -> vit [1,3840,28,28],
#                    cnn [1,768,56,56]
#  ahiq_head.onnx    (vit_d, vit_r [1,3840,28,28], cnn_d, cnn_r
#                    [1,768,56,56]) -> score [1,1,1,1]
#  ahiq_full.onnx    (r, d) -> score
#  ahiq.meta         `channels 3840,768`, `min_size 224`
#  ahiq_goldens.tsv  upstream 9-crop ensemble score on the ladder
#  ahiq_export.meta  versions + sha256 of outputs
import hashlib, os, sys, platform
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image

LADDER, OUT, CKPT, PATCH = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)
# upstream modules: loaded standalone (the repo has no package init)
import importlib.util
def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

AHQ_SRC = os.environ['AHIQ_SRC']
dr = load_mod('deform_regressor', os.path.join(AHQ_SRC, 'model/deform_regressor.py'))

class Opt:  # deform_fusion only reads patch_size
    patch_size = PATCH

vit = timm.create_model(
    'vit_base_patch8_224' if PATCH == 8 else 'vit_base_patch16_224',
    pretrained=True).eval()
res = timm.create_model('resnet50', pretrained=True).eval()
deform_net = dr.deform_fusion(Opt()).eval()
regressor = dr.Pixel_Prediction().eval()
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
deform_net.load_state_dict(ck['deform_net_model_state_dict'])
regressor.load_state_dict(ck['regressor_model_state_dict'])
for m in (vit, res, deform_net, regressor):
    m.requires_grad_(False)

TOK = 784 if PATCH == 8 else 196
G = 28 if PATCH == 8 else 14


def vit_taps(x):
    """blocks 0-4 token maps, CLS dropped, concat -> [1, 3840, G, G]."""
    x = vit.patch_embed(x)
    x = vit._pos_embed(x)
    feats = []
    for blk in vit.blocks[:5]:
        x = blk(x)
        feats.append(x[:, 1:, :])
    f = torch.cat(feats, dim=2)          # [1, TOK, 3840]
    return f.transpose(1, 2).reshape(1, 3840, G, G)


def cnn_taps(x):
    """layer1's three Bottleneck outputs concat -> [1, 768, 56, 56]."""
    x = res.conv1(x)
    x = res.bn1(x)
    x = res.act1(x)
    x = res.maxpool(x)
    feats = []
    for blk in res.layer1:
        x = blk(x)
        feats.append(x)
    return torch.cat(feats, dim=1)


def deform_gs(x, offset, weight, bias):
    """DeformConv2d via 9 grid_sample taps + 1x1 conv. Offsets are
    [B,18,H,W] with channel 2k = dy, 2k+1 = dx at kernel pos k = ky*3+kx."""
    B, C, H, W = x.shape
    off = offset.reshape(B, 9, 2, H, W)
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=x.device),
        torch.arange(W, dtype=torch.float32, device=x.device), indexing='ij')
    outs = []
    for k in range(9):
        pos_y = yy + (k // 3 - 1) + off[:, k, 0]
        pos_x = xx + (k % 3 - 1) + off[:, k, 1]
        gx = 2 * (pos_x + 0.5) / W - 1
        gy = 2 * (pos_y + 0.5) / H - 1
        grid = torch.stack([gx, gy], dim=-1)
        outs.append(F.grid_sample(x, grid, mode='bilinear',
                                  padding_mode='zeros', align_corners=False))
    cat = torch.cat(outs, dim=1)                       # [B, 9C, H, W]
    w2 = weight.reshape(C, C, 9).permute(0, 2, 1).reshape(C, 9 * C, 1, 1)
    return F.conv2d(cat, w2, bias)


class Trunk(nn.Module):
    def forward(self, x):
        return vit_taps(x), cnn_taps(x)


class Head(nn.Module):
    def forward(self, vit_d, vit_r, cnn_d, cnn_r):
        # offsets come from the REFERENCE vit only (upstream test loop)
        vit_up = F.interpolate(vit_r, size=cnn_r.shape[-2:], mode='nearest')
        offset = deform_net.conv_offset(vit_up)
        def fuse(cnn):
            d = deform_gs(cnn, offset, deform_net.deform.weight,
                          deform_net.deform.bias)
            return deform_net.conv1(d)
        return regressor(vit_d, vit_r, fuse(cnn_d), fuse(cnn_r)).reshape(1, 1, 1, 1)


class Full(nn.Module):
    def forward(self, r, d):
        vd, cd = vit_taps(d), cnn_taps(d)
        vr, cr = vit_taps(r), cnn_taps(r)
        return Head()(vd, vr, cd, cr).reshape(())


# --- self-checks -----------------------------------------------------------
x = torch.rand(1, 3, 224, 224) * 2 - 1
vt, ct = Trunk()(x)
assert vt.shape == (1, 3840, G, G) and ct.shape == (1, 768, 56, 56)

# deform_gs vs torchvision DeformConv2d (random data, offset from conv_offset)
ref_off = deform_net.conv_offset(F.interpolate(vt, size=(56, 56), mode='nearest'))
ref = deform_net.deform(ct, ref_off)
got = deform_gs(ct, ref_off, deform_net.deform.weight, deform_net.deform.bias)
assert float((ref - got).abs().max()) < 1e-5, float((ref - got).abs().max())

# Head == literal upstream composition (deform_net interpolates vit internally)
a, b = torch.rand(1, 3, 224, 224) * 2 - 1, torch.rand(1, 3, 224, 224) * 2 - 1
vd, cd_, vr, cr = vit_taps(b), cnn_taps(b), vit_taps(a), cnn_taps(a)
ref_score = regressor(vd, vr, deform_net(cd_, vr), deform_net(cr, vr))
got_score = float(Head()(vd, vr, cd_, cr))
assert abs(got_score - float(ref_score)) < 1e-6, (got_score, float(ref_score))
assert abs(float(Full()(a, b)) - got_score) < 1e-6

# --- export ----------------------------------------------------------------
for name, mod, args, inputs, outs in (
    ('ahiq_trunk', Trunk(), (x,), ['x'], ['vit', 'cnn']),
    ('ahiq_head', Head(),
     (torch.rand(1, 3840, G, G), torch.rand(1, 3840, G, G),
      torch.rand(1, 768, 56, 56), torch.rand(1, 768, 56, 56)),
     ['vit_d', 'vit_r', 'cnn_d', 'cnn_r'], ['score']),
    ('ahiq_full', Full(), (x, x.clone()), ['r', 'd'], ['ahiq']),
):
    torch.onnx.export(mod, args, os.path.join(OUT, name + '.onnx'),
                      opset_version=17, dynamo=False, do_constant_folding=True,
                      input_names=inputs, output_names=outs)

# --- goldens: upstream 9-crop ensemble --------------------------------------
def crops(h, w):
    """five_point_crop idx 0..8, top/left pairs (needs h,w >= 224)."""
    ch, cw = h // 2, w // 2
    return [(0, 0), (0, w - 224), (h - 224, 0), (h - 224, w - 224),
            (ch - 112, cw - 112), (ch - 112, 0), (ch - 112, w - 224),
            (0, cw - 112), (h - 224, cw - 112)]


def prep(pil):
    a = np.asarray(pil.convert('RGB'), dtype=np.float32) / 255.0
    a = (a - 0.5) / 0.5
    return torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0)


def score_pair(r_img, d_img):
    a = prep(r_img); b = prep(d_img)
    h, w = a.shape[2], a.shape[3]
    s = 0.0
    for (t, l) in crops(h, w):
        s += float(Full()(a[:, :, t:t + 224, l:l + 224],
                          b[:, :, t:t + 224, l:l + 224]))
    return s / 9.0


rows = []
for f in sorted(os.listdir(LADDER)):
    if not f.endswith('_dist.png'):
        continue
    stem = f[:-9]
    size = stem.rsplit('_', 1)[1]
    if int(size) < 224:
        continue                      # upstream cannot score <224 crops
    r = Image.open(os.path.join(LADDER, stem + '_ref.png'))
    d = Image.open(os.path.join(LADDER, stem + '_dist.png'))
    rows.append((stem.rsplit('_', 1)[0], size, f'{score_pair(r, d):.9e}'))

with open(os.path.join(OUT, 'ahiq_goldens.tsv'), 'w') as f:
    f.write('image\tsize\tahiq\n')
    for r in rows:
        f.write('\t'.join(r) + '\n')

with open(os.path.join(OUT, 'ahiq.meta'), 'w') as f:
    f.write('channels 3840,768\nmin_size 224\n')

meta = {'python': platform.python_version(), 'torch': torch.__version__,
        'timm': timm.__version__, 'patch_size': PATCH,
        'ckpt_epoch': int(ck['epoch'])}
with open(os.path.join(OUT, 'ahiq_export.meta'), 'w') as f:
    f.write(f'# ahiq export {meta}\n')
    for n in sorted(os.listdir(OUT)):
        if n.endswith(('.onnx', '.f32', '.meta', '.tsv')):
            h = hashlib.sha256(open(os.path.join(OUT, n), 'rb').read()).hexdigest()
            f.write(f'{h}  {n}\n')
print('wrote', len(rows), 'goldens; ckpt epoch', ck['epoch'])
