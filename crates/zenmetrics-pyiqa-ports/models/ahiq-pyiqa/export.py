"""Export pyiqa's `ahiq` (AHIQ, checkpoint AHIQ_vit_p8_epoch33-da3ea303.pth) for zenmetrics-pyiqa-ports (src/ahiq.rs).

usage: export.py OUT_DIR [LADDER_PAIRS_TSV]

Run with the oracle venv (torch 2.14.1+cpu, pyiqa 0.1.16, timm 1.0.30), TORCH_HOME at its weight
cache. The backbones are what pyiqa's `timm.create_model(name, pretrained=True)` resolves under
timm 1.0.30: `vit_base_patch8_224` -> augreg2_in21k_ft_in1k, `resnet50` -> a1_in1k (the public
zenmetrics port pins upstream's timm 0.5.4 resolution, augreg_in21k_ft_in1k, instead).

pyiqa's eval protocol (archs/ahiq_arch.py, num_crop=20, crop_size=224):
  - ToTensor input in [0,1]; if min(h, w) <= 224 the image is first upscaled with
    F.interpolate(scale_factor=225/min(h, w), mode='bilinear') (align_corners False, no antialias);
  - uniform_crop: step = (d - 224) // 4 per axis, crops at (i*step_h, j*step_w) for i in 0..4,
    j in 0..5 (the first 20 of a 5x5 grid), batch 20;
  - ImageNet mean/std normalisation, ViT-B/8 blocks 0-4 (CLS dropped, 3840 @ 28x28) and ResNet-50
    layer1's three Bottleneck outputs (768 @ 56x56);
  - DeformFusion with offsets from the REFERENCE ViT map for both images, Pixel_Prediction
    pooling f*w / sum(w); score = mean over the 20 crops.

Graphs (batch 1, one crop per call):
  ahiq_pyiqa_trunk.onnx  x[1,3,224,224] in [0,1] -> vit[1,3840,28,28], cnn[1,768,56,56]
  ahiq_pyiqa_ref.onnx    vit_r, cnn_r -> offset[1,18,56,56], f_ref[1,256,28,28]
                         (offset = conv_offset(nearest(vit_r)); f_ref = down_channel(cat(vit_r,
                         conv1(deform(cnn_r, offset)))))
  ahiq_pyiqa_dist.onnx   vit_d, cnn_d, offset, f_ref -> score[1,1]
DeformConv2d is decomposed as nine bilinear grid_sample taps + a 1x1 conv (as zenmetrics'
models/ahiq/export.py does); the script checks it against torchvision.
Also writes ahiq_pyiqa.meta, ahiq_pyiqa_export.meta (versions, sha256s) and, with a pairs TSV,
replays those pairs through onnxruntime CPU against pyiqa's own score.
"""
import csv, hashlib, os, platform, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import pyiqa, timm, onnxruntime as ort
from PIL import Image

OUT = sys.argv[1]
PAIRS = sys.argv[2] if len(sys.argv) > 2 else None
os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)
COL = 'ahiq_pyiqa'
metric = pyiqa.create_metric('ahiq', device='cpu', as_loss=False)
net = metric.net.eval()
vit, res, dn, rg = net.vit, net.resnet50, net.deform_net, net.regressor
net.requires_grad_(False)  # timm/torchvision params require grad; tracing then refuses them as constants
print('vit', vit.pretrained_cfg.get('tag'), 'resnet', res.pretrained_cfg.get('tag'), flush=True)


def vit_taps(x):
    x = vit.patch_embed(x)
    x = vit._pos_embed(x)
    x = vit.patch_drop(x)
    x = vit.norm_pre(x)
    feats = []
    for blk in vit.blocks[:5]:
        x = blk(x)
        feats.append(x[:, 1:, :])
    f = torch.cat(feats, dim=2)
    return f.transpose(1, 2).reshape(1, 3840, 28, 28)


def cnn_taps(x):
    x = res.maxpool(res.act1(res.bn1(res.conv1(x))))
    feats = []
    for blk in res.layer1:
        x = blk(x)
        feats.append(x)
    return torch.cat(feats, dim=1)


def deform_gs(x, offset, weight, bias):
    """DeformConv2d (3x3, pad 1, stride 1) via 9 grid_sample taps + 1x1 conv. Offset channel
    2k = dy, 2k+1 = dx at kernel position k = ky*3 + kx."""
    B, C, H, W = x.shape
    off = offset.reshape(B, 9, 2, H, W)
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32),
                            indexing='ij')
    outs = []
    for k in range(9):
        pos_y = yy + (k // 3 - 1) + off[:, k, 0]
        pos_x = xx + (k % 3 - 1) + off[:, k, 1]
        grid = torch.stack([2 * (pos_x + 0.5) / W - 1, 2 * (pos_y + 0.5) / H - 1], dim=-1)
        outs.append(F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False))
    cat = torch.cat(outs, dim=1)
    co = weight.shape[0]
    w2 = weight.reshape(co, C, 9).permute(0, 2, 1).reshape(co, 9 * C, 1, 1)
    return F.conv2d(cat, w2, bias)


class Trunk(nn.Module):
    def forward(self, x):
        x = net.preprocess(x)
        return vit_taps(x), cnn_taps(x)


def fuse(cnn, offset):
    return dn.conv1(deform_gs(cnn, offset, dn.deform.weight, dn.deform.bias))


class RefHead(nn.Module):
    def forward(self, vit_r, cnn_r):
        offset = dn.conv_offset(F.interpolate(vit_r, size=cnn_r.shape[-2:], mode='nearest'))
        f_ref = rg.down_channel(torch.cat((vit_r, fuse(cnn_r, offset)), 1))
        return offset, f_ref


class DistHead(nn.Module):
    def forward(self, vit_d, cnn_d, offset, f_ref):
        f_dis = rg.down_channel(torch.cat((vit_d, fuse(cnn_d, offset)), 1))
        f_cat = torch.cat((f_dis - f_ref, f_dis, f_ref), 1)
        feat = rg.conv1(rg.feat_smoothing(f_cat))
        f, w = rg.conv(feat), rg.conv_attent(feat)
        return (f * w).sum(dim=-1).sum(dim=-1) / w.sum(dim=-1).sum(dim=-1)


trunk, refh, disth = Trunk().eval(), RefHead().eval(), DistHead().eval()

# --- self-checks against pyiqa's own modules ---------------------------------------------
a, b = torch.rand(1, 3, 224, 224), torch.rand(1, 3, 224, 224)
vt, ct = trunk(a)
hv = net.get_vit_feature(net.preprocess(a)).transpose(1, 2).reshape(1, 3840, 28, 28)
hc = net.get_resnet_feature(net.preprocess(a))
print('trunk vs pyiqa hooks:', float((hv - vt).abs().max()), float((hc - ct).abs().max()), flush=True)
assert float((hv - vt).abs().max()) < 1e-5 and float((hc - ct).abs().max()) < 1e-5
off = dn.conv_offset(F.interpolate(vt, size=(56, 56), mode='nearest'))
dref = dn.deform(ct, off)
dgot = deform_gs(ct, off, dn.deform.weight, dn.deform.bias)
print('deform_gs vs torchvision:', float((dref - dgot).abs().max()), flush=True)
assert float((dref - dgot).abs().max()) < 1e-4
vd, cd = trunk(b)
o, fr = refh(vt, ct)
got = float(disth(vd, cd, o, fr))
want = float(net.regress_score(b, a))
print('split head vs pyiqa regress_score (one crop):', got, want, abs(got - want), flush=True)
assert abs(got - want) < 1e-5

# --- export ------------------------------------------------------------------------------
for name, mod, args, ins, outs in (
    ('trunk', trunk, (a,), ['x'], ['vit', 'cnn']),
    ('ref', refh, (vt, ct), ['vit_r', 'cnn_r'], ['offset', 'f_ref']),
    ('dist', disth, (vd, cd, o, fr), ['vit_d', 'cnn_d', 'offset', 'f_ref'], ['score']),
):
    torch.onnx.export(mod, args, os.path.join(OUT, f'{COL}_{name}.onnx'), opset_version=17, dynamo=False,
                      do_constant_folding=True, input_names=ins, output_names=outs)
with open(os.path.join(OUT, f'{COL}.meta'), 'w') as f:
    f.write('channels 3840,768\nmin_size 1\n')

# --- replay through onnxruntime ----------------------------------------------------------
worst = None
if PAIRS:
    sess = {n: ort.InferenceSession(os.path.join(OUT, f'{COL}_{n}.onnx'), providers=['CPUExecutionProvider'])
            for n in ('trunk', 'ref', 'dist')}

    def load(p):
        x = np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8).astype(np.float32) / np.float32(255)
        t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)
        h, w = t.shape[2:]
        if min(h, w) <= 224:
            t = F.interpolate(t, scale_factor=225 / min(h, w), mode='bilinear')
        return t.numpy()

    worst = 0.0
    for r in csv.DictReader(open(PAIRS), delimiter='\t'):
        # Our own traced forwards fired pyiqa's SaveOutput hooks; drop those outputs first.
        net.save_output.clear(torch.device('cpu'))
        s = float(metric(r['dist_path'], r['ref_path']).flatten()[0])
        xr, xd = load(r['ref_path']), load(r['dist_path'])
        h, w = xr.shape[2:]
        sh, sw = (h - 224) // 4, (w - 224) // 4
        tot = 0.0
        for i in range(4):
            for j in range(5):
                cr = np.ascontiguousarray(xr[:, :, i * sh:i * sh + 224, j * sw:j * sw + 224])
                cdd = np.ascontiguousarray(xd[:, :, i * sh:i * sh + 224, j * sw:j * sw + 224])
                vr_, cr_ = sess['trunk'].run(None, {'x': cr})
                vd_, cd_ = sess['trunk'].run(None, {'x': cdd})
                of, f_r = sess['ref'].run(None, {'vit_r': vr_, 'cnn_r': cr_})
                tot += float(sess['dist'].run(None, {'vit_d': vd_, 'cnn_d': cd_, 'offset': of, 'f_ref': f_r})[0].item())
        o = tot / 20
        worst = max(worst, abs(o - s))
        print(r['distorted'], s, o, abs(o - s), flush=True)
    print('worst |onnx - pyiqa|:', worst)

with open(os.path.join(OUT, f'{COL}_export.meta'), 'w') as f:
    f.write(f'pyiqa {pyiqa.__version__}\ntorch {torch.__version__}\ntimm {timm.__version__}\n'
            f'onnxruntime {ort.__version__}\nhost {platform.node()}\n'
            f"vit_tag {vit.pretrained_cfg.get('tag')}\nresnet_tag {res.pretrained_cfg.get('tag')}\n"
            f'worst_onnx_vs_pyiqa {worst!r}\n')
    for n in sorted(os.listdir(OUT)):
        if n.startswith(COL) and not n.endswith('_export.meta'):
            f.write(f'sha256 {n} {hashlib.sha256(open(os.path.join(OUT, n), "rb").read()).hexdigest()}\n')
print('done')
