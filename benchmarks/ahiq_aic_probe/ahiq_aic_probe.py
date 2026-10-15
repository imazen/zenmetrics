# Canonical AHIQ AIC-4 convention probe (i265).
#
#   venv/bin/python ahiq_aic_probe.py <img_dir> <metrics_csv> <out_tsv> [conv ...]
#
# Scores every row of <metrics_csv> with UPSTREAM's own inference path
# (SaveOutput hooks + get_resnet_feature/get_vit_feature + deform_fusion
# with torchvision DeformConv2d + Pixel_Prediction, epoch-32 checkpoint)
# under each named convention and writes <out_tsv>:
#   distorted, source, published, <conv>..., delta columns (published units
#   and JND units, AHIQ's KonJND scale = 0.024882), jitter std.
#
# Conventions:
#   s9           native [-1,1], the port's deterministic nine-crop mean
#   stream20     native [-1,1], upstream default: 20 random crops from a
#                continuing np.seed(1919) stream in dataset order
#   s9_imnet     ImageNet mean/std normalisation, nine-crop
#   stream20_imnet  ImageNet norm, continuing-stream 20 crops
#   s20          native [-1,1], 20 random crops, np.seed(1919) per image
#
# AHIQ_VIT_TAG overrides the ViT backbone tag (default
# vit_base_patch8_224.augreg_in21k_ft_in1k, upstream's timm 0.5.4
# resolution).
#
# Backbones are the timm==0.5.4 default_cfg resolutions upstream loads,
# named explicitly: vit_base_patch8_224.augreg_in21k_ft_in1k + resnet50.a1_in1k.
import csv, os, sys
import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
import importlib.util
from timm.models.resnet import Bottleneck
from timm.models.vision_transformer import Block

AHQ = os.environ.get('AHIQ_SRC', '/home/lilith/tmp/nn-port-scratch/ahiq')
CKPT = os.environ.get(
    'AHIQ_CKPT',
    '/home/lilith/tmp/nn-port-scratch/ahiq-ckpt/1Nk-IpjnDNXbWacoh3T69wkSYhYYWip2W.pth')
IMG, CSV, OUT = sys.argv[1:4]
CONVS = sys.argv[4:] or ['s9', 'stream20']
# AHIQ's KonJND-1k pooled median |d| scale (crates/zenmetrics-nn/models/
# jnd_scale.tsv); overridable for reruns.
JND = float(os.environ.get('AHIQ_JND', '0.024882'))

def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

dr = load_mod('deform_regressor', os.path.join(AHQ, 'model/deform_regressor.py'))
xf = load_mod('extract_feature', os.path.join(AHQ, 'script/extract_feature.py'))

class Opt:
    patch_size = 8

torch.set_grad_enabled(False)
# Upstream's timm 0.5.4 resolves the bare name to augreg; AHIQ_VIT_TAG
# overrides the tag when probing other timm resolutions.
VIT_TAG = os.environ.get(
    'AHIQ_VIT_TAG', 'vit_base_patch8_224.augreg_in21k_ft_in1k')
vit = timm.create_model(VIT_TAG, pretrained=True).eval()
res = timm.create_model('resnet50.a1_in1k', pretrained=True).eval()
deform_net = dr.deform_fusion(Opt()).eval()
regressor = dr.Pixel_Prediction().eval()
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
deform_net.load_state_dict(ck['deform_net_model_state_dict'])
regressor.load_state_dict(ck['regressor_model_state_dict'])
for m in (vit, res, deform_net, regressor):
    m.requires_grad_(False)


class SaveOutput:  # upstream utils/util.py verbatim
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


save = SaveOutput()
for layer in res.modules():
    if isinstance(layer, Bottleneck):
        layer.register_forward_hook(save)
for layer in vit.modules():
    if isinstance(layer, Block):
        layer.register_forward_hook(save)


def score_crop(d_img, r_img):
    """test.py's per-crop branch verbatim (batch 1, CPU)."""
    save.clear()
    _ = vit(d_img)
    vit_dis = xf.get_vit_feature(save)
    save.clear()
    _ = vit(r_img)
    vit_ref = xf.get_vit_feature(save)
    save.clear()
    B, N, C = vit_ref.shape
    vit_ref = vit_ref.transpose(1, 2).view(B, C, 28, 28)
    vit_dis = vit_dis.transpose(1, 2).view(B, C, 28, 28)
    _ = res(d_img)
    cnn_dis = xf.get_resnet_feature(save)
    save.clear()
    cnn_dis = deform_net(cnn_dis, vit_ref)
    _ = res(r_img)
    cnn_ref = xf.get_resnet_feature(save)
    save.clear()
    cnn_ref = deform_net(cnn_ref, vit_ref)
    return float(regressor(vit_dis, vit_ref, cnn_dis, cnn_ref))


M = np.array([.485, .456, .406], np.float32)
S = np.array([.229, .224, .225], np.float32)


def prep(pil, imnet=False):
    a = np.asarray(pil.convert('RGB'), dtype=np.float32) / 255.0
    a = (a - M) / S if imnet else (a - 0.5) / 0.5
    return torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0)


def crops9(h, w):
    ch, cw = h // 2, w // 2
    return [(0, 0), (0, w - 224), (h - 224, 0), (h - 224, w - 224),
            (ch - 112, cw - 112), (ch - 112, 0), (ch - 112, w - 224),
            (0, cw - 112), (h - 224, cw - 112)]


def conv_score(conv, r, d):
    h, w = d.shape[2], d.shape[3]
    base = conv[:-6] if conv.endswith('_imnet') else conv
    if base == 's9':
        return sum(score_crop(d[:, :, t:t + 224, l:l + 224],
                              r[:, :, t:t + 224, l:l + 224])
                   for t, l in crops9(h, w)) / 9
    n = int(base[1:])
    s = 0.0
    for _ in range(n):
        t = np.random.randint(0, h - 224)
        l = np.random.randint(0, w - 224)
        s += score_crop(d[:, :, t:t + 224, l:l + 224],
                        r[:, :, t:t + 224, l:l + 224])
    return s / n


rows = list(csv.DictReader(open(CSV)))
out_rows = []
jit = {}  # per-pair seed-jitter std for the continuing-stream conventions

# stream conventions run the np.seed(1919) ONCE for the whole list, in
# dataset order (upstream's test.py draws per image off one stream).
streams = {c: np.random.RandomState(1919) for c in CONVS if 'stream' in c}

for row in rows:
    rn, dn = row['source'], row['distorted']
    pub = float(row['AHIQ'])
    rec = {'distorted': dn, 'source': rn, 'published': pub}
    for conv in CONVS:
        imnet = conv.endswith('_imnet')
        r = prep(Image.open(os.path.join(IMG, rn)), imnet)
        d = prep(Image.open(os.path.join(IMG, dn)), imnet)
        h, w = d.shape[2], d.shape[3]
        if conv in streams:
            rs = streams[conv]
            s = 0.0
            for _ in range(20):
                t = rs.randint(0, h - 224)
                l = rs.randint(0, w - 224)
                s += score_crop(d[:, :, t:t + 224, l:l + 224],
                                r[:, :, t:t + 224, l:l + 224])
            got = s / 20
        elif conv == 's20':
            np.random.seed(1919)
            s = 0.0
            for _ in range(20):
                t = np.random.randint(0, h - 224)
                l = np.random.randint(0, w - 224)
                s += score_crop(d[:, :, t:t + 224, l:l + 224],
                                r[:, :, t:t + 224, l:l + 224])
            got = s / 20
        else:  # s9 / s9_imnet
            got = conv_score(conv, r, d)
        rec[conv] = got
        rec[f'{conv}_delta'] = got - pub
        rec[f'{conv}_delta_jnd'] = (got - pub) / JND
    out_rows.append(rec)
    print(dn, ' '.join(f'{c}={rec[c]:.4f} d={rec[c+"_delta"]:+.4f}'
                       for c in CONVS), flush=True)

# seed-jitter std: for each stream convention, re-score K seeds per pair on
# the FIRST 10 pairs (jitter is per-draw; 10 pairs estimates it fine).
K = 5
for conv in CONVS:
    if 'stream' not in conv and conv != 's20':
        continue  # jitter applies to seeded draws only
    imnet = conv.endswith('_imnet')
    stds = []
    for row in rows[:10]:
        rn, dn = row['source'], row['distorted']
        r = prep(Image.open(os.path.join(IMG, rn)), imnet)
        d = prep(Image.open(os.path.join(IMG, dn)), imnet)
        h, w = d.shape[2], d.shape[3]
        vals = []
        for seed in range(1919, 1919 + K):
            rs = np.random.RandomState(seed)
            s = 0.0
            for _ in range(20):
                t = rs.randint(0, h - 224)
                l = rs.randint(0, w - 224)
                s += score_crop(d[:, :, t:t + 224, l:l + 224],
                                r[:, :, t:t + 224, l:l + 224])
            vals.append(s / 20)
        stds.append(float(np.std(vals)))
    jit[conv] = float(np.median(stds))
    print(f'{conv}: median per-pair seed-jitter std {jit[conv]:.4f}', flush=True)

fields = ['distorted', 'source', 'published']
for c in CONVS:
    fields += [c, f'{c}_delta', f'{c}_delta_jnd']
with open(OUT, 'w', newline='') as f:
    wcsv = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
    wcsv.writeheader()
    for rec in out_rows:
        wcsv.writerow(rec)
    for c, j in jit.items():
        wcsv.writerow({'distorted': f'# seed-jitter std {c}', 'published': j})
print('wrote', OUT)
