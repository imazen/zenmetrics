import os, sys, numpy as np, torch, torch.nn.functional as F, timm
from PIL import Image
import importlib.util

AHQ = '/home/lilith/tmp/nn-port-scratch/ahiq'
CKPT = '/home/lilith/tmp/nn-port-scratch/ahiq-ckpt/1Nk-IpjnDNXbWacoh3T69wkSYhYYWip2W.pth'
sys.path.insert(0, AHQ)
import importlib.util
def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
dr = load_mod('deform_regressor', os.path.join(AHQ, 'model/deform_regressor.py'))
class Opt: patch_size = 8
torch.set_grad_enabled(False)
vit = timm.create_model('vit_base_patch8_224', pretrained=False).eval()
res = timm.create_model('resnet50', pretrained=False).eval()
deform_net = dr.deform_fusion(Opt()).eval(); regressor = dr.Pixel_Prediction().eval()
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
deform_net.load_state_dict(ck['deform_net_model_state_dict']); regressor.load_state_dict(ck['regressor_model_state_dict'])
for m in (vit, res, deform_net, regressor): m.requires_grad_(False)
import torchvision.models as tvm
sw = tvm.vit_b_16(weights=None); 
# need pretrained backbones - load from timm hub state already in export's cache? use pretrained=True
vit = timm.create_model('vit_base_patch8_224', pretrained=True).eval()
res = timm.create_model('resnet50', pretrained=True).eval()
for m in (vit, res): m.requires_grad_(False)

def vit_taps(x):
    x = vit.patch_embed(x); x = vit._pos_embed(x); feats = []
    for blk in vit.blocks[:5]: x = blk(x); feats.append(x[:, 1:, :])
    return torch.cat(feats, dim=2).transpose(1, 2).reshape(1, 3840, 28, 28)
def cnn_taps(x):
    x = res.conv1(x); x = res.bn1(x); x = res.act1(x); x = res.maxpool(x); feats = []
    for blk in res.layer1: x = blk(x); feats.append(x)
    return torch.cat(feats, dim=1)
def score(d, r):
    vd, cd = vit_taps(d), cnn_taps(d); vr, cr = vit_taps(r), cnn_taps(r)
    return float(regressor(vd, vr, deform_net(cd, vr), deform_net(cr, vr)))

def prep(pil):
    a = np.asarray(pil.convert('RGB'), dtype=np.float32) / 255.0
    a = (a - 0.5) / 0.5
    return torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0)

def crops9(h, w):
    ch, cw = h // 2, w // 2
    return [(0,0),(0,w-224),(h-224,0),(h-224,w-224),(ch-112,cw-112),(ch-112,0),(ch-112,w-224),(0,cw-112),(h-224,cw-112)]

import random
IMG = '/home/lilith/tmp/aic-sub-crop/img'
pairs = [('PTC_S01_Ref_00.png','PTC_S01_AVIF_01.png',0.5894),('PTC_S03_Ref_00.png',None,None)]
import csv
pub = {r['distorted']: float(r['AHIQ']) for r in csv.DictReader(open('/home/lilith/tmp/aic-sub-crop/metrics_subset.csv'))}
names = [('PTC_S01_Ref_00.png','PTC_S01_AVIF_01.png'),('PTC_S02_Ref_00.png','PTC_S02_CCM_01.png'),('PTC_S03_Ref_00.png',None)]
# pick dist names
import os
files = set(os.listdir(IMG))
sel = [f for f in sorted(files) if '_Ref_' not in f][:3]
print('using', sel)
for dn in sel:
    rn = dn.split('_')[0]+'_'+dn.split('_')[1]+'_Ref_00.png'
    r = prep(Image.open(os.path.join(IMG, rn)))
    d = prep(Image.open(os.path.join(IMG, dn)))
    h, w = d.shape[2], d.shape[3]
    # 9-crop
    s9 = sum(score(d[:,:,t:t+224,l:l+224], r[:,:,t:t+224,l:l+224]) for t,l in crops9(h,w)) / 9
    # 20 random crops, np seeded 1919 (first image in stream)
    np.random.seed(1919)
    s20 = 0.0
    for i in range(20):
        t = np.random.randint(0, h-224); l = np.random.randint(0, w-224)
        s20 += score(d[:,:,t:t+224,l:l+224], r[:,:,t:t+224,l:l+224])
    s20 /= 20
    # resize to 224 then single view
    dr2 = F.interpolate(d, size=(224,224), mode='bilinear', align_corners=False, antialias=True)
    rr2 = F.interpolate(r, size=(224,224), mode='bilinear', align_corners=False, antialias=True)
    s_resize = score(dr2, rr2)
    # PIL bilinear to 224 (approximates upstream's cv2.resize default bilinear)
    def pilprep(t, rn_dn):
        return prep(Image.open(rn_dn).resize((224,224), Image.BILINEAR))
    s_cv2 = score(pilprep(d, os.path.join(IMG, dn)), pilprep(r, os.path.join(IMG, rn)))
    # 5-crop idx 0-4
    s5 = sum(score(d[:,:,t:t+224,l:l+224], r[:,:,t:t+224,l:l+224]) for t,l in crops9(h,w)[:5]) / 5
    print(f'{dn}: pub={pub[dn]:.4f} s9={s9:.4f} s20={s20:.4f} s_resize={s_resize:.4f} s_cv2={s_cv2:.4f} s5={s5:.4f}')
