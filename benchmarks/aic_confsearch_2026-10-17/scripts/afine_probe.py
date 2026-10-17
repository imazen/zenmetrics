"""A-FINE preprocessing probe: upstream ChrisDud0257/AFINE QuickInference modules + afine.pth.
usage: afine_probe.py <up_quickinference_dir> <ckpt_dir> <pairs.tsv> <out.tsv> <variant>[,<variant>...]
Writes per (variant, pair): raw heads nd/nr/fid, scaled nds/nrs/fids, afine_all; tiled variants
average each term across tiles (all_m = adapter applied to the tile-mean scaled terms)."""
import csv, os, sys, time, importlib.util
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
QI, CK, pairs, out, variants = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5].split(',')
sys.path.insert(0, QI)
spec = importlib.util.spec_from_file_location('clip_model', os.path.join(QI, 'CLIP_ReturnFea', 'clip', 'model.py'))
cm = importlib.util.module_from_spec(spec); spec.loader.exec_module(cm)
from utils.network_archs import AFINEQhead, AFINEDhead, AFINENLM_NR_Fit, AFINENLM_FR_Fit_with_limit, AFINELearnLambda
torch.set_grad_enabled(False)
DEV = torch.device(os.environ.get('DEV', 'cpu'))
AMP = os.environ.get('AMP') == '1'
ck = torch.load(os.path.join(CK, 'afine.pth'), map_location='cpu', weights_only=False)
clip = cm.build_model(torch.jit.load(os.path.join(CK, 'ViT-B-32.pt'), map_location='cpu').state_dict()).float()
clip.load_state_dict(ck['finetuned_clip']); clip.eval()
qh = AFINEQhead(); qh.load_state_dict(ck['natural']); qh.eval()
dh = AFINEDhead(); dh.load_state_dict(ck['fidelity']); dh.eval()
sfr = AFINENLM_FR_Fit_with_limit(yita1=2, yita2=-2, yita3=0.5, yita4=0.15, yita3_upper=0.95, yita3_lower=0.05, yita4_upper=0.70, yita4_lower=0.01)
sfr.load_state_dict(ck['fidelity_scale']); sfr.eval()
snr = AFINENLM_NR_Fit(yita1=2, yita2=-2, yita3=4.9592, yita4=21.5968); snr.load_state_dict(ck['natural_scale']); snr.eval()
ad = AFINELearnLambda(k=5); ad.load_state_dict(ck['adapter']); ad.eval()
for _m in (clip, qh, dh, sfr, snr, ad): _m.to(DEV)
MEAN = torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1)
STD = torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1)
def raw(img):  # PIL -> 1x3xHxW float in [0,1]
    return torch.from_numpy(np.asarray(img, np.float32) / 255.).permute(2, 0, 1).unsqueeze(0).contiguous()
def norm(t, k=1):
    for _ in range(k): t = (t - MEAN) / STD
    return t
def pad32_bicubic(t):
    _, _, h, w = t.shape; nh, nw = h + (32 - h % 32) % 32, w + (32 - w % 32) % 32
    return F.interpolate(t, size=(nh, nw), mode='bicubic', align_corners=False) if (nh, nw) != (h, w) else t
def m32(x, up): return (x + 31) // 32 * 32 if up else x // 32 * 32
def terms(r, d, half=False):
    if half:
        clip.half(); _, fd = clip.encode_image(d.half()); _, fr = clip.encode_image(r.half()); clip.float()
        fd = [f.float() for f in fd]; fr = [f.float() for f in fr]
    else:
        d, r = d.to(DEV), r.to(DEV)
        with torch.autocast('cuda', enabled=AMP):
            _, fd = clip.encode_image(d); _, fr = clip.encode_image(r)
        fd = [x.float() for x in fd]; fr = [x.float() for x in fr]
    nd, nr, fid = qh(d, fd), qh(r, fr), dh(d, r, fd, fr)
    nds, nrs, fids = snr(nd), snr(nr), sfr(fid)
    return dict(nd=nd.item(), nr=nr.item(), fid=fid.item(), nds=nds.item(), nrs=nrs.item(), fids=fids.item(), all=ad(nds, nrs, fids).item())
def tiles(R, D, size, stride):
    _, _, h, w = R.shape; out = []
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            out.append((R[..., y:y + size, x:x + size], D[..., y:y + size, x:x + size]))
    return out
def variant(v, ri, di):
    """-> list of (ref, dis) normalized tensor pairs"""
    R, D = raw(ri), raw(di); h, w = R.shape[2:]
    if v == 'native': return [(pad32_bicubic(norm(R)), pad32_bicubic(norm(D)))]
    if v == 'swap': return [(pad32_bicubic(norm(D)), pad32_bicubic(norm(R)))]
    if v.startswith('norm') and v[4:].isdigit(): k = int(v[4:]); return [(pad32_bicubic(norm(R, k)), pad32_bicubic(norm(D, k)))]
    if v == 'norm2dis': return [(pad32_bicubic(norm(R)), pad32_bicubic(norm(D, 2)))]
    if v == 'norm2ref': return [(pad32_bicubic(norm(R, 2)), pad32_bicubic(norm(D)))]
    if v == 'nonorm': return [(pad32_bicubic(R), pad32_bicubic(D))]
    if v in ('cc', 'tl'):
        H, W = m32(h, 0), m32(w, 0); y0, x0 = ((h - H) // 2, (w - W) // 2) if v == 'cc' else (0, 0)
        return [(norm(R[..., y0:y0 + H, x0:x0 + W]), norm(D[..., y0:y0 + H, x0:x0 + W]))]
    if v.startswith('pad'):
        mode = v[3:]; H, W = m32(h, 1), m32(w, 1)
        p = (0, W - w, 0, H - h)
        if mode == 'zero': return [(norm(F.pad(R, p)), norm(D, 1) if False else norm(F.pad(D, p)))]
        if mode == 'nzero': return [(F.pad(norm(R), p), F.pad(norm(D), p))]  # zero in normalized space
        return [(norm(F.pad(R, p, mode={'refl': 'reflect', 'rep': 'replicate'}[mode])), norm(F.pad(D, p, mode={'refl': 'reflect', 'rep': 'replicate'}[mode])))]
    if v.startswith('rsdn') or v.startswith('rsup'):
        up = v.startswith('rsup'); H, W = m32(h, up), m32(w, up); kind = v[4:]
        if kind == 'pil':
            return [(norm(raw(ri.resize((W, H), Image.BICUBIC))), norm(raw(di.resize((W, H), Image.BICUBIC))))]
        mode, aa = {'bic': ('bicubic', False), 'bicaa': ('bicubic', True), 'bil': ('bilinear', False), 'bilaa': ('bilinear', True), 'near': ('nearest', False)}[kind]
        f = lambda t: F.interpolate(t, size=(H, W), mode=mode, align_corners=False if mode != 'nearest' else None, antialias=aa) if mode != 'nearest' else F.interpolate(t, size=(H, W), mode='nearest')
        return [(norm(f(R)), norm(f(D)))]
    if v.startswith('half'):
        f = lambda im: raw(im.resize((w // 2, h // 2), Image.BICUBIC))
        return [(pad32_bicubic(norm(f(ri))), pad32_bicubic(norm(f(di))))]
    if v.startswith('tile'):  # tile<size>s<stride>
        size, stride = (int(x) for x in v[4:].split('s'))
        return tiles(norm(R), norm(D), size, stride)
    if v == 'fp16': return 'fp16'
    if v == 'bgr': return [(pad32_bicubic(norm(R.flip(1))), pad32_bicubic(norm(D.flip(1))))]
    if v == 'x255': return [(pad32_bicubic(norm(R * 255)), pad32_bicubic(norm(D * 255)))]
    if v == 'imnet':
        m, s_ = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return [(pad32_bicubic((R - m) / s_), pad32_bicubic((D - m) / s_))]
    raise SystemExit('unknown variant ' + v)
rows = list(csv.DictReader(open(pairs), delimiter='\t'))
new = not os.path.exists(out)
f = open(out, 'a')
if new: f.write('variant\tdistorted\tntiles\tnd\tnr\tfid\tnds\tnrs\tfids\tall\tall_m\tsec\n')
for v in variants:
    for r in rows:
        t0 = time.time()
        ri, di = Image.open(r['ref_path']).convert('RGB'), Image.open(r['dist_path']).convert('RGB')
        pp = variant(v, ri, di)
        if pp == 'fp16': res = [terms(*variant('native', ri, di)[0], half=True)]
        else: res = [terms(a, b) for a, b in pp]
        mean = {k: float(np.mean([x[k] for x in res])) for k in res[0]}
        all_m = ad(torch.tensor([[mean['nds']]], device=DEV), torch.tensor([[mean['nrs']]], device=DEV), torch.tensor([[mean['fids']]], device=DEV)).item()
        f.write('\t'.join([v, r['distorted'], str(len(res))] + [repr(mean[k]) for k in ('nd', 'nr', 'fid', 'nds', 'nrs', 'fids', 'all')] + [repr(all_m), f'{time.time() - t0:.1f}']) + '\n'); f.flush()
