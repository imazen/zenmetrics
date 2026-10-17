"""DMM probe: upstream Baoliang93/DMM (DMM_PyTorch/DMM.py) and pyiqa 0.1.16 `dmm` as black boxes,
under resize / device / dtype variants.
usage: dmm_probe.py <up_dir_containing_DMM_PyTorch> <pairs.tsv> <out.tsv> <variant>[,...]
variants: <impl>_<resize>_<device>[_f64]  impl=up|pyiqa  resize=pilaspect|tsquare|tsquareaa|pilsquare|tensoraspect|none
"""
import csv, sys, time, os
import torch, torch.nn.functional as F, torchvision.transforms.functional as TF
from PIL import Image
up, pairs, out, variants = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(',')
sys.path.insert(0, up)
torch.set_grad_enabled(False)
from DMM_PyTorch.DMM import DMM as UpDMM
import pyiqa
def tar(ms): return max(int(ms / 48.0) * 32, 128)
def prep(img, mode):
    w, h = img.size; ms = min(w, h)
    if mode == 'pilaspect': img = TF.resize(img, tar(ms)) if ms > 128 else img  # upstream prepare_image_adt
    elif mode == 'pilsquare': img = img.resize((tar(ms), tar(ms)), Image.BILINEAR) if ms > 128 else img
    t = TF.to_tensor(img).unsqueeze(0)
    if mode == 'tsquare' and ms > 128: t = F.interpolate(t, size=tar(ms), mode='bilinear', align_corners=False)  # pyiqa prepare_image_adt
    if mode == 'tsquareaa' and ms > 128: t = F.interpolate(t, size=(tar(ms), tar(ms)), mode='bilinear', align_corners=False, antialias=True)
    if mode == 'tensoraspect' and ms > 128:
        s = tar(ms); size = (s, int(s * w / h)) if h <= w else (int(s * h / w), s)
        t = F.interpolate(t, size=size, mode='bilinear', align_corners=False)
    return t
rows = list(csv.DictReader(open(pairs), delimiter='\t'))
f = open(out, 'a')
models = {}
for v in variants:
    parts = v.split('_'); impl, mode, dev = parts[:3]; f64 = 'f64' in parts[3:]
    dev = torch.device(dev)
    if impl == 'up':
        net = UpDMM().to(dev).eval()
        if f64: net = net.double()
        call = lambda r, d: net(r, d)
    else:
        m = pyiqa.create_metric('dmm', device=dev)
        net = m.net
        net.prepare_image_adt = lambda t: t  # resizing done here so all variants share prep()
        if f64: net.double()
        call = lambda r, d: net(d, r)
    for r in rows:
        t0 = time.time()
        R = prep(Image.open(r['ref_path']).convert('RGB'), mode).to(dev)
        D = prep(Image.open(r['dist_path']).convert('RGB'), mode).to(dev)
        if f64: R, D = R.double(), D.double()
        s = float(call(R, D).item())
        f.write(f"{v}\t{r['distorted']}\t{s!r}\t{time.time()-t0:.2f}\n"); f.flush()
