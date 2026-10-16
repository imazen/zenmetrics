import sys, types, csv, numpy as np, torch
sys.modules.setdefault('imp', types.ModuleType('imp'))
from DeepDC_PyTorch import DeepDC
from DeepDC_PyTorch.DeepDC import prepare_image
from PIL import Image
torch.set_grad_enabled(False)
m = DeepDC().eval()
rows = list(csv.DictReader(open('/home/lilith/tmp/aic-sub-crop/metrics_subset.csv')))[:8]
for r in rows:
    R = Image.open('/home/lilith/tmp/aic-sub-crop/img/' + r['source']).convert('RGB')
    D = Image.open('/home/lilith/tmp/aic-sub-crop/img/' + r['distorted']).convert('RGB')
    a = float(m(prepare_image(R, resize=False), prepare_image(D, resize=False)))
    b = float(m(prepare_image(R, resize=True), prepare_image(D, resize=True)))
    c = float(m(prepare_image(D, resize=True), prepare_image(R, resize=True)))
    print(f"{r['distorted']:28s} published {float(r['DeepDC']):.6f}  noresize {a:.6f}  resize224 {b:.6f}  resize224 swapped {c:.6f}")
