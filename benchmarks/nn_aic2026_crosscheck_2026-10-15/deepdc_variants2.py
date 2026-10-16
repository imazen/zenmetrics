import sys, types, csv, torch
sys.modules.setdefault('imp', types.ModuleType('imp'))
from DeepDC_PyTorch import DeepDC
from DeepDC_PyTorch.DeepDC import prepare_image
from PIL import Image
torch.set_grad_enabled(False)
base = DeepDC().eval()
norm = DeepDC().eval()
fe = norm.features_extractor
fe.use_input_norm = True
fe.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
fe.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
relu = DeepDC(features_to_compute=('relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4')).eval()
rows = list(csv.DictReader(open('/home/lilith/tmp/aic-sub-crop/metrics_subset.csv')))[:8]
for r in rows:
    R = prepare_image(Image.open('/home/lilith/tmp/aic-sub-crop/img/' + r['source']).convert('RGB'), resize=False)
    D = prepare_image(Image.open('/home/lilith/tmp/aic-sub-crop/img/' + r['distorted']).convert('RGB'), resize=False)
    print(f"{r['distorted']:24s} pub {float(r['DeepDC']):.6f}  base {float(base(R, D)):.6f}  inputnorm {float(norm(R, D)):.6f}  relu-taps {float(relu(R, D)):.6f}")
