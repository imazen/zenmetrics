#!/usr/bin/env python3
"""Import only the explicitly selected oracle packages and inspect public signatures."""
import importlib,importlib.metadata,inspect,json,sys,types
import torch,torchvision
stub=types.ModuleType('IPython');stub.embed=lambda *a,**k:None
sys.modules.setdefault('IPython',stub);sys.modules.setdefault('imp',types.ModuleType('imp'))
names=['torch','torchvision','piq','skimage','torchmetrics','pytorch_msssim','lpips','DISTS_pytorch','stlpips_pytorch.stlpips','DeepDC_PyTorch','pycvvdp']
modules={name:importlib.import_module(name) for name in names}
records=[]
for name,member in [('piq','ssim'),('piq','multi_scale_ssim'),('piq','information_weighted_ssim'),('piq','fsim'),('pycvvdp','cvvdp')]:
 records.append(dict(module=name,member=member,signature=str(inspect.signature(getattr(modules[name],member)))))
print(json.dumps(dict(imported=names,public_apis=records),indent=2))
