#!/usr/bin/env python3
"""Export WD's raw-input VGG16 trunk and score the unmodified Cool-Chic oracle.

Reference source is caller-supplied, pinned by SHA-256 in the export receipt.
No weights are bundled. Torchvision VGG16 IMAGENET1K_V1 is used verbatim.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

import numpy as np
from PIL import Image
import torch
import torchvision


def reference(path):
    spec = importlib.util.spec_from_file_location('coolchic_wd_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WassersteinLoss().eval()


def load(path):
    return torchvision.transforms.functional.to_tensor(Image.open(path).convert('RGB')).unsqueeze(0)


def scores(loss, a, b):
    # Call the upstream statistics and weighting code, channel-chunked to bound
    # memory. The channel means are combined with their exact channel counts.
    out = []
    for k in range(1, 6):
        layers = []
        for fa, fb in zip(a, b):
            n, _, h, w = fa.shape
            value = 0.0
            chunk = n if n*h*w <= 64*1024*1024 else 16
            for start in range(0, n, chunk):
                ca, cb = fa[start:start+chunk], fb[start:start+chunk]
                sigma = torch.full((1, 1, h, w), float(k))
                d = loss.wasserstein_distortion(ca, cb, sigma, num_levels=5)
                value += float(d) * len(ca) / n
            layers.append(value)
        # Upstream accumulates layer contributions in f32.
        total = torch.zeros(1)
        for value in layers:
            total += value
        out.append(float(total))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['export', 'oracle', 'head-golden'])
    ap.add_argument('--reference', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ladder', action='append', default=[])
    ap.add_argument('--sizes', default='32,64,128,256,512,1024,2048,448x333,333x448,1024x684,684x1024,240x320,320x240')
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    loss = reference(args.reference)
    out = Path(args.out)
    if args.stage == 'export':
        out.mkdir(parents=True, exist_ok=True)
        graph = out/'wd_trunk.onnx'
        assert not graph.exists(), graph
        # The upstream feature reshaping is undone only for the graph boundary;
        # all values, tap positions and weights remain unchanged.
        # Register the actual parameters instead of capturing them as constants.
        class ExportTrunk(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = loss.wd_net.features
            def forward(self, x):
                result = []
                for i, layer in enumerate(self.features):
                    x = layer(x)
                    if i in (3, 8, 15, 22):
                        result.append(x)
                return tuple(result)
        torch.onnx.export(ExportTrunk().eval(), (torch.zeros(1,3,64,64),), graph,
                          dynamo=False, opset_version=17, input_names=['x'],
                          output_names=[f'tap{i}' for i in range(4)],
                          dynamic_axes={'x': {2:'h',3:'w'}, **{f'tap{i}':{2:f'h{i}',3:f'w{i}'} for i in range(4)}})
        (out/'wd.meta').write_text('channels 64,128,256,512\nmin_size 8\n')
        checkpoint = Path(torch.hub.get_dir())/'checkpoints/vgg16-397923af.pth'
        (out/'wd_export.json').write_text(json.dumps({
            'python':platform.python_version(), 'torch':torch.__version__,
            'torchvision':torchvision.__version__, 'reference_sha256':hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
            'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'graph_sha256':hashlib.sha256(graph.read_bytes()).hexdigest(),
            'input':'RGB f32 0..1, no ImageNet normalization', 'taps':[3,8,15,22]}, indent=2)+'\n')
    elif args.stage == 'head-golden':
        a = (torch.arange(2*5*7).reshape(2,1,5,7)%17).float()/7
        b = a*0.91 + 0.13
        chunked = scores(loss,[a],[b])
        direct = [float(loss.wasserstein_distortion(a,b,torch.full((1,1,5,7),float(k)),5)) for k in range(1,6)]
        assert np.array_equal(np.array(chunked), np.array(direct))
        out.write_text(json.dumps({'a':a.flatten().tolist(),'b':b.flatten().tolist(),'scores':direct},indent=2)+'\n')
        print('head golden',direct,flush=True)
    else:
        sizes = set(args.sizes.split(','))
        files = {}
        for folder in args.ladder:
            for p in sorted(Path(folder).glob('*_dist.png')):
                stem,size = p.stem[:-5].split('_',1)
                if size in sizes:
                    files[stem,size] = p
        assert files, 'no matching ladder pairs'
        with out.open('x') as stream:
            writer = csv.writer(stream,delimiter='\t')
            writer.writerow(['image','size',*[f'wd_s{k}' for k in range(1,6)]])
            for (stem,size),dist in sorted(files.items()):
                ref = dist.with_name(dist.name.replace('_dist.png','_ref.png'))
                a = loss.wd_net(load(ref)); b = loss.wd_net(load(dist))
                values = scores(loss,a,b)
                writer.writerow([stem,size,*[f'{v:.17e}' for v in values]])
                stream.flush()
                print(stem,size,*[f'{v:.9g}' for v in values],flush=True)

if __name__ == '__main__':
    main()
