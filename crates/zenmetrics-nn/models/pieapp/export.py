#!/usr/bin/env python3
"""Export the independent paper/tensor-metadata reconstruction, never official code.

The fixed convention is BGR code values and NCHW flattening, established
through black-box probes.
The build is not a reference-parity claim; the complete nn-parity gate must
pass using independently generated official-release goldens and KonJND scale.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import torch
from torch import nn


class Ingress(nn.Module):
    def __init__(self, network):
        super().__init__();self.network=network

    def forward(self, reference, distorted):
        reference,distorted=reference*255,distorted*255
        reference,distorted=reference[:,[2,1,0]],distorted[:,[2,1,0]]
        return self.network(reference,distorted)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    torch.set_grad_enabled(False);torch.set_num_threads(16)
    spec=importlib.util.spec_from_file_location('pieapp_reconstruction',Path(__file__).with_name('network.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    digest=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    if digest!=module.CHECKPOINT_SHA256:raise ValueError('official checkpoint SHA-256 mismatch')
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=Ingress(module.PieAPP(state)).eval()
    a.out.mkdir(exist_ok=True,parents=True)
    for name in ('pieapp.onnx','pieapp.meta','pieapp_export.json'):
        if (a.out/name).exists():raise FileExistsError(f'Preserve existing artifacts before exporting: {a.out/name}')
    generator=torch.Generator().manual_seed(1301)
    r=torch.rand((2,3,64,64),generator=generator);d=torch.rand((2,3,64,64),generator=generator)
    graph=a.out/'pieapp.onnx'
    torch.onnx.export(model,(r,d),graph,dynamo=False,input_names=['reference','distorted'],output_names=['patches'],dynamic_axes={'reference':{0:'n'},'distorted':{0:'n'},'patches':{0:'n'}},opset_version=17)
    (a.out/'pieapp.meta').write_text('channels 3\nmin_size 64\n')
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=16
    session=ort.InferenceSession(str(graph),options,providers=['CPUExecutionProvider'])
    worst=0.0
    for n in (1,3,32):
        r=torch.rand((n,3,64,64),generator=generator);d=torch.rand((n,3,64,64),generator=generator)
        expected=model(r,d).numpy()
        got=session.run(None,{'reference':r.numpy(),'distorted':d.numpy()})[0]
        error=float(abs(expected-got).max());worst=max(worst,error)
        # Conversion check only, not a replacement for the measured JND gate.
        if not bool(__import__('numpy').allclose(expected,got,rtol=1e-4,atol=1e-5)):
            raise ValueError(f'ONNX conversion failed at batch {n}: {error}')
    metadata=dict(python=platform.python_version(),torch=torch.__version__,onnxruntime=ort.__version__,checkpoint_sha256=digest,graph_sha256=hashlib.sha256(graph.read_bytes()).hexdigest(),flatten='channels-first',input_units='u8',bgr=True,host_input='planar RGB [0,1]',opset=17,batch_sizes_checked=[1,3,32],conversion_max_absolute=worst,reference_parity_qualified=False,qualification='Run nn-pieapp-gates with independent official-release goldens and measured KonJND scale.')
    (a.out/'pieapp_export.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata),flush=True)

if __name__=='__main__':main()
