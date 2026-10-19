"""Canonical pilot: resident RGB pair -> both NR scores, including both ingresses."""
import argparse
import json
import os
import time
from pathlib import Path
from panel import digest
from workload import ADAPTER, VERSION_SUFFIX


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference',type=Path)
    parser.add_argument('distorted',type=Path)
    parser.add_argument('weights',type=Path)
    args=parser.parse_args()
    import numpy as np
    import torch
    import pyiqa
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    model=pyiqa.create_metric('brisque',device='cpu',pretrained_model_path=str(args.weights))
    r=np.fromfile(args.reference,dtype=np.uint8).reshape(512,512,3)
    d=np.fromfile(args.distorted,dtype=np.uint8).reshape(512,512,3)
    print('IQA_PROCESS '+json.dumps({'pid':os.getpid(),'pgid':os.getpgrp()}),flush=True)
    print('VERSION pyiqa-'+pyiqa.__version__+VERSION_SUFFIX,flush=True)
    print('WORKLOAD '+json.dumps(dict(metric_name='brisque',width=512,height=512,threads=torch.get_num_threads(),mode='score',timing_phase='warm',reference=str(args.reference.absolute()),distorted=str(args.distorted.absolute()),adapter=ADAPTER,adapter_sha256=digest(__file__),weights_sha256=digest(args.weights),input_sha256={'reference':digest(args.reference),'distorted':digest(args.distorted)},sample_count=10)),flush=True)

    def score_pair():
        # Public Rust PairedNoReference evaluates reference and distorted separately.
        x=torch.from_numpy(r.copy()).permute(2,0,1).unsqueeze(0).contiguous().float()/255
        y=torch.from_numpy(d.copy()).permute(2,0,1).unsqueeze(0).contiguous().float()/255
        ref_score=model(x)
        distorted_score=model(y)
        return float(ref_score.item()),float(distorted_score.item())

    score_pair()
    for i in range(10):
        start=time.perf_counter_ns()
        scores=score_pair()
        elapsed=time.perf_counter_ns()-start
        print('SAMPLE '+json.dumps(dict(sample_id=i+1,ns=elapsed,reference_score=scores[0],score=scores[1])),flush=True)


if __name__=='__main__':main()
