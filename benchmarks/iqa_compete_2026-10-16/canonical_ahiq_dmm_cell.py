#!/usr/bin/env python3
"""Original AHIQ epoch-32 nine-crop and original DMM resident RGB callables."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from canonical_author_network_cell import functions, setup_prefix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path)
    p.add_argument('metric', choices=['ahiq', 'dmm'])
    p.add_argument('width', type=int);p.add_argument('height', type=int)
    p.add_argument('reference', type=Path);p.add_argument('distorted', type=Path)
    p.add_argument('threads', type=int);p.add_argument('samples', type=int)
    p.add_argument('--setup-only', action='store_true')
    a = p.parse_args();cfg=json.loads(a.config.read_text())
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
    if a.metric == 'ahiq':
        os.environ['AHIQ_SRC']=cfg['ahiq_source']
        export=Path(cfg['exports'])/'ahiq/export.py'
        namespace=setup_prefix(export, 'vit_taps', ['unused', Path.cwd(), cfg['ahiq_checkpoint'], 8])
        if namespace['ck']['epoch'] != 32:
            raise ValueError('frozen AHIQ model requires original epoch 32')
        functions(export, ['crops', 'prep', 'score_pair'], namespace)
        def score(r, d):return namespace['score_pair'](r, d)
    else:
        sys.path.insert(0, cfg['dmm_source'])
        from DMM_PyTorch.DMM import DMM, prepare_image_adt
        model=DMM().eval()
        def score(r, d):return float(model(prepare_image_adt(r), prepare_image_adt(d)).item())
    if a.setup_only:
        print(json.dumps(dict(metric=a.metric,setup='ready',torch=torch.__version__,threads=a.threads)))
        return
    r,d=[np.fromfile(path,np.uint8).reshape(a.height,a.width,3) for path in [a.reference,a.distorted]]
    with torch.inference_mode():
        for i in range(a.samples+1):
            start=time.perf_counter_ns();value=score(Image.fromarray(r),Image.fromarray(d));elapsed=time.perf_counter_ns()-start
            print(('COLD' if i==0 else 'SAMPLE')+f' {elapsed} {value:.17g}',flush=True)


if __name__ == '__main__':
    main()
