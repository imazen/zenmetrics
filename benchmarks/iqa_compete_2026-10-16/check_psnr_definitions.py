#!/usr/bin/env python3
"""Check eight PSNR routes against exact integer color definitions and scikit-image.

Definition oracle only: no upstream implementation is claimed for the color
variants, no external acceptance bound is selected, and no timing is qualified.
Integer coefficient numerators avoid floating-point rounding at color ties.
Caller supplies both heavy locks and run-heavy around the complete grid.
"""
import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import numpy as np
import skimage
from skimage.metrics import peak_signal_noise_ratio

from haarpsi_primary import execute
from octave_primary import sha


def db(mse):
    return math.inf if mse == 0 else 10*math.log10(255**2/mse)


def exact_planes(rgb):
    # All coefficients below are integer numerators; division is deferred.
    r,g,b=(rgb[...,c].astype(np.int64) for c in range(3))
    round_positive=lambda n,d:(2*n+d)//(2*d)
    return dict(y709=2126*r+7152*g+722*b,y601=299*r+587*g+114*b,
        studio_y=round_positive(16*255000+65481*r+128553*g+24966*b,255000),
        cb=round_positive(128*255000-37797*r-74203*g+112000*b,255000),
        cr=round_positive(128*255000+112000*r-93786*g-18214*b,255000),
        studio709=round_positive(16*2550000+219*(2126*r+7152*g+722*b),2550000))


def values(r,d):
    a,b=exact_planes(r),exact_planes(d)
    def exact_sum_squares(delta):
        squared=np.square(delta).reshape(-1)
        # Each partial fits int64 even for the largest coefficient numerator;
        # Python integers then accumulate the partials without a size limit.
        return sum(int(squared[i:i+1024].sum(dtype=np.int64)) for i in range(0,squared.size,1024))
    mse={k:exact_sum_squares(a[k]-b[k])/a[k].size for k in a}
    return {'psnr':float(peak_signal_noise_ratio(r,d,data_range=255)),
        'psnr-y':db(mse['y709']/10000**2),'psnr-y601':db(mse['y601']/1000**2),
        'psnr-y-studio601':db(mse['studio_y']),'psnr-y-libvmaf':db(mse['studio709']),
        'psnr-cb':db(mse['cb']),'psnr-cr':db(mse['cr']),
        'psnr-ycbcr611':db((6*mse['studio_y']+mse['cb']+mse['cr'])/8)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['build','inputs','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.inputs=a.inputs.resolve(strict=True)
    receipt=json.loads((a.build/'build-receipt.json').read_text())
    binary=(a.build/'target/release/iqa-compete').resolve(strict=True)
    assert sha(binary)==receipt['binary_sha256'] and receipt['source_revision']=='e2d414c33a6d3c7da50a818fa36e56654d3ebeef'
    a.output.mkdir();root=a.output.resolve()
    # Exact identity, channel separation and neutral half-tie controls.
    control=np.array([[[0,0,0],[255,255,255],[255,0,0],[0,255,0],[0,0,255]]],dtype=np.uint8)
    assert all(math.isinf(v) for v in values(control,control).values())
    planes=exact_planes(control)
    assert planes['studio_y'][0,:2].tolist()==[16,235] and planes['cb'][0,:2].tolist()==[128,128]
    identity=dict(worker_sha256=sha(Path(__file__)),inputs_sha256=sha(a.inputs),receipt_sha256=sha(a.build/'build-receipt.json'),
        binary_sha256=sha(binary),numpy=np.__version__,skimage=skimage.__version__,
        skimage_psnr_source_sha256=sha(Path(__import__('inspect').getfile(peak_signal_noise_ratio))),
        definitions='exact integer coefficients matching documented RGB/full-709/full-601/studio-601/studio-709 conventions; round half upward',
        bound='UNBOUND',qualified_timing_cells=0)
    (root/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    rows=[]
    for item in json.loads(a.inputs.read_text())['pairs']:
        paths={r:(a.inputs.parent/item[r]).resolve(strict=True) for r in ['reference','distorted']}
        assert all(sha(path)==item[r+'_sha256'] for r,path in paths.items())
        pixels={r:np.fromfile(path,dtype=np.uint8).reshape(item['height'],item['width'],3) for r,path in paths.items()}
        scores=values(pixels['reference'],pixels['distorted']);assert scores==values(pixels['reference'],pixels['distorted'])
        for threads in [1,max(1,len(os.sched_getaffinity(0))-4)]:
            env=dict(os.environ,LC_ALL='C')
            for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS']:env[key]=str(threads)
            for metric,canonical in scores.items():
                directory=root/f"{metric}-{item['pair']}-{item['size_name']}-t{threads}";directory.mkdir()
                config=dict(width=item['width'],height=item['height'],reference=str(paths['reference']),distorted=str(paths['distorted']),
                    metric_name=metric,threads=threads,probe_only=True,only_mode='score',timing_phase='warm')
                (directory/'config.json').write_text(json.dumps(config,indent=2)+'\n')
                env['TMPDIR']=str(directory)
                rc,command=execute([str(binary),str(directory/'config.json')],env,directory,'rust',120)
                assert rc==0
                records=[json.loads(s.removeprefix('IQA_PROBE ')) for s in (directory/'rust-stdout.log').read_text().splitlines() if s.startswith('IQA_PROBE ')]
                assert len(records)==1 and records[0]['metric_name']==metric and len(records[0]['scores'])==1
                ours=struct.unpack('>d',bytes.fromhex(records[0]['scores'][0][1]))[0]
                row=dict(metric=metric,pair=item['pair'],size=item['size_name'],threads=threads,ours=ours,canonical=canonical,
                    abs_delta=abs(ours-canonical),bound='UNBOUND',status='observation',qualified_ratio=False,
                    command=command,files={p.name:sha(p) for p in directory.iterdir() if p.is_file()})
                (directory/'results.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
            print('PSNR_GROUP',item['pair'],item['size_name'],threads,flush=True)
        assert sha(binary)==identity['binary_sha256'] and sha(Path(__file__))==identity['worker_sha256']
    (root/'completion.json').write_text(json.dumps(dict(cases=rows,qualified_timing_cells=0),indent=2)+'\n')


if __name__=='__main__':main()
