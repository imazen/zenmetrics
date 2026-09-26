#!/usr/bin/env python3
"""Independent NumPy equation oracle for the preregistered MS-GMSD variant.

This does not claim author-software parity. Oracle-only: never a model input.
Inputs are the frozen raw-RGB8 oracle TSV, not labels or evaluation corpora.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path('/var/tmp/gmsd-chroma')


def main():
    import numpy as np
    assert not Path('/home/lilith/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md').exists()
    out = ROOT/'ms_numpy'
    out.mkdir(exist_ok=False)
    matrix = np.array([[.299,.587,.114], [.595716,-.274453,-.321263],
                       [.211456,-.522591,.311135]])

    def gradient(y):
        z = np.pad(y, 1)
        a,b,c = z[:-2,:-2],z[1:-1,:-2],z[2:,:-2]
        d,f = z[:-2,1:-1],z[2:,1:-1]
        g,h,i = z[:-2,2:],z[1:-1,2:],z[2:,2:]
        # Direct convolution of the normalized Prewitt stencils.
        gx = (-a-b-c+g+h+i)/3
        gy = (-a+c-d+f-g+i)/3
        return np.hypot(gx,gy)

    def half(a):
        h,w,_ = a.shape
        z = np.pad(a,((0,h%2),(0,w%2),(0,0)),mode='edge')
        return (z[::2,::2]+z[::2,1::2]+z[1::2,::2]+z[1::2,1::2])/4

    records = []
    with (ROOT/'oracle_inputs.tsv').open() as f:
        rows = list(csv.DictReader(f,delimiter='\t'))
    for row in rows:
        w,h = int(row['width']),int(row['height'])
        r,d = [np.fromfile(row[side+'_rgb'],dtype=np.uint8).reshape(h,w,3).astype(float)@matrix.T
               for side in ['ref','dist']]
        variances = []
        for scale in range(4):
            gr,gd = gradient(r[:,:,0]),gradient(d[:,:,0])
            q = (2*gr*gd-.5*gr*gd+170)/(gr*gr+gd*gd-.5*gr*gd+170)
            q.astype('<f8').tofile(out/f'{row["id"]}.s{scale}.f64')
            variances.append(float(np.mean((q-np.mean(q))**2)))
            if scale!=3:
                r,d = half(r),half(d)
        ms = float(np.sqrt(np.dot([.096,.596,.289,.019],variances)))
        chroma = float(np.sqrt(np.mean(np.sum((r[:,:,1:]-d[:,:,1:])**2,axis=2))))
        gamma = 2/(1+.32*np.exp(-15*ms))-1
        records.append(dict(id=row['id'],ms_gmsd=ms,ms_gmsdc=float(gamma*ms+(1-gamma)*.01*chroma)))
    with (out/'scores.tsv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['id','ms_gmsd','ms_gmsdc'],delimiter='\t',lineterminator='\n')
        writer.writeheader();writer.writerows(records)
    print(json.dumps(dict(pairs=len(records),numpy_version=np.__version__,
                         scores_sha256=hashlib.sha256((out/'scores.tsv').read_bytes()).hexdigest()),sort_keys=True))


if __name__ == '__main__':
    main()
