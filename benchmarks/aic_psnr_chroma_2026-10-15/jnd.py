import csv, json, numpy as np
from PIL import Image
exec(open('probe.py').read().split('variants = {')[0])
M=json.load(open('/home/lilith/tmp/aic-site/config.json'))['mappings']
def jnd(col,x): m=M[col]; return m['a']*max(0.0,m['b']-x)**m['c']
out={'PSNR-Cb':[],'PSNR-Cr':[]}
for p in pairs:
    R,D=load(p['ref_path']),load(p['dist_path']); row=pub[p['distorted']]
    a=[rnd(c) for c in yuv(R)]; b=[rnd(c) for c in yuv(D)]
    for i,col in ((1,'PSNR-Cb'),(2,'PSNR-Cr')):
        v=psnr(a[i],b[i]); out[col].append(abs(jnd(col,v)-jnd(col,float(row[col]))))
for k,v in out.items(): print(k, f"JND |d| median {np.median(v):.2e} max {np.max(v):.2e}")
