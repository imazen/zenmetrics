"""Stagewise release/native/SciPy controls, with no score fitting or tolerance edits."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import scipy
from scipy.interpolate import PchipInterpolator, CubicSpline

v221,v222,v307,probe,out=map(Path,sys.argv[1:]); assert not out.exists()
out.mkdir(parents=True); here=Path(__file__).parent
for name in ['spectral_dump.m','cubic_convolution_control.m','spectral_control.py']:
    shutil.copy2(here/name,out/name)
commands=[]
def run(args):
    commands.append(list(map(str,args))); subprocess.run(commands[-1],check=True)
for label,tree,version,mode in [('v221',v221,2,'release'),('v222',v222,2,'release'),
                               ('v307',v307,3,'release'),('modern-model',v222,2,'modern')]:
    run(['octave','--quiet',out/'spectral_dump.m',tree,out/(label+'.tsv'),version,mode])
for ver,tree,base in [(2,v222,'v222'),(3,v307/'data','v307')]:
    run([probe,tree,out/f'native{ver}.tsv',out/(base+'.tsv'),ver])
def load(path):
    records=list(csv.DictReader(path.open(),delimiter='\t')); data={}
    keys=set()
    for r in records:
        key=(r['stage'],int(r['row']),int(r['column'])); assert key not in keys; keys.add(key)
        data.setdefault(r['stage'],{})[(key[1],key[2])]=float(r['value'])
    arrays={}
    for stage,items in data.items():
        nr=max(k[0] for k in items)+1;nc=max(k[1] for k in items)+1
        assert len(items)==nr*nc
        a=np.array([[items[r,c] for c in range(nc)] for r in range(nr)])
        assert np.isfinite(a).all(); arrays[stage]=a
    return arrays
reports=[]
def compare(name,a,b):
    assert a.shape==b.shape
    d=np.abs(a-b); index=np.unravel_index(np.argmax(d),d.shape)
    reports.append(dict(comparison=name,n=a.size,exact=int(np.count_nonzero(a==b)),
                        median_abs=float(np.median(d)),max_abs=float(d[index]),
                        max_row=int(index[0]),max_column=int(index[1]),
                        value_a=float(a[index]),value_b=float(b[index])))
releases={k:load(out/(k+'.tsv')) for k in ['v221','v222','v307','modern-model']}
for stage,a in releases['v222'].items():
    compare('221_cubic_vs_222_pchip/'+stage,releases['v221'][stage],a)
    compare('modern_model_vs_222_pchip/'+stage,releases['modern-model'][stage],a)
table_names=['log_cone_smith_pokorny_1975.csv','cie_scotopic_lum.txt']
table_properties=[]
for ver,root,base,display in [(2,v222,'v222','ccfl-lcd'),(3,v307/'data','v307','oled')]:
    native=load(out/f'native{ver}.tsv'); release=releases[base]; q=release['grid'][:,0]
    for stage in release:
        compare(f'native{ver}_vs_{base}/'+stage,native[stage],release[stage])
    for filename,label in zip(table_names+[f'emission_spectra_{display}.csv'],['cone_log','rod','display']):
        p=root/filename;d=np.loadtxt(p,delimiter=',');x,y=d[:,0],d[:,1:]
        uniform=bool(np.allclose(np.diff(x),np.diff(x)[0],rtol=1e-10,atol=0))
        table_properties.append(dict(version=ver,name=filename,n=len(x),columns=y.shape[1],
                                     first=float(x[0]),last=float(x[-1]),uniform=uniform,
                                     sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        pc=PchipInterpolator(x,y,axis=0)(q); pc[(q<x[0])|(q>x[-1])]=0
        compare(f'scipy_pchip_vs_{base}/raw_'+label,pc,release['raw_'+label])
        # Same queries isolate PCHIP arithmetic from linspace rounding.
        pn=PchipInterpolator(x,y,axis=0)(q)
        if ver==2: pn[(q<x[0])|(q>x[-1])]=0
        compare(f'native{ver}_matched_grid_vs_scipy/raw_'+label,native['matched_raw_'+label],pn)
        if ver==2:
            if uniform:
                h=np.diff(x)[0]
                slopes=np.empty_like(y);slopes[1:-1]=(y[2:]-y[:-2])/(2*h)
                slopes[0]=(-3*y[0]+4*y[1]-y[2])/(2*h)
                slopes[-1]=(3*y[-1]-4*y[-2]+y[-3])/(2*h)
                # Independent polynomial-coefficient evaluation of Keys Hermite segments.
                i=np.clip(np.searchsorted(x,q,side='right')-1,0,len(x)-2)
                t=(q-x[i])[:,None]
                delta=(y[i+1]-y[i])/h
                c=(3*delta-2*slopes[i]-slopes[i+1])/h
                b=(slopes[i]+slopes[i+1]-2*delta)/(h*h)
                modern=y[i]+t*(slopes[i]+t*(c+t*b))
            else:
                modern=CubicSpline(x,y,axis=0,bc_type='not-a-knot')(q)
            modern[(q<x[0])|(q>x[-1])]=0
            compare('independent_modern_vs_octave_model/raw_'+label,modern,releases['modern-model']['raw_'+label])
    if ver==3:
        # Extrapolated raw receptor tails are discarded by native v3 before use.
        for label in ['cone_log','rod']:
            a=native['raw_'+label].copy();a[:20]=0
            compare('native3_discarded_tail_then_vs_v307/raw_'+label,a,release['raw_'+label])
fields=list(reports[0])
with (out/'comparisons.tsv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(reports)
upstream_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [v221/'load_spectral_resp.m',v222/'load_spectral_resp.m',v307/'utils/load_spectral_resp.m',v307/'hdrvdp_visual_pathway.m',v307/'utils/hdrvdp_iturgb2native.m']}
meta=dict(scipy=scipy.__version__,numpy=np.__version__,commands=commands,tables=table_properties,upstream_files=upstream_files,
          probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),
          files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},
          native_sources_sha256=hashlib.sha256((probe.parent/'source-sha256.json').read_bytes()).hexdigest(),
          qualification='stage differences only; no published-score gate changed; modern model not MATLAB runtime qualified')
(out/'receipt.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(dict(n_comparisons=len(reports),tables=table_properties),indent=2),flush=True)
