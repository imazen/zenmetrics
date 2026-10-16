"""Record stage-vs-oracle residuals without changing the JND/1000 gate."""
import pathlib,json,sys,numpy as np
s=pathlib.Path.home()/'tmp/aic-variants-scratch/round6';label=sys.argv[1];jnd=15.138428
rows=json.load(open(s/'oracle_panel.json'));own=[x.split('\t') for x in (s/f'{label}-panel.tsv').read_text().splitlines()];assert len(own)==len(rows)==370
summary={'stage':label,'JND':jnd,'gate':jnd/1000,'panels':{}}
alpha=[0,2,6,10,14,18,20,24,28,32]
for dataset in ['golden','extra']:
 ds=[];flips=0;flip_images=0;errors=0
 for ref,native in zip(rows,own):
  assert ref['path']==native[0]
  if ref['set']!=dataset:continue
  if 'error' in ref:
   assert native[1]=='NaN';errors+=1;continue
  assert native[1]!='NaN',native[0]
  bits=[int(x,16) for x in native[2].split(',') if x];assert len(bits)==36
  f=sum(bits[i]!=ref['features_bits'][i] for i in alpha);flips+=f;flip_images+=f>0
  ds.append((abs(float(native[1])-ref['score']),native[0]))
 maximum,worst=max(ds);summary['panels'][dataset]=dict(rows=len(ds),matched_errors=errors,argmin_feature_flips=flips,argmin_flip_images=flip_images,max_abs=maximum,max_jnd=maximum/jnd,median_abs=float(np.median([d for d,p in ds])),worst=worst,within_gate=all(d<=jnd/1000 for d,p in ds))
dtype=[('ggd','<f4'),('gi','<u4'),('aggd','<f4'),('ai','<u4')]
a=np.fromfile(s/'ratios-oracle.bin',dtype=dtype);b=np.fromfile(s/f'{label}-ratios.bin',dtype=dtype);assert len(a)==len(b)==400000
summary['synthetic']={'rows':len(a)}
for v,i in [('ggd','gi'),('aggd','ai')]:
 summary['synthetic'][v]=dict(ratio_bit_mismatches=int(np.sum(a[v].view('u4')!=b[v].view('u4'))),argmin_flips=int(np.sum(a[i]!=b[i])),max_ratio_abs=float(np.max(abs(a[v]-b[v]))))
if label!='baseline':
 old=[x.split('\t') for x in (s/'baseline-panel.tsv').read_text().splitlines()];delta=max(abs(float(a[1])-float(b[1])) for a,b in zip(old,own) if a[1]!='NaN' and b[1]!='NaN');summary['change_from_baseline']=dict(max_abs=delta,max_jnd=delta/jnd)
with (s/f'{label}-summary.json').open('x') as f:json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))
if '--gate' in sys.argv:
 assert all(p['within_gate'] for p in summary['panels'].values()),'published-unit/JND gate failed'
