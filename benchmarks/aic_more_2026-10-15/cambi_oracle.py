"""Pinned libvmaf 3.2.1 CAMBI default and v1-setting control, AIC + ladder."""
import csv,json,pathlib,subprocess,statistics
import numpy as np
from PIL import Image
home=pathlib.Path.home();s=home/'tmp/aic-variants-scratch/round4';binary=s/'oracles/vmaf-3.2.1/libvmaf/build-aic/tools/vmaf';cfg=json.load(open(home/'tmp/aic-site/config.json'))['mappings']['CAMBI'];pub={r['distorted']:r for r in csv.DictReader(open(home/'tmp/aic-site/site_fullres_subset61.csv'))}
def jnd(x):return cfg['a']*max(0,x-cfg['b'])**cfg['c']
for dataset in ['aic','ladder']:
 pairs=list(csv.DictReader(open(home/'tmp/aic-audit/pairs.tsv'),delimiter='\t')) if dataset=='aic' else [dict(distorted=p.stem,ref_path=str(p),dist_path=str(p).replace('_ref.png','_dist.png')) for p in sorted((home/'tmp/nn-phase0/ladder').glob('*_ref.png'))]
 rows=[]
 for pair in pairs:
  a=np.asarray(Image.open(pair['dist_path']).convert('RGB'),dtype=np.float64);h,w=a.shape[:2];h-=h%2;w-=w%2;a=a[:h,:w];y=np.floor(16+(.0+65.481*a[:,:,0]+128.553*a[:,:,1]+24.966*a[:,:,2])/255+.5).astype(np.uint8)
  with (s/'cambi.yuv').open('wb') as f:f.write(y.tobytes());f.write(bytes([128])*(w*h//2))
  row={'key':pair['distorted']}
  for name,feature in [('default','cambi'),('v1','cambi=cambi_vis_lum_threshold=0.06:cambi_max_val=17:cambi_high_res_speedup=1080'),('uncapped_v1','cambi=cambi_vis_lum_threshold=0.06:cambi_max_val=1000:cambi_high_res_speedup=1080')]:
   target=s/'cambi-oracle-output.json'
   # Disposable per-call transport file; retained results are written exclusively below.
   result=subprocess.run([str(binary),'-r',str(s/'cambi.yuv'),'-d',str(s/'cambi.yuv'),'-w',str(w),'-h',str(h),'-p','420','-b','8','--feature',feature,'--no_prediction','--json','-o',str(target)],capture_output=True,text=True)
   if result.returncode:row[name+'_error']=result.stderr.strip();continue
   values=json.load(target.open())['frames'][0]['metrics'];val=next(v for k,v in values.items() if k.startswith('cambi'))
   row[name]=val
   if dataset=='aic':targetval=float(pub[pair['distorted']]['CAMBI']);row['published']=targetval;row[name+'_published_error']=abs(val-targetval);row[name+'_published_jnd_error']=abs(jnd(val)-jnd(targetval))
  rows.append(row);print(dataset,json.dumps(row),flush=True)
 with (s/f'cambi-precise-settings-{dataset}.json').open('x') as f:json.dump(rows,f,indent=2)
 if dataset=='aic':print('SUMMARY',json.dumps({k:{'median':statistics.median(r[k] for r in rows if k in r),'max':max(r[k] for r in rows if k in r)} for k in rows[0] if k.endswith('_error')}),flush=True)
