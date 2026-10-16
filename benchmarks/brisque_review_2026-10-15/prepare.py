"""Pinned piq/torch oracle captures; no pyiqa. All generated data stays external."""
import pathlib,sys,json,hashlib,importlib,platform,os
import numpy as np,cv2,torch,piq
from piq.functional import gaussian_filter
s=pathlib.Path.home()/'tmp/aic-variants-scratch/round6';s.mkdir(exist_ok=True)
root=pathlib.Path(os.environ.get('ZENMETRICS_NR_GOLDEN_ROOT',str(pathlib.Path.home()/'tmp')))
torch.set_num_threads(1);torch.set_grad_enabled(False)
m=importlib.import_module('piq.brisque');original=m._natural_scene_statistics;capture=[]
def traced(*a,**k):
 v=original(*a,**k);capture.append(v);return v
m._natural_scene_statistics=traced
meta=dict(piq=piq.__version__,torch=torch.__version__,glibc=platform.libc_ver(),cpu=torch.backends.cpu.get_cpu_capability(),torch_config=torch.__config__.show(),environment={k:os.environ.get(k) for k in ['MKL_CBWR','MKL_ENABLE_INSTRUCTIONS','ATEN_CPU_CAPABILITY']},threads=1)
(s/'oracle_environment.json').write_text(json.dumps(meta,indent=2))
gold=[]
for line in pathlib.Path('crates/brisque/tests/goldens_brisque.tsv').read_text().splitlines():
 if line and not line.startswith('#'):gold.append(str(root/line.split('\t')[0]))
assert len(gold)==170
lines=(root/'konjnd-pairs/pairs_abs.tsv').read_text().splitlines();keys=lines[0].split('\t');pairs=[dict(zip(keys,x.split('\t'))) for x in lines[1:]]
by_codec=[[p for p in pairs if p['codec']==codec] for codec in ['JPEG','BPG']]
extra=[];seen=set(gold)
for a,b in zip(*by_codec):
 for p in [a,b]:
  for k in ['ref_path','dist_path']:
   path=p[k]
   if path not in seen:extra.append(path);seen.add(path)
   if len(extra)==200:break
  if len(extra)==200:break
 if len(extra)==200:break
assert len(extra)==200 and not set(extra)&set(gold)
paths=gold+extra
(s/'panel.paths').write_text('\n'.join(paths)+'\n')
rows=[]
for i,path in enumerate(paths):
 capture.clear();bgr=cv2.imread(path);assert bgr is not None,path
 rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);x=torch.from_numpy(rgb.astype(np.float32)).permute(2,0,1).unsqueeze(0)
 row=dict(path=path,set='golden' if i<170 else 'extra',sha256=hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest())
 try:
  score=piq.brisque(x,data_range=255.,reduction='none').item();features=torch.cat(capture,dim=-1).flatten().numpy()
  row.update(score=score,features_bits=features.view(np.uint32).tolist())
 except (AssertionError,RuntimeError,ValueError) as e:row.update(error=str(e))
 rows.append(row)
 if i%10==0:print('oracle panel',i+1,'/',len(paths),flush=True)
with (s/'oracle_panel.json').open('x') as f:json.dump(rows,f)
# Broad ratios and gamma-boundary-enriched cases, deterministic and finite.
n=400000;rng=np.random.default_rng(20261015)
gam=torch.arange(.2,10.001,.001,dtype=torch.float32)
ggd=torch.exp(torch.lgamma(1/gam)+torch.lgamma(3/gam)-2*torch.lgamma(2/gam))
aggd=torch.exp(2*torch.lgamma(2/gam)-torch.lgamma(1/gam)-torch.lgamma(3/gam))
data=(10**rng.uniform(-3,3,size=(n,6))).astype(np.float32)
data[:,4]=np.sqrt(data[:,5])*rng.uniform(.15,.95,n).astype(np.float32)
indices=rng.integers(0,len(gam)-1,n//2)
target=(aggd.numpy()[indices].astype(np.float64)+aggd.numpy()[indices+1])/2
gh=data[n//2:,2].astype(np.float64)/data[n//2:,3]
correction=((gh**3+1)*(gh+1))/(gh**2+1)**2
data[n//2:,4]=np.sqrt(target*data[n//2:,5]/correction).astype(np.float32)
targetg=(ggd.numpy()[indices].astype(np.float64)+ggd.numpy()[indices+1])/2
data[n//2:,1]=np.sqrt(data[n//2:,0]/targetg).astype(np.float32)
data.astype('<f4').tofile(s/'ratios-input.bin');a=torch.from_numpy(data)
g=a[:,2]/a[:,3];ro=a[:,4].pow(2)/a[:,5]
ratio=(ro*(g.pow(3)+1)*(g+1))/(g.pow(2)+1).pow(2)
gratio=a[:,0]/a[:,1].pow(2)
assert torch.equal(g.pow(3),g*g*g) and torch.equal(a[:,1].pow(2),a[:,1]*a[:,1])
output=np.empty(n,dtype=[('ggd','<f4'),('gi','<u4'),('aggd','<f4'),('ai','<u4')]);output['ggd']=gratio.numpy();output['aggd']=ratio.numpy()
for start in range(0,n,512):
 stop=min(n,start+512)
 output['gi'][start:stop]=(gratio[start:stop,None]-ggd[None,:]).abs().argmin(dim=1).numpy()
 output['ai'][start:stop]=(ratio[start:stop,None]-aggd[None,:]).abs().argmin(dim=1).numpy()
output.tofile(s/'ratios-oracle.bin')
np.savez(s/'oracle_grids.npz',gam=gam.numpy(),r_ggd=ggd.numpy(),r_aggd=aggd.numpy())
print('oracle ratios',n,'pow-vs-multiplication mismatches=0',flush=True)
