import argparse,json,os,pathlib,subprocess,sys
p=argparse.ArgumentParser(description='Compare raw DSSIM maps from explicitly identified native and reference libraries.')
for name in ['native-target','reference-target','reference-source','inputs','output']:p.add_argument('--'+name,type=pathlib.Path,required=True)
p.add_argument('--native-revision',required=True);p.add_argument('--reference-revision',required=True);p.add_argument('--reference-origin',required=True);p.add_argument('--case')
a=p.parse_args();here=pathlib.Path(__file__).resolve().parent
from compare_canonical import sha
from haarpsi_primary import execute
import numpy as np
for name in ['native_target','reference_target','reference_source','inputs']:setattr(a,name,getattr(a,name).resolve(strict=True))
a.output=a.output.absolute();a.output.mkdir();source=here/'dssim_map_dump.rs'
reference_files=[path for path in a.reference_source.rglob('*') if path.is_file() and (path.suffix=='.rs' or path.name in ['Cargo.toml','Cargo.lock'])]
source_hashes={str(path):sha(path) for path in reference_files}
receipt={'source_sha256':sha(source),'libraries':{},'commands':{},'binaries':{},'native_revision':a.native_revision,'reference_revision':a.reference_revision,'reference_origin':a.reference_origin,'reference_source_files':source_hashes}
for role,target in [('canonical',a.reference_target),('native',a.native_target)]:
 deps=target/'release/deps';libraries={}
 for name in ['dssim_core','rgb','imgref','rayon']:
  found=list(deps.glob('lib'+name+'-*.rlib'));assert len(found)==1,(role,name,found);libraries[name]=found[0]
 binary=a.output/(role+'-map');command=['rustc','--edition=2024','-C','opt-level=3',*(['-C','panic=abort'] if role=='canonical' else []),'-L','dependency='+str(deps),*[item for name,path in libraries.items() for item in ['--extern',name+'='+str(path)]],str(source),'-o',str(binary)]
 subprocess.run(command,check=True);receipt['libraries'][role]={str(p):sha(p) for p in libraries.values()};receipt['commands'][role]=command;receipt['binaries'][role]=sha(binary)
(a.output/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
inputs=a.inputs;rows=[]
for pair in json.loads(inputs.read_text())['pairs']:
 w,h=pair['width'],pair['height'];paths={role:(inputs.parent/pair[role]).resolve() for role in ['reference','distorted']}
 assert all(sha(path)==pair[role+'_sha256'] for role,path in paths.items())
 for threads in [1,28]:
  root=a.output/f'{pair["pair"]}-{pair["size_name"]}-t{threads}'
  if a.case and root.name!=a.case:continue
  root.mkdir();observed={};errors={}
  env=dict(os.environ,TMPDIR=str(root),RAYON_NUM_THREADS=str(threads),OMP_NUM_THREADS=str(threads))
  for role in ['canonical','native']:
   command=[str(a.output/(role+'-map')),str(w),str(h),str(paths['reference']),str(paths['distorted']),str(threads),str(root/role),role]
   rc,_=execute(command,env,root,role,300)
   if rc:errors[role]='exit '+str(rc);continue
   lines=(root/(role+'-stdout.log')).read_text().splitlines();head=lines[0].split();assert head[0]=='SCORE';maps=[]
   for line in lines[1:]:
    _,index,mw,mh,mean=line.split();file=root/role/f'scale-{index}.f32';values=np.fromfile(file,dtype='<f4');assert len(values)==int(mw)*int(mh)
    maps.append(dict(index=int(index),width=int(mw),height=int(mh),mean_ssim=float(mean),sha256=sha(file),values=values))
   observed[role]=dict(score=float(head[1]),plain_score=float(head[head.index('PLAIN_SCORE')+1]),map_bits=head[head.index('MAP_BITS')+1],plain_bits=head[head.index('PLAIN_BITS')+1],maps=maps)
  value=dict(pair=pair['pair'],size=pair['size_name'],width=w,height=h,threads=threads,errors=errors,phase='accuracy-only',map_bound=None)
  if len(observed)==2:
   n,c=observed['native'],observed['canonical'];assert len(n['maps'])==len(c['maps']);maps=[]
   order=lambda item:(item['width'],item['height'])
   assert len({order(item) for item in n['maps']})==len(n['maps'])
   assert len({order(item) for item in c['maps']})==len(c['maps'])
   for nm,cm in zip(sorted(n['maps'],key=order,reverse=True),sorted(c['maps'],key=order,reverse=True)):
    assert order(nm)==order(cm)
    nv,cv=nm.pop('values'),cm.pop('values');finite=np.isfinite(nv)&np.isfinite(cv);same=(np.isnan(nv)&np.isnan(cv))|((nv==cv)&~finite)
    delta=np.abs(nv[finite].astype(np.float64)-cv[finite]);maps.append(dict(native=nm,canonical=cm,max_abs_delta=float(delta.max()) if len(delta) else None,bit_identical=nm['sha256']==cm['sha256'],nonfinite_mismatches=int((~(finite|same)).sum())))
   value.update(native_score=n['score'],canonical_score=c['score'],scalar_abs_delta=abs(n['score']-c['score']),maps=maps,native_plain_score=n['plain_score'],canonical_plain_score=c['plain_score'],native_map_scalar_bits_match=n['map_bits']==n['plain_bits'],canonical_map_scalar_bits_match=c['map_bits']==c['plain_bits'])
  (root/'results.json').write_text(json.dumps(value,indent=2)+'\n');rows.append(value);print('DSSIM_MAP_CASE',root.name,value.get('scalar_abs_delta'),[m['bit_identical'] for m in value.get('maps',[])],flush=True)
assert rows,'explicit case selection did not match the declared input grid'
assert all(sha(path)==digest for path,digest in source_hashes.items()),'reference source changed during acquisition'
(a.output/'completion.json').write_text(json.dumps(dict(cases=rows,build_receipt_sha256=sha(a.output/'build-receipt.json'),inputs_sha256=sha(inputs),qualification='Raw scalar and all five saved SSIM maps, matched by unique geometry because the original parallel callable returns maps in completion order; original returned indices retained; no timing claim and no new map bound'),indent=2)+'\n')
