import argparse,json,os,pathlib,shutil,subprocess,sys
p=argparse.ArgumentParser(description='Compare FLIP raw maps and both scalar mean conventions.')
p.add_argument('build',type=pathlib.Path);p.add_argument('inputs',type=pathlib.Path);p.add_argument('output',type=pathlib.Path)
p.add_argument('--case');a=p.parse_args();here=pathlib.Path(__file__).resolve().parent
from compare_canonical import sha
from haarpsi_primary import execute
import numpy as np
build=a.build.resolve(strict=True);a.output=a.output.absolute();receipt=json.loads((build/'build-receipt.json').read_text())
a.output.mkdir();source=build/'work/zenmetrics/crates/zenmetrics-cli/src/metrics/flip.rs';original=subprocess.check_output(['git','show',receipt['source_revision']+':crates/zenmetrics-cli/src/metrics/flip.rs']);assert original==source.read_bytes()
shutil.copy2(source,a.output/'frozen-flip.rs');shutil.copy2(here/'flip_map_dump.rs',a.output/'flip_map_dump.rs');deps=build/'target/release/deps';libs={}
fingerprints=build/'target/release/.fingerprint'
math_fingerprint=list(fingerprints.glob('zenmetrics-math-*/lib-zenmetrics_math.json'));assert len(math_fingerprint)==1
required={row[1]:row[-1] for row in json.loads(math_fingerprint[0].read_text())['deps'] if row[1] in ['archmage','magetypes']}
for name in ['archmage','magetypes','zenmetrics_math','rayon']:
 found=list(deps.glob('lib'+name+'-*.rlib'))
 if name in required:
  found=[file for file in found if int.from_bytes(bytes.fromhex((fingerprints/file.stem.removeprefix('lib')/('lib-'+name)).read_text().strip()),'little')==required[name]]
 assert len(found)==1,(name,found);libs[name]=found[0]
binary=a.output/'native-map';cmd=['rustc','--edition=2024','-C','opt-level=3','--cfg','feature="parallel"','--cfg','feature="avx512"','-L','dependency='+str(deps),*[x for n,path in libs.items() for x in ['--extern',n+'='+str(path)]],str(a.output/'flip_map_dump.rs'),'-o',str(binary)];subprocess.run(cmd,check=True)
metadata=dict(source_revision=receipt['source_revision'],source_sha256=sha(source),native_probe_sha256=sha(here/'flip_map_dump.rs'),canonical_probe_sha256=sha(here/'flip_reference_map.py'),libraries={str(p):sha(p) for p in libs.values()},command=cmd,binary_sha256=sha(binary),parent_build_receipt_sha256=sha(build/'build-receipt.json'))
(a.output/'build-receipt.json').write_text(json.dumps(metadata,indent=2)+'\n');inputs=a.inputs.resolve(strict=True);rows=[]
for pair in json.loads(inputs.read_text())['pairs']:
 w,h=pair['width'],pair['height'];paths={role:(inputs.parent/pair[role]).resolve() for role in ['reference','distorted']};assert all(sha(p)==pair[role+'_sha256'] for role,p in paths.items())
 for threads in [1,28]:
  root=a.output/f'{pair["pair"]}-{pair["size_name"]}-t{threads}'
  if a.case and root.name!=a.case:continue
  root.mkdir();env=dict(os.environ,TMPDIR=str(root),RAYON_NUM_THREADS=str(threads),OMP_NUM_THREADS=str(threads));observed={};errors={}
  for role,prefix in [('native',[str(binary)]),('canonical',[sys.executable,str(here/'flip_reference_map.py')])]:
   command=[*prefix,str(w),str(h),str(paths['reference']),str(paths['distorted']),str(threads),str(root/(role+'.f32'))];rc,_=execute(command,env,root,role,300)
   if rc:errors[role]='exit '+str(rc);continue
   line=(root/(role+'-stdout.log')).read_text().splitlines()[-1].split();assert line[0]=='FLIP';observed[role]=dict(flip=float(line[1]),flip_v17=float(line[3]))
  result=dict(pair=pair['pair'],size=pair['size_name'],width=w,height=h,threads=threads,errors=errors,observed=observed,map_bound=None,qualification='Accuracy only; map means preserve f64 and row-major f32 conventions; no timing claim')
  if len(observed)==2:
   native,canonical=[np.fromfile(root/(role+'.f32'),'<f4') for role in ['native','canonical']];assert native.size==canonical.size==w*h
   finite=np.isfinite(native)&np.isfinite(canonical);matching=(np.isnan(native)&np.isnan(canonical))|((native==canonical)&~finite);delta=np.abs(native[finite].astype(np.float64)-canonical[finite]);result.update(max_map_abs_delta=float(delta.max()),mean_map_abs_delta=float(delta.mean()),nonfinite_mismatches=int((~(finite|matching)).sum()),native_map_sha256=sha(root/'native.f32'),canonical_map_sha256=sha(root/'canonical.f32'))
  (root/'results.json').write_text(json.dumps(result,indent=2)+'\n');rows.append(result);print('FLIP_MAP_CASE',root.name,result.get('max_map_abs_delta'),result.get('nonfinite_mismatches'),flush=True)
assert rows,'explicit selection did not match input grid'
(a.output/'completion.json').write_text(json.dumps(dict(cases=rows,build_receipt_sha256=sha(a.output/'build-receipt.json'),inputs_sha256=sha(inputs)),indent=2)+'\n')
