#!/usr/bin/env python3
"""Original VIFvec with a locally compiled original matlabPyrTools corrDn MEX.

An alias binds the original uppercase helper file to the lowercase caller;
the M-file bytes and every numerical expression remain unchanged. Score
parity uses the existing 1e-9 absolute gate. The controller supplies both
heavy locks and a per-child cap, releasing them between calls.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess

from contained_primary import execute
from octave_primary import sha, quoted


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','build','inputs','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    for name in ['source','build','inputs']:setattr(a,name,getattr(a,name).resolve(strict=True))
    sources=json.loads((a.source/'source.json').read_text())
    assert {r['name']:r['rev'] for r in sources}=={'image-quality-tools':'163fff128a86260efcc962de52196c71a3d3d06e','matlabPyrTools':'f02ab059d0f220fdee840c5a06769104c3e73d37'}
    original=json.loads((a.build/'build-receipt.json').read_text())
    binary=a.build/'target/release/iqa-compete'
    assert sha(binary)==original['binary_sha256'] and original['source_revision']=='e2d414c33a6d3c7da50a818fa36e56654d3ebeef'
    a.output.mkdir();root=a.output.resolve()
    author=a.source/'image-quality-tools/metrix_mux/metrix/vif/vifvec_release'
    pyr=a.source/'matlabPyrTools';mex=pyr/'MEX'
    alias=root/'oracle-bindings';alias.mkdir()
    # Octave declares its MATLAB matrix API in mex.h rather than matrix.h.
    # Supply only the header-name binding; original MEX bodies stay unchanged.
    (alias/'matrix.h').write_text('#pragma once\n#include <mex.h>\n')
    shutil.copyfile(author/'vifsub_est_M.m',alias/'vifsub_est_m.m')
    assert sha(author/'vifsub_est_M.m')==sha(alias/'vifsub_est_m.m')
    dev=a.source/'octave-dev';compiler=dev/'usr/bin/mkoctfile'
    assert compiler.is_file()
    env=dict(os.environ,INCFLAGS=f'-I{dev}/usr/include/octave-6.4.0 -I{dev}/usr/include/octave-6.4.0/octave')
    for key in ['CFLAGS','CXXFLAGS','FFLAGS']:env.pop(key,None)
    command=[str(compiler),'--mex','-O3','-I'+str(alias),*[str(mex/name) for name in ['corrDn.c','convolve.c','wrap.c','edges.c']],'-o',str(alias/'corrDn.mex')]
    print('MEX_BUILD',json.dumps(command),flush=True)
    rc,_=execute(command,env,root,'mex-build',300)
    assert rc==0, 'MEX build failed; see mex-build-stderr.log'
    bound_files={str(p):sha(p) for directory in [author,pyr] for p in directory.rglob('*') if p.is_file() and p.suffix in ['.m','.c','.h']}
    identity=dict(sources=sources,source_receipt_sha256=sha(a.source/'source.json'),source_files=bound_files,
        alias_sha256=sha(alias/'vifsub_est_m.m'),matrix_header_sha256=sha(alias/'matrix.h'),mex_sha256=sha(alias/'corrDn.mex'),build_command=command,
        compiler_sha256=sha(compiler),compiler_flags=subprocess.check_output([str(compiler),'-p','ALL_CFLAGS'],env=env,text=True),
        octave_dev_deb_sha256=sha(a.source/'octave-dev_6.4.0-2_amd64.deb'),binary_sha256=sha(binary),
        build_receipt_sha256=sha(a.build/'build-receipt.json'),inputs_sha256=sha(a.inputs),worker_sha256=sha(Path(__file__)),
        gate=1e-9,gate_source='crates/vif/src/tests.rs:vifvec_plane_goldens and rgb_goldens',qualified_timing_cells=0)
    (root/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    rows=[]
    for item in json.loads(a.inputs.read_text())['pairs']:
        paths={r:(a.inputs.parent/item[r]).resolve(strict=True) for r in ['reference','distorted']}
        assert all(sha(path)==item[r+'_sha256'] for r,path in paths.items())
        w,h=item['width'],item['height']
        for threads in [1,max(1,len(os.sched_getaffinity(0))-4)]:
            directory=root/f"vifvec-{item['pair']}-{item['size_name']}-t{threads}";directory.mkdir()
            env=dict(os.environ,LC_ALL='C',TMPDIR=str(directory))
            for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS']:env[key]=str(threads)
            for key in ['DISPLAY','WAYLAND_DISPLAY','XAUTHORITY']:env.pop(key,None)
            entry=directory/'entry.m'
            entry.write_text(f"""more off;
addpath({quoted(author)},'-begin');addpath({quoted(pyr)},'-begin');addpath({quoted(alias)},'-begin');
assert(strcmp(which('vifvec'),{quoted(author/'vifvec.m')}));assert(strcmp(which('corrDn'),{quoted(alias/'corrDn.mex')}));
assert(strcmp(which('vifsub_est_m'),{quoted(alias/'vifsub_est_m.m')}));
for name={{'vifvec','buildSpyr','corrDn','vifsub_est_m','refparams_vecgsm','rgb2gray'}};fprintf('BIND %s %s\\n',name{{1}},which(name{{1}}));end
f=fopen({quoted(paths['reference'])},'rb');r=fread(f,Inf,'*uint8');assert(fclose(f)==0);
f=fopen({quoted(paths['distorted'])},'rb');d=fread(f,Inf,'*uint8');assert(fclose(f)==0);
r=double(permute(reshape(r,[3,{w},{h}]),[3,2,1]));d=double(permute(reshape(d,[3,{w},{h}]),[3,2,1]));
r=round(0.299*r(:,:,1)+0.587*r(:,:,2)+0.114*r(:,:,3));d=round(0.299*d(:,:,1)+0.587*d(:,:,2)+0.114*d(:,:,3));
for call=0:1;score=vifvec(r,d);fprintf('SCORE %.17g\\n',score);fflush(stdout);end
""")
            rc,cc=execute(['octave-cli','--no-gui','--no-window-system','--no-history','--no-init-file','--quiet',str(entry)],env,directory,'canonical',600)
            config=dict(width=w,height=h,reference=str(paths['reference']),distorted=str(paths['distorted']),metric_name='vifvec',threads=threads,probe_only=True,only_mode='score',timing_phase='warm')
            (directory/'config.json').write_text(json.dumps(config,indent=2)+'\n')
            rr,cr=execute([str(binary),str(directory/'config.json')],env,directory,'rust',300)
            row=dict(metric='vifvec',pair=item['pair'],size=item['size_name'],threads=threads,canonical_returncode=rc,rust_returncode=rr,
                canonical_command=cc,rust_command=cr,bound=1e-9,status='process-finding',qualified_ratio=False)
            if rc==rr==0:
                vals=[float(s.split()[1]) for s in (directory/'canonical-stdout.log').read_text().splitlines() if s.startswith('SCORE ')]
                records=[json.loads(s.removeprefix('IQA_PROBE ')) for s in (directory/'rust-stdout.log').read_text().splitlines() if s.startswith('IQA_PROBE ')]
                assert len(vals)==2 and vals[0]==vals[1] and len(records)==1 and len(records[0]['scores'])==1
                ours=struct.unpack('>d',bytes.fromhex(records[0]['scores'][0][1]))[0]
                delta=abs(ours-vals[0]);row.update(ours=ours,canonical=vals[0],max_abs_delta=delta,status='pass' if math.isfinite(ours) and delta<=1e-9 else 'numeric-finding')
            elif min(w,h)<72 and rc==rr==1:
                cs=(directory/'canonical-stderr.log').read_text();rs=(directory/'rust-stderr.log').read_text()
                if ('Cannot build pyramid higher' in cs or 'pyramid' in cs.lower()) and '72' in rs:row['status']='matching-size-refusal'
            row['files']={p.name:sha(p) for p in directory.iterdir() if p.is_file()}
            (directory/'result.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
            print('VIFVEC_CASE',directory.name,row['status'],row.get('max_abs_delta'),flush=True)
            assert sha(binary)==identity['binary_sha256'] and sha(alias/'corrDn.mex')==identity['mex_sha256']
            assert all(sha(Path(path))==digest for path,digest in bound_files.items())
    (root/'completion.json').write_text(json.dumps(dict(cases=rows,qualified_timing_cells=0),indent=2)+'\n')


if __name__=='__main__':main()
