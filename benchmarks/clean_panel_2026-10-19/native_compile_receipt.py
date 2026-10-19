"""Preflight and execute the frozen native build plan, retaining actual tool evidence."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
import native_build

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(root,receipts,attempt):
    plan=json.loads((receipts/'plan.json').read_text())
    evidence=root/attempt;evidence.mkdir(exist_ok=False)
    if any((root/p).exists() for p in ('target','native-build','binaries.json')):raise ValueError('fresh build output required')
    for name,h in plan['receipts'].items():
        if sha(receipts/name)!=h:raise ValueError('receipt changed '+name)
    for name,h in plan['sources'].items():
        if sha(root/name)!=h:raise ValueError('source changed '+name)
    for r in plan['inputs']:
        if sha(r['path'])!=r['sha256']:raise ValueError('input changed')
    adapter=json.loads((receipts/'adapter-binding.json').read_text())
    for r in adapter['files']:
        if sha(root/'adapter'/Path(r['path']).name)!=r['sha256']:raise ValueError('adapter changed')
    for name,h in json.loads((root/'libjxl-source.json').read_text()).items():
        if sha(root/'libjxl'/name)!=h:raise ValueError('native source changed '+name)
    overlay=json.loads((root/'rust-overlay.json').read_text())
    for name,entry in json.loads((root/'work/zenmetrics.mtimes.json').read_text()).items():
        p=root/'work/zenmetrics'/name;h=overlay['after_sha256'] if str(p)==overlay['path'] else entry[0]
        if sha(p)!=h:raise ValueError('frozen Rust source changed '+name)
    tools={}
    for name,argv in [('rustc',['rustc','-vV']),('cargo',['cargo','-vV']),('cxx',['/usr/bin/c++','--version']),('cc',['/usr/bin/cc','--version']),('cmake',['cmake','--version']),('ninja',['ninja','--version']),('linker',['ld','--version'])]:
        result=subprocess.check_output(argv,text=True)
        import shutil
        path=Path(shutil.which(argv[0])).resolve();tools[name]=dict(argv=argv,path=str(path),sha256=sha(path),version=result)
    (evidence/'preflight.json').write_text(json.dumps(dict(plan_sha256=sha(receipts/'plan.json'),helper_sha256=sha(__file__),builder_sha256=sha(native_build.__file__),tools=tools,environment={k:os.environ.get(k) for k in ('PATH','RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','LD_LIBRARY_PATH','CARGO_BUILD_JOBS','RAYON_NUM_THREADS','OMP_NUM_THREADS')}),indent=2))
    tree=root/'work/zenmetrics';before=sha(tree/'Cargo.lock')
    native_build.call(['cargo','metadata','--locked','--offline','--format-version','1'],tree,evidence/'locked-metadata.json')
    if sha(tree/'Cargo.lock')!=before:raise ValueError('lock changed')
    native_build.build(root)
    if sha(tree/'Cargo.lock')!=before:raise ValueError('lock changed during build')
    (evidence/'build-complete.json').write_text(json.dumps(dict(binaries=json.loads((root/'binaries.json').read_text()),scorer_executed=False,parity_executed=False,timing_acquired=False),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('receipts',type=Path);p.add_argument('attempt');a=p.parse_args();main(a.root,a.receipts,a.attempt)
