"""Fresh build-only native cohort; caller holds both existing locks/run-heavy."""
import argparse,hashlib,json,os,subprocess,tarfile,shutil
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def call(argv,cwd,log):
    with log.open('wb') as stream:subprocess.run(argv,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,check=True)
def prepare(root,cache):
    tree=root/'work/zenmetrics';receipts=[]
    for line in (tree/'ci/sibling-pins.tsv').read_text().splitlines():
        if not line or line.startswith('#'):continue
        name,url,dest,rev,flags=line.split('\t');source=(cache/'work/zenmetrics'/dest).resolve();target=(tree/dest).resolve()
        if not target.is_relative_to(root.resolve()):raise ValueError('sibling escapes cohort')
        target.mkdir(parents=True,exist_ok=False)
        archive=root/(name+'.tar')
        with archive.open('wb') as f:subprocess.run(['git','-C',str(source),'archive',rev],stdout=f,check=True)
        with tarfile.open(archive) as t:
            for member in t.getmembers():
                if flags=='sparse-no-benchmarks' and (member.name=='benchmarks' or member.name.startswith('benchmarks/')):continue
                if not (target/member.name).resolve().is_relative_to(target):raise ValueError('archive escape')
                t.extract(member,target)
        receipts.append(dict(name=name,revision=rev,archive_sha256=sha(archive),source_cache=str(source)))
    (root/'sibling-exports.json').write_text(json.dumps(receipts,indent=2))
    before=sha(tree/'Cargo.lock')
    call(['cargo','metadata','--locked','--offline','--format-version','1'],tree,root/'locked-metadata.json')
    if sha(tree/'Cargo.lock')!=before:raise ValueError('lock changed')
    (root/'locked-check.json').write_text(json.dumps(dict(cargo_lock_sha256=before,passed=True)))
def native_source(root,source,inputs):
    shutil.copytree(source,root/'libjxl',ignore=shutil.ignore_patterns('.git'))
    files={str(p.relative_to(root/'libjxl')):sha(p) for p in sorted((root/'libjxl').rglob('*')) if p.is_file()}
    (root/'libjxl-source.json').write_text(json.dumps(files,indent=2))
    (root/'inputs').mkdir();pairs=[]
    for role in ('reference','distorted'):
        old=inputs/('cid22-'+role+'.rgb');new=root/'inputs'/(role+'.rgb');shutil.copyfile(old,new)
        if new.stat().st_size!=512*512*3:raise ValueError('input size')
        pairs.append(dict(role=role,path=str(new),sha256=sha(new),width=512,height=512,stride_bytes=1536,layout='RGB8 packed sRGB',source=str(old)))
    (root/'inputs.json').write_text(json.dumps(pairs,indent=2))

def build(root):
    tree=root/'work/zenmetrics'
    env=os.environ;env['CARGO_TARGET_DIR']=str(root/'target');env['CARGO_INCREMENTAL']='0'
    # Baseline graph, native FR only; no NR or NN feature inherited.
    features='compete,metrics-parallel'
    call(['cargo','tree','--locked','--offline','-p','cpu-profile','--features',features,'-e','features'],tree,root/'features.txt')
    call(['rustc','-vV'],tree,root/'rustc.txt')
    call(['cargo','build','--locked','--offline','--release','-p','cpu-profile','--features',features,'--bin','iqa-compete'],tree,root/'rust-build.log')
    call(['cmake','-S',str(root/'adapter'),'-B',str(root/'native-build'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DLIBJXL_SOURCE='+str(root/'libjxl')],root,root/'cmake-configure.log')
    call(['cmake','--build',str(root/'native-build'),'--target','iqa-jxl','--parallel',os.environ['CARGO_BUILD_JOBS']],root,root/'native-build.log')
    binaries=[root/'target/release/iqa-compete',root/'native-build/iqa-jxl']
    (root/'binaries.json').write_text(json.dumps([dict(path=str(p),sha256=sha(p)) for p in binaries],indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','native-source','build']);p.add_argument('root',type=Path);p.add_argument('--cache',type=Path);p.add_argument('--source',type=Path);p.add_argument('--inputs',type=Path);a=p.parse_args()
    if a.phase=='prepare':prepare(a.root,a.cache)
    elif a.phase=='native-source':native_source(a.root,a.source,a.inputs)
    else:build(a.root)
