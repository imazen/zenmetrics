"""Record actual compile artifacts and ELF closure without executing scorers."""
import argparse,hashlib,json,os,re,subprocess,sys
from pathlib import Path
import runtime_closure

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def record(path):
    p=Path(path);return dict(path=str(p.absolute()),resolved=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size)
def collect(root,output):
    output.mkdir(exist_ok=False);native=root/'native-build'
    binaries=json.loads((root/'binaries.json').read_text())
    for r in binaries:
        if sha(r['path'])!=r['sha256']:raise ValueError('compiled binary changed')
    loader=runtime_closure.loader_environment()
    modules=set()
    for m in list(sys.modules.values()):
        for name in ('__file__','__cached__'):
            value=getattr(m,name,None)
            if value and Path(value).is_file():modules.add(Path(value).absolute())
    (output/'python-loaded-files.json').write_text(json.dumps([record(p) for p in sorted(modules)],indent=2))
    extensions=[str(p) for p in modules if runtime_closure.elf(p)]
    closure=runtime_closure.closure([r['path'] for r in binaries]+[sys.executable]+extensions,loader)
    (output/'runtime-closure.json').write_text(json.dumps(closure,indent=2))
    commands=subprocess.check_output(['ninja','-C',str(native),'-t','commands','iqa-jxl'],text=True)
    (output/'native-commands.txt').write_text(commands)
    deps=subprocess.check_output(['ninja','-C',str(native),'-t','deps'],text=True)
    (output/'native-deps.txt').write_text(deps)
    headers=set()
    for line in deps.splitlines():
        if line.startswith('    '):
            p=Path(line.strip());p=p if p.is_absolute() else native/p
            if not p.is_file():raise ValueError('missing compiled dependency '+str(p))
            headers.add(p.absolute())
    if not headers:raise ValueError('no compiler dependency evidence')
    (output/'compiled-native-dependencies.json').write_text(json.dumps([record(p) for p in sorted(headers)],indent=2))
    libs=set(native.rglob('*.a'))
    for path in re.findall(r'(/[^\s;]+\.(?:a|so(?:\.[0-9]+)*))',commands):
        if Path(path).is_file():libs.add(Path(path))
    (output/'native-link-files.json').write_text(json.dumps([record(p) for p in sorted(libs)],indent=2))
    cargo_files=[p for p in (root/'target/release').rglob('*') if p.is_file() and (p.suffix in ('.rlib','.rmeta','.d') or '.fingerprint' in p.parts and p.suffix=='.json')]
    (output/'rust-artifacts.json').write_text(json.dumps([record(p) for p in sorted(cargo_files)],indent=2))
    configs=[Path('/etc/ld.so.cache'),Path('/etc/ld.so.conf')]+list(Path('/etc/ld.so.conf.d').glob('*'))
    (output/'loader-config.json').write_text(json.dumps(dict(environment=loader,files=[record(p) for p in configs if p.is_file()]),indent=2))
    # Retain actual dependency-file membership and source bytes for owned/pinned
    # source trees. Registry package archives/checksums remain separately bound
    # by Cargo.lock; do not inspect the global registry source checkout.
    source_files={}
    for base in (root/'work',root/'butteraugli',root/'dssim',root/'adapter'):
        for p in base.rglob('*'):
            if p.is_file():source_files[str(p)]=sha(p)
    (output/'owned-sources.json').write_text(json.dumps(source_files,indent=2))
    controller=Path(__file__).parent
    (output/'controller-files.json').write_text(json.dumps([record(p) for p in sorted(controller.rglob('*')) if p.is_file()],indent=2))
    (output/'summary.json').write_text(json.dumps(dict(binaries=binaries,compiled_dependency_files=len(headers),rust_artifacts=len(cargo_files),metric_execution=False,scope_qualified=False,attribution_ready=False,remaining=['independent cohort/protocol/scope review','unchanged parity and actual scorer gate evidence','generic and statistical gates']),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();collect(a.root,a.output)
