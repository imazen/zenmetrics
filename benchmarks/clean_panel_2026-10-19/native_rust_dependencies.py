"""Bind compiled Rust dependency records to package archives and Git-source inputs."""
import argparse,hashlib,json,re
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def capture(root,output):
    if output.exists():raise ValueError('retain prior evidence')
    deps='\n'.join(p.read_text() for p in (root/'target/release').rglob('*.d'))
    registry=sorted(set(re.findall(r'/\.cargo/registry/src/[^/\s]+/([^/\s]+)/',deps)))
    packages=[];lock=(root/'work/zenmetrics/Cargo.lock').read_text()
    for name in registry:
        found=list((Path.home()/'.cargo/registry/cache').glob('*/'+name+'.crate'))
        if not found:raise ValueError('compiled registry package archive missing '+name)
        hashes={sha(p) for p in found}
        if len(hashes)!=1:raise ValueError('ambiguous package archive '+name)
        checksum=next(iter(hashes))
        if 'checksum = "'+checksum+'"' not in lock:raise ValueError('archive checksum not in frozen lock '+name)
        packages.append(dict(package=name,sha256=checksum,archive_paths=[str(p) for p in found]))
    gitfiles=sorted(set(re.findall(r'(/[^\s:]+/\.cargo/git/checkouts/[^\s:]+)',deps)))
    files=[]
    for path in gitfiles:
        p=Path(path)
        if p.is_file():files.append(dict(path=str(p),sha256=sha(p)))
    output.write_text(json.dumps(dict(registry_packages=packages,git_source_files=files,source_membership='actual rustc dependency files',registry_binding='package archive SHA matches frozen Cargo.lock; global registry source files not inspected',scorer_executed=False),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();capture(a.root,a.output)
