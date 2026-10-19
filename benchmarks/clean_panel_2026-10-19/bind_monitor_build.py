"""Bind a successful isolated nonmetric monitor lib-test build to actual files."""
import argparse,hashlib,json,subprocess
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def bind(root,log):
    manifest=root/'source-manifest.json';data=json.loads(manifest.read_text())
    for name,digest in data['files'].items():
        if sha(root/'source'/name)!=digest:raise ValueError('source changed '+name)
    events=[json.loads(line) for line in log.read_text().splitlines() if line.startswith('{')]
    if not any(e.get('reason')=='build-finished' and e.get('success') is True for e in events):raise ValueError('build not successful')
    executables=[Path(e['executable']) for e in events if e.get('reason')=='compiler-artifact' and e.get('executable') and e['target']['name']=='zenbench' and e['target']['kind']==['lib']]
    if len(executables)!=1:raise ValueError('ambiguous lib-test binary')
    binary=executables[0]
    value=dict(binary=str(binary),binary_sha256=sha(binary),source_manifest_sha256=sha(manifest),cargo_lock_sha256=sha(root/'source/Cargo.lock'),rustc=subprocess.check_output(['rustc','-vV'],text=True),build_log=str(log),build_log_sha256=sha(log),command=['cargo','test','--manifest-path',str(root/'source/Cargo.toml'),'--locked','--offline','--no-default-features','--lib','--no-run','--message-format=json'],metric_build=False)
    with (root/'build-receipt.json').open('x') as f:json.dump(value,f,indent=2)
    print(binary)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('log',type=Path);a=p.parse_args();bind(a.root,a.log)
