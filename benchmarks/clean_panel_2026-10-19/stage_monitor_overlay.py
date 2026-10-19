"""Copy pinned zenbench into a new isolated diagnostic workspace; never edit input."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path

def stage(source,out):
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if not revision.startswith('ac675d3'):raise ValueError('requires reviewed ac675d3 source')
    out.mkdir(parents=True,exist_ok=False)
    subprocess.run(['git','-C',str(source),'archive','--format=tar','HEAD','-o',str(out/'source.tar')],check=True)
    import tarfile
    with tarfile.open(out/'source.tar') as archive:archive.extractall(out/'source',filter='data')
    tree=out/'source';original={}
    def replace(name,old,new):
        path=tree/name;data=path.read_text();original.setdefault(name,hashlib.sha256(path.read_bytes()).hexdigest())
        if data.count(old)!=1:raise ValueError('source anchor mismatch '+name)
        path.write_text(data.replace(old,new))
    replace('src/lib.rs','mod monitor;','mod monitor;\nmod monitor_overlay;')
    replace('src/platform.rs','pub struct SystemState {','pub struct SystemState {\n    pub(crate) attribution: serde_json::Value,')
    replace('src/platform.rs','pub fn snapshot_excluding_processes(&self, workers: &[u32]) -> SystemState {','pub fn snapshot_excluding_processes(&self, workers: &[u32]) -> SystemState {\n        let attribution_before=crate::monitor_overlay::before();')
    replace('src/platform.rs','        SystemState {\n            cpu_load,','        let attribution=crate::monitor_overlay::snapshot(attribution_before, &sys, &owned, &user_tasks, heavy_process_count);\n        SystemState {\n            attribution,\n            cpu_load,')
    replace('src/gate.rs','        let clean = reason.is_none();','        crate::monitor_overlay::check(&state.attribution);\n        let clean = reason.is_none();')
    shutil.copyfile(Path(__file__).with_name('monitor_overlay.rs'),tree/'src/monitor_overlay.rs')
    files={str(p.relative_to(tree)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tree.rglob('*')) if p.is_file()}
    (out/'source-manifest.json').write_text(json.dumps(dict(revision=revision,originals=original,files=files,scope='isolated nonmetric monitor controls only'),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();stage(a.source,a.output)
