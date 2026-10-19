"""Copy pinned zenbench into a new isolated diagnostic workspace; never edit input."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path

def stage(source,out):
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if revision!='ac675d3a3f5334f4be6749a61b37995c9f3d34fb':raise ValueError('requires reviewed ac675d3 source')
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
    replace('src/monitor.rs','                    *target.lock().unwrap() = Some((Instant::now(), state));',        '                    let published = Instant::now();\n                    crate::monitor_overlay::publication(&state.attribution, published);\n                    *target.lock().unwrap() = Some((published, state));')
    replace('src/monitor.rs',        '        let latest = self.latest.lock().ok()?;\n        let (time, state) = latest.as_ref()?;\n        let age = time.elapsed();',        '        let latest = match self.latest.lock() { Ok(v)=>v, Err(_)=>{crate::monitor_overlay::read(None,None,Instant::now(),None,"missing");return None;} };\n        let Some((time,state))=latest.as_ref() else {crate::monitor_overlay::read(None,None,Instant::now(),None,"missing");return None;};\n        let observed=Instant::now();\n        let age=observed.duration_since(*time);\n        crate::monitor_overlay::read(Some(&state.attribution),Some(*time),observed,Some(age),if age<=Duration::from_secs(1){"fresh"}else{"stale"});')
    replace('src/monitor.rs','    pub(crate) fn latest(&self)',        '    #[cfg(test)]\n    pub(crate) fn control_stop(&mut self) { self.stop.store(true,Ordering::Release);if let Some(worker)=self.worker.get_mut().unwrap().take(){worker.join().unwrap();} }\n    #[cfg(test)]\n    pub(crate) fn control_clear(&mut self) { assert!(self.stop.load(Ordering::Acquire));*self.latest.lock().unwrap()=None; }\n    pub(crate) fn latest(&self)')
    replace('src/gate.rs','    pub(crate) fn check_and_record_status(&mut self) -> Option<bool> {',        '    #[cfg(test)]\n    pub(crate) fn control_stop_sampler(&mut self) { self.background.as_mut().unwrap().control_stop(); }\n    #[cfg(test)]\n    pub(crate) fn control_clear_cache(&mut self) { self.background.as_mut().unwrap().control_clear(); }\n    pub(crate) fn check_and_record_status(&mut self) -> Option<bool> {\n        crate::monitor_overlay::begin();')
    replace('src/gate.rs','                self.missing_observations = true;\n                return None;',        '                self.missing_observations = true;\n                crate::monitor_overlay::gate(None,self.observation_age_ns,None,None);\n                return None;')
    replace('src/gate.rs','        let clean = reason.is_none();',        '        let clean = reason.is_none();\n        crate::monitor_overlay::gate(Some(&state),self.observation_age_ns,Some(clean),Some(format!("{reason:?}")));')
    shutil.copyfile(Path(__file__).with_name('monitor_overlay.rs'),tree/'src/monitor_overlay.rs')
    files={str(p.relative_to(tree)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tree.rglob('*')) if p.is_file()}
    (out/'source-manifest.json').write_text(json.dumps(dict(revision=revision,originals=original,files=files,scope='isolated nonmetric monitor controls only'),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();stage(a.source,a.output)
