#!/usr/bin/env python3
"""Instrument an owned zenbench copy for resource-accounting diagnostics only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('source', type=Path, help='Read-only pinned zenbench source')
p.add_argument('copy', type=Path, help='New owned diagnostic source directory')
p.add_argument("--snapshot", type=Path, help="Owned diagnostic zenmetrics snapshot to override, never the source checkout")
p.add_argument("--original", type=Path, help="Source checkout to protect from aliasing")
a = p.parse_args()
revision = subprocess.check_output(["git", "-C", str(a.source), "rev-parse", "HEAD"], text=True).strip()
if not revision.startswith("ac675d3"):
    p.error("resource trace requires the pinned ac675d3 source")
if a.copy.exists():
    p.error('preserve previous diagnostic source; choose a new copy')
shutil.copytree(a.source, a.copy, ignore=shutil.ignore_patterns('.git', 'target'))
f = a.copy / 'src/platform.rs'
original = f.read_text()
needle = '        let (cpu_load, heavy_process_count) = foreign_activity(\n'
if original.count(needle) != 1:
    raise ValueError('pinned resource snapshot insertion point changed')
trace = '''        if let Some(path) = std::env::var_os("ZENBENCH_RAW_CPU_TRACE") {
            use std::io::Write as _;
            let processes: Vec<_> = sys.processes().values()
                .filter(|p| p.cpu_usage() > 0.0 || owned.contains(&p.pid()))
                .map(|p| serde_json::json!({"pid":p.pid().as_u32(),
                    "parent":p.parent().map(|p|p.as_u32()),
                    "name":p.name().to_string_lossy(),
                    "cpu_pct":p.cpu_usage(),"accumulated_cpu_ms":p.accumulated_cpu_time(),
                    "thread_kind":format!("{:?}",p.thread_kind()),
                    "owned":owned.contains(&p.pid()),"user_task":user_tasks.contains(&p.pid())}))
                .collect();
            let value = serde_json::json!({"diagnostic_only":true,
                "unix_ns":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("clock").as_nanos().to_string(),
                "global_fraction":cpu_load,"cores":cpus.len(),
                "owned":owned.iter().map(|p|p.as_u32()).collect::<Vec<_>>(),"processes":processes});
            let mut output = std::fs::OpenOptions::new().create(true).append(true).open(path).expect("raw diagnostic trace");
            writeln!(output,"{}",value).expect("write raw diagnostic trace");
        }
'''
f.write_text(original.replace(needle, trace + needle))
sha = lambda data: hashlib.sha256(data).hexdigest()
receipt = dict(qualification=False, source=str(a.source.resolve()), copy=str(a.copy.resolve()),
    file='src/platform.rs', original_sha256=sha(original.encode()), adapted_sha256=sha(f.read_bytes()),
    trigger='ZENBENCH_RAW_CPU_TRACE', scope='append raw internal resource snapshots; no gate or threshold changes')
(a.copy / 'resource-trace-patch.json').write_text(json.dumps(receipt, indent=2)+'\n')
if a.snapshot:
    if not a.original or (a.snapshot / 'Cargo.toml').samefile(a.original / 'Cargo.toml'):
        raise ValueError('owned snapshot and distinct original checkout required')
    manifest = a.snapshot / 'Cargo.toml'
    before = manifest.read_bytes()
    lock = a.snapshot / 'Cargo.lock'
    lock_before = lock.read_bytes()
    marker = '[patch."https://github.com/imazen/zenbench"]'
    if marker in before.decode():
        raise ValueError('existing zenbench override; preserve it')
    manifest.write_text(before.decode() + '\n' + marker + '\nzenbench = { path = ' + json.dumps(str(a.copy.resolve())) + ' }\n')
    subprocess.run(['cargo','metadata','--format-version','1'], cwd=a.snapshot, stdout=subprocess.DEVNULL, check=True)
    receipt['snapshot_files'] = {'Cargo.toml':{'original_sha256':sha(before),'adapted_sha256':sha(manifest.read_bytes())},
        'Cargo.lock':{'original_sha256':sha(lock_before),'adapted_sha256':sha(lock.read_bytes())}}
    receipt['zenbench_revision'] = revision
    (a.copy / 'resource-trace-patch.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt))
