#!/usr/bin/env python3
"""Overlay only the verified private counter correction on pinned zenbench."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import uuid

PIN = 'ac675d3a3f5334f4be6749a61b37995c9f3d34fb'
CORRECTION = '3b1d0f714deac0140d6653b6aa42747466b0460a'
ORIGINAL = '99ca5793392f7cb3aa09675f5f11fbf9767151182ac8f3d549e0a3d1b07c9af9'
sha = lambda data: hashlib.sha256(data).hexdigest()


def adapt(source):
    if sha(source.encode()) != ORIGINAL:
        raise ValueError('private correction requires the exact frozen ac675d3 platform source')
    source = source.replace('mod affinity;', '#[cfg(any(target_os = "macos", test))]\nmod cpu_counters;\nmod affinity;', 1)
    source = source.replace('    exclude_benchmark_owner: bool,', '    exclude_benchmark_owner: bool,\n    #[cfg(target_os = "macos")]\n    cpu_counters: Mutex<cpu_counters::CpuCounters>,', 1)
    source = source.replace('        Self {\n            sys: Mutex::new(sys),', '''        #[cfg(target_os = "macos")]
        let cpu_counters = cpu_counters::CpuCounters::new(Instant::now(), sys.processes().values().map(|p| (p.pid(), p.start_time(), p.accumulated_cpu_time())));
        Self {
            #[cfg(target_os = "macos")]
            cpu_counters: Mutex::new(cpu_counters),
            sys: Mutex::new(sys),''', 1)
    source = source.replace('        sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);', '        sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);\n        #[cfg(target_os = "macos")]\n        let process_sample_at = Instant::now();', 1)
    source = source.replace('        let (cpu_load, heavy_process_count) = foreign_activity(', '''        #[cfg(target_os = "macos")]
        let process_rates = self.cpu_counters.lock().unwrap().refresh(process_sample_at,
            sys.processes().values().map(|p| (p.pid(), p.start_time(), p.accumulated_cpu_time())));
        let (cpu_load, heavy_process_count) = foreign_activity(''', 1)
    needle = '.map(|p| (p.pid(), f64::from(p.cpu_usage()))),'
    if source.count(needle) != 1:
        raise ValueError('process accounting integration changed')
    source = source.replace(needle, '''.map(|p| {
                    #[cfg(target_os = "macos")]
                    let percent = cpu_counters::conservative_percent(f64::from(p.cpu_usage()), process_rates.get(&p.pid()).copied(), owned.contains(&p.pid()));
                    #[cfg(not(target_os = "macos"))]
                    let percent = f64::from(p.cpu_usage());
                    (p.pid(), percent)
                }),''', 1)
    return source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot', type=Path)
    p.add_argument('original', type=Path)
    p.add_argument('receipt', type=Path)
    p.add_argument('--source', type=Path)
    a = p.parse_args()
    if (a.snapshot/'Cargo.toml').samefile(a.original/'Cargo.toml') or (a.snapshot/'Cargo.lock').samefile(a.original/'Cargo.lock'):
        raise ValueError('snapshot aliases original source')
    candidates = [a.source] if a.source else list((Path.home()/'.cargo/git/checkouts').glob('zenbench-*/ac675d3'))
    candidates = [d for d in candidates if (d/'src/platform.rs').exists() and sha((d/'src/platform.rs').read_bytes()) == ORIGINAL]
    if len(candidates) != 1:
        raise ValueError('resolve exactly one pinned zenbench source, or pass --source')
    source = candidates[0]
    copy = a.receipt.parent / ('zenbench-mac-counters-' + uuid.uuid4().hex[:12])
    shutil.copytree(source, copy, ignore=shutil.ignore_patterns('.git','target'))
    helper = Path(__file__).with_name('zenbench_cpu_counters.rs')
    (copy/'src/platform/cpu_counters.rs').write_bytes(helper.read_bytes())
    platform = copy/'src/platform.rs'
    platform.write_text(adapt(platform.read_text()))
    before = {name:(a.snapshot/name).read_bytes() for name in ('Cargo.toml','Cargo.lock')}
    marker = '[patch."https://github.com/imazen/zenbench"]'
    if marker in before['Cargo.toml'].decode():
        raise ValueError('existing zenbench override; preserve it')
    (a.snapshot/'Cargo.toml').write_text(before['Cargo.toml'].decode()+'\n'+marker+'\nzenbench = { path = '+json.dumps(str(copy.resolve()))+' }\n')
    subprocess.run(['cargo','metadata','--format-version','1'],cwd=a.snapshot,stdout=subprocess.DEVNULL,check=True)
    value = dict(adapter_sha256=sha(Path(__file__).read_bytes()), zenbench_base=PIN, correction_commit=CORRECTION, private_only=True,
        source=str(source.resolve()), copy=str(copy.resolve()), original_platform_sha256=ORIGINAL,
        corrected_platform_sha256=sha(platform.read_bytes()), counter_module_sha256=sha(helper.read_bytes()),
        files={name:dict(original_sha256=sha(data),adapted_sha256=sha((a.snapshot/name).read_bytes())) for name,data in before.items()},
        scope='Only private Mac cumulative process rates; original CPU gates, Linux arithmetic and public API unchanged; fresh worker/bundle required')
    if a.receipt.exists():
        a.receipt.rename(a.receipt.with_name(a.receipt.stem+'-'+uuid.uuid4().hex[:12]+'.bak.json'))
    a.receipt.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
