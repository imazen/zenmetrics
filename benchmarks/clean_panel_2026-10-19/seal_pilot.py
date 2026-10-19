"""Bind a newly built baseline and installed canonical environment; never backdate."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import platform
import runtime_closure
import workload
import bytecode_inventory
from panel import digest, identity


def record(path):
    path=Path(path).absolute()
    return {'path':str(path),'sha256':digest(path)}


def source_inventory(root):
    rows=[]
    for directory,dirs,files in os.walk(root):
        dirs[:]=sorted(d for d in dirs if d not in {'.git','.jj','target','__pycache__'})
        for name in sorted(files):
            path=Path(directory)/name
            if path.is_file() and name not in {'.workongoing'}:
                rows.append({'path':str(path.relative_to(root)),'sha256':digest(path)})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--python',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--distorted',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--revision',required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    root=args.output.resolve()
    here=Path(__file__).resolve().parent
    source=args.build/'work/zenmetrics'
    baseline=root/'iqa-compete'
    shutil.copy2(args.build/'target/release/iqa-compete',baseline)
    baseline.chmod(0o555)
    weights=root/'brisque_svm_weights.pth'
    shutil.copy2(args.weights,weights)
    inputs=[]
    for label,src in [('reference',args.reference),('distorted',args.distorted)]:
        dest=root/(label+'.rgb');shutil.copy2(src,dest)
        if dest.stat().st_size!=512*512*3:raise ValueError('wrong input dimensions')
        inputs.append(record(dest))
    inventory=root/'sources.json'
    inventory.write_text(json.dumps(source_inventory(args.build),sort_keys=True))
    receipts=[record(inventory),record(baseline),record(weights),record(here/'seal_pilot.py'),record(here/'build_baseline.sh')]
    for name in ['build-resource.log','rustc.txt','cargo.txt','features.txt','metadata.json','build-command.txt','model-preload.meta.json','luma-maps.meta.json']:
        dest=root/name;shutil.copy2(args.build/name,dest);receipts.append(record(dest))
    for name in ['Cargo.lock','ci/sibling-pins.tsv']:
        dest=root/Path(name).name;shutil.copy2(source/name,dest);receipts.append(record(dest))
    # The source tree hash is newly observed. Only the sealed binary is executed.
    build_identity={'revision':args.revision,'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'receipts':list(receipts),'rustflags':'--cfg iqa_compete_ingress --cfg iqa_compete_cached','features':'compete,metrics-parallel,zenmetrics-cli/nr'}
    build_file=root/'baseline.json';build_file.write_text(json.dumps(build_identity,indent=2));receipts.append(record(build_file))
    # Bind all installed environment source/native/data files, not just the adapter.
    envroot=args.python.parent.parent
    environment=[]
    for directory,dirs,files in os.walk(envroot/'lib'):
        dirs[:]=sorted(dirs)
        for name in sorted(files):
            path=Path(directory)/name
            if path.is_file():environment.append(record(path))
    environment.extend([record(args.python),record(sys.executable),record(here/'brisque_oracle.py')])
    runtime=json.loads(subprocess.check_output([str(args.python),'-B','-c','import sys,sysconfig,json;print(json.dumps(dict(version=sys.version,stdlib=sysconfig.get_path("stdlib"))))'],env=bytecode_inventory.environment(),text=True))
    stdlib=Path(runtime['stdlib'])
    for path in sorted(stdlib.rglob('*.py')):
        if 'site-packages' not in path.parts and '__pycache__' not in path.parts:environment.append(record(path))
    for path in sorted((stdlib/'lib-dynload').glob('*.so')):environment.append(record(path))
    loader=runtime_closure.loader_environment()
    native_roots=[r['path'] for r in environment if runtime_closure.elf(r['path'])]
    native_roots += [str(p) for p in [baseline,args.python,Path(sys.executable).absolute()] if runtime_closure.elf(p)]
    native=runtime_closure.closure(native_roots,loader)
    environment.extend(native['files'])
    # Loader configuration and inspection tools are part of the binding too.
    for path in [Path('/etc/ld.so.cache'),Path('/etc/ld.so.conf'),*Path('/etc/ld.so.conf.d').glob('*'),Path('/usr/bin/ldd'),Path('/usr/bin/readelf')]:
        if path.is_file(): environment.append(record(path))
    packages=json.loads(subprocess.check_output([str(args.python),'-c',
        'import importlib.metadata as m,json;print(json.dumps(sorted((d.metadata.get("Name",""),d.version) for d in m.distributions())))'],env=bytecode_inventory.environment(),text=True))
    contention=source/'benchmarks/iqa_compete_2026-10-16/contention.py'
    probes=[dict(executable=str(args.python),script_directory=str(here)),
            dict(executable=str(Path(sys.executable).absolute()),script_directory=str(here)),
            dict(executable=str(Path(sys.executable).absolute()),script_directory=str(contention.parent))]
    bytecode=bytecode_inventory.capture(probes,[str(envroot/'lib'),str(stdlib)])
    canonical={'bytecode_binding':bytecode,'runtime':runtime,'packages':packages,'artifacts':environment,'weights':record(weights),'native_closure':native}
    canonical_file=root/'canonical.json';canonical_file.write_text(json.dumps(canonical,indent=2))
    config=root/'cell.json'
    config.write_text(json.dumps(dict(width=512,height=512,reference=str(root/'reference.rgb'),distorted=str(root/'distorted.rgb'),metric_name='brisque',only_mode='score',timing_phase='warm',locked_process_accounting=True,threads=1,chunk_budget_ms=60000)))
    contention=source/'benchmarks/iqa_compete_2026-10-16/contention.py'
    containment=here/'containment.py'
    heavy=Path.home()/'work/claudehints/scripts/run-heavy'
    scope='resident RGB8 reference and distorted; separate NR scores for both; ingress+metric+result allocation timed; immutable model load and input file I/O excluded'
    artifacts=receipts+environment+[record(canonical_file),record(config),record(contention),record(containment),record(heavy),record(here/'panel.py'),record(here/'workload.py'),record(here/'runtime_closure.py'),record(here/'bytecode_inventory.py'),record('/usr/bin/taskset')]
    # Duplicate resolved library paths carry one authoritative expected hash.
    artifacts=list({r['path']:r for r in artifacts}.values())
    work=dict(metric_name='brisque',width=512,height=512,threads=1,mode='score',timing_phase='warm',reference=str(root/'reference.rgb'),distorted=str(root/'distorted.rgb'))
    pyiqa_version=next(version for name,version in packages if name.lower()=='pyiqa')
    manifest=dict(schema=2,metric='brisque',shape=[512,512,1,'score'],scope={'ours':scope,'canonical':scope},cpu=2,parity_abs_bound=15.138428/1000,
                  inputs=inputs,artifacts=artifacts,host_observation=platform.uname()._asdict(),contention_script=str(contention),containment_script=str(containment),run_heavy=str(heavy),controller_python=str(Path(sys.executable).absolute()),taskset='/usr/bin/taskset',workload=work,native_closure=native,bytecode_binding=bytecode,arms={
                      'ours':dict(build_identity=identity(build_identity),argv=[str(baseline),str(config)],protocol='iqa-row',sample_count=10),
                      'canonical':dict(build_identity=identity(canonical),argv=[str(args.python),str(here/'brisque_oracle.py'),str(root/'reference.rgb'),str(root/'distorted.rgb'),str(weights)],protocol='canonical-v2',adapter=workload.ADAPTER,pyiqa_version=pyiqa_version,version='pyiqa-'+pyiqa_version+workload.VERSION_SUFFIX,sample_count=10)})
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('SEALED',root,'artifacts',len(artifacts),flush=True)


if __name__=='__main__':main()
