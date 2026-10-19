"""Owned nonmetric runtime control for an isolated instrumented zenbench test binary."""
import argparse,copy,fcntl,hashlib,json,os,select,subprocess,sys,time
from pathlib import Path
import admission_receipt,gate_trace

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_outer(value,outer):
    rival=outer['rival'];leader=outer['launch']['child']
    if not any(p['pid']==rival['pid'] and p['start_ticks']==rival['start_ticks'] and p['counted'] for s in value['snapshots'] for p in s['processes']):raise ValueError('owned rival not counted')
    if not any(p['pid']==leader['pid'] and p['start_ticks']==leader['start_ticks'] and p['owned'] and not p['counted'] for s in value['snapshots'] for p in s['processes']):raise ValueError('monitor owner identity not joined')
    for c in value['checks']:
        earliest=outer['launch_before_ns']+c['checked_monotonic_ns'];latest=outer['ready_received_ns']+c['checked_monotonic_ns']
        if earliest>outer['exit_observed_ns'] or latest<outer['go_ns']:raise ValueError('outer clock interval disjoint')
    if outer.get('cleanup_complete') is not True or outer.get('child_returncode')!=0:raise ValueError('missing owned cleanup')
    if outer['launch']['owner']['pid']!=outer['rival']['ppid']:raise ValueError('outer parent mismatch')
    return True

def run(binary,source,out):
    build=json.loads(source.with_name('build-receipt.json').read_text())
    if sha(binary)!=build['binary_sha256'] or sha(source)!=build['source_manifest_sha256']:raise ValueError('compiled cohort binding mismatch')
    manifest=json.loads(source.read_text())
    for name,digest in manifest['files'].items():
        if sha(source.parent/'source'/name)!=digest:raise ValueError('source changed: '+name)
    out.mkdir(parents=True,exist_ok=False)
    binary=binary.resolve();config={'test':'monitor_overlay::controls::owned_nonmetric_monitor','gate':'unmodified GateConfig::default','source':str(source),'diagnostic_only':True,'clock':'child Instant origin bracketed by parent launch/READY monotonic interval','locks':'two owned fixture flocks; enclosing run-heavy/dev-heavy admission retained separately'}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    descriptors=[];busy=None;child=None;outer={}
    try:
        for name in ('primary','secondary'):
            path=out/(name+'.lock');fd=path.open('a');fcntl.flock(fd,fcntl.LOCK_EX);descriptors.append((fd,path))
        held=admission_receipt.Held(descriptors,out/'ownership')
        # Sibling is owned by this controller, intentionally foreign to the
        # monitor's registered set. It exercises real >10% classification.
        busy=subprocess.Popen([sys.executable,'-c','while True: pass'])
        outer['rival']=admission_receipt.process_identity(busy.pid)
        args=[str(binary),'--exact',config['test'],'--nocapture','--test-threads=1']
        env=os.environ.copy();env['CLEAN_PANEL_TRACE']=str(out/'trace.jsonl');env['ZENBENCH_LAUNCHER_PIDS']=str(os.getpid())
        outer['build_receipt_sha256']=sha(source.with_name('build-receipt.json'))
        outer['binary_sha256']=sha(binary)
        outer['controller_sha256']=sha(__file__)
        outer['launch_before_ns']=time.monotonic_ns()
        with (out/'stderr.log').open('w') as err:
            child=subprocess.Popen(args,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err)
            # Wait for an actual byte marker without buffered readline read-ahead.
            captured=b'';deadline=time.monotonic()+15
            while b'MONITOR_READY\n' not in captured:
                if time.monotonic()>=deadline:raise TimeoutError('nonmetric monitor ready')
                if select.select([child.stdout],[],[],.1)[0]:
                    chunk=os.read(child.stdout.fileno(),4096)
                    if not chunk:raise RuntimeError('child exited before READY')
                    captured+=chunk
            outer['ready_received_ns']=time.monotonic_ns();outer['launch']=held.bind_launch(args,child.pid)
            if sha(Path('/proc')/str(child.pid)/'exe')!=outer['binary_sha256']:raise ValueError('launched binary mismatch')
            outer['go_ns']=time.monotonic_ns();child.stdin.write(b'GO\n');child.stdin.flush()
            rest,_=child.communicate(timeout=20);(out/'stdout.log').write_bytes(captured+rest)
            if child.returncode!=0:raise RuntimeError('nonmetric child failed')
            outer['exit_observed_ns']=time.monotonic_ns();outer['child_returncode']=child.returncode
        # Reaped child cannot be reused as a signal target. Only our still-owned
        # unreaped rival receives terminate; never name/group-wide cleanup.
        busy.terminate();busy.wait(timeout=5);outer['rival_reaped']=True
        outer['locks_still_held']=[admission_receipt.lock_record(fd,p,i) for i,(fd,p) in enumerate(descriptors)]
        outer['cleanup_complete']=True
    finally:
        for proc in (child,busy):
            if proc is not None and proc.poll() is None:
                proc.kill();proc.wait(timeout=5)
        outer['finally_ns']=time.monotonic_ns()
        (out/'outer.json').write_text(json.dumps(outer,indent=2))
        for fd,_ in reversed(descriptors):fd.close()
    if sha(binary)!=outer['binary_sha256'] or sha(__file__)!=outer['controller_sha256']:raise ValueError('runtime artifact changed')
    events=[json.loads(s) for s in (out/'trace.jsonl').read_text().splitlines()]
    value=dict(schema=2,synthetic=False,snapshots=[e['value'] for e in events if e['kind']=='snapshot'],checks=[e['value'] for e in events if e['kind']=='check'])
    paths=dict(binary=binary,config=out/'config.json',source=source,trace=out/'trace.jsonl',outer_receipt=out/'outer.json')
    value.update({k+'_sha256':sha(p) for k,p in paths.items()})
    for name,digest in manifest['files'].items():
        if sha(source.parent/'source'/name)!=digest:raise ValueError('source changed after control: '+name)
    (out/'attribution.json').write_text(json.dumps(value,indent=2))
    result=gate_trace.verify_artifact_join(value,paths)
    if any(s['identity_errors'] for s in value['snapshots']):raise ValueError('unknown counted/owned process identity')
    from collections import Counter
    uses=Counter(c['snapshot_id'] for c in value['checks'])
    if not any(n>1 for n in uses.values()):raise ValueError('cached reuse not exercised')
    verify_outer(value,outer)
    rival=outer['rival']
    negatives=[]
    for mutation in ('age','identity','artifact','outer-owner'):
        bad=copy.deepcopy(value)
        try:
            if mutation=='age':bad['checks'][0]['observation_age_ns']+=1
            elif mutation=='identity':bad['checks'][0]['counted_identities'].append([1,1])
            elif mutation=='artifact':bad['binary_sha256']='0'*64
            else:
                wrong=copy.deepcopy(outer);wrong['launch']['child']['start_ticks']+=1
                verify_outer(bad,wrong)
            gate_trace.verify_artifact_join(bad,paths)
        except ValueError:negatives.append(mutation)
        else:raise AssertionError('negative accepted '+mutation)
    result.update(missing=['independent runtime overlay qualification','metric cohort integration not performed'],runtime_nonmetric_join=True,clock_correlation='bounded origin interval only',cached_reuse=dict(uses),owned_rival=rival,negative_controls=negatives,attribution_ready=False,remaining=['independent overlay review','full metric binary integration prohibited','counter interval brackets sysinfo refresh but does not reconstruct internal denominator','no historical attribution'])
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('binary',type=Path);p.add_argument('source_manifest',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.binary,a.source_manifest,a.output)
