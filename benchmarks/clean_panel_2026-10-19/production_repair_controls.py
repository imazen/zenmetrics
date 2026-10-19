"""Isolated owned-child fault quarantine and retained full-chain negative controls."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
import production_locks as locks,production_lock_run as runner,production_join,monitor_join

def waitfile(p):
    end=time.monotonic()+10
    while time.monotonic()<end:
        if p.exists():return
        time.sleep(.01)
    raise AssertionError('missing '+str(p))
def blocked(paths):
    for p in paths:
        with p.open('r+') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            raise AssertionError('admission released before proven cleanup')
def worker(mode,d):
    ready=d/'recover';original=locks.identity
    if mode=='identity':
        def identity(pid):
            if pid!=os.getpid() and not ready.exists():raise OSError('post-Popen identity fault')
            return original(pid)
        locks.identity=identity
    if mode in ('inventory','interrupt'):
        old=runner.session_members
        def inventory(sid):
            waitfile(d/'child.json')
            if not ready.exists():
                if mode=='interrupt':raise KeyboardInterrupt('post-launch interrupt')
                raise OSError('post-launch inventory fault')
            return old(sid)
        runner.session_members=inventory
    if mode=='receipt':
        old=Path.write_text
        def write(p,*a,**kw):
            if p.name=='wrapper.json' and not ready.exists():raise OSError('receipt fault')
            return old(p,*a,**kw)
        Path.write_text=write
    if mode in ('cleanup','logging'):
        old=locks.event
        def event(p,kind,**kw):
            if ((mode=='cleanup' and kind=='cleanup-confirmed') or (mode=='logging' and kind in ('quarantined','cleanup-confirmed'))) and not ready.exists():raise OSError('cleanup/log fault')
            return old(p,kind,**kw)
        locks.event=event
    runner.run(d/'inventory.json',d/'wrapper',[sys.executable,__file__,'child',str(d)])
def controls(root):
    root.mkdir(exist_ok=False);results={}
    for mode in ('identity','inventory','interrupt','receipt','cleanup','logging'):
        d=root/mode;d.mkdir();paths=[d/'a.lock',d/'b.lock']
        for p in paths:p.write_text('owned fixture')
        (d/'inventory.json').write_text(json.dumps(dict(locks=[locks.inspect(p) for p in paths])))
        with (d/'worker.log').open('wb') as log:
            proc=subprocess.Popen([sys.executable,__file__,mode,str(d)],stdout=log,stderr=subprocess.STDOUT)
            waitfile(d/'child.json');blocked(paths)
            pending=[]
            for p in paths:pending.append(locks.inspect(p))
            if mode=='identity':time.sleep(.25)
            else:waitfile(d/'child-finished');time.sleep(.25)
            assert proc.poll() is None;blocked(paths)
            held=json.loads((d/'wrapper/production-held.json').read_text())
            kernel=[locks.fdinfo(proc.pid,r['fd']) for r in held['locks']]
            assert all(r['kernel_lock'] in raw.splitlines() for r,raw in zip(held['locks'],kernel))
            (d/'pending-cleanup.json').write_text(json.dumps(dict(wrapper=locks.identity(proc.pid),locks=pending,held=held,kernel_fdinfo=kernel,both_nonblocking_acquisitions_refused=True,child_finished=(d/'child-finished').exists()),indent=2))
            (d/'recover').write_text('remove injected fault')
            assert proc.wait(timeout=10)==0
        waitfile(d/'child-finished')
        for p in paths:
            with p.open('r+') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
        w=json.loads((d/'wrapper/wrapper.json').read_text());assert w['quarantine_observed'] is True
        results[mode]='retained both locks pending cleanup; recovered and released after proof'
    archive=Path('/mnt/v/aic2026-evidence-private/clean-panel-production-monitor-2026-10-19')
    results['positive']=production_join.verify(archive)
    mutants=Path('/home/lilith/tmp/production-lock-independent-controls/fully-rehashed-mutations')
    for d in sorted(mutants.iterdir()):
        v=json.loads((d/'runtime/attribution.json').read_text());c=json.loads((d/'runtime/outer.json').read_text())
        paths=dict(binary=archive/'zenbench-061293ee59bcf2c1',config=archive/'runtime/config.json',source=archive/'source-manifest.json',trace=archive/'runtime/trace.jsonl',outer_receipt=d/'runtime/outer.json',held_receipt=archive/'runtime/ownership/held.json',launch_receipt=archive/'runtime/ownership/launch.json')
        monitor_join.verify_artifacts(v,paths);monitor_join.verify_outer(v,c)
        try:production_join.verify(d)
        except ValueError as e:results[d.name]=str(e)
        else:raise AssertionError('full-chain mutant accepted '+d.name)
    (root/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))
if __name__=='__main__':
    mode=sys.argv[1];d=Path(sys.argv[2])
    if mode=='controls':controls(d)
    elif mode=='child':
        (d/'child.json').write_text(json.dumps(locks.identity(os.getpid())));time.sleep(1);(d/'child-finished').write_text('natural exit')
    else:worker(mode,d)
