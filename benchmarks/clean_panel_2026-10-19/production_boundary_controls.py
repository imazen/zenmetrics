"""Exact post-Popen line-boundary interruption plus adjacent retry transitions."""
import inspect,json,os,subprocess,sys,time,fcntl
from pathlib import Path
import production_lock_run as runner,production_locks as locks
from production_repair_controls import waitfile,blocked

def site(function,text):
    lines,start=inspect.getsourcelines(function)
    return start+next(i for i,l in enumerate(lines) if l.strip()==text)
def worker(mode,d):
    function,line={
        'popen-return':(runner.run,site(runner.run,'rc=supervise(child,output,control_output,receipt,outer,manager,state)')),
        'retry-header':(runner.supervise,site(runner.supervise,'while True:')),
        'retry-try':(runner.supervise,site(runner.supervise,'try:')),
    }[mode]
    def boundary(frame,event,arg):
        if event=='line' and frame.f_code is function.__code__ and frame.f_lineno==line:
            waitfile(d/'child.json')
            (d/'injected.json').write_text(json.dumps(dict(function=function.__name__,line=line)))
            raise KeyboardInterrupt('owned launch-to-supervision boundary injection')
        return boundary
    sys.settrace(boundary)
    runner.run(d/'inventory.json',d/'wrapper',[sys.executable,__file__,'child',str(d)])
def controls(root):
    root.mkdir(exist_ok=False);result={}
    for mode in ('popen-return','retry-header','retry-try'):
        d=root/mode;d.mkdir();paths=[d/'a.lock',d/'b.lock']
        for p in paths:p.write_text('owned boundary fixture')
        (d/'inventory.json').write_text(json.dumps(dict(locks=[locks.inspect(p) for p in paths])))
        with (d/'worker.log').open('wb') as log:
            worker_process=subprocess.Popen([sys.executable,__file__,mode,str(d)],stdout=log,stderr=subprocess.STDOUT)
            waitfile(d/'injected.json');child=json.loads((d/'child.json').read_text())
            assert locks.identity(child['pid'])['start_ticks']==child['start_ticks']
            assert not (d/'child-finished').exists();assert worker_process.poll() is None
            blocked(paths)
            held=json.loads((d/'wrapper/production-held.json').read_text())
            proof=[locks.fdinfo(worker_process.pid,row['fd']) for row in held['locks']]
            assert all(row['kernel_lock'] in raw.splitlines() for row,raw in zip(held['locks'],proof))
            (d/'pending-locks.json').write_text(json.dumps(dict(held=held,fdinfo=proof,child=child,wrapper=locks.identity(worker_process.pid))))
            rc=worker_process.wait(timeout=10)
            assert (d/'child-finished').exists()
            for p in paths:
                with p.open('r+') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
            events=[json.loads(l) for l in (d/'wrapper/events.jsonl').read_text().splitlines()]
            assert events[-1]['kind']=='released'
            cleanup=[e for e in events if e['kind']=='cleanup-confirmed'][-1]
            assert cleanup['cleanup']['command_reaped'] is True
            result[mode]=dict(worker_rc=rc,held_while_child_alive=True,released_after_proven_cleanup=True)
    (root/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
    mode=sys.argv[1];d=Path(sys.argv[2])
    if mode=='controls':controls(d)
    elif mode=='child':
        (d/'child.json').write_text(json.dumps(locks.identity(os.getpid())));time.sleep(2);(d/'child-finished').write_text('natural exit')
    else:worker(mode,d)
