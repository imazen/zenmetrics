"""Hold actual existing admission locks across run-heavy and verified child cleanup."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
import production_locks as locks

def session_members(sid):
    members=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():continue
        try:
            value=locks.identity(int(entry.name))
            if value['sid']==sid:members.append(value)
        except (FileNotFoundError,ProcessLookupError):pass
    return members

def await_exit(recorded,events):
    # PID/start identity is retained. Never signal a reused PID or foreign task.
    while True:
        try:current=locks.identity(recorded['pid'])
        except (FileNotFoundError,ProcessLookupError):return
        if current['start_ticks']!=recorded['start_ticks']:return
        locks.event(events,'cleanup-wait',identity=current)
        time.sleep(1)

def run(inventory,output,command,control_output=None):
    expected=json.loads(inventory.read_text())['locks'];manager=locks.ExistingLocks(expected,output)
    held=manager.acquire();receipt=output/'production-held.json';receipt.write_text(json.dumps(held,indent=2))
    env=os.environ.copy();env['CLEAN_PANEL_PRODUCTION_RECEIPT']=str(receipt);env['ZENBENCH_LAUNCHER_PIDS']=str(os.getpid())
    outer=dict(wrapper=locks.identity(os.getpid()),inventory_sha256=hashlib.sha256(inventory.read_bytes()).hexdigest(),command=command,production_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
    locks.event(output/'events.jsonl','command-launch',command=command,owner=outer['wrapper'])
    with (output/'command.log').open('wb') as stream:
        child=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        outer['command_child']=locks.identity(child.pid);observed={}
        while child.poll() is None:
            for row in session_members(child.pid):observed[(row['pid'],row['start_ticks'])]=row
            time.sleep(.05)
        rc=child.wait()
        for row in session_members(child.pid):observed[(row['pid'],row['start_ticks'])]=row
        for row in observed.values():await_exit(row,output/'events.jsonl')
        outer['observed_session_members_exited']=list(observed.values())
    outer['returncode']=rc;outer['command_reaped_ns']=time.monotonic_ns()
    # The outer run-heavy process is reaped. On control success also verify
    # every retained wrapper/controller chain member and the two owned workers.
    checked=[]
    if control_output is not None and rc==0:
        control=json.loads((control_output/'outer.json').read_text())
        for witness in (control['production_before'],control['production_after']):
            if witness['held_sha256']!=outer['production_receipt_sha256']:raise ValueError('production receipt not joined')
        identities=control['production_after']['chain'][:-1]+[control['rival'],control['launch']['child']]
        for recorded in identities:
            try:current=locks.identity(recorded['pid'])
            except FileNotFoundError:current=None
            if current is not None and current['start_ticks']==recorded['start_ticks']:await_exit(recorded,output/'events.jsonl')
            checked.append(recorded)
        outer['control_outer_sha256']=hashlib.sha256((control_output/'outer.json').read_bytes()).hexdigest()
        outer['descendants_confirmed_exited']=checked
    outer['before_release']=locks.witness(receipt)
    (output/'wrapper.json').write_text(json.dumps(outer,indent=2))
    manager.release(dict(command_reaped=True,returncode=rc,retained_descendants_exited=checked,monitor_control_complete=control_output is not None and rc==0))
    return rc
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('inventory',type=Path);p.add_argument('output',type=Path);p.add_argument('--control-output',type=Path);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();command=a.command
    if command and command[0]=='--':command=command[1:]
    if not command or Path(command[0]).name!='run-heavy':raise SystemExit('normal run-heavy command required')
    raise SystemExit(run(a.inventory,a.output,command,a.control_output))
