"""Hold existing admission locks; quarantine on uncertain post-launch cleanup."""
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
    while True:
        try:current=locks.identity(recorded['pid'])
        except (FileNotFoundError,ProcessLookupError):return
        if current['start_ticks']!=recorded['start_ticks']:return
        locks.event(events,'cleanup-wait',identity=current);time.sleep(.1)

def quarantine(output,error):
    # Logging itself may be the fault. Never let that release admission.
    try:locks.event(output/'events.jsonl','quarantined',error=repr(error))
    except BaseException:pass
    try:time.sleep(.1)
    except BaseException:pass

def run(inventory,output,command,control_output=None):
    expected=json.loads(inventory.read_text())['locks'];manager=locks.ExistingLocks(expected,output)
    held=manager.acquire();receipt=output/'production-held.json';receipt.write_text(json.dumps(held,indent=2))
    env=os.environ.copy();env['CLEAN_PANEL_PRODUCTION_RECEIPT']=str(receipt);env['ZENBENCH_LAUNCHER_PIDS']=str(os.getpid())
    outer=dict(wrapper=locks.identity(os.getpid()),inventory_sha256=hashlib.sha256(inventory.read_bytes()).hexdigest(),command=command,production_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
    locks.event(output/'events.jsonl','command-launch',command=command,owner=outer['wrapper'])
    stream=(output/'command.log').open('wb')
    try:child=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    except BaseException as error:
        # Popen may have started a child before an interruption. Without a
        # returned handle/identity, absence of execution cannot be proven.
        while True:quarantine(output,error)
    # From this point every exception is contained while BOTH descriptors stay
    # live. A persistent inventory/receipt fault deliberately quarantines this
    # wrapper even after direct-child exit. No timeout authorizes release.
    observed={};child_identity=None;faults=False
    while True:
        try:
            if child_identity is None:
                child_identity=locks.identity(child.pid);outer['command_child']=child_identity
            for row in session_members(child.pid):observed[(row['pid'],row['start_ticks'])]=row
            if child.poll() is None:time.sleep(.05);continue
            rc=child.wait();outer['returncode']=rc;outer['command_reaped_ns']=time.monotonic_ns()
            for row in session_members(child.pid):observed[(row['pid'],row['start_ticks'])]=row
            for row in observed.values():await_exit(row,output/'events.jsonl')
            # Fresh successful enumeration after all retained identities exit.
            if session_members(child.pid):raise ValueError('owned session not empty')
            outer['observed_session_members_exited']=list(observed.values())
            checked=[]
            if control_output is not None and rc==0:
                control=json.loads((control_output/'outer.json').read_text())
                for witness in (control['production_before'],control['production_after']):
                    if witness['held_sha256']!=outer['production_receipt_sha256']:raise ValueError('production receipt not joined')
                identities=control['production_after']['chain'][:-1]+[control['rival'],control['launch']['child']]
                for row in identities:await_exit(row,output/'events.jsonl');checked.append(row)
                outer['control_outer_sha256']=hashlib.sha256((control_output/'outer.json').read_bytes()).hexdigest()
                outer['descendants_confirmed_exited']=checked
            outer['before_release']=locks.witness(receipt)
            outer['quarantine_observed']=faults
            (output/'wrapper.json').write_text(json.dumps(outer,indent=2))
            cleanup=dict(command_reaped=True,returncode=rc,retained_descendants_exited=checked,monitor_control_complete=control_output is not None and rc==0)
            # Persist all fallible evidence BEFORE touching either flock.
            manager.prepare_release(cleanup)
            break
        except BaseException as error:
            faults=True;quarantine(output,error)
    manager.release_prepared();stream.close()
    return rc
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('inventory',type=Path);p.add_argument('output',type=Path);p.add_argument('--control-output',type=Path);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();command=a.command
    if command and command[0]=='--':command=command[1:]
    if not command or Path(command[0]).name!='run-heavy':raise SystemExit('normal run-heavy command required')
    raise SystemExit(run(a.inventory,a.output,command,a.control_output))
