"""Existing-only Linux admission; receipts state owner release, never global freedom."""
import fcntl,hashlib,json,os,stat,time
from pathlib import Path
import admission_receipt

def identity(pid):
    value=admission_receipt.process_identity(pid)
    value['cgroup']=Path(f'/proc/{pid}/cgroup').read_text()
    value['argv']=[p.decode() for p in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if p]
    return value

def inspect(path):
    path=Path(path).absolute()
    if path.resolve(strict=True)!=path:raise ValueError('symlink/noncanonical lock path')
    s=path.lstat()
    if not stat.S_ISREG(s.st_mode):raise ValueError('not a regular existing lock')
    return dict(path=str(path),device=s.st_dev,inode=s.st_ino)

def fdinfo(pid,fd):return Path(f'/proc/{pid}/fdinfo/{fd}').read_text()
def event(path,kind,**data):
    with Path(path).open('a') as f:
        f.write(json.dumps(dict(kind=kind,monotonic_ns=time.monotonic_ns(),**data))+'\n');f.flush();os.fsync(f.fileno())

class ExistingLocks:
    def __init__(self,expected,output):
        if len(expected)!=2 or len({(r['device'],r['inode']) for r in expected})!=2:raise ValueError('two distinct existing locks required')
        self.expected=expected;self.output=Path(output);self.output.mkdir(exist_ok=False);self.fds=[];self.owner=identity(os.getpid());self.released=False
        for r in expected:
            if inspect(r['path'])!=r:raise ValueError('lock identity differs from live inventory')
    def acquire(self):
        try:
            for i,r in enumerate(self.expected):
                fd=os.open(r['path'],os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC)
                stream=os.fdopen(fd,'r+');self.fds.append((stream,Path(r['path'])))
                actual=os.fstat(fd)
                if (actual.st_dev,actual.st_ino)!=(r['device'],r['inode']) or inspect(r['path'])!=r:raise ValueError('opened lock mismatch')
                event(self.output/'events.jsonl','waiting',order=i,identity=r)
                # Normal blocking flock: never interrupt another lock owner.
                fcntl.flock(fd,fcntl.LOCK_EX)
                if inspect(r['path'])!=r:raise ValueError('lock replaced while queued')
                proof=admission_receipt.lock_record(stream,r['path'],i)
                event(self.output/'events.jsonl','acquired',order=i,proof=proof,owner=self.owner)
            held=admission_receipt.Held(self.fds,self.output/'held')
            self.held=json.loads((self.output/'held/held.json').read_text())
            self.held['owner_cgroup']=self.owner['cgroup']
            return self.held
        except BaseException:
            # No child can have launched from acquire; release partial admission.
            self.release(dict(no_child_launched=True));raise
    def prepare_release(self,cleanup):
        if self.released:raise ValueError('duplicate release')
        # All pathname/kernel/receipt operations that can fail are attempted
        # while both locks remain held. No release until caller proves cleanup.
        for i,(stream,path) in enumerate(self.fds):
            if inspect(path)!=self.expected[i]:raise ValueError('release path changed')
            admission_receipt.lock_record(stream,path,i)
        event(self.output/'events.jsonl','cleanup-confirmed',cleanup=cleanup)
        self.prepared=True
    def release(self,cleanup):
        # Acquisition failures have no child; retain the existing partial path.
        if cleanup.get('no_child_launched') is True:
            event(self.output/'events.jsonl','cleanup-confirmed',cleanup=cleanup)
            self.prepared=True
        else:self.prepare_release(cleanup)
        self.release_prepared()
    def release_prepared(self):
        if self.released or not getattr(self,'prepared',False):raise ValueError('unprepared release')
        for i,(stream,path) in reversed(list(enumerate(self.fds))):
            before=fdinfo(os.getpid(),stream.fileno())
            fcntl.flock(stream,fcntl.LOCK_UN)
            after=fdinfo(os.getpid(),stream.fileno())
            if any(line.startswith('lock:') and 'FLOCK' in line for line in after.splitlines()):raise ValueError('own descriptor still flocked')
            event(self.output/'events.jsonl','owner-unlocked',order=i,fd=stream.fileno(),before_fdinfo=before,after_fdinfo=after,path_identity=inspect(path))
            stream.close()
        self.released=True
        event(self.output/'events.jsonl','released',owner=self.owner,claim='this owner released; another owner may now hold the locks')

def witness(receipt):
    held=json.loads(Path(receipt).read_text());owner=held['owner'];now=identity(owner['pid'])
    for key in ('pid','ppid','start_ticks','pgid','sid'):
        if now[key]!=owner[key]:raise ValueError('production wrapper identity changed')
    if now['cgroup']!=held['owner_cgroup']:raise ValueError('wrapper cgroup changed')
    proofs=[]
    for row in held['locks']:
        named=inspect(row['path'])
        if (named['device'],named['inode'])!=(row['device'],row['inode']):raise ValueError('production lock path identity changed')
        raw=fdinfo(owner['pid'],row['fd'])
        if row['kernel_lock'] not in raw.splitlines():raise ValueError('production wrapper no longer owns flock')
        proofs.append(raw)
    chain=[];pid=os.getpid()
    for _ in range(32):
        item=identity(pid);chain.append(item)
        if pid==owner['pid']:break
        pid=item['ppid']
    else:raise ValueError('controller not descendant of production owner')
    if chain[-1]['pid']!=owner['pid']:raise ValueError('missing wrapper ancestry')
    return dict(observed_monotonic_ns=time.monotonic_ns(),held_sha256=hashlib.sha256(Path(receipt).read_bytes()).hexdigest(),owner=now,fdinfo=proofs,chain=chain)
