"""Bind caller-held Linux flocks and owned launch. Never acquires/releases locks."""
import json
import os
from pathlib import Path
import time
import uuid


def process_identity(pid):
    raw=Path(f'/proc/{pid}/stat').read_text();fields=raw[raw.rindex(')')+2:].split()
    return dict(pid=pid,ppid=int(fields[1]),start_ticks=int(fields[19]),pgid=int(fields[2]),sid=int(fields[3]))


def write(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())


def lock_record(fd,path,index):
    stat=os.fstat(fd.fileno());named=Path(path).stat()
    if (stat.st_dev,stat.st_ino)!=(named.st_dev,named.st_ino):raise ValueError('lock pathname/inode mismatch')
    lines=Path(f'/proc/self/fdinfo/{fd.fileno()}').read_text().splitlines()
    proof=[line for line in lines if line.startswith('lock:')]
    # Linux fdinfo supplies the actual attached lock; no acquiring a missing
    # lock as a side effect and no inference from a filename or resource log.
    matches=[]
    for line in proof:
        fields=line.split()
        if 'FLOCK' in fields and 'WRITE' in fields:
            at=fields.index('WRITE')
            if fields[at+1]==str(os.getpid()) and fields[at+2].split(':')[-1]==str(stat.st_ino):matches.append(line)
    if len(matches)!=1:raise ValueError('caller exclusive flock proof missing')
    return dict(order=index,path=str(Path(path).absolute()),device=stat.st_dev,inode=stat.st_ino,fd=fd.fileno(),kernel_lock=matches[0])


class Held:
    def __init__(self,descriptors,directory):
        if len(descriptors)!=2:raise ValueError('two caller-held locks in fixed order required')
        self.descriptors=descriptors;self.directory=Path(directory);self.directory.mkdir(exist_ok=False)
        self.owner=process_identity(os.getpid());self.token=str(uuid.uuid4());self.launched=False
        self.records=[lock_record(fd,path,i) for i,(fd,path) in enumerate(descriptors)]
        if len({(r['device'],r['inode']) for r in self.records})!=2:raise ValueError('duplicate lock inode')
        write(self.directory/'held.json',dict(schema=1,owner=self.owner,token=self.token,locks=self.records,
              observed_monotonic_ns=time.monotonic_ns(),historical_proof=False))
    def bind_launch(self,argv,child_pid):
        if self.launched or not argv:raise ValueError('one launch per active receipt')
        if process_identity(os.getpid())!=self.owner:raise ValueError('controller identity changed')
        current=[lock_record(fd,path,i) for i,(fd,path) in enumerate(self.descriptors)]
        if current!=self.records:raise ValueError('lock identity changed')
        child=process_identity(child_pid)
        if child['ppid']!=self.owner['pid']:raise ValueError('not a direct owned child')
        executable=str(Path(f'/proc/{child_pid}/exe').resolve(strict=True))
        command=Path(f'/proc/{child_pid}/cmdline').read_bytes().split(b'\0')
        actual=[p.decode() for p in command if p]
        if actual!=list(argv) or process_identity(child_pid)!=child:raise ValueError('launch argv/identity changed')
        value=dict(schema=1,token=self.token,owner=self.owner,locks=current,argv=actual,child=child,
                   executable=executable,observed_monotonic_ns=time.monotonic_ns(),ownership_verified=True)
        write(self.directory/'launch.json',value);self.launched=True
        return value
