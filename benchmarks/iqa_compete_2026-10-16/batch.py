"""Optional bounded lock holds spanning several separately supervised cells."""
import json,math,os,pathlib,time

class BatchLease:
    def __init__(self,path,seconds,events,reserve=120):
        if not 1<=seconds<=1100:raise ValueError('batch hold must be within 1..1100 seconds')
        self.path=pathlib.Path(path);self.seconds=seconds;self.events=pathlib.Path(events)
        self.reserve=min(reserve,seconds);self.fd=None;self.started=None;self.calls=0
    def prepare(self,scope):
        import fcntl
        if self.fd is not None and self.remaining()<self.reserve:self.close()
        if self.fd is None:
            self.fd=os.open(self.path,os.O_CREAT|os.O_RDWR,0o600)
            try:fcntl.flock(self.fd,fcntl.LOCK_EX)
            except BaseException:self.close();raise
            self.started=time.monotonic();self.calls=0
            self.event('acquired')
        fresh=self.calls==0;self.calls+=1
        return max(1,min(scope,self.remaining())),fresh
    def remaining(self):
        return max(0,math.floor(self.seconds-(time.monotonic()-self.started)))
    def event(self,kind):
        with self.events.open('a') as f:
            f.write(json.dumps(dict(event=kind,pid=os.getpid(),configured_hold_seconds=self.seconds,
                observed_parent_hold_seconds=time.monotonic()-self.started,cells_submitted=self.calls))+'\n')
            f.flush();os.fsync(f.fileno())
    def close(self):
        if self.fd is not None:
            # Close our reference, never explicitly unlock the shared OFD.
            # A supervised child inheriting it keeps the lock if its parent dies.
            fd=self.fd;self.fd=None
            try:
                if self.started is not None:self.event('parent_reference_closed')
            finally:
                os.close(fd);self.started=None
