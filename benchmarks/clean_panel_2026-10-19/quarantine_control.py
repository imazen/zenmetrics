"""Owned fault-injection child for test_repair2; never a scorer."""
import errno
import json
import os
from pathlib import Path
import signal
import sys
from unittest.mock import patch
import containment
import panel
from test_repairs import fixture


def main():
    root=Path(sys.argv[1]);kind=sys.argv[2]
    (root/'tmp').mkdir()
    child_source=root/'owned_child.py'
    child_source.write_text('import os,signal,time,json,sys\nfrom pathlib import Path\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\ns=Path("/proc/self/stat").read_text();fields=s[s.rindex(")")+2:].split()\nPath(sys.argv[1]).write_text(json.dumps(dict(pid=os.getpid(),start=int(fields[19]))))\ntime.sleep(30)\n')
    m=fixture(root)
    original_open=containment.os.pidfd_open
    original_send=containment.signal.pidfd_send_signal
    original_processes=containment.processes
    original_write=containment.write
    def broken():
        if (root/'child.json').exists() and (root/'fault').exists():
            (root/'fault-observed').write_text(kind)
            return True
        return False
    def pidfd(pid,*a,**kw):
        if pid!=os.getpid() and kind=='pidfd' and broken():raise OSError(errno.EMFILE,'owned injected pidfd exhaustion')
        return original_open(pid,*a,**kw)
    def send(fd,sig,*a,**kw):
        if sig and kind=='send' and broken():raise OSError(errno.EPERM,'owned injected signal failure')
        return original_send(fd,sig,*a,**kw)
    def processes():
        if kind=='refresh' and broken():raise OSError(errno.EIO,'owned injected process observation failure')
        return original_processes()
    def write(path,value):
        if path.name=='owned.containment.json' and kind=='write' and broken():raise OSError(errno.ENOSPC,'owned injected receipt write failure')
        return original_write(path,value)
    def arm(name,manifest,bid,out):
        with (out/'owned.log').open('w') as log:
            containment.execute([sys.executable,str(child_source),str(root/'child.json')],log,os.environ.copy(),out/'owned.containment.json',timeout=.25,grace=.1)
        raise TimeoutError('owned negative control completed cleanup')
    with patch.object(panel.Path,'home',return_value=root),patch.object(panel,'run_arm',side_effect=arm),patch.object(containment.os,'pidfd_open',side_effect=pidfd),patch.object(containment.signal,'pidfd_send_signal',side_effect=send),patch.object(containment,'processes',side_effect=processes),patch.object(containment,'write',side_effect=write):
        blocks=panel.acquire(m,root/'output',1)
    (root/'done.json').write_text(json.dumps(blocks))


if __name__=='__main__':main()
