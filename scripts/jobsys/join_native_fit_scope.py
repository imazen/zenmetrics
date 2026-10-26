#!/usr/bin/env python3
"""Admit a bounded native worker launcher into an existing capped, locked scope.

Linux pidfd_getfd shares the original lock's open file description. Keeping it
open here retains heavy admission until this launcher and its children finish,
even if the initial launcher exits. Cell claiming stays in zenfleet-worker.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import pwd
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--runtime', type=Path, required=True)
p.add_argument('--lock-pid', type=int, required=True)
p.add_argument('--lock-fd', type=int, default=3)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
if os.geteuid() != 0 or not a.command or a.command[0] != '--':
    p.error('run through sudo with -- followed by the bounded native launcher')
owner = pwd.getpwuid(int(os.environ['SUDO_UID']))
lock = Path(owner.pw_dir) / 'tmp/heavy.lock'
if Path(f'/proc/{a.lock_pid}/fd/{a.lock_fd}').resolve() != lock.resolve():
    raise ValueError('recorded descriptor is not this owner\'s heavy lock')
relative = (a.runtime/'scope.cgroup').read_text().strip().split('::', 1)[1]
cgroup = Path('/sys/fs/cgroup') / relative.lstrip('/')
if (cgroup/'memory.max').read_text().strip() != '45097156608' or (cgroup/'memory.swap.max').read_text().strip() != '0':
    raise ValueError('expected live 42GiB no-swap scope')
members = (cgroup/'cgroup.procs').read_text().split()
if str(int((a.runtime/'batch.pid').read_text())) not in members and str(a.lock_pid) not in members:
    raise ValueError('original batch no longer owns this scope')
# pidfd_getfd is syscall 438 on the x86_64 Linux worker host.
if os.uname().machine != 'x86_64':
    raise ValueError('native scope admission currently requires x86_64 Linux')
handle = os.pidfd_open(a.lock_pid)
libc = ctypes.CDLL(None, use_errno=True)
shared_fd = libc.syscall(438, handle, a.lock_fd, 0)
if shared_fd < 0:
    raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
os.close(handle)
fcntl.flock(shared_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
(cgroup/'cgroup.procs').write_text(str(os.getpid()))
if Path('/proc/self/cgroup').read_text().strip() != '0::'+relative:
    raise ValueError('failed to join the original resource scope')
os.initgroups(owner.pw_name, owner.pw_gid)
os.setgid(owner.pw_gid)
os.setuid(owner.pw_uid)
os.nice(19)
os.environ.update(HOME=owner.pw_dir, USER=owner.pw_name, TMPDIR=owner.pw_dir+'/tmp',
                  PATH=owner.pw_dir+'/.local/bin:/usr/local/bin:/usr/bin:/bin')
report = dict(observed_at=datetime.now(timezone.utc).isoformat(), scope=str(cgroup), pid=os.getpid(),
              original_lock_pid=a.lock_pid, shared_lock_descriptor=True, shared_lock_fd=shared_fd, command=a.command[1:])
print(json.dumps(report), flush=True)
rc = 1
try:
    rc = subprocess.run(['ionice', '-c', '3', *a.command[1:]], check=False).returncode
finally:
    report.update(exit_code=rc, finished_at=datetime.now(timezone.utc).isoformat(),
                  scope_memory={k:(cgroup/k).read_text().strip() for k in
                                ('memory.current','memory.peak','memory.max','memory.events')})
    out = a.runtime/f'admission-{os.getpid()}-{time.time_ns()}.json'
    with out.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    os.close(shared_fd)
raise SystemExit(rc)
