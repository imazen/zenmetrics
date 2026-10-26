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
from native_scope_limits import admission_budget, member, scope_path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--runtime', type=Path, required=True)
p.add_argument('--lock-pid', type=int, required=True)
p.add_argument('--lock-fd', type=int, default=3)
p.add_argument('--memory-max-bytes', type=int,
               help='serialize a postprocessor in a nested no-swap cgroup with this hard bound')
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
if os.geteuid() != 0 or not a.command or a.command[0] != '--':
    p.error('run through sudo with -- followed by the bounded native launcher')
owner = pwd.getpwuid(int(os.environ['SUDO_UID']))
lock = Path(owner.pw_dir) / 'tmp/heavy.lock'
if Path(f'/proc/{a.lock_pid}/fd/{a.lock_fd}').resolve() != lock.resolve():
    raise ValueError('recorded descriptor is not this owner\'s heavy lock')
cgroup = scope_path(a.runtime)
if not member(int((a.runtime/'batch.pid').read_text()), cgroup) and not member(a.lock_pid, cgroup):
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
target = cgroup / 'science' if (cgroup / 'science').is_dir() else cgroup
post_lock = None
if a.memory_max_bytes is not None:
    if a.memory_max_bytes <= 0 or a.memory_max_bytes > 12 * 1024**3:
        raise ValueError('postprocessor hard bound must be positive and at most 12GiB')
    if 'memory' not in (cgroup/'cgroup.subtree_control').read_text().split():
        raise ValueError('native batch must prepare nested memory control before joining')
    target = cgroup/f'postprocess-{os.getpid()}'
    target.mkdir()
    (target/'memory.max').write_text(str(a.memory_max_bytes))
    (target/'memory.swap.max').write_text('0')
    # Keep a waiting admission inside the aggregate too, with its own hard cap.
    (target/'cgroup.procs').write_text(str(os.getpid()))
    post_lock = os.open(a.runtime/'postprocess.lock', os.O_CREAT | os.O_RDWR, 0o600)
    os.fchown(post_lock, owner.pw_uid, owner.pw_gid)
    print(json.dumps(dict(postprocess_waiting=True, scope=str(target))), flush=True)
    fcntl.flock(post_lock, fcntl.LOCK_EX)
    while True:
        permitted, reserved = admission_budget(
            a.memory_max_bytes, int((a.runtime/'slots').read_text()),
            member(int((a.runtime/'batch.pid').read_text()), cgroup),
            int((cgroup/'science/memory.current').read_text()),
            int((cgroup/'memory.current').read_text()))
        if permitted:
            break
        print(json.dumps(dict(postprocess_waiting=True, reserved_science_bytes=reserved,
                              requested_bytes=a.memory_max_bytes)), flush=True)
        time.sleep(10)
else:
    (target/'cgroup.procs').write_text(str(os.getpid()))
if not member(os.getpid(), target):
    raise ValueError('failed to join the original resource scope')
os.initgroups(owner.pw_name, owner.pw_gid)
os.setgid(owner.pw_gid)
os.setuid(owner.pw_uid)
os.nice(19)
os.environ.update(HOME=owner.pw_dir, USER=owner.pw_name, TMPDIR=owner.pw_dir+'/tmp',
                  PATH=owner.pw_dir+'/.local/bin:/usr/local/bin:/usr/bin:/bin')
report = dict(observed_at=datetime.now(timezone.utc).isoformat(), scope=str(cgroup), pid=os.getpid(),
              original_lock_pid=a.lock_pid, shared_lock_descriptor=True, shared_lock_fd=shared_fd,
              child_scope=str(target), child_memory_max_bytes=(target/'memory.max').read_text().strip(),
              child_swap_max_bytes=(target/'memory.swap.max').read_text().strip(), command=a.command[1:])
print(json.dumps(report), flush=True)
rc = 1
try:
    rc = subprocess.run(['ionice', '-c', '3', *a.command[1:]], check=False).returncode
finally:
    report.update(exit_code=rc, finished_at=datetime.now(timezone.utc).isoformat(),
                  scope_memory={k:(cgroup/k).read_text().strip() for k in
                                ('memory.current','memory.peak','memory.max','memory.events')},
                  child_memory={k:(target/k).read_text().strip() for k in
                                ('memory.current','memory.peak','memory.max','memory.swap.max','memory.events')})
    out = a.runtime/f'admission-{os.getpid()}-{time.time_ns()}.json'
    with out.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    os.close(shared_fd)
    if post_lock is not None:
        os.close(post_lock)
raise SystemExit(rc)
