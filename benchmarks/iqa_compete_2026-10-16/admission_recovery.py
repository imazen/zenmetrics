"""Preserve a killed benchmark's own reservation after verifying it has no work."""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import subprocess
import sys


def exited(pid, pgid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    except PermissionError:
        return False
    else:
        return False
    if sys.platform == 'darwin':
        text = subprocess.check_output(['ps', '-axo', 'pid=,pgid=,stat='], text=True)
        return not any(int(group) == pgid and not state.startswith('Z')
                       for _, group, state in (line.split()[:3] for line in text.splitlines()))
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid:
                return False
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            return False
    return True


def recover(proof, output, cache=None):
    import fcntl
    pid, pgid, started = (proof[key] for key in ['pid', 'pgid', 'unix_s'])
    if not all(type(v) is int and v > 0 for v in [pid, pgid, started]):
        raise ValueError('invalid process proof')
    cache = cache or Path.home() / '.cache/zenbench/admission-v1'
    recovered = []
    for ticket in cache.glob('*.json'):
        original = ticket.read_bytes()
        record = json.loads(original)
        if (record.get('pid') != pid or record.get('activity') != 'benchmark'
                or record.get('running') is not True):
            continue
        stamp = record.get('start')
        if type(stamp) is not int or not started - 120 <= stamp <= int(time.time()):
            continue
        if not exited(pid, pgid):
            continue
        with ticket.with_suffix('.guard').open('a+') as guard:
            try:
                fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if not ticket.exists() or ticket.read_bytes() != original or not exited(pid, pgid):
                continue
            output.mkdir(parents=True, exist_ok=True)
            archived = output / f'{ticket.stem}-{uuid.uuid4().hex}.json'
            with archived.open('xb') as saved:
                saved.write(original)
            preserved = ticket.with_name(ticket.name + '.bak-iqa-' + uuid.uuid4().hex)
            ticket.rename(preserved)
            recovered.append(dict(process=proof, sequence=record.get('sequence'),
                                  original_sha256=hashlib.sha256(original).hexdigest(),
                                  archive=str(archived), preserved=str(preserved),
                                  verification='owner absent, no process-group members, guard acquired'))
    return recovered
