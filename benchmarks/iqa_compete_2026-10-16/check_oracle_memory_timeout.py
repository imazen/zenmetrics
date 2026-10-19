#!/usr/bin/env python3
"""Real-child deadline control: no metric, weights, input images or scoring."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from supervision import scoring_command, timeout_command

CHILD = '''import os,signal,time,json,sys
from pathlib import Path
signal.signal(signal.SIGTERM,signal.SIG_IGN)
pid=os.getpid()
fields=Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()
Path(sys.argv[1]).write_text(json.dumps({"pid":pid,"start_ticks":fields[19]}))
time.sleep(60)
'''


def live(identity):
    path = Path('/proc') / str(identity['pid'])
    if not path.exists():
        return False
    fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    if fields[19] != identity['start_ticks']:
        raise RuntimeError('owned PID was reused')
    return fields[0] != 'Z'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('evidence', type=Path)
    a = p.parse_args()
    a.evidence.mkdir()
    root = a.evidence.resolve()
    child = root / 'owned_child.py'
    child.write_text(CHILD)
    records = []
    for name, positive in [('deliberately-wrong-order', False), ('production-order', True)]:
        marker = root / (name + '.pid.json')
        scorer = [sys.executable, str(child), str(marker)]
        memory = root / (name + '.memory')
        command = (scoring_command(scorer, system='Linux', seconds=1, memory_log=memory)
                   if positive else [*timeout_command('Linux', 1), '/usr/bin/time', '-v', '-o', str(memory), *scorer])
        identity = None
        start = time.monotonic()
        with (root / (name + '.stdout')).open('x') as stdout, (root / (name + '.stderr')).open('x') as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            try:
                rc = process.wait(timeout=35)
                identity = json.loads(marker.read_text())
                was_live = live(identity)
                record = {'case': name, 'command': command, 'rc': rc,
                          'elapsed_seconds': time.monotonic() - start,
                          'child_live_when_parent_returned': was_live, 'owned_child': identity}
                records.append(record)
                print(json.dumps(record), flush=True)
                if positive:
                    assert not was_live, 'scorer outlived production deadline wrapper'
                else:
                    assert was_live, 'negative control did not exercise the orphan class'
            finally:
                if identity is None and marker.exists():
                    identity = json.loads(marker.read_text())
                if identity is not None and live(identity):
                    os.kill(identity['pid'], signal.SIGKILL)
                if process.poll() is None:
                    process.kill()
                    process.wait()
        if identity is not None:
            for _ in range(50):
                if not live(identity):
                    break
                time.sleep(0.02)
            assert not live(identity), 'owned stand-in cleanup did not finish'
    (root / 'results.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
