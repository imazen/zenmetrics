#!/usr/bin/env python3
"""Small real-cgroup proof: a child OOM leaves the aggregate and science intact."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from native_scope_limits import member, scope_path


def probe(runtime):
    scope = scope_path(runtime)
    owner = int(os.environ['SUDO_UID'])
    results = []
    for name, bound, allocation, expected in (
            ('proof-4G', 4*1024**3, 1024**2, 0),
            ('proof-small', 16*1024**2, 32*1024**2, -9)):
        leaf = scope/name
        leaf.mkdir()
        (leaf/'memory.max').write_text(str(bound))
        (leaf/'memory.swap.max').write_text('0')
        def enter():
            (leaf/'cgroup.procs').write_text(str(os.getpid()))
            os.setgid(int(os.environ['SUDO_GID']))
            os.setuid(owner)
        rc = subprocess.run([sys.executable, '-c',
                             f'x=bytearray({allocation}); x[-1]=1'],
                            preexec_fn=enter, check=False).returncode
        values={k:(leaf/k).read_text().strip() for k in
                ('memory.max','memory.swap.max','memory.peak','memory.events')}
        assert rc == expected, (name, rc, values)
        assert values['memory.max'] == str(bound)
        assert values['memory.swap.max'] == '0'
        if expected == -9:
            assert 'oom_kill 1' in values['memory.events']
        results.append(dict(child=str(leaf),exit_code=rc,**values))
    assert member(int((runtime/'batch.pid').read_text()),scope/'science')
    assert (runtime/'slots').read_text().strip() == '6'
    assert (scope/'memory.events').read_text().split('oom_kill ')[1].split()[0] == '1'
    return dict(observed_at=datetime.now(timezone.utc).isoformat(), passed=True,
                aggregate_cap_bytes=int((scope/'memory.max').read_text()),
                aggregate_swap_max_bytes=int((scope/'memory.swap.max').read_text()),
                science_slots_unchanged=6,children=results)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--inside',action='store_true')
    a=p.parse_args()
    if a.inside:
        print(json.dumps(probe(a.runtime)))
        return
    a.runtime.mkdir()
    (a.runtime/'scope.cgroup').write_text(Path('/proc/self/cgroup').read_text())
    (a.runtime/'batch.pid').write_text(str(os.getpid()))
    (a.runtime/'slots').write_text('6\n')
    unit=Path(Path('/proc/self/cgroup').read_text().strip().split('::',1)[1]).name
    for prop, expected in (('Delegate','yes'),('OOMPolicy','continue')):
        assert subprocess.check_output(['systemctl','--user','show',unit,'-p',prop,'--value'],text=True).strip()==expected
    subprocess.run(['sudo','-n',sys.executable,str(Path(__file__).with_name('native_scope_limits.py')),
                    '--runtime',str(a.runtime)],check=True)
    raw=subprocess.check_output(['sudo','-n',sys.executable,str(Path(__file__).resolve()),
                    '--runtime',str(a.runtime),'--out',str(a.out),'--inside'],text=True)
    report=json.loads(raw)
    with a.out.open('x') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
