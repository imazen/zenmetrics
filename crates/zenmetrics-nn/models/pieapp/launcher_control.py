#!/usr/bin/env python3
"""Verify immutable workflow launch against an in-flight file-edit control."""
from pathlib import Path
import select
import shlex
import subprocess
import tempfile


def run(command, fixture):
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True) as child:
        ready, _, _ = select.select([child.stdout], [], [], 10)
        assert ready, 'launcher did not reach the controlled input barrier'
        assert child.stdout.readline() == 'ready\n'
        fixture.write_text("printf 'unterminated\n")
        out, err = child.communicate('continue\n', timeout=10)
        return child.returncode, out, err


def main():
    shown = subprocess.run(['just', '--show', 'nn-pieapp-finish'],
                           capture_output=True, text=True, check=True).stdout
    launch = next(line.strip() for line in shown.splitlines()
                  if line.strip().startswith('bash -c '))
    source_path = 'benchmarks/pieapp_finish_2026-10-16.sh'
    assert source_path in launch
    payload = "printf 'ready\\n'\nread -r token\n#" + 'padding ' * 6000 + "\nprintf 'original-tail\\n'\n"
    scratch = Path.home() / 'tmp'
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pieapp-launch-control-', dir=scratch) as work:
        legacy = Path(work) / 'legacy.sh'
        frozen = Path(work) / 'frozen.sh'
        legacy.write_text(payload)
        frozen.write_text(payload)
        old = run(['bash', str(legacy)], legacy)
        assert old[0] != 0 or old[1] != 'original-tail\n', old
        command = launch.replace(source_path, shlex.quote(str(frozen)))
        fixed = run(['bash', '-c', command], frozen)
        assert fixed == (0, 'original-tail\n', ''), fixed
        print('Legacy file reader loses its original tail after controlled mutation:', old)
        print('Actual immutable launcher preserves its original tail, rc=0')


if __name__ == '__main__':
    main()
