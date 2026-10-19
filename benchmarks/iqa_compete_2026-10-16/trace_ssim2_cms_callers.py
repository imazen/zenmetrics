#!/usr/bin/env python3
"""Observe CMS callers after the retained zero-transform hypothesis failed.

This is a separate, untimed call-stack observation. It leaves the failed
check_ssim2_cms_runtime.py assertion and every numerical bound unchanged.
Caller supplies both heavy locks, run-heavy and a complete log.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from octave_primary import sha
from ssim2_primary_panel import PRIMARY_SHA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['canonical', 'panel', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert sha(args.canonical) == PRIMARY_SHA
    identity = json.loads((args.panel / 'identity.json').read_text())
    assert identity['canonical_sha256'] == PRIMARY_SHA
    cases = json.loads((args.panel / 'completion.json').read_text())['cases']
    case = next(c for c in cases if c['pair'] == 'pairA' and c['size'] == '64' and c['threads'] == 1)
    args.output.mkdir()
    root = args.output.resolve()
    script = '''set pagination off
set confirm off
set debuginfod enabled off
python
import gdb, json
stacks = {}
class CmsCaller(gdb.Breakpoint):
    def __init__(self, name):
        self.label = name
        super().__init__(name, internal=True)
    def stop(self):
        frame = gdb.newest_frame()
        names = []
        for i in range(16):
            if frame is None: break
            names.append(frame.name() or '<unknown>')
            frame = frame.older()
        key = json.dumps(names)
        stacks[key] = stacks.get(key, 0) + 1
        return False
for name in ['cmsDoTransform', 'cmsDoTransformStride', 'cmsDoTransformLineStride']:
    CmsCaller(name)
end
run
python
print('CMS_CALLERS '+json.dumps([{'frames':json.loads(k),'calls':v} for k,v in stacks.items()], sort_keys=True))
print('INFERIOR_EXIT '+str(gdb.parse_and_eval('$_exitcode')))
end
'''
    (root / 'trace.gdb').write_text(script)
    command = case['canonical_command']
    argv = command[command.index(str(args.canonical)):]
    command = ['timeout', '--kill-after=20s', '120s', 'gdb', '--batch', '--quiet',
               '-x', str(root / 'trace.gdb'), '--args', *argv]
    env = dict(os.environ, LC_ALL='C')
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
        env[name] = '1'
    with (root / 'stdout.log').open('x') as stdout, (root / 'stderr.log').open('x') as stderr:
        result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
    lines = (root / 'stdout.log').read_text().splitlines()
    assert result.returncode == 0 and 'INFERIOR_EXIT 0' in lines
    scores = [float(line.split()[2]) for line in lines if line.startswith(('COLD ', 'SAMPLE '))]
    assert len(scores) == 2 and all(score == case['canonical'] for score in scores)
    records = [json.loads(line.removeprefix('CMS_CALLERS ')) for line in lines if line.startswith('CMS_CALLERS ')]
    assert len(records) == 1
    assert sha(args.canonical) == PRIMARY_SHA
    output = dict(case={key: case[key] for key in ['pair', 'size', 'threads', 'canonical']},
                  stacks=records[0], scores=scores, command=command,
                  canonical_sha256=PRIMARY_SHA, worker_sha256=sha(Path(__file__)),
                  panel_identity_sha256=sha(args.panel / 'identity.json'),
                  panel_completion_sha256=sha(args.panel / 'completion.json'),
                  script_sha256=sha(root / 'trace.gdb'),
                  stdout_sha256=sha(root / 'stdout.log'), stderr_sha256=sha(root / 'stderr.log'),
                  qualified_timing_cells=0)
    (root / 'result.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
