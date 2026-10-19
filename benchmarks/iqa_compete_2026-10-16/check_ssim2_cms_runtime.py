#!/usr/bin/env python3
"""Trace CMS entry points in the exact retained primary executable with GDB.

This diagnostic is untimed. It verifies returned scalar bits against the prior
grid; debugger stop/resume overhead is never used as metric timing evidence.
Caller supplies both host heavy locks, run-heavy and a complete log.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from octave_primary import sha
from ssim2_primary_panel import PRIMARY_SHA


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['canonical', 'panel', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    assert sha(a.canonical) == PRIMARY_SHA
    identity = json.loads((a.panel/'identity.json').read_text())
    assert identity['canonical_sha256'] == PRIMARY_SHA
    cases = json.loads((a.panel/'completion.json').read_text())['cases']
    assert len(cases) == 24
    a.output.mkdir()
    rows = []
    script = '''set pagination off
set confirm off
set debuginfod enabled off
python
import gdb, json
counts = {}
class CountEntry(gdb.Breakpoint):
    def __init__(self, label, spec):
        self.label = label
        counts[label] = 0
        super().__init__(spec, internal=True)
    def stop(self):
        counts[self.label] += 1
        return False
CountEntry('JxlCmsInit', "'jxl::(anonymous namespace)::JxlCmsInit(void*, unsigned long, unsigned long, JxlColorProfile const*, JxlColorProfile const*, float)'")
CountEntry('cmsCreateTransformTHR', 'cmsCreateTransformTHR')
for name in ['cmsDoTransform', 'cmsDoTransformStride', 'cmsDoTransformLineStride']:
    CountEntry(name, name)
end
run
python
print('CMS_COUNTS '+json.dumps(counts, sort_keys=True))
print('INFERIOR_EXIT '+str(gdb $_exitcode))
end
'''
    # GDB's Python cannot use convenience variables as Python expressions.
    script = script.replace('str(gdb $_exitcode)', 'str(gdb.parse_and_eval("$_exitcode"))')
    (a.output/'trace.gdb').write_text(script)
    env = dict(os.environ, LC_ALL='C')
    for case in cases:
        args = case['canonical_command']
        start = args.index(str(a.canonical))
        argv = args[start:]
        directory = a.output/f"{case['pair']}-{case['size']}-t{case['threads']}"
        directory.mkdir()
        for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS']:
            env[key] = str(case['threads'])
        command = ['timeout','--kill-after=20s','120s','gdb','--batch','--quiet','-x',str((a.output/'trace.gdb').resolve()),'--args',*argv]
        with (directory/'stdout.log').open('x') as stdout, (directory/'stderr.log').open('x') as stderr:
            result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
        text = (directory/'stdout.log').read_text().splitlines()
        assert result.returncode == 0 and 'INFERIOR_EXIT 0' in text
        counts = [json.loads(s.removeprefix('CMS_COUNTS ')) for s in text if s.startswith('CMS_COUNTS ')]
        values = [float(s.split()[2]) for s in text if s.startswith(('COLD ','SAMPLE '))]
        assert len(counts) == 1 and len(values) == 2 and all(v == case['canonical'] for v in values)
        assert counts[0]['JxlCmsInit'] > 0 and counts[0]['cmsCreateTransformTHR'] > 0
        assert all(counts[0][name] == 0 for name in ['cmsDoTransform','cmsDoTransformStride','cmsDoTransformLineStride'])
        row = dict(pair=case['pair'],size=case['size'],threads=case['threads'],counts=counts[0],scores=values,
                   command=command,stdout_sha256=sha(directory/'stdout.log'),stderr_sha256=sha(directory/'stderr.log'))
        rows.append(row)
        (directory/'result.json').write_text(json.dumps(row,indent=2)+'\n')
        print('CMS_CASE',directory.name,json.dumps(counts[0]),flush=True)
        assert sha(a.canonical) == PRIMARY_SHA
    output = dict(canonical_sha256=PRIMARY_SHA,panel_identity_sha256=sha(a.panel/'identity.json'),
                  panel_completion_sha256=sha(a.panel/'completion.json'),script_sha256=sha(a.output/'trace.gdb'),
                  worker_sha256=sha(Path(__file__)),gdb_version=subprocess.check_output(['gdb','--version'],text=True),
                  cases=rows,qualified_timing_cells=0)
    (a.output/'completion.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':
    main()
