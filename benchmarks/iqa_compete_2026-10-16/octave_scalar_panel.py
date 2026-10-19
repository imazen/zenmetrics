#!/usr/bin/env python3
"""Primary MDSI scalar grid using unchanged author source through Octave.

Caller supplies both heavy locks and run-heavy around the complete grid.
Uses the existing raw-RGB, memory and frozen-probe conventions.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct

from haarpsi_primary import execute, write
from octave_primary import sha, quoted, process_max_rss

SOURCE_SHA = '4dff198b28e57b39b20877c7bcb75af2c6278c95f9f321362a2046ee3e37351b'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'inputs', 'binary', 'build-receipt', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    for name in ['source', 'inputs', 'binary', 'build_receipt']:
        setattr(a, name, getattr(a, name).resolve(strict=True))
    assert a.source.name == 'MDSI.m' and sha(a.source) == SOURCE_SHA
    receipt = json.loads(a.build_receipt.read_text())
    assert receipt['source_revision'] == 'e2d414c33a6d3c7da50a818fa36e56654d3ebeef'
    assert not receipt['adapters_applied'] and sha(a.binary) == receipt['binary_sha256']
    pairs = json.loads(a.inputs.read_text())['pairs']
    cap = max(1, len(os.sched_getaffinity(0)) - 4)
    a.output.mkdir()
    root = a.output.resolve()
    octave = Path(shutil.which('octave-cli')).resolve(strict=True)
    base = Path(__file__).resolve().parent
    helpers = {name: sha(base / name) for name in ['octave_scalar_panel.py', 'haarpsi_primary.py', 'octave_primary.py', 'supervision.py']}
    identity = dict(metric='mdsi', source_sha256=SOURCE_SHA, source=str(a.source), helpers=helpers,
                    origin='https://www.mathworks.com/matlabcentral/fileexchange/59809-mean-deviation-similarity-index-mdsi',
                    binary_sha256=receipt['binary_sha256'], build_receipt_sha256=sha(a.build_receipt),
                    octave_sha256=sha(octave), inputs_sha256=sha(a.inputs), threads=[1, cap],
                    relative_bound=1e-9, gate_source='crates/gmsd/README.md: MDSI', qualified_timing_cells=0,
                    map_status='author MDSI.m exposes only scalar Q; no synthetic map comparison')
    write(root / 'identity.json', identity)
    outcomes = []
    for row in pairs:
        paths = {role: (a.inputs.parent / row[role]).resolve(strict=True) for role in ['reference', 'distorted']}
        assert all(sha(path) == row[role + '_sha256'] for role, path in paths.items())
        w, h = row['width'], row['height']
        for threads in [1, cap]:
            assert sha(a.source) == SOURCE_SHA and sha(a.binary) == receipt['binary_sha256']
            assert all(sha(base / name) == value for name, value in helpers.items())
            directory = root / f'mdsi-{row["pair"]}-{row["size_name"]}-t{threads}'
            directory.mkdir()
            env = dict(os.environ, TMPDIR=str(directory), LC_ALL='C')
            for key in ['DISPLAY', 'WAYLAND_DISPLAY', 'XAUTHORITY']:
                env.pop(key, None)
            for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
                env[key] = str(threads)
            entry = directory / 'entry.m'
            entry.write_text(f"""more off;
pkg load image;
addpath({quoted(a.source.parent)},'-begin');
assert(strcmp(which('MDSI'),{quoted(a.source)}));
fprintf('VERSION octave-%s\\n',version());
for name={{'fspecial','conv2','mad','mean'}};fprintf('BIND %s %s\\n',name{{1}},which(name{{1}}));end
f=fopen({quoted(paths['reference'])},'rb');assert(f>=0);[r,n]=fread(f,Inf,'*uint8');assert(n=={w*h*3});assert(fclose(f)==0);
f=fopen({quoted(paths['distorted'])},'rb');assert(f>=0);[d,n]=fread(f,Inf,'*uint8');assert(n=={w*h*3});assert(fclose(f)==0);
r=permute(reshape(r,[3,{w},{h}]),[3,2,1]);
d=permute(reshape(d,[3,{w},{h}]),[3,2,1]);
for call=0:1
 clock=tic();score=MDSI(r,d);elapsed=toc(clock)*1e9;
 assert(isreal(score)&&isfinite(score));
 label='SAMPLE';if call==0;label='COLD';end
 fprintf('%s %.0f %.17g\\n',label,elapsed,score);fflush(stdout);
end
""")
            canonical_rc, canonical_command = execute([str(octave), '--no-gui', '--no-window-system', '--no-history',
                    '--no-init-file', '--quiet', str(entry)], env, directory, 'canonical', 300)
            config = dict(width=w, height=h, reference=str(paths['reference']), distorted=str(paths['distorted']),
                          metric_name='mdsi', threads=threads, probe_only=True, only_mode='score', timing_phase='warm')
            write(directory / 'config.json', config)
            rust_rc, rust_command = execute([str(a.binary), str(directory / 'config.json')], env, directory, 'rust', 120)
            result = dict(metric='mdsi', pair=row['pair'], size=row['size_name'], width=w, height=h, threads=threads,
                          canonical_returncode=canonical_rc, rust_returncode=rust_rc, canonical_command=canonical_command,
                          rust_command=rust_command, relative_bound=1e-9)
            if canonical_rc == rust_rc == 0:
                values = [float(s.split()[2]) for s in (directory / 'canonical-stdout.log').read_text().splitlines() if s.startswith(('COLD ', 'SAMPLE '))]
                probes = [json.loads(s.removeprefix('IQA_PROBE ')) for s in (directory / 'rust-stdout.log').read_text().splitlines() if s.startswith('IQA_PROBE ')]
                assert len(values) == 2 and values[0] == values[1] and len(probes) == 1 and probes[0]['metric_name'] == 'mdsi'
                assert len(probes[0]['scores']) == 1
                ours = struct.unpack('>d', bytes.fromhex(probes[0]['scores'][0][1]))[0]
                delta = max(abs(ours - value) for value in values)
                relative = max(abs(ours - value) / max(abs(value), 1e-300) for value in values)
                passed = math.isfinite(ours) and relative <= 1e-9
                result.update(ours=ours, reference_values=values, max_abs_delta=delta, max_relative_delta=relative,
                              status='pass' if passed else 'numeric-finding',
                              canonical_process_rss_kbytes=process_max_rss(directory / 'canonical-memory.txt'),
                              rust_process_rss_kbytes=process_max_rss(directory / 'rust-memory.txt'))
            else:
                result['status'] = 'process-failure'
            assert sha(a.source) == SOURCE_SHA and all(sha(path) == row[role + '_sha256'] for role, path in paths.items())
            write(directory / 'results.json', result)
            outcomes.append(result)
            print('CASE_DONE', len(outcomes), directory.name, result['status'], flush=True)
    write(root / 'completion.json', dict(cases=outcomes, passes=sum(row['status'] == 'pass' for row in outcomes), qualified_timing_cells=0))
    if any(row['status'] != 'pass' for row in outcomes):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
