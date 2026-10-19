#!/usr/bin/env python3
"""Primary HaarPSI RGB/luma scalar grid; caller holds locks and run-heavy.

Uses the pinned unchanged author M-file and the existing frozen Rust probe.
The author exposes orientation maps; this scalar adapter does not qualify
map parity or acquire paired timing ratios.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess

from octave_primary import sha, quoted, process_max_rss
from supervision import scoring_command

SOURCE_SHA = 'ef1bed2872fd1f74842afdccfd1d6b0e3a93bc5479b14bf845460f08dec8ee4a'
ORIGIN = 'https://github.com/rgcda/haarpsi/blob/3b9599b0c1911ee08f1bb991bb19e19d86727819/HaarPSI.m'


def write(path, value):
    with path.open('x') as f:
        f.write(json.dumps(value, indent=2) + '\n')


def execute(command, env, directory, role, seconds):
    command = scoring_command(command, system='Linux', seconds=seconds, memory_log=directory / (role + '-memory.txt'))
    with (directory / (role + '-stdout.log')).open('x') as stdout, (directory / (role + '-stderr.log')).open('x') as stderr:
        result = subprocess.run(command, cwd=directory, env=env, stdout=stdout, stderr=stderr)
    return result.returncode, command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'inputs', 'binary', 'build-receipt', 'output']:
        p.add_argument('--' + name, required=True, type=Path)
    a = p.parse_args()
    for name in ['source', 'inputs', 'binary', 'build_receipt']:
        setattr(a, name, getattr(a, name).resolve(strict=True))
    assert a.source.name == 'HaarPSI.m' and sha(a.source) == SOURCE_SHA
    receipt = json.loads(a.build_receipt.read_text())
    assert receipt['source_revision'] == 'e2d414c33a6d3c7da50a818fa36e56654d3ebeef'
    assert not receipt['adapters_applied'] and sha(a.binary) == receipt['binary_sha256']
    inventory = json.loads(subprocess.check_output([str(a.binary), 'inventory'], text=True))
    columns = {name: next(r['columns'] for r in inventory if r['metric_name'] == name and r['compiled'])
               for name in ['haarpsi', 'haarpsi-y']}
    assert all(len(v) == 1 for v in columns.values())
    pairs = json.loads(a.inputs.read_text())['pairs']
    cap = max(1, len(os.sched_getaffinity(0)) - 4)
    a.output.mkdir()
    root = a.output.resolve()
    octave = Path(shutil.which('octave-cli')).resolve(strict=True)
    base = Path(__file__).resolve().parent
    helpers = {name: sha(base / name) for name in ['haarpsi_primary.py', 'octave_primary.py', 'supervision.py']}
    write(root / 'identity.json', dict(source_sha256=SOURCE_SHA, origin=ORIGIN, helpers=helpers,
          binary_sha256=receipt['binary_sha256'], build_receipt_sha256=sha(a.build_receipt),
          octave_sha256=sha(octave), inputs_sha256=sha(a.inputs), threads=[1, cap],
          bound=5e-5, gate_source='crates/haarpsi/README.md: Verification', qualified_timing_cells=0,
          map_status='missing: scalar-only probe; no map acceptance gate inferred'))
    outcomes = []
    for row in pairs:
        paths = {role: (a.inputs.parent / row[role]).resolve(strict=True) for role in ['reference', 'distorted']}
        for role, path in paths.items():
            assert sha(path) == row[role + '_sha256'] and path.stat().st_size == row['width'] * row['height'] * 3
        for metric in ['haarpsi', 'haarpsi-y']:
            for threads in [1, cap]:
                assert sha(a.source) == SOURCE_SHA and sha(a.binary) == receipt['binary_sha256']
                assert all(sha(base / name) == digest for name, digest in helpers.items())
                directory = root / f'{metric}-{row["pair"]}-{row["size_name"]}-t{threads}'
                directory.mkdir()
                env = dict(os.environ, TMPDIR=str(directory), LC_ALL='C')
                for key in ['DISPLAY', 'WAYLAND_DISPLAY', 'XAUTHORITY']:
                    env.pop(key, None)
                for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
                    env[key] = str(threads)
                w, h = row['width'], row['height']
                convert = ''
                if metric == 'haarpsi-y':
                    convert = '\n'.join(f'{name}=0.299*double({name}(:,:,1))+0.587*double({name}(:,:,2))+0.114*double({name}(:,:,3));'
                                        for name in ['reference', 'distorted'])
                entry = directory / 'entry.m'
                entry.write_text(f"""more off;
addpath({quoted(a.source.parent)},'-begin');
assert(strcmp(which('HaarPSI'),{quoted(a.source)}));
fprintf('VERSION octave-%s\\n',version());
for name={{'conv2','exp','log'}}; fprintf('BIND %s %s\\n',name{{1}},which(name{{1}})); end
f=fopen({quoted(paths['reference'])},'rb');assert(f>=0);[r,n]=fread(f,Inf,'*uint8');assert(n=={w*h*3});assert(fclose(f)==0);
f=fopen({quoted(paths['distorted'])},'rb');assert(f>=0);[d,n]=fread(f,Inf,'*uint8');assert(n=={w*h*3});assert(fclose(f)==0);
for call=0:1
 clock=tic();
 reference=permute(reshape(r,[3,{w},{h}]),[3,2,1]);
 distorted=permute(reshape(d,[3,{w},{h}]),[3,2,1]);
 {convert}
 score=HaarPSI(reference,distorted);
 clear reference distorted;
 elapsed=toc(clock)*1e9;
 label='SAMPLE';if call==0;label='COLD';end
 fprintf('%s %.0f %.17g\\n',label,elapsed,score);fflush(stdout);
end
""")
                canonical_rc, canonical_command = execute([str(octave), '--no-gui', '--no-window-system', '--no-history',
                        '--no-init-file', '--quiet', str(entry)], env, directory, 'canonical', 300)
                config = dict(width=w, height=h, reference=str(paths['reference']), distorted=str(paths['distorted']),
                              metric_name=metric, threads=threads, probe_only=True, only_mode='score', timing_phase='warm')
                write(directory / 'config.json', config)
                rust_rc, rust_command = execute([str(a.binary), str(directory / 'config.json')], env, directory, 'rust', 120)
                result = dict(metric=metric, pair=row['pair'], size=row['size_name'], width=w, height=h, threads=threads,
                              canonical_returncode=canonical_rc, rust_returncode=rust_rc,
                              canonical_command=canonical_command, rust_command=rust_command, bound=5e-5)
                if canonical_rc == rust_rc == 0:
                    references = [float(s.split()[2]) for s in (directory / 'canonical-stdout.log').read_text().splitlines()
                                  if s.startswith(('COLD ', 'SAMPLE '))]
                    probes = [json.loads(s.removeprefix('IQA_PROBE ')) for s in
                              (directory / 'rust-stdout.log').read_text().splitlines() if s.startswith('IQA_PROBE ')]
                    assert len(references) == 2 and len(probes) == 1 and probes[0]['metric_name'] == metric
                    assert [r[0] for r in probes[0]['scores']] == columns[metric]
                    ours = struct.unpack('>d', bytes.fromhex(probes[0]['scores'][0][1]))[0]
                    finite = all(math.isfinite(v) for v in [ours, *references])
                    delta = max(abs(ours - v) for v in references) if finite else None
                    result.update(ours=ours, reference_values=references, max_abs_delta=delta,
                                  status='pass' if finite and delta <= 5e-5 else 'numeric-finding',
                                  canonical_process_rss_kbytes=process_max_rss(directory / 'canonical-memory.txt'),
                                  rust_process_rss_kbytes=process_max_rss(directory / 'rust-memory.txt'))
                else:
                    result['status'] = 'process-failure'
                assert sha(a.source) == SOURCE_SHA and all(sha(path) == row[role + '_sha256'] for role, path in paths.items())
                write(directory / 'results.json', result)
                outcomes.append(result)
                print('CASE_DONE', len(outcomes), metric, row['pair'], row['size_name'], threads, result['status'], flush=True)
    write(root / 'completion.json', {'cases': len(outcomes), 'passes': sum(r['status'] == 'pass' for r in outcomes),
                                    'qualified_timing_cells': 0})
    if any(r['status'] != 'pass' for r in outcomes):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
