#!/usr/bin/env python3
"""Unchanged author PSNR-HVS/HVS-M with upstream Octave signal DCT.

Caller holds both host locks and run-heavy. RGB means three independent
author plane calls; luma uses rounded BT.601. No local DCT shim is loaded.
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

PINS = {'psnrhvsm.m': '4a373c54603f98adf1a4340b0bcdee8b0b8e1c7d4a2dce5e5699fb08147841b7',
        'dct.m': '88e7eeebd55133315c8136571019f64d70e335c4b21fcc656f0e77388a6a8da2',
        'dct2.m': 'f5eb242f04a9e83aa6a4482b1dfc66ff715650a8eaeab85f43f357a4534dec2d'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'signal', 'inputs', 'binary', 'build-receipt', 'output']:
        p.add_argument('--' + name, required=True, type=Path)
    a = p.parse_args()
    for name in ['source', 'signal', 'inputs', 'binary', 'build_receipt']:
        setattr(a, name, getattr(a, name).resolve(strict=True))
    sources = {'psnrhvsm.m': a.source, 'dct.m': a.signal / 'dct.m', 'dct2.m': a.signal / 'dct2.m'}
    assert all(sha(path) == PINS[name] for name, path in sources.items())
    receipt = json.loads(a.build_receipt.read_text())
    assert receipt['source_revision'] == 'e2d414c33a6d3c7da50a818fa36e56654d3ebeef'
    assert not receipt['adapters_applied'] and sha(a.binary) == receipt['binary_sha256']
    cap = max(1, len(os.sched_getaffinity(0)) - 4)
    octave = Path(shutil.which('octave-cli')).resolve(strict=True)
    a.output.mkdir()
    root = a.output.resolve()
    base = Path(__file__).resolve().parent
    helpers = {name: sha(base / name) for name in ['psnrhvs_primary_panel.py', 'haarpsi_primary.py', 'octave_primary.py', 'supervision.py']}
    identity = dict(metrics=['psnrhvs', 'psnrhvs-y'], sources={name: str(path) for name, path in sources.items()},
                    source_sha256=PINS, helpers=helpers, inputs_sha256=sha(a.inputs),
                    binary_sha256=receipt['binary_sha256'], receipt_sha256=sha(a.build_receipt),
                    origin='http://ponomarenko.info/psnrhvsm.htm', signal_version='1.4.1 / Ubuntu 1.4.1-4build2',
                    signal_reference='https://octave.sourceforge.io/signal/function/dct2.html',
                    absolute_bound=1e-3, gate_source='crates/psnrhvs/README.md: Verification', threads=[1, cap],
                    octave=str(octave), octave_sha256=sha(octave), qualified_timing_cells=0,
                    map_status='author psnrhvsm.m returns two scalars; no native map API')
    write(root / 'identity.json', identity)
    outcomes = []
    for row in json.loads(a.inputs.read_text())['pairs']:
        paths = {role: (a.inputs.parent / row[role]).resolve(strict=True) for role in ['reference', 'distorted']}
        assert all(sha(path) == row[role + '_sha256'] for role, path in paths.items())
        w, h = row['width'], row['height']
        for metric in ['psnrhvs', 'psnrhvs-y']:
            for threads in [1, cap]:
                directory = root / f'{metric}-{row["pair"]}-{row["size_name"]}-t{threads}'
                directory.mkdir()
                env = dict(os.environ, TMPDIR=str(directory), LC_ALL='C')
                for key in ['DISPLAY', 'WAYLAND_DISPLAY', 'XAUTHORITY']:
                    env.pop(key, None)
                for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
                    env[key] = str(threads)
                planes = ("r=round(0.299*r(:,:,1)+0.587*r(:,:,2)+0.114*r(:,:,3));"
                          "d=round(0.299*d(:,:,1)+0.587*d(:,:,2)+0.114*d(:,:,3));") if metric.endswith('-y') else ''
                bindings = '\n'.join(f"assert(strcmp(which('{name[:-2]}'),{quoted(path)}));" for name, path in sources.items())
                entry = directory / 'entry.m'
                entry.write_text(f"""more off;
addpath({quoted(a.source.parent)},'-begin');addpath({quoted(a.signal)},'-begin');
{bindings}
fftw('threads',{threads});fprintf('VERSION octave-%s fftw-threads-%d\\n',version(),fftw('threads'));
for name={{'psnrhvsm','dct2','dct','fft','var'}};fprintf('BIND %s %s\\n',name{{1}},which(name{{1}}));end
f=fopen({quoted(paths['reference'])},'rb');assert(f>=0);[r,n]=fread(f,Inf,'*uint8');assert(n=={w*h*3});assert(fclose(f)==0);
f=fopen({quoted(paths['distorted'])},'rb');assert(f>=0);[d,n]=fread(f,Inf,'*uint8');assert(n=={w*h*3});assert(fclose(f)==0);
r=double(permute(reshape(r,[3,{w},{h}]),[3,2,1]));d=double(permute(reshape(d,[3,{w},{h}]),[3,2,1]));
{planes}
values=zeros(size(r,3),2);
for c=1:size(r,3);[masked,unmasked]=psnrhvsm(r(:,:,c),d(:,:,c));values(c,:)=[unmasked,masked];end
scores=mean(values,1);assert(all(isfinite(scores)));
fprintf('SCORES %.17g %.17g\\n',scores(1),scores(2));fflush(stdout);
""")
                canonical_rc, canonical_command = execute([str(octave), '--no-gui', '--no-window-system', '--no-history',
                    '--no-init-file', '--quiet', str(entry)], env, directory, 'canonical', 300)
                write(directory / 'config.json', dict(width=w, height=h, reference=str(paths['reference']), distorted=str(paths['distorted']),
                     metric_name=metric, threads=threads, probe_only=True, only_mode='score', timing_phase='warm'))
                rust_rc, rust_command = execute([str(a.binary), str(directory / 'config.json')], env, directory, 'rust', 120)
                result = dict(metric=metric, pair=row['pair'], size=row['size_name'], width=w, height=h, threads=threads,
                              canonical_returncode=canonical_rc, rust_returncode=rust_rc, canonical_command=canonical_command,
                              rust_command=rust_command, bound=1e-3, status='process-finding', qualified_ratio=False)
                if canonical_rc == rust_rc == 0:
                    scores = [list(map(float, s.split()[1:])) for s in (directory / 'canonical-stdout.log').read_text().splitlines() if s.startswith('SCORES ')]
                    probes = [json.loads(s.removeprefix('IQA_PROBE ')) for s in (directory / 'rust-stdout.log').read_text().splitlines() if s.startswith('IQA_PROBE ')]
                    assert len(scores) == len(probes) == 1 and len(scores[0]) == len(probes[0]['scores']) == 2
                    assert probes[0]['metric_name'] == metric
                    compared = []
                    for (column, bits), canonical in zip(probes[0]['scores'], scores[0]):
                        ours = struct.unpack('>d', bytes.fromhex(bits))[0]
                        assert math.isfinite(ours)
                        compared.append(dict(column=column, ours=ours, canonical=canonical, abs_delta=abs(ours-canonical)))
                    delta = max(v['abs_delta'] for v in compared)
                    result.update(scores=compared, max_abs_delta=delta, status='pass' if delta <= 1e-3 else 'numeric-finding',
                                  canonical_process_rss_kbytes=process_max_rss(directory / 'canonical-memory.txt'),
                                  rust_process_rss_kbytes=process_max_rss(directory / 'rust-memory.txt'))
                assert all(sha(path) == PINS[name] for name, path in sources.items())
                assert sha(a.binary) == receipt['binary_sha256'] and all(sha(base / name) == value for name, value in helpers.items())
                assert all(sha(path) == row[role + '_sha256'] for role, path in paths.items())
                write(directory / 'results.json', result); outcomes.append(result)
                print('CASE_DONE', len(outcomes), directory.name, result['status'], result.get('max_abs_delta'), flush=True)
    write(root / 'completion.json', dict(cases=outcomes, qualified_timing_cells=0))
    if any(r['status'] != 'pass' for r in outcomes):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
