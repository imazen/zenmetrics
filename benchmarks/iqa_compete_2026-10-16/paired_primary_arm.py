#!/usr/bin/env python3
"""HaarPSI child timing transport for zenbench pair; no startup/IO timing."""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

from haarpsi_primary import SOURCE_SHA
from octave_primary import sha, quoted, process_max_rss
from supervision import scoring_command


def main():
    if len(sys.argv) == 4 and sys.argv[1] == '--check':
        a, b = [json.loads((Path(p) / 'arm.json').read_text()) for p in sys.argv[2:]]
        assert a['input_hashes'] == b['input_hashes'] and a['metric'] == b['metric']
        assert a['phase'] == b['phase'] and a['threads'] == b['threads']
        assert all(abs(x - y) <= 5e-5 for x in a['scores'] for y in b['scores'])
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=['canonical', 'rust'], required=True)
    p.add_argument('--metric', choices=['haarpsi', 'haarpsi-y'], required=True)
    p.add_argument('--phase', choices=['cold', 'warm'], required=True)
    p.add_argument('--canonical-samples', type=int, default=10)
    for name in ['reference', 'distorted', 'source', 'binary', 'receipt', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    for name in ['width', 'height', 'threads']:
        p.add_argument('--' + name, type=int, required=True)
    a = p.parse_args()
    assert 1 <= a.canonical_samples <= 1000
    assert min(a.width, a.height, a.threads) > 0 and sha(a.source) == SOURCE_SHA
    receipt = json.loads(a.receipt.read_text())
    assert receipt['source_revision'] == 'e2d414c33a6d3c7da50a818fa36e56654d3ebeef'
    assert sha(a.binary) == receipt['binary_sha256']
    paths = {name: getattr(a, name).resolve(strict=True) for name in ['reference', 'distorted']}
    assert all(path.stat().st_size == a.width * a.height * 3 for path in paths.values())
    root = a.output.resolve(strict=True)
    input_hashes = {name: sha(path) for name, path in paths.items()}
    env = dict(os.environ, TMPDIR=str(root), LC_ALL='C')
    for key in ['DISPLAY', 'WAYLAND_DISPLAY', 'XAUTHORITY']:
        env.pop(key, None)
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
        env[key] = str(a.threads)
    if a.arm == 'rust':
        config = dict(width=a.width, height=a.height, reference=str(paths['reference']), distorted=str(paths['distorted']),
                      metric_name=a.metric, threads=a.threads, only_mode='score', timing_phase=a.phase,
                      locked_process_accounting=True, sample_target_ns=1_000_000, chunk_budget_ms=60_000)
        (root / 'config.json').write_text(json.dumps(config) + '\n')
        command = [str(a.binary), str(root / 'config.json')]
        env.update(ZENBENCH_CPU_POLICY='unrestricted', ZENBENCH_NO_SAVE='1')
    else:
        convert = ''
        if a.metric == 'haarpsi-y':
            convert = '\n'.join(f'{n}=0.299*double({n}(:,:,1))+0.587*double({n}(:,:,2))+0.114*double({n}(:,:,3));' for n in ['r', 'd'])
        entry = root / 'entry.m'
        entry.write_text(f"""more off;
addpath({quoted(a.source.parent)},'-begin');
assert(strcmp(which('HaarPSI'),{quoted(a.source)}));
f=fopen({quoted(paths['reference'])},'rb');assert(f>=0);[r,n]=fread(f,Inf,'*uint8');assert(n=={a.width*a.height*3});assert(fclose(f)==0);
f=fopen({quoted(paths['distorted'])},'rb');assert(f>=0);[d,n]=fread(f,Inf,'*uint8');assert(n=={a.width*a.height*3});assert(fclose(f)==0);
r0=permute(reshape(r,[3,{a.width},{a.height}]),[3,2,1]);
d0=permute(reshape(d,[3,{a.width},{a.height}]),[3,2,1]);clear r d;
for call=0:{a.canonical_samples if a.phase == 'warm' else 0}
 clock=tic();
 r=r0;d=d0;
 {convert}
 score=HaarPSI(r,d);clear r d;
 elapsed=toc(clock)*1e9;
 label='SAMPLE';if call==0;label='COLD';end
 fprintf('%s %.0f %.17g\\n',label,elapsed,score);fflush(stdout);
end
""")
        command = [str(Path(shutil.which('octave-cli')).resolve(strict=True)), '--no-gui', '--no-window-system',
                   '--no-history', '--no-init-file', '--quiet', str(entry)]
    command = scoring_command(command, system='Linux', seconds=300, memory_log=root / 'process-memory.txt')
    with (root / 'score-stdout.log').open('x') as stdout, (root / 'score-stderr.log').open('x') as stderr:
        result = subprocess.run(command, cwd=root, env=env, stdout=stdout, stderr=stderr)
    assert result.returncode == 0, 'child failed; see score-stderr.log'
    lines = (root / 'score-stdout.log').read_text().splitlines()
    if a.arm == 'rust':
        values = [json.loads(s.removeprefix('IQA_ROW ')) for s in lines if s.startswith('IQA_ROW ')]
        assert len(values) == 1 and values[0]['status'] == 'ok'
        row = values[0]
        timings = [row['cold_elapsed_ns']] if a.phase == 'cold' else row['round_means_ns']
        scores = [row['score']]
        assert len(timings) == (1 if a.phase == 'cold' else 10)
    else:
        selected = [s.split() for s in lines if s.startswith('COLD ' if a.phase == 'cold' else 'SAMPLE ')]
        assert len(selected) == (1 if a.phase == 'cold' else a.canonical_samples)
        timings = [float(s[1]) for s in selected]
        scores = [float(s[2]) for s in selected]
        all_scores = [float(s.split()[2]) for s in lines if s.startswith(('COLD ', 'SAMPLE '))]
        assert all(struct.pack('>d', v) == struct.pack('>d', all_scores[0]) for v in all_scores)
    assert all(t > 0 for t in timings)
    assert sha(a.source) == SOURCE_SHA and input_hashes == {name: sha(path) for name, path in paths.items()}
    record = dict(arm=a.arm, metric=a.metric, phase=a.phase, threads=a.threads, input_hashes=input_hashes,
                  canonical_samples=a.canonical_samples,
                  scores=scores, timings_ns=timings, command=command, process_rss_kbytes=process_max_rss(root / 'process-memory.txt'),
                  source_sha256=SOURCE_SHA, rust_binary_sha256=receipt['binary_sha256'])
    (root / 'arm.json').write_text(json.dumps(record, indent=2) + '\n')
    print('PAIR_TIME_NS', sum(timings) / len(timings), flush=True)


if __name__ == '__main__':
    main()
