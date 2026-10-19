#!/usr/bin/env python3
"""Explicit CUDA functional pilot, compared with frozen CPU headline scores.

This diagnoses routes on one 64x64 photo pair. It is not the full primary
parity grid, a new numerical gate, or a qualified timing experiment.
Caller holds both heavy locks and run-heavy around the complete pilot.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess

from haarpsi_primary import execute, write
from octave_primary import sha, process_max_rss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['gpu_build', 'cpu_build', 'inputs', 'output']:
        p.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    a = p.parse_args()
    builds = {role: getattr(a, role+'_build').resolve(strict=True) for role in ['cpu', 'gpu']}
    binaries = {role: root / 'target/release/iqa-compete' for role, root in builds.items()}
    receipts = {role: json.loads((root / 'build-receipt.json').read_text()) for role, root in builds.items()}
    for role, binary in binaries.items():
        assert sha(binary) == receipts[role]['binary_sha256']
    assert 'study-gpu-cuda' in receipts['gpu']['features']
    a.inputs = a.inputs.resolve(strict=True)
    row, = [r for r in json.loads(a.inputs.read_text())['pairs'] if r['pair']=='pairA' and r['size_name']=='64']
    paths = {role: (a.inputs.parent / row[role]).resolve(strict=True) for role in ['reference', 'distorted']}
    assert all(sha(path) == row[role+'_sha256'] for role, path in paths.items())
    a.output.mkdir(); root = a.output.resolve()
    env = dict(os.environ, TMPDIR=str(root), CUDA_VISIBLE_DEVICES='0', LC_ALL='C',
               CUDA_PATH='/usr/local/cuda', LD_LIBRARY_PATH='/usr/local/cuda/lib64:'+os.environ.get('LD_LIBRARY_PATH',''))
    driver = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'], text=True)
    identity = dict(build_receipt_sha256={role:sha(path/'build-receipt.json') for role,path in builds.items()},
                    binary_sha256={role:sha(binary) for role,binary in binaries.items()},
                    inputs_sha256=sha(a.inputs), input=row, driver=driver, cuda_path=str(Path('/usr/local/cuda').resolve()),
                    worker_sha256=sha(Path(__file__)), scope=__doc__, qualified_timing_cells=0)
    write(root / 'identity.json', identity)
    results = []
    for metric in ['ssim2', 'butteraugli', 'dssim', 'iwssim', 'cvvdp', 'zensim']:
        for threads in [1, max(1,len(os.sched_getaffinity(0))-4)]:
            directory = root / f'{metric}-t{threads}'; directory.mkdir()
            record = dict(metric=metric+'-gpu', threads=threads, status='finding', qualified_ratio=False)
            for key in ['RAYON_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
                env[key] = str(threads)
            for role, binary in binaries.items():
                config = dict(width=64, height=64, reference=str(paths['reference']), distorted=str(paths['distorted']),
                              metric_name=metric if role=='cpu' else metric+'-gpu', threads=threads,
                              only_mode='score', gpu_runtime='cuda', locked_process_accounting=True,
                              timing_phase='warm' if role=='cpu' else 'cold', probe_only=role=='cpu')
                config_path = directory / (role+'-config.json'); write(config_path, config)
                rc, command = execute([str(binary), str(config_path)], env, directory, role, 600)
                value = dict(returncode=rc, command=command, process_rss_kbytes=process_max_rss(directory/(role+'-memory.txt')))
                lines = (directory/(role+'-stdout.log')).read_text().splitlines()
                if role=='cpu' and rc==0:
                    probes = [json.loads(s.removeprefix('IQA_PROBE ')) for s in lines if s.startswith('IQA_PROBE ')]
                    assert len(probes)==1
                    value.update(score=struct.unpack('>d',bytes.fromhex(probes[0]['scores'][0][1]))[0], columns=probes[0]['scores'])
                elif role=='gpu' and rc==0:
                    scored = [json.loads(s.removeprefix('IQA_ROW ')) for s in lines if s.startswith('IQA_ROW ')]
                    assert len(scored)==1
                    value.update(row=scored[0])
                    if scored[0]['status']=='ok':
                        value['score'] = struct.unpack('>d',bytes.fromhex(scored[0]['score_bits']))[0]
                record[role] = value
            if all('score' in record[role] for role in ['cpu','gpu']):
                record.update(status='UNQUALIFIED-internal-comparison', abs_delta=abs(record['cpu']['score']-record['gpu']['score']))
            assert sha(Path(__file__))==identity['worker_sha256']
            assert all(sha(binary)==receipts[role]['binary_sha256'] for role,binary in binaries.items())
            write(directory/'results.json',record); results.append(record)
            print('CASE_DONE',len(results),record['metric'],threads,record['status'],record.get('abs_delta'),flush=True)
    write(root/'completion.json',dict(cases=results,qualified_timing_cells=0))


if __name__=='__main__':
    main()
