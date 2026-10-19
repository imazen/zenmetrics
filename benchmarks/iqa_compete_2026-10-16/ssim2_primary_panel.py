#!/usr/bin/env python3
"""Compare the pinned primary RGB adapter with the frozen Rust scalar probe.

Caller holds both heavy locks and run-heavy. Scores are observations: no
cross-implementation acceptance bound or paired timing ratio is invented.
"""
import argparse
import json
import math
import os
from pathlib import Path
import struct

from haarpsi_primary import execute, write
from octave_primary import sha, process_max_rss


PRIMARY_SHA = '9d0fc033b44580055909b52e2cabad87d2c5f6fb28c4fe970391738fe3f67ac1'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['canonical', 'inputs', 'binary', 'build-receipt', 'output']:
        p.add_argument('--' + name, required=True, type=Path)
    a = p.parse_args()
    for name in ['canonical', 'inputs', 'binary', 'build_receipt']:
        setattr(a, name, getattr(a, name).resolve(strict=True))
    assert sha(a.canonical) == PRIMARY_SHA
    receipt = json.loads(a.build_receipt.read_text())
    assert receipt['source_revision'] == 'e2d414c33a6d3c7da50a818fa36e56654d3ebeef'
    assert not receipt['adapters_applied'] and sha(a.binary) == receipt['binary_sha256']
    primary_root = a.canonical.parent.parent
    primary_source = primary_root / 'src-ssim2-extract/src/ssimulacra2.cc'
    build_log = primary_root.with_suffix('.log')
    assert primary_source.is_file() and build_log.is_file()
    cap = max(1, len(os.sched_getaffinity(0)) - 4)
    a.output.mkdir()
    root = a.output.resolve()
    base = Path(__file__).resolve().parent
    helpers = {name: sha(base / name) for name in ['ssim2_primary_panel.py', 'haarpsi_primary.py',
                                                 'octave_primary.py', 'supervision.py']}
    identity = dict(metric='ssim2', canonical_sha256=PRIMARY_SHA, primary_source_sha256=sha(primary_source),
                    primary_build_log_sha256=sha(build_log), helpers=helpers,
                    origin='https://github.com/cloudinary/ssimulacra2/tree/fb5d0dbe866e5113d320666a09ddbc8d145b5fde',
                    binary_sha256=receipt['binary_sha256'], rust_revision=receipt['source_revision'],
                    receipt_sha256=sha(a.build_receipt), inputs_sha256=sha(a.inputs), threads=[1, cap],
                    bound='UNBOUND', qualified_timing_cells=0,
                    map_status='primary v2.1 exposes scalar/features, no native pixel map',
                    runtime_cms_branch='not established by this scalar experiment')
    write(root / 'identity.json', identity)
    results = []
    for row in json.loads(a.inputs.read_text())['pairs']:
        paths = {role: (a.inputs.parent / row[role]).resolve(strict=True) for role in ['reference', 'distorted']}
        assert all(sha(path) == row[role + '_sha256'] for role, path in paths.items())
        for threads in [1, cap]:
            directory = root / f'ssim2-{row["pair"]}-{row["size_name"]}-t{threads}'
            directory.mkdir()
            env = dict(os.environ, TMPDIR=str(directory), LC_ALL='C')
            for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
                env[key] = str(threads)
            canonical_rc, canonical_command = execute([str(a.canonical), 'ssim2', str(row['width']), str(row['height']),
                str(paths['reference']), str(paths['distorted']), str(threads), 'score', '1'], env, directory, 'canonical', 300)
            config = dict(width=row['width'], height=row['height'], reference=str(paths['reference']),
                          distorted=str(paths['distorted']), metric_name='ssim2', threads=threads,
                          probe_only=True, only_mode='score', timing_phase='warm')
            write(directory / 'config.json', config)
            rust_rc, rust_command = execute([str(a.binary), str(directory / 'config.json')], env, directory, 'rust', 300)
            result = dict(metric='ssim2', pair=row['pair'], size=row['size_name'], threads=threads,
                          width=row['width'], height=row['height'], canonical_returncode=canonical_rc,
                          rust_returncode=rust_rc, canonical_command=canonical_command, rust_command=rust_command,
                          bound='UNBOUND', status='process-finding', qualified_ratio=False)
            if canonical_rc == rust_rc == 0:
                values = [float(s.split()[2]) for s in (directory / 'canonical-stdout.log').read_text().splitlines()
                          if s.startswith(('COLD ', 'SAMPLE '))]
                probes = [json.loads(s.removeprefix('IQA_PROBE ')) for s in (directory / 'rust-stdout.log').read_text().splitlines()
                          if s.startswith('IQA_PROBE ')]
                assert len(values) == 2 and values[0] == values[1] and len(probes) == 1
                assert probes[0]['metric_name'] == 'ssim2' and len(probes[0]['scores']) == 1
                ours = struct.unpack('>d', bytes.fromhex(probes[0]['scores'][0][1]))[0]
                assert math.isfinite(ours)
                result.update(ours=ours, canonical=values[0], max_abs_delta=abs(ours-values[0]), status='UNQUALIFIED',
                              canonical_process_rss_kbytes=process_max_rss(directory / 'canonical-memory.txt'),
                              rust_process_rss_kbytes=process_max_rss(directory / 'rust-memory.txt'))
            assert sha(a.binary) == receipt['binary_sha256'] and sha(a.canonical) == PRIMARY_SHA
            assert all(sha(base / name) == value for name, value in helpers.items())
            assert all(sha(path) == row[role + '_sha256'] for role, path in paths.items())
            write(directory / 'results.json', result)
            results.append(result)
            print('CASE_DONE', len(results), directory.name, result['status'], result.get('max_abs_delta'), flush=True)
    write(root / 'completion.json', dict(cases=results, qualified_timing_cells=0))
    if any(r['status'] != 'UNQUALIFIED' for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
