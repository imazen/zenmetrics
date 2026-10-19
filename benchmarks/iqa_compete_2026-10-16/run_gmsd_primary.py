#!/usr/bin/env python3
"""Compare primary C maps/scores and the frozen Rust route; caller owns scope."""
import argparse
import array
import csv
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

from supervision import scoring_command
from octave_primary import process_max_rss


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as f:
        f.write(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['build', 'rust-build', 'inputs', 'output']:
        p.add_argument('--' + name, required=True, type=Path)
    a = p.parse_args()
    for name in ['build', 'rust_build', 'inputs']:
        setattr(a, name, getattr(a, name).resolve(strict=True))
    receipt = json.loads((a.build / 'build-receipt.json').read_text())
    for name, digest in receipt['binaries'].items():
        assert sha(a.build / name) == digest
    rust = a.rust_build / 'target/release/iqa-compete'
    rust_receipt = json.loads((a.rust_build / 'build-receipt.json').read_text())
    assert sha(rust) == rust_receipt['binary_sha256']
    assert rust_receipt['source_revision'] == receipt['rust_source_revision']
    pairs = json.loads(a.inputs.read_text())['pairs']
    cap = max(1, len(os.sched_getaffinity(0)) - 4)
    a.output.mkdir()
    root = a.output.resolve()
    write(root / 'identity.json', dict(build_receipt_sha256=sha(a.build / 'build-receipt.json'),
          rust_build_receipt_sha256=sha(a.rust_build / 'build-receipt.json'), binaries=receipt['binaries'],
          rust_cli_sha256=sha(rust), inputs_sha256=sha(a.inputs), threads=[1, cap],
          worker_sha256=sha(Path(__file__)), score_relative_bound=1e-12, mean_absolute_bound=1e-14,
          map_gate='bit-identical', gate_source='benchmarks/gmsd_parity_2026-09-22.md and crates/gmsd/src/tests.rs',
          qualified_timing_cells=0))
    rows = []
    for row in pairs:
        paths = {role: (a.inputs.parent / row[role]).resolve(strict=True) for role in ['reference', 'distorted']}
        assert all(sha(path) == row[role + '_sha256'] for role, path in paths.items())
        w, h = row['width'], row['height']
        for threads in [1, cap]:
            directory = root / f'{row["pair"]}-{row["size_name"]}-t{threads}'
            directory.mkdir()
            env = dict(os.environ, LC_ALL='C', TMPDIR=str(directory), ASAN_OPTIONS='detect_leaks=1:halt_on_error=1')
            for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'RAYON_NUM_THREADS', 'MKL_NUM_THREADS']:
                env[key] = str(threads)
            outcome = dict(pair=row['pair'], size=row['size_name'], width=w, height=h, threads=threads,
                           canonical_instrumented=bool(w % 2 or h % 2), qualified_ratio=False)

            def execute(role, command):
                wrapped = scoring_command(list(map(str, command)), system='Linux', seconds=120,
                                          memory_log=directory / (role + '-memory.txt'))
                with (directory / (role + '-stdout.log')).open('x') as out, (directory / (role + '-stderr.log')).open('x') as err:
                    result = subprocess.run(wrapped, env=env, cwd=directory, stdout=out, stderr=err)
                outcome[role + '_command'] = wrapped
                outcome[role + '_returncode'] = result.returncode
                outcome[role + '_process_rss_kbytes'] = process_max_rss(directory / (role + '-memory.txt'))
                return result.returncode

            binary = a.build / ('gmsd-primary-asan' if w % 2 or h % 2 else 'gmsd-primary')
            canonical_rc = execute('canonical', [binary, w, h, paths['reference'], paths['distorted'], directory / 'canonical.f32'])
            map_rc = execute('rust_map', [a.build / 'gmsd-rust-map', w, h, paths['reference'], paths['distorted'], threads, directory / 'rust.f32'])
            config = dict(width=w, height=h, reference=str(paths['reference']), distorted=str(paths['distorted']),
                          metric_name='gmsd', threads=threads, probe_only=True, only_mode='score', timing_phase='warm')
            write(directory / 'config.json', config)
            cli_rc = execute('rust_cli', [rust, directory / 'config.json'])
            assert map_rc == cli_rc == 0
            fields = (directory / 'rust_map-stdout.log').read_text().split()
            assert fields[0] == 'GMSD' and fields[2] == 'MEAN'
            ours, mean = float(fields[1]), float(fields[3])
            probes = [json.loads(s.removeprefix('IQA_PROBE ')) for s in (directory / 'rust_cli-stdout.log').read_text().splitlines() if s.startswith('IQA_PROBE ')]
            assert len(probes) == 1 and len(probes[0]['scores']) == 1
            cli = struct.unpack('>d', bytes.fromhex(probes[0]['scores'][0][1]))[0]
            assert ours == cli
            outcome.update(ours=ours, mean_gms=mean, cli_score=cli, rust_map_sha256=sha(directory / 'rust.f32'))
            if canonical_rc:
                outcome.update(status='canonical-process-failure', canonical_error=(directory / 'canonical-stderr.log').read_text())
            else:
                scores = [float(s.split()[2]) for s in (directory / 'canonical-stdout.log').read_text().splitlines() if s.startswith(('COLD ', 'SAMPLE '))]
                assert len(scores) == 2 and scores[0] == scores[1]
                original = (directory / 'canonical.f32').read_bytes()
                port = (directory / 'rust.f32').read_bytes()
                assert len(original) == len(port) == (w // 2) * (h // 2) * 4
                values = array.array('f'); values.frombytes(original)
                assert struct.pack('=I', 1) == struct.pack('<I', 1), 'little-endian map reader required'
                canonical_mean = sum(values) / len(values)
                delta = max(abs(ours - value) for value in scores)
                relative = delta / max(abs(scores[0]), 1e-300)
                outcome.update(canonical=scores[0], absolute_delta=delta, relative_delta=relative,
                               canonical_mean=canonical_mean, mean_absolute_delta=abs(mean - canonical_mean),
                               map_bit_identical=original == port, canonical_map_sha256=sha(directory / 'canonical.f32'),
                               status='pass' if relative <= 1e-12 and abs(mean - canonical_mean) < 1e-14 and original == port else 'numeric-finding')
            write(directory / 'results.json', outcome)
            rows.append(outcome)
            print('CASE_DONE', len(rows), directory.name, outcome['status'], flush=True)
    write(root / 'completion.json', dict(cases=rows, qualified_timing_cells=0))
    with (root / 'files.sha256.jsonl').open('x') as f:
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.name != 'files.sha256.jsonl':
                f.write(json.dumps(dict(path=str(path.relative_to(root)), sha256=sha(path))) + '\n')
    if any(row['status'] == 'numeric-finding' for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
