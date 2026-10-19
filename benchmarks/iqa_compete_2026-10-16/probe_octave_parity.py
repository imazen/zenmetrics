#!/usr/bin/env python3
"""Untimed frozen-Rust probe against a bound primary FSIM/VSI execution.

Keeps the documented base gates unchanged for fixed-factor owner variants.
This is numerical evidence only, not timing acquisition or matrix completion.
"""
import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess

from check_fixed_factor_oracle import check_manifest
from octave_primary import process_max_rss, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['binary', 'build-receipt', 'canonical', 'manifest', 'reference', 'distorted', 'evidence']:
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--threads', type=int, required=True)
    p.add_argument('--measure-memory', action='store_true', help='retain /usr/bin/time -v Rust process peak RSS')
    a = p.parse_args()
    if a.threads <= 0:
        p.error('positive thread budget required')
    manifest = check_manifest(a.manifest)
    canonical = json.loads(a.canonical.read_text())
    build = json.loads(a.build_receipt.read_text())
    binary_sha = sha(a.binary)
    if binary_sha != build['binary_sha256']:
        p.error('binary differs from frozen build receipt')
    if build['source_revision'] != 'e2d414c33a6d3c7da50a818fa36e56654d3ebeef' or build['adapters_applied']:
        p.error('this gate record requires the unmodified frozen source revision')
    if canonical['manifest_sha256'] != sha(a.manifest) or canonical['source_sha256'] != manifest['patch']['after_sha256']:
        p.error('canonical execution/source manifest mismatch')
    inputs = [a.reference.resolve(strict=True), a.distorted.resolve(strict=True)]
    hashes = [sha(path) for path in inputs]
    if canonical['input_sha256'] != hashes or canonical['threads_requested'] != a.threads or canonical['returncode'] != 0:
        p.error('canonical input/thread/completion mismatch')
    source_name = manifest['source']['name']
    if source_name not in ['vsi-matlab', 'fsim-matlab']:
        p.error('primary VSI/FSIM source required')
    family = source_name.split('-')[0]
    factor = manifest['patch']['factor']
    metric = f'{family}-nodownscale' if factor == 1 else f'{family}-k{factor}'
    bound = 1e-4 if family == 'vsi' else 5e-5
    gate_source = 'docs/METRIC_PROVENANCE.md @e2d414c33a6d3c7da50a818fa36e56654d3ebeef; existing base gate inherited by owner fixed-factor variants'
    if not canonical['score_rows'] or not all(row['all_finite'] for row in canonical['score_rows']):
        p.error('nonfinite or absent canonical score is a finding, not successful parity')
    a.evidence.mkdir()
    evidence = a.evidence.resolve()
    env = os.environ.copy()
    for key in ['RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        env[key] = str(a.threads)
    if a.measure_memory:
        env['LC_ALL'] = 'C'
    inventory_run = subprocess.run([str(a.binary), 'inventory'], capture_output=True, text=True, check=True, env=env, timeout=30)
    (evidence / 'inventory.json').write_text(inventory_run.stdout)
    inventory = json.loads(inventory_run.stdout)
    selected = [row for row in inventory if row['metric_name'] == metric]
    if len(selected) != 1 or not selected[0]['compiled']:
        raise RuntimeError('metric absent or not compiled in frozen binary')
    columns = selected[0]['columns']
    config = {'width': canonical['width'], 'height': canonical['height'],
              'reference': str(inputs[0]), 'distorted': str(inputs[1]),
              'metric_name': metric, 'threads': a.threads, 'probe_only': True,
              'only_mode': 'score', 'timing_phase': 'warm'}
    config_path = evidence / 'config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    command = [str(a.binary), str(config_path)]
    if a.measure_memory:
        command = ['/usr/bin/time', '-v', '-o', str(evidence / 'process-memory.txt'), *command]
    with (evidence / 'stdout.log').open('w') as stdout, (evidence / 'stderr.log').open('w') as stderr:
        run = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, timeout=120)
    if run.returncode:
        raise RuntimeError(f'Rust probe failed rc={run.returncode}; see {evidence}/stderr.log')
    lines = (evidence / 'stdout.log').read_text().splitlines()
    probes = [json.loads(line.removeprefix('IQA_PROBE ')) for line in lines if line.startswith('IQA_PROBE ')]
    if len(probes) != 1 or probes[0]['metric_name'] != metric:
        raise RuntimeError('missing or ambiguous Rust probe')
    probe = probes[0]
    if [column[0] for column in probe['scores']] != columns:
        raise RuntimeError('observed score columns differ from frozen inventory')
    ours = [struct.unpack('>d', bytes.fromhex(row[1]))[0] for row in probe['scores']]
    if not all(math.isfinite(value) for value in ours):
        raise RuntimeError('nonfinite Rust score')
    compared = []
    for index, column in enumerate(columns):
        references = [float(row['score_strings'][index]) for row in canonical['score_rows']]
        delta = max(abs(ours[index] - value) for value in references)
        compared.append({'column': column, 'ours': ours[index], 'canonical_values': references,
                         'max_abs_delta': delta, 'bound': bound, 'pass': delta <= bound})
    if hashes != [sha(path) for path in inputs] or sha(a.binary) != binary_sha:
        raise RuntimeError('input or executable changed during probe')
    result = {'metric': metric, 'factor': factor, 'command': command, 'config_sha256': sha(config_path),
              'binary_sha256': binary_sha, 'build_receipt_sha256': sha(a.build_receipt),
              'canonical_record_sha256': sha(a.canonical), 'manifest_sha256': sha(a.manifest),
              'input_sha256': hashes, 'width': canonical['width'], 'height': canonical['height'],
              'threads_requested': a.threads, 'gate_source': gate_source, 'comparisons': compared,
              'pass': all(row['pass'] for row in compared),
              'maps': 'not compared: selected primary functions expose scalar outputs',
              'timing': 'not acquired; native and canonical warmup clocks do not supply paired ratios'}
    if a.measure_memory:
        result['process_memory_sha256'] = sha(evidence / 'process-memory.txt')
        result['process_max_rss_kbytes'] = process_max_rss(evidence / 'process-memory.txt')
    (evidence / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    if not result['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
