#!/usr/bin/env python3
"""Serialize source-bound primary FSIM/VSI parity cases under host heavy locks.

Numerical failures are retained and the remaining cases are still attempted.
This controller acquires no qualified timing ratios. Canonical failures also
receive an independent Rust probe, rather than hiding that arm's disposition.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time

from check_fixed_factor_oracle import check_manifest
from fixed_factor_oracle import write_artifact
from octave_primary import process_max_rss, sha
from supervision import scoring_command
import image_resize_source

BASE = Path(__file__).resolve().parent


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['inputs', 'references', 'rust-build', 'output']:
        p.add_argument('--' + key, required=True, type=Path)
    p.add_argument('--threads', default='1,all')
    p.add_argument('--families', default='fsim,vsi')
    p.add_argument('--factors', default='1,2,3,4')
    p.add_argument('--pairs', default='all')
    p.add_argument('--sizes', default='all')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--image-archive', type=Path)
    p.add_argument('--octave', type=Path, default=Path(shutil.which('octave-cli') or 'octave-cli'))
    p.add_argument('--run-heavy', type=Path, default=Path.home() / 'work/claudehints/scripts/run-heavy')
    a = p.parse_args()
    for key in ['inputs', 'references', 'rust_build', 'run_heavy', 'octave']:
        setattr(a, key, getattr(a, key).resolve(strict=True))
    a.output = a.output.resolve()
    if a.image_archive:
        a.image_archive = a.image_archive.resolve(strict=True)
        image_resize_source.validate_archive(a.image_archive)
    affinity = sorted(os.sched_getaffinity(0))
    cap = max(1, len(affinity) - 4)
    threads = sorted(set(cap if value == 'all' else int(value) for value in a.threads.split(',')))
    families = a.families.split(',')
    factors = [None] if a.factors == 'auto' else [int(value) for value in a.factors.split(',')]
    if not threads or min(threads) <= 0 or max(threads) > cap:
        p.error(f'thread budgets must be 1..{cap} under this host resource policy')
    if not families or not set(families) <= {'fsim', 'vsi'} or not factors or not set(factors) <= {None, 1, 2, 3, 4}:
        p.error('FSIM/VSI factors1-4 required')
    if len(families) != len(set(families)) or len(factors) != len(set(factors)):
        p.error('duplicate family or factor selection')
    inputs = json.loads(a.inputs.read_text())
    for option, field in [(a.pairs, 'pair'), (a.sizes, 'size_name')]:
        if option != 'all' and not set(option.split(',')) <= {row[field] for row in inputs['pairs']}:
            p.error(f'unknown {field} selection')
    references = json.loads(a.references.read_text())
    build = json.loads((a.rust_build / 'build-receipt.json').read_text())
    binary = a.rust_build.resolve() / 'target/release/iqa-compete'
    if sha(binary) != build['binary_sha256']:
        p.error('Rust binary differs from frozen receipt')
    selected = [row for row in inputs['pairs']
                if (a.pairs == 'all' or row['pair'] in a.pairs.split(','))
                and (a.sizes == 'all' or row['size_name'] in a.sizes.split(','))]
    if not selected:
        p.error('no selected input pairs')
    manifests = {}
    for family, factor in itertools.product(families, factors):
        carrier = 1 if factor is None else factor
        path = Path(references[family][str(carrier)]).resolve(strict=True)
        manifest = check_manifest(path)
        if manifest['source']['name'] != family + '-matlab' or manifest['patch']['factor'] != carrier:
            p.error('reference identity/factor mismatch')
        manifests[(family, factor)] = path
    for row in selected:
        for role in ['reference', 'distorted']:
            if sha(a.inputs.parent / row[role]) != row[role + '_sha256']:
                p.error('input SHA256 mismatch')
    helpers = {name: sha(BASE / name) for name in
               ['run_octave_panel.py', 'octave_primary.py', 'probe_octave_parity.py',
                'check_fixed_factor_oracle.py', 'fixed_factor_oracle.py', 'supervision.py', 'image_resize_source.py']}
    identity = {'schema': 'primary-octave-panel/1', 'inputs_sha256': sha(a.inputs),
                'references_sha256': sha(a.references), 'build_receipt_sha256': sha(a.rust_build / 'build-receipt.json'),
                'binary_sha256': sha(binary), 'helpers': helpers, 'families': families,
                'factors': factors, 'threads': threads, 'input_selection': selected,
                'affinity': affinity, 'heavy_job_thread_cap': cap, 'run_heavy_sha256': sha(a.run_heavy),
                'octave_executable': str(a.octave), 'octave_sha256': sha(a.octave),
                'qualified_ratios': False,
                'image_archive_sha256': sha(a.image_archive) if a.image_archive else None}
    if a.resume:
        if (a.output / 'identity.json').read_bytes() != encoded(identity):
            p.error('resume identity differs; preserve this attempt and use a fresh output')
    else:
        a.output.mkdir()
        write_artifact(a.output / 'identity.json', encoded(identity))
    output = a.output.resolve()
    scratch = output / 'scratch'
    scratch.mkdir(exist_ok=True)
    env = os.environ.copy()
    env['TMPDIR'] = str(scratch)
    locks = [Path.home() / 'tmp/lilith-heavy.lock', Path.home() / 'tmp/lilith-wsl-heavy.lock']
    outcomes = []
    cases = list(itertools.product(selected, families, factors, threads))
    print(f'PLAN {len(cases)} cases; thread budgets {threads}; ratios unqualified', flush=True)

    def execute(command, jobs, log):
        wrapped = ['flock', str(locks[0]), 'flock', str(locks[1]), str(a.run_heavy),
                   '--mem', '12G', '--jobs', str(jobs), '--', *map(str, command)]
        print('COMMAND', repr(wrapped), flush=True)
        with log.open('x') as stream:
            return subprocess.run(wrapped, cwd=BASE, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT).returncode

    for index, (row, family, factor, budget) in enumerate(cases, 1):
        for name, expected in helpers.items():
            if sha(BASE / name) != expected:
                raise RuntimeError('helper changed during frozen panel')
        if sha(a.octave) != identity['octave_sha256']:
            raise RuntimeError('Octave executable changed during frozen panel')
        if a.image_archive:
            image_resize_source.validate_archive(a.image_archive)
        factor_name = 'auto' if factor is None else f'k{factor}'
        key = f'{family}-{factor_name}-{row["pair"]}-{row["size_name"]}-t{budget}'
        directory = output / key
        if a.resume and (directory / 'outcome.json').exists():
            outcome = json.loads((directory / 'outcome.json').read_text())
            for relative, expected in outcome['artifact_sha256'].items():
                if sha(directory / relative) != expected:
                    raise RuntimeError('retained case artifact changed')
            outcomes.append(outcome)
            print(f'RETAINED {index}/{len(cases)} {key} {outcome["status"]}', flush=True)
            continue
        if a.resume and directory.exists():
            # Preserve partial attempts, including logs and crash workspaces.
            partials = output / 'partials'
            partials.mkdir(exist_ok=True)
            attempt = 1
            while (partials / f'{key}.partial.{attempt}').exists():
                attempt += 1
            directory.rename(partials / f'{key}.partial.{attempt}')
        directory.mkdir()
        manifest = manifests[(family, factor)]
        reference, distorted = [a.inputs.parent.resolve() / row[role] for role in ['reference', 'distorted']]
        command = ['python3', '-B', str(BASE / 'octave_primary.py'), '--manifest', str(manifest),
                   '--reference', str(reference), '--distorted', str(distorted),
                   '--width', str(row['width']), '--height', str(row['height']),
                   '--threads', str(budget), '--samples', '1', '--evidence', str(directory / 'canonical'),
                   '--measure-memory', '--octave', str(a.octave)]
        if a.image_archive:
            command += ['--image-archive', str(a.image_archive)]
        if factor is None:
            command += ['--automatic-factor']
        canonical_rc = execute(command, budget, directory / 'canonical-wrapper.log')
        canonical = None
        if canonical_rc == 0:
            canonical = json.loads((directory / 'canonical/results.json').read_text())
        comparable = canonical is not None and all(r['all_finite'] for r in canonical['score_rows'])
        if comparable:
            command = ['python3', '-B', str(BASE / 'probe_octave_parity.py'), '--binary', str(binary),
                       '--build-receipt', str(a.rust_build.resolve() / 'build-receipt.json'),
                       '--canonical', str(directory / 'canonical/results.json'), '--manifest', str(manifest),
                       '--reference', str(reference), '--distorted', str(distorted), '--threads', str(budget),
                       '--evidence', str(directory / 'rust'), '--measure-memory']
            rust_rc = execute(command, budget, directory / 'rust-wrapper.log')
            result_path = directory / 'rust/results.json'
            if result_path.exists():
                result = json.loads(result_path.read_text())
                status = ('parity_pass' if result['pass'] and rust_rc == 0 else
                          'parity_fail' if not result['pass'] and rust_rc == 1 else
                          'rust_process_or_contract_failure')
            else:
                if rust_rc == 0:
                    raise RuntimeError('successful Rust comparison has no result record')
                result, status = None, 'rust_process_or_contract_failure'
        else:
            # Still execute Rust independently when the primary reference failed.
            config = {'width': row['width'], 'height': row['height'], 'reference': str(reference),
                      'distorted': str(distorted), 'metric_name': family if factor is None else f'{family}-nodownscale' if factor == 1 else f'{family}-k{factor}',
                      'threads': budget, 'probe_only': True, 'only_mode': 'score', 'timing_phase': 'warm'}
            write_artifact(directory / 'rust-independent.json', encoded(config))
            scorer = scoring_command([str(binary), str(directory / 'rust-independent.json')],
                                     system='Linux', seconds=120,
                                     memory_log=directory / 'rust-process-memory.txt')
            command = ['env', f'RAYON_NUM_THREADS={budget}', f'OMP_NUM_THREADS={budget}',
                       f'OPENBLAS_NUM_THREADS={budget}', f'MKL_NUM_THREADS={budget}', 'LC_ALL=C',
                       *scorer]
            rust_rc = execute(command, budget, directory / 'rust-wrapper.log')
            result = None
            status = 'canonical_process_failure' if canonical_rc else 'canonical_nonfinite'
            if rust_rc == 0:
                rows = [json.loads(line.removeprefix('IQA_PROBE ')) for line in
                        (directory / 'rust-wrapper.log').read_text().splitlines()
                        if line.startswith('IQA_PROBE ')]
                if len(rows) != 1:
                    raise RuntimeError('independent Rust probe returned no unique score record')
                scores = [{'column': column, 'score_bits': bits,
                           'value_string': format(struct.unpack('>d', bytes.fromhex(bits))[0], '.17g')}
                          for column, bits in rows[0]['scores']]
                write_artifact(directory / 'rust-independent-result.json',
                               encoded({'scores': scores, 'process_max_rss_kbytes':
                                        process_max_rss(directory / 'rust-process-memory.txt')}))
        for role, path in [('reference', reference), ('distorted', distorted)]:
            if sha(path) != row[role + '_sha256']:
                raise RuntimeError('input changed during case')
        if sha(binary) != build['binary_sha256']:
            raise RuntimeError('Rust executable changed during case')
        artifacts = {str(path.relative_to(directory)): sha(path) for path in directory.rglob('*') if path.is_file()}
        outcome = {'case': key, 'pair': row['pair'], 'size': row['size_name'], 'width': row['width'],
                   'height': row['height'], 'family': family, 'factor': factor, 'threads': budget,
                   'status': status, 'canonical_rc': canonical_rc, 'rust_rc': rust_rc,
                   'comparison': result, 'artifact_sha256': artifacts}
        write_artifact(directory / 'outcome.json', encoded(outcome))
        outcomes.append(outcome)
        print(f'CASE_DONE {index}/{len(cases)} {key} {status}', flush=True)
        with (output / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps({'case': key, 'status': status, 'completed': index,
                                     'total': len(cases), 'unix_seconds': time.time()}) + '\n')
    counts = {status: sum(r['status'] == status for r in outcomes) for status in sorted({r['status'] for r in outcomes})}
    summary = {'attempted_cases': len(outcomes), 'planned_cases': len(cases), 'counts': counts,
               'execution_complete': len(outcomes) == len(cases), 'qualified_timing_cells': 0}
    write_artifact(output / 'summary.json', encoded(summary))
    print(json.dumps(summary), flush=True)
    if counts.get('parity_pass', 0) != len(cases):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
