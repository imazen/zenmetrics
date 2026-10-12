#!/usr/bin/env python3
"""Expand declared IQA studies into immutable cases and zenbench bundle specs.

This tool never measures a workload, invents correctness, or enrolls a worker.
Unknown/unsupported/unqualified cells remain in the inventory.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / 'benchmarks/heaptrack/drivers/cpu_profile/src'
PASSES = {'wall', 'os_peak', 'heaptrack', 'driver'}
WARM = {'ssim2', 'butteraugli', 'dssim', 'zensim'}
HDR = {'hdrvdp', 'hdrvdp3'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError('metadata exceeds 8 MiB')
    return json.loads(path.read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f'nonfinite JSON: {x}')))


def keys(value, required, optional=()):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ValueError(f'expected fields {sorted(required)} with optional {sorted(optional)}')


def positive(value, label, maximum=None, zero=False):
    if type(value) is not int or value < (0 if zero else 1) or (maximum and value > maximum):
        raise ValueError(f'invalid {label}')
    return value


def token(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9._-]{1,256}', value):
        raise ValueError('identity needs 1..256 ASCII letters/digits/dot/dash/underscore')
    return value


def distinct(values, label):
    if not isinstance(values, list) or not values or len({canonical(x) for x in values}) != len(values):
        raise ValueError(f'{label} must be nonempty and distinct')
    return values


def inventory():
    registry = ROOT / 'crates/zenmetrics-cli/src/metrics/mod.rs'
    harness = DRIVER / 'bin/new_metrics_wall.rs'
    source = registry.read_text().split('pub enum MetricKind {', 1)[1].split('impl MetricKind', 1)[0]
    names = re.findall(r'#\[value\(name = "([^"]+)"\)\]', source)
    cold = set(re.findall(r'\("([^"]+)", MetricKind::', harness.read_text().split('const METRICS:', 1)[1].split('];', 1)[0]))
    if not names or len(set(names)) != len(names) or not cold <= set(names):
        raise ValueError('registry/driver inventory cannot be resolved')
    rows = []
    for name in names:
        reasons = []
        if name.endswith('-gpu'):
            reasons.append('GPU workload driver not qualified')
        if name in HDR:
            reasons.append('absolute-nits workload required')
        if name not in cold and not reasons:
            reasons.append('SDR fleet driver does not expose this variant')
        rows.append({'metric': name, 'cold_rgb8': name in cold,
                     'warm_rgb8': name in WARM, 'unsupported_reasons': reasons})
    return {'schema': 1, 'kind': 'zenmetrics_benchmark_inventory', 'metrics': rows,
            'sources': {str(p.relative_to(ROOT)): file_sha(p) for p in [registry, harness, DRIVER / 'fleet_case.rs']}}


def bound_file(base, value):
    keys(value, ['path', 'sha256'])
    path = (base / value['path']).resolve()
    if not re.fullmatch('[a-f0-9]{64}', value['sha256']) or not path.is_file() or file_sha(path) != value['sha256']:
        raise ValueError(f'file hash mismatch: {path}')
    return {'path': str(path), 'sha256': value['sha256'], 'bytes': path.stat().st_size}


def expand(spec_path):
    spec_path = Path(spec_path).resolve()
    spec, base = load(spec_path), spec_path.parent
    keys(spec, ['schema', 'study', 'metrics', 'pairs', 'testbeds', 'reuse', 'threads',
                'passes', 'replicates', 'operations', 'timeout_seconds', 'admission_seconds', 'peak_mem_bytes'])
    if spec['schema'] != 1:
        raise ValueError('unknown study schema')
    token(spec['study'])
    for label in ['metrics', 'pairs', 'testbeds', 'reuse', 'threads', 'passes']:
        distinct(spec[label], label)
    for label, maximum in [('replicates', 1000), ('operations', 1024), ('timeout_seconds', 86400)]:
        positive(spec[label], label, maximum)
    positive(spec['admission_seconds'], 'admission_seconds', 86400, zero=True)
    positive(spec['peak_mem_bytes'], 'peak_mem_bytes')
    for threads in spec['threads']:
        positive(threads, 'threads', 1024)
    if not set(spec['passes']) <= PASSES or not set(spec['reuse']) <= {'cold', 'warm_reference'}:
        raise ValueError('unknown pass or reuse boundary')
    inv = inventory()
    metrics = {r['metric']: r for r in inv['metrics']}
    if not set(spec['metrics']) <= metrics.keys():
        raise ValueError('unknown CLI metric')
    pairs = []
    for pair in spec['pairs']:
        keys(pair, ['id', 'origin', 'content', 'width', 'height', 'stride_bytes', 'pixel_format', 'reference', 'distorted'])
        token(pair['id']); token(pair['origin']); token(pair['content'])
        positive(pair['width'], 'width'); positive(pair['height'], 'height')
        positive(pair['stride_bytes'], 'stride_bytes')
        item = dict(pair)
        for side in ['reference', 'distorted']:
            item[side] = bound_file(base, pair[side])
            if item[side]['bytes'] != pair['stride_bytes'] * pair['height']:
                raise ValueError('input byte length differs from stride * height')
        pairs.append(item)
    if len({p['id'] for p in pairs}) != len(pairs):
        raise ValueError('duplicate pair ID')
    testbeds = []
    for bed in spec['testbeds']:
        keys(bed, ['id', 'machine', 'domain', 'target'], ['artifacts', 'qualification'])
        for label in ['id', 'machine', 'domain', 'target']:
            token(bed[label])
        item = dict(bed, artifacts={})
        for variant, artifact in bed.get('artifacts', {}).items():
            if variant not in {'latency', 'memory'}:
                raise ValueError('unknown driver artifact')
            keys(artifact, ['binary', 'build'])
            item['artifacts'][variant] = {name: bound_file(base, artifact[name]) for name in ['binary', 'build']}
            build = load(Path(item['artifacts'][variant]['build']['path']))
            keys(build, ['schema', 'target', 'binary_sha256', 'source_revision', 'dependency_lock_sha256',
                         'compiler', 'features', 'rustflags', 'profile', 'instrumented'])
            if (build['schema'] != 1 or build['target'] != bed['target'] or build['profile'] != 'release'
                    or build['instrumented'] != (variant == 'memory')
                    or build['binary_sha256'] != item['artifacts'][variant]['binary']['sha256']):
                raise ValueError('build metadata differs from declared artifact/target/boundary')
        if 'qualification' in bed:
            item['qualification'] = bound_file(base, bed['qualification'])
        testbeds.append(item)
    if len({b['id'] for b in testbeds}) != len(testbeds):
        raise ValueError('duplicate testbed ID')
    identity = dict(spec, pairs=pairs, testbeds=testbeds, inventory_sha=digest(inv))
    count = (len(spec['metrics']) * len(pairs) * len(testbeds) * len(spec['reuse'])
             * len(spec['threads']) * len(spec['passes']) * spec['replicates'])
    if count > 100000:
        raise ValueError('study exceeds 100,000 declared cells; split into explicit panels')
    # Relocation must not change content identity; local source paths are packaging hints.
    def without_paths(value):
        if isinstance(value, dict):
            return {k: without_paths(v) for k, v in value.items() if k not in {'path'}}
        if isinstance(value, list):
            return [without_paths(v) for v in value]
        return value
    study_sha = digest(without_paths(identity))
    rows = []
    for metric, pair, bed, reuse, threads, measurement, replicate in itertools.product(
            spec['metrics'], pairs, testbeds, spec['reuse'], spec['threads'], spec['passes'], range(spec['replicates'])):
        case = {'schema': 1, 'metric': metric, 'width': pair['width'], 'height': pair['height'],
                'stride_bytes': pair['stride_bytes'], 'reference': 'reference.rgb',
                'distorted': 'distorted.rgb', 'pixel_format': pair['pixel_format'], 'reuse': reuse, 'threads': threads}
        contract = {'study_sha': study_sha, 'metric': metric, 'pair': without_paths(pair),
                    'testbed': without_paths(bed), 'reuse': reuse, 'threads': threads,
                    'measurement': measurement, 'replicate': replicate, 'operations': spec['operations']}
        cell_sha = digest(contract)
        reasons = list(metrics[metric]['unsupported_reasons'])
        if pair['pixel_format'] != 'rgb8_srgb' or pair['stride_bytes'] < pair['width'] * 3:
            reasons.append('driver requires RGB8 sRGB with valid row stride')
        if reuse == 'warm_reference' and not metrics[metric]['warm_rgb8']:
            reasons.append('cached-reference route unavailable in driver')
        if metric.startswith('iwssim') and min(pair['width'], pair['height']) < 176:
            reasons.append('IW-SSIM requires minimum dimension 176')
        if metric.startswith('vmaf') and (pair['width'] % 2 or pair['height'] % 2):
            reasons.append('VMAF requires even dimensions')
        if measurement == 'heaptrack' and 'linux' not in bed['target']:
            reasons.append('heaptrack interception backend requires Linux')
        variant = 'memory' if measurement == 'driver' else 'latency'
        artifact = bed['artifacts'].get(variant)
        status, expected = ('unsupported' if reasons else 'unqualified'), None
        if not reasons and artifact and 'qualification' in bed:
            q = load(Path(bed['qualification']['path']))
            keys(q, ['schema', 'target', 'native_arch_verified', 'inventory_sha', 'cases'])
            if q['schema'] != 1 or q['target'] != bed['target'] or q['native_arch_verified'] is not True or q['inventory_sha'] != digest(inv):
                raise ValueError('qualification target/native execution/inventory mismatch')
            # Qualification binds build bytes, exact input bytes and all timing settings.
            proof_id = digest({'case': case, 'inputs': [pair[s]['sha256'] for s in ['reference', 'distorted']],
                               'binary_sha256': artifact['binary']['sha256'], 'operations': spec['operations']})
            expected = q['cases'].get(proof_id)
            if expected is not None:
                if (not isinstance(expected, dict) or set(expected) != {'score_bits'}
                        or not isinstance(expected['score_bits'], list)
                        or len(expected['score_bits']) != spec['operations']
                        or any(type(x) is not int or not 0 <= x < 2**64 for x in expected['score_bits'])):
                    raise ValueError('qualification needs exactly one primary score-bit value per operation')
                status = 'ready'
        if status == 'unqualified':
            reasons.append('missing matching release artifact and independently reviewed case qualification')
        row = {'schema': 1, 'study_sha': study_sha, 'cell_sha': cell_sha, 'status': status,
               'reasons': reasons, 'contract': contract, 'case': case, 'testbed': bed['id'],
               'machine': bed['machine'], 'domain': bed['domain'], 'target': bed['target'],
               'measurement': measurement, 'replicate': replicate, 'expected_correctness': expected,
               'sources': {'inputs': {s: pair[s] for s in ['reference', 'distorted']}, 'artifact': artifact}}
        rows.append(row)
    if len({r['cell_sha'] for r in rows}) != len(rows):
        raise ValueError('duplicate study cell')
    summary = {'schema': 1, 'kind': 'zenmetrics_benchmark_study', 'study': spec['study'],
               'study_sha': study_sha, 'inventory_sha': digest(inv), 'declared_cells': len(rows),
               'statuses': {s: sum(r['status'] == s for r in rows) for s in ['ready', 'unsupported', 'unqualified']},
               'planned_driver_launches': sum(r['status'] == 'ready' for r in rows),
               'operations_per_launch': spec['operations'], 'timeout_seconds': spec['timeout_seconds'],
               'admission_seconds': spec['admission_seconds'], 'peak_mem_bytes': spec['peak_mem_bytes'],
               'inference': 'descriptive observations; no confirmatory decision'}
    return inv, summary, rows


def write_study(spec, output):
    inv, summary, rows = expand(spec)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'inventory.json').write_bytes(canonical(inv) + b'\n')
    (output / 'study.json').write_bytes(canonical(summary) + b'\n')
    with (output / 'cells.jsonl').open('wb') as stream:
        for row in rows:
            stream.write(canonical(row) + b'\n')
    for row in rows:
        if row['status'] != 'ready':
            continue
        cell = output / row['cell_sha']
        cell.mkdir()
        (cell / 'case.json').write_bytes(canonical(row['case']) + b'\n')
        files = {'case.json': {'source': str(cell / 'case.json'), 'executable': False},
                 'driver': {'source': row['sources']['artifact']['binary']['path'], 'executable': True},
                 'build.json': {'source': row['sources']['artifact']['build']['path'], 'executable': False}}
        for side in ['reference', 'distorted']:
            files[f'{side}.rgb'] = {'source': row['sources']['inputs'][side]['path'], 'executable': False}
        plan = {'schema': 1, 'program': 'driver', 'args': ['fleet', 'case.json'],
                'env': {'RAYON_NUM_THREADS': str(row['case']['threads'])}, 'backend': row['measurement'],
                'operations': summary['operations_per_launch'], 'timeout_seconds': summary['timeout_seconds'],
                'expected_correctness': row['expected_correctness'],
                'workload': {'study_sha': summary['study_sha'], 'cell_sha': row['cell_sha'],
                             'case': row['case'], 'timing_boundary': 'decoded_input_score'}}
        (cell / 'bundle-spec.json').write_bytes(canonical({'schema': 1, 'target': row['target'], 'files': files, 'plan': plan}) + b'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    subs.add_parser('inventory')
    dry = subs.add_parser('plan'); dry.add_argument('spec', type=Path)
    emit = subs.add_parser('write'); emit.add_argument('spec', type=Path); emit.add_argument('output', type=Path)
    args = parser.parse_args()
    try:
        result = inventory() if args.command == 'inventory' else (expand(args.spec)[1] if args.command == 'plan' else write_study(args.spec, args.output))
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f'benchmark study refused: {error}\n')


if __name__ == '__main__':
    main()
