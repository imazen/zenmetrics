#!/usr/bin/env python3
"""Publish compact retained HDR reference controls, without qualifying timings."""
import argparse
import json
from pathlib import Path

from compare_canonical import sha
from reference_publication import register_outputs


def read(path):
    return json.loads(path.read_text())


def pointer(path):
    return dict(path=str(path.resolve()), sha256=sha(path))


def golden(root):
    result = read(root / 'result.json')
    stages = result['stages']
    raw = stages['byte-control']
    if raw['passed'] is not True or len(raw['files']) != 14:
        raise ValueError('fourteen exact original raw fixtures required')
    for name, row in raw['files'].items():
        if (row['exact_equal'] is not True or row['expected_sha256'] != row['observed_sha256']
                or sha(root / 'original-raw' / name) != row['expected_sha256']):
            raise ValueError('original raw fixture differs: ' + name)
    baseline = read(root / 'runtime-before/snapshot.json')
    for name in ('bytes', 'system212', 'resize2201'):
        if read(root / ('runtime-after-' + name) / 'snapshot.json') != baseline:
            raise ValueError('retained runtime closure changed: ' + name)
    variants = {}
    for name in ('system212', 'resize2201'):
        stage = stages[name]
        for stream in ('stdout', 'stderr'):
            if sha(root / (name + '-' + stream + '.log')) != stage[stream + '_sha256']:
                raise ValueError('retained stage log changed')
        rows = [line.split() for line in (root / (name + '-stdout.log')).read_text().splitlines()
                if line.startswith('SRGB_GOLDEN ')]
        if len(rows) != 24:
            raise ValueError('all twenty-four original golden observations required')
        failures = sum(float(row[7]) > 0.005 for row in rows)
        if (stage['passed'] is True) != (failures == 0):
            raise ValueError('golden gate and retained process receipt disagree')
        variants[name] = dict(cases=24, failures=failures,
            max_absolute_difference=max(float(row[7]) for row in rows),
            stdout=pointer(root / (name + '-stdout.log')), returncode=stage['returncode'])
    return dict(raw=pointer(root / 'result.json'), original_raw_exact=14,
        absolute_bound=0.005, variants=variants,
        runtime_files=len(baseline['files']), dependency_nodes=len(baseline['dependencies']),
        runtime_snapshot=pointer(root / 'runtime-before/snapshot.json'),
        original_overall_passed=result['passed'], selected_reference_passed=stages['resize2201']['passed'],
        qualification='Original HDR-VDP-2 sRGB goldens only. Octave 6.4/image 2.12 with unchanged '
            'image 2.20.1 imresize.m and private/interp_method.m; the full newer package is not installed. '
            'No HDR-VDP-3 numerical bound or paired timing qualification follows from this control.')


def preflight(root):
    result = read(root / 'result.json')
    if (result['passed'] is not True or result['cases'] != []
            or result['coverage'] != dict(score_cases_completed=0, score_cases_required=104, preflight_only=True)
            or result['matched_scope'] is not False or result['qualified_timing_cells'] != 0
            or len(result['preflight']) != 9):
        raise ValueError('complete passed preflight-only receipt required')
    for row in result['preflight']:
        if row['passed'] is not True or row['returncode'] != 0:
            raise ValueError('failed resident parameter control')
        child = root / ('preflight-' + row['id'])
        for stream in ('stdout', 'stderr'):
            if sha(child / ('control-' + stream + '.log')) != row[stream + '_sha256']:
                raise ValueError('preflight log changed')
        if read(child / 'runtime-before/snapshot.json') != read(child / 'runtime-after/snapshot.json'):
            raise ValueError('preflight runtime changed')
    return dict(raw=pointer(root / 'result.json'), controls=result['preflight'],
        source_plan_sha256=result['plan_sha256'], matched_scope=False, qualified_timing_cells=0,
        qualification='Parameter-array identity, copy-on-write, sealed cache and missing-key controls. '
            'Default/resident metric score controls and native comparisons remain separate.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('golden', type=Path)
    parser.add_argument('preflight', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    values = dict(goldens=golden(args.golden), resident_parameters=preflight(args.preflight))
    args.output.mkdir()
    path = args.output / 'controls.json'
    path.write_text(json.dumps(values, indent=2, allow_nan=False) + '\n')
    if path.stat().st_size > 30000:
        raise ValueError('compact publication exceeds generated-data limit')
    register_outputs([(path, 'provenance')])
    print(json.dumps(dict(output=str(path), golden_variants=values['goldens']['variants'],
                         resident_controls=9, qualified_timing_cells=0)))


if __name__ == '__main__':
    main()
