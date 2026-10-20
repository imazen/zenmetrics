#!/usr/bin/env python3
"""Prepare a new immutable sRGB-display cohort from retained pair identities.

No scorer, compiler or Octave process is launched. The caller supplies the
runtime identity and source-bound full current worker in a later new plan.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from first_boundary_hdrvdp_octave import script

REVISION = '4f89155cd8ef766c285fcfbfed9d40def047a836'
PRODUCTION_SHA = '6b790071435a1bcc943f7d3da17b2353f6de434d4a526abb3dfeee4560c31b94'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def prepare(frozen, predecessor, output, octave, octave_version, image_version):
    frozen, predecessor, octave = [p.resolve(strict=True) for p in (frozen, predecessor, octave)]
    source = frozen / 'first_boundary_production_hdrvdp_aic.rs'
    if sha(source) != PRODUCTION_SHA:
        raise ValueError('successor production body differs from 4f89155')
    old = json.loads(predecessor.read_text())
    pairs = [case for case in old['cases'] if case['metric'] == 'hdrvdp-aic2026']
    if not pairs:
        raise ValueError('retained AIC pair inventory required; its scoring convention is not reused')
    output.mkdir()
    root = output.resolve()
    # Freeze this emitter and adapter, too; root never edits a running helper.
    helpers = root / 'helpers'
    helpers.mkdir()
    for name in ('first_boundary_hdrvdp_plan.py', 'first_boundary_hdrvdp_octave.py'):
        shutil.copy2(Path(__file__).with_name(name), helpers / name)
    bindings = {str(source): sha(source), str(octave): sha(octave), str(predecessor): sha(predecessor)}
    for path in helpers.iterdir():
        bindings[str(path)] = sha(path)
    for tag in ('official-v2', 'official-v3', 'statistics'):
        for path in sorted((frozen / tag).rglob('*')):
            if path.is_file():
                bindings[str(path.resolve())] = sha(path)
    for case in pairs:
        for field in ('reference', 'distorted'):
            path = Path(case[field]).resolve(strict=True)
            expected = old['bindings'].get(str(path))
            if expected is None or sha(path) != expected:
                raise ValueError('predecessor raw pair changed or unbound: ' + str(path))
            bindings[str(path)] = expected
    cases = []
    for pair in pairs:
        for metric, tag in [('hdrvdp', 'official-v2'), ('hdrvdp3', 'official-v3')]:
            for phase, samples in [('first', 1), ('warm', 5)]:
                name = pair['id'].replace('hdrvdp-aic2026', metric, 1) + '-' + phase + '-srgb-display'
                if Path(name).name != name:
                    raise ValueError('unsafe case id')
                config = dict(schema=1, metric=metric, phase=phase, samples=samples,
                    width=pair['width'], height=pair['height'], threads=pair['threads'],
                    reference=pair['reference'], distorted=pair['distorted'],
                    tree=str(frozen / tag), octave=str(octave), octave_version=octave_version,
                    image_version=image_version, geomean=str(frozen / 'statistics/geomean.m'), bindings=bindings)
                config_path = root / (name + '.json')
                write(config_path, config)
                entries = {}
                for role in ('candidate', 'direct-control'):
                    path = root / (name + '-' + role + '.m')
                    text = script(config, direct_control=role == 'direct-control')
                    with path.open('x') as stream:
                        stream.write(text)
                    entries[role] = dict(command=[str(octave), '--no-gui', '--no-init-file',
                        '--no-history', '--quiet', str(path)], script_sha256=sha(path))
                cases.append(dict(id=name, metric=metric, phase=phase, width=pair['width'],
                    height=pair['height'], threads=pair['threads'], samples=samples,
                    config=str(config_path), config_sha256=sha(config_path), **entries,
                    native_config=dict(width=pair['width'], height=pair['height'],
                        reference=pair['reference'], distorted=pair['distorted'],
                        metric_name=metric, only_mode='score', timing_phase='cold' if phase == 'first' else 'warm',
                        probe_only=False, required_samples=samples, locked_process_accounting=True),
                    absolute_bound=None, native_parity_qualified=False, matched_scope=False,
                    reference_output='Q_JOD' if metric == 'hdrvdp3' else 'Q'))
    result = dict(schema=1, production_revision=REVISION, production_module_sha256=PRODUCTION_SHA,
        predecessor_plan_sha256=sha(predecessor), cases=cases, bindings=bindings,
        runtime_contract=dict(octave_version=octave_version, image_version=image_version,
            matches_documented_v2_golden_runtime=octave_version == '11.1.0' and image_version == '2.20.1'),
        missing=['Current full-worker binary/source/build receipt must be bound in a new execution plan',
                 'Official runtime shared-library and image-package closure hashes must be bound before acquisition',
                 'Independent original-call exact-bit controls in fresh processes',
                 'Native-versus-original numerical observations and any applicable acceptance gate'],
        qualified_timing_cells=0, scope='RGB8 /255, original sRGB-display 30 ppd; v3 quality, age24, surround none, LED-LCD-sRGB, Q_JOD; original parameter-file I/O timed',
        qualification='Preparation only. Prior typed-input refusals and 200-nit AIC results remain historical, separate cohorts.')
    write(root / 'cohort.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frozen', type=Path)
    parser.add_argument('predecessor', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--octave', required=True, type=Path)
    parser.add_argument('--octave-version', required=True)
    parser.add_argument('--image-version', required=True)
    args = parser.parse_args()
    result = prepare(args.frozen, args.predecessor, args.output, args.octave,
                     args.octave_version, args.image_version)
    print(json.dumps(dict(cases=len(result['cases']), qualified_timing_cells=0, output=str(args.output))))


if __name__ == '__main__':
    main()
