#!/usr/bin/env python3
"""Freeze new matched model-lifetime cases after exact original-reference controls.

This reads receipts only. It runs no compiler, model, scorer or inference library.
Existing plans, observations and numerical bounds are preserved.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

if not __debug__:
    raise RuntimeError('Receipt assertions require Python without optimization')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def prepare(source, build_path, controls_path, native, native_receipt, output):
    source, build_path, controls_path, native, native_receipt = [
        Path(p).resolve(strict=True) for p in
        (source, build_path, controls_path, native, native_receipt)]
    old, build, controls, worker = map(read, (source, build_path, controls_path, native_receipt))
    assert build['passed'] is True and controls['passed'] is True
    assert controls['build_receipt_sha256'] == sha(build_path)
    assert controls['freeze_sha256'] == build['freeze_sha256'] == sha(build['freeze'])
    freeze_path = Path(build['freeze']); freeze = read(freeze_path)
    assert freeze['assets']['plan']['sha256'] == sha(source)
    bindings = dict(old['bindings'])

    def bind(path, expected=None):
        path = str(Path(path).resolve(strict=True)); actual = sha(path)
        assert expected is None or actual == expected, path
        assert path not in bindings or bindings[path] == actual, path
        bindings[path] = actual

    for path, expected in list(bindings.items()): bind(path, expected)
    for relative, expected in freeze['files'].items(): bind(freeze_path.parent / relative, expected)
    for records in (freeze['assets'], freeze['tools'], build['runtime_files']):
        for record in records.values(): bind(record['path'], record['sha256'])
    bind(freeze_path); bind(build_path); bind(controls_path); bind(source)
    bind(native, worker['binary_sha256']); bind(native_receipt); bind(__file__)
    binary = build['binary']; bind(binary['path'], binary['sha256'])
    cases = []; observed = 0
    expected_grid = {(name, t) for name in ('vmaf', 'vmaf-neg', 'vmaf-4k', 'vmaf-v1',
                     'vmaf-studio601', 'vmaf-neg-studio601') for t in (1, 8)}
    assert len(controls['cases']) == 12 and controls['passed_comparisons'] == 12
    for item in controls['cases']:
        assert item['passed'] is True
        bind(item['case'], item['case_sha256']); actual = read(item['case'])
        assert actual['passed'] is True and not actual['errors']
        bits = []
        for arm in actual['arms'].values():
            assert arm['process']['returncode'] == 0
            for key in ('stdout', 'stderr', 'command'):
                bind(arm['process'][key], arm['process'][key+'_sha256'])
            observations = arm['protocol']['observations']
            assert [o['label'] for o in observations] == ['COLD', 'SAMPLE', 'SAMPLE']
            bits.extend(o['f64_bits'] for o in observations)
        assert len(bits) == 6 and len(set(bits)) == 1
        observed += len(bits)
        originals = [c for c in old['cases'] if c['id'] == item['id']]
        assert len(originals) == 1
        case = copy.deepcopy(originals[0])
        assert (case['metric'], case['threads']) in expected_grid
        expected_grid.remove((case['metric'], case['threads']))
        assert case['width'] == case['height'] == 512
        case['id'] = case['metric']+'-fullcall-pair-cid22-t'+str(case['threads'])
        case['canonical']['command'][0] = binary['path']
        case['canonical']['label'] = 'libvmaf 3.2.0 / complete embedded-model lifetime'
        case['scope'] = ('Resident RGB8 conversion, embedded-model load/destroy, full single-frame '
                         'context/picture lifetime and scoring; input file reads and startup excluded')
        case['matched_scope'] = True
        cases.append(case)
    assert not expected_grid and observed == 72
    plan = dict(schema=1, binary=str(native), rust_source=worker['source_revision'],
                bindings=bindings, cases=cases,
                reference_control_gate='12 old/new comparisons; all72 scalar observations exact f64 bits',
                scope_qualification='Complete per-call model lifetime aligned; different parsers and '
                    'representations remain implementation details. No utilization or timing acceptance inferred.',
                ratio_definition='Rust/canonical; descriptive alternating-block medians')
    with Path(output).open('x') as stream:
        json.dump(plan, stream, indent=2, allow_nan=False); stream.write('\n')
    return dict(cases=len(cases), plan=str(output), plan_sha256=sha(output))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'build', 'controls', 'native', 'native-receipt', 'output'):
        p.add_argument('--'+name, required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.build, a.controls, a.native, a.native_receipt, a.output)))


if __name__ == '__main__': main()
