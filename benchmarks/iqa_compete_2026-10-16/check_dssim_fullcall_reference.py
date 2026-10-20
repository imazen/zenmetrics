#!/usr/bin/env python3
"""Root-admitted, fresh-process old/full-call DSSIM exact-bit controls.

Twelve bound raw pairs x budgets1/8 =24 comparisons /48 child processes.
No timing ratios, tolerance adjustments, skipped refusals or source recovery
claim for the older executable. Imports are standard library plus the frozen
read-only utility definitions in the sibling builder; no metric is imported.
"""
import argparse
import math
import os
from pathlib import Path
import re
import signal
import struct
import sys

sys.dont_write_bytecode = True
from build_dssim_fullcall_reference import (
    CHECKER, SOURCE, checked_process, clean_environment, digest, identity,
    positive_int, read_json, require, run_logged, runtime_snapshot,
    stop_requested, verify_assets, verify_freeze, write_json,
)


def parse_protocol(path, samples, threads, fullcall):
    try:
        lines = Path(path).read_text().splitlines()
    except UnicodeError as e:
        return dict(observations=[], errors=[f'invalid text protocol: {e}'], passed=False)
    errors = []
    if fullcall:
        expected = ['VERSION dssim-core-3.5.1 fullcall-v1',
                    'SCOPE context-images-compare-free',
                    f'THREADS requested={threads} environment=explicit effective=unverified']
        if lines[:3] != expected:
            errors.append('missing/altered fullcall metadata header')
        lines = lines[3:]
    observations = []
    for index, line in enumerate(lines):
        try:
            parts = line.split()
            require(len(parts) == 3, 'expected label elapsed_ns score')
            label, elapsed, scalar = parts
            require(label == ('COLD' if index == 0 else 'SAMPLE'), 'wrong label/order')
            require(re.fullmatch(r'[0-9]+', elapsed) is not None, 'invalid elapsed ns')
            ns = int(elapsed)
            require(re.fullmatch(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?', scalar)
                    is not None, 'invalid decimal score')
            score = float(scalar)
            require(math.isfinite(score), 'nonfinite score')
            observations.append(dict(label=label, elapsed_ns=ns, score_text=scalar,
                score=score, f64_bits=struct.pack('>d', score).hex()))
        except (ValueError, OverflowError) as e:
            errors.append(f'line{index}: {e}; raw={line!r}')
    if len(lines) != samples+1 or len(observations) != samples+1:
        errors.append(f'required one COLD plus {samples} SAMPLE records')
    return dict(observations=observations, errors=errors, passed=not errors)


def input_records(pair, base):
    width = positive_int(pair['width'], 'width')
    height = positive_int(pair['height'], 'height')
    expected_bytes = width*height*3
    records = {}
    for role in ('reference', 'distorted'):
        relative = Path(pair[role])
        require(not relative.is_absolute(), 'grid input must be relative to fixed directory')
        target = (base/relative).resolve(strict=True)
        require(target.is_relative_to(base), 'input escapes fixed directory')
        record = identity(target)
        require(record['bytes'] == expected_bytes, f'{role} raw RGB length mismatch')
        require(record['sha256'] == pair[role+'_sha256'], f'{role} raw RGB hash mismatch')
        records[role] = record
    return records


def validate_build(receipt_path, expected_hash, freeze_path, freeze):
    require(digest(receipt_path) == expected_hash, 'build receipt digest mismatch')
    build = read_json(receipt_path)
    require(build['schema'] == 'dssim-fullcall-build/1' and build['passed'] is True,
            'build receipt is not passed')
    require(Path(build['freeze']).resolve(strict=True) == freeze_path and
            build['freeze_sha256'] == digest(freeze_path), 'build freeze differs')
    require(build['source_revision'] == freeze['source_revision'], 'build source revision differs')
    require(build['new_source_sha256'] == freeze['files'][SOURCE], 'built C source differs')
    require(build['header_sha256'] == freeze['files']['dssim.h'], 'built header differs')
    require(build['library_sha256'] == freeze['assets']['library']['sha256'], 'built archive differs')
    expected_binary = receipt_path.parent/'dssim-fullcall-reference'
    require(build['binary']['path'] == str(expected_binary), 'unexpected build binary path')
    require(identity(expected_binary) == build['binary'], 'new binary drift')
    checked_process(build['compile'])
    return build


def check(args):
    freeze_path, freeze = verify_freeze(args.freeze, args.freeze_sha)
    require(Path(__file__).resolve(strict=True) == freeze_path.parent/CHECKER,
            'control driver must be the frozen copy')
    receipt_path = args.build_receipt.resolve(strict=True)
    build = validate_build(receipt_path, args.build_receipt_sha, freeze_path, freeze)
    output = args.output.absolute()
    output.mkdir()  # No resume/overwrite of successful or failed observations.
    (output/'scratch').mkdir()
    env = clean_environment(output/'scratch', 1)
    result = dict(schema='dssim-fullcall-controls/1', passed=False,
        freeze=str(freeze_path), freeze_sha256=args.freeze_sha,
        build_receipt=str(receipt_path), build_receipt_sha256=args.build_receipt_sha,
        expected_comparisons=24, expected_child_processes=48,
        samples_per_child=freeze['samples'], expected_observations=48*(1+freeze['samples']),
        cases=[], error=None, numerical_bound_changed=False,
        timing_qualification=False, old_wrapper_source_recovered=False,
        actual_effective_library_scheduling_verified=False)
    roots = [freeze['assets']['old_binary']['path'], build['binary']['path'],
             freeze['tools']['python']['path'], freeze['tools']['shell']['path']]
    runtime_before = None
    interrupted = None
    try:
        assets_before = verify_assets(freeze)
        write_json(output/'assets-before.json', assets_before)
        runtime_before = runtime_snapshot(roots, output, 'runtime-before', env,
                                         freeze['tools']['ldd']['path'])
        for path, record in build['runtime_files'].items():
            require(runtime_before['files'].get(path) == record, 'linked runtime differs from build')
        grid = read_json(freeze_path.parent/'pairs.json')
        require(grid['schema'] == 'primary-raw-crops/1' and len(grid['pairs']) == 12,
                'expected exact twelve-pair raw grid')
        require(digest(freeze_path.parent/'pairs.json') == freeze['assets']['pairs_manifest']['sha256'],
                'frozen grid differs from original manifest')
        pairs = grid['pairs']
        identities = {(p['pair'],p['size_name']) for p in pairs}
        require(identities == {(p,s) for p in ('pairA','pairB','pairC')
                 for s in ('64','odd','1mp','4k')}, 'grid missing/duplicate/unexpected pair')
        base = Path(freeze['input_directory']).resolve(strict=True)
        # Every refusal/mismatch is retained and fails the final gate. Independent
        # remaining cases still run; none is silently skipped or treated as passed.
        for pair in pairs:
            for threads in freeze['threads']:
                case_id = f"{pair['pair']}-{pair['size_name']}-t{threads}"
                case_dir = output/case_id
                case_dir.mkdir()
                case = dict(id=case_id, pair_manifest_record=pair, threads=threads,
                            requested_samples=freeze['samples'], passed=False,
                            arms={}, errors=[])
                child_env = clean_environment(output/'scratch', threads)
                try:
                    inputs = input_records(pair, base)
                    write_json(case_dir/'inputs-before.json', inputs)
                    case['inputs'] = inputs
                    for arm, binary in [('old',freeze['assets']['old_binary']['path']),
                                        ('fullcall',build['binary']['path'])]:
                        expected = (freeze['assets']['old_binary']['sha256'] if arm == 'old'
                                    else build['binary']['sha256'])
                        require(digest(binary) == expected, f'{arm} binary changed')
                        require(input_records(pair,base) == inputs, 'input drift before child')
                        argv = [binary, str(pair['width']), str(pair['height']),
                                inputs['reference']['path'], inputs['distorted']['path'],
                                str(threads), str(freeze['samples'])]
                        process = run_logged(argv, case_dir, arm, child_env,
                                             freeze['child_timeout_seconds'])
                        protocol = parse_protocol(process['stdout'], freeze['samples'],
                                                  threads, arm == 'fullcall')
                        observation = dict(process=process, protocol=protocol,
                            binary_sha256=expected,
                            thread_control=('outer environment; old C ignores argv5' if arm == 'old'
                                            else 'outer environment plus C setenv before first API call'),
                            library_utilization_qualified=False)
                        case['arms'][arm] = observation
                        write_json(case_dir/(arm+'.observations.json'), observation)
                        if process['returncode'] != 0 or process['timed_out'] or process['launch_error']:
                            case['errors'].append(arm+': process failure/refusal; raw logs retained')
                        if not protocol['passed']:
                            case['errors'].append(arm+': '+ '; '.join(protocol['errors']))
                        require(digest(binary) == expected, f'{arm} binary drift after child')
                        after = input_records(pair,base)
                        write_json(case_dir/(arm+'.inputs-after.json'), after)
                        require(after == inputs, 'input drift after child')
                    if not case['errors']:
                        old = case['arms']['old']['protocol']['observations']
                        new = case['arms']['fullcall']['protocol']['observations']
                        bits = [o['f64_bits'] for o in old+new]
                        require(len(bits) == 2*(1+freeze['samples']) and all(b==bits[0] for b in bits),
                                'old/new COLD/SAMPLE exact f64 bit mismatch')
                        case.update(passed=True, exact_f64_bits=bits[0])
                except Exception as e:
                    case['errors'].append(f'{type(e).__name__}: {e}')
                write_json(case_dir/'case.json', case)
                result['cases'].append(dict(id=case_id,passed=case['passed'],
                    case_path=str(case_dir/'case.json'), case_sha256=digest(case_dir/'case.json'),
                    exact_f64_bits=case.get('exact_f64_bits'), errors=case['errors']))
        after = verify_assets(freeze)
        write_json(output/'assets-after.json', after)
        require(after == assets_before, 'source/asset drift across controls')
        validate_build(receipt_path, args.build_receipt_sha, freeze_path, freeze)
        verify_freeze(freeze_path, args.freeze_sha)
        result['passed'] = len(result['cases']) == 24 and all(c['passed'] for c in result['cases'])
    except BaseException as e:
        result['error'] = f'{type(e).__name__}: {e}'
        if not isinstance(e, Exception):
            interrupted = e
    finally:
        # Preserve a terminal runtime snapshot even when a score/protocol failed.
        if interrupted is not None:
            result['passed'] = False
            result['runtime_error'] = 'interrupted; no after snapshot or qualification'
        else:
            try:
                runtime_after = runtime_snapshot(roots, output, 'runtime-after', env,
                                                freeze['tools']['ldd']['path'])
                require(runtime_before is not None and runtime_before['files'] == runtime_after['files'],
                        'runtime dependency closure drift or missing before snapshot')
            except Exception as e:
                result['passed'] = False
                result['runtime_error'] = f'{type(e).__name__}: {e}'
    result['completed_comparisons'] = len(result['cases'])
    result['passed_comparisons'] = sum(c['passed'] for c in result['cases'])
    write_json(output/'result.json', result)
    if interrupted is not None:
        raise interrupted
    return result


def main():
    signal.signal(signal.SIGTERM, stop_requested)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('freeze', 'build-receipt', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--freeze-sha', required=True)
    parser.add_argument('--build-receipt-sha', required=True)
    args = parser.parse_args()
    result = check(args)
    print(f"passed={result['passed']} comparisons={result['passed_comparisons']}/24; "
          f"result={args.output/'result.json'}")
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
