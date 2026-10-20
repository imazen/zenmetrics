#!/usr/bin/env python3
"""Verify retained full-call VMAF evidence and publish a new plan; never execute it."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile

# Reviewed producer and the two clock bodies. Never import execution machinery.
PRODUCER_SHA = '01e2f8d6089d6e3087a46ca726ba78c17af2946e5519d9c6166c7d694244470c'
SOURCE_HASHES = {
    'vmaf_fullcall_oracle.py': PRODUCER_SHA,
    'libvmaf_rgb_compare.cpp': 'd08698529ee31bc7861b568329ad884e57e08c9775189e535a7eb40a450aa326',
    'libvmaf_rgb_fullcall.cpp': '87946a32c20215ac65b4f7e90c1dd5efdd899f1d0765ee3caca237d75a1b7af3',
}
ROUTES = {'vmaf', 'vmaf-neg', 'vmaf-4k', 'vmaf-v1',
          'vmaf-studio601', 'vmaf-neg-studio601'}
FLAGS = ['-O3', '-Wall', '-Wextra', '-ffp-contract=off', '-std=c++17']
THREAD_VARIABLES = ('RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                    'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
NATIVE_FILES = (
    'benchmarks/heaptrack/drivers/cpu_profile/src/bin/iqa_compete.rs',
    'benchmarks/heaptrack/drivers/cpu_profile/src/bin/iqa_compete_timing/mod.rs',
    'crates/zenmetrics-cli/src/metrics/mod.rs',
    'crates/zenmetrics-cli/src/metrics/vmaf.rs', 'crates/vmaf/src/v0.rs',
    'crates/vmaf/src/lib.rs', 'crates/vmaf/src/v0_score.rs', 'crates/vmaf/src/score.rs',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: '+key)
            result[key] = value
        return result

    def finite(text):
        value = float(text)
        require(math.isfinite(value), 'nonfinite JSON number')
        return value

    def invalid(text):
        raise ValueError('nonfinite JSON constant: '+text)

    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_float=finite, parse_constant=invalid)


class Evidence:
    def __init__(self):
        self.bindings = {}

    def bind(self, path, expected=None):
        path = Path(path).absolute()
        require(path.is_file(), 'missing evidence: '+str(path))
        actual = sha(path)
        require(expected is None or actual == expected, 'hash mismatch: '+str(path))
        require(str(path) not in self.bindings or self.bindings[str(path)] == actual,
                'evidence changed during verification: '+str(path))
        self.bindings[str(path)] = actual
        if path.suffix == '.json':
            read(path)  # Bound JSON is evidence too, even when otherwise opaque.
        return path

    def json(self, path, expected=None):
        return read(self.bind(path, expected))

    def identity(self, record):
        path = self.bind(record['path'], record['sha256'])
        require(set(record) == {'path', 'resolved', 'bytes', 'sha256'} and
                Path(record['path']).is_absolute() and
                str(path.resolve(strict=True)) == record['resolved'] and
                type(record['bytes']) is int and path.stat().st_size == record['bytes'],
                'file identity mismatch: '+str(path))
        return record

    def process(self, directory, stem, argv, scratch, threads, timeout, recorded=None,
                ldd=False):
        directory = Path(directory)
        process = self.json(directory/(stem+'.process.json'))
        require(recorded is None or process == recorded, 'inline process receipt differs')
        require(recorded is None or (type(recorded['returncode']) is int and
                recorded['timed_out'] is False and recorded['launch_error'] is None and
                recorded['interrupted'] is None), 'invalid inline process outcome')
        require(process['argv'] == argv and type(process['returncode']) is int and
                process['timed_out'] is False and process['launch_error'] is None and
                process['interrupted'] is None, 'process argv/outcome mismatch: '+stem)
        require(process['returncode'] == 0 or (ldd and process['returncode'] == 1),
                'unsuccessful process: '+stem)
        for key, suffix in [('command', '.command.json'), ('stdout', '.stdout'),
                            ('stderr', '.stderr')]:
            expected_path = directory/(stem+suffix)
            require(process[key] == str(expected_path), 'process output path mismatch')
            self.bind(expected_path, process[key+'_sha256'])
        environment = {key: str(threads) for key in THREAD_VARIABLES}
        environment.update(OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE',
                           TMPDIR=str(scratch), LC_ALL='C', PATH='/usr/bin:/bin')
        require(self.json(process['command']) == dict(argv=argv, cwd=str(directory),
                timeout_seconds=timeout, environment=environment),
                'retained command/environment mismatch: '+stem)
        return process

    def snapshot(self, directory, roots, frozen):
        """Replay the producer's closure walk from retained text, never call ldd."""
        directory = Path(directory)
        snapshot = self.json(directory/'snapshot.json')
        require(snapshot['roots'] == roots, 'runtime roots mismatch')
        queue = list(roots)
        found, scanned, used = {}, set(), 0
        while queue:
            name = queue.pop(0)
            if name in found:
                continue
            record = snapshot['files'][name]
            require(record['path'] == name, 'runtime map key mismatch')
            self.identity(record)
            found[name] = record
            if record['resolved'] in scanned:
                continue
            scanned.add(record['resolved'])
            with Path(name).open('rb') as stream:
                elf = stream.read(4) == b'\x7fELF'
            if not elf:
                continue
            require(used < len(snapshot['ldd_logs']), 'missing ldd transcript')
            process = self.process(directory, f'ldd-{used:03}',
                [frozen['tools']['ldd']['path'], record['resolved']],
                directory.parent/'scratch', 1, 60, snapshot['ldd_logs'][used], ldd=True)
            used += 1
            text = Path(process['stdout']).read_text()+Path(process['stderr']).read_text()
            static = 'not a dynamic executable' in text or 'statically linked' in text
            require(process['returncode'] == 0 or static, 'ldd failed')
            require('not found' not in text, 'unresolved runtime dependency')
            for line in text.splitlines():
                match = re.match(r'^\s*(?:\S+\s+=>\s+)?(/\S+)\s+\(0x[0-9a-fA-F]+\)', line)
                if match:
                    queue.append(match.group(1))
                elif '=>' in line and not static:
                    raise ValueError('unparsed retained ldd dependency: '+line)
        require(found == snapshot['files'] and used == len(snapshot['ldd_logs']),
                'extra/missing runtime file or transcript')
        return found


def protocol(path, fullcall):
    # Same protocol as the reviewed producer, with no new stderr/score threshold.
    lines = Path(path).read_text().splitlines()
    require(len(lines) == 5 and lines[0].startswith('VERSION ') and
            len(lines[0].split()) == 2, 'wrong VERSION/protocol length')
    expected_scope = 'SCOPE resident-rgb-full-frame'+('-model-lifetime' if fullcall else '')
    require(lines[1] == expected_scope, 'wrong SCOPE')
    observations = []
    for line, label in zip(lines[2:], ['COLD', 'SAMPLE', 'SAMPLE']):
        fields = line.split()
        require(len(fields) == 3 and fields[0] == label, 'wrong observation order')
        _, elapsed, scalar = fields
        require(re.fullmatch(r'[0-9]+', elapsed) is not None and int(elapsed) > 0,
                'invalid elapsed duration')
        require(re.fullmatch(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?',
                             scalar) is not None, 'invalid decimal scalar')
        score = float(scalar)
        require(math.isfinite(score), 'nonfinite scalar')
        observations.append(dict(label=label, elapsed_ns=int(elapsed), score_text=scalar,
                                 score=score, f64_bits=struct.pack('>d', score).hex()))
    return dict(version=lines[0], observations=observations)


def assets(evidence, frozen):
    expected = {}
    for category in ('assets', 'tools'):
        for name, record in frozen[category].items():
            expected[category+':'+name] = evidence.identity(record)
    compiler = {k: v for k, v in frozen['compiler'].items() if k != 'flags'}
    expected['compiler'] = evidence.identity(compiler)
    return expected


def root_paths(frozen, tools, binary=None):
    result = [frozen['compiler']['path'], frozen['assets']['old_binary']['path'],
              frozen['tools']['python']['path'], frozen['tools']['shell']['path']]
    result += [record['path'] for record in tools.values()]
    if binary is not None:
        result.append(binary)
    return result


def verify_build(e, path, built, frozen, freeze_path):
    require(built['schema'] == 'vmaf-fullcall-build/1' and built['passed'] is True and
            built['error'] is None and built.get('after_error') is None and
            built['caller_admitted'] is True and
            built['numerical_or_timing_qualification'] is False and
            built['source_revision'] == frozen['source_revision'], 'invalid build result')
    require(built['new_source_sha256'] == frozen['files']['libvmaf_rgb_fullcall.cpp'] and
            built['old_source_sha256'] == frozen['files']['libvmaf_rgb_compare.cpp'] and
            built['library_sha256'] == frozen['assets']['library']['sha256'],
            'build source/archive mismatch')
    directory = path.parent
    require(built['binary']['path'] == str(directory/'libvmaf-rgb-fullcall') and
            built['copied_library']['path'] == str(directory/'inputs/libvmaf.a'),
            'build artifact path mismatch')
    e.identity(built['binary'])
    with Path(built['binary']['path']).open('rb') as stream:
        require(stream.read(4) == b'\x7fELF', 'build output is not ELF')
    e.identity(built['copied_library'])
    require(built['copied_library']['sha256'] == built['library_sha256'], 'copied archive drift')
    expected_assets = assets(e, frozen)
    for phase in ('before', 'after'):
        require(e.json(directory/f'assets-{phase}.json') == expected_assets, 'build asset drift')
        tools = e.json(directory/f'compiler-tools-{phase}.json')
        require(tools == built['compiler_tools'] and
                set(tools) == {'cc1plus', 'collect2', 'as', 'ld'}, 'compiler tools drift')
        for name, record in tools.items():
            e.identity(record)
            process = e.process(directory, 'compiler-tools-'+phase+'-'+name,
                [frozen['compiler']['path'], '-print-prog-name='+name], directory/'scratch', 1, 60)
            selected = Path(process['stdout']).read_text().strip()
            if not os.path.isabs(selected):
                candidates = [Path(p)/selected for p in ('/usr/bin', '/bin')]
                selected = next((str(p) for p in candidates if p.is_file()), '')
            require(selected and Path(selected).resolve(strict=True) == Path(record['resolved']),
                    'compiler query selection mismatch')
    e.process(directory, 'compiler-version', [frozen['compiler']['path'], '--version'],
              directory/'scratch', 1, 60, built['compiler_version'])
    argv = [frozen['compiler']['path'], *FLAGS, '-I', str(freeze_path.parent/'include-source'),
            '-I', str(freeze_path.parent/'include-build'),
            str(freeze_path.parent/'libvmaf_rgb_fullcall.cpp'),
            built['copied_library']['path'], '-pthread', '-lm', '-o', built['binary']['path']]
    e.process(directory, 'compile', argv, directory/'scratch', 1, 120, built['compile'])
    roots = root_paths(frozen, built['compiler_tools'])
    before = e.snapshot(directory/'runtime-before', roots, frozen)
    after = e.snapshot(directory/'runtime-after', roots+[built['binary']['path']], frozen)
    require(all(after.get(k) == v for k, v in before.items()) and
            after == built['runtime_files'], 'build runtime closure drift')


def verify_native(e, native, receipt_path, receipt):
    """Consume the normal verified-build receipt, not a scorer/probe result."""
    require(re.fullmatch(r'[0-9a-f]{40}', receipt['source_revision']) is not None,
            'invalid native revision')
    e.bind(native, receipt['binary_sha256'])
    base = receipt_path.parent
    build = e.json(base/'build-receipt.json', receipt['build_receipt_sha256'])
    require(build['source_revision'] == receipt['source_revision'] and
            build['binary_sha256'] == receipt['binary_sha256'] and
            build['adapters_applied'] is False and build['features'] ==
            ['compete', 'metrics-parallel', 'zenmetrics-cli/nr'], 'native build mismatch')
    e.bind(base.parent/(base.name+'.log'), receipt['build_log_sha256'])
    e.bind(base/'cargo.log')
    require(set(NATIVE_FILES).issubset(receipt['files']) and
            all(receipt['files'].get(k) == v for k, v in build['files'].items()),
            'incomplete/inconsistent native source map')
    for relative, expected in receipt['files'].items():
        path = (base/'work/zenmetrics'/relative).resolve(strict=True)
        require(path.is_relative_to(base/'work/zenmetrics'), 'native source path escape')
        e.bind(path, expected)


def verify_controls(e, path, controls, built, frozen, selected, old):
    require(controls['schema'] == 'vmaf-fullcall-controls/1' and controls['passed'] is True and
            controls['error'] is None and controls.get('after_error') is None and
            controls['source_revision'] == frozen['source_revision'] and
            controls['caller_admitted'] is True, 'invalid control result')
    for key, expected in [('expected_comparisons', 12), ('expected_children', 24),
                          ('expected_observations', 72), ('samples_per_child', 2),
                          ('completed_comparisons', 12), ('passed_comparisons', 12)]:
        require(type(controls[key]) is int and controls[key] == expected, 'wrong '+key)
    for key in ('numerical_bound_changed', 'timing_qualification',
                'native_parity_qualification', 'effective_library_utilization_verified'):
        require(controls[key] is False, 'unexpected qualification: '+key)
    require(len(controls['cases']) == len(selected) == 12, 'wrong case count')
    wanted = {c['id']: c for c in selected}
    require(len(wanted) == 12 and {(c['metric'], c['threads']) for c in selected} ==
            {(m, t) for m in ROUTES for t in (1, 8)}, 'wrong frozen grid')
    result, versions = [], set()
    for summary in controls['cases']:
        case = wanted.pop(summary['id'])
        metric, threads = case['metric'], case['threads']
        require(type(threads) is int and case['width'] == case['height'] == 512 and
                case['id'] == metric+'-pair-cid22-t'+str(threads), 'wrong frozen case identity')
        originals = [c for c in old['cases'] if c['id'] == case['id']]
        require(len(originals) == 1 and originals[0] == case['original_plan_case'],
                'original plan case mismatch')
        original = originals[0]
        require((original['metric'], original['threads'], original['width'], original['height']) ==
                (metric, threads, 512, 512) and
                all(type(original[k]) is int for k in ('threads', 'width', 'height')),
                'source-plan case mismatch')
        require(set(case['inputs']) == {'reference', 'distorted'}, 'wrong input roles')
        for role, record in case['inputs'].items():
            e.identity(record)
            require(record['bytes'] == 512*512*3 and record['path'] == original[role] and
                    old['bindings'][record['path']] == record['sha256'], 'input contract mismatch')
        argv = [frozen['assets']['old_binary']['path'], metric, '512', '512',
                original['reference'], original['distorted'], str(threads), '2',
                '601' if 'studio601' in metric else '709']
        require(argv == case['argv'] == [s.format_map(dict(original, samples=2))
                for s in original['canonical']['command']], 'frozen argv mismatch')
        directory = path.parent/case['id']
        require(summary['case'] == str(directory/'case.json') and summary['passed'] is True and
                summary['errors'] == [], 'wrong control summary')
        actual = e.json(summary['case'], summary['case_sha256'])
        require((actual['id'], actual['metric'], actual['threads']) ==
                (case['id'], metric, threads) and actual['passed'] is True and
                type(actual['threads']) is int and actual['errors'] == [] and
                set(actual['arms']) == {'old', 'fullcall'},
                'wrong control case/arms')
        require(e.json(directory/'inputs-before.json') == case['inputs'], 'initial input drift')
        bits = []
        for arm, binary in [('old', frozen['assets']['old_binary']), ('fullcall', built['binary'])]:
            record = actual['arms'][arm]
            require(record['binary'] == binary, 'wrong arm binary')
            process = e.process(directory, arm, [binary['path'], *argv[1:]],
                path.parent/'scratch', threads, frozen['child_timeout_seconds'], record['process'])
            parsed = protocol(process['stdout'], arm == 'fullcall')
            require(parsed == record['protocol'], 'recorded protocol differs from raw output')
            # Check bits separately: JSON numeric equality alone loses signed zero.
            for observed in record['protocol']['observations']:
                require(type(observed['elapsed_ns']) is int and
                        type(observed['score']) in (int, float) and
                        math.isfinite(observed['score']) and
                        re.fullmatch(r'[0-9a-f]{16}', observed['f64_bits'] or '') is not None and
                        struct.pack('>d', observed['score']).hex() == observed['f64_bits'],
                        'nonfinite/malformed scalar bits')
                bits.append(observed['f64_bits'])
            versions.add(parsed['version'])
            require(e.json(directory/(arm+'.inputs-after.json')) == case['inputs'], 'input drift')
        require(len(bits) == 6 and len(set(bits)) == 1 and
                actual['exact_f64_bits'] == summary['exact_f64_bits'] == bits[0],
                'old/new exact scalar mismatch')
        result.append(copy.deepcopy(original))
    require(not wanted and len(versions) == 1, 'incomplete grid/library version mismatch')
    expected_assets = assets(e, frozen)
    roots = root_paths(frozen, built['compiler_tools'], built['binary']['path'])
    for phase in ('before', 'after'):
        require(e.json(path.parent/f'assets-{phase}.json') == expected_assets, 'control asset drift')
        require(e.snapshot(path.parent/f'runtime-{phase}', roots, frozen) == built['runtime_files'],
                'build/control runtime drift')
    return result, next(iter(versions)).split()[1]


def publish(output, payload):
    """Complete bytes first; hard-link publication is atomic and refuses replacement."""
    output = Path(output).absolute()
    data = (json.dumps(payload, indent=2, allow_nan=False)+'\n').encode()
    require(not os.path.lexists(output), 'output already exists')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.'+output.name+'.',
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)  # EEXIST also handles a concurrent publisher.
    finally:
        if temporary is not None:
            temporary.unlink()


def prepare(source, build_path, controls_path, native, native_receipt, output, *,
            freeze_sha, build_sha, controls_sha, native_receipt_sha):
    require(not os.path.lexists(output), 'output already exists')
    source, build_path, controls_path, native, native_receipt = [Path(p).resolve(strict=True)
        for p in (source, build_path, controls_path, native, native_receipt)]
    e = Evidence()
    built = e.json(build_path, build_sha)
    controls = e.json(controls_path, controls_sha)
    worker = e.json(native_receipt, native_receipt_sha)
    freeze_path = Path(built['freeze']).resolve(strict=True)
    frozen = e.json(freeze_path, freeze_sha)
    require(frozen['schema'] == 'vmaf-fullcall-freeze/1' and
            controls['freeze'] == built['freeze'] == str(freeze_path) and
            controls['freeze_sha256'] == built['freeze_sha256'] == freeze_sha and
            controls['build_receipt'] == str(build_path) and
            controls['build_receipt_sha256'] == build_sha, 'receipt link mismatch')
    require(frozen['compiler']['flags'] == FLAGS and frozen['threads'] == [1, 8] and
            all(type(t) is int for t in frozen['threads']) and type(frozen['samples']) is int and
            frozen['samples'] == 2 and type(frozen['child_timeout_seconds']) is int and
            1 <= frozen['child_timeout_seconds'] <= 1100, 'producer contract mismatch')
    require(all(frozen['files'].get(k) == v for k, v in SOURCE_HASHES.items()) and
            'cases.json' in frozen['files'], 'unreviewed producer/clock source')
    headers = [('include-source', 'libvmaf/'+name) for name in
               ('feature.h', 'model.h', 'picture.h', 'libvmaf.h', 'libvmaf_cuda.h')]
    headers += [('include-build', name) for name in ('vcs_version.h', 'libvmaf/version.h')]
    for folder, name in headers:
        require(frozen['files'].get(folder+'/'+name) ==
                frozen['assets'][folder+':'+name]['sha256'], 'header/source binding mismatch')
    for relative, expected in frozen['files'].items():
        path = (freeze_path.parent/relative).resolve(strict=True)
        require(path.is_relative_to(freeze_path.parent), 'bundle path escape')
        e.bind(path, expected)
    require(frozen['assets']['plan']['path'] == str(source), 'wrong source plan')
    old = e.json(source, frozen['assets']['plan']['sha256'])
    require(type(old['schema']) is int and old['schema'] == 1, 'wrong original plan schema')
    for path, expected in old['bindings'].items():
        e.bind(path, expected)
    verify_build(e, build_path, built, frozen, freeze_path)
    verify_native(e, native, native_receipt, worker)
    selected = e.json(freeze_path.parent/'cases.json')
    cases, version = verify_controls(e, controls_path, controls, built, frozen, selected, old)
    for case in cases:
        case['id'] = case['metric']+'-fullcall-pair-cid22-t'+str(case['threads'])
        case['canonical']['command'][0] = built['binary']['path']
        case['canonical']['label'] = 'libvmaf '+version+' / complete embedded-model lifetime'
        case['scope'] = ('Resident RGB8 conversion, embedded-model load/destroy, full single-frame '
                         'context/picture lifetime and scoring; input file reads and startup excluded')
        case['matched_scope'] = True
    e.bind(__file__)
    plan = dict(schema=1, binary=str(native), rust_source=worker['source_revision'],
        bindings=e.bindings, cases=cases, reference_control_gate=
        '12 old/new comparisons; all72 scalar observations exact f64 bits',
        scope_qualification='Complete per-call model lifetime aligned; different parsers and '
        'representations remain implementation details. No utilization or timing acceptance inferred.',
        ratio_definition='Rust/canonical; descriptive alternating-block medians',
        receipt_verifier='vmaf-fullcall-plan/2', numerical_bound_changed=False,
        native_parity_qualification=False, timing_qualification=False,
        effective_library_utilization_verified=False)
    # Recheck every historical/current byte binding before publication; later
    # mutations are detected by the downstream plan's ordinary binding checks.
    for path, expected in list(e.bindings.items()):
        e.bind(path, expected)
    publish(output, plan)
    return dict(cases=len(cases), plan=str(Path(output).absolute()), plan_sha256=sha(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'build', 'controls', 'native', 'native-receipt', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    for name in ('freeze-sha', 'build-sha', 'controls-sha', 'native-receipt-sha'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.build, args.controls, args.native,
        args.native_receipt, args.output, freeze_sha=args.freeze_sha, build_sha=args.build_sha,
        controls_sha=args.controls_sha, native_receipt_sha=args.native_receipt_sha)))


if __name__ == '__main__':
    main()
