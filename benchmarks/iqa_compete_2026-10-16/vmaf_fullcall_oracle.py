#!/usr/bin/env python3
"""Source-bound C++ link-only full-call VMAF oracle and exact-bit controls.

No Cargo/install step. The caller must supply heavy admission; freeze preparation
never runs this module. Generic stdlib receipt/process/closure primitives derive
from build_dssim_fullcall_reference.py at zenmetrics 2fc4faf0.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
SOURCE = 'libvmaf_rgb_fullcall.cpp'
DRIVER = 'vmaf_fullcall_oracle.py'
OLD_SOURCE = 'libvmaf_rgb_compare.cpp'
SCHEMA = 'vmaf-fullcall-freeze/1'
FLAGS = ['-O3', '-Wall', '-Wextra', '-ffp-contract=off', '-std=c++17']
ROUTES = {'vmaf', 'vmaf-neg', 'vmaf-4k', 'vmaf-v1', 'vmaf-studio601', 'vmaf-neg-studio601'}
THREAD_VARIABLES = ('RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                    'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def read_json(path):
    def pairs(items):
        obj = {}
        for key, value in items:
            require(key not in obj, f'duplicate JSON key {key}')
            obj[key] = value
        return obj
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def positive_int(value, label):
    require(type(value) is int and value > 0, f'{label} must be a positive integer')
    return value


def stop_requested(signum, _frame):
    raise KeyboardInterrupt(f'stop signal {signum}')


def identity(path):
    path = Path(path).absolute()
    resolved = path.resolve(strict=True)
    require(resolved.is_file(), f'not a regular file: {path}')
    return dict(path=str(path), resolved=str(resolved),
                bytes=resolved.stat().st_size, sha256=digest(resolved))


def clean_environment(scratch, threads):
    positive_int(threads, 'threads')
    env = os.environ.copy()
    for key in ('CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS', 'GCC_EXEC_PREFIX',
                'COMPILER_PATH', 'LIBRARY_PATH', 'CPATH', 'C_INCLUDE_PATH',
                'CPLUS_INCLUDE_PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT',
                'PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    env.update(PATH='/usr/bin:/bin', LC_ALL='C', LANG='C',
               TMPDIR=str(scratch), PYTHONDONTWRITEBYTECODE='1',
               OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE')
    for key in THREAD_VARIABLES:
        env[key] = str(threads)
    return env


def run_logged(argv, directory, stem, env, timeout):
    """Retain argv/stdout/stderr even on nonzero exit or a bounded timeout."""
    directory = Path(directory)
    argv = list(map(str, argv))
    command = directory/(stem+'.command.json')
    stdout = directory/(stem+'.stdout')
    stderr = directory/(stem+'.stderr')
    write_json(command, dict(argv=argv, cwd=str(directory), timeout_seconds=timeout,
        environment={k:env[k] for k in (*THREAD_VARIABLES, 'OMP_DYNAMIC',
                                     'MKL_DYNAMIC', 'TMPDIR', 'LC_ALL', 'PATH')}))
    timed_out = False
    launch_error = None
    returncode = None
    interrupted = None
    with stdout.open('xb') as out, stderr.open('xb') as err:
        try:
            child = subprocess.Popen(argv, cwd=directory, env=env, stdout=out,
                                     stderr=err, start_new_session=True)
        except OSError as e:
            launch_error = str(e)
        else:
            try:
                returncode = child.wait(timeout=timeout)
            except BaseException as e:
                timed_out = isinstance(e, subprocess.TimeoutExpired)
                if not timed_out:
                    interrupted = e
                # Only this invocation's process group is owned by the driver.
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    returncode = child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    returncode = child.wait(timeout=5)
    result = dict(argv=argv, returncode=returncode, timed_out=timed_out,
                  launch_error=launch_error, command=str(command),
                  interrupted=(None if interrupted is None else str(interrupted)),
                  stdout=str(stdout), stderr=str(stderr),
                  command_sha256=digest(command), stdout_sha256=digest(stdout),
                  stderr_sha256=digest(stderr))
    write_json(directory/(stem+'.process.json'), result)
    if interrupted is not None:
        raise interrupted
    return result


def checked_process(result):
    require(result['returncode'] == 0 and not result['timed_out'] and
            result['launch_error'] is None, f'process failed: {result}')


def runtime_snapshot(roots, output, label, env, ldd):
    """Hash recursive ldd file closure; retain each unedited ldd transcript.

    Address-bearing transcripts are evidence, not part of equality comparisons.
    ldd is used only on locally retained/trusted ELF roots/shared dependencies.
    Scripts are hashed; their interpreters are explicit roots supplied by caller.
    """
    folder = Path(output)/label
    folder.mkdir()
    queue = [str(Path(x).absolute()) for x in roots]
    files = {}
    scanned = set()
    logs = []
    while queue:
        path = queue.pop(0)
        if path in files:
            continue
        record = identity(path)
        files[path] = record
        resolved = record['resolved']
        if resolved in scanned:
            continue
        scanned.add(resolved)
        with Path(resolved).open('rb') as f:
            elf = f.read(4) == b'\x7fELF'
        if not elf:
            continue
        process = run_logged([ldd, resolved], folder, f'ldd-{len(logs):03}', env, 60)
        logs.append(process)
        text = Path(process['stdout']).read_text()+Path(process['stderr']).read_text()
        require(not process['timed_out'] and process['launch_error'] is None,
                f'ldd failed: {resolved}')
        # Some ELF roots are genuinely static; unresolved dynamic dependencies fail.
        static = ('not a dynamic executable' in text or 'statically linked' in text)
        require(process['returncode'] == 0 or (process['returncode'] == 1 and static),
                f'ldd returned failure: {resolved}')
        require('not found' not in text, f'unresolved dependency: {resolved}')
        for line in text.splitlines():
            match = re.match(r'^\s*(?:\S+\s+=>\s+)?(/\S+)\s+\(0x[0-9a-fA-F]+\)', line)
            if match:
                queue.append(match.group(1))
            elif '=>' in line and not static:
                raise ValueError(f'unparsed ldd dependency: {line}')
    result = dict(roots=list(map(str, roots)), files=files, ldd_logs=logs)
    write_json(folder/'snapshot.json', result)
    return result


def copy_checked(source, target, expected):
    with Path(source).open('rb') as src, Path(target).open('xb') as dst:
        shutil.copyfileobj(src, dst, length=1 << 20)
    require(digest(target) == expected, f'copied asset mismatch: {source}')


def verify_freeze(path, expected):
    path = Path(path).resolve(strict=True)
    require(digest(path) == expected, 'freeze digest mismatch')
    frozen = read_json(path)
    require(frozen['schema'] == SCHEMA, 'wrong freeze schema')
    require({SOURCE, DRIVER, OLD_SOURCE, 'cases.json'}.issubset(frozen['files']),
            'missing frozen source/cases')
    for relative, sha256 in frozen['files'].items():
        target = (path.parent/relative).resolve(strict=True)
        require(target.is_relative_to(path.parent), 'bundle path escape')
        require(digest(target) == sha256, 'frozen file drift: '+relative)
    require(Path(__file__).resolve(strict=True) == path.parent/DRIVER,
            'execute the frozen driver copy')
    require(frozen['compiler']['flags'] == FLAGS, 'compiler flags changed')
    require(frozen['samples'] == 2 and type(frozen['samples']) is int,
            'preserve two samples after COLD')
    require(frozen['threads'] == [1, 8] and
            all(type(v) is int for v in frozen['threads']), 'preserve budgets1/8')
    require(type(frozen['child_timeout_seconds']) is int and
            1 <= frozen['child_timeout_seconds'] <= 1100, 'invalid child deadline')
    return path, frozen


def verify_assets(frozen):
    records = {}
    for category in ('assets', 'tools'):
        for name, wanted in frozen[category].items():
            actual = identity(wanted['path'])
            require(actual == wanted, 'asset/tool drift: '+name)
            records[category+':'+name] = actual
    actual = identity(frozen['compiler']['path'])
    require(actual == {k:v for k,v in frozen['compiler'].items() if k != 'flags'},
            'compiler drift')
    records['compiler'] = actual
    return records


def compiler_tools(frozen, output, stem, env):
    tools = {}
    for tool in ('cc1plus', 'collect2', 'as', 'ld'):
        result = run_logged([frozen['compiler']['path'], '-print-prog-name='+tool],
                            output, stem+'-'+tool, env, 60)
        checked_process(result)
        selected = Path(result['stdout']).read_text().strip()
        selected = selected if os.path.isabs(selected) else shutil.which(selected, path=env['PATH'])
        require(selected is not None, 'missing compiler backend: '+tool)
        tools[tool] = identity(selected)
    write_json(output/(stem+'.json'), tools)
    return tools


def roots(frozen, tools, binary=None):
    result = [frozen['compiler']['path'], frozen['assets']['old_binary']['path'],
              frozen['tools']['python']['path'], frozen['tools']['shell']['path']]
    result += [v['path'] for v in tools.values()]
    if binary is not None:
        result.append(str(binary))
    return result


def build(path, frozen, output):
    output = Path(output).absolute()
    output.mkdir()  # Never resume or overwrite failed/successful evidence.
    (output/'scratch').mkdir()
    env = clean_environment(output/'scratch', 1)
    receipt = dict(schema='vmaf-fullcall-build/1', passed=False, error=None,
        freeze=str(path), freeze_sha256=digest(path),
        source_revision=frozen['source_revision'], caller_admitted=True,
        numerical_or_timing_qualification=False)
    before = tools = runtime_before = None
    interrupted = None
    binary = output/'libvmaf-rgb-fullcall'
    library = output/'inputs/libvmaf.a'
    try:
        before = verify_assets(frozen)
        write_json(output/'assets-before.json', before)
        version = run_logged([frozen['compiler']['path'], '--version'], output,
                             'compiler-version', env, 60)
        checked_process(version)
        receipt['compiler_version'] = version
        tools = compiler_tools(frozen, output, 'compiler-tools-before', env)
        runtime_before = runtime_snapshot(roots(frozen, tools), output, 'runtime-before',
                                         env, frozen['tools']['ldd']['path'])
        (output/'inputs').mkdir()
        copy_checked(frozen['assets']['library']['path'], library,
                     frozen['assets']['library']['sha256'])
        argv = [frozen['compiler']['path'], *FLAGS,
                '-I', str(path.parent/'include-source'),
                '-I', str(path.parent/'include-build'),
                str(path.parent/SOURCE), str(library), '-pthread', '-lm', '-o', str(binary)]
        receipt['compile'] = run_logged(argv, output, 'compile', env, 120)
        checked_process(receipt['compile'])
        with binary.open('rb') as stream:
            require(stream.read(4) == b'\x7fELF', 'compiler output is not ELF')
        receipt['binary'] = identity(binary)
        receipt['new_source_sha256'] = frozen['files'][SOURCE]
        receipt['old_source_sha256'] = frozen['files'][OLD_SOURCE]
        receipt['library_sha256'] = frozen['assets']['library']['sha256']
        receipt['passed'] = True
    except BaseException as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
        if not isinstance(error, Exception):
            interrupted = error
    finally:
        if interrupted is not None:
            receipt['passed'] = False
            receipt['after_error'] = 'interrupted; no after qualification'
        else:
            try:
                after = verify_assets(frozen)
                write_json(output/'assets-after.json', after)
                require(before is not None and before == after, 'asset drift/missing before')
                require(digest(library) == frozen['assets']['library']['sha256'],
                        'copied link archive drift')
                receipt['copied_library'] = identity(library)
                verify_freeze(path, receipt['freeze_sha256'])
                after_tools = compiler_tools(frozen, output, 'compiler-tools-after', env)
                require(tools is not None and tools == after_tools, 'compiler backend drift')
                after_roots = roots(frozen, after_tools, binary if binary.is_file() else None)
                after_runtime = runtime_snapshot(after_roots, output, 'runtime-after', env,
                                                frozen['tools']['ldd']['path'])
                require(runtime_before is not None and all(
                    after_runtime['files'].get(k) == v
                    for k,v in runtime_before['files'].items()), 'compiler/runtime closure drift')
                receipt['compiler_tools'] = after_tools
                receipt['runtime_files'] = after_runtime['files']
            except Exception as error:
                receipt['passed'] = False
                receipt['after_error'] = f'{type(error).__name__}: {error}'
    write_json(output/'build-receipt.json', receipt)
    if interrupted is not None:
        raise interrupted
    return receipt


def protocol(path, fullcall):
    lines = Path(path).read_text().splitlines()
    require(len(lines) == 5, 'require VERSION/SCOPE, one COLD and two SAMPLE')
    require(lines[0].startswith('VERSION ') and len(lines[0].split()) == 2,
            'missing version header')
    scope = ('SCOPE resident-rgb-full-frame-model-lifetime' if fullcall
             else 'SCOPE resident-rgb-full-frame')
    require(lines[1] == scope, 'wrong scope header')
    observations = []
    for index, line in enumerate(lines[2:]):
        fields = line.split()
        require(len(fields) == 3, 'expected label ns score')
        label, elapsed, scalar = fields
        require(label == ('COLD' if index == 0 else 'SAMPLE'), 'wrong observation order')
        require(re.fullmatch(r'[0-9]+', elapsed) is not None and int(elapsed) > 0,
                'invalid elapsed duration')
        require(re.fullmatch(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?',
                             scalar) is not None, 'invalid decimal scalar')
        score = float(scalar)
        require(math.isfinite(score), 'nonfinite scalar')
        observations.append(dict(label=label, elapsed_ns=int(elapsed), score_text=scalar,
                                 score=score, f64_bits=struct.pack('>d', score).hex()))
    return dict(version=lines[0], observations=observations)


def inputs(case):
    result = {}
    for role in ('reference', 'distorted'):
        actual = identity(case['inputs'][role]['path'])
        require(actual == case['inputs'][role], role+' input identity drift')
        require(actual['bytes'] == 512*512*3, 'wrong raw RGB8 length')
        result[role] = actual
    return result


def cases(path, frozen):
    selected = read_json(path.parent/'cases.json')
    require(type(selected) is list and len(selected) == 12, 'require12 explicit cases')
    require({(c['metric'],c['threads']) for c in selected} ==
            {(m,t) for m in ROUTES for t in (1,8)}, 'wrong/duplicate metric-budget grid')
    plan = read_json(frozen['assets']['plan']['path'])
    for c in selected:
        require(c['width'] == 512 and c['height'] == 512 and
                type(c['threads']) is int, 'wrong geometry/budget')
        require(c['id'] == c['metric']+'-pair-cid22-t'+str(c['threads']),
                'unexpected case identity/output path')
        source = [row for row in plan['cases'] if row['id'] == c['id']]
        require(len(source) == 1 and source[0] == c['original_plan_case'],
                'case differs from unchanged original plan')
        row = source[0]
        require((row['metric'],row['threads'],row['width'],row['height']) ==
                (c['metric'],c['threads'],512,512), 'case selection mismatch')
        for role in ('reference','distorted'):
            require(row[role] == c['inputs'][role]['path'], 'plan input path mismatch')
            require(plan['bindings'].get(row[role]) == c['inputs'][role]['sha256'],
                    'input absent from original hash bindings')
        expected = [s.format_map(dict(row, samples=2)) for s in row['canonical']['command']]
        require(expected == c['argv'] and expected[0] == frozen['assets']['old_binary']['path'],
                'canonical command drift')
        require(expected[1:] == [c['metric'],'512','512',c['inputs']['reference']['path'],
                c['inputs']['distorted']['path'],str(c['threads']),'2',
                '601' if 'studio601' in c['metric'] else '709'], 'unexpected RGB convention')
    return selected


def controls(path, frozen, build_receipt, output):
    build_path = Path(build_receipt).resolve(strict=True)
    build_hash = digest(build_path)
    built = read_json(build_path)
    require(built['schema'] == 'vmaf-fullcall-build/1' and built['passed'] is True,
            'build is not passed')
    require(built['freeze'] == str(path) and built['freeze_sha256'] == digest(path) and
            built['source_revision'] == frozen['source_revision'], 'wrong build source/freeze')
    require(built['new_source_sha256'] == frozen['files'][SOURCE] and
            built['library_sha256'] == frozen['assets']['library']['sha256'], 'build source/archive drift')
    binary = build_path.parent/'libvmaf-rgb-fullcall'
    require(identity(binary) == built['binary'], 'built ELF drift')
    require(identity(built['copied_library']['path']) == built['copied_library'] and
            built['copied_library']['sha256'] == frozen['assets']['library']['sha256'],
            'copied build archive drift')
    output = Path(output).absolute()
    output.mkdir()
    (output/'scratch').mkdir()
    env = clean_environment(output/'scratch', 1)
    result = dict(schema='vmaf-fullcall-controls/1', passed=False, error=None,
        freeze=str(path), freeze_sha256=digest(path), source_revision=frozen['source_revision'],
        build_receipt=str(build_path), build_receipt_sha256=build_hash,
        expected_comparisons=12, expected_children=24, expected_observations=72,
        samples_per_child=2, cases=[], caller_admitted=True,
        numerical_bound_changed=False, timing_qualification=False,
        native_parity_qualification=False, effective_library_utilization_verified=False)
    before = runtime_before = None
    interrupted = None
    try:
        before = verify_assets(frozen)
        write_json(output/'assets-before.json', before)
        runtime_before = runtime_snapshot(roots(frozen, built['compiler_tools'], binary), output,
                                         'runtime-before', env, frozen['tools']['ldd']['path'])
        require(runtime_before['files'] == built['runtime_files'], 'build/control runtime closure drift')
        for c in cases(path, frozen):
            directory = output/c['id']
            directory.mkdir()
            row = dict(id=c['id'], metric=c['metric'], threads=c['threads'],
                       passed=False, errors=[], arms={})
            child_env = clean_environment(output/'scratch', c['threads'])
            try:
                baseline = inputs(c)
                write_json(directory/'inputs-before.json', baseline)
                for arm, executable, record in [('old',c['argv'][0],frozen['assets']['old_binary']),
                                               ('fullcall',str(binary),built['binary'])]:
                    try:
                        require(identity(executable) == record, arm+' binary drift')
                        require(inputs(c) == baseline, 'input drift before child')
                        process = run_logged([executable,*c['argv'][1:]], directory, arm,
                                             child_env, frozen['child_timeout_seconds'])
                        row['arms'][arm] = dict(process=process, binary=record)
                        checked_process(process)
                        row['arms'][arm]['protocol'] = protocol(process['stdout'], arm == 'fullcall')
                    except Exception as error:
                        row['errors'].append(arm+': '+f'{type(error).__name__}: {error}')
                    finally:
                        require(identity(executable) == record, arm+' binary drift after child')
                        after_inputs = inputs(c)
                        write_json(directory/(arm+'.inputs-after.json'), after_inputs)
                        require(after_inputs == baseline, 'input drift after child')
                if not row['errors']:
                    old = row['arms']['old']['protocol']
                    new = row['arms']['fullcall']['protocol']
                    require(old['version'] == new['version'], 'library version changed')
                    bits = [v['f64_bits'] for p in (old,new) for v in p['observations']]
                    require(len(bits) == 6 and all(b == bits[0] for b in bits),
                            'exact old/new COLD/SAMPLE f64 mismatch')
                    row.update(passed=True, exact_f64_bits=bits[0])
            except Exception as error:
                row['errors'].append(f'{type(error).__name__}: {error}')
            write_json(directory/'case.json', row)
            result['cases'].append(dict(id=c['id'], passed=row['passed'], errors=row['errors'],
                exact_f64_bits=row.get('exact_f64_bits'), case=str(directory/'case.json'),
                case_sha256=digest(directory/'case.json')))
        result['passed'] = len(result['cases']) == 12 and all(c['passed'] for c in result['cases'])
    except BaseException as error:
        result['error'] = f'{type(error).__name__}: {error}'
        if not isinstance(error, Exception):
            interrupted = error
    finally:
        if interrupted is not None:
            result['passed'] = False
            result['after_error'] = 'interrupted; no after qualification'
        else:
            try:
                after = verify_assets(frozen)
                write_json(output/'assets-after.json', after)
                require(before is not None and before == after, 'asset drift/missing before')
                require(digest(build_path) == build_hash and identity(binary) == built['binary'],
                        'build receipt/ELF drift')
                require(identity(built['copied_library']['path']) == built['copied_library'],
                        'copied build archive drift after controls')
                for tool in built['compiler_tools'].values():
                    require(identity(tool['path']) == tool, 'compiler tool drift')
                verify_freeze(path, result['freeze_sha256'])
                runtime_after = runtime_snapshot(roots(frozen, built['compiler_tools'], binary),
                    output, 'runtime-after', env, frozen['tools']['ldd']['path'])
                require(runtime_before is not None and
                        runtime_before['files'] == runtime_after['files'], 'runtime closure drift')
            except Exception as error:
                result['passed'] = False
                result['after_error'] = f'{type(error).__name__}: {error}'
    result['completed_comparisons'] = len(result['cases'])
    result['passed_comparisons'] = sum(c['passed'] for c in result['cases'])
    write_json(output/'result.json', result)
    if interrupted is not None:
        raise interrupted
    return result


def main():
    signal.signal(signal.SIGTERM, stop_requested)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', required=True, type=Path)
    parser.add_argument('--freeze-sha', required=True)
    parser.add_argument('--build-output', required=True, type=Path)
    parser.add_argument('--controls-output', required=True, type=Path)
    parser.add_argument('--admitted', action='store_true',
                        help='Caller attests both shared locks and run-heavy12G admission')
    args = parser.parse_args()
    require(args.admitted, 'root heavy admission is required before compiler/controls')
    require(not args.build_output.exists() and not args.controls_output.exists(),
            'build/control outputs must both be new paths')
    path, frozen = verify_freeze(args.freeze, args.freeze_sha)
    built = build(path, frozen, args.build_output)
    if not built['passed']:
        print('BUILD_FAILED '+str(args.build_output/'build-receipt.json'), flush=True)
        return 1
    result = controls(path, frozen, args.build_output/'build-receipt.json', args.controls_output)
    print(f"passed={result['passed']} comparisons={result['passed_comparisons']}/12; "
          f"result={args.controls_output/'result.json'}", flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
