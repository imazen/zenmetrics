#!/usr/bin/env python3
"""Root-admitted link-only builder for the full-call author DSSIM oracle.

Never invokes Cargo. All compilation/runtime inspection belongs under caller
admission. The optional `all` mode links then launches the separately bound
exact-bit control driver. Importing this module launches no subprocess.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys

sys.dont_write_bytecode = True
SCHEMA = 'dssim-fullcall-freeze/1'
SOURCE = 'dssim_fullcall_reference.c'
BUILDER = 'build_dssim_fullcall_reference.py'
CHECKER = 'check_dssim_fullcall_reference.py'
FLAGS = ['-O3', '-std=c11', '-Wall', '-Wextra', '-Werror',
         '-fno-fast-math', '-ffp-contract=off']
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
    for key in ('CFLAGS', 'CPPFLAGS', 'LDFLAGS', 'GCC_EXEC_PREFIX',
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


def verify_freeze(path, expected):
    path = Path(path).resolve(strict=True)
    require(digest(path) == expected, 'freeze digest mismatch')
    freeze = read_json(path)
    require(freeze['schema'] == SCHEMA, 'wrong freeze schema')
    base = path.parent
    require({SOURCE, BUILDER, CHECKER, 'dssim.h'}.issubset(freeze['files']),
            'missing required frozen source/header')
    for relative, expected_hash in freeze['files'].items():
        target = (base/relative).resolve(strict=True)
        require(target.is_relative_to(base), 'bundle path escapes source directory')
        require(digest(target) == expected_hash, f'bundle drift: {relative}')
    own = Path(__file__).resolve(strict=True)
    require(own == base/BUILDER and digest(own) == freeze['files'][BUILDER],
            'builder must be the bound frozen copy')
    require(freeze['compiler']['flags'] == FLAGS, 'unexpected compiler flags')
    require(freeze['threads'] == [1, 8] and all(type(t) is int for t in freeze['threads']),
            'preserve exact integer one/eight budgets')
    positive_int(freeze['samples'], 'samples')
    require(1 <= freeze['samples'] <= 1000, 'samples outside controlled fixture range')
    positive_int(freeze['child_timeout_seconds'], 'child timeout')
    positive_int(freeze['suite_timeout_seconds'], 'suite timeout')
    return path, freeze


def verify_assets(freeze):
    records = {}
    for name, asset in freeze['assets'].items():
        record = identity(asset['path'])
        require(record['sha256'] == asset['sha256'], f'asset drift: {name}')
        records[name] = record
    for name, tool in freeze['tools'].items():
        record = identity(tool['path'])
        require(record['sha256'] == tool['sha256'], f'tool drift: {name}')
        records['tool:'+name] = record
    record = identity(freeze['compiler']['path'])
    require(record['sha256'] == freeze['compiler']['sha256'], 'compiler drift')
    records['compiler'] = record
    return records


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


def build(args):
    path, freeze = verify_freeze(args.freeze, args.freeze_sha)
    output = Path(args.output).absolute()
    output.mkdir()  # Existing outputs are immutable, including failed builds.
    (output/'scratch').mkdir()
    env = clean_environment(output/'scratch', 1)
    receipt = dict(schema='dssim-fullcall-build/1', passed=False,
                   freeze=str(path), freeze_sha256=args.freeze_sha,
                   source_revision=freeze['source_revision'], error=None)
    try:
        before = verify_assets(freeze)
        write_json(output/'assets-before.json', before)
        compiler = freeze['compiler']['path']
        roots = [compiler, freeze['assets']['old_binary']['path'],
                 freeze['tools']['python']['path'], freeze['tools']['shell']['path']]
        version = run_logged([compiler, '--version'], output, 'compiler-version', env, 60)
        checked_process(version)
        # Freeze the actual compiler backend/linker paths selected by this driver.
        toolchain = {}
        for tool in ('cc1', 'collect2', 'as', 'ld'):
            process = run_logged([compiler, '-print-prog-name='+tool], output,
                                 'compiler-tool-'+tool, env, 60)
            checked_process(process)
            selected = Path(process['stdout']).read_text().strip()
            selected = selected if os.path.isabs(selected) else shutil.which(selected, path=env['PATH'])
            require(selected is not None, f'compiler tool unavailable: {tool}')
            toolchain[tool] = identity(selected)
        write_json(output/'compiler-tools-before.json', toolchain)
        roots += [record['path'] for record in toolchain.values()]
        runtime_before = runtime_snapshot(roots, output, 'runtime-before', env,
                                          freeze['tools']['ldd']['path'])
        (output/'inputs').mkdir()
        library = output/'inputs/libdssim_core.a'
        copy_checked(freeze['assets']['library']['path'], library,
                     freeze['assets']['library']['sha256'])
        for name in (SOURCE, 'dssim.h'):
            copy_checked(path.parent/name, output/'inputs'/name, freeze['files'][name])
        binary = output/'dssim-fullcall-reference'
        argv = [compiler, *FLAGS, '-I', str(output/'inputs'),
                str(output/'inputs'/SOURCE), str(library), '-ldl', '-pthread', '-lm',
                '-o', str(binary)]
        compiled = run_logged(argv, output, 'compile', env, 300)
        checked_process(compiled)
        with binary.open('rb') as f:
            require(f.read(4) == b'\x7fELF', 'compiler did not produce ELF')
        runtime_after = runtime_snapshot(roots, output, 'runtime-after', env,
                                         freeze['tools']['ldd']['path'])
        require(runtime_before['files'] == runtime_after['files'], 'build runtime closure drift')
        toolchain_after = {key:identity(value['path']) for key,value in toolchain.items()}
        write_json(output/'compiler-tools-after.json', toolchain_after)
        require(toolchain_after == toolchain, 'compiler backend/linker drift')
        linked = runtime_snapshot([str(binary)], output, 'linked-runtime', env,
                                  freeze['tools']['ldd']['path'])
        after = verify_assets(freeze)
        write_json(output/'assets-after.json', after)
        require(before == after, 'source/library/tool assets changed during link')
        verify_freeze(path, args.freeze_sha)
        require(digest(library) == freeze['assets']['library']['sha256'], 'link archive copy drift')
        for name in (SOURCE, 'dssim.h'):
            require(digest(output/'inputs'/name) == freeze['files'][name], 'link source copy drift')
        receipt.update(passed=True, binary=identity(binary), compile=compiled,
            compiler_version=version, compiler_tools=toolchain,
            runtime_files=linked['files'], assets=after,
            new_source_sha256=freeze['files'][SOURCE],
            header_sha256=freeze['files']['dssim.h'],
            library_sha256=freeze['assets']['library']['sha256'],
            numerical_or_timing_qualification=False)
    except Exception as e:
        receipt['error'] = f'{type(e).__name__}: {e}'
    write_json(output/'build-receipt.json', receipt)
    return receipt


def main():
    signal.signal(signal.SIGTERM, stop_requested)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['build', 'all'])
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--freeze-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control-output', type=Path)
    args = parser.parse_args()
    require(args.mode != 'all' or args.control_output is not None, 'all requires control-output')
    require(not args.output.exists(), 'build output already exists')
    if args.mode == 'all':
        require(not args.control_output.exists(), 'control output already exists')
        require(args.output.absolute() != args.control_output.absolute(),
                'build and control outputs must differ')
    receipt = build(args)
    if not receipt['passed']:
        print(json.dumps(receipt))
        return 1
    if args.mode == 'all':
        path = args.freeze.resolve(strict=True)
        freeze = read_json(path)
        output = args.output.absolute()
        control = run_logged([freeze['tools']['python']['path'], str(path.parent/CHECKER),
            '--freeze', str(path), '--freeze-sha', args.freeze_sha,
            '--build-receipt', str(output/'build-receipt.json'),
            '--build-receipt-sha', digest(output/'build-receipt.json'),
            '--output', str(args.control_output.absolute())], output, 'control-driver',
            clean_environment(output/'scratch', 8), freeze['suite_timeout_seconds'])
        write_json(output/'suite.json', dict(schema='dssim-fullcall-suite/1',
            build_receipt_sha256=digest(output/'build-receipt.json'),
            control_process=control, control_output=str(args.control_output.absolute()),
            passed=control['returncode']==0 and not control['timed_out'] and
                   control['launch_error'] is None))
        return 0 if control['returncode']==0 and not control['timed_out'] else 1
    print(json.dumps(dict(passed=True, build_receipt=str(args.output/'build-receipt.json'),
                          build_receipt_sha256=digest(args.output/'build-receipt.json'))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
