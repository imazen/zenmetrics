#!/usr/bin/env python3
"""Freeze/run original-byte and all24 HDR controls; execution needs root admission.

The prepare subcommand emits files only. The run subcommand captures runtime
closure, checks original-generator bytes, then runs BOTH metric variants even
when the system variant fails its unchanged numerical gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

GENERATOR_SHA = '907fab1cd50619ce930df4d63c9fa5fcb765f7b36899da78c4476de6dacdb071'
ROOTS = ('/usr/share/octave/6.4.0', '/usr/share/octave/packages/image-2.12.0',
         '/usr/lib/x86_64-linux-gnu/octave/6.4.0',
         '/usr/lib/x86_64-linux-gnu/octave/packages/image-2.12.0')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def verify(bindings):
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise ValueError('bound artifact changed: ' + path)


def original_emitter(source):
    if hashlib.sha256(source).hexdigest() != GENERATOR_SHA:
        raise ValueError('original integer generator changed')
    start = source.index(b'function img = reference_codes(H, W)\n')
    end = source.index(b"\nfid = fopen(fullfile(outdir, 'goldens_srgb.tsv'), 'w');", start)
    functions = source[start:end]
    # Exact byte slice, including unchanged reference_codes/minstd/distort.
    prefix = b'1; % Script marker; original function bodies below are unchanged.\n'
    suffix = r'''
function emit_raw(path, img)
  assert(all(isfinite(img(:))) && all(img(:)==fix(img(:))) && all(img(:)>=0) && all(img(:)<=255));
  raw=permute(img,[3,2,1]);
  f=fopen(path,'wb');assert(f>=0);
  n=fwrite(f,raw(:),'uint8');assert(n==numel(img));assert(fclose(f)==0);
end
args=argv();assert(numel(args)==1);outdir=args{1};assert(exist(outdir,'dir')==7);
sizes={[128,128],[96,160]};
names={'identical','noise','blur','dark','bright','chroma'};
for si=1:numel(sizes)
  H=sizes{si}(1);W=sizes{si}(2);ref=reference_codes(H,W);
  emit_raw(fullfile(outdir,sprintf('ref-%dx%d.rgb',W,H)),ref);
  for ni=1:numel(names)
    tst=distort(names{ni},ref);
    emit_raw(fullfile(outdir,sprintf('%s-%dx%d.rgb',names{ni},W,H)),tst);
  end
end
fprintf('ORIGINAL_RAW_COMPLETE files=14\n');fflush(stdout);
'''.encode()
    return prefix + functions + suffix, hashlib.sha256(functions).hexdigest()


def prepare(parent_path, output):
    parent_path = parent_path.resolve(strict=True)
    parent_root = parent_path.parent
    parent = json.loads(parent_path.read_text())
    verify(parent['bindings'])
    output.mkdir()
    root = output.resolve()
    bindings = {}
    for name, expected in parent['bindings'].items():
        path = Path(name)
        if path.is_relative_to(parent_root):
            dest = root / path.relative_to(parent_root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            bindings[str(dest)] = expected
        else:
            bindings[name] = expected
    controls = []
    for original in parent['controls']:
        role = original['variant']
        path = root / (role + '.m')
        text = path.read_text().replace(str(parent_root), str(root))
        # Only the NEW copy is changed to point at its own frozen inputs.
        path.write_text(text)
        bindings[str(path)] = sha(path)
        command = [str(root / Path(v).relative_to(parent_root))
                   if Path(v).is_relative_to(parent_root) else v for v in original['command']]
        controls.append(dict(original, command=command, script_sha256=sha(path)))
    emitter, function_hash = original_emitter((root/'original-gen_srgb_goldens.m').read_bytes())
    (root/'original-raw-emitter.m').write_bytes(emitter)
    shutil.copy2(Path(__file__), root/Path(__file__).name)
    for path in (root/'original-raw-emitter.m', root/Path(__file__).name):
        bindings[str(path)] = sha(path)
    bindings[str(parent_path)] = sha(parent_path)
    expected_raw = {p.name: sha(p) for p in (root/'inputs').iterdir() if p.is_file()}
    if len(expected_raw) != 14 or any(not name.endswith('.rgb') for name in expected_raw):
        raise ValueError('exact fourteen generated raw files required')
    paths = [Path('/usr/bin/ldd'), Path('/usr/bin/time'), Path('/usr/bin/timeout')]
    for path in paths:
        bindings[str(path.resolve(strict=True))] = sha(path)
    plan = dict(schema=1, parent_plan_sha256=sha(parent_path), controls=controls,
        bindings=bindings, expected_raw=expected_raw, original_generator_sha256=GENERATOR_SHA,
        unchanged_function_body_sha256=function_hash, emitter_sha256=sha(root/'original-raw-emitter.m'),
        octave=controls[0]['command'][0], runtime_roots=list(ROOTS), seconds_per_stage=1100,
        existing_absolute_bound=5e-3, expected_cases_per_variant=24, executed=False,
        qualification='Preparation only; byte control and runtime closure must pass before either metric process',
        failure_policy='Byte mismatch refuses both metric arms. Numerical failure in system212 never suppresses resize2201. Closure or source drift refuses further execution.',
        environment={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS')})
    write(root/'driver-plan.json', plan)
    return plan


def file_record(path):
    path = Path(path)
    resolved = path.resolve(strict=True)
    return dict(resolved=str(resolved), symlink=os.readlink(path) if path.is_symlink() else None,
                bytes=resolved.stat().st_size, sha256=sha(resolved))


def elf(path):
    with Path(path).open('rb') as stream:
        return stream.read(4) == b'\x7fELF'


def capture(plan, bundle, directory, environment):
    directory.mkdir()
    files = {}
    roots = [Path(p) for p in plan['runtime_roots']] + [bundle/'official-v2', bundle/'resize2201']
    # Inventory installed core and image M/OCT files, loaders and package hooks.
    for root in roots:
        if not root.is_dir():
            raise ValueError('required runtime root missing: ' + str(root))
        for path in sorted(root.rglob('*')):
            if path.is_file() and (path.suffix in ('.m','.oct','.mex','.so')
                                   or path.name in ('PKG_ADD','PKG_DEL')):
                files[str(path)] = file_record(path)
    for name in (plan['octave'], '/usr/bin/ldd', '/usr/bin/time', '/usr/bin/timeout',
                 '/usr/bin/python3', '/etc/ld.so.cache', '/etc/octaverc', '/etc/ld.so.conf'):
        files[name] = file_record(name)
    for path in sorted(Path('/etc/ld.so.conf.d').rglob('*')):
        if path.is_file():
            files[str(path)] = file_record(path)
    pending = sorted({row['resolved'] for row in files.values() if elf(row['resolved'])})
    seen = set()
    dependencies = {}
    counter = 0
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        name = f'ldd-{counter:04d}'
        counter += 1
        with (directory/(name+'-stdout.log')).open('x') as out, (directory/(name+'-stderr.log')).open('x') as err:
            completed = subprocess.run(['/usr/bin/ldd', path], env=environment,
                stdout=out, stderr=err, timeout=30)
        text = (directory/(name+'-stdout.log')).read_text() + (directory/(name+'-stderr.log')).read_text()
        if 'not found' in text:
            raise ValueError('unresolved shared library: ' + path)
        static = 'statically linked' in text or 'not a dynamic executable' in text
        if completed.returncode and not static:
            raise ValueError('ldd failed: ' + path + '; see ' + str(directory))
        resolved = []
        for line in text.splitlines():
            match = re.match(r'^\s*(?:\S+\s+=>\s+)?(/[^()]+?)(?:\s+\(|\s*$)', line)
            if match:
                library = str(Path(match.group(1).strip()).resolve(strict=True))
                files[library] = file_record(library)
                resolved.append(library)
                if library not in seen:
                    pending.append(library)
            elif line.strip() and not (static or 'linux-vdso' in line):
                raise ValueError('unparsed ldd resolution: ' + line)
        dependencies[path] = sorted(set(resolved))
        write(directory/(name+'-identity.json'),dict(path=path,returncode=completed.returncode,
            stdout_sha256=sha(directory/(name+'-stdout.log')),stderr_sha256=sha(directory/(name+'-stderr.log'))))
    # ASLR addresses in raw ldd logs may vary. Compare content hashes and
    # resolved paths/edges; retain complete raw resolution logs separately.
    result = dict(files=files, dependencies=dependencies,
        environment={k:environment.get(k) for k in ('PATH','LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT',
            'OCTAVE_PATH','OCTAVE_EXEC_PATH','OCTAVE_HOME','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    write(directory/'snapshot.json',result)
    return result


def execute(command, env, root, name, seconds):
    wrapped = ['/usr/bin/time','-v','-o',str(root/(name+'-memory.txt')),
        '/usr/bin/timeout','--kill-after=20s',str(seconds)+'s',*command]
    write(root/(name+'-command.json'),wrapped)
    with (root/(name+'-stdout.log')).open('x') as out, (root/(name+'-stderr.log')).open('x') as err:
        completed = subprocess.run(wrapped,env=env,cwd=root,stdout=out,stderr=err)
    return dict(returncode=completed.returncode,stdout_sha256=sha(root/(name+'-stdout.log')),
                stderr_sha256=sha(root/(name+'-stderr.log')))


def run(bundle, output):
    bundle = bundle.resolve(strict=True)
    plan_path = bundle/'driver-plan.json'
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    verify(plan['bindings'])
    output.mkdir()
    root = output.resolve()
    scratch = root/'scratch';scratch.mkdir()
    generated = root/'original-raw';generated.mkdir()
    env = dict(os.environ,**plan['environment'],TMPDIR=str(scratch),LC_ALL='C')
    write(root/'identity.json',dict(driver_plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        controller_sha256=sha(__file__),status='started',existing_absolute_bound=5e-3))
    result = dict(stages={},passed=False,qualified_timing_cells=0)
    try:
        overrides = [key for key in ('OCTAVE_PATH','OCTAVE_EXEC_PATH','OCTAVE_HOME',
                                     'OCTAVE_SITE_INITFILE','OCTAVE_VERSION_INITFILE',
                                     'LD_PRELOAD','LD_AUDIT') if env.get(key)]
        if overrides:
            raise ValueError('unbound runtime overrides: ' + ', '.join(overrides))
        baseline = capture(plan,bundle,root/'runtime-before',env)
        byte_command = [plan['octave'],'--no-gui','--no-init-file','--no-history','--quiet',
                        str(bundle/'original-raw-emitter.m'),str(generated)]
        byte_result = execute(byte_command,env,root,'byte-control',plan['seconds_per_stage'])
        actual = {p.name:p for p in generated.iterdir() if p.is_file()}
        byte_result['files'] = {name:dict(expected_sha256=expected,
            observed_sha256=sha(actual[name]) if name in actual else None,
            exact_equal=name in actual and actual[name].read_bytes()==(bundle/'inputs'/name).read_bytes())
            for name,expected in plan['expected_raw'].items()}
        byte_result['passed'] = (byte_result['returncode']==0 and set(actual)==set(plan['expected_raw'])
                                and all(v['exact_equal'] for v in byte_result['files'].values()))
        result['stages']['byte-control'] = byte_result
        write(root/'byte-control.json',byte_result)
        verify(plan['bindings'])
        if capture(plan,bundle,root/'runtime-after-bytes',env)!=baseline:
            raise ValueError('runtime closure changed during original-byte control')
        if not byte_result['passed']:
            raise ValueError('original-generator raw-byte control failed; metric execution refused')
        for control in plan['controls']:
            verify(plan['bindings'])
            if plan_path.read_bytes()!=plan_bytes:
                raise ValueError('driver plan changed before metric control')
            name=control['variant']
            observation=execute(control['command'],env,root,name,plan['seconds_per_stage'])
            rows=[line.split() for line in (root/(name+'-stdout.log')).read_text().splitlines()
                  if line.startswith('SRGB_GOLDEN ')]
            observation.update(cases=len(rows),passed=observation['returncode']==0 and len(rows)==24
                and all(len(row)==10 and row[8]=='1' for row in rows))
            result['stages'][name]=observation
            write(root/(name+'-result.json'),observation)
            verify(plan['bindings'])
            if capture(plan,bundle,root/('runtime-after-'+name),env)!=baseline:
                raise ValueError('runtime closure changed during '+name)
            # Numerical/exit failure is retained; always continue to next arm.
        if plan_path.read_bytes()!=plan_bytes:
            raise ValueError('driver plan changed during controls')
        result['passed']=all(stage['passed'] for stage in result['stages'].values())
    except Exception as error:
        result['error']=str(error)
    write(root/'result.json',result)
    print(json.dumps(result),flush=True)
    return 0 if result['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('parent',type=Path);prep.add_argument('output',type=Path)
    active=sub.add_parser('run');active.add_argument('bundle',type=Path);active.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.mode=='prepare':
        plan=prepare(args.parent,args.output)
        print(json.dumps(dict(controls=len(plan['controls']),raw_files=len(plan['expected_raw']),executed=False)))
    else:
        raise SystemExit(run(args.bundle,args.output))


if __name__=='__main__':main()
