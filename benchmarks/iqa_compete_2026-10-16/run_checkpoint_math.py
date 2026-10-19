#!/usr/bin/env python3
"""Observe unmodified checkpoint math on Linux and macOS in isolated scratch.

The staging directory must contain the hash-bound original modules, probe and
main.rs on both hosts. Caller supplies the two Linux host locks and run-heavy.
Remote commands also acquire the macOS host lock and its resource wrapper.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', type=Path)
    parser.add_argument('--remote-only', action='store_true',
                        help='Continue the remote observation after the retained local probe')
    args = parser.parse_args()
    root = args.stage.resolve(strict=True)
    identity = json.loads((root / 'identity.json').read_text())
    for name, digest in identity['files'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    env = dict(os.environ, TMPDIR=str(root))
    rows = None if args.remote_only else []
    commands = [('compile', ['rustc', '--edition=2024', '-O', 'main.rs', '-o', 'probe']),
                ('probe', [str(root / 'probe')]),
                ('test-compile', ['rustc', '--edition=2024', '-O', '--test', 'main.rs', '-o', 'tests']),
                ('tests', [str(root / 'tests'), '--nocapture'])]
    for name, command in ([] if args.remote_only else commands):
        with (root / ('linux-' + name + '.log')).open('x') as log:
            result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        rows.append(dict(name=name, command=command, returncode=result.returncode))
        if name != 'tests':
            assert result.returncode == 0, name
    remote = '''import hashlib,json,os,pathlib,subprocess
root=pathlib.Path.home()/'tmp'/STAGE_NAME
identity=json.loads((root/'identity.json').read_text())
for name,digest in identity['files'].items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
env=dict(os.environ,TMPDIR=str(root))
commands=[('compile',['rustc','--edition=2024','-O','main.rs','-o','probe']),
 ('probe',[str(root/'probe')]),
 ('test-compile',['rustc','--edition=2024','-O','--test','main.rs','-o','tests']),
 ('tests',[str(root/'tests'),'--nocapture'])]
rows=[]
for name,command in commands:
 with (root/('mac-'+name+'.log')).open('x') as log:
  result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 rows.append(dict(name=name,command=command,returncode=result.returncode))
 if name!='tests':assert result.returncode==0,name
(root/'mac-results.json').write_text(json.dumps(rows,indent=2)+'\\n')
print(json.dumps(rows,indent=2))
'''.replace('STAGE_NAME', repr(root.name))
    # Python code is passed on stdin; remote shell arguments contain only fixed
    # executables and this task's quoted scratch paths.
    remote_command = ' '.join(['flock', '"$HOME/tmp/mac-heavy.lock"',
        '"$HOME/work/claudehints/scripts/run-heavy"', '--mem', '12G', '--jobs', '1', '--',
        'env', 'TMPDIR="$HOME/tmp/' + root.name + '"', 'python3', '-'])
    wrapper_log = 'mac-wrapper-remote-only.log' if args.remote_only else 'mac-wrapper.log'
    with (root / wrapper_log).open('x') as log:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'mac', remote_command], input=remote,
                                text=True, stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == 0
    for name in ['mac-results.json', 'mac-compile.log', 'mac-probe.log', 'mac-test-compile.log', 'mac-tests.log']:
        remote_path = '~/tmp/' + root.name + '/' + name
        data = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', 'mac', 'cat ' + remote_path])
        with (root / name).open('xb') as stream:
            stream.write(data)
    record = dict(source_identity=identity, linux=rows, mac=json.loads((root / 'mac-results.json').read_text()),
                  continued_remote_only=args.remote_only,
                  worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()
                         if p.is_file() and p.suffix in ['.log', '.rs', '.json']}, qualified_timing_cells=0)
    (root / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(linux=rows, mac=record['mac']), indent=2))


if __name__ == '__main__':
    main()
