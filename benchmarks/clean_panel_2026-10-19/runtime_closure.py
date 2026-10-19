"""Resolve ELF dependency closure with the same explicit loader environment.

This inspects installed trusted ELF files; it never imports torch or runs a
scorer. Every environment extension is a root, including libraries dlopened by
Python. Runtime dlopen names not represented by those roots remain a limitation.
"""
import os
from pathlib import Path
import re
import subprocess


def loader_environment():
    # Refuse injection rather than silently describing a different execution.
    unsupported = {k:v for k,v in os.environ.items() if k.startswith('LD_') and k != 'LD_LIBRARY_PATH' and v}
    if unsupported:
        raise ValueError('unsupported loader environment: ' + ','.join(sorted(unsupported)))
    value = os.environ.get('LD_LIBRARY_PATH')
    if value is not None and any(not p or not Path(p).is_absolute() for p in value.split(':')):
        raise ValueError('loader search paths must be absolute and nonempty')
    return {'LD_LIBRARY_PATH': value}


def environment(binding):
    env = {k:v for k,v in os.environ.items() if not k.startswith('LD_')}
    env.update({k:v for k,v in binding.items() if v is not None})
    env['LC_ALL'] = 'C'
    return env


def elf(path):
    with Path(path).open('rb') as f:
        return f.read(4) == b'\x7fELF'


def inspect(path, loader):
    dynamic = subprocess.run(['/usr/bin/readelf','-d',str(path)],env=environment(loader),capture_output=True,text=True,check=True).stdout
    result = subprocess.run(['/usr/bin/ldd',str(path)],env=environment(loader),capture_output=True,text=True)
    if 'not found' in result.stdout:
        raise ValueError('unresolved native dependency: ' + str(path))
    if result.returncode and 'statically linked' not in result.stdout+result.stderr and 'not a dynamic executable' not in result.stdout+result.stderr:
        raise ValueError('ldd failed: ' + str(path) + result.stderr)
    paths=[]
    for line in result.stdout.splitlines():
        match=re.search(r'(?:=>\s+|^\s*)(/[^\s]+)',line)
        if match: paths.append(match[1])
    search=[line.strip() for line in dynamic.splitlines() if any(k in line for k in ('(RPATH)','(RUNPATH)','(NEEDED)'))]
    return dict(dependencies=sorted(set(paths)),dynamic_search=search)


def closure(roots, loader, inspector=inspect):
    from panel import digest
    pending=list(sorted(set(str(Path(p).absolute()) for p in roots)))
    files={}; graph={}
    while pending:
        alias=pending.pop()
        resolved=str(Path(alias).resolve(strict=True))
        if alias in files: continue
        files[alias]=dict(path=alias,resolved=resolved,sha256=digest(alias))
        if resolved in graph: continue
        detail=inspector(resolved,loader)
        graph[resolved]=detail
        pending.extend(detail['dependencies'])
    return dict(roots=sorted(set(str(Path(p).absolute()) for p in roots)),loader=loader,
                files=sorted(files.values(),key=lambda x:x['path']),graph=graph)


def verify(binding, inspector=inspect):
    if loader_environment() != binding['loader']:
        raise ValueError('loader environment changed')
    if closure(binding['roots'],binding['loader'],inspector) != binding:
        raise ValueError('native dependency closure changed')
