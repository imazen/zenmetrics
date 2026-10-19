"""Bind import search roots and executable caches without modifying environments."""
import os
from pathlib import Path
import subprocess
import json


def environment(base=None):
    env = dict(os.environ if base is None else base)
    for key in list(env):
        if key.startswith('PYTHON'): del env[key]
    env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    return env


def search_paths(executable, script_directory):
    code = 'import sys,sysconfig,json;print(json.dumps(dict(stdlib=sysconfig.get_path("stdlib"),paths=sys.path,cache_prefix=sys.pycache_prefix)))'
    result = json.loads(subprocess.check_output([executable,'-B','-c',code],env=environment(),text=True))
    if result['cache_prefix'] is not None:
        raise ValueError('unexpected Python cache prefix')
    paths = [script_directory if i==0 and p=='' else p for i,p in enumerate(result['paths'])]
    if any(not Path(p).is_absolute() for p in paths):
        raise ValueError('relative Python import path')
    return sorted(set(paths+[result['stdlib'],script_directory]))


def inventory(roots):
    from panel import digest
    root_states=[]; files={}; links={}
    for raw in sorted(set(roots)):
        root=Path(raw)
        state=dict(path=str(root),resolved=str(root.resolve()),exists=root.exists())
        root_states.append(state)
        if not root.exists(): continue  # Absent zip/import root is still bound above.
        if root.is_file():
            state.update(kind='file',sha256=digest(root))  # zip/egg roots include bytecode too.
            continue
        if not root.is_dir(): raise ValueError('unsupported import root: '+str(root))
        state['kind']='directory'
        visited=set()
        def fail(error): raise error
        for directory,dirs,names in os.walk(root,followlinks=True,onerror=fail):
            path=Path(directory);target=str(path.resolve(strict=True))
            if path.is_symlink():links[str(path)]=target
            if target in visited:
                dirs[:]=[];continue
            visited.add(target);dirs.sort()
            for name in sorted(names):
                if Path(name).suffix not in ('.pyc','.pyo'):continue
                entry=path/name
                files[str(entry)]=dict(path=str(entry),resolved=str(entry.resolve(strict=True)),sha256=digest(entry))
    return dict(roots=root_states,files=sorted(files.values(),key=lambda f:f['path']),directory_links=links)


def capture(probes, extra_roots=()):
    descriptions=[];roots=list(extra_roots)
    for spec in probes:
        paths=search_paths(spec['executable'],spec['script_directory'])
        descriptions.append(dict(spec,paths=paths));roots.extend(paths)
    return dict(probes=probes,extra_roots=list(extra_roots),search_paths=descriptions,inventory=inventory(roots))


def verify(binding):
    if capture(binding['probes'],binding['extra_roots']) != binding:
        raise ValueError('Python import/bytecode inventory changed')
