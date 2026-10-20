#!/usr/bin/env python3
"""Bind installed reference code and explicit assets without importing models.

Run with the reference interpreter under shared resource admission. The output
is a flat source-binding inventory compatible with canonical_provenance.py.
Existing copy receipts can provide expected asset hashes through --artifacts.
"""
import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import sys

from canonical_input_hashes import stable_hash
from canonical_run import sha


def inventory(bindings):
    if not bindings:
        raise ValueError('reference inventory must contain files')
    observed = {}
    for index, (source, expected) in enumerate(bindings.items(), 1):
        path = Path(source)
        if not path.is_absolute():
            raise ValueError('reference source path must be absolute: '+source)
        digest = stable_hash(path)
        if expected is not None and expected != digest:
            raise ValueError('reference source differs from expected bytes: '+source)
        observed[source] = digest
        if index % 1000 == 0:
            print(f'bound {index}/{len(bindings)} reference files', flush=True)
    return observed


def package_bindings(names, module_runtime_only=False):
    bindings, versions = {}, {}
    for name in names:
        dist = metadata.distribution(name)
        files = dist.files
        if not files:
            raise ValueError('package has no installed file inventory: '+name)
        # Tokenizer vocabularies, priors and versioned shared libraries matter
        # alongside Python sources. Exclude only mutable bytecode caches.
        selected = [entry for entry in files
                    if '__pycache__' not in entry.parts and entry.suffix not in ('.pyc', '.pyo')]
        root = Path(os.path.abspath(dist.locate_file('')))
        excluded = []
        if not selected:
            raise ValueError('package has no selected reference files: '+name)
        for entry in selected:
            path = Path(os.path.abspath(dist.locate_file(entry)))
            if module_runtime_only and not path.is_relative_to(root):
                excluded.append(dict(entry=str(entry), path=str(path),
                                     reason='wheel entry outside module installation root'))
                continue
            bindings[str(path)] = None
        versions[name] = dict(name=dist.metadata['Name'], version=dist.version,
                              home_page=dist.metadata.get('Home-page'),
                              project_urls=dist.metadata.get_all('Project-URL', []),
                              selected_files=len(selected)-len(excluded),
                              scope='module files' if module_runtime_only else 'full distribution',
                              excluded_files=excluded)
    return bindings, versions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('receipt', type=Path)
    p.add_argument('--packages', nargs='*', default=[])
    p.add_argument('--artifacts', type=Path)
    p.add_argument('--extra-path', type=Path)
    p.add_argument('--module-runtime-only', action='store_true',
                   help='Record module files for copied runtimes; list wheel entries outside the library root')
    a = p.parse_args()
    if a.output.exists() or a.receipt.exists():
        raise ValueError('retain prior inventory/receipt; supply new paths')
    if not a.packages and not a.artifacts:
        raise ValueError('explicit packages or artifact bindings are required')
    if a.extra_path:
        sys.path.insert(0, str(a.extra_path))
    started = datetime.now(timezone.utc).isoformat()
    bindings, packages = package_bindings(a.packages, a.module_runtime_only)
    if a.artifacts:
        bindings.update(json.loads(a.artifacts.read_text()))
    observed = inventory(bindings)
    with a.output.open('x') as stream:
        stream.write(json.dumps(observed, indent=2)+'\n')
    receipt = dict(started=started, finished=datetime.now(timezone.utc).isoformat(),
                   executable=sys.executable, python=sys.version, packages=packages,
                   extra_path=str(a.extra_path) if a.extra_path else None,
                   module_runtime_only=a.module_runtime_only,
                   artifact_plan_sha256=sha(a.artifacts) if a.artifacts else None,
                   unique_files=len(observed), inventory_sha256=sha(a.output),
                   worker_sha256=sha(__file__), read_chunk_bytes=1024*1024,
                   concurrent_read_streams=1)
    with a.receipt.open('x') as stream:
        stream.write(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
