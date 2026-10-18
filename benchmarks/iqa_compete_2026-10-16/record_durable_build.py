#!/usr/bin/env python3
"""Record a supplied native durable worker, its source and actual inventory."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
from prepare_durable import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('binary', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--features', required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--snapshot-id', help='Observed snapshot export revision from the build log')
    p.add_argument('--command-file', type=Path, required=True, help='JSON argv or structured build-command receipt; never interpreted by a shell')
    a = p.parse_args()
    if a.output.exists():
        p.error('preserve the existing build receipt; choose a new path')
    binary = a.binary.resolve()
    source_commit = subprocess.check_output(['git', 'rev-parse', a.source_commit], text=True).strip()
    inventory = subprocess.check_output([str(binary), 'inventory'])
    sources = ['Cargo.lock', 'benchmarks/heaptrack/drivers/cpu_profile/Cargo.toml',
        'benchmarks/heaptrack/drivers/cpu_profile/src/bin/iqa_compete.rs',
        'benchmarks/heaptrack/drivers/cpu_profile/src/bin/iqa_compete_dvifmish/mod.rs']
    tree = a.snapshot / 'work/zenmetrics'
    dependency_patch = a.snapshot / 'dvifmish-patch.meta.json'
    overlay = json.loads(dependency_patch.read_bytes()) if dependency_patch.exists() else None
    # The snapshot may have been exported again for tests. Source must still
    # agree with the recorded commit; benchmark-only engine patches have receipts.
    for name in sources:
        committed = subprocess.check_output(['git', 'show', source_commit + ':' + name])
        if committed != (tree / name).read_bytes():
            expected = (overlay or {}).get('files', {}).get(name, {})
            digest = __import__('hashlib').sha256(committed).hexdigest()
            if expected.get('original_sha256') != digest or expected.get('adapted_sha256') != sha(tree / name):
                raise ValueError('compiled source receipt differs from committed source: ' + name)
    dependency_command = ['otool', '-L', str(binary)] if platform.system() == 'Darwin' else ['ldd', str(binary)]
    value = dict(source_commit=source_commit, binary_sha256=sha(binary),
        inventory_sha256=__import__('hashlib').sha256(inventory).hexdigest(), features=a.features,
        command=json.loads(a.command_file.read_bytes()), compiler=subprocess.check_output(['rustc', '-Vv'], text=True),
        platform=platform.platform(), architecture=platform.machine(),
        binary_dependencies=subprocess.check_output(dependency_command, text=True),
        source_files={name:sha(tree / name) for name in sources})
    if a.snapshot_id:
        value['snapshot_id'] = a.snapshot_id
    patch = a.snapshot / 'ort-threads.meta.json'
    if patch.exists():
        value['benchmark_patch'] = json.loads(patch.read_bytes())
    if overlay:
        value['benchmark_dependency_patch'] = overlay
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(value, indent=2) + '\n')
    a.output.with_name(a.output.stem + '.inventory.json').write_bytes(inventory)
    print(json.dumps(dict(build=str(a.output.resolve()), binary_sha256=value['binary_sha256'],
                         inventory_sha256=value['inventory_sha256'])))


if __name__ == '__main__':
    main()
