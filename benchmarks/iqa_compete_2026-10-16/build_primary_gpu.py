#!/usr/bin/env python3
"""Build a separate CUDA cohort from the unchanged CPU reference snapshot.

Caller supplies both heavy locks, a resource cap, TMPDIR and a full log.
No neural features or benchmark source overlays are enabled.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess

from prepare_durable import sha
from record_durable_build import verify_source_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    snapshot = a.snapshot.resolve(strict=True)
    baseline = json.loads((snapshot / 'build-receipt.json').read_text())
    revision = baseline['source_revision']
    assert revision == 'e2d414c33a6d3c7da50a818fa36e56654d3ebeef' and not baseline['adapters_applied']
    tree = snapshot / 'work/zenmetrics'
    for name, digest in baseline['files'].items():
        assert sha(tree / name) == digest
        original = subprocess.check_output(['git', 'show', revision + ':' + name])
        verify_source_file(name, original, tree / name, [])
    a.output.mkdir()
    root = a.output.resolve()
    features = 'compete,metrics-parallel,study-gpu-cuda'
    command = ['cargo', 'build', '--manifest-path', str(tree / 'Cargo.toml'), '--locked', '--release',
               '-p', 'cpu-profile', '--features', features, '--bin', 'iqa-compete']
    env = dict(os.environ, CARGO_TARGET_DIR=str(root / 'target'), CARGO_PROFILE_RELEASE_LTO='thin',
               CUDARC_CUDA_VERSION='12000')
    for name in ['RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CARGO_BUILD_RUSTFLAGS']:
        env.pop(name, None)
    metadata = dict(command=command, source_revision=revision, features=features, adapters_applied=False,
                    source_snapshot=str(snapshot), baseline_receipt_sha256=sha(snapshot / 'build-receipt.json'),
                    source_files=baseline['files'], native_cpu_flags='unset', lto='thin',
                    CUDARC_CUDA_VERSION='12000', platform=platform.platform(),
                    compiler=subprocess.check_output(['rustc', '-Vv'], text=True))
    (root / 'build-command.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('COMMAND', json.dumps(command), flush=True)
    subprocess.run(command, env=env, check=True)
    binary = root / 'target/release/iqa-compete'
    fingerprints = {str(path.relative_to(root)): json.loads(path.read_text())['features']
                    for path in (root / 'target/release/.fingerprint').glob('zenmetrics-api-*/lib-zenmetrics_api.json')}
    assert fingerprints and any('"cuda"' in value for value in fingerprints.values())
    metadata.update(binary_sha256=sha(binary), api_feature_fingerprints=fingerprints,
                    binary_dependencies=subprocess.check_output(['ldd', str(binary)], text=True))
    for name, digest in baseline['files'].items():
        assert sha(tree / name) == digest
    (root / 'build-receipt.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(dict(binary=str(binary), sha256=metadata['binary_sha256'])), flush=True)


if __name__ == '__main__':
    main()
