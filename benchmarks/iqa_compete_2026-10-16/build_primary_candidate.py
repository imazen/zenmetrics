#!/usr/bin/env python3
"""Compile explicit source-file overlays on a fresh CI-pinned baseline export.

Caller supplies heavy-job locks/cap and a complete build log. No source in
an existing build is modified. The receipt separates base and overlay commits.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from prepare_durable import sha
from record_durable_build import verify_source_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-build', type=Path, required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--file', action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    baseline = a.baseline_build.resolve(strict=True)
    previous = json.loads((baseline / 'build-receipt.json').read_text())
    base_rev = previous['source_revision']
    assert base_rev=='e2d414c33a6d3c7da50a818fa36e56654d3ebeef' and not previous['adapters_applied']
    repo = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(['git','-C',str(repo),'rev-parse',a.revision],text=True).strip()
    assert not a.output.exists() and not a.output.is_symlink()
    env = dict(os.environ, LOCK_SNAP_DIR=str(a.output.resolve()))
    subprocess.run(['bash',str(repo/'scripts/ci/lock.sh'),'--check','--rev',base_rev],env=env,cwd=repo,check=True)
    root = a.output.resolve(strict=True); tree = root/'work/zenmetrics'
    overlays = {}
    for name in a.file:
        path = Path(name)
        assert not path.is_absolute() and '..' not in path.parts and name.startswith('crates/') and name.endswith('.rs')
        original = subprocess.check_output(['git','-C',str(repo),'show',base_rev+':'+name])
        verify_source_file(name,original,tree/name,[])
        replacement = subprocess.check_output(['git','-C',str(repo),'show',revision+':'+name])
        before = sha(tree/name); (tree/name).write_bytes(replacement)
        overlays[name] = dict(original_sha256=before,adapted_sha256=sha(tree/name))
    features = 'compete,metrics-parallel'
    command = ['cargo','build','--manifest-path',str(tree/'Cargo.toml'),'--locked','--release',
               '-p','cpu-profile','--features',features,'--bin','iqa-compete']
    env.update(CARGO_TARGET_DIR=str(root/'target'),CARGO_PROFILE_RELEASE_LTO='thin')
    for name in ['RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','CARGO_BUILD_RUSTFLAGS']:
        env.pop(name,None)
    metadata = dict(source_revision=revision,baseline_revision=base_rev,source_overlay=overlays,
                    baseline_receipt_sha256=sha(baseline/'build-receipt.json'),features=features.split(','),
                    adapters_applied=False,command=command,lto='thin',native_cpu_flags='unset',
                    compiler=subprocess.check_output(['rustc','-Vv'],text=True))
    (root/'build-command.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('COMMAND',json.dumps(command),flush=True)
    subprocess.run(command,env=env,check=True)
    metadata['binary_sha256'] = sha(root/'target/release/iqa-compete')
    assert all(sha(tree/name)==record['adapted_sha256'] for name,record in overlays.items())
    (root/'build-receipt.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata),flush=True)


if __name__=='__main__':
    main()
