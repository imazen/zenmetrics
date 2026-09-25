#!/usr/bin/env python3
"""Build the user-authorized, pinned main-branch fast-ssim2 CPU peer."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/var/tmp/gmsd-chroma')
DRIVER = Path('scripts/gmsd-chroma/fast_ssim2_peer')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    subprocess.run(['python3','scripts/gmsd-chroma/preflight.py'],check=True)
    source = json.loads((ROOT/'fast_ssim2_main_source.json').read_text())
    assert source['commit'] == 'f011259a0cd2fb7f538e08398ba41c0ccd0dfcb4'
    for name,digest in source['source_sha256'].items():
        assert sha(ROOT/'fast-ssim2-main'/name) == digest, name
    subprocess.run(['cargo','fmt','--manifest-path',str(DRIVER/'Cargo.toml')],check=True)
    inputs = {str(p):sha(p) for p in [DRIVER/'Cargo.toml',DRIVER/'src/main.rs']}
    env = {**os.environ,'CARGO_TARGET_DIR':str(ROOT/'peer-target')}
    mains = json.loads((ROOT/'src/main_snapshots.json').read_text())
    patches = {'archmage': mains['archmage']['path'],
               'archmage-macros': mains['archmage']['path']+'/archmage-macros',
               'magetypes': mains['archmage']['path']+'/magetypes',
               'enough': mains['enough']['path']+'/crates/enough'}
    command = ['cargo']
    for name,path in patches.items():
        command += ['--config',f'patch.crates-io.{name}.path={json.dumps(path)}']
    command += ['build','--release','--manifest-path',str(DRIVER/'Cargo.toml')]
    subprocess.run(command,check=True,env=env)
    assert inputs == {p:sha(Path(p)) for p in inputs}
    binary = ROOT/'peer-target/release/gmsd-chroma-fast-ssim2-peer'
    manifest = dict(producer='fast-ssim2-cpu',commit=source['commit'],
                    source_manifest_sha256=sha(ROOT/'fast_ssim2_main_source.json'),
                    driver_source_sha256=inputs,lock_sha256=sha(DRIVER/'Cargo.lock'),
                    binary_sha256=sha(binary),argv=command,
                    rustc=subprocess.check_output(['rustc','--version'],text=True).strip(),
                    features=['imgref'],rayon_enabled=False,
                    main_snapshots=mains)
    (ROOT/'fast_peer_build.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,sort_keys=True))
    # Use the existing job-system owner on the authorized single worker.
    subprocess.run(['python3','scripts/gmsd-chroma/ensure_cargo_patch.py'],check=True)
    owner_files = [Path('Cargo.toml'), Path('Cargo.lock')]
    for crate in ['zenfleet-worker','zenfleet-core','zenfleet-ledger']:
        owner_files.extend(sorted(Path('crates',crate).rglob('*.rs')))
        owner_files.append(Path('crates',crate,'Cargo.toml'))
    owner_inputs = {str(p):sha(p) for p in owner_files}
    worker_command = ['cargo','build','--release','--locked','-p','zenfleet-worker',
                      '--bin','zenfleet-worker']
    subprocess.run(worker_command,check=True)
    assert owner_inputs == {p:sha(Path(p)) for p in owner_inputs}
    worker = ROOT/'target/release/zenfleet-worker'
    (ROOT/'peer_worker_build.json').write_text(json.dumps(dict(
        argv=worker_command, source_sha256=owner_inputs,
        binary_sha256=sha(worker)),indent=2)+'\n')
    print('zenfleet_worker_sha256',sha(worker),flush=True)


if __name__=='__main__':
    main()
