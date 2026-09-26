#!/usr/bin/env python3
"""Build lane examples and bind their binary hashes to the exact source tree."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/var/tmp/gmsd-chroma')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    subprocess.run(['python3','scripts/gmsd-chroma/ensure_cargo_patch.py'],check=True)
    sources = [Path('Cargo.toml'),Path('crates/gmsd/Cargo.toml'),
               *sorted(Path('crates/gmsd/src').glob('*.rs')),
               *sorted(Path('crates/gmsd/examples').glob('*.rs'))]
    before = {str(p):sha(p) for p in sources}
    lock_before = sha(Path('Cargo.lock'))
    command = ['cargo','--config',
               'patch.crates-io.zenbench.path="/var/tmp/gmsd-chroma/zenbench-snapshot"',
               'build','-p','gmsd','--release','--features','parallel,avx512','--examples']
    subprocess.run(command,check=True)
    assert {str(p):sha(p) for p in sources} == before, 'source changed during example build'
    # Cargo may resolve the pinned harness patch in the lockfile. The actual
    # post-resolution lockfile is the build input retained for verification.
    before['Cargo.lock'] = sha(Path('Cargo.lock'))
    binaries = {name:sha(ROOT/'target/release/examples'/name)
                for name in ['mdsi_oracle','chroma_speed','colour_score','mdsi_vs_reference']}
    record = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source_sha256=before,binary_sha256=binaries,argv=command,
                  lock_sha256_before=lock_before,
                  rustc=subprocess.check_output(['rustc','--version'],text=True).strip(),
                  zenbench_snapshot_sha256=sha(ROOT/'zenbench_snapshot.json'))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    history = ROOT/('build_examples_'+stamp+'.json')
    history.write_text(json.dumps(record,indent=2)+'\n')
    (ROOT/'build_examples_latest.json').write_text(history.read_text())
    print('build_examples_manifest',str(history),sha(history),flush=True)


if __name__ == '__main__':
    main()
