#!/usr/bin/env python3
"""Run frozen Rust distance predictors and record provenance, without labels."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import argparse
import re

ROOT = Path('/var/tmp/gmsd-chroma')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-manifest',type=Path)
    ap.add_argument('--out-dir',type=Path,default=ROOT/'colour_v2',
                    help='where predictions.tsv and its manifest go (inputs stay in colour_v2)')
    args = ap.parse_args()
    subprocess.run(['python3','scripts/gmsd-chroma/preflight.py'],check=True)
    for name in ['parity.json','ms_parity.json']:
        assert json.loads((ROOT/name).read_text())['passed'] is True
    binary = ROOT/'target/release/examples/colour_score'
    inputs = ROOT/'colour_v2/raw_pairs.tsv'
    out = args.out_dir/'predictions.tsv'
    assert ROOT in args.out_dir.resolve().parents
    args.out_dir.mkdir(exist_ok=True)
    manifest = out.with_suffix('.manifest.json')
    assert not out.exists() and not manifest.exists()
    source = [Path('crates/gmsd/Cargo.toml'),Path('Cargo.lock'),
              Path('crates/gmsd/examples/colour_score.rs'),
              *sorted(Path('crates/gmsd/src').glob('*.rs'))]
    source_hashes = {str(p):sha(p) for p in source}
    build_path = ROOT/'build_examples_latest.json'
    build = json.loads(build_path.read_text())
    for name in ['parity.json','ms_parity.json']:
        gate = json.loads((ROOT/name).read_text())
        assert gate['source_sha256'] == build['source_sha256'], 'parity source differs from scorer build'
        assert gate['binary_sha256'] == build['binary_sha256']['mdsi_oracle']
    assert build['binary_sha256']['colour_score'] == sha(binary)
    assert all(build['source_sha256'][p] == digest for p,digest in source_hashes.items())
    # Require the current source tree to be committed. Documentation may
    # remain dirty; it does not define these predictor outputs.
    if args.source_manifest:
        frozen = json.loads(args.source_manifest.read_text())
        commit = frozen['implementation_commit']
        assert re.fullmatch('[0-9a-f]{40}',commit)
        for name,digest in source_hashes.items():
            if name.startswith('crates/'):
                assert frozen['files'][name] == digest, ('frozen metric source changed',name)
    else:
        changed = set(subprocess.check_output(['jj','diff','--name-only'],text=True).splitlines())
        assert not changed.intersection(source_hashes), 'commit metric sources before prediction freeze'
        commit = subprocess.check_output(['jj','log','--no-graph','-r','@-',
                                         '-T','commit_id'],text=True).strip()
    binary_sha = sha(binary)
    env = os.environ.copy()
    env['RAYON_NUM_THREADS'] = '8'
    subprocess.run([str(binary),str(inputs),str(out)],check=True,env=env)
    assert sha(binary) == binary_sha
    assert {str(p):sha(p) for p in source} == source_hashes
    data = dict(table_sha256=sha(out),binary_sha256=binary_sha,
                implementation_commit=commit,source_sha256=source_hashes,
                input_sha256=sha(inputs),build_manifest_sha256=sha(build_path),
                population_sha256=sha(ROOT/'colour_v2/population.json'),
                human_labels_read=False)
    if args.source_manifest:
        data['frozen_source_manifest_sha256'] = sha(args.source_manifest)
    manifest.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,sort_keys=True))


if __name__ == '__main__':
    main()
