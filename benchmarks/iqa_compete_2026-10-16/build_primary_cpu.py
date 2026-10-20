#!/usr/bin/env python3
"""Build a fresh pinned master cohort and record actual AVX-512 feature receipts.

Caller supplies both heavy locks, run-heavy, TMPDIR and a full build log.
No source overlays or native-CPU compiler flags are applied.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from prepare_durable import sha
from record_durable_build import verify_source_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('revision');p.add_argument('output',type=Path)
    p.add_argument('--feature', action='append', default=[],
                   help='Additional explicit Cargo feature, e.g. zenmetrics-cli/nr')
    a=p.parse_args()
    repo=Path(__file__).resolve().parents[2]
    revision=subprocess.check_output(['git','-C',str(repo),'rev-parse',a.revision],text=True).strip()
    assert not a.output.exists() and not a.output.is_symlink()
    env=dict(os.environ,LOCK_SNAP_DIR=str(a.output.resolve()))
    subprocess.run(['bash',str(repo/'scripts/ci/lock.sh'),'--check','--rev',revision],env=env,cwd=repo,check=True)
    root=a.output.resolve(strict=True);tree=root/'work/zenmetrics'
    files=['Cargo.toml','Cargo.lock','ci/sibling-pins.tsv','crates/zenmetrics-cli/Cargo.toml',
        'crates/zenmetrics-cli/src/metrics/mod.rs','benchmarks/heaptrack/drivers/cpu_profile/Cargo.toml',
        'benchmarks/heaptrack/drivers/cpu_profile/src/bin/iqa_compete.rs']
    for crate in ['vmaf','haarpsi','fsim','vsi','gmsd','msssim','vif','psnrhvs','iwssim','cvvdp','nlpd','mad-iqa']:
        files += [str(p.relative_to(tree)) for p in (tree/'crates'/crate).rglob('*') if p.is_file() and (p.suffix=='.rs' or p.name=='Cargo.toml')]
    hashes={}
    for name in files:
        original=subprocess.check_output(['git','-C',str(repo),'show',revision+':'+name])
        verify_source_file(name,original,tree/name,[]);hashes[name]=sha(tree/name)
    features=['compete','metrics-parallel',*a.feature]
    command=['cargo','build','--manifest-path',str(tree/'Cargo.toml'),'--locked','--release','-p','cpu-profile','--features',','.join(features),'--bin','iqa-compete']
    env.update(CARGO_TARGET_DIR=str(root/'target'),CARGO_PROFILE_RELEASE_LTO='thin')
    for key in ['RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','CARGO_BUILD_RUSTFLAGS']:env.pop(key,None)
    metadata=dict(source_revision=revision,features=features,adapters_applied=False,files=hashes,command=command,lto='thin',
        native_cpu_flags='RUSTFLAGS,CARGO_ENCODED_RUSTFLAGS,CARGO_BUILD_RUSTFLAGS unset',worker_sha256=sha(Path(__file__)),
        compiler=subprocess.check_output(['rustc','-Vv'],text=True))
    (root/'build-command.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('BUILD',json.dumps(command),flush=True);subprocess.run(command,env=env,check=True)
    binary=root/'target/release/iqa-compete'
    fingerprints={}
    for crate in ['vmaf','haarpsi','fsim','vsi','gmsd','msssim','vif','psnrhvs','iwssim','cvvdp','nlpd','mad-iqa']:
        records={str(p.relative_to(root)):json.loads(p.read_text())['features'] for p in (root/'target/release/.fingerprint').glob(crate+'-*/lib-*.json')}
        assert records and all('"avx512"' in s for s in records.values()),(crate,records)
        fingerprints[crate]=records
    metadata.update(binary_sha256=sha(binary),avx512_feature_fingerprints=fingerprints,
        binary_dependencies=subprocess.check_output(['ldd',str(binary)],text=True))
    assert all(sha(tree/name)==digest for name,digest in hashes.items())
    (root/'build-receipt.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('BUILT',revision,metadata['binary_sha256'],flush=True)


if __name__=='__main__':main()
