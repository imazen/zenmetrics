#!/usr/bin/env python3
"""Prove the new RGB regression fails on old coefficients and passes the fix.

Uses a new CI-pinned export/target. Existing golden values stay unchanged.
Caller supplies both host locks, run-heavy and a complete log.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from octave_primary import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('revision');p.add_argument('output',type=Path);a=p.parse_args()
    repo=Path(__file__).resolve().parents[2]
    revision=subprocess.check_output(['git','-C',str(repo),'rev-parse',a.revision],text=True).strip()
    assert not a.output.exists()
    env=dict(os.environ,LOCK_SNAP_DIR=str(a.output.resolve()))
    subprocess.run(['bash',str(repo/'scripts/ci/lock.sh'),'--check','--rev',revision],cwd=repo,env=env,check=True)
    root=a.output.resolve(strict=True);tree=root/'work/zenmetrics';lib=tree/'crates/vif/src/lib.rs'
    before=subprocess.check_output(['git','-C',str(repo),'show','e2d414c33a6d3c7da50a818fa36e56654d3ebeef:crates/vif/src/lib.rs'])
    after=subprocess.check_output(['git','-C',str(repo),'show',revision+':crates/vif/src/lib.rs'])
    assert lib.read_bytes()==after and b'const LUMA_COEF: [f64; 3]' in after and b'const LUMA_COEF: [f32; 3]' in before
    (root/'baseline-lib.rs').write_bytes(before);(root/'candidate-lib.rs').write_bytes(after)
    env.update(CARGO_TARGET_DIR=str(root/'target'))
    for key in ['RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','CARGO_BUILD_RUSTFLAGS']:env.pop(key,None)
    base=['cargo','test','--manifest-path',str(tree/'Cargo.toml'),'--locked','-p','vif','--all-features']
    logs=[]
    def run(name,command,expected=0):
        print('RUN',name,json.dumps(command),flush=True)
        with (root/(name+'.log')).open('x') as f:r=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
        text=(root/(name+'.log')).read_text();print(text,flush=True)
        assert r.returncode==expected,(name,r.returncode)
        logs.append(dict(name=name,command=command,returncode=r.returncode,log_sha256=sha(root/(name+'.log'))))
        return text
    lib.write_bytes(before)
    text=run('baseline-regression',base+['tests::rgb_luma_coefficients_keep_double_precision','--','--exact','--nocapture'],101)
    # Rust's full test path includes tests::. Use the unambiguous full selector.
    assert 'rgb_luma_coefficients_keep_double_precision' in text and 'FAILED' in text and '0.7120196377747934' in text
    lib.write_bytes(after)
    run('candidate-all-tests',base)
    run('candidate-clippy',['cargo','clippy','--manifest-path',str(tree/'Cargo.toml'),'--locked','-p','vif','--all-features','--all-targets','--','-D','warnings'])
    assert lib.read_bytes()==after
    (root/'result.json').write_text(json.dumps(dict(revision=revision,baseline_revision='e2d414c33a6d3c7da50a818fa36e56654d3ebeef',
        baseline_sha256=sha(root/'baseline-lib.rs'),candidate_sha256=sha(root/'candidate-lib.rs'),
        tests_sha256=sha(tree/'crates/vif/src/tests.rs'),logs=logs,worker_sha256=sha(Path(__file__))),indent=2)+'\n')


if __name__=='__main__':main()
