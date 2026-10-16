"""Compile the computed-grid experiment in the isolated pinned snapshot.

Caller holds the heavy lock. The tracked source tree is never modified.
The snapshot's library source is restored even if compilation fails.
"""
import os
import pathlib
import shutil
import subprocess

snapshot = pathlib.Path(os.environ['LOCK_SNAP_DIR'])
root = snapshot / 'work/zenmetrics'
source = root / 'crates/brisque/src/lib.rs'
original = source.read_bytes()
text = original.decode()
text = text.replace('mod params;', '#[path = "../../../benchmarks/brisque_review_2026-10-15/computed_experiment.rs"]\nmod computed;\n#[allow(dead_code)]\nmod params;')
text = text.replace('use params::{ETA_FACTOR, KERNEL49, R_AGGD, R_GGD, RHO, SV, SV_COEF};',
                    'use computed::{ETA_FACTOR, R_AGGD, R_GGD};\nuse params::{KERNEL49, RHO, SV, SV_COEF};')
text = text.replace('first_argmin(&R_GGD, ratio)', 'first_argmin(&R_GGD[..], ratio)')
text = text.replace('first_argmin(&R_AGGD, ratio)', 'first_argmin(&R_AGGD[..], ratio)')
assert text != original.decode() and 'use computed::' in text
try:
    source.write_text(text)
    env = dict(os.environ, CARGO_TARGET_DIR=str(snapshot / 'target'))
    subprocess.run(['cargo', '--locked', 'build', '-p', 'brisque', '--example', 'review', '--release'],
                   cwd=root, env=env, check=True)
    shutil.copy2(snapshot / 'target/release/examples/review',
                 pathlib.Path.home() / 'tmp/aic-variants-scratch/round6/computed-independent-review')
finally:
    source.write_bytes(original)
    # Restore ordinary Cargo artifacts too: a later snapshot export may restore
    # the earlier source mtime, so leaving computed artifacts here is unsafe.
    subprocess.run(['cargo', '--locked', 'build', '-p', 'brisque', '--example', 'review', '--release'],
                   cwd=root, env=dict(os.environ, CARGO_TARGET_DIR=str(snapshot / 'target')), check=True)
