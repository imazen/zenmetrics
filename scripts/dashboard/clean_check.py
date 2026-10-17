#!/usr/bin/env python3
"""Verify a tracked revision independently of ignored files and the working tree."""
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

root=Path(__file__).resolve().parents[2]
revision=subprocess.check_output(['jj','log','-r','@','--no-graph','-T','commit_id'],cwd=root,text=True).strip()
git_dir=subprocess.check_output(['jj','git','root'],cwd=root,text=True).strip()
with tempfile.TemporaryDirectory(prefix='dashboard-clean-',dir=Path.home()/'tmp') as scratch:
    base=Path(scratch);source=base/'source';source.mkdir()
    archive=base/'source.tar'
    with archive.open('wb') as f:
        subprocess.run(['git','--git-dir='+git_dir,'archive',revision,'scripts/dashboard','web/dashboard',
                        'benchmarks','docs/METRIC_PROVENANCE.md','scripts/ci/check_hygiene.py','scripts/lib/hygiene_patterns.txt','Cargo.toml','DIVERGENCES.md','docs/AIC2026_METRICS_AND_FITTING.md','docs/AIC2026_COVERAGE_AUDIT_2026-10-15.md','crates'],stdout=f,check=True)
    with tarfile.open(archive) as f:f.extractall(source,filter='data')
    subprocess.run([sys.executable,str(source/'scripts/dashboard/catalog.py'),'--check'],check=True)
    subprocess.run([sys.executable,str(source/'scripts/dashboard/extract_markdown.py'),'--check'],check=True)
    subprocess.run([sys.executable,str(source/'scripts/dashboard/build.py'),'--output',str(base/'site')],check=True)
    subprocess.run([sys.executable,str(source/'scripts/dashboard/check.py'),str(base/'site')]+(['--screenshots',os.environ['DASHBOARD_SCREENSHOTS']] if os.environ.get('DASHBOARD_SCREENSHOTS') else []),check=True)
    print('Clean tracked-source build and browser verification passed:',revision)
