#!/usr/bin/env python3
"""Merge the checked AIC extension into the existing viewer, preserving its assets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from origins import check as check_origins


def merge(viewer, dashboard, source):
    manifest=json.loads((dashboard/'aic2026/manifest.json').read_text())
    provenance=json.loads((viewer/'site-source.json').read_text())
    if manifest['source_commit']!=source or provenance['source_commit']!=source:
        raise ValueError('viewer and AIC source commits differ')
    if (viewer/'aic2026').exists():raise ValueError('AIC extension already present')
    shutil.copytree(dashboard/'aic2026',viewer/'aic2026')
    index=viewer/'index.html';text=index.read_text()
    if '<h1>' not in text:raise ValueError('viewer heading changed')
    navigation='<nav aria-label="Results" style="flex-basis:100%;display:flex;flex-wrap:wrap;gap:10px 22px;padding:12px 0;border-bottom:2px solid var(--accent)"><a href="/">Image explorer</a><a href="/aic2026/">AIC2026 results</a><a href="/aic2026/reproduction.html">Reproduction evidence</a><a href="/aic2026/complexity.html">Benchmarks</a><a href="https://github.com/imazen/zenmetrics/tree/'+source+'">Source</a><a href="/aic2026/#downloads">Downloads</a></nav>'
    text=text.replace('<h1>',navigation+'\n<h1>',1)
    index.write_text(text)
    check_origins(viewer/'aic2026')
    provenance['aic2026']={'source_commit':source,'manifest':'aic2026/manifest.json'}
    provenance['files']={p.relative_to(viewer).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(viewer.rglob('*')) if p.is_file() and p.name!='site-source.json'}
    (viewer/'site-source.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Merged AIC2026 into existing viewer:',source)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--viewer',type=Path,required=True);parser.add_argument('--dashboard',type=Path,required=True);parser.add_argument('--source',required=True);a=parser.parse_args();merge(a.viewer,a.dashboard,a.source)
