#!/usr/bin/env python3
"""Merge the checked AIC extension into the existing viewer, preserving its assets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from origins import check as check_origins


BRIEF = 'https://metric-brief.pages.dev/'


def style_landing(text, source):
    if 'class="evidence-header"' in text:raise ValueError("landing theme already present")
    if '<h1>' not in text:raise ValueError('viewer heading changed')
    # The workbench stays here; benchmark and reproduction evidence lives on its own site (the metric brief).
    navigation='<nav aria-label="Results" style="flex-basis:100%;display:flex;flex-wrap:wrap;gap:10px 22px;padding:12px 0;border-bottom:2px solid var(--accent)"><a href="/">Image explorer</a><a href="'+BRIEF+'">Speed, reproduction and model cards ↗</a><a href="https://github.com/imazen/zenmetrics/tree/'+source+'">Source</a><a href="https://github.com/imazen/zenmetrics/releases">Downloads</a></nav>'
    navigation=navigation.replace('</nav>','<a href="https://gigapixels.imazen.org/">Gigapixels</a></nav>')
    text=text.replace('<h1>',navigation+'\n<h1>',1)
    # Keep the interactive explorer intact while giving its entrance a clear identity.
    text=text.replace('<header>','<header class="evidence-header"><div class="evidence-kicker">IMAGE QUALITY · EXPLORE / MEASURE / VERIFY</div>',1)
    theme='''<style>
    .evidence-header{position:relative;isolation:isolate;overflow:hidden;padding:24px 28px 28px;border:1px solid var(--line);border-radius:18px;background:var(--panel);box-shadow:0 12px 40px #00000008;gap:14px 22px}
    .evidence-header:before{content:"";position:absolute;inset:0;z-index:-1;pointer-events:none;background:radial-gradient(ellipse at 100% 0%,color-mix(in srgb,var(--accent) 15%,transparent),transparent 60%),repeating-linear-gradient(135deg,transparent 0 26px,color-mix(in srgb,var(--ink) 3%,transparent) 26px 27px)}
    .evidence-header .evidence-kicker{flex-basis:100%;color:var(--accent);font:600 10px/1.5 var(--mono);letter-spacing:.2em}
    .evidence-header nav{order:3;font-size:12px}.evidence-header nav a{text-decoration:none}.evidence-header nav a:hover{text-decoration:underline}
    .evidence-header h1{font:normal clamp(38px,5vw,62px)/1.05 Georgia,serif;letter-spacing:-.045em}
    .evidence-header p{font-size:15px;max-width:49ch}.evidence-header> a{font-size:12px}
    @media(max-width:600px){.evidence-header{padding:20px 18px;border-radius:12px}.evidence-header nav{gap:10px 18px!important}.evidence-header .evidence-kicker{letter-spacing:.12em}.seg{flex-wrap:wrap;max-width:100%}}
    </style>'''
    text=text.replace('</head>',theme+'</head>',1)
    return text


def refresh_landing(viewer, source):
    """Apply only landing HTML/CSS, preserving separately versioned viewer/AIC assets."""
    provenance=json.loads((viewer/'site-source.json').read_text())
    for name,digest in provenance['files'].items():
        if hashlib.sha256((viewer/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('existing asset hash mismatch: '+name)
    index=viewer/'index.html'
    index.write_text(style_landing(index.read_text(),source))
    provenance['landing']={'source_commit':source,'scope':'HTML/CSS navigation only'}
    provenance['files']['index.html']=hashlib.sha256(index.read_bytes()).hexdigest()
    (viewer/'site-source.json').write_text(json.dumps(provenance,indent=2)+'\n')


def merge(viewer, dashboard, source):
    manifest=json.loads((dashboard/'aic2026/manifest.json').read_text())
    provenance=json.loads((viewer/'site-source.json').read_text())
    if manifest['source_commit']!=source or provenance['source_commit']!=source:
        raise ValueError('viewer and AIC source commits differ')
    if (viewer/'aic2026').exists():raise ValueError('AIC extension already present')
    shutil.copytree(dashboard/'aic2026',viewer/'aic2026')
    index=viewer/'index.html';text=index.read_text()
    text=style_landing(text,source)
    index.write_text(text)
    check_origins(viewer/'aic2026')
    provenance['aic2026']={'source_commit':source,'manifest':'aic2026/manifest.json'}
    provenance['files']={p.relative_to(viewer).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(viewer.rglob('*')) if p.is_file() and p.name!='site-source.json'}
    (viewer/'site-source.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Merged AIC2026 into existing viewer:',source)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--viewer',type=Path,required=True);parser.add_argument('--dashboard',type=Path);parser.add_argument('--landing-only',action='store_true');parser.add_argument('--source',required=True);a=parser.parse_args()
    if a.landing_only:refresh_landing(a.viewer,a.source)
    elif a.dashboard:merge(a.viewer,a.dashboard,a.source)
    else:parser.error('--dashboard is required unless --landing-only is selected')
